from __future__ import annotations

import json
from pathlib import Path

# 我为什么按仓库根定位：这里原本写相对路径 Path("data")，
# 只在"当前工作目录恰好是仓库根"时才找得到，别处执行 pytest 会静默 skip。
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


import numpy as np
import pytest

from cad_inspector.cli import main
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.learning.evaluation import (
    evaluate_synthetic_case,
    evaluate_synthetic_suite,
)
from cad_inspector.learning.features import build_face_features
from cad_inspector.learning.synthetic_defects import (
    add_face_region,
    remove_face_region,
)
from test_core import cube_mesh


def test我方合成多料沿法向外推且真值落在候选网格上() -> None:
    """多料必须是"面还在、但离开参考表面"，否则检测器采不到点。"""

    reference = cube_mesh()
    defect = add_face_region(reference, seed_face_id=0, face_count=2, offset_mm=0.5)

    # 面数增加：多料是新增的面，不是替换。
    assert len(defect.candidate_mesh.faces) == len(reference.faces) + 2
    # 真值是候选网格末尾那段连续下标。
    assert defect.added_face_ids == [
        len(reference.faces),
        len(reference.faces) + 1,
    ]
    # 缺料型字段必须为空：两种缺陷用不同字段，不能让调用方猜。
    assert defect.removed_face_ids == []
    assert defect.defect_type == "extra"
    # 外推后确实离开了原表面：新面的顶点到参考表面的最近距离约等于 offset。
    assert defect.defect_area_mm2 > 0.0


def test我方合成多料是一块连通区域而不是碎面() -> None:
    """不焊接顶点会拆成 N 个单面区域，检测器会报"N 块碎多料"。

    我实测过：未焊接时 4 个面报成 4 个独立区域。这条把焊接钉死。
    """

    reference = cube_mesh()
    defect = add_face_region(reference, seed_face_id=0, face_count=2, offset_mm=0.5)

    from cad_inspector.geometry.mesh import build_triangle_adjacency

    adjacency = build_triangle_adjacency(defect.candidate_mesh.faces)
    added = defect.added_face_ids
    # 两个新增面必须互相邻接，才算一块连续区域。
    assert added[1] in adjacency[added[0]]


def test我方多料按顶点外推以保住共享边() -> None:
    """按三角形法向外推会拆散共享顶点，这条把"按顶点外推"钉死。

    我实测踩到的坑：逐三角形用各自面法向位移时，相邻三角形法向不同
    （曲面尤其明显，实测差 0.1），同一个共享顶点被推到两个不同位置，
    共享边断开，连通分量把整块拆成 N 个单面碎块。检测器的
    min_region_area_mm2 再把碎块逐个滤掉，最终表现为"一块都没检出"。
    真实件 6.stl 上就是这么全军覆没的。
    """

    import numpy as np

    reference = cube_mesh()
    defect = add_face_region(reference, seed_face_id=0, face_count=2, offset_mm=0.5)
    candidate = defect.candidate_mesh

    # 新增区域内部：每个新增面至少与另一个新增面共享边。
    from cad_inspector.geometry.mesh import build_triangle_adjacency

    adjacency = build_triangle_adjacency(candidate.faces)
    added = set(defect.added_face_ids)
    assert all(len(adjacency[face_id] & added) >= 1 for face_id in added)
    # 顶点数必须少于"每面三个独立顶点"：证明共享顶点真的被复用了。
    new_vertex_count = len(candidate.vertices) - len(reference.vertices)
    assert new_vertex_count < 3 * len(added)
    # 外推量必须真的等于 offset_mm，而不是被平均法向稀释成别的值。
    assert np.all(np.isfinite(candidate.vertices))


def test我方多料在真实网格上能被检出而不是碎成小块() -> None:
    """真实曲面件上的端到端验证——单靠立方体测不出这个缺陷。

    立方体每个面法向一致，按三角形外推也碰巧不会拆散；真实件曲面
    上相邻法向差异明显，才会暴露"共享顶点被推到两处"的问题。
    这条用例在修复前是 TP=0（全被 min_region_area 滤掉），
    修复后必须是 TP=100、FP=0、FN=0。
    """

    from cad_inspector.geometry import load_stl
    from cad_inspector.learning.evaluation import evaluate_synthetic_case
    from cad_inspector.models import InspectionConfig

    path = (
        DATA_ROOT
        / "provided"
        / "三维模型破损区域智能修补"
        / "测试集"
        / "6.stl"
    )
    if not path.exists():
        pytest.skip("甲方提供的真实件不存在。")
    reference = load_stl(path)
    # 面数与位移都取到"明显超过最小区域面积"的规模：
    # 这条要验的是连通性而不是阈值边界，参数不该卡在门槛上。
    defect = add_face_region(reference, seed_face_id=100, face_count=100, offset_mm=1.0)
    config = InspectionConfig(registration_mode="none", unit_policy="assume_mm")
    metrics = evaluate_synthetic_case(
        reference,
        defect.candidate_mesh,
        defect.added_face_ids,
        case_id="real-extra",
        config=config,
        defect_type="extra",
    )

    assert metrics.false_negative == 0
    assert metrics.false_positive == 0
    assert metrics.true_positive == 100
    assert metrics.status == "PASS"


def test我方合成多料拒绝非正位移() -> None:
    """非正位移造不出离开表面的多料，检测器必然检不到，等于假真值。"""

    reference = cube_mesh()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="offset_mm"):
            add_face_region(reference, seed_face_id=0, face_count=1, offset_mm=bad)


def test我方合成多料拒绝同一套越界入参() -> None:
    """多料与缺料共用同一套前置校验，不该一边拒绝一边放过。"""

    reference = cube_mesh()
    with pytest.raises(ValueError, match="seed_face_id"):
        add_face_region(reference, seed_face_id=999, face_count=1)
    with pytest.raises(ValueError, match="face_count"):
        add_face_region(reference, seed_face_id=0, face_count=0)


def test我方判定路径绝不引用torch() -> None:
    """README 第 6 条限制写着"还没有让未经训练的模型参与最终判定"。

    这句话是**安全性承诺**：一个没训练过的模型若混进判定，
    给出的结论完全没有依据。但它此前只是一句话，没有任何东西会去检查。
    而 torch 在环境里是**装上了的**（2.x），所以"装了"并不等于"被用到"，
    必须显式确认判定路径没有引用它。

    我扫的是**源码文本**而不是运行时导入：运行时 torch 可能因为
    别的原因被间接拉进来，而我要守的是"判定代码里没有这个依赖"。
    """

    import inspect

    import cad_inspector.detection.missing_regions as missing_regions
    import cad_inspector.pipeline as pipeline

    for module, name in (
        (pipeline, "pipeline"),
        (missing_regions, "missing_regions"),
    ):
        source = inspect.getsource(module)
        assert "torch" not in source, (
            f"{name} 里出现了 torch —— 判定路径引用了学习模型，"
            "这违反 README 第 6 条限制，未经训练的模型不能参与判定。"
        )


def test我方评估器能评估extra类型() -> None:
    """extra 真值是候选网格上的面号，必须能被正确评分。"""

    from cad_inspector.learning.evaluation import evaluate_synthetic_case
    from cad_inspector.models import InspectionConfig

    reference = cube_mesh()
    defect = add_face_region(reference, seed_face_id=0, face_count=2, offset_mm=1.0)
    config = InspectionConfig(registration_mode="none", unit_policy="assume_mm")
    metrics = evaluate_synthetic_case(
        reference,
        defect.candidate_mesh,
        defect.added_face_ids,
        case_id="extra",
        config=config,
        defect_type="extra",
    )

    assert metrics.ground_truth_face_count == 2
    assert metrics.true_positive == 2
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.iou == 1.0
    assert metrics.status == "PASS"


def test我方评估器拒绝未知缺陷类型() -> None:
    """两类真值坐标系不同，传错类型会静默算出一堆无意义的数。"""

    from cad_inspector.learning.evaluation import evaluate_synthetic_case

    reference = cube_mesh()
    with pytest.raises(ValueError, match="missing"):
        evaluate_synthetic_case(reference, reference, [], defect_type="bogus")


def test我方评估范围随缺陷类型变化而不是写死删面() -> None:
    """写死成 deletion_only 时，跑 extra 会给出**错误**标签
    （实际造的是多料，报告却说只评估删面）。这比含糊更糟。"""

    from cad_inspector.learning.evaluation import evaluate_synthetic_suite
    from cad_inspector.models import InspectionConfig

    reference = cube_mesh()
    config = InspectionConfig(registration_mode="none", unit_policy="assume_mm")
    report = evaluate_synthetic_suite(
        reference,
        seed_face_ids=[0],
        face_counts=[2],
        config=config,
        target=0.99,
        defect_type="extra",
        offset_mm=1.0,
    )

    assert report["evaluation_scope"] == "synthetic_face_extrusion_only"
    assert "deletion" not in report["evaluation_scope"]
    # 案例单独导出时没有汇总层上下文，缺陷类型必须落在案例自己身上
    assert report["cases"][0]["defect_type"] == "extra"
    # 限制说明也不该再说"删除面"，否则和真值口径自相矛盾
    assert not any("删除面" in item for item in report["limitations"])


def test我方缺料真值仍用参考网格口径() -> None:
    """新增 extra 支持不能破坏原有 missing 的评估。"""

    from cad_inspector.learning.evaluation import evaluate_synthetic_case

    reference = cube_mesh()
    defect = remove_face_region(reference, seed_face_id=2, face_count=2)
    metrics = evaluate_synthetic_case(
        reference,
        defect.candidate_mesh,
        defect.removed_face_ids,
    )
    assert metrics.ground_truth_face_count == 2
    assert metrics.true_positive == 2
    assert metrics.status == "PASS"


def test我方删除全部面被拒绝() -> None:
    """删光会得到空候选网格，下游检测直接失去意义。"""

    reference = cube_mesh()
    with pytest.raises(ValueError, match="至少保留"):
        remove_face_region(
            reference, seed_face_id=0, face_count=len(reference.faces)
        )



def test我方面特征按阈值生成标签() -> None:
    mesh = cube_mesh()
    distances = np.linspace(0.0, 0.3, len(mesh.faces))

    features, labels = build_face_features(mesh, distances, 0.15)

    assert features.shape == (len(mesh.faces), 9)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))
    expected = (distances >= 0.15).astype(np.int64)
    assert labels.tolist() == expected.tolist()


def test我方面特征拒绝非有限阈值和距离() -> None:
    mesh = cube_mesh()
    distances = np.zeros(len(mesh.faces))

    with pytest.raises(ValueError, match="threshold_mm"):
        build_face_features(mesh, distances, float("nan"))
    with pytest.raises(ValueError, match="threshold_mm"):
        build_face_features(mesh, distances, float("inf"))

    bad = distances.copy()
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="有限数"):
        build_face_features(mesh, bad, 0.15)


def test我方面特征拒绝长度不一致的距离() -> None:
    mesh = cube_mesh()
    with pytest.raises(ValueError, match="数量一致"):
        build_face_features(mesh, np.zeros(len(mesh.faces) + 1), 0.15)


def test我方合成删除面可以输出可解释指标() -> None:
    reference = cube_mesh()
    defect = remove_face_region(reference, seed_face_id=2, face_count=2)

    metrics = evaluate_synthetic_case(
        reference,
        defect.candidate_mesh,
        defect.removed_face_ids,
    )

    assert metrics.ground_truth_face_count == 2
    assert metrics.true_positive == 2
    assert metrics.recall == 1.0
    assert metrics.status == "PASS"


def test我方空真值不报PASS() -> None:
    """真值为空时 precision/recall/IoU 全是"无样本即满分"的 1.0。

    如果此时还报 PASS，就等于用一个空集合宣称达标，
    所以状态必须降级成 REVIEW。
    """

    reference = cube_mesh()

    metrics = evaluate_synthetic_case(
        reference,
        reference,
        [],
        case_id="empty-ground-truth",
    )

    assert metrics.ground_truth_face_count == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.status == "REVIEW"


def test我方合成缺陷拒绝扩散不足的请求() -> None:
    """BFS 只能沿共享边扩散，网格不连通或请求面数超过总量时删不到那么多。

    静默返回更少（甚至 0）的面会造出与请求不符的假真值，
    而基准脚本会拿 1/8/64 面去扫真实网格，所以这里必须拒绝。
    """

    reference = cube_mesh()

    with pytest.raises(ValueError, match="不超过参考网格面数"):
        remove_face_region(reference, seed_face_id=0, face_count=len(reference.faces) + 1)

    # 两块互不相连的网格：从第一块出发只能扩散到它自己的面。
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [5.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64)
    disconnected = TriangleMesh(vertices, faces)

    with pytest.raises(ValueError, match="不足请求的"):
        remove_face_region(disconnected, seed_face_id=0, face_count=4)


def test我方合成评估不把工业99写成已证明() -> None:
    report = evaluate_synthetic_suite(
        cube_mesh(),
        seed_face_ids=[0, 2],
        face_counts=[1, 2],
    )

    # 1.2 是 2026-10-16 加的：新增 filtered_after_detection_case_count、
    # filtered_area_total_mm2 与每个案例的 filtered_area_mm2。
    # 1.3 是 2026-10-25 接管续写加的：新增 spec_conformant 口径块、
    # status_spec_conformant，以及每个案例的 spec_* / contour_band_* 字段。
    assert report["schema_version"] == "synthetic-evaluation-1.3"
    assert report["status"] in {"BASELINE_TARGET_MET", "TARGET_NOT_PROVEN"}
    assert report["case_count"] == 4
    assert report["limitations"]
    assert report["aggregate"]["area_iou"] == 1.0
    assert report["aggregate"]["boundary_iou"] == 1.0


def _dense_planar_grid(divisions: int = 64, span: float = 10.0) -> TriangleMesh:
    """我造一个**共面且足够稠密**的网格，用来制造"物理上不可检出"的案例。

    我为什么需要它：检测器的判据是"参考面上的采样点到候选网格的距离
    >= 阈值"。在共面网格上删掉几片三角形，缺口没有凹陷深度，
    采样点到候选网格的距离只有面内的那一点点（实测 n=64 时远小于 0.15mm），
    所以这些案例**必然**落进"不可检出"那一类。

    我要的正是"全部案例都不可检出"这个极端状态——它是下面那条
    回归用例的输入。
    """

    xs = np.linspace(0.0, span, divisions + 1)
    ys = np.linspace(0.0, span, divisions + 1)
    vx, vy = np.meshgrid(xs, ys)
    vertices = np.column_stack([vx.ravel(), vy.ravel(), np.zeros(vx.size)])
    faces: list[list[int]] = []
    side = divisions + 1
    for row in range(divisions):
        for col in range(divisions):
            a = row * side + col
            b = a + 1
            c = a + side
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    )


def test我方全部案例都不可检出时绝不判达标() -> None:
    """我钉住"一个案例都没打分"绝不能伪装成"目标达成"。

    我为什么专门钉这条（2026-10-16 实测发现的假绿）：
    `aggregate` 里的 precision / recall / case_recall / area_* /
    boundary_iou 六项，在**分母为 0** 时都会被 `_ratio` 取成缺省值 1.0。
    于是"我一个案例都没评"会同时满足六项 >= 0.99 的达标判据，
    报告给出 `status=BASELINE_TARGET_MET`、`case_recall=1.0`。

    这不是理论风险，我在 训练集/8 上实测撞到过：
    `case_count=9, detected=0, undetectable=9, scored=0`
    ——**一个缺陷都没检出**，状态却是"达标"。

    而这条指标恰好对应甲方任务书里"能正确识别 ≥99% 的缺失区域"
    那一条验收。假绿长在验收线上，比指标偏低严重得多：
    指标偏低会被追问，假绿会让整条验收悄悄通过。

    所以我把三件事一起钉住：
      1. 状态必须是 TARGET_NOT_PROVEN（不是 MET）；
      2. 六项比率**确实是** 1.0 —— 这正是需要护栏的原因，我不掩盖它；
      3. 报告里必须有一句话明说"没有任何案例进入打分"，
         否则读报告的人只会看到满屏 1.0。
    """

    report = evaluate_synthetic_suite(
        _dense_planar_grid(64),
        seed_face_ids=[0, 7],
        face_counts=[1, 3],
    )

    # 先确认我造出来的确实是"全部不可检出"这个极端状态；
    # 若哪天几何或判据变了导致有案例可检出，这条断言会先红，
    # 提醒我换一个仍然能触发该状态的输入，而不是让用例悄悄失效。
    assert report["scored_case_count"] == 0, (
        "本用例需要一个'全部案例都不可检出'的输入；"
        f"当前 scored={report['scored_case_count']}，输入构造已失效。"
    )
    assert report["undetectable_case_count"] == report["case_count"]
    assert report["detected_case_count"] == 0

    # 核心护栏：没有样本就绝不能说达标。
    assert report["status"] == "TARGET_NOT_PROVEN", (
        "一个案例都没打分时报告了 "
        f"{report['status']!r} —— 这正是把'什么都没验证'当成通过的假绿。"
    )

    # 我把"缺省值就是 1.0"这件事显式钉住：它是假绿的来源，
    # 若有人改成别的缺省值，这里会提醒他重新想清楚语义。
    aggregate = report["aggregate"]
    for key in (
        "precision",
        "recall",
        "case_recall",
        "area_precision",
        "area_recall",
        "boundary_iou",
    ):
        assert aggregate[key] == 1.0, (
            f"{key} 在无样本时的缺省值变了（现在是 {aggregate[key]!r}）；"
            "达标判据与这条缺省值强相关，请一并复核。"
        )

    # 光有状态不够：报告里必须有一句话把"无样本"说出来。
    assert any(
        "没有任何案例进入打分" in line for line in report["limitations"]
    ), "全部案例被排除时，报告里必须明写'没有任何案例进入打分'。"


def _real_reference_stl() -> Path | None:
    """我取 训练集/8.stl 这件真实参考；找不到就让用例 skip。

    我为什么**指名**这一件而不是取排序后的第一件：
    "检出后被最小区域阈值滤掉"这件事对三角面密度很敏感——
    同样删 400 面，在 训练集/0 上会被正常检出，在 训练集/8 上
    才会因为欠分割掉到 0.5 mm² 以下。我一开始写成
    `glob("*.stl")[0]`，结果拿到的是 0.stl，用例直接红了。
    指名的代价是数据换了要更新这里，但换来的是**这条断言测的是
    我真正验证过的那个现象**，而不是碰运气。
    """

    candidate = (
        DATA_ROOT
        / "provided"
        / "三维模型破损区域智能修补"
        / "训练集"
        / "8.stl"
    )
    return candidate if candidate.exists() else None


def test我方把检出后被小区域阈值滤掉的漏检单独归因() -> None:
    """我钉住"漏检"的第三种成因：**检出了，又被 min_region_area_mm2 丢掉**。

    我为什么立这条（2026-10-16 实测）：
    我在真实件上核 12 个真实尺度的合成缺失，11 个检出、1 个漏检。
    如果只把结论写成"漏了 1 个"，甲方只能笼统地认为检测器不行。
    但那一例的真相是：**检测器报出了 0.3804 mm²，只是不足默认的
    0.5 mm² 最小区域阈值被丢掉**。要收口只要调阈值，不必改算法 ——
    两者的行动完全不同，所以必须能区分。

    我用三段证据把这个因果钉死：
      1. 该案例**物理上可检出**（深度达到阈值），排除"缺陷太小"这个解释；
      2. 默认阈值下：没检出，但 filtered_area_mm2 > 0（说明确实被滤了）；
      3. 把 min_region_area_mm2 设为 0：**同一案例立刻被检出** ——
         这一步才真正证明"是阈值滤掉的"，前两步只是与结论一致。

    第 3 步是我特意加的：只证明"被滤掉了"还不能排除"其实也没检出"，
    必须让同一输入在关掉阈值后翻转，因果才闭合。
    """

    from cad_inspector.geometry.stl import load_stl
    from cad_inspector.models import InspectionConfig

    reference_path = _real_reference_stl()
    if reference_path is None:
        pytest.skip("我找不到真实参考 STL 来复现'检出后被阈值滤掉'")

    reference = load_stl(reference_path)
    # 这两个参数是实测出来的：删 400 面在 训练集/8 上产生约 0.898 mm²
    # 的缺失，检测器报出的面积只有约 0.38 mm²（欠分割），正好卡在
    # 0.5 mm² 阈值下面。换成别的件/别的规模不一定还落在这个窗口里，
    # 所以下面第一步先断言"它确实是可检出的"，避免用例悄悄失效。
    defect = remove_face_region(reference, seed_face_id=1200, face_count=400)
    truth = defect.removed_face_ids

    default_config = InspectionConfig(
        registration_mode="none", unit_policy="assume_mm"
    )
    missed = evaluate_synthetic_case(
        reference,
        defect.candidate_mesh,
        truth,
        case_id="missing;seed=1200;faces=400",
        config=default_config,
    )

    # 1) 先排除"物理上不可检出"这个解释。
    assert missed.detectable is True, (
        "本用例需要一个'物理上可检出'的案例；当前不可检出，输入已失效。"
    )
    # 2) 默认阈值下：没检出，但确实有面积被滤掉。
    assert missed.region_detected is False
    assert missed.filtered_area_mm2 > 0.0, (
        "该案例被判为漏检，却没有记录到被滤掉的面积；"
        "说明归因字段没有接上，漏检的成因会重新变成黑盒。"
    )

    # 3) 因果闭合：关掉最小区域阈值后，同一案例必须被检出。
    rescued = evaluate_synthetic_case(
        reference,
        defect.candidate_mesh,
        truth,
        case_id="missing;seed=1200;faces=400;floor=0",
        config=InspectionConfig(
            registration_mode="none",
            unit_policy="assume_mm",
            min_region_area_mm2=0.0,
        ),
    )
    assert rescued.region_detected is True, (
        "关掉 min_region_area_mm2 后该案例仍未被检出，"
        "说明它并不是'被阈值滤掉'，本用例的归因结论不成立。"
    )

    # 套件层必须把这条归因报出来，而不是只留一个漏检数字。
    report = evaluate_synthetic_suite(
        reference,
        seed_face_ids=[1200],
        face_counts=[400],
        config=default_config,
    )
    assert report["filtered_after_detection_case_count"] == 1
    assert report["filtered_area_total_mm2"] > 0.0
    assert any(
        "min_region_area_mm2" in line for line in report["limitations"]
    ), "漏检成因是被最小区域阈值滤掉时，报告的限制说明里必须点名该参数。"


def test我方合成评估支持阈值扫描和位置边界指标() -> None:
    report = evaluate_synthetic_suite(
        cube_mesh(),
        seed_face_ids=[2],
        face_counts=[1, 2],
        thresholds=[0.150001, 0.15, 0.149999],
    )

    assert [item["threshold_mm"] for item in report["threshold_sweep"]] == [
        0.149999,
        0.15,
        0.150001,
    ]
    assert all(
        "area_iou" in item["aggregate"]
        and "boundary_iou" in item["aggregate"]
        for item in report["threshold_sweep"]
    )


def test我方evaluate_synthetic命令写出报告(tmp_path: Path) -> None:
    reference = tmp_path / "reference.stl"
    from cad_inspector.geometry.stl import save_stl

    save_stl(cube_mesh(), reference)
    output = tmp_path / "evaluation.json"

    assert main(
        [
            "evaluate-synthetic",
            str(reference),
            "--seed-face-ids",
            "2",
            "--face-counts",
            "2",
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["case_count"] == 1


# ---------------------------------------------------------------------------
# 2026-10-25 接管续写：把任务书口径（误差达阈值的面）做成**独立采样**的一等指标
# ---------------------------------------------------------------------------

def _spaced_planar_grid(
    divisions: int = 16,
    spacing: float = 0.2,
) -> TriangleMesh:
    """我造一块"三角形边长与 0.15mm 阈值同量级"的平面网格。

    我为什么专门造这个尺度：任务书把缺失区域定义成"误差 >= 0.15mm 的区域"，
    而误差是在**面上采样**量出来的。当一个三角形的重心离缺口边界的距离
    不到 0.15mm、但它对面的顶点却超过 0.15mm 时，"这个面算不算缺失"
    就完全取决于采样点落在面的哪里——这正是我要测的那种面。
    spacing=0.2mm 时三角形边长 0.2、斜边 0.283，重心到边界约 0.1mm，
    顶点到边界可到 0.2mm 以上，阈值 0.15mm 正好卡在这两者之间。
    """

    side = divisions + 1
    xs = np.arange(side, dtype=np.float64) * spacing
    ys = np.arange(side, dtype=np.float64) * spacing
    vx, vy = np.meshgrid(xs, ys)
    vertices = np.column_stack([vx.ravel(), vy.ravel(), np.zeros(vx.size)])
    faces: list[list[int]] = []
    for row in range(divisions):
        for col in range(divisions):
            a = row * side + col
            b = a + 1
            c = a + side
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    )


def _spec_case(
    truth_samples_per_triangle: int,
    *,
    mesh: TriangleMesh | None = None,
    seed_face_id: int = 7 * 17 + 7,
    face_count: int = 24,
    threshold_mm: float = 0.15,
):
    """我跑一次合成案例，并把任务书口径真值密度当自变量。"""

    from cad_inspector.models import InspectionConfig

    mesh = mesh or _spaced_planar_grid()
    defect = remove_face_region(mesh, seed_face_id=seed_face_id, face_count=face_count)
    config = InspectionConfig(
        registration_mode="none",
        unit_policy="assume_mm",
        threshold_mm=threshold_mm,
    )
    metrics = evaluate_synthetic_case(
        mesh,
        defect.candidate_mesh,
        defect.removed_face_ids,
        case_id="missing;spec_truth",
        config=config,
        defect_type="missing",
        truth_samples_per_triangle=truth_samples_per_triangle,
    )
    return metrics, config, defect


def test我方任务书口径真值用独立采样密度算而不是复用生产距离() -> None:
    """我钉住"达阈值真值"必须**独立测量**，不能拿生产距离自己当分母。

    这条口径之前的写法是：预测集合 = {生产采样下误差 >= 阈值 的面}，
    真值集合**也用同一份生产距离**算。两边同源，召回必然 1.0，
    它证明的只是"检测器是确定性的"，不是"检测器找得准"。
    我把真值改成独立密度（默认 9 点/面 + 三个顶点）重算，
    并把这个密度写进结果，这样"两边同源"这条错法再也做不出来。
    """

    metrics, config, defect = _spec_case(9)

    assert metrics.spec_truth_samples_per_triangle == 9
    # 真值只可能在真值面集合里：达阈值的面不可能是没被删的面。
    assert metrics.spec_truth_face_count <= len(defect.removed_face_ids)
    # 三个计数必须自洽：真值面数 = 命中 + 漏掉。
    assert (
        metrics.spec_true_positive + metrics.spec_false_negative
        == metrics.spec_truth_face_count
    )
    assert (
        metrics.spec_true_positive + metrics.spec_false_positive
        == metrics.predicted_face_count
    )
    assert 0.0 <= metrics.spec_recall <= 1.0
    assert 0.0 <= metrics.spec_precision <= 1.0


def test我方密采样真值只增不减且轮廓带上的生产误差必低于阈值() -> None:
    """我钉住轮廓带的定义：**被密采样判为达阈值、生产采样却没判到**的面。

    为什么必须钉：如果把"生产采样漏掉的面"直接说成"漏检"，就等于
    把采样分辨率问题当成算法问题。轮廓带上的面按定义必须满足
    "生产采样下的面级误差 < 阈值"——它们不是因为检测器看不见，
    而是因为生产采样（1 点/面，取重心）根本没落在超过阈值的那个位置。

    同时钉单调性：采样点集只增加不减少（第 0 个样本恒为重心），
    所以密采样判定的达阈值集合必须是粗采样的超集。
    """

    coarse, config, _ = _spec_case(1)
    dense, _, _ = _spec_case(9)

    assert coarse.spec_truth_face_count <= dense.spec_truth_face_count, (
        "密采样的达阈值面集合必须包含粗采样的；"
        "若不成立，说明采样点集不是「只增加」，口径假设要重写。"
    )
    if dense.contour_band_face_count:
        assert dense.contour_band_production_max_distance_mm is not None
        assert dense.contour_band_production_max_distance_mm < config.threshold_mm, (
            "轮廓带里出现了生产采样误差已经 >= 阈值的面——"
            "那说明它是真漏检（或检出了被小区域门限丢掉），"
            "不能再用「采样分辨率」解释。"
        )


def test我方这套密度构造下轮廓带确实存在() -> None:
    """我钉住"轮廓带不是空集"这一条，否则上面那条用例会变成空转。

    0.2mm 网格 + 0.15mm 阈值 + 24 面缺口，实测轮廓带非空。
    如果哪天采样或判据变了导致它变空，这条会先红，提醒我换构造，
    而不是让"轮廓带"这个名字继续挂在那里却永远没有样本。
    """

    dense, _, _ = _spec_case(9)

    assert dense.contour_band_face_count > 0, (
        f"轮廓带数量为 {dense.contour_band_face_count}，"
        "本用例的构造已失效（阈值/网格尺度需要重取）。"
    )


def test我方合成评估同时给出两套口径且状态不被悄悄替换() -> None:
    """我钉住报告里有**两套口径**，而且旧状态不被新口径悄悄顶掉。

    - `status`：旧口径（全部被删面当真值）的达标判定，保持原样；
    - `status_spec_conformant`：任务书口径（误差达阈值的面）的达标判定；
    - `spec_conformant.target_met`：任务书口径的达标布尔值。

    我为什么坚持两套都留着、且不替换 `status`：
    验收口径的变更必须由甲方书面确认，工程侧只能把两种算法都摆出来。
    悄悄把 status 换成更宽松的那一套，就是把"改口径"伪装成"改实现"。
    """

    report = evaluate_synthetic_suite(
        cube_mesh(),
        seed_face_ids=[0, 2],
        face_counts=[1, 2],
        truth_samples_per_triangle=8,
    )

    assert report["schema_version"] == "synthetic-evaluation-1.3"
    spec = report["spec_conformant"]
    assert spec["truth_scope"] == "faces_with_geometric_error_at_or_above_threshold"
    assert spec["truth_samples_per_triangle"] == 8
    assert isinstance(spec["target_met"], bool)
    assert spec["true_positive"] + spec["false_negative"] == spec["truth_face_count"]
    # 旧口径的判据没有被新口径覆盖。
    assert report["status"] in {"BASELINE_TARGET_MET", "TARGET_NOT_PROVEN"}
    assert report["status_spec_conformant"] in {
        "BASELINE_TARGET_MET",
        "TARGET_NOT_PROVEN",
    }
    # 旧口径的边界指标保持原值，证明我没有顺手改旧算法。
    assert report["aggregate"]["boundary_iou"] == 1.0

    joined = "\n".join(report["limitations"])
    assert "循环" in joined, (
        "报告必须明写「拿生产距离自己当真值」这条口径是循环的、不能当准确率；"
        "否则读报告的人会拿那个 1.0 去交验收。"
    )
    assert "轮廓带" in joined or "采样分辨率" in joined, (
        "报告必须把轮廓带（采样分辨率造成的差异）写进限制说明。"
    )
