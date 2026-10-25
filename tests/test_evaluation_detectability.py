"""我给"合成案例是否物理可检出"这个判据上回归。

起因是一个真实的误读风险：在精细三角化的小件上（实测 训练集/8，
4.8x5.9x2.5mm 却切了 19228 个面，平均边长 0.079mm），删掉 60 个面
只造成 0.1355mm 的凹陷，低于 0.15mm 阈值。此时"没检出"是**物理必然**，
不是检测器漏检。若把它算进召回率，评估结论就会把"我造的缺陷太小"
错记成"检测器不行"。

所以我在评估里加了 max_achievable_depth_mm / detectable 两个字段，
并把不可检出的案例从准确率统计中排除。这一组用例钉住这个语义。
"""

from __future__ import annotations

from pathlib import Path

# 我为什么按仓库根定位：这里原本写相对路径 Path("data")，
# 只在"当前工作目录恰好是仓库根"时才找得到，别处执行 pytest 会静默 skip。
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


import numpy as np
import pytest

from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.geometry.stl import load_stl
from cad_inspector.learning.evaluation import (
    _max_achievable_depth,
    evaluate_synthetic_case,
)
from cad_inspector.learning.synthetic_defects import remove_face_region
from cad_inspector.models import InspectionConfig


def _slab_mesh(n: int = 8) -> TriangleMesh:
    """我造一块平面板：两面各 n x n 个三角形，便于按面号定位。"""

    vertices: list[list[float]] = []
    for iy in range(n + 1):
        for ix in range(n + 1):
            vertices.append([float(ix), float(iy), 0.0])
    faces: list[list[int]] = []
    for iy in range(n):
        for ix in range(n):
            v = iy * (n + 1) + ix
            faces.append([v, v + 1, v + n + 1])
            faces.append([v + 1, v + n + 2, v + n + 1])
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    )


def test我给每个案例都量出可达深度() -> None:
    """真值非空时 max_achievable_depth_mm 必须是一个有限非负数。"""

    mesh = _slab_mesh()
    defect = remove_face_region(mesh, seed_face_id=0, face_count=2)
    config = InspectionConfig(registration_mode="none", unit_policy="assume_mm")

    metrics = evaluate_synthetic_case(
        mesh,
        defect.candidate_mesh,
        defect.removed_face_ids,
        config=config,
        defect_type="missing",
    )

    assert metrics.max_achievable_depth_mm is not None
    assert metrics.max_achievable_depth_mm >= 0.0
    assert np.isfinite(metrics.max_achievable_depth_mm)


def test我把深度不足阈值的案例标成不可检出() -> None:
    """我钉住"缺陷深度低于阈值"这一类的判定。

    同一块平面板上删 2 个面，实测可达深度 0.6667mm：
       - 阈值 0.15mm  -> 可检出，且两个真值面全部命中；
       - 阈值 5.0mm   -> 深度不足，必须标成不可检出。
    我同时断言"可检出时确实命中"，是为了证明这个判据
    和检测结果是一致的，而不是两套各说各话的口径。
    """

    mesh = _slab_mesh()
    defect = remove_face_region(mesh, seed_face_id=0, face_count=2)

    easy = InspectionConfig(
        registration_mode="none", unit_policy="assume_mm", threshold_mm=0.15
    )
    easy_metrics = evaluate_synthetic_case(
        mesh,
        defect.candidate_mesh,
        defect.removed_face_ids,
        config=easy,
        defect_type="missing",
    )
    assert easy_metrics.max_achievable_depth_mm is not None
    assert easy_metrics.detectable is True
    # 判据与结果必须自洽：说能检出，就应当真的检出。
    assert easy_metrics.recall == 1.0

    hard = InspectionConfig(
        registration_mode="none", unit_policy="assume_mm", threshold_mm=5.0
    )
    hard_metrics = evaluate_synthetic_case(
        mesh,
        defect.candidate_mesh,
        defect.removed_face_ids,
        config=hard,
        defect_type="missing",
    )
    # 深度是纯几何量，不因阈值而改变；改变的是"够不够"。
    assert hard_metrics.max_achievable_depth_mm == pytest.approx(
        easy_metrics.max_achievable_depth_mm
    )
    assert hard_metrics.detectable is False


def test真值为空时我不谎报深度() -> None:
    """真值空集时返回 None，而不是 0.0——0.0 会被读成"深度为零"。"""

    mesh = _slab_mesh()
    config = InspectionConfig(registration_mode="none", unit_policy="assume_mm")

    depth = _max_achievable_depth(
        reference=mesh,
        candidate=mesh,
        ground_truth=set(),
        config=config,
        defect_type="missing",
    )

    assert depth is None


def test可达深度随阈值不变但检出判定随之改变() -> None:
    """同一个几何缺陷，阈值抬高会把它从可检出变成不可检出。

    这条用例锁的是"detectable 依赖阈值"这一点：如果哪天有人把
    判据写死成常量，这条会立刻红。
    """

    mesh = _slab_mesh()
    defect = remove_face_region(mesh, seed_face_id=0, face_count=2)
    depth = None
    for threshold in (1e-9, 1e9):
        config = InspectionConfig(
            registration_mode="none",
            unit_policy="assume_mm",
            threshold_mm=threshold,
        )
        metrics = evaluate_synthetic_case(
            mesh,
            defect.candidate_mesh,
            defect.removed_face_ids,
            config=config,
            defect_type="missing",
        )
        if depth is None:
            depth = metrics.max_achievable_depth_mm
        # 深度是纯几何量，不随阈值变化。
        assert metrics.max_achievable_depth_mm == pytest.approx(depth or 0.0)
        assert metrics.detectable == ((depth or 0.0) >= threshold)


def test浅碗效应让面级召回收敛在086而不是10() -> None:
    """我在 README 里写了"0.99 这个目标在面级召回口径下不可能达到"，
    依据是实测召回只收敛到约 0.86。这条把那个结论钉住——
    它是"必须与甲方重新约定命中口径"的依据，
    如果哪天代码改了、收敛值变了，这条会红，README 的结论也就得改。

    我钉的是**方向**而不是精确值：召回必须随删除面数增大而上升，
    但最终明显低于 1.0，说明存在结构性上限而不是随机波动。
    """

    path = DATA_ROOT / "provided" / "三维模型破损区域智能修补" / "训练集" / "8.stl"
    if not path.exists():
        pytest.skip("我找不到真实语料")

    mesh = load_stl(path)
    config = InspectionConfig(
        registration_mode="none", unit_policy="assume_mm"
    )
    recalls = []
    for count in (400, 1500, 5000):
        defect = remove_face_region(mesh, seed_face_id=0, face_count=count)
        metrics = evaluate_synthetic_case(
            mesh,
            defect.candidate_mesh,
            defect.removed_face_ids,
            config=config,
            defect_type="missing",
        )
        recalls.append(metrics.recall)
        # 零误报是我目前最硬的一条正面证据，必须始终成立。
        assert metrics.precision == pytest.approx(1.0)

    # 单调上升：删除面越多，碗越深，召回越高。
    assert recalls[0] < recalls[1] < recalls[2]
    # 但存在结构性上限：即便删到 5000 面也远达不到 1.0。
    assert recalls[-1] < 0.95, f"召回收敛到 {recalls[-1]}，README 结论需重估"


def test我方按任务书阈值口径的召回是满值_086是分母造成的() -> None:
    """我钉住"0.86 是分母口径造成的，不是检测器漏检"。

    上一条用例证明面级召回上限约 0.86，README 据此写了
    "0.99 在面级召回口径下不可能达到"。这个结论**只对"拿全部被删面
    当分母"成立**，容易被读成"检测器找不到缺陷"，那是误读。

    真实原因是：删除块的外圈紧贴破损件边界，误差天然低于 0.15mm，
    按任务书定义（缺失区域 = 误差 >= 0.15mm 的区域）它们本来就
    不属于缺陷。把分母换成"误差确实达阈值的那些面"之后，实测
    检测器**一个不漏、一个不多**——预测集合与达阈值集合完全相等。

    我同时钉住那个真正卡住满分的东西：默认 min_region_area_mm2=0.5
    会把一块 0.38mm2 的真实缺料**整块丢掉**（seed=1200, faces=400），
    于是那一例召回直接掉到 0。这条用例把两个口径的区别钉死，
    避免把分母问题当成能力问题、也避免有人以为"调小门限就万事大吉"
    （调小门限在真实件上会不会放进噪声，本用例不覆盖）。
    """

    path = DATA_ROOT / "provided" / "三维模型破损区域智能修补" / "训练集" / "8.stl"
    if not path.exists():
        pytest.skip("我找不到真实语料")

    from cad_inspector.detection.missing_regions import detect_missing_regions
    from cad_inspector.learning.evaluation import _identity_registration

    mesh = load_stl(path)
    registration = _identity_registration()

    def _predict(candidate, floor: float) -> tuple[set[int], np.ndarray]:
        config = InspectionConfig(
            registration_mode="none",
            unit_policy="assume_mm",
            min_region_area_mm2=floor,
        )
        regions, _metrics, _warnings, distances = detect_missing_regions(
            reference=mesh,
            candidate=candidate,
            config=config,
            registration=registration,
        )
        predicted = {
            int(face_id)
            for region in regions
            if region.defect_type == "missing"
            for face_id in region.reference_face_ids
        }
        return predicted, np.asarray(distances)

    # 默认门限下这一例整块被丢：检出面积是有的，只是低于 0.5mm2。
    small = remove_face_region(mesh, seed_face_id=1200, face_count=400)
    assert _predict(small.candidate_mesh, 0.5)[0] == set(), (
        "默认最小面积门限本该把这块 0.38mm2 的真实缺料整块丢掉；"
        "若这里不再是空集，说明门限或面积算法变了，请重新实测。"
    )

    for seed, count in ((0, 400), (0, 5000), (1200, 400), (1200, 5000)):
        defect = remove_face_region(mesh, seed_face_id=seed, face_count=count)
        predicted, distances = _predict(defect.candidate_mesh, 0.3)
        ground_truth = {int(face_id) for face_id in defect.removed_face_ids}
        detectable = {f for f in ground_truth if distances[f] >= 0.15}

        assert detectable, f"seed={seed} faces={count} 本该有达阈值的面"
        # 达阈值的面一个不漏。
        assert not (detectable - predicted), (
            f"seed={seed} faces={count} 漏了 {len(detectable - predicted)} 个"
            "误差已达阈值的面——这是真漏检，不是分母问题。"
        )
        # 也不多报低于阈值的面。
        assert not (predicted - detectable), (
            f"seed={seed} faces={count} 多报了 {len(predicted - detectable)} 个"
            "误差低于阈值的面——阈值口径下的假阳性。"
        )
