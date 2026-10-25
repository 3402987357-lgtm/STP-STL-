"""我给任务书里那两条**细节要求**写回归。

任务书细节要求原文有两条：

1. 缺失区域判定阈值默认为几何误差≥0.15mm，并提供阈值调参接口；
2. 识别结果应包含缺失区域的位置、范围及对应误差数据。

两条都是明确验收项。我用 rg 查过整个 tests/：`--threshold-mm` 这个开关
**一次都没被测过**（它只出现在 run_real_pair_acceptance.py 与
run_soak_stability.py 两个脚本里），主 result.json 的位置/范围/误差
三件套也没有一条用例直接断言过。没有测试意味着两种坏情况都可能是真的：
开关没接上管线（传了没用），或者字段漏了没人发现。这个文件把两条钉住。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cad_inspector.cli import main
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.geometry.stl import save_stl


def _cube(omit_top: bool = False) -> TriangleMesh:
    """我造一个 10mm 立方体；omit_top=True 时挖掉顶面。

    实测这条缺料的**最大几何误差是 3.33mm**，不是 10mm：默认
    samples_per_triangle=1 只取每个三角面的重心，顶面重心到候选网格
    最近点落在缺口的边界上，距离是 10/3。用例里的阈值必须按这个
    真实误差选，按"看起来有 10mm 深"去选会两个阈值都滤掉、误判成开关失灵。
    """

    vertices = np.asarray(
        [
            [0, 0, 0],
            [10, 0, 0],
            [10, 10, 0],
            [0, 10, 0],
            [0, 0, 10],
            [10, 0, 10],
            [10, 10, 10],
            [0, 10, 10],
        ],
        dtype=np.float64,
    )
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    if omit_top:
        faces = faces[:2] + faces[4:]
    return TriangleMesh(vertices, np.asarray(faces, dtype=np.int64))


def _run_inspect(tmp_path: Path, threshold_mm: float) -> dict:
    reference = tmp_path / "reference.stl"
    candidate = tmp_path / "candidate.stl"
    save_stl(_cube(), reference)
    save_stl(_cube(omit_top=True), candidate)
    output_dir = tmp_path / f"out_{threshold_mm}"

    # 关掉配准：这个用例测的是阈值开关，不该被质心对齐的残差干扰。
    main(
        [
            "inspect",
            str(candidate),
            str(reference),
            "--output-dir",
            str(output_dir),
            "--threshold-mm",
            str(threshold_mm),
            "--registration-mode",
            "none",
        ]
    )
    return json.loads(
        (output_dir / "result.json").read_text(encoding="utf-8")
    )


def test我方阈值开关真的改变检出结果(tmp_path: Path) -> None:
    """缺陷实测误差 3.33mm：阈值 2 应检出，阈值 5 应检不出。"""

    pytest.importorskip("numpy")

    detected = _run_inspect(tmp_path, 2.0)
    suppressed = _run_inspect(tmp_path, 5.0)

    assert len(detected.get("regions") or []) > 0, "阈值低于缺陷深度时必须报出区域"
    assert not (suppressed.get("regions") or []), "阈值高于缺陷深度时不该报区域"


def test我方把实际生效的阈值写进结果(tmp_path: Path) -> None:
    """事后要能看出这次跑用的是哪个阈值，否则"可调参"无法追溯。"""

    pytest.importorskip("numpy")

    payload = _run_inspect(tmp_path, 2.0)

    assert payload["threshold_mm"] == 2.0
    assert (payload.get("config") or {}).get("threshold_mm") == 2.0


def test我方阈值默认值是任务书要求的015(tmp_path: Path) -> None:
    """不传 --threshold-mm 时必须是 0.15mm。"""

    pytest.importorskip("numpy")

    from cad_inspector.models import InspectionConfig

    assert InspectionConfig().threshold_mm == 0.15

    reference = tmp_path / "reference.stl"
    candidate = tmp_path / "candidate.stl"
    save_stl(_cube(), reference)
    save_stl(_cube(omit_top=True), candidate)
    output_dir = tmp_path / "out_default"
    main(
        [
            "inspect",
            str(candidate),
            str(reference),
            "--output-dir",
            str(output_dir),
            "--registration-mode",
            "none",
        ]
    )
    payload = json.loads(
        (output_dir / "result.json").read_text(encoding="utf-8")
    )
    assert payload["threshold_mm"] == 0.15


def test我方结果里位置范围误差三件套齐全(tmp_path: Path) -> None:
    """任务书：识别结果应包含缺失区域的**位置、范围及对应误差数据**。

    我逐项钉住这三类，而不是只检查字段存在：

    - 位置：center_xyz_mm 必须是三个有限数；
    - 范围：bbox_min/max_xyz_mm 各三个有限数，且下界不得大于上界，
      area_mm2 是有限非负数；
    - 误差：max/mean/p95 三个误差量都是有限数，且满足
      mean <= p95 <= max 的基本大小关系——只查"是数字"的话，
      把三个字段写反了也照样通过。
    """

    pytest.importorskip("numpy")

    import math

    payload = _run_inspect(tmp_path, 2.0)
    regions = payload.get("regions") or []
    assert regions, "这个用例需要至少一个检出区域才能核对字段"

    for region in regions:
        center = region["center_xyz_mm"]
        assert len(center) == 3 and all(math.isfinite(v) for v in center)

        lower = region["bbox_min_xyz_mm"]
        upper = region["bbox_max_xyz_mm"]
        assert len(lower) == 3 and len(upper) == 3
        assert all(math.isfinite(v) for v in lower + upper)
        assert all(lo <= hi for lo, hi in zip(lower, upper))

        area = region["area_mm2"]
        assert math.isfinite(area) and area >= 0.0

        mean_error = region["mean_error_mm"]
        p95_error = region["p95_error_mm"]
        max_error = region["max_error_mm"]
        for value in (mean_error, p95_error, max_error):
            assert math.isfinite(value) and value >= 0.0
        assert mean_error <= p95_error <= max_error


def test我方按任务书的输入形态跑通STP候选加STL参考(tmp_path: Path) -> None:
    """任务书功能验收第一条：能正确完成 **STP 部件**与**参考 STL** 的几何比对。

    我为什么要专门钉这个组合（2026-10-16 查覆盖时发现）：
    仓库里调用 inspect 的用例用的都是 STL 对 STL；唯一拿真实 STEP 当输入的
    test_pipeline_units 用例，是 **STEP 当参考、STL 当候选**——正好和任务书
    相反。也就是说任务书写的那个输入形态（待检测 STP + 完整参考 STL）
    **没有任何一条端到端用例**。这个组合不是对称的：候选侧是 STEP 才会
    走 OCCT 三角化与 CAD 面号映射，参考侧是 STL 则缺料区域落在 STL 的
    三角面上，两条路径的编号口径完全不同，STL 对 STL 的用例证明不了它。

    这条跑的是甲方真实语料里最小的一对（训练集/8），实测约 17 秒。
    """

    pytest.importorskip("numpy")
    pytest.importorskip("OCP")

    from cad_inspector.geometry.stl import load_stl
    from cad_inspector.models import InspectionConfig
    from cad_inspector.pipeline import inspect_pair

    corpus = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "provided"
        / "三维模型破损区域智能修补"
        / "训练集"
    )
    candidate = corpus / "8.stp"
    reference = corpus / "8.stl"
    if not (candidate.is_file() and reference.is_file()):
        pytest.skip("我找不到这一对真实语料")

    result = inspect_pair(
        candidate,
        reference,
        tmp_path / "stp_vs_stl",
        InspectionConfig(registration_mode="none", distance_backend="auto"),
    )

    # 候选侧确实是 STEP：这条用例存在的意义就是走 STEP 那条路径。
    assert str(result.candidate_file).lower().endswith(".stp")

    regions = result.regions
    assert regions, "这一对是有缺料的件，跑完不该一个区域都没有"

    # 缺料区域落在**参考 STL** 上，面号必须是指向该网格三角面的合法下标。
    # 只断言"有区域"是不够的：编号口径错位（比如误用候选侧的面号）
    # 同样能产出非空结果，却把位置指到完全不相干的地方。
    reference_triangle_count = len(load_stl(reference).faces)
    missing = [r for r in regions if r.defect_type == "missing"]
    assert missing, "这一对报的是缺料，必须有 missing 区域"
    for region in missing:
        face_ids = list(region.reference_face_ids)
        assert face_ids, "缺料区域必须给出它覆盖了参考件的哪些面"
        assert min(face_ids) >= 0
        assert max(face_ids) < reference_triangle_count, (
            "缺料区域的面号必须落在参考 STL 的三角面范围内"
        )


def test我方全部真实件的两侧面号都不越界() -> None:
    """我把编号口径检查从 1 对扩到全部 18 对，并且两侧编号都查。

    为什么值得单独一条：面号越界会把区域位置指到完全不相干的地方，
    而任务书明确要求输出缺失区域的**位置**。跑一次真实件要十几秒，
    18 对重跑太贵，所以我改查**已存的正式结果**与它们各自的输入——
    结果目录被清掉时这条 skip，不让它变成假通过。

    我为什么两侧都查，而不是只查参考侧：两套编号是不同口径，错法也不同。
    参考侧是 STL 的**三角面下标**，越界只是内部数据错；候选侧是 STEP 的
    **CAD 面号**（1 起编号），那是工程师拿去在 CAD 里定位缺陷的编号，
    越界意味着交给甲方的定位编号根本不存在。所以三角面号和 CAD 面号
    要分别对各自的上界。
    """

    import sys

    root = Path(__file__).resolve().parents[1]
    results_root = root / "outputs" / "current_draft_source"
    if not results_root.is_dir():
        pytest.skip("正式结果目录不存在，跳过编号口径复核")

    sys.path.insert(0, str(root / "scripts"))
    try:
        from run_real_pair_acceptance import _count_stl_faces
    finally:
        sys.path.remove(str(root / "scripts"))

    checked = 0
    for result_path in sorted(results_root.glob("*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = payload.get("global_metrics") or {}
        reference = Path(payload.get("reference_file") or "")
        if not reference.is_file():
            # 2026-10-18：项目被整体移动过，旧 result.json 里记的是旧根的
            # 绝对路径。按 "data/" 之后的部分重新锚到当前根，否则一移动
            # 整个编号口径复核就会静默跳过（跳过比失败更危险）。
            parts = reference.parts
            if "data" in parts:
                reference = root.joinpath(*parts[parts.index("data"):])
        if not reference.is_file():
            pytest.skip(f"参考件不在本机：{reference}")
        face_count = _count_stl_faces(reference)
        if face_count is None:
            continue
        checked += 1

        # 先核对"记录下来的面数"和"文件里实际的面数"是否一致——
        # 若记录值本身是错的，后面拿它当上界就只是自证。
        assert metrics.get("reference_face_count") == face_count, (
            f"{result_path.parent.name} 记录的参考面数 {metrics.get('reference_face_count')} "
            f"与实际文件 {face_count} 不一致"
        )
        candidate_triangles = metrics.get("candidate_face_count")
        mapping = metrics.get("cad_face_mapping") or {}
        cad_face_count = mapping.get("cad_face_count")
        declared = (metrics.get("candidate_step_face_mapping") or {}).get(
            "declared_face_count"
        )
        if cad_face_count is not None and declared is not None:
            assert cad_face_count == declared, (
                f"{result_path.parent.name} 的 cad_face_count={cad_face_count} "
                f"与 STEP 映射声明的 {declared} 不一致"
            )

        for region in payload.get("regions") or []:
            rid = region.get("region_id")
            face_ids = list(region.get("reference_face_ids") or [])
            if face_ids:
                assert min(face_ids) >= 0, (result_path, rid)
                assert max(face_ids) < face_count, (
                    f"{result_path.parent.name} 区域 {rid} "
                    f"参考三角面号越界：max={max(face_ids)} 参考面数={face_count}"
                )

            candidate_ids = list(region.get("candidate_face_ids") or [])
            if candidate_ids and candidate_triangles:
                assert min(candidate_ids) >= 0, (result_path, rid)
                assert max(candidate_ids) < candidate_triangles, (
                    f"{result_path.parent.name} 区域 {rid} "
                    f"候选三角面号越界：max={max(candidate_ids)} / {candidate_triangles}"
                )

            # CAD 面号是 1 起编号，上界是 CAD 面总数。
            cad_ids = list(region.get("candidate_step_face_ids") or [])
            if cad_ids:
                assert cad_face_count is not None, (
                    f"{result_path.parent.name} 区域 {rid} 报了 CAD 面号却没有 cad_face_count"
                )
                assert min(cad_ids) >= 1, (result_path, rid)
                assert max(cad_ids) <= cad_face_count, (
                    f"{result_path.parent.name} 区域 {rid} "
                    f"CAD 面号越界：max={max(cad_ids)} / 1..{cad_face_count}"
                )

    assert checked >= 18, f"我只查到 {checked} 组，正式结果应当有 18 组"


def test我方全部源码模块都用第一人称撰写() -> None:
    """任务要求"用第一人称方式来去撰写所有代码"。

    我只检查**模块文档字符串**，不逐行扫注释：模块文档是一个文件的开场白，
    最能代表这份代码是谁在什么立场上写的；逐行做字符串匹配只会把用例写成
    匹配游戏，换个措辞就误报，反而没有约束力。

    我为什么要专门钉这条：这是任务里唯一一条**纯风格**要求，而风格最容易
    被后来的自动生成代码冲掉——新加的模块若写成第三人称说明书，
    整套代码的语气就断了，但没有任何测试会拦。
    """

    import ast

    root = Path(__file__).resolve().parents[1]
    checked = 0
    offenders: list[str] = []
    for base_name in ("src", "scripts"):
        base = root / base_name
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            checked += 1
            doc = ast.get_docstring(
                ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            ) or ""
            if "我" not in doc:
                offenders.append(str(path.relative_to(root)))

    assert offenders == [], f"这些模块的文档不是第一人称：{offenders}"
    # 防止遍历意外变空：源码与脚本合计有一百多个模块。
    assert checked >= 100, f"我只查到 {checked} 个模块"
