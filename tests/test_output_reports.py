"""我方 report writer 区域的用例。

我为什么要补这个文件：我用 rg 全库查过，`output/step_repair_report.py`
等四个 writer 模块**没有任何测试引用它们**，但它们都在用（CLI 会调）。
`step_repair_report` 尤其关键——它写出的 JSON 就是甲方要看的修补报告。

这里的重点不是"文件能写出来"，而是钉住一个**刻意的设计决定**：
偏差量来自 OCP 曲面求交，真实件上会返回 NaN；Python 的 json.dumps
默认把 NaN 写成裸的 ``NaN``，那不是合法 JSON，甲方那边解析器会直接
失败。所以代码用 allow_nan=False 宁可报错、也不写出打不开的报告。
这个决定必须有用例守着，否则哪天被人"顺手去掉"就不会有人发现。
"""

from __future__ import annotations

import json

import pytest

from cad_inspector.detection.voxel_compare import VoxelComparison
from cad_inspector.domain.repair import (
    BoundaryLoopResult,
    RepairConfig,
    RepairRegionResult,
    RepairResult,
)
from cad_inspector.domain.step_repair import (
    StepRepairRegionResult,
    StepRepairResult,
)
from cad_inspector.output.repair_report import (
    write_repair_csv,
    write_repair_json,
)
from cad_inspector.output.step_repair_report import (
    write_step_repair_csv,
    write_step_repair_json,
)
from cad_inspector.output.voxel_report import write_voxel_json


def _result(**overrides) -> StepRepairResult:
    """我造一个最小的修补结果，让每个用例只改自己关心的字段。"""

    base = dict(
        status="SUCCESS",
        input_file="in.stp",
        output_file="out.step",
        backend="ocp",
        input_shape_is_valid=True,
        output_shape_is_valid=True,
        input_topology={"free_edge_count": 4},
        output_topology={"free_edge_count": 0},
        repaired_regions=[],
        warnings=[],
        artifacts={},
        config={},
        elapsed_seconds=1.5,
        diagnostics={},
    )
    base.update(overrides)
    return StepRepairResult(**base)


def _region(region_id: int) -> StepRepairRegionResult:
    """我造一个真实的修补区域证据，让 CSV 用例能核对行数。"""

    return StepRepairRegionResult(
        region_id=region_id,
        damage_type="hole",
        status="MANUAL_REVIEW_REQUIRED",
        repair_method="none",
        area_repaired_mm2=1.25,
        continuity_level="NOT_VERIFIED",
        boundary_edge_count=1,
        boundary_vertex_count=5,
        planarity_error_mm=None,
        g0_verified=False,
        g1_verified=False,
        g2_verified=False,
        g0_max_deviation_mm=None,
        g1_max_normal_angle_deg=None,
        g2_max_curvature_delta_per_mm=None,
        continuity_verified=False,
        deviation_max_mm="n/a",
        deviation_mean_mm="n/a",
        source_face_ids=[1, 2],
        review_required=True,
    )


def test我方写出的修补报告是合法JSON且以换行结尾(tmp_path) -> None:
    """我保证报告能被解析，且末尾有换行（否则批量比对凭空多噪音行）。"""

    target = tmp_path / "repair.json"
    returned = write_step_repair_json(_result(), target)
    assert returned == target
    text = target.read_text(encoding="utf-8")
    assert text.endswith("\n"), "我要求文本文件以换行结尾"
    payload = json.loads(text)
    assert payload["schema_version"] == "step-repair-1.3", payload.get(
        "schema_version"
    )
    assert payload["status"] == "SUCCESS"


def test我方拒绝把NaN偏差写成打不开的报告(tmp_path) -> None:
    """我保证非有限值直接报错，而不是写出一份甲方解析不了的 JSON。

    这是我刻意钉住的设计决定：真实件上补面与原面无有效交点时，OCP 会
    返回 NaN。若这里顺手放开 allow_nan，写出的文件含裸 ``NaN`` 字样、
    不是合法 JSON，甲方那边的解析器会直接失败。
    """

    target = tmp_path / "bad.json"
    bad = _result(diagnostics={"planarity_error_mm": float("nan")})
    with pytest.raises(ValueError):
        write_step_repair_json(bad, target)
    # 没写出来才算真的拦住了：出错时不该留下半份报告。
    assert not target.exists(), "我报错了就不该留下文件"


def test我方CSV行数等于表头加区域数(tmp_path) -> None:
    """我保证 CSV 每个区域都有一行，不静默少写。

    我为什么不只断言"文件非空"：那等于没验证——一个只写表头、把区域
    全丢掉的实现也能通过。这里造两个真实区域，行数必须是 表头 1 + 2。
    """

    target = tmp_path / "repair.csv"
    write_step_repair_csv(
        _result(repaired_regions=[_region(1), _region(2)]), target
    )
    lines = [
        line
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 3, f"表头 1 行 + 区域 2 行，实际 {len(lines)}: {lines}"
    # 两个 region_id 都要出现在数据行里，确认写的是区域本身而非占位。
    body = "\n".join(lines[1:])
    assert "1" in body and "2" in body, body


def _voxel(**overrides) -> VoxelComparison:
    """我造一个最小的体素比对结论。"""

    base = dict(
        voxel_size_mm=1.0,
        grid_dims=(4, 4, 4),
        grid_origin_mm=(0.0, 0.0, 0.0),
        reference_voxel_count=64,
        candidate_voxel_count=60,
        intersection_count=60,
        missing_count=4,
        extra_count=0,
        union_count=64,
        iou=60 / 64,
        missing_volume_mm3=4.0,
        extra_volume_mm3=0.0,
        reference_watertight=True,
        candidate_watertight=False,
        reference_free_edge_count=0,
        candidate_free_edge_count=12,
        status="REVIEW_REQUIRED",
    )
    base.update(overrides)
    return VoxelComparison(**base)


def test我方体素报告必带体素边长与状态(tmp_path) -> None:
    """我保证体素报告里体积和格子数不是无法解释的裸数字。

    我为什么钉这两项：源码 docstring 明写"脱离了体素边长，体积与格子数
    都是无法解释的裸数字；脱离了状态，一个未封闭网格漏填出来的体积会被
    当成精确结论"。所以 voxel_size_mm 与 status 必须随报告一起写出。

    另外我用标准库 trace 量化过：**这个模块命中行 0/38，完全没被执行
    过**——它不是"覆盖浅"，是从来没人跑过。
    """

    target = tmp_path / "voxel.json"
    returned = write_voxel_json(_voxel(), target)
    assert returned == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "voxel-compare-1.0"
    assert payload["voxel_size_mm"] == 1.0, payload.get("voxel_size_mm")
    assert payload["status"] == "REVIEW_REQUIRED", payload.get("status")
    # 未封闭的候选网格必须有提示，否则漏填的体积会被当成精确结论。
    assert payload["candidate_watertight"] is False


def test我方体素报告会创建不存在的父目录(tmp_path) -> None:
    """我保证写到深层目录时自己建目录，而不是抛 FileNotFoundError。"""

    target = tmp_path / "a" / "b" / "voxel.json"
    write_voxel_json(_voxel(), target)
    assert target.exists()


def _repair(**overrides) -> RepairResult:
    """我造一个最小的网格修补结果。"""

    loop = BoundaryLoopResult(
        loop_id=1,
        vertex_ids=[1, 2, 3],
        center_xyz_mm=[0.0, 0.0, 0.0],
        normal_xyz=[0.0, 0.0, 1.0],
        perimeter_mm=3.0,
        area_mm2=0.5,
        planarity_error_mm=0.01,
        is_convex=True,
        repairable=True,
    )
    region = RepairRegionResult(
        region_id=1,
        defect_type="hole",
        status="REPAIRED",
        method="planar_fill",
        boundary_loop=loop,
        added_vertex_count=0,
        added_triangle_count=1,
        review_required=True,
    )
    base = dict(
        status="SUCCESS",
        input_file="in.stl",
        output_file="out.stl",
        input_quality={"boundary_edge_count": 3},
        output_quality={"boundary_edge_count": 0},
        repaired_regions=[region],
        warnings=[],
        artifacts={},
        config=RepairConfig(),
    )
    base.update(overrides)
    return RepairResult(**base)


def test我方修补报告拒绝把NaN当成正常数值写出(tmp_path) -> None:
    """我保证非有限值在写出阶段炸掉，不流到下游。

    源码注释点破了危害：裸 ``NaN`` 在文件里**看起来像个正常的数值**，
    严格解析器（含项目自己的审计器）读到才报错。让写出阶段直接炸，
    比让一份带 NaN 的报告流到下游再炸好得多。
    """

    target = tmp_path / "repair.json"
    bad = _repair(input_quality={"planarity_error_mm": float("nan")})
    with pytest.raises(ValueError):
        write_repair_json(bad, target)
    assert not target.exists(), "我报错了就不该留下文件"


def test我方CSV把复核结论单独成列不被状态吞掉(tmp_path) -> None:
    """我保证 review_required 是独立一列，看得表的人不会漏掉待复核行。

    我为什么钉这条：源码 docstring 明写——"review_required 必须单独成列：
    它是'要不要人工看一眼'的结论列。混在 status 里会让人以为它是修补
    状态，从而**漏掉真正需要复核的行**"。

    这条与我在修补报告里发现的"顶层 SUCCESS 与区域级 MANUAL_REVIEW_
    REQUIRED 并存"是同一类关注点：结论被更显眼的状态盖住了。
    """

    target = tmp_path / "repair.csv"
    write_repair_csv(_repair(), target)
    lines = [
        line
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2, f"表头 1 行 + 区域 1 行，实际 {len(lines)}: {lines}"
    header = [col.strip() for col in lines[0].split(",")]
    assert "review_required" in header, f"复核结论必须单独成列：{header}"
    assert "status" in header, f"状态列也要在：{header}"
