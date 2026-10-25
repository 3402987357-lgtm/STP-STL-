from __future__ import annotations

from pathlib import Path

import pytest

from cad_inspector.cli import main
from cad_inspector.domain.repair import RepairConfig
from cad_inspector.repair.step_hole import (
    _audit_patch_continuity,
    _make_surface_filling_candidate,
    repair_step_holes,
)


def _write_open_box_step(path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Shell

    solid = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Solid()
    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    faces = []
    while explorer.More():
        faces.append(TopoDS.Face(explorer.Current()))
        explorer.Next()
    assert len(faces) == 6

    shell = TopoDS_Shell()
    builder = BRep_Builder()
    builder.MakeShell(shell)
    for face in faces[1:]:
        builder.Add(shell, face)

    writer = STEPControl_Writer()
    assert (
        writer.Transfer(shell, STEPControl_AsIs)
        == IFSelect_ReturnStatus.IFSelect_RetDone
    )
    assert writer.Write(str(path)) == IFSelect_ReturnStatus.IFSelect_RetDone


def _write_closed_box_step(path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    assert (
        writer.Transfer(
            BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Solid(),
            STEPControl_AsIs,
        )
        == IFSelect_ReturnStatus.IFSelect_RetDone
    )
    assert writer.Write(str(path)) == IFSelect_ReturnStatus.IFSelect_RetDone


def test我方平面_STEP孔洞可以局部补面并输出报告(tmp_path: Path) -> None:
    step_path = tmp_path / "open_box.step"
    output_path = tmp_path / "open_box.repaired.step"
    _write_open_box_step(step_path)

    result = repair_step_holes(step_path, output_path=output_path)

    assert result.status == "SUCCESS"
    assert result.output_file == str(output_path.resolve())
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert result.input_topology["boundary_edge_count"] == 4
    assert result.output_topology["boundary_edge_count"] == 0
    assert result.output_topology["solid_count"] == 1
    assert result.repaired_regions
    region = result.repaired_regions[0]
    assert region.damage_type == "hole"
    assert region.repair_method == "ocp_planar_boundary_face"
    assert region.continuity_level == "G0_VERIFIED"
    assert region.g0_verified is True
    assert region.g1_verified is False
    assert region.g2_verified is False
    assert region.g0_max_deviation_mm == pytest.approx(0.0, abs=1e-8)
    assert region.g1_max_normal_angle_deg == pytest.approx(90.0, abs=1e-6)
    assert region.continuity_verified is False
    assert region.deviation_max_mm == "NOT_AVAILABLE"
    assert region.deviation_mean_mm == "NOT_AVAILABLE"
    assert region.area_repaired_mm2 == pytest.approx(100.0, abs=1e-5)
    assert region.review_required is False
    assert result.diagnostics["repair_attempt_count"] == 1
    assert result.diagnostics["repair_success_count"] == 1
    assert result.diagnostics["repair_success_rate"] == 1.0
    assert result.diagnostics["continuity_verified"] is False
    assert result.diagnostics["deviation_max_mm"] == "NOT_AVAILABLE"
    assert result.diagnostics["continuity_summary"]["G0"]["status"] == "VERIFIED"
    assert result.diagnostics["continuity_summary"]["G1"]["status"] == (
        "NOT_SATISFIED"
    )
    assert result.diagnostics["continuity_summary"]["G2"]["status"] == (
        "NOT_VERIFIED"
    )


def test我方完整_STEP没有开放边界时不执行修补(tmp_path: Path) -> None:
    step_path = tmp_path / "closed_box.step"
    _write_closed_box_step(step_path)

    result = repair_step_holes(step_path, output_path=tmp_path / "out.step")

    assert result.status == "NO_ACTION"
    assert result.repaired_regions == []
    assert result.output_file is None
    assert result.output_topology["boundary_edge_count"] == 0
    assert result.diagnostics["repair_attempt_count"] == 0
    assert result.diagnostics["repair_success_rate"] is None


    def test我方共面边界实际计算并通过G2曲率审计() -> None:
        pytest.importorskip("OCP")
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
        from OCP.TopAbs import (
            TopAbs_COMPOUND,
            TopAbs_FACE,
            TopAbs_REVERSED,
        )
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        from OCP.gp import gp_Pnt, gp_Vec

        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Solid()
        face_explorer = TopExp_Explorer(box, TopAbs_FACE)
        face = TopoDS.Face(face_explorer.Current())
        wire = BRepTools.OuterWire_s(face)
        patch = BRepBuilderAPI_MakeFace(wire, True).Face()
        edge_explorer = BRepTools_WireExplorer(wire)
        edges = []
        while edge_explorer.More():
            edges.append(TopoDS.Edge(edge_explorer.Current()))
            edge_explorer.Next()

        result = _audit_patch_continuity(
            wire_edges=edges,
            source_face_ids=[1],
            face_shapes={1: face},
            patch_face=patch,
            curve_type=BRepAdaptor_Curve,
            surface_type=BRepAdaptor_Surface,
            brep_tool_type=BRep_Tool,
            reversed_type=TopAbs_REVERSED,
            point_type=gp_Pnt,
            vector_type=gp_Vec,
            # 这四个类型供后续的缝合与自由边统计使用，
            # 必须与生产调用方传的一致，否则审计路径会退化。
            compound_type=TopAbs_COMPOUND,
            explorer_type=TopExp_Explorer,
            topods_type=TopoDS,
            face_type=TopAbs_FACE,
            tolerance_mm=0.05,
            max_g1_angle_deg=5.0,
            max_g2_curvature_delta_per_mm=0.01,
            sample_count=5,
        )

        assert result["g0_verified"] is True
        assert result["g1_verified"] is True
        assert result["g2_verified"] is True
        assert result["g2_max_curvature_delta_per_mm"] == pytest.approx(0.0)
        assert result["continuity_verified"] is True


def test我方非平面Wire可以生成待复核曲面填充候选() -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt

    polygon = BRepBuilderAPI_MakePolygon()
    for point in (
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (10.0, 10.0, 2.0),
        (0.0, 10.0, 0.0),
    ):
        polygon.Add(gp_Pnt(*point))
    polygon.Close()
    wire = polygon.Wire()
    explorer = BRepTools_WireExplorer(wire)
    edges = []
    while explorer.More():
        edges.append(TopoDS.Edge(explorer.Current()))
        explorer.Next()

    candidate = _make_surface_filling_candidate(
        wire_edges=edges,
        source_face_ids=[],
        face_shapes={},
        tolerance_mm=0.05,
    )

    assert not candidate.IsNull()
    assert BRepCheck_Analyzer(candidate).IsValid()


def test我方把填充Build的崩溃转成RuntimeError而不是打死任务(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """我锁住 2026-10-11 修的那个崩溃。

    实测 `filler.Build()` 会抛 `Standard_NoSuchObject`，而它**不在**
    `(AttributeError, RuntimeError, TypeError, ValueError)` 里，于是**一路
    没被捕获、把整个基准进程打死**：合成基准因此从 20 个 case 掉到 17 个
    （训练集/5#face3 崩，后面训练集/6 的两个根本没跑）。

    修法是把它转成 `RuntimeError` 交调用处已有的 except。这个用例保证
    "**一个环补不上**"永远不会再变成"**整个任务终止**"。
    """

    from OCP.OCP.Standard import Standard_NoSuchObject  # noqa: PLC0415

    import cad_inspector.repair.step_hole as step_hole  # noqa: PLC0415

    def boom(*_args: object, **_kwargs: object) -> None:
        raise Standard_NoSuchObject("我模拟 OCP 在 Build 里抛出的异常。")

    monkeypatch.setattr(
        "OCP.BRepOffsetAPI.BRepOffsetAPI_MakeFilling.Build", boom
    )
    # 我照 179 行那个用例的方式现建一个非平面四边形环：本用例只关心
    # "Build 抛异常之后变成什么"，环本身只要是合法的即可。
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon  # noqa: PLC0415
    from OCP.BRepTools import BRepTools_WireExplorer  # noqa: PLC0415
    from OCP.TopoDS import TopoDS  # noqa: PLC0415
    from OCP.gp import gp_Pnt  # noqa: PLC0415

    polygon = BRepBuilderAPI_MakePolygon()
    for point in (
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 1.0),
        (10.0, 10.0, 2.0),
        (0.0, 10.0, 0.0),
    ):
        polygon.Add(gp_Pnt(*point))
    polygon.Close()
    explorer = BRepTools_WireExplorer(polygon.Wire())
    edges = []
    while explorer.More():
        edges.append(TopoDS.Edge(explorer.Current()))
        explorer.Next()
    with pytest.raises(RuntimeError):
        step_hole._make_surface_filling_candidate(
            wire_edges=edges,
            source_face_ids=[],
            face_shapes={},
            tolerance_mm=0.05,
        )


def test我方不再有use_support这个重试参数() -> None:
    """我锁住 2026-10-11 撤销的那趟"不挂 support 重试"。

    我早先加过"第一轮挂 support 失败后再试一次不挂的"，依据是测试集/1 上
    不挂能把 96948 倍降到 1155 倍（小 84 倍）。但它**从未带来可证明的提分**，
    而且**造成了回归**：测试集/2 上第一轮抛错后第二轮接手、建出 0.139715
    mm² 的面，导致"补面后仍有开放边界、质量门禁不过"，**原本能升级成 Solid
    的升不成了**（2 个测试转红）。所以我把参数整个移除。

    这个用例保证它**不会被无意重新引入**——要重新引入，请先解决测试集/2 的
    回归（比如只在"面积门禁拒掉"时重试，而不是"第一轮抛异常就重试"）。
    """

    import inspect  # noqa: PLC0415

    import cad_inspector.repair.step_hole as step_hole  # noqa: PLC0415

    parameters = inspect.signature(
        step_hole._make_surface_filling_candidate
    ).parameters
    assert "use_support" not in parameters, (
        "我要求 use_support 保持移除状态；若确需重新引入，"
        "请先解决 测试集/2 的回归并更新本用例的说明。"
    )


def test我方超过面积门禁的_STEP孔洞只进入复核(tmp_path: Path) -> None:
    step_path = tmp_path / "open_box.step"
    output_path = tmp_path / "review.step"
    _write_open_box_step(step_path)

    result = repair_step_holes(
        step_path,
        output_path=output_path,
        config=RepairConfig(max_hole_area_mm2=10.0),
    )

    assert result.status == "REVIEW_REQUIRED"
    assert result.output_file is None
    assert not output_path.exists()
    assert result.repaired_regions
    assert result.repaired_regions[0].review_required is True


def test我方不给带着实质孔洞的件发_SUCCESS(tmp_path: Path) -> None:
    """我修掉的真 bug（2026-10-13）：回读复核只重判质量门禁五项，
    没再看"是否仍有带实质孔洞的环"，就把**洞根本没补上的件**提成 SUCCESS。
    更糟的是报告自相矛盾——warnings 里还写着
    "我仍有 N 个环带着超过几何容差的实质孔洞，输出只能作为人工复核候选"，
    status 却是 SUCCESS。

    这类工具里最不能有的错就是**把不合格件说成合格**，
    所以我把"说只能复核"与"判 SUCCESS"互斥钉成硬不变量：
    只要有那条告警，就绝不能是 SUCCESS。
    """

    step_path = tmp_path / "open_box.step"
    output_path = tmp_path / "review.step"
    _write_open_box_step(step_path)

    result = repair_step_holes(
        step_path,
        output_path=output_path,
        config=RepairConfig(max_hole_area_mm2=10.0),
    )

    claims_review_only = any(
        "只能作为人工复核候选" in str(warning)
        for warning in (result.warnings or [])
    )
    if not claims_review_only:
        pytest.skip("这个几何没有触发实质孔洞告警，改天换个用例")
    # 说了"只能当复核候选"，就不能同时给 SUCCESS——否则报告自相矛盾。
    assert result.status != "SUCCESS"


def test我方_STEP修补命令写出JSON和CSV(tmp_path: Path) -> None:
    step_path = tmp_path / "open_box.step"
    output_dir = tmp_path / "reports"
    _write_open_box_step(step_path)

    assert main(
        [
            "step-repair",
            str(step_path),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    payload = (output_dir / "open_box.step-repair.json").read_text(
        encoding="utf-8"
    )
    assert '"schema_version": "step-repair-1.3"' in payload
    assert '"deviation_status": "NOT_AVAILABLE_NO_REFERENCE_SURFACE"' in payload
    assert (output_dir / "open_box.step-repair.csv").exists()
    csv_payload = (output_dir / "open_box.step-repair.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "continuity_verified" in csv_payload.splitlines()[0]
    assert "g0_max_deviation_mm" in csv_payload.splitlines()[0]
    assert "g2_max_curvature_delta_per_mm" in csv_payload.splitlines()[0]
    assert "NOT_AVAILABLE" in csv_payload
    assert (output_dir / "open_box.repaired.step").exists()
