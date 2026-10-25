from __future__ import annotations

from pathlib import Path

import pytest

from cad_inspector.detection.step_parting import extract_step_parting_geometry
from cad_inspector.domain.parting import PartingConfig, PartingPipelineResult
from cad_inspector.errors import GeometryProcessingError
from cad_inspector.output.cad_parting import (
    _atomic_write_cad,
    write_step_parting_cad_outputs,
)
from cad_inspector.output.parting_report import write_parting_result


def test我方cad写出失败时不留下半个正式文件(tmp_path: Path) -> None:
    target = tmp_path / "box.parting_surface.step"
    target.write_bytes(b"OLD")

    def failing_write(temporary: Path) -> bool:
        temporary.write_bytes(b"PARTIAL")
        return False

    with pytest.raises(GeometryProcessingError):
        _atomic_write_cad(target, failing_write)

    assert target.read_bytes() == b"OLD"
    assert [item.name for item in tmp_path.iterdir() if item.suffix == ".tmp"] == []


def test我方cad写出成功后原子替换旧文件(tmp_path: Path) -> None:
    target = tmp_path / "box.parting_surface.step"

    def successful_write(temporary: Path) -> bool:
        temporary.write_bytes(b"NEW")
        return True

    assert _atomic_write_cad(target, successful_write) == str(target)
    assert target.read_bytes() == b"NEW"
    assert [item.name for item in tmp_path.iterdir() if item.suffix == ".tmp"] == []


def _write_box_step(path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    shape = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Solid()
    writer = STEPControl_Writer()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_ReturnStatus.IFSelect_RetDone
    assert writer.Write(str(path)) == IFSelect_ReturnStatus.IFSelect_RetDone


def test我方真实_ocp截交可以提取盒体分型线(tmp_path: Path) -> None:
    step_path = tmp_path / "box.step"
    _write_box_step(step_path)

    geometry = extract_step_parting_geometry(
        step_path,
        PartingConfig(plane_z_mm=5.0, plane_scan_count=1),
    )

    assert geometry.line.method == "ocp_brep_section"
    assert geometry.line.closed is True
    assert geometry.line.area_xy_mm2 == pytest.approx(100.0, abs=1e-6)
    assert geometry.face is not None
    assert geometry.shape_is_valid is True
    assert geometry.line.diagnostics["surface_face_is_valid"] is True
    assert geometry.line.diagnostics["stage6_status"] == "BASELINE"
    assert geometry.line.diagnostics["source_solid_count"] == 1
    assert geometry.line.diagnostics["surface_boundary_is_closed"] is True
    assert geometry.line.diagnostics["surface_intersects_source_shape"] is True
    assert geometry.transition_surface is not None
    assert geometry.line.diagnostics["transition_surface_status"] == "CANDIDATE"
    assert geometry.line.diagnostics["transition_surface_shape_is_valid"] is True
    assert geometry.line.diagnostics["transition_surface_face_count"] > 0
    assert geometry.line.diagnostics["transition_surface_edge_count"] > 0
    assert geometry.line.diagnostics["transition_surface_bare_edge_count"] > 0
    assert geometry.line.diagnostics["sewing_status"] == (
        geometry.line.diagnostics["transition_surface_sewing_status"]
    )
    assert "基础缝合" in " ".join(
        geometry.line.diagnostics["stage6_limitations"]
    )
    assert (
        geometry.line.diagnostics["transition_surface_penetration_evidence"]
        == "SECTION_EDGES_PRESENT"
    )
    assert geometry.sewn_transition_surface is not None
    assert geometry.line.diagnostics["transition_surface_sewing_status"] == (
        "SEWN_OPEN_SHELL"
    )
    assert (
        geometry.line.diagnostics["transition_surface_sewing_shape_is_valid"]
        is True
    )
    assert (
        geometry.line.diagnostics["transition_surface_sewing_free_edge_count"]
        > 0
    )
    assert (
        geometry.line.diagnostics["surface_penetration_evidence"]
        == "SURFACE_SECTION_EDGES_PRESENT"
    )
    assert geometry.line.diagnostics["surface_face_method"] == "ocp_extended_planar_face"
    assert geometry.line.diagnostics["surface_margin_mm"] == pytest.approx(5.0)
    assert geometry.line.diagnostics["source_face_ids"]
    assert geometry.line.diagnostics["source_face_count"] >= 1
    assert geometry.line.diagnostics["section_edge_ids"]
    assert geometry.line.diagnostics["edge_source"] == "ocp_section_result"
    assert (
        geometry.line.diagnostics["topology_index_method"]
        == "hash_bucket_is_same"
    )
    assert len(geometry.line_candidates) == 1


def test我方真实_ocp分型结果可以写出并再次读取(tmp_path: Path) -> None:
    step_path = tmp_path / "box.step"
    _write_box_step(step_path)
    geometry = extract_step_parting_geometry(
        step_path,
        PartingConfig(plane_scan_count=3),
    )
    assert geometry.line_candidates

    artifacts = write_step_parting_cad_outputs(geometry, tmp_path / "out", "box")

    assert Path(artifacts["parting_line_iges"]).stat().st_size > 0
    surface_step = Path(artifacts["parting_surface_step"])
    assert surface_step.stat().st_size > 0
    transition_step = Path(artifacts["transition_surface_step"])
    assert transition_step.stat().st_size > 0
    assert Path(artifacts["transition_surface_iges"]).stat().st_size > 0
    assert (
        Path(artifacts["transition_surface_sewn_step"]).stat().st_size > 0
    )
    assert (
        Path(artifacts["transition_surface_sewn_iges"]).stat().st_size > 0
    )
    out_dir = tmp_path / "out"
    assert [item.name for item in out_dir.iterdir() if item.suffix == ".tmp"] == []

    report_artifacts = write_parting_result(
        PartingPipelineResult(
            status="BASELINE",
            input_file=str(step_path),
            backend="ocp_brep_section",
            line=geometry.line,
            surface=None,
            line_candidates=geometry.line_candidates,
        ),
        tmp_path / "report",
    )
    assert Path(report_artifacts["parting_candidates_json"]).stat().st_size > 0

    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    assert reader.ReadFile(str(surface_step)) == IFSelect_ReturnStatus.IFSelect_RetDone
    assert reader.TransferRoots() > 0
    assert not reader.OneShape().IsNull()
    transition_reader = STEPControl_Reader()
    assert (
        transition_reader.ReadFile(str(transition_step))
        == IFSelect_ReturnStatus.IFSelect_RetDone
    )
    assert transition_reader.TransferRoots() > 0
    assert not transition_reader.OneShape().IsNull()
    sewn_reader = STEPControl_Reader()
    assert (
        sewn_reader.ReadFile(
            str(Path(artifacts["transition_surface_sewn_step"]))
        )
        == IFSelect_ReturnStatus.IFSelect_RetDone
    )
    assert sewn_reader.TransferRoots() > 0
    assert not sewn_reader.OneShape().IsNull()


def test我方倾斜环的真实面积不随摆放角度坍塌() -> None:
    """我证明闭环面积用的是自身拟合平面，而不是 XY 投影。

    回归背景：修复前用 ``abs(_signed_area_xy(...))``。一个 10x10 的方形环
    竖直摆放时 XY 投影面积是 0，会把真正的最大外轮廓判成"面积为零的垃圾候选"。
    """

    from cad_inspector.detection.step_parting import _true_loop_area_mm2

    import numpy as np

    square = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 10.0], [0.0, 0.0, 10.0]],
        dtype=np.float64,
    )
    area, status = _true_loop_area_mm2(square, tolerance_mm=1e-6)
    assert status == "COMPUTED"
    assert area == pytest.approx(100.0, abs=1e-6)


def test我方非凸L形环得到真实围成面积() -> None:
    """我证明非凸 L 形环得到的是真实围成面积。

    回归背景：修复前的口径是 abs(_signed_area_xy(...))，它依赖环平面与 XY 平行。
    我把口径换成环自身拟合平面上的面积后，L 形这类非凸环也必须仍然算对
    （鞋带公式对简单非凸多边形是精确的，这里锁定这一点，防止后续有人改坏）。
    """

    from cad_inspector.detection.step_parting import _true_loop_area_mm2

    import numpy as np

    # L 形环：真实面积 = 7x3 + 3x4 = 33。我锁定的是"拟合平面上的真实面积"
    # 这个口径本身，它不受环的摆放角度和绕行方向影响。
    l_shape = np.array(
        [
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [7.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [3.0, 7.0, 0.0],
            [0.0, 7.0, 0.0],
        ],
        dtype=np.float64,
    )
    area, status = _true_loop_area_mm2(l_shape, tolerance_mm=1e-6)
    assert status == "COMPUTED"
    assert area == pytest.approx(33.0, abs=1e-6)


def test我方共线退化环不给出有效面积() -> None:
    """我证明近似共线的点列不会被当成一个有面积的环。"""

    from cad_inspector.detection.step_parting import _true_loop_area_mm2

    import numpy as np

    line = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    _, status = _true_loop_area_mm2(line, tolerance_mm=1e-6)
    assert status == "DEGENERATE_POINTS"


def test我方分型方向检测在主轴为Z时给出Z轴(tmp_path: Path) -> None:
    """盒体的主惯量轴与最短跨度方向一致，我应得到 Z 轴且置信度为 HIGH。"""

    from cad_inspector.detection.step_parting import _detect_parting_axis

    pytest.importorskip("OCP")
    step_path = tmp_path / "box.step"
    _write_box_step(step_path)
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    assert reader.ReadFile(str(step_path)) == IFSelect_ReturnStatus.IFSelect_RetDone
    reader.TransferRoots()
    shape = reader.OneShape()
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    bounds = (
        box.GetXMin(), box.GetYMin(), box.GetZMin(),
        box.GetXMax(), box.GetYMax(), box.GetZMax(),
    )
    axis, method, diagnostics = _detect_parting_axis(
        shape, bounds, PartingConfig()
    )
    # 立方体三轴惯量相同，主惯量轴本身不唯一，所以我不在这里断言 HIGH 置信度；
    # 我只要求方向与 Z 平行且方法合法。置信度语义由长方体用例单独锁定。
    assert abs(abs(axis[2]) - 1.0) < 1e-6
    assert method in {"PRINCIPAL_INERTIA_AXIS", "FALLBACK_MIN_BBOX_SPAN"}
    assert diagnostics["parting_axis_method"] == method
    assert diagnostics["parting_axis_confidence"] in {"HIGH", "MEDIUM", "LOW"}


def test我方扁平长方体分型方向取最短跨度且置信度为HIGH() -> None:
    """我证明模具那种扁平件能拿到高置信度的开模方向。

    回归背景：开模方向通常是厚度最小的方向。扁平长方体在 Z 上最薄，
    主惯量轴与最短跨度方向应当一致，此时我给出的置信度为 HIGH。
    """

    from cad_inspector.detection.step_parting import _detect_parting_axis

    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    shape = BRepPrimAPI_MakeBox(100.0, 80.0, 10.0).Solid()
    # 包围盒与实体一致：X 100、Y 80、Z 10，最短跨度是 Z。
    bounds = (0.0, 0.0, 0.0, 100.0, 80.0, 10.0)
    axis, method, diagnostics = _detect_parting_axis(
        shape, bounds, PartingConfig()
    )
    assert method == "PRINCIPAL_INERTIA_AXIS"
    assert abs(abs(axis[2]) - 1.0) < 1e-6
    assert diagnostics["parting_axis_confidence"] == "HIGH"


def test我方截面平面范围按分型方向投影而不是固定XY() -> None:
    """我证明倾斜分型方向下，截面矩形仍能覆盖整个包围盒。

    回归背景：修复前 ``_build_section`` 固定用 XY 包围盒建面。分型方向不是 Z 时，
    XY 矩形投影到倾斜平面上会露出零件，截出的轮廓被平面边界截断。
    """

    from cad_inspector.detection.step_parting import (
        _plane_extent,
        _plane_frame,
    )

    import numpy as np

    bounds = (0.0, 0.0, 0.0, 10.0, 20.0, 30.0)
    for axis in ([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.847, -0.027, 0.531]):
        u, v = _plane_frame(axis)
        # 基向量必须与法向正交，否则平面退化。
        normal = np.asarray(axis, dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        assert abs(float(np.dot(u, normal))) < 1e-9
        assert abs(float(np.dot(v, normal))) < 1e-9
        u_min, u_max, v_min, v_max = _plane_extent(bounds, u, v)
        # 外扩后的范围必须严格大于包围盒在该方向的投影跨度。
        corners = np.asarray(
            [
                [x, y, z]
                for x in (bounds[0], bounds[3])
                for y in (bounds[1], bounds[4])
                for z in (bounds[2], bounds[5])
            ],
            dtype=np.float64,
        )
        assert u_min < float((corners @ u).min())
        assert u_max > float((corners @ u).max())
        assert v_min < float((corners @ v).min())
        assert v_max > float((corners @ v).max())
