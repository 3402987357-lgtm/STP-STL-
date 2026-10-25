from __future__ import annotations

from pathlib import Path

import pytest

from cad_inspector.detection.step_features import classify_step_face_features
from cad_inspector.io.step_faces import inspect_step_face_features


def _write_box_step(path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    shape = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Solid()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_ReturnStatus.IFSelect_RetDone
    assert writer.Write(str(path)) == IFSelect_ReturnStatus.IFSelect_RetDone


def _write_compound_step(path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopoDS import TopoDS_Compound

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Solid())
    builder.Add(compound, BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Solid())
    writer = STEPControl_Writer()
    assert writer.Transfer(compound, STEPControl_AsIs) == IFSelect_ReturnStatus.IFSelect_RetDone
    assert writer.Write(str(path)) == IFSelect_ReturnStatus.IFSelect_RetDone


def test我方可以提取盒体的_face_类型面积和法向(tmp_path: Path) -> None:
    step_path = tmp_path / "box.step"
    _write_box_step(step_path)

    report = inspect_step_face_features(step_path)

    assert report.status == "COMPLETE"
    assert report.face_count == 6
    assert report.surface_type_counts["GeomAbs_Plane"] == 6
    assert all(face.edge_ids for face in report.faces)
    assert all(len(face.edge_ids) == face.edge_count for face in report.faces)
    assert all(
        len(face.edge_curve_types) == len(face.edge_ids)
        and len(face.edge_lengths_mm) == len(face.edge_ids)
        and len(face.edge_radii_mm) == len(face.edge_ids)
        for face in report.faces
    )
    assert all(
        curve_type == "GeomAbs_Line"
        for face in report.faces
        for curve_type in face.edge_curve_types
    )
    assert all(
        length is not None and length > 0.0
        for face in report.faces
        for length in face.edge_lengths_mm
    )
    assert all(
        len(face.edge_endpoint_vertex_ids) == len(face.edge_ids)
        and all(
            len(vertex_ids) in {1, 2}
            for vertex_ids in face.edge_endpoint_vertex_ids.values()
        )
        for face in report.faces
    )
    assert all(not face.circular_edge_ids for face in report.faces)
    assert len({edge_id for face in report.faces for edge_id in face.edge_ids}) == 12
    assert sum(face.area_mm2 or 0.0 for face in report.faces) == pytest.approx(
        2.0 * (10.0 * 20.0 + 10.0 * 30.0 + 20.0 * 30.0),
        abs=1e-6,
    )
    assert all(face.normal_xyz is not None for face in report.faces)


def test我方真实_ocp语义采样分支可运行(tmp_path: Path) -> None:
    step_path = tmp_path / "box_semantics.step"
    _write_box_step(step_path)

    report = inspect_step_face_features(step_path, classify_semantics=True)

    assert report.status == "COMPLETE"
    assert len(report.faces) == 6
    assert all(face.semantic_evidence for face in report.faces)
    assert all(
        face.semantic_evidence["method"] == "normal_offset_solid_classifier"
        for face in report.faces
    )


def test我方记录盒体外部面语义证据而不伪造最终结论(tmp_path: Path) -> None:
    step_path = tmp_path / "box_semantics.step"
    _write_box_step(step_path)

    report = inspect_step_face_features(
        step_path,
        classify_semantics=True,
    )

    assert report.status == "COMPLETE"
    assert all(
        face.semantic_evidence["status"] == "EXTERNAL_CANDIDATE"
        for face in report.faces
    )
    assert all(
        face.semantic_evidence["method"]
        == "normal_offset_solid_classifier"
        for face in report.faces
    )
    assert report.to_dict()["schema_version"] == "step-face-features-1.8"
    assert report.vertex_coordinates_mm
    assert all(
        len(coordinates) == 3
        and all(isinstance(value, float) for value in coordinates)
        for coordinates in report.vertex_coordinates_mm.values()
    )
    assert all(face.geometry_signature for face in report.faces)
    assert report.diagnostics["geometry_signature_count"] > 0


def test我方语义采样上限拒绝非整数(tmp_path: Path) -> None:
    step_path = tmp_path / "box_semantics.step"
    _write_box_step(step_path)

    with pytest.raises(ValueError, match="至少为 1"):
        inspect_step_face_features(
            step_path,
            classify_semantics=True,
            semantic_sample_limit=1.5,
        )


def test我方多_solid_compound语义采样不漏_face(tmp_path: Path) -> None:
    step_path = tmp_path / "compound.step"
    _write_compound_step(step_path)

    report = inspect_step_face_features(
        step_path,
        classify_semantics=True,
        semantic_sample_limit=128,
    )

    assert report.status == "COMPLETE"
    assert report.shape_is_valid is True
    assert report.face_count == 12
    assert report.diagnostics["semantic_skipped_face_count"] == 0
    assert report.diagnostics["status_counts"] == {"EXTERNAL_CANDIDATE": 12}
    assert report.diagnostics["duplicate_geometry_signature_count"] > 0
    assert all(
        face.semantic_evidence["status"] == "EXTERNAL_CANDIDATE"
        for face in report.faces
    )


def test我方识别圆柱体圆弧_edge和半径(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    step_path = tmp_path / "cylinder.step"
    writer = STEPControl_Writer()
    shape = BRepPrimAPI_MakeCylinder(5.0, 20.0).Solid()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_ReturnStatus.IFSelect_RetDone
    assert writer.Write(str(step_path)) == IFSelect_ReturnStatus.IFSelect_RetDone

    report = inspect_step_face_features(step_path)

    circular_radii = [
        radius
        for face in report.faces
        for radius in face.edge_radii_mm
        if radius is not None
    ]
    assert circular_radii
    assert all(radius == pytest.approx(5.0, abs=1e-6) for radius in circular_radii)
    assert any(face.circular_edge_ids for face in report.faces)
    cylinder_faces = [
        face for face in report.faces if face.surface_type == "GeomAbs_Cylinder"
    ]
    assert cylinder_faces
    assert all(face.surface_radius_mm == pytest.approx(5.0, abs=1e-6) for face in cylinder_faces)
    assert all(face.surface_axis_origin_xyz_mm is not None for face in cylinder_faces)
    assert all(face.surface_axis_direction_xyz is not None for face in cylinder_faces)
    assert all(
        sum(value * value for value in face.surface_axis_direction_xyz or []) == pytest.approx(1.0, abs=1e-6)
        for face in cylinder_faces
    )


def test我方为圆柱体采样边折线并算出圆面投影面积(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    step_path = tmp_path / "cylinder_area.step"
    writer = STEPControl_Writer()
    shape = BRepPrimAPI_MakeCylinder(5.0, 20.0).Solid()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_ReturnStatus.IFSelect_RetDone
    assert writer.Write(str(step_path)) == IFSelect_ReturnStatus.IFSelect_RetDone

    report = inspect_step_face_features(step_path)

    # 我方按需采样边折线：整圆边被细分出多于两个点，直线边只取两个端点。
    sampler = report.edge_polyline_sampler
    assert callable(sampler)
    circular_edge_ids = sorted(
        {edge_id for face in report.faces for edge_id in face.circular_edge_ids}
    )
    assert circular_edge_ids
    sampled_circles = [sampler(edge_id) for edge_id in circular_edge_ids]
    assert all(
        points is not None and len(points) > 2 for points in sampled_circles
    )
    line_edge_ids = [
        edge_id
        for face in report.faces
        for edge_id, curve_type in zip(face.edge_ids, face.edge_curve_types)
        if curve_type == "GeomAbs_Line"
    ]
    assert line_edge_ids
    assert len(sampler(line_edge_ids[0])) == 2

    classification = classify_step_face_features(report)
    circular_regions = [
        region
        for region in classification.regions
        if "GeomAbs_Plane" in region.surface_type_counts
    ]
    assert circular_regions
    # 圆柱上下圆面由整圆闭合边组成；采样前这里会因为点数不足而算不出面积。
    assert all(
        region.primary_boundary_projection_status == "COMPUTED"
        for region in circular_regions
    )
    # 16 段内接多边形面积约为精确圆面积的 97.4%。
    exact_area = 3.141592653589793 * 5.0 * 5.0
    assert all(
        region.primary_boundary_projected_area_mm2
        == pytest.approx(exact_area, rel=0.03)
        for region in circular_regions
    )
