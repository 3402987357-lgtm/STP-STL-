from __future__ import annotations

import pytest

from cad_inspector.detection.step_engineering import (
    _region_category_evidence,
    _region_measurements,
    _region_review_queue,
    classify_step_engineering_features,
)
from cad_inspector.domain.step_engineering import (
    StepEngineeringCandidate,
    StepEngineeringReport,
)
from cad_inspector.domain.step_features import StepFeatureRegion
from cad_inspector.io.step_faces import StepFaceFeature, StepFaceFeatureReport
from cad_inspector.output.step_engineering_report import (
    _json_for_csv,
    _compact_boundary_topology_evidence,
    _compact_loop_relation_evidence,
    _limited_list,
    write_step_engineering_report,
)


def _face(
    face_id: int,
    center: list[float],
    normal: list[float],
    edges: list[int],
    *,
    surface_type: str = "GeomAbs_Plane",
    radii: list[float | None] | None = None,
    circular_edges: list[int] | None = None,
    surface_radius: float | None = None,
    axis_origin: list[float] | None = None,
    axis_direction: list[float] | None = None,
) -> StepFaceFeature:
    x, y, z = center
    return StepFaceFeature(
        face_id=face_id,
        surface_type=surface_type,
        area_mm2=100.0,
        center_xyz_mm=center,
        normal_xyz=normal,
        bbox_min_xyz_mm=[x - 1.0, y - 1.0, z - 1.0],
        bbox_max_xyz_mm=[x + 1.0, y + 1.0, z + 1.0],
        edge_count=len(edges),
        edge_ids=edges,
        orientation="FORWARD",
        edge_curve_types=["GeomAbs_Circle" if edge_id in (circular_edges or []) else "GeomAbs_Line" for edge_id in edges],
        edge_lengths_mm=[10.0] * len(edges),
        edge_radii_mm=radii or [None] * len(edges),
        circular_edge_ids=circular_edges or [],
        surface_radius_mm=surface_radius,
        surface_axis_origin_xyz_mm=axis_origin,
        surface_axis_direction_xyz=axis_direction,
    )


def _report() -> StepFaceFeatureReport:
    return StepFaceFeatureReport(
        status="COMPLETE",
        file_path="box.step",
        unit="mm",
        backend="OCP",
        shape_is_valid=True,
        face_count=4,
        surface_type_counts={"GeomAbs_Plane": 3, "GeomAbs_Cylinder": 1},
        faces=[
            _face(1, [5.0, 5.0, 0.0], [0.0, 0.0, -1.0], [1, 2]),
            _face(2, [5.0, 5.0, 10.0], [0.0, 0.0, 1.0], [1, 3]),
            _face(4, [10.0, 5.0, 5.0], [1.0, 0.0, 0.0], [2, 4]),
            _face(
                3,
                [2.0, 2.0, 5.0],
                [1.0, 0.0, 0.0],
                [4, 5],
                surface_type="GeomAbs_Cylinder",
                radii=[5.0, 5.0],
                circular_edges=[4, 5],
                surface_radius=5.0,
                axis_origin=[0.0, 0.0, 0.0],
                axis_direction=[0.0, 0.0, 1.0],
            ),
        ],
    )


def test我方输出台阶圆柱和拔模候选() -> None:
    result = classify_step_engineering_features(_report())

    assert isinstance(result, StepEngineeringReport)
    assert result.status == "BASELINE"
    kinds = {candidate.kind for candidate in result.candidates}
    assert "STEP_CANDIDATE" in kinds
    assert "CYLINDER_OR_HOLE_CANDIDATE" in kinds
    assert len(result.draft_evidence) == 4
    assert result.to_dict()["schema_version"] == "step-engineering-features-2.5"
    assert result.regions
    assert result.rule_summary["region_engineering"]["region_count"] == len(
        result.regions
    )
    assert result.rule_summary["plane_roles"]["top_plane_candidates"]["face_count"] == 1
    assert result.rule_summary["plane_roles"]["bottom_plane_candidates"]["face_count"] == 1
    hole = next(
        candidate
        for candidate in result.candidates
        if candidate.kind == "CYLINDER_OR_HOLE_CANDIDATE"
    )
    assert hole.evidence["closed_boundary_proxy"] is True
    assert hole.evidence["hole_semantics"] == "HOLE_WALL_CANDIDATE"
    cylinder_region = next(
        region for region in result.regions if 3 in region.face_ids
    )
    assert "CYLINDER_OR_HOLE_REGION_CANDIDATE" in cylinder_region.role_candidates
    assert "CYLINDER_OR_HOLE_CANDIDATE" in cylinder_region.candidate_kinds
    assert cylinder_region.engineering_status == "REVIEW_REQUIRED"
    assert cylinder_region.category_candidates
    assert cylinder_region.category_review_required is True
    assert "SEMANTICS_UNRESOLVED" in cylinder_region.review_reason_codes
    assert cylinder_region.review_priority in {"MEDIUM", "HIGH"}
    # 我方按修正后的边界规则：圆柱面只被自身引用的自由边 5 也计入区域边界。
    assert cylinder_region.boundary_edge_ids == [2, 5]
    assert cylinder_region.boundary_edge_count == len(
        cylinder_region.boundary_edge_ids
    )
    assert cylinder_region.boundary_topology_status == "UNRESOLVED_ENDPOINTS"
    assert cylinder_region.boundary_loop_count == 0
    assert "BOUNDARY_TOPOLOGY_UNRESOLVED" in cylinder_region.review_reason_codes
    assert cylinder_region.boundary_wire_selection_status == "UNRESOLVED"
    assert cylinder_region.primary_boundary_edge_ids == []
    assert cylinder_region.evidence["boundary_edge_ids"] == [2, 5]
    assert cylinder_region.region_geometry_signature is not None
    assert cylinder_region.region_geometry_signature.startswith("regiongeom-")
    assert "PARTING_PROFILE_REGION_CANDIDATE" in (
        cylinder_region.parting_role_candidates
    )
    assert "PARTING_PROFILE_EDGE_CANDIDATE" in (
        cylinder_region.parting_role_candidates
    )
    assert cylinder_region.parting_confidence > 0.0
    assert cylinder_region.parting_evidence["method"] == (
        "region_role_boundary_neighbor_parting_rules"
    )
    assert result.rule_summary["region_engineering"]["review_reason_counts"][
        "SEMANTICS_UNRESOLVED"
    ] > 0
    measurements = cylinder_region.evidence["measurements"]
    assert measurements["bbox_size_xyz_mm"] == pytest.approx([10.0, 5.0, 2.0])
    assert measurements["cylinder_radius_samples_mm"] == [5.0]
    assert measurements["circular_edge_count"] == 2
    assert measurements["face_geometry_signatures"] == []
    assert "neighbor_evidence" in cylinder_region.evidence
    assert "role_transition_candidates" in cylinder_region.evidence[
        "neighbor_evidence"
    ]


def test我方输出区域工程_csv并保留人工复核证据(tmp_path) -> None:
    result = classify_step_engineering_features(_report())

    artifacts = write_step_engineering_report(result, tmp_path)

    assert "regions_csv" in artifacts
    regions_csv = tmp_path / "box.regions.csv"
    assert regions_csv.exists()
    text = regions_csv.read_text(encoding="utf-8-sig")
    assert "区域工程状态" in text
    assert "包围盒尺寸_xyz_mm" in text
    assert "边界Edge编号" in text
    assert "区域几何签名" in text
    assert "CYLINDER_OR_HOLE_REGION_CANDIDATE" in text
    assert "类别置信度" in text
    assert "复核优先级" in text
    assert "SEMANTICS_UNRESOLVED" in text
    assert "分型区域候选" in text
    assert "分型区域置信度" in text
    assert "分型区域证据" in text
    assert "边界拓扑状态" in text
    assert "边界拓扑证据" in text
    assert "主边界Wire Edge" in text
    assert "边界Wire选择状态" in text
    assert "主边界Wire投影面积_mm2" in text
    assert "主边界Wire投影状态" in text
    assert "主边界Wire投影证据" in text
    assert "主边界Wire环角色" in text
    assert "主边界Wire环关系状态" in text
    assert "主边界Wire环关系证据" in text


def test我方区域类别候选保留低置信度和混合区域复核() -> None:
    report = _report()
    result = classify_step_engineering_features(report)

    assert result.rule_summary["region_engineering"]["category_candidate_counts"]
    assert result.rule_summary["region_engineering"]["category_review_required_count"] == len(
        result.regions
    )
    assert all(region.category_evidence for region in result.regions)
    assert all(region.category_review_required for region in result.regions)


def test我方区域类别字段可稳定序列化() -> None:
    result = classify_step_engineering_features(_report())
    payload = result.to_dict()
    region = payload["regions"][0]

    assert isinstance(region["category_candidates"], list)
    assert isinstance(region["category_confidence"], float)
    assert isinstance(region["category_evidence"], dict)
    assert isinstance(region["category_review_required"], bool)
    assert isinstance(region["parting_role_candidates"], list)
    assert isinstance(region["parting_confidence"], float)
    assert isinstance(region["parting_review_required"], bool)
    assert isinstance(region["parting_evidence"], dict)
    assert "primary_boundary_projected_area_mm2" in region
    assert "primary_boundary_projection_status" in region
    assert isinstance(region["primary_boundary_projection_evidence"], dict)
    assert "primary_boundary_loop_role" in region
    assert "primary_boundary_loop_relation_status" in region
    assert isinstance(region["primary_boundary_loop_relation_evidence"], dict)
    assert region["region_geometry_signature"].startswith("regiongeom-")


def test我方CSV证据单元格不会超过Excel上限() -> None:
    payload = {
        "items": [
            {
                "index": index,
                "description": "长证据" * 200,
            }
            for index in range(200)
        ],
        "large_text": "完整证据" * 5000,
    }

    encoded = _json_for_csv(payload)

    assert len(encoded) <= 30000
    assert "truncated" in encoded


def test我方区域邻接过渡和缺失引用进入可审计证据() -> None:
    report = _report()
    result = classify_step_engineering_features(report)
    neighbor_evidence = next(
        region.evidence["neighbor_evidence"]
        for region in result.regions
        if region.neighbor_region_ids
    )

    assert neighbor_evidence["neighbor_count"] > 0
    assert neighbor_evidence["neighbor_source_roles"]
    assert neighbor_evidence["role_transition_candidates"]
    assert neighbor_evidence["missing_neighbor_region_ids"] == []
    assert result.rule_summary["region_engineering"][
        "boundary_transition_counts"
    ]


def test我方把主边界投影冲突写入人工复核队列() -> None:
    region = _category_region("RESOLVED")
    region.boundary_wire_selection_status = "PROJECTED_AREA_CONFLICT"
    region.primary_boundary_projection_evidence = {
        "projection_conflict": True,
    }

    reasons, priority = _region_review_queue(
        region,
        [],
        [],
        0.0,
        False,
        {},
        {},
        shape_is_valid=True,
    )

    assert "BOUNDARY_PROJECTION_CONFLICT" in reasons
    assert priority == "MEDIUM"


def test我方把主边界环关系冲突写入人工复核队列() -> None:
    region = _category_region("RESOLVED")
    region.boundary_wire_selection_status = "LOOP_RELATION_CONFLICT"
    region.primary_boundary_loop_relation_evidence = {
        "loop_relation_conflict": True,
    }

    reasons, priority = _region_review_queue(
        region,
        [],
        [],
        0.0,
        False,
        {},
        {},
        shape_is_valid=True,
    )

    assert "BOUNDARY_LOOP_RELATION_CONFLICT" in reasons
    assert priority == "MEDIUM"


def _category_region(
    semantic_status: str,
    *,
    neighbor_region_ids: list[int] | None = None,
    surface_type_counts: dict[str, int] | None = None,
) -> StepFeatureRegion:
    return StepFeatureRegion(
        region_id=1,
        role="side_wall_region",
        face_ids=[1],
        area_mm2=100.0,
        mean_center_xyz_mm=[0.0, 0.0, 0.0],
        mean_normal_xyz=[0.0, 0.0, 1.0],
        surface_type_counts=surface_type_counts or {"GeomAbs_Plane": 1},
        neighbor_region_ids=neighbor_region_ids or [],
        boundary_edge_count=4,
        semantic_status=semantic_status,
        score=0.8,
    )


def _category_face(
    *,
    surface_type: str,
    bbox_max: list[float],
    semantic_status: str,
    radius: float | None = None,
    circular_edges: list[int] | None = None,
    axis: bool = False,
) -> StepFaceFeature:
    return StepFaceFeature(
        face_id=1,
        surface_type=surface_type,
        area_mm2=100.0,
        center_xyz_mm=[0.0, 0.0, 0.0],
        normal_xyz=[0.0, 0.0, 1.0],
        bbox_min_xyz_mm=[0.0, 0.0, 0.0],
        bbox_max_xyz_mm=bbox_max,
        edge_count=len(circular_edges or []),
        edge_ids=list(circular_edges or []),
        orientation="FORWARD",
        edge_curve_types=["GeomAbs_Circle"] * len(circular_edges or []),
        edge_lengths_mm=[10.0] * len(circular_edges or []),
        edge_radii_mm=[radius] * len(circular_edges or []),
        circular_edge_ids=list(circular_edges or []),
        surface_radius_mm=radius,
        surface_axis_origin_xyz_mm=[0.0, 0.0, 0.0] if axis else None,
        surface_axis_direction_xyz=[0.0, 0.0, 1.0] if axis else None,
        semantic_evidence={"status": semantic_status},
    )


def test我方六类部件候选均可由合成区域证据触发() -> None:
    cases = [
        (
            "PLATE_COMPONENT_CANDIDATE",
            _category_region("EXTERNAL_CANDIDATE"),
            [
                _category_face(
                    surface_type="GeomAbs_Plane",
                    bbox_max=[100.0, 30.0, 2.0],
                    semantic_status="EXTERNAL_CANDIDATE",
                )
            ],
            [],
        ),
        (
            "SHAFT_COMPONENT_CANDIDATE",
            _category_region("EXTERNAL_CANDIDATE"),
            [
                _category_face(
                    surface_type="GeomAbs_Cylinder",
                    bbox_max=[10.0, 10.0, 80.0],
                    semantic_status="EXTERNAL_CANDIDATE",
                    radius=5.0,
                    circular_edges=[1, 2],
                    axis=True,
                )
            ],
            [
                StepEngineeringCandidate(
                    1,
                    "CYLINDER_OR_HOLE_CANDIDATE",
                    [1],
                    [1, 2],
                    0.8,
                )
            ],
        ),
        (
            "GUIDE_BUSH_COMPONENT_CANDIDATE",
            _category_region("INTERNAL_CAVITY_CANDIDATE"),
            [
                _category_face(
                    surface_type="GeomAbs_Cylinder",
                    bbox_max=[20.0, 20.0, 10.0],
                    semantic_status="INTERNAL_CAVITY_CANDIDATE",
                    radius=5.0,
                    circular_edges=[1, 2],
                )
            ],
            [
                StepEngineeringCandidate(
                    1,
                    "COAXIAL_HOLE_ARRAY_CANDIDATE",
                    [1, 2],
                    [1, 2],
                    0.8,
                )
            ],
        ),
        (
            "EJECTOR_COMPONENT_CANDIDATE",
            _category_region("INTERNAL_CAVITY_CANDIDATE"),
            [
                _category_face(
                    surface_type="GeomAbs_Cylinder",
                    bbox_max=[8.0, 8.0, 50.0],
                    semantic_status="INTERNAL_CAVITY_CANDIDATE",
                    radius=4.0,
                    axis=True,
                )
            ],
            [],
        ),
        (
            "MOLDED_SURFACE_CANDIDATE",
            _category_region("EXTERNAL_CANDIDATE"),
            [
                _category_face(
                    surface_type="GeomAbs_BSplineSurface",
                    bbox_max=[40.0, 30.0, 20.0],
                    semantic_status="EXTERNAL_CANDIDATE",
                )
            ],
            [],
        ),
        (
            "STRUCTURAL_COMPONENT_CANDIDATE",
            _category_region(
                "EXTERNAL_CANDIDATE",
                neighbor_region_ids=[2, 3],
                surface_type_counts={
                    "GeomAbs_Plane": 1,
                    "GeomAbs_Cylinder": 1,
                },
            ),
            [
                _category_face(
                    surface_type="GeomAbs_Plane",
                    bbox_max=[10.0, 10.0, 10.0],
                    semantic_status="EXTERNAL_CANDIDATE",
                ),
                _category_face(
                    surface_type="GeomAbs_Cylinder",
                    bbox_max=[10.0, 10.0, 10.0],
                    semantic_status="EXTERNAL_CANDIDATE",
                    radius=5.0,
                ),
            ],
            [],
        ),
    ]

    for expected, region, faces, candidates in cases:
        measurements = {
            "bbox_size_xyz_mm": [100.0, 30.0, 2.0]
            if expected == "PLATE_COMPONENT_CANDIDATE"
            else (
                [10.0, 10.0, 80.0]
                if expected in {
                    "SHAFT_COMPONENT_CANDIDATE",
                    "EJECTOR_COMPONENT_CANDIDATE",
                }
                else [20.0, 20.0, 10.0]
            ),
            "circular_edge_count": (
                2
                if expected
                in {
                    "SHAFT_COMPONENT_CANDIDATE",
                    "GUIDE_BUSH_COMPONENT_CANDIDATE",
                }
                else 0
            ),
        }
        if expected == "MOLDED_SURFACE_CANDIDATE":
            measurements["bbox_size_xyz_mm"] = [40.0, 30.0, 20.0]
        if expected == "STRUCTURAL_COMPONENT_CANDIDATE":
            measurements["bbox_size_xyz_mm"] = [10.0, 10.0, 10.0]
        categories, confidence, evidence, review_required, _ = _region_category_evidence(
            region,
            candidates,
            faces,
            measurements,
        )
        assert expected in categories
        assert confidence >= 0.35
        assert evidence["scores"]
        assert isinstance(review_required, bool)


def test我方圆弧共享边可以生成_r角候选() -> None:
    report = _report()
    report.faces[0].edge_ids = [10]
    report.faces[0].edge_radii_mm = [2.0]
    report.faces[0].circular_edge_ids = [10]
    report.faces[2].edge_ids = [10]
    report.faces[2].edge_radii_mm = [2.0]
    report.faces[2].circular_edge_ids = [10]

    result = classify_step_engineering_features(report)

    assert any(candidate.kind == "R_CORNER_CANDIDATE" for candidate in result.candidates)


def test我方拒绝零拔模方向() -> None:
    with pytest.raises(ValueError):
        classify_step_engineering_features(_report(), draw_direction=(0.0, 0.0, 0.0))


def test我方拒绝越界平面和孔半径容差() -> None:
    with pytest.raises(ValueError):
        classify_step_engineering_features(_report(), horizontal_tolerance_deg=45.0)
    with pytest.raises(ValueError):
        classify_step_engineering_features(_report(), hole_radius_tolerance_ratio=1.0)


def test我方拒绝非有限台阶高度和轴线距离容差() -> None:
    with pytest.raises(ValueError):
        classify_step_engineering_features(_report(), min_step_height_mm=float("nan"))
    with pytest.raises(ValueError):
        classify_step_engineering_features(_report(), min_step_height_mm=float("inf"))
    with pytest.raises(ValueError):
        classify_step_engineering_features(
            _report(),
            axis_distance_tolerance_mm=float("nan"),
        )


def test我方半径不一致时保留圆柱候选并要求复核() -> None:
    report = _report()
    report.faces[3].edge_radii_mm = [5.0, 6.0]

    result = classify_step_engineering_features(
        report,
        hole_radius_tolerance_ratio=0.05,
    )

    hole = next(
        candidate
        for candidate in result.candidates
        if candidate.kind == "CYLINDER_OR_HOLE_CANDIDATE"
    )
    assert hole.evidence["closed_boundary_proxy"] is False
    assert hole.evidence["radius_consistent"] is False
    assert hole.evidence["hole_semantics"] == "CYLINDER_FACE_CANDIDATE"
    assert result.rule_summary["hole_topology"]["topology_review_required_count"] == 1
    assert hole.warnings


def test我方外部圆柱面不会直接解释成孔壁() -> None:
    report = _report()
    report.faces[3].semantic_evidence = {
        "status": "EXTERNAL_CANDIDATE",
        "confidence": 0.75,
    }

    result = classify_step_engineering_features(report)

    hole = next(
        candidate
        for candidate in result.candidates
        if candidate.kind == "CYLINDER_OR_HOLE_CANDIDATE"
    )
    assert hole.evidence["closed_boundary_proxy"] is True
    assert hole.evidence["semantic_status"] == "EXTERNAL_CANDIDATE"
    assert hole.evidence["hole_semantics"] == "EXTERNAL_CYLINDER_FACE_CANDIDATE"
    assert any("不能把它解释成孔壁" in warning for warning in hole.warnings)
    assert result.rule_summary["hole_topology"]["external_cylinder_candidate_count"] == 1
    assert result.rule_summary["hole_topology"]["internal_hole_candidate_count"] == 0


def test我方近似共轴圆柱面生成孔阵列候选() -> None:
    report = _report()
    report.face_count = 5
    report.faces.append(
        _face(
            5,
            [0.0, 0.0, 12.0],
            [1.0, 0.0, 0.0],
            [6, 7],
            surface_type="GeomAbs_Cylinder",
            radii=[5.0, 5.0],
            circular_edges=[6, 7],
            surface_radius=5.0,
            axis_origin=[0.0, 0.0, 12.0],
            axis_direction=[0.0, 0.0, 1.0],
        )
    )

    result = classify_step_engineering_features(report)

    arrays = [
        candidate
        for candidate in result.candidates
        if candidate.kind == "COAXIAL_HOLE_ARRAY_CANDIDATE"
    ]
    assert len(arrays) == 1
    assert arrays[0].face_ids == [3, 5]
    assert arrays[0].evidence["member_count"] == 2
    assert arrays[0].evidence["radius_consistent"] is True
    assert result.rule_summary["hole_topology"]["coaxial_array_candidate_count"] == 1


def test我方半径差异过大时不生成孔阵列候选() -> None:
    report = _report()
    report.face_count = 5
    report.faces.append(
        _face(
            5,
            [0.0, 0.0, 12.0],
            [1.0, 0.0, 0.0],
            [6, 7],
            surface_type="GeomAbs_Cylinder",
            radii=[7.0, 7.0],
            circular_edges=[6, 7],
            surface_radius=7.0,
            axis_origin=[0.0, 0.0, 12.0],
            axis_direction=[0.0, 0.0, 1.0],
        )
    )

    result = classify_step_engineering_features(report)

    assert not any(
        candidate.kind == "COAXIAL_HOLE_ARRAY_CANDIDATE"
        for candidate in result.candidates
    )


def test我方无效_shape降级工程复核() -> None:
    report = _report()
    report.status = "REVIEW_REQUIRED"
    report.shape_is_valid = False

    result = classify_step_engineering_features(report)

    assert result.status == "REVIEW_REQUIRED"
    assert all("SHAPE_INVALID" in region.review_reason_codes for region in result.regions)
    assert all(region.review_priority == "HIGH" for region in result.regions)


def test我方区域复核队列汇总极端原因() -> None:
    region = _category_region(
        "UNRESOLVED",
        neighbor_region_ids=[99],
        surface_type_counts={
            "GeomAbs_Plane": 1,
            "GeomAbs_Cylinder": 1,
        },
    )
    region.semantic_evidence = {"method": "resource_guard"}
    reason_codes, priority = _region_review_queue(
        region,
        [],
        [],
        0.2,
        True,
        {},
        {"missing_neighbor_region_ids": [99]},
        shape_is_valid=False,
    )

    assert priority == "HIGH"
    assert {
        "SHAPE_INVALID",
        "SEMANTICS_UNRESOLVED",
        "RESOURCE_GUARD",
        "BOUNDARY_EDGES_PRESENT",
        "MISSING_NEIGHBOR_REGION",
        "NO_ENGINEERING_CANDIDATE",
        "MIXED_SURFACE_TYPES",
        "CATEGORY_MISSING",
        "CATEGORY_LOW_CONFIDENCE",
        "CATEGORY_AMBIGUOUS",
    }.issubset(reason_codes)


def test我方非法_face不会阻断其余工程证据() -> None:
    report = _report()
    report.faces[0].normal_xyz = [float("nan"), 0.0, 0.0]

    result = classify_step_engineering_features(report)

    assert result.status == "REVIEW_REQUIRED"
    assert result.diagnostics["valid_face_count"] == 3
    assert any("隔离" in warning for warning in result.warnings)


def test我方没有有效_face时返回复核而不是抛异常() -> None:
    report = _report()
    for face in report.faces:
        face.center_xyz_mm = None

    result = classify_step_engineering_features(report)

    assert result.status == "REVIEW_REQUIRED"
    assert result.diagnostics["valid_face_count"] == 0


def test我方继承既有分类的复核状态() -> None:
    report = _report()
    report.faces[1].face_id = report.faces[0].face_id

    result = classify_step_engineering_features(report)

    assert result.status == "REVIEW_REQUIRED"


def test我方区域测量不被个别坏面的NaN污染() -> None:
    """我锁定"先过滤坏值再聚合"这条顺序。

    真实 STEP 里个别面的包围盒可能是 None 或含 NaN（退化面、
    求值失败的面）。一个 NaN 进 np.min 会污染整行结果，
    让整个区域的 bbox 全变成 NaN，而这个区域里其余面其实都是好的。

    我用一个好面配一个坏面验证：好面的 bbox 必须原样得出，
    不能被坏面带偏。
    """

    nan = float("nan")
    good = _face(1, [5.0, 5.0, 5.0], [0.0, 0.0, 1.0], [])
    bad = _face(2, [5.0, 5.0, 5.0], [0.0, 0.0, 1.0], [])
    bad.bbox_min_xyz_mm = [nan, nan, nan]
    bad.bbox_max_xyz_mm = None

    region = StepFeatureRegion(
        region_id=1,
        role="top_plane_region",
        face_ids=[1, 2],
        area_mm2=100.0,
        mean_center_xyz_mm=[5.0, 5.0, 5.0],
        mean_normal_xyz=[0.0, 0.0, 1.0],
        surface_type_counts={"plane": 2},
        neighbor_region_ids=[],
        boundary_edge_count=0,
        semantic_status="EXTERNAL_CANDIDATE",
        score=1.0,
    )
    measurements = _region_measurements(region, [good, bad])

    assert measurements["bbox_min_xyz_mm"] == [4.0, 4.0, 4.0]
    assert measurements["bbox_max_xyz_mm"] == [6.0, 6.0, 6.0]


def test我方区域测量全为坏面时返回None而不是零向量() -> None:
    """我覆盖"一个有效值都不剩"的情形。

    零向量会被下游当成"这个区域尺寸为零"的真实测量值，
    只有 None 才能正确表达"测不出来"。
    """

    nan = float("nan")
    bad = _face(2, [5.0, 5.0, 5.0], [0.0, 0.0, 1.0], [])
    bad.bbox_min_xyz_mm = [nan, nan, nan]
    bad.bbox_max_xyz_mm = None

    region = StepFeatureRegion(
        region_id=1,
        role="top_plane_region",
        face_ids=[2],
        area_mm2=100.0,
        mean_center_xyz_mm=None,
        mean_normal_xyz=None,
        surface_type_counts={"plane": 1},
        neighbor_region_ids=[],
        boundary_edge_count=0,
        semantic_status="EXTERNAL_CANDIDATE",
        score=1.0,
    )
    measurements = _region_measurements(region, [bad])

    assert measurements["bbox_min_xyz_mm"] is None
    assert measurements["bbox_max_xyz_mm"] is None


def test我方CSV压缩不污染原始证据字典() -> None:
    """我锁定"先拷贝再改"这条契约。

    证据字典是 region 上的共享引用。若就地删除 wire_selection，
    会连 JSON 报告里的数据一起改掉——CSV 与 JSON 必须各走各的副本。
    """

    from types import SimpleNamespace

    evidence = {
        "components": [
            {
                "edge_ids": list(range(200)),
                "vertex_ids": list(range(200)),
                "closed": True,
            }
        ],
        "wire_selection": {"deep": {"nested": "value"}},
    }
    region = SimpleNamespace(boundary_topology_evidence=evidence)

    compact = _compact_boundary_topology_evidence(region)

    # 原对象不受影响
    assert "wire_selection" in evidence
    assert "evidence_truncated_in_csv" not in evidence
    # 压缩副本带摘要与截断标记
    assert "wire_selection_summary" in compact
    assert compact["evidence_truncated_in_csv"] is True
    assert len(compact["components"][0]["edge_ids"]) == 64


def test我方CSV摘要同时给出真实数量与截断列表() -> None:
    """我锁定 count 与 indices 的分工。

    count 是真实的完整数量，用于判断是否超限；
    indices 供人查看前若干个。只给截断后的列表会让人误以为
    "总共就这几个"，拿去汇报就会报出一个被低估的数字。
    """

    from types import SimpleNamespace

    region = SimpleNamespace(
        primary_boundary_loop_relation_evidence={
            "outer_candidate_indices": list(range(100)),
            "hole_candidate_indices": [],
            "intersecting_pairs": [],
            "touching_pairs": [],
            "loop_count": 3,
        },
        primary_boundary_loop_relation_status="RESOLVED",
    )

    relation = _compact_loop_relation_evidence(region)

    assert relation["outer_candidate_count"] == 100
    assert len(relation["outer_candidate_indices"]) == 64
    # status 优先取 region 属性而非常规证据里的中间量
    assert relation["status"] == "RESOLVED"


def test我方CSV摘要对非列表取值返回空列表而不是报错() -> None:
    """我锁定 CSV 摘要的"尽力而为"语义。

    证据字段在部分路径下可能缺失或类型不同。CSV 摘要的取值本来就是
    尽力而为，不该因为一个字段缺失让整份报告写不出来。
    """

    assert _limited_list(None) == []
    assert _limited_list("str") == []
    assert _limited_list({"a": 1}) == []
    assert _limited_list([1, 2, 3]) == [1, 2, 3]
