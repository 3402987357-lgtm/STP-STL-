from __future__ import annotations

import numpy as np
import pytest

from cad_inspector.detection.step_features import (
    _annotate_cross_region_bridges,
    _boundary_wire_selection,
    _boundary_topology_evidence,
    _combined_bbox,
    _make_group,
    _make_region,
    _open_boundary_chains,
    _point_in_polygon_2d,
    _point_in_polygon_2d_batch,
    _polygon_containment,
    _edge_bbox_candidates,
    _polygon_edges_intersect,
    _wire_projection_evidence,
    classify_step_face_features,
)
from cad_inspector.domain.step_features import StepFeatureClassificationReport
from cad_inspector.io.step_faces import StepFaceFeature, StepFaceFeatureReport


def _face(
    face_id: int,
    center: list[float],
    normal: list[float],
    edges: list[int],
) -> StepFaceFeature:
    x, y, z = center
    return StepFaceFeature(
        face_id=face_id,
        surface_type="GeomAbs_Plane",
        area_mm2=100.0,
        center_xyz_mm=center,
        normal_xyz=normal,
        bbox_min_xyz_mm=[x - 0.1, y - 0.1, z - 0.1],
        bbox_max_xyz_mm=[x + 0.1, y + 0.1, z + 0.1],
        edge_count=len(edges),
        edge_ids=edges,
        orientation="FORWARD",
    )


def _box_report() -> StepFaceFeatureReport:
    return StepFaceFeatureReport(
        status="COMPLETE",
        file_path="box.step",
        unit="mm",
        backend="OCP",
        shape_is_valid=True,
        face_count=6,
        surface_type_counts={"GeomAbs_Plane": 6},
        faces=[
            _face(1, [5.0, 5.0, 0.0], [0.0, 0.0, -1.0], [1, 2, 3, 4]),
            _face(2, [5.0, 5.0, 10.0], [0.0, 0.0, 1.0], [5, 6, 7, 8]),
            _face(3, [0.0, 5.0, 5.0], [-1.0, 0.0, 0.0], [1, 9, 10, 11]),
            _face(4, [10.0, 5.0, 5.0], [1.0, 0.0, 0.0], [2, 12, 13, 14]),
            _face(5, [5.0, 0.0, 5.0], [0.0, -1.0, 0.0], [3, 12, 15, 16]),
            _face(6, [5.0, 10.0, 5.0], [0.0, 1.0, 0.0], [4, 13, 15, 16]),
        ],
    )


def test我方从_face报告识别上下平面侧壁和邻接() -> None:
    result = classify_step_face_features(_box_report())

    assert isinstance(result, StepFeatureClassificationReport)
    assert result.status == "BASELINE"
    groups = {group.role: group for group in result.groups}
    assert groups["top_plane_candidates"].face_ids == [2]
    assert groups["bottom_plane_candidates"].face_ids == [1]
    assert groups["side_wall_candidates"].face_ids == [3, 4, 5, 6]
    assert len(result.adjacency) == 6
    assert set(result.adjacency[0].neighbor_face_ids) == {3, 4, 5, 6}
    assert result.adjacency[0].shared_edge_ids == [1, 2, 3, 4]
    assert result.adjacency[0].normal_angle_deg == pytest.approx(90.0)
    regions_by_role: dict[str, list] = {}
    for region in result.regions:
        regions_by_role.setdefault(region.role, []).append(region)
    assert set(regions_by_role) == {
        "top_plane_region",
        "bottom_plane_region",
        "side_wall_region",
    }
    assert regions_by_role["top_plane_region"][0].face_ids == [2]
    side_regions = regions_by_role["side_wall_region"]
    assert [region.face_ids for region in side_regions] == [[3], [4, 5, 6]]
    assert regions_by_role["bottom_plane_region"][0].neighbor_region_ids == [3, 4]
    assert side_regions[0].neighbor_region_ids == [2]
    assert side_regions[1].neighbor_region_ids == [2]
    # 我方按修正后的边界规则断言：只被一个 Face 引用的自由边也计入区域边界。
    # 这个合成盒体不是闭合实体，面 3 的边 9/10/11 和面 4 的边 14 都是自由边。
    assert side_regions[0].boundary_edge_ids == [1, 9, 10, 11]
    assert side_regions[1].boundary_edge_ids == [2, 3, 4, 14]
    assert all(
        region.boundary_edge_count == len(region.boundary_edge_ids)
        for region in result.regions
    )
    assert all(
        region.boundary_edge_ids == sorted(region.boundary_edge_ids)
        for region in result.regions
    )
    assert all(
        region.region_geometry_signature is not None
        and region.region_geometry_signature.startswith("regiongeom-")
        for region in result.regions
    )
    assert result.diagnostics["region_geometry_signature_count"] == len(
        result.regions
    )
    assert result.diagnostics["duplicate_region_geometry_signature_count"] == 0
    assert all(region.semantic_status == "UNRESOLVED" for region in result.regions)
    assert sorted(
        face_id for region in result.regions for face_id in region.face_ids
    ) == [1, 2, 3, 4, 5, 6]
    assert result.to_dict()["schema_version"] == "step-feature-classification-1.11"


def test我方拒绝无效分类参数() -> None:
    with pytest.raises(ValueError):
        classify_step_face_features(_box_report(), horizontal_tolerance_deg=45.0)
    with pytest.raises(ValueError):
        classify_step_face_features(_box_report(), z_band_ratio=0.5)


def test我方按顶点关联识别闭合和开放边界环() -> None:
    faces = [
        _face(1, [5.0, 5.0, 0.0], [0.0, 0.0, -1.0], [1, 2, 3, 4]),
    ]
    faces[0].edge_endpoint_vertex_ids = {
        1: [1, 2],
        2: [2, 3],
        3: [3, 4],
        4: [4, 1],
    }

    closed = _boundary_topology_evidence(faces, [1, 2, 3, 4])
    assert closed["status"] == "CLOSED_LOOP_CANDIDATE"
    assert closed["loop_count"] == 1
    assert closed["closed_loop_count"] == 1
    assert closed["edge_components"] == [[1, 2, 3, 4]]

    faces[0].edge_endpoint_vertex_ids[4] = [4, 5]
    open_boundary = _boundary_topology_evidence(faces, [1, 2, 3, 4])
    assert open_boundary["status"] == "OPEN_BOUNDARY_CANDIDATE"
    assert open_boundary["closed_loop_count"] == 0


def test我方把自由边计入区域边界并算出闭合投影() -> None:
    """我方按 OCP 真实边面关联复核后的规则：只被一个 Face 引用的自由边仍是区域边界边。"""

    face = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1, 2, 3, 4])
    face.edge_endpoint_vertex_ids = {
        1: [1, 2],
        2: [2, 3],
        3: [3, 4],
        4: [4, 1],
    }
    face.edge_lengths_mm = [10.0, 10.0, 10.0, 10.0]
    report = StepFaceFeatureReport(
        status="COMPLETE",
        file_path="open_shell.step",
        unit="mm",
        backend="OCP",
        shape_is_valid=True,
        face_count=1,
        surface_type_counts={"GeomAbs_Plane": 1},
        faces=[face],
        vertex_coordinates_mm={
            1: [0.0, 0.0, 0.0],
            2: [10.0, 0.0, 0.0],
            3: [10.0, 10.0, 0.0],
            4: [0.0, 10.0, 0.0],
        },
    )

    result = classify_step_face_features(report)

    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.boundary_edge_ids == [1, 2, 3, 4]
    assert region.boundary_topology_status == "CLOSED_LOOP_CANDIDATE"
    assert region.primary_boundary_projection_status == "COMPUTED"
    assert region.primary_boundary_projected_area_mm2 == pytest.approx(100.0)


def test我方把开放边界拆成链并给出桥接候选() -> None:
    """我方要求开放区域边界给出链端点和桥接提案，且提案默认不自动应用。"""

    face = _face(1, [15.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1, 2, 3, 4])
    face.edge_endpoint_vertex_ids = {
        1: [1, 2],
        2: [2, 3],
        3: [3, 4],
        4: [5, 6],
    }
    face.edge_lengths_mm = [10.0, 10.0, 10.0, 10.0]
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [10.0, 0.0, 0.0],
        3: [20.0, 0.0, 0.0],
        4: [30.0, 0.0, 0.0],
        5: [40.0, 0.0, 0.0],
        6: [50.0, 0.0, 0.0],
    }

    evidence = _boundary_topology_evidence(
        [face],
        [1, 2, 3, 4],
        coordinates,
    )

    assert evidence["status"] == "OPEN_BOUNDARY_CANDIDATE"
    assert evidence["closed_loop_count"] == 0
    assert evidence["open_endpoint_vertex_ids"] == [1, 4, 5, 6]
    assert evidence["open_chain_count"] == 2
    assert evidence["bridge_required"] is True
    proposals = evidence["bridge_proposals"]
    assert len(proposals) == 2
    assert all(item["applied"] is False for item in proposals)
    matched = {
        vertex_id
        for item in proposals
        for vertex_id in (item["from_vertex_id"], item["to_vertex_id"])
    }
    assert matched == {1, 4, 5, 6}
    assert all(item["to_vertex_id"] is not None for item in proposals)
    assert min(item["gap_distance_mm"] for item in proposals) == pytest.approx(10.0)
    chain = evidence["components"][0]["open_chains"][0]
    assert chain["closed"] is False
    assert chain["edge_ids"] == [1, 2, 3]
    assert chain["start_vertex_id"] == 1
    assert chain["end_vertex_id"] == 4
    assert chain["length_mm"] == pytest.approx(30.0)


def test我方不把同一条开放链的两个端点桥接成退化小环() -> None:
    """我方要求桥接候选连接不同链，而不是把链自己闭成小环。"""

    # 链 A：1-2-3-4（长 30 mm）；链 B：5-6（长 2 mm）。
    # 纯最近端点贪心会先选 5-6（2 mm），把链 B 自己闭上，
    # 再选 1-4 把链 A 自己也闭上；我要的是 4-5 或 1-6 这类跨链桥接。
    face = _face(1, [15.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1, 2, 3, 4, 5])
    face.edge_endpoint_vertex_ids = {
        1: [1, 2],
        2: [2, 3],
        3: [3, 4],
        4: [5, 6],
    }
    face.edge_lengths_mm = [10.0, 10.0, 10.0, 2.0]
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [10.0, 0.0, 0.0],
        3: [20.0, 0.0, 0.0],
        4: [30.0, 0.0, 0.0],
        5: [32.0, 0.0, 0.0],
        6: [34.0, 0.0, 0.0],
    }

    evidence = _boundary_topology_evidence(
        [face],
        [1, 2, 3, 4],
        coordinates,
    )

    proposals = [
        item
        for item in evidence["bridge_proposals"]
        if item["to_vertex_id"] is not None
    ]
    assert proposals, "我要求至少给出一条跨链桥接候选"
    same_chain_pairs = {(1, 4), (5, 6)}
    for item in proposals:
        pair = (item["from_vertex_id"], item["to_vertex_id"])
        assert pair not in same_chain_pairs, f"同链自连：{pair}"
        assert item["method"] == "nearest_open_endpoint_pairing_no_same_chain"
        assert item["applied"] is False
    # 链 A 端点 4 应当接到链 B 的某个端点，缺口为 2 mm 到 4 mm 之间。
    bridged = {
        item["from_vertex_id"]: item["to_vertex_id"] for item in proposals
    }
    assert set(bridged) | set(bridged.values()) == {1, 4, 5, 6}
    assert any(
        item["from_vertex_id"] in {1, 4} and item["to_vertex_id"] in {5, 6}
        or item["from_vertex_id"] in {5, 6} and item["to_vertex_id"] in {1, 4}
        for item in proposals
    )


def test我方把零间距的跨区域端点标成跨区域桥接候选() -> None:
    """我方要求缺口另一侧在别的区域时也能给出桥接候选。

    单区域的 _bridge_proposals 只看本区域端点，两个区域在零间距上贴着
    却互相看不见。MD270324 上 102 个开放区域里有 33 个属于这种情况。
    """

    first = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [11])
    first.surface_type = "GeomAbs_Plane"
    first.edge_endpoint_vertex_ids = {11: [101, 102]}
    first.edge_lengths_mm = [1.0]
    first.edge_ids = [11]

    second = _face(2, [10.0, 0.0, 0.0], [0.0, 0.0, 1.0], [12])
    second.surface_type = "GeomAbs_Plane"
    second.edge_endpoint_vertex_ids = {12: [201, 202]}
    second.edge_lengths_mm = [1.0]
    second.edge_ids = [12]

    coordinates = {
        101: [0.0, 0.0, 0.0],
        102: [1.0, 0.0, 0.0],
        # 201 与 101 坐标完全一致：跨区域的同一个物理点
        201: [0.0, 0.0, 0.0],
        202: [5.0, 0.0, 0.0],
    }

    group = _make_group("side_wall_candidates", [first, second], None, None)
    regions = [
        _make_region(1, group, [first], [11], coordinates),
        _make_region(2, group, [second], [12], coordinates),
    ]
    _annotate_cross_region_bridges(regions, coordinates)

    bridges = {
        region.region_id: (region.boundary_topology_evidence or {}).get(
            "cross_region_bridges"
        )
        for region in regions
    }
    assert bridges[1] and bridges[2]
    assert any(item["gap_distance_mm"] == 0.0 for item in bridges[1])
    # 两侧都要记到对方区域，且 applied 一律为 False 交人工确认
    assert {item["to_region_id"] for item in bridges[1]} == {2}
    assert all(item["applied"] is False for item in bridges[1])


def test我方把两条链共享同一顶点的自接触边界判为自接触而非缺口() -> None:
    """我方要求两条开放链端点顶点 id 相同时判为自接触闭合，不是开放缺口。

    MD270324 区域 2260：两条链的端点顶点 id 完全相同（851 与 851），
    边界在这里自己碰到自己，却被当成两个独立自由端、算成开放边界。
    """

    face = _face(2260, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1329, 1766, 1764])
    face.surface_type = "GeomAbs_Cone"
    face.edge_endpoint_vertex_ids = {
        # MD270324 区域 2260 的真实形态：一条自环链 [1329]（851 -> 851）
        # 与另一条链 [1766, 1764, 1765, 1769] 共用顶点 851。
        1329: [851],  # 自环链
        1766: [851, 1182],  # 另一条链，起点也是 851
        1764: [1181, 1182],
        1765: [1181, 1182],
        1769: [933, 1182],
    }
    face.edge_lengths_mm = [25.1327, 13.8309, 8.0, 12.5664, 104.9172]
    face.edge_ids = [1329, 1766, 1764, 1765, 1769]
    coordinates = {
        851: [0.0, 0.0, 0.0],
        1182: [1.0, 0.0, 0.0],
        1181: [2.0, 0.0, 0.0],
        933: [3.0, 0.0, 0.0],
    }

    evidence = _boundary_topology_evidence(
        [face], [1329, 1766, 1764, 1765, 1769], coordinates
    )

    assert evidence["status"] == "SELF_CONTACT_CLOSED_CANDIDATE"
    assert evidence["self_touch_shared_vertex_ids"] == [851]


def test我方不把单一自环链重复贡献同一顶点误判成自接触() -> None:
    """我方要求共享顶点必须由**两条不同**的链提供，一条自环链不算。

    MD270324 区域 2784：链 [8704, 8705] 是自环（5682 -> 5682），它把同一个
    顶点 id 贡献了两次。若按"出现次数 > 1"判定，一条自环链自己就能凑够
    两次、被误判成自接触；必须按"拥有该顶点的链数"来判。
    """

    face = _face(2784, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [11, 12, 13])
    face.surface_type = "GeomAbs_Cone"
    face.edge_endpoint_vertex_ids = {
        11: [1, 2],  # 自环链 1 -> 1（实际是 5682 -> 5682）
        12: [1, 2],
        13: [3, 4],  # 另一条独立链，不共用顶点
    }
    face.edge_lengths_mm = [1.0, 1.0, 1.0]
    face.edge_ids = [11, 12, 13]
    coordinates = {1: [0.0, 0.0, 0.0], 2: [1.0, 0.0, 0.0],
                   3: [5.0, 0.0, 0.0], 4: [6.0, 0.0, 0.0]}

    evidence = _boundary_topology_evidence([face], [11, 12, 13], coordinates)

    # 顶点 1 只被一条链（自环链）拥有，不构成自接触
    assert evidence["self_touch_shared_vertex_ids"] == []


def test我方不把两条不共享顶点的链判成自接触() -> None:
    """我方要求只有真正共用的顶点才算自接触，否则仍是开放缺口。"""

    face = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [11, 12])
    face.surface_type = "GeomAbs_Plane"
    face.edge_endpoint_vertex_ids = {11: [1, 2], 12: [3, 4]}
    face.edge_lengths_mm = [1.0, 1.0]
    face.edge_ids = [11, 12]
    coordinates = {1: [0.0, 0.0, 0.0], 2: [1.0, 0.0, 0.0],
                   3: [5.0, 0.0, 0.0], 4: [6.0, 0.0, 0.0]}

    evidence = _boundary_topology_evidence([face], [11, 12], coordinates)

    assert evidence["status"] == "OPEN_BOUNDARY_CANDIDATE"
    assert evidence["self_touch_shared_vertex_ids"] == []


def test我方把圆锥顶点的退化圆也算作接缝圆而非缺口() -> None:
    """我方要求圆锥顶点（半径 0 的退化圆）被算进接缝圆判定。

    MD270324 区域 4157：Edge 1282 是底圆（周长 21.2058 -> 半径 3.375 mm），
    Edge 1283 是圆锥顶点、长度 0.0（顶点圆的半径为 0），Edge 1284 长 3.9374
    是母线。零长边在上游会被当成退化边剥离，我必须把它单独收回来参与判定，
    否则 96 个圆锥顶点全被误判成开放缺口。
    """

    face = _face(4157, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1282, 1283, 1284])
    face.surface_type = "GeomAbs_Cone"
    face.edge_endpoint_vertex_ids = {
        1282: [825],  # 底圆塌缩成圆心
        1283: [826],  # 圆锥顶点，长度 0
        1284: [825, 826],  # 母线
    }
    face.edge_lengths_mm = [21.2058, 0.0, 3.9374]
    coordinates = {825: [0.0, 0.0, 0.0], 826: [0.0, 0.0, 3.9374]}

    evidence = _boundary_topology_evidence(
        [face],
        [1282, 1283, 1284],
        coordinates,
        degenerate_edge_vertices_override={1283: (826,)},
    )

    assert evidence["status"] == "REVOLUTION_SEAM_CLOSED_CANDIDATE"
    closure = evidence["seam_closure"]
    assert closure["reason"] == "two_collapsed_seam_circles"
    assert closure["seam_edge_ids"] == [1282, 1283]
    # 底圆半径 3.375，顶点半径 0，差 3.375 不超过母线 3.9374
    assert closure["first_radius_mm"] == pytest.approx(3.375, rel=1e-3)
    assert closure["second_radius_mm"] == pytest.approx(0.0, abs=1e-9)
    assert closure["radius_delta_mm"] <= closure["link_length_mm"]


def test我方把塌缩圆加闭合环的完整回转面判为闭合() -> None:
    """我方要求"一个接缝圆塌缩成圆心 + 另一端是闭合环 + 一条桥边"时判闭合。

    MD270324 区域 3732 的真实形态：Edge 18512 只给出端点 12209（塌缩圆，
    周长 19.297 mm -> 半径约 3.0712 mm），其余边围成一个各顶点度数均为 2
    的闭合环，Edge 18513 把圆心连到环上。按度数判定会把圆心当成度 1 自由端，
    误判成开放缺口；实际这是完整回转面，边界闭合。
    """

    face = _face(3732, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [18440, 18512, 18513])
    face.surface_type = "GeomAbs_Cone"
    face.edge_endpoint_vertex_ids = {
        18512: [12209],  # 塌缩成圆心的接缝圆
        18513: [12139, 12209],  # 桥边
        18440: [12138, 12139],  # 闭合环边 1
        18441: [12139, 12140],  # 闭合环边 2
        18485: [12183, 12184],  # 闭合环边 3
        18514: [12138, 12184],  # 闭合环边 4
        18515: [12140, 12183],  # 闭合环边 5
    }
    # 环上五条边长度之和 = 19.297，与塌缩圆周长一致 -> 半径相等
    face.edge_lengths_mm = [0.4526, 3.856, 4.3087, 5.9015, 5.9015, 19.297, 2.0514]
    face.edge_ids = [18440, 18441, 18485, 18514, 18515, 18512, 18513]
    coordinates = {
        12209: [0.0, 0.0, 0.0],
        12139: [1.0, 0.0, 0.0],
        12138: [1.0, 1.0, 0.0],
        12140: [1.0, -1.0, 0.0],
        12183: [2.0, -1.0, 0.0],
        12184: [2.0, 1.0, 0.0],
    }

    evidence = _boundary_topology_evidence(
        [face],
        [18440, 18441, 18485, 18512, 18513, 18514, 18515],
        coordinates,
    )

    assert evidence["status"] == "REVOLUTION_SEAM_CLOSED_CANDIDATE"
    closure = evidence["seam_closure"]
    assert closure["reason"] == "collapsed_seam_circle_plus_closed_loop"
    assert closure["seam_edge_ids"] == [18512]
    assert closure["axial_edge_ids"] == [18513]
    # 真实几何里两个半径并不完全相等（3.0712 / 3.2500），差 0.1788 mm，
    # 远小于桥边长 2.0514 mm —— 判据是"半径差不超过桥边长"，不是要求相等。
    assert closure["first_radius_mm"] == pytest.approx(3.0712, rel=1e-3)
    assert closure["second_radius_mm"] == pytest.approx(3.2500, rel=1e-3)
    assert closure["radius_delta_mm"] <= closure["link_length_mm"]


def test我方不把未闭合的环判成回转面闭合() -> None:
    """我方要求剩余边必须真正闭合（各顶点度数恰为 2），否则不判闭合。

    这里环缺了一条边，12140 的度数只有 1，是真实缺口，不能算闭合回转面。
    """

    face = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [11, 12, 13, 14])
    face.surface_type = "GeomAbs_Cone"
    face.edge_endpoint_vertex_ids = {
        11: [1],  # 塌缩圆
        12: [2, 1],  # 桥边
        13: [2, 3],  # 环边 1
        14: [3, 4],  # 环边 2 —— 4 号顶点度数为 1，环没闭合
    }
    face.edge_lengths_mm = [20.0, 2.0, 10.0, 10.0]
    face.edge_ids = [11, 12, 13, 14]
    coordinates = {1: [0.0, 0.0, 0.0], 2: [1.0, 0.0, 0.0],
                   3: [2.0, 0.0, 0.0], 4: [3.0, 0.0, 0.0]}

    evidence = _boundary_topology_evidence([face], [11, 12, 13, 14], coordinates)

    assert evidence["status"] != "REVOLUTION_SEAM_CLOSED_CANDIDATE"
    assert evidence["seam_closure"]["closed"] is False


def test我方把两个塌缩接缝圆的完整回转面判为闭合() -> None:
    """我方要求两个接缝圆各自塌缩成圆心、由一条边相连时判为闭合。

    MD270324 区域 399 的真实形态：Edge 891 与 1315 各只给出一个端点
    （568 / 842），周长 31.4159 mm 由 2πr 反推得半径 5.0 mm，Edge 1316
    长 10.0 mm 把两个圆心连起来。按度数判定会得到两个度 1 端点，
    被误判成开放缺口；实际这是完整圆柱侧面，边界闭合。
    """

    face = _face(399, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [891, 1315, 1316])
    face.surface_type = "GeomAbs_Cylinder"
    face.edge_endpoint_vertex_ids = {891: [568], 1315: [842], 1316: [568, 842]}
    face.edge_lengths_mm = [31.41592653589793, 31.41592653589793, 10.0]
    coordinates = {568: [0.0, 0.0, 0.0], 842: [10.0, 0.0, 0.0]}

    evidence = _boundary_topology_evidence(
        [face], [891, 1315, 1316], coordinates
    )

    assert evidence["status"] == "REVOLUTION_SEAM_CLOSED_CANDIDATE"
    assert evidence["bridge_required"] is False
    closure = evidence["seam_closure"]
    assert closure["reason"] == "two_collapsed_seam_circles"
    assert closure["seam_edge_ids"] == [891, 1315]
    assert closure["axial_edge_ids"] == [1316]
    assert closure["first_radius_mm"] == pytest.approx(5.0, rel=1e-9)
    assert closure["second_radius_mm"] == pytest.approx(5.0, rel=1e-9)


def test我方不把半径差超过连接边的开放边界判成回转面闭合() -> None:
    """我方要求半径差超过连接边长时不判闭合——那在几何上连不上。

    两个"圆"半径分别约 15.9155 mm 和 0.1592 mm，相差 15.7563 mm，
    而连接边只有 1.0 mm，圆台侧面在几何上根本不可能闭合成这个形状。
    """

    face = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [11, 12, 13])
    face.surface_type = "GeomAbs_Cone"
    face.edge_endpoint_vertex_ids = {11: [1], 12: [2], 13: [1, 2]}
    face.edge_lengths_mm = [100.0, 1.0, 1.0]  # 半径约 15.92 / 0.16，差 15.76
    coordinates = {1: [0.0, 0.0, 0.0], 2: [1.0, 0.0, 0.0]}

    evidence = _boundary_topology_evidence([face], [11, 12, 13], coordinates)

    assert evidence["status"] != "REVOLUTION_SEAM_CLOSED_CANDIDATE"
    assert evidence["seam_closure"]["closed"] is False


def test我方把OCP塌缩圆接缝的完整回转面判为闭合而非开放缺口() -> None:
    """我方要求"两条弧长之和等于圆周长"时判为闭合，不误报缺口。

    MD270324 区域 433 的真实形态：Edge 1423 只给出端点 943（接缝圆被 OCP
    塌缩成圆心点），Edge 1807/1808 挂在 1195/1196 上、弧长 2.2253 + 24.4782
    = 26.7035 正好等于圆的周长，Edge 1806 是连接圆心的轴向接缝假象。
    按度数判定会得到两个度 1 端点、被误判成开放缺口。
    """

    face = _face(433, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1423, 1806, 1807, 1808])
    face.surface_type = "GeomAbs_Cylinder"
    face.edge_endpoint_vertex_ids = {
        1423: [943],  # 单端点：接缝圆塌缩成圆心
        1806: [943, 1195],  # 轴向边（接缝假象）
        1807: [1195, 1196],  # 弧 1
        1808: [1195, 1196],  # 弧 2
    }
    face.edge_lengths_mm = [26.7035, 11.0, 2.2253, 24.4782]
    coordinates = {
        943: [0.0, 0.0, 0.0],
        1195: [0.0, 0.0, 11.0],
        1196: [2.2, 0.0, 11.0],
    }

    evidence = _boundary_topology_evidence(
        [face],
        [1423, 1806, 1807, 1808],
        coordinates,
    )

    assert evidence["status"] == "REVOLUTION_SEAM_CLOSED_CANDIDATE"
    assert evidence["bridge_required"] is False
    assert evidence["bridge_proposals"] == []
    assert evidence["open_endpoint_vertex_ids"] == []
    closure = evidence["seam_closure"]
    assert closure["closed"] is True
    assert closure["arc_edge_ids"] == [1807, 1808]
    assert closure["seam_edge_ids"] == [1423]
    assert closure["axial_edge_ids"] == [1806]
    assert closure["arc_length_sum_mm"] == pytest.approx(
        closure["seam_length_mm"], rel=1e-9
    )


def test我方不把真实缺口误判成回转面闭合() -> None:
    """我方要求弧长之和不等于圆周长时不判闭合，真实缺口必须留着。"""

    face = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [11, 12, 13])
    face.surface_type = "GeomAbs_Cylinder"
    face.edge_endpoint_vertex_ids = {
        11: [1],  # 单端点接缝圆，周长 20
        12: [2, 3],  # 弧 1
        13: [2, 3],  # 弧 2 —— 但两条弧之和只有 9，缺口是真的
    }
    face.edge_lengths_mm = [20.0, 4.0, 5.0]
    coordinates = {1: [0.0, 0.0, 0.0], 2: [0.0, 0.0, 1.0], 3: [1.0, 0.0, 1.0]}

    evidence = _boundary_topology_evidence([face], [11, 12, 13], coordinates)

    assert evidence["status"] != "REVOLUTION_SEAM_CLOSED_CANDIDATE"
    assert evidence["seam_closure"]["closed"] is False
    assert evidence["seam_closure"]["reason"] == (
        "no_arc_pair_matching_seam_circle"
    )


def test我方把分支点也算作可桥接端点() -> None:
    """我方要求度大于 1 的分支点作为链端点时同样可桥接。

    BOTTLE 区域 43 的真实形态：短链两端是度 3 的分支点，只按度 1 收集
    会漏掉它们，区域只剩 2 个端点、拿不到任何候选。
    """

    # 我照 BOTTLE 区域 43 的真实度数搭：{v1: 1, v2: 3, v3: 3, v4: 1}。
    #   e201: v1-v2   e202: v2-v3（短链）  e203: v3-v4   e204: v2-v3（第二条）
    # 于是 v2、v3 度为 3（分支点），v1、v4 度为 1。
    # 长链 e201-e202-e203 的自由端是 v1 / v4；短链 e204 两端是分支点 v2 / v3，
    # 它们同样是链端、必须可桥接。
    face = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [201, 202, 203, 204])
    face.edge_endpoint_vertex_ids = {
        201: [1, 2],
        202: [2, 3],
        203: [3, 4],
        204: [2, 3],
    }
    face.edge_lengths_mm = [10.0, 10.0, 10.0, 2.0]
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [10.0, 0.0, 0.0],
        3: [20.0, 0.0, 0.0],
        4: [30.0, 0.0, 0.0],
    }

    evidence = _boundary_topology_evidence(
        [face],
        [201, 202, 203, 204],
        coordinates,
    )

    assert evidence["open_endpoint_vertex_ids"] == [1, 2, 3, 4]
    proposals = [
        item
        for item in evidence["bridge_proposals"]
        if not item.get("unmatched")
    ]
    assert proposals, "我要求给出跨链桥接候选而不是全部落单"
    assert all(item["to_vertex_id"] is not None for item in proposals)
    covered = {
        vertex_id
        for item in proposals
        for vertex_id in (item["from_vertex_id"], item["to_vertex_id"])
    }
    assert covered == {1, 2, 3, 4}


def test我方把单端点退化自环收成链而不让整条分量变空() -> None:
    """我方要求端点塌缩成单点的自环单独成链，不拖垮整条分量的分解。

    MD270324 区域 399 的真实形态：边 891、1315 各只给出一个端点（圆柱接缝处
    的自环），边 1316 正常连接 568/842。旧实现遇到“端点数不是 2”就整体返回
    空列表，于是该区域没有端点、被误判为开放边界（全模型 1779 个）。
    """

    edge_vertices = {891: [568], 1315: [842], 1316: [568, 842]}
    lengths = {891: 31.41592653589793, 1315: 31.41592653589793, 1316: 10.0}
    chains = _open_boundary_chains([891, 1315, 1316], edge_vertices, lengths)

    self_loops = [item for item in chains if item.get("degenerate_self_loop")]
    assert len(self_loops) == 2
    assert {item["edge_ids"][0] for item in self_loops} == {891, 1315}
    assert all(item["closed"] is True for item in self_loops)
    assert all(
        item["start_vertex_id"] == item["end_vertex_id"] for item in self_loops
    )
    # 正常的单边链必须仍然被分解出来，端点 568 / 842 得以暴露。
    normal = [
        item for item in chains if not item.get("degenerate_self_loop")
    ]
    assert len(normal) == 1
    assert normal[0]["edge_ids"] == [1316]
    assert normal[0]["start_vertex_id"] == 568
    assert normal[0]["end_vertex_id"] == 842
    assert normal[0]["closed"] is False


def test我方隔离近零长度退化自环不误判闭合环为开放边界() -> None:
    """我方要求极点/接缝处 1e-9 mm 量级的退化自环不参与闭合环判定。"""

    face = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1, 2, 3, 4, 5])
    face.edge_endpoint_vertex_ids = {
        1: [1, 2],
        2: [2, 3],
        3: [3, 4],
        4: [4, 1],
        5: [1],
    }
    face.edge_lengths_mm = [10.0, 10.0, 10.0, 10.0, 1e-10]
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [10.0, 0.0, 0.0],
        3: [10.0, 10.0, 0.0],
        4: [0.0, 10.0, 0.0],
    }

    evidence = _boundary_topology_evidence(
        [face],
        [1, 2, 3, 4, 5],
        coordinates,
    )

    assert evidence["degenerate_edge_ids"] == [5]
    assert evidence["degenerate_edge_count"] == 1
    assert evidence["status"] == "CLOSED_LOOP_CANDIDATE"
    assert evidence["loop_count"] == 1
    assert evidence["closed_loop_count"] == 1
    assert evidence["edge_components"] == [[1, 2, 3, 4]]
    assert evidence["open_endpoint_vertex_ids"] == []
    assert evidence["bridge_required"] is False


def test我方把全部退化自环的边界标成退化专用状态() -> None:
    """我方要求边界只剩退化自环时给出专用状态，而不是冒充闭合环。"""

    face = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1, 2])
    face.edge_endpoint_vertex_ids = {1: [1], 2: [1]}
    face.edge_lengths_mm = [1e-10, 1e-10]

    evidence = _boundary_topology_evidence([face], [1, 2], {1: [0.0, 0.0, 0.0]})

    assert evidence["status"] == "DEGENERATE_ONLY_BOUNDARY"
    assert evidence["degenerate_edge_count"] == 2
    assert evidence["loop_count"] == 0
    assert evidence["closed_loop_count"] == 0
    assert evidence["bridge_required"] is False


def test我方把没有自由端的自接触边界判为闭合并记录接触点() -> None:
    """我方要求度数 4 的接触点仍算闭合边界，但单独记录并要求复核。"""

    face = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1, 2, 3, 4, 5, 6])
    face.edge_endpoint_vertex_ids = {
        1: [1, 2],
        2: [2, 3],
        3: [3, 1],
        4: [1, 4],
        5: [4, 5],
        6: [5, 1],
    }
    face.edge_lengths_mm = [10.0] * 6
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [10.0, 0.0, 0.0],
        3: [0.0, 10.0, 0.0],
        4: [-10.0, 0.0, 0.0],
        5: [0.0, -10.0, 0.0],
    }

    evidence = _boundary_topology_evidence([face], [1, 2, 3, 4, 5, 6], coordinates)

    assert evidence["status"] == "SELF_CONTACT_CLOSED_CANDIDATE"
    assert evidence["loop_count"] == 1
    assert evidence["closed_loop_count"] == 1
    assert evidence["branch_vertex_ids"] == [1]
    assert evidence["branch_vertex_count"] == 1
    component = evidence["components"][0]
    assert component["closed"] is True
    assert component["branch_vertex_ids"] == [1]
    assert component["open_endpoint_vertex_ids"] == []
    assert component["projection_status"] == "UNORDERED_COMPONENT"
    assert evidence["open_endpoint_vertex_ids"] == []
    assert evidence["bridge_required"] is False


def test我方按最大闭合边界_wire选择主轮廓并识别平局() -> None:
    components = [
        {
            "edge_ids": [1, 2, 3, 4],
            "vertex_ids": [1, 2, 3, 4],
            "closed": True,
        },
        {
            "edge_ids": [5, 6, 7],
            "vertex_ids": [5, 6, 7],
            "closed": True,
        },
    ]

    selected = _boundary_wire_selection(components)

    assert selected["status"] == "MULTIPLE_CLOSED_CANDIDATES"
    assert selected["primary_edge_ids"] == [1, 2, 3, 4]
    assert selected["candidate_count"] == 2
    assert selected["primary_usable"] is True
    assert selected["selection_metric"] == "edge_count"

    tied = _boundary_wire_selection(
        [
            {
                "edge_ids": [1, 2, 3],
                "vertex_ids": [1, 2, 3],
                "closed": True,
            },
            {
                "edge_ids": [4, 5, 6],
                "vertex_ids": [4, 5, 6],
                "closed": True,
            },
        ]
    )
    assert tied["status"] == "AMBIGUOUS_MAXIMUM_TIE"
    assert tied["primary_usable"] is False


def test我方优先按有限边界周长选择_wire并保留主周长() -> None:
    selected = _boundary_wire_selection(
        [
            {
                "edge_ids": [1, 2, 3, 4],
                "vertex_ids": [1, 2, 3, 4],
                "closed": True,
                "perimeter_length_mm": 40.0,
                "perimeter_complete": True,
            },
            {
                "edge_ids": [5, 6, 7, 8, 9],
                "vertex_ids": [5, 6, 7, 8, 9],
                "closed": True,
                "perimeter_length_mm": 30.0,
                "perimeter_complete": True,
            },
        ]
    )

    assert selected["selection_metric"] == "perimeter_length_mm"
    assert selected["primary_edge_ids"] == [1, 2, 3, 4]
    assert selected["primary_perimeter_length_mm"] == pytest.approx(40.0)


def test我方按有序顶点计算闭合_wire投影面积() -> None:
    evidence = _wire_projection_evidence(
        [1, 2, 3, 4],
        {
            1: (1, 2),
            2: (2, 3),
            3: (3, 4),
            4: (4, 1),
        },
        {
            1: [0.0, 0.0, 0.0],
            2: [10.0, 0.0, 0.0],
            3: [10.0, 10.0, 0.0],
            4: [0.0, 10.0, 0.0],
        },
        closed=True,
    )

    assert evidence["status"] == "COMPUTED"
    assert evidence["point_count"] == 4
    assert evidence["projected_area_mm2"] == pytest.approx(100.0)
    assert evidence["max_out_of_plane_mm"] == pytest.approx(0.0, abs=1e-9)
    assert evidence["ordered_vertex_ids"] == [1, 2, 3, 4]


def _circle_polyline(
    radius: float,
    start_deg: float,
    end_deg: float,
    segments: int,
) -> list[list[float]]:
    angles = np.radians(np.linspace(start_deg, end_deg, segments + 1))
    return [
        [float(radius * np.cos(angle)), float(radius * np.sin(angle)), 0.0]
        for angle in angles
    ]


_RADIUS_10_SIXTEEN_GON_AREA_MM2 = 306.14674587548704


def test我方用边曲线采样把两段圆弧闭环投影成非零面积() -> None:
    # 两段半圆弧共用两个顶点，只用端点会退化成两个点，投影面积为零。
    upper = _circle_polyline(10.0, 0.0, 180.0, 8)
    lower = _circle_polyline(10.0, 180.0, 360.0, 8)

    evidence = _wire_projection_evidence(
        [1, 2],
        {1: (1, 2), 2: (2, 1)},
        {1: [10.0, 0.0, 0.0], 2: [-10.0, 0.0, 0.0]},
        closed=True,
        edge_polyline_points_mm={1: upper, 2: lower},
    )

    assert evidence["status"] == "COMPUTED"
    assert evidence["point_source"] == "edge_curve_sampling"
    assert evidence["point_count"] == 16
    assert evidence["projected_area_mm2"] == pytest.approx(
        _RADIUS_10_SIXTEEN_GON_AREA_MM2,
        rel=1e-6,
    )
    assert evidence["max_out_of_plane_mm"] == pytest.approx(0.0, abs=1e-9)


def test我方用边曲线采样把单条闭合圆边投影成非零面积() -> None:
    # 整圆在 STEP 里常常是一条只有一个 Vertex 的闭合 Edge。
    circle = _circle_polyline(10.0, 0.0, 360.0, 16)

    evidence = _wire_projection_evidence(
        [7],
        {7: (1,)},
        {1: [10.0, 0.0, 0.0]},
        closed=True,
        edge_polyline_points_mm={7: circle},
    )

    assert evidence["status"] == "COMPUTED"
    assert evidence["point_source"] == "edge_curve_sampling"
    assert evidence["point_count"] == 16
    assert evidence["projected_area_mm2"] == pytest.approx(
        _RADIUS_10_SIXTEEN_GON_AREA_MM2,
        rel=1e-6,
    )


def test我方缺少边采样时两段圆弧仍回退为点数不足() -> None:
    evidence = _wire_projection_evidence(
        [1, 2],
        {1: (1, 2), 2: (2, 1)},
        {1: [10.0, 0.0, 0.0], 2: [-10.0, 0.0, 0.0]},
        closed=True,
    )

    assert evidence["status"] == "INSUFFICIENT_POINTS"
    assert evidence["point_source"] == "vertex_endpoints"
    assert evidence["point_count"] == 2


def test我方标记周长最大与投影面积最大候选冲突() -> None:
    selected = _boundary_wire_selection(
        [
            {
                "edge_ids": [1, 2, 3, 4],
                "vertex_ids": [1, 2, 3, 4],
                "closed": True,
                "perimeter_length_mm": 40.0,
                "perimeter_complete": True,
                "projected_area_mm2": 50.0,
                "projection_status": "COMPUTED",
                "projection_evidence": {"status": "COMPUTED"},
            },
            {
                "edge_ids": [5, 6, 7],
                "vertex_ids": [5, 6, 7],
                "closed": True,
                "perimeter_length_mm": 30.0,
                "perimeter_complete": True,
                "projected_area_mm2": 100.0,
                "projection_status": "COMPUTED",
                "projection_evidence": {"status": "COMPUTED"},
            },
        ]
    )

    assert selected["status"] == "PROJECTED_AREA_CONFLICT"
    assert selected["primary_edge_ids"] == [1, 2, 3, 4]
    assert selected["primary_projected_area_mm2"] == pytest.approx(50.0)
    assert selected["projection_max_component_index"] == 1
    assert selected["projection_max_area_mm2"] == pytest.approx(100.0)
    assert selected["projection_conflict"] is True
    assert selected["primary_usable"] is False
    assert selected["primary_projection_evidence"]["projection_conflict"] is True


def test我方识别嵌套边界环和孔洞候选() -> None:
    outer_edges = {
        1: (1, 2),
        2: (2, 3),
        3: (3, 4),
        4: (4, 1),
    }
    inner_edges = {
        5: (5, 6),
        6: (6, 7),
        7: (7, 8),
        8: (8, 5),
    }
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [100.0, 0.0, 0.0],
        3: [100.0, 100.0, 0.0],
        4: [0.0, 100.0, 0.0],
        5: [25.0, 25.0, 0.0],
        6: [75.0, 25.0, 0.0],
        7: [75.0, 75.0, 0.0],
        8: [25.0, 75.0, 0.0],
    }
    outer_projection = _wire_projection_evidence(
        [1, 2, 3, 4],
        outer_edges,
        coordinates,
        closed=True,
    )
    inner_projection = _wire_projection_evidence(
        [5, 6, 7, 8],
        inner_edges,
        coordinates,
        closed=True,
    )
    records = [
        {
            "edge_ids": [1, 2, 3, 4],
            "vertex_ids": [1, 2, 3, 4],
            "closed": True,
            "perimeter_length_mm": 400.0,
            "perimeter_complete": True,
            "projected_area_mm2": outer_projection["projected_area_mm2"],
            "projection_status": outer_projection["status"],
            "projection_evidence": outer_projection,
        },
        {
            "edge_ids": [5, 6, 7, 8],
            "vertex_ids": [5, 6, 7, 8],
            "closed": True,
            "perimeter_length_mm": 200.0,
            "perimeter_complete": True,
            "projected_area_mm2": inner_projection["projected_area_mm2"],
            "projection_status": inner_projection["status"],
            "projection_evidence": inner_projection,
        },
    ]

    selected = _boundary_wire_selection(
        records,
        vertex_coordinates_mm=coordinates,
    )

    assert selected["status"] == "MULTIPLE_CLOSED_CANDIDATES"
    assert selected["primary_edge_ids"] == [1, 2, 3, 4]
    assert selected["loop_relation_status"] == "NESTED_LOOPS"
    assert selected["primary_loop_role"] == "OUTER_WITH_HOLES_CANDIDATE"
    assert selected["primary_loop_is_hole_candidate"] is False
    assert selected["loop_relation_conflict"] is False
    assert selected["loop_relation_evidence"]["hole_candidate_indices"] == [1]
    assert selected["loop_relation_evidence"]["parent_by_component"] == {
        "1": 0
    }


def test我方拒绝把内环选成主边界Wire() -> None:
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [100.0, 0.0, 0.0],
        3: [100.0, 100.0, 0.0],
        4: [0.0, 100.0, 0.0],
        5: [25.0, 25.0, 0.0],
        6: [75.0, 25.0, 0.0],
        7: [75.0, 75.0, 0.0],
        8: [25.0, 75.0, 0.0],
    }
    outer_edges = {1: (1, 2), 2: (2, 3), 3: (3, 4), 4: (4, 1)}
    inner_edges = {5: (5, 6), 6: (6, 7), 7: (7, 8), 8: (8, 5)}
    outer_projection = _wire_projection_evidence(
        [1, 2, 3, 4],
        outer_edges,
        coordinates,
        closed=True,
    )
    inner_projection = _wire_projection_evidence(
        [5, 6, 7, 8],
        inner_edges,
        coordinates,
        closed=True,
    )
    selected = _boundary_wire_selection(
        [
            {
                "edge_ids": [1, 2, 3, 4],
                "vertex_ids": [1, 2, 3, 4],
                "closed": True,
                "perimeter_length_mm": 400.0,
                "perimeter_complete": True,
                "projected_area_mm2": outer_projection["projected_area_mm2"],
                "projection_status": outer_projection["status"],
                "projection_evidence": outer_projection,
            },
            {
                "edge_ids": [5, 6, 7, 8],
                "vertex_ids": [5, 6, 7, 8],
                "closed": True,
                "perimeter_length_mm": 500.0,
                "perimeter_complete": True,
                "projected_area_mm2": inner_projection["projected_area_mm2"],
                "projection_status": inner_projection["status"],
                "projection_evidence": inner_projection,
            },
        ],
        vertex_coordinates_mm=coordinates,
    )

    assert selected["status"] == "LOOP_RELATION_CONFLICT"
    assert selected["primary_edge_ids"] == [5, 6, 7, 8]
    assert selected["primary_loop_role"] == "INNER_HOLE_CANDIDATE"
    assert selected["primary_loop_is_hole_candidate"] is True
    assert selected["loop_relation_conflict"] is True
    assert selected["primary_usable"] is False
    assert selected["loop_relation_evidence"]["parent_by_component"] == {
        "1": 0
    }


def test我方标记相交边界环关系冲突() -> None:
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [100.0, 0.0, 0.0],
        3: [100.0, 100.0, 0.0],
        4: [0.0, 100.0, 0.0],
        5: [50.0, 50.0, 0.0],
        6: [120.0, 50.0, 0.0],
        7: [120.0, 120.0, 0.0],
        8: [50.0, 120.0, 0.0],
    }
    outer_edges = {1: (1, 2), 2: (2, 3), 3: (3, 4), 4: (4, 1)}
    overlapping_edges = {5: (5, 6), 6: (6, 7), 7: (7, 8), 8: (8, 5)}
    outer_projection = _wire_projection_evidence(
        [1, 2, 3, 4],
        outer_edges,
        coordinates,
        closed=True,
    )
    overlapping_projection = _wire_projection_evidence(
        [5, 6, 7, 8],
        overlapping_edges,
        coordinates,
        closed=True,
    )
    selected = _boundary_wire_selection(
        [
            {
                "edge_ids": [1, 2, 3, 4],
                "vertex_ids": [1, 2, 3, 4],
                "closed": True,
                "perimeter_length_mm": 400.0,
                "perimeter_complete": True,
                "projected_area_mm2": outer_projection["projected_area_mm2"],
                "projection_status": outer_projection["status"],
                "projection_evidence": outer_projection,
            },
            {
                "edge_ids": [5, 6, 7, 8],
                "vertex_ids": [5, 6, 7, 8],
                "closed": True,
                "perimeter_length_mm": 280.0,
                "perimeter_complete": True,
                "projected_area_mm2": overlapping_projection[
                    "projected_area_mm2"
                ],
                "projection_status": overlapping_projection["status"],
                "projection_evidence": overlapping_projection,
            },
        ],
        vertex_coordinates_mm=coordinates,
    )

    assert selected["loop_relation_status"] == "INTERSECTING_LOOPS"
    assert selected["status"] == "LOOP_RELATION_CONFLICT"
    assert selected["loop_relation_conflict"] is True
    assert selected["loop_relation_evidence"]["intersecting_pairs"] == [[0, 1]]


def test我方把面积相同的相交环判为相交() -> None:
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [100.0, 0.0, 0.0],
        3: [100.0, 100.0, 0.0],
        4: [0.0, 100.0, 0.0],
        5: [50.0, 50.0, 0.0],
        6: [150.0, 50.0, 0.0],
        7: [150.0, 150.0, 0.0],
        8: [50.0, 150.0, 0.0],
    }
    first_edges = {1: (1, 2), 2: (2, 3), 3: (3, 4), 4: (4, 1)}
    second_edges = {5: (5, 6), 6: (6, 7), 7: (7, 8), 8: (8, 5)}
    first_projection = _wire_projection_evidence(
        [1, 2, 3, 4],
        first_edges,
        coordinates,
        closed=True,
    )
    second_projection = _wire_projection_evidence(
        [5, 6, 7, 8],
        second_edges,
        coordinates,
        closed=True,
    )

    selected = _boundary_wire_selection(
        [
            {
                "edge_ids": [1, 2, 3, 4],
                "vertex_ids": [1, 2, 3, 4],
                "closed": True,
                "perimeter_length_mm": 400.0,
                "perimeter_complete": True,
                "projected_area_mm2": first_projection["projected_area_mm2"],
                "projection_status": first_projection["status"],
                "projection_evidence": first_projection,
            },
            {
                "edge_ids": [5, 6, 7, 8],
                "vertex_ids": [5, 6, 7, 8],
                "closed": True,
                "perimeter_length_mm": 400.0,
                "perimeter_complete": True,
                "projected_area_mm2": second_projection["projected_area_mm2"],
                "projection_status": second_projection["status"],
                "projection_evidence": second_projection,
            },
        ],
        vertex_coordinates_mm=coordinates,
    )

    assert selected["loop_relation_status"] == "INTERSECTING_LOOPS"
    assert selected["loop_relation_conflict"] is True
    assert selected["loop_relation_evidence"]["intersecting_pairs"] == [[0, 1]]


def test我方不把面积相同的分离环误判为相交() -> None:
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [100.0, 0.0, 0.0],
        3: [100.0, 100.0, 0.0],
        4: [0.0, 100.0, 0.0],
        5: [150.0, 0.0, 0.0],
        6: [250.0, 0.0, 0.0],
        7: [250.0, 100.0, 0.0],
        8: [150.0, 100.0, 0.0],
    }
    first_edges = {1: (1, 2), 2: (2, 3), 3: (3, 4), 4: (4, 1)}
    second_edges = {5: (5, 6), 6: (6, 7), 7: (7, 8), 8: (8, 5)}
    first_projection = _wire_projection_evidence(
        [1, 2, 3, 4],
        first_edges,
        coordinates,
        closed=True,
    )
    second_projection = _wire_projection_evidence(
        [5, 6, 7, 8],
        second_edges,
        coordinates,
        closed=True,
    )

    selected = _boundary_wire_selection(
        [
            {
                "edge_ids": [1, 2, 3, 4],
                "vertex_ids": [1, 2, 3, 4],
                "closed": True,
                "perimeter_length_mm": 400.0,
                "perimeter_complete": True,
                "projected_area_mm2": first_projection["projected_area_mm2"],
                "projection_status": first_projection["status"],
                "projection_evidence": first_projection,
            },
            {
                "edge_ids": [5, 6, 7, 8],
                "vertex_ids": [5, 6, 7, 8],
                "closed": True,
                "perimeter_length_mm": 400.0,
                "perimeter_complete": True,
                "projected_area_mm2": second_projection["projected_area_mm2"],
                "projection_status": second_projection["status"],
                "projection_evidence": second_projection,
            },
        ],
        vertex_coordinates_mm=coordinates,
    )

    assert selected["loop_relation_status"] == "MULTIPLE_DISJOINT_LOOPS"
    assert selected["loop_relation_conflict"] is False
    assert selected["loop_relation_evidence"]["intersecting_pairs"] == []
    assert selected["loop_relation_evidence"]["touching_pairs"] == []


def test我方对投影退化非平面和缺失坐标给出稳定状态() -> None:
    square_edges = {
        1: (1, 2),
        2: (2, 3),
        3: (3, 4),
        4: (4, 1),
    }
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [10.0, 0.0, 0.0],
        3: [10.0, 10.0, 0.0],
        4: [0.0, 10.0, 0.0],
    }

    not_closed = _wire_projection_evidence(
        [1, 2, 3, 4],
        square_edges,
        coordinates,
        closed=False,
    )
    assert not_closed["status"] == "NOT_CLOSED"

    non_planar = _wire_projection_evidence(
        [1, 2, 3, 4],
        square_edges,
        {
            **coordinates,
            4: [0.0, 10.0, 5.0],
        },
        closed=True,
    )
    assert non_planar["status"] == "NON_PLANAR_CANDIDATE"
    assert non_planar["projected_area_mm2"] is not None

    missing = _wire_projection_evidence(
        [1, 2, 3, 4],
        square_edges,
        {1: [0.0, 0.0, 0.0], 2: [10.0, 0.0, 0.0]},
        closed=True,
    )
    assert missing["status"] == "MISSING_VERTEX_COORDINATES"

    unordered = _wire_projection_evidence(
        [1, 2, 3, 4],
        {
            1: (1, 2),
            2: (2, 3),
            3: (3, 1),
            4: (1, 4),
        },
        {1: [0.0, 0.0, 0.0], 2: [1.0, 0.0, 0.0], 3: [0.0, 1.0, 0.0], 4: [0.0, 0.0, 1.0]},
        closed=True,
    )
    assert unordered["status"] == "UNORDERED_COMPONENT"


def test我方隔离非有限或退化_face并降级复核() -> None:
    report = _box_report()
    report.faces[0].center_xyz_mm = [float("nan"), 5.0, 0.0]
    report.faces[0].bbox_min_xyz_mm = [float("nan"), 4.9, -0.1]
    report.faces[0].bbox_max_xyz_mm = [float("nan"), 5.1, 0.1]
    report.faces[1].normal_xyz = [0.0, 0.0, 0.0]

    result = classify_step_face_features(report)

    assert result.status == "REVIEW_REQUIRED"
    assert result.bbox_min_xyz_mm is not None
    assert result.bbox_max_xyz_mm is not None
    assert all(
        np.isfinite(value)
        for value in (*result.bbox_min_xyz_mm, *result.bbox_max_xyz_mm)
    )
    assert "不能参加方向分类" in " ".join(result.warnings)


def test我方跳过或不完整的_shape证据必须复核() -> None:
    report = _box_report()
    report.status = "PARTIAL"
    report.shape_is_valid = None

    result = classify_step_face_features(report)

    assert result.status == "REVIEW_REQUIRED"


def test我方重复_face编号不会静默覆盖区域() -> None:
    report = _box_report()
    report.faces[1].face_id = report.faces[0].face_id

    result = classify_step_face_features(report)

    assert result.status == "REVIEW_REQUIRED"
    assert result.diagnostics["duplicate_face_id_count"] == 1
    assert result.diagnostics["duplicate_region_face_count"] > 0
    assert any("重复 Face 编号" in warning for warning in result.warnings)


def test我方非法_face编号会被隔离并要求复核() -> None:
    report = _box_report()
    report.faces[0].face_id = 0
    report.faces[1].face_id = "2"  # type: ignore[assignment]

    result = classify_step_face_features(report)

    assert result.status == "REVIEW_REQUIRED"
    assert result.diagnostics["valid_face_count"] == 4
    assert any("Face 编号不可用" in warning for warning in result.warnings)


def test我方中间水平_face进入兜底区域而不被漏掉() -> None:
    report = _box_report()
    report.faces.append(
        _face(
            7,
            [5.0, 5.0, 5.0],
            [0.0, 0.0, 1.0],
            [17, 18, 19, 20],
        )
    )
    report.face_count = 7

    result = classify_step_face_features(report)

    groups = {group.role: group for group in result.groups}
    assert groups["horizontal_other_candidates"].face_ids == [7]
    assert result.diagnostics["unclassified_valid_face_count"] == 0
    assert result.status == "BASELINE"


def test我方组法向按面积加权而不是简单平均() -> None:
    """我锁定加权平均这条契约。

    一组面里常有一大片主面加几块很小的倒角面。简单平均会让小面
    获得与主面相同的发言权，算出的"组法向"被小面拉偏，
    而这个组本来应该代表主面的朝向。

    数值对比很直观：加权得到 (0.0002, 0, 1)，
    简单平均得到 (0.667, 0, 0.333) —— 后者几乎不再是 +Z 面。
    """

    main_face = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [])
    main_face.area_mm2 = 100.0
    chamfer_a = _face(2, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [])
    chamfer_a.area_mm2 = 0.01
    chamfer_b = _face(3, [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [])
    chamfer_b.area_mm2 = 0.01

    group = _make_group("top", [main_face, chamfer_a, chamfer_b], None, None)

    assert group.mean_normal_xyz[2] == pytest.approx(1.0, abs=1e-6)
    assert abs(group.mean_normal_xyz[0]) < 1e-3


def test我方面积缺失的面不被丢弃且法向抵消时不产生NaN() -> None:
    """我覆盖两个兜底路径。

    area_mm2 为 None 时用极小权重兜底而不是跳过：丢弃会让面从
    face_ids 里消失（漏面），极小权重只是让它在加权里几乎不发声。

    法向互相抵消时（例如闭合环）平均后长度接近 0，
    此时不能做除法，否则除零会产出 NaN 污染下游所有夹角计算。
    """

    main_face = _face(1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [])
    main_face.area_mm2 = 100.0

    # 面积缺失的面
    unknown_area = _face(4, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [])
    unknown_area.area_mm2 = None
    group = _make_group("top", [main_face, unknown_area], None, None)
    assert group.face_ids == [1, 4]

    # 法向互相抵消
    up = _face(5, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [])
    down = _face(6, [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [])
    opposed = _make_group("top", [up, down], None, None)
    assert np.isfinite(opposed.mean_normal_xyz).all()


def test我方合并包围盒在无有效值时返回None对() -> None:
    """我锁定 None 与零包围盒的区分。

    零包围盒会被下游当成一个真实的退化包围盒，
    只有 None 才能表达"这个组合并没测出范围"。
    """

    with_bbox = _face(7, [5.0, 5.0, 5.0], [0.0, 0.0, 1.0], [])
    without_bbox = _face(8, [5.0, 5.0, 5.0], [0.0, 0.0, 1.0], [])
    without_bbox.bbox_min_xyz_mm = None
    without_bbox.bbox_max_xyz_mm = None

    # 混合时只取有效面
    assert _combined_bbox([with_bbox, without_bbox]) == (
        [4.9, 4.9, 4.9],
        [5.1, 5.1, 5.1],
    )
    # 全无效与空输入都返回 None 对
    assert _combined_bbox([without_bbox]) == (None, None)
    assert _combined_bbox([]) == (None, None)


def test我方半径内点对与暴力全遍历逐对一致() -> None:
    """我钉住 KD-tree 宽相位与 O(n^2) 全遍历结果必须逐对一致。

    跨区域桥接原先是 O(n^2) 全遍历，我换成 cKDTree.query_pairs 做宽相位。
    换的只是"怎么找到候选对"，判定语义一个字没改，所以两种实现必须给出
    完全相同的点对集合。这里用随机点云在多个尺度与多个半径下对拍。
    """

    from cad_inspector.detection.step_features import _pairs_within_tolerance

    rng = np.random.default_rng(20241009)
    for _ in range(8):
        count = int(rng.integers(2, 120))
        scale = float(rng.choice([0.001, 0.01, 1.0, 50.0]))
        coordinates = rng.normal(scale=scale, size=(count, 3))
        radius = float(rng.choice([1e-6, 0.01, 0.5, 5.0]))
        got = _pairs_within_tolerance(coordinates, radius)
        reference = sorted(
            (i, j)
            for i in range(count)
            for j in range(i + 1, count)
            if float(np.linalg.norm(coordinates[i] - coordinates[j])) <= radius
        )
        assert got == reference


def test我方点对查询在退化输入上返回空而不是抛错() -> None:
    """我钉住空输入、单点、非正容差这些边界都安静返回空列表。"""

    from cad_inspector.detection.step_features import _pairs_within_tolerance

    assert _pairs_within_tolerance(np.zeros((0, 3)), 0.1) == []
    assert _pairs_within_tolerance(np.zeros((1, 3)), 0.1) == []
    assert _pairs_within_tolerance(np.zeros((3, 3)), 0.0) == []
    assert _pairs_within_tolerance(np.zeros((3, 3)), float("nan")) == []


def test我方scipy缺失时点对查询退回全遍历且结果不变() -> None:
    """我钉住 scipy 装不上时的降级路径：行为不变，只是慢。

    scipy 已在 requirements.txt 里声明，但部署环境可能没装上；
    那种情况下我必须仍然给出正确结果，而不是崩掉或返回空。
    """

    import builtins

    from cad_inspector.detection import step_features

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("我方在测试里屏蔽 scipy")
        return real_import(name, *args, **kwargs)

    coordinates = np.array(
        [[0.0, 0.0, 0.0], [0.005, 0.0, 0.0], [9.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    builtins.__import__ = blocked_import
    try:
        assert step_features._pairs_within_tolerance(coordinates, 0.01) == [
            (0, 1)
        ]
    finally:
        builtins.__import__ = real_import


def test我方批量射线法与逐点判定结果完全一致() -> None:
    """我锁定批量版与逐点版的语义等价性。

    `_point_in_polygon_2d_batch` 是性能优化产物：它把"逐点调用射线法"
    改成"一次算完一批点"。优化最容易犯的错是判定式悄悄走偏，
    于是快了但结果不对——所以这条契约必须钉死：
    同一个多边形、同一批点，两种算法必须给出逐位相同的结果。

    我特意用凹多边形（L 形）与随机星形多边形，
    因为凸多边形上射线法的很多边界分支根本走不到，
    只在凹形和自相交形态下才会暴露差异。
    """

    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    probes = np.array(
        [
            [5.0, 5.0],  # 内部
            [15.0, 5.0],  # 外部
            [-5.0, 5.0],  # 外部
            [0.0, 5.0],  # 边界
            [10.0, 10.0],  # 顶点
            [5.0, 0.0],  # 边界
        ]
    )
    expected = np.array(
        [_point_in_polygon_2d(point, square) for point in probes], dtype=np.int8
    )
    assert np.array_equal(_point_in_polygon_2d_batch(probes, square), expected)
    assert expected.tolist() == [1, 0, 0, -1, -1, -1]

    ell = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [4.0, 4.0], [4.0, 10.0], [0.0, 10.0]]
    )
    rng = np.random.default_rng(7)
    for _ in range(5):
        points = rng.uniform(-2.0, 12.0, size=(200, 2))
        expected = np.array(
            [_point_in_polygon_2d(point, ell) for point in points], dtype=np.int8
        )
        assert np.array_equal(_point_in_polygon_2d_batch(points, ell), expected)

    for _ in range(5):
        count = int(rng.integers(20, 120))
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, count))
        radii = rng.uniform(1.0, 5.0, count)
        polygon = np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii])
        points = rng.uniform(-6.0, 6.0, size=(200, 2))
        expected = np.array(
            [_point_in_polygon_2d(point, polygon) for point in points],
            dtype=np.int8,
        )
        assert np.array_equal(
            _point_in_polygon_2d_batch(points, polygon), expected
        )


def test我方包含判定对边界点仍优先报相切() -> None:
    """我锁定"边界优先于内部"这条优先级不被优化改掉。

    相切是两个环共享边或点的信号，真实外轮廓与孔洞之间不应发生，
    必须上报而不是被当成包含。批量化之后如果边界判定被
    奇偶统计盖过，这个异常就静默消失了。
    """

    outer = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    # 三个内部点 + 一个恰好落在边界上的点
    inner = np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0], [0.0, 5.0]])
    assert _polygon_containment(inner, outer) == "TOUCHING"

    # 全部在内部时报 INSIDE
    assert _polygon_containment(inner[:3], outer) == "INSIDE"

    # 全部在包围盒之外时报 DISJOINT（走前置快速路径）
    far = np.array([[20.0, 20.0], [30.0, 30.0]])
    assert _polygon_containment(far, outer) == "DISJOINT"


def test我方分块边相交判定在巨型环上仍给出正确结论() -> None:
    """我锁定分块优化不改变边相交判定的结论。

    `_polygon_edges_intersect` 按行分块是为了让中间阵列留在
    缓存里（真实环可达上万顶点，整块 n*m 是几千万元素）。
    分块最容易出的错是"块边界漏算"或"共线分支拿到块尺寸阵列
    而广播错位"，两者都不会报错，只会悄悄给出错的结论。

    所以我特意用远超分块大小的环（1200 顶点，n*m 达 144 万），
    并覆盖相交、相离、共享边、包含四种形态——
    只测小环根本走不到分块路径，等于没测。
    """

    def square(origin_x: float, origin_y: float, side: float) -> np.ndarray:
        return np.array(
            [
                [origin_x, origin_y],
                [origin_x + side, origin_y],
                [origin_x + side, origin_y + side],
                [origin_x, origin_y + side],
            ]
        )

    assert _polygon_edges_intersect(square(0, 0, 4), square(2, 2, 4)) is True
    assert _polygon_edges_intersect(square(0, 0, 4), square(10, 10, 4)) is False
    assert _polygon_edges_intersect(square(0, 0, 4), square(4, 0, 4)) is True
    assert _polygon_edges_intersect(square(0, 0, 8), square(2, 2, 2)) is False

    rng = np.random.default_rng(11)

    def ring(center_x: float, radius: float, count: int) -> np.ndarray:
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, count))
        radii = radius + rng.uniform(-0.4, 0.4, count)
        return np.column_stack(
            [center_x + np.cos(angles) * radii, np.sin(angles) * radii]
        )

    # 同一圆心的两个大环必然相交；远距离的两个大环必然不相交。
    big_a = ring(0.0, 4.0, 1200)
    big_b = ring(0.0, 4.0, 1200)
    assert _polygon_edges_intersect(big_a, big_b) is True

    far_a = ring(-50.0, 2.0, 1200)
    far_b = ring(50.0, 2.0, 1200)
    assert _polygon_edges_intersect(far_a, far_b) is False


def test我方包围盒预筛不会把相交的边错筛掉() -> None:
    """我锁定预筛是"保守剔除"而不是"激进剔除"。

    `_edge_bbox_candidates` 只剔掉"包围盒都不重叠"的边对。
    如果它漏掉了真正相交的边，后面的精确判定就再也看不到它们，
    于是相交被静默报成不相交——这是最危险的失败模式，
    因为它不报错，只让报告少一条异常。

    所以我钉死这条契约：凡是真正相交的边，
    必须出现在预筛的候选里。
    """

    outer = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    # 一个明确穿过 outer 的环：四个顶点中两个在内、两个在外，
    # 它的边必然与 outer 的边相交。
    crossing = np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 6.0], [5.0, 6.0]])
    keep = _edge_bbox_candidates(crossing, outer)
    assert bool(np.any(keep)) is True
    assert _polygon_edges_intersect(crossing, outer) is True

    # 完全远离的环：没有任何边的包围盒会重叠，可以整体剔除。
    far = np.array([[100.0, 100.0], [110.0, 100.0], [110.0, 110.0], [100.0, 110.0]])
    keep_far = _edge_bbox_candidates(far, outer)
    assert bool(np.any(keep_far)) is False
    assert _polygon_edges_intersect(far, outer) is False

    # 完全被包含的环：它不与外环任何一条边相交，预筛把它整条剔掉
    # 是对的——但结论仍必须是"不相交"，不能因为剔掉了就走捷径。
    inside = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]])
    keep_inside = _edge_bbox_candidates(inside, outer)
    assert bool(np.any(keep_inside)) is False
    assert _polygon_edges_intersect(inside, outer) is False

    # 关键回归：预筛剔掉全部边时，最终结论必须与"不过滤直接判定"一致。
    # 这是我真正要防的失败——过滤把候选清空后若直接返回 True，
    # 所有内含/相离的环都会被报成相交。
    assert _polygon_edges_intersect(inside, outer) is False
    assert _polygon_edges_intersect(far, outer) is False
