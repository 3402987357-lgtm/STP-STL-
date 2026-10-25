"""我为体素/截面辅助比对区域写回归测试。

我重点覆盖三件事：体积算得对（用解析解校验）、能抓到面级距离看不见的
内部空腔、以及网格不封闭时必须降级而不是硬报数字。
"""

from __future__ import annotations

import numpy as np
import pytest

from cad_inspector.detection.voxel_compare import (
    build_voxel_occupancy,
    compare_meshes_voxel,
    free_edge_count,
)
from cad_inspector.geometry.mesh import TriangleMesh


_UNIT_VERTICES = np.array(
    [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64
)


def _vertex_index(x: int, y: int, z: int) -> int:
    """我按 x*4 + y*2 + z 给单位立方体的八个顶点编号。"""

    return x * 4 + y * 2 + z


# 六个面各两个三角形，绕序一致朝外（右手法则指向立方体外部）。
_UNIT_FACES = np.array(
    [
        [_vertex_index(0, 0, 0), _vertex_index(0, 1, 0), _vertex_index(0, 1, 1)],
        [_vertex_index(0, 0, 0), _vertex_index(0, 1, 1), _vertex_index(0, 0, 1)],
        [_vertex_index(1, 0, 0), _vertex_index(1, 1, 1), _vertex_index(1, 1, 0)],
        [_vertex_index(1, 0, 0), _vertex_index(1, 0, 1), _vertex_index(1, 1, 1)],
        [_vertex_index(0, 0, 0), _vertex_index(1, 0, 1), _vertex_index(1, 0, 0)],
        [_vertex_index(0, 0, 0), _vertex_index(0, 0, 1), _vertex_index(1, 0, 1)],
        [_vertex_index(0, 1, 0), _vertex_index(1, 1, 0), _vertex_index(1, 1, 1)],
        [_vertex_index(0, 1, 0), _vertex_index(1, 1, 1), _vertex_index(0, 1, 1)],
        [_vertex_index(0, 0, 0), _vertex_index(1, 0, 0), _vertex_index(1, 1, 0)],
        [_vertex_index(0, 0, 0), _vertex_index(1, 1, 0), _vertex_index(0, 1, 0)],
        [_vertex_index(0, 0, 1), _vertex_index(1, 1, 1), _vertex_index(1, 0, 1)],
        [_vertex_index(0, 0, 1), _vertex_index(0, 1, 1), _vertex_index(1, 1, 1)],
    ],
    dtype=np.int64,
)


def _unit_cube(lower, upper) -> TriangleMesh:
    """我造一个轴对齐长方体，绕序一致朝外，自由边必须为 0。"""

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    vertices = _UNIT_VERTICES * (upper - lower) + lower
    return TriangleMesh(vertices=vertices, faces=_UNIT_FACES.copy())


def test_free_edge_count_closed_cube_is_zero() -> None:
    """封闭长方体没有自由边——这是体积可信的前提。"""

    assert free_edge_count(_unit_cube([0, 0, 0], [10, 10, 10])) == 0


def test_free_edge_count_detects_open_box() -> None:
    """我拆掉一个面，自由边必须被数出来。"""

    mesh = _unit_cube([0, 0, 0], [10, 10, 10])
    mesh.faces = mesh.faces[:-2]
    assert free_edge_count(mesh) == 4


def test_free_edge_count_empty_mesh() -> None:
    """空网格返回 0 而不是抛异常。"""

    empty = TriangleMesh(
        vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)
    )
    assert free_edge_count(empty) == 0


def test_occupancy_volume_matches_analytic() -> None:
    """10x10x10 的立方体，体素体积必须收敛到 1000 mm^3。"""

    mesh = _unit_cube([0, 0, 0], [10, 10, 10])
    for voxel_size in (2.0, 1.0, 0.5):
        occupancy, _sections = build_voxel_occupancy(mesh, voxel_size=voxel_size)
        volume = int(np.count_nonzero(occupancy)) * voxel_size**3
        assert volume == pytest.approx(1000.0)


@pytest.mark.parametrize(
    "extent",
    [(10, 10, 10), (12, 10, 10), (5, 7, 9), (7, 3, 5), (3, 10, 4), (20, 20, 20)],
)
def test_occupancy_volume_exact_for_many_extents(extent) -> None:
    """多个尺寸的盒子都必须填出精确体积。

    这条是踩坑回归：包围盒尺寸为体素边长整数倍时，切片平面会恰好
    穿过一排顶点；同时同一交点被相邻三角形算两次会有 1e-15 级差异。
    两个问题叠加曾让 12x10x10 只填出 856 格（应为 1200），
    而 10x10x10 恰好躲过——换个尺寸就错的 bug 必须靠多尺寸守住。
    """

    mesh = _unit_cube([0, 0, 0], list(extent))
    occupancy, _sections = build_voxel_occupancy(mesh, voxel_size=1.0)
    expected = int(extent[0] * extent[1] * extent[2])
    assert int(np.count_nonzero(occupancy)) == expected


def test_slice_at_vertex_row_still_yields_closed_loop() -> None:
    """切片平面穿过顶点排时，轮廓必须仍然闭合。"""

    mesh = _unit_cube([0, 0, 0], [12, 10, 10])
    _occupancy, sections = build_voxel_occupancy(mesh, voxel_size=1.0)
    assert len(sections) == 10
    for item in sections:
        assert item.open_loop_count == 0
        assert item.area_mm2 == pytest.approx(120.0)


def test_cross_section_area_matches_analytic() -> None:
    """每一层截面面积都必须是 10x10=100 mm^2。"""

    mesh = _unit_cube([0, 0, 0], [10, 10, 10])
    _occupancy, sections = build_voxel_occupancy(mesh, voxel_size=1.0)
    assert len(sections) == 10
    for item in sections:
        assert item.area_mm2 == pytest.approx(100.0)
        assert item.loop_count >= 1
        assert item.open_loop_count == 0


def test_identical_meshes_give_iou_one() -> None:
    """同一个件对自己，必须完全一致。"""

    mesh = _unit_cube([0, 0, 0], [10, 10, 10])
    result = compare_meshes_voxel(mesh, mesh, voxel_size=1.0)
    assert result.status == "OK"
    assert result.iou == pytest.approx(1.0)
    assert result.missing_count == 0
    assert result.extra_count == 0
    assert result.warnings == []


def test_half_volume_is_detected_as_missing() -> None:
    """候选件只有一半，缺失体积必须是 500 mm^3。"""

    full = _unit_cube([0, 0, 0], [10, 10, 10])
    half = _unit_cube([0, 0, 0], [5, 10, 10])
    result = compare_meshes_voxel(full, half, voxel_size=1.0)
    assert result.missing_count == 500
    assert result.extra_count == 0
    assert result.missing_volume_mm3 == pytest.approx(500.0)
    assert result.iou == pytest.approx(0.5)


def test_extra_material_is_detected() -> None:
    """候选件多出一截，多余体必须被数出来且方向不能反。"""

    full = _unit_cube([0, 0, 0], [10, 10, 10])
    bigger = _unit_cube([0, 0, 0], [12, 10, 10])
    result = compare_meshes_voxel(full, bigger, voxel_size=1.0)
    assert result.missing_count == 0
    assert result.extra_count == 200
    assert result.extra_volume_mm3 == pytest.approx(200.0)


def test_internal_cavity_is_detected() -> None:
    """这是体素比对的核心价值：外壳完整但内部被挖空。

    面级距离判据在这里会漏检——空腔壁上的点在参考件上照样能找到
    很近的面。体素占用必须抓到这 4x4x4=64 的空洞。
    """

    outer = _unit_cube([0, 0, 0], [10, 10, 10])
    inner = _unit_cube([3, 3, 3], [7, 7, 7])
    # 翻转绕序让内壳法向朝内，等价于从实体里挖掉这块。
    inner.faces = inner.faces[:, ::-1]
    shell = TriangleMesh(
        vertices=np.concatenate([outer.vertices, inner.vertices]),
        faces=np.concatenate(
            [outer.faces, inner.faces + len(outer.vertices)]
        ),
    )
    assert free_edge_count(shell) == 0
    result = compare_meshes_voxel(outer, shell, voxel_size=1.0)
    assert result.status == "OK"
    assert result.missing_count == 64
    assert result.extra_count == 0


def test_non_watertight_mesh_is_downgraded() -> None:
    """网格有破口时我拒绝给出体积结论。

    填充会从破口漏到外部，此时"占用"不代表实体内部，
    继续报体积等于给甲方一个看起来精确的错误数字。
    """

    full = _unit_cube([0, 0, 0], [10, 10, 10])
    open_box = _unit_cube([0, 0, 0], [10, 10, 10])
    open_box.faces = open_box.faces[:-2]
    result = compare_meshes_voxel(full, open_box, voxel_size=1.0)
    assert result.status == "REVIEW_REQUIRED"
    assert result.candidate_watertight is False
    assert result.candidate_free_edge_count == 4
    assert any("自由边" in warning for warning in result.warnings)


def test_non_cubic_extent_uses_common_grid() -> None:
    """X/Y/Z 尺寸不同也必须能比——这里曾因轴序写反直接崩过。"""

    full = _unit_cube([0, 0, 0], [6, 10, 4])
    half = _unit_cube([0, 0, 0], [3, 10, 4])
    result = compare_meshes_voxel(full, half, voxel_size=1.0)
    assert result.grid_dims == (6, 10, 4)
    assert result.missing_count == 120


def test_voxel_size_validation() -> None:
    """非法体素边长必须被拒绝，包括 bool——bool 是 int 的子类。"""

    mesh = _unit_cube([0, 0, 0], [10, 10, 10])
    for bad in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            compare_meshes_voxel(mesh, mesh, voxel_size=bad)


def test_max_voxels_scales_voxel_size_up() -> None:
    """体素总数超上限时我放大体素边长，而不是直接失败。"""

    mesh = _unit_cube([0, 0, 0], [10, 10, 10])
    result = compare_meshes_voxel(mesh, mesh, voxel_size=0.1, max_voxels=1000)
    assert result.voxel_size_mm > 0.1
    assert int(np.prod(result.grid_dims)) <= (
        1000 * 8
    )  # 取整后允许略超，但不能量级失控


def test_empty_candidate_reports_all_missing() -> None:
    """候选件为空 = 全部材料都没了。"""

    full = _unit_cube([0, 0, 0], [10, 10, 10])
    empty = TriangleMesh(
        vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)
    )
    result = compare_meshes_voxel(full, empty, voxel_size=1.0)
    assert result.missing_count == 1000
    assert result.iou == pytest.approx(0.0)


def test_result_is_json_serializable() -> None:
    """我要求结果能直接进 JSON，不能含 numpy 标量。"""

    import json

    full = _unit_cube([0, 0, 0], [10, 10, 10])
    result = compare_meshes_voxel(full, _unit_cube([0, 0, 0], [5, 10, 10]))
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "missing_count" in payload
    assert isinstance(result.to_dict()["voxel_size_mm"], float)
    assert isinstance(result.to_dict()["grid_dims"][0], int)
