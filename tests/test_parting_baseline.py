from __future__ import annotations

import numpy as np

from cad_inspector.detection.parting_line import extract_parting_line
from cad_inspector.detection.parting_surface import build_planar_parting_surface
from cad_inspector.domain.parting import PartingConfig
from cad_inspector.geometry.mesh import TriangleMesh


def _box_mesh() -> TriangleMesh:
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
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=int,
    )
    return TriangleMesh(vertices, faces, name="box")


def test我方从封闭网格提取水平闭合分型线() -> None:
    result = extract_parting_line(
        _box_mesh(),
        PartingConfig(plane_scan_count=3, plane_margin_ratio=0.1),
    )

    assert result.closed is True
    assert result.status == "BASELINE"
    assert result.length_mm > 0
    assert result.area_xy_mm2 > 0
    assert result.points_xyz_mm[0] == result.points_xyz_mm[-1]


def test我方从分型线生成平面网格基线() -> None:
    line = extract_parting_line(_box_mesh(), PartingConfig(plane_z_mm=5.0))
    surface = build_planar_parting_surface(line, PartingConfig(surface_margin_mm=2.0))

    assert surface.status == "BASELINE"
    assert len(surface.vertices_xyz_mm) == 4
    assert surface.faces == [[0, 1, 2], [0, 2, 3]]
    assert all(point[2] == 5.0 for point in surface.vertices_xyz_mm)
