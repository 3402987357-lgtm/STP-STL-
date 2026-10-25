from __future__ import annotations

import numpy as np

from cad_inspector.domain.repair import RepairConfig
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.repair import repair_mesh_holes
from cad_inspector.validation.mesh_quality import (
    boundary_edges,
    boundary_loops,
)


def _cube_mesh(omit_top: bool = False) -> TriangleMesh:
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


def test我方可以识别缺顶面的单个边界环() -> None:
    mesh = _cube_mesh(omit_top=True)
    assert len(boundary_edges(mesh)) == 4
    loops = boundary_loops(mesh)
    assert len(loops) == 1
    assert len(loops[0]) == 4


def test我方可以安全填补简单平面孔洞() -> None:
    mesh = _cube_mesh(omit_top=True)
    repaired, result = repair_mesh_holes(
        mesh,
        RepairConfig(
            max_hole_area_mm2=200.0,
            max_planarity_error_mm=0.01,
        ),
    )
    assert result.status == "success"
    assert result.repaired_regions[0].status == "repaired"
    assert len(repaired.faces) == len(mesh.faces) + 4
    assert len(boundary_edges(repaired)) == 0
    assert repaired.total_area == mesh.total_area + 100.0


def test我方拒绝超过安全面积的孔洞() -> None:
    mesh = _cube_mesh(omit_top=True)
    _, result = repair_mesh_holes(
        mesh,
        RepairConfig(max_hole_area_mm2=10.0),
    )
    assert result.status == "review"
    assert result.repaired_regions[0].status == "manual_review_required"
