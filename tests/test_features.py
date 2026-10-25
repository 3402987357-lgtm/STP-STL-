from __future__ import annotations

import numpy as np

from cad_inspector.detection.features import recognize_mesh_features
from cad_inspector.geometry.mesh import TriangleMesh


def _box_mesh() -> TriangleMesh:
    vertices = np.asarray(
        [
            [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
            [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10],
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


def test我方识别盒体上下平面候选() -> None:
    result = recognize_mesh_features(_box_mesh())

    assert result.status == "BASELINE"
    assert result.top_plane is not None
    assert result.bottom_plane is not None
    assert result.top_plane.mean_z_mm > result.bottom_plane.mean_z_mm
    assert result.vertical_face_count > 0
