"""我验证网格、距离、区域聚类和 STL 读写的最小闭环。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cad_inspector.detection.distance import closest_points_on_triangle
from cad_inspector.detection.missing_regions import detect_missing_regions
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.geometry.stl import load_stl, save_stl
from cad_inspector.models import InspectionConfig, RegistrationResult


def cube_mesh(omit_top: bool = False) -> TriangleMesh:
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
        dtype=np.float64,
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


class CoreTest(unittest.TestCase):
    def test_closest_point_on_triangle(self) -> None:
        triangle = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        points = np.asarray([[0.25, 0.25, 1.0]], dtype=float)
        closest = closest_points_on_triangle(points, triangle)
        np.testing.assert_allclose(closest, [[0.25, 0.25, 0.0]], atol=1e-8)

    def test_missing_top_is_one_region(self) -> None:
        reference = cube_mesh()
        candidate = cube_mesh(omit_top=True)
        candidate.metadata["face_ids_by_triangle"] = list(
            range(1, len(candidate.faces) + 1)
        )
        registration = RegistrationResult(
            mode="none",
            rotation_matrix=np.eye(3).tolist(),
            translation_mm=[0.0, 0.0, 0.0],
            scale=1.0,
            mean_residual_mm=0.0,
            p95_residual_mm=0.0,
            quality="good",
        )
        config = InspectionConfig(
            threshold_mm=0.15,
            min_region_area_mm2=0.1,
            min_triangle_count=1,
            registration_mode="none",
        )
        regions, metrics, _, face_distances = detect_missing_regions(
            reference, candidate, config, registration
        )
        self.assertEqual(len(regions), 1)
        self.assertTrue(
            regions[0].region_geometry_signature.startswith("meshregion-")
        )
        self.assertAlmostEqual(regions[0].area_mm2, 100.0, places=6)
        self.assertEqual(metrics["missing_region_count"], 1)
        self.assertEqual(int(np.sum(face_distances >= 0.15)), 2)
        self.assertTrue(regions[0].candidate_face_ids)
        self.assertTrue(regions[0].candidate_step_face_ids)

    def test_ascii_stl_round_trip(self) -> None:
        mesh = cube_mesh()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cube.stl"
            save_stl(mesh, path)
            loaded = load_stl(path)
            self.assertEqual(len(loaded.faces), len(mesh.faces))
            self.assertEqual(len(loaded.vertices), len(mesh.vertices))


if __name__ == "__main__":
    unittest.main()
