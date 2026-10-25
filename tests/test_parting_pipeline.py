from __future__ import annotations

from pathlib import Path

import numpy as np

from cad_inspector.detection import build_planar_parting_surface, extract_parting_line
from cad_inspector.domain.parting import PartingConfig, PartingPipelineResult
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.output import write_parting_result


def _cube() -> TriangleMesh:
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
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices=vertices, faces=faces, name="cube")


def test我方分型线和分型面可以完整输出(tmp_path: Path) -> None:
    config = PartingConfig(plane_z_mm=5.0, surface_margin_mm=1.0)
    line = extract_parting_line(_cube(), config)
    surface = build_planar_parting_surface(line, config)
    result = PartingPipelineResult(
        status="BASELINE",
        input_file="cube.stl",
        backend="mesh_horizontal_section",
        line=line,
        surface=surface,
        warnings=line.warnings + surface.warnings,
        config=config,
    )

    artifacts = write_parting_result(result, tmp_path)

    assert line.closed
    assert abs(line.area_xy_mm2 - 100.0) < 1e-6
    assert surface.status == "BASELINE"
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["parting_line_csv"]).exists()
    assert Path(artifacts["sharp_corners_csv"]).exists()
    assert Path(artifacts["parting_surface_stl"]).exists()
    assert line.diagnostics["sharp_corners"]
