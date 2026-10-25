from __future__ import annotations

import numpy as np
import pytest

from cad_inspector.detection.parting_line import extract_parting_line
from cad_inspector.detection.parting_surface import build_planar_parting_surface
from cad_inspector.domain.parting import PartingConfig, PartingLineResult
from cad_inspector.errors import GeometryProcessingError
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.validation.parting_quality import (
    planar_surface_diagnostics,
    parting_line_diagnostics,
    parting_line_warnings,
    surface_mesh_warnings,
)


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
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=int,
    )
    return TriangleMesh(vertices, faces, name="quality_box")


def test我方拒绝非整数网格顶点索引() -> None:
    with pytest.raises(GeometryProcessingError, match="faces"):
        TriangleMesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0.0, 1.5, 2.0]],
        )


def test我方对自交分型线给出质量警告() -> None:
    points = np.asarray(
        [
            [0, 0, 0],
            [10, 10, 0],
            [0, 10, 0],
            [10, 0, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )
    warnings = parting_line_warnings(
        points,
        closed=True,
        tolerance_mm=1e-6,
    )
    assert any("自交" in warning for warning in warnings)


def test我方对非共面分型面给出质量警告() -> None:
    warnings = surface_mesh_warnings(
        np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0.1],
                [0, 1, 0],
            ],
            dtype=float,
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int),
        tolerance_mm=1e-6,
    )
    assert any("共面" in warning for warning in warnings)


@pytest.mark.parametrize(
    "faces",
    [
        np.asarray([[0, 1, 4]], dtype=int),
        np.asarray([[-1, 1, 2]], dtype=int),
    ],
)
def test我方对越界分型面索引只给警告不抛异常(faces: np.ndarray) -> None:
    warnings = surface_mesh_warnings(
        np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=float,
        ),
        faces,
        tolerance_mm=1e-6,
    )
    assert any("索引越界" in warning for warning in warnings)


def test我方扫描数量为一时选择模型中间高度() -> None:
    result = extract_parting_line(
        _box_mesh(),
        PartingConfig(plane_scan_count=1),
    )
    assert result.plane_z_mm == pytest.approx(5.0)


def test我方恶意自交线不会标记为可靠平面() -> None:
    line = PartingLineResult(
        status="BASELINE",
        method="test",
        plane_z_mm=0.0,
        points_xyz_mm=[
            [0, 0, 0],
            [10, 10, 0],
            [0, 10, 0],
            [10, 0, 0],
            [0, 0, 0],
        ],
        closed=True,
        length_mm=1.0,
        area_xy_mm2=1.0,
        score=1.0,
    )
    surface = build_planar_parting_surface(line)
    assert surface.status == "REVIEW_REQUIRED"
    assert any("自交" in warning for warning in surface.warnings)


def test我方阶段5识别桥接候选和尖角() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [60.0, 0.0, 0.0],
            [70.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    diagnostics = parting_line_diagnostics(
        points,
        closed=True,
        tolerance_mm=1e-6,
    )

    assert diagnostics["bridge_required"] is True
    assert diagnostics["bridge_candidate_segments"] == [11]
    assert len(diagnostics["bridge_proposals"]) == 1
    proposal = diagnostics["bridge_proposals"][0]
    assert proposal["segment_index"] == 11
    assert proposal["method"] == "straight_linear_bridge_candidate"
    assert proposal["applied"] is False
    assert proposal["requires_engineering_review"] is True
    assert proposal["intermediate_points_xyz_mm"]
    assert diagnostics["sharp_corner_indices"]
    assert diagnostics["sharp_corners"]
    for corner in diagnostics["sharp_corners"]:
        assert corner["index"] in diagnostics["sharp_corner_indices"]
        assert np.allclose(
            corner["point_xyz_mm"],
            points[corner["index"]],
        )


def test我方阶段5对非有限输入只输出安全诊断() -> None:
    diagnostics = parting_line_diagnostics(
        np.asarray([[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0]]),
        closed=False,
        tolerance_mm=1e-6,
    )

    assert diagnostics["finite"] is False
    assert diagnostics["bridge_required"] is False
    assert diagnostics["sharp_corner_indices"] == []
    assert diagnostics["sharp_corners"] == []


def test我方阶段6确认平面分型面覆盖边界() -> None:
    diagnostics = planar_surface_diagnostics(
        np.asarray(
            [
                [0.0, 0.0, 5.0],
                [10.0, 0.0, 5.0],
                [10.0, 10.0, 5.0],
                [0.0, 10.0, 5.0],
            ]
        ),
        np.asarray(
            [
                [-2.0, -2.0, 5.0],
                [12.0, -2.0, 5.0],
                [12.0, 12.0, 5.0],
                [-2.0, 12.0, 5.0],
            ]
        ),
        margin_mm=2.0,
        tolerance_mm=1e-6,
    )

    assert diagnostics["finite"] is True
    assert diagnostics["contains_boundary"] is True
    assert diagnostics["surface_area_xy_mm2"] == pytest.approx(196.0)


def test我方阶段6拒绝不完整或非有限分型面诊断() -> None:
    diagnostics = planar_surface_diagnostics(
        np.asarray([[0.0, 0.0, 5.0], [np.nan, 1.0, 5.0]]),
        np.asarray([[0.0, 0.0, 5.0]]),
        margin_mm=1.0,
        tolerance_mm=1e-6,
    )

    assert diagnostics["finite"] is False
    assert diagnostics["contains_boundary"] is False


def test我方拒绝非正过渡曲面偏移参数() -> None:
    with pytest.raises(ValueError, match="transition_offset_mm"):
        PartingConfig(transition_offset_mm=0.0)
    with pytest.raises(ValueError, match="transition_offset_mm"):
        PartingConfig(transition_offset_mm=-1.0)


def test我方允许自动推导过渡曲面偏移参数() -> None:
    config = PartingConfig(transition_offset_mm=None)
    assert config.to_dict()["transition_offset_mm"] is None
