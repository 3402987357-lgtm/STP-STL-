from __future__ import annotations

from pathlib import Path

from cad_inspector.domain.parting import (
    PartingLineResult,
    PartingPipelineResult,
    PartingSurfaceResult,
)
from cad_inspector.parting_cli import main
from cad_inspector.parting_pipeline import PartingRun


def test我方parting_cli对_review_required返回非零(tmp_path: Path, monkeypatch) -> None:
    line = PartingLineResult(
        status="REVIEW_REQUIRED",
        method="test",
        plane_z_mm=0.0,
        points_xyz_mm=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        closed=True,
        length_mm=4.0,
        area_xy_mm2=1.0,
        score=1.0,
        warnings=["我要求人工复核。"],
    )
    surface = PartingSurfaceResult(
        status="REVIEW_REQUIRED",
        method="test",
        plane_z_mm=0.0,
        normal_xyz=[0.0, 0.0, 1.0],
        boundary_points_xyz_mm=line.points_xyz_mm,
        vertices_xyz_mm=[
            [-1.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        margin_mm=1.0,
    )
    result = PartingPipelineResult(
        status="REVIEW_REQUIRED",
        input_file=str(tmp_path / "box.stl"),
        backend="test",
        line=line,
        surface=surface,
    )
    monkeypatch.setattr(
        "cad_inspector.parting_cli.run_parting",
        lambda path, config: PartingRun(result),
    )

    exit_code = main(
        [
            str(tmp_path / "box.stl"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1


def test我方parting_cli传递过渡曲面配置(tmp_path: Path, monkeypatch) -> None:
    captured = {}
    line = PartingLineResult(
        status="BASELINE",
        method="test",
        plane_z_mm=0.0,
        points_xyz_mm=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        closed=True,
        length_mm=4.0,
        area_xy_mm2=1.0,
        score=1.0,
    )
    result = PartingPipelineResult(
        status="BASELINE",
        input_file=str(tmp_path / "box.stl"),
        backend="test",
        line=line,
        surface=None,
    )

    def fake_run(path, config):
        captured["config"] = config
        return PartingRun(result)

    monkeypatch.setattr("cad_inspector.parting_cli.run_parting", fake_run)
    assert (
        main(
            [
                str(tmp_path / "box.stl"),
                "--output-dir",
                str(tmp_path / "out"),
                "--section-tolerance-mm",
                "0.02",
                "--curve-samples-per-edge",
                "12",
                "--transition-offset-mm",
                "3.5",
            ]
        )
        == 0
    )
    config = captured["config"]
    assert config.section_tolerance_mm == 0.02
    assert config.curve_samples_per_edge == 12
    assert config.transition_offset_mm == 3.5
