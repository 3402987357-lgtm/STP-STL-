from __future__ import annotations

import numpy as np

from cad_inspector.domain.parting import PartingLineResult
from cad_inspector.detection.step_parting import StepPartingGeometry
from cad_inspector.geometry.stl import save_stl
from cad_inspector.parting_pipeline import run_parting


def test我方不会把需要复核的分型线汇总为_baseline(monkeypatch) -> None:
    from pathlib import Path
    import tempfile

    line = PartingLineResult(
        status="REVIEW_REQUIRED",
        method="test",
        plane_z_mm=5.0,
        points_xyz_mm=[
            [0.0, 0.0, 5.0],
            [10.0, 0.0, 5.0],
            [10.0, 10.0, 5.0],
            [0.0, 10.0, 5.0],
            [0.0, 0.0, 5.0],
        ],
        closed=True,
        length_mm=40.0,
        area_xy_mm2=100.0,
        score=100.4,
        warnings=["测试输入需要人工复核。"],
    )
    geometry = StepPartingGeometry(
        line=line,
        wire=None,
        face=None,
        bbox_min_xyz_mm=[0.0, 0.0, 0.0],
        bbox_max_xyz_mm=[10.0, 10.0, 10.0],
        section_wire_count=1,
        section_edge_count=4,
        shape_is_valid=False,
    )
    monkeypatch.setattr(
        "cad_inspector.parting_pipeline.extract_step_parting_geometry",
        lambda path, config: geometry,
    )

    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "invalid.step"
        source.write_text("placeholder", encoding="ascii")
        result = run_parting(source).result

    assert result.status == "REVIEW_REQUIRED"
    assert result.line is line
    assert result.surface is not None


def test我方分型结果保留输入哈希和格式元数据(tmp_path) -> None:
    source = tmp_path / "box.stl"
    from test_core import cube_mesh

    save_stl(cube_mesh(), source, "box")

    result = run_parting(source).result

    payload = result.to_dict()
    assert payload["schema_version"] == "parting-1.5"
    assert payload["input_metadata"]["name"] == "box.stl"
    assert len(payload["input_metadata"]["sha256"]) == 64
