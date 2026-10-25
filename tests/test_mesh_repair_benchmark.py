from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_mesh_repair.py"
    spec = importlib.util.spec_from_file_location("benchmark_mesh_repair", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test我方挖洞种子固定且落在网格范围内() -> None:
    module = _load_module()
    seeds = [module.case_seed(100, index) for index in range(5)]
    assert seeds == [module.case_seed(100, index) for index in range(5)]
    assert all(0 <= seed < 100 for seed in seeds)
    assert len(set(seeds)) == len(seeds)


def test我方基准汇总区分补上与补不上() -> None:
    module = _load_module()
    records = [
        {"case": "a", "hole_closed": True, "repaired_region_count": 1,
         "requested_face_count": 1, "repair_status": "success",
         "elapsed_seconds": 1.0},
        {"case": "b", "hole_closed": False, "repaired_region_count": 1,
         "requested_face_count": 8, "repair_status": "review",
         "elapsed_seconds": 2.0},
    ]
    aggregate = module.aggregate(records)
    assert aggregate["case_count"] == 2
    assert aggregate["hole_closed_count"] == 1
    assert aggregate["hole_closed_ratio"] == 0.5
    assert aggregate["hole_closed_with_region_count"] == 1
    assert aggregate["by_face_count"]["8"]["hole_closed_count"] == 0
    assert aggregate["by_status"] == {"review": 1, "success": 1}


def test我方基准报告写出失败时保留旧文件(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "output"
    output_root.mkdir()
    summary_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    summary_path.write_text("old-json", encoding="utf-8")
    csv_path.write_text("old-csv", encoding="utf-8")

    with pytest.raises(ValueError):
        module.write_reports(
            output_root=output_root,
            data_root=tmp_path / "data",
            config=module.RepairConfig(),
            records=[{"case": "bad", "value": float("nan")}],
            skipped=[],
        )

    assert summary_path.read_text(encoding="utf-8") == "old-json"
    assert csv_path.read_text(encoding="utf-8") == "old-csv"


def test我方基准报告可序列化(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "output"
    module.write_reports(
        output_root=output_root,
        data_root=tmp_path / "data",
        config=module.RepairConfig(allow_nonconvex=True),
        records=[
            {
                "case": "样本/1#face8#1",
                "pair": "样本/1",
                "seed_face_id": 13,
                "requested_face_count": 8,
                "removed_face_count": 8,
                "removed_area_mm2": 0.5,
                "damaged_boundary_edge_count": 10,
                "repair_status": "success",
                "repaired_region_count": 1,
                "repaired_area_mm2": 0.5,
                "output_boundary_edge_count": 0,
                "hole_closed": True,
                "elapsed_seconds": 0.4,
                "warning_count": 0,
            }
        ],
        skipped=[{"pair": "样本/2", "reason": "不是水密网格"}],
    )
    payload = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == module.SCHEMA_VERSION
    assert payload["config"]["allow_nonconvex"] is True
    assert payload["aggregate"]["hole_closed_ratio"] == 1.0
    assert payload["skipped_mesh_count"] == 1
    assert "样本/1#face8#1" in (output_root / "summary.csv").read_text(
        encoding="utf-8-sig"
    )
