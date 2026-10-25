from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from cad_inspector.reproducibility import content_sha256


ROOT = Path(__file__).parents[1]
csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))


def test我方正式区域输出与当前源码schema一致() -> None:
    stage3_dir = ROOT / "outputs" / "stage3_regions_v3"
    engineering_dir = ROOT / "outputs" / "step_engineering"

    for name in ("BOTTLE", "mould_04"):
        stage3 = json.loads(
            (stage3_dir / f"{name}_step_regions_v3.json").read_text(
                encoding="utf-8"
            )
        )
        engineering = json.loads(
            (
                engineering_dir
                / name
                / f"{name}.engineering.json"
            ).read_text(encoding="utf-8")
        )
        assert stage3["schema_version"] == "step-feature-classification-1.11"
        assert engineering["schema_version"] == "step-engineering-features-2.5"
        assert stage3["content_sha256"] == content_sha256(stage3)
        assert engineering["content_sha256"] == content_sha256(engineering)
        assert len(stage3["regions"]) == len(engineering["regions"])
        assert all(
            "boundary_perimeter_length_mm" in region
            and "primary_boundary_perimeter_length_mm" in region
            and "primary_boundary_projected_area_mm2" in region
            and "primary_boundary_projection_status" in region
            and "primary_boundary_loop_role" in region
            and "primary_boundary_loop_relation_status" in region
            for region in stage3["regions"]
        )
        projection_statuses = {
            "UNAVAILABLE",
            "NOT_CLOSED",
            "MISSING_VERTEX_COORDINATES",
            "UNORDERED_COMPONENT",
            "INSUFFICIENT_POINTS",
            "PROJECTION_FAILED",
            "DEGENERATE_POINTS",
            "DEGENERATE",
            "NON_PLANAR_CANDIDATE",
            "COMPUTED",
            "PROJECTED_AREA_CONFLICT",
            "LOOP_RELATION_CONFLICT",
        }
        for region in stage3["regions"]:
            status = region["primary_boundary_projection_status"]
            area = region["primary_boundary_projected_area_mm2"]
            evidence = region["primary_boundary_projection_evidence"]
            assert status in projection_statuses
            assert area is None or (
                isinstance(area, (int, float))
                and math.isfinite(float(area))
                and float(area) >= 0.0
            )
            assert isinstance(evidence, dict)
            assert isinstance(evidence.get("projection_conflict"), bool)
            loop_evidence = region["primary_boundary_loop_relation_evidence"]
            assert isinstance(loop_evidence, dict)
            assert isinstance(loop_evidence.get("loop_relation_conflict"), bool)
            if status == "PROJECTED_AREA_CONFLICT":
                assert evidence["projection_conflict"] is True

        with (
            engineering_dir
            / name
            / f"{name}.regions.csv"
        ).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = 0
            max_field_length = 0
            for row in reader:
                rows += 1
                if row:
                    max_field_length = max(
                        max_field_length,
                        max(len(field) for field in row),
                    )
        assert len(header) == 47
        assert "主边界Wire投影面积_mm2" in header
        assert "主边界Wire投影状态" in header
        assert "主边界Wire投影证据" in header
        assert "主边界Wire环角色" in header
        assert "主边界Wire环关系状态" in header
        assert "主边界Wire环关系证据" in header
        assert rows == len(engineering["regions"])
        assert max_field_length < 32000


def test我方正式_BOM输出与当前源码schema一致() -> None:
    bom_json = ROOT / "outputs" / "bom_bottle" / "BOTTLE.bom.json"
    bom_csv = ROOT / "outputs" / "bom_bottle" / "BOTTLE.bom.csv"
    bom_xlsx = ROOT / "outputs" / "bom_bottle" / "BOTTLE.bom.xlsx"
    assert bom_json.exists()
    assert bom_csv.exists()
    assert bom_xlsx.exists()

    payload = json.loads(bom_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "mold-bom-1.9"
    assert payload["content_sha256"] == content_sha256(payload)
    assert payload["components"]
    for component in payload["components"]:
        assert "category_candidates" in component
        assert "category_review_required" in component
        assert "category_review_reason_codes" in component

    with bom_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)
    assert "类别候选" in header
    assert "类别需要复核" in header
    assert "类别复核原因" in header
    assert len(rows) == payload["component_count"]


def test我方正式Face特征输出与当前源码schema一致() -> None:
    for name in ("BOTTLE", "mould_04"):
        path = ROOT / "outputs" / f"step_face_features_{name.lower()}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["schema_version"] == "step-face-features-1.8"
        assert payload["content_sha256"] == content_sha256(payload)
        assert payload["face_count"] == len(payload["faces"])
        vertex_coordinates = payload["vertex_coordinates_mm"]
        assert vertex_coordinates

        for face in payload["faces"]:
            endpoints = face["edge_endpoint_vertex_ids"]
            assert set(map(int, endpoints)) == set(face["edge_ids"])
            for vertex_ids in endpoints.values():
                assert len(vertex_ids) in {1, 2}
                assert all(str(vertex_id) in vertex_coordinates for vertex_id in vertex_ids)
