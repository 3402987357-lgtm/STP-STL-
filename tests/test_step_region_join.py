from __future__ import annotations

import json
from pathlib import Path

from cad_inspector.detection.step_region_join import (
    join_step_engineering_regions,
    load_step_engineering_region_map,
)
from cad_inspector.models import RegionResult
from cad_inspector.output.writers import write_csv


def _region() -> RegionResult:
    return RegionResult(
        region_id=1,
        defect_type="missing",
        center_xyz_mm=[0.0, 0.0, 0.0],
        bbox_min_xyz_mm=[0.0, 0.0, 0.0],
        bbox_max_xyz_mm=[1.0, 1.0, 1.0],
        area_mm2=1.0,
        estimated_volume_mm3=None,
        max_error_mm=1.0,
        mean_error_mm=1.0,
        p95_error_mm=1.0,
        triangle_count=1,
        confidence=0.8,
        reference_face_ids=[0],
        review_required=False,
        reference_step_face_ids=[12],
        candidate_step_face_ids=[13],
    )


def test我方把缺失区域关联到参考和候选工程区域(tmp_path: Path) -> None:
    path = tmp_path / "engineering.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "step-engineering-features-1.5",
                "status": "BASELINE",
                "regions": [
                    {
                        "region_id": 3,
                        "face_ids": [12],
                        "category_candidates": ["PLATE_COMPONENT_CANDIDATE"],
                    },
                    {
                        "region_id": 4,
                        "face_ids": [13],
                        "category_candidates": ["STRUCTURAL_COMPONENT_CANDIDATE"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    region_map = load_step_engineering_region_map(path)
    region = _region()

    metrics, warnings, review_required = join_step_engineering_regions(
        [region],
        reference_map=region_map,
        candidate_map=region_map,
    )

    assert metrics["enabled"] is True
    assert metrics["mapped_region_count"] == 1
    assert warnings == []
    assert review_required is False
    assert region.reference_step_region_ids == [3]
    assert region.candidate_step_region_ids == [4]
    assert region.reference_step_category_candidates == [
        "PLATE_COMPONENT_CANDIDATE"
    ]
    assert region.candidate_step_category_candidates == [
        "STRUCTURAL_COMPONENT_CANDIDATE"
    ]


def test我方损坏工程区域索引必须降级复核(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    region = _region()

    metrics, warnings, review_required = join_step_engineering_regions(
        [region],
        reference_map=load_step_engineering_region_map(path),
    )

    assert metrics["reference"]["valid"] is False
    assert review_required is True
    assert any("索引不完整" in warning for warning in warnings)


def test我方区域_csv写出工程区域回溯字段(tmp_path: Path) -> None:
    region = _region()
    region.reference_step_region_ids = [3]
    region.reference_step_category_candidates = [
        "PLATE_COMPONENT_CANDIDATE"
    ]
    result = type(
        "Result",
        (),
        {"regions": [region]},
    )()

    path = write_csv(result, tmp_path / "regions.csv")
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]

    assert "candidate_step_region_ids" in header
    assert "reference_step_category_candidates" in header


def test我方重复区域编号不会静默覆盖(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "step-engineering-features-1.5",
                "status": "BASELINE",
                "regions": [
                    {
                        "region_id": 3,
                        "face_ids": [12],
                        "category_candidates": ["FIRST"],
                    },
                    {
                        "region_id": 3,
                        "face_ids": [13],
                        "category_candidates": ["SECOND"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = load_step_engineering_region_map(path)

    assert result.valid is False
    assert result.face_to_region_ids == {12: (3,)}
    assert result.region_categories == {3: ("FIRST",)}
    assert any("重复的区域编号" in warning for warning in result.warnings)


def test我方空面列表和错误类别字段必须进入复核(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "step-engineering-features-1.5",
                "status": "BASELINE",
                "regions": [
                    {
                        "region_id": 3,
                        "face_ids": [],
                        "category_candidates": "PLATE_COMPONENT_CANDIDATE",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = load_step_engineering_region_map(path)

    assert result.valid is False
    assert result.face_to_region_ids == {}
    assert any("没有可回溯的 Face" in warning for warning in result.warnings)
    assert any("category_candidates 不是数组" in warning for warning in result.warnings)


def test我方拒绝字符串形式的区域和_face编号(tmp_path: Path) -> None:
    path = tmp_path / "string_ids.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "step-engineering-features-1.6",
                "status": "BASELINE",
                "regions": [
                    {
                        "region_id": "3",
                        "face_ids": ["12"],
                        "category_candidates": ["PLATE_COMPONENT_CANDIDATE"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = load_step_engineering_region_map(path)

    assert result.valid is False
    assert result.face_to_region_ids == {}
    assert result.region_categories == {}
    assert any("缺少正整数 region_id" in warning for warning in result.warnings)
