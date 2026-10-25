from __future__ import annotations

import json
from pathlib import Path

import pytest

from cad_inspector.detection.threshold import (
    bbox_diagonal_mm,
    recommend_threshold_mm,
)
from cad_inspector.geometry.stl import save_stl
from cad_inspector.models import InspectionConfig
from cad_inspector.pipeline import _threshold_warnings, inspect_pair

from test_core import cube_mesh


def _small_mesh():
    """我方构造一个约 8 mm 对角线的小件，复现真实样例的尺度。"""

    mesh = cube_mesh()
    mesh.vertices = mesh.vertices / 2.0 + 2.0
    return mesh


def test我方推荐阈值取尺度与残差分量的较大值() -> None:
    reference = cube_mesh()
    config = InspectionConfig()
    recommendation = recommend_threshold_mm(reference, 1.0, config)
    # 10 mm 立方体对角线约 17.32 mm，尺度分量 = 0.002 × 17.32 ≈ 0.0346 mm。
    assert recommendation.bbox_diagonal_mm == pytest.approx(
        bbox_diagonal_mm(reference)
    )
    assert recommendation.scale_component_mm == pytest.approx(0.0346, abs=1e-3)
    # 噪声分量 = 1.5 × 1.0 = 1.5 mm，压过尺度分量。
    assert recommendation.residual_component_mm == pytest.approx(1.5)
    assert recommendation.recommended_mm == pytest.approx(1.5)
    assert recommendation.residual_dominates is True


def test我方推荐阈值受绝对下限保护() -> None:
    reference = _small_mesh()
    recommendation = recommend_threshold_mm(reference, 0.0, InspectionConfig())
    # 残差为 0 时推荐值退回绝对下限，避免阈值退化成 0。
    assert recommendation.recommended_mm == pytest.approx(
        InspectionConfig().auto_threshold_floor_mm
    )
    assert recommendation.residual_dominates is False


def test我方绝对模式沿用配置阈值但记录推荐值() -> None:
    reference = _small_mesh()
    config = InspectionConfig(threshold_mm=0.15)
    recommendation = recommend_threshold_mm(reference, 0.37, config)
    assert recommendation.mode == "absolute"
    assert recommendation.effective_mm == pytest.approx(0.15)
    # 小件 + 大残差：推荐阈值应当远高于配置的绝对阈值。
    assert recommendation.recommended_mm == pytest.approx(0.555, abs=1e-3)
    assert recommendation.configured_below_recommended is True


def test我方自动模式改用推荐阈值() -> None:
    reference = _small_mesh()
    config = InspectionConfig(threshold_mode="auto", threshold_mm=0.15)
    recommendation = recommend_threshold_mm(reference, 0.37, config)
    assert recommendation.mode == "auto"
    assert recommendation.effective_mm == pytest.approx(
        recommendation.recommended_mm
    )
    assert recommendation.effective_mm > config.threshold_mm


def test我方配准残差过大时给出尺度过粗标记() -> None:
    reference = cube_mesh()
    diagonal = bbox_diagonal_mm(reference)
    # 残差超过对角线的 5%，配准精度不足以支撑可靠检测。
    recommendation = recommend_threshold_mm(
        reference,
        0.06 * diagonal,
        InspectionConfig(),
    )
    assert recommendation.scale_too_coarse is True
    assert _threshold_warnings(recommendation)


def test我方阈值配置拒绝非法值() -> None:
    with pytest.raises(ValueError):
        InspectionConfig(threshold_mode="relative")
    with pytest.raises(ValueError):
        InspectionConfig(auto_threshold_scale_ratio=0.0)
    with pytest.raises(ValueError):
        InspectionConfig(auto_threshold_residual_factor=-1.0)
    with pytest.raises(ValueError):
        InspectionConfig(auto_threshold_floor_mm=0.0)


def test我方阈值配置可以JSON往返() -> None:
    config = InspectionConfig(
        threshold_mode="auto",
        auto_threshold_scale_ratio=0.005,
        auto_threshold_residual_factor=2.0,
        auto_threshold_floor_mm=0.05,
    )
    restored = InspectionConfig.from_dict(config.to_dict())
    assert restored.threshold_mode == "auto"
    assert restored.auto_threshold_scale_ratio == pytest.approx(0.005)
    assert restored.auto_threshold_residual_factor == pytest.approx(2.0)
    assert restored.auto_threshold_floor_mm == pytest.approx(0.05)


def test我方阈值配置拒绝未知键() -> None:
    with pytest.raises(ValueError):
        InspectionConfig.from_dict({"threshold_modes": "auto"})


def test我方绝对模式在阈值偏低时给出复核警告() -> None:
    recommendation = recommend_threshold_mm(
        _small_mesh(),
        0.37,
        InspectionConfig(threshold_mm=0.15),
    )
    warnings = _threshold_warnings(recommendation)
    assert any("低于" in item and "threshold_mode=auto" in item for item in warnings)


def test我方检测管线记录并应用阈值审计(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.stl"
    candidate_path = tmp_path / "candidate.stl"
    save_stl(cube_mesh(), reference_path)
    save_stl(cube_mesh(omit_top=True), candidate_path)

    absolute = inspect_pair(
        candidate_path,
        reference_path,
        tmp_path / "absolute",
        config=InspectionConfig(threshold_mode="absolute"),
    )
    absolute_audit = absolute.global_metrics["threshold_audit"]
    assert absolute_audit["mode"] == "absolute"
    assert absolute.threshold_mm == pytest.approx(0.15)
    assert absolute_audit["effective_mm"] == pytest.approx(0.15)

    automatic = inspect_pair(
        candidate_path,
        reference_path,
        tmp_path / "auto",
        config=InspectionConfig(threshold_mode="auto"),
    )
    auto_audit = automatic.global_metrics["threshold_audit"]
    assert auto_audit["mode"] == "auto"
    assert automatic.threshold_mm == pytest.approx(auto_audit["effective_mm"])
    assert automatic.threshold_mm == pytest.approx(auto_audit["recommended_mm"])
    assert auto_audit["bbox_diagonal_mm"] > 0
    # 两套模式都应当检测到被删掉的顶面。
    assert any(region.defect_type == "missing" for region in absolute.regions)
    assert any(region.defect_type == "missing" for region in automatic.regions)


def test我方检测结果JSON带出阈值审计(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.stl"
    candidate_path = tmp_path / "candidate.stl"
    save_stl(cube_mesh(), reference_path)
    save_stl(cube_mesh(omit_top=True), candidate_path)
    inspect_pair(
        candidate_path,
        reference_path,
        tmp_path / "out",
        config=InspectionConfig(threshold_mode="auto"),
    )
    payload = json.loads(
        (tmp_path / "out" / "result.json").read_text(encoding="utf-8")
    )
    assert payload["global_metrics"]["threshold_audit"]["mode"] == "auto"
    assert payload["threshold_mm"] == pytest.approx(
        payload["global_metrics"]["threshold_audit"]["effective_mm"]
    )
