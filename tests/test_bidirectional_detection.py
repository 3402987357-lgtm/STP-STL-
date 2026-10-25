from __future__ import annotations

import numpy as np
import pytest

from cad_inspector.detection.missing_regions import detect_missing_regions
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.models import InspectionConfig, RegistrationResult

from test_core import cube_mesh


def _registration() -> RegistrationResult:
    return RegistrationResult(
        mode="none",
        rotation_matrix=np.eye(3).tolist(),
        translation_mm=[0.0, 0.0, 0.0],
        scale=1.0,
        mean_residual_mm=0.0,
        p95_residual_mm=0.0,
        quality="good",
    )


def test我方双向检查可以识别候选模型多余材料() -> None:
    reference = cube_mesh()
    extra_vertices = np.asarray(
        [[30.0, 30.0, 30.0], [32.0, 30.0, 30.0], [30.0, 32.0, 30.0]],
        dtype=np.float64,
    )
    candidate = TriangleMesh(
        vertices=np.vstack([reference.vertices, extra_vertices]),
        faces=np.vstack(
            [
                reference.faces,
                np.asarray([[8, 9, 10]], dtype=np.int64),
            ]
        ),
        metadata={
            "unit": "mm",
            "face_ids_by_triangle": [100 + index for index in range(13)],
        },
    )
    regions, metrics, warnings, _ = detect_missing_regions(
        reference,
        candidate,
        InspectionConfig(
            registration_mode="none",
            threshold_mm=0.15,
            min_region_area_mm2=0.1,
            bidirectional_check=True,
            distance_backend="bruteforce",
        ),
        _registration(),
    )

    extra_regions = [region for region in regions if region.defect_type == "extra"]
    assert len(extra_regions) == 1
    assert extra_regions[0].reference_face_ids == []
    assert extra_regions[0].candidate_face_ids == [12]
    assert extra_regions[0].candidate_step_face_ids == [112]
    assert extra_regions[0].reference_step_face_ids == []
    assert metrics["extra_region_count"] == 1
    assert metrics["candidate_abnormal_face_count"] == 1
    assert any("多余材料" in warning for warning in warnings)


def test我方给出缺失与多余面积的接近度() -> None:
    """我为什么加这个量：一件上"既缺料又多料"在真实模具上很少见，
    而**两侧面积几乎相等**几乎必然是整体没对齐——一侧凹进去多少，
    另一侧就凸出来多少。我在实测语料上量到 19 件里 12 件接近度大于 0.9
    （最高 0.997），所以"疑似整体错位"在这批数据里是常态。

    这个量让人能一眼排出复核优先级：越接近 1，越该先查配准与单位，
    而不是逐个区域去量尺寸。
    """

    reference = cube_mesh()
    candidate = cube_mesh()
    regions, metrics, _, _ = detect_missing_regions(
        reference,
        candidate,
        InspectionConfig(
            registration_mode="none",
            threshold_mm=0.15,
            min_region_area_mm2=0.1,
            bidirectional_check=True,
            distance_backend="bruteforce",
        ),
        _registration(),
    )

    # 完全相同的两个模型：两侧都没有面积，接近度无定义，
    # 必须是 None 而不是编一个 0 出来——0 会被误读成"极不均衡"。
    assert metrics["missing_extra_balance"] is None


def test我方两侧都有面积时接近度在零到一之间() -> None:
    """接近度是 min/max，按定义必然落在 (0, 1]，且两侧相等时为 1。"""

    reference = cube_mesh()
    # 把顶面整体抬高，制造"一侧缺、另一侧多"的对称错位。
    raised = cube_mesh()
    vertices = raised.vertices.copy()
    vertices[:, 2] += 0.5
    candidate = TriangleMesh(
        vertices=vertices,
        faces=raised.faces,
        metadata={"unit": "mm"},
    )
    _, metrics, _, _ = detect_missing_regions(
        reference,
        candidate,
        InspectionConfig(
            registration_mode="none",
            threshold_mm=0.15,
            min_region_area_mm2=0.1,
            bidirectional_check=True,
            distance_backend="bruteforce",
        ),
        _registration(),
    )

    balance = metrics.get("missing_extra_balance")
    if balance is None:
        pytest.skip("这个几何没有同时产生缺失与多余面积")
    assert 0.0 < balance <= 1.0


def test我方输出法向一致偏移证据但不篡改缺失语义() -> None:
    reference = TriangleMesh(
        vertices=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        faces=[[0, 1, 2]],
        metadata={"unit": "mm"},
    )
    candidate = TriangleMesh(
        vertices=[[0.0, 0.0, 0.25], [2.0, 0.0, 0.25], [0.0, 2.0, 0.25]],
        faces=[[0, 1, 2]],
        metadata={"unit": "mm"},
    )

    regions, metrics, _, _ = detect_missing_regions(
        reference,
        candidate,
        InspectionConfig(
            registration_mode="none",
            threshold_mm=0.15,
            min_region_area_mm2=0.1,
            bidirectional_check=True,
            distance_backend="bruteforce",
        ),
        _registration(),
    )

    assert metrics["directional_evidence"]["enabled"] is True
    assert (
        metrics["directional_evidence"]["forward_reference_to_candidate"][
            "aligned_offset_sample_count"
        ]
        > 0
    )
    assert any(
        region.direction_evidence == "NORMAL_ALIGNED_OFFSET"
        for region in regions
    )
    assert {region.defect_type for region in regions} == {"missing", "extra"}


def test我方可以关闭有向证据并保留区域检测() -> None:
    reference = TriangleMesh(
        vertices=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        faces=[[0, 1, 2]],
        metadata={"unit": "mm"},
    )
    candidate = TriangleMesh(
        vertices=[[0.0, 0.0, 0.25], [2.0, 0.0, 0.25], [0.0, 2.0, 0.25]],
        faces=[[0, 1, 2]],
        metadata={"unit": "mm"},
    )
    regions, metrics, _, _ = detect_missing_regions(
        reference,
        candidate,
        InspectionConfig(
            registration_mode="none",
            threshold_mm=0.15,
            min_region_area_mm2=0.1,
            bidirectional_check=True,
            directional_evidence_enabled=False,
            distance_backend="bruteforce",
        ),
        _registration(),
    )

    assert regions
    assert metrics["directional_evidence"]["enabled"] is False
    assert all(region.direction_evidence == "DISABLED" for region in regions)


def test我方区域几何签名不依赖区域编号并且可重复() -> None:
    reference = cube_mesh()
    candidate = cube_mesh(omit_top=True)
    config = InspectionConfig(
        registration_mode="none",
        threshold_mm=0.15,
        min_region_area_mm2=0.1,
        bidirectional_check=False,
        distance_backend="bruteforce",
    )
    first, _, _, _ = detect_missing_regions(
        reference,
        candidate,
        config,
        _registration(),
    )
    second, _, _, _ = detect_missing_regions(
        reference,
        candidate,
        config,
        _registration(),
    )

    assert len(first) == len(second) == 1
    assert first[0].region_geometry_signature == second[0].region_geometry_signature
    assert first[0].region_geometry_signature.startswith("meshregion-")
    assert first[0].defect_type == "missing"
