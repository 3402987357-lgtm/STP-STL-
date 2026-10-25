from __future__ import annotations

import numpy as np
import pytest

from cad_inspector.cli import build_parser
from cad_inspector.detection.missing_regions import detect_missing_regions
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.models import InspectionConfig, RegistrationResult
from cad_inspector.registration import align_mesh


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


def test我方残差量到曲面而不是量到顶点() -> None:
    """同一个平面用两套三角化，顶点对不上但曲面完全重合，残差必须接近 0。

    这正是 STP 与 STL 的关系：同一零件、两套三角化。按"到最近顶点"量，
    残差会被采样间距支配（甲方件上实测 0.13~0.76 mm）；按"到曲面"量，
    应该是网格化误差量级（实测 0.004~0.005 mm）。这个用例把口径钉死。
    """

    reference = TriangleMesh(
        vertices=np.asarray(
            [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    # 同一个方板切成 8 个三角形，顶点位置与参考完全不同
    grid = np.linspace(0.0, 10.0, 3)
    vertices = [[x, y, 0.0] for y in grid for x in grid]
    faces = []
    for row in range(2):
        for col in range(2):
            a = row * 3 + col
            faces.append([a, a + 1, a + 4])
            faces.append([a, a + 4, a + 3])
    candidate = TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    )

    _, result = align_mesh(
        candidate,
        reference,
        InspectionConfig(registration_mode="none"),
    )

    assert result.mean_residual_mm < 1e-6, (
        f"同一曲面两套三角化，残差应接近 0，实际 {result.mean_residual_mm}"
    )


def _asymmetric_mesh() -> TriangleMesh:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 1.0, 3.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [1, 3, 4],
            [0, 3, 4],
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices, faces)


def test我方PCA对非对称旋转和大平移保持低残差() -> None:
    reference = _asymmetric_mesh()
    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    candidate = reference.transformed(
        rotation=rotation,
        translation=np.asarray([100.0, -50.0, 25.0]),
    )

    aligned, result = align_mesh(
        candidate,
        reference,
        InspectionConfig(registration_mode="pca", distance_backend="bruteforce"),
    )

    np.testing.assert_allclose(aligned.vertices, reference.vertices, atol=1e-8)
    assert result.quality == "good"
    assert result.p95_residual_mm < 1e-8


def test我方PCA尺度估计不依赖旋转后的包围盒() -> None:
    reference = _asymmetric_mesh()
    angle = 0.61
    rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    candidate = reference.transformed(
        rotation=rotation,
        translation=np.asarray([-20.0, 30.0, 40.0]),
        scale=2.0,
    )

    aligned, result = align_mesh(
        candidate,
        reference,
        InspectionConfig(
            registration_mode="pca",
            allow_scale=True,
            distance_backend="bruteforce",
        ),
    )

    np.testing.assert_allclose(aligned.vertices, reference.vertices, atol=1e-8)
    assert result.scale == 0.5
    assert result.scale_review_required is True


def test我方ICP复用PCA初始姿态并返回最终残差() -> None:
    reference = _asymmetric_mesh()
    angle = 1.13
    rotation = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    candidate = reference.transformed(
        rotation=rotation,
        translation=np.asarray([70.0, -10.0, 12.0]),
    )

    aligned, result = align_mesh(
        candidate,
        reference,
        InspectionConfig(registration_mode="icp"),
    )

    np.testing.assert_allclose(aligned.vertices, reference.vertices, atol=1e-8)
    assert result.p95_residual_mm < 1e-8


def test我方配准残差分块计算与朴素最近邻一致() -> None:
    rng = np.random.default_rng(20261006)
    reference = TriangleMesh(
        vertices=rng.normal(size=(700, 3)) * 10.0,
        faces=np.empty((0, 3), dtype=np.int64),
    )
    candidate = TriangleMesh(
        vertices=rng.normal(size=(1200, 3)) * 10.0 + np.asarray([3.0, -2.0, 1.5]),
        faces=np.empty((0, 3), dtype=np.int64),
    )

    aligned, result = align_mesh(
        candidate,
        reference,
        InspectionConfig(registration_mode="centroid"),
    )

    differences = aligned.vertices[:, None, :] - reference.vertices[None, :, :]
    naive = np.sqrt(np.min(np.sum(differences * differences, axis=2), axis=1))
    assert result.mean_residual_mm == pytest.approx(float(naive.mean()))
    assert result.p95_residual_mm == pytest.approx(float(np.percentile(naive, 95)))


def test我方none模式实测残差而不是写死零() -> None:
    """none 只表示"不做变换"，不表示"输入一定对齐"。

    写死零残差会让整体缩放 2.5 倍的候选报出 quality=good、平均残差 0，
    后续所有距离判定都建立在这个谎上。这里锁住"错位必须被看出来"。
    """

    reference = _asymmetric_mesh()
    identical = TriangleMesh(
        vertices=reference.vertices.copy(),
        faces=reference.faces.copy(),
    )
    scaled = reference.transformed(scale=2.5)

    _, exact = align_mesh(
        identical,
        reference,
        InspectionConfig(registration_mode="none"),
    )
    _, misaligned = align_mesh(
        scaled,
        reference,
        InspectionConfig(registration_mode="none"),
    )

    assert exact.quality == "good"
    assert exact.p95_residual_mm < 1e-8
    assert misaligned.mean_residual_mm > 0.1
    assert misaligned.quality in {"warning", "bad"}


def test我方采样不足时全局误差不把未采样面当成零() -> None:
    reference = TriangleMesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [10.0, 2.0, 0.0],
        ],
        faces=[[0, 1, 2], [3, 4, 5]],
        metadata={"unit": "mm"},
    )
    candidate = TriangleMesh(
        vertices=[
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [10.0, 2.0, 0.0],
        ],
        faces=[[0, 1, 2], [3, 4, 5]],
        metadata={"unit": "mm"},
    )

    _, metrics, _, _ = detect_missing_regions(
        reference,
        candidate,
        InspectionConfig(
            registration_mode="none",
            max_query_points=1,
            min_region_area_mm2=0.0,
            distance_backend="bruteforce",
        ),
        _registration(),
    )

    assert metrics["reference_unsampled_face_count"] == 1
    assert metrics["sampled_reference_error_face_count"] == 1
    assert metrics["mean_error_mm"] == 1.0
    assert metrics["p95_error_mm"] == 1.0
    assert metrics["max_error_mm"] == 1.0


def test我方inspect命令暴露关键检测覆盖参数() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "inspect",
            "candidate.stl",
            "reference.stl",
            "--allow-scale",
            "--samples-per-triangle",
            "3",
            "--max-query-points",
            "99",
            "--distance-backend",
            "bruteforce",
            "--distance-candidate-k",
            "7",
            "--distance-chunk-size",
            "11",
            "--no-bidirectional-check",
            "--unit-policy",
            "require_mm",
            "--write-intermediate",
            "--no-colored-mesh",
        ]
    )

    assert args.allow_scale is True
    assert args.samples_per_triangle == 3
    assert args.max_query_points == 99
    assert args.distance_backend == "bruteforce"
    assert args.distance_candidate_k == 7
    assert args.distance_chunk_size == 11
    assert args.bidirectional_check is False
    assert args.unit_policy == "require_mm"
    assert args.write_intermediate is True
    assert args.no_colored_mesh is True


def test我方配置直接构造时拒绝非布尔开关() -> None:
    for field_name in (
        "allow_scale",
        "bidirectional_check",
        "directional_evidence_enabled",
        "write_intermediate",
        "write_colored_mesh",
    ):
        try:
            InspectionConfig(**{field_name: 1})  # type: ignore[arg-type]
        except ValueError as exc:
            assert field_name in str(exc)
        else:
            raise AssertionError(f"{field_name} 没有拒绝非布尔值")
