"""我集中验证异常输入、极限参数和跨后端一致性。"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from cad_inspector.detection.distance import (
    closest_points_on_triangle,
    closest_points_on_triangle_batch,
    points_to_mesh,
)
from cad_inspector.detection.missing_regions import detect_missing_regions
from cad_inspector.detection.missing_regions import (
    _map_step_face_ids,
    _step_face_mapping_diagnostics,
)
from cad_inspector.domain.parting import PartingConfig
from cad_inspector.errors import GeometryProcessingError, InputValidationError
from cad_inspector.geometry.mesh import TriangleMesh, mesh_quality_report
from cad_inspector.geometry.stl import load_stl
from cad_inspector.io.batch_probe import expand_inputs, run_batch
from cad_inspector.io.step_faces import StepFaceFeature, _face_semantic_evidence
from cad_inspector.io.step_probe import probe_step
from cad_inspector.models import InspectionConfig, RegistrationResult
from cad_inspector.registration import align_mesh
from cad_inspector.repair import repair_mesh_holes
from cad_inspector.validation.mesh_quality import boundary_loops

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


def _triangle_mesh(offset: float = 0.0) -> TriangleMesh:
    return TriangleMesh(
        vertices=[
            [offset + 0.0, 0.0, 0.0],
            [offset + 2.0, 0.0, 0.0],
            [offset + 0.0, 2.0, 0.0],
        ],
        faces=[[0, 1, 2]],
        metadata={"unit": "mm"},
    )


def _semantic_face() -> StepFaceFeature:
    return StepFaceFeature(
        face_id=1,
        surface_type="GeomAbs_Plane",
        area_mm2=1.0,
        center_xyz_mm=[0.0, 0.0, 0.0],
        normal_xyz=[1.0, 0.0, 0.0],
        bbox_min_xyz_mm=[-1.0, -1.0, -1.0],
        bbox_max_xyz_mm=[1.0, 1.0, 1.0],
        edge_count=0,
        edge_ids=[],
        orientation="FORWARD",
    )


class _TraceClassifier:
    def __init__(self, states: list[str]) -> None:
        self._states = states
        self._index = -1

    def Perform(self, point: object, tolerance: float) -> None:
        self._index += 1

    def State(self) -> str:
        return self._states[self._index]


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (["IN"] * 5 + ["OUT"] * 5, "EXTERNAL_CANDIDATE"),
        (
            ["IN"] * 5 + ["OUT", "IN", "IN", "OUT", "OUT"],
            "INTERNAL_CAVITY_CANDIDATE",
        ),
        (["ON"] * 5 + ["OUT"] * 5, "UNRESOLVED"),
        (["UNKNOWN"] * 10, "UNRESOLVED"),
    ],
)
def test我方实体语义采样轨迹覆盖_in_out_on_unknown(
    states: list[str],
    expected: str,
) -> None:
    evidence = _face_semantic_evidence(
        _semantic_face(),
        _TraceClassifier(states),
        lambda x, y, z: (x, y, z),
        [0.001, 0.01, 0.1, 1.0, 2.0],
        "IN",
        "ON",
        "OUT",
    )
    assert evidence["status"] == expected


@pytest.mark.parametrize(
    "bad_vertices",
    [
        [[0.0, 0.0, np.nan], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, np.inf], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[-np.inf, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ],
)
def test我方拒绝非有限网格坐标(bad_vertices: list[list[float]]) -> None:
    with pytest.raises(GeometryProcessingError, match="NaN 或无穷大"):
        TriangleMesh(bad_vertices, [[0, 1, 2]])


def test我方明确处理空网格和全退化网格() -> None:
    empty = TriangleMesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))
    assert mesh_quality_report(empty)["has_empty_geometry"] is True
    degenerate = TriangleMesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0, 1, 2]],
    )
    assert mesh_quality_report(degenerate)["degenerate_triangle_count"] == 1


def test我方拒绝非法距离查询点() -> None:
    with pytest.raises(ValueError, match="points"):
        points_to_mesh(np.asarray([1.0, 2.0, 3.0]), _triangle_mesh())
    with pytest.raises(ValueError, match="无穷大"):
        points_to_mesh(
            np.asarray([[0.0, np.nan, 0.0]]),
            _triangle_mesh(),
            backend="bruteforce",
        )


def test我方拒绝非整数计数和非法变换参数() -> None:
    mesh = _triangle_mesh()
    with pytest.raises(ValueError, match="candidate_k"):
        points_to_mesh(
            np.asarray([[0.0, 0.0, 0.0]]),
            mesh,
            candidate_k=1.5,
        )
    with pytest.raises(ValueError, match="samples_per_triangle"):
        mesh.sample_points(samples_per_triangle=1.5)
    with pytest.raises(GeometryProcessingError, match="rotation"):
        mesh.transformed(rotation=np.zeros((2, 2)))
    with pytest.raises(GeometryProcessingError, match="scale"):
        mesh.transformed(scale=0.0)
    with pytest.raises(ValueError, match="整数"):
        InspectionConfig(min_triangle_count=1.5)
    with pytest.raises(ValueError, match="有限数值"):
        InspectionConfig(threshold_mm="0.15")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="有限数值"):
        PartingConfig(endpoint_tolerance_mm=float("nan"))
    with pytest.raises(ValueError, match="整数"):
        PartingConfig(curve_samples_per_edge=2.5)


def test我方空网格配准和空目标距离进入明确审计路径() -> None:
    empty = TriangleMesh(
        np.empty((0, 3), dtype=float),
        np.empty((0, 3), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="空网格"):
        align_mesh(empty, _triangle_mesh(), InspectionConfig())
    distances, face_ids, closest = points_to_mesh(
        np.asarray([[0.0, 0.0, 0.0]]),
        empty,
        backend="scipy_kdtree",
    )
    assert np.isinf(distances[0])
    assert face_ids.tolist() == [-1]
    assert np.isnan(closest).all()


def test我方批量最近点拒绝非有限三角形() -> None:
    with pytest.raises(ValueError, match="NaN 或无穷大"):
        closest_points_on_triangle_batch(
            np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            np.asarray(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan, 1.0, 0.0]]],
                dtype=float,
            ),
        )


def test我方采样上限优先覆盖三角面() -> None:
    mesh = TriangleMesh(
        vertices=np.asarray(
            [
                [0, 0, 0], [1, 0, 0], [0, 1, 0],
                [10, 0, 0], [11, 0, 0], [10, 1, 0],
                [20, 0, 0], [21, 0, 0], [20, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64),
    )
    points, face_ids = mesh.sample_points(samples_per_triangle=3, max_points=3)
    assert len(points) == 3
    assert face_ids.tolist() == [0, 1, 2]


def test我方采样截断时返回人工复核状态() -> None:
    reference = TriangleMesh(
        vertices=np.asarray(
            [
                [0, 0, 0], [1, 0, 0], [0, 1, 0],
                [10, 0, 0], [11, 0, 0], [10, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        metadata={"unit": "mm"},
    )
    regions, metrics, warnings, _ = detect_missing_regions(
        reference,
        reference,
        InspectionConfig(
            registration_mode="none",
            max_query_points=1,
            bidirectional_check=True,
            distance_backend="bruteforce",
            min_region_area_mm2=0.0,
        ),
        _registration(),
    )
    assert regions == []
    assert metrics["reference_sampling_coverage_ratio"] == 0.5
    assert metrics["sampling_review_required"] is True
    assert metrics["status"] == "review"
    assert any("采样没有覆盖全部" in warning for warning in warnings)


def test我方过滤区域后仍保持缺失和多余区域编号唯一() -> None:
    base = cube_mesh()
    small_vertices = np.asarray(
        [[40.0, 40.0, 0.0], [41.0, 40.0, 0.0], [40.0, 41.0, 0.0]],
        dtype=float,
    )
    reference = TriangleMesh(
        vertices=np.vstack([base.vertices, small_vertices]),
        faces=np.vstack([base.faces, [[8, 9, 10]]]),
        metadata={"unit": "mm"},
    )
    extra_vertices = np.asarray(
        [[30.0, 30.0, 30.0], [32.0, 30.0, 30.0], [30.0, 32.0, 30.0]],
        dtype=float,
    )
    candidate = TriangleMesh(
        vertices=np.vstack([base.vertices, extra_vertices]),
        faces=np.vstack([base.faces, [[8, 9, 10]]]),
        metadata={"unit": "mm"},
    )
    regions, _, _, _ = detect_missing_regions(
        reference,
        candidate,
        InspectionConfig(
            registration_mode="none",
            distance_backend="bruteforce",
            min_region_area_mm2=1.0,
        ),
        _registration(),
    )
    assert regions
    assert all(region.defect_type == "extra" for region in regions)
    assert [region.region_id for region in regions] == list(
        range(1, len(regions) + 1)
    )
    assert len({region.region_id for region in regions}) == len(regions)


@pytest.mark.parametrize("candidate_k", [1, 2, 64, 10_000])
def test我方KD树极限候选数仍与暴力基线一致(candidate_k: int) -> None:
    rng = np.random.default_rng(1234)
    vertices = rng.normal(size=(90, 3))
    faces = np.asarray(
        [(index, index + 1, index + 2) for index in range(0, 87, 3)],
        dtype=np.int64,
    )
    mesh = TriangleMesh(vertices, faces)
    points = rng.normal(size=(31, 3))
    brute = points_to_mesh(points, mesh, backend="bruteforce", chunk_size=5)
    indexed = points_to_mesh(
        points,
        mesh,
        backend="scipy_kdtree",
        candidate_k=candidate_k,
        chunk_size=5,
    )
    np.testing.assert_allclose(indexed[0], brute[0], atol=1e-10)
    np.testing.assert_allclose(indexed[2], brute[2], atol=1e-10)


def test我方随机三角网格后端结果可复现() -> None:
    rng = np.random.default_rng(99)
    vertices = rng.uniform(-100.0, 100.0, size=(120, 3))
    faces = np.asarray(
        [(index, index + 1, index + 2) for index in range(0, 117, 3)],
        dtype=np.int64,
    )
    mesh = TriangleMesh(vertices, faces)
    points = rng.uniform(-120.0, 120.0, size=(64, 3))
    first = points_to_mesh(
        points,
        mesh,
        backend="scipy_kdtree",
        candidate_k=1,
        chunk_size=1,
    )
    second = points_to_mesh(
        points,
        mesh,
        backend="scipy_kdtree",
        candidate_k=1,
        chunk_size=64,
    )
    np.testing.assert_allclose(first[0], second[0], atol=1e-12)
    np.testing.assert_allclose(first[2], second[2], atol=1e-12)


def test我方向量化最近点覆盖普通和退化三角面() -> None:
    points = np.asarray(
        [[0.2, 0.2, 1.0], [3.0, 3.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=float,
    )
    expected = np.vstack(
        [
            closest_points_on_triangle(point[None, :], triangle)[0]
            for point, triangle in zip(points, triangles)
        ]
    )
    actual = closest_points_on_triangle_batch(points, triangles)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test我方配准可以处理大平移和尺度差异() -> None:
    reference = cube_mesh()
    candidate = reference.transformed(
        translation=np.asarray([1000.0, -500.0, 250.0]),
        scale=2.0,
    )
    aligned, registration = align_mesh(
        candidate,
        reference,
        InspectionConfig(
            registration_mode="centroid",
            allow_scale=True,
        ),
    )
    np.testing.assert_allclose(aligned.bbox_min, reference.bbox_min, atol=1e-12)
    np.testing.assert_allclose(aligned.bbox_max, reference.bbox_max, atol=1e-12)
    assert registration.quality == "good"


def test我方修补器拒绝非平面孔洞() -> None:
    base = cube_mesh(omit_top=True)
    nonplanar = base.vertices.copy()
    nonplanar[6, 2] += 1.0
    nonplanar_mesh = TriangleMesh(nonplanar, base.faces)
    _, result = repair_mesh_holes(nonplanar_mesh)
    assert result.status == "review"
    assert any("不够平面" in warning for warning in result.warnings)


def test我方修补器对非流形边界不自动处理() -> None:
    mesh = TriangleMesh(
        vertices=[
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 1, 1],
        ],
        faces=[[0, 1, 2], [0, 1, 3], [0, 1, 4]],
    )
    assert boundary_loops(mesh) == []
    _, result = repair_mesh_holes(mesh)
    assert result.status == "no_action"


def test我方拒绝截断binary_stl(tmp_path: Path) -> None:
    path = tmp_path / "truncated.stl"
    payload = bytearray(84)
    payload[:5] = b"solid"
    struct.pack_into("<I", payload, 80, 1)
    path.write_bytes(bytes(payload))
    with pytest.raises(Exception, match="Binary STL"):
        load_stl(path)


def test我方对损坏和非UTF8_STEP给出可审计状态(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.step"
    malformed.write_bytes(b"\xff\xfeISO-10303-21;\nHEADER;\n")
    result = probe_step(malformed)
    assert result.status.value == "PARTIAL"
    assert any(issue.code == "STEP_NON_UTF8_TEXT" for issue in result.issues)

    missing = tmp_path / "missing.step"
    missing.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\n", encoding="utf-8")
    missing_result = probe_step(missing)
    assert missing_result.status.value == "PARTIAL"
    assert any(issue.code == "STEP_SECTION_MISSING" for issue in missing_result.issues)


def test我方拒绝零值布尔值和超范围的_step_face映射() -> None:
    mesh = _triangle_mesh()
    mesh.metadata.update(
        {
            "face_count": 1,
            "face_ids_by_triangle": [0],
        }
    )
    zero_report = _step_face_mapping_diagnostics(mesh)
    assert zero_report["review_required"] is True
    assert zero_report["invalid_entry_count"] == 1
    assert _map_step_face_ids(mesh, [0]) == []

    mesh.metadata["face_ids_by_triangle"] = [True]
    bool_report = _step_face_mapping_diagnostics(mesh)
    assert bool_report["review_required"] is True
    assert bool_report["invalid_entry_count"] == 1
    assert _map_step_face_ids(mesh, [0]) == []

    mesh.metadata["face_ids_by_triangle"] = [2]
    range_report = _step_face_mapping_diagnostics(mesh)
    assert range_report["review_required"] is True
    assert range_report["out_of_range_entry_count"] == 1
    assert _map_step_face_ids(mesh, [0]) == []


def test我方发现_step_face映射长度和声明计数异常() -> None:
    mesh = _triangle_mesh()
    mesh.metadata.update(
        {
            "face_count": 1,
            "face_ids_by_triangle": [1, 1],
        }
    )
    report = _step_face_mapping_diagnostics(mesh)
    assert report["review_required"] is True
    assert report["length_matches"] is False
    assert report["declared_face_count"] == 1
    assert _map_step_face_ids(mesh, [0, 1]) == [1]

    mesh.metadata["face_count"] = "bad"
    invalid_count_report = _step_face_mapping_diagnostics(mesh)
    assert invalid_count_report["declared_face_count_valid"] is False
    assert invalid_count_report["review_required"] is True
    assert _map_step_face_ids(mesh, [0]) == []


def test我方空目录和重复路径批量探测稳定(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert expand_inputs([empty, empty / "unknown.bin"]) == []
    batch = run_batch([empty, empty])
    assert batch.total == 0


def test我方拒绝非法配置极限值() -> None:
    with pytest.raises(ValueError, match="有限数值"):
        InspectionConfig(threshold_mm=float("nan"))
    with pytest.raises(ValueError, match="大于 0"):
        InspectionConfig(mesh_linear_deflection_mm=0.0)
    with pytest.raises(ValueError, match="有限数值"):
        InspectionConfig(mesh_angular_deflection_deg=float("inf"))
    with pytest.raises(ValueError, match="max_input_file_size_bytes"):
        InspectionConfig(max_input_file_size_bytes=0)


def test我方配置文件不静默截断非法整数() -> None:
    with pytest.raises(ValueError, match="整数"):
        InspectionConfig.from_dict({"min_triangle_count": 1.5})


def test我方配置枚举字段拒绝非字符串() -> None:
    with pytest.raises(ValueError, match="registration_mode"):
        InspectionConfig(registration_mode=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="distance_backend"):
        InspectionConfig(distance_backend={})  # type: ignore[arg-type]


def test我方配置拒绝未知参数名而不是静默退回默认值() -> None:
    with pytest.raises(ValueError, match="不认识的参数"):
        InspectionConfig.from_dict({"threshold": 0.2})
    with pytest.raises(ValueError, match="min_triangle_counts"):
        InspectionConfig.from_dict(
            {"threshold_mm": 0.2, "min_triangle_counts": 2}
        )


def test我方配置文件加载拒绝把其它_JSON_当成检测配置(tmp_path: Path) -> None:
    from cad_inspector.config import load_config

    other = tmp_path / "input_formats.json"
    other.write_text(
        json.dumps({"step": {}, "stl": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="不认识的参数"):
        load_config(other)


def test我方输入大小上限拒绝超限文件(tmp_path: Path) -> None:
    source = tmp_path / "x.stp"
    source.write_text("ISO-10303-21;", encoding="utf-8")
    from cad_inspector.io.input_validator import validate_input_path

    with pytest.raises(InputValidationError) as exc_info:
        validate_input_path(source, max_size_bytes=1)
    assert "> 1 bytes" in str(exc_info.value)


def test我方损坏的_step_face映射只进入_review不抛异常() -> None:
    reference = cube_mesh()
    candidate = cube_mesh(omit_top=True)
    candidate.metadata["face_ids_by_triangle"] = [1, "bad", None, float("nan")]

    regions, metrics, warnings, _ = detect_missing_regions(
        reference,
        candidate,
        InspectionConfig(
            registration_mode="none",
            distance_backend="bruteforce",
            min_region_area_mm2=0.1,
        ),
        _registration(),
    )

    assert regions
    assert metrics["status"] == "review"
    assert metrics["step_face_mapping_review_required"] is True
    mapping = metrics["candidate_step_face_mapping"]
    assert mapping["mapping_length"] == 4
    assert mapping["invalid_entry_count"] == 3
    assert any("STEP Face 映射元数据" in warning for warning in warnings)


def test我方普通_stl没有_step映射时不被误判为_review() -> None:
    reference = cube_mesh()
    candidate = cube_mesh()

    regions, metrics, warnings, _ = detect_missing_regions(
        reference,
        candidate,
        InspectionConfig(
            registration_mode="none",
            distance_backend="bruteforce",
            min_region_area_mm2=0.1,
        ),
        _registration(),
    )

    assert regions == []
    assert metrics["status"] == "pass"
    assert metrics["step_face_mapping_review_required"] is False
    assert not any("STEP Face 映射元数据" in warning for warning in warnings)
