from __future__ import annotations

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from cad_inspector.detection.distance import points_to_mesh
from cad_inspector.domain.repair import RepairConfig
from cad_inspector.errors import GeometryProcessingError
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.geometry.stl import load_stl, save_stl
from cad_inspector.io.file_validation import validate_mesh
from cad_inspector.models import InspectionConfig
from cad_inspector.pipeline import inspect_pair
from cad_inspector.repair import repair_mesh_holes
from cad_inspector.validation.mesh_quality import boundary_loops

from test_core import cube_mesh


def _write_binary_stl(path: Path, triangle_count: int = 1) -> None:
    prefix = b"solid binary facet vertex"
    header = prefix + b" " * (80 - len(prefix))
    payload = bytearray(header)
    payload.extend(struct.pack("<I", triangle_count))
    for _ in range(triangle_count):
        payload.extend(
            struct.pack(
                "<12fH",
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0,
            )
        )
    path.write_bytes(bytes(payload))


def test我方拒绝空文件和截断_binary_stl(tmp_path: Path) -> None:
    empty = tmp_path / "empty.stl"
    empty.write_bytes(b"")
    with pytest.raises(GeometryProcessingError):
        load_stl(empty)

    valid = tmp_path / "valid.stl"
    _write_binary_stl(valid)
    truncated = tmp_path / "truncated.stl"
    truncated.write_bytes(valid.read_bytes()[:-1])
    with pytest.raises(GeometryProcessingError, match="长度不匹配"):
        load_stl(truncated)


def test我方拒绝凑够_vertex_但缺少_endfacet的_ascii_stl(tmp_path: Path) -> None:
    path = tmp_path / "truncated_ascii.stl"
    path.write_text(
        "solid broken\n"
        "  facet normal 0 0 1\n"
        "    outer loop\n"
        "      vertex 0 0 0\n"
        "      vertex 1 0 0\n"
        "      vertex 0 1 0\n"
        "    endloop\n"
        "endsolid broken\n",
        encoding="ascii",
    )
    with pytest.raises(GeometryProcessingError, match="结构不完整"):
        load_stl(path)


def test我方能识别_header伪装的_binary_stl(tmp_path: Path) -> None:
    path = tmp_path / "solid_header_binary.stl"
    _write_binary_stl(path)
    mesh = load_stl(path)
    assert mesh.metadata["format"] == "binary_stl"
    assert len(mesh.faces) == 1


def test我方能识别头部带换行的_binary_stl(tmp_path: Path) -> None:
    # 二进制 STL 的 80 字节头部是自由文本，部分导出器会写成 "solid 名称\n"。
    # 头部里的换行不能把合法二进制 STL 误判成空 ASCII STL。
    for name, header_text in (("lf", b"solid part\n"), ("crlf", b"solid part\r\n")):
        path = tmp_path / f"header_newline_{name}.stl"
        payload = bytearray(header_text + b" " * (80 - len(header_text)))
        payload.extend(struct.pack("<I", 1))
        payload.extend(
            struct.pack(
                "<12fH",
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0,
            )
        )
        path.write_bytes(bytes(payload))
        mesh = load_stl(path)
        assert mesh.metadata["format"] == "binary_stl"
        assert len(mesh.faces) == 1


def test我方拒绝全退化和非有限坐标网格() -> None:
    degenerate = TriangleMesh(
        vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        faces=[[0, 1, 2]],
    )
    with pytest.raises(Exception, match="全部是退化"):
        validate_mesh(degenerate, "极端")
    with pytest.raises(GeometryProcessingError, match="NaN"):
        TriangleMesh(
            vertices=[[0, 0, 0], [np.nan, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
        )


def test我方坏网格自动距离后端回退且不产生_runtime_warning() -> None:
    mesh = TriangleMesh(
        vertices=[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        faces=[[0, 1, 2], [0, 3, 3]],
    )
    points = np.asarray([[0.2, 0.2, 0.5]], dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = points_to_mesh(points, mesh, backend="auto")
    assert actual[0][0] == pytest.approx(0.5)
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


def test我方把非流形和未知单位升级为_review(tmp_path: Path) -> None:
    reference = cube_mesh()
    candidate = TriangleMesh(
        vertices=np.vstack(
            [
                reference.vertices,
            ]
        ),
        faces=np.vstack([reference.faces, [[0, 2, 1], [0, 2, 1]]]),
    )
    candidate_path = tmp_path / "候选 模型.stl"
    reference_path = tmp_path / "参考 模型.stl"
    save_stl(candidate, candidate_path)
    save_stl(reference, reference_path)
    result = inspect_pair(
        candidate_path,
        reference_path,
        tmp_path / "output",
        InspectionConfig(registration_mode="none", distance_backend="bruteforce"),
    )
    assert result.status == "review"
    assert result.global_metrics["quality_review_required"] is True
    assert result.global_metrics["unit_review_required"] is True
    assert any("非流形边" in warning for warning in result.warnings)
    assert any("单位" in warning for warning in result.warnings)


def test我方修补异常边界时只给_review不强行生成面() -> None:
    mesh = cube_mesh(omit_top=True)
    warped_vertices = mesh.vertices.copy()
    warped_vertices[6, 2] += 1.0
    extra = TriangleMesh(warped_vertices, mesh.faces)
    repaired, result = repair_mesh_holes(
        extra,
        RepairConfig(max_planarity_error_mm=0.01),
    )
    assert result.status in {"partial", "review"}
    assert len(boundary_loops(extra)) >= 1
    assert len(repaired.faces) >= len(extra.faces)


def test我方修补质量失败时默认保留原始网格() -> None:
    mesh = cube_mesh(omit_top=True)
    warped_vertices = mesh.vertices.copy()
    warped_vertices[6, 2] += 1.0
    candidate = TriangleMesh(warped_vertices, mesh.faces)

    repaired, result = repair_mesh_holes(candidate)

    assert result.status in {"partial", "review"}
    np.testing.assert_array_equal(repaired.vertices, candidate.vertices)
    assert any("保留原始网格" in warning for warning in result.warnings)


def test我方修补配置拒绝非整数和非有限值() -> None:
    with pytest.raises(ValueError, match="max_boundary_edges"):
        RepairConfig(max_boundary_edges=3.5)
    with pytest.raises(ValueError, match="有限数值"):
        RepairConfig(max_hole_area_mm2=float("nan"))
    with pytest.raises(ValueError, match="allow_nonconvex"):
        RepairConfig(allow_nonconvex=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="allow_nonplanar_surface_candidate"):
        RepairConfig(allow_nonplanar_surface_candidate=1)  # type: ignore[arg-type]


def test我方连续性配置拒绝非法法向阈值和采样数() -> None:
    with pytest.raises(ValueError, match="max_g1_normal_angle_deg"):
        RepairConfig(max_g1_normal_angle_deg=0.0)
    with pytest.raises(ValueError, match="max_g1_normal_angle_deg"):
        RepairConfig(max_g1_normal_angle_deg=90.1)
    with pytest.raises(ValueError, match="continuity_sample_count"):
        RepairConfig(continuity_sample_count=2)
    with pytest.raises(ValueError, match="max_g2_curvature_delta_per_mm"):
        RepairConfig(max_g2_curvature_delta_per_mm=-0.001)


def test我方多边界孔洞和非凸孔洞进入可审计_review路径() -> None:
    mesh = cube_mesh(omit_top=True)
    disconnected = TriangleMesh(
        vertices=np.vstack([mesh.vertices, mesh.vertices + [30.0, 0.0, 0.0]]),
        faces=np.vstack([mesh.faces, mesh.faces + 8]),
    )
    _, multi_result = repair_mesh_holes(disconnected)
    assert len(multi_result.repaired_regions) >= 2

    nonconvex = TriangleMesh(
        vertices=[
            [0, 0, 0],
            [4, 0, 0],
            [4, 1, 0],
            [2, 0.5, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        faces=[
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 1],
        ],
    )
    # 我为什么在这里显式传 allow_nonconvex=False：这条用例钉的是"非凸环
    # 进入**可审计的 review 路径**"这件事本身，它有价值、要保留。但我把
    # `allow_nonconvex` 的**默认值改成了 True**（依据见 domain/repair.py
    # 该字段注释），默认下这个环现在会被补掉、拿不到 review 了。所以显式
    # 关掉开关，让这条覆盖**继续成立**，而不是删掉它。
    _, nonconvex_result = repair_mesh_holes(
        nonconvex, RepairConfig(allow_nonconvex=False)
    )
    assert nonconvex_result.status == "review"
    assert any(
        "不是凸多边形" in warning
        for warning in nonconvex_result.warnings
    )


def test我方大三角面批量查询保持有限结果() -> None:
    base = cube_mesh()
    mesh = TriangleMesh(
        vertices=base.vertices,
        faces=np.tile(base.faces, (40, 1)),
    )
    points = np.zeros((4097, 3), dtype=float)
    distances, face_ids, closest = points_to_mesh(
        points,
        mesh,
        backend="bruteforce",
        chunk_size=37,
    )
    assert distances.shape == (4097,)
    assert face_ids.shape == (4097,)
    assert closest.shape == (4097, 3)
    assert np.isfinite(distances).all()
    assert np.isfinite(closest).all()


def test我方对零三角面_STL_给出与事实相符的错误(tmp_path: Path) -> None:
    zero_triangles = tmp_path / "zero_binary.stl"
    payload = bytearray(b" " * 80)
    payload.extend(struct.pack("<I", 0))
    zero_triangles.write_bytes(bytes(payload))
    with pytest.raises(GeometryProcessingError, match="不包含任何三角面"):
        load_stl(zero_triangles)

    empty_ascii = tmp_path / "empty_ascii.stl"
    empty_ascii.write_text("solid empty\nendsolid empty\n", encoding="ascii")
    with pytest.raises(GeometryProcessingError, match="不包含任何三角面"):
        load_stl(empty_ascii)


def test我方空网格写回_STL_后读取仍给出明确错误(tmp_path: Path) -> None:
    empty = TriangleMesh(
        vertices=np.empty((0, 3), dtype=float),
        faces=np.empty((0, 3), dtype=int),
    )
    path = tmp_path / "empty_roundtrip.stl"
    save_stl(empty, path)

    assert path.exists()
    with pytest.raises(GeometryProcessingError, match="不包含任何三角面"):
        load_stl(path)
