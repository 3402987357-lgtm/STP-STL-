"""我给 TriangleMesh 的几何属性缓存上回归。

起因：triangle_vertices / triangle_normals / triangle_areas 原本是
每次访问都重算的 property。实测 训练集/3（候选侧 245022 面）里
_make_region 被调用 1205 次，光这三个属性就占 258.8s / 315.0s，
是整条链路最大的单点开销。缓存后端到端 624.6s -> 82.0s。

这里钉住三件事，缺一不可：
1. 缓存确实生效（同一对象）；
2. 缓存不改变任何数值（与未缓存路径逐位一致）；
3. 不同网格之间不共享缓存对象。
"""

from __future__ import annotations

import numpy as np
import pytest

from cad_inspector.geometry.mesh import TriangleMesh


def _mesh() -> TriangleMesh:
    return TriangleMesh(
        vertices=np.array(
            [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64),
    )


def test我缓存三个几何属性() -> None:
    """同一网格重复访问必须拿到同一个数组对象，否则缓存没生效。"""

    mesh = _mesh()
    assert mesh.triangle_vertices is mesh.triangle_vertices
    assert mesh.triangle_areas is mesh.triangle_areas
    assert mesh.triangle_normals is mesh.triangle_normals


def test缓存结果与原算法逐位一致() -> None:
    """我最在意这条：缓存只能加速，绝不能改变任何一个数。

    做法是不走缓存地重算一份真值，再与缓存结果按位比较。
    """

    mesh = _mesh()
    faces = mesh.faces
    expected_triangles = mesh.vertices[faces]
    cross = np.cross(
        expected_triangles[:, 1] - expected_triangles[:, 0],
        expected_triangles[:, 2] - expected_triangles[:, 0],
    )
    expected_areas = 0.5 * np.linalg.norm(cross, axis=1)
    lengths = np.linalg.norm(cross, axis=1)
    expected_normals = cross.copy()
    valid = lengths > 1e-15
    expected_normals[valid] /= lengths[valid, None]
    expected_normals[~valid] = 0.0

    assert np.array_equal(mesh.triangle_vertices, expected_triangles)
    assert np.array_equal(mesh.triangle_areas, expected_areas)
    assert np.array_equal(mesh.triangle_normals, expected_normals)


def test不同网格不共享缓存() -> None:
    """两个网格的缓存必须各自独立，否则会互相污染。"""

    a = _mesh()
    b = TriangleMesh(vertices=a.vertices.copy(), faces=a.faces.copy())
    assert a.triangle_vertices is not b.triangle_vertices
    assert a.triangle_areas is not b.triangle_areas


def test空网格的缓存不抛异常() -> None:
    """空网格要返回形状正确的空数组，缓存不能改变这个约定。"""

    mesh = TriangleMesh(
        vertices=np.zeros((0, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    assert mesh.triangle_vertices.shape == (0, 3, 3)
    assert mesh.triangle_areas.shape == (0,)
    assert mesh.triangle_normals.shape == (0, 3)
    # 空网格也要能重复访问而不炸。
    assert mesh.triangle_areas.shape == (0,)


def test退化面法向仍是零向量() -> None:
    """三点共线的退化面给零法向而不是 NaN——缓存不能改变这个约定。"""

    mesh = TriangleMesh(
        vertices=np.array(
            [[0.0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    normals = mesh.triangle_normals
    assert np.all(np.isfinite(normals))
    assert np.allclose(normals, 0.0)
    assert pytest.approx(float(mesh.triangle_areas[0])) == 0.0
