"""我钉住 KD-tree 后端与暴力后端必须给出同一结果。

我为什么钉这条（2026-10-15）：我为了够任务书"8 秒/件"，改过 KD-tree 的
停止下界（想让它在小 k 处就判定安全、不再翻倍扩 k）。改动**已回退**，
但这类"加速"最容易在边界上漏解——下界一旦不保守，最近面就会被漏掉，
miss 掉真实破损。任务书要的是能用于返修决策的距离，漏解比慢严重得多。

所以我把"两个后端必须对得上"钉成断言：以后任何加速改动，
只要让 KD-tree 给出比暴力更远的距离，这里立刻红。
"""

from __future__ import annotations

import numpy as np

from cad_inspector.detection.distance import points_to_mesh
from cad_inspector.geometry.mesh import TriangleMesh


def _mesh_with_one_huge_triangle() -> TriangleMesh:
    """我造一个刻意混入超大三角形的网格。

    这样能复现真实件上"全局最大包围半径远大于绝大多数三角形"的情形
    （实测 测试集/8 的半径最大值 0.0792 是中位数 0.0331 的 2.4 倍），
    正是停止下界最容易写错的场景。
    """

    grid = np.linspace(0.0, 1.0, 6)
    vx, vy = np.meshgrid(grid, grid)
    bottom = np.column_stack([vx.ravel(), vy.ravel(), np.zeros(vx.size)])
    top = np.column_stack([vx.ravel(), vy.ravel(), np.ones(vx.size)])
    vertices = np.vstack([bottom, top])

    faces: list[list[int]] = []
    side = 6
    for layer in range(2):
        base = layer * side * side
        for row in range(side - 1):
            for col in range(side - 1):
                a = base + row * side + col
                b = a + 1
                c = a + side
                d = c + 1
                faces.append([a, b, c])
                faces.append([b, d, c])
    face_array = np.asarray(faces, dtype=np.int64)

    huge = np.array(
        [[-50.0, -50.0, 0.0], [50.0, -50.0, 0.0], [0.0, 50.0, 0.0]],
        dtype=np.float64,
    )
    vertices = np.vstack([vertices, huge])
    n = len(vertices)
    face_array = np.vstack(
        [face_array, np.array([[n - 3, n - 2, n - 1]], dtype=np.int64)]
    )
    return TriangleMesh(vertices=vertices, faces=face_array)


def test我方kdtree后端绝不给比暴力更远的距离() -> None:
    """KD-tree 可以更近（取到更优面），但绝不能更远——更远就是漏解。"""

    mesh = _mesh_with_one_huge_triangle()
    rng = np.random.default_rng(20261015)
    probe = rng.uniform(-0.3, 1.3, size=(1500, 3))

    d_kd, _, _ = points_to_mesh(
        probe, mesh, chunk_size=256, backend="scipy_kdtree", candidate_k=64
    )
    d_bf, _, _ = points_to_mesh(
        probe, mesh, chunk_size=256, backend="bruteforce", candidate_k=64
    )

    worse = d_kd > d_bf + 1e-9
    assert int(worse.sum()) == 0, (
        f"我方 KD-tree 后端在 {int(worse.sum())} 个点上给出了比暴力更远的距离，"
        "这说明停止下界不保守、漏掉了真正的最近面。"
        "漏解会让真实破损 miss 掉，比慢严重得多。"
    )


def test我方两个后端的距离在本网格上完全相等() -> None:
    """同一度量下两者应当逐点吻合（本用例不含并列最近面）。"""

    mesh = _mesh_with_one_huge_triangle()
    rng = np.random.default_rng(20261015)
    probe = rng.uniform(-0.3, 1.3, size=(1500, 3))

    d_kd, _, _ = points_to_mesh(
        probe, mesh, chunk_size=256, backend="scipy_kdtree", candidate_k=64
    )
    d_bf, _, _ = points_to_mesh(
        probe, mesh, chunk_size=256, backend="bruteforce", candidate_k=64
    )

    assert np.all(np.isfinite(d_kd)), "我方 KD-tree 后端给出了非有限距离"
    gap = float(np.abs(d_kd - d_bf).max())
    assert gap <= 1e-9, f"我方两个后端距离最大差 {gap:.3e}，超出可接受范围"
