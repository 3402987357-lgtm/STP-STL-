"""我给三角面邻接构建上回归，锁定"共享边才算相邻"的语义。

这一组用例是在我把实现从 Python 循环改成整阵列运算时加的。改性能实现
最容易丢的是**边界语义**，而不是主流程，所以这里专门钉住三类边界：
退化面自环、孤立面、以及"共享一个顶点但不相接"的面。
"""

from __future__ import annotations

import numpy as np
import pytest

from cad_inspector.geometry.mesh import build_triangle_adjacency


def test我按共享边建立邻接() -> None:
    """面 0/1 共享边 (1,2)，面 1/2 共享边 (2,3)，面 3 完全孤立。"""

    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [5, 6, 7]])

    adjacency = build_triangle_adjacency(faces)

    assert [sorted(item) for item in adjacency] == [[1], [0, 2], [1], []]


def test我把退化面自环里的自身排除掉() -> None:
    """我锁定退化面不会把自己算成自己的邻居。

    退化面（两个顶点重合）会产生自环边。原实现用 `other != face_id`
    显式排除自身；我在向量化时踩过一次坑——按**组内下标**排除而不是
    按面号排除，会把真正的邻居一起丢掉。这条用例就是钉住这个区别。
    """

    faces = np.array([[19, 19, 56], [19, 56, 99]])

    adjacency = build_triangle_adjacency(faces)

    assert [sorted(item) for item in adjacency] == [[1], [0]]


def test我只共享一个顶点不算相邻() -> None:
    """共享顶点不等于共享边，否则连通区域会沿尖角错误地连成一片。"""

    faces = np.array([[0, 1, 2], [0, 3, 4], [0, 5, 6]])

    adjacency = build_triangle_adjacency(faces)

    assert [sorted(item) for item in adjacency] == [[], [], []]


def test我处理空网格() -> None:
    faces = np.zeros((0, 3), dtype=np.int64)

    assert build_triangle_adjacency(faces) == []


@pytest.mark.parametrize("size", [300, 3000, 20000])
def test我向量化结果与逐个建表的朴素实现一致(size: int) -> None:
    """我用一个独立的朴素实现交叉验证，不依赖被测代码的写法。

    这一条是性能重构的**等价性锚点**：朴素实现按"边 -> 面列表"逐条
    建表再反查，逻辑与被测代码完全不同，两边一致才能说明向量化
    没有悄悄改变语义。我特意让随机网格里混入退化面（顶点数取
    size//2 会让重复顶点自然出现），因为退化面正是上次踩坑的地方。
    """

    rng = np.random.default_rng(2026)
    faces = rng.integers(0, max(4, size // 2), size=(size, 3))

    vectorized = [sorted(item) for item in build_triangle_adjacency(faces)]

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(faces):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        for left, right in ((a, b), (b, c), (c, a)):
            key = (left, right) if left < right else (right, left)
            edge_to_faces.setdefault(key, []).append(face_id)
    naive: list[set[int]] = [set() for _ in range(len(faces))]
    for related in edge_to_faces.values():
        for face_id in related:
            naive[face_id].update(other for other in related if other != face_id)

    assert vectorized == [sorted(item) for item in naive]
