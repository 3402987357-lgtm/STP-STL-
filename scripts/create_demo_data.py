"""我生成一个不依赖外部 CAD 软件的演示数据集。

算法原理：在没有甲方真实 CAD 的情况下也要能打通"检测 -> 报告"全链路，所以我手工
构造一对网格：完整闭合立方体作参考模型，缺顶面的立方体作候选模型。缺顶面是通过
移除立方体顶面那两个三角面实现的，于是候选模型出现一个开放边界环——这正是检测算法
要找的"缺失区域"。

三角面的顶点顺序按逆时针（从外侧看）给出，保证法向朝外；这一点很重要，因为后续的
配准、距离计算和可视化都依赖一致的朝向。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.geometry.stl import save_stl


def cube_mesh(size: float = 10.0, omit_top: bool = False) -> TriangleMesh:
    """我生成一个边长为 size 的闭合立方体或缺顶面立方体。"""

    # 八个角点：底面四个（z=0）加顶面四个（z=size）。
    vertices = np.asarray(
        [
            [0, 0, 0],
            [size, 0, 0],
            [size, size, 0],
            [0, size, 0],
            [0, 0, size],
            [size, 0, size],
            [size, size, size],
            [0, size, size],
        ],
        dtype=np.float64,
    )
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    # 前两个三角面正是顶面；去掉它们就得到一个带开放边界的缺顶立方体。
    if omit_top:
        faces = faces[:2] + faces[4:]
    return TriangleMesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def main() -> None:
    """我写出参考立方体与缺顶候选立方体两个 STL。"""
    output_dir = ROOT / "data" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_stl(cube_mesh(), output_dir / "reference_cube.stl", "reference_cube")
    save_stl(
        cube_mesh(omit_top=True),
        output_dir / "candidate_cube_missing_top.stl",
        "candidate_cube_missing_top",
    )
    print(f"我已经生成演示数据：{output_dir}")


if __name__ == "__main__":
    main()
