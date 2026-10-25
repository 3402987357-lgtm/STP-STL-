"""我用完整参考网格生成可追踪的合成缺失区域和标签。

算法原理：甲方给的 STL 全部是水密闭合网格，没有现成孔洞可统计修补成功率。所以我用
"自造真值"的办法造孔洞：拿真实水密网格当完整模型，从种子面开始按共享边邻接做
**广度优先扩散**，删掉指定数量的连续三角面。这样造出来的孔洞是真值已知的——删了哪些
面、孔中心在哪、孔面积多大，全部记录在 ``SyntheticDefect`` 里，可以直接当标签用。

用 BFS 而不是随机选面，是为了保证删掉的是**一块连续区域**，模拟真实破损（掉一块），
而不是撒胡椒面式的随机碎面。面积按三角形面积加权算质心，保证缺陷中心是几何重心
而不是顶点平均。

我同时造两类缺陷，因为检测器本来就报两类：

- ``missing``：删掉一块面，模拟缺料。真值记在 ``removed_face_ids``。
- ``extra``：**复制并外推**一块面，模拟多料。真值记在
  ``added_face_ids``（候选网格上多出来的面下标）。

我为什么必须补 ``extra``：评估器原先只统计 missing，而生产管线对每件真实
件都会报 extra 区域。没有 extra 真值，"多余材料"这一类结论的准确率就是
**零证据**——报了多少、报对了多少，一个数都拿不出来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..geometry.mesh import TriangleMesh, build_triangle_adjacency


@dataclass
class SyntheticDefect:
    """我记录一次合成缺陷的来源和面级标签。"""

    candidate_mesh: TriangleMesh
    # 缺料：从参考网格上删掉的面下标。
    removed_face_ids: list[int]
    # 多料：候选网格上多出来的面下标（相对候选网格自身）。
    # 缺料型缺陷下这个列表为空，两种缺陷用不同字段而不是共用一个，
    # 避免调用方猜"这次的 id 是删的还是加的"。
    added_face_ids: list[int]
    defect_center_xyz_mm: list[float]
    defect_area_mm2: float
    defect_type: str


def _validate_seed(reference: TriangleMesh, seed_face_id: int, face_count: int) -> None:
    """我校验种子面与数量，两类缺陷共用同一套前置检查。

    我为什么把校验抽出来：缺料与多料是两条生成路径，若各写一份，
    下次改校验规则很容易只改一边，于是"删面拒绝越界种子、加面却放过"
    这种不一致会悄悄出现。
    """

    if len(reference.faces) == 0:
        raise ValueError("我无法从空网格生成合成缺陷。")
    if seed_face_id < 0 or seed_face_id >= len(reference.faces):
        raise ValueError("我要求 seed_face_id 位于参考网格面索引范围内。")
    if face_count < 1:
        raise ValueError("我要求 face_count 至少为 1。")


def _bfs_region(
    faces: np.ndarray,
    seed_face_id: int,
    face_count: int,
) -> list[int]:
    """我从种子面按共享边 BFS 扩出一块连续面。

    BFS 只能沿共享边扩散，所以网格不连通时扩不到另一块。静默返回更少
    （甚至 0）的面会造出与请求不符的假真值，所以不足即拒绝，
    绝不"有多少给多少"。
    """

    adjacency = build_triangle_adjacency(faces)
    selected: list[int] = []
    queue = [seed_face_id]
    visited = {seed_face_id}
    while queue and len(selected) < face_count:
        current = queue.pop(0)
        selected.append(current)
        for neighbor in sorted(adjacency[current]):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    if len(selected) != face_count:
        raise ValueError(
            f"我只从种子面 {seed_face_id} 扩散到 {len(selected)} 个连续三角面，"
            f"不足请求的 {face_count} 个；请检查网格是否连通。"
        )
    return selected


def _area_weighted_centroid(mesh: TriangleMesh, face_ids: Sequence[int]) -> list[float]:
    """我按三角形面积加权求质心，退化时回落到顶点平均。

    用面积加权而不是顶点平均：一块由大小悬殊的三角形拼成的区域，
    顶点平均会被大量小面拉偏，得到的"缺陷中心"不在几何重心上，
    拿它做定位误差评测会系统性偏大。
    """

    triangles = mesh.triangle_vertices[list(face_ids)]
    areas = mesh.triangle_areas[list(face_ids)]
    total_area = float(areas.sum())
    centroids = triangles.mean(axis=1)
    center = (
        (centroids * areas[:, None]).sum(axis=0) / total_area
        if total_area > 1e-12
        else centroids.mean(axis=0)
    )
    return np.asarray(center, dtype=np.float64).tolist()


def remove_face_region(
    reference: TriangleMesh,
    seed_face_id: int,
    face_count: int = 1,
    defect_type: str = "missing",
) -> SyntheticDefect:
    """我从共享边相邻面中删除一块连续区域，生成训练候选模型。"""

    # 输入先校验：空网格、越界种子面、非正数量都不接受，
    # 静默容错只会造出假真值。
    _validate_seed(reference, seed_face_id, face_count)
    if face_count > len(reference.faces):
        raise ValueError(
            f"我要求 face_count 不超过参考网格面数（{len(reference.faces)}），"
            f"当前请求 {face_count}。"
        )
    selected = _bfs_region(reference.faces, seed_face_id, face_count)
    # 删掉全部面会让候选网格变成空网格，下游检测直接失去意义。
    # 我放在 BFS **之后**判：BFS 扩散不足时（网格不连通）它自己的报错
    # 更有信息量，若在这里先拦就只会说"要留一个面"，把真正的
    # "只扩散到 N 个"原因盖掉。
    if len(selected) == len(reference.faces):
        raise ValueError(
            "我要求删除后至少保留一个三角面：参考网格共 "
            f"{len(reference.faces)} 个面，当前请求删除 {len(selected)} 个。"
        )
    # 保留掩码取反得到候选网格；顶点表整体保留，方便与参考网格直接对应。
    selected_set = set(selected)
    keep_mask = np.asarray(
        [face_id not in selected_set for face_id in range(len(reference.faces))],
        dtype=bool,
    )
    candidate = TriangleMesh(
        vertices=reference.vertices.copy(),
        faces=reference.faces[keep_mask].copy(),
        name=f"{reference.name}_synthetic_{defect_type}",
        metadata={
            **reference.metadata,
            "synthetic_defect": defect_type,
            "removed_face_ids": selected,
        },
    )
    return SyntheticDefect(
        candidate_mesh=candidate,
        removed_face_ids=selected,
        added_face_ids=[],
        defect_center_xyz_mm=_area_weighted_centroid(reference, selected),
        defect_area_mm2=float(reference.triangle_areas[selected].sum()),
        defect_type=defect_type,
    )


def add_face_region(
    reference: TriangleMesh,
    seed_face_id: int,
    face_count: int = 1,
    offset_mm: float = 1.0,
    defect_type: str = "extra",
) -> SyntheticDefect:
    """我复制一块连续面并沿其法向外推，造出"多出的料"。

    我为什么用"复制 + 外推"而不是凭空加面：

    检测器的 extra 判据是**反向采样**——从候选模型取点，到参考模型求最近
    距离，距离超过阈值才算多料。所以合成的多料必须满足两件事：
    一是它是候选网格上真实存在的一块面（否则采不到点），
    二是它离参考模型足够远（否则距离不超阈值，检测不到）。
    沿法向外推正好同时满足：面还在、位移可控、离参考面 offset_mm。

    offset_mm 必须**大于检测阈值**才检得到，我不在这里猜一个默认值去
    掩盖这个关系，调用方要按自己的 threshold_mm 来定。
    """

    _validate_seed(reference, seed_face_id, face_count)
    if not np.isfinite(offset_mm) or offset_mm <= 0.0:
        raise ValueError(
            f"我要求 offset_mm 为正的有限值，当前为 {offset_mm}；"
            "非正位移造不出离开参考表面的多料，检测器必然检不到。"
        )
    selected = _bfs_region(reference.faces, seed_face_id, face_count)
    # 我按**顶点**外推，而不是按三角形外推。这是我实测踩到的坑：
    # 逐三角形用各自的面法向位移时，相邻三角形的法向不同（曲面尤其明显，
    # 实测相邻法向差可达 0.1），于是本来共享的那个顶点被推到两个不同
    # 位置，共享边**断开**，连通分量把整块区域拆成 face_count 个单面碎块。
    # 检测器的 min_region_area_mm2 再把这些碎块逐个滤掉，
    # 最终表现为"一块多料都没检出"。
    # 改成按顶点位移、法向取相邻面的面积加权平均，共享顶点才会一起走。
    region_faces = reference.faces[selected]
    unique_vertices, inverse = np.unique(region_faces, return_inverse=True)
    inverse = inverse.reshape(-1, 3)
    face_normals = reference.triangle_normals[selected]
    lengths = np.linalg.norm(face_normals, axis=1)
    # 退化三角形没有法向：拿它算位移会得到零向量甚至 nan，
    # 造出来的"多料"与原始面重合，是真值污染。
    if np.any(lengths <= 1e-15):
        raise ValueError(
            "我发现所选区域里有退化三角形（法向为零），无法给出外推方向。"
        )
    unit_face_normals = face_normals / lengths[:, None]
    face_areas = reference.triangle_areas[selected]
    # 面积加权平均：大面对该顶点法向的贡献更大，
    # 简单算术平均会让一堆细长小面把方向带偏。
    vertex_normals = np.zeros((len(unique_vertices), 3), dtype=np.float64)
    vertex_weights = np.zeros(len(unique_vertices), dtype=np.float64)
    # 每个三角形把自己的（单位法向 × 面积）加到它的三个顶点上，
    # 再把面积加到权重上；最后相除即得面积加权平均法向。
    flat_vertex_ids = inverse.ravel()
    # (F,1,3) 广播到 (F,3,3)：同一个三角形的三个顶点各得到同一份贡献，
    # 再摊平成 (F*3, 3) 才能与 flat_vertex_ids 一一对应。
    contributions = np.repeat(
        unit_face_normals[:, None, :] * face_areas[:, None, None],
        3,
        axis=1,
    ).reshape(-1, 3)
    np.add.at(vertex_normals, flat_vertex_ids, contributions)
    np.add.at(vertex_weights, flat_vertex_ids, np.repeat(face_areas, 3))
    if np.any(vertex_weights <= 1e-15):
        raise ValueError("我发现所选区域里有顶点未被任何有效面引用。")
    vertex_normals /= vertex_weights[:, None]
    vertex_lengths = np.linalg.norm(vertex_normals, axis=1)
    if np.any(vertex_lengths <= 1e-15):
        raise ValueError(
            "我发现所选区域里有顶点法向被相反方向抵消为零，无法给出外推方向。"
        )
    vertex_normals /= vertex_lengths[:, None]
    # 新顶点 = 原顶点 + 平均法向 × 位移：共享顶点位移一致，拓扑得以保留。
    new_vertices = reference.vertices[unique_vertices] + vertex_normals * offset_mm
    new_faces = inverse
    base_vertex_count = len(reference.vertices)
    candidate_vertices = np.vstack([reference.vertices, new_vertices])
    candidate_faces = np.vstack([reference.faces, new_faces + base_vertex_count])
    # 真值是候选网格上新增的那段面下标：它们追加在 faces 表末尾，
    # 所以是"原有面数之后"的连续区间。
    base_face_count = len(reference.faces)
    added_face_ids = list(range(base_face_count, base_face_count + len(selected)))
    candidate = TriangleMesh(
        vertices=candidate_vertices,
        faces=candidate_faces,
        name=f"{reference.name}_synthetic_{defect_type}",
        metadata={
            **reference.metadata,
            "synthetic_defect": defect_type,
            "added_face_ids": added_face_ids,
            "offset_mm": float(offset_mm),
        },
    )
    return SyntheticDefect(
        candidate_mesh=candidate,
        removed_face_ids=[],
        added_face_ids=added_face_ids,
        # 中心取外推后那块面的质心：这才是多料在候选模型上实际的位置。
        defect_center_xyz_mm=_area_weighted_centroid(candidate, added_face_ids),
        defect_area_mm2=float(candidate.triangle_areas[added_face_ids].sum()),
        defect_type=defect_type,
    )
