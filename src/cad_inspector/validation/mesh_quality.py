"""我提供局部修补需要的边界环和前后质量比较能力。

算法原理：网格修补的验收不能只看"补完了没有"，要同时看三个量：

一、边界边数。水密闭合网格里每条边恰好被两个三角面共享，所以只被一个面引用的边就是
   开放边。修补前后各数一次，边界边从 N 降到 0 才算真的补上了。这是最硬的判据。

二、质量是否变差。修补不能"把洞堵上、把网格搞烂"。所以我检查修补后的非流形边数和
   退化三角面数是否比修补前增加——只要增加就判 review，不管洞补得有多好。

三、新增面积是否符合预期。有预期孔洞面积时，用"实际新增面积 vs 预期面积"的误差与
   ``area_tolerance_ratio`` 比较；没有预期值时退化为"总面积变化是否超过容差"，并明确
   提示这是弱判据、建议人工复核。

边界环提取只接受**每个边界顶点恰好连两条边**的简单环；分叉边界直接返回空列表，
避免修补器把复杂拓扑误当成普通孔洞去填。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..geometry.mesh import TriangleMesh, mesh_quality_report


def edge_to_faces(mesh: TriangleMesh) -> dict[tuple[int, int], list[int]]:
    """我建立无向边到三角面编号的映射。"""

    # 边统一按 (小, 大) 存键，保证同一条边不因顶点顺序不同被当成两条。
    mapping: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, face in enumerate(mesh.faces):
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = (first, second) if first < second else (second, first)
            mapping[edge].append(face_id)
    return dict(mapping)


def boundary_edges(mesh: TriangleMesh) -> set[tuple[int, int]]:
    """我返回只被一个三角面使用的开放边。"""

    # 被一个面引用 = 开放边；被两个引用 = 正常流形边；三个及以上 = 非流形。
    return {
        edge
        for edge, related_faces in edge_to_faces(mesh).items()
        if len(related_faces) == 1
    }


def boundary_loops(mesh: TriangleMesh) -> list[list[int]]:
    """我按边界连接关系提取闭合顶点环。

    只有每个边界顶点恰好连接两条边时，我才把它认定为可遍历的简单环。
    分叉边界会被跳过，避免修补器把复杂拓扑误当成普通孔洞。
    """

    edges = boundary_edges(mesh)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    if not edges:
        return []
    # 分叉边界（顶点度不为 2）不是简单环，直接放弃，避免误填复杂拓扑。
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return []

    # 从编号最小的边出发遍历，保证同一网格每次得到相同的环顺序（结果可复现）。
    unused = set(edges)
    loops: list[list[int]] = []
    while unused:
        # 每次都从编号最小的剩余边起步：起点选法固定，
        # 同一份网格得到的环列表才完全可复现。
        first, second = min(unused)
        loop = [first]
        previous = first
        current = second
        unused.discard((first, second))
        while current != first:
            loop.append(current)
            # 候选邻点排除来路：不排除会沿原路退回，
            # 在两个点之间来回打转。
            candidates = sorted(
                neighbor
                for neighbor in adjacency[current]
                if neighbor != previous
            )
            # 没有可走的下一点说明这条路是断的，
            # 不是环，作废这一轮已走出的部分。
            if not candidates:
                loop = []
                break
            # 取编号最小的候选：与上面"从最小边起步"同一条理由，
            # 保证遍历顺序确定，不依赖集合的迭代顺序。
            next_vertex = candidates[0]
            # 边键排序后查：unused 里的键是排序后的顶点对，
            # 不排序会查不到，于是误判成"这条路断了"。
            edge = (
                (current, next_vertex)
                if current < next_vertex
                else (next_vertex, current)
            )
            # 这条边已用过、又没回到起点，说明走进了死路。
            # 唯独 next_vertex == first 是例外——那是收口的那条边，
            # 它在上一步已经作为起始边被丢弃了。
            if edge not in unused and next_vertex != first:
                loop = []
                break
            unused.discard(edge)
            previous, current = current, next_vertex
        # 至少三个点且确实走回了起点才算一个环：
        # 两个点组成的"环"是一条边，不构成可填充的边界。
        if len(loop) >= 3 and current == first:
            loops.append(loop)
    return loops


def compare_mesh_quality(
    before: TriangleMesh,
    after: TriangleMesh,
    area_tolerance_ratio: float = 0.05,
    expected_added_area_mm2: float | None = None,
) -> dict[str, Any]:
    """我比较修补前后的关键质量指标，并给出是否需要复核的结论。"""

    before_report = mesh_quality_report(before)
    after_report = mesh_quality_report(after)
    before_boundary = len(boundary_edges(before))
    after_boundary = len(boundary_edges(after))
    before_loops = len(boundary_loops(before))
    after_loops = len(boundary_loops(after))
    before_area = float(before.total_area)
    after_area = float(after.total_area)
    area_ratio = (
        abs(after_area - before_area) / before_area
        if before_area > 1e-12
        else 0.0
    )
    non_manifold_increased = (
        after_report["non_manifold_edge_count"]
        > before_report["non_manifold_edge_count"]
    )
    degenerate_increased = (
        after_report["degenerate_triangle_count"]
        > before_report["degenerate_triangle_count"]
    )
    # 面积变化率用于无预期值时的弱判据；有预期值时用绝对误差判据，更准。
    return {
        "before": before_report,
        "after": after_report,
        "boundary_edge_count_before": before_boundary,
        "boundary_edge_count_after": after_boundary,
        "boundary_loop_count_before": before_loops,
        "boundary_loop_count_after": after_loops,
        "area_delta_mm2": after_area - before_area,
        "area_delta_ratio": area_ratio,
        "expected_added_area_mm2": expected_added_area_mm2,
        "added_area_error_mm2": (
            abs((after_area - before_area) - expected_added_area_mm2)
            if expected_added_area_mm2 is not None
            else None
        ),
        "non_manifold_increased": non_manifold_increased,
        "degenerate_increased": degenerate_increased,
        "topology_improved": after_boundary < before_boundary
        or (before_boundary == 0 and after_boundary == 0),
        # 质量是否变差只看是否"增加"，不看绝对值——原网格本身可能有历史缺陷。
        "quality_status": (
            "review"
            if non_manifold_increased
            or degenerate_increased
            else "pass"
        ),
    }


def validate_repair(
    before: TriangleMesh,
    after: TriangleMesh,
    area_tolerance_ratio: float = 0.05,
    expected_added_area_mm2: float | None = None,
) -> tuple[bool, dict[str, Any], list[str]]:
    """我用拓扑、质量和预期孔洞面积判断修补结果是否可继续。"""

    comparison = compare_mesh_quality(
        before,
        after,
        area_tolerance_ratio=area_tolerance_ratio,
        expected_added_area_mm2=expected_added_area_mm2,
    )
    warnings: list[str] = []
    if comparison["non_manifold_increased"]:
        warnings.append("我发现修补后非流形边数量增加。")
    if comparison["degenerate_increased"]:
        warnings.append("我发现修补后退化三角形数量增加。")
    if not comparison["topology_improved"]:
        warnings.append("我没有观察到开放边数量下降，请人工复核修补结果。")
    # 有预期面积走绝对误差判据；没有就退化成面积变化率，并明确标注这是弱判据。
    if expected_added_area_mm2 is None:
        if comparison["area_delta_ratio"] > area_tolerance_ratio:
            warnings.append(
                "我发现修补前后总面积变化超过当前容差，"
                "但没有提供预期孔洞面积，建议人工复核。"
            )
    else:
        expected = max(float(expected_added_area_mm2), 0.0)
        allowed_error = max(1e-6, expected * area_tolerance_ratio)
        actual_error = float(comparison["added_area_error_mm2"] or 0.0)
        if actual_error > allowed_error:
            warnings.append(
                "我发现修补实际新增面积与边界环预计面积不一致，"
                "建议人工复核。"
            )
    passed = (
        comparison["quality_status"] == "pass"
        and comparison["topology_improved"]
        and not any("新增面积与边界环预计面积不一致" in item for item in warnings)
    )
    return passed, comparison, warnings
