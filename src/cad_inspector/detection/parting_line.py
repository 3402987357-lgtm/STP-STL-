"""我方基于三角网格的分型线候选基线。

这个模块不是最终的工业级 B-Rep 分型算法。我把它作为没有 OCC 时仍可测试的
几何基线：在若干水平截面上求三角网格与平面的交线，连接闭合环，并选择
面积和长度都合理的最大候选。后续 OCC 版本可以复用结果模型和质量门禁。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.parting import PartingConfig, PartingLineResult
from ..errors import GeometryProcessingError
from ..geometry.mesh import TriangleMesh
from ..validation.parting_quality import (
    parting_line_diagnostics,
    parting_line_warnings,
)


@dataclass
class _Segment:
    """我用这个内部结构保存一个三角形与水平面的交线段。"""

    start: np.ndarray
    end: np.ndarray


def extract_parting_line(
    mesh: TriangleMesh,
    config: PartingConfig | None = None,
) -> PartingLineResult:
    """我从网格中提取最长闭合水平截面作为分型线候选。"""

    config = config or PartingConfig()
    if len(mesh.faces) == 0:
        raise GeometryProcessingError("我无法从空网格中提取分型线。")

    candidates = (
        [float(config.plane_z_mm)]
        if config.plane_z_mm is not None
        else _scan_plane_levels(mesh, config)
    )
    best: PartingLineResult | None = None
    for plane_z in candidates:
        result = _extract_at_plane(mesh, plane_z, config)
        if result is not None and (best is None or _line_score(result) > _line_score(best)):
            best = result

    if best is None:
        raise GeometryProcessingError(
            "我没有找到有效的水平截面闭环；请确认模型是封闭实体，或改用 OCC B-Rep 后端。"
        )
    return best


def _scan_plane_levels(mesh: TriangleMesh, config: PartingConfig) -> list[float]:
    """我在网格 Z 跨度内等距取若干截面高度，两侧按边距内缩。"""
    z_values = mesh.vertices[:, 2]
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    span = z_max - z_min
    # Z 跨度太小或只扫描一次时退化为中位平面，避免切出退化截面。
    if span <= config.endpoint_tolerance_mm:
        return [float((z_min + z_max) / 2.0)]
    if config.plane_scan_count == 1:
        return [float((z_min + z_max) / 2.0)]
    margin = span * config.plane_margin_ratio
    return np.linspace(
        z_min + margin,
        z_max - margin,
        config.plane_scan_count,
        dtype=np.float64,
    ).tolist()


def _extract_at_plane(
    mesh: TriangleMesh,
    plane_z: float,
    config: PartingConfig,
) -> PartingLineResult | None:
    """我在给定高度切网格，把交线段连成闭环并给出分型线候选。

    流程是：逐三角面求交线段 -> 按端点容差连成环 -> 按评分选主环 ->
    简化点列 -> 跑质量诊断。每一步的取舍如下：

      1. 逐面求交而不是整体布尔：网格没有 B-Rep 拓扑，只能按三角形切。
         我只收"恰好两个交点"的面，因为 0 个是不相交、3 个是退化接触；
      2. 连环用端点量化键匹配，容差取 endpoint_tolerance_mm。
         网格顶点是浮点坐标，直接比相等会大量漏配，环会断成碎段；
      3. 多环时按评分选主环，但**必须显式告警存在多环**。
         真实模具在同一高度常切出多个环（型腔+冷却水路），静默取最大
         会让甲方以为"这就是分型线"，所以我把它写进 warnings；
      4. 出现严重几何问题（NaN、自交、面积近零、重复点、零长度）时
         一律降级 REVIEW_REQUIRED，不给 BASELINE。
    """
    # 逐三角面求与平面的交线段；只有恰好两个交点的面才产生有效线段。
    segments: list[_Segment] = []
    for triangle in mesh.triangle_vertices:
        points = _triangle_plane_intersections(
            triangle,
            plane_z,
            config.endpoint_tolerance_mm,
        )
        if len(points) == 2:
            segments.append(_Segment(points[0], points[1]))

    if not segments:
        return None
    loops = _connect_segments(
        segments,
        tolerance=config.endpoint_tolerance_mm,
        min_loop_points=config.min_loop_points,
    )
    if not loops:
        return None

    # 多环时按"投影面积为主、周长为辅"的评分选主环，并显式提示存在多环。
    loops.sort(key=_loop_score, reverse=True)
    points = _simplify_closed_loop(loops[0], config.endpoint_tolerance_mm)
    closed = _points_close(points[0], points[-1], config.endpoint_tolerance_mm)
    warnings = [
        "我当前使用的是三角网格水平截面基线，不代表最终 B-Rep 分型线。",
        "我当前未识别真实面类型、R 角语义或模具加工规则。",
    ]
    if not closed:
        warnings.append("我找到的是开放候选线，不能直接作为最终分型线。")
    if len(loops) > 1:
        warnings.append("我发现多个截面环，当前按候选评分选择了一个主环。")
    warnings.extend(
        parting_line_warnings(
            points,
            closed=closed,
            tolerance_mm=config.endpoint_tolerance_mm,
            min_area_mm2=config.minimum_line_area_mm2,
        )
    )
    quality_diagnostics = parting_line_diagnostics(
        points,
        closed=closed,
        tolerance_mm=config.endpoint_tolerance_mm,
    )
    if quality_diagnostics["sharp_corner_indices"]:
        warnings.append("我发现分型线存在尖角候选，已输出坐标供复核。")
    # 只要诊断出现严重几何问题关键字，就降级为 REVIEW_REQUIRED。
    severe_quality_issue = any(
        keyword in warning
        for warning in warnings
        for keyword in (
            "包含 NaN",
            "自交",
            "投影面积接近零",
            "重复点",
            "零长度",
        )
    )

    return PartingLineResult(
        status="BASELINE" if closed and not severe_quality_issue else "REVIEW_REQUIRED",
        method="mesh_horizontal_section",
        plane_z_mm=float(plane_z),
        points_xyz_mm=points.tolist(),
        closed=closed,
        length_mm=_polyline_length(points),
        area_xy_mm2=abs(_signed_area_xy(points)),
        score=_loop_score(points),
        warnings=warnings,
        diagnostics={
            "segment_count": len(segments),
            "loop_count": len(loops),
            "candidate_plane_z_mm": float(plane_z),
            **quality_diagnostics,
        },
    )


def _triangle_plane_intersections(
    triangle: np.ndarray,
    plane_z: float,
    tolerance: float,
) -> list[np.ndarray]:
    """我求一个三角形与水平面的交点；退化情况按最长点对返回两个点。

    原理说明：我用有符号距离判断边的两端相对平面的位置。两端异号
    （start_delta * end_delta < 0）说明边穿过平面，用线性插值求交点；
    端点恰好落在平面上（|delta| <= tolerance）时直接取该顶点。

    为什么要把每个交点强制压到 z = plane_z：插值出来的点 z 分量只有
    浮点近似精度，同一个几何交点在不同三角形上算出来可能有 1e-16 的差异。
    后续连环用量化整数键匹配，z 不一致会让同一个点得到不同的键，
    环就拼不起来。压平是让"同一交点得到同一键"的必要一步。

    三个以上交点是退化情形（三角形恰好躺在平面上，或顶点正好落在平面上），
    此时我取距离最远的一对作为交线，避免把整个三角形当成一条线段。
    """
    points: list[np.ndarray] = []
    for start, end in (
        (triangle[0], triangle[1]),
        (triangle[1], triangle[2]),
        (triangle[2], triangle[0]),
    ):
        start_delta = float(start[2] - plane_z)
        end_delta = float(end[2] - plane_z)
        if abs(start_delta) <= tolerance and abs(end_delta) <= tolerance:
            points.extend([start.copy(), end.copy()])
        elif abs(start_delta) <= tolerance:
            points.append(start.copy())
        elif abs(end_delta) <= tolerance:
            points.append(end.copy())
        elif start_delta * end_delta < 0:
            ratio = start_delta / (start_delta - end_delta)
            points.append(start + ratio * (end - start))

    # 交点按容差去重并强制压到平面上，避免浮点误差让同一交点产生多个键。
    unique: list[np.ndarray] = []
    for point in points:
        if not any(_points_close(point, existing, tolerance) for existing in unique):
            point = point.copy()
            point[2] = plane_z
            unique.append(point)
    if len(unique) < 2:
        return []
    if len(unique) == 2:
        return unique

    # 出现两个以上交点时（顶点恰好落在平面上），取距离最远的一对作为该面交线。
    pair = max(
        (
            (a, b)
            for index, a in enumerate(unique)
            for b in unique[index + 1 :]
        ),
        key=lambda values: float(np.linalg.norm(values[0] - values[1])),
    )
    return [pair[0], pair[1]]


def _connect_segments(
    segments: list[_Segment],
    tolerance: float,
    min_loop_points: int,
) -> list[np.ndarray]:
    """我根据量化端点连接截面线段，优先保留闭合环。

    原理说明：截面得到的是一堆无序线段，要把它们还原成环，本质是在
    "端点重合"这个关系上做图遍历。我先把每个端点量化成整数键
    （_point_key），相同键视为同一点，于是线段变成图上的边、端点变成节点，
    再沿邻接表走链。

    为什么必须量化而不能直接比坐标：交点是插值算出来的浮点值，
    同一个几何端点在相邻两个三角形上算出的坐标可能差 1e-15。
    直接比较会得到"两个不同的点"，于是每条线段都成了孤边，
    一个环都拼不出来。量化让"近似相等"变成"完全相等"。

    走链时我用 used 集合保证每条边只用一次，并在回到起点时判定为闭合环；
    走不下去就作为开放链保留。分叉（一个点连着三条以上边）时交给
    _choose_next_edge 按夹角选最平顺的延续，避免在两个环交叉处乱跳。
    """

    key_to_point: dict[tuple[int, int], np.ndarray] = {}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    adjacency: dict[tuple[int, int], list[int]] = {}
    # 先建"量化坐标 -> 顶点"的图：把浮点坐标按容差量化成整数键，
    # 相邻线段的端点才能被认作同一个点。
    # 不先做这一步的话，两条本应相接的线段会因 1e-9 级差异断开，
    # 后面永远拼不出闭合环。
    for segment in segments:
        start_key = _point_key(segment.start, tolerance)
        end_key = _point_key(segment.end, tolerance)
        # 起终点量化后相同说明这是条退化线段（长度小于容差），
        # 它对成环没有贡献，直接丢弃。
        if start_key == end_key:
            continue
        key_to_point.setdefault(start_key, segment.start)
        key_to_point.setdefault(end_key, segment.end)
        edge_id = len(edges)
        edges.append((start_key, end_key))
        # 邻接表记的是"边号"而不是"邻居点"：一个点上可能有多条边，
        # 选下一条边时要按转角偏好挑（见 _choose_next_edge），
        # 只存邻居点会丢掉这个选择能力。
        adjacency.setdefault(start_key, []).append(edge_id)
        adjacency.setdefault(end_key, []).append(edge_id)

    used: set[int] = set()
    loops: list[np.ndarray] = []
    # 从每条未用过的边出发尽量走一圈。遍历全部边而不是只从端点出发，
    # 是因为分型线可能由多个互不相连的环组成，
    # 只从度为 1 的端点出发会漏掉中间的闭合环。
    for start_edge_id, (start_key, _) in enumerate(edges):
        if start_edge_id in used:
            continue
        path_keys = [start_key]
        current_key = start_key
        current_edge_id = start_edge_id
        previous_key: tuple[int, int] | None = None
        # 步数上限取 len(edges)+1：一条边最多用一次，
        # 走超过边数还没回到起点说明进了死循环（图有异常），
        # 硬上限保证一定会退出。
        for _ in range(len(edges) + 1):
            used.add(current_edge_id)
            edge_start, edge_end = edges[current_edge_id]
            # 边是无向的：从哪端进来就从另一端出去。
            next_key = edge_end if edge_start == current_key else edge_start
            path_keys.append(next_key)
            # 回到起点即成环，这是唯一正常的结束方式。
            if next_key == start_key:
                break
            choices = [
                edge_id
                for edge_id in adjacency.get(next_key, [])
                if edge_id not in used
            ]
            # 没有可继续的边说明走到断头，这条路径成不了环。
            if not choices:
                break
            previous_key, current_key = current_key, next_key
            current_edge_id = _choose_next_edge(
                choices,
                current_key,
                previous_key,
                edges,
                key_to_point,
            )

        if path_keys[-1] == start_key and len(path_keys) >= min_loop_points:
            loops.append(
                np.asarray(
                    [key_to_point[key] for key in path_keys],
                    dtype=np.float64,
                )
            )
    return loops


def _choose_next_edge(
    choices: list[int],
    current_key: tuple[int, int],
    previous_key: tuple[int, int] | None,
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    key_to_point: dict[tuple[int, int], np.ndarray],
) -> int:
    """我在多个候选边里挑转折最平缓的一条，避免连出尖锐折返。"""
    if len(choices) == 1 or previous_key is None:
        return choices[0]
    previous = key_to_point[previous_key]
    current = key_to_point[current_key]
    incoming = current - previous
    incoming_norm = np.linalg.norm(incoming)
    if incoming_norm <= 1e-12:
        return choices[0]
    incoming /= incoming_norm

    # 转折评分取入向与出向的夹角余弦；越接近 1 越平直，取最大者。
    def turn_score(edge_id: int) -> float:
        """我给候选边打平直度分：入向与出向夹角余弦，越接近 1 越平直。"""
        start, end = edges[edge_id]
        next_key = end if start == current_key else start
        outgoing = key_to_point[next_key] - current
        outgoing_norm = np.linalg.norm(outgoing)
        if outgoing_norm <= 1e-12:
            return -1.0
        outgoing /= outgoing_norm
        return float(np.dot(incoming, outgoing))

    return min(choices, key=lambda edge_id: (-turn_score(edge_id), edge_id))


def _simplify_closed_loop(points: np.ndarray, tolerance: float) -> np.ndarray:
    """我删除重复点和共线中间点，保持闭环和原始几何顺序。"""

    values = np.asarray(points, dtype=np.float64)
    if len(values) < 4:
        return values
    cleaned: list[np.ndarray] = []
    for point in values:
        if not cleaned or not _points_close(cleaned[-1], point, tolerance):
            cleaned.append(point.copy())
    if len(cleaned) > 1 and _points_close(cleaned[0], cleaned[-1], tolerance):
        cleaned[-1] = cleaned[0].copy()
    changed = True
    while changed and len(cleaned) > 4:
        changed = False
        for index in range(1, len(cleaned) - 1):
            previous = cleaned[index - 1]
            current = cleaned[index]
            following = cleaned[index + 1]
            first = current - previous
            second = following - current
            cross = abs(float(first[0] * second[1] - first[1] * second[0]))
            if cross <= tolerance * max(
                np.linalg.norm(first),
                np.linalg.norm(second),
                1.0,
            ):
                del cleaned[index]
                changed = True
                break
    return np.asarray(cleaned, dtype=np.float64)


def _line_score(result: PartingLineResult) -> float:
    """我给候选线评分：闭合环加一个大常数，保证闭环永远优先于开放线。

    我为什么用一个大常数而不是"只收闭环"：后者看起来更干净，但实测
    有些破损件在所有高度上都切不出严格闭环，只收闭环会直接抛错、
    连一条候选线都不给。而甲方复核时"一条开放但形状接近的候选线"
    比"什么都没有"有价值得多。加常数保证只要存在闭环就优先选它，
    没有闭环时仍能退到开放线并明确告警。
    """
    return result.score + (1_000_000.0 if result.closed else 0.0)


def _loop_score(points: np.ndarray) -> float:
    """我用闭环真实围成面积为主、周长为辅给闭环评分。

    我不再用 XY 投影面积：网格分型线是水平截面环，但若模型整体倾斜摆放，
    XY 投影面积等于真实面积乘 cos(夹角)，会让"最大外轮廓"排序选错候选。
    我改成环自身拟合平面上的真实面积，与摆放角度无关。

    我复用 step_parting 里的同一份实现，而不是在这里抄一份：面积口径
    必须和 B-Rep 路线一致，否则两条路线对同一个零件会给出两个不同的
    "最大外轮廓"，甲方无法判断该信哪个。

    我在函数内延迟导入：step_parting 依赖本模块，放到模块顶层会形成
    循环导入，所以这里只在真正需要评分时才导入。
    """


    from .step_parting import _true_loop_area_mm2

    area, _ = _true_loop_area_mm2(points, tolerance_mm=1e-6)
    return area + 0.01 * _polyline_length(points)


def _polyline_length(points: np.ndarray) -> float:
    """我返回折线相邻点欧氏距离之和。

    我为什么用 ``np.diff`` 一次算完而不是逐段循环：分型线的点数是
    几百到几千量级，逐段 Python 循环在这个规模上比向量化慢一个量级，
    而这个函数在候选筛选里会被每个高度调一次。
    """
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _signed_area_xy(points: np.ndarray) -> float:
    """我用鞋带公式算折线在 XY 平面的有符号投影面积（自动闭合）。"""
    if len(points) < 3:
        return 0.0
    xy = points[:, :2]
    if not _points_close(points[0], points[-1], 1e-12):
        xy = np.vstack([xy, xy[0]])
    return float(
        0.5 * np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1])
    )


def _point_key(point: np.ndarray, tolerance: float) -> tuple[int, int]:
    """我把二维点按容差量化成整数键，用于端点匹配与去重。

    我为什么只量化 x、y 两个分量：截面是水平面，所有交点的 z 都已被
    强制压到同一个 plane_z（见 _triangle_plane_intersections），
    z 不携带区分信息。带上它反而会让 z 上的残余误差重新引入失配。

    我为什么用 round 而不是 int() 截断：截断会让落在同一格内、
    分处格线两侧的两个点得到不同的键——它们明明该算同一点。
    round 至少保证"离得最近的两个点常被分到同一格"，虽然格线附近
    仍有边界效应，但那是量化法的固有代价，我已在文档里写明。
    """
    scale = 1.0 / tolerance
    return (int(round(float(point[0]) * scale)), int(round(float(point[1]) * scale)))


def _points_close(first: np.ndarray, second: np.ndarray, tolerance: float) -> bool:
    """我判断两点欧氏距离是否在容差以内。

    我用闭区间（<=）而不是开区间：容差的语义是"可接受的误差上限"，
    恰好等于上限的点属于可接受。用开区间会让"距离正好等于容差"
    这一边界被判为不等，而这条边界在量化格线上是真实会碰到的。
    """
    return float(np.linalg.norm(first - second)) <= tolerance
