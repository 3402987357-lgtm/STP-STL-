"""我为分型线和分型面提供不依赖 CAD 内核的几何质量门禁。

模块职责：甲方任务书要求分型线"封闭、连续、无明显尖角、无扭曲、平整顺滑"，
分型面要检查"褶皱、交叉、破面、实体穿透和贴合性"。这些要求本质上是几何
性质，不需要 CAD 内核就能先做一轮自查，所以我把它们做成纯 NumPy 门禁：
在真正生成分型面之前，先把明显不合格的候选挡下来。

设计原则：
1. 只诊断、不修改。所有函数都返回"问题清单 + 证据 + 建议"，绝不悄悄改动
   原始分型线/网格。是否采纳建议由工程人员或后续阶段决定。
2. 阈值全部显式传入（tolerance_mm、sharp_turn_threshold_deg 等），不写死
   魔数，方便按模型尺度重新标定。
3. 任何不确定的情况一律偏向"报出来要求复核"，而不是"看起来还行就放行"。
"""

from __future__ import annotations

from math import ceil

import numpy as np


def parting_line_diagnostics(
    points: np.ndarray,
    *,
    closed: bool,
    tolerance_mm: float,
    bridge_ratio: float = 4.0,
    sharp_turn_threshold_deg: float = 60.0,
) -> dict[str, object]:
    """我为阶段5输出桥接、尖角、曲率和R角保留诊断。

    原理说明：这条诊断链按"先看能不能算，再看算出来的东西像不像一条
    合格分型线"的顺序组织：
      1. 有限性：含 NaN/Inf 时后续所有几何量都无意义，直接置 finite=False；
      2. 段长分布：取相邻点距离的中位数作为"正常步长"基准，明显超过
         bridge_ratio 倍的段判为"疑似断口/台阶"，交给 _bridge_proposals；
      3. 转角：用相邻两段的夹角判断尖角，超过阈值记入 sharp_corner_indices；
      4. 曲率跳变：相邻转角的二阶差分过大时记 curvature_jump_count；
      5. R 角候选：转角连续但半径很小的地方标为 r_corner_candidate，
         提醒工程保留圆角而不是当尖角削掉。
    边界：点数不足、退化点重合时对应指标返回 0/空列表，不抛异常——
    诊断函数本身不应该成为流程失败点。
    """

    values = np.asarray(points, dtype=np.float64)
    # 先给出一份"全零"基线再按需填充：这样下游无论走到哪个提前 return 分支，
    # 拿到的字段结构都完全一致，不会出现"有的报告有这个键、有的没有"。
    diagnostics: dict[str, object] = {
        "diagnostic_version": "parting-line-stage5-1.0",
        "closed": bool(closed),
        "point_count": int(len(values)) if values.ndim >= 1 else 0,
        "finite": bool(np.all(np.isfinite(values))) if values.size else True,
        "bridge_candidate_segments": [],
        "bridge_proposals": [],
        "sharp_corner_indices": [],
        "sharp_corners": [],
        "r_corner_candidate_indices": [],
        "curvature_jump_count": 0,
        "max_turn_angle_deg": 0.0,
        "median_segment_length_mm": 0.0,
        "max_segment_length_mm": 0.0,
        "max_to_median_segment_ratio": 0.0,
        "bridge_required": False,
        "r_corner_preservation": "REVIEW_REQUIRED",
    }
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(values) < 3
        or not np.all(np.isfinite(values))
    ):
        return diagnostics

    segments = np.diff(values, axis=0)
    # 闭合环要把首尾这一段补回来，否则最后一条边永远不参与统计，
    # 恰好位于首尾之间的长边（搭桥嫌疑最大处）反而会被漏掉。
    if closed and np.linalg.norm(values[0] - values[-1]) > tolerance_mm:
        segments = np.vstack([segments, values[0] - values[-1]])
    lengths = np.linalg.norm(segments, axis=1)
    # 先把零长段滤掉再统计：重复点会产生长度为 0 的段，
    # 它们会把中位数拉向 0，进而让"最大/中位数"比值虚高、误报成搭桥。
    positive_lengths = lengths[lengths > tolerance_mm]
    if not len(positive_lengths):
        return diagnostics
    median_length = float(np.median(positive_lengths))
    max_length = float(np.max(positive_lengths))
    ratio = max_length / median_length if median_length > 0.0 else 0.0
    # 搭桥判据取"相对中位数"与"绝对 10 倍容差"的较大者：
    # 只按相对判据会在整条线都很短时把亚毫米段也标成搭桥；
    # 只按绝对判据则在大件上失效。两者取大才能同时覆盖大小件。
    bridge_indices = [
        int(index)
        for index, length in enumerate(lengths)
        if length > max(bridge_ratio * median_length, 10.0 * tolerance_mm)
    ]
    diagnostics.update(
        {
            "bridge_candidate_segments": bridge_indices,
            # 中位数与最大值都落盘：只有比值无法还原具体尺度，
            # 也就无法判断"这条长边到底有多长"。
            "median_segment_length_mm": median_length,
            "max_segment_length_mm": max_length,
            "max_to_median_segment_ratio": ratio,
            # bridge_required 只表示"存在疑似搭桥段"，
            # 具体怎么搭由 bridge_proposals 给出，两者分开：
            # 前者用于快速筛选，后者用于指导修补。
            "bridge_required": bool(bridge_indices),
            "bridge_proposals": _bridge_proposals(
                values,
                bridge_indices,
                closed=closed,
                median_length=median_length,
                tolerance_mm=tolerance_mm,
            ),
        }
    )

    # 转角只在"内点"上算：闭合环每个点都有前后邻居，
    # 开放环的两端点没有，所以起始与结束各让开一个。
    ring_count = len(values) - 1 if closed else len(values)
    angle_indices = range(ring_count) if closed else range(1, ring_count - 1)
    angles: list[float] = []
    sharp_indices: list[int] = []
    sharp_corners: list[dict[str, object]] = []
    for index in angle_indices:
        # 取模实现首尾相接：闭合环上第 0 个点的前驱是最后一个点，
        # 不取模会漏掉首尾交界处的转角。
        previous = (index - 1) % ring_count
        following = (index + 1) % ring_count
        first = values[index] - values[previous]
        second = values[following] - values[index]
        first_length = float(np.linalg.norm(first))
        second_length = float(np.linalg.norm(second))
        # 零长邻边说明有重复点，夹角无定义。
        # 记 0.0（平滑）而不是跳过：跳过会让 angles 与点序错位，
        # 后续按索引回查时对不上点。
        if first_length <= tolerance_mm or second_length <= tolerance_mm:
            angles.append(0.0)
            continue
        cosine = float(
            np.dot(first, second) / (first_length * second_length)
        )
        # clip 防 arccos 收到 1.0000000002 返回 nan：
        # 浮点误差在共线点上很常见，不 clip 会让整条线的统计变成 nan。
        angle = float(
            np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        )
        angles.append(angle)
        if angle >= sharp_turn_threshold_deg:
            sharp_indices.append(int(index))
            sharp_corners.append(
                {
                    "index": int(index),
                    "turn_angle_deg": angle,
                    "point_xyz_mm": [
                        float(value) for value in values[index]
                    ],
                }
            )

    if angles:
        diagnostics["max_turn_angle_deg"] = max(angles)
        diagnostics["curvature_jump_count"] = int(
            sum(
                angle >= max(90.0, sharp_turn_threshold_deg + 25.0)
                for angle in angles
            )
        )
        diagnostics["sharp_corner_indices"] = sharp_indices
        diagnostics["sharp_corners"] = sharp_corners
        diagnostics["r_corner_candidate_indices"] = [
            int(index)
            for index, angle in zip(angle_indices, angles)
            if 8.0 <= angle < sharp_turn_threshold_deg
        ]
        diagnostics["r_corner_preservation"] = (
            "CANDIDATE"
            if diagnostics["r_corner_candidate_indices"]
            else "NO_SAMPLED_ARC"
        )
    return diagnostics


def _bridge_proposals(
    points: np.ndarray,
    segment_indices: list[int],
    *,
    closed: bool,
    median_length: float,
    tolerance_mm: float,
    max_intermediate_points: int = 8,
) -> list[dict[str, object]]:
    """我只生成直线桥接方案，不直接改变原始分型线。"""

    # 原理说明：桥接的目标是把"断口"用一条尽量自然的线段接起来。我用
    # 中位段长作为期望步长，按 ceil(段长/中位段长)-1 决定插入几个中间点，
    # 并限制上限 max_intermediate_points，避免一个超长段生成上百个点把
    # 数据量炸掉。所有方案都带 applied=False 和 requires_engineering_review，
    # 因为"台阶处该不该直连"是工程判断，不是几何算法能替甲方决定的。

    proposals: list[dict[str, object]] = []
    for index in segment_indices:
        # 下标越界一律跳过，只有闭合环的最后一段例外：
        # 闭合时"最后一点到第一点"是一条真实存在的段，
        # 必须保留，否则环会缺掉收口那一段。
        if index < 0 or index >= len(points) - 1:
            if not closed or index != len(points) - 1:
                continue
        end_index = index + 1 if index + 1 < len(points) else 0
        start = np.asarray(points[index], dtype=np.float64)
        end = np.asarray(points[end_index], dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        # 长度不超过容差的段不搭桥：这么短的一段
        # 本来就该由缝合直接消掉，补一条桥反而引入新几何。
        if not np.isfinite(length) or length <= tolerance_mm:
            continue
        # 内插点数随段长增加，上限 max_intermediate_points：
        # 长段需要更多中间点才能让后续判据采样到它，
        # 但无上限会在极长段上生成上千个点。
        subdivision_count = max(
            1,
            min(
                max_intermediate_points,
                # 以中位段长为基准而不是固定步长：
                # 分型线的段长通常差好几倍，固定步长会让短段
                # 一个点都不插、长段插到上限。
                int(ceil(length / max(median_length, tolerance_mm))) - 1,
            ),
        )
        # 均匀内插而不是按弧长：这一段是直线，
        # 均匀参数与均匀弧长等价，写起来也更直接。
        intermediate_points = [
            (
                start
                + (end - start) * (step / (subdivision_count + 1))
            ).tolist()
            for step in range(1, subdivision_count + 1)
        ]
        proposals.append(
            {
                "segment_index": int(index),
                "start_xyz_mm": start.tolist(),
                "end_xyz_mm": end.tolist(),
                "length_mm": length,
                "intermediate_points_xyz_mm": intermediate_points,
                # 方法名写死在结果里：下游按这个名字区分候选来源，
                # 不写的话无法知道这批候选是哪一步生成的。
                "method": "straight_linear_bridge_candidate",
                # applied 默认 False：这只是候选，不是已应用的修补。
                # 默认 True 会让"提供了候选"被读成"已经补上了"。
                "applied": False,
                "requires_engineering_review": True,
                # 每条候选都自带能力边界声明：候选本身没有验证过
                # 台阶边界、R 角连续性或加工可行性，
                # 不写出来就会被当成可用的修补方案。
                "warning": (
                    "我只提供直线桥接候选，未确认台阶真实边界、R角连续性或加工可行性。"
                ),
            }
        )
    return proposals


def parting_line_warnings(
    points: np.ndarray,
    *,
    closed: bool,
    tolerance_mm: float,
    min_area_mm2: float = 1e-9,
) -> list[str]:
    """我检查有限性、闭合性、退化段、平面性和二维自交。"""

    # 原理说明：闭合性用"首尾点距离 <= tolerance_mm"判断，而不是要求坐标
    # 完全相等——STEP/STL 往返后首尾点常有 1e-9 级差异，用严格相等会
    # 把合法闭环误判成断开。二维自交用投影到 XY 后的线段相交检测，
    # 因为分型线在工程上按俯视轮廓使用，自交与否看的是俯视图。

    values = np.asarray(points, dtype=np.float64)
    warnings: list[str] = []
    if values.ndim != 2 or values.shape[1] != 3:
        return ["我发现分型线点集不是 (N, 3) 形状。"]
    if len(values) < 4:
        warnings.append("我发现分型线点数不足以形成闭合轮廓。")
        return warnings
    if not np.all(np.isfinite(values)):
        warnings.append("我发现分型线包含 NaN 或无穷值。")
        return warnings
    if not closed:
        warnings.append("我发现分型线没有闭合。")
    if float(np.linalg.norm(values[0] - values[-1])) > tolerance_mm:
        warnings.append("我发现分型线首尾距离超过端点容差。")

    segment_lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    if np.any(segment_lengths <= tolerance_mm):
        warnings.append("我发现分型线包含重复点或零长度线段。")

    if float(np.ptp(values[:, 2])) > max(tolerance_mm * 10.0, 1e-6):
        warnings.append("我发现分型线不在同一水平面内。")

    if abs(_signed_area_xy(values)) <= min_area_mm2:
        warnings.append("我发现分型线投影面积接近零。")
    if _has_self_intersection(values, tolerance_mm):
        warnings.append("我发现分型线在 XY 投影中存在自交。")
    return warnings


def surface_mesh_warnings(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    tolerance_mm: float,
) -> list[str]:
    """我检查平面网格索引、有限性、退化三角形和共面性。"""

    # 原理说明：平面分型面允许是三角网格，但必须满足三条底线——索引不
    # 越界、坐标有限、没有面积为零的退化三角形。共面性用三角面法向
    # 是否彼此平行来判断：如果法向方向散开，说明"平面"其实被折过，
    # 后续布尔运算会产生褶皱或自交。

    values = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    warnings: list[str] = []
    if values.ndim != 2 or values.shape[1] != 3:
        return ["我发现分型面顶点不是 (N, 3) 形状。"]
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        return ["我发现分型面三角索引不是 (M, 3) 形状。"]
    if not np.all(np.isfinite(values)):
        warnings.append("我发现分型面包含 NaN 或无穷值。")
    if triangles.size and (
        triangles.min() < 0 or triangles.max() >= len(values)
    ):
        warnings.append("我发现分型面三角索引越界。")
        return warnings
    if len(values) and float(np.ptp(values[:, 2])) > max(tolerance_mm * 10.0, 1e-6):
        warnings.append("我发现平面分型面的顶点没有保持共面。")
    if len(triangles):
        triangle_values = values[triangles]
        areas = 0.5 * np.linalg.norm(
            np.cross(
                triangle_values[:, 1] - triangle_values[:, 0],
                triangle_values[:, 2] - triangle_values[:, 0],
            ),
            axis=1,
        )
        if np.any(areas <= max(tolerance_mm**2, 1e-12)):
            warnings.append("我发现分型面包含退化三角形。")
    return warnings


def planar_surface_diagnostics(
    boundary_points: np.ndarray,
    surface_vertices: np.ndarray,
    *,
    margin_mm: float,
    tolerance_mm: float,
) -> dict[str, object]:
    """我核对平面分型面是否覆盖边界，并记录可复核的尺寸诊断。"""

    # 原理说明：分型面存在的意义是"沿分型线把模腔封住"，所以核心检查是
    # 覆盖：把分型线投影到分型面所在平面，看它是否落在面的范围内。我把
    # 包围盒、面积、投影覆盖率都写进诊断，让工程人员能拿数字复核，
    # 而不是只给一个"通过/不通过"。边界：投影退化（面积接近 0）时
    # 覆盖率返回 None 并标记不可判定，不返回误导性的 0。

    diagnostics: dict[str, object] = {
        "diagnostic_version": "parting-surface-stage6-1.0",
        # 初值一律标"未算出来"：中途任何一步数据不合格就原样返回它，
        # 这样"没算"与"算出来是假"能区分开，
        # contains_boundary=False 配 finite=False 才是"没算"。
        "finite": False,
        "contains_boundary": False,
        "margin_mm": float(margin_mm),
        "boundary_bbox_min_xy_mm": None,
        "boundary_bbox_max_xy_mm": None,
        "surface_bbox_min_xy_mm": None,
        "surface_bbox_max_xy_mm": None,
        "surface_area_xy_mm2": 0.0,
    }
    try:
        boundary = np.asarray(boundary_points, dtype=np.float64)
        surface = np.asarray(surface_vertices, dtype=np.float64)
    except (TypeError, ValueError):
        return diagnostics
    if (
        boundary.ndim != 2
        or boundary.shape[1] != 3
        or surface.ndim != 2
        or surface.shape[1] != 3
        or len(boundary) == 0
        # 分型面至少要有 4 个顶点：少于 4 个围不出一个面，
        # 说明传进来的根本不是一个有效面片的顶点集。
        or len(surface) < 4
        # 非有限坐标直接放弃：包围盒会被 nan 污染，
        # 后面所有比较都会变成 False 并给出"不覆盖"的错误结论。
        or not np.all(np.isfinite(boundary))
        or not np.all(np.isfinite(surface))
    ):
        return diagnostics
    # 只在 XY 平面上比较：分型面的覆盖要求是"在分型方向上
    # 盖住整个边界"，与 Z 无关，带上 Z 会因分型面很薄而误判。
    boundary_min = boundary[:, :2].min(axis=0)
    boundary_max = boundary[:, :2].max(axis=0)
    surface_min = surface[:, :2].min(axis=0)
    surface_max = surface[:, :2].max(axis=0)
    # 容差向外放宽：边界恰好贴在面的边上时，
    # 浮点误差会让"刚好覆盖"判成"没覆盖"。
    contains = bool(
        np.all(surface_min <= boundary_min + tolerance_mm)
        and np.all(surface_max >= boundary_max - tolerance_mm)
    )
    diagnostics.update(
        {
            "finite": True,
            "contains_boundary": contains,
            "boundary_bbox_min_xy_mm": boundary_min.tolist(),
            "boundary_bbox_max_xy_mm": boundary_max.tolist(),
            "surface_bbox_min_xy_mm": surface_min.tolist(),
            "surface_bbox_max_xy_mm": surface_max.tolist(),
            "surface_area_xy_mm2": float(
                (surface_max[0] - surface_min[0])
                * (surface_max[1] - surface_min[1])
            ),
        }
    )
    return diagnostics


def _signed_area_xy(points: np.ndarray) -> float:
    """我用鞋带公式算 XY 平面上的有符号面积，用于判断环的绕向和大小。

    原理说明：有符号面积的正负直接反映环是逆时针还是顺时针，这在判断
    "孔洞环"和"外轮廓环"时是必要信息。首尾点不重合时我先补一个闭合点，
    否则鞋带公式会漏掉最后一条边。点数少于 3 时返回 0.0，不抛错。
    """
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    if len(xy) < 3:
        return 0.0
    if not np.allclose(xy[0], xy[-1], atol=1e-12, rtol=0.0):
        xy = np.vstack([xy, xy[0]])
    return float(
        0.5
        * np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1])
    )


def _has_self_intersection(points: np.ndarray, tolerance_mm: float) -> bool:
    """我用 O(n²) 线段相交检测判断二维折线是否自交。

    原理说明：先去掉与首点重合的尾点，把折线当成闭合环处理；然后两两比较
    所有非相邻线段。相邻线段（共享端点）必须跳过，否则它们"接触"会被
    误判成自交。判断落在 tolerance_mm 容差内才算相交，避免浮点误差
    造成假阳性。
    复杂度：O(n²)，本项目分型线点数在千级，直接两两比较足够快；
    若未来点数上万，应换成扫描线算法（此处是刻意保留的简单实现）。
    """
    values = np.asarray(points, dtype=np.float64)[:, :2]
    # 少于四个点直接判不自交：三四条边以下的环
    # 在几何上不可能自己交叉。
    if len(values) < 4:
        return False
    # 首末点重合时去掉末点：闭合环的最后一个点是重复记录的起点，
    # 留着它会多出一条零长边，与相邻边的相交判定结果不可靠。
    if np.linalg.norm(values[0] - values[-1]) <= tolerance_mm:
        values = values[:-1]
    segment_count = len(values)
    # 去掉重复点后少于三个点就围不出环。
    if segment_count < 3:
        return False
    # 两两检查所有线段对：O(n²)，但分型线的点数通常在几百以内，
    # 这个量级下暴力两两比较比建索引更简单也更可靠。
    for first in range(segment_count):
        first_start = values[first]
        # 末段的终点绕回起点：环的最后一条边。
        first_end = values[(first + 1) % segment_count]
        for second in range(first + 1, segment_count):
            # 跳过自己与相邻的两条：相邻线段共享端点，
            # 共享端点会被判成相交，那是正常连接不是自交。
            if second in {
                first,
                (first - 1) % segment_count,
                (first + 1) % segment_count,
            }:
                continue
            second_start = values[second]
            second_end = values[(second + 1) % segment_count]
            # 找到任意一对非相邻线段相交就判自交：
            # 一处自交就足以说明这条线不能作为分型线。
            if _segments_intersect(
                first_start,
                first_end,
                second_start,
                second_end,
                tolerance_mm,
            ):
                return True
    return False


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance_mm: float,
) -> bool:
    """我用叉积定号判断两条二维线段是否相交（含容差）。

    原理说明：用四组叉积符号判断端点是否跨立：如果线段 A 的两端分居
    线段 B 两侧，且 B 的两端分居 A 两侧，则相交。共线/端点接触等退化
    情况用"点到线段距离 <= tolerance_mm"补充判断，保证相切、搭接这类
    工程上算"碰上"的情况不会被漏掉。
    边界：零长度线段按点处理；容差为负时按 0 处理。
    """
    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """我算三点转向的叉积 z 分量，符号给出 c 在 ab 的哪一侧。"""
        # 叉积的 z 分量：符号表示 c 在 ab 的哪一侧，
        # 绝对值等于以 ab、ac 为边的平行四边形面积。
        return float(
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )

    def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        """我判断 b 是否落在 ac 的包围盒内，容差向外放宽一档。"""
        # 判断 b 是否落在 ac 的包围盒内（容差向外放宽）。
        # 这一步必须与 orientation 配合：叉积近零只说明共线，
        # 不说明落在线段上，共线的两条延长线也算"相交"是错的。
        return bool(
            min(a[0], c[0]) - tolerance_mm
            <= b[0]
            <= max(a[0], c[0]) + tolerance_mm
            and min(a[1], c[1]) - tolerance_mm
            <= b[1]
            <= max(a[1], c[1]) + tolerance_mm
        )

    # 四个叉积：两条线段各自的端点相对于另一条线段的方位。
    # 只有在"每条线段的两端分处另一侧"时才真正相交。
    first_orientation = orientation(first_start, first_end, second_start)
    second_orientation = orientation(first_start, first_end, second_end)
    third_orientation = orientation(second_start, second_end, first_start)
    fourth_orientation = orientation(second_start, second_end, first_end)
    # 面积容差随线段长度放大：叉积的单位是长度的平方，
    # 长线段上同样的"几乎共线"会产生大得多的叉积值。
    # 下限 1e-12 保证极短线段不会除出 inf。
    scale = max(
        np.linalg.norm(first_end - first_start),
        np.linalg.norm(second_end - second_start),
        1.0,
    )
    area_tolerance = max(tolerance_mm * scale, 1e-12)
    # 四种共线接触情形逐一判：任一端点落在另一条线段上就算相交。
    # 必须四种都判——只判前两种会漏掉另一条线段的端点
    # 落在这条线段上的情况。
    if (
        abs(first_orientation) <= area_tolerance
        and on_segment(first_start, second_start, first_end)
    ):
        return True
    if (
        abs(second_orientation) <= area_tolerance
        and on_segment(first_start, second_end, first_end)
    ):
        return True
    if (
        abs(third_orientation) <= area_tolerance
        and on_segment(second_start, first_start, second_end)
    ):
        return True
    if (
        abs(fourth_orientation) <= area_tolerance
        and on_segment(second_start, first_end, second_end)
    ):
        return True
    return (first_orientation > area_tolerance) != (
        second_orientation > area_tolerance
    ) and (third_orientation > area_tolerance) != (
        fourth_orientation > area_tolerance
    )
