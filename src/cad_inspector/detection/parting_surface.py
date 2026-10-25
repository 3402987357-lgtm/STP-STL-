"""我方平面分型面的网格基线生成器。

算法原理：分型线本身是一条位于某个 Z 平面上的闭环折线。要给出"分型面"的最简可
检查基线，我取分型线在 XY 平面的包围盒，四边各外扩 ``surface_margin_mm``，得到
一个覆盖分型线的矩形平面，再用两个三角形组成网格。

这条基线只用于打通"分型线 -> 分型面 -> 质量诊断"的流程链路，并不等于最终 CAD
B-Rep 分型面：它不做实体穿透、不做曲面缝合、不做面体夹角与加工可行性检查。所有
结构化输出都显式带 method 与 warnings，提醒下游这是基线而不是成品。
"""

from __future__ import annotations

import numpy as np

from ..domain.parting import PartingConfig, PartingLineResult, PartingSurfaceResult
from ..validation.parting_quality import (
    planar_surface_diagnostics,
    parting_line_warnings,
    surface_mesh_warnings,
)


def build_planar_parting_surface(
    line: PartingLineResult,
    config: PartingConfig | None = None,
) -> PartingSurfaceResult:
    """我用分型线包围盒外扩生成一个可检查的平面网格基线。

    面用两个三角形组成一个矩形，而不是对分型线做三角化：
    分型线的形状通常复杂且可能自交，对它直接三角化本身就是
    一个未解决的问题。包围盒矩形虽然粗糙，但它是**确定**的——
    任何分型线都能得到一个合法平面，流程不会卡在这里。

    外扩 surface_margin_mm 的意义：正好贴着包围盒的面与分型线
    共边界，后续任何几何比较都会落在"相切"这种退化情形上。
    留出余量让面严格大于分型线，包含性判断才有明确的余量可用。

    质量警告我一路带到结果里而不中途丢弃：分型线的闭环性、
    面积、端点重合这些问题是**输入**的问题，
    把它们丢掉会让下游以为这个面是基于一条好线生成的。
    """

    config = config or PartingConfig()
    # 先取分型线点并做闭环/面积/端点质量诊断，这些警告会一路带到最终结果里。
    points = np.asarray(line.points_xyz_mm, dtype=np.float64)
    line_quality_warnings = parting_line_warnings(
        points,
        closed=line.closed,
        tolerance_mm=config.endpoint_tolerance_mm,
        min_area_mm2=config.minimum_line_area_mm2,
    )
    # 少于三个点无法围成面，直接返回 FAILED 并附带质量警告。
    if len(points) < 3:
        return PartingSurfaceResult(
            status="FAILED",
            method="mesh_bounded_rectangle",
            plane_z_mm=line.plane_z_mm,
            normal_xyz=[0.0, 0.0, 1.0],
            boundary_points_xyz_mm=points.tolist(),
            vertices_xyz_mm=[],
            faces=[],
            margin_mm=config.surface_margin_mm,
            warnings=["我没有足够的分型线点生成平面。"] + line_quality_warnings,
        )

    # 在 XY 平面按外扩量取包围盒，四个角构成矩形顶点。
    min_xy = points[:, :2].min(axis=0) - config.surface_margin_mm
    max_xy = points[:, :2].max(axis=0) + config.surface_margin_mm
    z = float(line.plane_z_mm)
    vertices = np.asarray(
        [
            [min_xy[0], min_xy[1], z],
            [max_xy[0], min_xy[1], z],
            [max_xy[0], max_xy[1], z],
            [min_xy[0], max_xy[1], z],
        ],
        dtype=np.float64,
    )
    # 明确声明基线边界，避免下游把它当最终分型面使用。
    warnings = [
        "我当前输出的是外扩矩形平面网格基线，不是最终 CAD B-Rep 分型面。",
        "我还没有执行实体穿透、曲面缝合、面体夹角和加工可行性检查。",
    ]
    if not line.closed:
        warnings.append("我收到的分型线不是闭环，当前平面只用于诊断和流程联调。")
    warnings.extend(line_quality_warnings)
    surface_quality_warnings = surface_mesh_warnings(
        vertices,
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        tolerance_mm=config.endpoint_tolerance_mm,
    )
    warnings.extend(surface_quality_warnings)
    surface_diagnostics = planar_surface_diagnostics(
        points,
        vertices,
        margin_mm=config.surface_margin_mm,
        tolerance_mm=config.endpoint_tolerance_mm,
    )
    if not surface_diagnostics["contains_boundary"]:
        warnings.append("我发现平面分型面没有完整覆盖分型线包围盒。")
    # 只要诊断里出现严重几何问题关键字，就把状态降级为 REVIEW_REQUIRED。
    severe_quality_issue = any(
        keyword in warning
        for warning in warnings
        for keyword in (
            "NaN",
            "越界",
            "退化",
            "没有保持共面",
            "自交",
            "重复点",
            "零长度",
            "投影面积接近零",
        )
    )

    return PartingSurfaceResult(
        status=(
            "BASELINE"
            if line.closed and not severe_quality_issue
            else "REVIEW_REQUIRED"
        ),
        method="mesh_bounded_rectangle",
        plane_z_mm=z,
        normal_xyz=[0.0, 0.0, 1.0],
        boundary_points_xyz_mm=points.tolist(),
        vertices_xyz_mm=vertices.tolist(),
        faces=[[0, 1, 2], [0, 2, 3]],
        margin_mm=float(config.surface_margin_mm),
        warnings=warnings,
        diagnostics={
            "line_area_xy_mm2": line.area_xy_mm2,
            "rectangle_area_xy_mm2": float(
                (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
            ),
            **surface_diagnostics,
        },
    )
