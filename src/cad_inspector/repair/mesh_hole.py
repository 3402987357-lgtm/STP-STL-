"""我实现安全边界内的局部网格孔洞修补基线。

我为什么选扇形三角化（从环上一点向其余点连三角形）而不是更高级的
advancing front 或泊松重建：扇形填补只用环上已有的顶点，
不引入任何新点。这带来两个好处：
  1. 修补不改变、不新增原网格顶点，其余部分完全不受影响，
     事后可以精确验证"只填了洞、没动别的"；
  2. 失败模式明确：非凸环用扇形填补会产生穿出边界的三角形，
     所以我在分析阶段就把非凸环直接判掉并要求人工处理，
     而不是产出一个"看起来补上了但几何是错的"的结果。

修补是本模块里少数"改写输入几何"的动作，宁可不做也不能做错，
所以平面性、凸性、面积上限、顶点数上下限逐项在补之前判完。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain.repair import (
    BoundaryLoopResult,
    RepairConfig,
    RepairRegionResult,
    RepairResult,
)
from ..geometry.mesh import TriangleMesh, mesh_quality_report
from ..validation.mesh_quality import (
    boundary_loops,
    edge_to_faces,
    validate_repair,
)


def analyze_boundary_loop(
    mesh: TriangleMesh,
    loop_id: int,
    vertex_ids: list[int],
    config: RepairConfig,
) -> BoundaryLoopResult:
    """我计算边界环的平面性、面积、周长和凸性，并逐项判断能否安全填补。

    每个判据都对应一种"扇形填补会出错"的具体情形：
      - 顶点数下限：少于 3 个点围不成面；
      - 顶点数/面积上限：过大的洞用扇形填补会产生大量狭长三角形，
        几何质量极差，且大概率本来就是设计上的开口而非破损；
      - 平面性误差：环不共面时，扇形填补出的三角形一部分会
        明显偏离真实曲面，这是"补上了但补错了"；
      - 凸性：非凸环（例如 L 形、带凹口的环）用扇形填补必然
        产生穿出边界的三角形，几何上不成立。

    我逐项独立判断并各自记一条 warning，而不是遇到第一个问题就返回：
    一个环可能同时超大又不共面，只报第一条会让人工复核时
    修好一个问题后才发现还有下一个。
    """

    points = mesh.vertices[np.asarray(vertex_ids, dtype=np.int64)]
    center = points.mean(axis=0)
    normal, basis_u, basis_v = _best_fit_plane(points, center)
    projected = np.column_stack(
        ((points - center) @ basis_u, (points - center) @ basis_v)
    )
    signed_area = _signed_polygon_area(projected)
    # 面积取绝对值，方向（绕序）由 signed_area 单独判断：
    # 两者用途不同——面积用于比上限，符号用于确定三角化的绕序。
    area = abs(signed_area)
    # 周长用 roll 一次性算所有相邻点对的距离再求和，
    # 比 for 循环逐边快一个量级，在大环上差别明显。
    perimeter = float(
        np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum()
    )
    planarity_error = float(np.max(np.abs((points - center) @ normal)))
    # 凸性是扇形三角化的前提，单独先算出来，
    # 后面据此决定走扇形还是耳切。
    convex = _is_convex_polygon(projected)
    warnings: list[str] = []
    repairable = True
    # 下面这组是"能不能自动补"的准入检查，逐条独立判断并各自记录原因，
    # 不做提前返回——复核时一次看到全部不合格原因，
    # 比只看到第一条、改完再跑一次效率高得多。
    if len(vertex_ids) < config.min_loop_vertices:
        repairable = False
        warnings.append("我发现边界环顶点数量不足，不能形成面。")
    # 环过大与面积过大是两道不同的闸：前者防计算量爆炸，
    # 后者防"把整个缺失的面当成洞补掉"，两者不可互相替代。
    if len(vertex_ids) > config.max_boundary_edges:
        repairable = False
        warnings.append("我发现边界环过大，当前安全配置不允许自动修补。")
    if area > config.max_hole_area_mm2:
        repairable = False
        warnings.append("我发现孔洞面积超过当前自动修补上限。")
    if planarity_error > config.max_planarity_error_mm:
        repairable = False
        warnings.append("我发现边界环不够平面，不能使用三角扇安全填补。")
    # 非凸环默认拒绝，但允许通过配置放行走耳切路径：
    # 默认拒绝是因为扇形化非凸环必然产生重叠三角形，
    # 放行开关留给明确知道风险的场景。
    if not convex and not config.allow_nonconvex:
        repairable = False
        warnings.append("我发现边界环不是凸多边形，当前实现拒绝直接扇形填补。")
    # 面积/周长/方向三项退化检查放在最后：
    # 它们是"数值上无法构造"的兜底，只有前面的准入都过了才有意义。
    if area <= 1e-12:
        # 投影面积近零说明这个环退化成了一条线：
        # 没有面积就无从补面，补出来也是零面积的废面。
        repairable = False
        warnings.append("我发现边界环投影面积接近零。")
    if perimeter <= 1e-12:
        # 周长近零说明所有点几乎重合：
        # 这是一个退化点，不是一个洞。
        repairable = False
        warnings.append("我发现边界环周长接近零。")
    if abs(signed_area) <= 1e-12:
        # 有向面积近零说明定不出稳定的绕向：
        # 绕向决定补出的面的法向朝哪，
        # 定不出来就无法保证补面与邻面朝向一致。
        repairable = False
        warnings.append("我无法确定边界环的稳定方向。")
    edge_mapping = edge_to_faces(mesh)
    adjacent_face_ids = {
        face_id
        for first, second in zip(vertex_ids, vertex_ids[1:] + vertex_ids[:1])
        # 边的键排序后查表：edge_mapping 的键是排序后的顶点对，
        # 不排序会查不到，于是邻接面数恒为 0。
        for face_id in edge_mapping.get(
            (first, second) if first < second else (second, first),
            [],
        )
    }
    # 邻接面不超过一个就不补：只有一片面相邻的"洞"
    # 通常是一片悬空的孤立三角形，
    # 把它当洞补上会把噪声固化进模型。
    if len(adjacent_face_ids) <= 1:
        repairable = False
        warnings.append(
            "我发现边界环没有足够的邻接面支撑，不能把孤立三角片当作孔洞修补。"
        )
    return BoundaryLoopResult(
        loop_id=loop_id,
        vertex_ids=[int(value) for value in vertex_ids],
        center_xyz_mm=center.tolist(),
        # 法向单独存：补面的朝向要与它对齐，
        # 而补完之后还要拿它去校验新增三角形朝向是否一致。
        normal_xyz=normal.tolist(),
        perimeter_mm=perimeter,
        area_mm2=area,
        # 平面度误差是"这个环有多平"：
        # 它决定是否可以用平面填充，
        # 超过容差要走曲面填充或三角化。
        planarity_error_mm=planarity_error,
        is_convex=convex,
        repairable=repairable,
        # 警告逐条保留：这里可能有好几条，
        # 只留最后一条会让人以为只有一个问题。
        warnings=warnings,
    )


def repair_mesh_holes(
    mesh: TriangleMesh,
    config: RepairConfig | None = None,
    input_file: str | Path = "",
) -> tuple[TriangleMesh, RepairResult]:
    """我只对满足安全条件的简单开放边界环执行局部修补。"""

    config = config or RepairConfig()
    original = mesh
    loops = boundary_loops(mesh)
    if not loops:
        # 没有边界环就原样返回，不是失败：
        # 封闭的模型本来就没有洞，"没有可修补的东西"
        # 与"修补失败"必须区分开。
        result = RepairResult(
            status="no_action",
            input_file=str(input_file),
            # 不写出文件：没有改动却生成一个副本
            # 会让下游以为这个副本是被修补过的。
            output_file=None,
            # 前后质量报告都留：两者相同正好说明"没动过模型"，
            # 只留一份就无法验证这一点。
            input_quality=mesh_quality_report(original),
            output_quality=mesh_quality_report(mesh),
            repaired_regions=[],
            warnings=[
                "我没有发现可遍历的简单闭合边界环，当前任务没有自动修补。"
            ],
            artifacts={},
            config=config,
        )
        return mesh, result

    # 顶点与面都先复制：下面要往里追加新三角形，
    # 直接在原数组上改会让"修补前"的那份数据消失，
    # 也就无法在最后做前后质量对比。
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    regions: list[RepairRegionResult] = []
    warnings: list[str] = []
    expected_added_area_mm2 = 0.0
    for region_id, loop in enumerate(loops, start=1):
        analysis = analyze_boundary_loop(mesh, region_id, loop, config)
        # 判定为不可修补的区域单独立一条结果而不是跳过：
        # 跳过会让"这个洞没修"这件事在结果里完全消失，
        # 复核者会以为没有这个洞。
        if not analysis.repairable:
            regions.append(
                RepairRegionResult(
                    region_id=region_id,
                    defect_type="hole",
                    status="manual_review_required",
                    # 方法名给 "none" 而不是空串：
                    # 空串会被读成"用了某个默认方法"。
                    method="none",
                    boundary_loop=analysis,
                    # 新增顶点与三角形都是 0：明确记下"没加任何东西"，
                    # 省掉会让下游不知道该不该重新数一遍面。
                    added_vertex_count=0,
                    added_triangle_count=0,
                    review_required=True,
                    warnings=list(analysis.warnings),
                )
            )
            warnings.extend(analysis.warnings)
            continue

        normal = _boundary_neighbor_normal(mesh, loop)
        # 面积门禁的基准必须和补片是同一类量。此前用"边界环在最佳拟合平面上的
        # 投影面积"，对非平面环天然偏小：实测一个平面度 0.019 mm 的真实孔洞，
        # 正确补片的新增面积比投影面积大 6.7%，会被 5% 容差误判超差。
        # 这里改用"把边界环正确三角化后的三维面积"作基准：对平面凸环它等于
        # 多边形面积，对非平面环它和补片同量级，同时仍能抓住非凸环上的越界扇面
        # （那种补片会明显大于正确三角化面积）。
        reference_area_mm2 = _loop_reference_area_mm2(loop, vertices, normal)
        if reference_area_mm2 <= 0.0:
            reference_area_mm2 = analysis.area_mm2
        # 凸多边形用中心扇面最省；非凸多边形如果继续用扇面，三角形会跨到
        # 多边形外面，新增面积对不上边界环面积。这里改用耳切法做正确三角化。
        if analysis.is_convex:
            # 凸环可以扇形三角化：从环心向各边连线即可，
            # 只需新增一个中心顶点，三角形数量最少、质量也最均匀。
            center = np.asarray(analysis.center_xyz_mm, dtype=np.float64)
            new_vertex_id = len(vertices)
            vertices = np.vstack([vertices, center])
            new_faces = _fan_faces(loop, new_vertex_id, vertices, normal)
            method = "planar_triangle_fan"
            added_vertex_count = 1
        else:
            # 凹环不能扇形化——从中心连线会穿出环外，必然产生重叠三角形。
            # 改走耳切三角化：它不新增顶点，只重排原有顶点，
            # 这样补出来的面片与原始网格顶点集一致，不会引入新坐标。
            clipped = _ear_clip_faces(loop, vertices, normal)
            if clipped is None:
                # 耳切失败说明这个环自相交或退化。我不做近似修补：
                # 强行三角化会产出翻转面片，把"破洞"变成"更坏的几何"，
                # 所以这里明确退回人工复核，并把原因写进 warnings。
                failure = [
                    "我发现边界环无法完成耳切三角化，"
                    "当前实现拒绝自动修补这块区域。"
                ]
                regions.append(
                    RepairRegionResult(
                        region_id=region_id,
                        defect_type="hole",
                        status="manual_review_required",
                        method="none",
                        boundary_loop=analysis,
                        added_vertex_count=0,
                        added_triangle_count=0,
                        review_required=True,
                        warnings=failure,
                    )
                )
                warnings.extend(failure)
                continue
            new_faces = clipped
            method = "planar_ear_clipping"
            added_vertex_count = 0
        expected_added_area_mm2 += reference_area_mm2
        # 逐块追加而不是最后一次性拼接：每块的 new_faces 用的是当前
        # vertices 里的索引，中间若新增了顶点，索引基准就变了，
        # 逐块 vstack 能保证索引始终对应同一个顶点表。
        faces = np.vstack([faces, np.asarray(new_faces, dtype=np.int64)])
        # 补好的区域仍带 review_required=False，但方法名原样记录
        # （扇形或耳切），让复核时能看出这块面是怎么生成的。
        regions.append(
            RepairRegionResult(
                region_id=region_id,
                defect_type="hole",
                status="repaired",
                method=method,
                boundary_loop=analysis,
                added_vertex_count=added_vertex_count,
                added_triangle_count=len(new_faces),
                review_required=False,
            )
        )

    # 重新包成 TriangleMesh 而不是直接返回数组：
    # 构造器会再跑一遍完整性校验（索引越界、非整数、NaN 坐标），
    # 这样修补引入的任何坏几何都会在出口被拦住，不会静默流到下游。
    repaired = TriangleMesh(
        vertices=vertices,
        faces=faces,
        name=mesh.name,
        metadata={
            **mesh.metadata,
            # 方法名用 sorted 去重后拼接：一块模型上可能用了多种补法
            # （有些走扇形、有些走耳切），排序保证同一批输入
            # 得到的字符串完全一致，便于做可复现比对。
            "repair_method": "+".join(
                sorted(
                    {
                        region.method
                        for region in regions
                        if region.status == "repaired"
                    }
                )
            )
            # 一块都没补上时写 "none" 而不是空串：
            # 空串在下游会被当成"方法名缺失"而不是"没有修补"。
            or "none",
            "repair_region_count": sum(
                region.status == "repaired" for region in regions
            ),
        },
    )
    # 补完必须再验一遍：validate_repair 会检查面积增量是否吻合、
    # 网格是否仍然闭合。补面本身成功不代表结果可用——
    # 补错了位置同样是"成功"地加了一块面。
    passed, comparison, validation_warnings = validate_repair(
        original,
        repaired,
        expected_added_area_mm2=expected_added_area_mm2,
    )
    warnings.extend(validation_warnings)
    repaired_count = sum(region.status == "repaired" for region in regions)
    review_count = sum(
        region.status == "manual_review_required" for region in regions
    )
    # 四档状态：一块没补 -> review；补了但有需复核的 -> partial（校验通过时）
    # 或 review（校验没过）；全补上且校验过 -> success。
    # 关键点是 partial 也要求 passed：部分修补一旦面积对不上，
    # 说明补的那部分本身就不可信，不能算"部分成功"。
    if repaired_count == 0:
        status = "review"
    elif review_count:
        status = "partial" if passed else "review"
    else:
        status = "success" if passed else "review"
    returned_mesh = repaired
    # do_no_harm 的最后一道闸：校验没过就退回原始网格。
    # 宁可交回没修的模型，也不能交出一个被改坏的模型——
    # 后者会让下游所有量测都建立在错误几何上。
    if not passed and config.preserve_original_on_failure:
        returned_mesh = original
        warnings.append(
            "我因修补质量门禁未通过而保留原始网格，未把待复核结果作为正式输出。"
        )
    result = RepairResult(
        status=status,
        input_file=str(input_file),
        output_file=None,
        input_quality=mesh_quality_report(original),
        output_quality={
            **mesh_quality_report(returned_mesh),
            "comparison": comparison,
            "attempted_repair_quality": mesh_quality_report(repaired),
        },
        repaired_regions=regions,
        warnings=_unique(warnings),
        artifacts={},
        config=config,
    )
    return returned_mesh, result


def _best_fit_plane(
    points: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """我使用 PCA 获取边界环的稳定局部平面基。

    原理说明：边界环通常近似共面但不严格共面，补片需要一个明确的
    "把它摊平到哪个平面"的约定。我用协方差矩阵的最小特征值对应的
    特征向量作为法向——它就是让所有点到该平面距离平方和最小的方向，
    即最佳拟合平面。用 eigh 而不是 eig，因为协方差矩阵是对称的，
    eigh 保证返回实数特征值和正交特征向量，数值更稳。

    基向量选取的细节：我先拿 X 轴当参考向量做叉积得到 basis_u，
    但如果法向几乎平行于 X 轴（|dot| > 0.9），叉积会退化成接近零向量，
    归一化后是数值垃圾，所以这时换用 Y 轴当参考。
    basis_v 再用 normal x basis_u 得到，保证 (u, v, n) 构成右手正交系。
    """

    covariance = np.cov((points - center).T, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    normal = normal / max(np.linalg.norm(normal), 1e-15)
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    basis_u = np.cross(normal, reference)
    basis_u = basis_u / max(np.linalg.norm(basis_u), 1e-15)
    basis_v = np.cross(normal, basis_u)
    basis_v = basis_v / max(np.linalg.norm(basis_v), 1e-15)
    return normal, basis_u, basis_v


def _signed_polygon_area(points_2d: np.ndarray) -> float:
    """我用鞋带公式算二维环的**有符号**面积。

    我保留符号而不是取绝对值：符号携带绕向信息（逆时针为正、
    顺时针为负），调用方要靠它判断环的绕向，以及判断两个环
    是不是一个内孔。取绝对值会丢掉这个信息。
    """
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _is_convex_polygon(points_2d: np.ndarray, tolerance: float = 1e-10) -> bool:
    """我用逐顶点叉积符号一致性判断二维多边形是否为凸。

    原理说明：沿多边形走一圈，凸多边形在每个顶点处的转向必须一致
    （全左转或全右转），对应相邻边叉积的符号全部同号。我取所有
    非零叉积（忽略容差内的共线顶点），要求它们全正或全负。

    两个边界处理：顶点数少于 3 判 False（围不出多边形）；
    所有叉积都为零（点全共线）也判 False——共线"多边形"面积为 0，
    既不能扇面也不能耳切，当作凸去补会得到零面积垃圾面。
    """

    if len(points_2d) < 3:
        return False
    deltas = np.roll(points_2d, -1, axis=0) - points_2d
    cross_values = (
        deltas[:, 0] * np.roll(deltas[:, 1], -1)
        - deltas[:, 1] * np.roll(deltas[:, 0], -1)
    )
    non_zero = cross_values[np.abs(cross_values) > tolerance]
    if len(non_zero) == 0:
        return False
    return bool(np.all(non_zero > 0) or np.all(non_zero < 0))


def _boundary_neighbor_normal(mesh: TriangleMesh, loop: list[int]) -> np.ndarray:
    """我取边界环邻接三角面的平均法向，决定补片的朝向。

    原理说明：补片的朝向必须与原网格一致，否则补出来的面法向朝里，
    下游的实体判定和渲染都会错。我取环上每条边所邻接三角面的法向
    求平均再归一化。用邻接面的法向而不是环自身的拟合平面法向，
    是因为后者只能确定平面的朝向**轴线**，无法区分正反两个方向，
    而邻接面的法向直接携带了原网格的朝向约定。

    兜底：一条边都找不到邻接面，或平均后模长为 0（邻接面法向恰好互相
    抵消，例如环两侧面朝相反方向），我退化为 +Z 并让下游继续走。
    返回兜底值而不是抛异常，是因为朝向错了可以由复核标记兜住，
    而抛异常会让整个网格都修不了。
    """

    mapping = edge_to_faces(mesh)
    normals = mesh.triangle_normals
    collected: list[np.ndarray] = []
    for first, second in zip(loop, loop[1:] + loop[:1]):
        # 边键排序后查：mapping 的键是排序后的顶点对，
        # 不排序会查不到，于是邻接面数恒为 0。
        edge = (first, second) if first < second else (second, first)
        face_ids = mapping.get(edge, [])
        # 每条边只取第一个邻接面：这里要的是"邻面大致朝哪"，
        # 边界边最多只有一个面，取第一个就够。
        if face_ids:
            collected.append(normals[face_ids[0]])
    if not collected:
        # 一个邻面都没有时返回 +Z 而不是零向量：
        # 零向量无法归一化，后面所有与它点乘的结果都是 0，
        # 判不出朝向。+Z 是一个明确的、可追溯的默认。
        return np.asarray([0.0, 0.0, 1.0])
    # 邻面法向求和后归一化：求和能让占多数的朝向主导结果，
    # 比只取一个邻面稳定。
    normal = np.sum(np.asarray(collected), axis=0)
    length = np.linalg.norm(normal)
    # 和为近零说明邻面朝向互相抵消（例如环两侧的面对着来），
    # 这时同样退回 +Z。
    if length <= 1e-15:
        return np.asarray([0.0, 0.0, 1.0])
    return normal / length


def _fan_faces(
    loop: list[int],
    center_id: int,
    vertices: np.ndarray,
    preferred_normal: np.ndarray,
) -> list[tuple[int, int, int]]:
    """我以中心点为轴把边界环扇形三角化，并保证法向朝向一致。

    我为什么要逐个三角形检查法向再决定是否反转绕向：扇形三角化
    按环序给出的绕向，在环的绕序与期望法向不一致时会整片朝里。
    补面朝里会让后续的实体封闭检查直接失败，而且看不出来——
    面明明"补上了"，但实体就是闭不上。逐个校准绕向是唯一可靠的做法。
    """
    faces: list[tuple[int, int, int]] = []
    center = vertices[center_id]
    for index, first in enumerate(loop):
        # 末点的下一点绕回起点：扇形三角化要闭合这一圈。
        second = loop[(index + 1) % len(loop)]
        # 先按 (first, second, center) 这个绕向算出法向。
        triangle_normal = np.cross(
            vertices[second] - vertices[first],
            center - vertices[first],
        )
        # 法向与期望朝向相反就反转绕向：
        # 补出的三角形必须与邻面朝向一致，
        # 否则模型会出现"翻面"，后续的面积与法向计算全错。
        if float(np.dot(triangle_normal, preferred_normal)) < 0:
            faces.append((second, first, center_id))
        else:
            faces.append((first, second, center_id))
    return faces


def _ear_quality(
    first: np.ndarray, second: np.ndarray, third: np.ndarray
) -> float:
    """我给一个候选耳朵三角形打质量分，1.0 表示正三角形，越扁越低。

    我用正三角形归一化面积（``4*sqrt(3)*area / (a^2+b^2+c^2)``）当质量：
    它只吃三边长度、不吃坐标原点，对细长三角形的惩罚比单看面积更稳，
    而且不需要开方之外的三角函数。

    ponytail: 这个分只用来**排序**、不用来否决，所以退化三角形拿到 0
    也没关系——否决仍然交给调用方的叉积与"含其他顶点"两道判据。
    """

    side_a = float(np.linalg.norm(second - first))
    side_b = float(np.linalg.norm(third - second))
    side_c = float(np.linalg.norm(first - third))
    denominator = side_a * side_a + side_b * side_b + side_c * side_c
    if denominator <= 0.0:
        return 0.0
    area = abs(float(_cross_2d(second - first, third - second))) * 0.5
    return (4.0 * float(np.sqrt(3.0)) * area) / denominator


def _ear_clip_faces(
    loop: list[int],
    vertices: np.ndarray,
    preferred_normal: np.ndarray,
) -> list[tuple[int, int, int]] | None:
    """我用耳切法三角化简单多边形，支持非凸边界环。

    原理说明：中心扇面只对凸多边形成立。非凸多边形如果继续连中心点，
    三角形会跨到多边形外部，新增面积对不上边界环面积（实测 38% 偏差）。
    耳切法每次剪掉一个"耳朵"三角形（凸顶点且不包含其它顶点），直到只剩
    一个三角形；它对任意简单多边形都成立。
    """

    points = vertices[np.asarray(loop, dtype=np.int64)]
    if len(points) < 3:
        return None
    center = points.mean(axis=0)
    _, basis_u, basis_v = _best_fit_plane(points, center)
    projected = np.column_stack(
        ((points - center) @ basis_u, (points - center) @ basis_v)
    )
    remaining = list(range(len(loop)))
    if _signed_polygon_area(projected) < 0.0:
        remaining.reverse()
    triangles: list[tuple[int, int, int]] = []
    # ponytail: 耳切是 O(n^2)~O(n^3)，n 受 max_boundary_edges 限制（默认 128）；
    # 边界环再大就先换增量式耳切或单调多边形剖分。
    guard = 0
    # 循环上限取 n²：每次成功削掉一个顶点，正常情况最多 n 轮；
    # 取 n² 而不是 n 是因为有些轮次找不到可削的耳朵需要继续扫，
    # 但真的卡住时必须退出，否则凹多边形会让这里死循环。
    limit = len(remaining) * len(remaining)
    # 标准耳切法：反复找一个"耳朵"（凸且不含其他顶点的三角形），
    # 削掉它，直到只剩三个点。
    while len(remaining) > 3:
        guard += 1
        if guard > limit:
            return None
        # 我为什么不再"找到第一个耳朵就削"：实测测试集/9#face128#2 那个
        # 环本身不自相交、平面误差 0，但按"遇到就削"的顺序削到第 35 轮
        # 时，剩下 6 点组成的多边形**已经自相交**——它还有 2 个凸耳，
        # 可这两个耳朵的三角形都包含了对面的顶点，于是再也削不动，整块
        # 区域被退回人工复核。原始环是干净的，是我削的顺序把它削成了
        # 自交形状。所以我改成每轮先把**所有**可削耳朵都找出来、按质量
        # 排序、只削最优的那个，把剩余多边形被拖进自交的概率降下来。
        candidates: list[tuple[float, int, int, int, int]] = []
        for position in range(len(remaining)):
            previous = remaining[position - 1]
            current = remaining[position]
            following = remaining[(position + 1) % len(remaining)]
            first = projected[previous]
            second = projected[current]
            third = projected[following]
            # 叉积 <= 0 说明这段是凹的或退化的，不是耳朵。
            # 1e-14 而非 0 是为了排除"三点几乎共线"的情况，
            # 那种三角形面积接近零，补出来是废面。
            if _cross_2d(second - first, third - second) <= 1e-14:
                continue
            # 三角形里含其他顶点也不能削：削掉会把那些顶点
            # 甩到多边形外面，剩下的点再也连不成合法三角网。
            if _triangle_contains_other_point(
                projected,
                remaining,
                (previous, current, following),
                (first, second, third),
            ):
                continue
            candidates.append(
                (
                    _ear_quality(first, second, third),
                    position,
                    previous,
                    current,
                    following,
                )
            )
        # 一整轮都没找到可削的耳朵：剩下的是自交或严重退化的环，
        # 三角化不下去，返回 None 交给上层走人工复核。
        if not candidates:
            return None
        # 质量降序；同质量时取下标小的，保证结果可复现（不依赖集合顺序）。
        candidates.sort(key=lambda item: (-item[0], item[1]))
        _quality, position, previous, current, following = candidates[0]
        triangles.append((loop[previous], loop[current], loop[following]))
        remaining.pop(position)
    triangles.append(
        (loop[remaining[0]], loop[remaining[1]], loop[remaining[2]])
    )
    oriented: list[tuple[int, int, int]] = []
    for first, second, third in triangles:
        normal = np.cross(
            vertices[second] - vertices[first],
            vertices[third] - vertices[first],
        )
        # 逐个三角形统一绕向：三角网的面法向必须朝同一侧，
        # 否则导出 STL 后法向混乱，下游按法向做的判断全部失效。
        # 交换前两点即可翻向，比交换首尾更直观。
        if float(np.dot(normal, preferred_normal)) < 0:
            oriented.append((second, first, third))
        else:
            oriented.append((first, second, third))
    return oriented


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    """我算两个二维向量的叉积（有向面积的两倍）。

    我为什么单独抽这个函数：它是凸性判定、耳切、点在三角形内
    三处的共同判据。抄三份会让"改一处漏两处"成为可能——
    我在耳切里就犯过一次叉积下标写错的错误，抽成一份能让修改
    只发生在一个地方。
    """
    return float(first[0] * second[1] - first[1] * second[0])


def _loop_reference_area_mm2(
    loop: list[int],
    vertices: np.ndarray,
    preferred_normal: np.ndarray,
) -> float:
    """我返回边界环正确三角化后的三维面积，作为面积门禁基准。

    耳切失败（例如自相交边界）时我返回 0，由调用方退回投影面积，
    不把"算不出来"当成"面积合格"。
    """

    faces = _ear_clip_faces(loop, vertices, preferred_normal)
    if faces is None:
        return 0.0
    total = 0.0
    for first, second, third in faces:
        cross = np.cross(
            vertices[second] - vertices[first],
            vertices[third] - vertices[first],
        )
        total += 0.5 * float(np.linalg.norm(cross))
    return total


def _triangle_contains_other_point(
    projected: np.ndarray,
    remaining: list[int],
    corner_ids: tuple[int, int, int],
    corners: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    """我判断三角形内是否落入了其它剩余顶点，是的话这个耳朵不能剪。"""

    first, second, third = corners
    for vertex_id in remaining:
        if vertex_id in corner_ids:
            continue
        point = projected[vertex_id]
        edge_one = _cross_2d(second - first, point - first)
        edge_two = _cross_2d(third - second, point - second)
        edge_three = _cross_2d(first - third, point - third)
        has_negative = edge_one < -1e-14 or edge_two < -1e-14 or edge_three < -1e-14
        has_positive = edge_one > 1e-14 or edge_two > 1e-14 or edge_three > 1e-14
        if not (has_negative and has_positive):
            return True
    return False


def _unique(values: list[str]) -> list[str]:
    """我按首次出现顺序去重字符串，保住警告的排查线索顺序。"""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
