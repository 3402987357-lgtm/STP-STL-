"""我从 STEP Face 和共享 Edge 中提取模具工程特征候选。

关键算法原理（本模块的设计依据，供甲方与工程师追溯）：

一、只读一次，复用优先。我先调用 ``inspect_step_face_features`` 读一次 STEP，拿到
    每个 Face 的面积、中心、法向、包围盒、圆弧边半径和圆柱轴线证据，再复用
    ``classify_step_face_features`` 已经算好的平面方向分组与共享 Edge 邻接。
    工程特征分类阶段不再读盘、不再三角化，保证单套零件在 3 分钟预算内可跑完。

二、台阶候选（STEP_CANDIDATE）。我遍历共享 Edge 相邻的 Face 对：法向夹角大于等于
    ``min_step_angle_deg``，且两 Face 中心在抽模方向上的高度差大于等于
    ``max(min_step_height_mm, span_z * step_height_ratio)`` 时记为台阶边界候选。
    高度阈值取"绝对毫米下限"和"零件高度比例"两者的较大值，目的是让大零件不被
    固定毫米阈值淹没、小零件不被比例阈值放大，兼顾不同尺寸模具。

三、圆柱/孔候选（CYLINDER_OR_HOLE_CANDIDATE）。对每个圆柱面统计圆弧边数量与半径
    相对离散度：离散度小于 ``hole_radius_tolerance_ratio``、Edge 特征数组对齐、
    且具备内部腔体语义时，才给出"闭合孔壁代理"证据。语义不稳定的圆柱面只保留为
    几何候选，绝不直接当作孔，避免把外圆柱误判成孔。

四、R 角候选（R_CORNER_CANDIDATE）。共享 Edge 带可用半径、且两侧 Face 法向夹角落在
    (1°, 179°) 时记为圆角过渡候选。圆弧 Edge 只作为过渡证据，不作为最终圆角加工特征。

五、同轴孔阵列（COAXIAL_HOLE_ARRAY_CANDIDATE）。为避免 O(n^2) 全量两两比较，我先把
    圆柱轴线量化成"方向桶 + 垂足桶"，再只在桶内用轴线夹角、轴线距离、半径比三个
    容差聚类，把候选规模压到线性级别。

六、区域级证据。把上面候选回填到 ``StepFeatureRegion``，生成角色候选、类别候选、
    分型候选和人工复核队列。所有结构化输出都带 ``method`` 与 ``interpretation``
    字段，方便甲方追溯每条结论来自哪条规则、还缺什么证据。

设计边界：本模块只输出"候选 + 证据"，不输出型腔、型芯、最终分型线或加工工艺；
所有阈值都是保守默认值，必须结合甲方坐标系规则与人工标注样本验收后再收紧。
"""

from __future__ import annotations

import time
from math import acos, degrees, radians, sin
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np

from ..domain.step_engineering import (
    StepDraftEvidence,
    StepEngineeringCandidate,
    StepEngineeringRegion,
    StepEngineeringReport,
)
from ..domain.step_features import StepFeatureRegion
from ..io.step_faces import StepFaceFeature, StepFaceFeatureReport, inspect_step_face_features
from .step_features import classify_step_face_features


def inspect_step_engineering_features(
    path: str | Path,
    *,
    draw_direction: Sequence[float] = (0.0, 0.0, 1.0),
    draft_tolerance_deg: float = 15.0,
    step_height_ratio: float = 0.02,
    min_step_height_mm: float = 0.5,
    min_step_angle_deg: float = 15.0,
    horizontal_tolerance_deg: float = 15.0,
    z_band_ratio: float = 0.1,
    hole_radius_tolerance_ratio: float = 0.05,
    axis_angle_tolerance_deg: float = 5.0,
    axis_distance_tolerance_mm: float = 0.5,
    semantic_sample_limit: int = 16,
    validate_shape: bool = True,
) -> StepEngineeringReport:
    """我读取 STEP 一次，并输出模具工程特征候选。"""

    started = time.perf_counter()
    face_report = inspect_step_face_features(
        path,
        validate_shape=validate_shape,
        # 语义分类必须开：工程特征（孔、台阶、R 角）的判据
        # 依赖面的内外语义，不开就判不出来。
        classify_semantics=True,
        # 采样上限透传：语义分类在大件上是主要耗时项，
        # 不给上限会在几万面的件上跑很久。
        semantic_sample_limit=semantic_sample_limit,
    )
    result = classify_step_engineering_features(
        face_report,
        # 以下九个参数全部显式透传而不是让被调用方用默认值：
        # 这些阈值是本模块的对外契约，
        # 由调用方决定，写死在被调用方会让调参无法进行。
        draw_direction=draw_direction,
        draft_tolerance_deg=draft_tolerance_deg,
        step_height_ratio=step_height_ratio,
        min_step_height_mm=min_step_height_mm,
        min_step_angle_deg=min_step_angle_deg,
        horizontal_tolerance_deg=horizontal_tolerance_deg,
        z_band_ratio=z_band_ratio,
        hole_radius_tolerance_ratio=hole_radius_tolerance_ratio,
        axis_angle_tolerance_deg=axis_angle_tolerance_deg,
        axis_distance_tolerance_mm=axis_distance_tolerance_mm,
    )
    # 耗时由这一层统一记：读文件与特征分类都算在内，
    # 于是"这一趟工程识别多久"有唯一出处。
    result.elapsed_seconds = time.perf_counter() - started
    return result


def classify_step_engineering_features(
    report: StepFaceFeatureReport,
    *,
    draw_direction: Sequence[float] = (0.0, 0.0, 1.0),
    draft_tolerance_deg: float = 15.0,
    step_height_ratio: float = 0.02,
    min_step_height_mm: float = 0.5,
    min_step_angle_deg: float = 15.0,
    horizontal_tolerance_deg: float = 15.0,
    z_band_ratio: float = 0.1,
    hole_radius_tolerance_ratio: float = 0.05,
    axis_angle_tolerance_deg: float = 5.0,
    axis_distance_tolerance_mm: float = 0.5,
) -> StepEngineeringReport:
    """我复用已有 Face 分类和共享 Edge 结果，避免重复读取 STEP。"""

    # 抽模方向必须能归一化，否则后面所有"沿方向投影"的判据都无意义。
    direction = _unit_vector(draw_direction, "draw_direction")
    # 以下逐项校验阈值区间。区间边界都由几何含义决定，例如法向夹角不可能达到 90°
    # 以上仍称为"台阶"，所以上限收在 90° 以内；任何越界都直接报错而不是静默截断。
    if not 0.0 < draft_tolerance_deg < 45.0:
        raise ValueError("我要求 draft_tolerance_deg 位于 (0, 45) 区间。")
    if not 0.0 < step_height_ratio < 0.5:
        raise ValueError("我要求 step_height_ratio 位于 (0, 0.5) 区间。")
    if not np.isfinite(min_step_height_mm) or min_step_height_mm <= 0.0:
        raise ValueError("我要求 min_step_height_mm 是大于 0 的有限数。")
    if not 0.0 < min_step_angle_deg < 90.0:
        raise ValueError("我要求 min_step_angle_deg 位于 (0, 90) 区间。")
    if not 0.0 < horizontal_tolerance_deg < 45.0:
        raise ValueError("我要求 horizontal_tolerance_deg 位于 (0, 45) 区间。")
    if not 0.0 < z_band_ratio < 0.5:
        raise ValueError("我要求 z_band_ratio 位于 (0, 0.5) 区间。")
    if not 0.0 <= hole_radius_tolerance_ratio < 1.0:
        raise ValueError("我要求 hole_radius_tolerance_ratio 位于 [0, 1) 区间。")
    if not 0.0 < axis_angle_tolerance_deg < 45.0:
        raise ValueError("我要求 axis_angle_tolerance_deg 位于 (0, 45) 区间。")
    if (
        not np.isfinite(axis_distance_tolerance_mm)
        or axis_distance_tolerance_mm <= 0.0
    ):
        raise ValueError("我要求 axis_distance_tolerance_mm 是大于 0 的有限数。")

    # 先隔离几何字段缺失或 Face 编号非法的输入，后续所有规则只在有效 Face 上运行。
    valid_faces = [
        face
        for face in report.faces
        if _valid_face(face) and _valid_face_id(face.face_id)
    ]
    face_by_id = {face.face_id: face for face in valid_faces}
    warnings = list(report.warnings)
    if len(valid_faces) != len(report.faces):
        warnings.append("我隔离了缺少有限几何字段或非法 Face 编号的输入。")

    # 用整体包围盒高度 span_z 把"比例阈值"折算成毫米，与绝对下限取较大者。
    bbox_min, bbox_max = _combined_bbox(valid_faces)
    span_z = (
        max(float(bbox_max[2] - bbox_min[2]), 1e-9)
        if bbox_min is not None and bbox_max is not None
        else 1.0
    )
    height_threshold = max(float(min_step_height_mm), span_z * float(step_height_ratio))
    classification = classify_step_face_features(
        report,
        horizontal_tolerance_deg=horizontal_tolerance_deg,
        z_band_ratio=z_band_ratio,
    )
    adjacency_by_face = {item.face_id: item for item in classification.adjacency}

    candidates: list[StepEngineeringCandidate] = []
    candidate_id = 0
    seen_pairs: set[tuple[int, int]] = set()
    # 第一轮：台阶候选。按排序后的 Face 编号遍历共享 Edge 邻接，seen_pairs 去重保证
    # 每个无序 Face 对只处理一次，避免 (i,j) 与 (j,i) 重复产出候选。
    for left_id in sorted(adjacency_by_face):
        adjacency = adjacency_by_face[left_id]
        left = face_by_id.get(left_id)
        if left is None:
            # 邻接表里有编号但找不到面对应：数据不一致，
            # 跳过这一条而不是当成"没有邻居"。
            continue
        for right_id in adjacency.neighbor_face_ids:
            # 面号排序后成对去重：邻接是双向的，
            # 不排序去重会把同一对面当两个候选各算一次。
            pair = tuple(sorted((left_id, right_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            right = face_by_id.get(right_id)
            if right is None:
                continue
            shared_edge_ids = _shared_edge_ids(left, right)
            # 没有公共边就不是真的相邻：邻接表里可能记录了
            # "包围盒接近"这类弱关系，没有公共边的两面
            # 之间不存在台阶。
            if not shared_edge_ids:
                continue
            angle = _unsigned_normal_angle(left.normal_xyz, right.normal_xyz)
            # 高度差只取 Z 分量：台阶是"沿拔模方向的高度突变"，
            # 用三维距离会把斜面上的两个面也算成台阶。
            height_difference = abs(
                float(left.center_xyz_mm[2]) - float(right.center_xyz_mm[2])
            )
            # 夹角与高度差两个条件都要满足：
            # 只有夹角说明是"折弯"，只有高度差说明是"平缓错开"，
            # 两者都不是台阶。
            if angle is not None and angle >= min_step_angle_deg and height_difference >= height_threshold:
                candidate_id += 1
                # 两个分项各自归一化到 [0,1] 再加权：
                # 夹角单位是度、高度差单位是毫米，
                # 不归一化直接相加会让量纲大的那项支配整个分数。
                angle_score = min(angle / 90.0, 1.0)
                height_score = min(height_difference / max(height_threshold * 4.0, 1e-9), 1.0)
                # 夹角占 0.6、高度占 0.4：台阶的本质是"面的朝向突变"，
                # 高度差是辅助佐证，所以夹角权重更高。
                score = round(0.6 * angle_score + 0.4 * height_score, 6)
                candidates.append(
                    StepEngineeringCandidate(
                        candidate_id=candidate_id,
                        kind="STEP_CANDIDATE",
                        face_ids=[left_id, right_id],
                        edge_ids=shared_edge_ids,
                        score=score,
                        # 证据里同时给出三个原始量：分数是压缩后的结论，
                        # 复核者需要看到原始量才能判断这个分数该不该采信。
                        evidence={
                            "normal_angle_deg": round(angle, 6),
                            "height_difference_mm": round(height_difference, 6),
                            "center_distance_mm": round(
                                _center_distance(left.center_xyz_mm, right.center_xyz_mm),
                                6,
                            ),
                            "surface_types": [left.surface_type, right.surface_type],
                            "area_mm2": [left.area_mm2, right.area_mm2],
                            "height_threshold_mm": round(height_threshold, 6),
                            "method": "shared_edge_angle_and_center_height",
                        },
                        warnings=[
                            "我把该记录作为台阶边界候选，不把它直接判定为最终分型线。"
                        ],
                    )
                )

    # 第二轮：圆柱/孔候选。只对圆柱面提取半径证据，其他曲面类型直接跳过。
    for face in valid_faces:
        if "Cylinder" not in face.surface_type:
            continue
        radii = _edge_radii(face)
        hole_evidence = _hole_topology_evidence(
            face,
            radii,
            hole_radius_tolerance_ratio,
        )
        candidate_id += 1
        warning = list(hole_evidence.pop("warnings"))
        candidates.append(
            StepEngineeringCandidate(
                candidate_id=candidate_id,
                kind="CYLINDER_OR_HOLE_CANDIDATE",
                face_ids=[face.face_id],
                edge_ids=list(face.circular_edge_ids),
                score=round(float(hole_evidence["score"]), 6),
                evidence={
                    "surface_type": face.surface_type,
                    "radius_mm": round(float(median(radii)), 6) if radii else None,
                    "radius_samples_mm": [round(float(value), 6) for value in radii],
                    "circular_edge_count": len(face.circular_edge_ids),
                    "area_mm2": face.area_mm2,
                    "method": "cylindrical_face_and_circular_edge",
                    **hole_evidence,
                },
                warnings=warning,
            )
        )

    # 第三轮：同轴孔阵列候选。用分桶后的轴线聚类，返回的候选编号紧接前面的计数。
    coaxial_candidates = _coaxial_hole_array_candidates(
        valid_faces,
        start_candidate_id=candidate_id + 1,
        axis_angle_tolerance_deg=axis_angle_tolerance_deg,
        axis_distance_tolerance_mm=axis_distance_tolerance_mm,
        radius_tolerance_ratio=hole_radius_tolerance_ratio,
    )
    candidates.extend(coaxial_candidates)
    candidate_id += len(coaxial_candidates)

    # 第四轮：R 角候选。再次遍历共享 Edge，只挑带可用半径且法向夹角合理的边。
    for left_id in sorted(adjacency_by_face):
        left = face_by_id.get(left_id)
        # 邻接表里有的面可能不在面表里（被前面过滤掉了），跳过。
        if left is None:
            continue
        for right_id in adjacency_by_face[left_id].neighbor_face_ids:
            # 只处理编号左小右大的一对：
            # 邻接是对称的，不加这个判断每一对会被处理两次，
            # 于是同一个 R 角会产出两条编号不同的重复候选。
            if left_id >= right_id:
                continue
            right = face_by_id.get(right_id)
            if right is None:
                continue
            # 两个面共用的圆边才是 R 角的候选边：
            # 直边相接是尖角或台阶，不是圆角过渡。
            shared_edge_ids = _shared_edge_ids(left, right)
            for edge_id in shared_edge_ids:
                radius = _shared_edge_radius(left, right, edge_id)
                angle = _unsigned_normal_angle(left.normal_xyz, right.normal_xyz)
                # 夹角必须在 1 到 179 度之间：
                # 接近 0 说明两个面几乎共面（那是平面拼接不是角），
                # 接近 180 说明它们背靠背（那是薄壁的两面）。
                if radius is None or angle is None or not 1.0 <= angle <= 179.0:
                    continue
                candidate_id += 1
                # 打分 = 0.5 + 半径相对整体尺寸的比例（上限 0.5）：
                # 半径越大的 R 角越显著，但基础分 0.5 保证
                # 小圆角也不会因为分低被直接滤掉。
                # 分母用 max(span_z, 1e-9) 防止零尺寸时除零。
                score = min(1.0, 0.5 + min(radius / max(span_z, 1e-9), 0.5))
                candidates.append(
                    StepEngineeringCandidate(
                        candidate_id=candidate_id,
                        kind="R_CORNER_CANDIDATE",
                        face_ids=[left_id, right_id],
                        edge_ids=[edge_id],
                        score=round(float(score), 6),
                        evidence={
                            # 半径与法向夹角：这两个量就是判 R 角的全部依据，
                            # 复核者靠它们决定这是圆角还是别的东西。
                            "radius_mm": round(float(radius), 6),
                            "normal_angle_deg": round(float(angle), 6),
                            # 两个面的曲面类型一并记：圆角两侧的材质
                            # 决定了它是不是加工出来的 R 角。
                            "surface_types": [left.surface_type, right.surface_type],
                            "method": "circular_edge_radius_and_face_angle",
                        },
                        warnings=[
                            # 能力边界：几何上的圆边不等于工艺上的圆角，
                            # 建模里用圆弧边的地方未必都要倒角。
                            "我把该记录作为 R 角或圆弧过渡候选，不把圆弧 Edge 自动解释成最终圆角加工特征。"
                        ],
                    )
                )

    # 第五轮：拔模方向证据。逐个 Face 记录法向与抽模方向的夹角关系。
    draft_evidence = [
        _draft_evidence(face, direction, draft_tolerance_deg)
        for face in valid_faces
    ]
    if not candidates:
        warnings.append("我没有得到满足当前阈值的台阶、圆柱或 R 角候选。")
    # 这两条是"能力边界"声明而不是错误：本模块只产出证据，
    # 不替甲方判定型腔/型芯与最终工艺，所以每次都要随结果一起带出去，
    # 避免下游把候选名单当成最终结论。
    warnings.extend(
        [
            "我输出的是模具工程特征证据，仍需要甲方坐标规则、完整拓扑和人工样本验收。",
            "我没有自动确定型腔、型芯、最终分型线或加工工艺。",
        ]
    )
    # 下面一串 *_counts 都是"分布"而非"总数"：总数只说明有多少候选，
    # 分布才能暴露阈值是否把某一类特征整体误判掉了，
    # 例如 step 为 0 往往意味着台阶阈值偏严，需要回看阈值而不是只看总量。
    candidate_counts: dict[str, int] = {}
    for candidate in candidates:
        candidate_counts[candidate.kind] = candidate_counts.get(candidate.kind, 0) + 1
    # 拔模证据按分类计数同理：全落到 same_direction 说明抽模方向可能取反了。
    draft_counts: dict[str, int] = {}
    for item in draft_evidence:
        draft_counts[item.classification] = draft_counts.get(item.classification, 0) + 1
    region_evidence = _build_region_evidence(
        classification.regions,
        candidates,
        valid_faces,
        shape_is_valid=report.shape_is_valid,
    )
    region_status_counts: dict[str, int] = {}
    role_candidate_counts: dict[str, int] = {}
    category_candidate_counts: dict[str, int] = {}
    boundary_transition_counts: dict[str, int] = {}
    review_reason_counts: dict[str, int] = {}
    review_priority_counts: dict[str, int] = {}
    parting_role_counts: dict[str, int] = {}
    boundary_topology_status_counts: dict[str, int] = {}
    # 逐区域累加七个分布。一个区域可能同时带多个角色/类别候选，
    # 所以这里是"候选出现次数"计数，不是"区域数"计数，两者不可混用。
    for region in region_evidence:
        region_status_counts[region.engineering_status] = (
            region_status_counts.get(region.engineering_status, 0) + 1
        )
        # 角色/类别/分型角色三者的计数基数不同：
        # 区域状态是"每区域计一次"，而这三者是"每出现一个候选计一次"。
        # 一个区域可以有多个角色候选，所以它们的总和会大于区域数，这是预期的。
        for role_candidate in region.role_candidates:
            role_candidate_counts[role_candidate] = (
                role_candidate_counts.get(role_candidate, 0) + 1
            )
        # 四套分布各自独立统计：角色候选、类别候选、
        # 边界过渡类型、复核原因。它们由不同的判据得出，
        # 合并成一个计数就看不出"卡在哪个环节"。
        for category in region.category_candidates:
            category_candidate_counts[category] = (
                category_candidate_counts.get(category, 0) + 1
            )
        # 过渡类型埋在邻域证据里，取不到就按空列表处理：
        # 老版本的数据没有这个键，不设默认值会 KeyError。
        for transition in region.evidence.get(
            "neighbor_evidence",
            {},
        ).get("role_transition_candidates", []):
            boundary_transition_counts[transition] = (
                boundary_transition_counts.get(transition, 0) + 1
            )
        for reason_code in region.review_reason_codes:
            review_reason_counts[reason_code] = (
                review_reason_counts.get(reason_code, 0) + 1
            )
        # 复核优先级每条区域只有一个，直接取不加循环：
        # 它是区域级的结论，上面的原因码则是清单。
        review_priority_counts[region.review_priority] = (
            review_priority_counts.get(region.review_priority, 0) + 1
        )
        for parting_role in region.parting_role_candidates:
            parting_role_counts[parting_role] = (
                parting_role_counts.get(parting_role, 0) + 1
            )
        # 边界拓扑状态也是区域级单值：它描述这块区域的
        # 边界本身可不可信，是后面所有边界结论的前提。
        boundary_topology_status = region.boundary_topology_status
        boundary_topology_status_counts[boundary_topology_status] = (
            boundary_topology_status_counts.get(boundary_topology_status, 0) + 1
        )
    plane_role_summary = _plane_role_summary(classification.groups)
    # 孔候选单独筛出一份：孔是这套工程语义里
    # 最需要单独体检的一类特征。
    hole_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "CYLINDER_OR_HOLE_CANDIDATE"
    ]
    hole_topology_summary = {
        # 这一组是"孔候选的质量体检表"：除了总数，我刻意统计三类异常计数——
        # 语义未定、半径不一致、闭合代理缺失。它们的用途是让甲方一眼看出
        # "孔候选多"到底是因为真有孔，还是因为阈值松了把外圆柱也算了进来。
        # 任一异常计数偏高时，应当先调阈值而不是直接采信孔的数量。
        "candidate_count": len(hole_candidates),
        # 闭合代理为真的孔壁，才可能真的是一个贯通的孔；
        # 没有闭合代理的圆柱面多半只是零件外表面的柱段。
        "closed_boundary_proxy_count": sum(
            bool(candidate.evidence.get("closed_boundary_proxy"))
            for candidate in hole_candidates
        ),
        "hole_wall_topology_candidate_count": sum(
            candidate.evidence.get("hole_semantics") == "HOLE_WALL_CANDIDATE"
            for candidate in hole_candidates
        ),
        # 内孔与外圆柱几何上都是圆柱面，工程含义却相反（一个是孔、一个是轴）。
        # 我靠实体内外语义区分；语义未定时两者都不计入内孔，宁可少报也不误报。
        "internal_hole_candidate_count": sum(
            candidate.evidence.get("hole_semantics") == "HOLE_WALL_CANDIDATE"
            and candidate.evidence.get("semantic_status")
            == "INTERNAL_CAVITY_CANDIDATE"
            for candidate in hole_candidates
        ),
        "external_cylinder_candidate_count": sum(
            candidate.evidence.get("hole_semantics")
            == "EXTERNAL_CYLINDER_FACE_CANDIDATE"
            for candidate in hole_candidates
        ),
        "semantic_unresolved_count": sum(
            candidate.evidence.get("semantic_status")
            in {"UNRESOLVED", "REVIEW_REQUIRED"}
            for candidate in hole_candidates
        ),
        # 半径不一致说明这个"孔"可能是锥孔、阶梯孔，或若干不同半径的柱面
        # 被误聚成一个候选。它不直接否定判定，但必须计数让复核时看得见。
        "radius_inconsistent_count": sum(
            not bool(candidate.evidence.get("radius_consistent"))
            for candidate in hole_candidates
        ),
        # 需复核总数：闭合代理缺失或语义未定二者居其一。
        # 这是给甲方排复核工作量的口径，除以 candidate_count 即得可信比例。
        "topology_review_required_count": sum(
            not bool(candidate.evidence.get("closed_boundary_proxy"))
            or candidate.evidence.get("semantic_status")
            in {"UNRESOLVED", "REVIEW_REQUIRED"}
            for candidate in hole_candidates
        ),
        # 同轴阵列按"组"计数，面数按去重后的 Face 计数：
        # 同一个面可能同时属于多个同轴组，所以两者不可互相推算。
        "coaxial_array_candidate_count": len(coaxial_candidates),
        "coaxial_array_face_count": len(
            {
                face_id
                for candidate in coaxial_candidates
                for face_id in candidate.face_ids
            }
        ),
        "interpretation": "孔壁候选仍需要完整闭合拓扑、轴向连续性和实体语义复核。",
    }
    intermediate_draft_count = draft_counts.get(
        "INTERMEDIATE_OR_DRAFT_CANDIDATE",
        0,
    )
    # rule_summary 是给甲方看的"规则体检总表"，三段各自独立：
    #   平面角色分布 / 孔拓扑质量 / 拔模方向分布 / 区域工程汇总。
    # 每段都同时给"计数分布"和"需复核数量"，因为只有分布看不出
    # 结论是否可信——必须知道这里面有多少是待复核的。
    rule_summary = {
        "plane_roles": plane_role_summary,
        "hole_topology": hole_topology_summary,
        # 拔模方向：只要存在中间态/拔模候选就置 review_required。
        # 中间态意味着这批面既不顺向也不逆向，无法据此判定拔模可行性。
        "draft_direction": {
            "classification_counts": dict(sorted(draft_counts.items())),
            "intermediate_or_draft_count": intermediate_draft_count,
            "review_required": intermediate_draft_count > 0,
        },
        # 区域工程汇总里所有 dict 都 sorted 后再落盘：
        # 键顺序稳定才能让两次运行的报告逐字节相同，便于做可复现比对。
        "region_engineering": {
            # 区域总数先给：后面各分布的总量都应与它对得上，
            # 对不上就说明某一步漏统计了。
            "region_count": len(region_evidence),
            # 以下各分布全部排序后输出：
            # 键的顺序固定，跨批次 diff 才能看出真变化。
            "status_counts": dict(sorted(region_status_counts.items())),
            "role_candidate_counts": dict(sorted(role_candidate_counts.items())),
            "category_candidate_counts": dict(sorted(category_candidate_counts.items())),
            # 两类"需要复核"的计数用求和而不是再走一遍字典：
            # 它们是区域级的布尔，求和即总数，
            # 不需要维护第二套计数。
            "category_review_required_count": sum(
                region.category_review_required for region in region_evidence
            ),
            "boundary_transition_counts": dict(
                sorted(boundary_transition_counts.items())
            ),
            "review_reason_counts": dict(sorted(review_reason_counts.items())),
            "review_priority_counts": dict(sorted(review_priority_counts.items())),
            "parting_role_counts": dict(sorted(parting_role_counts.items())),
            "parting_review_required_count": sum(
                region.parting_review_required for region in region_evidence
            ),
            "boundary_topology_status_counts": dict(
                sorted(boundary_topology_status_counts.items())
            ),
            # 含闭合边界环的区域数：闭合环是"这块区域边界完整"的标志，
            # 有闭合环的区域更适合作为分型线的载体。
            "closed_boundary_loop_region_count": sum(
                region.closed_boundary_loop_count > 0
                for region in region_evidence
            ),
            # 语义采样的诊断整份继承：它记录了采样覆盖率等可信度指标，
            # 由上游模块产出，这里透传而不是重算。
            "semantic_sampling": dict(report.diagnostics),
            "review_required_count": sum(
                status == "REVIEW_REQUIRED"
                for status in (region.engineering_status for region in region_evidence)
            ),
            # 方法名固定：这一层是"区域角色与各类候选的连接"。
            "method": "shared_edge_region_role_and_candidate_join",
            # 能力边界：区域角色只是几何上的候选关联，
            # 不是型腔型芯的判定，也不是加工语义结论。
            "interpretation": (
                "区域工程角色只表示当前 Face/Edge 证据的候选关联，"
                "不能替代型腔、型芯和最终加工语义确认。"
            ),
        },
        "review_gates": {
            # 这一组是"交付闸门"：每一项都是一个布尔检查，
            # 供甲方在验收时逐条打勾，也供自动化流程判断能否放行。
            # 它们只陈述事实（形状是否有效、上下平面候选是否存在……），
            # 不做综合判定——最终语义一律写 REVIEW_REQUIRED，不自动放行。
            "shape_valid": report.shape_is_valid is True,
            "all_face_geometry_valid": len(valid_faces) == len(report.faces),
            # 上下平面候选缺失是硬信号：没有这两个面，
            # 就无法建立"型腔在上、型芯在下"的坐标基准，后续角色全不可信。
            "upper_plane_candidate_present": bool(
                plane_role_summary.get("top_plane_candidates", {}).get("face_ids")
            ),
            "lower_plane_candidate_present": bool(
                plane_role_summary.get("bottom_plane_candidates", {}).get("face_ids")
            ),
            "hole_topology_review_required": (
                hole_topology_summary["topology_review_required_count"] > 0
            ),
            # 同轴阵列只要有一组半径不一致就需复核：
            # 阵列孔通常要求同规格，半径不齐意味着分组把不同孔混在了一起。
            "coaxial_array_review_required": bool(
                coaxial_candidates
                and any(
                    not bool(candidate.evidence.get("radius_consistent"))
                    for candidate in coaxial_candidates
                )
            ),
            "draw_direction_review_required": intermediate_draft_count > 0,
            "region_semantics_review_required": any(
                region.engineering_status == "REVIEW_REQUIRED"
                for region in region_evidence
            ),
            # 最终语义恒定 REVIEW_REQUIRED：本模块只产出证据与闸门，
            # 不输出"已确认的型腔/型芯结论"，这是刻意的能力边界。
            "final_engineering_semantics": "REVIEW_REQUIRED",
        },
    }
    status = (
        "BASELINE"
        if (
            report.status == "COMPLETE"
            and classification.status == "BASELINE"
            and report.shape_is_valid is True
            and valid_faces
        )
        else "REVIEW_REQUIRED"
    )
    return StepEngineeringReport(
        status=status,
        file_path=report.file_path,
        unit=report.unit,
        backend="OCP",
        shape_is_valid=report.shape_is_valid,
        face_count=report.face_count,
        bbox_min_xyz_mm=bbox_min,
        bbox_max_xyz_mm=bbox_max,
        candidates=candidates,
        draft_evidence=draft_evidence,
        # dict.fromkeys 去重但保持首次出现顺序：警告文本会重复产生，
        # 去重能让报告可读；保序则保证同一输入两次运行输出完全一致。
        warnings=list(dict.fromkeys(warnings)),
        diagnostics={
            "valid_face_count": len(valid_faces),
            "candidate_count": len(candidates),
            "candidate_counts": dict(sorted(candidate_counts.items())),
            "draft_evidence_count": len(draft_evidence),
            "draft_classification_counts": dict(sorted(draft_counts.items())),
            # 下面这组把本次实际生效的全部阈值原样写回报告：
            # 甲方拿到的结论必须能回答"是用哪套参数算的"，
            # 否则同一份 STEP 换个阈值结果不同时无法追溯。
            "draw_direction_xyz": direction.tolist(),
            # 本轮用到的全部容差都写进诊断：这些值决定了这批结论，
            # 不写进去的话同一份结果换一组容差就无从比较。
            "draft_tolerance_deg": draft_tolerance_deg,
            "step_height_ratio": step_height_ratio,
            "min_step_height_mm": min_step_height_mm,
            "min_step_angle_deg": min_step_angle_deg,
            "horizontal_tolerance_deg": horizontal_tolerance_deg,
            "z_band_ratio": z_band_ratio,
            "hole_radius_tolerance_ratio": hole_radius_tolerance_ratio,
            "axis_angle_tolerance_deg": axis_angle_tolerance_deg,
            "axis_distance_tolerance_mm": axis_distance_tolerance_mm,
            "step_height_threshold_mm": height_threshold,
            # 区域总数与各类"需要复核"的计数：
            # 总数给分母，复核数给分子，
            # 两者一起才能算出"这批有多大比例要人工看"。
            "region_count": len(region_evidence),
            "region_review_required_count": sum(
                region.engineering_status == "REVIEW_REQUIRED"
                for region in region_evidence
            ),
            # 工程、类别、分型三个维度的复核数分开统计：
            # 它们由不同的判据触发，合并成一个数
            # 就看不出"卡在哪个环节"。
            "category_review_required_count": sum(
                region.category_review_required for region in region_evidence
            ),
            # 原因与优先级的分布都排序后输出：
            # 排序保证跨批次 diff 时不会因为字典顺序变化
            # 产生大量无意义的差异。
            "review_reason_counts": dict(sorted(review_reason_counts.items())),
            "review_priority_counts": dict(sorted(review_priority_counts.items())),
            "parting_role_counts": dict(sorted(parting_role_counts.items())),
            "parting_review_required_count": sum(
                region.parting_review_required for region in region_evidence
            ),
            "boundary_topology_status_counts": dict(
                sorted(boundary_topology_status_counts.items())
            ),
            # 有闭合环的区域数：闭合环是"这块区域能可靠量出边界"的信号，
            # 这个计数直接反映本批结果的可信程度。
            "closed_boundary_loop_region_count": sum(
                region.closed_boundary_loop_count > 0
                for region in region_evidence
            ),
            "semantic_sampling": dict(report.diagnostics),
            # 整体语义一律标 REVIEW_REQUIRED：这套工程语义
            # 在任何一批输入上都不能直接当结论用，
            # 写死而不是按条件给，避免某一批恰好都通过时
            # 被读成"这批可以免检"。
            "engineering_semantics": "REVIEW_REQUIRED",
        },
        rule_summary=rule_summary,
        regions=region_evidence,
    )


def _build_region_evidence(
    regions: Iterable[StepFeatureRegion],
    candidates: Sequence[StepEngineeringCandidate],
    faces: Sequence[StepFaceFeature],
    *,
    shape_is_valid: bool | None,
) -> list[StepEngineeringRegion]:
    """我把区域角色和工程候选关联起来，形成可追踪的区域级证据。

    这个函数是整个工程特征识别的汇聚点，所以我把它刻意拆成
    "关联 -> 分类 -> 复核"三段，每段都留下可查的证据：

    1. 关联：区域与候选靠 **Face 交集**挂钩，而不是靠几何距离。
       距离判定需要容差，容差一变结论就变，且无法解释"为什么
       这两个算一组"；共享 Face 是拓扑事实，可逐条核对。
       一个区域可以同时命中多个候选，所以 matched_candidates
       是列表而非单个值。

    2. 分类：角色候选来自两个来源——区域自身的方向角色
       （顶/底/侧壁/斜面）和命中候选的种类（台阶/孔/R角/同轴阵列）。
       两者叠加而非二选一：一个侧壁区域上打了孔，
       "侧壁"和"孔"都是真的，丢掉任何一个都会让下游误判。

    3. 复核：engineering_status 只在"语义稳定 + 角色已知"两个条件
       同时成立时才给 CANDIDATE，否则一律 REVIEW_REQUIRED。
       这里的保守是刻意的：工程特征无法靠几何单独判死，
       给一个不确定的 CANDIDATE 等于让甲方把它当结论采信。

    边界：regions 为空时返回空列表；候选与区域无交集时
    该区域仍会输出，只是角色退化为 UNRESOLVED_REGION_CANDIDATE，
    不会从结果里消失——漏掉一个区域比多标一个不确定角色更严重。
    """

    # 基础角色映射：把上游区域方向角色翻译成"区域级工程角色候选"。
    base_role_candidates = {
        "top_plane_region": "TOP_HORIZONTAL_REGION_CANDIDATE",
        "bottom_plane_region": "BOTTOM_HORIZONTAL_REGION_CANDIDATE",
        "horizontal_other_region": "HORIZONTAL_OTHER_REGION_CANDIDATE",
        "side_wall_region": "SIDE_WALL_REGION_CANDIDATE",
        "inclined_region": "INCLINED_REGION_CANDIDATE",
    }
    # 候选种类到区域角色的映射：区域只要命中了某类候选，就补上对应角色线索。
    candidate_role_by_kind = {
        "STEP_CANDIDATE": "STEP_BOUNDARY_REGION_CANDIDATE",
        "CYLINDER_OR_HOLE_CANDIDATE": "CYLINDER_OR_HOLE_REGION_CANDIDATE",
        "R_CORNER_CANDIDATE": "R_CORNER_REGION_CANDIDATE",
        "COAXIAL_HOLE_ARRAY_CANDIDATE": "COAXIAL_ARRAY_MEMBER_REGION_CANDIDATE",
    }
    face_by_id = {face.face_id: face for face in faces}
    source_region_by_id = {
        region.region_id: region
        for region in regions
    }
    output: list[StepEngineeringRegion] = []
    for source_region in regions:
        # 一个区域可能与多个候选共享 Face，用 Face 交集把候选关联到区域。
        face_ids = set(source_region.face_ids)
        matched_candidates = [
            candidate
            for candidate in candidates
            if face_ids.intersection(candidate.face_ids)
        ]
        role_candidates: list[str] = []
        base_role = base_role_candidates.get(source_region.role)
        if base_role is not None:
            role_candidates.append(base_role)
        if source_region.semantic_status == "EXTERNAL_CANDIDATE":
            role_candidates.append("EXTERNAL_SURFACE_REGION_CANDIDATE")
        elif source_region.semantic_status == "INTERNAL_CAVITY_CANDIDATE":
            role_candidates.append("INTERNAL_CAVITY_REGION_CANDIDATE")
        elif source_region.semantic_status in {"REVIEW_REQUIRED", "UNRESOLVED"}:
            role_candidates.append("REGION_SEMANTICS_REVIEW_REQUIRED")

        candidate_kinds = sorted({candidate.kind for candidate in matched_candidates})
        for kind in candidate_kinds:
            role_candidate = candidate_role_by_kind.get(kind)
            if role_candidate is not None:
                role_candidates.append(role_candidate)
        role_candidates = list(dict.fromkeys(role_candidates))
        if not role_candidates:
            role_candidates = ["UNRESOLVED_REGION_CANDIDATE"]

        # 只有语义稳定（内部/外部候选）且角色已知时才算 CANDIDATE，否则进人工复核。
        engineering_status = (
            "CANDIDATE"
            if source_region.semantic_status
            in {"EXTERNAL_CANDIDATE", "INTERNAL_CAVITY_CANDIDATE"}
            and base_role is not None
            else "REVIEW_REQUIRED"
        )
        candidate_kind_counts: dict[str, int] = {}
        candidate_ids: list[int] = []
        for candidate in matched_candidates:
            candidate_ids.append(candidate.candidate_id)
            candidate_kind_counts[candidate.kind] = (
                candidate_kind_counts.get(candidate.kind, 0) + 1
            )
        # 汇总区域尺寸、曲面参数和圆弧边，供类别规则与复核队列复用。
        region_measurements = _region_measurements(
            source_region,
            [
                face_by_id[face_id]
                for face_id in source_region.face_ids
                if face_id in face_by_id
            ],
        )
        (
            category_candidates,
            category_confidence,
            category_evidence,
            category_review_required,
            category_warnings,
        ) = _region_category_evidence(
            source_region,
            matched_candidates,
            [
                face_by_id[face_id]
                for face_id in source_region.face_ids
                if face_id in face_by_id
            ],
            region_measurements,
        )
        # 邻接证据：记录邻区角色、语义状态与角色过渡候选，用于分型与复核判断。
        neighbor_evidence = _region_neighbor_evidence(
            source_region,
            source_region_by_id,
            matched_candidates,
        )
        # 分型证据依赖邻接证据：分型面发生在"本区域与邻居的交界"，
        # 没有邻居就没有交界，所以必须先算邻居再算分型。
        (
            parting_role_candidates,
            parting_confidence,
            parting_review_required,
            parting_evidence,
            parting_warnings,
        ) = _region_parting_evidence(
            source_region,
            matched_candidates,
            neighbor_evidence,
        )
        # 复核队列同时给出原因码与优先级：
        # 原因码回答"为什么要复核"，优先级回答"先复核哪个"。
        # 只有原因码的话，几十个待复核区域之间没有处理顺序。
        review_reason_codes, review_priority = _region_review_queue(
            source_region,
            matched_candidates,
            category_candidates,
            category_confidence,
            category_review_required,
            region_measurements,
            neighbor_evidence,
            parting_review_required=parting_review_required,
            boundary_topology_status=source_region.boundary_topology_status,
            boundary_wire_selection_status=source_region.boundary_wire_selection_status,
            shape_is_valid=shape_is_valid,
        )
        # 能力边界声明无条件放在 warnings 第一条：
        # 它说明这份输出是"证据关联"而不是"最终型腔/型芯结论"。
        # 无条件加入是因为这条边界对每个区域都成立。
        warnings = [
            "我只把区域角色、实体采样和工程候选关联为证据，不把它们写成最终型腔或型芯。"
        ]
        # 语义未定与多类候选是两条独立的复核提示：
        # 前者是"不敢下结论"，后者是"结论互相矛盾"，
        # 分开写才能让复核者知道该补数据还是该做裁决。
        if source_region.semantic_status in {"UNRESOLVED", "REVIEW_REQUIRED"}:
            warnings.append(
                "我发现该区域缺少稳定的实体内外语义，区域工程角色必须人工复核。"
            )
        if len(candidate_kinds) > 1:
            warnings.append(
                "我发现该区域同时关联多类工程候选，不能按单一特征自动下结论。"
            )
        if neighbor_evidence["missing_neighbor_region_ids"]:
            warnings.append(
                "我发现该区域引用了不存在的邻接区域编号，区域边界关系需要复核。"
            )
        if neighbor_evidence["role_transition_candidates"]:
            warnings.append(
                "我保留了该区域与邻接区域的角色过渡证据，不把过渡类型解释成最终加工工艺。"
            )
        warnings.extend(category_warnings)
        warnings.extend(parting_warnings)
        if source_region.boundary_topology_status == "OPEN_BOUNDARY_CANDIDATE":
            warnings.append(
                "我发现该区域边界尚未形成闭合环，不能直接作为最终分型轮廓。"
            )
        elif source_region.boundary_topology_status == "UNRESOLVED_ENDPOINTS":
            warnings.append(
                "我发现该区域缺少完整 Vertex 端点证据，边界 Wire 需要人工复核。"
            )
        if category_review_required:
            engineering_status = "REVIEW_REQUIRED"
        output.append(
            StepEngineeringRegion(
                # 下面这批字段是从区域分类结果原样透传的"身份与量测"信息：
                # 我刻意不在这里重算或改写，保证工程报告与上游分类报告
                # 可以逐字段对账，出现分歧时能立刻定位是哪一层改了值。
                region_id=source_region.region_id,
                source_role=source_region.role,
                face_ids=list(source_region.face_ids),
                neighbor_region_ids=list(source_region.neighbor_region_ids),
                area_mm2=float(source_region.area_mm2),
                boundary_edge_count=source_region.boundary_edge_count,
                boundary_edge_ids=list(source_region.boundary_edge_ids),
                mean_center_xyz_mm=source_region.mean_center_xyz_mm,
                mean_normal_xyz=source_region.mean_normal_xyz,
                bbox_min_xyz_mm=region_measurements["bbox_min_xyz_mm"],
                bbox_max_xyz_mm=region_measurements["bbox_max_xyz_mm"],
                semantic_status=source_region.semantic_status,
                engineering_status=engineering_status,
                role_candidates=role_candidates,
                candidate_ids=sorted(candidate_ids),
                candidate_kinds=candidate_kinds,
                category_candidates=category_candidates,
                category_confidence=category_confidence,
                category_evidence=category_evidence,
                category_review_required=category_review_required,
                evidence={
                    # evidence 里放的是"可解释性"内容：计数、量测、邻接与边界状态，
                    # 供人工复核时回答"为什么是这个角色"，不属于判定字段本身。
                    "surface_type_counts": dict(source_region.surface_type_counts),
                    # 证据里同时保留"量测值"与"来源区域的信息"：
                    # 量测回答"多大"，来源信息回答"由哪些面构成"，
                    # 缺少后者时复核者无法回到模型上核对。
                    "semantic_evidence": dict(source_region.semantic_evidence),
                    "candidate_count": len(matched_candidates),
                    "candidate_kind_counts": dict(sorted(candidate_kind_counts.items())),
                    "measurements": region_measurements,
                    "boundary_edge_ids": list(source_region.boundary_edge_ids),
                    "boundary_perimeter_length_mm": (
                        source_region.boundary_perimeter_length_mm
                    ),
                    # 两个周长并列：整条边界的与主边界 Wire 的。
                    # 边界有多条 Wire 时二者不等，差值本身就是"边界不单一"的证据。
                    "primary_boundary_perimeter_length_mm": (
                        source_region.primary_boundary_perimeter_length_mm
                    ),
                    "primary_boundary_projected_area_mm2": (
                        source_region.primary_boundary_projected_area_mm2
                    ),
                    "primary_boundary_projection_status": (
                        source_region.primary_boundary_projection_status
                    ),
                    "primary_boundary_loop_role": (
                        source_region.primary_boundary_loop_role
                    ),
                    "primary_boundary_loop_relation_status": (
                        source_region.primary_boundary_loop_relation_status
                    ),
                    "neighbor_evidence": neighbor_evidence,
                    # 方法标识随证据落盘：区域关联是靠共享边做的弱关联，
                    # 换个实现后历史报告的结论口径就不同了，靠这个字段可区分。
                    "method": "shared_edge_region_role_and_candidate_join",
                    # interpretation 直接写进证据而不是只写在报告总述里：
                    # 甲方常常只看单一区域的证据块，
                    # 能力边界必须跟着每一块走，否则会被断章取义。
                    "interpretation": (
                        "区域级工程证据需要结合完整实体拓扑、装配层级、"
                        "坐标系和甲方标注继续复核。"
                    ),
                },
                # 警告去重：同一条警告可能在多个判据分支里各加一次，
                # 重复出现会让复核者以为有多个不同的问题。
                warnings=list(dict.fromkeys(warnings)),
                review_reason_codes=review_reason_codes,
                review_priority=review_priority,
                # 几何签名独立于面号：面号会随区域划分顺序变，
                # 跨批次追踪"同一个区域"只能靠签名。
                region_geometry_signature=source_region.region_geometry_signature,
                parting_role_candidates=parting_role_candidates,
                parting_confidence=parting_confidence,
                parting_review_required=parting_review_required,
                parting_evidence=parting_evidence,
                # 边界相关的字段成组排列：连通分量 -> 环数 -> 拓扑状态，
                # 顺序就是复核者的判断顺序，
                # 从"有几块边界"到"边界连不连得上"。
                boundary_edge_components=list(
                    source_region.boundary_edge_components
                ),
                boundary_loop_count=source_region.boundary_loop_count,
                closed_boundary_loop_count=source_region.closed_boundary_loop_count,
                boundary_topology_status=source_region.boundary_topology_status,
                # 证据用 dict() 复制而不是直接引用：
                # 直接引用会让外部改动源头对象时证据跟着变，
                # 这份记录就不再代表当时的判定依据了。
                boundary_topology_evidence=dict(
                    source_region.boundary_topology_evidence
                ),
                primary_boundary_edge_ids=list(
                    source_region.primary_boundary_edge_ids
                ),
                boundary_wire_candidate_count=source_region.boundary_wire_candidate_count,
                boundary_wire_selection_status=source_region.boundary_wire_selection_status,
                boundary_wire_selection_evidence=dict(
                    source_region.boundary_wire_selection_evidence
                ),
                # 周长与投影面积三个量相邻排列：
                # 它们回答"这条边界围出的东西有多大"，
                # 放在一起才能一次看全。
                boundary_perimeter_length_mm=(
                    source_region.boundary_perimeter_length_mm
                ),
                primary_boundary_perimeter_length_mm=(
                    source_region.primary_boundary_perimeter_length_mm
                ),
                primary_boundary_projected_area_mm2=(
                    source_region.primary_boundary_projected_area_mm2
                ),
                # 投影状态与投影证据紧挨着：
                # 状态给出结论，证据给出依据，
                # 两者必须同时可查才能复核这个状态。
                primary_boundary_projection_status=(
                    source_region.primary_boundary_projection_status
                ),
                primary_boundary_projection_evidence=dict(
                    source_region.primary_boundary_projection_evidence
                ),
                # 环角色与环关系放在最后：
                # 它们是"这条边界在整块区域里处于什么地位"，
                # 要理解了前面的尺度信息之后才有意义。
                # 所有 dict/list 字段都复制一份而不是直接引用：
                # 直接引用会让外部改动源头对象时这里的证据跟着变，
                # 这份记录就不再代表当时的判定依据。
                primary_boundary_loop_role=(
                    source_region.primary_boundary_loop_role
                ),
                primary_boundary_loop_relation_status=(
                    source_region.primary_boundary_loop_relation_status
                ),
                primary_boundary_loop_relation_evidence=dict(
                    source_region.primary_boundary_loop_relation_evidence
                ),
            )
        )
    return output


def _region_parting_evidence(
    source_region: StepFeatureRegion,
    candidates: Sequence[StepEngineeringCandidate],
    neighbor_evidence: dict[str, object],
) -> tuple[list[str], float, bool, dict[str, object], list[str]]:
    """我按区域角色和邻接关系生成分型候选证据，不直接伪造最终分型线。

    函数名里的"候选"二字是重点：分型线在工程上由开模方向、拔模角
    和工艺规则共同决定，这里能给出的只是"这个区域**像不像**分型面"
    的证据，不能给出"分型线就在这里"。所以我返回的是
    候选列表 + 置信度 + 证据字典，而不是一条线。

    为什么把邻接关系纳入分型判断：分型面的一个典型特征是
    "一侧是水平面、另一侧是侧壁"的角色过渡。只看区域自身角色
    无法区分"顶面中央的一块水平面"和"顶面与侧壁交界处的一块水平面"，
    而后者才可能是分型面。所以邻区角色是必要输入。

    边界：邻区信息缺失（neighbor_evidence 里没有对应键）时
    退化为只按自身角色判断，并把 review_required 置为 True，
    不会因为缺数据而直接放弃给出候选。
    """

    horizontal_roles = {
        "top_plane_region",
        "bottom_plane_region",
        "horizontal_other_region",
    }
    # 型面角色指的是侧壁与斜面：分型线往往沿着这类面的边界走。
    profile_roles = {"side_wall_region", "inclined_region"}
    neighbor_roles = sorted(
        {
            str(role)
            for role in neighbor_evidence.get("neighbor_source_roles", [])
        }
    )
    # 邻区里属于水平角色的那些：一块型面紧邻着水平面，
    # 说明它可能就是分型的立面部分。
    horizontal_neighbor_roles = sorted(
        role for role in neighbor_roles if role in horizontal_roles
    )
    candidate_kinds = sorted({candidate.kind for candidate in candidates})
    # 六个判据各自追加一个候选，彼此不互斥：
    # 一块区域可以同时是"分型面"又是"闭合边界环"，
    # 只保留最强的那个会丢掉其他有效线索。
    role_candidates: list[str] = []
    warnings: list[str] = []
    if source_region.role in horizontal_roles:
        role_candidates.append("PARTING_SURFACE_REGION_CANDIDATE")
    if source_region.role in profile_roles:
        role_candidates.append("PARTING_PROFILE_REGION_CANDIDATE")
    if source_region.boundary_edge_ids:
        role_candidates.append("PARTING_BOUNDARY_EDGE_REGION_CANDIDATE")
    # 型面紧邻水平面：这个组合比单独满足任一条件都更指向分型。
    if horizontal_neighbor_roles and source_region.role in profile_roles:
        role_candidates.append("PARTING_PROFILE_EDGE_CANDIDATE")
    if source_region.closed_boundary_loop_count > 0:
        role_candidates.append("PARTING_CLOSED_BOUNDARY_LOOP_CANDIDATE")
    if source_region.primary_boundary_edge_ids:
        role_candidates.append("PARTING_PRIMARY_BOUNDARY_WIRE_CANDIDATE")
    # 两个"需要复核"的候选：台阶缺口与 R 角都是
    # 分型面上必须人工确认的工艺细节。
    if "STEP_CANDIDATE" in candidate_kinds:
        role_candidates.append("PARTING_STEP_BRIDGE_REVIEW_CANDIDATE")
    if "R_CORNER_CANDIDATE" in candidate_kinds:
        role_candidates.append("PARTING_R_CORNER_REVIEW_CANDIDATE")
    # 去重但保留出现顺序：靠前的判据更强，
    # 顺序本身携带信息，用 set 会打乱它。
    role_candidates = list(dict.fromkeys(role_candidates))
    # 一个判据都没命中时给"未定"而不是空列表：
    # 空列表会被读成"这块区域与分型无关"，
    # 而实际是"看不出来"。
    if not role_candidates:
        role_candidates = ["UNRESOLVED_PARTING_REGION_CANDIDATE"]

    # 分型置信度是若干弱证据的加权和：区域角色、边界边、水平邻区、台阶/R 角支撑、
    # 闭合环与最大边界 Wire 各自给固定权重，最后截断到 1.0。权重是保守的经验值，
    # 只用于排序人工复核优先级，不作为自动判定分型线的阈值。
    score_components = {
        # 侧壁/倾斜这类"轮廓角色"给最高权重，因为分型线在几何上就落在
        # 这些面上；水平面给较低权重——它们通常是型腔底面而非分型位置。
        "role": (
            0.35
            if source_region.role in profile_roles
            else 0.2 if source_region.role in horizontal_roles else 0.0
        ),
        "boundary_edge": 0.25 if source_region.boundary_edge_ids else 0.0,
        "horizontal_neighbor": 0.3 if horizontal_neighbor_roles else 0.0,
        # 台阶与 R 角只是"支撑性"证据，权重刻意压得很低：
        # 它们常出现在非分型位置，给高权重会把普通倒角误判成分型线索。
        "step_support": 0.05 if "STEP_CANDIDATE" in candidate_kinds else 0.0,
        "r_corner_support": (
            0.05 if "R_CORNER_CANDIDATE" in candidate_kinds else 0.0
        ),
        "closed_boundary_loop": (
            0.15 if source_region.closed_boundary_loop_count > 0 else 0.0
        ),
        "primary_boundary_wire": (
            0.1
            if source_region.boundary_wire_selection_status
            in {"SINGLE_CLOSED_CANDIDATE", "MULTIPLE_CLOSED_CANDIDATES"}
            else 0.0
        ),
    }
    confidence = round(min(1.0, sum(score_components.values())), 6)
    # 下面每一条都是"必须人工看一眼"的硬条件，与置信度高低无关：
    # 置信度再高，只要缺边界边或边界未闭合，结论就没有几何依据，
    # 所以 review_required 是独立的布尔门，不参与上面的加权求和。
    review_required = False
    # 没有可追踪的边界边就无法做分型关联：
    # 分型轮廓的本质就是一条边界，没有边界等于没有可分析的输入。
    if not source_region.boundary_edge_ids:
        review_required = True
        warnings.append("我发现该区域没有可追踪的边界 Edge，分型关联只能人工复核。")
    # 侧壁/倾斜区域必须邻接水平区域才可能是分型轮廓：
    # 分型面发生在"竖直面与水平面交界"处，
    # 孤立的侧壁没有这个交界，也就没有分型可言。
    if source_region.role in profile_roles and not horizontal_neighbor_roles:
        review_required = True
        warnings.append(
            "我发现侧壁或倾斜区域没有明确邻接水平区域，不能自动作为分型轮廓候选。"
        )
    # 同时命中多类候选说明判据在这块几何上矛盾，必须人工裁决。
    if len(candidate_kinds) > 1:
        review_required = True
        warnings.append("我发现该区域同时存在多类工程候选，分型角色需要人工确认。")
    # 边界必须已确认闭合：只有闭合环才可能围出分型轮廓。
    if source_region.boundary_topology_status != "CLOSED_LOOP_CANDIDATE":
        review_required = True
        warnings.append(
            "我发现该区域没有得到完整闭合边界环证据，分型轮廓只能进入人工复核。"
        )
    # 主 Wire 选不出来（多个候选、周长打平、开闭混合）时，
    # "哪条是外轮廓"未定，分型角色也就不能自动给。
    if source_region.boundary_wire_selection_status in {
        "MULTIPLE_CLOSED_CANDIDATES",
        "AMBIGUOUS_MAXIMUM_TIE",
        "OPEN_AND_CLOSED_CANDIDATES",
        "UNRESOLVED_WITH_CLOSED_CANDIDATES",
    }:
        review_required = True
        warnings.append(
            "我发现该区域存在多个同等最大或混合开放/闭合 Wire 候选，"
            "最大边界轮廓需要人工确认。"
        )
    # 置信度为 0 说明没有任何正面证据支撑，直接要求复核。
    if confidence <= 0.0:
        # 置信度为 0 强制复核：它说明所有规则都没给出支持，
        # 此时"需不需要复核"不该再由其他条件决定。
        review_required = True
    evidence: dict[str, object] = {
        "status": "REVIEW_REQUIRED" if review_required else "CANDIDATE",
        # 方法名写进证据：这条结论来自"区域角色 + 边界 + 邻域 + 分型规则"，
        # 不写清来源的话，复核者无法判断该去找哪套规则核对。
        "method": "region_role_boundary_neighbor_parting_rules",
        "source_role": source_region.role,
        # 全部邻居角色与水平方向邻居角色分开记：
        # 分型面只发生在水平方向的交界上，
        # 把全部邻居混进来会让"水平方向有几个"看不出来。
        "neighbor_source_roles": neighbor_roles,
        "horizontal_neighbor_roles": horizontal_neighbor_roles,
        "candidate_kinds": candidate_kinds,
        # 边界字段整段复制进来：这份证据要能独立回答
        # "边界本身可不可信"，不复制就得回头去查区域对象。
        "boundary_edge_ids": list(source_region.boundary_edge_ids),
        "boundary_topology_status": source_region.boundary_topology_status,
        "boundary_loop_count": source_region.boundary_loop_count,
        "closed_boundary_loop_count": source_region.closed_boundary_loop_count,
        "boundary_edge_components": list(
            source_region.boundary_edge_components
        ),
        "primary_boundary_edge_ids": list(source_region.primary_boundary_edge_ids),
        "boundary_wire_candidate_count": source_region.boundary_wire_candidate_count,
        "boundary_wire_selection_status": source_region.boundary_wire_selection_status,
        "boundary_wire_selection_evidence": dict(
            source_region.boundary_wire_selection_evidence
        ),
        "score_components": score_components,
        # 能力边界声明：这些字段只是"追踪入口"，
        # 不含 B-Rep 截交、Wire 闭合、实体穿透这些真正的验收动作。
        "interpretation": (
            "我把区域分型字段作为最大外轮廓、台阶桥接和分型面候选的追踪入口；"
            "它不能替代 B-Rep 截交、Wire 闭合、实体穿透和甲方标注验收。"
        ),
    }
    return (
        role_candidates,
        confidence,
        review_required,
        evidence,
        # 警告去重：同一条理由可能在多条规则里各触发一次，
        # 重复出现会让复核者以为有多个不同的问题。
        list(dict.fromkeys(warnings)),
    )


def _region_review_queue(
    source_region: StepFeatureRegion,
    candidates: Sequence[StepEngineeringCandidate],
    category_candidates: Sequence[str],
    category_confidence: float,
    category_review_required: bool,
    measurements: dict[str, object],
    neighbor_evidence: dict[str, object],
    *,
    parting_review_required: bool = False,
    boundary_topology_status: str = "UNRESOLVED_ENDPOINTS",
    boundary_wire_selection_status: str = "NO_BOUNDARY",
    shape_is_valid: bool | None,
) -> tuple[list[str], str]:
    """我把区域人工复核原因压缩成稳定代码和优先级。

    原理说明：甲方真正要用的是"先复核哪些区域"，所以我不只给一堆警告文本，
    而是把每类问题压成一个稳定英文代码，再按严重程度排出优先级。
    这样做有两个实际收益：
      1. 代码是常量，下游可以按代码统计各类问题有多少个、按代码过滤队列，
         不用去解析中文警告文本；
      2. 原因代码只增不改，甲方历史报告的口径不会因为我改文案就漂移。

    一个区域可以同时命中多个代码（例如 Shape 无效且边界开放），
    所以我返回的是列表而不是单个值，优先级取其中最严重的一档。
    """

    # 下面每条 if 对应一类复核原因代码。代码保持稳定英文常量，方便下游按代码统计
    # 与过滤；原因只增不改，避免甲方历史报告口径漂移。
    reasons: list[str] = []
    # 复核原因逐条累加而不是命中一条就返回：
    # 复核者一次看到所有理由，才能判断"这是单一问题还是复合问题"，
    # 只给第一条会让他改完一个再跑一次，反复多轮。
    if shape_is_valid is False:
        reasons.append("SHAPE_INVALID")
    # 空区域与语义未定是不同的问题：空区域是"没数据"，
    # 语义未定是"有数据但不敢下结论"，处理方式完全不同。
    if not source_region.face_ids:
        reasons.append("EMPTY_REGION")
    if source_region.semantic_status in {"UNRESOLVED", "REVIEW_REQUIRED"}:
        reasons.append("SEMANTICS_UNRESOLVED")
    semantic_evidence = source_region.semantic_evidence
    # resource_guard 单独标出：这类"未定"是资源上限导致的主动跳过，
    # 提高上限就能解决，与真正的几何歧义要区别对待。
    if semantic_evidence.get("method") == "resource_guard":
        reasons.append("RESOURCE_GUARD")
    if source_region.boundary_edge_count > 0:
        reasons.append("BOUNDARY_EDGES_PRESENT")
    # 边界拓扑两档分开：确定开放 vs 端点未解析。
    # 前者是结论，后者是"数据不足以下结论"，混为一谈会让复核者白找一遍。
    if boundary_topology_status == "OPEN_BOUNDARY_CANDIDATE":
        reasons.append("BOUNDARY_LOOP_OPEN")
    elif boundary_topology_status == "UNRESOLVED_ENDPOINTS":
        reasons.append("BOUNDARY_TOPOLOGY_UNRESOLVED")
    # 这四种都说明"主环选不出来"：多个候选、周长打平、
    # 开闭混合、闭合候选但仍有游离边。
    if boundary_wire_selection_status in {
        "MULTIPLE_CLOSED_CANDIDATES",
        "AMBIGUOUS_MAXIMUM_TIE",
        "OPEN_AND_CLOSED_CANDIDATES",
        "UNRESOLVED_WITH_CLOSED_CANDIDATES",
    }:
        reasons.append("BOUNDARY_WIRE_AMBIGUOUS")
    projection_evidence = source_region.primary_boundary_projection_evidence
    # 冲突判据看两个来源：状态字符串或证据里的标志位。
    # 只看状态会在证据先于状态更新时漏报，两者取或是稳妥做法。
    if (
        boundary_wire_selection_status == "PROJECTED_AREA_CONFLICT"
        or projection_evidence.get("projection_conflict") is True
    ):
        reasons.append("BOUNDARY_PROJECTION_CONFLICT")
    loop_relation_evidence = source_region.primary_boundary_loop_relation_evidence
    if (
        boundary_wire_selection_status == "LOOP_RELATION_CONFLICT"
        or loop_relation_evidence.get("loop_relation_conflict") is True
    ):
        reasons.append("BOUNDARY_LOOP_RELATION_CONFLICT")
    if neighbor_evidence.get("missing_neighbor_region_ids"):
        # 邻居区域缺失说明区域划分有洞：有些面既不在本区域、
        # 也不在任何邻居里，区域覆盖不完整。
        reasons.append("MISSING_NEIGHBOR_REGION")
    candidate_kinds = {candidate.kind for candidate in candidates}
    # 一个区域同时命中多种候选类型（既是平面又是柱面）
    # 说明候选判据在这块几何上互相矛盾，必须人工裁决。
    if len(candidate_kinds) > 1:
        reasons.append("MULTI_CANDIDATE_KINDS")
    if not candidates:
        # 没有候选与"类别缺失"是两条不同的原因：
        # 前者是几何上没找到任何工程特征，
        # 后者是找到了区域但定不出类别，补救方式不同。
        reasons.append("NO_ENGINEERING_CANDIDATE")
    if len(source_region.surface_type_counts) > 1:
        reasons.append("MIXED_SURFACE_TYPES")
    # 类别相关的三条原因只在"类别需要复核"时才追加：
    # 类别已经确认的情况下再报"类别缺失/置信低"会自相矛盾。
    if category_review_required:
        if not category_candidates:
            reasons.append("CATEGORY_MISSING")
        # 0.6 是置信度下限：低于它的类别结论
        # 与"猜一个"没有区别，不该被直接采信。
        if category_confidence < 0.6:
            reasons.append("CATEGORY_LOW_CONFIDENCE")
        # 候选数不等于 1 都算歧义：0 个已在上面单独报成
        # CATEGORY_MISSING，这里覆盖"多个候选不分胜负"。
        if len(category_candidates) != 1:
            reasons.append("CATEGORY_AMBIGUOUS")
    if parting_review_required:
        reasons.append("PARTING_REGION_AMBIGUOUS")
    # 原因码去重但保持原顺序：同一条原因可能被多个判据追加，
    # 顺序保持不变才能让"最先命中的原因"排在最前。
    reasons = list(dict.fromkeys(reasons))
    if not reasons:
        # 没有任何原因就给空列表与 "NONE"：
        # 给一个"无原因的默认原因"会让下游
        # 无法区分"不用复核"和"没跑复核判据"。
        return [], "NONE"
    # 高优先级三档：形状无效、区域为空、邻接区域缺失。
    # 这三类的共同点是"数据本身不成立"，
    # 在它们之上做任何其他判定都没有意义，所以必须最先处理。
    high_priority = {
        "SHAPE_INVALID",
        "EMPTY_REGION",
        "MISSING_NEIGHBOR_REGION",
    }
    # 中优先级是"结论可能不准"这一类：数据成立但判据不足，
    # 需要人工看一眼才能确认，不必立刻阻断后续流程。
    medium_priority = {
        "SEMANTICS_UNRESOLVED",
        "RESOURCE_GUARD",
        "BOUNDARY_EDGES_PRESENT",
        "MULTI_CANDIDATE_KINDS",
        "MIXED_SURFACE_TYPES",
        "NO_ENGINEERING_CANDIDATE",
        "PARTING_REGION_AMBIGUOUS",
        "BOUNDARY_LOOP_OPEN",
        "BOUNDARY_TOPOLOGY_UNRESOLVED",
        "BOUNDARY_WIRE_AMBIGUOUS",
        "BOUNDARY_PROJECTION_CONFLICT",
        "BOUNDARY_LOOP_RELATION_CONFLICT",
    }
    # 取交集而不是遍历：一条区域可能同时命中多个原因，
    # 只要有一个属于该档就整条升到那一档，
    # 交集判断正好表达"存在即升档"。
    if high_priority.intersection(reasons):
        priority = "HIGH"
    elif medium_priority.intersection(reasons):
        priority = "MEDIUM"
    else:
        priority = "LOW"
    return reasons, priority


def _region_neighbor_evidence(
    source_region: StepFeatureRegion,
    region_by_id: dict[int, StepFeatureRegion],
    candidates: Sequence[StepEngineeringCandidate],
) -> dict[str, object]:
    """我把区域邻接和角色过渡压缩成可追踪的边界证据。"""

    role_by_neighbor: dict[str, list[int]] = {}
    semantic_by_neighbor: dict[str, list[int]] = {}
    missing_neighbor_ids: list[int] = []
    role_transitions: set[str] = set()
    for neighbor_id in sorted(set(source_region.neighbor_region_ids)):
        neighbor = region_by_id.get(neighbor_id)
        # 邻居编号存在但找不到区域时单独记下来：
        # 这是"引用了不存在的区域"，属于数据不一致，
        # 必须与"邻居角色是什么"分开报，否则会被静默吞掉。
        if neighbor is None:
            missing_neighbor_ids.append(int(neighbor_id))
            continue
        # 按角色与语义各建一份索引：角色回答"邻居是什么形状特征"，
        # 语义回答"它的内外判定可不可信"。两者维度不同，
        # 合成一份会丢失其中一维。
        role_by_neighbor.setdefault(neighbor.role, []).append(int(neighbor_id))
        semantic_by_neighbor.setdefault(
            neighbor.semantic_status,
            [],
        ).append(int(neighbor_id))
        # 角色跃迁用集合：同一对角色可能出现在多个邻居上，
        # 集合自动去重，"本区域与邻居之间存在哪几种跃迁"才是关心的量。
        role_transitions.add(
            _region_role_transition(source_region.role, neighbor.role)
        )
    # 边界特征候选只取这四类：它们才是"分型/清根/孔"这类
    # 工程含义明确的候选，其余类型在这个判据里没有角色。
    boundary_feature_candidates = sorted(
        {
            candidate.kind
            for candidate in candidates
            if candidate.kind
            in {
                "STEP_CANDIDATE",
                "R_CORNER_CANDIDATE",
                "CYLINDER_OR_HOLE_CANDIDATE",
                "COAXIAL_HOLE_ARRAY_CANDIDATE",
            }
        }
    )
    return {
        # 原始邻居数与去重后的编号列表都给：
        # 两者不等说明邻接表里有重复编号，那本身就是一处数据瑕疵。
        "neighbor_count": len(source_region.neighbor_region_ids),
        "neighbor_region_ids": sorted(set(source_region.neighbor_region_ids)),
        # 角色与语义两个索引都排序后输出：
        # 保证同一份输入得到字节一致的输出，便于跨批次 diff。
        "neighbor_source_roles": {
            role: sorted(ids)
            for role, ids in sorted(role_by_neighbor.items())
        },
        "neighbor_semantic_statuses": {
            status: sorted(ids)
            for status, ids in sorted(semantic_by_neighbor.items())
        },
        "role_transition_candidates": sorted(role_transitions),
        "boundary_feature_candidates": boundary_feature_candidates,
        "missing_neighbor_region_ids": missing_neighbor_ids,
        "method": "region_role_and_shared_edge_neighbor_summary",
        "interpretation": (
            "邻接和过渡只表示当前区域拓扑关系，"
            "不能替代最终分型边界、加工顺序或装配关系。"
        ),
    }


def _region_role_transition(left_role: str, right_role: str) -> str:
    """我用稳定名称记录两个区域角色之间的过渡候选。"""

    pair = frozenset((left_role, right_role))
    known = {
        frozenset(("top_plane_region", "side_wall_region")): "TOP_TO_SIDE_TRANSITION",
        frozenset(("bottom_plane_region", "side_wall_region")): "BOTTOM_TO_SIDE_TRANSITION",
        frozenset(("horizontal_other_region", "side_wall_region")): "HORIZONTAL_TO_SIDE_TRANSITION",
        frozenset(("inclined_region", "side_wall_region")): "INCLINED_TO_SIDE_TRANSITION",
    }
    if left_role == right_role:
        return "SAME_ROLE_BOUNDARY"
    return known.get(pair, "OTHER_ROLE_TRANSITION")


def _region_category_evidence(
    source_region: StepFeatureRegion,
    candidates: Sequence[StepEngineeringCandidate],
    faces: Sequence[StepFaceFeature],
    measurements: dict[str, object],
) -> tuple[list[str], float, dict[str, object], bool, list[str]]:
    """我用有限几何证据给区域生成部件类别候选，不输出最终类别。

    设计原则（这条是我方与甲方任务书对齐的关键）：我**只给候选，不给结论**。
    原因是区域级几何证据只能区分"像板件""像回转体"这类粗粒度形态，
    而任务书的 8 个部件大类（模架件 vs 辅助结构件）要靠装配位置和主数据
    才能分开。硬凑一个结论等于把不确定说成确定，所以我：
      - 每条候选都带 scores、signals 和来源，甲方能逐条复核是哪几个几何量
        把它推上去的；
      - 跨大类或证据不足时置 review_required=True，保留空白交人工确认。

    信号设计：我用平面占比、圆柱占比、长宽比、圆弧边数和同轴阵列候选
    这五个量，它们互不冗余且都能从 B-Rep 稳定算出。长宽比用包围盒
    最长边/最短边，且只统计大于 1e-9 的尺寸，避免薄板件被一个
    数值噪声维度算出荒谬的比值。
    """

    surface_counts = dict(source_region.surface_type_counts)
    face_count = len(faces)
    # 平面数与圆柱面数用子串匹配而不是等值比较：
    # 曲面类型的实际取值可能是复合写法，
    # 等值比较会漏掉它们而让计数偏小。
    planar_count = sum(
        "Plane" in face.surface_type for face in faces
    )
    cylindrical_count = sum(
        "Cylinder" in face.surface_type for face in faces
    )
    # 曲面数按总数减平面数推：除平面与圆柱之外
    # 还有锥面、球面等，它们一律归入"曲面"。
    curved_count = face_count - planar_count
    # 圆边数取不到就按 0：这里的默认值必须是 0 而不是 None，
    # 否则后面的比值计算要先判空。
    circular_edge_count = int(measurements.get("circular_edge_count", 0) or 0)
    bbox_size = _finite_vector(measurements.get("bbox_size_xyz_mm"))
    # 只保留为正的尺寸：某一维尺寸为 0 说明这块区域是
    # 退化的一片（例如平面区域在法向上尺寸为 0），
    # 让它参与长宽比会把比值算成无穷大。
    positive_sizes = (
        [float(value) for value in bbox_size if float(value) > 1e-9]
        if bbox_size is not None
        else []
    )
    # 长宽比 = 最大尺寸 / 最小尺寸：
    # 它区分"细长条"与"近似方块"，
    # 是判这块区域像板件还是像柱件的关键量。
    # 分母夹到 1e-9 防止除零，无有效尺寸时给 0.0。
    max_size = max(positive_sizes, default=0.0)
    min_size = min(positive_sizes, default=0.0)
    aspect_ratio = max_size / max(min_size, 1e-9) if positive_sizes else 0.0
    # 两个占比：平面占多少、圆柱占多少。
    # 分母夹到 1 防止空区域除零。
    planar_ratio = planar_count / max(face_count, 1)
    cylindrical_ratio = cylindrical_count / max(face_count, 1)
    # 语义状态统一转大写：上游的取值大小写不统一，
    # 不统一会让这里的等值比较时好时坏。
    semantic_status = str(source_region.semantic_status).upper()
    # 三个"有没有"标志：分别问这块区域是否带孔阵列候选、
    # 是否带圆柱/孔候选、面上是否量得出轴线。
    # 它们是类别判定的开关，各自独立。
    has_array = any(
        candidate.kind == "COAXIAL_HOLE_ARRAY_CANDIDATE"
        for candidate in candidates
    )
    has_cylinder_candidate = any(
        candidate.kind == "CYLINDER_OR_HOLE_CANDIDATE"
        for candidate in candidates
    )
    # 轴向与原点必须同时量到才算"有轴"：
    # 只有方向没有原点定不出这条轴在空间里的位置。
    has_axis = any(
        face.surface_axis_direction_xyz is not None
        and face.surface_axis_origin_xyz_mm is not None
        for face in faces
    )
    scores: dict[str, float] = {}
    # signals 保留全部原始量化信号，便于甲方复核每条类别候选是由哪些几何量支撑的。
    signals: dict[str, object] = {
        "surface_type_counts": surface_counts,
        "face_count": face_count,
        "planar_ratio": round(planar_ratio, 6),
        "cylindrical_ratio": round(cylindrical_ratio, 6),
        "curved_face_count": curved_count,
        "circular_edge_count": circular_edge_count,
        "aspect_ratio": round(aspect_ratio, 6),
        "semantic_status": semantic_status,
        "has_coaxial_array_candidate": has_array,
        "has_cylinder_candidate": has_cylinder_candidate,
        "has_axis_evidence": has_axis,
    }

    # 类别规则一：平面占比高且长宽比大 -> 板状件候选。
    if planar_count > 0 and planar_ratio >= 0.75 and aspect_ratio >= 4.0:
        scores["PLATE_COMPONENT_CANDIDATE"] = min(
            0.9,
            0.55 + 0.2 * planar_ratio + 0.15 * min(aspect_ratio / 10.0, 1.0),
        )
    # 类别规则二：圆柱面带轴线且细长 -> 轴类件候选。
    if cylindrical_count > 0 and has_axis and aspect_ratio >= 2.0:
        scores["SHAFT_COMPONENT_CANDIDATE"] = min(
            0.9,
            0.5 + 0.2 * cylindrical_ratio + 0.2 * min(aspect_ratio / 8.0, 1.0),
        )
    # 类别规则三：圆柱面 + 多个圆弧边 + 内部腔体语义 -> 导套类件候选。
    if (
        cylindrical_count > 0
        and circular_edge_count >= 2
        and (semantic_status == "INTERNAL_CAVITY_CANDIDATE" or has_array)
    ):
        scores["GUIDE_BUSH_COMPONENT_CANDIDATE"] = min(
            0.85,
            0.45
            + 0.15 * cylindrical_ratio
            + 0.15 * min(circular_edge_count / 4.0, 1.0)
            + 0.15 * float(semantic_status == "INTERNAL_CAVITY_CANDIDATE")
            + 0.1 * float(has_array),
        )
    # 类别规则四：圆柱面 + 轴线 + 内部腔体 + 细长 -> 顶针类件候选。
    if (
        cylindrical_count > 0
        and has_axis
        and semantic_status == "INTERNAL_CAVITY_CANDIDATE"
        and aspect_ratio >= 2.0
    ):
        scores["EJECTOR_COMPONENT_CANDIDATE"] = min(
            0.8,
            0.45 + 0.2 * cylindrical_ratio + 0.15 * min(aspect_ratio / 8.0, 1.0),
        )
    # 类别规则五：存在曲面且语义为外部候选 -> 成型面类件候选。
    if (
        curved_count > 0
        and semantic_status == "EXTERNAL_CANDIDATE"
    ):
        scores["MOLDED_SURFACE_CANDIDATE"] = min(
            0.9,
            0.5
            + 0.2 * (curved_count / max(face_count, 1))
            + 0.15 * float(has_cylinder_candidate),
        )
    # 类别规则六：多面且邻接或曲面类型丰富 -> 结构件候选（兜底前的最后一档）。
    if (
        face_count >= 2
        and (curved_count > 0 or planar_count > 0)
        and (
            len(source_region.neighbor_region_ids) >= 2
            or len(surface_counts) >= 2
        )
    ):
        scores["STRUCTURAL_COMPONENT_CANDIDATE"] = min(
            0.8,
            0.4
            + 0.15 * min(len(source_region.neighbor_region_ids) / 4.0, 1.0)
            + 0.15 * min(len(surface_counts) / 3.0, 1.0),
        )

    # 没有任何专用规则命中时，保留低置信度的结构件兜底候选，而不是返回空类别。
    if not scores:
        # 一个专用类别都没命中时补一个兜底候选：
        # 返回空列表会让下游"没有类别"，
        # 而"像结构件但证据不足"与"完全没有线索"是两回事。
        scores["STRUCTURAL_COMPONENT_CANDIDATE"] = 0.25
        signals["fallback"] = "我没有得到足够的专用类别证据，保留结构件兜底候选。"
    category_candidates = [
        category
        for category, score in sorted(
            scores.items(),
            # 先按分数降序、同分按名称升序：
            # 名称作为次级键保证输出稳定，
            # 否则字典顺序变化会让候选顺序抖动，跨批次无法 diff。
            key=lambda item: (-item[1], item[0]),
        )
        # 0.35 是入选下限：低于它的类别只是"有微弱迹象"，
        # 列进来会让候选列表变长并稀释真正有把握的那个。
        if score >= 0.35
    ]
    category_confidence = round(max(scores.values(), default=0.0), 6)
    # 平面与圆柱同时出现说明是混合曲面区域：
    # 这种区域在几何上不属于任何单一类别，
    # 必须并列保留候选而不是挑一个。
    mixed_geometry = planar_count > 0 and cylindrical_count > 0
    # 四个复核条件任一成立即可：语义未定、混合曲面、
    # 置信不足、候选不唯一。它们各自独立，
    # 任一成立都说明这个类别结论不能直接用。
    category_review_required = (
        semantic_status in {"UNRESOLVED", "REVIEW_REQUIRED"}
        or mixed_geometry
        or category_confidence < 0.6
        or len(category_candidates) != 1
    )
    warnings: list[str] = []
    # 三条警告与上面三个复核条件一一对应：
    # 每条都说清"卡在哪"，而不是笼统地说"需要复核"。
    if mixed_geometry:
        warnings.append("我发现该区域同时包含平面和圆柱等混合曲面，类别只能并列保留候选。")
    if len(category_candidates) != 1:
        warnings.append("我发现该区域类别候选不唯一或证据不足，必须人工复核。")
    if semantic_status in {"UNRESOLVED", "REVIEW_REQUIRED"}:
        warnings.append("我没有得到稳定实体内外语义，类别候选不能作为最终部件分类。")
    return (
        category_candidates,
        category_confidence,
        {
            # 全部类别的分数都留：只留选中的那个，
            # 复核者就无法判断"第二名差多少"。
            "scores": {
                key: round(float(value), 6)
                for key, value in sorted(scores.items())
            },
            "signals": signals,
            "method": "region_surface_bbox_topology_semantic_rules",
            # 能力边界：类别是几何候选，不是零件名称/装配层级/工程标注。
            "interpretation": "类别字段是几何候选证据，不替代零件名称、装配层级和工程标注。",
        },
        category_review_required,
        warnings,
    )


def _region_measurements(
    source_region: StepFeatureRegion,
    faces: Sequence[StepFaceFeature],
) -> dict[str, object]:
    """我提取区域级尺寸和曲面参数，供后续部件规则继续使用。

    为什么每个 face 的 bbox 都要先过 _finite_vector 再参与聚合：
    真实 STEP 里个别面的包围盒可能是 None 或含 NaN
    （退化面、求值失败的面）。一个 NaN 进 np.min 会污染整行结果，
    让整个区域的 bbox 全变成 NaN，而这个区域里其余 99 个面
    其实都是好的。所以我逐个过滤坏值，而不是先聚合再检查。

    过滤后可能一个有效值都不剩，此时 bbox 返回 None 而不是
    (0,0,0)：零向量会被下游当成"这个区域尺寸为零"的真实测量值，
    None 才能正确表达"测不出来"。
    """

    lower_bounds = [
        _finite_vector(face.bbox_min_xyz_mm)
        for face in faces
        # 逐个过滤掉非有限值：某个面的包围盒可能是 nan，
        # 放进数组再求 min/max 会把整个区域的包围盒污染成 nan。
        if _finite_vector(face.bbox_min_xyz_mm) is not None
    ]
    upper_bounds = [
        _finite_vector(face.bbox_max_xyz_mm)
        for face in faces
        if _finite_vector(face.bbox_max_xyz_mm) is not None
    ]
    # 区域的包围盒取所有面包围盒的外包：
    # 所有面的下界取逐轴最小值、上界取逐轴最大值。
    # 一个有效面都没有时留 None，而不是给一个零向量——
    # 零向量会被下游当成"区域退化成原点上的一个点"。
    bbox_min = (
        np.min(np.asarray(lower_bounds), axis=0)
        if lower_bounds
        else None
    )
    bbox_max = (
        np.max(np.asarray(upper_bounds), axis=0)
        if upper_bounds
        else None
    )
    bbox_size = (
        bbox_max - bbox_min
        if bbox_min is not None and bbox_max is not None
        else None
    )
    cylinder_radii = sorted(
        round(float(face.surface_radius_mm), 6)
        for face in faces
        # 三道过滤：半径存在、有限、且为正。
        # 半径 0 或非有限在退化柱面上会出现，
        # 混进来会让"半径分布"统计失真。
        if face.surface_radius_mm is not None
        and np.isfinite(float(face.surface_radius_mm))
        and float(face.surface_radius_mm) > 0.0
    )
    # 圆形边去重后排序：同一条边可能被相邻的两个面各记一次，
    # 不去重会让"圆边数"翻倍。排序保证输出稳定可比对。
    circular_edge_ids = sorted(
        {
            int(edge_id)
            for face in faces
            for edge_id in face.circular_edge_ids
            # 无效面号过滤掉：面特征里可能带 -1 之类的占位值，
            # 不过滤会让"圆边数"虚高。
            if _valid_face_id(edge_id)
        }
    )
    # 几何签名去重排序：同一区域里的面可能是同一形状的多个实例，
    # 去重后的签名数才反映"这块区域由几种不同的面构成"。
    geometry_signatures = sorted(
        {
            face.geometry_signature
            for face in faces
            if face.geometry_signature
        }
    )
    return {
        "mean_center_xyz_mm": source_region.mean_center_xyz_mm,
        "mean_normal_xyz": source_region.mean_normal_xyz,
        # 包围盒三件套各自判空：量不出来就留 None，
        # 给零向量会被读成"这块区域退化到了原点"。
        "bbox_min_xyz_mm": (
            bbox_min.tolist() if bbox_min is not None else None
        ),
        "bbox_max_xyz_mm": (
            bbox_max.tolist() if bbox_max is not None else None
        ),
        "bbox_size_xyz_mm": (
            bbox_size.tolist() if bbox_size is not None else None
        ),
        "surface_area_mm2": round(float(source_region.area_mm2), 6),
        # 曲面类型计数整份保留：它是判类别的第一手依据，
        # 只给"主要类型"会丢掉混合曲面这个关键信息。
        "surface_type_counts": dict(source_region.surface_type_counts),
        # 圆柱面数用子串匹配而不是等值比较：
        # 曲面类型的实际取值可能是 "Cylinder" 之外的复合写法，
        # 等值比较会漏掉它们。
        "cylindrical_face_count": sum(
            "Cylinder" in face.surface_type for face in faces
        ),
        # 半径采样逐个保留：孔径是否一致要看这一组样本，
        # 只给均值会掩盖"两个明显不同的孔径"。
        "cylinder_radius_samples_mm": cylinder_radii,
        # 圆边计数与编号成对：编号用于定位，计数用于统计。
        "circular_edge_count": len(circular_edge_ids),
        "circular_edge_ids": circular_edge_ids,
        "face_geometry_signatures": geometry_signatures,
        # 方法名固定：这段聚合来自"面包围盒 + 曲面类型 + 圆边"这三个来源。
        "method": "face_bbox_surface_type_and_circular_edge_aggregation",
    }


def _valid_face(face: StepFaceFeature) -> bool:
    """我判断 Face 是否带足够有限几何字段，可作为后续规则的输入。"""
    values = (face.area_mm2, face.center_xyz_mm, face.normal_xyz)
    if values[0] is None or not np.isfinite(float(values[0])) or float(values[0]) < 0.0:
        return False
    return (
        _finite_vector(face.center_xyz_mm) is not None
        and _finite_vector(face.normal_xyz) is not None
        and float(np.linalg.norm(np.asarray(face.normal_xyz, dtype=float))) > 1e-12
    )


def _valid_face_id(value: object) -> bool:
    """我判断 Face/Edge 编号是否为大于 0 的整数（布尔值不算）。

    我为什么把 bool 明确排除：Python 里 ``True`` 是 ``int`` 的子类且
    ``True > 0``，不排除就会把布尔字段当成编号 1。而编号 1 几乎总是
    已存在，于是后写的覆盖先写的，一个面凭空消失且不报错。
    """
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value > 0


def _finite_vector(value: Sequence[float] | None) -> np.ndarray | None:
    """我把输入转成长度 3 的有限浮点向量；不合法时返回 None 而不是抛异常。

    我为什么返回 None 而不是抛错：这个函数服务的是"证据收集"，
    一个面的法向坏了不该让整份工程特征报告失败——那个面只是少一条
    证据，其余几十个面的台阶、孔、圆角信息仍然有效。抛错会为了
    一个坏面丢掉整件的结果。

    我为什么严格判 ``shape != (3,)``：嵌套一层但长度凑巧是 3 的数据
    说明上游字段结构变了，宽松接受会把结构错误藏到叉积时才炸，
    那时错误信息指向叉积而不是真正出错的读取环节。
    """
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    return array


def _unit_vector(value: Sequence[float], name: str) -> np.ndarray:
    """我返回单位向量；输入缺失、非有限或近似零向量时抛出带字段名的错误。

    我为什么在这里**抛错**而与 _finite_vector 的返回 None 相反：
    归一化是后续夹角、同轴判定的前提，拿不到方向就意味着这条证据
    根本无法成立；而调用方需要知道"是 left normal 还是 right normal
    坏了"。所以我把字段名带进错误信息，让排查能直接定位到字段。

    1e-12 这个门槛：真实法向的模长是 1 量级，退化到 1e-12 以下说明
    上游已经给出了零向量。这里若放行，除以极小数会把数值放大到
    1e12 量级，后面所有夹角判定全部失真。
    """
    vector = _finite_vector(value)
    if vector is None or float(np.linalg.norm(vector)) <= 1e-12:
        raise ValueError(f"我要求 {name} 是有限且非零的三维向量。")
    return vector / float(np.linalg.norm(vector))


def _combined_bbox(
    faces: Iterable[StepFaceFeature],
) -> tuple[list[float] | None, list[float] | None]:
    """我合并所有有效 Face 的包围盒，返回整体 min/max；没有有效输入时返回两个 None。"""
    mins = []
    maxs = []
    for face in faces:
        if face.bbox_min_xyz_mm is not None and face.bbox_max_xyz_mm is not None:
            lower = _finite_vector(face.bbox_min_xyz_mm)
            upper = _finite_vector(face.bbox_max_xyz_mm)
            if lower is not None and upper is not None and np.all(lower <= upper):
                mins.append(lower)
                maxs.append(upper)
    if not mins:
        return None, None
    return (
        np.min(np.asarray(mins), axis=0).tolist(),
        np.max(np.asarray(maxs), axis=0).tolist(),
    )


def _shared_edge_ids(left: StepFaceFeature, right: StepFaceFeature) -> list[int]:
    """我返回两个 Face 共用的 Edge 编号，升序排列并去重。"""
    return sorted(set(left.edge_ids).intersection(right.edge_ids))


def _edge_radii(face: StepFaceFeature) -> list[float]:
    """我挑出 Face 上属于圆弧边、且半径有限的边半径样本，供孔壁判据使用。"""
    return [
        float(radius)
        for edge_id, radius in zip(face.edge_ids, face.edge_radii_mm)
        if edge_id in face.circular_edge_ids
        and radius is not None
        and np.isfinite(float(radius))
        and float(radius) > 0.0
    ]


def _hole_topology_evidence(
    face: StepFaceFeature,
    radii: list[float],
    tolerance_ratio: float,
) -> dict[str, object]:
    """我用圆柱边界和半径一致性生成孔壁的拓扑代理证据。

    原理说明：一个真正的通孔在 B-Rep 上表现为"一个圆柱面，其边界至少含
    两条圆弧边，且两条圆弧半径一致（同一个钻头一次加工出来的）"。
    我据此构造三条互相独立的证据，三者缺一不可才给"闭合孔壁代理"：
      1. 至少两条圆弧边——单条圆弧可能是半圆槽而不是完整孔壁；
      2. 半径离散度在容差内——半径不一致说明是台阶孔或锥孔；
      3. Edge 特征数组长度对齐——数组错位会让"第 i 条边的半径"读到别的
         边上去，据此得出的任何结论都是幽灵证据，必须先拦掉。

    诚实边界：这是**拓扑代理**，不是实体布尔求差。我只能说"这个面看起来
    像孔壁"，不能说"这里确实存在一个孔"，所以只给候选和复核提示。

    score 从 0.4 起步按证据逐项累加并封顶 1.0：即使一条证据都没有，
    也不给 0 分——圆柱面本身已经是"可能是孔"的弱证据，保留排序区分度。
    """

    # 只保留确实出现在 Face 边集合里的圆弧边，防止数组错位带来幽灵边。
    circular_edge_ids = [
        int(edge_id)
        for edge_id in face.circular_edge_ids
        if edge_id in face.edge_ids
    ]
    unique_circular_edge_ids = sorted(set(circular_edge_ids))
    radius_consistent = False
    radius_spread_ratio = None
    if radii:
        median_radius = float(median(radii))
        if median_radius > 0.0:
            radius_spread_ratio = (
                (max(radii) - min(radii)) / median_radius
            )
            radius_consistent = radius_spread_ratio <= tolerance_ratio
    arrays_aligned = all(
        not values or len(values) == len(face.edge_ids)
        for values in (
            face.edge_curve_types,
            face.edge_lengths_mm,
            face.edge_radii_mm,
        )
    )
    # 闭合孔壁代理：至少两条圆弧边 + 半径一致 + 特征数组对齐，三者缺一不可。
    closed_boundary_proxy = (
        len(unique_circular_edge_ids) >= 2
        and bool(radii)
        and radius_consistent
        and arrays_aligned
    )
    # 基础分 0.4，按可用证据逐项加分；上限 1.0。
    score = 0.4
    if len(unique_circular_edge_ids) >= 2:
        score += 0.2
    if radius_consistent:
        score += 0.2
    if arrays_aligned:
        score += 0.1
    if closed_boundary_proxy:
        score += 0.1
    warnings: list[str] = []
    if len(unique_circular_edge_ids) < 2:
        warnings.append(
            "我没有得到至少两个可用圆弧边，只能保留圆柱面候选，不能确认闭合孔壁。"
        )
    if not radii:
        warnings.append("我没有得到可用圆弧半径，不能确认孔径一致性。")
    elif not radius_consistent:
        warnings.append(
            "我发现圆弧半径离散度超过当前容差，孔壁候选需要人工复核。"
        )
    if not arrays_aligned:
        warnings.append("我发现 Face 的 Edge 特征数组未完全对齐，拓扑证据需要复核。")
    semantic_status = str(
        face.semantic_evidence.get("status", "UNRESOLVED")
    ).upper()
    if semantic_status == "EXTERNAL_CANDIDATE":
        warnings.append(
            "我发现该圆柱面具备实体外部候选语义，因此不能把它解释成孔壁。"
        )
    elif semantic_status in {"UNRESOLVED", "REVIEW_REQUIRED"}:
        warnings.append(
            "我没有得到稳定的实体内外语义，孔壁解释仍只能作为候选。"
        )
    # 结合实体语义给出最终孔壁解释：内部腔体且闭合才叫孔壁，外圆柱一律排除。
    if semantic_status == "INTERNAL_CAVITY_CANDIDATE":
        hole_semantics = (
            "HOLE_WALL_CANDIDATE"
            if closed_boundary_proxy
            else "INTERNAL_CYLINDER_FACE_CANDIDATE"
        )
    elif semantic_status == "EXTERNAL_CANDIDATE":
        hole_semantics = "EXTERNAL_CYLINDER_FACE_CANDIDATE"
    elif closed_boundary_proxy:
        hole_semantics = "HOLE_WALL_CANDIDATE"
    else:
        hole_semantics = "CYLINDER_FACE_CANDIDATE"
    return {
        "circular_edge_ids": unique_circular_edge_ids,
        "closed_boundary_proxy": closed_boundary_proxy,
        "radius_consistent": radius_consistent,
        "semantic_status": semantic_status,
        "semantic_confidence": face.semantic_evidence.get("confidence"),
        "radius_spread_ratio": (
            round(float(radius_spread_ratio), 6)
            if radius_spread_ratio is not None
            else None
        ),
        "edge_feature_arrays_aligned": arrays_aligned,
        "hole_semantics": hole_semantics,
        "score": min(1.0, score),
        "warnings": warnings,
        "interpretation": (
            "圆柱面具备闭合边界代理证据且实体语义支持内部腔体，"
            "可作为孔壁候选；仍需轴向连续性和工程标注复核。"
            if hole_semantics == "HOLE_WALL_CANDIDATE"
            else "圆柱面保留为几何候选，但当前实体语义不能确认孔壁。"
        ),
    }


def _plane_role_summary(groups: Iterable[object]) -> dict[str, dict[str, object]]:
    """我把已有方向分组压缩成可供下游规则读取的平面角色摘要。"""

    summary: dict[str, dict[str, object]] = {}
    for group in groups:
        # 每个方向分组压成"面数 + 面编号 + 面积"三要素，供下游按角色读取。
        role = str(group.role)
        face_ids = sorted(
            int(face_id)
            for face_id in group.face_ids
            if _valid_face_id(face_id)
        )
        summary[role] = {
            "face_count": len(face_ids),
            "face_ids": face_ids,
            "area_mm2": round(float(group.area_mm2), 6),
        }
    return dict(sorted(summary.items()))


def _coaxial_hole_array_candidates(
    faces: Iterable[StepFaceFeature],
    *,
    start_candidate_id: int,
    axis_angle_tolerance_deg: float,
    axis_distance_tolerance_mm: float,
    radius_tolerance_ratio: float,
) -> list[StepEngineeringCandidate]:
    """我按轴线分桶提取同轴孔阵列候选，避免大模型全量两两比较。

    原理说明：孔阵列（同一规格的一圈螺丝孔、冷却水路）在工程上是
    "轴线共线或平行、半径一致的一组圆柱面"。朴素做法是对所有圆柱面
    两两比较，复杂度 O(n^2)；真实模具常有上千个孔，会直接卡死。

    所以我用**空间哈希分桶**降复杂度：把每条轴线量化成一个整数键，
    只在桶内做精确比较。桶键由两部分组成，缺一不可：
      1. 方向键：把归一化的方向向量按角度容差量化取整。我先做方向
         正负归一（_canonical_axis_direction），否则同一条轴线会因为
         朝向相反被拆进两个桶，阵列被拦腰截断；
      2. 位置键：取轴线上离原点最近的垂足点，按距离容差量化取整。
         用垂足而不是轴线原点是关键——沿轴平移不改变垂足，
         所以同一条轴线上的孔不会因为起点不同而落进不同桶。

    分桶只是加速手段，不是判据：桶内仍要用夹角/轴距/半径比三个容差
    逐个验证，量化带来的边界误差（同一条轴线可能被切到相邻桶）
    由桶内精确比较兜住，不会漏判。
    """

    # 量化步长取容差本身，把"近似同轴"的轴线归到同一个桶，桶内再精确比较。
    angular_step = max(sin(radians(axis_angle_tolerance_deg)), 1e-4)
    distance_step = max(float(axis_distance_tolerance_mm), 1e-6)
    buckets: dict[tuple[tuple[int, ...], tuple[int, ...]], list[StepFaceFeature]] = {}
    for face in faces:
        direction = _finite_vector(face.surface_axis_direction_xyz)
        origin = _finite_vector(face.surface_axis_origin_xyz_mm)
        radius = face.surface_radius_mm
        if (
            direction is None
            or origin is None
            or radius is None
            or not np.isfinite(float(radius))
            or float(radius) <= 0.0
        ):
            continue
        direction = direction / float(np.linalg.norm(direction))
        if float(np.linalg.norm(direction)) <= 1e-12:
            continue
        # 统一轴线正负，否则同一条轴线会被拆进两个方向桶。
        direction = _canonical_axis_direction(direction)
        # 用轴线上离原点最近的点（垂足）作为位置桶键，消除沿轴平移带来的差异。
        closest_origin = origin - direction * float(np.dot(origin, direction))
        key = (
            tuple(int(round(float(value) / angular_step)) for value in direction),
            tuple(int(round(float(value) / distance_step)) for value in closest_origin),
        )
        buckets.setdefault(key, []).append(face)

    candidates: list[StepEngineeringCandidate] = []
    next_id = start_candidate_id
    for bucket_faces in buckets.values():
        # 桶内少于两个圆柱面，谈不上阵列，直接跳过。
        if len(bucket_faces) < 2:
            continue
        # 以桶内第一个面为参考，逐个用三容差（夹角/轴距/半径比）验证成员。
        # 只跟参考面比而不是两两互比：两两互比是 O(m^2)，且与参考面
        # 都满足容差并不意味着成员之间也满足（容差不传递），
        # 反而会给出"两两都合规"的错觉。与参考面比是明确的语义：
        # 这一组面都以参考轴为准。
        representative = bucket_faces[0]
        accepted = [representative]
        pair_metrics: list[tuple[float, float, float]] = []
        for face in bucket_faces[1:]:
            # 与代表面逐对比较三个量：轴线夹角、轴线距离、半径比。
            # 三者必须同时满足才算同轴——只判夹角会收进
            # "平行但偏移很远"的孔，只判距离会收进朝向不同的孔。
            metrics = _axis_pair_metrics(representative, face)
            if metrics is None:
                # 量不出轴向的面跳过：它没有足够信息参与同轴判定，
                # 跳过比强行比更安全。
                continue
            angle_deg, axis_distance_mm, radius_ratio = metrics
            if (
                angle_deg <= axis_angle_tolerance_deg
                and axis_distance_mm <= axis_distance_tolerance_mm
                and radius_ratio <= radius_tolerance_ratio
            ):
                accepted.append(face)
                # 成对指标收集起来：它给出"这组孔同轴到什么程度"，
                # 只给一个布尔值无法说明判定的余量有多大。
                pair_metrics.append(metrics)
        # 少于两个就不成"阵列"：单个孔没有阵列可言，
        # 报成阵列会让下游去做毫无意义的分组统计。
        if len(accepted) < 2:
            continue
        face_ids = sorted({int(face.face_id) for face in accepted})
        # 圆形边去重：同一个孔的两条圆边可能被两个面各记一次，
        # 不去重会让"孔数"翻倍。
        edge_ids = sorted(
            {
                int(edge_id)
                for face in accepted
                for edge_id in face.circular_edge_ids
                if _valid_face_id(edge_id)
            }
        )
        direction = _unit_vector(
            representative.surface_axis_direction_xyz or (),
            "cylinder axis",
        )
        # 轴向原点取不到就放弃：没有原点就排不出孔的顺序，
        # 而"孔在轴上的分布"正是阵列的核心信息。
        origin = _finite_vector(representative.surface_axis_origin_xyz_mm)
        if origin is None:
            continue
        # 把每个孔的轴向原点投影到共同轴向上，得到一维位置序列。
        # 降到一维是为了判断"孔是否等距排列"——
        # 在三维空间里比较间距会因方向差异而无法比。
        projections = sorted(
            float(
                np.dot(
                    _finite_vector(face.surface_axis_origin_xyz_mm) - origin,
                    direction,
                )
            )
            for face in accepted
            if _finite_vector(face.surface_axis_origin_xyz_mm) is not None
        )
        spacing = [
            round(float(right - left), 6)
            for left, right in zip(projections, projections[1:])
        ]
        max_angle = max((item[0] for item in pair_metrics), default=0.0)
        max_distance = max((item[1] for item in pair_metrics), default=0.0)
        max_radius_ratio = max((item[2] for item in pair_metrics), default=0.0)
        # 评分封顶 0.95 而不是 1.0：这是几何候选，没有孔径标准、
        # 装配层级和实体内外证据，我保留 0.05 的余量表示"未确认"。
        # 成员数、角度余量、距离余量三项相加，越多越齐越可信。
        score = min(
            # 上限 0.95：几何证据再完美也不给满分，
            # 因为这类候选还需要孔径、内外语义、装配层级来确认。
            0.95,
            # 基础分 0.55：只要形成了一个合格的同轴组就给到的分，
            # 下面三项是在此之上按"有多齐"加分。
            0.55
            # 成员数加分，最多 0.2：孔越多越像阵列，
            # 10 个以上不再加分——再多也不改变"这是阵列"这个结论。
            + min(len(accepted) / 10.0, 0.2)
            # 角度余量加分，最多 0.15：实际夹角越接近 0 越齐。
            # 用 (1 - 实测/容差) 而不是直接用角度值，
            # 是为了让加分与容差尺度无关。
            + max(0.0, 0.15 * (1.0 - max_angle / axis_angle_tolerance_deg))
            # 距离余量加分，最多 0.1：轴线越重合越齐。
            # 权重比角度低是因为轴线的微小偏移在加工上比角度偏差更常见。
            + max(0.0, 0.1 * (1.0 - max_distance / axis_distance_tolerance_mm)),
        )
        candidates.append(
            StepEngineeringCandidate(
                candidate_id=next_id,
                kind="COAXIAL_HOLE_ARRAY_CANDIDATE",
                face_ids=face_ids,
                edge_ids=edge_ids,
                score=round(float(score), 6),
                evidence={
                    # 轴向与代表面原点：这组孔的共同轴线由它们确定，
                    # 复核者要靠它把孔画出来看。
                    "axis_direction_xyz": direction.tolist(),
                    "representative_axis_origin_xyz_mm": origin.tolist(),
                    "member_count": len(accepted),
                    # 三个"最大偏差"量：角度、轴线距离、半径比。
                    # 记最大值而不是均值——最差的那一个决定这组能不能用。
                    "max_axis_angle_deg": round(float(max_angle), 6),
                    "max_axis_distance_mm": round(float(max_distance), 6),
                    "max_radius_difference_ratio": round(float(max_radius_ratio), 6),
                    "axis_spacing_mm": spacing,
                    # 孔径是否一致存成布尔：下游判断走布尔，
                    # 不必各处重复写一遍这个比较。
                    "radius_consistent": max_radius_ratio <= radius_tolerance_ratio,
                    # 方法名固定：这组候选来自"轴向量化分桶"这一路分法。
                    "method": "quantized_axis_line_bucket",
                    # 能力边界：同轴只是几何关系，
                    # 不等于这些孔是标准件、也不等于加工方案已定。
                    "interpretation": (
                        "同轴圆柱/孔阵列候选，需要孔径、实体内外、"
                        "装配层级和工程标准件规则复核。"
                    ),
                },
                warnings=[
                    "我把多个近似共轴圆柱面记录为孔阵列候选，不自动判定标准件或最终孔加工方案。"
                ],
            )
        )
        # 候选编号在成功加入之后才递增：
        # 中途 continue 的那些组不会占掉编号，
        # 于是编号始终连续，不会出现空洞。
        next_id += 1
    return candidates


def _axis_pair_metrics(
    left: StepFaceFeature,
    right: StepFaceFeature,
) -> tuple[float, float, float] | None:
    """我计算两圆柱轴线的夹角、轴线间距和半径相对差；任一轴线证据缺失时返回 None。

    三个判据各自回答一个问题，缺一就无法判定"同轴"：
      - 夹角：两轴是否平行（同轴的前提）；
      - 轴线间距：两轴是否重合。我用叉积的模长
        |(origin2-origin1) x dir| 来算，这等价于点到直线的距离，
        直接算两原点距离会把"平行但错开"误判成同轴；
      - 半径相对差：两孔是否同规格。不同直径的孔即使同轴也不该
        并成同一个孔阵列。

    夹角取绝对点积：轴线是无向的，d 与 -d 是同一条轴，
    不取绝对值会把完全重合的轴算成 180 度夹角。

    任一输入缺失就返回 None 而不是填 0：填 0 会让"证据缺失"
    伪装成"完全匹配"，把一批没有轴线数据的面误聚成同轴阵列。
    """
    left_direction = _finite_vector(left.surface_axis_direction_xyz)
    right_direction = _finite_vector(right.surface_axis_direction_xyz)
    left_origin = _finite_vector(left.surface_axis_origin_xyz_mm)
    right_origin = _finite_vector(right.surface_axis_origin_xyz_mm)
    left_radius = left.surface_radius_mm
    right_radius = right.surface_radius_mm
    if (
        left_direction is None
        or right_direction is None
        or left_origin is None
        or right_origin is None
        or left_radius is None
        or right_radius is None
        or float(left_radius) <= 0.0
        or float(right_radius) <= 0.0
    ):
        return None
    left_direction = left_direction / float(np.linalg.norm(left_direction))
    right_direction = right_direction / float(np.linalg.norm(right_direction))
    angle_deg = degrees(
        acos(float(np.clip(abs(np.dot(left_direction, right_direction)), -1.0, 1.0)))
    )
    axis_distance_mm = float(
        np.linalg.norm(np.cross(right_origin - left_origin, left_direction))
    )
    radius_ratio = abs(float(left_radius) - float(right_radius)) / max(
        float(np.median([left_radius, right_radius])),
        1e-12,
    )
    return angle_deg, axis_distance_mm, radius_ratio


def _canonical_axis_direction(direction: np.ndarray) -> np.ndarray:
    """我统一轴线方向正负，避免同一条轴线被分成两个桶。

    原理说明：一条轴线在数学上是一个**无向**直线，d 和 -d 描述同一条线。
    但分桶用的是向量分量的量化值，不归一时 [0,0,1] 和 [0,0,-1] 会落进
    完全不同的桶，一组同轴孔会被拆成两半，阵列检测直接失效。

    我按"第一个显著非零分量必须为正"来定向：从前往后找第一个
    绝对值大于 1e-12 的分量，它为负就把整个向量取反。用第一个显著分量
    而不是固定用某一轴，是因为轴线可能与 X 轴垂直（该分量为 0），
    固定轴会在这种情况下失去判别力。
    """

    result = np.asarray(direction, dtype=np.float64).copy()
    for value in result:
        if abs(float(value)) > 1e-12:
            if value < 0.0:
                result *= -1.0
            break
    return result


def _shared_edge_radius(
    left: StepFaceFeature,
    right: StepFaceFeature,
    edge_id: int,
) -> float | None:
    """我取共享 Edge 在两侧 Face 上的半径中位数，作为圆角半径的代表值。

    为什么取中位数而不是直接取第一个：同一条边在两侧面上各有一次记录，
    两次采样可能略有差异（弦近似精度不同）。两个值取中位数是稳定的；
    只取一侧面会在那一侧数据缺失时直接丢掉这条证据。

    我逐个过滤掉 None、非有限和非正值：半径缺失在自由曲面上很常见，
    而负半径没有物理意义，让它进中位数会把结果整体带偏。
    """
    radii = []
    for face in (left, right):
        for current_edge_id, radius in zip(face.edge_ids, face.edge_radii_mm):
            if current_edge_id == edge_id and radius is not None:
                value = float(radius)
                if np.isfinite(value) and value > 0.0:
                    radii.append(value)
    return float(median(radii)) if radii else None


def _unsigned_normal_angle(
    left: Sequence[float] | None,
    right: Sequence[float] | None,
) -> float | None:
    """我返回两法向的无向夹角（0°~90°），忽略正负朝向。

    为什么取绝对值（无向）：法向的正负取决于面的朝向约定，同一个
    "两个面互相垂直"的几何关系，可以算成 90 度也可以算成 90 度取反的
    270 度。台阶、圆角这类判据关心的是**方向差多大**，不是朝向正负，
    所以我把夹角压到 0~90 度区间，让"差 90 度"和"差 270 度"得到同一个值。

    np.clip 不能省：点积在数值上可能略超出 [-1, 1]（例如 1.0000000002），
    直接 acos 会返回 nan，而 nan 会静默传播到后续所有比较，
    最终表现为"这个面判不出来"而不是报错。
    """
    left_vector = _unit_vector(left if left is not None else (), "left normal")
    right_vector = _unit_vector(right if right is not None else (), "right normal")
    dot = float(np.clip(abs(np.dot(left_vector, right_vector)), -1.0, 1.0))
    return degrees(acos(dot))


def _center_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """我返回两个三维中心点之间的欧氏距离。"""
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def _draft_evidence(
    face: StepFaceFeature,
    direction: np.ndarray,
    tolerance_deg: float,
) -> StepDraftEvidence:
    """我按 Face 法向与抽模方向的夹角，把 Face 分成平面/法向/中间三类拔模证据。

    原理说明：拔模分析的本质是"面法向与开模方向的夹角"。三类语义是：
      - 法向与开模方向**垂直**（夹角约 90 度）：面平行于开模方向，
        即典型的侧壁/立面，脱模时需要拔模斜度，记为
        DRAW_DIRECTION_PARALLEL_PLANE_CANDIDATE；
      - 法向与开模方向**平行**（夹角约 0 度）：面垂直于开模方向，
        即型腔底面或分型面，脱模时不产生侧向阻力，记为
        DRAW_DIRECTION_NORMAL_CANDIDATE；
      - 其余是过渡面或已带拔模斜度的面，记为 INTERMEDIATE_OR_DRAFT_CANDIDATE。

    我对点积取绝对值后再算夹角，因为法向的正反取决于面的朝向约定，
    而"平行/垂直"这个语义与朝向正负无关；不取绝对值会把同一几何的
    +90 度和 -90 度判成两类。

    score 用"与垂直状态的偏离程度"归一化到 0~1，只用于排序候选，
    不用于判定型腔、型芯或合模方向——那需要实体内外测试，我没有偷换结论。
    """
    normal = _unit_vector(face.normal_xyz or (), "face normal")
    angle = degrees(acos(float(np.clip(abs(np.dot(normal, direction)), -1.0, 1.0))))
    plane_angle = abs(90.0 - angle)
    if plane_angle <= tolerance_deg:
        classification = "DRAW_DIRECTION_PARALLEL_PLANE_CANDIDATE"
    elif angle <= tolerance_deg:
        classification = "DRAW_DIRECTION_NORMAL_CANDIDATE"
    else:
        classification = "INTERMEDIATE_OR_DRAFT_CANDIDATE"
    score = max(0.0, 1.0 - min(plane_angle, 90.0) / 90.0)
    return StepDraftEvidence(
        face_id=face.face_id,
        direction_xyz=direction.tolist(),
        normal_vs_direction_angle_deg=round(angle, 6),
        plane_vs_direction_angle_deg=round(plane_angle, 6),
        classification=classification,
        score=round(score, 6),
        warnings=[
            "我只记录拔模方向角度证据，不自动判定型腔、型芯或合模方向。"
        ],
    )
