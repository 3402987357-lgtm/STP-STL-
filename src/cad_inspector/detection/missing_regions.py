"""我根据几何距离把参考模型中的异常面聚类成缺失区域。

原理说明（任务书要求关键算法附原理，这里写完整流程与取舍理由）：

一、为什么用"双向表面检查"而不是直接做布尔差
    真实件和参考件都是 STL 网格，两者三角化参数往往不同。直接做布尔差会
    先被"同一张平面被切成不同的三角片"这种网格噪声淹没。所以我方不比较
    拓扑，而是比较"点到面的最近距离"：参考件上的采样点在候选件表面找不到
    足够近的对应物，就说明参考件在这里有一块面是候选件没有的，也就是缺失候选。

二、完整流程（detect_missing_regions）
    1. 在参考网格上按 samples_per_triangle 确定性采样，最多 max_query_points 个点；
       注意配额是"按面计数"：每个三角面分到相同数量的采样点，与面积无关，
       所以大面上单个采样点代表的面积更大；这不是按面积成正比的配额。
    2. 每个采样点求到候选网格的最近距离、最近点、最近三角面编号；
    3. 把采样点级距离归约到 Face 级：取该 Face 上所有采样点的最大距离；
    4. Face 距离 >= threshold_mm 记为 abnormal；
    5. 用三角面邻接把 abnormal 面做连通分量聚类，每个连通块是一个候选区域；
    6. 过滤面积或三角面数低于阈值的区域（记进 warnings，不静默丢弃）；
    7. 对候选件反过来再跑一遍 1~6，得到"多余材料"候选；
    8. 汇总全局指标（面积比、误差分位数、采样覆盖率、STEP Face 映射诊断）并定状态。

三、为什么归约取"最大距离"而不是"平均距离"
    缺失区域里通常只有一部分采样点真的落在孔洞里，其余落在周边正常面上。
    取平均会把信号摊薄甚至摊没；取最大值能保证"只要这个 Face 上有一处对不上"
    就报出来，宁可多报进复核，也不漏检——这正是任务书"漏检率 ≤ 1%"的方向。

四、有向证据只做辅助，不改分类
    _directional_sample_evidence 用"最近点位移在源面法向上的投影"和
    "源面法向与目标面法向的夹角"给每个采样点贴 aligned / boundary 标签。
    它回答的是"这块区域是整体偏移，还是拓扑真的变了"，但最终缺失/多余分类
    仍只由距离加连通性决定，避免有向证据把结论带偏。

五、为什么状态要留 review
    配准质量不是 good、采样没覆盖全部面、STEP Face 映射不完整，这三种情况
    都会让"面积和结论不完整"。我方不把它们写成 pass/fail，而是降级成 review，
    要求人工复核或提高采样上限后重跑。
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from typing import Any

import numpy as np

from ..geometry.mesh import TriangleMesh, build_triangle_adjacency
from ..models import InspectionConfig, RegionResult, RegistrationResult
from .distance import points_to_mesh


def detect_missing_regions(
    reference: TriangleMesh,
    candidate: TriangleMesh,
    config: InspectionConfig,
    registration: RegistrationResult,
) -> tuple[list[RegionResult], dict[str, object], list[str], np.ndarray]:
    """我执行参考到候选、候选到参考的双向表面检查。

    返回值是四元组：(区域列表, 全局指标, 警告列表, 参考面距离数组)。
    区域列表里 missing 和 extra 混在一起，靠 RegionResult.defect_type 区分，
    并且我在过滤小区域之后统一重编号，保证 region_id 连续且不重复。
    """

    warnings: list[str] = []
    # 第一步：在参考网格上做确定性采样。采样用固定算法而不是随机数，
    # 这样同一份输入每次跑出来的区域和面积都可复现。
    sample_points, sample_face_ids = reference.sample_points(
        samples_per_triangle=config.samples_per_triangle,
        max_points=config.max_query_points,
    )
    # 第二步：每个采样点求到候选网格的最近距离、最近点、最近三角面编号。
    sample_distances, sample_candidate_face_ids, sample_closest_points = points_to_mesh(
        sample_points,
        candidate,
        chunk_size=config.distance_chunk_size,
        backend=config.distance_backend,
        candidate_k=config.distance_candidate_k,
    )
    # 第三步：有向证据（辅助诊断）。它只标记 aligned/boundary，不参与分类。
    forward_direction = _directional_sample_evidence(
        source_mesh=reference,
        target_mesh=candidate,
        source_face_ids=sample_face_ids,
        target_face_ids=sample_candidate_face_ids,
        source_points=sample_points,
        target_points=sample_closest_points,
        distances=sample_distances,
        threshold_mm=config.threshold_mm,
        normal_tolerance_deg=config.direction_normal_tolerance_deg,
        enabled=config.directional_evidence_enabled,
    )
    # 第四步：把采样点级证据归约到 Face 级。forward_face_direction 是各 Face
    # 上有向证据的累加统计，face_distances 是各 Face 的最大最近距离。
    forward_face_direction = _reduce_directional_samples(
        sample_face_ids,
        forward_direction,
        face_count=len(reference.faces),
    )
    face_distances = _reduce_sample_distances(
        sample_face_ids,
        sample_distances,
        face_count=len(reference.faces),
    )
    # 第四步补：可选地把每个面的三个顶点也纳入面级最大值。
    # 默认关闭，所以这一段不改变任何既有结果；打开时它是"轮廓带"漏检的收口。
    if config.face_corner_check:
        corner_distances = _face_corner_distances(
            reference,
            candidate,
            config=config,
        )
        face_distances = np.maximum(face_distances, corner_distances)
    # 第五步：显式记录"哪些 Face 真的被采样到了"。未被采样的 Face 在
    # face_distances 里是 0，如果不加掩码就会伪装成"完全吻合"，这是必须
    # 单独区分的一种情况——它就是后面 coverage_incomplete 的依据。
    reference_sampled_face_mask = _sampled_face_mask(
        sample_face_ids,
        face_count=len(reference.faces),
    )
    reference_sampled_face_count = len(np.unique(sample_face_ids))
    reference_unsampled_face_count = max(
        len(reference.faces) - reference_sampled_face_count,
        0,
    )
    # 打开面级顶点采样后，"有没有检查过这个面"的答案变了：顶点距离对**每个**
    # 面都算过。若仍按"面内采样点"统计覆盖率，报告会说"有面没被采样"，
    # 而实际上它们已按顶点判过。我把掩码同步成"全部面已测"，让覆盖率与
    # 真实判据一致；这是开关打开时才生效的口径变化，默认路径不变。
    if config.face_corner_check and len(reference.faces):
        reference_sampled_face_mask = np.ones(len(reference.faces), dtype=bool)
        reference_sampled_face_count = len(reference.faces)
        reference_unsampled_face_count = 0
    # 第六步：超过阈值的 Face 记为 abnormal，再按三角面邻接聚成连通区域。
    abnormal_faces = face_distances >= config.threshold_mm
    adjacency = build_triangle_adjacency(reference.faces)
    raw_regions = _connected_components(abnormal_faces, adjacency)
    regions: list[RegionResult] = []
    ignored_missing_area = 0.0
    for region_id, face_ids in enumerate(raw_regions, start=1):
        region = _make_region(
            region_id=region_id,
            face_ids=face_ids,
            source_mesh=reference,
            face_distances=face_distances,
            candidate_face_ids=_candidate_triangle_ids_for_region(
                face_ids=face_ids,
                sample_face_ids=sample_face_ids,
                sample_candidate_face_ids=sample_candidate_face_ids,
            ),
            step_mapping_mesh=candidate,
            config=config,
            registration=registration,
            defect_type="missing",
            directional_face_stats=forward_face_direction,
        )
        # 第七步：过滤太小的区域。我把它累加到 ignored_missing_area 并写进
        # warnings，而不是直接扔掉——甲方需要知道"有多少面积被阈值滤掉了"。
        if (
            region.area_mm2 < config.min_region_area_mm2
            or region.triangle_count < config.min_triangle_count
        ):
            ignored_missing_area += region.area_mm2
            continue
        regions.append(region)
    if ignored_missing_area:
        warnings.append(
            f"我过滤了 {ignored_missing_area:.6f} mm2 的小面积缺失候选区域，"
            "请确认最小区域阈值。"
        )

    # 下面是反向检查（候选 -> 参考），用来找"多余材料"。
    # 我先用空数组把 extra_* 全部初始化成安全默认值，这样即使
    # bidirectional_check 关闭，后面的全局指标代码也不需要写分支。
    extra_face_distances = np.zeros(len(candidate.faces), dtype=np.float64)
    candidate_sampled_face_mask = np.zeros(len(candidate.faces), dtype=bool)
    extra_region_count = 0
    extra_area = 0.0
    ignored_extra_area = 0.0
    candidate_sampled_face_count = 0
    candidate_unsampled_face_count = 0
    # 关闭状态下我也调用一次 _directional_sample_evidence（enabled=False），
    # 目的只是拿到一份"结构完整但全为无效值"的空证据，让输出字段形状一致。
    reverse_direction = _directional_sample_evidence(
        source_mesh=candidate,
        target_mesh=reference,
        source_face_ids=np.empty((0,), dtype=np.int64),
        target_face_ids=np.empty((0,), dtype=np.int64),
        source_points=np.empty((0, 3), dtype=np.float64),
        target_points=np.empty((0, 3), dtype=np.float64),
        distances=np.empty((0,), dtype=np.float64),
        threshold_mm=config.threshold_mm,
        normal_tolerance_deg=config.direction_normal_tolerance_deg,
        enabled=False,
    )
    reverse_face_direction = _reduce_directional_samples(
        np.empty((0,), dtype=np.int64),
        reverse_direction,
        face_count=len(candidate.faces),
    )
    if config.bidirectional_check:
        # 反向检查与正向完全对称：候选采样 -> 求到参考的最近距离 ->
        # 归约到 Face -> 连通分量聚类 -> 过滤小区域。
        reverse_points, reverse_face_ids = candidate.sample_points(
            samples_per_triangle=config.samples_per_triangle,
            max_points=config.max_query_points,
        )
        # 反向采样：从候选模型采样、到参考模型求距离。
        # 它与正向采样（参考->候选）回答的是两个不同问题：
        #   正向找"缺失"（参考上有、候选上没有）；
        #   反向找"多出"（候选上有、参考上没有）。
        # 只做单向会把"多出"误判成"没差异"。
        reverse_distances, reverse_reference_face_ids, reverse_closest_points = points_to_mesh(
            reverse_points,
            reference,
            chunk_size=config.distance_chunk_size,
            backend=config.distance_backend,
            candidate_k=config.distance_candidate_k,
        )
        reverse_direction = _directional_sample_evidence(
            source_mesh=candidate,
            target_mesh=reference,
            source_face_ids=reverse_face_ids,
            target_face_ids=reverse_reference_face_ids,
            source_points=reverse_points,
            target_points=reverse_closest_points,
            distances=reverse_distances,
            threshold_mm=config.threshold_mm,
            normal_tolerance_deg=config.direction_normal_tolerance_deg,
            enabled=config.directional_evidence_enabled,
        )
        reverse_face_direction = _reduce_directional_samples(
            reverse_face_ids,
            reverse_direction,
            face_count=len(candidate.faces),
        )
        candidate_sampled_face_count = len(np.unique(reverse_face_ids))
        # 采样掩码记录"哪些面真的取到了点"：三角面大小差异大时，
        # 均匀采样可能整片小面一个点都没落上，
        # 这些面上的"无差异"结论其实是"没检查"。
        candidate_sampled_face_mask = _sampled_face_mask(
            reverse_face_ids,
            face_count=len(candidate.faces),
        )
        # max(..., 0) 兜底：采样点数受 max_query_points 截断时，
        # 唯一面数不可能超过总面数，但下标异常会让差值为负，
        # 负的"未采样数"会让下游覆盖率算错。
        candidate_unsampled_face_count = max(
            len(candidate.faces) - candidate_sampled_face_count,
            0,
        )
        # 与正方向同样的口径同步：顶点采样让"每个面都被测过"。
        if config.face_corner_check and len(candidate.faces):
            candidate_sampled_face_mask = np.ones(
                len(candidate.faces), dtype=bool
            )
            candidate_sampled_face_count = len(candidate.faces)
            candidate_unsampled_face_count = 0
        # 反向距离：每个候选面到参考模型的最近距离。
        # 这是"多出的料"方向——参考模型上找不到对应面的那些面。
        extra_face_distances = _reduce_sample_distances(
            reverse_face_ids,
            reverse_distances,
            face_count=len(candidate.faces),
        )
        # 反向方向同样补顶点：两个方向必须用**同一套判据**，
        # 否则"缺失"严格、"多出"宽松，两类结论就无法放在一起比较。
        if config.face_corner_check:
            extra_face_distances = np.maximum(
                extra_face_distances,
                _face_corner_distances(candidate, reference, config=config),
            )
        # 同一个 threshold_mm 用于两个方向：阈值语义必须一致，
        # 否则"缺失"与"多出"的判定尺度不同，两者无法放在一起比较。
        extra_abnormal = extra_face_distances >= config.threshold_mm
        extra_adjacency = build_triangle_adjacency(candidate.faces)
        # 连通分量即"多出的区域"：散落的单个异常面
        # 是噪声，连成一片的才是一块真实的多料。
        extra_raw_regions = _connected_components(
            extra_abnormal,
            extra_adjacency,
        )
        # 编号从已有区域数之后接着排：两个方向的区域
        # 放在同一份列表里，编号必须全局唯一且不重叠。
        first_extra_id = len(regions) + 1
        for offset, face_ids in enumerate(extra_raw_regions):
            region = _make_region(
                region_id=first_extra_id + offset,
                face_ids=face_ids,
                # 源网格是候选而不是参考：这块区域的位置与面积
                # 要按候选模型上的面来算。
                source_mesh=candidate,
                face_distances=extra_face_distances,
                candidate_face_ids=face_ids,
                step_mapping_mesh=candidate,
                config=config,
                registration=registration,
                defect_type="extra",
                # 方向统计也用反向的那一份：
                # "多出的料偏向哪一侧"与"缺失的料"是两套不同的采样。
                directional_face_stats=reverse_face_direction,
            )
            # 面积与三角形数两个门槛都要过：
            # 只看面积会把"一个巨大但只有一两个三角形的噪声片"放进来，
            # 只看三角形数会把"很多个极小三角形拼成的一片"放进来。
            if (
                region.area_mm2 < config.min_region_area_mm2
                or region.triangle_count < config.min_triangle_count
            ):
                # 被忽略的面积累加起来：它会在下面变成一条警告，
                # 否则"有一片多料被滤掉了"这件事会完全消失。
                ignored_extra_area += region.area_mm2
                continue
            regions.append(region)
            extra_region_count += 1
            extra_area += region.area_mm2
        if ignored_extra_area:
            warnings.append(
                f"我过滤了 {ignored_extra_area:.6f} mm2 的小面积多余材料候选区域，"
                "请确认最小区域阈值。"
            )
        # 只要反向检查发现了多余材料，我就提醒甲方去看彩色网格和原始 CAD。
        # 因为"多余"的常见成因是配准姿态误差，不一定是真的多料。
        if extra_region_count:
            warnings.append(
                "我在候选模型反向检查中发现多余材料候选区域，"
                "请结合彩色网格和原始 CAD 复核。"
            )
    else:
        warnings.append(
            "我按配置关闭了候选到参考的反向检查，当前结果只覆盖缺失候选。"
        )

    # 以下四条是把"结论可能不完整"的原因显式抬到 warnings 里。
    # 配准不是 good、单位未知，都会让面积和状态失去确定性。
    if registration.quality != "good" or registration.scale_review_required:
        warnings.extend(registration.warnings)
    if reference.metadata.get("unit") == "unknown":
        warnings.append("我无法从参考 STL 判断单位，请确认当前数值确实使用 mm。")
    if candidate.metadata.get("unit") == "unknown":
        warnings.append("我无法从待检测 STL 判断单位，请确认当前数值确实使用 mm。")
    # STEP Face 映射诊断：如果三角面到 STEP Face 的映射缺失或含非法编号，
    # 那"这块缺失对应 STEP 上哪个面"就不可信，只能当人工复核线索。
    reference_mapping = _step_face_mapping_diagnostics(reference)
    candidate_mapping = _step_face_mapping_diagnostics(candidate)
    mapping_review_required = False
    for label, mapping in (
        ("参考模型", reference_mapping),
        ("待检测模型", candidate_mapping),
    ):
        if mapping["present"] and mapping["review_required"]:
            mapping_review_required = True
            warnings.append(
                f"我发现{label}的 STEP Face 映射元数据不完整或含非法编号，"
                "相关 Face 回溯结果只能作为人工复核线索。"
            )
    # 采样覆盖率：只要有 Face 没被采到，面积统计就是"部分覆盖"的，
    # 这种结论不能当作完整通过。
    coverage_incomplete = (
        reference_unsampled_face_count > 0
        or candidate_unsampled_face_count > 0
    )
    if coverage_incomplete:
        warnings.append(
            "我发现当前采样没有覆盖全部三角面，区域面积和通过结论不完整；"
            "请提高 max_query_points 后再验收。"
        )
    global_metrics = _global_metrics(
        reference=reference,
        face_distances=face_distances,
        regions=regions,
        candidate=candidate,
        config=config,
        extra_face_distances=extra_face_distances,
        missing_region_count=sum(
            region.defect_type == "missing" for region in regions
        ),
        extra_region_count=extra_region_count,
        extra_area_mm2=extra_area,
        reference_sampled_face_count=reference_sampled_face_count,
        reference_unsampled_face_count=reference_unsampled_face_count,
        candidate_sampled_face_count=candidate_sampled_face_count,
        candidate_unsampled_face_count=candidate_unsampled_face_count,
        reference_sampled_face_mask=reference_sampled_face_mask,
        candidate_sampled_face_mask=candidate_sampled_face_mask,
        reference_step_face_mapping=reference_mapping,
        candidate_step_face_mapping=candidate_mapping,
    )
    global_metrics["directional_evidence"] = _directional_summary(
        forward_direction,
        reverse_direction,
        normal_tolerance_deg=config.direction_normal_tolerance_deg,
    )
    # 我把"被小区域阈值滤掉多少面积"做成结构化字段，而不只留一条警告文本
    # （2026-10-16）。理由：警告是给人看的，程序读不到，于是这件事会从
    # 评估口径里彻底消失。而它会**直接改变召回**——我在 训练集/8 上实测：
    # 一个真值 0.898 mm2 的缺失，检测器确实检出了（报出 0.3804 mm2），
    # 但因为不足 min_region_area_mm2=0.5 被丢掉，最终表现为"漏检"。
    # 有了这两个数，评估层才能把漏检归因到"检出后被阈值滤掉"，
    # 而不是笼统记成"检测器没检出"——两者要采取的行动完全不同。
    global_metrics["ignored_missing_area_mm2"] = float(ignored_missing_area)
    global_metrics["ignored_extra_area_mm2"] = float(ignored_extra_area)
    # 我在过滤小区域后重新编号，保证缺失和多余区域不会复用 region_id。
    for region_id, region in enumerate(regions, start=1):
        region.region_id = region_id
    missing_region_count = int(global_metrics["missing_region_count"])
    has_defect = missing_region_count > 0 or extra_region_count > 0
    # 状态判定有明确的优先级，顺序不能调换：
    # 1) 覆盖率不完整或 STEP 映射可疑 -> review（结论不可信，先别下判定）；
    # 2) 没有任何缺陷 -> pass；
    # 3) 配准坏掉或尺度需要复核 -> review（可能是配准问题被误报成缺陷）；
    # 4) 其余才算 fail（确实有缺陷，且基础条件都可信）。
    if coverage_incomplete or mapping_review_required:
        status = "review"
    elif not has_defect:
        status = "pass"
    elif registration.quality == "bad" or registration.scale_review_required:
        status = "review"
    else:
        status = "fail"
    global_metrics["status"] = status
    return regions, global_metrics, warnings, face_distances


def _reduce_sample_distances(
    sample_face_ids: np.ndarray,
    sample_distances: np.ndarray,
    face_count: int,
) -> np.ndarray:
    """我把采样点级最近距离归约到 Face 级，取每个 Face 的最大距离。

    我特意用 np.maximum.at 而不是普通索引赋值（reduced[ids] = distances），
    因为一个 Face 上可能有很多采样点，普通赋值会互相覆盖、只留下最后一个。
    非法编号（越界、负数）和非有限距离我直接丢弃，不让它们污染结果。
    """

    sample_face_ids = np.asarray(sample_face_ids, dtype=np.int64)
    sample_distances = np.asarray(sample_distances, dtype=np.float64)
    if len(sample_face_ids) != len(sample_distances):
        raise ValueError("我要求采样面编号和采样距离数量一致。")
    # 默认 0 表示"这个 Face 目前还没看到异常"，稍后由掩码区分"未采样"。
    reduced = np.zeros(face_count, dtype=np.float64)
    valid = (
        (sample_face_ids >= 0)
        & (sample_face_ids < face_count)
        & np.isfinite(sample_distances)
    )
    np.maximum.at(
        reduced,
        sample_face_ids[valid],
        sample_distances[valid],
    )
    return reduced


def _face_corner_distances(
    source_mesh: TriangleMesh,
    target_mesh: TriangleMesh,
    *,
    config: InspectionConfig,
) -> np.ndarray:
    """我量每个三角面**三个顶点**到目标网格的最近距离，并按面取最大值。

    我为什么补这一路（2026-10-25 实测）：
    面级判据原先只看采样点的最大值，而默认采样是 1 点/面且恒取重心。
    重心是三角形内部一点，它到缺口边界的距离既不是最小也不是最大；
    当一个面的重心距离 0.1mm、对面顶点距离 0.28mm 时，
    这个面按任务书口径（面上存在 ≥0.15mm 的误差）**属于缺失区域**，
    却会被判成"没有误差"。实测在 0.2mm 网格 / 0.15mm 阈值 / 删 24 面上，
    这类"轮廓带"面占到真值面的 7/15。

    顶点不是随意的第二个采样点：三角形上到某一方向最远的点必在顶点或边上，
    顶点是面内离边界最远的**极值候选位置**，只取重心等于放弃了它。

    代价与边界：每面多 3 次距离查询（19228 面的件约 +57684 个查询点），
    所以它由 `config.face_corner_check` 控制、**默认关闭**；
    打开与否属于验收口径决策，我只负责让开关生效并把代价量出来。
    """

    triangles = source_mesh.triangle_vertices
    face_count = len(source_mesh.faces)
    if face_count == 0:
        return np.zeros(0, dtype=np.float64)
    points = triangles.reshape(-1, 3)
    distances, _, _ = points_to_mesh(
        points,
        target_mesh,
        chunk_size=config.distance_chunk_size,
        backend=config.distance_backend,
        candidate_k=config.distance_candidate_k,
    )
    # 每个面 3 个顶点，编号按面重复；再走同一套面级最大归约。
    corner_face_ids = np.repeat(
        np.arange(face_count, dtype=np.int64),
        3,
    )
    return _reduce_sample_distances(corner_face_ids, distances, face_count)


def _sampled_face_mask(
    sample_face_ids: np.ndarray,
    *,
    face_count: int,
) -> np.ndarray:
    """我显式记录实际被采样的 Face，避免未采样面零值污染统计。

    我为什么必须单独维护这个掩码：距离数组对未采样的 Face 是 0，
    而"距离 0"的语义是"完全贴合、无缺失"。把两者混在一起，
    未采样的面会被当成最完美的面统计进去，区域的 p95 与均值
    会被系统性拉低——实测表现为"看起来缺失很小"，实际是采样不全。
    """

    mask = np.zeros(face_count, dtype=bool)
    ids = np.asarray(sample_face_ids, dtype=np.int64)
    valid = (ids >= 0) & (ids < face_count)
    mask[ids[valid]] = True
    return mask


def _directional_sample_evidence(
    *,
    source_mesh: TriangleMesh,
    target_mesh: TriangleMesh,
    source_face_ids: np.ndarray,
    target_face_ids: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    distances: np.ndarray,
    threshold_mm: float,
    normal_tolerance_deg: float,
    enabled: bool,
) -> dict[str, object]:
    """我把最近点位移转成有向证据，但不替换缺失/多余分类。"""

    source_face_ids = np.asarray(source_face_ids, dtype=np.int64)
    target_face_ids = np.asarray(target_face_ids, dtype=np.int64)
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    size = len(source_face_ids)
    # 空结果一律用 nan / False 填充且长度对齐：
    # 长度不对齐会让下游与面号数组 zip 时错位，
    # 用 nan 而非 0 是因为 0 是一个合法的有符号偏移值。
    empty = {
        "enabled": bool(enabled),
        "signed_offset_mm": np.full(size, np.nan, dtype=np.float64),
        "normal_alignment": np.full(size, np.nan, dtype=np.float64),
        "valid": np.zeros(size, dtype=bool),
        "aligned": np.zeros(size, dtype=bool),
        "boundary": np.zeros(size, dtype=bool),
    }
    if not enabled:
        return empty
    # 五个数组长度必须完全一致，否则是调用方传错了参数：
    # 此时宁可返回空证据，也不能让广播机制悄悄把长度对不上
    # 的数组补成莫名其妙的结果。
    if (
        len(target_face_ids) != size
        or len(source_points) != size
        or len(target_points) != size
        or len(distances) != size
        or size == 0
    ):
        return empty
    # 有效样本的四道过滤：
    #   面号在范围内 —— 距离计算用 -1 表示无匹配，必须排除；
    #   坐标与距离有限 —— 排除 nan/inf；
    #   距离 >= 阈值 —— 只有真的偏了才有方向可言，
    #     完全贴合处的法向没有参考价值。
    valid = (
        (source_face_ids >= 0)
        & (source_face_ids < len(source_mesh.faces))
        & (target_face_ids >= 0)
        & (target_face_ids < len(target_mesh.faces))
        & np.isfinite(source_points).all(axis=1)
        & np.isfinite(target_points).all(axis=1)
        & np.isfinite(distances)
        & (distances >= threshold_mm)
    )
    # 一个有效样本都没有就返回空：继续算会得到空数组的 max/min 并抛异常。
    if not np.any(valid):
        # 一个有效样本都没有就返回全 False 的空壳：
        # 保持键齐全，下游按固定键取值才不会 KeyError。
        return empty
    source_normals = source_mesh.triangle_normals[source_face_ids[valid]]
    target_normals = target_mesh.triangle_normals[target_face_ids[valid]]
    # 两个法向模长的乘积：它等于两个单位法向点积的分母。
    # 任一个是零向量（退化三角形）时这个乘积为 0，
    # 下面据此把这类样本剔掉。
    normal_lengths = np.linalg.norm(source_normals, axis=1) * np.linalg.norm(
        target_normals,
        axis=1,
    )
    valid_indices = np.flatnonzero(valid)
    # 剔掉退化法向的样本：退化三角形没有法向，
    # 用它算出来的"偏移方向"是零向量，没有意义。
    valid[valid_indices[normal_lengths <= 1e-15]] = False
    if not np.any(valid):
        return empty
    # 剔除后重新取一遍下标：上面用的是 valid_indices 这个旧下标集，
    # 直接复用会把已剔除的样本又算回来。
    valid_indices = np.flatnonzero(valid)
    source_normals = source_mesh.triangle_normals[source_face_ids[valid_indices]]
    target_normals = target_mesh.triangle_normals[target_face_ids[valid_indices]]
    displacement = target_points[valid_indices] - source_points[valid_indices]
    # 有符号偏移：位移在源法向上的投影。
    # 符号表示"往法向正方向偏还是负方向偏"，
    # 这正是区分"缺料"与"多料"的依据。
    signed = np.sum(displacement * source_normals, axis=1)
    # 法向对齐度取绝对值：两个面朝向相同或相反都算"平行"，
    # 不加绝对值会把反向的（例如内表面）误判为不对齐。
    alignment = np.abs(np.sum(source_normals * target_normals, axis=1))
    # 容差角度先夹到 [0,90]：超出这个区间 cos 会变成负值，
    # 而"法向一致"的判据在超过 90 度后不再单调。
    cos_limit = float(
        np.cos(np.deg2rad(np.clip(normal_tolerance_deg, 0.0, 90.0)))
    )
    # 对齐的两个条件：法向夹角在容差内，
    # 且偏移量确实超过阈值的一半。
    # 第二个条件把"几乎重合"的样本排除掉——
    # 它们偏移太小，判不出方向。
    aligned_values = (
        (alignment >= cos_limit)
        & (np.abs(signed) >= max(threshold_mm * 0.5, 1e-12))
    )
    # 不对齐的算作在边界上：边界处两侧法向本来就不同，
    # 这类样本要单独统计而不是当成噪声丢掉。
    boundary_values = ~aligned_values
    result = dict(empty)
    # 只回填有效样本的位置，其余保持空壳里的 False：
    # 无效样本若填 True 会让下游把没量到的面当成量过了。
    result["signed_offset_mm"][valid_indices] = signed
    result["normal_alignment"][valid_indices] = alignment
    result["valid"][valid_indices] = True
    result["aligned"][valid_indices] = aligned_values
    result["boundary"][valid_indices] = boundary_values
    return result


def _reduce_directional_samples(
    face_ids: np.ndarray,
    evidence: dict[str, object],
    *,
    face_count: int,
) -> dict[str, object]:
    """我把采样级有向证据压缩到 Face，避免输出无法追溯的全局猜测。"""

    enabled = bool(evidence.get("enabled", False))
    # 五个数组都按面号建，长度等于面数：
    # 后面要把逐样本的证据"按面号累加"到这些数组上。
    signed_sum = np.zeros(face_count, dtype=np.float64)
    alignment_sum = np.zeros(face_count, dtype=np.float64)
    valid_count = np.zeros(face_count, dtype=np.int64)
    aligned_count = np.zeros(face_count, dtype=np.int64)
    boundary_count = np.zeros(face_count, dtype=np.int64)
    if not enabled:
        # 关闭时返回形状齐全的零数组而不是空字典：
        # 下游按固定键取值，返回空字典会在那里 KeyError，
        # 而"功能关闭"是一个合法的、需要继续往下走的状态。
        return {
            "enabled": False,
            "signed_sum": signed_sum,
            "alignment_sum": alignment_sum,
            "valid_count": valid_count,
            "aligned_count": aligned_count,
            "boundary_count": boundary_count,
        }
    ids = np.asarray(face_ids, dtype=np.int64)
    valid = np.asarray(evidence["valid"], dtype=bool)
    signed = np.asarray(evidence["signed_offset_mm"], dtype=np.float64)
    alignment = np.asarray(evidence["normal_alignment"], dtype=np.float64)
    aligned = np.asarray(evidence["aligned"], dtype=bool)
    boundary = np.asarray(evidence["boundary"], dtype=bool)
    # 可用样本的四道过滤：面号在范围内、样本本身有效、
    # 两个数值都有限。任一不满足都不能参与累加，
    # 否则一个 nan 会顺着累加污染掉整个面的统计。
    usable = (
        (ids >= 0)
        & (ids < face_count)
        & valid
        & np.isfinite(signed)
        & np.isfinite(alignment)
    )
    # 用 np.add.at 而不是 signed_sum[ids[usable]] += ...：
    # 同一个面号会出现很多次（一个面上有很多采样点），
    # 普通的花式索引赋值只保留最后一次，累加结果会严重偏小。
    # np.add.at 才是真正的逐个累加。
    np.add.at(signed_sum, ids[usable], signed[usable])
    np.add.at(alignment_sum, ids[usable], alignment[usable])
    np.add.at(valid_count, ids[usable], 1)
    # 两个计数的掩码是在 usable 之上再叠一层条件：
    # 只有可用样本里的"对齐的""在边界上的"才计数，
    # 分母仍是 usable 的总数，这样比值才有意义。
    np.add.at(aligned_count, ids[usable & aligned], 1)
    np.add.at(boundary_count, ids[usable & boundary], 1)
    return {
        "enabled": True,
        "signed_sum": signed_sum,
        "alignment_sum": alignment_sum,
        "valid_count": valid_count,
        "aligned_count": aligned_count,
        "boundary_count": boundary_count,
    }


def _region_directional_evidence(
    face_ids: list[int],
    stats: dict[str, object] | None,
) -> dict[str, object]:
    """我为区域生成有向证据摘要，并保留不确定状态。"""

    if not stats:
        # 无统计与"功能关闭"是两个不同的标签：
        # 前者说明方向证据从来没算过，后者说明算过但被关掉了。
        # 混成同一个标签会让"为什么没有方向证据"无法回答。
        return {
            "direction_evidence": "UNAVAILABLE",
            "signed_offset_mean_mm": None,
            "normal_alignment_mean": None,
            "direction_valid_sample_count": 0,
            "direction_aligned_sample_count": 0,
            "direction_boundary_sample_count": 0,
        }
    if not bool(stats.get("enabled", False)):
        return {
            "direction_evidence": "DISABLED",
            "signed_offset_mean_mm": None,
            "normal_alignment_mean": None,
            "direction_valid_sample_count": 0,
            "direction_aligned_sample_count": 0,
            "direction_boundary_sample_count": 0,
        }
    # 三个计数数组都按面号索引后求和：这些数组是"按面统计"的，
    # 区域只取属于它的那些面，直接对全表求和会把别的面算进来。
    ids = np.asarray(face_ids, dtype=np.int64)
    valid_count = np.asarray(stats["valid_count"], dtype=np.int64)[ids]
    aligned_count = np.asarray(stats["aligned_count"], dtype=np.int64)[ids]
    boundary_count = np.asarray(stats["boundary_count"], dtype=np.int64)[ids]
    total_valid = int(valid_count.sum())
    total_aligned = int(aligned_count.sum())
    total_boundary = int(boundary_count.sum())
    # 一个有效样本都没有时给 UNAVAILABLE 且均值为 None：
    # 均值给 0.0 会被读成"没有偏移"，
    # 而真实含义是"没有数据"，两者工程结论完全相反。
    if total_valid == 0:
        label = "UNAVAILABLE"
        signed_mean = None
        alignment_mean = None
    else:
        # 均值用"和/有效数"而不是直接 mean：
        # 面上样本数不同，直接对逐面均值再平均会让小面和大面同权。
        signed_mean = float(
            np.asarray(stats["signed_sum"], dtype=np.float64)[ids].sum()
            / total_valid
        )
        alignment_mean = float(
            np.asarray(stats["alignment_sum"], dtype=np.float64)[ids].sum()
            / total_valid
        )
        # 0.75 是"整体按法向偏移"的门槛：低于它说明
        # 偏移方向与法向不一致，更可能是错位而不是单纯的偏置。
        if total_aligned / total_valid >= 0.75:
            label = "NORMAL_ALIGNED_OFFSET"
        elif total_boundary / total_valid >= 0.75:
            label = "BOUNDARY_OR_TOPOLOGY_CHANGE"
        else:
            label = "MIXED_DIRECTIONAL_EVIDENCE"
    return {
        "direction_evidence": label,
        "signed_offset_mean_mm": signed_mean,
        "normal_alignment_mean": alignment_mean,
        "direction_valid_sample_count": total_valid,
        "direction_aligned_sample_count": total_aligned,
        "direction_boundary_sample_count": total_boundary,
    }


def _directional_summary(
    forward: dict[str, object],
    reverse: dict[str, object],
    *,
    normal_tolerance_deg: float,
) -> dict[str, object]:
    """我汇总双向有向证据，明确它仍是辅助诊断。"""

    def counts(value: dict[str, object]) -> dict[str, int]:
        """我把有向证据的布尔数组归约成计数，供置信度与告警使用。"""
        return {
            "valid_sample_count": int(np.sum(np.asarray(value["valid"], dtype=bool))),
            "aligned_offset_sample_count": int(
                np.sum(np.asarray(value["aligned"], dtype=bool))
            ),
            "boundary_or_topology_sample_count": int(
                np.sum(np.asarray(value["boundary"], dtype=bool))
            ),
        }

    return {
        "enabled": bool(forward.get("enabled", False)),
        "normal_tolerance_deg": float(normal_tolerance_deg),
        "classification_scope": "evidence_only_not_final_defect_semantics",
        "forward_reference_to_candidate": counts(forward),
        "reverse_candidate_to_reference": counts(reverse),
    }


def _connected_components(
    abnormal_faces: np.ndarray,
    adjacency: list[set[int]],
) -> list[list[int]]:
    """我用广度优先搜索把相邻的异常面聚成连通区域。

    输入是布尔掩码（哪些面异常）和三角面邻接表。我用 deque 做 BFS 而不是
    递归 DFS：大网格上递归会撞 Python 的递归深度上限，而 BFS 只受内存限制。
    每个连通块内部的 Face 编号我排序后再返回，保证同一份输入结果稳定。
    """

    visited: set[int] = set()
    components: list[list[int]] = []
    for face_id, is_abnormal in enumerate(abnormal_faces):
        # 只从"异常且还没访问过"的面出发，每个连通块只会被启动一次。
        if not is_abnormal or face_id in visited:
            continue
        component: list[int] = []
        queue: deque[int] = deque([face_id])
        visited.add(face_id)
        while queue:
            current = queue.popleft()
            component.append(current)
            # 只把"同样异常"的邻居加进来，正常面会切断连通性。
            for neighbor in adjacency[current]:
                if abnormal_faces[neighbor] and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def _make_region(
    region_id: int,
    face_ids: list[int],
    source_mesh: TriangleMesh,
    face_distances: np.ndarray,
    candidate_face_ids: list[int] | None,
    step_mapping_mesh: TriangleMesh | None,
    config: InspectionConfig,
    registration: RegistrationResult,
    defect_type: str,
    directional_face_stats: dict[str, object] | None = None,
) -> RegionResult:
    """我把一个连通面簇组装成一条完整的区域结果。

    这里做四件事：算面积加权的几何中心与包围盒、按误差分布算置信度、
    收集复核理由（warnings + review_required）、生成可追溯的几何签名。
    defect_type 决定"哪些 Face 编号要写进 reference_face_ids"：
    missing 记参考面，extra 记候选面，两边不会混。
    """

    ids = np.asarray(face_ids, dtype=np.int64)
    areas = source_mesh.triangle_areas[ids]
    total_area = float(areas.sum())
    triangles = source_mesh.triangle_vertices[ids]
    vertices = triangles.reshape(-1, 3)
    centroids = triangles.mean(axis=1)
    # 几何中心我按三角面面积加权：缺失区域常常由大小不一的碎片组成，
    # 不加权会被数量多但面积小的碎片把中心拉偏。
    if total_area > 1e-12:
        center = (centroids * areas[:, None]).sum(axis=0) / total_area
    else:
        center = vertices.mean(axis=0)
    errors = face_distances[ids]
    confidence = _confidence(errors, config.threshold_mm, registration.quality)
    warnings: list[str] = []
    review_required = False
    # 复核理由一：全局配准质量不好，区域位置本身就可能不准。
    if registration.quality != "good" or registration.scale_review_required:
        warnings.append("我建议人工复核该区域，因为全局配准质量不是 good。")
        review_required = True
    # 复核理由二：区域面积接近过滤阈值（不到阈值的 2 倍），
    # 说明它处在"报/不报"的边界上，值得人工看一眼。
    if total_area < max(config.min_region_area_mm2 * 2, 1e-12):
        warnings.append("我认为该区域面积较小，建议结合原始 CAD 面人工确认。")
        review_required = True
    # 候选面编号去重、排序并丢掉负数；这样下游拿到的是稳定且干净的列表。
    selected_candidate_face_ids = sorted(
        {
            int(value)
            for value in (
                candidate_face_ids
                if candidate_face_ids is not None
                else []
            )
            # 负数面号过滤掉：距离计算在无匹配时返回 -1，
            # 不过滤会把"没有对应的面"当成第 -1 个面传下去。
            if int(value) >= 0
        }
    )
    # "多出"缺陷在没有候选面号时回退到本区域的全部面：
    # extra 区域本身就只存在于候选模型上，没有对应的参考面，
    # 此时用它自己的面号列表才是正确表示。
    if defect_type == "extra" and not selected_candidate_face_ids:
        selected_candidate_face_ids = [int(value) for value in face_ids]
    # 映射到 STEP 面号时用 step_mapping_mesh：
    # 三角化网格与 STEP 的 Face 编号不是一套，
    # 必须走映射表才能让甲方按 STEP 里的面号去核对。
    candidate_step_face_ids = _map_step_face_ids(
        step_mapping_mesh or source_mesh,
        selected_candidate_face_ids,
    )
    # 参考侧只给 missing 类型：extra 类型本来就没有参考面，
    # 给一个空列表比给"参考面不存在"的误导性编号更诚实。
    reference_step_face_ids = _map_step_face_ids(
        source_mesh,
        face_ids if defect_type == "missing" else [],
    )
    direction = _region_directional_evidence(
        face_ids,
        directional_face_stats,
    )
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    # 几何签名独立于面号：面号会随三角化顺序变，签名不会，
    # 跨批次比对"是不是同一处缺陷"只能靠它。
    region_geometry_signature = _mesh_region_geometry_signature(
        defect_type=defect_type,
        area_mm2=total_area,
        center_xyz_mm=center,
        bbox_min_xyz_mm=bbox_min,
        bbox_max_xyz_mm=bbox_max,
        # 每个三角形的法向与面积都参与签名：只靠面积和包围盒
        # 区分不了"形状不同但外接盒相同的两块区域"。
        triangle_normals=source_mesh.triangle_normals[ids],
        triangle_areas=areas,
        triangle_count=len(face_ids),
    )
    return RegionResult(
        region_id=region_id,
        defect_type=defect_type,
        center_xyz_mm=center.tolist(),
        bbox_min_xyz_mm=bbox_min.tolist(),
        bbox_max_xyz_mm=bbox_max.tolist(),
        area_mm2=total_area,
        # 体积估算留 None：从网格面片推体积需要先封闭区域，
        # 而这里拿到的只是一片面，推算出的体积不可信。
        estimated_volume_mm3=None,
        # 三个偏差量都在无样本时给 0.0 而不是 None：
        # 它们是"这块区域偏了多少"的量，空区域偏 0 是成立的结论，
        # 而 len(errors)==0 只可能出现在零面积区域上。
        max_error_mm=float(errors.max()) if len(errors) else 0.0,
        mean_error_mm=float(errors.mean()) if len(errors) else 0.0,
        p95_error_mm=float(np.percentile(errors, 95)) if len(errors) else 0.0,
        triangle_count=len(face_ids),
        confidence=confidence,
        # 参考面号只在 missing 时给出：extra（多出的料）
        # 在参考模型上没有对应面，给面号会误导人去找根本不存在的面。
        reference_face_ids=(
            [int(value) for value in face_ids]
            if defect_type == "missing"
            else []
        ),
        review_required=review_required,
        region_geometry_signature=region_geometry_signature,
        candidate_face_ids=selected_candidate_face_ids,
        candidate_step_face_ids=candidate_step_face_ids,
        reference_step_face_ids=reference_step_face_ids,
        warnings=warnings,
        **direction,
    )


def _mesh_region_geometry_signature(
    *,
    defect_type: str,
    area_mm2: float,
    center_xyz_mm: np.ndarray,
    bbox_min_xyz_mm: np.ndarray,
    bbox_max_xyz_mm: np.ndarray,
    triangle_normals: np.ndarray,
    triangle_areas: np.ndarray,
    triangle_count: int,
) -> str:
    """我用区域几何摘要生成候选追踪签名，不把网格编号当作永久 ID。

    为什么不能用三角面编号当追踪 ID：三角面编号是**本次网格化的产物**。
    同一份 STEP 换一个弦高参数重新网格化，面的切分就完全不同，
    同一个物理缺陷会得到不同的编号集合，跨运行追踪直接失效。
    所以我用几何量（面积、中心、包围盒、面积加权法向、面数）生成签名，
    它们属于零件本身，不依赖三角化方式。

    三个让它稳定的细节：
      - 法向按面积加权后再归一化。区域内可能同时存在朝向不同的碎片，
        直接取平均会被小碎片主导，加权后才是区域的主导朝向；
      - 所有几何量先量化到 1e-6。不量化时浮点尾数差异会让同一个
        缺陷算出不同签名，追踪就失去意义；
      - ensure_ascii=True 并按 ascii 编码。让字节流与语言环境无关，
        签名在任何平台算出同一个值。
    """

    normals = np.asarray(triangle_normals, dtype=np.float64)
    areas = np.asarray(triangle_areas, dtype=np.float64)
    weighted_normal = np.sum(normals * areas[:, None], axis=0)
    normal_length = float(np.linalg.norm(weighted_normal))
    if normal_length > 1e-15:
        weighted_normal = weighted_normal / normal_length
    else:
        weighted_normal = np.zeros(3, dtype=np.float64)
    payload = {
        "signature_version": "mesh-region-geometry-v1",
        "defect_type": str(defect_type),
        "area_mm2": round(float(area_mm2), 6),
        "center_xyz_mm": _rounded_vector(center_xyz_mm),
        "bbox_min_xyz_mm": _rounded_vector(bbox_min_xyz_mm),
        "bbox_max_xyz_mm": _rounded_vector(bbox_max_xyz_mm),
        "mean_normal_xyz": _rounded_vector(weighted_normal),
        "triangle_count": int(triangle_count),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256(encoded).hexdigest()[:24]
    return f"meshregion-{digest}"


def _rounded_vector(value: np.ndarray) -> list[float]:
    """我把签名输入量化到固定精度，避免浮点尾数改变追踪结果。"""

    return [round(float(item), 6) for item in np.asarray(value, dtype=np.float64)]


def _candidate_triangle_ids_for_region(
    face_ids: list[int],
    sample_face_ids: np.ndarray,
    sample_candidate_face_ids: np.ndarray,
) -> list[int]:
    """我收集缺失区域采样点对应的候选三角面编号。

    这一步回答"参考件上这块缺失区域，在候选件上对应的是哪些面"。
    注意它常常是空集——参考件有而候选件没有的面，本来就没有对应物，
    这正是"缺失"的定义。所以空结果不是异常，是预期。

    我用 isin 做向量化筛选而不是逐点循环：采样点通常是几万到几十万，
    Python 层循环在这个量级上会成为瓶颈。

    编号逐个过 _safe_nonnegative_int：最近面查询在查不到时会返回 -1
    之类的哨兵值，不滤掉会让 -1 混进"对应的候选面"里，
    下游按编号取几何时会直接越界。
    """

    selected = np.isin(sample_face_ids, np.asarray(face_ids, dtype=np.int64))
    return sorted(
        {
            parsed
            for value in sample_candidate_face_ids[selected]
            if (parsed := _safe_nonnegative_int(value)) is not None
        }
    )


def _map_step_face_ids(
    mesh: TriangleMesh,
    triangle_ids: list[int],
) -> list[int]:
    """我把候选三角面编号映射到 STEP Face 编号。

    映射关系来自网格化时记录的 face_ids_by_triangle 元数据。它是
    "尽力而为"的信息，所以我做三道防御，任何一道不过就返回空列表
    让下游走复核，绝不猜：
      - 映射本身必须是 list/tuple，缺失或类型不对直接放弃；
      - 三角面编号必须在映射范围内，越界索引会抛 IndexError，
        不能让元数据长度和实际网格不一致搞崩整个检测；
      - 若元数据里声明了 face_count，映射出来的编号不得超过它。
        超出说明这份元数据属于另一个模型，编号体系对不上，
        用它做关联会张冠李戴。
    """

    raw_mapping = mesh.metadata.get("face_ids_by_triangle")
    if not isinstance(raw_mapping, (list, tuple)):
        return []
    raw_face_count = mesh.metadata.get("face_count")
    declared_face_count = (
        _safe_positive_int(raw_face_count)
        if raw_face_count is not None
        else None
    )
    if raw_face_count is not None and declared_face_count is None:
        return []
    return sorted(
        {
            parsed
            for triangle_id in triangle_ids
            if 0 <= triangle_id < len(raw_mapping)
            if (
                parsed := _safe_positive_int(raw_mapping[triangle_id])
            ) is not None
            and (
                declared_face_count is None
                or parsed <= declared_face_count
            )
        }
    )


def _safe_nonnegative_int(value: Any) -> int | None:
    """我只接受有限且无小数部分的非负编号，坏元数据只进入复核路径。

    我为什么连"有小数部分"都要拒：编号来自 STEP 实体编号，一定是整数。
    出现 1.5 说明上游把别的量（例如长度或角度）当成编号写进来了，
    直接 ``int()`` 截断会得到一个看起来合法、实际错位的编号，
    把 A 面的证据挂到 B 面上。拒掉它至少能进复核被人发现。

    我为什么用 ``math.floor`` 判整数而不是 ``float.is_integer()``：
    两者对有限值等价，但 floor 对负值的行为是我要显式确认的，
    配合上面 ``numeric < 0`` 的守卫能一次性把负整数也挡掉，
    不需要再写一条分支。
    """

    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0.0 or numeric != math.floor(numeric):
        return None
    return int(numeric)


def _safe_positive_int(value: Any) -> int | None:
    """我把 STEP Face 和声明计数限制为正整数，拒绝零和布尔值。"""

    parsed = _safe_nonnegative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _step_face_mapping_diagnostics(mesh: TriangleMesh) -> dict[str, object]:
    """我核对 STEP Face 映射长度和每一项编号，避免静默错配。"""

    raw_mapping = mesh.metadata.get("face_ids_by_triangle")
    # 映射缺失是正常情况：只有从 STEP 三角化来的网格才带这个元数据，
    # 纯 STL 输入本来就没有。此时 review_required=False——
    # 没有映射不等于映射有问题。
    if raw_mapping is None:
        return {
            "present": False,
            "review_required": False,
            "triangle_count": int(len(mesh.faces)),
            "mapping_length": 0,
            "valid_entry_count": 0,
            "invalid_entry_count": 0,
            "out_of_range_entry_count": 0,
            "declared_face_count": None,
            "declared_face_count_valid": True,
            "length_matches": False,
        }
    # 映射存在但类型不对就是数据损坏，必须要求复核：
    # 它与"没有映射"的区别在于——本该有却坏了，
    # 静默按"没有"处理会让所有面号溯源失效而不被发现。
    if not isinstance(raw_mapping, (list, tuple)):
        return {
            "present": True,
            "review_required": True,
            "triangle_count": int(len(mesh.faces)),
            "mapping_length": None,
            "valid_entry_count": 0,
            # 无效条目数按三角面数计：每个三角形都对应一个坏条目，
            # 这样"影响范围"是可度量的。
            "invalid_entry_count": int(len(mesh.faces)),
            "out_of_range_entry_count": 0,
            "declared_face_count": None,
            "declared_face_count_valid": False,
            "length_matches": False,
        }
    raw_face_count = mesh.metadata.get("face_count")
    # 声明的面数用于校验映射里的最大编号是否越界。
    # 取不到时留 None，并把"取不到"记为有效——
    # 缺少声明只是少了一道校验，不等于声明本身有问题。
    declared_face_count = (
        _safe_positive_int(raw_face_count)
        if raw_face_count is not None
        else None
    )
    declared_face_count_valid = (
        raw_face_count is None or declared_face_count is not None
    )
    # 逐条解析而不是整体解析：映射里可能混有 None 或字符串，
    # 逐条容错能让"部分损坏"仍保留其余可用条目。
    parsed_values = [
        _safe_positive_int(value)
        for value in raw_mapping
    ]
    valid_entry_count = sum(value is not None for value in parsed_values)
    invalid_entry_count = len(raw_mapping) - valid_entry_count
    out_of_range_entry_count = sum(
        value is not None
        and declared_face_count is not None
        and value > declared_face_count
        for value in parsed_values
    )
    length_matches = len(raw_mapping) == len(mesh.faces)
    return {
        "present": True,
        "review_required": (
            not length_matches
            or invalid_entry_count > 0
            or out_of_range_entry_count > 0
            or not declared_face_count_valid
        ),
        "triangle_count": int(len(mesh.faces)),
        "mapping_length": int(len(raw_mapping)),
        "valid_entry_count": int(valid_entry_count),
        "invalid_entry_count": int(invalid_entry_count),
        "out_of_range_entry_count": int(out_of_range_entry_count),
        "declared_face_count": declared_face_count,
        "declared_face_count_valid": declared_face_count_valid,
        "length_matches": bool(length_matches),
    }


def _confidence(
    errors: np.ndarray,
    threshold_mm: float,
    registration_quality: str,
) -> float:
    """我把区域误差超出阈值的程度折算成 0~1 的置信度。

    公式是 base = 1 - exp(-margin)，其中
    margin = 平均误差 / 阈值 - 1（小于 0 时截断为 0）。
    它表达的是：平均误差只是刚刚压线时置信度接近 0；超得越多越接近 1，
    但永远到不了 1，因为我不会把统计推断说成确定结论。
    配准质量再乘一个折扣：warning 打 75%，bad 打 35%。
    """

    if len(errors) == 0:
        return 0.0
    margin = max(float(np.mean(errors) / max(threshold_mm, 1e-12) - 1.0), 0.0)
    base = 1.0 - float(np.exp(-margin))
    if registration_quality == "warning":
        base *= 0.75
    elif registration_quality == "bad":
        base *= 0.35
    return float(np.clip(base, 0.0, 1.0))


def _global_metrics(
    reference: TriangleMesh,
    face_distances: np.ndarray,
    regions: list[RegionResult],
    candidate: TriangleMesh,
    config: InspectionConfig,
    extra_face_distances: np.ndarray,
    missing_region_count: int,
    extra_region_count: int,
    extra_area_mm2: float,
    reference_sampled_face_count: int,
    reference_unsampled_face_count: int,
    candidate_sampled_face_count: int,
    candidate_unsampled_face_count: int,
    reference_sampled_face_mask: np.ndarray,
    candidate_sampled_face_mask: np.ndarray,
    reference_step_face_mapping: dict[str, object],
    candidate_step_face_mapping: dict[str, object],
) -> dict[str, object]:
    """我汇总整对模型的全局指标，供报告和审计使用。

    我刻意把"采样到的面"和"未采样的面"分开统计：面积、误差分位数这些数
    只对采样掩码覆盖到的面计算，未采样面另记计数和覆盖率。这样甲方看到
    coverage < 1 就知道当前结论只覆盖了一部分表面，不会误读成"全模型通过"。
    """

    reference_sampled_face_mask = np.asarray(
        reference_sampled_face_mask,
        dtype=bool,
    )
    candidate_sampled_face_mask = np.asarray(
        candidate_sampled_face_mask,
        dtype=bool,
    )
    if len(reference_sampled_face_mask) != len(reference.faces):
        raise ValueError("我要求参考 Face 采样掩码长度与参考网格一致。")
    if len(candidate_sampled_face_mask) != len(candidate.faces):
        raise ValueError("我要求候选 Face 采样掩码长度与候选网格一致。")
    # 面积统计只统计"超过阈值且真的被采样到"的面，避免未采样面的 0 距离
    # 把异常面积算少，也避免把未采样面误当成异常。
    abnormal = (
        (face_distances >= config.threshold_mm)
        & reference_sampled_face_mask
    )
    missing_area = float(reference.triangle_areas[abnormal].sum())
    extra_abnormal = (
        (extra_face_distances >= config.threshold_mm)
        & candidate_sampled_face_mask
    )
    sampled_extra_area = float(candidate.triangle_areas[extra_abnormal].sum())
    total_area = reference.total_area
    sampled_reference_distances = face_distances[reference_sampled_face_mask]
    sampled_candidate_distances = extra_face_distances[candidate_sampled_face_mask]
    # 包围盒尺度比用来诊断"是不是单位或缩放对不上"：参考和候选的包围盒
    # 三个轴向尺寸逐轴相除，正常应该都接近 1。
    bbox_size_reference = reference.bbox_max - reference.bbox_min
    bbox_size_candidate = candidate.bbox_max - candidate.bbox_min
    # 某个轴向尺寸本来就接近 0（薄板）时不做除法，直接记 1，避免除零爆掉。
    nonzero = np.abs(bbox_size_reference) > 1e-12
    bbox_scale_ratios = np.ones(3, dtype=np.float64)
    bbox_scale_ratios[nonzero] = (
        bbox_size_candidate[nonzero] / bbox_size_reference[nonzero]
    )
    warnings: list[str] = []
    # 同时报出缺失和多余，是"配准/单位有问题"的典型信号，我在这里给两条
    # 不同的诊断：尺度接近说明可能是整体平移或姿态误差；尺度差得多说明
    # 更可能是单位、缩放或三角化参数不一致。
    if config.bidirectional_check and missing_region_count and extra_region_count:
        # 同时出现缺失与多余，通常不是真的"既少料又多料"，而是整体对不齐。
        # 我用包围盒尺度比值区分两类根因：
        #   尺度接近（<=3%）-> 平移/旋转/姿态误差，模型本身没问题；
        #   尺度差异超 5%  -> 单位或缩放不一致，该先查导入参数。
        # 中间地带不猜，保持沉默，避免给出没有依据的定性结论。
        if float(np.max(np.abs(bbox_scale_ratios - 1.0))) <= 0.03:
            warnings.append(
                "我同时观察到缺失和多余材料候选，且包围盒尺度接近；"
                "可能存在整体平移、姿态误差或网格边界误差。"
            )
        elif float(np.max(bbox_scale_ratios)) > 1.05 or float(
            np.min(bbox_scale_ratios)
        ) < 0.95:
            warnings.append(
                "我同时观察到缺失和多余材料候选，且包围盒尺度差异明显；"
                "请优先排查单位、缩放或三角化参数不一致。"
            )
    # 比值类指标在分母为 0 时统一给 0.0：面积为 0 的模型是空模型，
    # 此时"缺失比例"没有定义，用 0.0 表示"无可比基准"而不是抛错中断。
    return {
        # 面积类指标与距离类指标分开列，且各自标注来源：
        # 面积是整模型的几何量，距离是采样点上的统计量，
        # 两者量纲与可信度都不同，混在一起看会得出错误结论。
        "reference_area_mm2": float(total_area),
        "candidate_area_mm2": float(candidate.total_area),
        "missing_area_mm2": missing_area,
        "missing_area_ratio": float(missing_area / total_area)
        if total_area > 1e-12
        else 0.0,
        "extra_area_mm2": float(extra_area_mm2),
        "extra_area_ratio": float(extra_area_mm2 / candidate.total_area)
        if candidate.total_area > 1e-12
        else 0.0,
        # 缺失/多余面积接近度 = min/max，取值 0~1。
        #
        # 我为什么要单独给这个量：一件上"既缺料又多料"在真实模具上很少见，
        # 而**两者面积几乎相等**几乎必然是整体没对齐——一侧凹进去多少，
        # 另一侧就凸出来多少。我在实测语料上量到 19 件里有 12 件
        # 接近度大于 0.9（最高的 0.997），也就是说这批数据里
        # "疑似整体错位"是常态而不是例外。
        #
        # 这个量能让人一眼排出复核优先级：接近度越接近 1，
        # 越该先去查配准与单位，而不是逐个区域去量尺寸。
        # 只在两侧都有面积时才有意义，否则记 None 而不是编一个 0。
        "missing_extra_balance": (
            float(
                min(missing_area, extra_area_mm2)
                / max(missing_area, extra_area_mm2)
            )
            if missing_area > 1e-12
            and extra_area_mm2 > 1e-12
            and max(missing_area, extra_area_mm2) > 1e-12
            else None
        ),
        "sampled_extra_area_mm2": sampled_extra_area,
        # 误差统计用 max/mean/p95 三个口径并列：
        # mean 会被大面积的贴合区拉低，max 又会被单个离群点主导，
        # p95 才反映"绝大多数区域"的偏差水平，三者缺一不可。
        "max_error_mm": float(sampled_reference_distances.max())
        if len(sampled_reference_distances)
        else 0.0,
        "mean_error_mm": float(sampled_reference_distances.mean())
        if len(sampled_reference_distances)
        else 0.0,
        "p95_error_mm": float(np.percentile(sampled_reference_distances, 95))
        if len(sampled_reference_distances)
        else 0.0,
        # 采样面数必须一并给出：误差统计的可信度取决于样本量，
        # 只有 3 个采样面时的 p95 与 3000 个面的 p95 完全不是一回事。
        "sampled_reference_error_face_count": int(
            len(sampled_reference_distances)
        ),
        "sampled_candidate_error_face_count": int(
            len(sampled_candidate_distances)
        ),
        "sampled_reference_max_error_mm": float(
            sampled_reference_distances.max()
        )
        if len(sampled_reference_distances)
        else 0.0,
        "sampled_reference_mean_error_mm": float(
            sampled_reference_distances.mean()
        )
        if len(sampled_reference_distances)
        else 0.0,
        "sampled_reference_p95_error_mm": float(
            np.percentile(sampled_reference_distances, 95)
        )
        if len(sampled_reference_distances)
        else 0.0,
        "sampled_candidate_max_error_mm": float(
            sampled_candidate_distances.max()
        )
        if len(sampled_candidate_distances)
        else 0.0,
        # 参考侧与候选侧的误差必须分开统计：
        # 合并成一个"总体误差"会掩盖方向性偏差——
        # 比如候选整体比参考外扩，两侧误差符号相反、均值相消。
        "sampled_candidate_mean_error_mm": float(
            sampled_candidate_distances.mean()
        )
        if len(sampled_candidate_distances)
        else 0.0,
        "sampled_candidate_p95_error_mm": float(
            np.percentile(sampled_candidate_distances, 95)
        )
        if len(sampled_candidate_distances)
        else 0.0,
        # 阈值写回结果：缺失区域的判定完全由它决定，
        # 不给阈值就无法解释"为什么这块算缺失"。
        "threshold_mm": float(config.threshold_mm),
        # 缺失与多出分开计数：两者工程含义相反
        # （待检模型少了面 / 多了面），合并成"差异数"无法指导返工。
        "missing_region_count": int(missing_region_count),
        "extra_region_count": int(extra_region_count),
        # 异常面数与总面数成对给出：异常面数没有分母
        # 就无法判断"19 张异常面"到底是多还是少。
        "abnormal_face_count": int(np.sum(abnormal)),
        "reference_face_count": int(len(reference.faces)),
        "candidate_abnormal_face_count": int(np.sum(extra_abnormal)),
        "candidate_face_count": int(len(candidate.faces)),
        # 两个方向的"已采样/未采样"分别统计：
        # 采样是"逐面找最近距离"，抽样的面数不足
        # 会让下面的覆盖率成为关键可信度指标。
        "reference_sampled_face_count": int(reference_sampled_face_count),
        "reference_unsampled_face_count": int(reference_unsampled_face_count),
        # 空网格时覆盖率给 1.0 而不是 0.0：
        # 0 张面里采样了 0 张是"全部覆盖"，
        # 给 0 会被读成"完全没采到样"，从而错误地触发复核。
        "reference_sampling_coverage_ratio": float(
            reference_sampled_face_count / len(reference.faces)
        )
        if len(reference.faces)
        else 1.0,
        "candidate_sampled_face_count": int(candidate_sampled_face_count),
        "candidate_unsampled_face_count": int(candidate_unsampled_face_count),
        "candidate_sampling_coverage_ratio": float(
            candidate_sampled_face_count / len(candidate.faces)
        )
        if len(candidate.faces)
        else 1.0,
        # 是否检查与检查状态分成两个字段：
        # 布尔供机器判断，字符串说明"为什么是这个值"。
        # 没开双向检查时状态写 disabled_by_config 而不是 checked，
        # 否则"没查"会被读成"查过了且没问题"。
        "candidate_sampling_checked": bool(config.bidirectional_check),
        "candidate_sampling_status": (
            "checked"
            if config.bidirectional_check
            else "disabled_by_config"
        ),
        "reference_step_face_mapping": reference_step_face_mapping,
        "candidate_step_face_mapping": candidate_step_face_mapping,
        # 面号映射的复核标记取两个方向的或：
        # 任一侧映射不可信，整份"这块区域对应 STEP 上哪些面"
        # 的结论就不能直接用。
        "step_face_mapping_review_required": bool(
            reference_step_face_mapping["review_required"]
            or candidate_step_face_mapping["review_required"]
        ),
        # 采样复核标记：任一方向有未采样面就置位。
        # 它与上面的映射复核是两个不同的环节，
        # 分开才能定位是哪一步留下的不确定性。
        "sampling_review_required": bool(
            reference_unsampled_face_count > 0
            or candidate_unsampled_face_count > 0
        ),
        "bidirectional_check": bool(config.bidirectional_check),
        # 包围盒尺度比：它记录两个模型配准时被缩放了多少，
        # 缩放过大说明两个模型的单位或版本不一致。
        "bbox_scale_ratios": bbox_scale_ratios.tolist(),
        "diagnostic_warnings": warnings,
    }
