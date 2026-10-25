"""我把完整检测流程串起来，作为命令行和未来 API 的共同入口。

原理说明（任务书要求关键算法附原理）：

inspect_pair 的执行顺序是固定的，每一步都为下一步准备证据：

1. 校验输入对（存在性、大小上限、后缀）；
2. 读参考件和候选件成网格（STL 直接读，STEP 走 OCCT 三角化）；
3. 校验网格质量（退化面、重复面、非流形边、孤立点）；
4. 解析单位策略，决定结果单位是 mm 还是 unknown；
5. 配准：把候选件对齐到参考件，拿到刚体变换和残差；
6. 推导有效阈值：absolute 用配置值，auto 用包围盒尺度和配准残差算；
7. 跑双向缺失/多余区域检测；
8. 可选地把 STEP 工程区域 JSON 关联到缺失区域上（补充工程语义）；
9. 汇总所有复核理由（分辨率、质量、单位、配准、区域关联），定 review 状态；
10. 写彩色 PLY、regions.csv、result.json，返回 InspectionResult。

我在这里刻意做的一件事是"把警告当一等公民"：只要任何一步的证据不足以
支撑确定结论（配准不是 good、网格有退化面、三角化偏差大于阈值、单位未知、
STEP 映射断链），我就把整体状态降级成 review，而不是硬给一个 pass/fail。
任务书要的是 99% 准确率和 1% 漏检率，宁可控告复核也不能把不确定说成确定。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .detection import detect_missing_regions
from .detection.cad_face_mapping import (
    map_regions_to_cad_faces,
    summarize_region_cad_faces,
)
from .detection.threshold import ThresholdRecommendation, recommend_threshold_mm
from .detection.step_region_join import (
    join_step_engineering_regions,
    load_step_engineering_region_map,
)
from .errors import InspectorError
from .geometry import load_stl
from .io.file_validation import validate_mesh, validate_model_pair
from .io.step import load_step_as_mesh
from .models import InspectionConfig, InspectionResult, normalize_path
from .output.atomic import atomic_path
from .output import write_colored_ply, write_csv, write_json
from .registration import align_mesh


def inspect_pair(
    candidate_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
    config: InspectionConfig | None = None,
    reference_engineering_json: str | Path | None = None,
    candidate_engineering_json: str | Path | None = None,
) -> InspectionResult:
    """我执行一次候选模型与参考模型的完整几何比对。

    参数里的两个 engineering_json 是可选的 STEP 工程区域报告；
    不传就跳过工程语义关联，缺失区域检测本身照常进行。
    """

    config = config or InspectionConfig()
    # 第一步：校验输入对。这一步会检查文件存在、后缀、大小上限，
    # 并返回规范化后的路径和元数据，后面的读取都用它的结果。
    paths = validate_model_pair(
        candidate_path,
        reference_path,
        max_size_bytes=config.max_input_file_size_bytes,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 第二步 + 第三步：读网格并做质量体检。参考件在前，因为它决定阈值尺度。
    reference = _load_mesh(paths["reference"], config)
    reference_quality = validate_mesh(reference, "参考")
    candidate = _load_mesh(paths["candidate"], config)
    candidate_quality = validate_mesh(candidate, "待检测")

    # 第四步：单位策略。我用一个集合装两侧的声明单位，方便整体判断。
    known_units = {
        str(reference.metadata.get("unit", "")).lower(),
        str(candidate.metadata.get("unit", "")).lower(),
    }
    candidate_unit = str(candidate.metadata.get("unit", "unknown")).lower()
    reference_unit = str(reference.metadata.get("unit", "unknown")).lower()
    warnings_for_unit: list[str] = []
    # require_mm 是硬门禁：只要有一侧不是明确 mm 就直接失败，不进入检测。
    if config.unit_policy == "require_mm" and known_units != {"mm"}:
        raise InspectorError(
            "我要求参考模型和待检测模型都明确声明 mm，"
            "但当前至少有一个输入单位未知或明确不是 mm。"
        )
    if config.unit_policy == "unknown":
        # unknown 策略：我保留输入数值原样，但明确声明"这不代表毫米"。
        result_unit = "unknown"
        unit_source = "policy_unknown"
        warnings_for_unit.append(
            "我按 unknown 单位策略保留输入数值，不把结果解释为毫米；"
            "当前结果不能直接用于毫米级验收。"
        )
    elif known_units == {"mm"}:
        # 两侧都明确声明 mm，单位来源记为输入元数据，可信度最高。
        result_unit = "mm"
        unit_source = "input_metadata"
    else:
        # 兜底：按配置假定 mm。我会额外区分"STEP 读取器已经归一化过"
        # 和"完全没有可验证换算"两种情况，因为前者至少留下了审计记录。
        result_unit = "mm"
        unit_source = "config_assumption"
        explicit_non_mm = sorted(
            unit
            for unit in known_units
            # 我为什么要把 mm 也排除掉（2026-10-14 修的真 bug）：
            # 原来只排除了空/未知，于是"一侧声明 mm、另一侧未知"时，
            # mm 会被当成**非 mm** 单位报出去，警告变成
            # "我检测到输入文件声明了非 mm 单位 mm"——一句自相矛盾的话。
            # 甲方读到它只会更困惑：到底是不是 mm？
            # 注意 known_units 里的值已经在入口处统一 lower 过了。
            if unit not in {"", "unknown", "none", "mm"}
        )
        if explicit_non_mm:
            normalized_by_backend = any(
                str(mesh.metadata.get("unit_normalization", "")).lower()
                == "occt_step_reader_to_mm"
                for mesh in (candidate, reference)
            )
            if normalized_by_backend:
                unit_warning = (
                    "我检测到输入文件声明了非 mm 单位 "
                    f"{', '.join(explicit_non_mm)}；我不在检测管线重复换算，"
                    "STEP 读取器的归一化比例已记录在 unit_audit，"
                    "当前仍按复核策略处理。"
                )
            else:
                unit_warning = (
                    "我检测到输入文件声明了非 mm 单位 "
                    f"{', '.join(explicit_non_mm)}；当前没有可验证的自动换算，"
                    "按 assume_mm 处理只能作为复核线索，请改用 require_mm "
                    "或先完成单位归一化。"
                )
        else:
            unit_warning = (
                "我没有从全部输入模型得到可靠单位；当前按配置 assume_mm 处理，"
                "请确认源 CAD 和 STL 数值确实使用 mm。"
            )
        warnings_for_unit.append(unit_warning)

    aligned_candidate, registration = align_mesh(candidate, reference, config)
    # 我必须先配准再推阈值：阈值里的"噪声分量"用的就是配准残差。
    # 顺序反过来就只能拿上一轮或默认的残差去定阈值，那会让阈值
    # 与本次实际配准精度脱钩——实测在姿态差的数据上这样会把阈值
    # 定得偏小，把整件判成破损。
    # 第五步 + 第六步：先配准拿残差，再用残差和尺度推导有效阈值。
    threshold_audit = recommend_threshold_mm(
        reference,
        registration.mean_residual_mm,
        config,
    )
    # 我方把推荐阈值真正应用到本次检测：absolute 模式沿用配置值，
    # auto 模式改用按尺度和配准残差推导出来的阈值。
    effective_config = replace(
        config,
        threshold_mm=threshold_audit.effective_mm,
    )
    # 第七步：核心检测。注意我用的是 effective_config 和配准后的候选件。
    regions, global_metrics, warnings, face_distances = detect_missing_regions(
        reference=reference,
        candidate=aligned_candidate,
        config=effective_config,
        registration=registration,
    )
    global_metrics["threshold_audit"] = threshold_audit.to_dict()
    warnings.extend(_threshold_warnings(threshold_audit))
    # 第八步：可选工程语义关联。两边 JSON 都没传时这两个 map 是 None，
    # join_step_engineering_regions 会走"功能未启用"分支，不会报错。
    reference_region_map = (
        load_step_engineering_region_map(reference_engineering_json)
        if reference_engineering_json is not None
        else None
    )
    candidate_region_map = (
        load_step_engineering_region_map(candidate_engineering_json)
        if candidate_engineering_json is not None
        else None
    )
    region_join_metrics, region_join_warnings, region_join_review = (
        join_step_engineering_regions(
            regions,
            reference_map=reference_region_map,
            candidate_map=candidate_region_map,
        )
    )
    global_metrics["step_engineering_region_join"] = region_join_metrics
    # 第八步之二：把区域映射到 CAD 面编号。这一步只在参考件是 STEP 时
    # 才真正生效，其余情况会返回"未启用 + 原因"，不影响检测结果本身。
    # 我为什么要接进主流程：检测报告说"这里有个 11 mm2 的缺失"，甲方
    # 还得自己回 CAD 里找是哪个面；而"缺失落在第 17 号 CAD 面上"才是
    # 模具工程师能直接动手的那句话。
    cad_face_mapping = map_regions_to_cad_faces(
        regions=regions,
        reference_mesh=reference,
        reference_path=paths["reference"],
        reference_metadata=reference.metadata,
        # 多料区域只存在于候选件上，要拿候选件自己的映射表去翻译；
        # 缺料走参考件，两条路径各用各的表，不能混。
        # 这里必须传**配准后**的候选件：区域的面积是按它算的，
        # 传未配准的那份会让"逐面面积之和 == 区域面积"这一步对不上。
        candidate_mesh=aligned_candidate,
        candidate_path=paths["candidate"],
        candidate_metadata=aligned_candidate.metadata,
        # 三角化参数必须和 _load_mesh 读参考件时用的一致，
        # 否则三角数不同、下标对不上，映射出的面号全是错的。
        linear_deflection=config.mesh_linear_deflection_mm,
        angular_deflection_deg=config.mesh_angular_deflection_deg,
    )
    mapping_payload = cad_face_mapping.to_dict()
    # 我把"人能直接读的那一句"也写进去：甲方不该为了看懂结论
    # 先去解析 region_cad_face_ids 再自己算占比。
    mapping_payload["summary_lines"] = summarize_region_cad_faces(cad_face_mapping)
    global_metrics["cad_face_mapping"] = mapping_payload
    # 映射没启用时要留一条可读提示：静默留空会让"没做映射"
    # 看起来像"这个区域不属于任何面"。
    if not cad_face_mapping.enabled:
        warnings.append(cad_face_mapping.reason)
    warnings.extend(cad_face_mapping.warnings)
    global_metrics["input_metadata"] = {
        "candidate": paths["candidate_metadata"],
        "reference": paths["reference_metadata"],
    }
    warnings.extend(region_join_warnings)
    warnings.extend(registration.warnings)
    # 我把检测内部产生的诊断警告上浮到顶层 warnings，
    # 让甲方只需要看一个列表就能看到所有复核理由。
    warnings.extend(
        str(item)
        for item in global_metrics.pop("diagnostic_warnings", [])
    )
    resolution_warnings = _mesh_resolution_warnings(
        candidate=candidate,
        reference=reference,
        threshold_mm=effective_config.threshold_mm,
    )
    warnings.extend(resolution_warnings)
    global_metrics["resolution_review_required"] = bool(resolution_warnings)
    quality_warnings = _mesh_quality_warnings(
        candidate_quality=candidate_quality,
        reference_quality=reference_quality,
    )
    warnings.extend(quality_warnings)
    global_metrics["quality_review_required"] = bool(quality_warnings)
    global_metrics["quality_warning_count"] = len(quality_warnings)
    # 第九步：把所有复核来源聚合成一个 review_required。
    global_metrics["unit_review_required"] = bool(warnings_for_unit)
    global_metrics["unit_audit"] = {
        "policy": config.unit_policy,
        # 声明的单位与实际坐标单位分开记：
        # 两者不一致正是"单位有问题"的征兆，
        # 只记一个就无法发现这种不一致。
        "candidate_source_unit": candidate_unit,
        "reference_source_unit": reference_unit,
        "candidate_coordinate_unit": candidate.metadata.get(
            "coordinate_unit",
            "unknown",
        ),
        "reference_coordinate_unit": reference.metadata.get(
            "coordinate_unit",
            "unknown",
        ),
        # 缩放到 mm 的系数留 None 而不是默认 1.0：
        # 取不到时写 1.0 等于声称"已经确认是毫米"，
        # 而真实情况是"不知道"。
        "candidate_unit_scale_to_mm": candidate.metadata.get(
            "unit_scale_to_mm",
        ),
        "reference_unit_scale_to_mm": reference.metadata.get(
            "unit_scale_to_mm",
        ),
        "candidate_unit_normalization": candidate.metadata.get(
            "unit_normalization",
            "unknown",
        ),
        "reference_unit_normalization": reference.metadata.get(
            "unit_normalization",
            "unknown",
        ),
        # 明确声明"管线不做换算"：这点必须写死在结果里。
        # 自动换算会让阈值语义随输入单位漂移，
        # 而实测阈值都是按毫米标定的。
        "coordinate_conversion_applied_by_pipeline": False,
        "coordinate_conversion_note": (
            "我没有在检测管线内自动换算非 mm 坐标；"
            "STEP 几何单位行为由 CAD 读取后端负责，需结合后端审计。"
        ),
    }
    # 配准质量不为 good 就要求复核：配准不准时
    # 后面所有"差异面积"都没有意义，这个标记是全局闸门。
    global_metrics["registration_review_required"] = (
        registration.quality != "good"
        or registration.scale_review_required
    )
    global_metrics["registration_scale"] = float(registration.scale)
    global_metrics["registration_scale_review_required"] = bool(
        registration.scale_review_required
    )
    review_required = bool(
        resolution_warnings
        or quality_warnings
        or warnings_for_unit
        or registration.quality != "good"
        or registration.scale_review_required
        or region_join_review
    )
    if review_required:
        global_metrics["status"] = "review"
    # 第十步：写产物。彩色 PLY 可以关（大网格很占磁盘），JSON/CSV 必写。
    artifacts: dict[str, str] = {}
    if config.write_colored_mesh:
        colored_path = write_colored_ply(
            reference,
            face_distances,
            effective_config.threshold_mm,
            output_dir / "colored_reference.ply",
        )
        artifacts["colored_mesh"] = normalize_path(colored_path)

    json_path = output_dir / "result.json"
    csv_path = output_dir / "regions.csv"
    artifacts["json"] = normalize_path(json_path)
    artifacts["csv"] = normalize_path(csv_path)
    if config.write_intermediate:
        # 中间 JSON 只给审计用：它记录每个 Face 的距离和采样掩码，
        # 让"为什么这个面被判成异常"可以逐面核对。
        intermediate = output_dir / "intermediate_face_distances.json"
        _, sampled_face_ids = reference.sample_points(
            samples_per_triangle=config.samples_per_triangle,
            max_points=config.max_query_points,
        )
        sampled_face_mask = np.zeros(len(reference.faces), dtype=bool)
        valid_sampled_ids = sampled_face_ids[
            (sampled_face_ids >= 0)
            & (sampled_face_ids < len(reference.faces))
        ]
        sampled_face_mask[valid_sampled_ids] = True
        with atomic_path(intermediate) as temporary:
            temporary.write_text(
                json.dumps(
                    {
                        "face_distances_mm": face_distances.tolist(),
                        "sampled_face_ids": sorted(
                            {int(value) for value in valid_sampled_ids}
                        ),
                        "sampled_face_mask": sampled_face_mask.tolist(),
                        "threshold_mm": effective_config.threshold_mm,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
        artifacts["face_distances"] = normalize_path(intermediate)
    result = InspectionResult(
        # 我把 status 从 global_metrics 里 pop 出来再传给 InspectionResult：
        # 两份都留会在 JSON 里出现两个不一致的状态字段，甲方核对时
        # 不知道该信哪个。pop 掉保证只有一份权威状态。
        # status 从 global_metrics 里取出来后就 pop 掉，避免在报告里出现两份。
        status=str(global_metrics.pop("status", "review")),
        candidate_file=normalize_path(candidate_path),
        reference_file=normalize_path(reference_path),
        unit=result_unit,
        unit_source=unit_source,
        threshold_mm=effective_config.threshold_mm,
        registration=registration,
        global_metrics={
            **global_metrics,
            "candidate_quality": candidate_quality,
            "reference_quality": reference_quality,
        },
        regions=regions,
        artifacts=artifacts,
        warnings=_unique_strings(warnings + warnings_for_unit),
        config=effective_config,
    )
    write_csv(result, csv_path)
    write_json(result, json_path)
    # 我最后才写 CSV/JSON，而不是边算边写：这两份是"权威结果"，
    # 只有当整条流水线走完、所有警告都收集齐了才有意义。中途写
    # 会在异常时留下一份"看起来完整、实则缺了后半程证据"的报告。
    return result


def _load_mesh(path: str | Path, config: InspectionConfig):
    """我按后缀选读取器：.stl 直读，.stp/.step 走 OCCT 三角化。

    STEP 的三角化偏差来自配置，所以同一步文件在不同配置下会得到不同网格；
    这也是为什么我要求"参考件和候选件用一致的三角化参数"。
    其它后缀我直接抛 InspectorError，不做"猜格式"的兜底。
    """

    path = Path(path)
    if path.suffix.lower() == ".stl":
        return load_stl(path)
    if path.suffix.lower() in {".stp", ".step"}:
        return load_step_as_mesh(
            path,
            linear_deflection_mm=config.mesh_linear_deflection_mm,
            angular_deflection_deg=config.mesh_angular_deflection_deg,
        )
    raise InspectorError(f"我不支持的待检测文件类型：{path.suffix}")


def _unique_strings(values: list[str]) -> list[str]:
    """我按首次出现顺序去重字符串列表，用来清理重复的 warnings。

    我为什么保序去重而不用 set：警告的**出现顺序**本身就是排查线索
    （先报单位问题、再报配准问题、最后报区域问题），转成 set 会把这个
    顺序打乱、每次运行结果还不一样，两份报告无法逐行比对。
    """

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _mesh_resolution_warnings(
    candidate,
    reference,
    threshold_mm: float,
) -> list[str]:
    """我检查三角化分辨率是否足以支持当前缺失阈值。

    为什么这条检查是硬性的：线性三角化偏差是网格与真实曲面之间的
    固有误差。偏差 0.3 mm 的网格去查 0.15 mm 的缺失，
    "缺了多少"和"三角化差了多少"根本分不开——
    报出来的区域可能完全是离散化伪影。这属于证据不足以支撑结论，
    所以我把它升级成人工复核，而不是继续给出 pass/fail。

    两个网格的角度偏差不一致时同样要报：不一致意味着两边的
    离散化程度不同，即使各自都满足阈值，双向差值也会被
    离散化差异污染。

    元数据缺失时跳过而不报警：STL 输入本来就没有三角化偏差
    这个概念（它直接就是网格），报警会淹没真正的问题。
    """

    warnings: list[str] = []
    for role, mesh in (("待检测", candidate), ("参考", reference)):
        raw_deflection = mesh.metadata.get("linear_deflection_mm")
        if raw_deflection is None:
            continue
        try:
            deflection = float(raw_deflection)
        except (TypeError, ValueError):
            continue
        if deflection > threshold_mm:
            warnings.append(
                f"我发现{role}网格线性三角化偏差 {deflection:g} mm "
                f"大于检测阈值 {threshold_mm:g} mm，当前结果必须人工复核。"
            )

    candidate_angle = candidate.metadata.get("angular_deflection_deg")
    reference_angle = reference.metadata.get("angular_deflection_deg")
    if candidate_angle is not None and reference_angle is not None:
        try:
            if abs(float(candidate_angle) - float(reference_angle)) > 1e-9:
                warnings.append(
                    "我发现待检测和参考网格使用了不同的角度三角化偏差，"
                    "请用一致参数重新网格化后再验收。"
                )
        except (TypeError, ValueError):
            pass
    return warnings


def _mesh_quality_warnings(
    candidate_quality: dict[str, object],
    reference_quality: dict[str, object],
) -> list[str]:
    """我把会影响几何可信度的输入质量问题集中升级为人工复核。"""

    warnings: list[str] = []
    checks = (
        ("degenerate_triangle_count", "退化三角形"),
        ("duplicate_triangle_count", "重复三角面"),
        ("non_manifold_edge_count", "非流形边"),
        ("isolated_vertex_count", "孤立顶点"),
    )
    for role, report in (("待检测", candidate_quality), ("参考", reference_quality)):
        for key, label in checks:
            count = int(report.get(key, 0))
            if count:
                warnings.append(
                    f"我发现{role}网格存在 {count} 个{label}，"
                   "当前几何判定必须人工复核。"
                )
    return warnings


def _threshold_warnings(
    recommendation: ThresholdRecommendation,
) -> list[str]:
    """我把阈值与模型尺度、配准残差的关系升级成可复核警告。

    两个警告针对两类不同的失败，不能合并：

    1. 配置阈值低于推荐阈值（只在 absolute 模式下报）。
       小件配大阈值会把整件判成破损——零件的正常尺度起伏
       本身就超过了阈值。这条提示用户改用 auto 模式。
       auto 模式不报是因为它本来就是按尺度算出来的，
       "低于推荐"这个判断对它没有意义。

    2. 配准残差超过包围盒对角线的 5%。这是比第一条更严重的
       信号：残差这么大说明两个模型根本没对齐，
       此时无论阈值怎么设都不可靠——不是阈值的问题，
       是姿态/单位/采样的问题。所以它给出的建议是
       "先修正姿态、单位或采样"，而不是调阈值。
    """

    warnings: list[str] = []
    if (
        recommendation.mode == "absolute"
        and recommendation.configured_below_recommended
    ):
        warnings.append(
            f"我方发现配置阈值 {recommendation.configured_mm:g} mm 低于按尺度与"
            f"配准残差推导的推荐阈值 {recommendation.recommended_mm:g} mm"
            f"（包围盒对角线 {recommendation.bbox_diagonal_mm:g} mm，"
            f"平均残差 {recommendation.registration_mean_residual_mm:g} mm）；"
            "当前绝对阈值可能把整件误判成破损，建议改用 threshold_mode=auto。"
        )
    if recommendation.scale_too_coarse:
        warnings.append(
            f"我方发现配准平均残差 "
            f"{recommendation.registration_mean_residual_mm:g} mm 超过包围盒"
            f"对角线 {recommendation.bbox_diagonal_mm:g} mm 的 5%，"
            "当前配准精度不足以支撑可靠检测，请先修正姿态、单位或采样。"
        )
    return warnings
