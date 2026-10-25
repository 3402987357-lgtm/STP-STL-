"""我把缺失区域和 STEP 工程区域做可选、可审计的关联。

原理说明：距离检测只能告诉我"参考件的这一块面在候选件上找不到"，
但回答不了"这块面在工程语义上是什么"。要回答后者，我需要另一条链路：
STEP 工程特征识别会给每个 STEP Face 一个工程区域（台阶/孔/R角…）和
类别候选；如果缺失区域的三角面能回溯到 STEP Face，就能把这条语义接上去。

这条链路的关键是"每一步都可能断，断了必须说出来"：

1. 工程区域 JSON 可能读不出来、schema 不对、regions 不是数组
   -> 记成 invalid，关联结果降级为人工复核线索，绝不假装关联成功。
2. region_id 可能重复、face_ids 可能含非法编号
   -> 逐条记 warning 并跳过，不静默覆盖前一条记录。
3. 同一个 STEP Face 可能被多个工程区域引用
   -> 记 warning 并要求复核，因为"一个面属于两个区域"通常意味着上游有问题。
4. 缺失区域的 Face 可能压根找不到对应工程区域
   -> 计入 unmapped 计数，并把该区域标成 review_required。

我故意把"关联"设计成可选步骤（两个 map 都传 None 就是关闭），
因为工程区域 JSON 不是每次都有；没有它的时候缺失区域检测本身仍然要能跑。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from ..models import RegionResult


@dataclass(frozen=True)
class StepEngineeringRegionMap:
    """我保存一个工程区域 JSON 的最小回溯索引。"""

    # 源文件绝对路径，用于报告里写清"这份关联来自哪个文件"。
    path: str
    # 源 JSON 的 schema_version / status，原样透传，不做解释。
    schema_version: str | None
    status: str | None
    # 核心索引：STEP Face 编号 -> 引用它的工程区域编号（已排序、去重）。
    face_to_region_ids: dict[int, tuple[int, ...]]
    # 工程区域编号 -> 它的类别候选字符串（已排序、去重）。
    region_categories: dict[int, tuple[str, ...]]
    # valid=False 表示这份索引本身有问题，任何基于它的关联都只能当线索。
    valid: bool
    warnings: tuple[str, ...] = ()


def load_step_engineering_region_map(
    path: str | Path,
) -> StepEngineeringRegionMap:
    """我读取工程区域 JSON，只保留区域回溯所需的稳定字段。"""

    resolved = str(Path(path).expanduser().resolve())
    try:
        payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        # 读文件失败不是异常：我返回一个 valid=False 的映射，
        # 让调用方拿到"有关联请求但关联不可用"这个明确状态。
        return _invalid_map(
            resolved,
            f"我无法读取 STEP 工程区域 JSON：{exc}",
        )
    # 顶层必须是对象，数组/标量都不是合法的工程区域报告。
    if not isinstance(payload, dict):
        return _invalid_map(resolved, "我要求 STEP 工程区域 JSON 顶层必须是对象。")
    schema_version = payload.get("schema_version")
    status = payload.get("status")
    # 我按 schema 前缀判断"这是不是我认识的工程特征报告"，
    # 版本号后面还会升，所以我只卡前缀不卡完整字符串。
    if (
        not isinstance(schema_version, str)
        or not schema_version.startswith("step-engineering-features-")
    ):
        return _invalid_map(
            resolved,
            "我发现输入文件不是可识别的 STEP 工程特征 schema。",
            schema_version=str(schema_version) if schema_version is not None else None,
            status=str(status) if status is not None else None,
        )
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, list):
        return _invalid_map(
            resolved,
            "我要求 STEP 工程区域 JSON 包含 regions 数组。",
            schema_version=schema_version,
            status=str(status) if status is not None else None,
        )

    face_to_regions: dict[int, set[int]] = {}
    region_categories: dict[int, tuple[str, ...]] = {}
    warnings: list[str] = []
    seen_region_ids: set[int] = set()
    for item in raw_regions:
        if not isinstance(item, dict):
            warnings.append("我跳过了不是对象的区域记录。")
            continue
        region_id = _positive_int(item.get("region_id"))
        if region_id is None:
            warnings.append("我跳过了缺少正整数 region_id 的区域记录。")
            continue
        # 重复 region_id 我拒绝"后者覆盖前者"：那属于静默数据丢失，
        # 会让甲方看到的类别候选取决于 JSON 里哪条写在后面。
        if region_id in seen_region_ids:
            warnings.append(
                f"我发现重复的区域编号 {region_id}，"
                "不能把后续记录静默覆盖到前一条记录。"
            )
            continue
        seen_region_ids.add(region_id)
        raw_faces = item.get("face_ids", [])
        if not isinstance(raw_faces, list):
            warnings.append(f"我发现区域 {region_id} 的 face_ids 不是数组。")
            continue
        # 区域没有任何 Face 时我仍然登记它（类别可能还有用），但记一条 warning。
        if not raw_faces:
            warnings.append(f"我发现区域 {region_id} 没有可回溯的 Face。")
        face_ids: list[int] = []
        for value in raw_faces:
            face_id = _positive_int(value)
            if face_id is None:
                # 非法 Face 编号我逐条记 warning 并跳过，不让它进索引。
                warnings.append(
                    f"我发现区域 {region_id} 含有非法 STEP Face 编号。"
                )
                continue
            face_ids.append(face_id)
            # 反向索引：Face -> 引用它的所有区域。用 set 是因为
            # 同一个区域里同一个 Face 可能被列了两次。
            face_to_regions.setdefault(face_id, set()).add(region_id)
        raw_categories = item.get("category_candidates", [])
        if not isinstance(raw_categories, list):
            warnings.append(
                f"我发现区域 {region_id} 的 category_candidates 不是数组。"
            )
        categories = tuple(
            sorted(
                {
                    value.strip()
                    for value in raw_categories
                    if isinstance(value, str) and value.strip()
                }
            )
        ) if isinstance(raw_categories, list) else ()
        region_categories[region_id] = categories

    duplicate_face_count = sum(
        len(region_ids) > 1 for region_ids in face_to_regions.values()
    )
    # 一个 Face 被多个工程区域引用 = 上游区域划分重叠，关联结论不可全信。
    if duplicate_face_count:
        warnings.append(
            f"我发现 {duplicate_face_count} 个 STEP Face 被多个工程区域引用，"
            "区域关联必须人工复核。"
        )
    # 只有"至少解析出一个区域，且一条 warning 都没有"才算 valid。
    # 我故意这么严：valid 决定的是"能不能自动采信"，不是"能不能用"。
    valid = bool(region_categories) and not warnings
    return StepEngineeringRegionMap(
        path=resolved,
        schema_version=schema_version,
        status=str(status) if status is not None else None,
        face_to_region_ids={
            face_id: tuple(sorted(region_ids))
            for face_id, region_ids in sorted(face_to_regions.items())
        },
        region_categories=region_categories,
        valid=valid,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def join_step_engineering_regions(
    regions: list[RegionResult],
    *,
    reference_map: StepEngineeringRegionMap | None = None,
    candidate_map: StepEngineeringRegionMap | None = None,
) -> tuple[dict[str, Any], list[str], bool]:
    """我把缺失/多余区域关联到 STEP 工程区域和类别候选。

    返回值是 (指标字典, 警告列表, 是否需要人工复核)。
    我在这里直接改写传入的 RegionResult（补上 step_region_ids 与类别候选），
    因为调用方拿到的就是同一批对象；指标字典只做统计，不重复存数据。
    """

    warnings: list[str] = []
    review_required = False
    # 我把指标结构一次性初始化完整（包括没启用的字段），
    # 这样报告里的字段形状稳定，甲方写解析脚本不用做存在性判断。
    metrics: dict[str, Any] = {
        "enabled": reference_map is not None or candidate_map is not None,
        "reference": _map_summary(reference_map),
        "candidate": _map_summary(candidate_map),
        "region_count": len(regions),
        "mapped_region_count": 0,
        "unmapped_face_reference_count": 0,
        "unmapped_face_candidate_count": 0,
        "method": "step_face_to_engineering_region_join",
    }
    for label, region_map in (
        ("参考", reference_map),
        ("待检测", candidate_map),
    ):
        if region_map is None:
            continue
        # 索引本身的 warning 直接上浮；valid=False 会让整批关联降级成线索。
        warnings.extend(region_map.warnings)
        if not region_map.valid:
            review_required = True
            warnings.append(
                f"我发现{label} STEP 工程区域索引不完整，关联结果只能作为人工复核线索。"
            )

    for region in regions:
        # 参考侧用 region.reference_step_face_ids（缺失区域对应的参考面），
        # 候选侧用 region.candidate_step_face_ids（多余区域对应的候选面）。
        reference_ids, reference_categories, reference_unmapped = _lookup(
            region.reference_step_face_ids,
            reference_map,
        )
        candidate_ids, candidate_categories, candidate_unmapped = _lookup(
            region.candidate_step_face_ids,
            candidate_map,
        )
        region.reference_step_region_ids = reference_ids
        region.candidate_step_region_ids = candidate_ids
        region.reference_step_category_candidates = reference_categories
        region.candidate_step_category_candidates = candidate_categories
        # 累计"有 Face 但没有对应工程区域"的数量，它是关联完整性的直接指标。
        if reference_unmapped:
            metrics["unmapped_face_reference_count"] += reference_unmapped
        if candidate_unmapped:
            metrics["unmapped_face_candidate_count"] += candidate_unmapped
        if (
            # 只有当"确实提供了 map、区域也确实有 STEP Face、但一个区域都没查到"
            # 时才算断链。没有 map 属于功能没开，不是断链。
            (reference_map is not None and region.reference_step_face_ids and not reference_ids)
            or (
                candidate_map is not None
                and region.candidate_step_face_ids
                and not candidate_ids
            )
        ):
            region.review_required = True
            review_required = True
            region.warnings.append(
                "我没有把该区域的 STEP Face 回溯到工程区域，当前关联结果需要人工复核。"
            )
        if reference_ids or candidate_ids:
            metrics["mapped_region_count"] += 1
    if metrics["unmapped_face_reference_count"] or metrics[
        "unmapped_face_candidate_count"
    ]:
        review_required = True
        warnings.append(
            "我发现部分缺失区域 Face 没有对应工程区域，不能把类别候选当作完整语义。"
        )
    return metrics, list(dict.fromkeys(warnings)), review_required


def _lookup(
    face_ids: Iterable[int],
    region_map: StepEngineeringRegionMap | None,
) -> tuple[list[int], list[str], int]:
    """我把一组 STEP Face 编号查成 (区域编号, 类别候选, 未命中数)。

    区域编号和类别候选我都排序去重后返回，保证同一份输入结果稳定。
    region_map 为 None 表示这个方向没启用关联，我返回全空并把未命中数记 0，
    因为"没启用"和"启用了但查不到"必须区分开。
    """

    if region_map is None:
        return [], [], 0
    region_ids: set[int] = set()
    categories: set[str] = set()
    unmapped = 0
    for face_id in face_ids:
        mapped = region_map.face_to_region_ids.get(int(face_id))
        if not mapped:
            # 这个 Face 不在任何工程区域里，计入未命中。
            unmapped += 1
            continue
        region_ids.update(mapped)
        for region_id in mapped:
            # 通过区域编号再取类别候选，形成 Face -> 区域 -> 类别 的两跳回溯。
            categories.update(region_map.region_categories.get(region_id, ()))
    return sorted(region_ids), sorted(categories), unmapped


def _map_summary(
    region_map: StepEngineeringRegionMap | None,
) -> dict[str, Any]:
    """我把一份工程区域索引压成报告用的摘要字段。"""

    if region_map is None:
        return {"provided": False}
    return {
        "provided": True,
        "path": region_map.path,
        "schema_version": region_map.schema_version,
        "status": region_map.status,
        "valid": region_map.valid,
        "face_count": len(region_map.face_to_region_ids),
        "region_count": len(region_map.region_categories),
        "warning_count": len(region_map.warnings),
    }


def _invalid_map(
    path: str,
    warning: str,
    *,
    schema_version: str | None = None,
    status: str | None = None,
) -> StepEngineeringRegionMap:
    """我构造一份"读取失败"的映射，保留 path/schema/status 便于排查。"""

    return StepEngineeringRegionMap(
        path=path,
        schema_version=schema_version,
        status=status,
        face_to_region_ids={},
        region_categories={},
        valid=False,
        warnings=(warning,),
    )


def _positive_int(value: object) -> int | None:
    """我只接受正整数；bool、浮点、字符串一律返回 None。

    这里我故意不接受 "12" 这种数字字符串，也不接受 12.0 这种浮点：
    编号必须是 JSON 里的整数，宽松解析会把上游的格式错误藏起来。
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    else:
        return None
    if number <= 0:
        return None
    return number
