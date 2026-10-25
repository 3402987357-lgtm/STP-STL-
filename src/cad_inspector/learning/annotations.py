"""我用严格 JSON 契约承载真实标注，并在训练前阻止数据泄漏。

算法原理：真实标注是训练与验收的信任边界，一旦脏数据进来，后面所有指标都不可信。
所以我在"读入即校验"阶段做四件事：

1. 结构校验：顶层必须是带正确 schema 版本的 JSON 对象，records 非空，每条记录的
   sample_id / split / reference_file / candidate_file 都是非空字符串。
2. 泄漏校验：同一份 CAD 文件（按大小写归一、路径归一后比较）不允许跨 train/test
   重复出现，否则测试集等于见过训练数据，准确率会虚高。
3. 自比校验：参考文件和候选文件不能是同一个文件，否则天然"没有缺失区域"，是假通过。
4. 数值校验：Face/Edge 编号是严格非负整数（拒绝布尔与浮点），坐标与包围盒是有限
   三维向量，包围盒下界不得大于上界，面积是非负有限数，区域内编号不得重复。

我为什么在 1.1 里把"真值面号落在哪一侧"写进 schema：缺料的真值在**参考网格**
上（参考件有、破损件没有），多余材料的真值在**候选网格**上（破损件多出来）。
旧版只有 reference_face_ids 一个面号字段，于是多余材料永远无法被打分——
检测器每件都会报 extra 区域、评估器也支持 extra，只有"人工确认 -> 打分"
这一环断着（详见 annotation_scoring.py 的模块说明）。1.1 增加
candidate_face_ids 并把"extra 必须给候选面号、且不许同时给参考面号"
写成硬约束。同时我保留对 1.0 的读取：已交付的审计材料不能因为升版就变非法。

校验失败直接抛错而不是降级，宁可拒绝数据也不放过泄漏样本。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import ntpath
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "missing-region-annotations-1.1"
# 我保留 1.0 的读取路径：reports/draft_annotations.json 就是 1.0 的存量草稿，
# 它是已交付的审计材料。升版把它判成非法，等于顺手烧掉既有证据。
_LEGACY_SCHEMA_VERSIONS = ("missing-region-annotations-1.0",)
_ACCEPTED_SCHEMA_VERSIONS = (SCHEMA_VERSION, *_LEGACY_SCHEMA_VERSIONS)
_ALLOWED_SPLITS = {"train", "validation", "test"}
_REQUIRED_SPLITS = {"train", "test"}
# provenance 标明这条标注的来源：confirmed 是人工确认过的真值，
# draft 是机器预测自动生成的待复核草稿。
_ALLOWED_PROVENANCE = {"confirmed", "draft"}
_ALLOWED_DEFECT_TYPES = {"missing", "extra"}
# 每种缺陷的真值面号落在哪一侧：缺料看参考网格，多余材料看候选网格。
# 这一条是 extra 能被打分的前提——打分器按类型选字段，不许混着用。
_TRUTH_FACE_FIELD = {
    "missing": "reference_face_ids",
    "extra": "candidate_face_ids",
}


@dataclass(frozen=True)
class AnnotationManifest:
    """我保存已通过边界校验的标注清单摘要。"""

    schema_version: str
    record_count: int
    split_counts: dict[str, int]
    region_count: int
    missing_region_count: int
    extra_region_count: int
    missing_splits: list[str]
    path_overlap_count: int
    ready_for_training: bool
    confirmed_record_count: int
    draft_record_count: int

    def to_dict(self) -> dict[str, object]:
        """我把标注摘要转成可序列化字典，供报告与审计直接使用。"""
        return asdict(self)


def load_annotation_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], AnnotationManifest]:
    """我读取并校验标注 JSON，返回原始数据和可审计摘要。

    我同时返回**原始 payload 和校验摘要**两份数据，而不是只返回校验后的
    结果。理由：摘要只承载统计结论，训练时仍需要原始记录里的
    区域坐标和编号；只返回摘要会丢掉训练真正要用的东西。

    expanduser() 不能省：标注清单里的路径常写成 "~/data/..."，
    不展开会直接报文件不存在，而错误信息看起来像"清单有问题"，
    实际只是没展开家目录。
    """

    source = Path(path).expanduser()
    payload = json.loads(source.read_text(encoding="utf-8"))
    return payload, validate_annotation_manifest(payload)


def validate_annotation_manifest(payload: object) -> AnnotationManifest:
    """我拒绝非法编号、非有限坐标、重复样本和跨 split 文件泄漏。"""

    if not isinstance(payload, dict):
        raise ValueError("我要求标注清单顶层必须是 JSON 对象。")
    # 我先记下**文件自己的**版本而不是一律归一成当前版本：
    # 1.0 里没有 candidate_face_ids 这个字段，"extra 必须给候选面号"
    # 就只能对 1.1 生效。把版本抹平会让存量草稿突然变成非法数据。
    raw_version = payload.get("schema_version")
    if raw_version not in _ACCEPTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"我只接受标注 schema {list(_ACCEPTED_SCHEMA_VERSIONS)}，"
            f"当前收到 {raw_version!r}。"
        )
    schema_version = str(raw_version)
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("我要求 records 是非空 JSON 数组。")

    # 下面这组累加器在遍历过程中统计 split 分布、区域数、缺陷类型和路径重叠。
    sample_ids: set[str] = set()
    path_splits: dict[str, str] = {}
    split_counts = {split: 0 for split in sorted(_ALLOWED_SPLITS)}
    region_count = 0
    missing_region_count = 0
    extra_region_count = 0
    path_overlap_count = 0
    # 我为什么区分"已确认"和"草稿"：
    # 标注清单有两条完全不同的来源——人工确认过的真值，
    # 和机器预测自动生成的草稿。拿草稿当真值去评估生成它的那个
    # 预测器，等于自己给自己打分，会得到 100% 的假准确率。
    # 这个闸门因此必须知道清单里有多少条是草稿。
    confirmed_record_count = 0
    draft_record_count = 0

    # 逐条记录校验；任何一条不合法都立即抛错，并带上出错下标方便定位。
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"我要求 records[{index}] 必须是 JSON 对象。")
        sample_id = _required_string(raw_record, "sample_id", index)
        if sample_id in sample_ids:
            raise ValueError(f"我发现重复 sample_id：{sample_id}")
        sample_ids.add(sample_id)
        split = _required_string(raw_record, "split", index)
        if split not in _ALLOWED_SPLITS:
            raise ValueError(
                f"我不认识 records[{index}].split：{split}"
            )
        # provenance 决定这条能不能用于训练：draft 是机器预测，
        # 只有人工确认过（confirmed）才能当真值。
        provenance = str(raw_record.get("provenance") or "confirmed")
        if provenance not in _ALLOWED_PROVENANCE:
            raise ValueError(
                f"我不认识 records[{index}].provenance：{provenance}"
                f"（只允许 {sorted(_ALLOWED_PROVENANCE)}）"
            )
        if provenance == "draft":
            draft_record_count += 1
        else:
            confirmed_record_count += 1
        split_counts[split] += 1
        reference_file = _required_string(raw_record, "reference_file", index)
        candidate_file = _required_string(raw_record, "candidate_file", index)
        if _normalized_path(reference_file) == _normalized_path(candidate_file):
            raise ValueError(
                f"我发现 records[{index}] 的参考文件和候选文件是同一个文件："
                f"{reference_file}。"
                "同一文件同时充当参考和候选只能得到“没有缺失区域”的假通过，"
                "我拒绝把这种记录当成训练标注。"
            )
        # 路径泄漏检查：同一路径若已属于别的 split，直接判定训练数据泄漏。
        for role, file_path in (
            ("参考", reference_file),
            ("候选", candidate_file),
        ):
            path_key = _normalized_path(file_path)
            previous_split = path_splits.get(path_key)
            if previous_split is not None and previous_split != split:
                path_overlap_count += 1
                raise ValueError(
                    f"我发现同一{role} CAD 文件跨 split 重复出现，"
                    "这会造成训练数据泄漏。"
                )
            path_splits[path_key] = split

        raw_regions = raw_record.get("regions", [])
        if not isinstance(raw_regions, list):
            raise ValueError(f"我要求 records[{index}].regions 必须是数组。")
        for region_index, region in enumerate(raw_regions):
            _validate_region(
                region, index, region_index, schema_version=schema_version
            )
            region_count += 1
            if region["defect_type"] == "missing":
                missing_region_count += 1
            else:
                extra_region_count += 1

    # 缺少 train 或 test 时 ready_for_training 为假，不允许开始训练。
    # ready_for_training 必须同时满足三个条件：train/test 都不为空、
    # 没有跨 split 路径泄漏、且区域总数大于 0。只满足其中一部分就放行，
    # 会让"只有 train 没有 test"的清单看起来可以训练，
    # 实际上跑出来的指标无从验证。
    # 注意 path_overlap_count 在上方遇到泄漏时已经抛错，这里保留为 0，
    # 双重保险：即便将来改成只记数不抛错，这个闸门也不会误放行。
    missing_splits = [
        split for split in sorted(_ALLOWED_SPLITS) if split_counts[split] == 0
    ]
    return AnnotationManifest(
        schema_version=schema_version,
        record_count=len(raw_records),
        split_counts=split_counts,
        region_count=region_count,
        missing_region_count=missing_region_count,
        extra_region_count=extra_region_count,
        missing_splits=missing_splits,
        path_overlap_count=path_overlap_count,
        confirmed_record_count=confirmed_record_count,
        draft_record_count=draft_record_count,
        ready_for_training=(
            _REQUIRED_SPLITS.isdisjoint(
                split for split, count in split_counts.items() if count == 0
            )
            and path_overlap_count == 0
            and region_count > 0
            # 我为什么加"草稿必须为 0"这一条：
            # 上面三条只看格式和划分，看不出记录是人工真值还是机器预测。
            # 我这一轮刚加了"把验收结果转成标注草稿"的脚本，
            # 草稿能顺利通过前三条闸门并拿到 READY_FOR_TRAINING，
            # 于是机器预测可以悄悄流进训练——这是假准确率的直通车。
            and draft_record_count == 0
        ),
    )


def _validate_region(
    region: object,
    record_index: int,
    region_index: int,
    *,
    schema_version: str,
) -> None:
    """我校验单个缺失/多余区域的编号、坐标、包围盒和面积字段。

    ``schema_version`` 决定"真值面号"这一条的严格程度：1.1 起，
    extra 区域必须给候选网格面号、且不许同时给参考面号；1.0 的存量
    草稿按老规矩读（那时只有 reference_face_ids 一个面号字段）。
    """
    if not isinstance(region, dict):
        # 报错信息带上完整下标路径（第几条记录的第几个区域）：
        # 标注文件通常有几十条记录，只说"格式不对"
        # 等于让人自己去找是哪一条。
        raise ValueError(
            f"我要求 records[{record_index}].regions[{region_index}] "
            "必须是 JSON 对象。"
        )
    defect_type = region.get("defect_type")
    if defect_type not in _ALLOWED_DEFECT_TYPES:
        raise ValueError("我只接受 missing 或 extra 缺陷类型。")
    face_ids = region.get("reference_face_ids", [])
    if not isinstance(face_ids, list):
        raise ValueError("我要求 reference_face_ids 必须是数组。")
    # 逐条严格解析：JSON 里混进字符串数字或布尔值时，
    # 不逐个校验会让它们悄悄通过。
    parsed_face_ids = [_strict_nonnegative_int(value) for value in face_ids]
    # 面号不允许重复：同一个面在一块区域里出现两次
    # 会让面积统计重复计算，也会让"这块区域包含哪些面"含糊不清。
    if len(set(parsed_face_ids)) != len(parsed_face_ids):
        raise ValueError("我拒绝区域内重复的 reference_face_ids。")
    # 候选网格面号：多余材料的真值字段（1.1 起）。1.0 的文件里没有这个
    # 字段，所以"缺失即报错"和"必须非空"两条都要按版本分开。
    candidate_face_ids = region.get("candidate_face_ids", [])
    if not isinstance(candidate_face_ids, list):
        raise ValueError("我要求 candidate_face_ids 必须是数组。")
    parsed_candidate_ids = [
        _strict_nonnegative_int(value) for value in candidate_face_ids
    ]
    if len(set(parsed_candidate_ids)) != len(parsed_candidate_ids):
        raise ValueError("我拒绝区域内重复的 candidate_face_ids。")
    if schema_version == SCHEMA_VERSION and defect_type == "extra":
        if not parsed_candidate_ids:
            raise ValueError(
                "我要求 extra 区域给出 candidate_face_ids：多余材料的真值"
                "只存在于候选网格上，缺了它这一类结论无法被人工确认后打分。"
            )
        if parsed_face_ids:
            raise ValueError(
                "我拒绝给 extra 区域填 reference_face_ids：多出来的料"
                "在参考件上并不存在，两侧同时给会让真值口径自相矛盾。"
            )
    for key in ("center_xyz_mm",):
        if key in region and region[key] is not None:
            # 可选字段只在"提供了"时才校验：
            # 缺失是允许的（标注可以只标一部分），
            # 提供了却格式错则必须报错。
            _finite_vector(region[key], key)
    bbox_min = region.get("bbox_min_xyz_mm")
    bbox_max = region.get("bbox_max_xyz_mm")
    # 包围盒两端必须同时给：只有一端的话无法判断
    # 是"不标包围盒"还是"漏填了一半"。
    if (bbox_min is None) != (bbox_max is None):
        raise ValueError(
            "我要求 bbox_min_xyz_mm 和 bbox_max_xyz_mm 必须同时提供。"
        )
    if bbox_min is not None and bbox_max is not None:
        _finite_vector(bbox_min, "bbox_min_xyz_mm")
        _finite_vector(bbox_max, "bbox_max_xyz_mm")
        # 逐轴检查下界不大于上界：包围盒反了说明标注填反了，
        # 不拦住的话后面算出的体积会是负数。
        if any(
            float(lower) > float(upper)
            for lower, upper in zip(bbox_min, bbox_max)
        ):
            raise ValueError(
                "我要求 bbox_min_xyz_mm 的每个坐标不能大于 bbox_max_xyz_mm。"
            )
    if "area_mm2" in region and region["area_mm2"] is not None:
        area = region["area_mm2"]
        # 布尔值要单独排除：bool 是 int 的子类，
        # 只查 isinstance(area, (int, float)) 会把 True 当成面积 1。
        if (
            isinstance(area, bool)
            or not isinstance(area, (int, float))
            or not math.isfinite(float(area))
            or float(area) < 0.0
        ):
            raise ValueError("我要求 area_mm2 是有限非负数。")
    edge_ids = region.get("boundary_edge_ids", [])
    if not isinstance(edge_ids, list):
        raise ValueError("我要求 boundary_edge_ids 必须是数组。")
    parsed_edge_ids = [_strict_nonnegative_int(value) for value in edge_ids]
    if len(set(parsed_edge_ids)) != len(parsed_edge_ids):
        raise ValueError("我拒绝区域内重复的 boundary_edge_ids。")


def _required_string(record: dict[str, Any], key: str, index: int) -> str:
    """我读取必填字符串字段；缺失或全空白时报错并指明记录下标与字段名。"""
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"我要求 records[{index}].{key} 是非空字符串。")
    return value


def _strict_nonnegative_int(value: object) -> int:
    """我把输入校验为非负 JSON 整数，显式拒绝布尔值与浮点数。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            "我要求 Face/Edge 编号必须是 JSON 整数且不能小于 0。"
        )
    return value


def _finite_vector(value: object, key: str) -> None:
    """我校验输入是长度 3 的有限数值数组，否则抛出带字段名的错误。"""
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"我要求 {key} 是长度为 3 的有限数值数组。")


def _normalized_path(value: str) -> str:
    """我按 Windows 规则归一化路径（大小写、分隔符、相对段），用于泄漏比较。"""
    normalized = ntpath.normcase(ntpath.normpath(value.strip()))
    return normalized.replace("/", "\\")
