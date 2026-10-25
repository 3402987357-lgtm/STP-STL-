"""我记录模具装配体 BOM 基线结果。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from ..reproducibility import with_content_digest


# 甲方任务书"模具零件类型分类"给出的 8 个部件大类，顺序与任务书一致。
# 我方内部规则先产出带"候选"字样的几何/名称证据标签，再映射到这 8 个大类，
# 既保留"候选、需复核"的诚实语义，又让输出能对上任务书的大类口径。
PART_CATEGORY_GROUPS: tuple[str, ...] = (
    "成型零件",
    "模架系统零件",
    "导向定位零件",
    "顶出系统零件",
    "侧向抽芯零件",
    "标准紧固件",
    "冷却浇注零件",
    "辅助结构零件",
)

_CANDIDATE_TO_PART_CATEGORY_GROUP: dict[str, str] = {
    "成型件候选": "成型零件",
    "模架件候选": "模架系统零件",
    "导向件候选": "导向定位零件",
    "顶出件候选": "顶出系统零件",
    "侧向抽芯件候选": "侧向抽芯零件",
    "紧固件候选": "标准紧固件",
    "冷却浇注件候选": "冷却浇注零件",
    "辅助结构件候选": "辅助结构零件",
}


# 只能靠**名称**命中、几何规则永远不会给出候选的大类。
# 我为什么必须把它写死成一个常量：甲方看到某大类计数为 0 时，
# 会读成"这套模具里没有这类零件"。但这三个大类的 0 是**工具识别不到**，
# 不是零件不存在——尤其"成型零件"（型腔/型芯）是一套注塑模的定义性零件，
# 它报 0 几乎必然是漏识别。把这条边界显式记下来，
# 报告才能把"确实没有"和"我们认不出来"分开说，而不是都写成 0。
NAME_ONLY_PART_CATEGORY_GROUPS: frozenset[str] = frozenset(
    {"成型零件", "侧向抽芯零件", "冷却浇注零件"}
)


def canonical_part_category(candidate_labels: Sequence[str]) -> str | None:
    """我把内部候选标签映射成任务书的 8 个部件大类。

    只有所有候选标签都指向同一个大类时才返回该大类；出现跨大类候选
    （例如"板件/结构件候选"无法区分模架件与辅助结构件）时返回 None，
    由调用方继续保留"需复核"，不硬凑一个结论。
    """

    groups = {
        _CANDIDATE_TO_PART_CATEGORY_GROUP[label]
        for label in candidate_labels
        if label in _CANDIDATE_TO_PART_CATEGORY_GROUP
    }
    if len(groups) == 1:
        return groups.pop()
    return None


def canonical_part_category_from_candidates(
    candidates: Sequence[dict[str, Any]],
) -> tuple[str | None, str]:
    """我按证据强度选出部件大类，并返回 (大类, 强度)。

    我为什么不能沿用 ``canonical_part_category`` 那种"所有候选求交集"：

    实测 510 个零部件里，唯一有强证据的 11 件（名称明确叫
    ``Cap Screw`` / ``Ejector`` / ``Runner System``）**全部**被判成了
    None。原因是名称规则给出"紧固件候选"的同时，几何规则会无条件
    补上"板件/结构件候选""模架件候选""辅助结构件候选"三个弱候选，
    四者跨大类，交集为空。结果就是：证据最强的那批件反而一个大类
    都拿不到，而 158 个纯几何兜底件因为候选同源反倒能归上——
    恰好把可信度搞反了。

    我改成按证据来源分档，只在**同一档内**求交集：

    - ``named``：主类别来自 NAME_RULE（名称或主数据），最可信；
    - ``agreed``：名称与几何同时指向主类别，次之；
    - ``geometry``：只有几何证据，最弱——这时才要求所有同类档候选
      指向同一大类，否则返回 None 保留"需复核"。

    强度一并返回，让调用方能把"这条大类结论有多硬"写进输出，
    而不是让 0.2 置信度的兜底结论和名称命中的结论看起来一样。
    """

    if not candidates:
        return None, "none"
    primary = candidates[0]
    primary_category = str(primary.get("category", ""))
    group = _CANDIDATE_TO_PART_CATEGORY_GROUP.get(primary_category)
    if group is None:
        # 主类别本身就对不上 8 大类（例如"板件/结构件候选"是个
        # 跨大类的中间标签），没有强证据可依据。
        return None, "none"
    sources = {str(source) for source in primary.get("sources", [])}
    if "NAME_RULE" in sources and "GEOMETRY_RULE" in sources:
        return group, "agreed"
    if "NAME_RULE" in sources or "MASTER_DATA" in sources:
        return group, "named"
    # 只有几何证据：要求同一档内不出现跨大类，否则保留待复核。
    geometry_groups = {
        _CANDIDATE_TO_PART_CATEGORY_GROUP[str(item.get("category", ""))]
        for item in candidates
        if str(item.get("category", "")) in _CANDIDATE_TO_PART_CATEGORY_GROUP
    }
    if len(geometry_groups) == 1:
        return group, "geometry"
    return None, "none"


@dataclass
class BomComponent:
    """我记录一个装配体零部件的可追溯几何摘要。

    字段分成三组，各回答一个问题：
      - 身份与追溯：component_id、reference_shape_id、occurrence_path
        和 occurrence_ids/paths/locations，回答"这是哪个零件、装在哪里"；
      - 分类与可信度：category 与 category_candidates/confidence/
        source/evidence/review_required 并列，回答"我判它是什么、能不能信"；
      - 几何量：bbox、dimensions、各拓扑计数、表面积体积，
        回答"它多大"。

    occurrence_count 与 occurrence_ids 并存：数量是给 BOM 汇总用的，
    编号列表是给追溯用的。只留数量就无法定位具体装在哪，
    只留列表就每次都要 len()，且容易被误当成"编号即数量"。

    part_category_group 允许为 None：跨大类候选时我不硬凑结论，
    None 表达"待人工判定"，与空字符串区分开。
    """

    component_id: str
    reference_shape_id: str
    occurrence_path: str
    # 出现 ID 列表、路径列表、位姿列表三者下标一一对应：
    # 同一个零件被装了 N 次，这三个列表就都是 N 项，
    # 拆开看其中任何一个都会丢掉"第 i 处装在哪"的信息。
    occurrence_ids: list[str]
    occurrence_paths: list[str]
    occurrence_locations: list[dict[str, Any] | None]
    occurrence_count: int
    part_name: str
    category: str
    category_confidence: float
    category_source: str
    category_evidence: list[str]
    category_candidates: list[dict[str, Any]]
    category_review_required: bool
    category_review_reason_codes: list[str]
    part_category_group: str | None
    # 材料与采购规格都允许为 None，但各自的来源字段必填：
    # 值为 None 时来源说明"为什么没有"，
    # 两者缺一就无法区分"模型里没标"和"没读出来"。
    material: str | None
    material_source: str
    quantity: int
    purchase_specification: str | None
    purchase_specification_source: str
    remarks: str
    # 包围盒三件套都允许为 None：量不出来就如实留空，
    # 给零向量会被读成"这个零件退化到了原点"。
    bbox_min_xyz_mm: list[float] | None
    bbox_max_xyz_mm: list[float] | None
    dimensions_xyz_mm: list[float] | None
    # 五类拓扑计数：它们既是复杂度的度量，
    # 也是"这个件是否取全了几何"的依据。
    solid_count: int
    shell_count: int
    face_count: int
    edge_count: int
    vertex_count: int
    # 面积与体积都允许为 None：非实体（例如只有面或线）
    # 量不出体积，留 None 比给 0 更准确。
    surface_area_mm2: float | None
    volume_mm3: float | None
    # 大类结论的证据强度：named / agreed / geometry / none。
    # 我单独存这个字段而不是让调用方猜：同样是"辅助结构零件"，
    # 名称命中的和纯几何兜底的可信度完全不同，
    # 混在一起会让甲方把 0.2 置信度的兜底当结论采信。
    part_category_group_strength: str = "none"
    # 警告带默认空列表：绝大多数零件没有警告，
    # 不设默认值会让每个构造点都得显式传一个。
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """我把这条 BOM 行转成字典，键名与 CSV 表头一致。"""

        return asdict(self)


@dataclass
class BomReport:
    """我记录一次装配体解析、分类和 BOM 导出基线。

    warnings 与 errors 分开存放：warnings 是"结果可用但要留意"
    （例如某条候选需复核），errors 是"这一项没做出来"。
    混在一起会让调用方无法判断这批 BOM 能不能直接用。

    status 只描述**整体**是否可用，单个零件的问题落在各自组件的
    category_review_required 上。两级状态并存，
    避免"一个零件待复核"把整份 BOM 判成不可用，
    也避免"整份不可用"被淹没在逐条警告里。

    to_dict 里带 content digest：BOM 是要交付的正式产物，
    带摘要才能让甲方验证"收到的这份和我跑出来的是同一份"。
    """

    status: str
    file_path: str
    unit: str | None
    backend: str
    component_count: int
    components: list[BomComponent]
    input_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """我把整份 BOM 报告转成带 schema 版本与内容摘要的字典。"""

        # schema 升到 1.9：BomComponent 新增 part_category_group_strength，
        # CSV/XLSX 新增"大类证据强度"列。
        return with_content_digest({
            "schema_version": "mold-bom-1.9",
            "status": self.status,
            "file_path": self.file_path,
            "unit": self.unit,
            "backend": self.backend,
            "component_count": self.component_count,
            "components": [component.to_dict() for component in self.components],
            "input_metadata": dict(self.input_metadata),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "diagnostics": dict(self.diagnostics),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        })


def reconcile_bom_completeness(
    components: Sequence[BomComponent],
    assembly_occurrence_count: int,
    expected_occurrence_slots: int | None = None,
) -> dict[str, Any]:
    """我把"不遗漏任何零件"变成一条**可对账**的不变量。

    我为什么要单独做这件事：甲方要求"不遗漏任何零件"，但此前这条要求
    没有对应的判据——BOM 少算几个零件不会报错，甲方也看不出来。
    我在核对真实产物（mould_04）时确实发现它对不上：

        装配树实例数 156，而"数量"列合计 159（多了 3）

    根因是**多 Solid 拆分**：一个装配节点含 3 个 Solid 时会被拆成 3 行，
    这 3 行原本各自继承了原节点那 1 个 occurrence，于是同一个实例被计了
    3 次——同一个 occurrence id 在 3 行里重复出现，**没有任何判别依据**，
    看 BOM 表也发现不了。采购按这个数下单就会多买 3 件。

    我把账拆成两条口径，各自对一次，都对上才算不重不漏：

    - **覆盖口径（有没有漏）**：把 occurrence id 上的 `#solid_NNN` 派生
      后缀剥掉再取集合，集合大小 == 装配树实例数。
      剥后缀是因为拆分出来的行本来就指向同一个装配实例；
      不剥的话拆分会把"覆盖数"抬高，反而把真正的漏计掩盖掉。
    - **件数口径（有没有重复计）**：各行 quantity 之和 ==
      **预期槽位数**。预期槽位数 = 每个计量单元的实例数之和，
      拆分会让一个装配实例展开成 N 个 Solid 行，所以预期值
      通常**大于**装配树实例数，这个差值是拆分造成的、是设计使然，
      不是错误。用装配树实例数去比件数，等于拿节点数比零件数，
      永远不会相等，那条判据也就永远在报假警。

    两条分开是有意义的：只查覆盖会放过"重复计"，只查件数会放过"漏计"。
    两条一起才等价于"不重不漏"。

    我返回结构化的差异证据（多出/缺失的 occurrence id），
    而不是只回一个 bool——对账对不上时，甲方需要知道**差的是哪几个**，
    否则没法查。
    """

    seen: list[str] = []
    for component in components:
        for occurrence_id in component.occurrence_ids or []:
            seen.append(str(occurrence_id))
    unique_ids = set(seen)

    # 覆盖口径：剥掉拆分派生后缀，还原成"装配树里的那一个实例"。
    base_ids = {item.split("#solid_")[0] for item in seen}

    total_quantity = sum(int(component.quantity or 0) for component in components)
    expected_slots = (
        assembly_occurrence_count
        if expected_occurrence_slots is None
        else int(expected_occurrence_slots)
    )

    coverage_ok = len(base_ids) == assembly_occurrence_count
    quantity_ok = total_quantity == expected_slots
    # 拆分展开量：BOM 行数比装配节点多出来的那部分，是设计使然，如实记出。
    split_expansion_delta = expected_slots - assembly_occurrence_count

    if not coverage_ok and not quantity_ok:
        status = "FAIL"
        reason = (
            "既存在实例覆盖差异，也存在数量重复计："
            f"覆盖实例 {len(base_ids)}、数量合计 {total_quantity}、"
            f"装配树实例 {assembly_occurrence_count}、预期槽位 {expected_slots}"
        )
    elif not coverage_ok:
        status = "FAIL"
        reason = (
            f"实例覆盖对不上：覆盖实例 {len(base_ids)}，"
            f"装配树实例 {assembly_occurrence_count}（差值 "
            f"{len(base_ids) - assembly_occurrence_count}）"
        )
    elif not quantity_ok:
        status = "FAIL"
        reason = (
            f"数量合计对不上：合计 {total_quantity}，"
            f"预期槽位 {expected_slots}（差值 "
            f"{total_quantity - expected_slots}）。这个差值就是会被"
            "重复采购或漏采购的件数。"
        )
    else:
        status = "PASS"
        reason = (
            f"覆盖实例 {len(base_ids)} 与装配树实例数一致（不漏），"
            f"数量合计 {total_quantity} 与预期槽位一致（不重）；"
            f"其中拆分展开 {split_expansion_delta} 件，属设计使然。"
        )

    return {
        "status": status,
        "reason": reason,
        "assembly_occurrence_count": assembly_occurrence_count,
        "covered_occurrence_count": len(base_ids),
        "expected_occurrence_slots": expected_slots,
        "split_expansion_delta": split_expansion_delta,
        "occurrence_slot_count": len(seen),
        # 完全重复的 id（未剥后缀前）：出现即说明同一实例被计了多次。
        "duplicated_occurrence_count": len(seen) - len(unique_ids),
        "total_quantity": total_quantity,
        "coverage_ok": coverage_ok,
        "quantity_ok": quantity_ok,
        # 完全重复的 id 列表：对不上时甲方要能查"到底是哪几个被算了两次"，
        # 只给一个计数没法定位。
        "duplicated_occurrence_ids": sorted(
            {item for item in seen if seen.count(item) > 1}
        )[:50],
        "component_count": len(components),
    }


def summarize_category_coverage(
    components: Sequence[BomComponent],
) -> dict[str, Any]:
    """我逐个大类统计命中数，并把每个 0 的**原因**分开写清楚。

    我为什么必须做这一步：甲方任务书要求按 8 个大类分类。实测真实数据上
    有 4 个大类恒为 0，但"0"有两种完全不同的含义，此前报告把它们混在一起：

      - **工具认不出**：该大类只有名称规则能命中，而这份 STEP 的零件名是
        哈希串（实测 341 个未分类件里大量形如 ``52561FBC24A736ws``），
        没有任何可匹配的关键词。这时 0 不代表模具里没有这类零件。
      - **几何歧义**：几何规则确实给出了该大类的候选，但同时给出了
        跨大类的竞争候选，交集为空，所以不硬下结论。

    这两种 0 对甲方的含义天差地别：前者是**我方能力的缺口**，后者是
    **数据本身的歧义**。都写成 0，甲方会把"成型零件 0 件"当成"这套模具
    没有型腔型芯"——而那是任何一套注塑模都必然有的零件。

    所以我为每个 0 记一条原因，并在报告里显式提示哪几个 0 属于能力缺口。
    """

    counts: dict[str, int] = {group: 0 for group in PART_CATEGORY_GROUPS}
    unclassified = 0
    for component in components:
        group = component.part_category_group
        if group in counts:
            counts[group] += 1
        else:
            unclassified += 1

    zero_groups: list[str] = []
    zero_reasons: dict[str, str] = {}
    for group in PART_CATEGORY_GROUPS:
        if counts[group]:
            continue
        zero_groups.append(group)
        if group in NAME_ONLY_PART_CATEGORY_GROUPS:
            zero_reasons[group] = (
                "该大类只有名称规则能识别，本文件零件名没有对应关键词，"
                "0 命中**不代表模具里没有这类零件**，需要名称或主数据才能判定。"
            )
        else:
            zero_reasons[group] = (
                "几何规则给出了该大类候选，但同时存在跨大类的竞争候选，"
                "交集为空，我方不硬下结论，需人工或主数据复核。"
            )

    return {
        "counts": counts,
        "unclassified_count": unclassified,
        "zero_groups": zero_groups,
        # 这些 0 是能力缺口，不是"确实没有"；单独列出来防止被读成否定结论。
        "unreachable_groups": [
            group for group in zero_groups if group in NAME_ONLY_PART_CATEGORY_GROUPS
        ],
        "zero_reasons": zero_reasons,
    }
