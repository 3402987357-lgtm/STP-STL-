"""我记录 STEP 模具工程特征的可解释候选证据。

设计原则：这个模块只放数据契约，不放算法。每个 dataclass 都是"候选 + 证据 + 警告"
三件套，因为模具工程特征本质上没有一个是能靠几何单独判死的：台阶、孔、R 角、分型
面全都依赖装配语义与工艺规则。所以每条记录都必须能回答三个问题——

1. 结论是什么（``kind`` / ``classification`` / ``role_candidates`` / ``status``）；
2. 凭什么这么说（``evidence`` / ``score`` / ``confidence`` 里的原始量化信号）；
3. 还缺什么（``warnings`` / ``review_reason_codes`` / ``review_priority``）。

``review_priority`` 与 ``review_reason_codes`` 让甲方能按优先级排队复核，而不是面对
几百条候选无从下手。``to_dict`` 带 schema 版本与内容摘要，保证报告口径可追溯。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..reproducibility import with_content_digest


@dataclass
class StepEngineeringCandidate:
    """我记录一个台阶、圆柱或 R 角候选。"""

    candidate_id: int
    kind: str
    face_ids: list[int]
    edge_ids: list[int]
    score: float
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """我把候选记录转成字典。"""
        return asdict(self)


@dataclass
class StepDraftEvidence:
    """我记录单个 Face 相对指定拔模方向的角度证据。"""

    face_id: int
    direction_xyz: list[float]
    normal_vs_direction_angle_deg: float | None
    plane_vs_direction_angle_deg: float | None
    classification: str
    score: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """我把拔模证据记录转成字典。"""
        return asdict(self)


@dataclass
class StepEngineeringRegion:
    """我记录一个区域的工程角色、语义证据和候选特征关联。

    这个 dataclass 字段很多，但分组是刻意的，不是堆出来的：
      - 结论层：source_role / role_candidates / category_candidates；
      - 证据层：evidence / category_evidence / parting_evidence
        以及 boundary_* 一族量化信号，全部保留原始数值而不是只留
        判定结果，甲方复核时才能自己判断"这个结论站不站得住"；
      - 缺口层：warnings / review_reason_codes / review_priority。

    为什么默认 category_review_required 与 parting_review_required
    都是 True（而不是 False）：工程特征无法靠几何单独判死，
    默认"需要人工确认"意味着漏标一处只会多一次复核，
    默认"已确认"则意味着误判会被当成结论直接采信。
    这个方向的错误代价不对称，所以默认取保守的一侧。

    各类 status 字段的默认值刻意是 "UNRESOLVED_*" / "UNAVAILABLE"
    而非 ""：空字符串在下游 JSON 里看起来像"已处理"，
    而 UNRESOLVED 明确表示"这一步没能给出结论"。
    """

    region_id: int
    source_role: str
    face_ids: list[int]
    neighbor_region_ids: list[int]
    area_mm2: float
    boundary_edge_count: int
    # 中心、法向、包围盒都允许 None：量不出来就如实留空，
    # 给零向量会被下游当成"区域退化到原点"这种真实结论。
    mean_center_xyz_mm: list[float] | None
    mean_normal_xyz: list[float] | None
    bbox_min_xyz_mm: list[float] | None
    bbox_max_xyz_mm: list[float] | None
    semantic_status: str
    engineering_status: str
    role_candidates: list[str]
    candidate_ids: list[int]
    candidate_kinds: list[str]
    category_candidates: list[str]
    category_confidence: float
    category_evidence: dict[str, Any] = field(default_factory=dict)
    # 两个复核标记默认 True：拿不到证据时默认"需要人工看"，
    # 默认 False 会让判不出来的区域被当成已确认的结论流出去。
    category_review_required: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    review_reason_codes: list[str] = field(default_factory=list)
    # 优先级默认 "NONE" 而不是最紧急：没有原因码就没有优先级，
    # 默认给最高会让所有区域都排在最前面，等于没有排序。
    review_priority: str = "NONE"
    boundary_edge_ids: list[int] = field(default_factory=list)
    # 独立于面号的几何签名：面号随区域划分顺序变，
    # 跨批次追踪"同一个区域"只能靠它。
    region_geometry_signature: str | None = None
    parting_role_candidates: list[str] = field(default_factory=list)
    parting_confidence: float = 0.0
    parting_review_required: bool = True
    parting_evidence: dict[str, Any] = field(default_factory=dict)
    boundary_edge_components: list[list[int]] = field(default_factory=list)
    boundary_loop_count: int = 0
    closed_boundary_loop_count: int = 0
    # 拓扑与 Wire 选择状态的默认值都是"没判出来"而不是"没问题"：
    # 没跑过判定与判定为良好是两回事，
    # 默认给良好会让未计算的字段看起来像已通过。
    boundary_topology_status: str = "UNRESOLVED_ENDPOINTS"
    boundary_topology_evidence: dict[str, Any] = field(default_factory=dict)
    primary_boundary_edge_ids: list[int] = field(default_factory=list)
    boundary_wire_candidate_count: int = 0
    boundary_wire_selection_status: str = "NO_BOUNDARY"
    boundary_wire_selection_evidence: dict[str, Any] = field(default_factory=dict)
    # 周长留 None 而不是 0.0：周长 0 是一个可能的真实值（退化环），
    # 用 0.0 表示"没量"会与之混淆。
    boundary_perimeter_length_mm: float | None = None
    primary_boundary_perimeter_length_mm: float | None = None
    primary_boundary_projected_area_mm2: float | None = None
    primary_boundary_projection_status: str = "UNAVAILABLE"
    primary_boundary_projection_evidence: dict[str, Any] = field(default_factory=dict)
    primary_boundary_loop_role: str = "UNAVAILABLE"
    primary_boundary_loop_relation_status: str = "UNAVAILABLE"
    primary_boundary_loop_relation_evidence: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """我把区域工程记录转成字典。"""
        return asdict(self)


@dataclass
class StepEngineeringReport:
    """我记录一次模具工程特征候选识别结果。

    shape_is_valid 允许是 None 而不只是 True/False：OCP 的有效性检查
    在极端几何上可能抛异常而拿不到结果。用 None 表示"没查出来"，
    与 False 表示"查出来是无效的"区分开——这两者对下游的含义完全不同。

    warnings 与 errors 分开存放：warning 是"结论可用但需要留意"，
    error 是"这一步没跑完"。合并成一个列表会让调用方无法判断
    报告里的候选是否完整。
    """

    status: str
    file_path: str
    unit: str | None
    backend: str
    shape_is_valid: bool | None
    face_count: int
    bbox_min_xyz_mm: list[float] | None
    bbox_max_xyz_mm: list[float] | None
    candidates: list[StepEngineeringCandidate]
    draft_evidence: list[StepDraftEvidence]
    regions: list[StepEngineeringRegion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    rule_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """我把完整工程特征报告转成带 schema 版本和内容摘要的字典。

        schema_version 硬编码在报告里而不是由调用方传入：甲方归档的就是
        这个文件，版本号必须随数据一起走，否则半年后拿到一份 JSON
        无法判断该用哪套字段解读。

        elapsed_seconds 保留 6 位小数而不是取整：小样本上耗时常在
        毫秒量级，取整会全部变成 0，性能回归就看不出来了。
        """
        return with_content_digest({
            "schema_version": "step-engineering-features-2.5",
            "status": self.status,
            "file_path": self.file_path,
            "unit": self.unit,
            "backend": self.backend,
            "shape_is_valid": self.shape_is_valid,
            "face_count": self.face_count,
            "bbox_min_xyz_mm": self.bbox_min_xyz_mm,
            "bbox_max_xyz_mm": self.bbox_max_xyz_mm,
            "candidates": [item.to_dict() for item in self.candidates],
            "draft_evidence": [item.to_dict() for item in self.draft_evidence],
            "regions": [item.to_dict() for item in self.regions],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "diagnostics": dict(self.diagnostics),
            "rule_summary": dict(self.rule_summary),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        })
