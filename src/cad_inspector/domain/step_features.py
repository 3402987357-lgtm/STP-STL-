"""我方 STEP 面区域分类结果使用的稳定数据模型。

三层结构由粗到细，每层都保留可追溯的证据字段：

- ``StepFeatureGroup``：按方向角色（顶面、底面、侧壁、倾斜等）聚成的一类 Face；
- ``StepFaceAdjacency``：单个 Face 通过共享 Edge 连到哪些邻居，带边界边数与法向夹角；
- ``StepFeatureRegion``：同一角色内按共享 Edge 连通分量聚成的区域，带边界拓扑、主
  Wire 选择、投影面积与环关系证据。

区域层的拓扑字段（``boundary_topology_status`` / ``boundary_wire_selection_status`` /
``primary_boundary_loop_relation_status``）全部默认置为"未解析"而不是"正常"，因为拿不到
证据时默认为正常是最危险的默认值——它会让下游以为边界是闭合的。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..reproducibility import with_content_digest


@dataclass
class StepFeatureGroup:
    """我记录一组具有相同几何角色的 Face 候选。"""

    role: str
    face_ids: list[int]
    area_mm2: float
    mean_center_xyz_mm: list[float] | None
    mean_normal_xyz: list[float] | None
    score: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """我把方向分组记录转成字典。"""
        return asdict(self)


@dataclass
class StepFaceAdjacency:
    """我记录一个 Face 通过共享 Edge 连接到哪些相邻 Face。"""

    face_id: int
    neighbor_face_ids: list[int]
    shared_edge_count: int
    boundary_edge_count: int
    shared_edge_ids: list[int] = field(default_factory=list)
    normal_angle_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """我把邻接记录转成字典。"""
        return asdict(self)


@dataclass
class StepFeatureRegion:
    """我记录同一几何角色内由共享 Edge 连通的 Face 区域。"""

    region_id: int
    role: str
    face_ids: list[int]
    # 身份与量测段：编号、角色、面集合、面积与中心法向。
    # center/normal 允许 None——退化或取不到几何时如实留空，
    # 用零向量填充会让下游误以为"中心就在原点"。
    area_mm2: float
    mean_center_xyz_mm: list[float] | None
    mean_normal_xyz: list[float] | None
    surface_type_counts: dict[str, int]
    neighbor_region_ids: list[int]
    # 边界段：从"有多少条边界边"到"主边界环的角色与关系"。
    # 全部字段都给缺省值，保证区域在没有边界数据时也能被构造出来，
    # 状态字段用 UNRESOLVED/NO_BOUNDARY/UNAVAILABLE 明确表达"未算过"，
    # 而不是默认成"无问题"。
    boundary_edge_count: int
    semantic_status: str
    score: float
    semantic_evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    boundary_edge_ids: list[int] = field(default_factory=list)
    region_geometry_signature: str | None = None
    boundary_edge_components: list[list[int]] = field(default_factory=list)
    boundary_loop_count: int = 0
    closed_boundary_loop_count: int = 0
    boundary_topology_status: str = "UNRESOLVED_ENDPOINTS"
    boundary_topology_evidence: dict[str, Any] = field(default_factory=dict)
    primary_boundary_edge_ids: list[int] = field(default_factory=list)
    boundary_wire_candidate_count: int = 0
    boundary_wire_selection_status: str = "NO_BOUNDARY"
    boundary_wire_selection_evidence: dict[str, Any] = field(default_factory=dict)
    # 周长与投影面积允许 None：长度缺失时给 0.0 会被当成"周长为零"，
    # 与"计算过且为零"无法区分，所以这里必须是三态。
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
        """我把区域记录转成字典。"""
        return asdict(self)


@dataclass
class StepFeatureClassificationReport:
    """我记录一次 STEP 面区域候选识别和邻接分析。"""

    status: str
    file_path: str
    unit: str | None
    backend: str
    shape_is_valid: bool | None
    face_count: int
    bbox_min_xyz_mm: list[float] | None
    bbox_max_xyz_mm: list[float] | None
    groups: list[StepFeatureGroup]
    adjacency: list[StepFaceAdjacency]
    regions: list[StepFeatureRegion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """我把完整分类报告转成带 schema 版本和内容摘要的字典。"""
        return with_content_digest({
            "schema_version": "step-feature-classification-1.11",
            "status": self.status,
            "file_path": self.file_path,
            "unit": self.unit,
            "backend": self.backend,
            "shape_is_valid": self.shape_is_valid,
            "face_count": self.face_count,
            "bbox_min_xyz_mm": self.bbox_min_xyz_mm,
            "bbox_max_xyz_mm": self.bbox_max_xyz_mm,
            "groups": [group.to_dict() for group in self.groups],
            "adjacency": [item.to_dict() for item in self.adjacency],
            "regions": [region.to_dict() for region in self.regions],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "diagnostics": dict(self.diagnostics),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        })
