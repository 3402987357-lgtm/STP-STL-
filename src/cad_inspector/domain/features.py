"""我方几何特征识别区域使用的数据模型。

这里承载的是网格路线的特征识别结果：按法向把三角面分成水平、垂直、其他三类，再从
水平面里挑出顶平面与底平面候选。

``top_plane`` / ``bottom_plane`` 允许为 None，因为不是每个模型都有明确的顶底平面
（比如球形零件）。用 None 明确表达"没有找到"，而不是填一个假的面积为 0 的候选去凑
字段完整性——后者会让下游误以为"找到了但面积为零"。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PlaneCandidate:
    """我记录一个水平面候选的几何摘要。"""

    face_count: int
    area_mm2: float
    mean_z_mm: float
    min_z_mm: float
    max_z_mm: float
    normal_xyz: list[float]
    role: str

    def to_dict(self) -> dict[str, Any]:
        """我把平面候选转成字典。"""
        return asdict(self)


@dataclass
class MeshFeatureResult:
    """我把网格特征识别结果统一输出。"""

    status: str
    backend: str
    bbox_min_xyz_mm: list[float]
    bbox_max_xyz_mm: list[float]
    horizontal_face_count: int
    vertical_face_count: int
    other_face_count: int
    horizontal_area_mm2: float
    vertical_area_mm2: float
    top_plane: PlaneCandidate | None
    bottom_plane: PlaneCandidate | None
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """我把网格特征结果转成带 schema 版本的可序列化字典。"""
        return {
            "schema_version": "mesh-features-1.0",
            "status": self.status,
            "backend": self.backend,
            "bbox_min_xyz_mm": self.bbox_min_xyz_mm,
            "bbox_max_xyz_mm": self.bbox_max_xyz_mm,
            "horizontal_face_count": self.horizontal_face_count,
            "vertical_face_count": self.vertical_face_count,
            "other_face_count": self.other_face_count,
            "horizontal_area_mm2": self.horizontal_area_mm2,
            "vertical_area_mm2": self.vertical_area_mm2,
            "top_plane": self.top_plane.to_dict() if self.top_plane else None,
            "bottom_plane": self.bottom_plane.to_dict() if self.bottom_plane else None,
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
        }
