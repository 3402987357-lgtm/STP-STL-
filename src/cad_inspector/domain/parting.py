"""我方分型设计区域使用的稳定数据模型。

我把网格基线、后续 OCC B-Rep 后端和可能的 NX/Parasolid 适配器统一成
同一组可序列化结果。这样我方可以先验证流程和验收口径，再替换几何后端。

算法原理：分型有两种后端（网格基线 / OCP B-Rep），精度差很多，但**结果模型只有一套**。
这样设计的好处是验收口径统一：不管是哪条路线，都要求分型线闭合、状态 BASELINE，否则
降级 REVIEW_REQUIRED。换后端时报告 schema 不变，甲方已有的比对脚本和历史结论都能续用。

配置里的两个容差必须分清：``endpoint_tolerance_mm`` 是点重合判据（1e-6 mm 量级，用于
端点匹配与去重），``section_tolerance_mm`` 是截交容差（吸收 B-Rep 求交的微小误差）。
把两者合成一个参数会让"想放宽截交容差"变成"把点全部判为重合"，直接毁掉环的遍历。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from numbers import Integral, Real
from typing import Any


@dataclass
class PartingConfig:
    """我把分型线和分型面的关键参数集中保存。

    为什么把这些参数集中成一个配置对象而不是散成函数参数：
    分型识别的每个环节都依赖同一组容差（端点容差、截面容差、
    扫面数量）。散开传很容易出现"调用 A 用 1e-6、调用 B 用 1e-3"
    这种不一致，而分型线对容差极其敏感——同一批数据在不同容差下
    会得出不同的线。集中配置还能直接 asdict 做配置指纹，
    让报告能回答"这条线是用哪套参数算出来的"。

    几个容易用错的字段，说明它们各自的单位与含义边界：
      - plane_margin_ratio 是**比例**不是长度：它按包围盒尺寸
        缩放，所以大小件能用同一套默认值。上界取 0.5（不含）
        是因为超过一半就等于把扫描范围推到包围盒之外；
      - endpoint_tolerance_mm 与 section_tolerance_mm 是长度，
        必须严格大于 0：等于 0 在浮点比较上等于"永不闭合"，
        会让所有环都被判成开放环；
      - surface_margin_mm 与 minimum_line_area_mm2 可以为 0，
        0 表示"不额外外扩 / 不过滤微小环"，是有意义的取值；
      - minimum_line_area_mm2 存在的理由：截面运算在相切位置
        会产出面积近乎为零的退化环，它们不是分型线，
        不按面积过滤会让结果里混进上百条噪声线。
    """

    plane_z_mm: float | None = None
    plane_scan_count: int = 9
    plane_margin_ratio: float = 0.08
    endpoint_tolerance_mm: float = 1e-6
    min_loop_points: int = 4
    smoothing_iterations: int = 0
    surface_margin_mm: float = 5.0
    allow_open_loop: bool = False
    minimum_line_area_mm2: float = 1e-9
    section_tolerance_mm: float = 1e-6
    curve_samples_per_edge: int = 8
    transition_offset_mm: float | None = None

    def __post_init__(self) -> None:
        """我在构造时校验参数，而不是等算到一半才失败。

        为什么放在构造期校验：分型识别一趟要跑几十秒到几分钟。
        如果 plane_margin_ratio 传了 0.8，跑到扫描那一步才发现
        范围越界，浪费的是几分钟的算力与一次排查；构造期抛错
        则是立刻、带参数名、可定位。

        容差类字段必须严格大于 0 而不是 >= 0：0 在浮点比较上
        意味着"两个点必须完全相同才算闭合"，实际几何数据里
        几乎不可能成立，会让所有环都被判成开放环。

        允许为 0 的是 margin 与面积下限这类"强度"参数：
        0 有明确含义（不外扩、不过滤），不是无效输入。
        """
        _require_int(self.plane_scan_count, "plane_scan_count", 1)
        _require_int(self.min_loop_points, "min_loop_points", 4)
        _require_int(self.smoothing_iterations, "smoothing_iterations", 0)
        _require_int(self.curve_samples_per_edge, "curve_samples_per_edge", 2)
        for name, value in (
            ("plane_z_mm", self.plane_z_mm),
            ("plane_margin_ratio", self.plane_margin_ratio),
            ("endpoint_tolerance_mm", self.endpoint_tolerance_mm),
            ("surface_margin_mm", self.surface_margin_mm),
            ("minimum_line_area_mm2", self.minimum_line_area_mm2),
            ("section_tolerance_mm", self.section_tolerance_mm),
            ("transition_offset_mm", self.transition_offset_mm),
        ):
            if value is not None:
                _require_finite_real(value, name)
        if not 0.0 <= float(self.plane_margin_ratio) < 0.5:
            raise ValueError("我要求 plane_margin_ratio 位于 [0, 0.5) 区间。")
        if float(self.endpoint_tolerance_mm) <= 0:
            raise ValueError("我要求 endpoint_tolerance_mm 必须大于 0。")
        if float(self.surface_margin_mm) < 0:
            raise ValueError("我要求 surface_margin_mm 不能小于 0。")
        if float(self.minimum_line_area_mm2) < 0:
            raise ValueError("我要求 minimum_line_area_mm2 不能小于 0。")
        if float(self.section_tolerance_mm) <= 0:
            raise ValueError("我要求 section_tolerance_mm 必须大于 0。")
        if (
            self.transition_offset_mm is not None
            and float(self.transition_offset_mm) <= 0
        ):
            raise ValueError("我要求 transition_offset_mm 必须大于 0。")

    def to_dict(self) -> dict[str, Any]:
        """我把分型配置转成字典，供报告与配置指纹使用。"""
        return asdict(self)


def _require_finite_real(value: Any, name: str) -> float:
    """我在分型配置边界拒绝布尔值、字符串和非有限数值。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"我要求 {name} 必须是有限数值。")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"我要求 {name} 必须是有限数值。")
    return numeric


def _require_int(value: Any, name: str, minimum: int) -> int:
    """我要求分型计数参数是整数，避免底层 range 使用非法类型。"""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"我要求 {name} 必须是整数。")
    if int(value) < minimum:
        raise ValueError(f"我要求 {name} 至少为 {minimum}。")
    return int(value)


@dataclass
class PartingLineResult:
    """我记录一次分型线候选的几何结果和可追溯诊断。"""

    status: str
    method: str
    plane_z_mm: float
    points_xyz_mm: list[list[float]]
    closed: bool
    length_mm: float
    area_xy_mm2: float
    score: float
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """我把分型线结果转成字典。"""
        return asdict(self)


@dataclass
class PartingSurfaceResult:
    """我记录平面分型面的网格基线结果。

    boundary_points_xyz_mm 与 vertices_xyz_mm 要分开保存：
    前者是原始边界点（供核验"这条边是不是真的贴着零件轮廓"），
    后者是外扩后用于生成网格的顶点。合成一个字段会让人无法判断
    看到的坐标是原始轮廓还是已外扩的结果。

    margin_mm 必须随结果一起输出：分型面是外扩出来的，
    不知道外扩了多少就无法判断这块面是否超出模仁范围，
    也无法复现同一块面。
    """

    status: str
    method: str
    plane_z_mm: float
    normal_xyz: list[float]
    boundary_points_xyz_mm: list[list[float]]
    vertices_xyz_mm: list[list[float]]
    faces: list[list[int]]
    margin_mm: float
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """我把分型面结果转成字典。"""
        return asdict(self)


@dataclass
class PartingPipelineResult:
    """我把分型线和分型面放在一次任务结果中。

    line 与 surface 都允许为 None：分型线算不出来时（例如零件
    没有明显的分型高度）不应该连带把整份结果变成失败，
    status 字段负责表达这次任务的整体结论，
    用 None 表达"这一项没算出来"比抛异常更贴近下游需要。

    line_candidates 要保留全部候选而不只留最优：分型线常常
    有多个合理高度（一模多腔、阶梯分型），只留最优会让复核者
    无从判断最优是不是选错的。

    config 随结果一起存档：同一份输入在不同参数下会得出不同的线，
    没有配置指纹就无法复现，也无法解释两次结果为何不同。
    """

    status: str
    input_file: str
    backend: str
    line: PartingLineResult | None
    surface: PartingSurfaceResult | None
    line_candidates: list[PartingLineResult] = field(default_factory=list)
    input_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    config: PartingConfig = field(default_factory=PartingConfig)

    def to_dict(self) -> dict[str, Any]:
        """我把分型任务完整结果转成带 schema 版本的可序列化字典。"""
        return {
            "schema_version": "parting-1.5",
            "status": self.status,
            "input_file": self.input_file,
            "backend": self.backend,
            "input_metadata": dict(self.input_metadata),
            "line": self.line.to_dict() if self.line else None,
            "line_candidates": [
                candidate.to_dict() for candidate in self.line_candidates
            ],
            "surface": self.surface.to_dict() if self.surface else None,
            "warnings": list(self.warnings),
            "config": self.config.to_dict(),
        }
