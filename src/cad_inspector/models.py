"""我定义项目各模块之间共享的数据结构，避免模块之间传递无约束字典。

设计原则：配置在构造时就校验，而不是等到跑算法时才发现参数非法。我把校验放在
``InspectionConfig.__post_init__``，对所有字段做类型与区间检查，另外：

- 拒绝布尔值混入数值字段（Python 里 True 是 int 的子类，不显式排除会被当成 1）；
- 拒绝非有限数值（NaN/Inf 会在后续距离计算里静默传播成全 0 结果）；
- 枚举型字段（注册模式、距离后端、阈值模式、单位策略）只允许白名单取值；
- ``from_dict`` 遇到不认识的键直接报错，而不是静默忽略——参数名写错后悄悄退回默认值
  是最难排查的一类问题。

结果结构（RegistrationResult / RegionResult / InspectionResult）只做数据承载，
``to_dict`` 里带上 schema 版本号，保证报告口径可追溯。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any


@dataclass
class InspectionConfig:
    """我把一个检测任务的关键参数集中管理，并保留可复现配置。

    为什么在构造时就校验而不是等算法报错：这些参数一旦非法，
    后果往往是**静默的错误结果**而不是崩溃——NaN 阈值会让所有距离
    比较返回 False，得到"零缺陷"的漂亮报告；单位策略写错会让
    毫米和英寸混算。跑到最后才发现，排查成本比构造时报错高一个量级。

    校验的四类重点（每一类都对应一个真实踩过的坑）：
      - 布尔字段先于数值字段校验。Python 里 True 是 int 的子类，
        不先排除就会把 `allow_scale=True` 当数值 1 通过检查；
      - 拒绝 NaN/Inf。它们会在后续距离计算里静默传播，把结果全变 0；
      - 枚举字段只认白名单，避免 "centered" 拼成 "centre" 后悄悄失效；
      - from_dict 遇到不认识的键直接报错。参数名写错后静默退回默认值
        是最难排查的一类问题，我宁可启动失败也不让它悄悄跑完。
    """

    threshold_mm: float = 0.15
    min_region_area_mm2: float = 0.5
    min_triangle_count: int = 1
    # 默认改成 auto：实测 21/21 个真实件的"原始位姿"残差都低于
    # centroid 和 pca（件本身就在同一坐标系，质心平移反而把表面推开）。
    # auto 在这三种里取残差最小者，残差恒 <= 固定的 centroid，
    # 所以它严格优于旧的固定默认值，不是另一种口径。
    registration_mode: str = "auto"
    allow_scale: bool = False
    samples_per_triangle: int = 1
    # 2026-10-25 接管续写新增（默认关闭，不改任何既有行为）：
    # 面级判据是否额外取该面的**三个顶点**参与"最大误差"归约。
    # 我为什么加它：任务书把缺失区域定义成"几何误差 ≥ 阈值的区域"，
    # 而"面上是否存在达阈值的点"取决于采样点落在面的哪里。只取重心时，
    # 一个重心距离 0.1mm、顶点距离 0.28mm 的面会被判成"没有误差"——
    # 实测（0.2mm 网格 / 0.15mm 阈值 / 删 24 面）真值 15 面里因此漏 7 面。
    # 代价是每面多 3 次距离查询；是否接受这个代价属于验收口径决策，
    # 所以我做成开关并把证据留档，而不是直接改默认值。
    face_corner_check: bool = False
    max_query_points: int = 200_000
    max_input_file_size_bytes: int = 2_000_000_000
    distance_backend: str = "auto"
    distance_candidate_k: int = 64
    distance_chunk_size: int = 2048
    bidirectional_check: bool = True
    directional_evidence_enabled: bool = True
    direction_normal_tolerance_deg: float = 30.0
    unit_policy: str = "assume_mm"
    mesh_linear_deflection_mm: float = 0.04
    mesh_angular_deflection_deg: float = 10.0
    write_intermediate: bool = False
    write_colored_mesh: bool = True
    threshold_mode: str = "absolute"
    auto_threshold_scale_ratio: float = 0.002
    auto_threshold_residual_factor: float = 1.5
    auto_threshold_floor_mm: float = 0.02

    def __post_init__(self) -> None:
        """我在构造后立刻校验全部字段，防止坏配置静默跑出坏结果。"""
        # 先校验布尔字段：bool 是 int 子类，不先排除会混进数值校验。
        for name in (
            "allow_scale",
            "bidirectional_check",
            "directional_evidence_enabled",
            "face_corner_check",
            "write_intermediate",
            "write_colored_mesh",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"我要求 {name} 必须是布尔值。")
        # 数值字段：必须是有限实数。
        numeric_values = {
            "threshold_mm": self.threshold_mm,
            "min_region_area_mm2": self.min_region_area_mm2,
            "mesh_linear_deflection_mm": self.mesh_linear_deflection_mm,
            "mesh_angular_deflection_deg": self.mesh_angular_deflection_deg,
            "direction_normal_tolerance_deg": (
                self.direction_normal_tolerance_deg
            ),
            "auto_threshold_scale_ratio": self.auto_threshold_scale_ratio,
            "auto_threshold_residual_factor": (
                self.auto_threshold_residual_factor
            ),
            "auto_threshold_floor_mm": self.auto_threshold_floor_mm,
        }
        for name, value in numeric_values.items():
            _require_finite_real(value, name)
        # 整数字段：计数与采样参数必须为正整数，避免浮点进入 range。
        _require_int(self.min_triangle_count, "min_triangle_count", 1)
        _require_int(self.samples_per_triangle, "samples_per_triangle", 1)
        _require_int(self.max_query_points, "max_query_points", 1)
        _require_int(
            self.max_input_file_size_bytes,
            "max_input_file_size_bytes",
            1,
        )
        _require_int(self.distance_candidate_k, "distance_candidate_k", 1)
        _require_int(self.distance_chunk_size, "distance_chunk_size", 1)
        # 阈值与面积下限：阈值必须为正，面积下限可以为 0（表示不按面积过滤）。
        if float(self.threshold_mm) <= 0:
            # 阈值必须为正：阈值 0 会让"完全相同"也被判成有差异，
            # 于是整份结果全是区域，等于没有判据。
            raise ValueError("我要求 threshold_mm 必须大于 0。")
        # 面积下限允许为 0：取 0 表示"不过滤小区域"，
        # 这是一个合法的显式选择，与阈值不同。
        if float(self.min_region_area_mm2) < 0:
            raise ValueError("我要求 min_region_area_mm2 不能小于 0。")
        _require_string(self.registration_mode, "registration_mode")
        if self.registration_mode not in {"auto", "none", "centroid", "pca", "icp"}:
            raise ValueError("我只支持 auto、none、centroid、pca 和 icp 五种注册模式。")
        _require_string(self.distance_backend, "distance_backend")
        # 后端白名单校验而不是让下游自己炸：
        # 拼错的后端名如果不在这里拦住，
        # 会在真正开始算距离时才失败，那时已经跑了很久。
        if self.distance_backend not in {"auto", "bruteforce", "scipy_kdtree"}:
            raise ValueError(
                "我只支持 auto、bruteforce 和 scipy_kdtree 三种距离后端。"
            )
        # 法向容差上限 90 度：超过 90 度"方向一致"就失去意义
        # （两个面夹角大于 90 度仍算同向），判据会退化。
        if not 0.0 < float(self.direction_normal_tolerance_deg) <= 90.0:
            raise ValueError(
                "我要求 direction_normal_tolerance_deg 位于 (0, 90] 区间。"
            )
        # 网格化两个步长都必须为正：它们是离散精度，
        # 取 0 会让 OCP 网格化死循环或产出海量三角形。
        if float(self.mesh_linear_deflection_mm) <= 0:
            raise ValueError("我要求 mesh_linear_deflection_mm 必须大于 0。")
        if float(self.mesh_angular_deflection_deg) <= 0:
            raise ValueError("我要求 mesh_angular_deflection_deg 必须大于 0。")
        _require_string(self.threshold_mode, "threshold_mode")
        if self.threshold_mode not in {"absolute", "auto"}:
            raise ValueError("我只支持 absolute 和 auto 两种阈值模式。")
        # 自动阈值的三个参数都必须为正：
        # 它们是"由尺度与配准残差推算阈值"的系数与下限，
        # 任一个取 0 会让推算出来的阈值变成 0 或负数。
        if float(self.auto_threshold_scale_ratio) <= 0:
            raise ValueError(
                "我要求 auto_threshold_scale_ratio 必须大于 0。"
            )
        if float(self.auto_threshold_residual_factor) <= 0:
            raise ValueError(
                "我要求 auto_threshold_residual_factor 必须大于 0。"
            )
        # 阈值下限必须为正：它是最后一道保险，
        # 保证即使前两项算得很小，阈值也不会塌到 0。
        if float(self.auto_threshold_floor_mm) <= 0:
            raise ValueError(
                "我要求 auto_threshold_floor_mm 必须大于 0。"
            )
        _require_string(self.unit_policy, "unit_policy")
        # 单位策略白名单：拼错的策略名如果不拦住，
        # 会被当成"未知"静默走默认分支，
        # 于是用户以为自己配了 require_mm 实际没生效。
        if self.unit_policy not in {"assume_mm", "unknown", "require_mm"}:
            raise ValueError(
                "我只支持 assume_mm、unknown 和 require_mm 三种单位策略。"
            )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "InspectionConfig":
        """我从 JSON 或外部配置字典构造检测参数。"""

        # 未知键直接报错，防止参数名拼错后静默退回默认值。
        known_fields = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known_fields)
        # 未知参数直接抛错而不是忽略：配置里把 threashold_mm 拼错时，
        # 静默忽略意味着用户以为改了阈值、实际一直在用默认值，
        # 这类错误只会在比对结果时才暴露，排查成本极高。
        # 排序后再拼接是为了让错误信息每次都一致。
        if unknown:
            raise ValueError(
                "我发现配置文件里有我不认识的参数："
                + "、".join(unknown)
                + "。我拒绝静默忽略，避免参数名写错后悄悄退回默认值。"
            )
        # 布尔字段一律走 _as_bool：YAML/JSON 里的 "false" 字符串在
        # Python 里是真值，直接 .get() 会让 --flag=false 变成 True。
        return cls(
            threshold_mm=values.get("threshold_mm", 0.15),
            min_region_area_mm2=values.get("min_region_area_mm2", 0.5),
            min_triangle_count=values.get("min_triangle_count", 1),
            registration_mode=values.get("registration_mode", "auto"),
            # 布尔字段统一走 _as_bool 而不是直接取：
            # YAML 里写 "false" 得到字符串 "false"，它是真值，
            # 直接取会让关掉的开关实际是开着的。
            allow_scale=_as_bool(values.get("allow_scale", False), "allow_scale"),
            samples_per_triangle=values.get("samples_per_triangle", 1),
            # 面级顶点采样：默认关，只有配置里显式写 true 才打开。
            # 走 _as_bool 的理由与其它布尔字段一致（"false" 字符串是真值）。
            face_corner_check=_as_bool(
                values.get("face_corner_check", False),
                "face_corner_check",
            ),
            # 采样上限与输入体积上限都是资源保护：
            # 前者限制计算量，后者限制读入的模型大小。
            # 两者都取很大的默认值——默认不限制用户，
            # 只在明确配置时才收紧。
            max_query_points=values.get("max_query_points", 200_000),
            max_input_file_size_bytes=values.get(
                "max_input_file_size_bytes",
                2_000_000_000,
            ),
            distance_backend=values.get("distance_backend", "auto"),
            distance_candidate_k=values.get("distance_candidate_k", 64),
            # 分块大小只影响内存峰值、不影响结果：
            # 它是纯性能参数，所以不需要与几何容差那样写进报告的判据说明。
            distance_chunk_size=values.get("distance_chunk_size", 2048),
            # 双向检查默认开：只做单向会把"多出"误判成"没差异"，
            # 关掉它必须是有意为之。
            bidirectional_check=_as_bool(
                values.get("bidirectional_check", True),
                "bidirectional_check",
            ),
            directional_evidence_enabled=_as_bool(
                values.get("directional_evidence_enabled", True),
                "directional_evidence_enabled",
            ),
            direction_normal_tolerance_deg=values.get(
                "direction_normal_tolerance_deg",
                30.0,
            ),
            # 单位策略默认"按毫米处理"：STEP 有单位声明但 STL 没有，
            # 混合输入时必须有一个明确的默认口径。
            unit_policy=values.get("unit_policy", "assume_mm"),
            mesh_linear_deflection_mm=values.get(
                "mesh_linear_deflection_mm",
                0.04,
            ),
            mesh_angular_deflection_deg=values.get(
                "mesh_angular_deflection_deg",
                10.0,
            ),
            write_intermediate=_as_bool(
                values.get("write_intermediate", False),
                "write_intermediate",
            ),
            write_colored_mesh=_as_bool(
                values.get("write_colored_mesh", True),
                "write_colored_mesh",
            ),
            threshold_mode=values.get("threshold_mode", "absolute"),
            auto_threshold_scale_ratio=values.get(
                "auto_threshold_scale_ratio",
                0.002,
            ),
            auto_threshold_residual_factor=values.get(
                "auto_threshold_residual_factor",
                1.5,
            ),
            auto_threshold_floor_mm=values.get(
                "auto_threshold_floor_mm",
                0.02,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """我把配置转成字典，供 JSON 输出与配置指纹使用。"""
        return asdict(self)


@dataclass
class RegistrationResult:
    """我记录候选模型到参考模型的变换和配准质量。

    配准是后面一切偏差判定的地基，所以我除了变换本身，还一定记录
    mean/p95_residual_mm 和 quality 这三个质量量。理由很直接：
    两个模型没对齐时，距离算法会忠实地报出一大片"缺陷"，
    而那片缺陷完全是配准误差造出来的。不把配准质量随结果一起输出，
    甲方看到的就只是"大面积缺失"，无从判断是真缺陷还是没对齐。

    scale_review_required 单独成一个字段：允许缩放配准时
    （allow_scale=True）算出的 scale 可能是在补偿单位错误而非真实缩放，
    这种情况必须显性标记，不能混在普通 warning 里被忽略。
    """

    mode: str
    rotation_matrix: list[list[float]]
    translation_mm: list[float]
    scale: float
    mean_residual_mm: float
    p95_residual_mm: float
    quality: str
    warnings: list[str] = field(default_factory=list)
    scale_review_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        """我把配准结果转成字典。"""
        return asdict(self)


@dataclass
class RegionResult:
    """我用这个结构描述一个连续的缺失候选区域。

    字段设计要点：我刻意把"是什么"和"能不能信"分开存放。
      - 几何量（area_mm2、max/mean/p95_error_mm、center/bbox）回答
        "这个区域多大、偏了多少"；
      - 可信度量（confidence、review_required、warnings、
        direction_evidence、direction_valid_sample_count）回答
        "这个结论能不能用"。
    两者混在一起会让下游只看面积的调用方误把待复核区域当定论。

    direction_* 三个字段是有向证据的统计：signed_offset_mean_mm 区分
    材料偏多还是偏少，normal_alignment_mean 描述法向一致性，
    *_sample_count 暴露样本量——样本量不足时前两个比值没有统计意义，
    所以我把分母也一起存下来，不允许只看比值下结论。
    """

    region_id: int
    defect_type: str
    center_xyz_mm: list[float]
    bbox_min_xyz_mm: list[float]
    bbox_max_xyz_mm: list[float]
    area_mm2: float
    estimated_volume_mm3: float | None
    max_error_mm: float
    mean_error_mm: float
    p95_error_mm: float
    triangle_count: int
    confidence: float
    reference_face_ids: list[int]
    review_required: bool
    region_geometry_signature: str | None = None
    candidate_face_ids: list[int] = field(default_factory=list)
    candidate_step_face_ids: list[int] = field(default_factory=list)
    reference_step_face_ids: list[int] = field(default_factory=list)
    candidate_step_region_ids: list[int] = field(default_factory=list)
    reference_step_region_ids: list[int] = field(default_factory=list)
    candidate_step_category_candidates: list[str] = field(default_factory=list)
    reference_step_category_candidates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    direction_evidence: str = "UNAVAILABLE"
    signed_offset_mean_mm: float | None = None
    normal_alignment_mean: float | None = None
    direction_valid_sample_count: int = 0
    direction_aligned_sample_count: int = 0
    direction_boundary_sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """我把单个区域结果转成字典。"""
        return asdict(self)


@dataclass
class InspectionResult:
    """我把一次完整检测的全局指标和区域结果统一输出。"""

    status: str
    candidate_file: str
    reference_file: str
    unit: str
    unit_source: str
    threshold_mm: float
    registration: RegistrationResult
    global_metrics: dict[str, Any]
    regions: list[RegionResult]
    artifacts: dict[str, str]
    warnings: list[str]
    config: InspectionConfig

    def to_dict(self) -> dict[str, Any]:
        """我把完整检测结果转成带 schema 版本的可序列化字典。"""
        return {
            "schema_version": "missing-region-detection-1.3",
            "status": self.status,
            "candidate_file": self.candidate_file,
            "reference_file": self.reference_file,
            "unit": self.unit,
            "unit_source": self.unit_source,
            "threshold_mm": self.threshold_mm,
            "registration": self.registration.to_dict(),
            "global_metrics": self.global_metrics,
            "regions": [region.to_dict() for region in self.regions],
            "artifacts": self.artifacts,
            "warnings": self.warnings,
            "config": self.config.to_dict(),
        }


def normalize_path(value: str | Path) -> str:
    """我把路径统一成字符串，方便写入 JSON。"""

    return str(Path(value).expanduser().resolve())


def _require_finite_real(value: Any, name: str) -> float:
    """我在配置边界拒绝布尔值、字符串和非有限数值。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"我要求 {name} 必须是有限数值。")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"我要求 {name} 必须是有限数值。")
    return numeric


def _require_int(value: Any, name: str, minimum: int) -> int:
    """我要求计数参数是整数，避免浮点值进入 range 或采样逻辑。"""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"我要求 {name} 必须是整数。")
    if int(value) < minimum:
        raise ValueError(f"我要求 {name} 至少为 {minimum}。")
    return int(value)


def _as_bool(value: Any, name: str) -> bool:
    """我按 JSON 语义解析布尔值，拒绝把任意非空字符串当成 True。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"我要求 {name} 必须是布尔值。")


def _require_string(value: Any, name: str) -> str:
    """我校验字段必须是字符串。"""
    if not isinstance(value, str):
        raise ValueError(f"我要求 {name} 必须是字符串。")
    return value
