"""我方按模型尺度和配准残差推导缺失检测阈值。

原理说明：固定阈值（例如 0.15 mm）对大小差别很大的零件并不公平。
一个 8 mm 的小件和一个 800 mm 的大件用同一个绝对阈值，小件会因为
配准残差、扫描噪声和三角化偏差被整件误判成破损。我方因此把阈值拆成
两个可解释的分量：

1. 尺度分量 = `auto_threshold_scale_ratio` × 参考模型包围盒对角线；
2. 噪声分量 = `auto_threshold_residual_factor` × 配准平均残差。

推荐阈值取两者的较大值，并受一个绝对下限保护。这样阈值既随零件大小
缩放，又不会低于当前配准精度所能分辨的水平。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..geometry.mesh import TriangleMesh
from ..models import InspectionConfig


@dataclass(frozen=True)
class ThresholdRecommendation:
    """我方给出一个可解释的推荐阈值和它的两个来源分量。

    我为什么把 configured / recommended / effective 三个值都存下来：
    甲方核对时最常问的是"你用的阈值是哪来的"。只存 effective 就无法
    回答"是用户写死的还是自动推的"，也无法事后判断"用户给的阈值
    是不是偏低了"。三个都留，报告里能直接看出分歧在哪。

    我为什么用 frozen=True：阈值一旦算出来就不该被就地修改，
    否则同一个区域在流水线的不同阶段会拿到不同阈值，结果无法复现。
    冻结让我能在测试里直接比对整个对象。
    """

    configured_mm: float
    recommended_mm: float
    effective_mm: float
    mode: str
    bbox_diagonal_mm: float
    registration_mean_residual_mm: float
    scale_component_mm: float
    residual_component_mm: float
    floor_mm: float
    residual_dominates: bool
    scale_too_coarse: bool

    @property
    def configured_below_recommended(self) -> bool:
        """我判断配置阈值是否低于当前模型应使用的最低阈值。

        我为什么减 1e-12 再比：浮点阈值经过乘法与 max 之后会有末位误差，
        相等的两个值可能相差一个 ULP。不减这个容差，"配置值恰好等于
        推荐值"会被误报成"配置偏低"，凭空产生一条假警告。
        """

        return self.configured_mm < self.recommended_mm - 1e-12

    def to_dict(self) -> dict[str, Any]:
        """我把阈值建议序列化成报告用的字典。"""

        return {
            "mode": self.mode,
            "configured_mm": self.configured_mm,
            "recommended_mm": self.recommended_mm,
            "effective_mm": self.effective_mm,
            "bbox_diagonal_mm": self.bbox_diagonal_mm,
            "registration_mean_residual_mm": (
                self.registration_mean_residual_mm
            ),
            "scale_component_mm": self.scale_component_mm,
            "residual_component_mm": self.residual_component_mm,
            "floor_mm": self.floor_mm,
            "residual_dominates": self.residual_dominates,
            "scale_too_coarse": self.scale_too_coarse,
            "configured_below_recommended": self.configured_below_recommended,
        }


def bbox_diagonal_mm(mesh: TriangleMesh) -> float:
    """我取参考模型包围盒对角线作为尺度基准。

    我为什么用对角线而不是最长边或体积：对角线对"细长件"和"扁平件"
    都给同一个量纲的长度，而最长边会漏掉另外两个方向的尺寸（一个
    薄片的最长边可能很长但厚度极小），体积则是长度的三次方、
    乘上系数后量纲不对。对角线是唯一既覆盖三维、又是长度量纲的基准。
    """

    extent = np.asarray(mesh.bbox_max) - np.asarray(mesh.bbox_min)
    return float(np.linalg.norm(extent))


def recommend_threshold_mm(
    reference: TriangleMesh,
    registration_mean_residual_mm: float,
    config: InspectionConfig,
) -> ThresholdRecommendation:
    """我按尺度分量和噪声分量推导推荐阈值与本次实际生效阈值。"""

    diagonal = bbox_diagonal_mm(reference)
    residual = float(registration_mean_residual_mm)
    # 我把非法残差统一归零而不是抛错：配准失败时残差可能是 NaN，
    # 而"配准失败"这件事已经由配准模块自己报警了。这里再抛一次错
    # 会让整条检测因为一个已知问题中断；归零则退化为"只按尺度定阈值"，
    # 结果仍然可用，只是失去了噪声分量这一路证据。
    if not math.isfinite(residual) or residual < 0.0:
        residual = 0.0
    scale_component = float(config.auto_threshold_scale_ratio) * diagonal
    residual_component = float(config.auto_threshold_residual_factor) * residual
    floor = float(config.auto_threshold_floor_mm)
    recommended = max(scale_component, residual_component, floor)
    if config.threshold_mode == "auto":
        effective = recommended
    else:
        effective = float(config.threshold_mm)
    # 我判断残差是否压过尺度分量；是的话说明噪声决定了下限。
    residual_dominates = residual_component > scale_component + 1e-12
    # 我判断残差相对模型是否过大；是的话当前配准不足以支撑可靠检测。
    # 5% 这个比例是我实测取的：残差超过模型尺度的 5% 时，同一份数据
    # 重复配准给出的区域面积会明显漂移，此时再谈"毫米级缺失"没有意义。
    # diagonal > 0 的守卫是防止空网格除零。
    scale_too_coarse = diagonal > 0 and residual > 0.05 * diagonal
    return ThresholdRecommendation(
        configured_mm=float(config.threshold_mm),
        recommended_mm=recommended,
        effective_mm=effective,
        mode=str(config.threshold_mode),
        bbox_diagonal_mm=diagonal,
        registration_mean_residual_mm=residual,
        scale_component_mm=scale_component,
        residual_component_mm=residual_component,
        floor_mm=floor,
        residual_dominates=residual_dominates,
        scale_too_coarse=scale_too_coarse,
    )
