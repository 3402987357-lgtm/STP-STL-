"""我钉住三角化默认值必须与检测管线一致。

为什么需要这个测试：CAD 面映射要求"三角化参数与加载网格时完全相同"，
参数不同会得到不同数量的三角形，下标对不上，映射出的 CAD 面号全是错的
**而且不报错**。默认 0.5 对管线 0.04 就是一个默认就踩坑的入口——
用 tessellate-step 单独三角化再和检测结果比对，会静默得到对不上的结果。
"""

from __future__ import annotations

import math

from cad_inspector.geometry.occt_tessellation import TessellationConfig
from cad_inspector.models import InspectionConfig


def test我方三角化默认弦高与管线一致() -> None:
    """默认值不一致会让面号映射静默错位，所以必须相等而不是"接近"。"""

    assert TessellationConfig().linear_deflection == (
        InspectionConfig().mesh_linear_deflection_mm
    )


def test我方三角化默认角度偏差与管线一致() -> None:
    """管线用度、三角化用弧度，单位不同更要钉住换算。"""

    expected = math.radians(InspectionConfig().mesh_angular_deflection_deg)
    assert abs(TessellationConfig().angular_deflection - expected) < 1e-4
