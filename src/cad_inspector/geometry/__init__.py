"""我集中管理网格、距离和几何变换能力。

这一层是所有几何计算的公共底座，只依赖标准库与 NumPy，不依赖 OCP。这样设计的好处是：
在没有 CAD 内核的环境里（比如只装了最小依赖的 CI），网格路线仍能完整跑通，BOM 与检测
的基线结果依然可得；OCP 只影响 STEP 相关的 B-Rep 精度路线。

``TriangleMesh`` 是唯一的网格载体，统一提供重心、法向、面积、包围盒与三角面邻接；
``mesh_quality_report`` 给出退化面、非流形边等质量诊断，供修补前后对比使用。
"""

from .mesh import TriangleMesh, build_triangle_adjacency, mesh_quality_report
from .stl import load_stl, save_stl

__all__ = [
    "TriangleMesh",
    "build_triangle_adjacency",
    "load_stl",
    "mesh_quality_report",
    "save_stl",
]
