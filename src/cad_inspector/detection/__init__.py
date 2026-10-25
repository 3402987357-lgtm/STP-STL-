"""我方规则识别和缺失区域检测区域。

两条检测路线在这里汇合，用途不同、精度不同，绝不互相冒充：

- 网格路线（``missing_regions`` / ``features`` / ``parting_line`` / ``parting_surface``）：
  基于三角网格，在没有 OCC 的环境下也能跑通，是"能跑的基线"，明确标注为 BASELINE；
- B-Rep 路线（``step_features`` / ``step_region_join``）：基于 OCP 真实拓扑，能拿到
  Face/Edge/Vertex 级证据，是精度更高的路线。

导出时保留两条路线的名字，是为了让调用方显式选择走哪条，而不是被某个"自动选优"的
包装悄悄切换口径。
"""

from .missing_regions import detect_missing_regions
from .features import recognize_mesh_features
from .parting_line import extract_parting_line
from .parting_surface import build_planar_parting_surface
from .step_features import (
    classify_step_face_features,
    inspect_step_feature_classification,
)
from .step_region_join import (
    join_step_engineering_regions,
    load_step_engineering_region_map,
)
from .voxel_compare import (
    VoxelComparison,
    build_voxel_occupancy,
    compare_meshes_voxel,
    free_edge_count,
)

__all__ = [
    "build_planar_parting_surface",
    "detect_missing_regions",
    "extract_parting_line",
    "recognize_mesh_features",
    "classify_step_face_features",
    "inspect_step_feature_classification",
    "join_step_engineering_regions",
    "load_step_engineering_region_map",
    "VoxelComparison",
    "build_voxel_occupancy",
    "compare_meshes_voxel",
    "free_edge_count",
]
