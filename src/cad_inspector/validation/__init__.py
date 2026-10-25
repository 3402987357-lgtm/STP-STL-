"""我集中管理修补前后的网格质量检查。

修补必须有验收标准，否则"补完了"和"补好了"分不清。这里的检查围绕一个核心指标：
边界边。水密闭合网格的每条边恰好被两个三角面共享，边界边数必须为 0；修补前后各查
一次，边界边从 N 降到 0 才算真的补上了。

导出的能力分两类：诊断类（``boundary_edges`` / ``boundary_loops`` / ``edge_to_faces``）
给出问题在哪，验收类（``validate_repair`` / ``compare_mesh_quality``）给出修补前后
的量化对比，供修补成功率统计与甲方复核使用。
"""

from .mesh_quality import (
    boundary_edges,
    boundary_loops,
    compare_mesh_quality,
    edge_to_faces,
    validate_repair,
)

__all__ = [
    "boundary_edges",
    "boundary_loops",
    "compare_mesh_quality",
    "edge_to_faces",
    "validate_repair",
]
