"""我方结果输出、缺陷报告和可视化文件区域。

我把所有"写文件"的动作收口在这个包里，原因只有一条：原子写出必须只有一个入口。
中断、磁盘写满、并发替换都会留下半截文件，如果每个模块各自 ``open(path, "w")``，
就会到处出现能骗过"文件存在"检查、但打不开的坏产物。统一走 ``atomic_path`` 后，
一处加空间闸门、一处加临时文件替换，全项目的正式输出就都被覆盖到了。

各子模块的产出分工：

- ``writers``：检测结果 JSON/CSV 与彩色 PLY；
- ``parting_report``：分型结果 JSON/CSV/STL 与数据清单；
- ``cad_parting``：可交换的 IGS/STEP 分型几何；
- ``step_*_report``：STEP 拓扑、Face 特征、分类与工程特征报告；
- ``repair_report`` / ``step_repair_report``：修补结果；
- ``visualization``：可视化高亮与对比报告。
"""

from .parting_report import write_manifest, write_parting_result
from .atomic import atomic_path
from .cad_parting import write_step_parting_cad_outputs
from .repair_report import write_repair_csv, write_repair_json
from .step_report import write_step_topology_json
from .step_face_report import write_step_face_features_json
from .step_feature_report import write_step_feature_classification_json
from .step_engineering_report import write_step_engineering_report
from .step_repair_report import write_step_repair_csv, write_step_repair_json
from .visualization import (
    HighlightRegion,
    face_bboxes_from_step_face_features,
    highlight_regions_from_inspection,
    highlight_regions_from_step_repair,
    write_comparison_report,
)
from .writers import write_colored_ply, write_csv, write_json
from .flowchart import (
    FlowChart,
    build_flowcharts,
    verify_flowcharts,
    write_flowchart_markdown,
    write_flowcharts,
)

# 通用写盘工具排在前面：它们是其余各报告模块的基础，
# 按依赖顺序排能让"新增一个报告模块该插在哪"这个问题有明确答案。
# __all__ 与上面的 import 保持同一顺序，
# 两处位置一致就不容易漏登记或重复登记。
__all__ = [
    "write_colored_ply",
    "write_csv",
    "write_json",
    "atomic_path",
    "write_manifest",
    "write_parting_result",
    "write_step_parting_cad_outputs",
    "write_repair_csv",
    "write_repair_json",
    "write_step_topology_json",
    "write_step_face_features_json",
    "write_step_feature_classification_json",
    "write_step_engineering_report",
    "write_step_repair_json",
    "write_step_repair_csv",
    "HighlightRegion",
    "face_bboxes_from_step_face_features",
    "highlight_regions_from_inspection",
    "highlight_regions_from_step_repair",
    "write_comparison_report",
    "FlowChart",
    "build_flowcharts",
    "verify_flowcharts",
    "write_flowcharts",
    "write_flowchart_markdown",
]
