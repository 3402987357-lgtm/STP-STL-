"""我方的领域数据模型区域。

我把这个包当作纯数据契约层：这里只放 dataclass 与枚举，不放算法、不做 I/O。
各子模块的职责边界是：

- ``models``：文件格式探测与解析结果（DetectedFormat / ProbeResult / Issue / Severity）；
- ``repair``：孔洞修补的区域与结果模型；
- ``step_features``：STEP Face 邻接、方向分组与区域模型；
- ``step_engineering``：模具工程特征候选、拔模证据与区域工程模型。

统一从本包导出，是为了让上层只依赖一份稳定契约：下游改算法实现时，数据类名与字段
不变，报告 schema 与审计口径就不会漂移。
"""

from .models import (
    DetectedFormat,
    FileMetadata,
    Issue,
    ParseStatus,
    ProbeResult,
    Severity,
)
from .repair import (
    BoundaryLoopResult,
    RepairConfig,
    RepairRegionResult,
    RepairResult,
)
from .step_features import (
    StepFaceAdjacency,
    StepFeatureClassificationReport,
    StepFeatureGroup,
    StepFeatureRegion,
)
from .step_engineering import (
    StepDraftEvidence,
    StepEngineeringCandidate,
    StepEngineeringRegion,
    StepEngineeringReport,
)

# 分组按来源模块排列，__all__ 里保持同一顺序：
# 这样新增一个类时能在两处都放在相同的位置，
# 不容易漏加或加重。
__all__ = [
    "DetectedFormat",
    "FileMetadata",
    "Issue",
    "ParseStatus",
    "ProbeResult",
    "BoundaryLoopResult",
    "RepairConfig",
    "RepairRegionResult",
    "RepairResult",
    "Severity",
    "StepFaceAdjacency",
    "StepFeatureClassificationReport",
    "StepFeatureGroup",
    "StepFeatureRegion",
    "StepDraftEvidence",
    "StepEngineeringCandidate",
    "StepEngineeringRegion",
    "StepEngineeringReport",
]
