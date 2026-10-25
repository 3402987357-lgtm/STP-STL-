"""我集中管理输入文件校验、STEP 适配和批量探测。

输入层是所有算法的信任边界，这里出问题后面全错，所以每个入口都先校验再放行：

- ``file_validation``：校验输入文件存在、后缀受支持、可读性，杜绝"路径打错还能跑"；
- ``step``：STEP 三角化适配层，优先 pythonocc-core，回落 OCP；
- ``step_topology`` / ``step_faces`` / ``step_bom``：B-Rep 拓扑、Face 特征与装配 BOM；
- ``batch_probe``：批量探测，单文件失败只影响自身；
- ``artifact_audit``：输出文件内容级审计，挡住"存在但损坏"的产物。

统一从本包导出，让上层只依赖一份输入契约。
"""

from .batch_probe import BatchProbe, expand_inputs, run_batch, write_batch_reports
from .artifact_audit import audit_artifacts
from .file_validation import validate_input_file, validate_model_pair
from .step import load_step_as_mesh
from .step_topology import StepTopologyReport, inspect_step_topology
from .step_faces import StepFaceFeatureReport, inspect_step_face_features
from .step_bom import BomReport, inspect_step_bom

__all__ = [
    "BatchProbe",
    "audit_artifacts",
    "expand_inputs",
    "load_step_as_mesh",
    "StepTopologyReport",
    "inspect_step_topology",
    "StepFaceFeatureReport",
    "inspect_step_face_features",
    "BomReport",
    "inspect_step_bom",
    "run_batch",
    "validate_input_file",
    "validate_model_pair",
    "write_batch_reports",
]
