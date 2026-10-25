"""我把三维几何比对能力集中在这个包中，方便后续扩展为服务或桌面应用。

顶层只导出四个核心契约（配置、结果、区域、配准），子能力按包分层，各层职责不交叉：

- ``io``：输入校验与格式适配（信任边界，先校验再放行）；
- ``geometry``：网格、距离、变换与 STL 读写；
- ``detection``：网格路线与 B-Rep 路线两套识别；
- ``repair``：孔洞修补；
- ``output``：所有写文件动作（统一原子写出）；
- ``validation``：修补前后的网格质量验收；
- ``learning``：合成真值、特征与可选深度学习；
- ``domain``：纯数据契约，不含算法。

只导出契约而不导出实现，是为了让调用方依赖稳定的报告结构，替换几何后端时上层代码不动。
"""

from .models import InspectionConfig, InspectionResult, RegionResult, RegistrationResult

__all__ = [
    "InspectionConfig",
    "InspectionResult",
    "RegionResult",
    "RegistrationResult",
]

__version__ = "0.1.0"
