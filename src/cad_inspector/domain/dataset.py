"""我方数据集登记区域使用的数据模型。

算法原理：数据集清单是训练与验收的信任起点。每个样本登记的都是"事实"而不是"判断"：
路径、名称、后缀、字节数、sha256、探测出的格式。有了 sha256，任何一轮训练都能证明自己
用的是同一批文件；换一批数据、改一个字节，指纹就变，旧结论自动失效。

``dataset_split`` / ``label_status`` / ``probe_status`` 三个状态字段默认为"未分配"/
"未知"而不是"已就绪"——没做过划分就说已就绪，是数据泄漏最常见的来源。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DatasetRecord:
    """我把每个输入样本的事实和处理状态单独登记。"""

    sample_id: str
    path: str
    name: str
    extension: str
    size_bytes: int
    sha256: str
    detected_format: str
    dataset_split: str = "unassigned"
    label_status: str = "unknown"
    probe_status: str = "unknown"
    notes: list[str] = field(default_factory=list)
    # 我为什么加这个字段：我给 STL 补了真正的解析（顶点数、三角面数、
    # 退化面、非流形边），但 build_manifest 原先只取 probe_any().status，
    # 把这些计数全丢了——探测算出来了、清单里却看不到。放在最后加
    # 默认值，不破坏既有字段顺序。
    entity_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """我把样本登记记录转成字典。"""
        return asdict(self)


@dataclass
class DatasetManifest:
    """我把一次数据扫描保存成可复现的清单。"""

    schema_version: str
    root: str
    created_at_utc: str
    records: list[DatasetRecord]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """我把数据清单转成带 schema 版本的可序列化字典。"""
        return {
            "schema_version": self.schema_version,
            "root": self.root,
            "created_at_utc": self.created_at_utc,
            "records": [record.to_dict() for record in self.records],
            "warnings": list(self.warnings),
        }
