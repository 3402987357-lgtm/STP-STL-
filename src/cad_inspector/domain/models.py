"""我方第一阶段使用的稳定数据模型。

我把格式探测结果定义成普通 dataclass，而不是直接把解析器内部对象暴露给
后续模块。这样我方以后更换 OCC、Parasolid 或 NX 适配器时，不需要重写
BOM、日志和界面层。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DetectedFormat(str, Enum):
    """我方目前关心的输入格式。"""

    STEP = "STEP"
    PARASOLID_XT = "PARASOLID_XT"
    SIEMENS_NX_PRT = "SIEMENS_NX_PRT"
    STL = "STL"
    UNKNOWN = "UNKNOWN"


class ParseStatus(str, Enum):
    """我方统一的阶段状态。"""

    READY = "READY"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    FAILED = "FAILED"


class Severity(str, Enum):
    """我方对问题严重程度的统一定义。"""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Issue:
    """我方需要写入报告的结构化问题。"""

    code: str
    message: str
    severity: Severity
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """我把问题转成报告字典。

        severity 取 .value 而不是枚举对象：报告要写成 JSON，
        枚举对象不可序列化。写成裸字符串还能让下游按字面值
        过滤严重级，不必先 import 本模块的枚举。

        detail 允许为 None 并原样保留 None：
        "没有补充信息"与"补充信息是空字符串"是两回事，
        把 None 压成 "" 会让人分不清是没采集还是采集到空。
        """
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "detail": self.detail,
        }


@dataclass
class FileMetadata:
    """我方对输入文件的只读元数据。"""

    path: str
    name: str
    extension: str
    size_bytes: int
    sha256: str
    detected_format: DetectedFormat

    def to_dict(self) -> dict[str, Any]:
        """我把元数据转成字典。

        先 asdict 再把 detected_format 换成 .value：asdict 对枚举
        会原样保留枚举对象，而它需要写进 JSON，所以必须降级成字面值。
        只改这一个字段而不是手工重写整份字典，是为了让新增字段时
        不必同步改这里——漏改就是静默丢字段。

        sha256 是必填而不是可选：它是判断"两次运行是否在同一份
        输入上"的唯一依据。没有它，报告里的结论无法追溯到具体文件。
        """
        data = asdict(self)
        data["detected_format"] = self.detected_format.value
        return data


@dataclass
class ProbeResult:
    """我方格式探测区域的完整输出。"""

    file: FileMetadata
    status: ParseStatus
    header: dict[str, Any] = field(default_factory=dict)
    entity_counts: dict[str, int] = field(default_factory=dict)
    unit_hint: str | None = None
    backend_required: bool = True
    issues: list[Issue] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """我把探测结果转成字典，并顺带算出吞吐率。

        entity_counts 输出前排序：这份字典会写进 JSON 并参与
        报告比对。dict 的迭代顺序取决于插入顺序，同一批数据
        若遍历顺序不同就会产出不同的 JSON 文本，让两次运行的
        报告看起来"变了"——实际上内容完全一样。排序消除这种假差异。

        throughput 在 elapsed_seconds 为 0 时给 None 而不是 0 或
        inf：耗时为 0（或没采集）时算不出有意义的速率，
        给 0 会被读成"极慢"，给 inf 会直接让 JSON 不合法
        （严格 JSON 没有 inf）。None 才能正确表达"没测出来"。

        elapsed_seconds 保留 6 位小数：毫秒级精度对性能分析够用，
        再多位只是把报告撑大。
        """
        return {
            "file": self.file.to_dict(),
            "status": self.status.value,
            "header": self.header,
            "entity_counts": dict(sorted(self.entity_counts.items())),
            "unit_hint": self.unit_hint,
            "backend_required": self.backend_required,
            "issues": [issue.to_dict() for issue in self.issues],
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "throughput_bytes_per_second": (
                round(self.file.size_bytes / self.elapsed_seconds, 2)
                if self.elapsed_seconds > 0
                else None
            ),
        }
