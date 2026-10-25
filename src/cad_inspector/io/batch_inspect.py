"""我批量执行"候选件 vs 参考件"的完整检测，并汇总成一份可审计报告。

为什么需要这一层：``batch_probe`` 只做"这个文件能不能打开"的探测，
不跑检测。而真实交付场景是"一个目录下有几十个件，每个都要跟它的
参考件比一遍"——逐条敲命令既慢又容易漏，且失败了没有统一台账。

核心约定与 ``batch_probe`` 一致：**单个文件对失败只影响自身，不中断
整批**。我对每一对单独 try，异常转成一条 ``status=FAILED`` 的记录并
带上异常明细。这条约定在批量场景里比单件场景更关键——否则第 3 个件
打不开就会毁掉剩下 29 个件的结果。

我为什么坚持"按 stem 配对"而不是"按目录顺序配对"：真实数据集里
候选件与参考件的文件名有明确对应关系（``1.stl`` 对 ``1.stp``，
``训练集/0.stl`` 对 ``训练集/0.stp``）。按 stem 配对能在两侧顺序不一致、
或某个 stem 缺一半时准确定位，而不是靠下标静默配错——配错对会让
"缺失区域"整份结论反过来，属于最难发现的假结果。
"""

from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ..models import InspectionConfig
from ..output.atomic import atomic_path
from ..pipeline import inspect_pair


@dataclass
class BatchItemResult:
    """我记录一对模型在批量任务里的结果。"""

    candidate: str
    reference: str | None
    status: str
    output_dir: str | None = None
    missing_count: int = 0
    extra_count: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """我把单条结果转成可序列化字典。"""

        return {
            "candidate": self.candidate,
            "reference": self.reference,
            "status": self.status,
            "output_dir": self.output_dir,
            "missing_count": self.missing_count,
            "extra_count": self.extra_count,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "error": self.error,
            "warnings": list(self.warnings),
        }


@dataclass
class BatchInspectReport:
    """我给出整批检测的汇总台账。"""

    items: list[BatchItemResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        """我返回本批处理的对数。"""
        return len(self.items)

    @property
    def ok(self) -> int:
        """我返回成功完成检测的对数。"""
        return sum(item.status == "OK" for item in self.items)

    @property
    def failed(self) -> int:
        """我返回失败的对数。"""
        return sum(item.status == "FAILED" for item in self.items)

    @property
    def unpaired(self) -> int:
        """我返回找不到参考件的候选件数量。"""
        return sum(item.status == "UNPAIRED" for item in self.items)

    @property
    def elapsed_seconds(self) -> float:
        """我返回本批总耗时。"""
        return sum(item.elapsed_seconds for item in self.items)

    def to_dict(self) -> dict[str, object]:
        """我把整批汇总转成可序列化字典。"""

        return {
            "schema_version": "batch-inspect-1.0",
            "total": self.total,
            "ok": self.ok,
            "failed": self.failed,
            "unpaired": self.unpaired,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "items": [item.to_dict() for item in self.items],
        }


def _first_int(*values: object) -> int | None:
    """我按优先级取第一个可用的整数值。

    我为什么显式排除 bool：bool 是 int 的子类，``isinstance(True, int)``
    为真，不排除的话一个 True 会被当成计数 1 悄悄通过。
    """

    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _index_by_stem(paths: Sequence[Path]) -> dict[str, Path]:
    """我按文件名主干建索引，让候选件能找到它的参考件。"""

    index: dict[str, Path] = {}
    for path in paths:
        # 同一 stem 出现多次时保留第一个，并记录顺序由调用方保证：
        # 真实数据集里 stem 是唯一的，重复只可能来自通配符重复展开。
        index.setdefault(path.stem, path)
    return index


def pair_by_stem(
    candidates: Sequence[Path], references: Sequence[Path]
) -> list[tuple[Path, Path | None]]:
    """我把候选件按文件名主干与参考件配成对。

    找不到参考件的候选件我保留为 ``(候选, None)`` 而不是丢弃：
    "这一批里有个件没有参考件"是需要被看见的台账信息，
    静默跳过会让甲方以为所有件都检过了。
    """

    reference_index = _index_by_stem(references)
    return [
        (candidate, reference_index.get(candidate.stem)) for candidate in candidates
    ]


def run_batch_inspect(
    candidates: Iterable[str | Path],
    references: Iterable[str | Path],
    output_dir: str | Path,
    config: InspectionConfig | None = None,
    max_items: int | None = None,
) -> BatchInspectReport:
    """我批量检测多个"候选件 vs 参考件"对，返回汇总台账。

    每一对都写进 ``output_dir/<候选件stem>/``，与单件命令的输出布局
    一致——批量结果必须能被单件命令复现和复核，否则出了问题没法单独
    重跑那一个件。

    ``max_items`` 限制处理对数：批量任务在真实数据上可能跑很久，
    先用小数量试跑能快速验证参数是否正确，不至于等几十分钟才发现
    参数配错了。
    """

    candidate_paths = [Path(value).expanduser() for value in candidates]
    reference_paths = [Path(value).expanduser() for value in references]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pairs = pair_by_stem(candidate_paths, reference_paths)
    if max_items is not None:
        pairs = pairs[:max_items]

    report = BatchInspectReport()
    import time

    for candidate, reference in pairs:
        started = time.perf_counter()
        # 缺参考件：这不进入检测，但必须留在台账里。
        if reference is None:
            report.items.append(
                BatchItemResult(
                    candidate=str(candidate),
                    reference=None,
                    status="UNPAIRED",
                    warnings=[f"我没找到与 {candidate.stem} 同名的参考件。"],
                )
            )
            continue
        try:
            per_item_dir = output_root / candidate.stem
            result = inspect_pair(
                candidate_path=candidate,
                reference_path=reference,
                output_dir=per_item_dir,
                config=config,
            )
            # 区域计数不在 InspectionResult 的顶层，而在 global_metrics 里。
            # 我不猜字段：顶层拿不到就退回 global_metrics，再拿不到记 0
            # 并在 warnings 里留痕，绝不静默当成"没有缺失"。
            metrics = getattr(result, "global_metrics", None) or {}
            missing = _first_int(
                getattr(result, "missing_region_count", None),
                metrics.get("missing_region_count"),
            )
            extra = _first_int(
                getattr(result, "extra_region_count", None),
                metrics.get("extra_region_count"),
            )
            warnings = list(getattr(result, "warnings", []) or [])
            if missing is None or extra is None:
                warnings.append(
                    "我没能从检测结果里取到区域计数，台账中的计数按 0 记录，"
                    "请以该件的单件 JSON 报告为准。"
                )
            report.items.append(
                BatchItemResult(
                    candidate=str(candidate),
                    reference=str(reference),
                    status="OK",
                    output_dir=str(per_item_dir),
                    missing_count=int(missing or 0),
                    extra_count=int(extra or 0),
                    elapsed_seconds=time.perf_counter() - started,
                    warnings=warnings,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 批量任务的单点隔离点
            # 我在这里捕获所有异常是刻意的：批量任务里一个件读失败
            # 不该毁掉其余件。但我不吞掉信息——完整 traceback 进台账，
            # 让事后能定位"到底哪个件、哪一行炸了"。
            report.items.append(
                BatchItemResult(
                    candidate=str(candidate),
                    reference=str(reference),
                    status="FAILED",
                    elapsed_seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                    warnings=[traceback.format_exc(limit=8)],
                )
            )
    return report


def write_batch_inspect_reports(
    report: BatchInspectReport, output_dir: str | Path
) -> dict[str, Path]:
    """我把批量台账写成 JSON 与 CSV，返回实际路径。"""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "batch_inspect.json"
    csv_path = target_dir / "batch_inspect.csv"

    with atomic_path(json_path) as temporary:
        temporary.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # CSV 只放平铺的汇总字段：完整 warnings/traceback 是多行文本，
    # 塞进 CSV 会把行结构打乱，那些明细留在 JSON 里查。
    columns = [
        "candidate",
        "reference",
        "status",
        "missing_count",
        "extra_count",
        "elapsed_seconds",
        "error",
    ]
    with atomic_path(csv_path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for item in report.items:
                writer.writerow({key: getattr(item, key) for key in columns})
    return {"json": json_path, "csv": csv_path}
