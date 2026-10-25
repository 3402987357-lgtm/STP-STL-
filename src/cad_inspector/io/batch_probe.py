"""我提供批量文件探测、排序和汇总报告。

算法原理：批量探测面对的真实数据一定有脏文件（打不开的、空文件、非 CAD 文件被误放进
目录）。所以核心约定只有一条：**单个文件失败只影响自身，不中断整批任务**。
``run_batch`` 对每个文件单独 try，异常转成一条 status=FAILED 的 ProbeResult 并带上
异常明细；只有当连文件元信息都取不到时才抛 RuntimeError（那说明输入本身不可用）。

输入展开用规范化绝对路径的大小写无关键去重：同一文件通过不同相对路径、不同大小写被
传入两次时只处理一次，且按绝对路径排序，保证同一批输入在不同机器上的处理顺序一致。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cad_inspector.domain.models import Issue, ParseStatus, ProbeResult, Severity
from cad_inspector.io.input_validator import validate_input_path
from cad_inspector.io.step_probe import probe_any
from cad_inspector.output.atomic import atomic_path


SUPPORTED_SUFFIXES = {".step", ".stp", ".xt", ".prt", ".stl"}


@dataclass
class BatchProbe:
    """我把一批模型的结果保存在一个可审计对象中。

    我只存 results 一个字段，四个状态计数都现算成 property：
    计数是结果的派生量，存下来就要在每次追加时同步维护，
    漏维护一次就会给出和 results 不一致的数字——
    而这类不一致在报告里看起来完全正常，是最难发现的一种错。
    """

    results: list[ProbeResult]

    @property
    def total(self) -> int:
        """我返回本批探测的文件总数。"""
        return len(self.results)

    @property
    def complete(self) -> int:
        """我返回解析完整的样本数。"""
        return sum(result.status is ParseStatus.COMPLETE for result in self.results)

    @property
    def partial(self) -> int:
        """我返回部分解析的样本数。"""
        return sum(result.status is ParseStatus.PARTIAL for result in self.results)

    @property
    def unsupported(self) -> int:
        """我返回格式不受支持的样本数。"""
        return sum(result.status is ParseStatus.UNSUPPORTED for result in self.results)

    @property
    def failed(self) -> int:
        """我返回解析失败的样本数。"""
        return sum(result.status is ParseStatus.FAILED for result in self.results)

    def to_dict(self) -> dict:
        """我把批量探测汇总转成可序列化字典。"""
        # 先给四个状态计数，再附完整明细列表：
        # 计数用于快速判断这批数据整体健康度，
        # 明细用于定位"到底哪个文件出了问题"，两者不可互相替代。
        return {
            "total": self.total,
            "complete": self.complete,
            "partial": self.partial,
            "unsupported": self.unsupported,
            "failed": self.failed,
            "results": [result.to_dict() for result in self.results],
        }


def expand_inputs(paths: Iterable[str | Path]) -> list[Path]:
    """我递归展开目录，按规范化绝对路径排序，并去掉重复文件。

    去重键用 casefold 过的绝对路径：Windows 文件系统大小写不敏感，
    同一个文件以 "Model.STP" 和 "model.stp" 传入时，
    大小写敏感的键会把它当成两个文件探测两遍，
    报告里就出现重复行。casefold 比 lower 更稳——
    它对非 ASCII 的大小写也做了归一化。

    排序后返回而不是保持传入顺序：同一批输入在不同顺序下
    必须产出同一份报告，否则两次运行的文件差异会淹没真正的
    内容变化。

    后缀比较用 .lower()：甲方目录里 .STP / .Step / .stp 混用很常见。
    """

    # 用大小写无关的绝对路径做键去重，同一文件被多次传入只保留一份。
    # 大小写无关是必需的：Windows 上 A.stp 与 a.stp 是同一个文件，
    # 大小写敏感的键会让它在批量结果里出现两次。
    unique: dict[str, Path] = {}
    for raw in paths:
        # 不存在或既非文件又非目录的路径静默跳过：
        # 批量输入常常来自通配符展开，混入一个不存在的路径
        # 不该让整批探测失败。
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            children = candidate.rglob("*")
        elif candidate.is_file():
            children = (candidate,)
        else:
            continue
        for child in children:
            # 后缀比较统一转小写：STEP 文件常见 .STP/.stp 混用，
            # 大小写敏感比较会把合法文件误判成不支持。
            if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                resolved = child.resolve()
                unique[str(resolved).casefold()] = resolved
    # 按排序后的键返回：同一批输入每次都得到相同顺序，
    # 批量报告才能跨运行逐行比对。
    return [unique[key] for key in sorted(unique)]


def _failed_result(path: Path, exc: Exception) -> ProbeResult:
    """我把单文件异常转成一条 FAILED 记录，保留异常明细供人工定位。

    detail 里存 str(exc) 而不是只存异常类型：批量跑几百个文件时，
    只知道"某个文件抛了 ValueError"毫无用处，
    必须能看到具体消息才能判断是数据问题还是代码问题。

    仍然走 validate_input_path 取元信息：即使解析失败，
    也要留下"是哪个文件"（路径、大小、sha256），
    否则这条失败记录无法追溯到具体输入。
    """
    metadata = validate_input_path(path)
    return ProbeResult(
        file=metadata,
        status=ParseStatus.FAILED,
        issues=[
            Issue(
                code="BATCH_PROBE_EXCEPTION",
                message="我在批量探测中遇到单文件异常。",
                severity=Severity.ERROR,
                detail=str(exc),
            )
        ],
    )


def run_batch(paths: Iterable[str | Path]) -> BatchProbe:
    """我让单个文件失败只影响自身，不中断整批任务。

    为什么不做"失败即中止"：批量探测面对的是甲方整个目录，
    里面混着各种年代、各种导出器的文件，个别文件读不出来是常态。
    一个坏文件让整批中止，等于几百个好文件的结果全丢，
    而跑一遍要几十分钟。

    但连元信息都取不到时必须向上抛错（第二层 except）：
    这时连"哪个文件失败了"都记不下来，
    说明输入本身不可用（路径不存在、权限不足），
    继续跑下去产出的报告是不完整的却看起来正常——
    那比明确报错危险。
    """

    results: list[ProbeResult] = []
    for path in expand_inputs(paths):
        # 单文件异常只影响该文件；连元信息都取不到才说明输入不可用，向上抛错。
        try:
            results.append(probe_any(path))
        except Exception as exc:
            try:
                results.append(_failed_result(path, exc))
            except Exception as metadata_exc:
                raise RuntimeError(f"{path}: {metadata_exc}") from exc
    return BatchProbe(results)


def write_batch_reports(batch: BatchProbe, output_dir: str | Path) -> tuple[Path, Path]:
    """我同时输出 JSON 和 UTF-8 with BOM CSV。

    两种格式都写而不是只写一种：JSON 给程序读（下游再处理、
    归档比对），CSV 给人读（甲方直接在 Excel 里打开筛选）。
    只给 JSON 甲方打不开，只给 CSV 别人没法程序化处理。

    CSV 用 utf-8-sig 而不是 utf-8：这不是偏好问题。
    Excel 在没有 BOM 时会按系统 ANSI 代码页（中文环境是 GBK）
    去解 UTF-8 字节，中文路名与状态直接变成乱码。
    加 BOM 是让 Excel 正确识别编码的唯一实用手段。

    两份文件都走 atomic_path：写一半的 JSON/CSV 会让下游
    "文件存在即有效"的检查误判通过。
    """

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "batch_summary.json"
    csv_path = target_dir / "batch_summary.csv"
    with atomic_path(json_path) as temporary:
        temporary.write_text(
            json.dumps(
                batch.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    # CSV 列顺序固定，便于不同批次横向比对；issue_codes 用分号拼接成单单元格。
    fields = [
        "path",
        "name",
        "extension",
        "detected_format",
        "status",
        "size_bytes",
        "sha256",
        "unit_hint",
        "entity_total",
        "elapsed_seconds",
        "issue_codes",
    ]
    with atomic_path(csv_path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for result in batch.results:
                writer.writerow(
                    {
                        "path": result.file.path,
                        "name": result.file.name,
                        "extension": result.file.extension,
                        "detected_format": result.file.detected_format.value,
                        "status": result.status.value,
                        "size_bytes": result.file.size_bytes,
                        "sha256": result.file.sha256,
                        "unit_hint": result.unit_hint or "",
                        "entity_total": sum(result.entity_counts.values()),
                        "elapsed_seconds": round(result.elapsed_seconds, 6),
                        "issue_codes": ";".join(
                            issue.code for issue in result.issues
                        ),
                    }
                )
    return json_path, csv_path
