"""我方批量解析一个目录下的所有 STEP 装配体，连续生成几何 BOM 基线。

我按甲方"连续处理多套模具"和"不遗漏任何零件"的要求，把单文件 BOM 入口
扩成批处理：逐套解析、逐套落盘、逐套记录耗时和内存。任何一套失败都只写进
汇总，不中断后续模具，方便后续 AI 一眼看出是哪一套、哪一步出的问题。
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cad_inspector.io.bom_master_data import load_bom_master_data  # noqa: E402
from cad_inspector.io.step_bom import inspect_step_bom  # noqa: E402
from cad_inspector.output.atomic import atomic_path  # noqa: E402
from cad_inspector.output.bom_report import write_bom_report  # noqa: E402


SCHEMA_VERSION = "bom-batch-1.0"
_STEP_SUFFIXES = frozenset({".stp", ".step"})


def _rss_mb() -> float | None:
    """我读取当前进程工作集，用来观察连续处理是否持续涨内存。"""

    # 只在 Windows 上读 psapi；读不到就返回 None，不让监控反过来拖垮批处理。
    try:
        import ctypes
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            """我映射 Windows PROCESS_MEMORY_COUNTERS 结构体字段。"""
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        psapi = ctypes.WinDLL("psapi")
        function = psapi.GetProcessMemoryInfo
        function.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        function.restype = wintypes.BOOL
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if function(wintypes.HANDLE(handle), ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / (1024.0 * 1024.0), 3)
    except Exception:  # noqa: BLE001
        return None
    return None


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """我按块读文件算 SHA-256，避免一次性把大 STEP 读进内存。

    我为什么分块而不是 ``path.read_bytes()``：真实模具有接近 60 MB
    的 STEP（实测 训练集/1 是 57.98 MB），整读会让批量处理的峰值
    内存随文件大小线性上涨，几十个件并发时直接撑爆。分块读则把峰值
    固定在 1 MB，与文件大小无关。

    我选 1 MB 而不是更小的 4 KB：4 KB 会把系统调用次数放大上千倍，
    在机械盘上哈希本身就比读文件慢；1 MB 是实测吞吐与内存的平衡点。
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_step_files(root: Path) -> list[Path]:
    """我按路径稳定排序收集所有 .stp/.step 文件。

    排序用 str(path).lower() 而不是 path 本身：Windows 文件系统
    大小写不敏感但字符串比较敏感，不统一大小写会让
    "B.stp" 和 "a.stp" 的顺序取决于它们的实际大小写，
    不同机器上跑出来的批次顺序可能不同，
    而顺序直接决定了同名文件谁拿到无后缀的那个名字。

    rglob("*") 后逐个 is_file() 过滤：符号链接指向目录时
    is_file() 已经是 False，会自然排除，不需要额外处理。
    """

    # 我同时收 .stp 与 .step 两种后缀：模具厂导出的两种写法都会遇到，
    # 只收一种会静默漏掉一半的件——而且不报错，只在最后计数时少几个。
    # 大小写统一 lower 处理，因为 Windows 上 .STP 与 .stp 是同一个文件。

    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _STEP_SUFFIXES
    ]
    return sorted(files, key=lambda item: str(item).lower())


def _unique_target_names(files: list[Path]) -> dict[Path, str]:
    """我给同名不同目录的 STEP 分配唯一子目录，避免互相覆盖。

    用内容哈希前 8 位做后缀，而不是目录序号或父目录名：
      - 序号依赖排序，换一次遍历顺序同一个文件就会得到不同名字，
        两次运行的输出对不上；
      - 父目录名可能含中文和空格，拼进路径后在不同系统上的
        编码和转义行为不一致；
      - 哈希只依赖文件内容，同内容永远同名，可复现且不引入非法字符。

    第一次出现的名字保持原样（不加后缀）：绝大多数情况下名字本就唯一，
    给所有文件都加哈希后缀会让输出目录不可读，
    也让人误以为"这些文件都有重名"。
    """

    names: dict[Path, str] = {}
    used: dict[str, int] = {}
    for path in files:
        stem = path.stem
        count = used.get(stem, 0)
        used[stem] = count + 1
        names[path] = stem if count == 0 else f"{stem}__{_sha256(path)[:8]}"
    return names


def run_batch(
    root: Path,
    output_dir: Path,
    master_data_path: str | None,
) -> dict[str, object]:
    """我连续处理目录内所有 STEP，并返回可审计的批量汇总。

    设计要点（对应甲方"连续处理多套模具无崩溃无泄漏、不遗漏任何零件"）：
      1. **单套失败不中断批次**。我把 inspect + write 整段包在 try 里，
         失败只把异常写进该套的 error 字段，循环继续。十套模具里有一套
         几何损坏时，甲方仍然拿到另外九套的结果，而不是空手而归。
      2. **逐套记录前后内存**。连续批处理最怕的是内存单调增长——
         单套能跑不代表连跑十套不爆。我在每套结束后 gc.collect() 再读一次
         工作集，汇总里能直接看出是否真的释放了。
      3. **同名不同目录用哈希后缀区分**。直接按 stem 命名会让
         两个同名模具的输出互相覆盖，静默丢掉一整套零件，
         这正是"不遗漏任何零件"最容易破功的地方。
      4. 全部落盘走 atomic_path，避免断电留下半截 JSON/CSV 骗过审计。
    """

    master_data = (
        load_bom_master_data(master_data_path) if master_data_path else None
    )
    files = discover_step_files(root)
    # 同名不同目录的 STEP 用哈希后缀区分，避免输出互相覆盖。
    target_names = _unique_target_names(files)
    records: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        # 每套模具单独计时、单独记录前后内存，便于看出是否存在内存持续增长。
        before_mb = _rss_mb()
        started = time.perf_counter()
        record: dict[str, object] = {
            "order": index,
            "name": path.stem,
            "source_path": str(path.resolve()),
            "source_size_bytes": path.stat().st_size,
            "source_sha256": _sha256(path),
            "rss_before_mb": before_mb,
        }
        try:
            report = inspect_step_bom(path, master_data=master_data)
            artifacts = write_bom_report(
                report,
                output_dir / target_names[path],
            )
            record.update(
                {
                    "status": report.status,
                    "component_count": report.component_count,
                    "artifacts": {
                        key: str(Path(value).resolve())
                        for key, value in artifacts.items()
                    },
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            # 单套失败只写进记录，不中断后续模具，保证"不遗漏任何零件"。
            record.update(
                {
                    "status": "FAILED",
                    "component_count": None,
                    "artifacts": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        # 每套结束后强制回收，并记录回收后的内存，用于验证是否真的释放了。
        gc.collect()
        record["rss_after_mb"] = _rss_mb()
        records.append(record)
        print(
            f"[{index}/{len(files)}] {path.stem} -> {record['status']} "
            f"components={record['component_count']} "
            f"elapsed={record['elapsed_seconds']}s",
            flush=True,
        )

    # 汇总统计：总数、BASELINE/REVIEW/FAILED 分布、总耗时与最慢单套耗时。
    statuses = [str(item["status"]) for item in records]
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "file_count": len(records),
        # 三个状态分别计数而不是只给一个总数：
        # 批量跑几百个文件时，"多少个需要复核"才是排工作量的依据，
        # 只给总数等于让人自己重算一遍。
        "baseline_count": statuses.count("BASELINE"),
        "review_required_count": statuses.count("REVIEW_REQUIRED"),
        "failed_count": statuses.count("FAILED"),
        # 总耗时看整体成本，最大耗时看最坏样本：
        # 平均值会被大量小文件拉低，掩盖某个大文件的性能问题。
        "total_elapsed_seconds": round(
            sum(float(item["elapsed_seconds"]) for item in records),
            3,
        ),
        "max_elapsed_seconds": max(
            (float(item["elapsed_seconds"]) for item in records),
            default=0.0,
        ),
        "records": records,
    }
    # 只按 failed 判 PASS/FAIL：REVIEW_REQUIRED 是正常的工作产出，
    # 把它算进失败会让批量跑几乎永远是 FAIL，门禁就失去意义了。
    summary["status"] = "PASS" if summary["failed_count"] == 0 else "FAIL"

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "bom_batch_summary.json"
    # 两份产物都走 atomic_path：写到一半被中断会留下截断的 json/csv，
    # 下游解析时报错的位置与实际出错原因完全不同，极难定位。
    with atomic_path(summary_json) as temporary:
        temporary.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                # allow_nan=False 让 NaN/Infinity 直接抛错而不是写出非法 JSON：
                # 非法 JSON 会让下游解析器整份拒绝，
                # 而错误发生在读取时已经离产生它的计算很远了。
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    summary_csv = output_dir / "bom_batch_summary.csv"
    with atomic_path(summary_csv) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "order",
                    "name",
                    "status",
                    "component_count",
                    "elapsed_seconds",
                    "rss_before_mb",
                    "rss_after_mb",
                    "source_size_bytes",
                    "source_sha256",
                    "source_path",
                    "error",
                ],
            )
            writer.writeheader()
            for item in records:
                writer.writerow({key: item.get(key) for key in writer.fieldnames})
    summary["summary_json"] = str(summary_json.resolve())
    summary["summary_csv"] = str(summary_csv.resolve())
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """我构造批量 BOM 命令行解析器：扫描根目录、输出目录、可选主数据。"""
    parser = argparse.ArgumentParser(
        description="我方批量生成 STEP 装配体几何 BOM 基线。"
    )
    parser.add_argument(
        "--root",
        default=str(ROOT / "data" / "raw"),
        help="我递归扫描 STEP 的根目录。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "bom_batch"),
        help="我保存批量 BOM 的目录。",
    )
    parser.add_argument(
        "--master-data",
        default=None,
        help="我读取的可选 BOM 主数据 JSON。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """我跑批量 BOM 并打印汇总；无文件返回 2，有失败返回 1，全通过返回 0。"""
    args = _build_parser().parse_args(argv)
    summary = run_batch(
        Path(args.root),
        Path(args.output_dir),
        args.master_data,
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "file_count",
                    "baseline_count",
                    "review_required_count",
                    "failed_count",
                    "total_elapsed_seconds",
                    "max_elapsed_seconds",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if summary["file_count"] == 0:
        return 2
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
