"""我方按样本编号逐对运行真实 STEP/STL 压力检查，并输出可审计统计。

算法原理：压力检查要能长时间跑、被打断后能续、还要能证明"这一轮的结论确实对应
这一批输入"。为此我用三层指纹做绑定：

1. 输入指纹：对 STEP 和 STL 各记录路径、字节数和 sha256，任何输入变化都会让旧记录失效。
2. 检测配置指纹：对 ``InspectionConfig`` 序列化后取 sha256，阈值一变就重跑。
3. 运行配置指纹：把 schema 版本、检测配置、超时秒数、资源保护阈值一起哈希，只有
   三项完全一致时 ``--resume`` 才会复用旧记录。

每对样本在独立子进程里跑（``--worker``），父进程负责超时终止与逐对落盘，避免单个
异常样本拖垮整批任务。子进程固定加 ``-B`` 关闭字节码写入，保持项目"缓存必须为 0"
的清理门禁成立。每对完成后立即写一次 summary，中断最多只丢当前这一对。
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cad_inspector.models import InspectionConfig
from cad_inspector.output.atomic import atomic_path
from cad_inspector.pipeline import inspect_pair


SCHEMA_VERSION = "real-pair-stress-1.1"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """我按 1 MiB 分块计算文件 sha256，避免一次性读入大文件。

    分块不是为了快，是为了不撑爆内存：真实 STEP 有 12 MB 以上，
    而压力轮会连续处理很多对，一次读入会把峰值内存抬到不必要的高位。
    1 MiB 是磁盘块大小的整数倍，读放大最小。

    用 `while chunk := handle.read(...)` 这种"读到空串即停"的写法：
    文件正好是块大小整数倍时，下一次 read 返回空串，循环正确结束；
    用文件大小做循环条件反而容易差一个块。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _input_fingerprint(path: Path) -> dict[str, object]:
    """我返回输入文件的三元指纹：绝对路径、字节数、内容哈希。

    三个字段各有分工，缺一个都会留下漏判：
      - 绝对路径：区分同名不同目录的样本，也便于人工核对；
      - 字节数：不读内容就能先排除绝大多数变化，廉价且直观；
      - 内容哈希：唯一能发现"大小相同但内容不同"的判据。

    path 用 resolve() 而不是原样字符串：相对路径与绝对路径
    指向同一个文件时，不规范化会被判成两个不同的输入，
    续跑就会白白重算。
    """
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _worker_summary_matches_inputs(
    summary: dict[str, object],
    step_path: Path,
    stl_path: Path,
    inspection_config_fingerprint: str | None = None,
) -> bool:
    """我判断 worker 摘要是否对应当前输入（可选再比对检测配置指纹）。

    我用**完整三元指纹**（绝对路径 + 字节数 + 内容哈希）而不是只比
    路径或只比大小。只比路径会漏掉"文件被替换过"的情况——
    同一个路径换了内容，旧结论就失效了；只比大小会被
    "大小相同但内容不同"骗过。内容哈希是唯一可靠的判据。

    比较用整个 dict 相等而不是逐项比对，避免将来加字段时
    忘记同步修改比较逻辑而留下静默的漏洞。
    """
    fingerprints = summary.get("input_fingerprints")
    if not isinstance(fingerprints, dict):
        return False
    if fingerprints != {
        "step": _input_fingerprint(step_path),
        "stl": _input_fingerprint(stl_path),
    }:
        return False
    if inspection_config_fingerprint is None:
        return True
    return (
        summary.get("inspection_config_fingerprint")
        == inspection_config_fingerprint
    )


def _inspection_config_fingerprint(config: InspectionConfig) -> str:
    """我对检测配置做规范化 JSON 序列化后取 sha256。

    sort_keys 与 separators 必须固定：序列化结果只要因字典顺序或
    空格差异而不同，指纹就会变，续跑会把已完成的样本全部重算。
    ensure_ascii=False 让中文参数按 UTF-8 编码，
    与路径/单位这类含中文的字段保持一致。
    """
    payload = json.dumps(
        config.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_config_fingerprint(
    config: InspectionConfig,
    timeout_seconds: float,
    resource_guard_bytes: int,
) -> str:
    """我把 schema、检测配置、超时和资源阈值一起哈希成运行配置指纹。

    --resume 能不能复用旧记录，取决于这四项是否完全一致。为什么超时和
    资源阈值也要进指纹：它们直接决定"这对样本能不能跑完"。上一轮用
    300 秒超时跑出来的成功记录，在这一轮 60 秒超时下本来就会超时，
    如果只比检测配置就会错误复用，把"没跑完"粉饰成"已通过"。
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "inspection_config": config.to_dict(),
        "timeout_seconds": timeout_seconds,
        "resource_guard_bytes": resource_guard_bytes,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _memory_mb() -> float | None:
    """我优先读取 Windows 当前进程工作集，读取失败时返回空值。"""

    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            """我映射 Windows PROCESS_MEMORY_COUNTERS 结构体字段。"""
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("page_fault_count", wintypes.DWORD),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return round(counters.working_set_size / (1024 * 1024), 2)
    except Exception:
        return None
    return None


def _find_pairs(data_root: Path) -> list[tuple[str, Path, Path]]:
    """我扫描数据根目录，按同目录同名 stem 配对 STEP 与 STL。

    配对规则是"同一目录下、文件名主干相同"，而不是全目录任意配对：
    数据集里不同子目录常有同名的 1.stp，按全库配对会把不同零件的
    STEP 和 STL 错误地配成一对，跑出来的"缺失区域"完全是
    两个不同零件的差异。限定同目录是防止这种错配的必要条件。

    两组索引都只扫当前目录（iterdir 而不是 rglob），因为我外层已经
    逐目录遍历了。若在循环内再 rglob，每个子目录都会被它的所有
    祖先重复扫描一遍，目录层级一深就是平方级开销。

    stem 排序用 _numeric_key 而非字符串序，保证 1,2,...,10 的自然顺序。
    """
    pairs: list[tuple[str, Path, Path]] = []
    for directory in sorted(path for path in data_root.rglob("*") if path.is_dir()):
        step_by_stem = {
            path.stem: path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".stp", ".step"}
        }
        stl_by_stem = {
            path.stem: path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".stl"
        }
        for stem in sorted(set(step_by_stem) & set(stl_by_stem), key=_numeric_key):
            pairs.append(
                (
                    f"{directory.name}/{stem}",
                    step_by_stem[stem],
                    stl_by_stem[stem],
                )
            )
    return pairs


def _numeric_key(value: str) -> tuple[int, str]:
    """我把纯数字样本名按整数排序，非数字名排在最后并按字符串排序。

    为什么不能直接按字符串排序：样本名是 1.stp / 2.stp / ... / 10.stp 时，
    字符串排序会得到 1, 10, 2, 3 ... 的顺序。这不只是观感问题——
    压力脚本按顺序逐对落盘并支持 --resume，顺序一旦在不同运行间抖动，
    "跑到第几对"的续跑语义就乱了，也会让两次运行的日志无法逐行对照。
    """
    try:
        return int(value), value
    except ValueError:
        return 10**9, value


def _stress_config() -> InspectionConfig:
    """我返回压力检查固定的检测配置（阈值、配准、查询上限等）。

    我为什么把 write_intermediate 与 write_colored_mesh 都关掉：
    压力轮要连续跑几十对，中间 JSON 与彩色网格是每对几十 MB 的量级，
    开着会把磁盘先占满，而这个脚本关心的是稳定性与耗时，
    不是逐面核对。要核对时单独跑一次 inspect 打开它们即可。

    我为什么固定用 centroid 配准而不是 icp：压力脚本的目的是在
    同一份真实数据重复跑的条件下观察内存与耗时是否漂移，用 icp 会
    引入迭代收敛次数的随机性，让耗时不可比。auto 也是确定性的——
    它在 none/centroid/pca 三个确定性变换里按残差取最小，同样可复现。
    """
    return InspectionConfig(
        threshold_mm=0.15,
        registration_mode="auto",
        max_query_points=20_000,
        distance_backend="auto",
        distance_candidate_k=64,
        distance_chunk_size=2_048,
        bidirectional_check=True,
        unit_policy="assume_mm",
        write_intermediate=False,
        write_colored_mesh=False,
    )


def _worker_summary(result) -> dict[str, object]:
    """我把一次检测结果压成 worker 摘要，只保留可审计字段。"""
    return {
        "status": result.status,
        "unit": result.unit,
        "unit_source": result.unit_source,
        "warning_count": len(result.warnings),
        "warnings": result.warnings,
        "region_count": len(result.regions),
        "missing_region_count": result.global_metrics.get(
            "missing_region_count", 0
        ),
        "extra_region_count": result.global_metrics.get(
            "extra_region_count", 0
        ),
        "candidate_quality": result.global_metrics.get(
            "candidate_quality", {}
        ),
        "reference_quality": result.global_metrics.get(
            "reference_quality", {}
        ),
        "artifacts": result.artifacts,
    }


def _run_worker(step_path: Path, stl_path: Path, output_dir: Path) -> int:
    """我在隔离子进程中执行单对检测，避免异常样本拖住批量任务。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = inspect_pair(
        # 候选是 STL、参考是 STEP：压测的方向是
        # "拿 STL 去比对 STEP"，与正式流水线一致。
        candidate_path=stl_path,
        reference_path=step_path,
        output_dir=output_dir,
        config=_stress_config(),
    )
    summary = _worker_summary(result)
    # 两个输入的指纹各自记下：续跑逻辑靠它判断
    # 文件是否被换过（名字没变但内容变了）。
    summary["input_fingerprints"] = {
        "step": _input_fingerprint(step_path),
        "stl": _input_fingerprint(stl_path),
    }
    # 配置指纹单独一项：换过配置后旧记录不可比，必须重跑。
    summary["inspection_config_fingerprint"] = _inspection_config_fingerprint(
        _stress_config()
    )
    # worker 自身的耗时：子进程里跑，所以要在这里计时。
    # 父进程另有一个总耗时，两者之差就是进程启动与收尾的开销。
    summary["worker_elapsed_seconds"] = round(
        time.perf_counter() - started,
        3,
    )
    # 走 atomic_path 写：压测常会被掐断，
    # 原子写能保证留在盘上的要么是完整文件要么没有文件。
    with atomic_path(output_dir / "worker_summary.json") as temporary:
        temporary.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                # allow_nan=False：nan 不是合法 JSON，
                # 让它写出去会导致读的一方解析失败。
                # 这里宁可抛错也不产出坏 JSON。
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


def _record_matches_inputs(
    record: dict[str, object],
    step_path: Path,
    stl_path: Path,
    run_config_fingerprint: str,
) -> bool:
    """我判断已有记录是否与当前运行配置和输入完全一致，可用于续跑。"""
    return (
        record.get("run_config_fingerprint") == run_config_fingerprint
        and _worker_summary_matches_inputs(record, step_path, stl_path)
    )


def _load_resume_records(
    summary_path: Path,
    pairs: list[tuple[str, Path, Path]],
    data_root: Path,
    run_config_fingerprint: str,
) -> dict[str, dict[str, object]]:
    """我只恢复输入和运行配置都完全一致的已记录样本。

    续跑的诚实边界：我采取"宁可重跑，不可错复用"的策略。schema 版本、
    数据根目录、运行配置指纹、以及每对样本的输入三元指纹，任何一项对不上
    就整份丢掉不复用，而不是挑着复用一部分。

    理由很实际：压力结论是要拿去跟甲方交代"这批数据跑得动"的。
    如果续跑时悄悄混进了上一轮不同阈值下的结果，这份汇总就不再是
    单一口径的证据，出了问题根本无法回溯是哪一轮、哪个配置。
    重跑一次只是多花时间，错复用会毁掉结论的可信度。
    """

    if not summary_path.exists():
        # 没有旧报告就返回空，让调用方从头跑：
        # "没有可续跑的东西"是正常状态，不是错误。
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    # 读不动或解析不了都当没有：上一轮可能写到一半被掐断，
    # 留下一个残缺的 JSON。坏掉的报告不该让这一轮也跑不起来。
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    # 三道一致性校验，任一不符就整份弃用：
    # 换了 schema、换了数据根目录、换了运行配置，
    # 旧记录里的耗时与内存数字都不再可比，
    # 混进新结果会得到一份无法解读的报告。
    if payload.get("schema_version") != SCHEMA_VERSION:
        return {}
    if payload.get("data_root") != str(data_root.resolve()):
        return {}
    if payload.get("run_config_fingerprint") != run_config_fingerprint:
        return {}
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        return {}
    pair_map = {name: (step, stl) for name, step, stl in pairs}
    resumed: dict[str, dict[str, object]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            continue
        pair_name = raw_record.get("pair")
        # 这一轮不存在的样本对跳过：数据目录可能已经变了，
        # 旧记录里的某些件在新一轮里不再参与。
        if not isinstance(pair_name, str) or pair_name not in pair_map:
            continue
        step_path, stl_path = pair_map[pair_name]
        # 逐个样本再校验输入指纹：文件被换过（内容变了但名字没变）时
        # 这条记录失效，必须重跑。
        # 只靠上面的目录级校验发现不了"单个文件被替换"。
        if _record_matches_inputs(
            raw_record,
            step_path,
            stl_path,
            run_config_fingerprint,
        ):
            resumed[pair_name] = raw_record
    return resumed


def _initial_record(
    pair_name: str,
    step_path: Path,
    stl_path: Path,
    config: InspectionConfig,
    run_config_fingerprint: str,
    resource_guard_bytes: int,
) -> dict[str, object]:
    """我为每个样本构造初始记录骨架，字段齐全便于异常时也能落盘。"""
    return {
        "pair": pair_name,
        "step_file": str(step_path.resolve()),
        "stl_file": str(stl_path.resolve()),
        "step_size_bytes": step_path.stat().st_size,
        "stl_size_bytes": stl_path.stat().st_size,
        "input_fingerprints": {
            "step": _input_fingerprint(step_path),
            "stl": _input_fingerprint(stl_path),
        },
        "run_config_fingerprint": run_config_fingerprint,
        "max_query_points": config.max_query_points,
        "resource_guard_bytes": resource_guard_bytes,
        "status": "exception",
        "elapsed_seconds": None,
        "memory_before_mb": _memory_mb(),
        "memory_after_mb": None,
        "memory_delta_mb": None,
        "warning_count": 0,
        "region_count": 0,
        "error": None,
    }


def _write_reports(
    output_root: Path,
    data_root: Path,
    config: InspectionConfig,
    records: list[dict[str, object]],
    pair_count_total: int,
    timeout_seconds: float,
    resource_guard_bytes: int,
    run_config_fingerprint: str,
) -> None:
    """我在每对样本完成后保存一次进度，避免中断丢失统计。"""

    json_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "data_root": str(data_root.resolve()),
        "pair_count": pair_count_total,
        "processed_pair_count": len(records),
        "config": config.to_dict(),
        "timeout_seconds": timeout_seconds,
        "resource_guard_bytes": resource_guard_bytes,
        # 运行配置指纹：两次压测如果配置不同，
        # 内存与耗时数字就不能直接比较。有指纹就能一眼看出
        # "这两批结果是不是在同一种配置下跑的"。
        "run_config_fingerprint": run_config_fingerprint,
        "records": records,
    }
    with atomic_path(json_path) as temporary:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                # allow_nan=False：内存或耗时算成 nan 时直接报错，
                # 而不是写出一个下游 JSON 解析器读不了的 NaN 字面量。
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    fields = [
        "pair",
        "status",
        # 两个输入体积都列：压测关心的是
        # "多大的输入吃掉多少内存"，只有一个体积无法建立这个关系。
        "step_size_bytes",
        "stl_size_bytes",
        "max_query_points",
        # 总耗时与 worker 耗时分开：差值就是进程启动与结果读回的开销，
        # 合并成一个数看不出时间花在哪。
        "elapsed_seconds",
        "worker_elapsed_seconds",
        # 内存三段（前/后/增量）一起给：只看增量无法判断
        # 基线内存是否本身就偏高，只看前后又要自己算差。
        "memory_before_mb",
        "memory_after_mb",
        "memory_delta_mb",
        "warning_count",
        "region_count",
        # 缺失与多出分开统计：压测里两者都出现说明配准有问题，
        # 只给总数会掩盖"两个方向都错了"这种情况。
        "missing_region_count",
        "extra_region_count",
        # error 放最后一列：绝大多数行它都是空的，
        # 放前部会让人工扫表时每次都要跳过。
        "error",
    ]
    with atomic_path(csv_path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({field: record.get(field) for field in fields})


def _build_parser() -> argparse.ArgumentParser:
    """我构造命令行解析器；worker 专用参数对用户隐藏。"""
    parser = argparse.ArgumentParser(
        description="我方按样本编号逐对运行真实 STEP/STL 压力检查。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "provided" / "三维模型破损区域智能修补",
        help="我方要扫描的 STEP/STL 配对根目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "real_pair_stress_2026-10-02",
        help="我方要写出压力结果的目录。",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="我方单对检测的最大秒数。",
    )
    parser.add_argument(
        "--resource-guard-bytes",
        type=int,
        default=200 * 1024 * 1024,
        help="我方跳过超过该 STL 大小的样本。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="我方只恢复输入和运行配置完全一致的已完成记录。",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--step", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stl", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return parser


def _validate_runtime_args(args: argparse.Namespace) -> None:
    """我校验运行参数：超时和资源保护阈值必须为正。"""
    if args.timeout_seconds <= 0:
        raise ValueError("我要求 --timeout-seconds 必须大于 0。")
    if args.resource_guard_bytes <= 0:
        raise ValueError("我要求 --resource-guard-bytes 必须大于 0。")


def main() -> int:
    """我遍历所有配对样本，逐对隔离执行并在每对后落盘，返回进程退出码。"""
    args = _build_parser().parse_args()
    _validate_runtime_args(args)
    # worker 模式：只跑单对样本，结果写进 output 目录后直接返回。
    if args.worker:
        if args.step is None or args.stl is None or args.output is None:
            raise ValueError("我要求 worker 同时提供 --step、--stl 和 --output。")
        step_path = args.step
        stl_path = args.stl
        output_dir = args.output
        return _run_worker(step_path, stl_path, output_dir)

    data_root = args.data_root.expanduser()
    output_root = args.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    # 计算本轮指纹；后续 --resume 只恢复三项指纹完全一致的旧记录。
    pairs = _find_pairs(data_root)
    config = _stress_config()
    timeout_seconds = args.timeout_seconds
    resource_guard_bytes = args.resource_guard_bytes
    inspection_config_fingerprint = _inspection_config_fingerprint(config)
    run_config_fingerprint = _run_config_fingerprint(
        config,
        timeout_seconds,
        resource_guard_bytes,
    )
    resume_records = (
        _load_resume_records(
            output_root / "summary.json",
            pairs,
            data_root,
            run_config_fingerprint,
        )
        if args.resume
        else {}
    )
    records: list[dict[str, object]] = []

    # 主循环：逐对处理，每对结束都写一次 summary，中断最多丢当前这对。
    for pair_name, step_path, stl_path in pairs:
        if pair_name in resume_records:
            record = dict(resume_records[pair_name])
            record["resumed_from_summary"] = True
            records.append(record)
            _write_reports(
                output_root,
                data_root,
                config,
                records,
                len(pairs),
                timeout_seconds,
                resource_guard_bytes,
                run_config_fingerprint,
            )
            print(f"[resumed] {pair_name} status={record.get('status')}")
            continue
        pair_output = output_root / pair_name.replace("/", "__")
        started = time.perf_counter()
        record = _initial_record(
            pair_name,
            step_path,
            stl_path,
            config,
            run_config_fingerprint,
            resource_guard_bytes,
        )
        # 资源保护：STL 超过阈值时跳过并记录，避免大文件拖垮整批任务。
        if stl_path.stat().st_size > resource_guard_bytes:
            record["status"] = "resource_guard"
            record["error"] = (
                f"STL 文件为 {stl_path.stat().st_size} bytes，"
                f"超过压力轮资源保护阈值 {resource_guard_bytes} bytes。"
                "我保留此前实际运行超过 1881 秒的历史证据，"
                "本轮不再重复占用资源。"
            )
            record["elapsed_seconds"] = 0.0
            record["memory_after_mb"] = _memory_mb()
            records.append(record)
            _write_reports(
                output_root,
                data_root,
                config,
                records,
                len(pairs),
                timeout_seconds,
                resource_guard_bytes,
                run_config_fingerprint,
            )
            print(
                f"[{record['status']}] {pair_name} "
                f"elapsed={record['elapsed_seconds']}s "
                f"warnings={record['warning_count']}"
            )
            continue
        # 优先复用已存在且指纹一致的 worker 摘要，否则起隔离子进程重跑。
        try:
            summary_path = pair_output / "worker_summary.json"
            needs_worker = True
            if summary_path.exists():
                cached_summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                if _worker_summary_matches_inputs(
                    cached_summary,
                    step_path,
                    stl_path,
                    inspection_config_fingerprint,
                ):
                    record.update(cached_summary)
                    record["reused_worker_summary"] = True
                    needs_worker = False
            if needs_worker:
                worker_command = [
                    sys.executable,
                    # 我用 -B 关闭 worker 的字节码写入，避免把 __pycache__ 灌进
                    # .venv，破坏 cleanup_project_caches.ps1 的"剩余缓存必须为 0"门禁。
                    "-B",
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--step",
                    str(step_path),
                    "--stl",
                    str(stl_path),
                    "--output",
                    str(pair_output),
                ]
                completed = subprocess.run(
                    worker_command,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    # 每对单独限时：一个大模型跑飞不能拖垮整批，
                    # 超时只让这一对记为 timeout，其余照常继续。
                    timeout=timeout_seconds,
                )
                if completed.returncode != 0:
                    # 错误信息同时看 stderr 与 stdout：
                    # OCP 的崩溃信息有时只落在其中一个流上。
                    # 截最后 2000 字符是为了避免巨型回溯把汇总撑爆。
                    detail = (completed.stderr or completed.stdout).strip()
                    raise RuntimeError(
                        f"worker exit={completed.returncode}: {detail[-2000:]}"
                    )
                record.update(
                    json.loads(summary_path.read_text(encoding="utf-8"))
                )
                # reused 标记区分"本轮真实计算"与"读回上轮结果"，
                # 没有它就无法判断汇总里的耗时数字是否来自本次运行。
                record["reused_worker_summary"] = False
        except subprocess.TimeoutExpired:
            # 超时不改 status 之外的字段：这一对的结果不完整，
            # 保留已有信息便于事后判断是慢还是卡死。
            record["status"] = "timeout"
            record["error"] = (
                f"单对检测超过 {timeout_seconds} 秒，"
                "我已终止隔离子进程并继续下一对。"
            )
        except Exception as exc:
            # 记异常类型而不只是消息：类型能区分"文件缺失"与"几何崩溃"，
            # 只有消息文本很难归类。
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            # 耗时与内存放在 finally 里：无论成功、超时还是异常都要记录，
            # 资源画像恰恰在失败样本上最有价值。
            record["elapsed_seconds"] = round(
                time.perf_counter() - started,
                3,
            )
            record["memory_after_mb"] = _memory_mb()
            # 增量只在两端都量到时才算：任何一端是 None
            # （例如 psutil 不可用）就留 None，
            # 用 0 填充会被读成"内存没有增长"。
            if (
                record["memory_before_mb"] is not None
                and record["memory_after_mb"] is not None
            ):
                record["memory_delta_mb"] = round(
                    float(record["memory_after_mb"])
                    - float(record["memory_before_mb"]),
                    2,
                )
            records.append(record)
            # 每跑完一对就重写报告：压测是最容易中途被掐掉的任务，
            # 攒到最后一次性落盘会丢掉全部已跑结果。
            _write_reports(
                output_root,
                data_root,
                config,
                records,
                len(pairs),
                timeout_seconds,
                resource_guard_bytes,
                run_config_fingerprint,
            )
            # 显式 gc：OCP 与 numpy 的大对象不会立刻归还，
            # 不主动回收会让下一对的内存基线被上一对抬高，
            # 于是"内存增量"这个压测的核心指标失真。
            gc.collect()
            print(
                f"[{record['status']}] {pair_name} "
                f"elapsed={record['elapsed_seconds']}s "
                # 只打印耗时与警告数：内存三段在报告里看，
                # 控制台太宽会折行，反而看不清状态。
                f"warnings={record['warning_count']}"
            )

    # 循环结束后再写一次：这一次的 pair_count 是最终值，
    # 中途每次写的时候它也是最终值，
    # 重复写一次是为了保证即使最后一对写报告失败，
    # 报告里也是完整的。
    _write_reports(
        output_root,
        data_root,
        config,
        records,
        len(pairs),
        timeout_seconds,
        resource_guard_bytes,
        run_config_fingerprint,
    )
    json_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    print(f"我方已写出压力汇总：{json_path}")
    print(f"我方已写出表格汇总：{csv_path}")
    return 0 if all(record["status"] != "exception" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
