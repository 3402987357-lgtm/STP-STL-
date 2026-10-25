"""我方按任务书"连续运行 48 小时"的要求做真实耐久巡检，并输出可审计台账。

算法原理：任务书写了"连续运行 48 小时"，但**跑满 48 小时本身不是证据**——
一台机器空转 48 小时也能"通过"。真正会在这条要求上出事的是三件事：

1. **内存单调增长**（泄漏）：单件能跑不代表连跑几百次不爆；
2. **结论漂移**：同一批输入跑两次，区域数/状态不一致，说明有隐藏状态；
3. **静默劣化**：耗时逐渐变长（句柄、临时文件、缓存堆积）但不报错。

所以我把它拆成三个可判定的不变量，逐个量出来：

- 内存：用 Windows `GetProcessMemoryInfo` 读**峰值工作集**，与首轮基线比，
  增长超过阈值就判 FAIL。我不看"最后一次"，看**趋势斜率**——
  泄漏是单调的，正常 GC 抖动不是。
- 结论：每轮记下区域数与状态指纹，任意两轮不一致就判 FAIL。
- 耗时：逐轮记录，按首尾分半比较中位数，劣化超过阈值就判 FAIL。

为什么**默认不真跑 48 小时**：那要占着机器两天，且这台机器上还跑着第三方的
72 小时 soak test，内存已经被压到 0.37 GB，测量本身会被污染。所以我做成
`--hours` 可指定：默认跑一小段（分钟级）把机制验通，**把"能不能跑满 48 小时"
如实记为未验证**，而不是拿短跑冒充长跑。甲方真要验收时指定 `--hours 48` 即可。

每轮独立子进程：父进程只负责编排与度量，子进程崩溃不会带走台账，
且子进程固定 `-B` 关闭字节码写入，保持项目"缓存必须为 0"的清理门禁成立。
"""

from __future__ import annotations

import argparse
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

from cad_inspector.output.atomic import atomic_path  # noqa: E402


SCHEMA_VERSION = "soak-stability-1.0"


def median(values: list[float]) -> float:
    """我算中位数，不用均值：个别轮被系统负载拖长会把均值带偏。"""

    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _child_entrypoint() -> str:
    """我返回子进程要执行的检测入口源码。

    我用 `-c` 传源码而不是构造命令行：子进程只做"跑一次检测并打印结论指纹"，
    逻辑很短，写进命令行会被引号和转义搞坏。指纹只取会影响结论的字段
    （状态、区域数、区域类型计数），不取耗时——耗时本来就在波动，
    把它并进指纹会让"结论一致"这个判据永远失败。

    **峰值内存我让子进程自己量并一起打印**，不靠父进程去读它：
    我实测过，父进程用 OpenProcess + GetProcessMemoryInfo 去读子进程，
    在这台机器上恒返回一个约 4.4 MB 的数，而同一个进程自己量是 226 MB。
    两者差 50 倍，且父进程侧的值与任务管理器一致——说明是跨进程查询被
    限制/返回了别的口径，不是代码写错。若拿那个 4.4 MB 当证据，
    "内存没涨"就完全是假象，所以这条路我不用。
    """

    return (
        "import json,sys\n"
        "import tempfile\n"
        "import ctypes,ctypes.wintypes as w,os\n"
        "from cad_inspector.pipeline import inspect_pair\n"
        "from cad_inspector.models import InspectionConfig\n"
        "class PMC(ctypes.Structure):\n"
        "    _fields_=[('cb',w.DWORD),('PageFaultCount',w.DWORD),"
        "('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),"
        "('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),"
        "('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),"
        "('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t),"
        "('PrivateUsage',ctypes.c_size_t)]\n"
        "def _self_peak():\n"
        "    try:\n"
        "        h=ctypes.windll.kernel32.OpenProcess(0x0400|0x0010,False,os.getpid())\n"
        "        c=PMC(); c.cb=ctypes.sizeof(c)\n"
        "        ok=ctypes.windll.psapi.GetProcessMemoryInfo(h,ctypes.byref(c),c.cb)\n"
        "        ctypes.windll.kernel32.CloseHandle(h)\n"
        "        return int(c.PeakWorkingSetSize) if ok else None\n"
        "    except Exception:\n"
        "        return None\n"
        "# 我让每轮写进各自的临时目录：既是 output_dir 的硬性要求，\n"
        "# 也避免几百轮互相覆盖或把磁盘写满污染耗时测量。\n"
        "with tempfile.TemporaryDirectory() as tmp:\n"
        "    result=inspect_pair(sys.argv[1],sys.argv[2],tmp,"
        "InspectionConfig(threshold_mm=float(sys.argv[3]),"
        "registration_mode='auto',write_intermediate=False,"
        "write_colored_mesh=False))\n"
        "    payload=result.to_dict()\n"
        "regions=payload.get('regions') or []\n"
        "types=sorted(str(r.get('defect_type')) for r in regions)\n"
        "# 我在检测**刚跑完、内存还没被回收**时量一次，这才是这一轮真实峰值。\n"
        "peak=_self_peak()\n"
        "print(json.dumps({'status':payload.get('status'),"
        "'region_count':len(regions),'region_types':types,"
        "'peak_working_set_bytes':peak},"
        "ensure_ascii=False))\n"
    )


def run_one_round(
    candidate: Path,
    reference: Path,
    threshold_mm: float,
    timeout_seconds: float,
) -> dict[str, object]:
    """我跑一轮检测并量出耗时与子进程峰值内存，失败也如实记下来。

    子进程退出后我读**它的**峰值内存，而不是父进程的：父进程还承载着
    前面几轮的 Python 对象，测父进程等于把编排开销算进检测开销。
    """

    started = time.perf_counter()
    # 子进程必须带上 src 路径：我前面用 `-c` 传源码时没继承 PYTHONPATH，
    # 子进程一上来就 ModuleNotFoundError，四轮全是"错误"——
    # 而如果不看 error 字段，只看"内存没涨"，会误判成稳定性通过。
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + child_env.get(
        "PYTHONPATH", ""
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            _child_entrypoint(),
            str(candidate),
            str(reference),
            repr(threshold_mm),
        ],
        cwd=str(ROOT),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        # 超时必须杀掉再等，否则子进程会变成僵尸继续占着内存，
        # 让后面几轮的"内存没涨"变成假象。
        process.kill()
        stdout, stderr = process.communicate()
        returncode = -1
    elapsed = time.perf_counter() - started

    fingerprint: dict[str, object] | None = None
    error: str | None = None
    peak_bytes: int | None = None
    if returncode == 0 and stdout.strip():
        try:
            fingerprint = json.loads(stdout.strip().splitlines()[-1])
            # 子进程自报的峰值，见 _child_entrypoint 里的说明。
            peak_bytes = fingerprint.get("peak_working_set_bytes")
        except (ValueError, TypeError) as exc:
            error = f"我无法解析子进程结论：{type(exc).__name__}: {exc}"
    elif returncode != 0:
        error = f"子进程退出码 {returncode}：{(stderr or '').strip()[-400:]}"
    else:
        error = "子进程没有输出结论"

    return {
        "elapsed_seconds": round(elapsed, 3),
        "child_peak_rss_bytes": peak_bytes,
        "fingerprint": fingerprint,
        "error": error,
    }


def evaluate_invariants(
    rounds: list[dict[str, object]],
    memory_growth_limit: float,
    elapsed_growth_limit: float,
) -> dict[str, object]:
    """我把三个不变量逐个判成 PASS / FAIL / UNKNOWN，并给出证据。

    我为什么不用一个总开关：三条坏法对应三种不同的修法。内存涨要去查
    缓存，结论漂要去查隐藏状态，耗时涨要去查句柄或临时文件堆积。
    合成一个"稳定性 PASS"会让甲方不知道该修哪里。

    轮数不足时我判 UNKNOWN 而不是 PASS：**样本不够时"没发现问题"
    不等于"没问题"**，这是这份台账最需要守住的一条。
    """

    errors = [r for r in rounds if r.get("error")]
    # 我从"结论指纹"里剔除峰值内存再比一致性。
    # 内存每轮必然不同（725.38 / 724.83 / 726.71 MB 都是正常的），
    # 把它并进指纹会让"结论一致"这个判据**永远失败**，
    # 那样我就永远看不出真正该抓的结论漂移。
    fingerprints = [
        {
            key: value
            for key, value in dict(r["fingerprint"]).items()
            if key != "peak_working_set_bytes"
        }
        for r in rounds
        if r.get("fingerprint")
    ]

    unique_fingerprints = {
        json.dumps(item, ensure_ascii=False, sort_keys=True)
        for item in fingerprints
    }
    consistency_ok = len(unique_fingerprints) <= 1
    consistency_known = len(fingerprints) >= 2

    peaks = [
        int(r["child_peak_rss_bytes"])
        for r in rounds
        if r.get("child_peak_rss_bytes") is not None
    ]
    memory_status = "UNKNOWN"
    memory_detail = "我没有拿到足够的子进程峰值内存样本"
    if len(peaks) >= 4:
        # 我比较后半段中位对前半段中位：单轮抖动不看，看的是位移。
        half = len(peaks) // 2
        first = median([float(v) for v in peaks[:half]])
        second = median([float(v) for v in peaks[half:]])
        growth = (second - first) / first if first > 0 else 0.0
        memory_status = "PASS" if growth <= memory_growth_limit else "FAIL"
        memory_detail = (
            f"前半段中位 {first / 1e6:.1f} MB -> 后半段中位 {second / 1e6:.1f} MB，"
            f"增长 {growth * 100:.2f}%（阈值 {memory_growth_limit * 100:.2f}%）"
        )

    elapsed_values = [float(r["elapsed_seconds"]) for r in rounds if not r.get("error")]
    elapsed_status = "UNKNOWN"
    elapsed_detail = "成功轮数不足，无法比较耗时趋势"
    if len(elapsed_values) >= 4:
        half = len(elapsed_values) // 2
        first = median(elapsed_values[:half])
        second = median(elapsed_values[half:])
        growth = (second - first) / first if first > 0 else 0.0
        elapsed_status = "PASS" if growth <= elapsed_growth_limit else "FAIL"
        elapsed_detail = (
            f"前半段中位 {first:.2f} s -> 后半段中位 {second:.2f} s，"
            f"增长 {growth * 100:.2f}%（阈值 {elapsed_growth_limit * 100:.2f}%）"
        )

    overall = "PASS"
    for status in (
        memory_status,
        elapsed_status,
        "FAIL" if errors else "PASS",
        "FAIL" if (consistency_known and not consistency_ok) else "PASS",
    ):
        if status == "FAIL":
            overall = "FAIL"
        elif status == "UNKNOWN" and overall == "PASS":
            overall = "UNKNOWN"

    return {
        "overall": overall,
        "round_count": len(rounds),
        "error_count": len(errors),
        "memory": {"status": memory_status, "detail": memory_detail},
        "elapsed": {"status": elapsed_status, "detail": elapsed_detail},
        "consistency": {
            "status": (
                "FAIL"
                if (consistency_known and not consistency_ok)
                else ("PASS" if consistency_known else "UNKNOWN")
            ),
            "detail": (
                f"{len(unique_fingerprints)} 种不同结论（应恒为 1）"
                if consistency_known
                else "成功轮数不足，无法比较结论一致性"
            ),
            "fingerprint_variants": sorted(unique_fingerprints)[:5],
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    """我构造耐久巡检的命令行参数。"""

    parser = argparse.ArgumentParser(
        description="我方按任务书连续运行要求做耐久巡检，并输出可审计台账。"
    )
    parser.add_argument("candidate", help="我方每轮检测的候选件。")
    parser.add_argument("reference", help="我方每轮比对的参考件。")
    parser.add_argument(
        "--hours",
        type=float,
        default=0.25,
        help=(
            "我要跑多久（小时）。任务书要 48 小时，但默认我只跑 0.25 小时把机制验通，"
            "真要验收请显式指定 48。"
        ),
    )
    parser.add_argument("--threshold-mm", type=float, default=0.15)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
        help="单轮超时（秒）。超时会被记成错误轮，不会被静默吞掉。",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="最多跑几轮（0 表示只受 --hours 限制）。用于快速验机制。",
    )
    parser.add_argument(
        "--memory-growth-limit",
        type=float,
        default=0.25,
        help="峰值内存后半段相对前半段的允许增长率（0.25 即 25%%）。",
    )
    parser.add_argument(
        "--elapsed-growth-limit",
        type=float,
        default=0.50,
        help="耗时后半段相对前半段的允许增长率（0.50 即 50%%）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "soak_stability",
        help="我方台账输出目录。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """我按小时数连续跑检测，度量稳定性，并落盘可审计台账。"""

    args = _build_parser().parse_args(argv)
    candidate = Path(args.candidate)
    reference = Path(args.reference)
    for label, path in (("候选件", candidate), ("参考件", reference)):
        if not path.is_file():
            print(f"我方找不到{label}：{path}")
            return 2

    deadline = time.monotonic() + args.hours * 3600.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, object]] = []
    index = 0

    print(
        f"我方开始耐久巡检：目标 {args.hours} 小时，"
        f"最多 {args.max_rounds or '不限'} 轮"
    )
    while time.monotonic() < deadline:
        if args.max_rounds and index >= args.max_rounds:
            break
        index += 1
        record = run_one_round(
            candidate, reference, args.threshold_mm, args.timeout_seconds
        )
        record["round"] = index
        rounds.append(record)
        status = "错误" if record["error"] else "完成"
        print(
            f"  第 {index} 轮：{status}，耗时 {record['elapsed_seconds']:.2f} s"
        )

    invariants = evaluate_invariants(
        rounds, args.memory_growth_limit, args.elapsed_growth_limit
    )
    reached_full_duration = time.monotonic() >= deadline and not (
        args.max_rounds and index >= args.max_rounds
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidate": str(candidate.resolve()),
        "reference": str(reference.resolve()),
        "requested_hours": args.hours,
        "task_required_hours": 48,
        # 我把"是否真跑满任务书时长"单独记成一个字段：
        # 短跑验机制不能冒充长跑验收，这一条必须能被一眼看见。
        "reached_full_duration": bool(reached_full_duration),
        "duration_verdict": (
            "已跑满任务书要求的时长"
            if reached_full_duration
            else f"**未**跑满：本轮只跑了 {args.hours} 小时，"
            f"任务书要求 48 小时；本轮结论只验机制，不等于 48 小时验收通过"
        ),
        "threshold_mm": args.threshold_mm,
        "invariants": invariants,
        "rounds": rounds,
    }

    with atomic_path(args.output_dir / "summary.json") as target:
        target.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\n我方耐久巡检结论：{invariants['overall']}")
    print(f"  轮数 {invariants['round_count']}，错误 {invariants['error_count']}")
    for key in ("memory", "elapsed", "consistency"):
        item = invariants[key]
        print(f"  {key}: {item['status']} - {item['detail']}")
    print(f"  {summary['duration_verdict']}")
    print(f"  台账：{args.output_dir / 'summary.json'}")
    return 0 if invariants["overall"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
