"""我实测真实大件的端到端耗时，并把结果写成报告。

为什么要有这个脚本：门禁里的用例都用合成小件跑，
从来没在真实 STEP 上量过一次。"应该够快"不是证据。
我把它固化成脚本，这样预算是可以被反复验证的，
而不是某一次口头结论。

用法：
    python scripts/measure_real_budget.py [--limit N] [--budget 秒]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 我把项目根的 src 插进 sys.path：本脚本作为 CLI 直接运行时（而非经
# pytest 的 conftest 导入时）不会自动带上 src，import cad_inspector
# 会 ModuleNotFoundError。参照 scripts/run_real_pair_acceptance.py:27 的写法。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cad_inspector.detection.step_features import (
    inspect_step_feature_classification,
)

DEFAULT_BUDGET = 8.0
SUPPORTED = {".stp", ".step"}


def collect_files(root: Path) -> list[Path]:
    """我按体积从大到小收集真实 STEP 文件。"""

    found = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED
    ]
    return sorted(found, key=lambda path: path.stat().st_size, reverse=True)


def main() -> int:
    """我测量每件耗时并输出报告，返回 0 表示脚本跑完。"""

    parser = argparse.ArgumentParser(description="我实测真实大件端到端耗时。")
    parser.add_argument("--limit", type=int, default=8, help="我最多测几件。")
    parser.add_argument(
        "--budget", type=float, default=DEFAULT_BUDGET, help="我用的单件预算（秒）。"
    )
    parser.add_argument(
        "--root",
        default="data/raw",
        help="我从哪个目录收集真实 STEP 文件。",
    )
    parser.add_argument(
        "--report",
        default="reports/real_budget.json",
        help="我把结果写到哪里。",
    )
    args = parser.parse_args()

    files = collect_files(Path(args.root))
    if not files:
        print(f"我在 {args.root} 下没有找到 STEP 文件。")
        return 1

    records = []
    print(f"单件预算 {args.budget}s，目录 {args.root}，取最大的 {args.limit} 件")
    print(f"{'文件':<34s} {'大小MB':>8s} {'耗时s':>9s} {'区域数':>7s} {'达标':>6s}")
    for path in files[: args.limit]:
        size_mb = path.stat().st_size / 1e6
        started = time.perf_counter()
        try:
            report = inspect_step_feature_classification(path)
            elapsed = time.perf_counter() - started
            # JSON 里不允许 NaN，写不出就说明这份结果本身不可交付。
            nan_free = True
            try:
                json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False)
            except ValueError:
                nan_free = False
            ok = elapsed <= args.budget and nan_free
            records.append(
                {
                    "file": path.name,
                    "size_mb": round(size_mb, 3),
                    "elapsed_seconds": round(elapsed, 3),
                    "region_count": len(report.regions),
                    "nan_free": nan_free,
                    "within_budget": ok,
                }
            )
            print(
                f"{path.name[:32]:<34s} {size_mb:8.2f} {elapsed:9.1f} "
                f"{len(report.regions):7d} {'是' if ok else '否':>6s}"
            )
        except Exception as error:
            print(f"{path.name[:32]:<34s} {size_mb:8.2f}  失败 {type(error).__name__}")
            records.append(
                {
                    "file": path.name,
                    "size_mb": round(size_mb, 3),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    measured = [r for r in records if "elapsed_seconds" in r]
    within = [r for r in measured if r["within_budget"]]
    summary = {
        "budget_seconds": args.budget,
        "measured_count": len(measured),
        "within_budget_count": len(within),
        "within_budget_ratio": (
            round(len(within) / len(measured), 6) if measured else None
        ),
        "elapsed_seconds": {
            "min": min(r["elapsed_seconds"] for r in measured) if measured else None,
            "max": max(r["elapsed_seconds"] for r in measured) if measured else None,
            "mean": (
                round(
                    sum(r["elapsed_seconds"] for r in measured) / len(measured), 3
                )
                if measured
                else None
            ),
        },
        "records": records,
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if measured:
        print(
            f"\n达标 {len(within)}/{len(measured)}，"
            f"最快 {summary['elapsed_seconds']['min']:.1f}s，"
            f"最慢 {summary['elapsed_seconds']['max']:.1f}s，"
            f"均值 {summary['elapsed_seconds']['mean']:.1f}s"
        )
    print(f"报告：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
