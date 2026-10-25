"""我提供模具装配体 BOM 基线命令行入口。

算法原理：BOM 基线分三步走，每步都把不确定性留在报告里而不是藏起来：

1. 解析 STEP 装配树，把每个 Solid 拆成一个零部件记录（MD270324 这类多 Solid 装配体
   能从 1 个拆到 363 个）；
2. 可选的 BOM 主数据只做**精确参考 Shape 或零件名匹配**，不做模糊匹配——宁可匹配不上
   留空，也不用相似度猜出一个错零件名；
3. 写出 JSON/XLSX/CSV 三份，JSON 含完整证据，CSV/XLSX 供甲方扫读。

退出码约定：STATUS 为 BASELINE 返回 0，非 BASELINE 返回 1，执行报错返回 2。
"""

from __future__ import annotations

import argparse
import sys

from .errors import InspectorError
from .io.bom_master_data import load_bom_master_data
from .io.step_bom import inspect_step_bom
from .output.bom_report import write_bom_report


def _configure_utf8_stdio() -> None:
    """我把 Windows 控制台切到 UTF-8，避免 BOM 中文乱码。"""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _build_parser() -> argparse.ArgumentParser:
    """我构造 BOM 命令行解析器：装配体路径、输出目录、可选主数据。"""
    parser = argparse.ArgumentParser(
        description="我解析 STEP 装配树并生成几何 BOM 基线 JSON/XLSX/CSV。"
    )
    parser.add_argument("step", help="我读取的 .stp/.step 装配体文件。")
    parser.add_argument(
        "--output-dir",
        default="outputs/bom",
        help="我保存 BOM JSON/XLSX/CSV 的目录。",
    )
    parser.add_argument(
        "--master-data",
        help="我读取的可选 BOM 主数据 JSON；只应用精确参考 Shape 或零件名匹配。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """我解析装配体并输出 BOM；非 BASELINE 返回 1，执行报错返回 2。"""
    _configure_utf8_stdio()
    args = _build_parser().parse_args(argv)
    try:
        master_data = load_bom_master_data(args.master_data) if args.master_data else None
        report = inspect_step_bom(args.step, master_data=master_data)
        artifacts = write_bom_report(report, args.output_dir)
    except (InspectorError, OSError, ValueError) as exc:
        print(f"[ERROR] 我方 BOM 解析失败：{exc}", file=sys.stderr)
        return 2
    print(
        f"[{report.status}] 我方完成 BOM 基线："
        f"components={report.component_count}，"
        f"JSON={artifacts['json']}，CSV={artifacts['csv']}，XLSX={artifacts['xlsx']}"
    )
    return 0 if report.status == "BASELINE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
