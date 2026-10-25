"""我方分型基线命令行入口。

它把"从 STL/STEP 提取分型线和分型面候选"这条流水线包成一条命令：
读输入 -> 跑 run_parting -> 有 STEP 几何就额外写 IGS/STEP 分型 CAD ->
写 JSON/CSV/STL 报告。退出码 0 表示 BASELINE/COMPLETE，1 表示其它状态，
2 表示执行期报错。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .domain.parting import PartingConfig
from .errors import InspectorError
from .output.cad_parting import write_step_parting_cad_outputs
from .output.parting_report import write_parting_result
from .parting_pipeline import run_parting


def _build_parser() -> argparse.ArgumentParser:
    """我构造分型命令的参数表。"""

    parser = argparse.ArgumentParser(
        description="我方分型线和分型面候选生成工具。"
    )
    parser.add_argument("input", help="我方要处理的 STL 或 STEP 文件")
    parser.add_argument(
        "--output-dir",
        default="outputs/parting",
        help="我方保存结果的目录",
    )
    parser.add_argument("--plane-z", type=float, default=None)
    # scan-count 是我沿高度方向扫多少个候选截面来找分型线。
    parser.add_argument("--scan-count", type=int, default=9)
    # 分型面往外扩的余量，避免边界刚好压在零件轮廓上。
    parser.add_argument("--surface-margin", type=float, default=5.0)
    # 判断"两个点算不算重合"的截面容差，默认 1e-6 mm。
    parser.add_argument("--section-tolerance-mm", type=float, default=1e-6)
    # 每条边采多少个点来近似曲线；越大越准也越慢。
    parser.add_argument("--curve-samples-per-edge", type=int, default=8)
    parser.add_argument("--transition-offset-mm", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """我执行一次分型基线流程，并打印产物路径。"""

    # 我把 Windows 控制台切成 UTF-8，否则中文帮助和警告会乱码。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    args = _build_parser().parse_args(argv)
    try:
        config = PartingConfig(
            plane_z_mm=args.plane_z,
            plane_scan_count=args.scan_count,
            surface_margin_mm=args.surface_margin,
            section_tolerance_mm=args.section_tolerance_mm,
            curve_samples_per_edge=args.curve_samples_per_edge,
            transition_offset_mm=args.transition_offset_mm,
        )
        run = run_parting(args.input, config)
        # STEP 输入才可能有 BRep 分型几何；STL 输入这一步是空字典。
        cad_artifacts = {}
        if run.step_geometry is not None:
            cad_artifacts = write_step_parting_cad_outputs(
                run.step_geometry,
                args.output_dir,
                Path(args.input).stem,
            )
        result = run.result
        # 先写 JSON/CSV/STL，再把 CAD 产物并进 artifacts，最后一起打印。
        artifacts = write_parting_result(
            result,
            args.output_dir,
            additional_artifacts=cad_artifacts,
        )
        artifacts.update(cad_artifacts)
    except (InspectorError, OSError, ValueError) as exc:
        # 统一兜底：输入不存在、参数非法、几何处理失败都归到退出码 2。
        print(f"[ERROR] 我方分型基线失败：{exc}", file=sys.stderr)
        return 2
    print(
        f"[{result.status}] 我方完成 {result.input_file}，"
        f"line_closed={result.line.closed if result.line else False}"
    )
    for name, path in artifacts.items():
        print(f"{name}={path}")
    if result.warnings:
        print(f"warnings={len(result.warnings)}")
    return 0 if result.status in {"BASELINE", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
