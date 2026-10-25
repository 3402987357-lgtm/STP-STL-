"""我方可视化与对比分析命令行入口（任务书核心任务四）。

算法原理：可视化最容易骗人——一张标红的图看起来很有说服力，但高亮范围其实受配准
残差影响。所以我默认从检测结果里提炼一行真实质量说明（状态、配准质量、平均残差、
阈值）作为副标题，让看图的人同时看到"这个红框有多可信"。用户显式传了 ``--subtitle``
才覆盖，默认不隐藏这些信息。

区域 JSON 有两种 schema（检测结果 inspection、STEP 修补 step-repair），用
``--regions-kind`` 显式区分，不靠猜字段猜出是哪种。STEP 修补区域还可以配合
``--face-features`` 用来源 Face 包围盒定位，否则修补报告里只有 Face 编号、画不出框。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import InspectorError
from .output.visualization import (
    face_bboxes_from_step_face_features,
    highlight_regions_from_inspection,
    highlight_regions_from_step_repair,
    write_comparison_report,
)
from .parting_pipeline import load_mesh_for_parting


def _configure_utf8_stdio() -> None:
    """我把 Windows 控制台切到 UTF-8，避免中文路径和分类名乱码。

    我用 getattr 探测 reconfigure 而不是直接调用：这个方法在
    Python 3.7 之后才有，而且当 stdout 被重定向到非 TextIOWrapper
    （例如被 pytest 捕获、或被 IDE 替换）时并不存在。直接调用会在
    这些环境里抛 AttributeError，让一个"改善显示"的功能变成崩溃源。

    errors="replace" 而非默认严格模式：控制台编码切换失败的边缘情况下，
    严格模式会抛 UnicodeEncodeError 中断输出，而这里只是提示信息，
    损失个别字符远好于整个命令失败。

    OSError/ValueError 都要吞掉：控制台被关闭、或流已被 detach 时
    reconfigure 会抛这两类错，此时保持原样继续即可。
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _build_parser() -> argparse.ArgumentParser:
    """我构造可视化命令行解析器：前后模型、区域 JSON、输出目录与标题。"""
    parser = argparse.ArgumentParser(
        description=(
            "我输出破损区域高亮、修复前后对比和分类标记的可视化报告。"
        )
    )
    parser.add_argument(
        "--before",
        required=True,
        # 只有 before 是必填：单模型可视化（只看分类标记）是合法的，
        # 强制要求 after 会让"只看看这个模型"变得不可能。
        help="我方要展示的修复前模型（STL/STP/STEP）。",
    )
    parser.add_argument(
        "--after",
        help="我方可选展示的修复后模型（STL/STP/STEP）。",
    )
    parser.add_argument(
        "--regions",
        help="我方读取的区域 JSON：缺失区域检测或 STEP 修补报告。",
    )
    parser.add_argument(
        "--regions-kind",
        choices=("inspection", "step-repair"),
        # 区域来源必须显式声明：两种报告的 schema 不同，
        # 自动猜会在字段恰好重名时静默解析错。
        default="inspection",
        help="我方按哪种 schema 解析 --regions。",
    )
    parser.add_argument(
        "--face-features",
        help=(
            # 面特征 JSON 是画 STEP 修补区域的必要条件：
            # 修补报告里只有面号，没有坐标，
            # 要靠这份文件把面号映射回空间位置。
            "我方可选的 step-face-features JSON；"
            "用来源 Face 包围盒把 STEP 修补区域画到图上。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visualization",
        help="我方保存 PNG 和 HTML 的目录。",
    )
    parser.add_argument(
        "--title",
        default="我方破损区域检测与修补对比",
        help="我方写入图片和 HTML 的标题。",
    )
    parser.add_argument(
        "--subtitle",
        help=(
            "我方可选副标题；不填时按检测状态和配准质量自动生成，"
            "提醒甲方高亮范围是否受配准残差影响。"
        ),
    )
    parser.add_argument("--dpi", type=int, default=120)
    return parser


def _read_json(path: str | Path) -> dict:
    """我按 UTF-8 读取 JSON，并在结构异常时给出清晰错误。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"我要求 {path} 的顶层是 JSON 对象。")
    return payload


def _load_regions(args: argparse.Namespace):
    """我按 --regions-kind 把区域 JSON 转成可视化区域。"""

    # 没传区域就返回空区域与 None 副标题，由调用方回落到自动生成的副标题。
    if not args.regions:
        return [], None
    payload = _read_json(args.regions)
    if args.regions_kind == "step-repair":
        face_bboxes = None
        if args.face_features:
            face_bboxes = face_bboxes_from_step_face_features(
                _read_json(args.face_features)
            )
        regions = highlight_regions_from_step_repair(payload, face_bboxes)
        subtitle = f"修补报告状态：{payload.get('status', 'UNKNOWN')}"
        return regions, subtitle
    regions = highlight_regions_from_inspection(payload)
    return regions, _derive_inspection_subtitle(payload)


def _derive_inspection_subtitle(payload: dict) -> str:
    """我从检测结果里提炼一行真实质量说明，避免高亮范围被误读。

    可视化最容易骗人的地方：一张标红的图看着很有说服力，但红框的
    范围实际受配准残差支配——两个模型没对齐时，红框标的是配准误差
    而不是真实缺陷。所以我把状态、配准质量、平均残差、阈值一起
    写进副标题，让看图的人同时看到"这个红框有多可信"。

    每个字段都先判存在再拼接，缺失就整个略去。我不写 "配准质量：None"
    这种占位文本——它看起来像"质量是 None 这个结论"，
    比直接不显示更具误导性。
    """

    parts = [f"检测状态：{payload.get('status', 'UNKNOWN')}"]
    registration = payload.get("registration") or {}
    quality = registration.get("quality")
    residual = registration.get("mean_residual_mm")
    if quality is not None:
        parts.append(f"配准质量：{quality}")
    if isinstance(residual, (int, float)):
        parts.append(f"平均残差：{residual:.3f} mm")
    threshold = payload.get("threshold_mm")
    if isinstance(threshold, (int, float)):
        parts.append(f"阈值：{threshold:.3f} mm")
    return "；".join(parts)


def main(argv: list[str] | None = None) -> int:
    """我生成对比可视化报告；参数非法或执行失败时返回退出码 2。"""
    _configure_utf8_stdio()
    args = _build_parser().parse_args(argv)
    # DPI 下限 30：再低就画不出可辨认的对比图，不如直接拒绝。
    if args.dpi < 30:
        print("[ERROR] 我方要求 --dpi 至少为 30。", file=sys.stderr)
        return 2
    try:
        before = load_mesh_for_parting(args.before)
        after = load_mesh_for_parting(args.after) if args.after else None
        regions, derived_subtitle = _load_regions(args)
        artifacts = write_comparison_report(
            before,
            after,
            regions,
            args.output_dir,
            title=args.title,
            subtitle=args.subtitle or derived_subtitle,
            dpi=args.dpi,
        )
    except (InspectorError, OSError, ValueError, KeyError) as exc:
        print(f"[ERROR] 我方可视化失败：{exc}", file=sys.stderr)
        return 2
    print(
        f"[COMPLETE] 我方完成可视化：regions={len(regions)}，"
        f"HTML={artifacts['html']}，"
        f"对比图={artifacts['comparison_png']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
