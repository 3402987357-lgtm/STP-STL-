"""我把 STEP/STP 转成 STL，方便我方生成参考网格和检查三角化质量。

这个脚本只有一个职责：用 OCCT 按给定的线性/角度偏差把 STEP 三角化，
然后写成一个命名正确的 STL。三角化参数直接决定 STL 的疏密，
所以我把它暴露成命令行参数，方便甲方在"精度"和"体积"之间取舍。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cad_inspector.geometry.stl import save_stl
from cad_inspector.io.step import load_step_as_mesh
from cad_inspector.errors import InspectorError


def _configure_utf8_stdout() -> None:
    """我把 stdout 切成 UTF-8，避免中文提示在 Windows 控制台乱码。"""

    stream = getattr(sys, "stdout", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main() -> int:
    """我解析参数、执行转换，并把顶点/三角面数量打印出来。"""

    _configure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="我用 OCCT 读取 STEP 并输出 STL。",
    )
    parser.add_argument("step", help="我读取的 .step 或 .stp 文件。")
    parser.add_argument("output", help="我写出的 .stl 文件。")
    parser.add_argument(
        "--linear-deflection-mm",
        type=float,
        default=0.04,
        help="我控制的线性三角化偏差，默认 0.04 mm。",
    )
    # 角度偏差控制曲面被切成多少段：越小圆孔越圆，但三角面数会上升。
    parser.add_argument(
        "--angular-deflection-deg",
        type=float,
        default=10.0,
        help="我控制的角度三角化偏差，默认 10 度。",
    )
    args = parser.parse_args()
    try:
        # 先读成内存网格，再写 STL；两步都可能因为文件损坏或格式不支持而失败。
        mesh = load_step_as_mesh(
            args.step,
            linear_deflection_mm=args.linear_deflection_mm,
            angular_deflection_deg=args.angular_deflection_deg,
        )
        save_stl(mesh, args.output, Path(args.step).stem)
    except (InspectorError, OSError, ValueError) as exc:
        # 失败一律写 stderr 并返回 2，保证调用方能和"转换成功"区分开。
        print(f"[ERROR] 我方 STEP 转 STL 失败：{exc}", file=sys.stderr)
        return 2
    print(
        "我已经完成 STEP 到 STL 的转换："
        f"{args.output}，顶点 {len(mesh.vertices)}，三角面 {len(mesh.faces)}，"
        f"单位提示 {mesh.metadata.get('unit', 'unknown')}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
