"""我方几何特征识别命令行入口。

它只做一件事：读一个网格（STL 或已配置 OCC 的 STEP），识别上下平面候选、
法向和基础尺寸，然后把结果原子写成一个 JSON。退出码 0 表示成功，
2 表示执行期报错。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detection.features import recognize_mesh_features
from .errors import InspectorError
from .output.atomic import atomic_path
from .parting_pipeline import load_mesh_for_parting


def _build_parser() -> argparse.ArgumentParser:
    """我构造特征识别命令的参数表。"""

    parser = argparse.ArgumentParser(
        description="我方识别网格上下平面候选、法向和基础尺寸。"
    )
    parser.add_argument("input", help="我方要识别的 STL 或已配置 OCC 的 STEP 文件")
    parser.add_argument(
        "--output",
        default="outputs/features/features.json",
        help="我方保存特征结果的 JSON 文件",
    )
    # 判断一个面是否水平/垂直的法向容差，默认 15 度。
    parser.add_argument("--tolerance-deg", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """我执行一次特征识别并原子写出 JSON。"""

    args = _build_parser().parse_args(argv)
    try:
        mesh = load_mesh_for_parting(args.input)
        result = recognize_mesh_features(mesh, args.tolerance_deg)
        output = Path(args.output)
        # 我用 atomic_path 先写临时文件再替换，避免中断留下半个 JSON。
        with atomic_path(output) as temporary:
            temporary.write_text(
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
    except (InspectorError, OSError, ValueError) as exc:
        # 统一兜底：输入问题或几何问题都归到退出码 2，错误写 stderr。
        print(f"[ERROR] 我方特征识别失败：{exc}", file=sys.stderr)
        return 2
    print(f"[{result.status}] 我方完成特征识别：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
