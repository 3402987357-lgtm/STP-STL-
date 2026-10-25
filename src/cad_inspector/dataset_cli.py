"""我方数据清单命令行入口。

算法原理：训练与验收前必须先知道"手里到底有哪些样本"，所以我离线扫描数据目录，
对每个样本记录路径、名称、后缀、字节数、sha256、探测出的格式、数据集划分与标注
状态。清单本身带内容摘要，后续任何一轮训练都能回指到这份清单，证明用的是同一批数据。

本入口只做编排：调用 ``build_manifest`` 扫描，再调用 ``write_manifest`` 落盘 JSON
与 CSV，自身不含业务逻辑。出错时打印带前缀的错误并返回退出码 2，与项目其他 CLI
的退出码约定保持一致（0 成功 / 1 业务失败 / 2 执行报错）。
"""

from __future__ import annotations

import argparse
import sys

from .io.dataset_manifest import build_manifest
from .output.parting_report import write_manifest


def _build_parser() -> argparse.ArgumentParser:
    """我构造数据清单命令行解析器：一个位置参数加一个输出路径。"""
    parser = argparse.ArgumentParser(
        description="我方离线扫描 STEP、XT、PRT 和 STL 样本并生成数据清单。"
    )
    parser.add_argument("root", help="我方要扫描的数据目录")
    parser.add_argument(
        "--output",
        default="outputs/dataset_manifest.json",
        help="我方保存 JSON 清单的路径",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """我扫描数据目录并写出清单；失败时返回退出码 2。"""
    args = _build_parser().parse_args(argv)
    try:
        manifest = build_manifest(args.root)
        target = write_manifest(manifest, args.output)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] 我方生成数据清单失败：{exc}", file=sys.stderr)
        return 2
    print(
        f"[COMPLETE] 我方登记 {len(manifest.records)} 个样本，"
        f"JSON={target}，CSV={target.with_suffix('.csv')}"
    )
    if manifest.warnings:
        print(f"[WARNING] 我方记录了 {len(manifest.warnings)} 个目录级问题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
