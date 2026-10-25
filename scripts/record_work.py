"""我方开发过程 TXT 记录工具。

我把每次代码、配置、报告或输出文件的写入动作追加到 D 盘日志中，
让后续开发可以追溯“改了什么、为什么改、怎样验证、还有什么限制”。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """我构造开发过程记录工具的命令行参数。"""
    parser = argparse.ArgumentParser(description="我方追加开发过程 TXT 记录。")
    parser.add_argument("--log", required=True, help="我方要追加的 TXT 日志路径")
    parser.add_argument("--title", required=True, help="本次记录标题")
    parser.add_argument("--status", default="已记录", help="本次记录状态")
    parser.add_argument("--paths", nargs="*", default=[], help="涉及的文件路径")
    parser.add_argument("--details", required=True, help="我方完成的详细内容")
    parser.add_argument("--verification", default="尚未执行", help="验证命令和结果")
    parser.add_argument("--limitations", default="无新增限制", help="已知限制")
    return parser


def append_record(
    log_path: str | Path,
    title: str,
    status: str,
    paths: list[str],
    details: str,
    verification: str,
    limitations: str,
) -> Path:
    """我以 UTF-8 追加一条可读的开发记录。

    我强制 encoding="utf-8" 且 newline="\\n"：Windows 上文本模式默认
    用本地编码并做 CRLF 转换，中文内容会在某些查看器里变乱码，
    换行也会被写成 \\r\\n 造成不同平台下 diff 全量变动。显式指定
    才能让这份记录在任何环境打开都是同一份内容。

    时间戳用 UTC 转本地时区并带时区偏移（isoformat），而不是裸的本地
    时间：跨时区协作时"19:04"没有时区信息就无法判断先后顺序，
    带偏移的时间戳可以直接排序和对照。

    记录块结构是固定的七段（时间戳/标题/状态/涉及文件/完成的事/验证/
    限制），其中"我保留的限制"是强制项：只写成果不写限制的记录会让人
    误以为工作已经彻底完成，这比不记录更有害。
    """

    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    path_lines = "\n".join(f"- {path}" for path in paths) or "- 未指定文件"
    block = (
        f"\n\n===== 开发记录 {timestamp} =====\n"
        f"标题：{title}\n"
        f"状态：{status}\n"
        f"涉及文件：\n{path_lines}\n"
        f"我完成的事情：\n{details}\n"
        f"我执行的验证：\n{verification}\n"
        f"我保留的限制：\n{limitations}\n"
    )
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(block)
    return target


def main(argv: list[str] | None = None) -> int:
    """我把一条开发记录追加到 TXT 日志文件末尾。"""
    args = _build_parser().parse_args(argv)
    append_record(
        log_path=args.log,
        title=args.title,
        status=args.status,
        paths=args.paths,
        details=args.details,
        verification=args.verification,
        limitations=args.limitations,
    )
    print(f"我方已追加开发记录：{Path(args.log).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
