"""我方统计项目代码注释率，为任务书"代码注释率 ≥ 25%"提供可复现证据。

原理说明：注释率没有唯一官方口径，不同甲方可能按行、按字符或按"有多少函数
带文档字符串"来数。我方不挑对自己有利的算法，而是同时给出四个指标：

1. 行注释率（严格）=（整行注释行 + 文档字符串行）÷ 全部物理行；
2. 行注释率（含行尾）= 上一项再加上"代码后面带 # 注释"的行；
3. 文档字符串覆盖率 = 带文档字符串的函数/类 ÷ 全部函数/类；
4. 模块文档字符串覆盖率 = 带模块文档字符串的文件 ÷ 全部文件。

只有同时看清这四个数，才能判断任务书的 25% 到底卡在哪一项，也方便后续 AI
按同一口径继续补注释，而不是每次换一种算法得出不同结论。
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "comment-ratio-1.0"
DEFAULT_ROOTS = ("src", "scripts")


def _docstring_line_numbers(tree: ast.AST) -> set[int]:
    """我收集模块、类、函数文档字符串覆盖的所有物理行号。

    我用 ast 而不是正则：文档字符串是"函数体里第一个表达式语句且值是
    字符串"这个语法位置，正则只能匹配引号形状，会把普通字符串赋值
    （例如把一个多行字符串赋给常量）误当成 docstring，从而虚报覆盖率。

    必须记 end_lineno 到 start 的**整段行号范围**：多行文档字符串
    只记首行会严重低估注释量，本项目里大量函数有十几行的原理说明，
    漏记会让这个指标本身失去意义。
    """

    marked: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            start = first.lineno
            end = getattr(first, "end_lineno", None) or start
            marked.update(range(start, end + 1))
    return marked


def _definition_counts(tree: ast.AST) -> tuple[int, int]:
    """我统计函数/类总数，以及其中带文档字符串的数量。"""

    total = 0
    documented = 0
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        total += 1
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            documented += 1
    return total, documented


def _has_module_docstring(tree: ast.Module) -> bool:
    """我判断模块顶层是否带文档字符串。"""

    if not tree.body:
        return False
    first = tree.body[0]
    return (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )


def _comment_line_numbers(source: str) -> tuple[set[int], set[int]]:
    """我返回（整行注释行号，行尾注释行号），两者互斥。

    我为什么用 tokenize 而不是按 '#' 切分字符串：源码里的 '#' 不一定
    是注释——字符串字面量里的 '#'（例如颜色码 '#ff0000'）会被朴素切分
    误判成注释，反过来注释里的引号也可能骗过简单的引号状态机。
    tokenize 是 Python 自己的词法分析器，它按真实语法规则区分
    字符串与注释，是这里唯一可靠的口径。

    区分整行与行尾：我用该行注释起点之前的内容是否为空来判断，
    两者互斥，所以 line_ratio_with_inline 不会把同一行重复计数。
    """

    full_lines: set[int] = set()
    inline_lines: set[int] = set()
    lines = source.splitlines()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        line_number = token.start[0]
        prefix = lines[line_number - 1][: token.start[1]]
        if prefix.strip():
            inline_lines.add(line_number)
        else:
            full_lines.add(line_number)
    return full_lines, inline_lines


def measure(roots: list[Path]) -> dict[str, object]:
    """我统计给定根目录下所有 .py 文件的注释率指标。"""

    files: list[dict[str, object]] = []
    # 分母一律是"源文件全部行数"（含空行与代码行），不做任何剔除：
    # 若把空行剔掉，多空几行就能把比率刷上去，指标就失去意义了。
    total_lines = 0
    blank_lines = 0
    full_comment = 0
    docstring_lines = 0
    inline_comment = 0
    definition_total = 0
    definition_documented = 0
    module_total = 0
    module_documented = 0
    parse_failures: list[str] = []

    for root in roots:
        # sorted 保证遍历顺序稳定：同一份代码两次运行必须得到完全相同的结果，
        # 否则 lowest_ratio_files 的排序会在文件数相同时抖动，
        # 门禁报告就无法跨运行比对。
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            line_count = len(source.splitlines())
            total_lines += line_count
            blank_lines += sum(
                1 for line in source.splitlines() if not line.strip()
            )
            # 语法错误的文件只记名不中断：统计脚本不该因为某一个文件
            # 正在被编辑就整轮失败，记进 parse_failures 能被门禁单独暴露。
            try:
                tree = ast.parse(source)
            except SyntaxError as exc:
                parse_failures.append(f"{path}: {exc}")
                continue
            doc_lines = _docstring_line_numbers(tree)
            full_lines, inline_lines = _comment_line_numbers(source)
            defs, documented = _definition_counts(tree)
            module_total += 1
            module_documented += int(_has_module_docstring(tree))
            definition_total += defs
            definition_documented += documented
            full_comment += len(full_lines)
            docstring_lines += len(doc_lines)
            inline_comment += len(inline_lines)
            marked = len(full_lines | doc_lines)
            # 整行注释与文档字符串取并集：同一行不会既是注释又是文档字符串，
            # 但用并集能保证即便口径将来重叠也不会重复计数。
            files.append(
                {
                    "path": str(path),
                    "lines": line_count,
                    "comment_lines": marked,
                    "ratio": round(marked / line_count, 6) if line_count else 0.0,
                }
            )

    strict_marked = full_comment + docstring_lines
    # 两套口径并存：strict 只算整行注释与文档字符串，
    # loose 再加上行尾注释。门禁判定用 strict——行尾注释容易靠
    # 在每行末尾加一句敷衍话刷高，用它当考核口径会让指标失真。
    loose_marked = strict_marked + inline_comment
    return {
        "schema_version": SCHEMA_VERSION,
        # 版本号随结果落盘：口径本身会演进（比如将来把行尾注释纳入考核），
        # 没有版本号就无法判断一份历史报告是按哪套口径算的。
        "roots": [str(root) for root in roots],
        "file_count": len(files),
        "total_lines": total_lines,
        # 空行数单独给出：它能解释"为什么这份报告的行数比代码行数多"，
        # 也让分母的构成透明，避免有人误以为分母被人为缩小过。
        "blank_lines": blank_lines,
        "full_line_comments": full_comment,
        "docstring_lines": docstring_lines,
        "inline_comments": inline_comment,
        "strict_comment_lines": strict_marked,
        "loose_comment_lines": loose_marked,
        "line_ratio_strict": (
            round(strict_marked / total_lines, 6) if total_lines else 0.0
        ),
        "line_ratio_with_inline": (
            round(loose_marked / total_lines, 6) if total_lines else 0.0
        ),
        # 三个覆盖率互不等价，必须都给：
        #   行比率 —— 说明整体注释密度；
        #   定义覆盖率 —— 说明函数/类有没有文档字符串；
        #   模块覆盖率 —— 说明每个文件有没有模块级说明。
        # 只看行比率会漏掉"注释全堆在少数文件里"的情况。
        "definition_count": definition_total,
        "documented_definition_count": definition_documented,
        "docstring_coverage": (
            round(definition_documented / definition_total, 6)
            if definition_total
            else 0.0
        ),
        "module_count": module_total,
        "documented_module_count": module_documented,
        "module_docstring_coverage": (
            round(module_documented / module_total, 6) if module_total else 0.0
        ),
        "parse_failures": parse_failures,
        # 最低分文件清单是这份报告最有用的部分：
        # 全局比率只说"整体差多少"，这份清单直接指出该去改哪个文件。
        # 排序键用 (ratio, -lines)：比率相同时行数多的排前面，
        # 因为改大文件的收益更高。截断到 15 条避免报告过长。
        "lowest_ratio_files": sorted(
            files, key=lambda item: (item["ratio"], -item["lines"])
        )[:15],
    }


def _build_parser() -> argparse.ArgumentParser:
    """我构造注释率工具的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="我方统计代码注释率的四个口径。"
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        # action="append" 允许重复传：一次统计多个根目录，
        # 而不必用逗号分隔再自己 split。
        help="我要统计的根目录，可重复；默认 src 与 scripts。",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        # 默认不计 tests：测试文件的注释风格与源码不同
        # （大量断言本身即说明），混进来会让指标失去可比性。
        help="我把 tests 也计入统计。",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=None,
        # 给了阈值就以退出码表达达标与否：
        # 这样门禁能直接把它当成一个通过/失败阶段来用，
        # 不必再解析报告文本。
        help="我按行注释率（严格）判定是否达标；给定时低于该值返回 1。",
    )
    parser.add_argument("--output", default=None, help="我把结果 JSON 写到该路径。")
    return parser


def main(argv: list[str] | None = None) -> int:
    """我按给定根目录统计注释率并把结果打成一目了然的表格。"""
    args = _build_parser().parse_args(argv)
    if args.root:
        roots = [Path(item) for item in args.root]
    else:
        roots = [ROOT / name for name in DEFAULT_ROOTS]
        if args.include_tests:
            roots.append(ROOT / "tests")
    result = measure(roots)
    if args.min_ratio is not None:
        result["min_ratio"] = args.min_ratio
        result["status"] = (
            "PASS"
            if float(result["line_ratio_strict"]) >= args.min_ratio
            else "BELOW_TARGET"
        )
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    print(payload, flush=True)
    if result.get("status") == "BELOW_TARGET":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
