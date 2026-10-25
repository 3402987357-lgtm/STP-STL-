"""我守住院内 Python 源码里 docstring 的结构完整性。

我为什么需要这一层：我在 step_hole._residual_boundary_length_mm 上
真的踩过一次——docstring 在开头被写成了两行，第一行是闭合的完整
docstring，第二行又开了一个三引号。语法上完全合法、compileall 不报错、
所有测试照过，但那段"为什么这么判"的说明从此变成一个被丢弃的裸
字符串常量，谁也读不到。这类缺陷只能靠结构性检查发现，
靠 review 和靠跑测试都发现不了。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = ("src", "scripts", "tests")


def _python_files() -> list[Path]:
    """我收集仓库内所有要检查的 Python 源文件。"""

    files: list[Path] = []
    for root in SEARCH_ROOTS:
        files.extend(sorted((ROOT / root).rglob("*.py")))
    return files


def _duplicated_docstring_lines(path: Path) -> list[tuple[str, int]]:
    """我找出某个文件里 docstring 被写成多段的位置。

    我为什么按"开头连续字符串表达式"判：函数体开头若连续出现两个
    及以上裸字符串，第一个才是 docstring，其余全是写错位置被丢弃的
    文字。中间穿插语句的情况不算，那种是正常代码。
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        consecutive = 0
        for statement in body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                consecutive += 1
            else:
                break
        if consecutive > 1:
            offenders.append((getattr(node, "name", "<module>"), body[1].lineno))
    return offenders


def test我方源码里没有docstring被写成多段() -> None:
    """我确认没有任何函数/类/模块的 docstring 被拆成多段。"""

    files = _python_files()
    assert files, "我没找到任何待检查的 Python 源文件，检查路径可能已失效。"
    offenders: list[str] = []
    for path in files:
        for name, lineno in _duplicated_docstring_lines(path):
            offenders.append(f"{path.relative_to(ROOT)}:{lineno} {name}")
    assert not offenders, (
        "我发现以下位置把 docstring 写成了多段，第二段起会被静默丢弃："
        + "\n".join(offenders)
    )


def test我方能解析全部源码() -> None:
    """我确认所有源文件都能被 AST 解析，顺带覆盖上一个用例的取样面。"""

    for path in _python_files():
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
