"""我方注释率统计工具的边界测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "comment_ratio.py"
    spec = importlib.util.spec_from_file_location("comment_ratio", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test我方注释率统计区分整行注释文档字符串和行尾注释(
    tmp_path: Path,
) -> None:
    module = _load_module()
    (tmp_path / "sample.py").write_text(
        '"""模块说明\n第二行\n"""\n'
        "# 整行注释\n"
        "X = 1  # 行尾注释\n"
        "\n"
        "def f():\n"
        '    """函数说明"""\n'
        "    return X\n",
        encoding="utf-8",
    )

    result = module.measure([tmp_path])

    assert result["total_lines"] == 9
    assert result["full_line_comments"] == 1
    assert result["inline_comments"] == 1
    assert result["docstring_lines"] == 4
    assert result["strict_comment_lines"] == 5
    assert result["loose_comment_lines"] == 6
    assert result["definition_count"] == 1
    assert result["documented_definition_count"] == 1
    assert result["module_docstring_coverage"] == 1.0


def test我方注释率门限不达标时返回非零(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "plain.py").write_text("x = 1\n", encoding="utf-8")

    assert module.main(["--root", str(tmp_path), "--min-ratio", "0.25"]) == 1
    assert module.main(["--root", str(tmp_path), "--min-ratio", "0.0"]) == 0
