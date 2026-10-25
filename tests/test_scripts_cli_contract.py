"""我钉住「脚本当 CLI 用能起来」，而不只是「当模块导入能起来」。

背景（实测）：项目 2026-10-18 整体搬进大赛目录后，`.venv` 里的可编辑安装
``__editable__...pth`` 仍指向已不存在的旧路径
`D:\\STP_STL_Missing_Region_Recognition\\src`，于是任何
`python -m cad_inspector...` 都报 ModuleNotFoundError。项目自己的办法是：
每个脚本/测试入口自己把 `<root>/src` 插进 sys.path。17 个脚本这么做了，
但有 3 个漏了，其中 `bootstrap_annotations.py` 正是《验收操作手册》第 2 步，
`measure_real_budget.py` 是耗时预算证据的生产者。

为什么既有用例盖不住：`tests/test_bootstrap_annotations.py` 是把脚本当
**模块**导入（先 `sys.path.insert(SCRIPTS_DIR)`），模块导入能成功，于是
「当命令行跑会崩」这件事一路绿灯。所以我在这里用**子进程**跑，
并且明确**清掉 PYTHONPATH**——手动执行手册的人不会有这个变量。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# 我只列"带 --help 且不会真干活"的入口：跑 --help 必须只打印帮助就退出。
_CLI_ENTRY_SCRIPTS = (
    "bootstrap_annotations.py",
    "measure_real_budget.py",
)


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


@pytest.mark.parametrize("script", _CLI_ENTRY_SCRIPTS)
def test我方脚本在不设PYTHONPATH时也能打印帮助(script: str) -> None:
    """我钉住手册里的脚本命令照抄就能跑。"""

    path = ROOT / "scripts" / script
    assert path.is_file(), f"找不到脚本：{path}"
    result = subprocess.run(
        [sys.executable, "-B", str(path), "--help"],
        cwd=str(ROOT),
        env=_clean_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    assert result.returncode == 0, (
        f"{script} 在未设置 PYTHONPATH 时退出码 {result.returncode}；"
        f"stderr 末尾：{result.stderr[-800:]}"
    )
    assert "usage" in (result.stdout + result.stderr).lower(), (
        f"{script} 没有打印帮助：{result.stdout[:300]!r}"
    )