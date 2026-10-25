"""我用标准库 unittest 运行第一版回归测试，不强制要求 pytest。"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _purge_stale_bytecode() -> int:
    """我在导入被测代码前清掉项目内的 __pycache__。

    外部 AI 工具改写源码时可能保留原 mtime 和文件大小，Python 会误把旧
    .pyc 当成有效缓存加载，从而把正确的源码报成缺参之类的假失败。每次
    测试前清一次缓存，保证被测模块一定来自当前源码。
    """

    removed = 0
    for relative in ("src", "tests", "scripts"):
        base = ROOT / relative
        if not base.is_dir():
            continue
        for cache_dir in base.rglob("__pycache__"):
            if not cache_dir.is_dir():
                continue
            try:
                cache_dir.resolve().relative_to(ROOT)
            except ValueError:
                continue
            shutil.rmtree(cache_dir, ignore_errors=True)
            removed += 1
    return removed


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    purged = _purge_stale_bytecode()
    if purged:
        print(f"我已清理 {purged} 个项目字节码缓存目录，避免过期 .pyc 干扰测试。")
    try:
        import pytest  # type: ignore
    except ModuleNotFoundError:
        print(
            "我当前 Python 环境没有 pytest，只运行不依赖 pytest 的核心回归测试；"
            "如需运行全部测试，请执行 python -m pip install -r requirements.txt。"
        )
        suite = unittest.defaultTestLoader.discover(
            str(ROOT / "tests"),
            pattern="test_core.py",
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    raise SystemExit(pytest.main(["-q", str(ROOT / "tests")]))
