"""我方方便在未安装项目包时直接执行的 STEP 探测脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cad_inspector.cli import main  # noqa: E402


if __name__ == "__main__":
    # 我保留旧的直接调用方式：用户不写子命令时，我自动进入 probe。
    arguments = sys.argv[1:]
    if not arguments or arguments[0] not in {"inspect", "validate", "probe"}:
        arguments.insert(0, "probe")
    raise SystemExit(main(arguments))

