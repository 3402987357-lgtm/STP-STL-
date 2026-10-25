"""我让用户不需要先安装项目包，也能从项目根目录运行命令行。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cad_inspector.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
