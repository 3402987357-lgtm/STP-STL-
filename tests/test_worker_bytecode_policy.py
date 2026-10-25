"""我方要求所有 worker 子进程都用 `-B` 启动，禁止把字节码写进 `.venv`。

背景：`cleanup_project_caches.ps1` 的门禁是"项目内缓存目录必须为 0"。
只要有一个 worker 不带 `-B`，numpy/scipy/trimesh 就会在 `.venv` 里重建
`__pycache__`，门禁就会假失败。这个不变量已经在真实运行里破过一次，
所以用源码级检查把它钉死。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
_SCRIPTS_WITH_WORKERS = (
    "run_real_pair_stress.py",
    "calibrate_real_thresholds.py",
    "verify_project.py",
)


def test我方所有worker子进程都用_B启动() -> None:
    for name in _SCRIPTS_WITH_WORKERS:
        lines = (ROOT / "scripts" / name).read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.strip() != "sys.executable,":
                continue
            following = [
                candidate.strip()
                for candidate in lines[index + 1 : index + 4]
                if candidate.strip() and not candidate.strip().startswith("#")
            ]
            assert following and following[0] == '"-B",', (
                f"{name}:{index + 1} 的 worker 命令缺少 -B，"
                "会把 __pycache__ 写进 .venv 并让缓存门禁假失败。"
            )
