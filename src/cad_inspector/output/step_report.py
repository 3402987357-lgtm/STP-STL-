"""我把 STEP BRep 拓扑结果写成 JSON，供甲方复核和后续模块读取。

原子写出（``atomic_path``）在这里不是可选的讲究：拓扑报告常被下游模块直接读取，
中断留下的半个 JSON 会导致"文件在但解析失败"，比文件不存在更难排查。
``allow_nan=False`` 同理——非有限值直接报错，不写出一个违反 JSON 规范的文件。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..io.step_topology import StepTopologyReport
from .atomic import atomic_path


def write_step_topology_json(
    result: StepTopologyReport,
    path: str | Path,
) -> Path:
    """我输出 STEP 的真实几何后端报告。"""

    path = Path(path)
    # allow_nan=False：任何非有限值都直接报错，绝不写出打不开的 JSON。
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return path
