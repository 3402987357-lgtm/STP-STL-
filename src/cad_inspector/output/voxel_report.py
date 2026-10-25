"""我把体素/截面比对结果写成 JSON 报告。

我为什么单独开一个文件而不是塞进 ``writers``：``writers`` 里的函数都
绑定检测结果的 schema（``write_json`` 吃的是完整检测结果）。体素比对
是一份**独立**的辅助证据，有自己的 schema 与自己的状态机，硬塞进
``writers`` 会让那边多出一堆"这个字段体素结果没有"的特判。

我写的 JSON 一定带 ``voxel_size_mm`` 与 ``status``：脱离了体素边长，
体积与格子数都是无法解释的裸数字；脱离了状态，一个未封闭网格漏填
出来的体积会被当成精确结论。
"""

from __future__ import annotations

import json
from pathlib import Path

from .atomic import atomic_path
from ..detection.voxel_compare import VoxelComparison


def write_voxel_json(result: VoxelComparison, path: str | Path) -> Path:
    """我把体素比对结果原子地写成 JSON，返回实际路径。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "voxel-compare-1.0",
        **result.to_dict(),
    }
    # 我走 atomic_path 而不是直接 open：写到一半被中断会留下一个能骗过
    # "文件存在"检查的坏 JSON，下游解析时才炸，排查成本远高于这里多一层包装。
    with atomic_path(target) as temporary:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return target
