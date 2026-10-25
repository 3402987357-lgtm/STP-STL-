"""我读取 BOM 外部主数据，并拒绝无法审计的模糊输入。

算法原理：主数据（零件编号、名称、材料、采购规格）来自甲方外部系统，是 BOM 里唯一
不能靠几何算出来的部分。它只能靠**精确键**挂载：``by_reference_shape`` 按参考 Shape
名精确匹配，``by_part_name`` 按零件名精确匹配。

我刻意不做模糊匹配：相似度 0.8 就把 A 零件的编号挂到 B 零件上，在 BOM 里是致命错误
（采购会买错东西），而且这种错误没有任何几何证据能发现。匹配不上就留空，让报告里
明确显示"主数据缺失"，交给甲方补，绝不猜。

根字段只认两个已知键，多出一个就报错——避免因字段名拼写错误导致整份主数据被静默忽略。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domain.bom_master_data import BomMasterData, BomMasterRecord
from ..errors import InputValidationError


def load_bom_master_data(path: str | Path) -> BomMasterData:
    """我只读取明确的 JSON 主数据，不把缺失字段自动补全。"""

    source = Path(path).expanduser()
    if not source.exists() or not source.is_file():
        raise InputValidationError(f"我找不到 BOM 主数据文件：{source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"我无法读取 BOM 主数据 JSON：{source}") from exc
    if not isinstance(value, dict):
        raise InputValidationError("我要求 BOM 主数据根节点必须是 JSON 对象。")
    # 未知根字段直接报错：字段名拼错后静默忽略是最难排查的一类问题。
    unknown = sorted(set(value) - {"by_reference_shape", "by_part_name"})
    if unknown:
        raise InputValidationError(
            f"我发现不支持的 BOM 主数据根字段：{', '.join(unknown)}"
        )
    return BomMasterData(
        by_reference_shape=_load_index(value.get("by_reference_shape"), "参考 Shape"),
        by_part_name=_load_index(value.get("by_part_name"), "零件名"),
    )


def _load_index(value: Any, description: str) -> dict[str, BomMasterRecord]:
    """我把一个主数据索引段解析成记录字典，逐键校验类型与空值。"""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise InputValidationError(f"我要求 BOM 主数据的{description}索引必须是 JSON 对象。")
    records: dict[str, BomMasterRecord] = {}
    for key, record in value.items():
        if not isinstance(key, str) or not key.strip():
            raise InputValidationError(f"我拒绝空的 BOM 主数据{description}键。")
        if not isinstance(record, dict):
            raise InputValidationError(f"我要求 BOM 主数据键 {key} 的值必须是 JSON 对象。")
        try:
            records[key.strip()] = BomMasterRecord.from_mapping(record)
        except ValueError as exc:
            raise InputValidationError(f"我无法解析 BOM 主数据键 {key}：{exc}") from exc
    return records
