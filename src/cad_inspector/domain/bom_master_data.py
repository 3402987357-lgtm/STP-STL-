"""我记录 BOM 外部主数据的最小、可审计结构。

算法原理：主数据（零件名、类别、材料、采购规格）是 BOM 里唯一不能靠几何算出来的部分，
只能由甲方确认后精确挂载。所以这里的数据结构刻意做得"小且严"：

- 字段白名单，多一个就报错——字段名拼错会被立刻发现，不会被静默忽略；
- 值必须是非空字符串，空串与 None 等价处理；
- 完全为空的记录直接拒绝，不允许"挂了一条什么都没有的主数据"冒充已挂载；
- 匹配只做精确键查找（参考 Shape 优先，其次完整零件名），不做模糊匹配。BOM 里把 A 零件
  的编号挂到 B 零件上会导致采购买错东西，而且没有任何几何证据能发现这种错误。

用 frozen dataclass 保证主数据在传递过程中不会被下游意外修改。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BomMasterRecord:
    """我记录一条人工确认过的零件主数据。"""

    part_name: str | None = None
    category: str | None = None
    material: str | None = None
    purchase_specification: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BomMasterRecord":
        """我从映射构造主数据记录，逐字段校验类型、空值与未知键。"""
        allowed = {
            "part_name",
            "category",
            "material",
            "purchase_specification",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"我发现不支持的 BOM 主数据字段：{', '.join(unknown)}")
        fields: dict[str, str | None] = {}
        for key in allowed:
            item = value.get(key)
            if item is not None and (not isinstance(item, str) or not item.strip()):
                raise ValueError(f"我要求 BOM 主数据字段 {key} 必须是非空字符串。")
            fields[key] = item.strip() if isinstance(item, str) else None
        # 四个字段全空 -> 拒绝，避免"挂了一条空主数据"冒充已挂载。
        if not any(fields.values()):
            raise ValueError("我拒绝一条完全为空的 BOM 主数据记录。")
        return cls(**fields)


@dataclass(frozen=True)
class BomMasterData:
    """我记录按参考 Shape 和零件名精确索引的主数据。"""

    by_reference_shape: dict[str, BomMasterRecord]
    by_part_name: dict[str, BomMasterRecord]

    def match(self, reference_shape_id: str, part_name: str) -> BomMasterRecord | None:
        """我优先按参考 Shape 匹配，再按完整零件名匹配。"""

        return self.by_reference_shape.get(reference_shape_id) or self.by_part_name.get(
            part_name
        )
