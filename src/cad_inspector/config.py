"""我负责读取 JSON 配置，并把参数转换成强类型配置对象。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import InspectionConfig


def load_config(path: str | Path | None = None) -> InspectionConfig:
    """我优先读取 JSON；如果没有配置文件，就使用项目安全默认值。

    path 为 None 时直接返回默认配置，不读任何文件。这是有意的：
    命令行没传 --config 时应该走内置安全默认值，而不是去找一个
    可能存在的"隐式默认配置文件"——隐式文件会让不同机器跑出
    不同的结果，而没人意识到配置参与了。

    顶层必须校验是不是 JSON 对象：一个数组或标量文件能被 json.loads
    成功解析，但传给 from_dict 后会在深层报出难以定位的错。
    在这里先拦住，错误信息才指向真正的问题。
    """

    if path is None:
        return InspectionConfig()
    path = Path(path)
    values: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("我要求配置文件顶层必须是 JSON 对象。")
    return InspectionConfig.from_dict(values)
