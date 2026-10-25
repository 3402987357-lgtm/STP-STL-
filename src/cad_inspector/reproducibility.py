"""我方给正式报告提供跨运行稳定的内容摘要，用于复现校验。"""

from __future__ import annotations

import hashlib
import json
from typing import Any


# 我方把只反映本次运行耗时的字段排除在内容摘要之外。
VOLATILE_FIELDS: frozenset[str] = frozenset({"elapsed_seconds"})
CONTENT_DIGEST_FIELD = "content_sha256"


def stable_payload(value: Any) -> Any:
    """我递归去掉运行耗时字段和旧摘要，返回可复现负载。

    我必须剔除两类字段，否则"同一输入两次运行摘要应一致"这个前提不成立：
      - elapsed_seconds：它反映本次运行的机器负载，每次都不同。留着它
        两次运行必然算出不同摘要，复现校验就永远失败；
      - 已有的 content_sha256：不剔除旧摘要会导致摘要套摘要，
        形成一个依赖上一轮结果的值，同样无法稳定。

    递归处理而不是只处理顶层：报告里的区域列表、候选列表嵌套很深，
    耗时字段可能出现在任意层级，只在顶层剔除会漏掉。
    """

    if isinstance(value, dict):
        return {
            str(key): stable_payload(item)
            for key, item in value.items()
            if key not in VOLATILE_FIELDS and key != CONTENT_DIGEST_FIELD
        }
    if isinstance(value, list):
        return [stable_payload(item) for item in value]
    return value


def content_sha256(payload: dict[str, Any]) -> str:
    """我按规范化 JSON 计算内容摘要，同一输入跨运行稳定。

    三个序列化参数缺一不可，每一个都对应一个真实的失稳来源：
      - sort_keys=True：dict 的键顺序依赖插入顺序，不排序的话
        遍历顺序变化就会算出不同摘要；
      - separators=(",", ":")：去掉默认的可变空白，让字节流唯一；
      - allow_nan=False：不要允许 NaN/Infinity 被写进 JSON。
        它们不是合法 JSON，且同一个 NaN 在不同路径下序列化结果可能不同，
        会让摘要失去意义。宁可让它抛错暴露问题，也不要产出一个假的稳定值。
    """

    encoded = json.dumps(
        stable_payload(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def with_content_digest(payload: dict[str, Any]) -> dict[str, Any]:
    """我给报告负载补上内容摘要字段，便于跨运行比对同一输入。"""

    payload[CONTENT_DIGEST_FIELD] = content_sha256(payload)
    return payload
