"""我把产物里那句**自相矛盾**的单位警告换成修正后的文案。

我为什么要写这个脚本而不是直接改代码就完事：
代码修了只保证**以后**不再产出那句话，但磁盘上已经存了 128 份
带着它的 result.json，而甲方看的是**产物**不是源码。

我只替换这一条**派生文案**，不动任何实测数值（面积、残差、区域等），
也不会重新跑检测——重跑一遍要几十分钟，而且会把"当时跑出来是什么样"
这个留档信息抹掉。这条警告是由 unit_audit 推导出来的，
所以就地重算它是安全的。

替换规则：只处理"非 mm 单位 mm"这种 mm 被误当成非 mm 的情况。
真正声明了非 mm 单位（如 inch）的文件不受影响。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 修正后走兜底分支的文案，与 pipeline.py 里的兜底文案保持一致。
_FALLBACK_ASSUME_MM = (
    "我没有从全部输入模型得到可靠单位；当前按配置 assume_mm 处理，"
    "请确认源 CAD 和 STL 数值确实使用 mm。"
)


def _fix_warnings(warnings: list) -> tuple[list, int]:
    """我把"非 mm 单位 mm"这类自相矛盾的旧告警换成如实说明单位假定的兜底文案。"""

    fixed: list = []
    changed = 0
    for warning in warnings:
        text = str(warning)
        if "非 mm 单位 mm" in text:
            # mm 被误当成"非 mm"：这条警告本来就不该提"非 mm 单位"，
            # 换成兜底文案——它如实说"我没有得到可靠单位"。
            fixed.append(_FALLBACK_ASSUME_MM)
            changed += 1
        else:
            fixed.append(warning)
    return fixed, changed


def _refresh(path: Path) -> int:
    """我改写单份 result.json 的告警文本，返回实际改了几条（没改返回 0）。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        return 0
    fixed, changed = _fix_warnings(warnings)
    if not changed:
        return 0
    payload["warnings"] = fixed
    if isinstance(payload.get("warning_count"), int):
        payload["warning_count"] = len(fixed)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return changed


def main() -> int:
    """我遍历 outputs 下的 result.json 逐份刷新告警，并汇总改了哪些文件。"""

    total_files = 0
    total_fixed = 0
    for path in sorted(ROOT.glob("outputs/**/result.json")):
        try:
            changed = _refresh(path)
        except (OSError, ValueError) as exc:
            print(f"跳过 {path}：{exc}", file=sys.stderr)
            continue
        if changed:
            total_files += 1
            total_fixed += changed
    print(f"我已修正 {total_files} 份产物里的 {total_fixed} 条单位警告。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
