"""我钉住"21 件真实件里没有 >=0.15mm 几何缺料"这条全量复核结论。

我为什么要写这个文件：这条结论推翻了我自己此前对 155 个草稿区域的解读。
它们不是"待人工确认的真值"，而是旧 centroid 配准把质心对齐、把整张表面
推开之后产生的系统性伪区域。换成 auto 配准后，21 件只剩 3 个区域。

这条结论有两个用途：
1. 甲方若按任务书口径验收，会看到"检测器几乎检不出缺失"——需要知道
   原因是**数据里没有这类缺陷**，不是检测器坏了；
2. 后续任何人想再调阈值，都该先看这条，否则会在一批没有真值的数据上
   反复调参。

原始数据：`reports/full_recheck_20261018.json`。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "full_recheck_20261018.json"


def test我方全量复核报告与README结论一致() -> None:
    """报告里的数字与 README 里写下的结论必须对得上。"""

    if not REPORT.is_file():
        pytest.skip("我还没有 21 件全量复核报告")
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    records = data["records"]
    assert data["pair_count"] == 21
    assert len(records) == 21

    old_regions = sum(r.get("old_region_count") or 0 for r in records)
    new_regions = sum(r.get("new_region_count") or 0 for r in records)
    assert old_regions == 155, "旧 centroid 口径下的草稿区域数变了，先确认是不是又跑过草稿"
    assert new_regions == 3, "auto 口径下应只剩 3 个区域"

    # 全部落到 auto->none：说明两套模型本来就在同一坐标系，不需要做变换。
    assert {r["new_mode"] for r in records} == {"auto->none"}

    ratios = [r.get("new_missing_ratio") or 0.0 for r in records]
    assert max(ratios) < 0.08, "缺失面积比最大值应远小于'缺了一块料'的量级"
    # 仍有区域的三件要逐件钉住，避免以后悄悄变成别的件。
    still = {r["case"]: r["new_region_count"] for r in records if r.get("new_region_count")}
    assert still == {"训练集__7": 1, "训练集__8": 1, "测试集__11": 1}


def test我方README写清了这批数据没有几何缺料() -> None:
    """我钉住 README 里能查到这条结论，避免它只活在报告和开发记录里。"""

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for keyword in ("21 件全量复核", "155", "3 个区域", "没有 >=0.15mm"):
        assert keyword in text, (
            f"README 里找不到 {keyword!r}，请把'21 件全量复核'结论补进 README，"
            "否则甲方会把'检不出'误读成'检测器不行'。"
        )
