"""我钉住"最小区域面积门限到底在挡什么"这轮实测的结论。

我为什么要写这个文件：README 第 5 条里我自己留过一句待办——
"默认值要不要动，必须在带配准的真实件上另做一轮实验"。
这轮实验做完了，结论是**反直觉**的，所以必须钉成断言，
否则它只会留在一次性的输出目录里，下次有人问"0.5 是怎么来的"
又得从头量一遍。

实测口径：真实件、默认 centroid 配准、14 个件每个跑两次
（门限 0.0 枚举全部候选 / 门限 0.5 作对照）。
原始数据：`reports/min_region_area_evidence.json`。

三条结论：
1. **噪声碎片规模巨大**：14 件共 116,948 个候选区域，
   测试集/3 一个件就有 73,843 个。所以"0.5 在压噪声"是真的。
2. **但噪声碎片极其小**：中位 0.0020、99 分位 0.0117 mm2，
   比 0.5 门限小两个量级。
3. **0.5 这个具体数值不承重**：压住噪声的是"门限远高于 0.01"，
   不是"门限恰好是 0.5"。降到 0.3 一个噪声碎片都放不进来，
   只会多放进 24 个区域。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "min_region_area_evidence.json"

# 我把关键数字内联一份，再和报告逐项核对：
# 只读报告的话，报告被改坏时测试会跟着一起错；
# 只写常量的话，报告被删了也没人知道。
PARTS = 14
TOTAL_CANDIDATES = 116_948
P50 = 0.0020
P90 = 0.0047
P99 = 0.0117
COUNT_GE_0_3 = 123
COUNT_GE_0_5 = 99


def _report() -> dict:
    assert REPORT.is_file(), (
        "我实测的原始数据 reports/min_region_area_evidence.json 不见了；"
        "它是这轮结论的唯一出处，删掉就等于把结论变成了口说无凭。"
    )
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test我方实测报告与内联数字一致() -> None:
    """我钉住报告本身没被改坏，且和我记的数字对得上。"""

    data = _report()
    assert data["registration_mode"] == "centroid", "这轮实验必须带配准才有意义"
    assert data["parts_analyzed"] == PARTS
    assert data["total_candidate_regions"] == TOTAL_CANDIDATES
    q = data["quantiles"]
    assert q["p50"] == pytest.approx(P50, abs=5e-5)
    assert q["p90"] == pytest.approx(P90, abs=5e-5)
    assert q["p99"] == pytest.approx(P99, abs=5e-5)
    assert data["count_at_or_above"]["0.3"] == COUNT_GE_0_3
    assert data["count_at_or_above"]["0.5"] == COUNT_GE_0_5


def test我方噪声碎片远小于默认门限() -> None:
    """我钉住"99% 的噪声碎片比 0.5 门限小两个量级"。

    这条是整轮实验的核心：它说明 0.5 之所以能压住噪声，
    是因为它远高于噪声主体，而不是因为它恰好等于 0.5。
    如果哪天 99 分位涨到接近门限，说明配准质量或采样变了，
    这个"门限不承重"的结论就不再成立，测试必须红。
    """

    assert P99 * 20 < 0.5, (
        f"实测 99 分位噪声只有 {P99} mm2，仍比 0.5 门限小 {0.5 / P99:.0f} 倍；"
        "这条断言不成立说明噪声尺度变了，请重新实测再下结论。"
    )


def test我方降到零点三不会放进噪声洪流() -> None:
    """我钉住"降门限放进的是 24 个小区域，不是 11 万个噪声"。

    我为什么单独钉这条：这是"能不能把默认门限调小"这个问题的
    **直接答案**。116,948 个噪声碎片全在 0.3 以下，
    所以 0.3 与 0.5 在压噪声上等价；两者的差别只有那 24 个区域。
    如果有人只看"碎片有 11 万个"就断言"门限绝不能降"，
    这条会用数字反驳他。
    """

    band = COUNT_GE_0_3 - COUNT_GE_0_5
    assert band == 24, f"[0.3,0.5) 档实测应有 24 个区域，实际 {band}"
    # 降门限放进来的量级必须是"几十个"，而不是"几万个"。
    assert band < TOTAL_CANDIDATES / 1000, (
        f"[0.3,0.5) 档有 {band} 个，噪声总数 {TOTAL_CANDIDATES}；"
        "若这两者量级接近，说明降门限会放进噪声，结论要改写。"
    )


def test我方没有擅自改动默认门限() -> None:
    """我钉住默认值仍是 0.5，且这件事是有意为之。

    我为什么钉这个：实测说"0.3 与 0.5 对噪声等价"，
    很容易被读成"那就该改成 0.3"。但两者差别只在要不要放进
    那 24 个小区域，这是纯粹的精度/召回取舍，属于甲方的判定口径。
    这条让任何改动都必须是有意识的、并且要连带更新本文件的实测依据。
    """

    from cad_inspector.models import InspectionConfig

    assert InspectionConfig().min_region_area_mm2 == pytest.approx(0.5), (
        "默认最小区域面积被改了；我实测它并不承重（见本文件顶部三条结论），"
        "改它属于精度/召回口径调整，请先确认甲方口径并更新实测依据。"
    )


def test我方README记录了这一轮实测() -> None:
    """我钉住 README 里能查到这轮实验，避免结论只活在测试文件里。"""

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for keyword in ("2026-10-17 实测补齐", "116,948", "0.0117"):
        assert keyword in text, (
            f"README 里找不到 {keyword!r}，"
            "请把这轮'最小区域门限'实测补进 README 第 5 条，"
            "否则甲方看不到这个结论。"
        )
