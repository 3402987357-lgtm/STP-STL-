"""我钉住"0.15mm 固定阈值在真实语料上不可用"这轮汇总的结论。

我为什么要写这个：README 里原本只说"0.15mm 没有真实件支持"，
听起来像精度不够、还能凑合。我把 20 个真实件的验收汇总一算，
发现它其实是**可用性问题**——18/19 个件的推荐阈值都高于 0.15mm，
14/20 个件因此把一半以上的表面判成缺失。

这两件事必须分开说：前者是"再调调也许能行"，后者是"这个默认值
给出的 missing_area_ratio 没有参考价值"。断言钉住后者，
免得日后有人把这条又读回成前者。

**2026-10-17 后续**：上面这组数字是 **centroid 配准**下的基线。
后来我把默认配准改成 auto，重跑后推荐阈值中位从 0.552mm 降到 0.155mm，
和 0.15mm 基本重合——所以"0.15mm 不可用"这个结论**只在旧配准下成立**。
下面多了一条钉住修复后的状态；旧数据保留，作为"为什么要改配准"的证据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "threshold_vs_registration_evidence.json"
REPORT_AFTER = ROOT / "reports" / "threshold_after_auto_registration.json"


def _report() -> dict:
    assert REPORT.is_file(), (
        "我汇总的原始数据 reports/threshold_vs_registration_evidence.json 不见了；"
        "它是这条结论的唯一出处，删掉就等于把结论变成口说无凭。"
    )
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test我方固定阈值在几乎每个真实件上都偏低() -> None:
    """我钉住 18/19 这个比例，而不是只写一句"偏低"。"""

    data = _report()
    assert data["configured_threshold_mm"] == pytest.approx(0.15)
    assert data["parts_below_recommended"] == 18
    # 分母是"有推荐值的件"，不是全部 20 个；测试集/9 无配准、无推荐值。
    assert len([r for r in data["rows"] if r["recommended_threshold_mm"]]) == 19


def test我方一半以上的件被误判成大面积缺失() -> None:
    """我钉住"14/20 个件 missing>=50%"——这是可用性问题的直接证据。"""

    data = _report()
    assert data["parts_with_missing_ratio_ge_50pct"] == 14
    worst = max(data["rows"], key=lambda r: r["missing_area_ratio"])
    assert worst["pair"] == "测试集/11"
    assert worst["missing_area_ratio"] == pytest.approx(0.957, abs=5e-4)


def test我方推荐阈值跨度大且与配置值差一个量级() -> None:
    """我钉住推荐阈值的分布，说明它不是"稍微调高一点"能解决的。"""

    data = _report()
    assert data["recommended_threshold_max"] / data["recommended_threshold_min"] > 20, (
        "推荐阈值最大与最小相差应超过 20 倍；若差距变小，说明配准质量变了，"
        "这条'固定阈值不可用'的结论需要重新评估。"
    )
    assert data["recommended_threshold_median"] > 0.15 * 3, (
        "推荐阈值中位应明显高于 0.15mm；否则固定值可能是合理的。"
    )


def test我方没有擅自把默认阈值改掉() -> None:
    """我钉住默认仍是 0.15mm，改动必须是有意识的判定口径变更。

    实测说 0.15mm 不可用，很容易被读成"那就改成 auto 或改个数"。
    但默认阈值是甲方的判定口径，改它必须甲方拍板；
    这条让任何改动都要连带更新本文件的实测依据。
    """

    from cad_inspector.models import InspectionConfig

    assert InspectionConfig().threshold_mm == pytest.approx(0.15)


def test我方修好配准后推荐阈值回到0_15附近() -> None:
    """我钉住"0.15mm 不再偏低"这个结论，并记录它是怎么被两次修正拉回来的。

    第一版基线（centroid + 到最近顶点）推荐阈值中位 0.552mm，看起来
    0.15mm 差得很远。后面发现两个口径问题叠加：
      1. centroid 把缺料件的质心对齐，等于把表面推开（已改 auto）；
      2. 残差量的是"候选顶点到最近参考顶点"，量到的其实是两套三角化
         的采样间距，不是错位（已改为到参考曲面）。
    两次修正后中位降到 0.047mm，配置的 0.15mm 对 19/20 件都不低于推荐值。
    所以这条断言的下界必须是"远低于 0.15"——若它涨回 0.12 以上，
    说明又退回成"到顶点"那种偏大的口径了。
    """

    assert REPORT_AFTER.is_file(), f"配准修复后的阈值证据不见了：{REPORT_AFTER}"
    data = json.loads(REPORT_AFTER.read_text(encoding="utf-8"))
    assert data["parts"] == 20
    assert data["configured_threshold_mm"] == pytest.approx(0.15)
    assert 0.02 <= data["recommended_threshold_median"] <= 0.08, (
        "曲面残差口径下推荐阈值中位应在 0.02~0.08mm；"
        "偏高说明又退回到量采样间距的顶点口径了。"
    )
    assert data["recommended_threshold_max"] < 0.30, (
        "旧基线最大推荐值 3.46mm；改用曲面残差后应远小于 0.3mm。"
    )
    assert data["parts_below_recommended"] <= 1, (
        "配置 0.15mm 应覆盖绝大多数件的推荐值；低于推荐值的件数突然变多，"
        "说明配准或距离口径又退化了。"
    )
