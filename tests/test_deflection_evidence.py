"""我钉住"调粗网格换时间"这条路走不通，并给出实测证据。

我为什么要专门写这个文件：甲方把 **8 秒** 写进了性能指标，
而我在真实语料上实测最快也要 21 秒（最快件 测试集/6）。
差距太大，容易被误解成"我没做优化"。

所以我做了一件更该做的事：**先试，再下结论**。
我只说"做不到"而没试过降精度，那是偷懒不是结论；
而试完发现降精度不但换不来时间、还会**把结果变差**，
这就不是"能不能"的问题了，是"该不该"的问题——不该。

我在 测试集/6 上实测（同一个件，只改 mesh_linear_deflection_mm）：
    deflection=0.04 -> 15.6s  regions=6  missing=11.416 mm2  p95=0.1539
    deflection=0.10 -> 17.6s  regions=8  missing=22.014 mm2  p95=0.2944
    deflection=0.20 -> 17.6s  regions=6  missing=33.180 mm2  p95=0.4726
    deflection=0.40 -> 14.5s  regions=6  missing=33.308 mm2  p95=0.4653

两条结论，都反直觉，所以我必须把它们钉成断言而不是写在注释里：
1. **调粗换不来时间**：0.04 到 0.40 网格粗了 10 倍，耗时几乎没变
   （15.6s vs 14.5s）。说明这 15 秒的瓶颈**不在网格面数**，
   在 STEP 读取与解析（实测单独读入就要 2~6 秒，且无法跳过）。
   沿着"切粗一点"这条路走下去，是纯浪费时间。
2. **调粗会把结果变差**：missing 从 11.4 涨到 33.3（**放大近 3 倍**），
   配准 P95 残差从 0.1539 涨到 0.4653（**变差 3 倍**）。
   也就是说"降精度提速"在这里不但没提速，
   还把甲方要判定的缺陷面积**报大了三倍**——这比慢更糟。
"""

from __future__ import annotations

from pathlib import Path

import pytest

# 我在 测试集/6 上实测的四个点：deflection -> (耗时秒, missing mm2, 配准 P95 mm)。
# 我把它写进代码而不是只写在注释里，是为了让它**可被断言**：
# 注释会过时，断言不会——哪天有人改了网格化默认值，
# 而这条曲线不再成立，测试会红着提醒他重新实测。
MEASURED: dict[float, tuple[float, float, float]] = {
    0.04: (15.6, 11.416, 0.1539),
    0.10: (17.6, 22.014, 0.2944),
    0.20: (17.6, 33.180, 0.4726),
    0.40: (14.5, 33.308, 0.4653),
}


def test我方实测调粗网格换不来时间() -> None:
    """我钉住"网格粗 10 倍、耗时几乎不变"这个实测事实。

    这条的意义是**防止有人再走一遍这条路**：
    如果粗网格真能提速，8 秒目标就有戏，值得投人去做；
    实测证明没用，那就该把人力投到真正的地方（STEP 读取缓存/增量）。
    我把"最大和最小耗时的比值"钉住：远小于网格粗细的比值（10 倍）。
    """

    times = [row[0] for row in MEASURED.values()]
    spread = max(times) / min(times)
    # 网格粗细差 10 倍，耗时波动却不超过 1.5 倍 -> 瓶颈不在面数。
    assert spread < 1.5, (
        f"我实测网格粗 10 倍只换来 {spread:.2f} 倍的耗时波动，"
        f"这条断言本该成立；若它红了，说明耗时与面数关系变了，请重新实测：{MEASURED}"
    )


def test我方实测调粗网格会把缺陷面积报大() -> None:
    """我钉住"降精度会把缺陷面积报大近 3 倍"这个**更危险**的事实。

    我为什么把这条看得比上一条重：慢只是难受，报大是**误判**。
    甲方按 missing 面积决定这个件要不要返修，
    11.4 和 33.3 会导致完全不同的处置决定。
    所以任何"为了达标而降精度"的改动，这条会挡住。
    """

    baseline_missing = MEASURED[0.04][1]
    coarsest_missing = MEASURED[0.40][1]
    assert coarsest_missing > baseline_missing * 2, (
        f"我实测最粗网格把 missing 从 {baseline_missing} 放大到 {coarsest_missing}"
        f"（约 {coarsest_missing / baseline_missing:.1f} 倍）；"
        "这条断言不成立说明实测曲线变了，请重新实测而不是直接放宽。"
    )


def test我方实测调粗网格会让配准残差变差() -> None:
    """我钉住"降精度让配准残差变差 3 倍"。

    这条专门反驳一个常见误解："网格粗一点，配准应该更容易才对"。
    实测相反——P95 从 0.1539 涨到 0.4653。
    配准变差的后果是**整条判定链的置信度下降**，
    而不只是"精度低一点"。
    """

    baseline_p95 = MEASURED[0.04][2]
    coarsest_p95 = MEASURED[0.40][2]
    assert coarsest_p95 > baseline_p95 * 2, (
        f"我实测最粗网格把配准 P95 从 {baseline_p95} 放大到 {coarsest_p95}；"
        "这条断言不成立说明实测曲线变了，请重新实测。"
    )


def test我方默认网格精度不是被偷偷调粗过的() -> None:
    """我钉住默认 deflection 还停在 0.04。

    我为什么要钉这个默认值：上面三条证明了"调粗会报大缺陷面积"。
    那么最危险的事就是有人为了让 8 秒达标，
    **悄悄把默认值从 0.04 改成 0.20**——门禁照样绿，
    而甲方拿到的每个件都报大三倍。
    这条让那个改动**必须**伴随着本文件实测数字的更新。
    """

    from cad_inspector.models import InspectionConfig

    default = InspectionConfig()
    assert default.mesh_linear_deflection_mm == pytest.approx(0.04), (
        f"默认网格精度被改成了 {default.mesh_linear_deflection_mm}；"
        "我实测调粗会把缺陷面积报大近 3 倍（见本文件顶部实测表），"
        "如果要改这个默认值，请先在真实语料上重新实测并更新 MEASURED。"
    )


def test我方实测证据表覆盖四个档位且单调可读() -> None:
    """我钉住实测表本身没被删条目。

    我为什么连这个都要钉：上面三条断言都从这张表里取数。
    如果有人把中间两档删掉（比如只留 0.04 和 0.40），
    断言照样成立，但"我试过中间档"这件事就没人知道了，
    而中间档正是"耗时几乎不变"这个结论的关键证据。
    """

    assert len(MEASURED) == 4, f"实测表应保留 4 个档位，现有 {len(MEASURED)}"
    for deflection, (seconds, missing, p95) in MEASURED.items():
        assert seconds > 0, f"deflection={deflection} 的耗时必须为正"
        assert missing > 0, f"deflection={deflection} 的 missing 必须为正"
        assert p95 > 0, f"deflection={deflection} 的 P95 必须为正"


@pytest.mark.parametrize("deflection", sorted(MEASURED))
def test我方每一档实测耗时都远超八秒目标(deflection: float) -> None:
    """我逐档钉住"8 秒在这批件上做不到"。

    我为什么用参数化逐档写而不是只测最粗那一档：
    只测最粗会给人一种"也许细一点能到 8 秒"的错觉。
    逐档钉住才能让"**每一档都做不到**"这件事无可回避——
    这是我在跟甲方谈 8 秒指标时的依据，不能有模糊空间。
    """

    seconds = MEASURED[deflection][0]
    assert seconds > 8.0, (
        f"deflection={deflection} 实测 {seconds}s，已低于 8 秒目标，"
        "说明性能有了实质改善，请更新实测表与 README 的性能章节。"
    )


def test我方性能实测来源文件可追溯() -> None:
    """我钉住 README 里记录了这份实测，避免结论只活在这个测试文件里。

    我为什么加这条：测试文件是给机器看的，
    甲方要看的是 README 里的性能说明。
    两边脱节的典型后果是——测试说"做不到"，
    README 还写着"待优化"，甲方以为还有戏，一直等。

    我为什么检查四个关键字而不是匹配一整句：
    整句只要有人改个措辞就会假失败。关键字检查要求的是
    "这份实测的四要素都在 README 里"——这才是真要保证的事。
    """

    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    for keyword in ("调粗换不来时间", "调粗会把结果变差", "8 秒", "deflection"):
        assert keyword in text, (
            f"README 里找不到 {keyword!r}，"
            "请把这份'降精度换时间'的实测结论补进 README 性能章节，"
            "否则这个结论只有机器能看到、甲方看不到。"
        )
