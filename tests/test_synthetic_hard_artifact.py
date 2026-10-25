"""我钉住 synthetic_hard 产物必须自带"为什么没检出"的归因。

我为什么要写这个：这个产物曾经是旧版代码生成的，里面只有一个
`detected_case_count=0`，没有任何解释。README 却写着"我如实上报
`undetectable_case_count`"——两边对不上，读产物的人会以为检测器
在 24 个案例上全军覆没，而真相是其中 23 个案例的缺陷深度
物理上就够不到阈值。

这类"文字说报了、产物里没有"的不一致，靠人眼很难发现，
但一条断言就能钉住。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "outputs" / "evaluation" / "synthetic_hard.json"


def _artifact() -> dict:
    assert ARTIFACT.is_file(), f"合成难例产物不见了：{ARTIFACT}"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test我方难例产物带有不可检出归因() -> None:
    """我钉住产物里既有 undetectable 计数，也有"检出后被阈值滤掉"的计数。"""

    data = _artifact()
    assert data["undetectable_case_count"] == 23, (
        "训练集/8 上 24 个合成案例里实测 23 个缺陷深度不足阈值、物理上不可检出；"
        f"产物记的是 {data.get('undetectable_case_count')!r}。"
    )
    assert data["filtered_after_detection_case_count"] == 1, (
        "剩下 1 个是检出后被 min_region_area_mm2 滤掉才表现为漏检；"
        f"产物记的是 {data.get('filtered_after_detection_case_count')!r}。"
    )
    assert data["undetectable_case_count"] + data["filtered_after_detection_case_count"] == data["case_count"], (
        "不可检出 + 被阈值滤掉 应正好覆盖全部案例；"
        "对不上说明有案例既没检出、也没被归类，产物会误导人。"
    )


def test我方难例产物没有因为加归因就放宽判据() -> None:
    """我钉住状态仍是 TARGET_NOT_PROVEN，没有变成假绿。

    我为什么要单独钉这个：`_ratio` 在分母为空时返回 1.0，
    六个指标会同时"达标"。如果哪天有人把全部案例都判成不可检出，
    又没守住"scored 非空才判达标"这条，产物就会从"没检出"
    悄悄变成"达标"。这条断言让那种改动立刻红。
    """

    data = _artifact()
    assert data["status"] == "TARGET_NOT_PROVEN", (
        f"状态变成了 {data['status']!r}；本轮所有案例都没检出，"
        "绝不能因为归因字段的加入而被判为达标。"
    )


def test我方难例产物里的限制说明覆盖两种归因() -> None:
    """我钉住 limitations 里确实把两种成因都写给了人看，而不是只存字段。"""

    joined = "\n".join(_artifact()["limitations"])
    assert "物理上不可检出" in joined, "limitations 应说明大部分案例物理上不可检出"
    assert "滤掉" in joined, "limitations 应说明有一个案例是检出后被阈值滤掉的"


ALL_EVALUATION_ARTIFACTS = sorted(
    p.name for p in (ROOT / "outputs" / "evaluation").glob("synthetic_*.json")
)


@pytest.mark.parametrize("name", ALL_EVALUATION_ARTIFACTS)
def test我方每个合成评估产物都带齐归因字段(name: str) -> None:
    """我钉住 outputs/evaluation 下**每一个**产物都带齐两个归因字段。

    我为什么从"只钉 hard"扩到"钉全部"：上一轮我只修了 hard，
    结果留下 5 个产物里有的缺 `undetectable_case_count`、
    有的缺 `filtered_after_detection_case_count`——README 却写着
    "套件带这两个字段"。只修一个等于制造新的不一致，
    所以这里对目录下所有 synthetic_*.json 逐个检查。
    """

    data = json.loads((ROOT / "outputs" / "evaluation" / name).read_text(encoding="utf-8"))
    for field in ("undetectable_case_count", "filtered_after_detection_case_count"):
        assert field in data, (
            f"{name} 缺 {field!r}；它是旧版代码生成的，"
            "请用当前 evaluate_synthetic_suite 重跑，并先核对指标与旧产物一致。"
        )
