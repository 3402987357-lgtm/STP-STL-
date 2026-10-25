"""我钉住"需求来源是甲方原件、不是我的转述"这条可追溯性。

我为什么要钉这个（2026-10-16）：
我翻了十几轮自己的报告，反复看到同一句自我限制——"需求清单来源是我
自己的 README 转述，**无甲方原始任务书可核对**"。这条限制让
"补充所有需求"从根上不可验证：按自己的转述核对，漏掉的需求永远发现不了。

本轮我把找到的两份甲方原件正文抽进了 `data/task_book/`。这个文件用
断言把三件事钉住：

1. 原件**还在**（被删掉就红，而不是悄悄退回"没有原件"的状态）；
2. 原件里**确实写着**本项目那几条硬要求（阈值 0.15mm、≥99%、
   位置/范围/误差、源码与流程图、Python）；
3. 两份任务书**没有被我张冠李戴**（曲率任务书的特征词不能出现在
   缺失区域任务书里，反之亦然）。

第 3 条是我特意加的：两份任务书都在桌面、名字都是 UUID，
最容易犯的错就是把它们搞混，然后用错误的原件去"证明"需求。
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
TASK_BOOK = ROOT / "data" / "task_book"

MISSING_REGION_BOOK = (
    TASK_BOOK / "基于STL比对的STP部件缺失区域识别_任务书原文.txt"
)
CURVATURE_BOOK = (
    TASK_BOOK / "基于3D文件的曲率复杂变化区域检索功能_任务书原文.txt"
)
EXAMPLE_FIGURES = (
    TASK_BOOK / "基于STL比对的STP部件缺失区域识别_示例图"
)


def _read(path: Path) -> str:
    assert path.exists(), (
        f"甲方原件抽取稿 {path.name} 不在了。"
        "它是我核对需求的唯一原件依据，删掉它等于让需求来源重新变回我的转述。"
        "请重跑 scripts/import_task_books.py 恢复。"
    )
    return path.read_text(encoding="utf-8")


def test我方保留了本项目任务书原件() -> None:
    text = _read(MISSING_REGION_BOOK)
    assert "基于STL比对的STP部件缺失区域识别任务说明书" in text


def test我方保留了曲率任务书原件() -> None:
    text = _read(CURVATURE_BOOK)
    assert "曲率复杂变化区域检索" in text


def test我方保留了任务书里的示例图() -> None:
    """我钉住"示例图"这一列也在仓库里。

    我为什么补这条（2026-10-17）：`import_task_books.py` 当初只抽正文文字
    （注释里写明"不复制图片与排版"），于是需求表里那一列"示例图"在交付物里
    整整缺席 —— 而它本身就是甲方对需求的原始表述。尤其示例图1，
    画的是"缺陷位置 + 一条红色测量线 + 2.7847 mm 数值标签"，
    直接说明甲方想看到的是**位置与误差数值**，不是一段文字描述。

    我钉 3 张不是 15 张：原件 `word/media/` 下有 15 张 png，
    但正文 `document.xml` 里只有 3 个 `<w:drawing>`，
    另外 12 张是 rels 里登记、正文里一次都没引用的孤儿。
    把孤儿也收进来会让人误以为它们也是需求。
    """

    figures = sorted(EXAMPLE_FIGURES.glob("*.png"))
    assert len(figures) == 3, (
        f"示例图目录里应有 3 张被正文真正引用的图，实际 {len(figures)} 张："
        f"{[p.name for p in figures]}。"
        "若原件更新过，请按 README.md 里记的'正文位置'重新确认哪几张被引用。"
    )
    assert all(p.stat().st_size > 0 for p in figures), (
        "示例图里有 0 字节文件，说明复制时没写全。"
    )


@pytest.mark.parametrize(
    "keyword",
    [
        "0.15mm",          # 阈值默认值
        "≥99%",            # 识别率验收线
        "阈值调参接口",     # 可调阈值
        "位置、范围",       # 输出必须含位置与范围
        "几何误差数据",     # 输出必须含误差
        "算法流程图",       # 交付物之一
        "Python",          # 开发语言
    ],
)
def test本项目任务书写着我实现的那几条硬要求(keyword: str) -> None:
    """我逐条钉住原件里点名、且我已经实现的硬要求。

    我把这些词钉住的意义：它们是"需求 ↔ 实现"对表里的**左侧**。
    左侧若被改动（原件换了、或我贴错了文件），对表结论就全部失效，
    这时候必须红着提醒，而不是继续拿旧结论交差。
    """

    assert keyword in _read(MISSING_REGION_BOOK), (
        f"本项目任务书原件里找不到 {keyword!r}；"
        "若原件确实换了，请先重新对表，再更新本用例。"
    )


def test我方没有把两份任务书搞混() -> None:
    """我钉住两份原件没有被张冠李戴。

    两份都在桌面、名字都是 UUID，最危险的错误是拿错原件去"证明"需求。
    曲率任务书的特征词（CCR/曲率/2D 图纸映射）不能出现在缺失区域任务书里；
    反之，缺失区域任务书的特征词（STL 比对/缺失区域）也不能出现在曲率任务书里。
    """

    missing = _read(MISSING_REGION_BOOK)
    curvature = _read(CURVATURE_BOOK)

    for word in ("曲率复杂变化", "2D 图纸", "CCR_"):
        assert word not in missing, (
            f"{word!r} 出现在缺失区域任务书里，说明我可能贴错了文件"
        )
    for word in ("缺失区域识别任务说明书", "完整参考STL"):
        assert word not in curvature, (
            f"{word!r} 出现在曲率任务书里，说明我可能贴错了文件"
        )
