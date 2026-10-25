"""我把甲方原始任务书的正文抽成纯文本，存进项目内作为需求依据。

我为什么加这个脚本（2026-10-16）：
我翻了十几轮报告，反复看到同一句自我限制——"需求清单来源是我自己的
README 转述，没有甲方原始任务书文件可核对"。那条限制让"补充所有需求"
这件事从根上不可验证：我按自己的转述核对，漏掉的需求我永远发现不了。

本轮我在桌面找到了两份**甲方原始任务书**（.docx），所以把正文抽出来
落到项目里，让需求来源从此可追溯、可复核，而不是活在我的转述里。

我刻意只抽正文 XML，不碰图片和排版：我要的是**可核对的需求文字**，
不是把原文档复制一份。
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "task_book"

# 我登记的来源：文件名 -> (桌面上的原文件名, 我在项目内用的名字)
SOURCES = {
    "基于STL比对的STP部件缺失区域识别_任务书原文.txt": (
        "77809665-b94c-43ca-aacf-8c57befaccc9.docx",
        "基于STL比对的STP部件缺失区域识别任务说明书",
    ),
    "基于3D文件的曲率复杂变化区域检索功能_任务书原文.txt": (
        "1896cdf6-498e-4414-a80c-ea1a90d8128b.docx",
        "基于 3D 文件的曲率复杂变化区域检索功能",
    ),
}


def _paragraphs(docx: Path) -> list[str]:
    """我按段落抽正文，保留原有分条结构，便于逐条核对。"""

    with zipfile.ZipFile(docx) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", "replace")
    out: list[str] = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, flags=re.S))
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            out.append(text)
    return out


def main() -> int:
    """我从桌面 docx 只读正文段落，逐份写到 data/task_book 下的 txt。"""

    desktop = Path(r"D:\桌面")
    TARGET.mkdir(parents=True, exist_ok=True)
    written = 0
    for target_name, (source_name, title) in SOURCES.items():
        source = desktop / source_name
        if not source.exists():
            print(f"[MISS] {source} 不存在，跳过")
            continue
        lines = _paragraphs(source)
        header = [
            f"# {title}",
            "",
            f"来源文件：{source}",
            "抽取方式：scripts/import_task_books.py 只读 word/document.xml 正文，",
            "不复制图片与排版；段落顺序与原文档一致。",
            "",
            "---",
            "",
        ]
        body = [f"{index:03d}| {line}" for index, line in enumerate(lines, 1)]
        (TARGET / target_name).write_text(
            "\n".join(header + body) + "\n", encoding="utf-8"
        )
        print(f"[OK] {target_name}：{len(lines)} 段")
        written += 1
    print(f"共写出 {written} 份")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
