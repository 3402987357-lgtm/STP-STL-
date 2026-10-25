"""我为桌面界面区域写回归测试。

我不真的弹出窗口（那会让门禁跑不起来，也需要人在场点掉）。我改为
把 mainloop 换成"记录下 root 就返回"的哑实现，这样能验到**控件是否
真的建起来了、层级对不对**，又不会阻塞。

我真正要钉住的是"判定不重新实现"：界面的摘要函数必须直接吃
pipeline 的结果字典，界面与命令行的结论必须是同一套。
"""

from __future__ import annotations

import pytest

pytest.importorskip("tkinter")

import tkinter as tk  # noqa: E402

from cad_inspector.ui.desktop import (  # noqa: E402
    _run_and_report,
    _summarize,
    launch_desktop_app,
)


def _sample_result() -> dict:
    """我造一份和 pipeline 输出同构的结果字典。"""

    return {
        "status": "review",
        "threshold_mm": 0.15,
        "unit": "mm",
        "unit_source": "assume_mm",
        "regions": [
            {
                "defect_type": "missing",
                "area_mm2": 1.234,
                "max_error_mm": 0.9,
                "review_required": False,
            },
            {
                "defect_type": "extra",
                "area_mm2": 0.5,
                "max_error_mm": 0.3,
                "review_required": True,
            },
        ],
        "warnings": ["我无法从参考 STL 判断单位。"],
        "artifacts": {"json": "/tmp/a.json"},
    }


def test我方摘要区分缺料多料与待复核() -> None:
    """多料与缺料是两类结论，混在一起会让甲方看错处理方向。"""

    text = _summarize(_sample_result())

    assert "缺料 1" in text
    assert "多料 1" in text
    assert "待复核" in text
    assert "review" in text


def test我方摘要保留警告与产物() -> None:
    """警告是"这结论能不能信"的依据，摘要里丢掉等于误导。"""

    text = _summarize(_sample_result())

    assert "我无法从参考 STL 判断单位。" in text
    assert "/tmp/a.json" in text


def test我方摘要截断过多区域但不静默丢弃() -> None:
    """截断要写明还有多少，否则用户会以为只有列出的这些。"""

    result = _sample_result()
    result["regions"] = [
        {
            "defect_type": "missing",
            "area_mm2": 1.0,
            "max_error_mm": 0.5,
            "review_required": False,
        }
        for _ in range(25)
    ]

    text = _summarize(result)

    assert "另有 5 个区域" in text


def test我方窗口能建起来且有输入输出控件() -> None:
    """控件建不起来等于界面是空的，这条钉住最低可用形态。"""

    built: dict[str, tk.Tk] = {}
    original = tk.Tk.mainloop

    def fake_mainloop(self: tk.Tk) -> None:
        built["root"] = self

    tk.Tk.mainloop = fake_mainloop
    try:
        launch_desktop_app(default_output_dir="outputs/_probe/ui")
    finally:
        tk.Tk.mainloop = original

    root = built.get("root")
    assert root is not None
    classes: list[str] = []

    def walk(widget: tk.Misc) -> None:
        for child in widget.winfo_children():
            classes.append(child.winfo_class())
            walk(child)

    walk(root)
    # 两行"标签 + 输入框 + 浏览按钮"、开始按钮、进度条、状态、结果文本框。
    assert classes.count("Entry") == 2
    assert classes.count("Button") >= 3
    assert "TProgressbar" in classes
    assert "Text" in classes
    root.destroy()


def test我方界面失败也必须留痕而不是静默() -> None:
    """我钉住 GUI 的失败路径：错误与堆栈都要进队列，且**顺序**要对。

    我为什么补这条：界面是甲方最直接的使用入口，而"点了开始、
    转圈结束、什么也没说"是最糟的结果——用户会以为"没检测出缺陷"，
    实际是根本没跑起来。这和 HTTP 端那条竞态是同一类问题：
    失败必须**留痕且可见**，不能静默。

    这里不测控件怎么显示（那需要真窗口），而测**事件队列里有什么**——
    显示层读的就是这个队列，队列里没有就等于界面上永远看不到。
    """

    queue_module = pytest.importorskip("queue")
    events: "queue_module.Queue[tuple[str, object]]" = queue_module.Queue()

    # 故意指向一个不存在的候选件，逼 _run_and_report 走失败分支。
    _run_and_report(
        "outputs/_probe/definitely_missing.stl",
        "outputs/_probe/definitely_missing.stp",
        "outputs/_probe/ui",
        events,
    )

    kinds: list[str] = []
    payloads: dict[str, object] = {}
    while not events.empty():
        kind, payload = events.get_nowait()
        kinds.append(kind)
        payloads[kind] = payload

    assert "failed" in kinds, f"失败事件没进队列，界面会静默：{kinds}"
    assert "traceback" in kinds, "没有堆栈，事后没法定位炸在哪一步。"
    # 堆栈必须在 failed 之后：界面用 failed 弹错误框、用 traceback 填文本框，
    # 顺序反了会先清空文本再填，视觉上闪一下，不是正确性问题但我要钉住。
    assert kinds.index("traceback") > kinds.index("failed")
    assert str(payloads["failed"]).strip()
    assert "Traceback" in str(payloads["traceback"])
