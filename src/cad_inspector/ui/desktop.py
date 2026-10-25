"""我提供一个 tkinter 桌面界面，用来选件、跑检测、看结论。

我守住的三条边界：

1. **判定逻辑一律走 pipeline**，界面不重新实现任何判据。界面与命令行
   必须给出同一套结论，否则甲方会拿到两份互相矛盾的报告。
2. **检测放在后台线程跑**。单件要几十秒，直接在界面线程里跑会把窗口
   卡死成"未响应"，用户只能杀进程。我用队列把进度回传给界面线程，
   因为 tkinter 不是线程安全的，**只有主线程能碰控件**。
3. **失败要显示出来**，而不是静默无反应。跑挂了界面上必须看得到
   错误和堆栈，否则用户只会以为"软件坏了"。
"""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from tkinter import (
    BOTH,
    DISABLED,
    END,
    HORIZONTAL,
    LEFT,
    NORMAL,
    RIGHT,
    StringVar,
    filedialog,
    messagebox,
)
from tkinter import Button, Entry, Frame, Label, Scrollbar, Text, Tk
from tkinter.ttk import Progressbar
from typing import Any


def _run_and_report(
    candidate: str,
    reference: str,
    output_dir: str,
    events: "queue.Queue[tuple[str, Any]]",
) -> None:
    """我在后台线程跑检测，把进度与结果通过队列回传。

    我为什么用队列而不是直接改控件：tkinter 不是线程安全的，
    从子线程改控件会随机崩溃或静默出错。子线程只往队列里放事件，
    由界面线程的定时器取出来再改控件。
    """

    try:
        from ..pipeline import inspect_pair

        events.put(("progress", "我正在检测，请稍候…"))
        result = inspect_pair(candidate, reference, output_dir)
        events.put(("done", result.to_dict()))
    except Exception as error:  # noqa: BLE001 - 失败必须完整留痕
        events.put(("failed", f"{type(error).__name__}: {error}"))
        events.put(("traceback", traceback.format_exc()))


def _summarize(result: dict[str, Any]) -> str:
    """我把检测结果整理成一段人能直接读的中文摘要。"""

    regions = result.get("regions", [])
    missing = [item for item in regions if item.get("defect_type") == "missing"]
    extra = [item for item in regions if item.get("defect_type") == "extra"]
    review = [item for item in regions if item.get("review_required")]
    lines = [
        f"结论状态：{result.get('status')}",
        f"阈值：{result.get('threshold_mm')} mm，单位：{result.get('unit')}"
        f"（来源：{result.get('unit_source')}）",
        "",
        f"区域总数 {len(regions)}：缺料 {len(missing)}、多料 {len(extra)}、"
        f"待复核 {len(review)}",
    ]
    for index, region in enumerate(regions[:20], start=1):
        lines.append(
            f"  {index}. {region.get('defect_type')} "
            f"面积 {float(region.get('area_mm2', 0.0)):.3f} mm2，"
            f"最大偏差 {float(region.get('max_error_mm', 0.0)):.3f} mm"
            + ("（待复核）" if region.get("review_required") else "")
        )
    if len(regions) > 20:
        lines.append(f"  ……另有 {len(regions) - 20} 个区域，详见输出目录。")
    warnings = result.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append(f"警告 {len(warnings)} 条：")
        lines.extend(f"  - {item}" for item in warnings[:10])
    artifacts = result.get("artifacts") or {}
    if artifacts:
        lines.append("")
        lines.append("产物：")
        for key, value in artifacts.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def launch_desktop_app(
    default_output_dir: str = "outputs/desktop_inspect",
) -> None:
    """我启动桌面窗口。

    这个函数会阻塞到用户关闭窗口，所以它只在用户显式要求界面时调用。
    """

    root = Tk()
    root.title("我方缺失区域检测")
    root.geometry("900x640")

    candidate_var = StringVar()
    reference_var = StringVar()
    status_var = StringVar(value="我等待你选择候选件与参考件。")
    events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    running = {"flag": False}

    def _pick(variable: StringVar, title: str) -> None:
        """我弹出文件选择框并把结果填进输入框。"""

        selected = filedialog.askopenfilename(
            title=title,
            filetypes=[("模型文件", "*.stl *.stp *.step"), ("所有文件", "*.*")],
        )
        if selected:
            variable.set(selected)

    def _row(parent: Frame, label: str, variable: StringVar, title: str) -> None:
        """我造一行"标签 + 输入框 + 浏览按钮"。"""

        line = Frame(parent)
        line.pack(fill="x", pady=4)
        Label(line, text=label, width=10, anchor="w").pack(side=LEFT)
        Entry(line, textvariable=variable).pack(side=LEFT, fill="x", expand=True)
        Button(line, text="浏览", command=lambda: _pick(variable, title)).pack(
            side=RIGHT
        )

    top = Frame(root)
    top.pack(fill="x", padx=10, pady=10)
    _row(top, "候选件", candidate_var, "我选择待检测模型")
    _row(top, "参考件", reference_var, "我选择参考模型")

    button_bar = Frame(root)
    button_bar.pack(fill="x", padx=10)
    start_button = Button(button_bar, text="开始检测")
    start_button.pack(side=LEFT)
    progress = Progressbar(button_bar, mode="indeterminate", orient=HORIZONTAL)

    status_label = Label(root, textvariable=status_var, anchor="w", justify=LEFT)
    status_label.pack(fill="x", padx=10, pady=(6, 0))

    text = Text(root, wrap="word")
    text.pack(fill=BOTH, expand=True, padx=10, pady=6)
    scroll = Scrollbar(text)
    scroll.pack(side=RIGHT, fill="y")
    text.configure(yscrollcommand=scroll.set)
    scroll.configure(command=text.yview)

    def _drain() -> None:
        """我在界面线程里取出后台事件并更新控件。"""

        try:
            while True:
                kind, payload = events.get_nowait()
                if kind == "progress":
                    status_var.set(str(payload))
                elif kind == "done":
                    running["flag"] = False
                    progress.stop()
                    progress.pack_forget()
                    start_button.configure(state=NORMAL)
                    status_var.set("我已完成检测。")
                    text.delete("1.0", END)
                    text.insert(END, _summarize(payload))
                elif kind == "failed":
                    running["flag"] = False
                    progress.stop()
                    progress.pack_forget()
                    start_button.configure(state=NORMAL)
                    status_var.set(f"我检测失败：{payload}")
                    messagebox.showerror("我检测失败", str(payload))
                elif kind == "traceback":
                    text.delete("1.0", END)
                    text.insert(END, str(payload))
        except queue.Empty:
            pass
        # 100ms 轮询：足够及时，又不会把主线程占满。
        root.after(100, _drain)

    def _start() -> None:
        """我校验输入并在后台线程启动检测。"""

        candidate = candidate_var.get().strip()
        reference = reference_var.get().strip()
        if not candidate or not reference:
            messagebox.showwarning("我还不能开始", "请分别选择候选件与参考件。")
            return
        if not Path(candidate).is_file() or not Path(reference).is_file():
            messagebox.showwarning("我还不能开始", "我找不到你选择的模型文件。")
            return
        running["flag"] = True
        start_button.configure(state=DISABLED)
        progress.pack(side=LEFT, padx=(8, 0))
        progress.start()
        status_var.set("我正在检测，请稍候…")
        text.delete("1.0", END)
        threading.Thread(
            target=_run_and_report,
            args=(candidate, reference, default_output_dir, events),
            daemon=True,
        ).start()

    start_button.configure(command=_start)
    root.after(100, _drain)
    root.mainloop()
