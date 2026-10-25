"""我把桌面界面放在独立区域。

我为什么选 tkinter：它随 Python 自带，不引入新的重量级依赖，
也不需要本机装 Qt 运行时。这个项目的一贯原则是**可选依赖延迟到使用时**，
桌面界面同样如此——只有真正启动界面时才会 import tkinter，
命令行、批处理、HTTP 服务都不受它影响。

界面本身只做三件事：选文件、跑检测、看结论。它不重新实现任何判定逻辑，
判定全部走 ``pipeline.inspect_pair``，保证界面上看到的结论和命令行
产出的 JSON **是同一套口径**——两条路径给出不同结论是最危险的情况。
"""

from __future__ import annotations

from .desktop import launch_desktop_app

__all__ = ["launch_desktop_app"]
