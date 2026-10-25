"""我把 HTTP 服务放在独立区域，避免它成为几何基线的必需依赖。

我为什么把服务单独开一个包：FastAPI/uvicorn 是**可选**依赖，项目门禁里
不强制安装。放在独立包里、只在真正启动服务时才导入，缺了它也只是"起不了
服务"，几何检测、BOM、分型、修补全部照常可用——这一点和
``learning/model.py`` 里延迟导入 torch 是同一个原则。
"""

from __future__ import annotations

from .service import create_app

__all__ = ["create_app"]
