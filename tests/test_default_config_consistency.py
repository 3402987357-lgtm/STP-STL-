"""我钉住随附的默认配置与代码默认值一致，不许再漂移。

为什么值得单独一个文件：README 的「第一版运行方式」示例就是
`--config configs\\default.json`，也就是说甲方照抄命令时用的就是这个文件。
而 2026-10-17 我已经公告过「默认配准从 centroid 改成 auto」——实测
centroid 会把缺料件的质心对齐、把整张表面推开，20 件真实件的缺失面积比
被判成 13.3%~95.7%（见 reports/registration_default_change_evidence.json）。
如果配置文件没跟着改，照 README 抄命令的人就会静默退回这个已被否决的口径。

这个坑的形状是「同一个默认值分散在两处」：dataclass 一处、JSON 一处，
改了一处另一处不会报错，只会安静地覆盖回去。所以必须有护栏。
"""

from __future__ import annotations

import json
from pathlib import Path

from cad_inspector.models import InspectionConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.json"

# 这些键一旦在配置文件和代码默认值之间不一致，改变的是**检测口径**
# （检出什么、算不算缺失），不是运行便利性，所以必须逐字相等。
# 我故意不列 write_intermediate：它是"要不要留中间产物"的开关，
# 配置文件里开成 true 是有意的演示选择，不影响结论。
_VERDICT_KEYS = (
    "registration_mode",
    "threshold_mode",
    "threshold_mm",
    "min_region_area_mm2",
    "min_triangle_count",
    "allow_scale",
    "samples_per_triangle",
    "max_query_points",
    "distance_backend",
    "distance_candidate_k",
    "distance_chunk_size",
    "bidirectional_check",
    "unit_policy",
    "mesh_linear_deflection_mm",
    "mesh_angular_deflection_deg",
    "write_colored_mesh",
    "auto_threshold_scale_ratio",
    "auto_threshold_residual_factor",
    "auto_threshold_floor_mm",
)


def _payload() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test我方随附默认配置的配准模式与代码默认一致() -> None:
    """我钉住 centroid/auto 这次口径变更真的落到了配置文件上。"""

    payload = _payload()
    config = InspectionConfig()
    assert payload["registration_mode"] == config.registration_mode, (
        f"configs/default.json 的 registration_mode={payload['registration_mode']!r} "
        f"与代码默认 {config.registration_mode!r} 不一致："
        "照 README 示例加 --config 运行的人会用到已被否决的配准口径。"
    )


def test我方随附默认配置的判定口径键逐条与代码默认一致() -> None:
    """我把所有会改变结论的默认值都钉住，避免下一次又只改一处。"""

    payload = _payload()
    config = InspectionConfig()
    drifted = {
        key: (payload[key], getattr(config, key))
        for key in _VERDICT_KEYS
        if key in payload and payload[key] != getattr(config, key)
    }
    assert not drifted, f"这些判定口径键在配置文件与代码默认值之间已漂移：{drifted}"