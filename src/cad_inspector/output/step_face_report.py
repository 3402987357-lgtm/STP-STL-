"""我把 STEP Face 级特征报告写成 UTF-8 JSON。

Face 级报告是整个 STEP 链路里最大的产物之一（大模型可达数十 MiB），所以它最需要原子
写出：写到一半被中断会留下一个几十 MiB 的坏文件，既占盘又骗过"文件存在"的检查。
``allow_nan=False`` 保证不会把 NaN 写进 JSON——NaN 不是合法 JSON，下游解析必然失败。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..io.step_faces import StepFaceFeatureReport
from .atomic import atomic_path


def write_step_face_features_json(
    report: StepFaceFeatureReport,
    output_path: str | Path,
) -> Path:
    """我输出可审计的 Face 级几何特征报告。"""

    target = Path(output_path)
    # allow_nan=False：非有限值直接报错，不写出违反 JSON 规范的坏文件。
    with atomic_path(target) as temporary:
        temporary.write_text(
            json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return target
