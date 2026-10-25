"""我把 STEP 面区域分类结果写成 UTF-8 JSON。

这份报告带内容摘要（``with_content_digest``），所以下游能验证报告有没有被改过：重算
规范化内容的 sha256 与报告里的 ``content_sha256`` 比对即可。审计器
（``artifact_audit``）已经自动做这一步，半截写入或被篡改的报告会被判 INVALID。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..domain.step_features import StepFeatureClassificationReport
from .atomic import atomic_path


def write_step_feature_classification_json(
    report: StepFeatureClassificationReport,
    output_path: str | Path,
) -> Path:
    """我输出可审计的上下平面、侧壁和邻接候选报告。"""

    target = Path(output_path)
    # allow_nan=False：非有限值直接报错；报告带内容摘要，可被审计器复核完整性。
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
