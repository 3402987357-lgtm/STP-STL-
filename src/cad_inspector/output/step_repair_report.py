"""我把 STEP 修补结果写成 JSON 和 CSV，供甲方复核。

两份载体分工：JSON 含修补前后的完整拓扑、每个区域的 G0/G1/G2 连续性证据与偏差量，
是权威结果；CSV 是每个修补区域一行，供甲方用 Excel 逐条复核"这个区域到底补得怎么样"。

CSV 里的列表字段（来源 Face 编号、警告）一律用 JSON 序列化成单元格文本，避免 Python
列表 repr 的逗号被 Excel 当成分隔符拆列。两份文件都走 atomic_path 原子替换，CSV 用
utf-8-sig 带 BOM 保证中文不乱码。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..domain.step_repair import StepRepairResult
from .atomic import atomic_path


def write_step_repair_json(
    result: StepRepairResult,
    path: str | Path,
) -> Path:
    """我原子写出 STEP 修补的完整质量报告。

    我为什么在这里也要 allow_nan=False：偏差量来自 OCP 的曲面求交，
    真实件上会返回 NaN（例如补面与原面无有效交点时）。Python 的
    json.dumps 默认会把 NaN 写成裸的 ``NaN`` 字样，那不是合法 JSON，
    甲方那边的解析器会直接失败。我宁可这里报错、让人看见"这个件
    补面求交失败了"，也不写出一份打不开的报告。

    我为什么末尾补一个 "\\n"：POSIX 文本文件的约定是以换行结尾，
    缺了它 ``git diff`` 会显示 "\\ No newline at end of file"，
    批量比对时凭空多出噪音行。
    """

    target = Path(path)
    # allow_nan=False：偏差量为非有限值时直接报错，不写出打不开的 JSON。
    with atomic_path(target) as temporary:
        temporary.write_text(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return target


def write_step_repair_csv(
    result: StepRepairResult,
    path: str | Path,
) -> Path:
    """我原子写出每个 STEP 修补区域的复核表。

    我为什么用 utf-8-sig 而不是 utf-8：带 BOM 的 UTF-8 能让 Excel
    双击打开时正确识别中文，不带 BOM 会按系统 ANSI 码页解，中文警告
    全变问号。代价是文件头多三个字节，对这个规模无所谓。

    我为什么把 warnings 放最后一列：警告文本可能很长且含逗号，
    放中间会让人在 Excel 里横向滚动时视线被截断；放末尾既能被
    DictWriter 完整写出，又不干扰前面那些要横向比对的数值列。
    """

    target = Path(path)
    # 列顺序固定，保证不同批次产出的 CSV 可直接横向比对。
    fields = [
        "region_id",
        "damage_type",
        "status",
        "repair_method",
        "area_repaired",
        "continuity_level",
        "continuity_verified",
        "deviation_max_mm",
        "deviation_mean_mm",
        "boundary_edge_count",
        "boundary_vertex_count",
        "planarity_error_mm",
        "g0_verified",
        "g1_verified",
        "g2_verified",
        "g0_max_deviation_mm",
        "g1_max_normal_angle_deg",
        "g2_max_curvature_delta_per_mm",
        "source_face_ids",
        "review_required",
        # 环自身缝长：门禁要看"还有没有环带着实质孔洞"，
        # 这个字段是唯一的判据来源，不写进 CSV 就没法按环复核。
        # 它必须放在 warnings 之前以保持在列尾区的可读位置。
        "loop_gap_mm",
        # 补面是否被采纳：它决定上面这个 loop_gap_mm 还要不要参与
        # 否决整份输出（被采纳的环其 gap 是补面前的量）。
        # 不写进 CSV，复核者就无法核对门禁为什么放行了这个大 gap 的环。
        "patch_accepted",
        "warnings",
    ]
    with atomic_path(target) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for region in result.repaired_regions:
                row = region.to_dict()
                # 列表字段 JSON 序列化，避免被 CSV 分隔符拆开。
                row["source_face_ids"] = json.dumps(
                    row["source_face_ids"],
                    ensure_ascii=False,
                )
                row["warnings"] = json.dumps(
                    row["warnings"],
                    ensure_ascii=False,
                )
                writer.writerow(row)
    return target
