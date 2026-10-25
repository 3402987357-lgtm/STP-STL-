"""我把局部修补结果写成 JSON 和 CSV，便于甲方复核。

JSON 是权威结果（含每个孔洞的面积、周长、平面度误差、凸性与是否需人工复核），CSV 是
逐孔洞的复核表。CSV 里的警告列表用 JSON 序列化成单元格文本，避免 Python 列表 repr 的
逗号被 Excel 当分隔符拆列。两份都走 atomic_path 原子替换，CSV 用 utf-8-sig 带 BOM
保证中文不乱码。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..domain.repair import RepairResult
from .atomic import atomic_path


def write_repair_json(result: RepairResult, path: str | Path) -> Path:
    """我写出完整修补报告。

    为什么强制 allow_nan=False：JSON 标准里没有 NaN/Infinity。
    Python 默认会写出 NaN 这个裸记号，严格解析器（含我们自己的
    审计器）读到就报错，而它在文件里看起来像个正常的数值。
    让写出阶段直接炸掉，比让一个带 NaN 的报告流到下游更好。

    ensure_ascii=False 让中文原样落盘，便于人直接阅读；
    编码显式给 utf-8，不依赖运行环境的默认编码。
    """

    path = Path(path)
    # allow_nan=False：出现非有限值直接报错，不写出打不开的 JSON。
    with atomic_path(path) as temporary:
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
    return path


def write_repair_csv(result: RepairResult, path: str | Path) -> Path:
    """我写出区域级修补表格。

    这是给甲方在 Excel 里逐行看的，不是给程序读的——
    完整信息在 write_repair_json 里。所以每一行必须能独立
    支撑一个判断："这块区域在哪、多大、平不平、我用了什么方法、
    结果能不能信"。缺任何一列，看表的人就得回去翻 JSON。

    review_required 必须单独成列：它是"要不要人工看一眼"的
    结论列。混在 status 里会让人以为它是修补状态，
    从而漏掉真正需要复核的行。
    """

    path = Path(path)
    # 列顺序固定，保证不同批次产出的 CSV 可直接横向比对。
    # 列序按"先定位、再量测、再结论"组织：
    # region_id/defect_type 定位是哪个缺陷，area/perimeter/planarity/is_convex
    # 给出这块区域的几何量，status/method/review_required/warnings 说明
    # 我做了什么、能不能信。甲方在 Excel 里从左往右读即可形成完整判断。
    fields = [
        "region_id",
        "defect_type",
        "status",
        "method",
        "area_mm2",
        "perimeter_mm",
        "planarity_error_mm",
        "is_convex",
        "review_required",
        "warnings",
    ]
    with atomic_path(path) as temporary:
        # warnings 用 json 数组塞进单列：CSV 无法表达列表，
        # 用分号拼接会在警告文本本身含分号时解析错误，json 不会。
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for region in result.repaired_regions:
                writer.writerow(
                    {
                        "region_id": region.region_id,
                        "defect_type": region.defect_type,
                        "status": region.status,
                        "method": region.method,
                        "area_mm2": region.boundary_loop.area_mm2,
                        "perimeter_mm": region.boundary_loop.perimeter_mm,
                        "planarity_error_mm": (
                            region.boundary_loop.planarity_error_mm
                        ),
                        "is_convex": region.boundary_loop.is_convex,
                        "review_required": region.review_required,
                        "warnings": json.dumps(
                            region.warnings,
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                    }
                )
    return path
