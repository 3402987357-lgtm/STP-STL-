"""我把标注草稿导成人工可填的复核表，再把人填的结果并回清单。

我为什么要补这一环（2026-10-17 查链路时发现的断点）：
任务书要的是"能正确识别≥99%的缺失区域"，而这个数只能拿**人工确认过的
真值**去算。仓库里已经有前半条链——bootstrap_annotations.py 把验收结果
转成带 provenance=draft 的草稿，score-annotations 拿 confirmed 记录打分。
但**中间没有任何工具**：草稿里的 provenance 是 draft，要变成 confirmed
只能手工去编辑那份 155 个区域的大 JSON。这条路在现实中走不通，于是
"99% 算不出来"就永远停在"缺数据"上，而不是"缺一个能让人干活的东西"。

我的做法是导出/导入，而不是做交互式问答：复核是要反复看、要几个人
分工、要留痕的活，交互式命令行做不好，而且没法测。导出一张表、
人在表里填一列、再导回来，是能落地也能用测试钉住的形态。

三条我写死在实现里的纪律：

1. **漏填不等于接受**。没填的行算"未决"，该记录整体保持 draft。
   只有一条记录的所有区域都被明确判定过，它才会变成 confirmed。
   半条记录绝不能冒充真值——那正是"假准确率"最容易进来的缝。
2. **表格必须覆盖草稿的每一个区域**。少一行就报错，不静默跳过。
   否则删掉几行就能让最难判的区域消失，召回率凭空变好。
3. **表格绑定草稿指纹**。表里每一行都带草稿的指纹，导回时核对；
   草稿被重新生成过（我这几轮就重新生成过一次）就必须重填，
   不能拿旧表套新草稿。
4. **整件判定不许填到有区域的件上**（2026-10-25 补）。`scope=record`
   只对"一个区域都没报"的件有意义；有区域的件上多出一行整件判定，
   会在按区域判定的循环里被覆盖——人工填的那一格静默失效，
   填表的人却以为它生效了。我直接报错，不静默忽略。

面号这一列我按缺陷类型取：missing 数参考网格面号，extra 数
**候选网格**面号（多余材料的真值只存在于候选件上）。若像以前那样
一律数 reference_face_ids，extra 行的面数会恒为 0，复核人会以为
"这块多余材料没有面号可判"而把它剔掉，那正好把要打分的真值丢掉。
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..output.atomic import atomic_path

# 我人工填写的两列在表尾，方便连续填完一列。
_DECISION = "decision"
_REVIEWER = "reviewer"
_NOTE = "note"

_ACCEPT = "accept"
_REJECT = "reject"
_UNDECIDED = ""

# 表头顺序即导出顺序：前半是"判断依据"，后半是"人来填"。
SHEET_COLUMNS: tuple[str, ...] = (
    "draft_fingerprint",
    "scope",
    "sample_id",
    "split",
    "region_index",
    "defect_type",
    "area_mm2",
    "center_x_mm",
    "center_y_mm",
    "center_z_mm",
    "max_error_mm",
    "mean_error_mm",
    "p95_error_mm",
    "confidence",
    "face_count",
    "candidate_face_count",
    _DECISION,
    _REVIEWER,
    _NOTE,
)

# scope 取值：region 表示这一行判的是某个区域；record 表示这一行判的是
# 整条记录。我为什么需要 record 这一档（2026-10-17 实跑时踩到）：
# 有一条记录**一个区域都没有**（检测器在这件上没报任何区域）。
# 我最初只按"区域是否都判过"来判断整条是否确认，于是这条记录的
# 区域循环一次都没进、all_decided 恒为 True，**没人填过任何一格
# 就被自动标成 confirmed**。那等于把"机器什么都没找到"直接写成
# "这件确实没有缺料"的真值，而它真实的原因可能只是阈值把缺陷滤掉了——
# 这会把召回率凭空抬高。所以无区域的记录必须要求一次显式的整条判定。
_SCOPE_REGION = "region"
_SCOPE_RECORD = "record"


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """我给草稿内容算一个短指纹，用来把复核表绑死在它对应的那一版草稿上。

    我为什么不用生成时间或来源目录当标识：来源目录不变而内容变了是
    完全可能的（重新生成到同一个目录），时间戳也会被人为改写。
    直接对 records 做规范化序列化再取哈希，内容变则指纹必变。
    """

    canonical = json.dumps(
        manifest.get("records") or [],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _triple(values: Any) -> tuple[str, str, str]:
    """我把三元向量摊成三列，缺值时留空而不是写 0。"""

    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return ("", "", "")
    return tuple("" if v is None else str(v) for v in values)  # type: ignore[return-value]


def _cell(value: Any) -> str:
    """我把单元格值转成字符串；None 写成空串，绝不落成字面量 "None"。"""

    return "" if value is None else str(value)


def _truth_face_count(region: Mapping[str, Any]) -> int:
    """我按缺陷类型数这一行的真值面号：缺料看参考面，多余材料看候选面。

    我为什么不让它恒等于 len(reference_face_ids)：extra 行的真值在候选
    网格上，按参考面数会显示 0——复核人看到"0 个面"会以为这块多余材料
    没有面号可判，直接把它剔掉，而它恰恰是 extra 打分的唯一真值来源。
    """

    defect_type = str(region.get("defect_type") or "missing")
    field = "candidate_face_ids" if defect_type == "extra" else "reference_face_ids"
    return len(region.get(field) or [])


def export_review_sheet(
    manifest: Mapping[str, Any], sheet_path: Path
) -> dict[str, int]:
    """我把草稿导出成复核表，返回行数统计。

    我为什么把区域行和整件行分开报：两者含义不同，混成一个"待判定
    区域数"会让人以为表里有 156 个区域，实际是 155 个区域加 1 条
    "这件没有区域"的整件判定。
    """

    fingerprint = manifest_fingerprint(manifest)
    rows: list[dict[str, str]] = []
    region_rows = 0
    record_rows = 0
    for record in manifest.get("records") or []:
        sample_id = _cell(record.get("sample_id"))
        split = _cell(record.get("split"))
        regions = list(record.get("regions") or [])
        for region_index, region in enumerate(regions):
            center_x, center_y, center_z = _triple(region.get("center_xyz_mm"))
            rows.append(
                {
                    "draft_fingerprint": fingerprint,
                    "scope": _SCOPE_REGION,
                    "sample_id": sample_id,
                    "split": split,
                    "region_index": str(region_index),
                    "defect_type": _cell(region.get("defect_type")),
                    "area_mm2": _cell(region.get("area_mm2")),
                    "center_x_mm": center_x,
                    "center_y_mm": center_y,
                    "center_z_mm": center_z,
                    "max_error_mm": _cell(region.get("max_error_mm")),
                    "mean_error_mm": _cell(region.get("mean_error_mm")),
                    "p95_error_mm": _cell(region.get("p95_error_mm")),
                    "confidence": _cell(region.get("confidence")),
                    "face_count": str(_truth_face_count(region)),
                    "candidate_face_count": str(
                        len(region.get("candidate_face_ids") or [])
                    ),
                    _DECISION: _UNDECIDED,
                    _REVIEWER: "",
                    _NOTE: "",
                }
            )
        if not regions:
            # 无区域的记录也要占一行，否则它没有任何格子可填，
            # 就只能被"空循环"默认成已确认。
            rows.append(
                {
                    "draft_fingerprint": fingerprint,
                    "scope": _SCOPE_RECORD,
                    "sample_id": sample_id,
                    "split": split,
                    "region_index": "",
                    "defect_type": "",
                    "area_mm2": "",
                    "center_x_mm": "",
                    "center_y_mm": "",
                    "center_z_mm": "",
                    "max_error_mm": "",
                    "mean_error_mm": "",
                    "p95_error_mm": "",
                    "confidence": "",
                    "face_count": "0",
                    "candidate_face_count": "0",
                    _DECISION: _UNDECIDED,
                    _REVIEWER: "",
                    _NOTE: "",
                }
            )
            record_rows += 1
        region_rows += len(regions)

    sheet_path = Path(sheet_path)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    # 用 utf-8-sig 带 BOM：这份表是要人用 Excel 打开填的，
    # 不带 BOM 的中文表头在 Excel 里会变乱码。
    with atomic_path(sheet_path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(SHEET_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
    return {
        "rows": len(rows),
        "region_rows": region_rows,
        "record_rows": record_rows,
    }


def _read_sheet(sheet_path: Path) -> list[dict[str, str]]:
    """我读回填好的复核表，先核对表头一列不缺；缺列直接报错不猜。"""

    with Path(sheet_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in SHEET_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"复核表缺少这些列：{missing}")
        return [dict(row) for row in reader]


def apply_review_sheet(
    manifest: Mapping[str, Any],
    sheet_path: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    """我把填好的复核表并回清单，返回（新清单，统计）。"""

    rows = _read_sheet(Path(sheet_path))
    expected = manifest_fingerprint(manifest)

    # 先核对指纹：拿旧表套新草稿会让判定对到别的区域上，必须挡住。
    seen_fingerprints = {row.get("draft_fingerprint") or "" for row in rows}
    seen_fingerprints.discard("")
    if seen_fingerprints and seen_fingerprints != {expected}:
        raise ValueError(
            "这份复核表不是对着当前草稿填的："
            f"表里的指纹是 {sorted(seen_fingerprints)}，当前草稿是 {expected}。"
            "草稿已经重新生成过，请重新导出复核表再填。"
        )

    decisions: dict[tuple[str, int], str] = {}
    record_decisions: dict[str, str] = {}
    reviewers: dict[tuple[str, int], str] = {}
    record_reviewers: dict[str, str] = {}
    notes: dict[tuple[str, int], str] = {}
    record_notes: dict[str, str] = {}
    for line_number, row in enumerate(rows, start=2):
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"复核表第 {line_number} 行缺少 sample_id。")
        decision = (row.get(_DECISION) or "").strip().lower()
        if decision not in (_ACCEPT, _REJECT, _UNDECIDED):
            raise ValueError(
                f"复核表第 {line_number} 行的 decision 我不认识：{decision!r}"
                f"（只接受 {_ACCEPT} / {_REJECT} / 留空）"
            )
        reviewer = (row.get(_REVIEWER) or "").strip()
        note = (row.get(_NOTE) or "").strip()
        scope = (row.get("scope") or _SCOPE_REGION).strip() or _SCOPE_REGION

        if scope == _SCOPE_RECORD:
            if sample_id in record_decisions:
                raise ValueError(f"复核表里 {sample_id} 的整条判定出现了多次。")
            record_decisions[sample_id] = decision
            record_reviewers[sample_id] = reviewer
            record_notes[sample_id] = note
            continue
        if scope != _SCOPE_REGION:
            raise ValueError(
                f"复核表第 {line_number} 行的 scope 我不认识：{scope!r}"
                f"（只接受 {_SCOPE_REGION} / {_SCOPE_RECORD}）"
            )

        raw_index = (row.get("region_index") or "").strip()
        if not raw_index:
            raise ValueError(f"复核表第 {line_number} 行缺少 region_index。")
        key = (sample_id, int(raw_index))
        if key in decisions:
            raise ValueError(f"复核表里 {sample_id} 的区域 {raw_index} 出现了多次。")
        decisions[key] = decision
        reviewers[key] = reviewer
        notes[key] = note

    accepted = rejected = undecided = 0
    confirmed_records = 0
    draft_records = 0
    new_records: list[dict[str, Any]] = []

    # 整件判定只对"一个区域都没报"的件有意义。有区域的件上多填一行整件
    # 判定，会在下面的按区域判定循环里被覆盖——那是**静默丢掉人工填的
    # 一格**，和漏填一样危险（填表的人会以为自己判过了）。我直接报错。
    records_with_regions = {
        _cell(record.get("sample_id"))
        for record in manifest.get("records") or []
        if list(record.get("regions") or [])
    }
    unexpected_record_rows = sorted(set(record_decisions) & records_with_regions)
    if unexpected_record_rows:
        raise ValueError(
            "复核表里给这些有区域的件也填了整件判定（scope=record）："
            f"{unexpected_record_rows}。整件判定只用于一个区域都没报的件，"
            "有区域的件必须按区域逐行判定，否则那一格会被静默忽略。"
        )

    for record in manifest.get("records") or []:
        sample_id = _cell(record.get("sample_id"))
        kept_regions: list[dict[str, Any]] = []
        # 注意：这里的两个局部变量不能叫 record_reviewers / record_notes，
        # 那会盖住上面从表格读出来的同名字典，整条判定的评审人与备注就丢了。
        reviewer_names: set[str] = set()
        note_texts: list[str] = []
        all_decided = True
        regions = list(record.get("regions") or [])

        if not regions:
            # 无区域的记录：必须有一次显式的整条判定，否则保持草稿。
            # 空循环绝不能被当成"都判过了"。
            # 我也要求这一行**必须在表里**：只是"没填"和"整行被删掉"
            # 要分开——前者是还没判，后者多半是误删，静默当成未决
            # 会让人以为复核已经做完了。
            if sample_id not in record_decisions:
                raise ValueError(
                    f"复核表漏了 {sample_id} 的整件判定行；"
                    "这件没有任何区域，删掉这一行就等于它永远不会被确认。"
                )
            decision = record_decisions[sample_id]
            if decision == _UNDECIDED:
                all_decided = False
                undecided += 1
            elif decision == _ACCEPT:
                accepted += 1
            else:
                rejected += 1
            if record_reviewers.get(sample_id):
                reviewer_names.add(record_reviewers[sample_id])
            if record_notes.get(sample_id):
                note_texts.append(record_notes[sample_id])

        for region_index, region in enumerate(regions):
            key = (sample_id, region_index)
            if key not in decisions:
                raise ValueError(
                    f"复核表漏了 {sample_id} 的区域 {region_index}；"
                    "少一行就等于让这个区域消失，我不接受。"
                )
            decision = decisions[key]
            if decision == _UNDECIDED:
                all_decided = False
                undecided += 1
                kept_regions.append(dict(region))
                continue
            if decision == _ACCEPT:
                accepted += 1
                kept_regions.append(dict(region))
            else:
                rejected += 1
            if reviewers[key]:
                reviewer_names.add(reviewers[key])
            if notes[key]:
                note_texts.append(notes[key])

        new_record = dict(record)
        new_record["regions"] = kept_regions
        if all_decided:
            # 整条都判过了才算真值；全判成 reject 也是合法真值，
            # 表示"这件确实没有缺失区域"——负样本同样是有用的标注。
            new_record["provenance"] = "confirmed"
            new_record["review_state"] = "confirmed"
            confirmed_records += 1
        else:
            new_record["provenance"] = "draft"
            new_record["review_state"] = "in_review"
            draft_records += 1
        if reviewer_names:
            new_record["reviewers"] = sorted(reviewer_names)
        if note_texts:
            new_record["review_notes"] = note_texts
        new_records.append(new_record)

    new_manifest = dict(manifest)
    new_manifest["records"] = new_records
    # 顶层 provenance 只是给人看的第一眼提示；真正的判定按每条记录算，
    # 所以只有全部确认才把它写成 confirmed。
    new_manifest["provenance"] = (
        "confirmed" if confirmed_records and draft_records == 0 else "draft"
    )
    new_manifest["review_summary"] = {
        "accepted_region_count": accepted,
        "rejected_region_count": rejected,
        "undecided_region_count": undecided,
        "confirmed_record_count": confirmed_records,
        "draft_record_count": draft_records,
    }
    summary = {
        "accepted": accepted,
        "rejected": rejected,
        "undecided": undecided,
        "confirmed_records": confirmed_records,
        "draft_records": draft_records,
    }
    return new_manifest, summary
