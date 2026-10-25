"""我拿真实标注给预测打分——这是唯一能算出任务书那条 99% 的路径。

我为什么要单独写这一层：仓库此前只能"校验标注清单"，却**没有任何地方
用真值给预测打分**。也就是说，即便甲方明天把确认过的标注交过来，
99% 这个数还是算不出来——链子在最后一环断了。合成评估证明不了它
（合成真值只能证明"已知孔洞能不能找回来"），草稿更证明不了
（拿机器预测当真值去评同一个预测器，只会得到 100% 的假准确率）。

所以这个模块只做一件事：**逐件比对人工确认过的真值与预测，给出面级
混淆矩阵**。它遵守三条纪律：

1. **草稿一律不参与打分**。清单里 provenance=draft 的记录是机器预测，
   用它们打分等于自己给自己判满分。我把它们剔除并单独列出来，
   只要剔过就把状态标成 DRAFT_EXCLUDED，绝不让一个含草稿的数字
   看起来像真准确率。
2. **找不到预测的件按漏检算**，而不是静默跳过。静默跳过会让
   "漏了整整一件"从召回率里消失，那正是最该被看见的错误。
3. **面级口径**：真值与预测都展开成面号集合，
   用交集算 TP/FP/FN。区域级匹配（质心距离、IoU）留待与甲方
   约定命中口径后再加——任务书目前写的是"识别误差 >= 0.15mm 的
   缺失区域"，面级是能直接对上这句话的口径。

**多余材料（extra）在 2026-10-25 接上了**（此前这里写的是"未打分"）。
我实测过两侧的真值位置：`outputs/**/result.json` 里 extra 区域的
`reference_face_ids` 恒为空、`candidate_face_ids` 非空（缺料正相反）。
所以打分器改成**按缺陷类型选真值字段**：missing 用参考网格面号，
extra 用候选网格面号，绝不混用。这一条不只是"多支持一种类型"——
检测器对每件真实件都会报 extra，CAD 面映射、可视化、合成评估
（`--defect-type extra`）全都支持，只有"人工确认 -> 打分"这一段断着，
后果是甲方标了多余材料也算不出这一类结论的准确率。

我另外加了一条防止"假 0 召回"的守卫：extra 的真值字段在 1.1 之前的
清单里不存在，按空真值算会把每一个预测都记成假阳性、召回记成 0，
看起来却像一个正常数字。这种情况我直接报错，不返回数字。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .annotations import _normalized_path

# 两类缺陷都能打分：真值字段按类型选，见 _TRUTH_FACE_FIELD。
_SCORABLE_DEFECT_TYPES = ("missing", "extra")
# 每种缺陷的真值面号落在哪一侧：缺料在参考网格上，多余材料在候选网格上。
_TRUTH_FACE_FIELD = {
    "missing": "reference_face_ids",
    "extra": "candidate_face_ids",
}


def _face_ids(regions: Sequence[Mapping[str, Any]], defect_type: str) -> set[int]:
    """我把若干区域展开成一个面号集合，只取指定缺陷类型。

    我按 ``_TRUTH_FACE_FIELD`` 选字段而不是写死 reference_face_ids：
    写死会把 extra 的真值变成空集——那正是"这一类永远算不出准确率"的根因。
    """

    field = _TRUTH_FACE_FIELD[defect_type]
    faces: set[int] = set()
    for region in regions:
        if str(region.get("defect_type")) != defect_type:
            continue
        for face_id in region.get(field) or []:
            faces.add(int(face_id))
    return faces


def _ratio(numerator: float, denominator: float) -> float:
    """我算比值；分母为 0 时返回 0.0 而不是抛错或 NaN。"""

    return float(numerator) / float(denominator) if denominator else 0.0


def collect_predictions(root: str | Path) -> dict[str, dict[str, Any]]:
    """我扫描一个输出目录，把每份 result.json 按**归一化候选路径**索引。

    为什么按候选件路径而不是目录名：目录名（如 `训练集__8`）是产出侧的
    命名约定，会随调用方式变化；而 result.json 里记的 candidate_file
    是数据本身，两侧都拿得到，才是稳的键。
    """

    base = Path(root).expanduser()
    index: dict[str, dict[str, Any]] = {}
    for result_path in sorted(base.rglob("result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 坏文件不参与打分，但也不能假装它不存在——调用方可以从
            # 返回的索引里看出少了哪件（找不到预测会按漏检算）。
            continue
        candidate = payload.get("candidate_file")
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        index[_normalized_path(candidate)] = payload
    return index


def score_annotations(
    payload: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    defect_type: str = "missing",
) -> dict[str, Any]:
    """我逐件比对真实标注与预测，返回面级混淆矩阵与汇总。

    ``predictions`` 的键是归一化后的候选件路径（见 collect_predictions）。
    """

    if defect_type not in _SCORABLE_DEFECT_TYPES:
        raise ValueError(
            f"我只支持对 {list(_SCORABLE_DEFECT_TYPES)} 打分，当前收到 {defect_type!r}。"
        )
    truth_field = _TRUTH_FACE_FIELD[defect_type]
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("我要求标注清单里 records 是非空数组。")

    # 守卫：真值字段整个缺失时，按空真值分会把每个预测都算成假阳性、
    # 召回算成 0——那是一个看起来正常的数字，却完全是"没数据"造成的。
    # 1.0 的存量清单里就没有 candidate_face_ids，所以这条必须挡在前面。
    regions_without_truth_field = 0
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for region in record.get("regions") or []:
            if not isinstance(region, Mapping):
                continue
            if str(region.get("defect_type")) != defect_type:
                continue
            if truth_field not in region:
                regions_without_truth_field += 1
    if regions_without_truth_field:
        raise ValueError(
            f"我发现 {regions_without_truth_field} 个 {defect_type} 区域没有 "
            f"{truth_field} 字段：这份清单是 1.1 之前的版本或被人手改过，"
            "按空真值打分会把它们整批算成漏检，我不返回这种数字。"
            "请用 current 版本的 bootstrap-annotations 重新生成草稿并复核。"
        )

    per_record: list[dict[str, Any]] = []
    draft_sample_ids: list[str] = []
    no_prediction_sample_ids: list[str] = []
    unsupported_region_types: set[str] = set()
    tp = fp = fn = 0

    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("我要求每条 record 都是 JSON 对象。")
        sample_id = str(record.get("sample_id"))
        provenance = str(record.get("provenance") or "confirmed")
        regions = record.get("regions") or []
        for region in regions:
            if isinstance(region, Mapping):
                kind = str(region.get("defect_type"))
                if kind not in _SCORABLE_DEFECT_TYPES:
                    unsupported_region_types.add(kind)

        # 纪律一：草稿不参与打分。
        if provenance != "confirmed":
            draft_sample_ids.append(sample_id)
            continue

        ground_truth = _face_ids(regions, defect_type)
        candidate_key = _normalized_path(str(record.get("candidate_file") or ""))
        prediction = predictions.get(candidate_key)
        if prediction is None:
            # 纪律二：找不到预测 = 整件漏检，全部真值进 FN。
            no_prediction_sample_ids.append(sample_id)
            predicted: set[int] = set()
        else:
            predicted = _face_ids(prediction.get("regions") or [], defect_type)

        record_tp = len(ground_truth & predicted)
        record_fp = len(predicted - ground_truth)
        record_fn = len(ground_truth - predicted)
        tp += record_tp
        fp += record_fp
        fn += record_fn
        per_record.append(
            {
                "sample_id": sample_id,
                "split": record.get("split"),
                "ground_truth_face_count": len(ground_truth),
                "predicted_face_count": len(predicted),
                "true_positive": record_tp,
                "false_positive": record_fp,
                "false_negative": record_fn,
                "precision": _ratio(record_tp, record_tp + record_fp),
                "recall": _ratio(record_tp, record_tp + record_fn),
                "iou": _ratio(record_tp, record_tp + record_fp + record_fn),
                "prediction_found": prediction is not None,
            }
        )

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    scored = [r for r in per_record]
    macro_recall = (
        sum(float(r["recall"]) for r in scored) / len(scored) if scored else 0.0
    )
    # 状态里必须能一眼看出"这个数含不含草稿""是不是有件没跑到"。
    if not scored:
        status = "NO_CONFIRMED_RECORDS"
    elif draft_sample_ids:
        status = "DRAFT_EXCLUDED"
    else:
        status = "SCORED"
    return {
        "status": status,
        "defect_type": defect_type,
        # 我把这次用哪个字段当真值一起写进结果：事后核对时不用再猜
        # "这个 recall 是拿参考面号还是候选面号算的"。
        "truth_face_field": truth_field,
        "scored_record_count": len(scored),
        "draft_excluded_sample_ids": draft_sample_ids,
        "no_prediction_sample_ids": no_prediction_sample_ids,
        "unsupported_region_types": sorted(unsupported_region_types),
        "aggregate": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2.0 * precision * recall, precision + recall),
            "iou": _ratio(tp, tp + fp + fn),
            "macro_recall": macro_recall,
        },
        "records": per_record,
        "limitations": [
            "面级口径：真值与预测都展开成同一侧的面号集合，"
            "不做法向/距离容差。",
            "草稿（provenance=draft）一律不参与打分，只列出 sample_id。",
            "找不到预测的件按整件漏检计入 FN，不跳过。",
            "missing 的真值面号在参考网格上，extra 的真值面号在**候选网格**"
            "（三角化结果）上：extra 必须与预测使用同一套三角化参数，"
            "否则编号不可比（草稿里的 source_linear_deflection_mm 就是这套参数）。",
        ],
    }


def score_annotations_multi(
    payload: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    defect_types: Sequence[str] = ("missing", "extra"),
) -> dict[str, Any]:
    """我一次跑多类缺陷，分别出报告，不把两类混成一个数字。

    我为什么必须分开报：缺料与多余材料的真值在两套不同的网格上，
    混算出来的 recall 既不是缺料的也不是多余的，谁也复核不了。
    整体状态只在**每一类都算得出来**时才是 SCORED；只要有确认记录
    但某一类还缺数据，就是 PARTIAL——那是个明确的"没算完"。
    """

    reports: dict[str, dict[str, Any]] = {}
    for defect_type in defect_types:
        reports[str(defect_type)] = score_annotations(
            payload, predictions, defect_type=str(defect_type)
        )
    if all(report["status"] == "SCORED" for report in reports.values()):
        status = "SCORED"
    elif any(report["scored_record_count"] > 0 for report in reports.values()):
        status = "PARTIAL"
    else:
        status = "NO_CONFIRMED_RECORDS"
    return {
        "status": status,
        "defect_types": [str(item) for item in defect_types],
        "reports": reports,
    }
