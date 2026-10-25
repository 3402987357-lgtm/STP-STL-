"""我给"草稿 -> 人工复核 -> 确认清单"这条链写用例。

我为什么要补这一环：任务书那条"≥99%"只能拿人工确认过的真值算，
而仓库里原本只有两头——bootstrap 生成草稿、scorer 给确认过的清单打分，
中间**没有任何工具**。草稿要变真值只能手工编辑那份 155 个区域的大 JSON。

这里的用例重点不在"能导出 CSV"，而在几条**不许悄悄放行**的纪律：
留空不等于接受、漏行必须报错、旧表不许套新草稿，以及一条我实跑时
真的踩到的：**一个区域都没有的件，不能被空循环默认成已确认**。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cad_inspector.learning.annotation_review import (
    apply_review_sheet,
    export_review_sheet,
    manifest_fingerprint,
)
from cad_inspector.learning.annotations import validate_annotation_manifest


def _region(face_ids: list[int], defect_type: str = "missing") -> dict:
    return {
        "defect_type": defect_type,
        "reference_face_ids": face_ids,
        "center_xyz_mm": [1.0, 2.0, 3.0],
        "bbox_min_xyz_mm": [0.0, 0.0, 0.0],
        "bbox_max_xyz_mm": [2.0, 2.0, 2.0],
        "area_mm2": 4.0,
        "max_error_mm": 0.9,
        "mean_error_mm": 0.5,
        "p95_error_mm": 0.8,
        "confidence": 0.7,
    }


def _manifest() -> dict:
    """我造一份最小草稿：两条有区域，一条**一个区域都没有**。"""

    return {
        "schema_version": "missing-region-annotations-1.0",
        "provenance": "draft",
        "note": "机器预测，未经人工确认。",
        "records": [
            {
                "sample_id": "train_0",
                "split": "train",
                "reference_file": "D:/d/ref0.stl",
                "candidate_file": "D:/d/cand0.stp",
                "provenance": "draft",
                "review_state": "unconfirmed",
                "regions": [_region([1, 2]), _region([5])],
            },
            {
                "sample_id": "test_1",
                "split": "test",
                "reference_file": "D:/d/ref1.stl",
                "candidate_file": "D:/d/cand1.stp",
                "provenance": "draft",
                "review_state": "unconfirmed",
                "regions": [_region([3, 4])],
            },
            {
                "sample_id": "test_2",
                "split": "test",
                "reference_file": "D:/d/ref2.stl",
                "candidate_file": "D:/d/cand2.stp",
                "provenance": "draft",
                "review_state": "unconfirmed",
                "regions": [],
            },
        ],
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _export(tmp_path: Path) -> tuple[dict, Path, list[dict[str, str]]]:
    manifest = _manifest()
    sheet = tmp_path / "sheet.csv"
    export_review_sheet(manifest, sheet)
    return manifest, sheet, _read_rows(sheet)


def test我方导出把无区域的件也占一行(tmp_path: Path) -> None:
    """一个区域都没有的件也必须有格子可填，否则它无从被判定。"""

    manifest, sheet, rows = _export(tmp_path)

    assert len(rows) == 4  # 3 个区域 + 1 条整件
    record_rows = [r for r in rows if r["scope"] == "record"]
    assert [r["sample_id"] for r in record_rows] == ["test_2"]
    assert all(r["decision"] == "" for r in rows)
    assert rows[0]["draft_fingerprint"] == manifest_fingerprint(manifest)


def test我方留空的行不会被当成接受(tmp_path: Path) -> None:
    manifest, sheet, _rows = _export(tmp_path)

    confirmed, summary = apply_review_sheet(manifest, sheet)

    assert summary["confirmed_records"] == 0
    assert summary["undecided"] == 4
    info = validate_annotation_manifest(confirmed)
    assert info.confirmed_record_count == 0
    assert info.ready_for_training is False


def test我方全部填接受才得到可训练的确认清单(tmp_path: Path) -> None:
    manifest, sheet, rows = _export(tmp_path)
    for row in rows:
        row["decision"] = "accept"
        row["reviewer"] = "老张"
    _write_rows(sheet, rows)

    confirmed, summary = apply_review_sheet(manifest, sheet)

    assert summary == {
        "accepted": 4,
        "rejected": 0,
        "undecided": 0,
        "confirmed_records": 3,
        "draft_records": 0,
    }
    info = validate_annotation_manifest(confirmed)
    assert info.ready_for_training is True
    assert confirmed["provenance"] == "confirmed"
    assert confirmed["records"][0]["reviewers"] == ["老张"]


def test我方区域全填了但整件没填就不算确认(tmp_path: Path) -> None:
    """这是我实跑时真的踩到的 bug，必须钉死。

    我最初只按"区域是否都判过"决定整条是否确认，于是那条**没有区域**的
    记录循环一次都没进、all_decided 恒为 True，**没人填过任何一格就被
    自动标成 confirmed**。那等于把"检测器什么都没找到"写成"这件确实
    没有缺料"的真值，而真实原因可能只是阈值把缺陷滤掉了。
    """

    manifest, sheet, rows = _export(tmp_path)
    for row in rows:
        if row["scope"] == "region":
            row["decision"] = "accept"
    _write_rows(sheet, rows)

    confirmed, summary = apply_review_sheet(manifest, sheet)

    assert summary["confirmed_records"] == 2, "只有填过区域的那两条能确认"
    assert summary["draft_records"] == 1
    by_id = {r["sample_id"]: r for r in confirmed["records"]}
    assert by_id["test_2"]["provenance"] == "draft"
    assert by_id["test_2"]["review_state"] == "in_review"


def test我方整件显式判为无缺料也算确认(tmp_path: Path) -> None:
    """整件判定被填 accept 时，这条负样本应当成为确认真值。"""

    manifest, sheet, rows = _export(tmp_path)
    for row in rows:
        row["decision"] = "accept"
    _write_rows(sheet, rows)

    confirmed, _summary = apply_review_sheet(manifest, sheet)

    by_id = {r["sample_id"]: r for r in confirmed["records"]}
    assert by_id["test_2"]["provenance"] == "confirmed"
    assert by_id["test_2"]["regions"] == []


def test我方全部判为剔除时区域被清空但记录仍确认(tmp_path: Path) -> None:
    manifest, sheet, rows = _export(tmp_path)
    for row in rows:
        row["decision"] = "reject"
    _write_rows(sheet, rows)

    confirmed, summary = apply_review_sheet(manifest, sheet)

    assert summary["rejected"] == 4
    assert summary["accepted"] == 0
    by_id = {r["sample_id"]: r for r in confirmed["records"]}
    assert by_id["train_0"]["provenance"] == "confirmed"
    assert by_id["train_0"]["regions"] == []


def test我方漏掉一行会报错(tmp_path: Path) -> None:
    """删掉一行就能让最难判的区域消失，召回率凭空变好，必须挡住。"""

    manifest, sheet, rows = _export(tmp_path)
    _write_rows(sheet, rows[:-1])

    with pytest.raises(ValueError, match="漏了"):
        apply_review_sheet(manifest, sheet)


def test我方不认识的判定会报错(tmp_path: Path) -> None:
    manifest, sheet, rows = _export(tmp_path)
    rows[0]["decision"] = "maybe"
    _write_rows(sheet, rows)

    with pytest.raises(ValueError, match="decision"):
        apply_review_sheet(manifest, sheet)


def test我方拒绝把旧表套到重新生成过的草稿(tmp_path: Path) -> None:
    """草稿重新生成过就必须重填：旧表会把判定对到别的区域上。"""

    manifest, sheet, rows = _export(tmp_path)
    for row in rows:
        row["decision"] = "accept"
    _write_rows(sheet, rows)

    # 草稿变了：train_0 多出一个区域。
    regenerated = json.loads(json.dumps(manifest))
    regenerated["records"][0]["regions"].append(_region([9]))

    with pytest.raises(ValueError, match="指纹"):
        apply_review_sheet(regenerated, sheet)
