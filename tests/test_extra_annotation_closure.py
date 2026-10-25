"""我给"多余材料（extra）也能被人工确认并打分"这条闭环写回归。

我为什么要补这一整条：`score-annotations` 此前明确只打 missing
（`annotation_scoring.py` 里那几个 `_SCORABLE_DEFECT_TYPES = ("missing",)`
与"schema 里没有候选网格面号字段"的注释）。后果不是"少一个功能"，
而是**任务书之外的半条链**：检测器每一件真实件都会报 extra 区域，
CAD 面映射、可视化、合成评估（`--defect-type extra`）全都支持，
唯独走到"人工确认 -> 打分"这里断了——甲方就算把多余材料的真值标出来，
也算不出这一类结论的准确率，只能得到一句"未打分"。

真值面号落在哪一侧我实测过：`outputs/**/result.json` 里 extra 区域
的 `reference_face_ids` 恒为空、`candidate_face_ids` 非空（缺料正相反）。
所以这条链要接上，只要让标注 schema 承认"extra 的真值是候选网格面号"，
再让打分器按缺陷类型选字段。我这一组用例钉的正是这几处，以及
"不许因为加了 extra 就把 missing 的口径带歪"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cad_inspector.learning.annotation_review import (
    apply_review_sheet,
    export_review_sheet,
)
from cad_inspector.learning.annotation_scoring import score_annotations
from cad_inspector.learning.annotations import (
    SCHEMA_VERSION,
    validate_annotation_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

# 我按路径导入脚本模块（它不是包内模块），写法与 test_bootstrap_annotations.py 一致。
sys.path.insert(0, str(SCRIPTS_DIR))
try:
    import bootstrap_annotations as bootstrap
finally:
    sys.path.remove(str(SCRIPTS_DIR))


def _region(
    defect_type: str,
    face_ids: list[int],
    *,
    candidate_face_ids: list[int] | None = None,
) -> dict:
    """我造一个标注区域；缺料的面号在参考侧，多余的候选侧。"""

    region: dict = {
        "defect_type": defect_type,
        "center_xyz_mm": [1.0, 2.0, 3.0],
        "bbox_min_xyz_mm": [0.0, 0.0, 0.0],
        "bbox_max_xyz_mm": [2.0, 2.0, 2.0],
        "area_mm2": 4.0,
        # 缺料的面号在参考侧，多余材料在候选侧；face_ids 参数按调用方的
        # 语义填进 reference_face_ids，候选侧只用 candidate_face_ids 显式给。
        "reference_face_ids": face_ids,
    }
    if candidate_face_ids is not None:
        region["candidate_face_ids"] = candidate_face_ids
    return region


def _manifest(regions_per_record: list[list[dict]], *, version: str = SCHEMA_VERSION) -> dict:
    records = []
    for index, regions in enumerate(regions_per_record):
        records.append(
            {
                "sample_id": f"sample_{index}",
                "split": "train" if index == 0 else "test",
                "reference_file": f"D:/d/ref{index}.stl",
                "candidate_file": f"D:/d/cand{index}.stp",
                "provenance": "draft",
                "review_state": "unconfirmed",
                "regions": regions,
            }
        )
    return {
        "schema_version": version,
        "provenance": "draft",
        "records": records,
    }


def test我方schema新版本接受带候选面号的extra区域() -> None:
    """extra 的真值必须在候选网格上；带上了就应当通过校验。"""

    payload = _manifest(
        [
            [
                _region("missing", [1, 2]),
                _region("extra", [], candidate_face_ids=[7, 8, 9]),
            ]
        ]
    )

    summary = validate_annotation_manifest(payload)

    assert summary.schema_version == SCHEMA_VERSION
    assert summary.extra_region_count == 1
    assert summary.missing_region_count == 1


def test我方schema新版本拒绝没有候选面号的extra区域() -> None:
    """空真值比不上名单：让它过，打分时会被当成"这块多余材料不存在"。"""

    payload = _manifest([[_region("extra", [])]])

    with pytest.raises(ValueError, match="candidate_face_ids"):
        validate_annotation_manifest(payload)


def test我方schema新版本拒绝extra区域同时带参考面号() -> None:
    """两侧面号同时给，说明标的人自己也没分清真值在哪张网格上。"""

    payload = _manifest(
        [[_region("extra", [1, 2], candidate_face_ids=[7, 8])]]
    )

    with pytest.raises(ValueError, match="extra"):
        validate_annotation_manifest(payload)


def test我方继续接受1_0存量草稿() -> None:
    """存量证据不能因为升版就变非法：reports/draft_annotations.json 是 1.0。

    我为什么非要留这条兼容：那份草稿是**已交付的审计材料**，
    仓库里还有用例（test_bootstrap_annotations.py）在读它的来源字段；
    升版顺手把它判成非法，等于把既有证据烧掉。
    """

    legacy = _manifest(
        [[_region("extra", [])]],
        version="missing-region-annotations-1.0",
    )
    summary = validate_annotation_manifest(legacy)
    assert summary.extra_region_count == 1

    real = ROOT / "reports" / "draft_annotations.json"
    if not real.is_file():
        pytest.skip("仓库里没有草稿清单")
    loaded = json.loads(real.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "missing-region-annotations-1.0"
    assert validate_annotation_manifest(loaded).record_count > 0


def test我方复核表给extra行报出候选面数(tmp_path: Path) -> None:
    """复核人要看的是"这块多余材料覆盖了几个候选面"，不是恒为 0 的参考面数。"""

    payload = _manifest(
        [
            [
                _region("missing", [1, 2]),
                _region("extra", [], candidate_face_ids=[7, 8, 9]),
            ]
        ]
    )
    sheet = tmp_path / "sheet.csv"
    export_review_sheet(payload, sheet)

    rows = _read_rows(sheet)
    by_type = {row["defect_type"]: row for row in rows if row["scope"] == "region"}
    assert by_type["extra"]["face_count"] == "3"
    assert by_type["extra"]["candidate_face_count"] == "3"
    assert by_type["missing"]["face_count"] == "2"
    assert by_type["missing"]["candidate_face_count"] == "0"


def test我方复核表并回后extra真值面号不丢(tmp_path: Path) -> None:
    """人工确认过的那块多余材料，必须带着候选面号活到确认清单里。"""

    payload = _manifest(
        [
            [
                _region("missing", [1, 2]),
                _region("extra", [], candidate_face_ids=[7, 8, 9]),
            ]
        ]
    )
    sheet = tmp_path / "sheet.csv"
    export_review_sheet(payload, sheet)
    rows = _read_rows(sheet)
    for row in rows:
        row["decision"] = "accept"
    _write_rows(sheet, rows)

    confirmed, summary = apply_review_sheet(payload, sheet)

    assert summary["confirmed_records"] == 1
    extra = [r for r in confirmed["records"][0]["regions"] if r["defect_type"] == "extra"]
    assert extra and extra[0]["candidate_face_ids"] == [7, 8, 9]
    assert validate_annotation_manifest(confirmed).extra_region_count == 1


def test我方拒绝有区域记录上多余的整件判定行(tmp_path: Path) -> None:
    """整件判定只对"一个区域都没报"的件有意义；给了别的件就是被静默忽略。

    静默忽略人工填的一格，和漏填一样危险——填表的人会以为自己的判断
    生效了，实际那条判定从头到尾没进过清单。
    """

    payload = _manifest([[_region("missing", [1, 2])]])
    sheet = tmp_path / "sheet.csv"
    export_review_sheet(payload, sheet)
    rows = _read_rows(sheet)
    rows.append({**rows[0], "scope": "record", "region_index": "", "defect_type": ""})
    _write_rows(sheet, rows)

    with pytest.raises(ValueError, match="整件判定"):
        apply_review_sheet(payload, sheet)


def test我方bootstrap把extra候选面号写进草稿(tmp_path: Path) -> None:
    """草稿是人工复核的输入：extra 真值不在草稿里，复核完也没法打分。"""

    pair_dir = tmp_path / "训练集__0"
    pair_dir.mkdir(parents=True)
    (pair_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "review",
                "config": {"mesh_linear_deflection_mm": 0.04},
                "reference_file": "D:/data/ref.stl",
                "candidate_file": "D:/data/cand.stp",
                "regions": [
                    {
                        "defect_type": "missing",
                        "reference_face_ids": [1, 2],
                        # 缺料的候选面号是"采样落点"，不是真值，不能进标注。
                        "candidate_face_ids": [9, 9],
                        "area_mm2": 3.0,
                    },
                    {
                        "defect_type": "extra",
                        "reference_face_ids": [],
                        "candidate_face_ids": [4, 5],
                        "area_mm2": 1.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = bootstrap.build_draft_manifest(tmp_path, lambda pair: "train")

    regions = manifest["records"][0]["regions"]
    missing = [r for r in regions if r["defect_type"] == "missing"][0]
    extra = [r for r in regions if r["defect_type"] == "extra"][0]
    assert missing["reference_face_ids"] == [1, 2]
    assert missing["candidate_face_ids"] == []
    assert extra["candidate_face_ids"] == [4, 5]
    assert extra["reference_face_ids"] == []
    # 草稿本身必须能过校验，否则人工复核前就先崩。
    validate_annotation_manifest(manifest)


def test我方对extra也算面级混淆矩阵() -> None:
    """多料的真值 {1,2,3}/{5} 对预测 {2,3,4} 必须是 tp=2 / fp=1 / fn=2。"""

    payload = _manifest(
        [
            [
                _region("extra", [], candidate_face_ids=[1, 2, 3]),
                _region("missing", [7]),
            ],
            [_region("extra", [], candidate_face_ids=[5])],
        ]
    )
    payload["records"][0]["provenance"] = "confirmed"
    payload["records"][1]["provenance"] = "confirmed"
    predictions = {
        "d:\\d\\cand0.stp": {
            "candidate_file": "D:/d/cand0.stp",
            "regions": [
                {"defect_type": "extra", "candidate_face_ids": [2, 3, 4]},
                {"defect_type": "missing", "reference_face_ids": [7]},
            ],
        },
        "d:\\d\\cand1.stp": {"candidate_file": "D:/d/cand1.stp", "regions": []},
    }

    report = score_annotations(payload, predictions, defect_type="extra")

    assert report["status"] == "SCORED"
    assert report["defect_type"] == "extra"
    assert report["aggregate"]["true_positive"] == 2
    assert report["aggregate"]["false_positive"] == 1
    assert report["aggregate"]["false_negative"] == 2
    assert report["aggregate"]["recall"] == pytest.approx(2 / 4)

    # 同一条清单打 missing：真值只认参考面号，不能被 extra 的候选面号污染。
    missing_report = score_annotations(payload, predictions, defect_type="missing")
    assert missing_report["aggregate"]["true_positive"] == 1
    assert missing_report["aggregate"]["false_positive"] == 0
    assert missing_report["aggregate"]["false_negative"] == 0


def test我方extra打分口径写明依赖同一套三角化() -> None:
    """候选面号是三角化结果的编号：换精度就换了编号，必须写进限制里。"""

    payload = _manifest([[_region("extra", [], candidate_face_ids=[1])]])
    payload["records"][0]["provenance"] = "confirmed"
    predictions = {
        "d:\\d\\cand0.stp": {
            "candidate_file": "D:/d/cand0.stp",
            "regions": [{"defect_type": "extra", "candidate_face_ids": [1]}],
        }
    }

    report = score_annotations(payload, predictions, defect_type="extra")

    joined = "\n".join(report["limitations"])
    assert "三角化" in joined and "extra" in joined


def test我方多类型打分只在整个清单都算得出时才给SCORED() -> None:
    """一次跑两类的入口必须分别报，不能合成一个看不清来源的数字。"""

    from cad_inspector.learning.annotation_scoring import score_annotations_multi

    payload = _manifest(
        [
            [
                _region("missing", [1]),
                _region("extra", [], candidate_face_ids=[2]),
            ]
        ]
    )
    payload["records"][0]["provenance"] = "confirmed"
    predictions = {
        "d:\\d\\cand0.stp": {
            "candidate_file": "D:/d/cand0.stp",
            "regions": [
                {"defect_type": "missing", "reference_face_ids": [1]},
                {"defect_type": "extra", "candidate_face_ids": [2]},
            ],
        }
    }

    combined = score_annotations_multi(payload, predictions)

    assert combined["status"] == "SCORED"
    assert set(combined["reports"]) == {"missing", "extra"}
    assert combined["reports"]["missing"]["aggregate"]["recall"] == pytest.approx(1.0)
    assert combined["reports"]["extra"]["aggregate"]["recall"] == pytest.approx(1.0)

    # 只有 extra 打得出来、missing 一条确认都没有时，整体不能是 SCORED。
    draft_only = _manifest([[_region("missing", [1])]])
    partial = score_annotations_multi(
        draft_only,
        {
            "d:\\d\\cand0.stp": {
                "candidate_file": "D:/d/cand0.stp",
                "regions": [{"defect_type": "missing", "reference_face_ids": [1]}],
            }
        },
    )
    assert partial["status"] == "NO_CONFIRMED_RECORDS"


def test我方命令行能对extra打分并分列missing(tmp_path: Path, capsys) -> None:
    """真实入口：默认口径不许变，extra 与 both 都要能跑。"""

    from cad_inspector.cli import main

    predictions_root = tmp_path / "preds"
    (predictions_root / "p0").mkdir(parents=True)
    (predictions_root / "p0" / "result.json").write_text(
        json.dumps(
            {
                "candidate_file": "D:/d/cand0.stp",
                "regions": [
                    {"defect_type": "missing", "reference_face_ids": [1]},
                    {"defect_type": "extra", "candidate_face_ids": [2]},
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "confirmed.json"
    payload = _manifest(
        [
            [
                _region("missing", [1]),
                _region("extra", [], candidate_face_ids=[2]),
            ]
        ]
    )
    payload["records"][0]["provenance"] = "confirmed"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    # 默认口径：仍然是 missing，输出形状不许变（既有审计材料按这个读）。
    assert (
        main(
            [
                "score-annotations",
                str(manifest_path),
                "--predictions-root",
                str(predictions_root),
            ]
        )
        == 0
    )
    default_report = json.loads(capsys.readouterr().out)
    assert default_report["defect_type"] == "missing"
    assert "reports" not in default_report

    assert (
        main(
            [
                "score-annotations",
                str(manifest_path),
                "--predictions-root",
                str(predictions_root),
                "--defect-type",
                "extra",
            ]
        )
        == 0
    )
    extra_report = json.loads(capsys.readouterr().out)
    assert extra_report["defect_type"] == "extra"
    assert extra_report["aggregate"]["recall"] == pytest.approx(1.0)

    assert (
        main(
            [
                "score-annotations",
                str(manifest_path),
                "--predictions-root",
                str(predictions_root),
                "--defect-type",
                "both",
            ]
        )
        == 0
    )
    both_report = json.loads(capsys.readouterr().out)
    assert both_report["status"] == "SCORED"
    assert set(both_report["reports"]) == {"missing", "extra"}


def test我方命令行拒绝不认识的缺陷类型(tmp_path: Path) -> None:
    """写错类型必须当参数错误挡掉，不能悄悄退回默认口径。"""

    from cad_inspector.cli import main

    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(_manifest([[]])), encoding="utf-8")

    with pytest.raises(SystemExit):
        main(
            [
                "score-annotations",
                str(manifest_path),
                "--predictions-root",
                str(tmp_path),
                "--defect-type",
                "bogus",
            ]
        )


def _read_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)