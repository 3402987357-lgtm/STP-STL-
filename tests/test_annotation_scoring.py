"""我给"用真实标注给预测打分"这条链路写回归。

为什么必须有这一层：仓库此前只能校验标注清单，没有任何地方**用真值
给预测打分**。合成评估证明不了任务书那条 99%（合成真值只证明
"已知孔洞能不能找回来"），草稿更证明不了（拿机器预测当真值去评
同一个预测器，只会得到 100% 的假准确率）。缺了这一环，即便甲方把
确认过的标注交过来，99% 还是算不出来。

这一组用例钉住三件最容易出错的事：混淆矩阵算得对不对、草稿有没有
被挡在门外、找不到预测的件有没有被静默跳过。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cad_inspector.learning.annotation_scoring import (
    collect_predictions,
    score_annotations,
)
from cad_inspector.learning.annotations import SCHEMA_VERSION


def _record(
    sample_id: str,
    split: str,
    candidate: str,
    faces: list[int],
    *,
    provenance: str = "confirmed",
) -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "provenance": provenance,
        "reference_file": f"ref-{sample_id}.stl",
        "candidate_file": candidate,
        "regions": [
            {
                "defect_type": "missing",
                "reference_face_ids": faces,
                "area_mm2": 1.0,
            }
        ],
    }


def _prediction(candidate: str, faces: list[int]) -> dict:
    return {
        "candidate_file": candidate,
        "regions": [
            {"defect_type": "missing", "reference_face_ids": faces},
        ],
    }


def _key(candidate: str) -> str:
    from cad_inspector.learning.annotations import _normalized_path

    return _normalized_path(candidate)


def test我方用确认标注算出面级混淆矩阵() -> None:
    """真值 {1,2,3}、预测 {2,3,4} 必须是 tp=2 / fp=1 / fn=1。"""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [_record("a", "test", "c1.stl", [1, 2, 3])],
    }
    predictions = {_key("c1.stl"): _prediction("c1.stl", [2, 3, 4])}

    report = score_annotations(payload, predictions)

    assert report["status"] == "SCORED"
    aggregate = report["aggregate"]
    assert aggregate["true_positive"] == 2
    assert aggregate["false_positive"] == 1
    assert aggregate["false_negative"] == 1
    assert aggregate["precision"] == pytest.approx(2 / 3)
    assert aggregate["recall"] == pytest.approx(2 / 3)
    assert aggregate["iou"] == pytest.approx(2 / 4)


def test我方把草稿排除在打分之外并标出状态() -> None:
    """草稿是机器预测，混进来就是把 100% 假准确率端上桌。

    我要求：草稿记录不贡献任何 tp/fp/fn，sample_id 单独列出，
    且状态必须变成 DRAFT_EXCLUDED——不能是一个看起来干净的数字。
    """

    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            _record("real", "test", "c1.stl", [1, 2]),
            _record("machine", "train", "c2.stl", [9, 9, 9], provenance="draft"),
        ],
    }
    predictions = {
        _key("c1.stl"): _prediction("c1.stl", [1, 2]),
        # 草稿那件即便预测得"完美"，也绝不能进分母。
        _key("c2.stl"): _prediction("c2.stl", [9, 9, 9]),
    }

    report = score_annotations(payload, predictions)

    assert report["status"] == "DRAFT_EXCLUDED"
    assert report["draft_excluded_sample_ids"] == ["machine"]
    assert report["scored_record_count"] == 1
    assert report["aggregate"]["true_positive"] == 2
    assert report["aggregate"]["false_positive"] == 0
    assert report["aggregate"]["false_negative"] == 0
    # 只有确认那件被记进明细。
    assert [r["sample_id"] for r in report["records"]] == ["real"]


def test我方找不到预测的件按整件漏检计入而不跳过() -> None:
    """静默跳过会让"漏了整整一件"从召回率里消失，那正是最该看见的错误。"""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [_record("missing-pred", "test", "c1.stl", [1, 2, 3])],
    }

    report = score_annotations(payload, {})

    assert report["no_prediction_sample_ids"] == ["missing-pred"]
    assert report["aggregate"]["false_negative"] == 3
    assert report["aggregate"]["recall"] == 0.0
    assert report["records"][0]["prediction_found"] is False


def test我方只有草稿时不给任何准确率() -> None:
    """一条确认记录都没有时，状态必须是 NO_CONFIRMED_RECORDS。"""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            _record("m1", "train", "c1.stl", [1], provenance="draft"),
        ],
    }
    predictions = {_key("c1.stl"): _prediction("c1.stl", [1])}

    report = score_annotations(payload, predictions)

    assert report["status"] == "NO_CONFIRMED_RECORDS"
    assert report["scored_record_count"] == 0
    assert report["draft_excluded_sample_ids"] == ["m1"]


def test我方只对已知缺陷类型打分() -> None:
    """未知类型必须报错；extra 从 2026-10-25 起已经能打分（见 test_extra_annotation_closure.py）。

    我为什么把这条从"拒绝对 extra 打分"改掉：那一条钉的是当时的**能力缺口**
    （标注 schema 没有候选网格面号字段）。现在 schema 1.1 有了
    candidate_face_ids、打分器按类型选字段，缺口补上了；再断言
    "extra 必须报错"就等于把缺陷当规格钉死。未知类型仍然要挡。
    """

    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [_record("a", "test", "c1.stl", [1])],
    }
    with pytest.raises(ValueError):
        score_annotations(payload, {}, defect_type="hole")


def test我方扫描预测目录并按候选路径索引(tmp_path: Path) -> None:
    """索引键必须是 result.json 里的候选件路径，而不是目录名。"""

    nested = tmp_path / "训练集__0"
    nested.mkdir()
    (nested / "result.json").write_text(
        json.dumps(_prediction("D:/data/c1.stl", [1, 2])), encoding="utf-8"
    )
    # 坏文件必须被跳过而不是让整次扫描崩掉。
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "result.json").write_text("{not json", encoding="utf-8")

    index = collect_predictions(tmp_path)

    assert list(index) == [_key("D:/data/c1.stl")]


def test我方命令行能跑通打分并拒绝含草稿的清单(tmp_path: Path) -> None:
    """我验真实入口：SCORED 返回 0，含草稿返回 1（不给假绿灯）。"""

    from cad_inspector.cli import main

    predictions_root = tmp_path / "preds"
    (predictions_root / "p1").mkdir(parents=True)
    (predictions_root / "p1" / "result.json").write_text(
        json.dumps(_prediction("c1.stl", [1, 2])), encoding="utf-8"
    )
    (predictions_root / "p2").mkdir(parents=True)
    (predictions_root / "p2" / "result.json").write_text(
        json.dumps(_prediction("c2.stl", [1, 2])), encoding="utf-8"
    )

    clean = tmp_path / "clean.json"
    clean.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "records": [
                    # train/test 必须用不同的候选件，否则会撞上
                    # 标注校验里的跨 split 路径泄漏门禁。
                    _record("a", "train", "c1.stl", [1, 2]),
                    _record("b", "test", "c2.stl", [1, 2]),
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "score-annotations",
                str(clean),
                "--predictions-root",
                str(predictions_root),
            ]
        )
        == 0
    )

    dirty = tmp_path / "dirty.json"
    dirty.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "records": [
                    _record("a", "test", "c1.stl", [1, 2], provenance="draft"),
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "score-annotations",
                str(dirty),
                "--predictions-root",
                str(predictions_root),
            ]
        )
        == 1
    )


def test我方打分器与合成评估器在同一份数据上给出同一个混淆矩阵() -> None:
    """我拿两条独立实现的链路对撞，证明打分器算的数是对的。

    为什么要这条：`evaluate_synthetic_case` 是合成口径的老实现，
    `score_annotations` 是真实标注口径的新实现。两者算的是同一件事
    （面级 TP/FP/FN），但代码完全独立。如果我只测新实现的"手算小例子"，
    它和真实检测输出对不上时不会有人发现。

    我在这里用**同一份真实语料的同一次检测**喂给两边：一边走
    evaluate_synthetic_case，一边把这次检测包成 result.json 形状的预测、
    再走 score_annotations。两边的 tp/fp/fn 与 recall 必须逐位相等。

    真值用"被删掉的全部面"——这是人工标注者会圈的范围（整块缺料），
    所以召回会落在 0.7 附近而不是 1.0；这条用例钉的是**两条链路一致**，
    不是数值高低。
    """

    from cad_inspector.detection.missing_regions import detect_missing_regions
    from cad_inspector.geometry.stl import load_stl
    from cad_inspector.learning.evaluation import (
        _identity_registration,
        evaluate_synthetic_case,
    )
    from cad_inspector.learning.synthetic_defects import remove_face_region
    from cad_inspector.models import InspectionConfig

    root = Path(__file__).resolve().parents[1]
    corpus = root / "data" / "provided" / "三维模型破损区域智能修补" / "训练集" / "8.stl"
    if not corpus.exists():
        pytest.skip("我找不到真实语料")

    reference = load_stl(corpus)
    defect = remove_face_region(reference, seed_face_id=0, face_count=1500)
    config = InspectionConfig(registration_mode="none", unit_policy="assume_mm")

    expected = evaluate_synthetic_case(
        reference,
        defect.candidate_mesh,
        defect.removed_face_ids,
        config=config,
        defect_type="missing",
    )

    regions, _metrics, _warnings, _distances = detect_missing_regions(
        reference=reference,
        candidate=defect.candidate_mesh,
        config=config,
        registration=_identity_registration(),
    )
    prediction = {
        "candidate_file": "candidate.stl",
        "regions": [
            {
                "defect_type": region.defect_type,
                "reference_face_ids": list(region.reference_face_ids),
            }
            for region in regions
        ],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            {
                "sample_id": "cross-check",
                "split": "test",
                "provenance": "confirmed",
                "reference_file": "reference.stl",
                "candidate_file": "candidate.stl",
                "regions": [
                    {
                        "defect_type": "missing",
                        "reference_face_ids": list(defect.removed_face_ids),
                    }
                ],
            }
        ],
    }

    report = score_annotations(
        payload, {_key("candidate.stl"): prediction}
    )

    assert report["status"] == "SCORED"
    aggregate = report["aggregate"]
    assert aggregate["true_positive"] == expected.true_positive
    assert aggregate["false_positive"] == expected.false_positive
    assert aggregate["false_negative"] == expected.false_negative
    assert aggregate["precision"] == pytest.approx(expected.precision)
    assert aggregate["recall"] == pytest.approx(expected.recall)
    assert aggregate["iou"] == pytest.approx(expected.iou)
