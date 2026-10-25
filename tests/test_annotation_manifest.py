from __future__ import annotations

import pytest

from cad_inspector.learning.annotations import (
    SCHEMA_VERSION,
    validate_annotation_manifest,
)


def _record(sample_id: str, split: str, reference: str, candidate: str) -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "reference_file": reference,
        "candidate_file": candidate,
        "regions": [
            {
                "defect_type": "missing",
                "reference_face_ids": [1, 2],
                "center_xyz_mm": [1.0, 2.0, 3.0],
                "bbox_min_xyz_mm": [0.0, 0.0, 0.0],
                "bbox_max_xyz_mm": [2.0, 3.0, 4.0],
                "area_mm2": 5.0,
                "boundary_edge_ids": [10, 11],
            }
        ],
    }


def test我方标注清单统计split并阻止训练泄漏() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            _record("train-1", "train", "r1.stl", "c1.stl"),
            _record("validation-1", "validation", "r2.stl", "c2.stl"),
            _record("test-1", "test", "r3.stl", "c3.stl"),
        ],
    }

    summary = validate_annotation_manifest(payload)

    assert summary.ready_for_training is True
    assert summary.record_count == 3
    assert summary.split_counts == {"test": 1, "train": 1, "validation": 1}
    assert summary.region_count == 3
    assert summary.missing_splits == []


def test我方拒绝跨split复用同一文件对() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            _record("train-1", "train", "same.stl", "same-c.stl"),
            _record("test-1", "test", "same.stl", "same-c.stl"),
        ],
    }

    with pytest.raises(ValueError, match="数据泄漏"):
        validate_annotation_manifest(payload)


def test我方拒绝跨split复用单个参考文件() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            _record("train-1", "train", "same.stl", "train-c.stl"),
            _record("test-1", "test", "same.stl", "test-c.stl"),
        ],
    }

    with pytest.raises(ValueError, match="数据泄漏"):
        validate_annotation_manifest(payload)


def test我方拒绝同一条记录里参考文件和候选文件是同一个文件() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            _record("train-1", "train", "same.stl", "same.stl"),
            _record("test-1", "test", "r.stl", "c.stl"),
        ],
    }

    with pytest.raises(ValueError, match="同一个文件"):
        validate_annotation_manifest(payload)

    # 大小写、正反斜杠和相对路径写法不同，但指向同一文件时同样拒绝。
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            _record("train-1", "train", r"C:\Data\A.STP", r"c:/data/a.stp"),
            _record("test-1", "test", "r.stl", "c.stl"),
        ],
    }

    with pytest.raises(ValueError, match="同一个文件"):
        validate_annotation_manifest(normalized)


def test我方允许只有训练集和测试集的历史标注清单() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            _record("train-1", "train", "r1.stl", "c1.stl"),
            _record("test-1", "test", "r2.stl", "c2.stl"),
        ],
    }

    summary = validate_annotation_manifest(payload)

    assert summary.ready_for_training is True
    assert summary.missing_splits == ["validation"]


def test我方没有训练集或测试集时不能进入训练() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            _record("train-1", "train", "r1.stl", "c1.stl"),
        ],
    }

    summary = validate_annotation_manifest(payload)

    assert summary.ready_for_training is False
    assert summary.missing_splits == ["test", "validation"]


def test我方拒绝浮点Face编号和非有限坐标() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            {
                **_record("train-1", "train", "r.stl", "c.stl"),
                "regions": [
                    {
                        "defect_type": "missing",
                        "reference_face_ids": [1.0],
                        "center_xyz_mm": [0.0, float("nan"), 0.0],
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError):
        validate_annotation_manifest(payload)


def test我方拒绝反向包围盒和不完整包围盒() -> None:
    reversed_bbox = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            {
                **_record("train-1", "train", "r.stl", "c.stl"),
                "regions": [
                    {
                        "defect_type": "missing",
                        "bbox_min_xyz_mm": [2.0, 0.0, 0.0],
                        "bbox_max_xyz_mm": [1.0, 1.0, 1.0],
                    }
                ],
            }
        ],
    }
    incomplete_bbox = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            {
                **_record("train-2", "train", "r2.stl", "c2.stl"),
                "regions": [
                    {
                        "defect_type": "missing",
                        "bbox_min_xyz_mm": [0.0, 0.0, 0.0],
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="不能大于"):
        validate_annotation_manifest(reversed_bbox)
    with pytest.raises(ValueError, match="同时提供"):
        validate_annotation_manifest(incomplete_bbox)


def test我方把机器草稿挡在训练之外() -> None:
    """我验证草稿清单不能拿到 READY_FOR_TRAINING。

    这条守的是一个很具体的坑：我这一轮新加了"把验收结果转成标注草稿"
    的脚本，草稿在格式和 split 划分上**完全合法**，能顺利过掉原来
    那三条闸门。若不按 provenance 拦，机器预测就能当作真值去训练、
    再去评估生成它的同一个预测器——自己给自己打分，得到 100% 假准确率。
    """

    draft_payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            {
                **_record("train-1", "train", "r1.stl", "c1.stl"),
                "provenance": "draft",
            },
            {
                **_record("test-1", "test", "r3.stl", "c3.stl"),
                "provenance": "draft",
            },
        ],
    }

    draft = validate_annotation_manifest(draft_payload)

    assert draft.draft_record_count == 2
    assert draft.confirmed_record_count == 0
    assert draft.ready_for_training is False

    # 同一批记录只要把 provenance 标成 confirmed，就应当放行：
    # 这证明拦住草稿的不是格式，而是"来源未确认"这一件事。
    confirmed_payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            {
                **_record("train-1", "train", "r1.stl", "c1.stl"),
                "provenance": "confirmed",
            },
            {
                **_record("test-1", "test", "r3.stl", "c3.stl"),
                "provenance": "confirmed",
            },
        ],
    }

    confirmed = validate_annotation_manifest(confirmed_payload)

    assert confirmed.draft_record_count == 0
    assert confirmed.confirmed_record_count == 2
    assert confirmed.ready_for_training is True


def test我方把没写provenance的旧清单当成已确认() -> None:
    """我验证缺 provenance 字段时不误伤历史清单。

    这个字段是本轮新加的，历史清单都没有。默认按 confirmed 处理，
    否则所有已有标注会一夜之间全部变成草稿、训练全线被拦。
    """

    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            _record("train-1", "train", "r1.stl", "c1.stl"),
            _record("test-1", "test", "r3.stl", "c3.stl"),
        ],
    }

    summary = validate_annotation_manifest(payload)

    assert summary.confirmed_record_count == 2
    assert summary.draft_record_count == 0
    assert summary.ready_for_training is True


def test我方拒绝不认识的provenance取值() -> None:
    """我验证拼错的 provenance 直接报错而不是被当成已确认。"""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [
            {
                **_record("train-1", "train", "r1.stl", "c1.stl"),
                "provenance": "Draft",
            },
            _record("test-1", "test", "r3.stl", "c3.stl"),
        ],
    }

    with pytest.raises(ValueError, match="provenance"):
        validate_annotation_manifest(payload)
