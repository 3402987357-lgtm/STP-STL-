"""我方验证正式报告除运行耗时外可跨运行复现。"""

from __future__ import annotations

import json

from cad_inspector.detection.step_features import classify_step_face_features
from cad_inspector.io.step_faces import StepFaceFeature, StepFaceFeatureReport
from cad_inspector.output.step_feature_report import (
    write_step_feature_classification_json,
)
from cad_inspector.output.step_face_report import write_step_face_features_json
from cad_inspector.reproducibility import content_sha256, stable_payload


def _face(
    face_id: int,
    center: list[float],
    normal: list[float],
    edges: list[int],
) -> StepFaceFeature:
    x, y, z = center
    return StepFaceFeature(
        face_id=face_id,
        surface_type="GeomAbs_Plane",
        area_mm2=100.0,
        center_xyz_mm=center,
        normal_xyz=normal,
        bbox_min_xyz_mm=[x - 0.1, y - 0.1, z - 0.1],
        bbox_max_xyz_mm=[x + 0.1, y + 0.1, z + 0.1],
        edge_count=len(edges),
        edge_ids=edges,
        orientation="FORWARD",
    )


def _box_report() -> StepFaceFeatureReport:
    return StepFaceFeatureReport(
        status="COMPLETE",
        file_path="box.step",
        unit="mm",
        backend="OCP",
        shape_is_valid=True,
        face_count=6,
        surface_type_counts={"GeomAbs_Plane": 6},
        faces=[
            _face(1, [5.0, 5.0, 0.0], [0.0, 0.0, -1.0], [1, 2, 3, 4]),
            _face(2, [5.0, 5.0, 10.0], [0.0, 0.0, 1.0], [5, 6, 7, 8]),
            _face(3, [0.0, 5.0, 5.0], [-1.0, 0.0, 0.0], [1, 9, 10, 11]),
            _face(4, [10.0, 5.0, 5.0], [1.0, 0.0, 0.0], [2, 12, 13, 14]),
            _face(5, [5.0, 0.0, 5.0], [0.0, -1.0, 0.0], [3, 12, 15, 16]),
            _face(6, [5.0, 10.0, 5.0], [0.0, 1.0, 0.0], [4, 13, 15, 16]),
        ],
    )


def test我方稳定负载递归剔除耗时字段() -> None:
    payload = {
        "a": 1,
        "elapsed_seconds": 3.5,
        "nested": {"elapsed_seconds": 9.9, "value": [{"elapsed_seconds": 1.0}]},
    }
    stable = stable_payload(payload)
    assert stable == {"a": 1, "nested": {"value": [{}]}}
    assert content_sha256(payload) == content_sha256(stable)


def test我方同一输入只改耗时不改内容摘要() -> None:
    report_a = _box_report()
    report_a.elapsed_seconds = 0.25
    report_b = _box_report()
    report_b.elapsed_seconds = 987.5

    payload_a = report_a.to_dict()
    payload_b = report_b.to_dict()

    assert payload_a["elapsed_seconds"] == 0.25
    assert payload_b["elapsed_seconds"] == 987.5
    assert payload_a["content_sha256"] == payload_b["content_sha256"]
    assert payload_a["content_sha256"] == content_sha256(payload_a)


def test我方分类报告重复运行内容摘要一致() -> None:
    first = classify_step_face_features(_box_report()).to_dict()
    second = classify_step_face_features(_box_report()).to_dict()
    assert first["content_sha256"] == second["content_sha256"]


def test我方内容变化会改变内容摘要() -> None:
    base = _box_report().to_dict()
    changed = _box_report()
    changed.faces[0].area_mm2 = 123.0
    assert base["content_sha256"] != changed.to_dict()["content_sha256"]


def test我方写出的正式报告带可校验内容摘要(tmp_path) -> None:
    result = classify_step_face_features(_box_report())
    path = tmp_path / "regions.json"
    write_step_feature_classification_json(result, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["content_sha256"] == content_sha256(payload)


def test我方整数键字典写盘后内容摘要仍稳定(tmp_path) -> None:
    report = _box_report()
    for face in report.faces:
        face.edge_endpoint_vertex_ids = {
            edge_id: [edge_id, edge_id + 1] for edge_id in face.edge_ids
        }
    report.vertex_coordinates_mm = {index: [1.0, 2.0, 3.0] for index in range(1, 20)}
    path = tmp_path / "faces.json"
    write_step_face_features_json(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["content_sha256"] == content_sha256(payload)
