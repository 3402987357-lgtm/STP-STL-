"""我为"区域 -> CAD 面编号"映射写回归测试。

这里最关键的一条是**下标对齐门禁**：映射表长度和参考网格三角数不一致时
必须拒绝映射。我实测过这个错——用默认弦高 0.5 重新三角化得到 26302 个
三角，而参考网格是 26216 个，下标直接错位，映射出的面号全是错的
而且错得悄无声息。这条门禁就是那次实测留下的。

第二条关键是**缺料与多料各走各的侧**。上游对 extra 区域把候选件的面编号
写进 candidate_face_ids 并留空 reference_face_ids（多料只存在于候选件上），
所以多料必须用候选件的映射表和三角面积；沿用参考侧会张冠李戴。
"""

from __future__ import annotations

import numpy as np
import pytest

from cad_inspector.detection.cad_face_mapping import (  # noqa: E402
    CadFaceMapping,
    _degrees_to_radians,
    map_regions_to_cad_faces,
    summarize_region_cad_faces,
)
from test_core import cube_mesh  # noqa: E402


class _FakeRegion:
    """我造一个只有映射所需字段的区域对象。"""

    def __init__(
        self,
        region_id: int,
        face_ids: list[int],
        defect_type: str = "missing",
        candidate_face_ids: list[int] | None = None,
    ):
        self.region_id = region_id
        self.reference_face_ids = face_ids
        self.candidate_face_ids = (
            [] if candidate_face_ids is None else candidate_face_ids
        )
        self.defect_type = defect_type


def test我方对非STEP参考件拒绝映射() -> None:
    """STL 没有 B-Rep 面，硬编号就是编造语义。"""

    mesh = cube_mesh()

    result = map_regions_to_cad_faces(
        regions=[_FakeRegion(1, [0, 1])],
        reference_mesh=mesh,
        reference_path="part.stl",
        reference_metadata={"detected_format": "stl"},
    )

    assert result.enabled is False
    assert "STL" in result.reason
    assert result.region_cad_face_ids == {}


def test我方优先复用加载器已有的面ID表() -> None:
    """加载器的表与参考网格同一份，下标天然对齐，不该再三角化一次。"""

    mesh = cube_mesh()
    # 12 个三角，前 6 个属于 1 号面，后 6 个属于 2 号面。
    existing = [1] * 6 + [2] * 6

    result = map_regions_to_cad_faces(
        regions=[_FakeRegion(1, [0, 1, 7, 8])],
        reference_mesh=mesh,
        reference_path="part.stp",
        reference_metadata={
            "detected_format": "step",
            "face_ids_by_triangle": existing,
        },
    )

    assert result.enabled is True
    assert result.region_cad_face_ids[1] == [1, 2]
    assert result.cad_face_count == 2
    assert (
        result.config["reference"]["source"]
        == "step_loader_face_ids_by_triangle"
    )


def test我方在映射表长度不一致时拒绝() -> None:
    """这是实测踩过的坑：长度不一致会静默给出错的面号。"""

    mesh = cube_mesh()

    result = map_regions_to_cad_faces(
        regions=[_FakeRegion(1, [0])],
        reference_mesh=mesh,
        reference_path="part.stl",
        reference_metadata={"detected_format": "stl"},
    )

    # 走 STL 分支直接拒绝；这里钉住"长度不符的表不会被当成可用"。
    wrong_length = [1] * (len(mesh.faces) + 5)
    regions = [_FakeRegion(1, [0])]
    from cad_inspector.detection.cad_face_mapping import _mapping_from_metadata

    assert _mapping_from_metadata(mesh, {"face_ids_by_triangle": wrong_length}) is None
    assert result.enabled is False


def test我方多料区域用候选件的映射表() -> None:
    """多料只存在于候选件上，必须用候选侧的表，否则张冠李戴。"""

    reference = cube_mesh()
    candidate = cube_mesh()
    # 两侧故意用**不同**的面号体系：都是 12 个三角，
    # 参考侧前 6 个属于 1 号面，候选侧前 6 个属于 11 号面。
    # 若多料错用了参考侧的表，就会得到 1 而不是 11。
    reference_table = [1] * 6 + [2] * 6
    candidate_table = [11] * 6 + [12] * 6

    result = map_regions_to_cad_faces(
        regions=[
            _FakeRegion(1, [0, 1], defect_type="missing"),
            _FakeRegion(2, [], defect_type="extra", candidate_face_ids=[0, 1]),
        ],
        reference_mesh=reference,
        reference_path="ref.stp",
        reference_metadata={
            "detected_format": "step",
            "face_ids_by_triangle": reference_table,
        },
        candidate_mesh=candidate,
        candidate_path="cand.stp",
        candidate_metadata={
            "detected_format": "step",
            "face_ids_by_triangle": candidate_table,
        },
    )

    assert result.region_cad_face_ids[1] == [1]
    assert result.region_cad_face_ids[2] == [11]
    # 面号属于哪一侧必须记下来，否则甲方会拿候选件的面号去参考件上找。
    assert result.region_cad_face_side[1] == "reference"
    assert result.region_cad_face_side[2] == "candidate"
    assert result.cad_face_count == 2


def test我方多料面积按候选件的三角面积累加() -> None:
    """多料的面积要按候选侧算，用参考侧的面积会对不上区域面积。"""

    reference = cube_mesh()
    candidate = cube_mesh()
    # 候选件整体放大 2 倍：候选侧三角面积是参考侧的 4 倍，
    # 拿错一侧的面积会立刻暴露。
    candidate = candidate.transformed(scale=2.0)
    table = [1] * len(reference.faces)

    result = map_regions_to_cad_faces(
        regions=[
            _FakeRegion(
                1,
                [],
                defect_type="extra",
                candidate_face_ids=list(range(len(candidate.faces))),
            )
        ],
        reference_mesh=reference,
        reference_path="ref.stp",
        reference_metadata={
            "detected_format": "step",
            "face_ids_by_triangle": table,
        },
        candidate_mesh=candidate,
        candidate_path="cand.stp",
        candidate_metadata={
            "detected_format": "step",
            "face_ids_by_triangle": table,
        },
    )

    summed = sum(result.region_cad_face_areas_mm2[1].values())
    # 四条边放大 2 倍 -> 面积 4 倍。
    assert summed == pytest.approx(
        float(reference.triangle_areas.sum()) * 4.0, rel=1e-9
    )


def test我方候选件是STL时只出参考侧结果() -> None:
    """候选件是 STL 就没有 B-Rep 面，多料不能映射，但缺料照常出。"""

    reference = cube_mesh()
    candidate = cube_mesh()
    existing = [1] * 6 + [2] * 6

    result = map_regions_to_cad_faces(
        regions=[
            _FakeRegion(1, [0, 1], defect_type="missing"),
            _FakeRegion(2, [], defect_type="extra", candidate_face_ids=[0, 1]),
        ],
        reference_mesh=reference,
        reference_path="ref.stp",
        reference_metadata={
            "detected_format": "step",
            "face_ids_by_triangle": existing,
        },
        candidate_mesh=candidate,
        candidate_path="cand.stl",
        candidate_metadata={"detected_format": "stl"},
    )

    assert result.enabled is True
    assert result.region_cad_face_ids.get(1) == [1]
    # 多料的面号我不猜：留空比给一个编造的编号诚实。
    assert 2 not in result.region_cad_face_ids


def test我方两侧都不是STEP时整体不启用() -> None:
    mesh = cube_mesh()

    result = map_regions_to_cad_faces(
        regions=[_FakeRegion(1, [0, 1])],
        reference_mesh=mesh,
        reference_path="ref.stl",
        reference_metadata={"detected_format": "stl"},
        candidate_mesh=mesh,
        candidate_path="cand.stl",
        candidate_metadata={"detected_format": "stl"},
    )

    assert result.enabled is False


def test我方跳过越界下标并留痕() -> None:
    """越界下标直接拿去索引会拿到错的面号甚至报错，必须拦住。"""

    mesh = cube_mesh()
    existing = [1] * len(mesh.faces)

    result = map_regions_to_cad_faces(
        regions=[_FakeRegion(1, [0, 1, 999])],
        reference_mesh=mesh,
        reference_path="part.stp",
        reference_metadata={
            "detected_format": "step",
            "face_ids_by_triangle": existing,
        },
    )

    assert result.enabled is True
    assert any("越界" in item for item in result.warnings)
    assert result.region_cad_face_ids[1] == [1]


def test我方面积按面号累加且等于区域面积() -> None:
    """逐面累加之和必须与区域面积相等，不等说明漏了或重复计了。"""

    mesh = cube_mesh()
    existing = [1] * 6 + [2] * 6
    face_ids = list(range(len(mesh.faces)))

    result = map_regions_to_cad_faces(
        regions=[_FakeRegion(1, face_ids)],
        reference_mesh=mesh,
        reference_path="part.stp",
        reference_metadata={
            "detected_format": "step",
            "face_ids_by_triangle": existing,
        },
    )

    summed = sum(result.region_cad_face_areas_mm2[1].values())
    assert summed == pytest.approx(float(mesh.triangle_areas.sum()), rel=1e-9)


def test我方摘要给出主导面与占比() -> None:
    """甲方要看的是"主要落在第几号面"，不是一堆面号列表。"""

    mapping = CadFaceMapping(
        enabled=True,
        reason="ok",
        region_cad_face_ids={3: [26, 27]},
        region_cad_face_areas_mm2={3: {26: 0.001, 27: 11.0}},
    )

    lines = summarize_region_cad_faces(mapping)

    assert len(lines) == 1
    assert "第 27 号" in lines[0]
    assert "100.0%" in lines[0]


def test我方未启用时摘要给出原因() -> None:
    mapping = CadFaceMapping(enabled=False, reason="我因为 X 没做映射。")

    assert summarize_region_cad_faces(mapping) == ["我因为 X 没做映射。"]


def test我方摘要写明面号属于哪一侧() -> None:
    """不写清"候选件/参考件"，甲方会拿着面号去找错模型上的面。"""

    mapping = CadFaceMapping(
        enabled=True,
        reason="ok",
        region_cad_face_ids={3: [27], 4: [12]},
        region_cad_face_areas_mm2={3: {27: 11.0}, 4: {12: 5.0}},
        region_cad_face_side={3: "reference", 4: "candidate"},
    )

    lines = summarize_region_cad_faces(mapping)

    assert "参考件的第 27 号" in lines[0]
    assert "候选件的第 12 号" in lines[1]


def test我方结果可JSON序列化() -> None:
    import json

    mapping = CadFaceMapping(
        enabled=True,
        reason="ok",
        region_cad_face_ids={1: [1, 2]},
        region_cad_face_areas_mm2={1: {1: 1.5, 2: 2.5}},
    )

    payload = json.loads(json.dumps(mapping.to_dict(), ensure_ascii=False))

    # JSON 键只能是字符串：不统一转字符串的话读回来类型会变。
    assert payload["region_cad_face_ids"]["1"] == [1, 2]
    assert payload["region_cad_face_areas_mm2"]["1"]["2"] == 2.5


def test我方角度换算正确() -> None:
    """OCCT 要弧度、配置用度，混用会让三角化精细度差 57 倍。"""

    assert _degrees_to_radians(180.0) == pytest.approx(np.pi)
    assert _degrees_to_radians(10.0) == pytest.approx(np.deg2rad(10.0), rel=1e-12)
