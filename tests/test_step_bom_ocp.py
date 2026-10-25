from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from cad_inspector.io.step_bom import (
    _collect_leaf_occurrences,
    _category_candidate,
    _category_decision,
    _compose_affine_3x4,
    _expand_multi_solid_groups,
    _group_occurrences,
    _measure_component,
    inspect_step_bom,
)
from cad_inspector.io.bom_master_data import load_bom_master_data
from cad_inspector.errors import InputValidationError
from dataclasses import MISSING, fields

from cad_inspector.domain.bom import (
    PART_CATEGORY_GROUPS,
    canonical_part_category,
    canonical_part_category_from_candidates,
)
from cad_inspector.output.bom_report import write_bom_report


class _FakeSequence:
    def __init__(self) -> None:
        self.items: list[object] = []

    def Length(self) -> int:
        return len(self.items)

    def Value(self, index: int) -> object:
        return self.items[index - 1]


class _FakeLabel:
    def __init__(
        self,
        tag: int,
        *,
        children: list["_FakeLabel"] | None = None,
        reference_tag: int | None = None,
    ) -> None:
        self.tag = tag
        self.children = children or []
        self.reference_tag = reference_tag

    def Tag(self) -> int:
        return self.tag


class _FakeReferenceLabel:
    def __init__(self) -> None:
        self.tag = 0

    def Tag(self) -> int:
        return self.tag


class _FakeShapeTool:
    def GetComponents_s(
        self, label: _FakeLabel, sequence: _FakeSequence, get_subchildren: bool
    ) -> None:
        sequence.items = list(label.children)

    def IsReference_s(self, label: _FakeLabel) -> bool:
        return label.reference_tag is not None

    def GetReferredShape_s(
        self, label: _FakeLabel, reference_label: _FakeReferenceLabel
    ) -> bool:
        if label.reference_tag is None:
            return False
        reference_label.tag = label.reference_tag
        return True


class _FakeMeasureShapeTool:
    def __init__(self, shape: object) -> None:
        self.shape = shape

    def GetShape_s(self, label: object) -> object:
        return self.shape


class _FakeMaterialTool:
    def GetMaterial_s(self, *args: object) -> bool:
        return False


def _write_box_step(path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    shape = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Solid()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_ReturnStatus.IFSelect_RetDone
    assert writer.Write(str(path)) == IFSelect_ReturnStatus.IFSelect_RetDone


def _make_multi_solid_compound(box_count: int = 3) -> object:
    """我构造一个含多个独立 Solid 的复合 Shape，复现单节点多实体模具。"""

    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Pnt

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for index in range(box_count):
        builder.Add(
            compound,
            BRepPrimAPI_MakeBox(
                gp_Pnt(index * 50.0, 0.0, 0.0),
                10.0,
                20.0,
                30.0,
            ).Solid(),
        )
    return compound


def _fake_group(reference_shape_id: str = "label:1") -> dict[str, object]:
    return {
        "reference_shape_id": reference_shape_id,
        "measure_label": object(),
        "occurrence_ids": [reference_shape_id],
        "occurrence_paths": ["root:1"],
        "occurrence_locations": [None],
    }


def test我方把多_solid_复合体拆成逐个零件的计量单元() -> None:
    pytest.importorskip("OCP")
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    compound = _make_multi_solid_compound(box_count=3)
    shape_tool = _FakeMeasureShapeTool(compound)

    units, multi_solid_group_count = _expand_multi_solid_groups(
        [_fake_group()],
        shape_tool,
        TopExp_Explorer,
        TopAbs_SOLID,
        TopoDS,
    )

    assert multi_solid_group_count == 1
    assert len(units) == 3
    assert [unit["solid_index"] for unit in units] == [1, 2, 3]
    assert [unit["solid_total"] for unit in units] == [3, 3, 3]
    assert [unit["split_reference_shape_id"] for unit in units] == [
        "label:1#solid_001",
        "label:1#solid_002",
        "label:1#solid_003",
    ]
    assert all(unit["shape_override"] is not None for unit in units)


def test我方不拆分只有一个_solid_的节点() -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    single_solid = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Solid()
    shape_tool = _FakeMeasureShapeTool(single_solid)

    units, multi_solid_group_count = _expand_multi_solid_groups(
        [_fake_group()],
        shape_tool,
        TopExp_Explorer,
        TopAbs_SOLID,
        TopoDS,
    )

    assert multi_solid_group_count == 0
    assert len(units) == 1
    assert units[0]["shape_override"] is None
    assert units[0]["solid_index"] is None
    assert units[0]["solid_total"] == 1


def test我方生成_step几何_bom基线和_excel兼容_csv(tmp_path: Path) -> None:
    step_path = tmp_path / "box.step"
    _write_box_step(step_path)

    report = inspect_step_bom(step_path)
    artifacts = write_bom_report(report, tmp_path / "out")

    assert report.status == "BASELINE"
    assert report.backend == "OCP_XCAF_STEPCAFControl"
    assert report.to_dict()["schema_version"] == "mold-bom-1.9"
    assert report.input_metadata["detected_format"] == "STEP"
    assert len(report.input_metadata["sha256"]) == 64
    assert report.component_count == 1
    assert report.diagnostics["assembly_occurrence_count"] == 1
    assert report.diagnostics["unique_reference_shape_count"] == 1
    assert report.components[0].dimensions_xyz_mm == pytest.approx([10.0, 20.0, 30.0])
    assert report.components[0].occurrence_count == 1
    assert report.components[0].quantity == report.components[0].occurrence_count
    assert report.components[0].category_source == "GEOMETRY_RULE"
    assert report.components[0].category_review_required is True
    assert report.components[0].category_candidates
    assert "SEMANTICS_UNRESOLVED" in report.components[0].category_review_reason_codes
    assert report.diagnostics["category_review_required_count"] == 1
    assert report.diagnostics["category_candidate_counts"]
    assert report.components[0].material_source == "NOT_PRESENT"
    assert report.components[0].occurrence_paths == ["root:1"]
    assert len(report.components[0].occurrence_locations) == 1
    for actual_row, expected_row in zip(
        report.components[0].occurrence_locations[0]["global_matrix_3x4"],
        report.components[0].occurrence_locations[0]["matrix_3x4"],
    ):
        assert actual_row == pytest.approx(expected_row)
    assert report.components[0].volume_mm3 == pytest.approx(6000.0, abs=1e-6)
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["xlsx"]).exists()
    with ZipFile(artifacts["xlsx"]) as workbook:
        assert workbook.testzip() is None
        assert "xl/worksheets/sheet1.xml" in workbook.namelist()
        assert "xl/styles.xml" in workbook.namelist()
        assert "零件编号".encode("utf-8") in workbook.read(
            "xl/worksheets/sheet1.xml"
        )
    with Path(artifacts["csv"]).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["零件编号"] == "component_0001"
    assert rows[0]["实例数量"] == "1"
    assert rows[0]["装配路径"] == "root:1"
    assert '"matrix_3x4"' in rows[0]["实例局部变换"]
    assert "实例全局变换" in rows[0]
    assert rows[0]["数量"] == "1"
    assert rows[0]["材料"] == ""
    assert rows[0]["材料来源"] == "NOT_PRESENT"
    assert rows[0]["采购规格来源"] == "NOT_PRESENT"
    assert rows[0]["类别需要复核"] == "True"
    assert "SEMANTICS_UNRESOLVED" in rows[0]["类别复核原因"]
    assert "category" in rows[0]["类别候选"]


def test我方按参考_shape合并重复_occurrence并保留装配路径() -> None:
    root = _FakeLabel(
        100,
        children=[
            _FakeLabel(101, reference_tag=200),
            _FakeLabel(102, reference_tag=200),
            _FakeLabel(103, reference_tag=201),
        ],
    )
    occurrences: list[dict[str, object]] = []
    _collect_leaf_occurrences(
        root,
        "root:1",
        _FakeShapeTool(),
        _FakeSequence,
        _FakeReferenceLabel,
        occurrences,
    )

    groups = _group_occurrences(occurrences)

    assert [group["reference_shape_id"] for group in groups] == [
        "label:200",
        "label:201",
    ]
    assert groups[0]["occurrence_ids"] == ["label:101", "label:102"]
    assert groups[0]["occurrence_paths"] == [
        "root:1/component:1",
        "root:1/component:2",
    ]
    assert groups[1]["occurrence_ids"] == ["label:103"]


def test我方重复实例数量会写入_bom数量字段() -> None:
    pytest.importorskip("OCP")
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.Bnd import Bnd_Box
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import (
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_VERTEX,
    )
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    shape = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Solid()
    component = _measure_component(
        _FakeLabel(200),
        1,
        _FakeMeasureShapeTool(shape),
        object,
        BRepBndLib,
        Bnd_Box,
        BRepGProp,
        GProp_GProps,
        TopExp_Explorer,
        {
            "SOLID": TopAbs_SOLID,
            "SHELL": TopAbs_SHELL,
            "FACE": TopAbs_FACE,
            "EDGE": TopAbs_EDGE,
            "VERTEX": TopAbs_VERTEX,
        },
        adaptor_surface_type=BRepAdaptor_Surface,
        topods_type=TopoDS,
        material_tool=_FakeMaterialTool(),
        ascii_string_type=object,
        reference_shape_id="label:200",
        occurrence_ids=["label:101", "label:102"],
        occurrence_paths=["root:1/component:1", "root:1/component:2"],
        occurrence_locations=[None, None],
    )

    assert component.occurrence_count == 2
    assert component.quantity == 2
    assert component.occurrence_path == "root:1/component:1"


def test我方名称规则只输出可解释类别候选() -> None:
    category, confidence, source, evidence = _category_candidate(
        "ejector_pin_01",
        [5.0, 5.0, 40.0],
        {"SOLID": 1},
        100.0,
        500.0,
    )

    assert category == "顶出件候选"
    assert confidence == pytest.approx(0.75)
    assert source == "NAME_RULE"
    assert "ejector" in evidence[0]


def test我方英文名称规则不会把普通子串误认为工程类别() -> None:
    category, confidence, source, evidence = _category_candidate(
        "template",
        [10.0, 20.0, 30.0],
        {"SOLID": 1},
        100.0,
        500.0,
    )

    assert category == "板件/结构件候选"
    assert confidence == pytest.approx(0.3)
    assert source == "GEOMETRY_RULE"
    assert evidence


def test我方只应用精确主数据并记录字段来源(tmp_path: Path) -> None:
    step_path = tmp_path / "box.step"
    _write_box_step(step_path)
    master_path = tmp_path / "bom_master.json"
    master_path.write_text(
        json.dumps(
            {
                "by_reference_shape": {
                    "label:1": {
                        "part_name": "MOLD_BASE_01",
                        "category": "模架件候选",
                        "material": "P20",
                        "purchase_specification": "板厚30mm",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = inspect_step_bom(step_path, load_bom_master_data(master_path))
    component = report.components[0]

    assert report.to_dict()["schema_version"] == "mold-bom-1.9"
    assert component.part_name == "MOLD_BASE_01"
    assert component.category == "模架件候选"
    assert component.part_category_group == "模架系统零件"
    assert component.category_source == "MASTER_DATA"
    assert component.material == "P20"
    assert component.material_source == "MASTER_DATA"
    assert component.purchase_specification == "板厚30mm"
    assert component.purchase_specification_source == "MASTER_DATA"
    assert component.category_review_required is False
    assert component.category_review_reason_codes == ["MASTER_DATA_CONFIRMED"]
    assert component.category_candidates[0]["confidence"] == 1.0


def test我方拒绝主数据根节点拼写错误(tmp_path: Path) -> None:
    master_path = tmp_path / "invalid_master.json"
    master_path.write_text(json.dumps({"by_ref_shape": {}}), encoding="utf-8")

    with pytest.raises(InputValidationError, match="根字段"):
        load_bom_master_data(master_path)


def test我方沿嵌套装配路径合成全局变换() -> None:
    parent = [
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
    ]
    local = [
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 5.0],
        [0.0, 0.0, 1.0, 6.0],
    ]

    result = _compose_affine_3x4(parent, local)

    for actual_row, expected_row in zip(
        result,
        [
            [1.0, 0.0, 0.0, 14.0],
            [0.0, 1.0, 0.0, 25.0],
            [0.0, 0.0, 1.0, 36.0],
        ],
    ):
        assert actual_row == pytest.approx(expected_row)


def test我方类别决策输出多候选和复核原因() -> None:
    decision = _category_decision(
        "ejector_pin_01",
        [5.0, 5.0, 40.0],
        {"SOLID": 1, "FACE": 6},
        100.0,
        500.0,
        surface_type_counts={"GeomAbs_Cylinder": 4, "GeomAbs_Plane": 2},
    )

    assert decision["category"] == "顶出件候选"
    assert decision["confidence"] == pytest.approx(0.75)
    # 名称规则和圆柱面几何同时命中同一类别时，我保留双证据来源。
    assert decision["source"] == "NAME_AND_GEOMETRY_RULE"
    assert decision["review_required"] is True
    assert "NAME_GEOMETRY_AGREEMENT" in decision["review_reason_codes"]
    assert "SEMANTICS_UNRESOLVED" in decision["review_reason_codes"]
    assert len(decision["candidates"]) >= 2
    assert decision["candidates"][0]["category"] == "顶出件候选"
    assert set(decision["candidates"][0]["sources"]) == {
        "NAME_RULE",
        "GEOMETRY_RULE",
    }


def test我方类别决策使用圆柱面几何证据() -> None:
    decision = _category_decision(
        "unknown_part",
        [4.0, 4.0, 60.0],
        {"SOLID": 1, "FACE": 10},
        200.0,
        800.0,
        surface_type_counts={"GeomAbs_Cylinder": 8, "GeomAbs_Plane": 2},
    )

    categories = {item["category"] for item in decision["candidates"]}
    assert {"导向件候选", "顶出件候选", "紧固件候选"} <= categories
    assert decision["review_required"] is True
    assert "GEOMETRY_ONLY" in decision["review_reason_codes"]
    assert "SEMANTICS_UNRESOLVED" in decision["review_reason_codes"]


def test我方类别决策在板状几何保留辅助结构件兜底() -> None:
    decision = _category_decision(
        "unknown_part",
        [10.0, 20.0, 30.0],
        {"SOLID": 1, "FACE": 6},
        100.0,
        500.0,
    )

    categories = {item["category"] for item in decision["candidates"]}
    assert "辅助结构件候选" in categories
    assert "板件/结构件候选" in categories
    assert decision["review_required"] is True


def test我方类别决策在缺少几何时保持未确定() -> None:
    decision = _category_decision(
        "unknown_part",
        None,
        {"SOLID": 0},
        None,
        None,
    )

    assert decision["category"] == "未确定"
    assert decision["confidence"] == pytest.approx(0.0)
    assert decision["source"] == "INSUFFICIENT_GEOMETRY"
    assert decision["review_required"] is True
    assert "NO_CATEGORY_EVIDENCE" in decision["review_reason_codes"]
    assert "NO_SOLID_GEOMETRY" in decision["review_reason_codes"]
    assert "MISSING_BOUNDING_BOX" in decision["review_reason_codes"]


def test我方候选标签映射到任务书八个部件大类() -> None:
    # 任务书"模具零件类型分类"给出的 8 个大类，顺序和名称必须一一对上。
    assert PART_CATEGORY_GROUPS == (
        "成型零件",
        "模架系统零件",
        "导向定位零件",
        "顶出系统零件",
        "侧向抽芯零件",
        "标准紧固件",
        "冷却浇注零件",
        "辅助结构零件",
    )
    assert canonical_part_category(["成型件候选"]) == "成型零件"
    assert canonical_part_category(["冷却浇注件候选"]) == "冷却浇注零件"
    # 同一大类的重复候选仍然唯一。
    assert canonical_part_category(["模架件候选", "模架件候选"]) == "模架系统零件"
    # 跨大类候选（板件无法区分模架件与辅助结构件）不硬凑结论。
    assert canonical_part_category(["板件/结构件候选"]) is None
    assert canonical_part_category(["模架件候选", "辅助结构件候选"]) is None
    assert canonical_part_category([]) is None


def test我方类别决策带出部件大类字段() -> None:
    # 只命中名称规则、几何规则不产出候选时，应唯一映射到"导向定位零件"。
    decision = _category_decision(
        "导柱",
        [50.0, 30.0, 10.0],
        {"SOLID": 1},
        None,
        None,
    )

    assert decision["category"] == "导向件候选"
    assert decision["part_category_group"] == "导向定位零件"

    # 无证据时保持"未确定"，部件大类必须为 None，不能瞎猜。
    unknown = _category_decision("unknown_part", None, {"SOLID": 0}, None, None)
    assert unknown["part_category_group"] is None


def test我方空形状计数为0且质量属性为None() -> None:
    """我覆盖"形状是空的"这条合法输入路径。

    先判 IsNull 再建 explorer：对空形状建 explorer 在部分 OCP
    版本上会直接抛异常，而空形状应当得到计数 0 / 质量属性 None，
    不是让整份 BOM 失败。
    """

    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Shape

    from cad_inspector.io.step_bom import _count_shapes, _mass_property

    box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Solid()
    assert _count_shapes(box, TopExp_Explorer, TopAbs_FACE) == 6
    assert _count_shapes(box, TopExp_Explorer, TopAbs_SOLID) == 1

    empty = TopoDS_Shape()
    assert _count_shapes(empty, TopExp_Explorer, TopAbs_FACE) == 0
    assert _mass_property(empty, lambda *_args: None, lambda: None) is None


def test我方质量属性异常或负值时留空而不是报错或写错值() -> None:
    """我锁定"拿不到就留空"这条契约。

    质量属性在自相交或未闭合的形状上会抛异常，而"这一个零件的
    体积算不出来"不该让整份物料清单报废。

    负值同样要挡掉：算出一个负值说明这次计算本身不可信
    （形状未闭合时体积会给出荒谬的结果），宁可留空也不能
    把一个错数值写进 BOM 让人去报价。
    """

    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    from cad_inspector.io.step_bom import _mass_property

    box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Solid()

    def exploding(shape, props):  # noqa: ARG001
        raise RuntimeError("boom")

    assert _mass_property(box, exploding, lambda: None) is None

    class _NegativeProps:
        @staticmethod
        def Mass() -> float:  # noqa: N802
            return -5.0

    def negative(shape, props):  # noqa: ARG001
        return None

    # 直接验证负值被挡掉：用一个恒返回负值的 props 工厂。
    assert (
        _mass_property(box, lambda shape, props: None, _NegativeProps) is None
    )
def test我方按证据强度选出部件大类而不被弱候选拖成空() -> None:
    """强证据的件必须拿到大类——这是本轮修掉的真实缺陷。

    我为什么必须钉死这条：名称规则命中"紧固件候选"时，几何规则会
    无条件补上"板件/结构件候选""模架件候选""辅助结构件候选"三个弱
    候选。若按"所有候选求交集"的老做法，四者跨大类交集为空，结果
    就是唯一有强证据的那批件（实测 11 件 Cap Screw / Ejector /
    Runner System）反而全部掉成 None，而 158 个纯几何兜底件反倒能
    归上——恰好把可信度搞反。
    """

    named_only = [
        {"category": "紧固件候选", "confidence": 0.9, "sources": ["NAME_RULE"]},
        {"category": "板件/结构件候选", "confidence": 0.3, "sources": ["GEOMETRY_RULE"]},
        {"category": "模架件候选", "confidence": 0.3, "sources": ["GEOMETRY_RULE"]},
        {"category": "辅助结构件候选", "confidence": 0.25, "sources": ["GEOMETRY_RULE"]},
    ]
    group, strength = canonical_part_category_from_candidates(named_only)
    assert group == "标准紧固件"
    assert strength == "named"


def test我方名称与几何一致时标为agreed档() -> None:
    """两方证据同时指向主类别，强度应高于纯名称。"""

    agreed = [
        {
            "category": "顶出件候选",
            "confidence": 0.6,
            "sources": ["NAME_RULE", "GEOMETRY_RULE"],
        },
        {"category": "辅助结构件候选", "confidence": 0.25, "sources": ["GEOMETRY_RULE"]},
    ]
    group, strength = canonical_part_category_from_candidates(agreed)
    assert group == "顶出系统零件"
    assert strength == "agreed"


def test我方纯几何候选跨大类时仍保留为空() -> None:
    """没有强证据时我仍然不硬凑——这条契约不能被上一条的修复破坏。"""

    geometry_only = [
        {"category": "模架件候选", "confidence": 0.3, "sources": ["GEOMETRY_RULE"]},
        {"category": "辅助结构件候选", "confidence": 0.25, "sources": ["GEOMETRY_RULE"]},
    ]
    group, strength = canonical_part_category_from_candidates(geometry_only)
    assert group is None
    assert strength == "none"


def test我方纯几何候选同源时归入该大类并标geometry档() -> None:
    """纯几何但同档内不冲突时可以归类，但强度必须标成 geometry。"""

    same_group = [
        {"category": "辅助结构件候选", "confidence": 0.2, "sources": ["GEOMETRY_RULE"]},
    ]
    group, strength = canonical_part_category_from_candidates(same_group)
    assert group == "辅助结构零件"
    assert strength == "geometry"


def test我方主类别是跨大类中间标签时不给大类() -> None:
    """"板件/结构件候选"本身跨模架与辅助两个大类，不能当结论。"""

    cross = [
        {"category": "板件/结构件候选", "confidence": 0.3, "sources": ["GEOMETRY_RULE"]},
    ]
    group, strength = canonical_part_category_from_candidates(cross)
    assert group is None
    assert strength == "none"


def test我方空候选列表返回空且不抛异常() -> None:
    """没有候选时返回 (None, none)，不能炸。"""

    assert canonical_part_category_from_candidates([]) == (None, "none")


def test我方大类强度字段默认值为none() -> None:
    """新字段必须有默认值，且不破坏既有必填字段顺序。"""

    from cad_inspector.domain.bom import BomComponent

    field_names = [item.name for item in fields(BomComponent)]
    assert "part_category_group_strength" in field_names
    # 有默认值的字段必须全部排在无默认值字段之后，否则 dataclass 直接报错。
    has_default = [
        item.name
        for item in fields(BomComponent)
        if item.default is not MISSING
        or item.default_factory is not MISSING  # type: ignore[arg-type]
    ]
    assert "part_category_group_strength" in has_default
    assert has_default == field_names[len(field_names) - len(has_default) :]


def test我方bom数量与装配树实例数对得上才算不重不漏() -> None:
    """我把"不遗漏任何零件"从口号变成可对账的判据。

    我为什么钉这条：此前这条要求没有任何判据，BOM 少算零件不会报错、
    甲方也看不出来。实测 mould_04 就是"数量合计 159、装配树实例 156"，
    看表完全发现不了。
    """

    from cad_inspector.domain.bom import reconcile_bom_completeness

    class _Component:
        def __init__(self, quantity: int, occurrence_ids: list[str]) -> None:
            self.quantity = quantity
            self.occurrence_ids = occurrence_ids

    # 正常：2 个实例 -> 覆盖 2、合计 2
    ok = [
        _Component(1, ["label:1"]),
        _Component(1, ["label:2"]),
    ]
    result = reconcile_bom_completeness(ok, 2)
    assert result["status"] == "PASS"
    assert result["coverage_ok"] is True
    assert result["quantity_ok"] is True

    # 漏计：装配树 3 个实例，BOM 只覆盖 2 个
    missed = [
        _Component(1, ["label:1"]),
        _Component(1, ["label:2"]),
    ]
    result = reconcile_bom_completeness(missed, 3)
    assert result["status"] == "FAIL"
    assert result["coverage_ok"] is False

    # 重复计：同一 occurrence 出现在两行
    dup = [
        _Component(1, ["label:1"]),
        _Component(1, ["label:1"]),
    ]
    result = reconcile_bom_completeness(dup, 2)
    assert result["status"] == "FAIL"
    assert result["duplicated_occurrence_count"] == 1


def test我方拆分后的派生occurrence_id不会彼此重复() -> None:
    """这是本轮修掉的真实缺陷：拆分会让同一实例被计 N 次。

    一个装配节点含 3 个 Solid 时被拆成 3 行，若各行直接继承原节点
    那 1 个 occurrence，同一个 id 就出现 3 次——数量虚高 2 件。
    修法是给拆分行各自派生带 `#solid_NNN` 后缀的 id。
    """

    pytest.importorskip("OCP")
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    compound = _make_multi_solid_compound(box_count=3)
    shape_tool = _FakeMeasureShapeTool(compound)

    units, _ = _expand_multi_solid_groups(
        [_fake_group()],
        shape_tool,
        TopExp_Explorer,
        TopAbs_SOLID,
        TopoDS,
    )

    ids: list[str] = []
    for unit in units:
        ids += list(unit["occurrence_ids"])
    # 3 个拆分行 -> 3 个互不相同的派生 id，不再是同一个 id 重复 3 次。
    assert len(ids) == 3
    assert len(set(ids)) == 3
    # 但剥掉后缀后仍能还原成同一个装配实例，保证"没漏"仍可判定。
    assert {item.split("#solid_")[0] for item in ids} == {"label:1"}


def test我方已交付的bom产物没有带重复计的数量() -> None:
    """我守住**交付物**：磁盘上的 BOM 产物必须对账通过。

    我为什么单独加这条（2026-10-15）：上一轮我修好了代码与判据，
    但 `outputs/bom_batch` 下的产物仍是修复前生成的，数量列含重复计。
    甲方看的是产物不是源码——只改代码不改产物等于没改。
    这条测试的作用是把残留暴露出来，逼人显式重新生成。
    """

    root = Path(__file__).resolve().parents[1]
    stale: list[str] = []
    checked = 0
    for path in sorted(root.glob("outputs/bom_batch/*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        completeness = (payload.get("diagnostics") or {}).get("completeness")
        if completeness is None:
            # 根本没有对账字段 = 修复前生成的产物，必须重新生成。
            stale.append(str(path.relative_to(root)) + "（缺 completeness 字段）")
            continue
        checked += 1
        if completeness.get("status") != "PASS":
            stale.append(
                str(path.relative_to(root)) + f"（对账 {completeness.get('status')}）"
            )
        elif completeness.get("duplicated_occurrence_count"):
            stale.append(
                str(path.relative_to(root))
                + f"（重复计 {completeness['duplicated_occurrence_count']} 件）"
            )
    assert not stale, (
        "我方磁盘上还有带着重复计数的 BOM 产物，甲方拿去采购会多买：\n"
        + "\n".join(stale[:10])
    )
    # 至少确实查过产物，避免这条测试因为 glob 没匹配上而空过。
    assert checked > 0, "我方没找到任何 BOM 产物可对账"


def test我方把0命中的大类分成我方认不出和几何歧义两种() -> None:
    """我把"某大类 0 件"的两种含义分开，不让 0 被读成"没有这类零件"。

    我为什么钉这条：真实数据上有 7 个大类恒为 0，但它们的 0 含义不同——
    "成型零件"等 3 类**只有名称规则能命中**，而实测 509 个零件名里有
    363 个是哈希串（形如 `52561FBC24A736ws`），根本没有关键词可匹配。
    这类 0 是我方能力缺口，不是模具里没有。成型零件（型腔/型芯）是任何
    一套注塑模的定义性零件，它报 0 几乎必然是漏识别。
    """

    from cad_inspector.domain.bom import summarize_category_coverage

    class _Component:
        def __init__(self, group: str | None) -> None:
            self.part_category_group = group

    # 只命中辅助结构零件，其余全 0。
    result = summarize_category_coverage([_Component("辅助结构零件")])

    assert result["counts"]["辅助结构零件"] == 1
    assert result["counts"]["成型零件"] == 0
    # 能力缺口：只有名称能命中的三类必须被单列出来。
    assert set(result["unreachable_groups"]) == {
        "成型零件",
        "侧向抽芯零件",
        "冷却浇注零件",
    }
    # 几何歧义导致的 0 不能被误报成能力缺口。
    assert "模架系统零件" not in result["unreachable_groups"]
    assert "模架系统零件" in result["zero_groups"]
    # 每个 0 都必须有原因文案，不允许只给一个裸 0。
    assert len(result["zero_reasons"]) == len(result["zero_groups"])
    for group in result["zero_groups"]:
        assert group in result["zero_reasons"]


def test我方命中后不再列为能力缺口() -> None:
    """若有名称证据命中成型零件，它就不该再出现在 unreachable 里。"""

    from cad_inspector.domain.bom import summarize_category_coverage

    class _Component:
        def __init__(self, group: str | None) -> None:
            self.part_category_group = group

    result = summarize_category_coverage(
        [_Component("成型零件"), _Component("侧向抽芯零件"), _Component(None)]
    )

    assert result["counts"]["成型零件"] == 1
    assert "成型零件" not in result["unreachable_groups"]
    assert "成型零件" not in result["zero_groups"]
    assert result["unclassified_count"] == 1
