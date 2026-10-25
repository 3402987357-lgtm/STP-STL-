from __future__ import annotations

import re
from pathlib import Path

import pytest

from cad_inspector.io.step import load_step_as_mesh

# STEP 的 LENGTH_UNIT 声明，形如 NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.)
_LENGTH_UNIT_RE = re.compile(
    r"LENGTH_UNIT\s*\(\s*\)?NAMED_UNIT\s*\([^)]*\)\s*SI_UNIT\s*\(\s*"
    r"(?:\.(\w+)\.\s*,\s*)?\.(\w+)\."
)


def test我方OCP读取器把明确厘米STEP归一化到毫米(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    source = tmp_path / "box.step"
    writer = STEPControl_Writer()
    assert (
        writer.Transfer(
            BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Solid(),
            STEPControl_AsIs,
        )
        == IFSelect_ReturnStatus.IFSelect_RetDone
    )
    assert writer.Write(str(source)) == IFSelect_ReturnStatus.IFSelect_RetDone

    cm_source = tmp_path / "box_cm.step"
    cm_source.write_text(
        source.read_text(encoding="utf-8", errors="ignore").replace(
            ".MILLI.",
            ".CENTI.",
        ),
        encoding="utf-8",
    )

    mesh = load_step_as_mesh(cm_source)

    assert mesh.metadata["unit"] == "cm"
    assert mesh.metadata["source_unit"] == "cm"
    assert mesh.metadata["coordinate_unit"] == "mm"
    assert mesh.metadata["unit_scale_to_mm"] == 10.0
    assert mesh.metadata["unit_normalization"] == "occt_step_reader_to_mm"
    assert mesh.bbox_max.tolist() == pytest.approx([100.0, 200.0, 300.0])


def test我方真实语料里每个STEP都明确声明毫米() -> None:
    """我钉住甲方真实语料的单位声明，避免它悄悄变成别的单位。

    我为什么加这条：README 里"单位已确认"这条结论是靠**一次性人工扫描**
    得出的，写死在文档里。文档不会随数据变化而变化——
    哪天语料里混进一个厘米件，README 还是写着"全部是毫米"，
    而管线会继续按 mm 解释数值，误差直接放大 10 倍且没有任何告警。

    我把这个结论从"文档里的一句话"变成"每次门禁都会重跑的断言"：
    扫全部真实 STEP，要求每个都声明 SI_UNIT(.MILLI.,.METRE.)。
    语料缺失时我 skip 而不是"因为没数据所以通过"——
    后者会让这条在换机器上静默失效。
    """

    corpus = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "provided"
        / "三维模型破损区域智能修补"
    )
    sources = sorted(corpus.rglob("*.stp"))
    if not sources:
        pytest.skip("我找不到真实 STEP 语料，无法确认单位声明")

    offenders: list[str] = []
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = _LENGTH_UNIT_RE.search(text)
        if match is None:
            offenders.append(
                f"{path.relative_to(corpus)}：我找不到 LENGTH_UNIT 声明"
            )
            continue
        prefix, base = match.group(1), match.group(2)
        if (prefix, base) != ("MILLI", "METRE"):
            offenders.append(
                f"{path.relative_to(corpus)}：声明的是 {prefix}/{base}，不是毫米"
            )

    assert not offenders, (
        f"真实语料里有 {len(offenders)} 个 STEP 没有明确声明毫米，"
        "按 mm 解释数值会直接引入量级误差：\n" + "\n".join(offenders[:10])
    )
    # 我顺手钉住数量：语料被误删一半的话，上面那条会因为"扫得少"而照样通过。
    assert len(sources) >= 21, f"真实 STEP 语料只剩 {len(sources)} 个"
