from __future__ import annotations

from pathlib import Path

from cad_inspector.domain.models import DetectedFormat, ParseStatus
from cad_inspector.io.step_probe import (
    _detect_unit,
    _extract_balanced_call,
    probe_any,
    probe_step,
)


MINIMAL_STEP = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('test'),'2;1');
FILE_NAME('中文 模型.step','2026-10-02T12:00:00',('author'),('org'),'writer','system','auth');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = PRODUCT('p','','',());
#2 = MANIFOLD_SOLID_BREP('solid',#3);
#3 = CLOSED_SHELL('',());
#4 = ADVANCED_FACE('',(),$);
#5 = CARTESIAN_POINT('',(0.,0.,0.));
#6 = AXIS2_PLACEMENT_3D('',#5,$,$);
#7 = SI_UNIT(.MILLI.,.METRE.);
ENDSEC;
END-ISO-10303-21;
"""


def test我方探测_step_header_和实体(tmp_path: Path) -> None:
    source = tmp_path / "中文 模型.step"
    source.write_text(MINIMAL_STEP, encoding="utf-8")

    result = probe_step(source)

    assert result.status is ParseStatus.COMPLETE
    assert result.file.detected_format is DetectedFormat.STEP
    assert result.header["FILE_SCHEMA"] == ["AUTOMOTIVE_DESIGN"]
    assert result.entity_counts["MANIFOLD_SOLID_BREP"] == 1
    assert result.entity_counts["CLOSED_SHELL"] == 1
    assert result.unit_hint == "mm"
    assert not [issue for issue in result.issues if issue.severity.value == "ERROR"]


def test我方对伪_step扩展名给出失败(tmp_path: Path) -> None:
    source = tmp_path / "bad.stp"
    source.write_text("not a step file", encoding="utf-8")

    result = probe_step(source)

    assert result.status is ParseStatus.FAILED
    assert any(issue.code == "STEP_MAGIC_MISSING" for issue in result.issues)


def test我方对_xt返回需要适配器(tmp_path: Path) -> None:
    source = tmp_path / "model.xt"
    source.write_bytes(b"licensed adapter required")

    result = probe_any(source)

    assert result.status is ParseStatus.UNSUPPORTED
    assert result.file.detected_format is DetectedFormat.PARASOLID_XT
    assert result.issues[0].code == "XT_ADAPTER_REQUIRED"


def test我方括号扫描不被字符串里的括号和转义引号骗过() -> None:
    """我锁定括号扫描必须带引号状态。

    无状态扫描的两个具体失败模式：
      - 字符串字面量里含 '(' 时，depth 永远归不了零，字段会被截断
        或一路读到文件尾；
      - STEP 用两个连续单引号 '' 表示字面量里的一个单引号，
        不一次跳两格就会把引号状态翻转两次（等于没变），
        之后的括号扫描全部错位。
    """

    with_paren = "FILE_DESCRIPTION(('has ( paren'),'2;1');"
    assert _extract_balanced_call(
        with_paren, with_paren.index("(")
    ) == "('has ( paren'),'2;1'"

    escaped = "FILE_NAME('a''b');"
    assert _extract_balanced_call(escaped, escaped.index("(")) == "'a''b'"


def test我方括号不配对时返回None而不是抛错() -> None:
    """我覆盖截断文件：读不出的字段应当跳过，而不是让探测整体失败。"""

    truncated = "FILE_DESCRIPTION(('x');"
    assert _extract_balanced_call(truncated, truncated.index("(")) is None


def test我方单位识别按特异性排序且认不出就返回None() -> None:
    """我锁定"绝不猜单位"这条契约。

    认不出必须返回 None：把米当毫米会让所有尺寸差 1000 倍却不报错，
    零件看起来正常、只有偏差全部离谱，是最难排查的一类问题。
    """

    # 注释里提到 inch，但 SI_UNIT 明确写毫米：必须以 SI_UNIT 为准。
    assert _detect_unit("/* sized in inch */ SI_UNIT(.MILLI.,.METRE.);") == "mm"
    assert _detect_unit("SI_UNIT(.INCH.);") == "inch"
    assert _detect_unit("nothing here") is None


def test我方STL真正解析出网格计数而不是只认扩展名(tmp_path) -> None:
    """我保证 STL 探测给出真实顶点/面数，不只是 UNSUPPORTED。

    实测发现：dataset_cli 的 docstring 写着"扫描 STEP、XT、PRT 和 STL"，
    但 21 个 STL **全部**被判 UNSUPPORTED（code=STL_BACKEND_NOT_
    IMPLEMENTED，"当前阶段只识别扩展名"）——清单里一半样本没有任何
    实体信息。而项目里 `load_stl` 与 `mesh_quality_report` 早就有了。
    我改成真的解析 XT / PRT 仍然只识别不解析（它们需要已授权适配器，
    伪造成功比报 UNSUPPORTED 更糟）。
    """

    from cad_inspector.domain.models import ParseStatus
    from cad_inspector.geometry.stl import save_stl
    from cad_inspector.io.step_probe import probe_any
    from test_core import cube_mesh

    target = tmp_path / "cube.stl"
    save_stl(cube_mesh(), target)
    result = probe_any(target)
    assert result.status is ParseStatus.COMPLETE, result.status
    assert result.entity_counts.get("vertex_count", 0) > 0, result.entity_counts
    assert result.entity_counts.get("triangle_count", 0) > 0, result.entity_counts
    # 只认扩展名时 backend_required 为 True；真解析后不该再要后端。
    assert result.backend_required is False, result.backend_required
