"""我方的 STEP 文本格式探测器。

这里不是完整 STEP 几何解析器。我方只在第一阶段读取文本头和实体数量，
用于判断文件是否值得交给后续 OCC 适配器。实体的精确拓扑、尺寸和体积
必须由后续授权或合规的 B-Rep 几何后端完成。

算法原理：探测阶段只做"值不值得继续"的判断，绝不假装解析出几何。三件事：

1. 格式魔数。找 ``ISO-10303-21;`` 标记，找不到就判 FAILED——后缀是 .stp 但内容不是
   STEP 的文件很常见，靠后缀信任会一路错到底。
2. 头部解析。STEP 的 FILE_DESCRIPTION 之类字段是嵌套括号结构，而且字符串里可能出现
   转义的单引号，所以 ``_extract_balanced_call`` 必须带引号状态扫描，不能简单数括号。
3. 单位线索。只报告文件里显式写出的 SI_UNIT，认不出就返回 None，绝不猜——把米当毫米
   会让所有尺寸差 1000 倍，而且不会报错。

关键边界：XT / PRT 格式只识别、不解析（需要已授权的 Parasolid 或 NX 适配器），返回
UNSUPPORTED 并给出适配器需求代码，绝不伪造解析成功。
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from cad_inspector.domain.models import (
    DetectedFormat,
    Issue,
    ParseStatus,
    ProbeResult,
    Severity,
)
from cad_inspector.io.input_validator import InputValidationError, validate_input_path


_ENTITY_PATTERN = re.compile(r"#\s*\d+\s*=\s*([A-Z0-9_]+)\s*\(", re.IGNORECASE)
_QUOTED_PATTERN = re.compile(r"'((?:''|[^'])*)'")
_STEP_MAGIC = re.compile(r"ISO-10303-21\s*;", re.IGNORECASE)

_IMPORTANT_ENTITIES = {
    "PRODUCT",
    "PRODUCT_DEFINITION",
    "PRODUCT_DEFINITION_SHAPE",
    "SHAPE_REPRESENTATION",
    "MANIFOLD_SOLID_BREP",
    "CLOSED_SHELL",
    "OPEN_SHELL",
    "ADVANCED_FACE",
    "FACE_SURFACE",
    "EDGE_CURVE",
    "VERTEX_POINT",
    "CARTESIAN_POINT",
    "AXIS2_PLACEMENT_3D",
    "SI_UNIT",
    "CONVERSION_BASED_UNIT",
}


def _unquote_values(value: str) -> list[str]:
    """我方把 STEP header 中的单引号字段转成可读字符串。"""

    return [item.replace("''", "'") for item in _QUOTED_PATTERN.findall(value)]


def _extract_balanced_call(text: str, open_index: int) -> str | None:
    """我用带引号状态的括号扫描处理嵌套 FILE_DESCRIPTION。

    为什么不能简单数括号：STEP 的字符串字面量里可以出现括号，
    例如 FILE_DESCRIPTION(('这里有 ( 括号'),'2;1')。
    无状态扫描会把字符串里的 '(' 当成结构，depth 永远归不了零，
    要么截断字段要么读到文件尾。

    为什么引号要单独处理转义：STEP 用两个连续单引号 '' 表示
    字面量里的一个单引号。遇到 '' 必须一次跳两格，
    否则引号状态会被翻转两次等于没变，之后的括号扫描全部错位。

    括号不配对时返回 None 而不是抛错：文件被截断是真实存在的场景，
    探测阶段应当"这个字段读不出来就跳过"，
    让上层把它记成 issue 并继续处理其余字段。
    """

    depth = 0
    in_quote = False
    index = open_index
    while index < len(text):
        char = text[index]
        if char == "'":
            if in_quote and index + 1 < len(text) and text[index + 1] == "'":
                index += 2
                continue
            in_quote = not in_quote
        elif not in_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return text[open_index + 1 : index]
        index += 1
    return None


def _parse_header(text: str) -> dict[str, Any]:
    """我扫描 STEP 头部字段，用带引号状态的括号配对取出字段内容。"""
    header: dict[str, Any] = {}
    for field_match in re.finditer(r"\b(FILE_[A-Z_]+)\s*\(", text, re.IGNORECASE):
        body = _extract_balanced_call(text, field_match.end() - 1)
        if body is None:
            continue
        values = _unquote_values(body)
        if values:
            header[field_match.group(1).upper()] = values
    return header


def _detect_unit(text: str) -> str | None:
    """我方只报告文件中明确可见的单位线索，不自行猜测。

    认不出就返回 None，绝不兜底成"毫米"。这是探测阶段最不能猜的
    一个字段：把米当毫米会让所有尺寸差 1000 倍，而**不会报错**——
    零件看起来正常，只是所有偏差都远超阈值，最终报出一整片
    荒谬的缺陷。宁可返回 None 让上层显式要求用户确认单位。

    匹配顺序按"特异性从强到弱"：先认 SI_UNIT 的精确组合
    （.MILLI./.CENTI. 配 .METRE.），最后才退回裸的 INCH 关键字。
    反过来会出问题——一个写着 SI_UNIT(.MILLI.,.METRE.)
    但注释里提到 inch 的文件会被误判成英寸。
    """

    upper = text.upper()
    if re.search(r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)", upper):
        return "mm"
    if re.search(r"SI_UNIT\s*\(\s*\.CENTI\.\s*,\s*\.METRE\.\s*\)", upper):
        return "cm"
    if re.search(r"SI_UNIT\s*\(\s*\$\s*,\s*\.METRE\.\s*\)", upper):
        return "m"
    if "INCH" in upper or "INCHES" in upper:
        return "inch"
    return None


def _count_entities(text: str) -> dict[str, int]:
    """我统计关注的 STEP 实体出现次数，不做完整拓扑解析。"""
    counts: dict[str, int] = {}
    for match in _ENTITY_PATTERN.finditer(text):
        name = match.group(1).upper()
        if name in _IMPORTANT_ENTITIES:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _read_step_text(path: Path) -> tuple[str, list[Issue]]:
    """我读取 STEP 文本；非 UTF-8 时用兼容方式读并留下警告。

    回退到 latin-1 而不是直接放弃：老 CAD 系统导出的 STEP 常有
    非 UTF-8 编码的中文注释。latin-1 永不解码失败（任何字节都有
    对应码位），用它至少能把文件读完并解析出实体结构，
    只牺牲注释的可读性。

    用 errors="replace" 而非 "ignore"：replace 会留下 U+FFFD 标记，
    让"这里有乱码"在输出里看得见；ignore 会静默丢字节，
    把问题藏起来。
    """
    issues: list[Issue] = []
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")
        issues.append(
            Issue(
                code="STEP_NON_UTF8_TEXT",
                message="STEP 文件不是 UTF-8，已用兼容方式读取文本探测。",
                severity=Severity.WARNING,
            )
        )
    return text, issues


def probe_step(path: str | Path) -> ProbeResult:
    """我方只做 STEP 文本探测，并返回可序列化结果。"""

    started = time.perf_counter()
    metadata = validate_input_path(path)
    if metadata.detected_format is not DetectedFormat.STEP:
        raise InputValidationError(f"该方法只接受 STEP/STP: {metadata.path}")

    text, issues = _read_step_text(Path(metadata.path))
    header: dict[str, Any] = {}
    entity_counts: dict[str, int] = {}
    status = ParseStatus.COMPLETE

    # 没有 ISO-10303-21 标记就判 FAILED：后缀不可信，内容才是依据。
    if not _STEP_MAGIC.search(text):
        status = ParseStatus.FAILED
        issues.append(
            Issue(
                code="STEP_MAGIC_MISSING",
                message="文件扩展名是 STEP/STP，但未找到 ISO-10303-21 文件标记。",
                severity=Severity.ERROR,
            )
        )
    else:
        header = _parse_header(text)
        entity_counts = _count_entities(text)
        required_sections = ("HEADER;", "DATA;", "ENDSEC;", "END-ISO-10303-21;")
        missing_sections = [
            section
            for section in required_sections
            if not re.search(re.escape(section), text, re.IGNORECASE)
        ]
        # 缺标准段落只降级为 PARTIAL 并警告，因为部分工具导出的 STEP 确实少段落但可用。
        if missing_sections:
            # 缺标准段落只降级为 PARTIAL 而不是判失败：
            # 很多能正常解析的 STEP 文件并不严格齐备所有段落，
            # 判失败会挡掉这些其实可用的文件。
            status = ParseStatus.PARTIAL
            issues.append(
                Issue(
                    code="STEP_SECTION_MISSING",
                    message="STEP 文件缺少一个或多个标准段落。",
                    severity=Severity.WARNING,
                    # 把具体缺哪些段落写进 detail：
                    # 只说"缺少段落"无法判断影响有多大。
                    detail=", ".join(missing_sections),
                )
            )
        if "FILE_SCHEMA" not in header:
            issues.append(
                Issue(
                    code="STEP_SCHEMA_MISSING",
                    message="STEP header 中未找到 FILE_SCHEMA。",
                    # 缺 schema 只是警告不改状态：
                    # 没有 schema 就无法判断文件属于 AP203 还是 AP242，
                    # 但这不影响能否解析出几何。
                    severity=Severity.WARNING,
                )
            )
        if not entity_counts:
            # 一个实体都没探测到就降级为 PARTIAL：
            # 这通常是文件真的有问题，但也可能只是
            # 本探针关注的那几类实体恰好一个都没出现。
            status = ParseStatus.PARTIAL
            issues.append(
                Issue(
                    code="STEP_ENTITY_COUNT_EMPTY",
                    message="没有探测到我方关注的 STEP 实体；需要几何后端进一步判断。",
                    severity=Severity.WARNING,
                )
            )

    return ProbeResult(
        file=metadata,
        status=status,
        header=header,
        entity_counts=entity_counts,
        # 单位只给提示而不是断言：STEP 文件可能不标单位，
        # 猜错会让后面所有毫米量级的阈值失去依据。
        unit_hint=_detect_unit(text),
        # 后端必需恒为 True：这一步是纯文本探测，
        # 任何几何结论都必须由 OCP 给出，不能在这里下结论。
        backend_required=True,
        issues=issues,
        elapsed_seconds=time.perf_counter() - started,
    )


def probe_stl(
    metadata,
    *,
    severity_threshold=None,
) -> ProbeResult:
    """我真正解析 STL 并给出网格质量，而不是只认扩展名。

    我为什么要写这个函数：dataset_cli 的 docstring 明确写着"扫描 STEP、
    XT、PRT 和 STL 样本"，但实测 21 个 STL **全部**被判 UNSUPPORTED
    （code=STL_BACKEND_NOT_IMPLEMENTED，"当前阶段只识别扩展名"）——也就是
    清单里一半的样本没有任何实体信息。而项目里 `load_stl` 与
    `mesh_quality_report` 早就有了，它们能给出真实的顶点数/面数与网格
    质量指标。我只做编排，不重新实现解析。

    我刻意不改 XT / PRT：那两种确实需要已授权的 Parasolid / NX 适配器，
    伪造解析成功比报 UNSUPPORTED 更糟。只有 STL 是本仓库自己就能读的。
    """

    started = time.perf_counter()
    try:
        from cad_inspector.geometry.mesh import mesh_quality_report
        from cad_inspector.geometry.stl import load_stl

        mesh = load_stl(metadata.path)
        quality = mesh_quality_report(mesh)
    except (ImportError, OSError, ValueError) as exc:
        return ProbeResult(
            file=metadata,
            status=ParseStatus.FAILED,
            backend_required=False,
            issues=[
                Issue(
                    code="STL_PARSE_FAILED",
                    message=f"我解析 STL 失败：{exc}",
                    severity=Severity.ERROR,
                )
            ],
            elapsed_seconds=round(time.perf_counter() - started, 6),
        )

    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    counts = {
        "vertex_count": int(len(vertices)) if vertices is not None else 0,
        "triangle_count": int(len(faces)) if faces is not None else 0,
    }
    header = {"format": "STL", "backend": "triangle_mesh"}
    issues = []
    for key in (
        "degenerate_triangle_count",
        "non_manifold_edge_count",
        "boundary_edge_count",
        "isolated_vertex_count",
    ):
        value = quality.get(key)
        if isinstance(value, (int, float)):
            counts[key] = int(value)
    # 退化面与非流形边是"这网格可不可信"的硬指标，我照实报出来，
    # 但不据此改判 status——探测阶段只回答"值不值得继续"。
    if counts.get("degenerate_triangle_count"):
        issues.append(
            Issue(
                code="STL_DEGENERATE_TRIANGLES",
                message=(
                    f"我发现 {counts['degenerate_triangle_count']} 个退化三角面，"
                    "法向与最近点判定可能不可靠。"
                ),
                severity=Severity.WARNING,
            )
        )
    if counts.get("non_manifold_edge_count"):
        issues.append(
            Issue(
                code="STL_NON_MANIFOLD_EDGES",
                message=(
                    f"我发现 {counts['non_manifold_edge_count']} 条非流形边，"
                    "体积与内外判定可能不可信。"
                ),
                severity=Severity.WARNING,
            )
        )
    return ProbeResult(
        file=metadata,
        status=ParseStatus.COMPLETE,
        header=header,
        entity_counts=counts,
        backend_required=False,
        issues=issues,
        elapsed_seconds=round(time.perf_counter() - started, 6),
    )


def probe_any(path: str | Path) -> ProbeResult:
    """我方统一入口：支持的格式会被识别，不支持的格式不会伪造解析成功。"""

    started = time.perf_counter()
    metadata = validate_input_path(path)

    # 只有 STEP 走真正的文本探测；其余格式一律 UNSUPPORTED + 明确的适配器需求代码。
    if metadata.detected_format is DetectedFormat.STEP:
        return probe_step(metadata.path)

    # STL 我现在真的解析（见 probe_stl 的说明）；XT / PRT 仍然只识别
    # 不解析——它们需要已授权的 Parasolid / NX 适配器，我不伪造。
    if metadata.detected_format is DetectedFormat.STL:
        return probe_stl(metadata)

    if metadata.detected_format is DetectedFormat.PARASOLID_XT:
        issue = Issue(
            code="XT_ADAPTER_REQUIRED",
            message="XT 需要已授权的 Parasolid 或商业 CAD 适配器，当前阶段不直接读取。",
            severity=Severity.WARNING,
        )
    elif metadata.detected_format is DetectedFormat.SIEMENS_NX_PRT:
        issue = Issue(
            code="PRT_ADAPTER_REQUIRED",
            message="NX PRT 需要 NX Open 或已授权转换流程，当前阶段不直接读取。",
            severity=Severity.WARNING,
        )
    else:
        issue = Issue(
            code="FORMAT_UNSUPPORTED",
            message="当前阶段没有可用的输入适配器。",
            severity=Severity.ERROR,
        )

    return ProbeResult(
        file=metadata,
        status=ParseStatus.UNSUPPORTED,
        backend_required=True,
        issues=[issue],
        elapsed_seconds=time.perf_counter() - started,
    )
