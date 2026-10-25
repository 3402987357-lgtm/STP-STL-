"""我把 BOM 基线结果写成 JSON、XLSX 和 Excel 兼容 CSV。

原理说明：同一份 BOM 我输出三种格式，各有各的用途，不是简单重复：

1. JSON —— 给程序和审计器读。字段名是英文键、结构完整、带 schema_version，
   方便下游做自动比对；数字保持原生类型，不做字符串化。
2. CSV —— 给甲方用 Excel 直接打开。我用 utf-8-sig（带 BOM）编码，
   否则简体中文 Excel 会把表头当 ANSI 读成乱码；列顺序就是 _fieldnames 的顺序。
3. XLSX —— 给甲方当正式交付表。这里我不引入 openpyxl 之类的第三方依赖，
   而是用标准库 zipfile 手写一个最小但合规的 OOXML 包：Content_Types、
   包级关系、workbook、worksheet、styles 五个成员齐全，Excel 能正常打开，
   表头冻结、自动筛选和列宽都在。

三份文件都通过 atomic_path 先写同目录临时文件再替换正式文件，
所以中途失败不会留下半个 BOM。
"""

from __future__ import annotations

import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from ..domain.bom import BomReport
from .atomic import atomic_path


# OOXML 用到的四个命名空间。手写 XML 时必须显式带上，否则 Excel 会拒绝打开。
_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"


def write_bom_report(report: BomReport, output_dir: str | Path) -> dict[str, str]:
    """我输出结构化 JSON、原生 XLSX 和 UTF-8 CSV，并返回三个路径。"""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    # 三种输出共用同一个文件名主干，只有扩展名不同，方便甲方按零件名归档。
    stem = Path(report.file_path).stem
    json_path = target_dir / f"{stem}.bom.json"
    csv_path = target_dir / f"{stem}.bom.csv"
    xlsx_path = target_dir / f"{stem}.bom.xlsx"
    fieldnames = _fieldnames()
    rows = _rows(report)

    # JSON：我先写临时文件再替换，保证不会留下半个 JSON。
    with atomic_path(json_path) as temporary:
        temporary.write_text(
            json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    # CSV：utf-8-sig 是给 Excel 看的中文兼容写法，BOM 让 Excel 正确识别编码。
    with atomic_path(csv_path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    _write_xlsx(xlsx_path, fieldnames, rows)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "xlsx": str(xlsx_path),
    }


def _fieldnames() -> list[str]:
    """我返回 BOM 的列顺序。

    这个顺序是甲方报告契约的一部分：任务书点名的"序号、零件编号、零件名称、
    材料、数量、采购规格、备注"六列必须在表里能直接找到，其余是我方为了
    可追溯补上的证据列（参考 Shape、实例、类别证据、几何尺寸等）。
    我把它写成显式列表而不是从 dict 推导，因为列顺序一旦漂移，
    甲方已有的解析脚本就会错位。
    """

    return [
        # 基础标识：序号、零件编号、参考 Shape 与实例信息。
        "序号",
        "零件编号",
        "参考Shape编号",
        "实例数量",
        "实例编号",
        "装配路径",
        "实例局部变换",
        "实例全局变换",
        # 名称与类别：部件类别是内部标签，部件大类是任务书要求的 8 大类映射。
        "零件名称",
        "部件类别",
        "部件大类",
        # 大类证据强度紧随大类：空大类配 none 强度，
        # 让复核者一眼看出"没归上"是证据不足还是候选冲突。
        "大类证据强度",
        "类别置信度",
        "类别来源",
        "类别证据",
        "类别候选",
        "类别需要复核",
        "类别复核原因",
        # 任务书点名的物料列，加一列来源以便追溯是主数据还是推断。
        "材料",
        "材料来源",
        "数量",
        "采购规格",
        "采购规格来源",
        "备注",
        # 几何与拓扑证据：让甲方能判断"这个零件大概是什么"。
        "X尺寸_mm",
        "Y尺寸_mm",
        "Z尺寸_mm",
        "表面积_mm2",
        "体积_mm3",
        "Solid数",
        "Shell数",
        "Face数",
        "Edge数",
        "Vertex数",
        # 复核警告放最后一列，避免打断前面的主数据阅读。
        "人工复核警告",
    ]


def _rows(report: BomReport) -> list[dict[str, Any]]:
    """我把每个零部件拍成一行字典，键就是 _fieldnames 里的列名。"""

    rows: list[dict[str, Any]] = []
    # 序号从 1 开始按报告顺序编号，和 JSON 里的顺序保持一致。
    for index, component in enumerate(report.components, start=1):
        # 尺寸可能缺失；缺失时我用三个 None，让 XLSX 留空格而不是写 "None"。
        dimensions = component.dimensions_xyz_mm or [None, None, None]
        rows.append(
            {
                "序号": index,
                "零件编号": component.component_id,
                "参考Shape编号": component.reference_shape_id,
                "实例数量": component.occurrence_count,
                "实例编号": " | ".join(component.occurrence_ids),
                "装配路径": " | ".join(component.occurrence_paths),
                # 实例变换是嵌套结构，CSV/XLSX 里我用紧凑 JSON 字符串承载，
                # separators 去掉空格、allow_nan=False 拒绝非标准数值。
                "实例局部变换": json.dumps(
                    component.occurrence_locations,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "实例全局变换": json.dumps(
                    [
                        item.get("global_matrix_3x4") if item else None
                        for item in component.occurrence_locations
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "零件名称": component.part_name,
                "部件类别": component.category,
                # 任务书要求按 8 个部件大类归类；无法唯一归类时留空并保留复核标记。
                "部件大类": component.part_category_group or "",
                "大类证据强度": component.part_category_group_strength or "",
                "类别置信度": component.category_confidence,
                "类别来源": component.category_source,
                "类别证据": " | ".join(component.category_evidence),
                # 类别候选是列表，同样用紧凑 JSON 表达，避免中文顿号分隔歧义。
                "类别候选": json.dumps(
                    component.category_candidates,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "类别需要复核": component.category_review_required,
                "类别复核原因": " | ".join(component.category_review_reason_codes),
                # 材料/采购规格缺失时写空串而不是 None，避免 Excel 里出现 "None"。
                "材料": component.material or "",
                "材料来源": component.material_source,
                "数量": component.quantity,
                "采购规格": component.purchase_specification or "",
                "采购规格来源": component.purchase_specification_source,
                "备注": component.remarks,
                "X尺寸_mm": dimensions[0],
                "Y尺寸_mm": dimensions[1],
                "Z尺寸_mm": dimensions[2],
                "表面积_mm2": component.surface_area_mm2,
                "体积_mm3": component.volume_mm3,
                "Solid数": component.solid_count,
                "Shell数": component.shell_count,
                "Face数": component.face_count,
                "Edge数": component.edge_count,
                "Vertex数": component.vertex_count,
                "人工复核警告": " | ".join(component.warnings),
            }
        )
    return rows


def _write_xlsx(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """我用标准库写出最小、可复核且不依赖第三方包的 XLSX。

    我不用 openpyxl 是为了让"生成 BOM 表"这件事不引入新依赖：一个 xlsx
    本质上就是一个 zip 里装着几段 XML，只要成员齐全、引用正确，Excel 就能打开。
    """

    # 第一行是表头，后面是数据行；我统一按"行列表"处理，方便一次算维度。
    all_rows = [fieldnames] + [
        [row.get(fieldname) for fieldname in fieldnames] for row in rows
    ]
    # dimension 和 autoFilter 都需要知道右下角单元格，例如 "AH120"。
    last_cell = _cell_reference(len(all_rows), len(fieldnames))
    sheet_rows = "\n".join(
        _xlsx_row(row_index, values, header=row_index == 1)
        for row_index, values in enumerate(all_rows, start=1)
    )
    # 列宽我按表头字数估算：中文表头偏长，所以最小 12、最大 42 字符。
    columns = "\n".join(
        f'<col min="{index}" max="{index}" width="{_column_width(name):.1f}" customWidth="1"/>'
        for index, name in enumerate(fieldnames, start=1)
    )
    worksheet = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{_XLSX_NS}" xmlns:r="{_XLSX_REL_NS}">
  <dimension ref="A1:{last_cell}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>{columns}</cols>
  <sheetData>{sheet_rows}</sheetData>
  <autoFilter ref="A1:{last_cell}"/>
</worksheet>"""
    workbook = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="{_XLSX_NS}" xmlns:r="{_XLSX_REL_NS}">
  <sheets><sheet name="BOM" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    with atomic_path(path) as temporary:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            # 这五个成员是最小可用 xlsx 的必需集合，缺任何一个 Excel 都打不开。
            archive.writestr("[Content_Types].xml", _content_types_xml())
            archive.writestr("_rels/.rels", _package_relationships_xml())
            archive.writestr("xl/workbook.xml", workbook)
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                _workbook_relationships_xml(),
            )
            archive.writestr("xl/worksheets/sheet1.xml", worksheet)
            archive.writestr("xl/styles.xml", _styles_xml())


def _xlsx_row(row_index: int, values: list[Any], *, header: bool) -> str:
    """我把一行值拼成 <row> 片段，列号从 1 开始。"""

    cells = "".join(
        _xlsx_cell(_cell_reference(row_index, column), value, header=header)
        for column, value in enumerate(values, start=1)
    )
    return f'<row r="{row_index}">{cells}</row>'


def _xlsx_cell(reference: str, value: Any, *, header: bool) -> str:
    """我把一个单元格写成 OOXML 片段。

    分三种情况：None 写成真正的空单元格（不是 "None" 字符串）；bool 写成 0/1
    并保留数字类型；有限数值写成数字单元格；其余一律走 inlineStr 文本单元格，
    并做 XML 转义。header=True 时挂样式 s="1"（加粗、居中、浅蓝底）。
    """

    if value is None:
        return ""
    # bool 必须在 int 之前判断，因为 Python 里 bool 是 int 的子类。
    style = ' s="1"' if header else ""
    if isinstance(value, bool):
        return f'<c r="{reference}"{style}><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return f'<c r="{reference}"{style}><v>{value}</v></c>'
    text = escape(_xml_safe_text(str(value)), quote=False)
    return (
        f'<c r="{reference}" t="inlineStr"{style}>'
        f'<is><t xml:space="preserve">{text}</t></is></c>'
    )


def _xml_safe_text(value: str) -> str:
    """我把 XML 1.0 不允许的控制字符换成空格，避免写出打不开的 XLSX。

    XML 1.0 只允许 #x9、#xA、#xD 这三个控制字符。STEP 字符串、主数据或文件名
    一旦带进其它 C0 控制符，html.escape 不会处理它们，工作表 XML 就会不合法，
    Excel 直接拒绝打开。这里只做"换成空格"，保留列宽和可读性，不做静默删除。
    """

    return "".join(
        char if _is_xml_char(ord(char)) else " " for char in value
    )


def _is_xml_char(code_point: int) -> bool:
    """我按 XML 1.0 的 Char 产生式判断一个码点是否合法。

    我为什么把 0xD7FF 以下和 0xE000 以上分开写：中间那一段（U+D800
    到 U+DFFF）是 UTF-16 的代理区，XML 1.0 明确禁止它出现在字符数据里。
    Python 字符串里单个字符不会落在这个区间，但从外部读进来的
    损坏数据可能构造出它，不排除就会写出 Excel 打不开的表。
    """

    return (
        code_point in (0x9, 0xA, 0xD)
        or 0x20 <= code_point <= 0xD7FF
        or 0xE000 <= code_point <= 0xFFFD
        or 0x10000 <= code_point <= 0x10FFFF
    )


def _cell_reference(row: int, column: int) -> str:
    """我把 (行号, 列号) 转成 Excel 地址，例如 (3, 28) -> "AB3"。

    列号是 1-based 的"26 进制但用 A~Z 表示"，且没有 0 位，所以要先减 1 再取模。

    我为什么必须自己实现而不是用 f"A{row}"：BOM 表有二十多列，
    超过 Z 列（第 26 列）之后就是 AA、AB，不实现这段换算就只能写到
    26 列为止，后面的字段会被丢掉。而甲方给的 BOM 表头正好超过 26 列。
    """

    letters: list[str] = []
    while column:
        column, remainder = divmod(column - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters)) + str(row)


def _column_width(name: str) -> float:
    """我按表头字数估算列宽，并限制在 12~42 之间，避免超宽列挤掉别的内容。

    我为什么按**字符数**乘系数而不是按像素算：中文字宽约为数字的两倍，
    而表头是中文；按字符数乘 1.5 对中文表头接近真实宽度，不需要引入
    字体度量。系数 1.5 是我打开几份 BOM 表对出来的经验值。

    我为什么设 12 的下限：过窄的列会把中文表头截成"#"，甲方在 Excel
    里看到的是一排井号，根本不知道这一列是什么。
    """

    return min(max(len(name) * 1.5 + 2.0, 12.0), 42.0)


def _content_types_xml() -> str:
    """我返回 [Content_Types].xml，声明包内每个成员的内容类型。"""

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{_CONTENT_TYPES_NS}">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""


def _package_relationships_xml() -> str:
    """我返回包级关系 _rels/.rels，把包入口指向 xl/workbook.xml。"""

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{_PACKAGE_REL_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""


def _workbook_relationships_xml() -> str:
    """我返回 workbook 的关系表，声明 sheet1 和 styles 两个部件。"""

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{_PACKAGE_REL_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _styles_xml() -> str:
    """我返回最小样式表：字体 0 常规、字体 1 加粗；填充 1 是表头底色。

    cellXfs 的索引 1 就是 _xlsx_cell 里 header 用的 s="1"，
    改这里的顺序会直接改变表头外观，所以两者必须一起改。
    """

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="{_XLSX_NS}">
  <numFmts count="0"/>
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
    <xf numFmtId="0" fontId="1" fillId="1" borderId="0" applyAlignment="1"><alignment horizontal="center" wrapText="1"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""
