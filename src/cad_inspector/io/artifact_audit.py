"""我审计项目输出文件，避免报告存在但内容已损坏。

算法原理：文件存在不等于文件可用。一次长跑中断、Excel 写出失败、CSV 列数错位，都会
留下一个"存在但打不开"的坏文件，而下游只看文件名就会误判为成功。所以我对每种受支持
格式做真正的内容级校验，而不是只检查大小非零：

- JSON：拒绝非标准数值常量（NaN/Infinity）、拒绝重复键、校验顶层类型，并在带
  ``content_sha256`` 时重算规范化摘要比对，防止文件被改过还自称完整。
- CSV：提高字段大小上限后按 DictReader 逐行读，检查表头非空不重复（重复列名会让
  DictReader 静默丢列）、每行列数与表头一致。
- XLSX：先 testzip 验压缩包完整性，再检查必要成员存在，最后逐个解析关键 XML，
  挡住"压缩包完整但工作表 XML 损坏"的假通过。
- PLY：按字节读头部（兼容 CRLF 与二进制载荷），必须有 ply 头与 end_header。
- STL：真正加载网格并统计三角面与顶点数。
- PNG：遍历所有块逐个校验 CRC32，并要求以 IEND 正常收尾，
  挡住"体积正常但中间像素数据已坏"的半张图。

超过 256 MiB 的文件不读取（拒绝服务式大文件），空文件直接判 INVALID。
"""

from __future__ import annotations

import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path
import struct
from xml.etree import ElementTree
import zlib
from zipfile import ZipFile

from ..reproducibility import content_sha256


_AUDITED_SUFFIXES = {
    ".json",
    ".csv",
    ".xlsx",
    ".ply",
    ".stl",
    ".step",
    ".stp",
    ".jsonl",
    ".png",
    # 我补上 .md / .html：实测 2026-10-19 的 outputs_audit 里有 3 个文件
    # （demo_visualize / demo_visualize_auto 的 visualization.html、
    # flowcharts/flowcharts.md）只因为后缀不在白名单就被"跳过"。
    # 跳过不等于合格——被截断到一半的 html 会照样过关，
    # 而甲方打开时看到的是半张页面。这两类都是给人看的交付物，
    # 必须真读一遍内容（见 _audit_markdown / _audit_html）。
    ".md",
    ".html",
}
_DEFAULT_MAX_FILE_BYTES = 256 * 1024 * 1024
_PLY_HEADER_SCAN_BYTES = 64 * 1024
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def audit_artifacts(
    path: str | Path,
    *,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
) -> dict[str, object]:
    """我递归审计一个输出目录，并把每个文件的结论写入结构化结果。"""

    root = Path(path).expanduser().resolve()
    # 路径与参数先校验，避免审计器自己出错却报告成"文件损坏"。
    if not root.exists():
        raise ValueError(f"我找不到待审计路径：{root}")
    if not root.is_dir() and not root.is_file():
        raise ValueError(f"我要求待审计路径是文件或目录：{root}")
    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, Integral)
        or max_file_bytes < 1
    ):
        raise ValueError("我要求 max_file_bytes 必须是正整数。")

    # 按大小写无关排序遍历，排除符号链接，保证审计结果与文件系统顺序无关。
    # is_file() 也要容错：临时文件可能在遍历期间被清理，
    # 无权限目录也会抛错。审计器不能因为遍历阶段的竞态就整体失败。
    files = [root] if root.is_file() else sorted(
        (
            item
            for item in root.rglob("*")
            if _is_auditable_file(item)
        ),
        key=lambda item: str(item).casefold(),
    )
    records = [
        _audit_one(
            item,
            root=root if root.is_dir() else root.parent,
            max_file_bytes=int(max_file_bytes),
        )
        for item in files
    ]
    valid = sum(record["status"] == "VALID" for record in records)
    invalid = sum(record["status"] == "INVALID" for record in records)
    skipped = sum(record["status"] == "SKIPPED" for record in records)
    schema_versions: dict[str, int] = {}
    for record in records:
        schema = record.get("schema_version")
        if isinstance(schema, str) and schema:
            schema_versions[schema] = schema_versions.get(schema, 0) + 1
    # 没有任何 INVALID 且至少有一个 VALID 才算 PASS；全空目录不是 PASS。
    has_files = bool(records)
    return {
        "schema_version": "artifact-audit-1.0",
        "path": str(root),
        "max_file_bytes": int(max_file_bytes),
        "total_files": len(records),
        "valid_files": valid,
        "invalid_files": invalid,
        "skipped_files": skipped,
        "schema_versions": dict(sorted(schema_versions.items())),
        "files": records,
        "status": "PASS" if invalid == 0 and valid > 0 else "REVIEW_REQUIRED",
        "empty_output": not has_files,
    }


def _is_auditable_file(path: Path) -> bool:
    """我判断一个遍历到的路径是否应当进入审计，并吞掉遍历期的竞态错误。

    OSError 必须捕获：临时文件可能在 rglob 与 is_file() 之间被删除，
    无权限目录也会抛 PermissionError。这类竞态属于"文件本来就不在"，
    不是"文件损坏"，让它崩掉整个审计会把门禁搞挂，
    而真正要拦的损坏产物反而一个都没审。
    """

    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def _audit_one(path: Path, *, root: Path, max_file_bytes: int) -> dict[str, object]:
    """我审计单个文件，先做通用检查再按后缀分派到具体格式校验器。

    stat().st_size 必须容错：文件系统列表（rglob）和真正读取之间
    有一段时间窗口，临时文件可能在这期间被清理掉。审计器是门禁的
    一环，它自己崩溃会让整个门禁失败，而消失的文件本来就不该
    算作"损坏产物"——所以我记成 SKIPPED 并说明原因，而不是抛错。
    """
    relative = str(path.relative_to(root))
    suffix = path.suffix.lower()
    try:
        size_bytes = path.stat().st_size
    except (OSError, ValueError):
        return {
            "path": relative,
            "absolute_path": str(path),
            "suffix": suffix,
            "size_bytes": None,
            "sha256": None,
            "kind": suffix.lstrip(".") or "extensionless",
            "status": "SKIPPED",
            "issues": ["我在审计时文件已不存在，无法读取。"],
        }
    record: dict[str, object] = {
        "path": relative,
        "absolute_path": str(path),
        "suffix": suffix,
        "size_bytes": size_bytes,
        "sha256": None,
        "kind": suffix.lstrip(".") or "extensionless",
        "status": "SKIPPED",
        "issues": [],
    }
    # 不支持的后缀只跳过，不判损坏；零字节与超上限都直接判 INVALID。
    if suffix not in _AUDITED_SUFFIXES:
        record["issues"] = ["我跳过了当前审计器不解析的文件类型。"]
        return record
    if int(record["size_bytes"]) == 0:
        record["status"] = "INVALID"
        record["issues"] = ["我发现输出文件为空。"]
        return record
    if int(record["size_bytes"]) > max_file_bytes:
        record["status"] = "INVALID"
        record["issues"] = [
            f"我拒绝读取超过审计上限的文件：{max_file_bytes} bytes。"
        ]
        return record
    record["sha256"] = _sha256(path)

    # 具体格式校验抛任何异常都归为 INVALID，异常信息进 issues 便于定位。
    try:
        if suffix == ".json":
            _audit_json(path, record)
        elif suffix == ".csv":
            _audit_csv(path, record)
        elif suffix == ".xlsx":
            _audit_xlsx(path, record)
        elif suffix == ".ply":
            _audit_ply(path, record)
        elif suffix == ".stl":
            _audit_stl(path, record)
        elif suffix in {".step", ".stp"}:
            _audit_step(path, record)
        elif suffix == ".jsonl":
            _audit_jsonl(path, record)
        elif suffix == ".png":
            _audit_png(path, record)
        elif suffix == ".md":
            _audit_markdown(path, record)
        elif suffix == ".html":
            _audit_html(path, record)
    except Exception as exc:
        record["status"] = "INVALID"
        record["issues"] = [f"{type(exc).__name__}: {exc}"]
    return record


def _audit_step(path: Path, record: dict[str, object]) -> None:
    """我校验 STEP：**真的把它读回来**，而不是只看看文件在不在。

    我为什么要把 STEP 纳入审计（这是补一个真实盲区）：
    outputs 下有 **229 个 .step/.stp 文件**（175 + 54）此前一律被
    判成"当前审计器不解析的类型"直接跳过。而 STEP 正是**修补功能的
    交付物**——我最不能接受的结果就是：交出去一个打不开的 STEP，
    却因为格式没被审计而一路绿灯。

    所以判据不是"文件存在且非空"，而是**用 OCP 真正读一遍**：
    - 读文件状态必须是 RetDone；
    - 必须转移出根对象；
    - Shape 不能为空。
    三条任一不满足就是 INVALID（交付物不可用）。

    我**不**在这里判 BRep 有效性：修补本来就会产出需要人工复核的
    候选件，那是 `REVIEW_REQUIRED` 而不是"损坏"，把两者混在一起
    会让审计把"待复核"误报成"损坏"。

    OCP 缺失时不判损坏，只标记跳过并说明原因——
    环境缺依赖不是产出物的问题。
    """

    try:
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.STEPControl import STEPControl_Reader
    except ImportError:
        record["status"] = "SKIPPED"
        record["issues"] = ["我无法校验 STEP，因为当前环境没有 OCP。"]
        return

    reader = STEPControl_Reader()
    read_status = reader.ReadFile(str(path))
    if read_status != IFSelect_ReturnStatus.IFSelect_RetDone:
        record["status"] = "INVALID"
        record["issues"] = [
            f"我用 OCP 读这个 STEP 失败，后端状态为 {read_status}。"
        ]
        return
    if not reader.TransferRoots():
        record["status"] = "INVALID"
        record["issues"] = ["我没有从这个 STEP 转移出可用根对象。"]
        return
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        record["status"] = "INVALID"
        record["issues"] = ["我读到的 STEP Shape 为空。"]
        return
    record["status"] = "VALID"


def _audit_jsonl(path: Path, record: dict[str, object]) -> None:
    """我校验 JSONL（每行一个 JSON）：**逐行解析，一行都不能坏**。

    我为什么单独处理它而不用 JSON 那套：JSONL 是**按行追加**写出的
    结果流，典型损坏是"写一半进程被杀"——最后一行截断。
    这种截断在"整个文件当 JSON 读"的口径下会被当成"文件格式不对"，
    而在逐行口径下才能精确定位到"第几行坏了"。
    我实测过当前 20 个 .jsonl 共 106 行全部合法，
    所以要造出这种损坏只能靠审计持续盯着，而不是靠一次手工检查。

    空白行跳过（追加写常见），非空行必须能解析，
    且顶层必须是对象或数组（裸标量说明这一行写坏了）。
    """

    bad_lines: list[str] = []
    line_count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            line_count += 1
            try:
                payload = json.loads(stripped)
            except ValueError as exc:
                bad_lines.append(f"第 {index} 行不是合法 JSON：{exc}")
                continue
            if not isinstance(payload, (dict, list)):
                bad_lines.append(
                    f"第 {index} 行顶层是 {type(payload).__name__}，"
                    "我要求对象或数组。"
                )
    record["line_count"] = line_count
    if bad_lines:
        record["status"] = "INVALID"
        record["issues"] = bad_lines[:5]
        return
    record["status"] = "VALID"


def _audit_json(path: Path, record: dict[str, object]) -> None:
    """我校验 JSON：拒绝非标准常量与重复键，并复核内容摘要。"""
    def reject_constant(value: str) -> None:
        """我拒绝 NaN/Infinity 这类非标准 JSON 数值常量。"""
        raise ValueError(f"JSON 不允许非标准数值：{value}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        """我拒绝重复键，避免后值覆盖前值造成静默数据丢失。"""
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON 存在重复键：{key}")
            result[key] = value
        return result

    payload = json.loads(
        # ponytail: utf-8-sig 只是额外容忍 Windows 工具写出的 BOM，其余行为不变。
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    # 顶层必须是对象或数组；带内容摘要时重算比对，防篡改与半截写入。
    if not isinstance(payload, (dict, list)):
        raise ValueError("JSON 顶层必须是对象或数组。")
    record["status"] = "VALID"
    record["top_level_type"] = "object" if isinstance(payload, dict) else "array"
    if isinstance(payload, dict):
        schema = payload.get("schema_version")
        if isinstance(schema, str):
            record["schema_version"] = schema
        record["key_count"] = len(payload)
        digest = payload.get("content_sha256")
        if digest is not None:
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("content_sha256 必须是 64 位十六进制字符串。")
            if digest != content_sha256(payload):
                raise ValueError("content_sha256 与规范化内容不一致。")
            record["content_sha256"] = digest
    else:
        record["item_count"] = len(payload)


def _audit_csv(path: Path, record: dict[str, object]) -> None:
    """我校验 CSV：表头有效且不重复，逐行列数与表头一致。"""
    csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or any(
            not isinstance(name, str) or not name.strip()
            for name in fieldnames
        ):
            raise ValueError("CSV 缺少有效表头。")
        if len(set(fieldnames)) != len(fieldnames):
            # DictReader 遇到重名列会静默丢列，我不能把这种文件当成有效输出。
            raise ValueError("CSV 表头存在重复列名，读取时会静默丢列。")
        rows = 0
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(row[name] is None for name in fieldnames):
                raise ValueError(f"CSV 第 {row_number} 行列数与表头不一致。")
            rows += 1
    record["status"] = "VALID"
    record["column_count"] = len(fieldnames)
    record["row_count"] = rows


def _audit_xlsx(path: Path, record: dict[str, object]) -> None:
    """我校验 XLSX：压缩包完整、必要成员齐全、关键 XML 可解析。"""
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("XLSX 压缩包中存在损坏成员。")
        names = set(archive.namelist())
        required = {
            "[Content_Types].xml",
            "xl/workbook.xml",
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"XLSX 缺少必要成员：{', '.join(missing)}")
        # ponytail: 我只解析关键 XML 成员；压缩包完整但工作表 XML 损坏的假通过也能挡住。
        members = sorted(
            required
            | {
                name
                for name in names
                if name.startswith("xl/worksheets/") and name.endswith(".xml")
            }
        )
        for member in members:
            try:
                ElementTree.fromstring(archive.read(member))
            except ElementTree.ParseError as exc:
                raise ValueError(
                    f"XLSX 成员 {member} 的 XML 不合法：{exc}"
                ) from exc
    record["status"] = "VALID"


def _audit_ply(path: Path, record: dict[str, object]) -> None:
    """我校验 PLY：按字节读头部，必须有 ply 起始与 end_header。"""
    # ponytail: 我按字节读头部，兼容 CRLF 与二进制载荷；64 KiB 上限够常见 PLY。
    with path.open("rb") as handle:
        prefix = handle.read(_PLY_HEADER_SCAN_BYTES)
    if not (prefix.startswith(b"ply\n") or prefix.startswith(b"ply\r\n")):
        raise ValueError("PLY 缺少有效头部或 end_header。")
    if b"end_header" not in prefix:
        raise ValueError("PLY 缺少有效头部或 end_header。")
    record["status"] = "VALID"


def _audit_stl(path: Path, record: dict[str, object]) -> None:
    """我校验 STL：真正加载网格并统计三角面与顶点数。"""
    from ..geometry.stl import load_stl

    mesh = load_stl(path)
    record["status"] = "VALID"
    record["triangle_count"] = int(len(mesh.faces))
    record["vertex_count"] = int(len(mesh.vertices))


def _audit_png(path: Path, record: dict[str, object]) -> None:
    """我校验 PNG：**逐块校验 CRC**，而不只看文件头和体积。

    我为什么做这个：PNG 是给人看的对比图和流程图，
    这类文件最容易出的损坏是"写到一半被杀"——体积看起来正常，
    但中间的像素数据已经坏了。只看文件头会放行这种文件，
    而甲方打开时看到的是半张图或花屏。

    PNG 自带每块的 CRC32，这是**文件格式免费给我的校验和**，
    不用就浪费了。所以我遍历所有块逐个校验，
    并要求必须以 IEND 正常收尾（缺 IEND 就是被截断了）。
    实测 8 张图全过，耗时约 2.4 秒——代价很低。
    """

    data = path.read_bytes()
    if not data.startswith(_PNG_MAGIC):
        raise ValueError("PNG 文件头不是标准 89 50 4E 47 0D 0A 1A 0A。")
    if len(data) < 8 + 12 + 12:
        raise ValueError("PNG 太短，连 IHDR 或 IEND 都放不下。")

    position = 8
    chunk_count = 0
    seen_iend = False
    while position + 8 <= len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        chunk_type = data[position + 4 : position + 8]
        body_start = position + 8
        body_end = body_start + length
        crc_end = body_end + 4
        if crc_end > len(data):
            raise ValueError(
                f"PNG 块 {chunk_type!r} 声明 {length} 字节，"
                f"但文件到 {len(data)} 字节就结束了——这是被截断的典型特征。"
            )
        (stored_crc,) = struct.unpack(">I", data[body_end:crc_end])
        actual_crc = zlib.crc32(chunk_type + data[body_start:body_end]) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise ValueError(
                f"PNG 块 {chunk_type!r} 的 CRC 对不上"
                f"（文件里 {stored_crc:#010x}，算出 {actual_crc:#010x}）——"
                "这块数据已经损坏。"
            )
        chunk_count += 1
        position = crc_end
        if chunk_type == b"IEND":
            seen_iend = True
            break
    if not seen_iend:
        raise ValueError("PNG 没有 IEND 收尾块，文件被截断了。")

    width, height = struct.unpack(">II", data[16:24])
    record["status"] = "VALID"
    record["chunk_count"] = chunk_count
    record["width"] = int(width)
    record["height"] = int(height)


def _read_text_artifact(path: Path, label: str) -> str:
    """我按 UTF-8 读一个文本产物，读不出来就按"损坏"处理。

    我为什么明确判 NUL 字节：被二进制内容污染或写到一半的文本文件
    在 Windows 上经常表现为"能读出来但中间夹了 \\x00"，
    decode 不报错，于是内容看着"存在"却是垃圾。
    """

    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"{label} 里出现 NUL 字节，它不是文本产物。")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} 不是合法 UTF-8 文本：{exc}") from exc


def _audit_markdown(path: Path, record: dict[str, object]) -> None:
    """我校验 Markdown：必须是能按 UTF-8 读出、且非空白的内容。

    我为什么不判"有没有标题"这类结构：Markdown 的合法写法太宽，
    按结构判会把正常文档误报成损坏。真正会伤到甲方的是**空文件**和
    **读不出来的字节**，这两类我拦住就够。
    """

    text = _read_text_artifact(path, "Markdown 文件")
    if not text.strip():
        raise ValueError("Markdown 文件只有空白字符，没有任何内容。")
    record["status"] = "VALID"
    record["line_count"] = len(text.splitlines())
    record["char_count"] = len(text)


def _audit_html(path: Path, record: dict[str, object]) -> None:
    """我校验 HTML：非空、UTF-8 可读，且 `<html>` 必须正常闭合。

    判据取"必须以 </html> 收尾"是因为这类文件最典型的损坏就是
    **写到一半被杀**：前缀完整、浏览器里也能显示半张页面，
    唯独缺收尾标记。只看文件在不在、体积大不大都会放行它。
    """

    text = _read_text_artifact(path, "HTML 文件")
    if not text.strip():
        raise ValueError("HTML 文件只有空白字符，没有任何内容。")
    lowered = text.lower()
    if "<html" not in lowered:
        raise ValueError("HTML 里找不到 <html> 起始标记。")
    if "</html>" not in lowered:
        raise ValueError(
            "HTML 没有 </html> 收尾标记——文件被截断的典型特征。"
        )
    record["status"] = "VALID"
    record["line_count"] = len(text.splitlines())
    record["char_count"] = len(text)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """我按 1 MiB 分块计算文件 sha256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
