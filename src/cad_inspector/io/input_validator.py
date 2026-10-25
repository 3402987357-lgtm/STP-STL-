"""我方的输入文件校验。

我方在任何 CAD 几何操作之前先完成路径、格式、大小和哈希校验。
这样可以把“文件不存在”“格式暂不支持”和“几何解析失败”分开，
避免后续界面把不同原因混成一句模糊错误。
"""

from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path

from cad_inspector.domain.models import DetectedFormat, FileMetadata
from cad_inspector.errors import InputValidationError


_FORMAT_BY_SUFFIX = {
    ".step": DetectedFormat.STEP,
    ".stp": DetectedFormat.STEP,
    ".xt": DetectedFormat.PARASOLID_XT,
    ".prt": DetectedFormat.SIEMENS_NX_PRT,
    ".stl": DetectedFormat.STL,
}


def format_from_path(path: Path) -> DetectedFormat:
    """我方只把扩展名作为初步分类，不把它当成内容验证。

    扩展名只是命名约定，改个后缀不会改变文件内容。所以我用它做
    **路由**（决定走哪个解析器），不做**校验**。真正的格式校验
    只能由解析器在读到内容后完成——一个声称是 .stp 的文件完全
    可能是 HTML 错误页，这一点必须在解析阶段暴露，不能在这里假装确认。

    不认识的扩展名返回 UNKNOWN 而不是抛错，让调用方决定是拒绝
    还是继续尝试，避免这里擅自替上层做策略决定。
    """

    return _FORMAT_BY_SUFFIX.get(path.suffix.lower(), DetectedFormat.UNKNOWN)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """我按块计算文件哈希，避免把几十 MB 的模具备件整读进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_input_path(path: str | Path, max_size_bytes: int | None = None) -> FileMetadata:
    """我方只读校验并生成文件元数据。

    我把不同失败原因分成独立分支并给出各自的错误文案：不存在、
    不是文件、扩展名不识别、空文件、超限。理由很实际——
    "文件不存在"和"格式不支持"是用户两种不同的下一步动作，
    混成一句"输入无效"会让用户不知道该去确认路径还是去转格式。

    顺序也有讲究：先校验参数本身（max_size_bytes 必须是正整数），
    再校验文件系统，最后算哈希。哈希是这套校验里唯一有 I/O 成本的
    操作，放在最后可以让廉价检查先失败，避免为一个不存在的路径
    去读整个文件。

    resolve() 之后再做后续判断：相对路径和符号链接在解析前后
    行为不同，统一解析能保证报错里的路径和实际读取的是同一个。
    """

    if (
        max_size_bytes is not None
        and (
            isinstance(max_size_bytes, bool)
            or not isinstance(max_size_bytes, Integral)
            or max_size_bytes < 1
        )
    ):
        raise InputValidationError("我要求 max_size_bytes 必须是正整数。")
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise InputValidationError(f"输入文件不存在: {candidate}")
    if not candidate.is_file():
        raise InputValidationError(f"输入路径不是文件: {candidate}")

    resolved = candidate.resolve()
    file_format = format_from_path(resolved)
    if file_format is DetectedFormat.UNKNOWN:
        raise InputValidationError(f"暂不识别的文件扩展名: {resolved.suffix or '<无扩展名>'}")

    size_bytes = resolved.stat().st_size
    if size_bytes == 0:
        raise InputValidationError(f"输入文件为空: {resolved}")
    if max_size_bytes is not None and size_bytes > max_size_bytes:
        raise InputValidationError(
            f"文件大小超过限制: {size_bytes} > {max_size_bytes} bytes"
        )

    return FileMetadata(
        path=str(resolved),
        name=resolved.name,
        extension=resolved.suffix.lower(),
        size_bytes=size_bytes,
        sha256=_sha256(resolved),
        detected_format=file_format,
    )
