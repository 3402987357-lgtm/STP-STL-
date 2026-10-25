from __future__ import annotations

from pathlib import Path

import pytest

from cad_inspector.domain.models import DetectedFormat
from cad_inspector.io.file_validation import validate_model_pair
from cad_inspector.io.input_validator import InputValidationError, validate_input_path


def test我方生成文件元数据(tmp_path: Path) -> None:
    source = tmp_path / "中文 模型.stp"
    source.write_text("ISO-10303-21;", encoding="utf-8")

    metadata = validate_input_path(source)

    assert metadata.detected_format is DetectedFormat.STEP
    assert metadata.name == "中文 模型.stp"
    assert len(metadata.sha256) == 64
    assert metadata.size_bytes > 0


def test我方拒绝不存在文件(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="不存在"):
        validate_input_path(tmp_path / "missing.step")


def test我方拒绝未知扩展名(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"data")

    with pytest.raises(InputValidationError, match="扩展名"):
        validate_input_path(source)


def test我方拒绝空文件和非法大小参数(tmp_path: Path) -> None:
    source = tmp_path / "empty.stp"
    source.write_bytes(b"")

    with pytest.raises(InputValidationError, match="为空"):
        validate_input_path(source)
    with pytest.raises(InputValidationError, match="正整数"):
        validate_input_path(source, max_size_bytes=0)
    with pytest.raises(InputValidationError, match="正整数"):
        validate_input_path(source, max_size_bytes=True)


def test我方模型对比拒绝超出资源上限的输入(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.stl"
    reference = tmp_path / "reference.stl"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")

    with pytest.raises(InputValidationError, match="大小超过限制"):
        validate_model_pair(candidate, reference, max_size_bytes=1)


def test我方模型对比拒绝同一个文件当候选和参考(tmp_path: Path) -> None:
    source = tmp_path / "model.stl"
    source.write_bytes(b"solid\nendsolid\n")

    with pytest.raises(InputValidationError, match="不是同一个文件"):
        validate_model_pair(source, source)

    # 同一份内容写成两个不同路径仍然允许，只有同一个文件才拒绝。
    other = tmp_path / "copy.stl"
    other.write_bytes(source.read_bytes())
    pair = validate_model_pair(source, other)
    assert pair["candidate"] != pair["reference"]
