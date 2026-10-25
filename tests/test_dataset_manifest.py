from __future__ import annotations

from pathlib import Path

from cad_inspector.io.dataset_manifest import build_manifest


def test我方登记目录中的支持格式(tmp_path: Path) -> None:
    (tmp_path / "a.stp").write_text(
        "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    (tmp_path / "b.prt").write_bytes(b"licensed adapter required")
    (tmp_path / "ignore.txt").write_text("我不应进入 CAD 清单", encoding="utf-8")

    manifest = build_manifest(tmp_path)

    assert len(manifest.records) == 2
    assert {record.extension for record in manifest.records} == {".stp", ".prt"}
    assert all(record.sample_id.startswith("sample-") for record in manifest.records)
