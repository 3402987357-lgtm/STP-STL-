from __future__ import annotations

import json
from pathlib import Path

from cad_inspector.io.batch_probe import expand_inputs, run_batch, write_batch_reports


def test我方批量清单不覆盖同名文件(tmp_path: Path) -> None:
    first = tmp_path / "a" / "同名.step"
    second = tmp_path / "b" / "同名.step"
    first.parent.mkdir()
    second.parent.mkdir()
    content = (
        "ISO-10303-21; HEADER; "
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN')); ENDSEC; "
        "DATA; #1=PRODUCT('p','','',()); ENDSEC; END-ISO-10303-21;"
    )
    first.write_text(content, encoding="utf-8")
    second.write_text(content + "\n", encoding="utf-8")

    paths = expand_inputs([tmp_path])
    batch = run_batch(paths)
    json_path, csv_path = write_batch_reports(batch, tmp_path / "out")

    assert len(paths) == 2
    assert batch.total == 2
    assert json_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["total"] == 2
