"""我为批量检测区域写回归测试。

我不在测试里跑真实大件：单个件要几十秒，几十个件的测试会比整条流水线
还慢。我改为直接测"批量编排"这一层的四个关键契约——配对、单点隔离、
未配对留痕、台账序列化——真正的检测由 ``inspect_pair`` 自己的测试覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cad_inspector.io import batch_inspect
from cad_inspector.io.batch_inspect import (
    BatchInspectReport,
    BatchItemResult,
    _first_int,
    pair_by_stem,
    run_batch_inspect,
    write_batch_inspect_reports,
)


def test_pair_by_stem_matches_names() -> None:
    """我按文件名主干配对，而不是按下标——下标配错会让结论整个反过来。"""

    candidates = [Path("a/1.stl"), Path("a/2.stl"), Path("a/10.stl")]
    references = [Path("b/10.stp"), Path("b/2.stp"), Path("b/1.stp")]
    pairs = pair_by_stem(candidates, references)
    matched = {candidate.name: (ref.name if ref else None) for candidate, ref in pairs}
    assert matched == {"1.stl": "1.stp", "2.stl": "2.stp", "10.stl": "10.stp"}


def test_pair_by_stem_keeps_unpaired() -> None:
    """找不到参考件的候选件必须留在台账里，不能被静默跳过。"""

    pairs = pair_by_stem([Path("a/9.stl")], [Path("b/1.stp")])
    assert pairs == [(Path("a/9.stl"), None)]


def test_first_int_rejects_bool() -> None:
    """bool 是 int 的子类，不排除会把 True 当成计数 1。"""

    assert _first_int(None, True, 5) == 5
    assert _first_int(True, 7) == 7
    assert _first_int(None, None) is None
    assert _first_int("12") == 12
    assert _first_int("abc") is None


def test_run_batch_inspect_reports_unpaired(tmp_path: Path) -> None:
    """缺参考件的件记为 UNPAIRED 且带说明，不进检测也不消失。"""

    candidate = tmp_path / "solo.stl"
    candidate.write_bytes(b"solid x\nendsolid x\n")
    report = run_batch_inspect([candidate], [], tmp_path / "out")
    assert report.total == 1
    assert report.unpaired == 1
    assert report.ok == 0
    assert "参考件" in report.items[0].warnings[0]


def test_single_failure_does_not_stop_the_batch(tmp_path: Path, monkeypatch) -> None:
    """一个件炸掉只影响它自己——这是批量编排最重要的契约。"""

    good = tmp_path / "good.stl"
    bad = tmp_path / "bad.stl"
    good.write_bytes(b"solid x\nendsolid x\n")
    bad.write_bytes(b"solid x\nendsolid x\n")

    def fake_inspect_pair(candidate_path, reference_path, output_dir, config=None):
        if Path(candidate_path).stem == "bad":
            raise RuntimeError("我故意让这一对失败。")

        class FakeResult:
            global_metrics = {"missing_region_count": 2, "extra_region_count": 1}
            warnings = ["我的假告警"]

        return FakeResult()

    monkeypatch.setattr(batch_inspect, "inspect_pair", fake_inspect_pair)
    report = run_batch_inspect(
        [good, bad],
        [tmp_path / "good.stp", tmp_path / "bad.stp"],
        tmp_path / "out",
    )
    assert report.total == 2
    assert report.ok == 1
    assert report.failed == 1
    statuses = {Path(item.candidate).stem: item.status for item in report.items}
    assert statuses == {"good": "OK", "bad": "FAILED"}
    good_item = next(item for item in report.items if item.status == "OK")
    assert good_item.missing_count == 2
    assert good_item.extra_count == 1
    bad_item = next(item for item in report.items if item.status == "FAILED")
    assert "我故意让这一对失败。" in (bad_item.error or "")
    # traceback 必须进台账，否则事后无法定位哪个件、哪一行炸了。
    assert any("Traceback" in warning for warning in bad_item.warnings)


def test_max_items_limits_work(tmp_path: Path, monkeypatch) -> None:
    """max_items 用来先小数量试跑，避免参数错了还要等几十分钟。"""

    calls: list[str] = []

    def fake_inspect_pair(candidate_path, reference_path, output_dir, config=None):
        calls.append(Path(candidate_path).stem)

        class FakeResult:
            global_metrics = {"missing_region_count": 0, "extra_region_count": 0}
            warnings: list[str] = []

        return FakeResult()

    monkeypatch.setattr(batch_inspect, "inspect_pair", fake_inspect_pair)
    candidates = [tmp_path / f"{index}.stl" for index in range(5)]
    references = [tmp_path / f"{index}.stp" for index in range(5)]
    report = run_batch_inspect(
        candidates, references, tmp_path / "out", max_items=2
    )
    assert len(calls) == 2
    assert report.total == 2


def test_counts_fall_back_to_global_metrics(tmp_path: Path, monkeypatch) -> None:
    """区域计数在 global_metrics 里，不在顶层属性上。"""

    def fake_inspect_pair(candidate_path, reference_path, output_dir, config=None):
        class FakeResult:
            global_metrics = {"missing_region_count": 7, "extra_region_count": 3}
            warnings: list[str] = []

        return FakeResult()

    monkeypatch.setattr(batch_inspect, "inspect_pair", fake_inspect_pair)
    report = run_batch_inspect(
        [tmp_path / "a.stl"], [tmp_path / "a.stp"], tmp_path / "out"
    )
    assert report.items[0].missing_count == 7
    assert report.items[0].extra_count == 3


def test_missing_counts_are_flagged_not_silently_zero(
    tmp_path: Path, monkeypatch
) -> None:
    """取不到计数时我记 0 并留痕，绝不静默当成"没有缺失"。"""

    def fake_inspect_pair(candidate_path, reference_path, output_dir, config=None):
        class FakeResult:
            global_metrics: dict = {}
            warnings: list[str] = []

        return FakeResult()

    monkeypatch.setattr(batch_inspect, "inspect_pair", fake_inspect_pair)
    report = run_batch_inspect(
        [tmp_path / "a.stl"], [tmp_path / "a.stp"], tmp_path / "out"
    )
    item = report.items[0]
    assert item.missing_count == 0
    assert any("区域计数" in warning for warning in item.warnings)


def test_report_writes_json_and_csv(tmp_path: Path) -> None:
    """台账必须同时给出 JSON（含明细）与 CSV（平铺汇总）。"""

    report = BatchInspectReport(
        items=[
            BatchItemResult(
                candidate="a.stl",
                reference="a.stp",
                status="OK",
                missing_count=2,
                extra_count=1,
                elapsed_seconds=1.5,
            ),
            BatchItemResult(candidate="b.stl", reference=None, status="UNPAIRED"),
        ]
    )
    paths = write_batch_inspect_reports(report, tmp_path)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "batch-inspect-1.0"
    assert payload["total"] == 2
    assert payload["ok"] == 1
    assert payload["unpaired"] == 1
    # CSV 行数 = 表头 + 2 条记录
    rows = paths["csv"].read_text(encoding="utf-8-sig").strip().splitlines()
    assert len(rows) == 3
    assert rows[0].startswith("candidate,reference,status")


def test_report_summary_properties() -> None:
    """汇总计数必须与明细一致。"""

    report = BatchInspectReport(
        items=[
            BatchItemResult("a", "a", "OK"),
            BatchItemResult("b", "b", "OK"),
            BatchItemResult("c", "c", "FAILED"),
            BatchItemResult("d", None, "UNPAIRED"),
        ]
    )
    assert report.total == 4
    assert report.ok == 2
    assert report.failed == 1
    assert report.unpaired == 1


def test_cli_has_batch_inspect_command() -> None:
    """CLI 必须暴露这个子命令，否则这一层没法被调用。"""

    from cad_inspector.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["batch-inspect", "a.stl", "--references", "a.stp", "--max-items", "1"]
    )
    assert args.command == "batch-inspect"
    assert args.max_items == 1
    assert args.references == ["a.stp"]


def test_cli_batch_inspect_requires_references() -> None:
    """--references 是必填：不给参考件就没法配对。"""

    from cad_inspector.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["batch-inspect", "a.stl"])
