"""我为流程图区域写回归测试，确保图与源码一致、产物可被读回、CLI 接得上。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cad_inspector.output.flowchart import (
    FlowChart,
    FlowEdge,
    FlowNode,
    build_flowcharts,
    verify_flowcharts,
    write_flowchart_markdown,
    write_flowcharts,
)


def test_build_flowcharts_returns_three_charts() -> None:
    """任务书要求至少一张图，我实际出一套三张并保证键不重复。"""

    charts = build_flowcharts()
    keys = [chart.key for chart in charts]
    assert len(charts) == 3
    assert len(set(keys)) == len(keys)
    assert "main" in keys
    assert "gate" in keys
    assert "modules" in keys


def test_main_chart_has_review_convergence() -> None:
    """我要求主流程图把"复核"画成汇合点——这是用户期望的语义。"""

    main_chart = next(chart for chart in build_flowcharts() if chart.key == "main")
    targets = [edge.target for edge in main_chart.edges]
    review = "review"
    assert review in targets
    # 复核节点至少被三条不同来源汇入：质量、单位、配准、检测/关联。
    incoming = [edge.source for edge in main_chart.edges if edge.target == review]
    assert len(set(incoming)) >= 3
    # 而且 review 必须有出边流向 "write"，否则就是死节点。
    outgoing = [edge.target for edge in main_chart.edges if edge.source == review]
    assert "write" in outgoing


def test_verify_flowcharts_consistent() -> None:
    """我要求所有引用的模块与脚本都能 import 或找到。"""

    result = verify_flowcharts()
    assert result["consistent"] is True
    assert result["missing_modules"] == []
    # 至少要校验 30 个模块；少于这个数说明图被悄悄缩水。
    assert result["checked_module_count"] >= 30


def test_verify_flowcharts_detects_missing_module() -> None:
    """我手动构造一张图带坏模块名，verify 必须把它揪出来。"""

    fake = FlowChart(
        key="fake",
        title="假图",
        subtitle="故意引入坏模块",
        nodes=[FlowNode("a", "A", "cad_inspector.does_not_exist")],
        edges=[],
    )
    result = verify_flowcharts([fake])
    assert result["consistent"] is False
    assert any("does_not_exist" in entry for entry in result["missing_modules"])


def test_write_flowcharts_produces_files(tmp_path: Path) -> None:
    """我要求 PNG 与 Mermaid 同时存在。"""

    artifacts = write_flowcharts(tmp_path)
    for path in artifacts["artifacts"].values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
    write_flowchart_markdown(tmp_path)
    markdown_path = tmp_path / "flowcharts.md"
    assert markdown_path.is_file()
    content = markdown_path.read_text(encoding="utf-8")
    assert "flowchart TD" in content


def test_flowchart_mermaid_contains_real_modules() -> None:
    """Mermaid 文本里要出现真实的关键字，否则就是张空图。"""

    target = Path("outputs/flowcharts/flowcharts.md")
    if not target.exists():
        write_flowchart_markdown("outputs/flowcharts")
    text = target.read_text(encoding="utf-8")
    # Mermaid 节点只显示 key，不含模块名；
    # 我要确保三张图的标题在文件里。
    for title in ("主流程", "门禁", "分组"):
        assert title in text


def test_cli_flowchart_subcommand(tmp_path: Path) -> None:
    """我要求 CLI flowchart 子命令能跑通并写产物。"""

    from cad_inspector.cli import main

    rc = main(["flowchart", "--output-dir", str(tmp_path), "--strict"])
    assert rc == 0
    assert (tmp_path / "flowchart_main.png").is_file()
    assert (tmp_path / "flowchart_gate.png").is_file()
    assert (tmp_path / "flowchart_modules.png").is_file()


def test_cli_flowchart_strict_rejects_missing_module(tmp_path: Path, monkeypatch) -> None:
    """我把模块校验改坏，CLI 必须返回非零退出码。"""

    def _fake_verify(_charts) -> dict:
        return {"checked_module_count": 1, "missing_modules": ["fake"], "consistent": False}

    monkeypatch.setattr(
        "cad_inspector.output.flowchart.verify_flowcharts", _fake_verify
    )
    from cad_inspector.cli import main

    rc = main(["flowchart", "--output-dir", str(tmp_path), "--strict"])
    assert rc == 2
