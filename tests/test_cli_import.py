from __future__ import annotations

import argparse

import pytest


def test我方主命令入口可以导入() -> None:
    from cad_inspector.cli import build_parser, main

    parser = build_parser()
    assert parser.prog == "cad-inspector"
    assert callable(main)


def test我方每个子命令都能打印帮助并以零退出() -> None:
    from cad_inspector.cli import build_parser

    parser = build_parser()
    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert subparsers, "我找不到主命令注册的子命令表。"
    subcommands = sorted(subparsers[0].choices)
    assert len(subcommands) >= 13

    for name in subcommands:
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args([name, "--help"])
        assert excinfo.value.code == 0, f"子命令 {name} 的帮助没有正常退出。"
