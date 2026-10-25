"""我方批量 BOM 入口的边界测试。

`scripts/run_bom_batch.py` 是甲方"连续处理多套模具、不遗漏任何零件"的入口，
此前没有测试覆盖。这里只测纯函数边界和空目录退出码，不启动真实 OCCT 解析，
避免把慢用例塞进常规回归。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_batch_module():
    path = Path(__file__).parents[1] / "scripts" / "run_bom_batch.py"
    spec = importlib.util.spec_from_file_location("run_bom_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test我方批量扫描递归识别大小写STEP后缀并按路径稳定排序(
    tmp_path: Path,
) -> None:
    batch = _load_batch_module()
    (tmp_path / "sub").mkdir()
    upper = tmp_path / "sub" / "B.STEP"
    lower = tmp_path / "a.stp"
    ignored = tmp_path / "notes.txt"
    for item in (upper, lower, ignored):
        item.write_text("x", encoding="ascii")

    found = batch.discover_step_files(tmp_path)

    assert [item.name for item in found] == ["a.stp", "B.STEP"]


def test我方批量同名文件分配唯一目录且保持稳定(tmp_path: Path) -> None:
    batch = _load_batch_module()
    first = tmp_path / "one" / "同名.step"
    second = tmp_path / "two" / "同名.step"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first-body", encoding="ascii")
    second.write_text("second-body", encoding="ascii")

    names = batch._unique_target_names([first, second])

    assert names[first] == "同名"
    assert names[second].startswith("同名__")
    assert names[first] != names[second]
    # 同一输入重复调用必须给出同一结果，避免批量目录漂移。
    assert batch._unique_target_names([first, second]) == names


def test我方批量空目录写出汇总并返回非零退出码(tmp_path: Path) -> None:
    batch = _load_batch_module()
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    output_dir = tmp_path / "out"

    exit_code = batch.main(
        [
            "--root",
            str(empty_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 2
    assert (output_dir / "bom_batch_summary.json").exists()
