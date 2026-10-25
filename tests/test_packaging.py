"""我钉住打包交付物的契约。

我为什么要加这个文件：`pyproject.toml` 里的入口点、spec 里的
hiddenimports、build_installer 的校验清单，这三者任何一处悄悄坏掉，
都不会让现有 600+ 条用例变红——因为**没有一条会去看打包**。
而甲方拿到的是打包产物，不是我的源码目录。

我刻意**不在这里真的跑 PyInstaller**：一次构建要几分钟、
产出 500MB，放进门禁会让每次验证都变慢好几倍。
我改成钉住"契约"——spec 声明了什么、入口点叫什么、
打包脚本的校验逻辑对不对。真正的完整构建我用
`scripts/build_installer.py` 手工跑过并验证过（见 README）。
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_build_installer():
    import importlib.util

    path = ROOT / "scripts" / "build_installer.py"
    spec = importlib.util.spec_from_file_location("build_installer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test我方打包脚本在缺少exe时明确失败而不是产出空包() -> None:
    """我钉住"没先构建"这个最常见的误用会**明确报错**。

    我为什么钉这个：如果不检查就直接压，会产出一个空的或残缺的 zip，
    甲方解压后双击没反应，而我会以为打包成功了。
    静默产出坏包比直接失败糟得多——后者一眼能看懂。
    """

    module = _load_build_installer()
    empty = ROOT / "dist_pyi" / "__nonexistent__"
    output = ROOT / "dist" / "__should_not_exist__.zip"
    assert module.build(empty, output) == 2
    assert not output.exists(), "打包失败时不该留下产物"


def test我方打包脚本产出的校验清单可被逐文件核对(tmp_path: Path) -> None:
    """我钉住校验清单**真的能用来验货**，而不只是写了个文件。

    我为什么钉这个：校验清单的意义是让甲方解压后核实"没被改过"。
    如果路径写错或算法不一致，清单就成了摆设——
    看起来很专业，实际验不了任何东西。
    所以我造一个假目录、真跑一遍打包、再按清单逐文件重算 sha256，
    要求全部对得上。
    """

    module = _load_build_installer()
    dist_dir = tmp_path / "dist"
    (dist_dir / "nested").mkdir(parents=True)
    (dist_dir / "cad-inspector.exe").write_bytes(b"fake-exe")
    (dist_dir / "nested" / "lib.dll").write_bytes(b"fake-dll")

    output = tmp_path / "out.zip"
    assert module.build(dist_dir, output) == 0

    manifest = output.with_suffix(".sha256.txt")
    assert manifest.is_file(), "我要求校验清单单独落盘，否则甲方解压前无法核实"

    # 解压后逐文件重算：这正是甲方拿到手之后会做的事。
    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(output) as zf:
        zf.extractall(extract_dir)

    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        target = extract_dir / name
        assert target.is_file(), f"清单里的 {name} 在包里找不到"
        assert (
            hashlib.sha256(target.read_bytes()).hexdigest() == digest
        ), f"{name} 的校验和对不上，清单无法用于验货"
        checked += 1
    assert checked == 2, f"我造了 2 个文件，清单里应有 2 条，实际 {checked}"


def test我方spec声明了OCP与关键依赖的隐藏导入() -> None:
    """我钉住 spec 里 OCP 的收集还在。

    我为什么钉这个（这是打包**最容易坏**的地方）：
    OCP 有海量运行时动态加载的模块，PyInstaller 静态分析看不见。
    一旦 spec 里那两行收集被删，构建**照样成功**，
    exe 也照样生成，但一跑就 `No module named OCP.XXX`——
    而门禁里没有任何一条用例会发现。
    """

    spec_text = (ROOT / "cad_inspector.spec").read_text(encoding="utf-8")
    assert "collect_dynamic_libs(\"OCP\")" in spec_text.replace("'", '"')
    assert "collect_submodules(\"OCP\")" in spec_text.replace("'", '"')
    # 我还钉住 torch 被排除：不排除的话 exe 会大出好几个 GB，
    # 而 torch 只被可选的深度学习增强用到。
    assert '"torch"' in spec_text.replace("'", '"')


def test我方pyproject声明的入口点都能在包里找到() -> None:
    """我钉住 pyproject 写的入口点**真的存在**，不是写错的名字。

    我为什么钉这个：入口点写错（模块名或函数名拼错）时，
    `pip install` 会**成功**——pip 不校验目标是否存在。
    甲方装完双击没反应才发现，而这个错误本可以在装之前查出来。
    """

    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover
        pytest.skip("当前 Python 没有 tomllib，无法解析 pyproject")

    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]
    assert scripts, "pyproject 里没声明任何入口点"

    import importlib

    missing: list[str] = []
    for name, target in scripts.items():
        module_name, _, func = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing.append(f"{name} -> {target}（模块导入失败：{exc}）")
            continue
        if not callable(getattr(module, func, None)):
            missing.append(f"{name} -> {target}（{module_name} 里没有可调用的 {func}）")

    assert not missing, "这些入口点装完会打不开：\n" + "\n".join(missing)
