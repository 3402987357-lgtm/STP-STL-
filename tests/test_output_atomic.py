from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cad_inspector.output.atomic import (
    _replace_with_retry,
    atomic_path,
    ensure_free_space,
    minimum_free_bytes,
)


def test我方原子输出成功后替换目标(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text("old", encoding="utf-8")

    with atomic_path(target) as temporary:
        temporary.write_text("new", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.glob(".*.tmp")) == []


def test我方覆盖裸文件名不再因磁盘探测崩溃(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """我钉住一个实测踩到的 Windows 坑（2026-10-17）。

    旧写法 ``probe = target if target.exists() else target.parent`` 在
    目标**已存在**时会把文件本身交给 shutil.disk_usage，而 Windows 的
    nt._getdiskusage 对"裸文件名"（不含路径分隔符）会抛
    FileNotFoundError [WinError 3]。于是"覆盖当前目录下已存在的文件"
    这条最普通的路直接崩，而写成 "./out.csv" 或绝对路径就没事——
    差别只在有没有分隔符，与磁盘、权限、内容都无关，所以极容易被
    当成偶发。改成始终探测父目录后这条不再触发。
    """

    monkeypatch.chdir(tmp_path)
    target = Path("out.csv")
    target.write_text("旧内容", encoding="utf-8")

    with atomic_path(target) as temporary:
        temporary.write_text("新内容", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "新内容"


def test我方原子输出失败时保留旧目标并清理临时文件(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError):
        with atomic_path(target) as temporary:
            temporary.write_text("partial", encoding="utf-8")
            raise RuntimeError("故障注入")

    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".*.tmp")) == []


def test我方空间不足时拒绝写出(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "result.json"

    with pytest.raises(OSError, match="拒绝写出"):
        ensure_free_space(target, minimum_bytes=10**18)

    monkeypatch.setenv("CAD_INSPECTOR_MIN_FREE_MB", str(10**12))
    with pytest.raises(OSError, match="拒绝写出"):
        with atomic_path(target) as temporary:
            temporary.write_text("data", encoding="utf-8")

    assert not target.exists()


def test我方空间充足时正常写出(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    ensure_free_space(target, minimum_bytes=0)

    with atomic_path(target) as temporary:
        temporary.write_text("new", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "new"


def test我方空间下限缺省为512兆且可被环境变量覆盖(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAD_INSPECTOR_MIN_FREE_MB", raising=False)
    assert minimum_free_bytes() == 512 * 1024 * 1024

    monkeypatch.setenv("CAD_INSPECTOR_MIN_FREE_MB", "16")
    assert minimum_free_bytes() == 16 * 1024 * 1024

    monkeypatch.setenv("CAD_INSPECTOR_MIN_FREE_MB", "not-a-number")
    assert minimum_free_bytes() == 512 * 1024 * 1024


def test我方替换遇到Windows竞态会重试到成功(tmp_path: Path, monkeypatch) -> None:
    """我锁定"WinError 5 要重试"这条契约。

    os.replace 在 Windows 上是 rename 系统调用，只要目标文件被别的
    句柄短暂打开（杀毒扫描、Windows 索引器、刚写完还没释放的映射），
    它就会抛 PermissionError(WinError 5)。这种失败是**随机的**，
    同一个用例连跑几次可能只中一次。

    不重试的代价很实际：一个偶发的 WinError 5 会让整轮门禁报 FAIL，
    而它跟代码正确性毫无关系，白白把真问题淹没在噪音里。

    我让前三次抛 PermissionError、第四次成功，验证它确实重试到成功。
    """

    source = tmp_path / "src.txt"
    destination = tmp_path / "dst.txt"
    source.write_text("payload", encoding="utf-8")

    attempts = {"count": 0}
    real_replace = os.replace

    def flaky(src, dst, **kwargs):
        attempts["count"] += 1
        if attempts["count"] <= 3:
            raise PermissionError(5, "拒绝访问")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", flaky)
    _replace_with_retry(source, destination)

    assert destination.read_text(encoding="utf-8") == "payload"
    assert attempts["count"] == 4


def test我方替换持续失败时抛出原始错误而不是静默吞掉(
    tmp_path: Path, monkeypatch
) -> None:
    """我锁定"长期锁死不能静默通过"这条边界。

    重试是为了吃掉毫秒级的短占用，不是为了让"目标文件真的被长期
    锁死"这种情况假装成功。耗尽重试次数后必须抛出原始异常。
    """

    source = tmp_path / "src.txt"
    source.write_text("x", encoding="utf-8")

    def always_fail(src, dst, **kwargs):  # noqa: ARG001
        raise PermissionError(5, "拒绝访问")

    monkeypatch.setattr(os, "replace", always_fail)

    with pytest.raises(PermissionError) as caught:
        _replace_with_retry(source, tmp_path / "dst.txt")
    assert caught.value.errno == 5


def test我方替换不重试非权限错误(tmp_path: Path, monkeypatch) -> None:
    """我锁定重试范围的边界。

    只有 PermissionError 属于"可能被短暂占用"这一类。
    像 IsADirectoryError 这样的错误重试多少次都不会变好，
    重试只是白白拖慢失败。
    """

    source = tmp_path / "src.txt"
    source.write_text("x", encoding="utf-8")

    attempts = {"count": 0}

    def other_error(src, dst, **kwargs):  # noqa: ARG001
        attempts["count"] += 1
        raise IsADirectoryError(21, "Is a directory")

    monkeypatch.setattr(os, "replace", other_error)

    with pytest.raises(IsADirectoryError):
        _replace_with_retry(source, tmp_path / "dst.txt")
    assert attempts["count"] == 1


def test我方STEP拓扑报告拒绝写出含NaN的JSON(tmp_path: Path) -> None:
    """我为什么单独钉这一条：step_report.py 是 src 里唯一一个
    测试从未引用过的模块（我全库扫过），它的存在意义就是那两条不变量——
    原子写出、以及 allow_nan=False 拒绝非有限值。

    "非有限值直接报错，不写出一个违反 JSON 规范的文件"是它的文档串
    明确承诺的：拓扑报告常被下游模块直接读取，写出一个打不开的 JSON
    比文件不存在更难排查。这条同时验证失败时旧文件完好（原子性）。
    """

    from cad_inspector.output.step_report import write_step_topology_json

    target = tmp_path / "topology.json"
    target.write_text('{"old": true}', encoding="utf-8")

    class _WithNaN:
        def to_dict(self) -> dict[str, object]:
            return {"depth_mm": float("nan")}

    with pytest.raises(ValueError):
        write_step_topology_json(_WithNaN(), target)

    # 原子性：失败的写出不能把已经能用的旧文件毁掉。
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}


def test我方STEP拓扑报告写出可解析的JSON(tmp_path: Path) -> None:
    """正常路径：写出后必须能立刻被 json 解析，不留半个文件。"""

    from cad_inspector.io.step_topology import StepTopologyReport
    from cad_inspector.output.step_report import write_step_topology_json

    report = StepTopologyReport(
        status="ok",
        file_path="part.step",
        unit="mm",
        backend="occt",
        shape_is_null=False,
        shape_is_valid=True,
        entity_counts={},
        face_ids=[],
    )
    target = tmp_path / "topology.json"
    written = write_step_topology_json(report, target)
    assert written == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
