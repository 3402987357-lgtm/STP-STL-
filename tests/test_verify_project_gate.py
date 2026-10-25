from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_gate_module():
    path = Path(__file__).parents[1] / "scripts" / "verify_project.py"
    spec = importlib.util.spec_from_file_location("project_verify_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test我方门禁带出失败用例名而不只留汇总行(tmp_path: Path) -> None:
    """我锁定"摘要之外还要带 failures"这条可诊断性契约。

    摘要只留一行汇总（例如 "1 failed, 389 passed"），它会丢掉
    **到底是哪个用例失败**这个信息。偶发失败在有 failures 时
    能当场定位，没有 failures 时只能全量重跑碰运气。

    我用两个真实 pytest 子进程验证：一个必然失败的用例能抓到用例名，
    一个收集期就报错的文件能抓到 ERROR 行。

    用例目录不用 TemporaryDirectory：我实测过，它在 Windows 上
    会在 pytest 进程还没完全退出时就删目录，子进程随即报
    FileNotFoundError 而变成 "1 error"，抓不到想要的 FAILED 行
    （20 次里撞到 1 次）。改用 tmp_path 由 pytest 自己在
    测试会话结束后统一清理，避开这个竞态。
    """

    import subprocess
    import sys

    gate = _load_gate_module()
    prefer = r"\d+\s+(passed|failed|error)"

    def _run_until(command: list[str], *, timeout: int, want) -> dict:
        """我跑子命令，并在**没拿到想要的结果**时重试有限次。

        内层 pytest 偶尔会异常收场：超时被杀、或临时文件瞬时不可读而变成
        收集期报错。那种情况下根本不存在"失败用例名"这种东西 —— 那是环境
        问题，不是门禁缺陷。门禁的解析若真的坏了，重试也不会变好，照样红。
        """
        outcome = gate._run(command, timeout=timeout, prefer=prefer)
        for _ in range(2):
            if want(outcome):
                break
            outcome = gate._run(command, timeout=timeout, prefer=prefer)
        return outcome

    # 成功时不带 failures
    ok = _run_until(
        [sys.executable, "-c", "print('3 passed in 0.1s')"],
        timeout=60,
        want=lambda outcome: outcome["ok"] is True and outcome["failures"] == [],
    )
    # 我在全量跑里见过这个用例偶发红一次，但单独跑 11 次都过、复现不出来。
    # 复现不了就不能猜根因，所以我先把**每一步的原始结论**打进断言消息：
    # 下次再红，能直接看出是子进程非零退出、还是解析没抓到那行，
    # 不用再从头怀疑一遍。
    assert ok["ok"] is True, f"这条子命令本该成功，实际：{ok}"
    assert ok["failures"] == [], f"成功的命令不该带 failures：{ok}"

    # 失败的用例必须留下名字
    failing = tmp_path / "test_failing.py"
    failing.write_text("def test_alpha():\n    assert False\n", encoding="utf-8")
    result = _run_until(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(failing),
        ],
        timeout=300,
        want=lambda outcome: any(
            "test_alpha" in line for line in outcome["failures"]
        ),
    )

    assert result["ok"] is False, f"失败用例必须被判为不通过，实际：{result}"
    assert any("test_alpha" in line for line in result["failures"]), (
        f"失败用例名没被抓出来，实际 failures={result['failures']}，"
        f"摘要={result['summary']!r}"
    )

    # 收集期报错同样要留下文件名
    broken = tmp_path / "test_broken.py"
    broken.write_text("import does_not_exist_xyz\n", encoding="utf-8")
    errored = _run_until(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(broken),
        ],
        timeout=300,
        want=lambda outcome: any("ERROR" in line for line in outcome["failures"]),
    )

    assert errored["ok"] is False, f"收集期报错必须被判为不通过，实际：{errored}"
    assert any("ERROR" in line for line in errored["failures"]), (
        f"收集期报错行没被抓出来，实际 failures={errored['failures']}，"
        f"摘要={errored['summary']!r}"
    )


def test我方门禁不会被子进程留下的孙进程卡死() -> None:
    """我钉住"门禁不会因为孙进程还攥着输出而永久挂住"。

    我为什么立这条（2026-10-17）：02:34 那次门禁在 pytest 阶段
    **挂了 10 个半小时没退出**，期间一直占着门禁锁，后面的门禁全部 BUSY。

    根因是 `subprocess.run(capture_output=True, timeout=...)` 的清理路径：
    超时后它只 kill **直接子进程**，然后还要 `communicate()` 把管道读到 EOF；
    而 .venv 的 python.exe 会再拉起一个真解释器，孙进程继承了 stdout 管道 ——
    直接子进程死了，管道却还开着，于是 `communicate()` 永远等不到 EOF，
    连超时机制本身都被卡死。

    这里我复现最小形态：父进程打印一行就退出，但它留下一个还要活好几秒的
    孙进程继续攥着输出句柄。
      - 旧实现：`_run` 会一直等到孙进程退出才返回（下面的 8 秒）；
      - 新实现：输出写临时文件，不依赖任何进程关句柄，父进程退出即返回。
    我把阈值卡在 4 秒：新实现远小于它，旧实现一定超。
    """

    import sys
    import time

    gate = _load_gate_module()

    # 父进程立刻退出，孙进程再活 8 秒并继承输出句柄。
    child = (
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'])\n"
        "print('parent done')\n"
    )
    started = time.monotonic()
    result = gate._run([sys.executable, "-c", child], timeout=60)
    elapsed = time.monotonic() - started

    assert result["ok"] is True, f"父进程应当正常退出，实际：{result}"
    assert elapsed < 4.0, (
        f"父进程已退出，但 _run 仍被孙进程拖住了 {elapsed:.1f}s；"
        "说明输出还是走的管道，孙进程不关句柄就永远读不到 EOF。"
    )


def test我方门禁超时会杀掉整棵进程树() -> None:
    """我钉住超时之后**真的把进程干掉**，而不是只放弃等待。

    我为什么和上一条分开：上一条防的是"父进程已退出却被孙进程拖住"，
    这一条防的是"父进程自己不退"。两者都出现过，处理方式也不同——
    前者靠改输出通道，后者必须显式杀整棵树。
    只放弃等待不杀进程的话，被丢下的 pytest 会继续把 __pycache__ 写回来，
    紧接着的 cache_cleanup 阶段就会报"还剩 N 个"，变成假失败。
    """

    import sys

    gate = _load_gate_module()
    result = gate._run(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout=2,
    )

    assert result["ok"] is False
    assert "超时" in result["summary"], (
        f"超时必须变成明确的失败摘要，实际：{result['summary']!r}"
    )


def test我方门禁能正确读出子进程的中文输出(monkeypatch) -> None:
    """我钉住门禁读子进程输出时的**编码一致性**。

    我为什么立这条（2026-10-16）：门禁报告里 failures 那一行长期是
    `test�ҷ��Ž�����ʧ...` 这样的乱码，导致一条偶发失败的用例名
    一直读不出来，每次只能靠"再跑一遍碰运气"。

    根因是两个编码对不上：
      - 门禁 `_run` 用 `encoding="utf-8"` 解码子进程输出；
      - 但管道不是终端时，Python 子进程按 `locale.getpreferredencoding()`
        输出，本机是 GBK/cp936。
    于是中文全变成乱码。修法是给子进程显式设 PYTHONIOENCODING=utf-8，
    让"怎么写"和"怎么读"用同一个编码。

    我用 monkeypatch 把 PYTHONIOENCODING 删掉来模拟门禁的真实环境 ——
    如果不清掉，本用例会被外层环境里的 utf-8 掩盖，测不出这个缺陷。
    这一点很关键：我第一版没删环境变量，结果无论修没修都是绿的。
    """

    import sys

    gate = _load_gate_module()
    # 模拟门禁真实环境：PYTHONIOENCODING 未设置。
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)

    probe = "print('FAILED tests/test_某中文用例.py::test_中文名 - 断言失败')"
    result = gate._run([sys.executable, "-c", probe], timeout=60)

    assert result["ok"] is True
    assert "中文名" in result["summary"], (
        "门禁读子进程中文输出时出现了乱码，说明子进程编码与解码端不一致："
        f"实际 summary={result['summary']!r}"
    )
    assert any("中文名" in line for line in result["failures"]), (
        "failures 行里的中文没被正确读出："
        f"实际 failures={result['failures']!r}"
    )


def test我方门禁在GBK控制台下也要能把结论交出去() -> None:
    """我锁定"跑完之后不能因为打印而崩"这条契约。

    门禁报告里既有中文，也有从子进程输出里带出的替换字符
    （errors="replace" 产生的 U+FFFD）。这两类字符在 GBK 控制台上
    都会抛 UnicodeEncodeError。

    后果不是"打印难看"，而是门禁在**跑完之后**崩溃：报告已写盘、
    结论已算出，却因为打印不出来而变成一次"没有结果的运行"。

    我用一个会抛 UnicodeEncodeError 的假 stdout 验证字节兜底路径确实生效。
    """

    import io
    import sys

    gate = _load_gate_module()
    text = "我方门禁报告：状态 PASS，含替换字符 \ufffd 结束"

    class _GBKBuffer(io.RawIOBase):
        def __init__(self) -> None:
            self.written = b""

        def writable(self) -> bool:
            return True

        def write(self, payload) -> int:
            self.written += bytes(payload)
            return len(payload)

    raw = _GBKBuffer()

    class _GBKStream(io.TextIOBase):
        encoding = "gbk"

        def __init__(self, buffer: _GBKBuffer) -> None:
            self._buffer = buffer

        @property
        def buffer(self) -> _GBKBuffer:
            return self._buffer

        def write(self, value: str) -> int:
            raise UnicodeEncodeError(
                "gbk", "你", 0, 1, "illegal multibyte sequence"
            )

        def flush(self) -> None:
            return None

    original = sys.stdout
    sys.stdout = _GBKStream(raw)
    try:
        gate._print_utf8(text)
    finally:
        sys.stdout = original

    assert raw.written
    assert raw.written.decode("utf-8", errors="replace").strip() == text

    # 正常路径也要能写
    buffer = io.StringIO()
    sys.stdout = buffer
    try:
        gate._print_utf8(text)
    finally:
        sys.stdout = original
    assert text in buffer.getvalue()


def test我方门禁只有全部阶段通过才给_PASS() -> None:
    gate = _load_gate_module()

    assert gate.summarize([{"ok": True}, {"ok": True}]) == "PASS"
    assert gate.summarize([{"ok": True}, {"ok": False}]) == "FAIL"
    assert gate.summarize([]) == "FAIL"


def test我方门禁只清理项目根内缓存(tmp_path: Path) -> None:
    gate = _load_gate_module()

    inside = [
        tmp_path / "src" / "pkg" / "__pycache__",
        tmp_path / ".pytest_cache",
    ]
    for cache_dir in inside:
        cache_dir.mkdir(parents=True)
        (cache_dir / "stale.bin").write_bytes(b"stale")

    outside = tmp_path.parent / f"{tmp_path.name}-outside" / "__pycache__"
    outside.mkdir(parents=True)
    (outside / "keep.bin").write_bytes(b"keep")
    try:
        removed = gate.purge_cache_dirs(tmp_path)

        assert removed == len(inside)
        for cache_dir in inside:
            assert not cache_dir.exists()
        assert outside.exists()
    finally:
        if outside.exists():
            for child in outside.iterdir():
                child.unlink()
            outside.rmdir()
            outside.parent.rmdir()


def test我方门禁锁互斥且释放后可再次获取(tmp_path: Path) -> None:
    gate = _load_gate_module()

    first = gate._acquire_gate_lock(tmp_path)
    assert first is not None
    try:
        assert gate._acquire_gate_lock(tmp_path) is None
    finally:
        gate._release_gate_lock(first)

    second = gate._acquire_gate_lock(tmp_path)
    assert second is not None
    gate._release_gate_lock(second)


def test我方门禁并发时返回_BUSY_且不写报告(tmp_path: Path) -> None:
    gate = _load_gate_module()

    # 我在这里持有真实项目锁，模拟"另一个门禁正在跑"。
    lock = gate._acquire_gate_lock(gate.ROOT)
    report_path = tmp_path / "gate.json"
    log_path = tmp_path / "log.txt"
    try:
        report = gate.run_gate(report_path=report_path, log_path=log_path)
    finally:
        if lock is not None:
            gate._release_gate_lock(lock)

    assert report["status"] == "BUSY"
    assert report["stages"] == []
    assert not report_path.exists()
    assert not log_path.exists()


def test我方阶段汇总把空列表与缺_ok键都判为失败() -> None:
    """我锁定 summarize 的"绝不默认通过"这条契约。

    一个阶段都没跑成通常是门禁自己在装配阶段就出错了
    （配置缺失、依赖没装）。把"什么都没验证"当成通过是最危险的
    一种默认值——它会让一个坏掉的门禁持续报告绿灯。

    同理，阶段自己崩了只留下错误信息、没有 ok 键时，
    也必须视为不通过而不是跳过。
    """
    gate = _load_gate_module()

    assert gate.summarize([]) == "FAIL"
    assert gate.summarize([{"ok": True}, {"ok": True}]) == "PASS"
    assert gate.summarize([{"ok": True}, {"ok": False}]) == "FAIL"
    assert gate.summarize([{"ok": True}, {"error": "boom"}]) == "FAIL"


def test我方缓存清理只删项目根内的缓存目录(tmp_path: Path) -> None:
    """我锁定递归删除不外溢。

    这是全项目唯一的递归删除，我用一个"根外的真目录 + 根内同名目录"
    的场景验证边界：根外的目录及其内容必须原样保留。
    """
    gate = _load_gate_module()

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("must survive", encoding="utf-8")

    root = tmp_path / "proj"
    root.mkdir()
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "stale.pyc").write_bytes(b"\x00")

    removed = gate.purge_cache_dirs(root)

    assert removed == 1
    assert not cache.exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "must survive"


def test我方缓存清理遇到删不掉的文件只跳过不崩(
    tmp_path: Path, monkeypatch
) -> None:
    """我钉住"删不掉一个缓存文件不能把整轮门禁带崩"这条契约。

    为什么必须有这条：purge_cache_dirs 是全项目唯一的递归删除，而它在
    门禁里被调用时外面**没有** try/except。一个被别的进程握着的 .pyc
    （Windows 上打开的文件删不掉，实测抛 [WinError 32]）会让它抛
    PermissionError，异常一路冒到 main -> 整轮门禁崩溃、**连报告都写不出来**。
    所以这里强制 unlink 抛错，要求 purge 只跳过、不抛。
    """

    gate = _load_gate_module()
    cache = tmp_path / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    locked = cache / "locked.pyc"
    locked.write_bytes(b"\x00")

    real_unlink = Path.unlink

    def fake_unlink(self: Path, *args, **kwargs):
        if self == locked:
            raise PermissionError(13, "另一个程序正在使用此文件")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fake_unlink)

    removed = gate.purge_cache_dirs(tmp_path)  # 不得抛

    assert removed == 0, "删不掉的缓存目录不该被计数"
    assert locked.exists(), "删不掉的文件应原样留着，交给 cache_cleanup 报剩余"


def test我方拿不到持有者信息时保守判活不接管(tmp_path: Path) -> None:
    """我锁定"证明不了对方死了，就当它活着"这条安全默认。

    孤儿锁接管是**唯一**会绕过互斥的路径，所以它的守门条件必须是
    "保守优先"：holder 文件缺失（旧版门禁占的锁）或内容损坏时，
    我无法证明持锁方已死，就必须返回"还活着"、继续 BUSY。
    宁可暂时不可用，也不能误抢一个真在跑的门禁——那正是
    "并发导致 cache_cleanup 假失败"那个老坑。
    """

    gate = _load_gate_module()
    reports = tmp_path / "reports"
    reports.mkdir()

    # 场景一：完全没有 holder 文件。
    assert gate._holder_is_alive(tmp_path) is True

    # 场景二：holder 存在但内容是垃圾，解析不出 pid。
    (reports / ".verify_project.holder").write_text("{not json", encoding="utf-8")
    assert gate._holder_is_alive(tmp_path) is True


@pytest.mark.skipif(
    __import__("os").name != "nt",
    reason="判活走 tasklist，只在 Windows 上有意义。",
)
def test我方持有者进程已死时判为孤儿锁(tmp_path: Path) -> None:
    """我钉住"进程确实不存在时，才判为孤儿"。

    这条是接管的前提条件：只有它返回 False，调用方才敢删锁重建。
    我用一个实测查不到的 pid（999999）写 holder，断言判为孤儿；
    再用**当前进程自己的 pid** 写 holder，断言判为存活——
    一死一活两个方向都钉住，避免哪天判活写反了却没人发现。
    """

    import json
    import os

    gate = _load_gate_module()
    reports = tmp_path / "reports"
    reports.mkdir()
    holder = reports / ".verify_project.holder"

    holder.write_text(json.dumps({"pid": 999999}), encoding="utf-8")
    assert gate._holder_is_alive(tmp_path) is False

    holder.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    assert gate._holder_is_alive(tmp_path) is True


def test我方孤儿锁能被接管并重新获取(tmp_path: Path) -> None:
    """我钉住接管真的能把门禁从"永久 BUSY"里救回来。

    我为什么要这条：没有它时，一旦出现"进程已死但内核锁没释放"，
    单实例保护就从"防并发"退化成"永久不可用"，而唯一的出路是
    人工去删锁文件——我上一轮就是这么干的。

    接管后我要求两件事：拿到句柄（不是 None），且 holder 被重写成
    当前进程——否则下一个来的人还会以为锁是孤儿。
    """

    import json
    import os

    gate = _load_gate_module()
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / ".verify_project.lock").write_bytes(b"\x00")
    (reports / ".verify_project.holder").write_text(
        json.dumps({"pid": 999999}), encoding="utf-8"
    )

    handle = gate._take_over_orphan_lock(tmp_path)
    try:
        assert handle is not None, "孤儿锁必须能被接管，否则门禁永久不可用"
        holder = json.loads(
            (reports / ".verify_project.holder").read_text(encoding="utf-8")
        )
        assert holder["pid"] == os.getpid(), "接管后 holder 必须指向新的持锁方"
    finally:
        if handle is not None:
            gate._release_gate_lock(handle)


def test我方判活拿不到tasklist输出时保守判活不崩(monkeypatch) -> None:
    """我钉住"拿不到进程列表就当对方还活着"，并禁止在 None 上做 in 判断。

    我为什么要这条：中文 Windows 的 tasklist 输出是 GBK 字节，而
    subprocess 在 text=True 且**没有显式编码**时按解释器首选编码解码。
    UTF-8 模式（PYTHONUTF8=1）下一次 UnicodeDecodeError(0xd0) 发生在
    subprocess 的读取线程里 —— 它不抛给调用方，而是让 out.stdout 变成
    None，随后 `str(pid) in None` 抛 TypeError，把一次判活变成门禁崩溃。
    另一条路径（tasklist 查不动，例如被拒绝）给的是空串，同样没拿到列表。
    两种情况我都证明不了持锁方死了，所以必须保守判活、并且留下警告。
    """

    import subprocess

    gate = _load_gate_module()
    calls: list[dict] = []

    def _fake_run(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 1, stdout=payload, stderr="")

    for payload, why in (
        (None, "读取线程解码失败时 stdout 的真实形态"),
        ("", "tasklist 查不动时 stdout 的真实形态"),
    ):
        calls.clear()
        monkeypatch.setattr(subprocess, "run", _fake_run)
        with pytest.warns(RuntimeWarning):
            assert gate._pid_alive(999999) is True, f"{why}：必须保守判活"
        assert calls[0].get("encoding"), "我必须显式指定编码，不能交给首选编码"


@pytest.mark.skipif(
    __import__("os").name != "nt",
    reason="tasklist 只在 Windows 上有意义。",
)
def test我方判活遇到GBK输出在UTF8模式下不崩(tmp_path: Path) -> None:
    """我用真实的 subprocess + 真实 GBK 字节复现"判活把门禁打崩"。

    复现命令（红）：
      $env:PYTHONUTF8=1
      .venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider
        tests/test_verify_project_gate.py::test我方判活遇到GBK输出在UTF8模式下不崩

    为什么非要开子进程：UTF-8 模式只能在解释器启动时确定，本进程改不了；
    而缺陷恰好只在那个模式下出现。所以我在子进程里显式打开 UTF-8 模式，
    再用一个**真的** subprocess 产出 GBK 字节喂给 _pid_alive（把 tasklist
    换成写 GBK 字节的 python，其余参数原样透传，解码走的仍是 CPython
    真实的 text=True 路径）。修之前：读取线程解码失败 -> stdout=None ->
    TypeError -> 子进程非零退出，这条用例红；修之后应打印 ALIVE False。
    """

    import os
    import subprocess
    import sys
    import textwrap

    gate_path = Path(__file__).parents[1] / "scripts" / "verify_project.py"
    child = tmp_path / "child_pid_alive.py"
    child.write_text(
        textwrap.dedent(
            """
            import importlib.util
            import subprocess
            import sys

            spec = importlib.util.spec_from_file_location("gate", sys.argv[1])
            gate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gate)

            # 中文 Windows 的 tasklist 输出 GBK 字节（0xd0 0xc5 就是「信」）；
            # 这里用真实子进程产出同样的字节，其余 subprocess 参数
            # （含 text=True / encoding）原样透传，保证走真实解码路径。
            real_run = subprocess.run

            def fake_run(command, **kwargs):
                stub = [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(bytes([0xd0, 0xc5, 0xcf, 0xa2]))",
                ]
                return real_run(stub + list(command[1:]), **kwargs)

            subprocess.run = fake_run
            print("ALIVE", gate._pid_alive(999999))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"  # 缺陷只在 UTF-8 模式（首选编码 utf-8）下出现
    proc = subprocess.run(
        [sys.executable, "-B", str(child), str(gate_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, f"判活不得把进程打崩：\n{proc.stdout}\n{proc.stderr}"
    assert "ALIVE False" in proc.stdout, proc.stdout


def test我方门禁失败时另存报告免得证据被覆盖(tmp_path: Path) -> None:
    """我锁定"FAIL 报告必须留一份"这条可诊断性契约。

    默认路径 verification_gate.json 每次运行都覆盖，而开发记录只在 PASS 时写，
    两者一叠加，一次 FAIL 跑完就什么都不剩 —— 我实测丢过一次 FAIL 报告，
    连是哪个阶段失败都查不回来。所以失败时必须留一份固定名的副本。
    """

    import json

    gate = _load_gate_module()
    reports = tmp_path / "reports"
    target = reports / "verification_gate.json"
    failure_copy = reports / "verification_gate_last_failure.json"

    # PASS 不该产生失败副本
    gate._write_gate_report({"status": "PASS", "stages": []}, target)
    assert target.exists()
    assert not failure_copy.exists(), "PASS 不该产生失败副本"

    # FAIL 必须另存一份，且内容与报告一致
    gate._write_gate_report(
        {"status": "FAIL", "stages": [{"name": "pytest", "ok": False}]}, target
    )
    assert failure_copy.exists(), "FAIL 必须另存一份，否则证据会被下次运行覆盖"
    saved = json.loads(failure_copy.read_text(encoding="utf-8"))
    assert saved["status"] == "FAIL"
    assert saved["stages"][0]["name"] == "pytest"


def test我方门禁主体崩溃也要留下失败报告(
    tmp_path: Path, monkeypatch
) -> None:
    """我钉住"门禁在任何一步崩溃，报告都必须落盘"这条兜底契约。

    为什么必须有这条：报告是门禁唯一的产出。主体里有一堆没包异常的调用
    （输出审计、注释率统计、写开发记录、写报告本身、子进程根本拉不起来……），
    任何一处崩掉都会让整轮**没有任何报告**，调用方只看到一段 traceback。
    我在顶层兜底：崩溃也必须写一份 FAIL 报告（带 traceback），再把异常抛出。
    """

    import json

    gate = _load_gate_module()
    # 让门禁把锁抢在 tmp_path 里，别去碰真实项目的锁。
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    report_path = tmp_path / "reports" / "gate.json"

    def boom(**kwargs):
        raise RuntimeError("装配阶段炸了")

    monkeypatch.setattr(gate, "_run_gate_locked", boom)

    with pytest.raises(RuntimeError):
        gate.run_gate(report_path=report_path)

    assert report_path.exists(), "主体崩溃时也必须留下报告"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "FAIL"
    assert saved["stages"][0]["name"] == "gate_crash"
    assert "RuntimeError" in saved["stages"][0]["summary"]


def test我方门禁阶段超时要变成失败阶段而不是崩溃() -> None:
    """我锁定"阶段超时 -> 明确的失败阶段"这条契约。

    以前 `subprocess.TimeoutExpired` 没人接，会一路冒到 main 让门禁**崩溃退出**，
    报告与开发记录都写不出来 —— 事后连是哪个阶段卡住的都查不到。
    """

    import sys

    gate = _load_gate_module()
    result = gate._run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=2,
    )
    assert result["ok"] is False, f"超时必须算失败，实际：{result}"
    assert result["returncode"] == -1
    assert "超时" in result["summary"], f"摘要要说清是超时：{result}"


def test我方门禁能发现运行期间源码被改(tmp_path: Path, monkeypatch) -> None:
    """我锁定"报告要记下源码快照有没有中途变化"这条契约。

    门禁跑十几分钟，期间若有人改源码，PASS/FAIL 就不对应任何确定快照
    （可能是 pytest 读到了写了一半的文件）。指纹必须能察觉这种改动。
    """

    gate = _load_gate_module()
    for sub in ("src", "tests", "scripts"):
        (tmp_path / sub).mkdir()
    target = tmp_path / "src" / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", tmp_path)

    before = gate._source_fingerprint()
    assert gate._source_fingerprint() == before, "没改动时指纹必须稳定"

    # 改内容且改大小，避免依赖文件系统 mtime 的分辨率。
    target.write_text("x = 22222\n", encoding="utf-8")
    assert gate._source_fingerprint() != before, "改了源码指纹必须变"


def test我方门禁失败时要带出输出末尾便于定位断言() -> None:
    """我锁定"失败阶段必须带出输出末尾"这条契约。

    `failures` 只有 FAILED/ERROR 那一行，看不到断言消息 ——
    我抓一次偶发红时就因为缺这条而无法定位断在哪一步。
    """

    import sys

    gate = _load_gate_module()
    result = gate._run(
        [
            sys.executable,
            "-c",
            "print('MARKER_ASSERT_DETAIL'); raise SystemExit(1)",
        ],
        timeout=60,
    )
    assert result["ok"] is False, f"非零退出必须算失败：{result}"
    tail = result.get("output_tail") or []
    assert any("MARKER_ASSERT_DETAIL" in line for line in tail), (
        f"失败时必须带出输出末尾，实际：{result}"
    )
    # 成功的命令不该背这个包袱。
    ok = gate._run([sys.executable, "-c", "print('fine')"], timeout=60)
    assert "output_tail" not in ok, f"成功时不该带 output_tail：{ok}"
