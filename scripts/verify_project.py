"""我方用一条命令跑完整交付门禁，并把结论写成可追溯文件。

门禁顺序固定为：清理过期字节码 -> 全量回归 -> 编译检查 -> 依赖检查 ->
输出审计 -> 缓存收口 -> 追加第一人称开发记录。只有每一步都通过，
整体状态才是 PASS；任何一步失败都会保留真实原因，不会改写成通过。

两个设计要点，都是踩过坑之后才加的：

一、门禁必须串行。两个门禁并发时，A 刚删完缓存、B 的 pytest 又把 ``__pycache__``
   写回来，于是 A 的 cache_cleanup 阶段报"剩余 2 个"，整体被判 FAIL。那是并发造成的
   假失败，不是代码坏了。所以我用操作系统内核文件锁（Windows ``msvcrt.locking`` /
   POSIX ``fcntl.flock``，非阻塞）保证同一时刻只有一个门禁在跑；抢不到锁直接返回
   BUSY，不写报告、不追加日志。锁由内核维护，进程崩溃也会自动释放，不需要自处理过期锁。

二、退出码有三态。PASS 返回 0，真失败返回 1，BUSY 返回 2。把 BUSY 和真失败混成同一个
   非零码，会让调用方误以为代码坏了。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cad_inspector.io.artifact_audit import audit_artifacts  # noqa: E402


_CACHE_DIR_NAMES = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)


def _load_sibling_script(filename: str):
    """我按路径加载同目录脚本，避免把 scripts 目录变成 Python 包。"""

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(
        f"cad_verify_{path.stem}",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"我无法加载脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def purge_cache_dirs(root: str | Path = ROOT) -> int:
    """我删除项目根内的字节码与工具缓存目录，绝不越出项目根。

    这是全项目唯一的递归删除，所以三重防护一个都不能省：
     1. 先 resolve() 再判断 `base not in resolved.parents`。
        直接比较字符串路径会被符号链接骗过去：一个名为 __pycache__
        却链接到项目根外的目录，字符串上"看起来"在根内；
     2. 显式跳过 is_symlink() 的目录，不跟随链接进入；
     3. 拒绝 resolved == base 本身，避免极端配置下把项目根删掉。

    删除顺序是先删文件再删目录，且用 reverse=True 从深到浅，
    否则非空目录会 rmdir 失败。unlink 带 missing_ok=True：
    遍历与删除之间文件可能已被其他进程清掉，
    这种竞态不该让门禁失败。

    单个缓存目录删不掉时（里面的 .pyc 正被别的进程握着 —— Windows 上
    打开的文件删不掉）我只跳过、不抛：门禁必须照常出报告，真删不掉
    交给 cache_cleanup 阶段如实报"剩余 N 个"。整轮崩溃、连报告都没有，
    比一次说清楚了的失败糟得多。
    """

    base = Path(root).resolve()
    # 解析真实路径后再判断是否在项目根内，防止符号链接把删除带出项目根。
    removed = 0
    for candidate in base.rglob("*"):
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        if candidate.name not in _CACHE_DIR_NAMES:
            continue
        resolved = candidate.resolve()
        if resolved == base or base not in resolved.parents:
            continue
        try:
            for child in sorted(resolved.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
            for child in sorted(resolved.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            resolved.rmdir()
            removed += 1
        except OSError:
            # 尽力而为：删不掉这一项就跳过，绝不把异常抛给调用方
            # （抛出 = 整轮门禁崩溃、无报告）。
            continue
    return removed


def _source_fingerprint() -> tuple[tuple[str, int, int], ...]:
    """我取源码快照指纹（相对路径 + mtime + 大小），用来判断门禁跑的时候源码有没有被改。

    为什么需要它：门禁要跑十几分钟，期间若有人改源码（我实测过这种并发），
    报告里的 PASS/FAIL 就不再对应**任何一个**确定的快照 ——
    可能是 pytest 或 compileall 读到了写了一半的文件。
    把这件事记进报告，比事后猜"为什么偶发红"可靠得多。
    """

    items: list[tuple[str, int, int]] = []
    for base in ("src", "tests", "scripts"):
        for path in (ROOT / base).rglob("*.py"):
            try:
                st = path.stat()
            except OSError:
                # 遍历期间文件被删（并发编辑）：记哨兵值，同样算"变了"。
                items.append((str(path.relative_to(ROOT)), -1, -1))
                continue
            items.append((str(path.relative_to(ROOT)), st.st_mtime_ns, st.st_size))
    return tuple(sorted(items))


def summarize(stages: Iterable[dict[str, Any]]) -> str:
    """我要求所有阶段都通过才给出 PASS。

    空阶段列表判 FAIL 而不是 PASS：一个阶段都没跑成通常是门禁
    自己在装配阶段就出错了（配置缺失、依赖没装）。
    把"什么都没验证"当成通过是最危险的一种默认值，
    它会让一个坏掉的门禁持续报告绿灯。

    用 item.get("ok") 而非直接索引：阶段字典可能缺少 ok 键
    （阶段自己崩了只留下错误信息），取不到即视为不通过。
    """

    items = list(stages)
    if not items:
        return "FAIL"
    return "PASS" if all(item.get("ok") for item in items) else "FAIL"


# 一次门禁的子进程超时是 1800 秒（30 分钟），再留出余量后取 3600 秒：
# 持锁超过它，那个进程已经不可能在正常干活了（正常门禁 15~20 分钟）。
_GATE_STALE_SECONDS = 3600


def _acquire_gate_lock(root: str | Path = ROOT, *, allow_orphan_takeover: bool = True):
    """我用操作系统文件锁保证同一时刻只有一个门禁在跑。

    我上一轮真的踩过这个坑：两个门禁并发时，A 刚删完缓存、B 的 pytest
    又把 __pycache__ 写回来，于是 A 的 cache_cleanup 阶段报"剩余 2 个"，
    整体状态被写成 FAIL。那是并发造成的假失败，不是代码坏了。锁由操作
    系统在内核里维护，进程崩溃也会自动释放，所以我不需要自己处理过期锁。
    """

    lock_path = Path(root).resolve() / "reports" / ".verify_project.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if lock_path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        # 抢不到锁时，先看是不是**孤儿锁**：持锁进程已经不存在了。
        # 我为什么要处理这个（2026-10-15 实测）：门禁进程会僵死，
        # 进程没了但内核锁没释放（我实测到过 pid=24268 占用 1126 秒后
        # 进程已不存在）。那之后**所有**门禁都只能拿到 BUSY，
        # 单实例保护就从"防并发"退化成"永久不可用"。
        # 这里只在确认持锁方已死时才接管，不会引发并发。
        if not allow_orphan_takeover:
            return None
        if not _holder_is_alive(root):
            return _take_over_orphan_lock(root)
        return None
    # 抢到锁后，把持有者信息写进**另一个**文件，而不是锁文件本身。
    # 为什么必须分开写：Windows 的 msvcrt.locking 是强制锁，
    # 锁住期间连**读**都被拒绝（我实测过：普通 open 和 os.open 带
    # 共享标志都 PermissionError，哪怕只想读锁区域之后的字节）。
    # 所以把 pid/时间戳写进锁文件等于写进一个谁也读不到的地方。
    # 为什么要有这一层：门禁进程僵死会让锁被永久占住，
    # 而原先 BUSY 只说"有另一个门禁在跑"，
    # 后来的人既不知道是谁占的、也不知道占了多久、更不知道它是不是早死了。
    _write_lock_holder(lock_path)
    return handle


def _holder_path(lock_path: Path) -> Path:
    """我返回持有者信息文件的位置（与锁文件相邻，但**不被锁**）。"""

    return lock_path.with_suffix(".holder")


def _write_lock_holder(lock_path: Path) -> None:
    """我记录"当前是谁持锁"，供后来者诊断 BUSY。失败也不影响门禁。"""

    try:
        import json

        _holder_path(lock_path).write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "acquired_at": datetime.now(timezone.utc)
                    .astimezone()
                    .isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _describe_lock_holder(root: str | Path = ROOT) -> str:
    """我读出锁的持有者信息，用于让 BUSY 报告说清"谁占着、占了多久"。

    为什么需要这一层：门禁进程僵死时锁会被永久占住，
    而原先 BUSY 只说"有另一个门禁在跑"，无从判断该等还是该清理。
    现在我能给出 pid、占用时长，以及那个 pid 是否还活着——
    僵死的持锁方会明确显示为"进程已不存在"，
    这一眼就能区分"真在跑"和"死了但锁没放"。
    """

    lock_path = Path(root).resolve() / "reports" / ".verify_project.lock"
    # 锁文件被锁住时既读不到也写不了，所以持有者信息单独放在 .holder 文件。
    try:
        raw = _holder_path(lock_path).read_bytes()
    except OSError:
        return "（没有持有者信息，可能由旧版门禁占着锁）"
    try:
        import json

        meta = json.loads(raw.decode("utf-8"))
        pid = int(meta["pid"])
        acquired_at = meta.get("acquired_at", "")
    except (UnicodeDecodeError, ValueError, KeyError, TypeError):
        return "（持有者信息无法解析）"

    held = ""
    stale_seconds = None
    if acquired_at:
        try:
            t0 = datetime.fromisoformat(acquired_at)
            secs = (datetime.now(timezone.utc).astimezone() - t0).total_seconds()
            held = f"，已占用 {int(secs)} 秒"
            # 一次门禁的 subprocess 超时是 1800 秒，留出余量后取 3600 秒：
            # 超过它说明那个进程已经不可能在正常干活了。
            if secs > _GATE_STALE_SECONDS:
                stale_seconds = secs
        except ValueError:
            held = ""

    alive = _pid_alive(pid)
    if not alive:
        return f"pid={pid}{held}，进程已不存在（僵死后锁未释放）"
    # 进程还在，但占用时间已经远超一次门禁的正常耗时 —— 这几乎可以肯定
    # 是僵死（我实测过：pid=26156 占了 546 秒仍"活着"，
    # 但 14 个线程全部 Waiting、CPU 不再增长、内存反而降到 6MB）。
    # 为什么这里**只告警、不自动接管**：僵死进程是"存在"的，
    # 我无法在不看它等待栈的情况下 100% 断定它没在干活。
    # 误抢一个真在跑的门禁，正是"并发导致 cache_cleanup 假失败"那个老坑。
    # 所以我把结论交给读报告的人：告诉他该去查这个 pid。
    if stale_seconds is not None:
        return (
            f"pid={pid}{held}，进程仍在但**已远超正常耗时**，"
            f"极可能是僵死（我只能告警、不会自动接管，请人工检查该 pid）"
        )
    return f"pid={pid}{held}，进程仍在"


def _pid_alive(pid: int) -> bool:
    """我判断这个 pid 是否还存在，只看存在性，不去碰它。"""

    try:
        import subprocess

        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return True  # 查不动就保守地当成还活着
    return str(pid) in out.stdout


def _holder_is_alive(root: str | Path = ROOT) -> bool:
    """我判断当前持锁方是否还活着。

    我为什么**拿不到信息时返回 True**（保守方向很关键）：
    没有 holder 文件（旧版门禁占的锁）、或信息解析不了时，
    我无法证明对方死了，那就一律当成活着、不接管。
    宁可继续 BUSY，也绝不误抢一个真在跑的门禁的锁——
    那正是"并发导致 cache_cleanup 假失败"这个老坑。
    """

    lock_path = Path(root).resolve() / "reports" / ".verify_project.lock"
    try:
        raw = _holder_path(lock_path).read_bytes()
        import json

        pid = int(json.loads(raw.decode("utf-8"))["pid"])
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
        return True
    return _pid_alive(pid)


def _take_over_orphan_lock(root: str | Path = ROOT):
    """我接管一把孤儿锁（持锁方已确认不存在），然后重新抢一次。

    为什么可以安全地删：调用方已经确认持锁 pid 不存在，
    所以没有活着的门禁在跑，不存在并发风险。
    我删的是锁文件和 holder 文件，不碰任何产物或报告。
    """

    lock_path = Path(root).resolve() / "reports" / ".verify_project.lock"
    holder = _holder_path(lock_path)
    try:
        # 锁文件正被已死进程的内核锁占着，删不掉；
        # 删不掉就没法重建，所以失败就直接放弃接管。
        lock_path.unlink()
    except OSError:
        return None
    try:
        holder.unlink()
    except OSError:
        pass
    return _acquire_gate_lock(root, allow_orphan_takeover=False)


def _release_gate_lock(handle) -> None:
    """我释放门禁文件锁，让下一个门禁可以开始。"""

    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
    # 我为什么必须顺手清掉 holder 文件（2026-10-15 实测到的真问题）：
    # 只解内核锁而不删 holder，会留下"锁已空闲、holder 却还指向一个
    # 已经退出的 pid"的不一致状态（我实测到 pid=11876 的门禁已正常
    # 结束，holder 却仍声称它占了 1237 秒）。后来者读到就会以为
    # 那是孤儿锁。清掉它，状态才与"锁已释放"一致。
    # 我只删**自己写的**那份：万一锁已被别人接管，不能删掉别人的 holder。
    # 我为什么从句柄反推根目录而不是直接用 ROOT：acquire 接受任意 root，
    # 若 release 写死 ROOT，用别的 root 取的锁就会清错地方——真正那份
    # holder 留下来，后来者又会把它当成孤儿锁。两边必须对称。
    _clear_lock_holder(Path(handle.name).resolve().parents[1])


def _clear_lock_holder(root: str | Path = ROOT) -> None:
    """我删掉 holder 文件，但只删我自己写的那一份。"""

    lock_path = Path(root).resolve() / "reports" / ".verify_project.lock"
    holder = _holder_path(lock_path)
    try:
        import json

        meta = json.loads(holder.read_bytes().decode("utf-8"))
        if int(meta["pid"]) != os.getpid():
            return
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
        return
    try:
        holder.unlink()
    except OSError:
        pass


def _print_utf8(text: str) -> None:
    """我把文本以 UTF-8 写进 stdout，不受控制台默认编码影响。

    为什么必须要这一层：Windows 控制台默认编码常是 GBK，
    而门禁报告里既有中文，也有从子进程输出里带出的替换字符
    （errors="replace" 产生的 U+FFFD）。这两类字符在 GBK 上
    都会抛 UnicodeEncodeError。

    后果不是"打印难看"，而是门禁在**跑完之后**崩溃：
    报告已写盘、结论已算出，却因为打印不出来而变成一次
    "没有结果的运行"。所以我每次都重配 stdout 的编码，
    而不是依赖调用方的环境。
    """

    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        # 重配失败时退回字节写入，保证结论一定能交出去。
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            raise
        buffer.write((text + "\n").encode("utf-8", errors="replace"))
        buffer.flush()


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    """我杀整棵进程树，而不只是直接子进程。

    我为什么需要它：.venv 的 python.exe 会再拉起一个真解释器，
    pytest 和它派生的用例子进程都是"孙进程"。只 kill 直接子进程的话，
    孙进程会继续跑 —— 它会把 ``__pycache__`` 重新写回来，
    让紧跟着的 cache_cleanup 阶段报"还剩 N 个"，制造假失败；
    在 POSIX 上还会因为继承了管道而让父进程永远等不到 EOF。

    杀不掉不算错误：进程可能已经自己退出了，此时 taskkill 返回非零是正常的，
    门禁不该因为"清理得比它慢一步"而报错。
    """

    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            timeout=30,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(process.pid), 9)
    except (ProcessLookupError, PermissionError):
        process.kill()


def _run(
    command: list[str],
    *,
    timeout: int,
    prefer: str | None = None,
) -> dict[str, Any]:
    """我执行一条子命令，返回是否成功、退出码与用于展示的摘要行。

    摘要取行的策略：默认取最后一行非空输出，给了 prefer 正则时从后往前
    取最后一条匹配的行。pytest 的最后一行是 "N passed"，而它的 stream
    里混杂着进度和警告，所以必须用 prefer 精确抓那一行，
    否则门禁报告里的摘要会是不可读的噪音。

    两个不能省的细节：
      - encoding 显式指定 utf-8 且 errors="replace"。CI/Windows 控制台
        默认编码不固定，不指定会在中文输出上直接抛 UnicodeDecodeError，
        门禁自己先崩了；
      - 必须给 timeout。某个阶段卡死时，没有超时的门禁会永远挂着，
        比"明确报超时失败"糟糕得多。

    2026-10-16 补的第三个细节：**光把解码端设成 utf-8 不够，还要把
    子进程的编码也设成 utf-8**。我为什么发现这件事：门禁报告里
    failures 那一行长期是 `test�ҷ��Ž�����ʧ...` 这样的乱码，害得
    一条偶发失败的用例名一直读不出来。根因是管道不是终端时，
    Python 子进程按 `locale.getpreferredencoding()` 输出（本机是
    GBK/cp936），而这里按 utf-8 解码，两边对不上。
    所以我在子进程环境里显式加 PYTHONIOENCODING=utf-8，
    让"子进程怎么写"和"我这里怎么读"用同一个编码。
    这一步只影响子进程的输出编码，不改它的业务行为。

    除了摘要，我还额外收集 failures：所有以 FAILED / ERROR 开头的行。
    摘要只留一行汇总（例如 "1 failed, 389 passed"），它会**丢掉到底是
    哪个用例失败**这个信息。偶发失败在有 failures 时能当场定位，
    没有 failures 时只能全量重跑碰运气。所以这两者必须分开带出。
    """
    # 2026-10-17 改：**不再用管道抓输出，改写到临时文件；超时改成杀整棵进程树**。
    # 我为什么动这里：02:34 那次门禁在 pytest 阶段挂了 10 个半小时没退出。
    # 查下来是两个问题叠在一起：
    #   1) `subprocess.run(capture_output=True, timeout=...)` 超时后只 kill
    #      **直接子进程**，然后还要 communicate() 把管道读到 EOF；
    #      而 .venv 的 python.exe 会再拉起一个真解释器，
    #      孙进程继承了 stdout 管道 —— 直接子进程被杀掉，管道却还开着，
    #      于是 communicate() 永远等不到 EOF。超时机制本身被卡死了。
    #   2) 退一步说，就算没超时，只要子进程把管道写满而父进程还没开始读，
    #      子进程就会阻塞在写 stdout 上。
    # 改成写临时文件后这两条同时消失：文件写满不会阻塞，读到的是磁盘上
    # 已有的内容，不依赖任何进程把句柄关掉。
    # 超时那边改成 taskkill /T 杀整棵树（POSIX 用 killpg），
    # 否则被丢下的孙进程会继续跑，把缓存又写回来 —— 那正是门禁锁要防的事。
    # 我不用 TemporaryDirectory：它在退出时会**强行删除**目录，
    # 而孙进程可能还攥着这两个文件（Windows 报 WinError 32），
    # 于是"清理"这一步会把门禁自己炸掉 —— 我第一版就是这么翻的。
    # 改成 mkdtemp + 尽力而为的 rmtree：删不掉就留着，
    # 临时目录本来就在 %TEMP% 下，由系统兜底，比让门禁崩掉划算。
    scratch = Path(tempfile.mkdtemp(prefix="gate-stage-"))
    stdout_path = scratch / "stdout.txt"
    stderr_path = scratch / "stderr.txt"
    try:
        with (
            stdout_path.open("wb") as out_handle,
            stderr_path.open("wb") as err_handle,
        ):
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=out_handle,
                stderr=err_handle,
                # 子进程也强制 utf-8 输出：不设的话 Python 子进程在管道下会按
                # 本地编码（本机 cp936）写，而下面按 utf-8 读，中文全成乱码。
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                start_new_session=os.name != "nt",
            )
            timed_out = False
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_tree(process)
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    returncode = -1
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if timed_out:
        # 超时必须变成"明确的失败阶段"，不能让异常冒到 main：
        # 那样门禁会崩溃退出，报告与开发记录都写不出来，
        # 事后连"是哪个阶段卡住的"都查不到（与 FAIL 报告被覆盖同一类缺口）。
        # 顺手把已经写出来的最后一行也带上：超时的阶段往往到最后
        # 都在正常输出，那一行常常直接指出卡在哪个用例上。
        last_line = next(
            (
                line.strip()
                for line in reversed(stdout_text.splitlines())
                if line.strip()
            ),
            "",
        )
        return {
            "ok": False,
            "returncode": -1,
            "summary": (
                f"阶段超时（超过 {timeout} 秒），已放弃该阶段"
                + (f"；最后一行输出：{last_line[:160]}" if last_line else "")
            ),
            "failures": [],
            "command": " ".join(command),
        }
    tail = [
        line.strip() for line in stdout_text.splitlines() if line.strip()
    ]
    error_tail = [
        line.strip() for line in stderr_text.splitlines() if line.strip()
    ]
    # 优先取最后一行；给了 prefer 正则时，取最后一条匹配该模式的行作为摘要。
    lines = tail or error_tail or ["<无输出>"]
    summary = lines[-1]
    if prefer:
        pattern = re.compile(prefer)
        for line in reversed(lines):
            if pattern.search(line):
                summary = line
                break
    # 我单独抓失败用例行：摘要会丢掉"到底哪个用例失败了"，
    # 而偶发失败必须当场能定位，否则只能全量重跑碰运气。
    failure_pattern = re.compile(r"^(FAILED|ERROR)\b")
    failures = [
        line for line in (tail + error_tail) if failure_pattern.match(line)
    ]
    result: dict[str, Any] = {
        "ok": returncode == 0,
        "returncode": returncode,
        "summary": summary,
        "failures": failures,
        "command": " ".join(command),
    }
    # 失败时把输出末尾一并带上：failures 只有 FAILED/ERROR 那一行，
    # **看不到断言消息**。我抓一次偶发红时就因为缺这条而无法定位断在哪一步。
    # 只在失败时带，避免正常跑的报告被 30 行噪音撑大。
    if not result["ok"]:
        result["output_tail"] = tail[-30:]
    return result


def _run_gate_locked(
    *,
    log_path: str | Path | None = None,
    report_path: str | Path | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    """我在已持有门禁锁的前提下执行完整门禁并写出报告。"""

    stages: list[dict[str, Any]] = []
    source_before = _source_fingerprint()
    # 先清缓存再跑测试：__pycache__ 里可能残留已删除模块的字节码，
    # 不清的话 pytest 会 import 到已经不存在的代码，出现"本地改了却不生效"
    # 这种极难排查的假绿。放在最前面才能为后续每个阶段提供干净起点。
    purged = purge_cache_dirs(ROOT)
    stages.append(
        {
            "name": "bytecode_purge",
            "ok": True,
            "summary": f"我已清理 {purged} 个项目缓存目录。",
        }
    )

    # -B 与 -p no:cacheprovider 双保险：前者不生成新字节码，
    # 后者禁用 pytest 自己的缓存插件，避免 .pytest_cache 又造出残留。
    stages.append(
        {
            "name": "pytest",
            **_run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    str(ROOT / "tests"),
                ],
                timeout=timeout,
                # prefer 让摘要行优先取"N passed/failed"这类统计行，
                # 而不是被最后一条无关输出盖掉；失败时仍保留用例名以便定位。
                prefer=r"\d+\s+(passed|failed|error)",
            ),
        }
    )
    stages.append(
        {
            "name": "compileall",
            # compileall 覆盖 tests 而不只是 src/scripts：
            # 测试文件里的语法错误同样会让 CI 整轮红，
            # 只编译源码会在跑测试时才暴露，浪费一轮完整运行。
            **_run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "compileall",
                    "-q",
                    "src",
                    "scripts",
                    "tests",
                ],
                timeout=timeout,
            ),
        }
    )
    stages.append(
        {
            "name": "pip_check",
            # pip check 能抓出"装了但依赖版本冲突"的问题，
            # 这类问题只在 import 到具体模块时才炸，
            # 放在门禁里一次查出来比等到运行时便宜得多。
            **_run(
                [sys.executable, "-B", "-m", "pip", "check"],
                timeout=timeout,
            ),
        }
    )

    audit = audit_artifacts(ROOT / "outputs")
    # 审计产物默认写到 reports/，显式给了 --report 时写到报告同目录：
    # 让审计结果与门禁报告放在一起，便于事后按同一次运行归档。
    audit_target = Path(report_path) if report_path else (
        ROOT / "reports" / "outputs_audit.json"
    )
    audit_target.parent.mkdir(parents=True, exist_ok=True)
    audit_target.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    stages.append(
        {
            "name": "outputs_audit",
            "ok": audit["status"] == "PASS",
            "summary": (
                f"{audit['valid_files']}/{audit['total_files']} 有效、"
                f"{audit['invalid_files']} 损坏、{audit['skipped_files']} 跳过、"
                f"status={audit['status']}"
            ),
            "report": str(audit_target),
        }
    )

    purged_after = purge_cache_dirs(ROOT)
    remaining_caches = sum(
        1
        for item in ROOT.rglob("*")
        if item.is_dir() and item.name in _CACHE_DIR_NAMES
    )
    stages.append(
        {
            "name": "cache_cleanup",
            "ok": remaining_caches == 0,
            "summary": (
                f"我收口时清理 {purged_after} 个缓存目录，"
                f"剩余 {remaining_caches} 个。"
            ),
        }
    )

    status = summarize(stages)
    # 源码在门禁运行期间被改过 -> 这一轮的 PASS/FAIL 不对应单一快照。
    # 只记进报告、不改 status：改它会让"别人碰了一下文件"变成硬失败，
    # 但读报告的人必须知道这次的结论建立在哪个快照上。
    source_changed_during_run = _source_fingerprint() != source_before
    # 任务书要求"代码注释率 ≥ 25%"。这一项只做可复现报告，不参与 PASS/FAIL：
    # 它衡量的是文档完整度，不是功能正确性；把它混进门禁会让"代码没坏"和
    # "注释够不够"两件事互相掩盖。
    comment_ratio_tool = _load_sibling_script("comment_ratio.py")
    # 注释率只统计 src 与 scripts，与 comment_ratio 的默认口径一致。
    # 不把 tests 算进来是因为测试文件的注释风格不同，
    # 混算会让这个指标在两种文件比例变动时自己漂移。
    comment_ratio = comment_ratio_tool.measure([ROOT / "src", ROOT / "scripts"])
    report: dict[str, Any] = {
        "schema_version": "verification-gate-1.0",
        "status": status,
        # 时间戳带时区：门禁报告常与 CI 日志对照，
        # 没有时区的时间戳在跨时区排查时会差好几个小时。
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "root": str(ROOT),
        "stages": stages,
        "source_changed_during_run": source_changed_during_run,
        "comment_ratio": {
            "target": 0.25,
            # 注释率不达标只标 BELOW_TARGET、不影响 status：
            # 它是质量目标而非正确性门槛，
            # 让它卡住整轮门禁会让"代码正确但注释没写完"变成硬失败。
            "status": (
                "PASS"
                if float(comment_ratio["line_ratio_strict"]) >= 0.25
                else "BELOW_TARGET"
            ),
            "total_lines": comment_ratio["total_lines"],
            "strict_comment_lines": comment_ratio["strict_comment_lines"],
            # 严格与含行尾两个比率都给：前者用于判定，
            # 后者用于说明"离达标还差多少行"时给出可操作的数字。
            "line_ratio_strict": comment_ratio["line_ratio_strict"],
            "line_ratio_with_inline": comment_ratio["line_ratio_with_inline"],
            "docstring_coverage": comment_ratio["docstring_coverage"],
            "module_docstring_coverage": comment_ratio[
                "module_docstring_coverage"
            ],
        },
    }

    record_target = Path(log_path) if log_path else ROOT / "开发记录.txt"
    # 只在 PASS 时写开发记录：记录里写的是"已完成一次交付校验"，
    # 门禁没过时写它等于往交付记录里塞一条未通过的记录，
    # 与这份记录"只记已交付的东西"的语义冲突。
    if status == "PASS":
        record_work = _load_sibling_script("record_work.py")
        record_work.append_record(
            log_path=record_target,
            title="我方用统一门禁脚本完成一次交付校验",
            status="门禁 PASS",
            paths=[
                "scripts/verify_project.py",
                "reports/outputs_audit.json",
            ],
            details=(
                "我方按固定顺序执行字节码清理、全量回归、编译检查、依赖检查、"
                "输出审计和缓存收口；每一步的真实结论都写入验证报告。"
            ),
            # 每一步的真实摘要都拼进去，而不是只写"通过"两个字：
            # 只看"通过"无法知道跑了多少用例、审计了多少输出，
            # 也无法发现某一阶段其实是被跳过的。
            verification="；".join(
                f"{item['name']}={item.get('summary', '')}" for item in stages
            ),
            # 能力边界每次都写：门禁通过很容易被读成"项目完成"，
            # 这段声明是防止这种误读的最后一道。
            limitations=(
                "本门禁只证明当前源码快照在本地通过，不替代甲方真实标注集、"
                "XT/PRT 原生解析、工业 G0/G1/G2、GUI 或生产验收。"
            ),
        )
        report["record"] = str(record_target)

    if report_path:
        gate_target = Path(report_path)
    else:
        gate_target = ROOT / "reports" / "verification_gate.json"
    report["report"] = str(gate_target)
    _write_gate_report(report, gate_target)
    return report


def _write_gate_report(report: dict[str, Any], target: Path) -> None:
    """我把门禁报告落盘；失败时**另存一份**，免得证据被下次运行覆盖。

    为什么必须另存：默认路径 `reports/verification_gate.json` 每次运行都覆盖，
    而开发记录**只在 PASS 时写**（那是交付记录，不该塞未通过的）。两者一叠加，
    一次 FAIL 跑完就什么都不剩 —— 我实测丢过一次 FAIL 报告，
    连是哪个阶段失败都查不回来了。所以失败时留一份固定名的副本。
    """

    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    if report.get("status") != "PASS":
        (target.parent / "verification_gate_last_failure.json").write_text(
            payload, encoding="utf-8"
        )


def _crash_report(error: BaseException) -> dict[str, Any]:
    """我把门禁主体的意外异常变成一份 FAIL 报告（带 traceback）。

    我为什么需要它：报告是门禁唯一的产出。主体崩掉时若什么都不写，
    调用方只看到一段 traceback，连"是哪个阶段炸的、当时的结论是什么"
    都查不回来 —— 那比一次明确的 FAIL 糟得多。
    """

    return {
        "schema_version": "verification-gate-1.0",
        "status": "FAIL",
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "root": str(ROOT),
        "stages": [
            {
                "name": "gate_crash",
                "ok": False,
                "summary": (
                    f"门禁在装配或收尾阶段崩溃：{type(error).__name__}: {error}"
                ),
            }
        ],
        "traceback": traceback.format_exc(),
    }


def run_gate(
    *,
    log_path: str | Path | None = None,
    report_path: str | Path | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    """我执行完整门禁，返回结构化结果并写出报告。

    我先抢门禁锁；抢不到就返回 status=BUSY，并且不写任何报告文件。
    这样并发的第二个门禁不会把前一个门禁的 PASS 结论覆盖成假 FAIL。
    """

    lock = _acquire_gate_lock(ROOT)
    if lock is None:
        # 抢不到锁就返回 BUSY 而不是排队等待：
        # 并发跑门禁会让最后的缓存收口阶段
        # 把另一个进程正在生成的 __pycache__ 当成"残留缓存"报出来。
        # 返回一份带 status 的完整报告，
        # 调用方才能区分"门禁没过"和"门禁没跑成"。
        hint = _describe_lock_holder(ROOT)
        return {
            "schema_version": "verification-gate-1.0",
            "status": "BUSY",
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(
                timespec="seconds"
            ),
            "root": str(ROOT),
            # 阶段列表留空而不是填一堆未执行的阶段：
            # 空列表明确表示"一个阶段都没跑"，
            # 填上未执行的阶段会被读成"跑了且失败"。
            "stages": [],
            "message": (
                "已有另一个门禁在运行，我拒绝并发执行：并发会让缓存收口阶段"
                "误报剩余缓存。请等当前门禁结束后重跑。"
                + (f"（锁的持有者：{hint}）" if hint else "")
            ),
        }
    try:
        return _run_gate_locked(
            log_path=log_path,
            report_path=report_path,
            timeout=timeout,
        )
    except Exception as error:
        # 兜底，也是这一类缺口的**总根**修复：门禁主体在任何一步抛出
        # 意外异常时，都必须留下一份 FAIL 报告。以前只有"阶段超时"和
        # "缓存清理"两个调用点各自补过；其余（输出审计、注释率统计、
        # 写开发记录、写报告本身、子进程根本拉不起来……）崩掉仍然是
        # "没有报告的一次运行"。与其逐个补调用点，不如在顶层兜住：
        # 报告先落盘，再把异常照旧抛出去 —— 保留响亮的失败，
        # 不把一个编程错误静默降级成一次普通 FAIL。
        crash = _crash_report(error)
        target = (
            Path(report_path)
            if report_path
            else ROOT / "reports" / "verification_gate.json"
        )
        try:
            _write_gate_report(crash, target)
        except OSError:
            # 连报告都写不出去（磁盘满/被占用）时，别再掩盖原始异常。
            pass
        raise
    # 无论中间哪一步抛异常都要放锁：
    # 不放锁会让后续所有门禁永远停在 BUSY，
    # 只能靠人工去删锁文件。
    finally:
        _release_gate_lock(lock)


def main(argv: list[str] | None = None) -> int:
    """我跑完整交付门禁，六步全绿才算 PASS。"""
    parser = argparse.ArgumentParser(
        description="我方运行完整交付门禁并写出验证报告。"
    )
    parser.add_argument("--log", default=None, help="我方追加开发记录的 TXT 路径。")
    parser.add_argument(
        "--report",
        default=None,
        help="我方写出验证门禁 JSON 的路径。",
    )
    # 超时默认 1800 秒：全量回归在这台机器上
    # 慢的时候要十几分钟，给半小时留足余量。
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    report = run_gate(
        log_path=args.log,
        report_path=args.report,
        timeout=args.timeout,
    )
    # 我不直接 print，而是先强制 stdout 为 UTF-8：门禁报告里含中文与
    # 从子进程输出里带出的替换字符，在 GBK 控制台上 print 会抛
    # UnicodeEncodeError，把一次已经跑完的门禁变得"没有结果"——
    # 这是最糟的一类失败：结论已经算出来却交不出去。
    _print_utf8(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "PASS":
        return 0
    # 我用退出码 2 单独表示"另一个门禁在跑"，让调用方能区分并发和真失败。
    return 2 if report["status"] == "BUSY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
