"""我用同目录临时文件替换正式输出，避免中断留下半个文件。"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# 默认在写出前至少保留 512 MiB 空闲空间，避免重负载运行把 D 盘写满。
_DEFAULT_MIN_FREE_BYTES = 512 * 1024 * 1024

# Windows 上 os.replace 会随机性失败（WinError 5）：
# 杀毒软件、索引器、甚至同机的另一个进程短暂打开目标文件，
# 都会让 rename 被拒。这是**环境竞态而不是数据错误**，
# 重试几次就能成功。我实测在 pytest 全量并发里撞到过。
#
# 我为什么把窗口从 8 次/1.4 秒扩到 12 次/约 12 秒（实测依据）：
# 那个长年 flaky 的 STEP 修补用例，我查到病根就是这里——
# Windows Defender 实时防护（MsMpEng）在本机开着、项目目录未加排除，
# 它扫描刚写出的大 STEP 文件时会把目标句柄占住**远不止 1.4 秒**。
# 我做过对照实验：把 rename 注入失败一次，就能稳定复现出两种失败形态
#   - 目标原本不存在 -> 文件没写出 -> 测试报"应当写出结果，但没有写出"；
#   - 目标残留上一轮旧内容 -> 回读到未升级的旧文件 -> solid=0。
# 这与历史上两次 flaky 的报错原文完全对得上。
# 1.4 秒的窗口挡不住 Defender，所以它偶发穿透；扩到约 12 秒后
# 单次扫描能覆盖，而"目标被长期锁死"仍会在末尾抛出原始异常，不静默吞。
_REPLACE_RETRY_COUNT = 12
_REPLACE_RETRY_DELAY_SECONDS = 0.10
# 退避上限：不加帽的话线性退避到第 12 次单次就要等 1.2 秒，
# 累计反而不可控；封顶 1 秒后累计约 12 秒，是我实测够用的窗口。
_REPLACE_RETRY_MAX_DELAY_SECONDS = 1.0
# 退避里加抖动：多个进程同时重试时，固定间隔会让它们整齐地
# 一起撞上去（惊群），抖动把它们错开。
_REPLACE_RETRY_JITTER_SECONDS = 0.05


def _replace_with_retry(temporary: Path, target: Path) -> None:
    """我把临时文件替换成目标文件，并在 Windows 竞态下重试。

    为什么必须重试：os.replace 在 Windows 上是 rename 系统调用，
    只要目标文件被别的句柄短暂打开（杀毒扫描、Windows 索引器、
    刚写完还没释放的文件映射），它就会抛 PermissionError(WinError 5)。
    这种失败是**随机的**，同一个用例连跑几次可能只中一次。

    不重试的代价很实际：一个偶发的 WinError 5 会让整轮门禁报 FAIL，
    而它跟代码正确性毫无关系，白白把真问题淹没在噪音里。

    重试次数与间隔的取值理由：现在是 12 次、间隔从 0.1 秒线性加、
    单次封顶 1 秒，累计约 12 秒（旧的 8 次 × 50 ms 只有 1.4 秒，
    挡不住 Defender 扫大 STEP 文件，见上面常量处的实测说明）。
    这个量级足以覆盖杀毒扫描这类秒级占用；又不至于让
    "目标文件真的被长期锁死"这种情况挂住整个流程——
    那种情况下最终仍然会抛出原始异常，不静默吞掉。
    """

    last_error: OSError | None = None
    for attempt in range(_REPLACE_RETRY_COUNT):
        try:
            os.replace(temporary, target)
            return
        except PermissionError as error:
            last_error = error
            if attempt + 1 < _REPLACE_RETRY_COUNT:
                delay = min(
                    _REPLACE_RETRY_DELAY_SECONDS * (attempt + 1),
                    _REPLACE_RETRY_MAX_DELAY_SECONDS,
                )
                delay += random.uniform(0, _REPLACE_RETRY_JITTER_SECONDS)
                time.sleep(delay)
    assert last_error is not None
    # 我为什么要把裸 WinError 5 包一层：原来的报错只有一个
    # "PermissionError: [WinError 5]"，看不出是杀毒占用还是权限问题，
    # 甲方和复盘的人只能反复猜。这里把"重试了几次、等了多久、
    # 最可能是谁占着"直接写进消息，下次再撞上就不用重新查一遍。
    #
    # 我为什么仍然用 PermissionError 并**逐字段保留 errno**：既有用例
    # 钉住了"耗尽重试后必须抛 PermissionError 且 errno 不变"这条边界，
    # 而下游也可能按 errno 分派。只换消息、不换类型与 errno，
    # 既拿到可读的诊断，又不破坏这条契约。
    wrapped = PermissionError(
        last_error.errno,
        f"我方重试 {_REPLACE_RETRY_COUNT} 次仍无法用临时文件替换 "
        f"{target}。Windows 上最常见的原因是目标文件被另一个句柄短暂"
        "占用：杀毒实时扫描（MsMpEng）、Windows 索引器，或同机另一个"
        "进程正在读它。这不是数据错误。可将项目目录加入杀毒排除项，"
        "或稍后重试。原始错误：%s" % last_error,
    )
    wrapped.__cause__ = last_error
    raise wrapped


def minimum_free_bytes() -> int:
    """我读取空间下限：优先用 CAD_INSPECTOR_MIN_FREE_MB，缺省 512 MiB。"""

    raw = os.environ.get("CAD_INSPECTOR_MIN_FREE_MB", "").strip()
    if not raw:
        return _DEFAULT_MIN_FREE_BYTES
    try:
        megabytes = int(raw)
    except ValueError:
        return _DEFAULT_MIN_FREE_BYTES
    return max(0, megabytes) * 1024 * 1024


def ensure_free_space(path: str | Path, minimum_bytes: int | None = None) -> None:
    """我在写出前检查目标所在卷的空闲空间，低于下限就拒绝继续写。"""

    target = Path(path)
    # 我为什么始终探测**父目录**，而不是目标文件本身（2026-10-17 实测踩到）：
    # Windows 的 nt._getdiskusage 对"裸文件名"（不含路径分隔符、且文件存在）
    # 会抛 FileNotFoundError [WinError 3]。旧写法是
    # ``probe = target if target.exists() else target.parent``，于是
    # "覆盖当前工作目录下已存在的文件"这条最普通的路直接崩：
    #     atomic_path(Path("out.csv"))  # out.csv 已存在 -> WinError 3
    # 而加上 "./" 或写成绝对路径就没事——差别只在有没有分隔符，
    # 与磁盘、权限、内容都无关，所以它藏得深、也最容易被当成偶发。
    # 父目录一定是目录，既在同一卷上拿得到同样的空闲量，又没有这个坑。
    probe = target.parent
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    floor = minimum_free_bytes() if minimum_bytes is None else minimum_bytes
    if free < floor:
        raise OSError(
            f"我方拒绝写出：{probe} 所在磁盘仅剩 {free} 字节，"
            f"低于安全下限 {floor} 字节，请先清理磁盘再运行。"
        )
    # ponytail: 只看写前空闲量，不预估本次载荷大小；单文件远小于下限时足够，
    # 真出现接近下限的巨型单文件时再加“预估大小 + 安全余量”的判定。


@contextmanager
def atomic_path(path: str | Path) -> Iterator[Path]:
    """我让调用方先写临时文件，成功后再原子替换目标文件。

    为什么不能直接往目标文件里写：写出过程可能被中断（异常、断电、
    被杀进程），直接写会留下半个文件。而下游的检查往往是"文件存在即
    有效"，半个 JSON 会让审计通过，实际上内容已经损坏——
    静默的假通过比明确的失败危险得多。

    实现要点：
      - 临时文件建在**目标同目录**。跨卷的 os.replace 不是原子操作，
        同目录才能保证替换是单一 rename 系统调用；
      - 用 os.replace 而不是先删再写，避免出现目标短暂不存在的窗口；
      - finally 里清理临时文件。yield 期间抛异常时临时文件会残留，
        不清理会慢慢堆满磁盘，而且这些残留文件会污染输出目录的审计。

    替换走 _replace_with_retry 而不是裸 os.replace：Windows 上
    rename 会被第三方句柄（杀毒、索引器）短暂挡掉，抛 WinError 5。
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ensure_free_space(target)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        _replace_with_retry(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
