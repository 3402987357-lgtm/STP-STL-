from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_run_tests_module():
    path = Path(__file__).parents[1] / "scripts" / "run_tests.py"
    spec = importlib.util.spec_from_file_location("project_run_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 我为什么立这条护栏（2026-10-11）：我修过一个**偶发测试失败**，根因是
# `test_step_repair_sampling.py` 里 8 个测试**共用固定输出路径**
# `outputs/_probe/...`，而不是 pytest 的 `tmp_path`。并发跑同一个测试文件时，
# 一个进程刚 `unlink()`、另一个正好在断言 `out.exists()`，就偶发转红。
# 我把那 8 处改成 `tmp_path` 了，但**改完没有护栏，将来还会有人写回去**。
# 所以这里把"不许再往 outputs/ 下写固定路径"钉成硬不变量。
#
# 白名单里的文件是**排查后确认无同类隐患**的：它们要么只读已有产物做契约
# 校验、要么测的是清理/审计脚本本身、要么写的不是修补输出。新增文件若要用
# 固定路径，**必须在这里显式登记并写明理由**——这正是护栏的价值。
ALLOWED_FIXED_OUTPUT_PATH_FILES = {
    "test_acceptance_face_count.py",   # 只读 acceptance_v10/summary.json
    "test_artifact_audit.py",          # 审计逻辑，不写修补输出
    "test_cleanup_policy.py",          # 测清理脚本本身
    "test_desktop_ui.py",              # 写 outputs/_probe/ui，非修补输出
    "test_flowchart.py",               # 写 outputs/flowcharts/
    "test_formal_output_contract.py",  # 只读已有产物做契约校验
    "test_step_repair_baseline.py",    # glob 读基准 summary
}


def _USES_CWD_RELATIVE_PROJECT_PATH(stripped: str) -> bool:
    """我判断这行是否在用**相对当前工作目录**的项目路径。

    判定要点：`Path("data")` / `"data/..."` / `"src/..."` 这类写法的解析
    基准是 CWD，换个目录执行就找不到。已经锚定的 `ROOT / "src/..."` 或
    `DATA_ROOT / "provided"` 不算——它们前面有变量，不依赖 CWD。
    """

    # 我为什么用"任意位置出现"而不是 startswith（2026-10-14 补的漏判）：
    # 原来只查行首，于是 `a = Path("data") / "provided"` 这种最常见的
    # 写法**检测不到**——赋值前缀把 Path( 顶到了行中间。
    # 那条守卫因此形同虚设：它只会报一行行首就写 Path("data" 的代码，
    # 而真实代码几乎总是先赋值给变量。
    # 这里要求 Path( 前面不是 `/` 或标识符，放过 ROOT / Path("data") 之类。
    for literal in ('Path("data', 'Path("src', 'Path("scripts'):
        position = stripped.find(literal)
        while position != -1:
            before = stripped[:position].rstrip()
            if before and (before[-1].isidentifier() or before.endswith("/")):
                # 形如 `DATA_ROOT / Path("data")`、或 `myPath("data")`，跳过。
                position = stripped.find(literal, position + 1)
                continue
            return True
    for literal in ('"data/', '"src/', '"scripts/'):
        # 放过 `ROOT / "src/..."`：`/` 前面是变量而不是字符串开头。
        if literal in stripped and not _is_anchored(stripped, literal):
            return True
    return False


def _is_continuation_of_anchored(lines: list[str], index: int) -> bool:
    """我判断这一行是否是被锚定表达式**拆行**出来的续行。

    我为什么要单独处理（2026-10-14 修自己的假红）：这个守卫原来是
    **逐行独立**判断的，于是下面这种完全正确的写法会被误判——

        return (
            ROOT
            / "data"          <- 这一行单独看是 `Path("data" 的样子
            / "provided"
        )

    锚 `ROOT` 在**上一行**，逐行判断看不见它。假红的代价是
    逼人把正确代码改歪，比漏判更坏，所以我向上回溯找锚：
    只要本行以 `/` 或 `)` 开头，就往上找最近的、以变量名开头的行。
    """

    current = lines[index].strip()
    # 形如 `("data/..."` 的实参行也要算续行：字符串作为调用实参独占一行时，
    # 它的锚（`Path(__file__).resolve().parents[1].glob(`）在上一行，
    # 逐行判断看不见，会被误报成"依赖 CWD"。
    if not current.startswith(("/", ")", "]", "}", '"', "'")):
        return False
    # 向上回溯，直到遇到看起来像表达式开头的行。
    for back in range(index - 1, max(index - 6, -1), -1):
        previous = lines[back].strip()
        if not previous or previous.startswith("#"):
            continue
        # 行以 `(` 结尾 = 实参列表未闭合，本行就是它的实参，锚已成立。
        if previous.endswith("("):
            return True
        # 遇到另一个续行就继续往上（多行拼接的常见形态）。
        if previous.startswith(("/", ")", "]", "}")):
            continue
        # 到这里是表达式开头的那一行：以标识符/变量开头即视为已锚定。
        head = previous.split("(")[-1].strip()
        return bool(head) and (
            head[0].isidentifier() or head.startswith("Path(")
        )
    return False


def _is_anchored(stripped: str, literal: str) -> bool:
    """我判断该字面量是否挂在某个变量/路径对象后面。"""

    index = stripped.find(literal)
    if index <= 0:
        return False
    before = stripped[:index].rstrip()
    # 形如 `ROOT / `、`DATA_ROOT / `、`(ROOT / `——末尾是路径拼接或变量名。
    return before.endswith("/") or before[-1:].isalnum() or before.endswith(")")


def test我方测试不再往固定输出路径写文件() -> None:
    """我保证测试用 tmp_path，而不是共用的固定目录。

    我只查"看起来像往 outputs/ 下写路径"的字面量。这是**保守**的检查：
    宁可漏报，也不误伤——它拦的是最常见的写法（Path("outputs") / ...）。
    """

    tests_dir = Path(__file__).parent
    offenders: list[str] = []
    for path in sorted(tests_dir.glob("*.py")):
        # 我跳过自身：护栏自己的注释和代码里必然出现这些字样，会误报。
        if path.name in ALLOWED_FIXED_OUTPUT_PATH_FILES:
            continue
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        for index, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # 我只拦"**写**"的用法：`output_path=...` 或 `_probe` 目录。
            #
            # 我第一版写得更宽（只要出现 outputs + Path( 就拦），结果**误报**：
            # `test_step_repair_sampling.py` 有 3 处是**读**已有基准产物
            # （`outputs/step_repair_benchmark_relaxed/.../damaged.stp`）当输入，
            # 那不是并发写竞争。所以这里收紧到"写"这个动作上。
            is_write = ("output_path" in stripped) or ("_probe" in stripped)
            if "outputs" in stripped and is_write:
                offenders.append("%s:%d: %s"
                                 % (path.name, index, stripped[:100]))
    assert not offenders, (
        "我方发现测试又在用固定输出路径（应改用 tmp_path，否则并发跑同一个\n"
        "测试文件时会互相删除文件、偶发转红）。确有特殊理由的，请把它加进\n"
        "test_run_tests_guard.py 的白名单并写明原因：\n"
        + "\n".join(offenders)
    )


def test我方测试入口清理项目内过期字节码(tmp_path: Path) -> None:
    run_tests = _load_run_tests_module()
    run_tests.ROOT = tmp_path

    caches = []
    for relative in ("src", "tests", "scripts"):
        cache_dir = tmp_path / relative / "pkg" / "__pycache__"
        cache_dir.mkdir(parents=True)
        (cache_dir / "stale.cpython-312.pyc").write_bytes(b"stale")
        caches.append(cache_dir)

    outside = tmp_path.parent / f"{tmp_path.name}-outside" / "__pycache__"
    outside.mkdir(parents=True)
    try:
        removed = run_tests._purge_stale_bytecode()

        assert removed == len(caches)
        for cache_dir in caches:
            assert not cache_dir.exists()
        assert outside.exists()
    finally:
        if outside.exists():
            for child in outside.iterdir():
                child.unlink()
            outside.rmdir()
            outside.parent.rmdir()

def test我方测试不依赖当前工作目录() -> None:
    """我保证测试从**任何**目录执行都能找到项目文件。

    我为什么立这条（2026-10-14）：我把一批测试从仓库根之外执行，
    发现 7 条用例因为写了相对路径（`Path("data")` / `"data/..."` /
    `"src/..."`）而**静默 skip**——套件全绿，覆盖率却掉了。
    更糟的是其中 3 条不是 skip 而是直接 FileNotFoundError，
    在根目录跑永远发现不了。项目文件一律用 `Path(__file__)` 定位。
    """

    tests_dir = Path(__file__).parent
    offenders: list[str] = []
    for path in sorted(tests_dir.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # 已锚定表达式的续行（锚在上一行）不算依赖 CWD。
            if _is_continuation_of_anchored(lines, index - 1):
                continue
            # 我只拦"字面量相对路径"这一种写法：它解析基准是 CWD 而不是文件位置。
            # 只有"直接以相对路径开头"才算；`ROOT / "src/..."` 这种已锚定的写法放过。
            if _USES_CWD_RELATIVE_PROJECT_PATH(stripped):
                offenders.append("%s:%d: %s" % (path.name, index, stripped[:100]))
    assert not offenders, (
        "我方发现测试在用相对路径定位项目文件，从非仓库根执行时会静默"
        "skip 或直接 FileNotFoundError。请改用 "
        "ROOT = Path(__file__).resolve().parents[1]：\n"
        + "\n".join(offenders)
    )
