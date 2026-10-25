from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# 我为什么钉这条：清理脚本的 `damaged*` 通配规则会删掉
# `test_step_repair_sampling.py` 当作**输入**读的三个基准产物。
# 被删之后那 3 条测试走 `pytest.skip`，套件仍然全绿——
# 覆盖率静默掉了，看不出来。所以我把"测试要读的产物"钉成不得清理项。
TEST_INPUT_RELATIVE_PATHS = (
    "step_repair_benchmark_relaxed\\测试集__10_face3\\damaged.stp",
    "step_repair_benchmark_relaxed\\测试集__11_face3\\damaged.repaired.step",
)


def test我方清理策略保护正式_BOM三件套() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "cleanup_regenerable_outputs.ps1"
    ).read_text(encoding="utf-8")
    targets, formal_targets = script.split(
        "$formalTargets = @(",
        maxsplit=1,
    )
    formal_targets = formal_targets.split(")", maxsplit=1)[0]
    for name in (
        "bom_bottle\\BOTTLE.bom.json",
        "bom_bottle\\BOTTLE.bom.csv",
        "bom_bottle\\BOTTLE.bom.xlsx",
    ):
        assert name not in targets
        assert name in formal_targets


def test我方清理策略覆盖区域与工程再生日志() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "cleanup_regenerable_outputs.ps1"
    ).read_text(encoding="utf-8")
    targets, formal_targets = script.split(
        "$formalTargets = @(",
        maxsplit=1,
    )
    formal_targets = formal_targets.split(")", maxsplit=1)[0]
    for name in (
        "stage3_regions_v3\\BOTTLE_regeneration.log",
        "stage3_regions_v3\\mould_04_regeneration.log",
        "step_engineering\\_regeneration.log",
    ):
        assert name in targets
        assert name not in formal_targets


def test我方清理策略用模式兜底覆盖改名后的再生日志() -> None:
    root = Path(__file__).parents[1]
    script = (root / "scripts" / "cleanup_regenerable_outputs.ps1").read_text(
        encoding="utf-8"
    )
    assert '-like "*regeneration*.log"' in script

    _, formal_targets = script.split("$formalTargets = @(", maxsplit=1)
    formal_targets = formal_targets.split(")", maxsplit=1)[0]
    assert ".log" not in formal_targets

    outputs = root / "outputs"
    if outputs.is_dir():
        stray = [
            item
            for item in outputs.rglob("*.log")
            if "regeneration" not in item.name.lower()
        ]
        assert not stray, f"我发现清理策略没有覆盖的日志：{stray}"


@pytest.mark.parametrize("relative", TEST_INPUT_RELATIVE_PATHS)
def test我方清理策略不删测试当作输入读的产物(relative: str) -> None:
    """我保证清理脚本不会删掉回归测试**读**的基准产物。

    清理只该删"可随时重跑"的东西。但 `test_step_repair_sampling.py`
    是把 `step_repair_benchmark_relaxed/.../damaged*` 当**输入**读的，
    删掉之后该测试走 skip，套件照样全绿，实际却少测了真实产物上的守卫。
    我把这些路径钉进保护名单，让脚本一旦想删就先红。
    """

    script_text = (
        ROOT / "scripts" / "cleanup_regenerable_outputs.ps1"
    ).read_text(encoding="utf-8")
    targets = script_text.split("$formalTargets = @(", maxsplit=1)[0]
    formal_block = script_text.split("$formalTargets = @(", maxsplit=1)[1].split(
        ")", maxsplit=1
    )[0]

    # 我按脚本的匹配语义判断：只要进 $formalTargets 就算受保护。
    # 光看 targets 名单不够——damaged* 是**通配规则**，不在名单里，
    # 所以真正的判据是"有没有被显式保护"。
    assert relative not in targets, (
        f"清理脚本直接列了测试输入 {relative}，测试会静默 skip"
    )
    assert relative in formal_block, (
        f"测试输入 {relative} 没有进 $formalTargets 保护名单，"
        "damaged* 通配规则会把它删掉，对应测试将静默 skip"
    )


@pytest.mark.parametrize("relative", TEST_INPUT_RELATIVE_PATHS)
def test我方测试输入产物确实存在(relative: str) -> None:
    """我盯住这些被依赖的产物别消失。

    上一条保证脚本"不许删"，这一条保证它们"确实在"。
    两者叠起来，测试才不会以 skip 的方式静默失效。
    """

    path = ROOT / "outputs" / relative
    if not path.exists():
        pytest.skip(
            f"基准产物不存在，先跑 scripts\\benchmark_step_repair.py：{relative}"
        )
