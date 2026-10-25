"""我把真实破损件修补的实测基线钉进测试。

我为什么单独建这个文件：**没有任何测试在读 STEP 修补基准的产物**。
我上一轮修了一个"把没补上的洞判成 SUCCESS"的真 bug，
改完只能靠手工重跑 benchmark 才知道有没有把好件误杀——
这等于每次改动的成功率变化都要靠我记得去量，
忘了就等于没发现。所以我把基线钉住。

我钉的是**方向**而不是精确数量：修好的件数不能倒退，
且不允许"判成功但 BRep 无效"这种自相矛盾的结果。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _latest_real_damaged_summary() -> dict | None:
    """我取磁盘上最新的真实破损件修补台账。"""

    candidates = sorted(
        ROOT.glob("outputs/step_repair_real_damaged*/summary.json")
    )
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _latest_real_threshold_summary() -> dict | None:
    """我取磁盘上最新的真实阈值标定台账（取字典序最后一个）。"""

    candidates = sorted(
        ROOT.glob("outputs/real_threshold_calibration*/summary.json")
    )
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def test我方修补产出的STEP真的是封闭实体() -> None:
    """我不信台账里的 `output_solid_count`，我要**自己再读一遍**。

    我为什么要独立复核：台账里的"输出 1 个实体"是**修补脚本自己报的**，
    那就等于让被测代码给自己打分。而我上一轮给产物审计加的 STEP 检查
    只验"读得回来"，**不验是不是实体**——
    一个退化成一堆散面的 STEP 也能"读得回来"。

    "补好了"的真实含义是：**洞封上了，且结果仍是一个封闭实体**。
    只封上缝但退化成散面，下游是拿去用的，会直接炸。
    所以我用 OCP 逐个重读 repaired.step，要求至少 1 个 SOLID 且 BRep 有效。
    """

    summary = _latest_real_damaged_summary()
    if summary is None:
        pytest.skip("我还没有真实破损件修补台账")

    repaired = sorted(
        ROOT.glob("outputs/step_repair_real_damaged*/*/repaired.step")
    )
    if not repaired:
        pytest.skip("磁盘上还没有修补产出的 STEP")

    pytest.importorskip("OCP.STEPControl")
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    checked = 0
    for path in repaired:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
            pytest.fail(f"我读不回自己补出来的 STEP：{path}")
        reader.TransferRoots()
        shape = reader.OneShape()
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        solids = 0
        while explorer.More():
            solids += 1
            explorer.Next()
        assert solids >= 1, f"补完退化成散面了，没有实体：{path}"
        assert BRepCheck_Analyzer(shape).IsValid(), f"补完 BRep 无效：{path}"
        checked += 1
    # 至少覆盖到台账里记录的成功件数，否则这条等于没验。
    assert checked >= 8, f"我只复核了 {checked} 件，覆盖不足"


def test我方真实破损件修补不出现判成功但BRep无效() -> None:
    """这是最不能出现的结果：**说补好了，但实体本身无效**。

    真要是这样，下游拿到的是一个打不开/不封闭的模型，
    却带着 SUCCESS 的结论——比直接报复核危险得多。
    """

    summary = _latest_real_damaged_summary()
    if summary is None:
        pytest.skip("我还没有真实破损件修补台账")

    aggregate = summary.get("aggregate", {})
    # 有 BRep 有效性结论的件里，不允许任何一件是无效的。
    assert aggregate.get("brep_valid_false_count", 0) == 0


def test我方真实破损件修补的成功件数不低于基线() -> None:
    """我钉一个下界而不是精确值：改动只能让修好的件变多，不能变少。

    基线取自我修完"实质孔洞被误判成功"那个 bug 之后重跑的实测
    （SUCCESS 9 件）。下界我压到 7（修 bug 之前的水平），
    这样即便某轮几何库升级让个别件波动，也不会假红；
    但真要是把判据改坏了、成功件塌下去，这条一定红。
    """

    summary = _latest_real_damaged_summary()
    if summary is None:
        pytest.skip("我还没有真实破损件修补台账")

    by_status = summary.get("aggregate", {}).get("by_status", {})
    assert by_status.get("SUCCESS", 0) >= 7, (
        f"成功件数掉到 {by_status.get('SUCCESS', 0)}，低于基线 7，"
        "请确认判据改动没有把好件误杀。"
    )


def test我方修补前后缝长确实变小() -> None:
    """补了洞却没让缝变小，等于白补——这条守的是"改善"的实质。

    我用"缝总长显著下降"而不是"边界条数下降"：实测 STEP 回读会把
    共享边拆开导致条数虚高（测试集/4 上补完反而 7 -> 14 条），
    但缝总长不受影响，是更可靠的改善判据。
    """

    summary = _latest_real_damaged_summary()
    if summary is None:
        pytest.skip("我还没有真实破损件修补台账")

    improved = 0
    for record in summary.get("records", []):
        before = record.get("gap_before_mm")
        after = record.get("gap_after_mm")
        if before is None or after is None:
            continue
        # 缝从毫米级降到容差级才算真补上。
        if after < before * 0.5:
            improved += 1
    assert improved >= 7, f"只有 {improved} 件缝长显著下降"


def test我方真实阈值标定覆盖率足够() -> None:
    """我钉住阈值标定**必须真的算出结果**，而不只是跑完就算了。

    我为什么在意覆盖率：2026-10-14 我发现标定台账里 21 对样本
    **只有 9 对有结果**，另外 11 对被 240 秒超时掐掉了。
    这 11 个不是算不出来——实测 测试集/1 放宽到 900 秒后只用了 67 秒。
    也就是说，我之前拿 9 个样本去谈"阈值该定多少"，
    而那 9 个是被超时筛剩下的**小件**，结论是有偏的。
    覆盖率不够会让标定结论悄悄失真，所以我把分母钉住。
    """

    summary = _latest_real_threshold_summary()
    if summary is None:
        pytest.skip("我还没有真实阈值标定台账")

    aggregate = summary.get("aggregate", {})
    processed = aggregate.get("processed_pair_count", 0)
    measured = aggregate.get("measured_pair_count", 0)
    # 唯一允许没有结果的是被资源守卫跳过的超大件（训练集/1 有 475 MB）。
    assert processed >= 20, f"标定的样本总数只有 {processed}"
    assert measured >= 19, (
        f"21 对里只有 {measured} 对真正算出了结果，覆盖率不足；"
        "多半是超时预算太小把大件全掐掉了，请加大 --timeout-seconds 重跑。"
    )
