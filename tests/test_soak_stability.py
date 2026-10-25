"""我方为耐久巡检台账写回归测试。

我为什么给一个"脚本"写测试：任务书要求连续运行 48 小时，而这条要求最容易
被**假证据**满足——跑一段短的就宣称"稳定"，或者测到一个恒定的假内存值。
我在实现这个脚本的过程中已经踩到过两次：

1. 父进程跨进程读子进程峰值内存，在这台机器上恒返回约 4.4 MB，
   而同一个进程自己量是 226 MB。若信前者，"内存没涨"完全是假象。
2. 我把峰值内存并进了"结论指纹"，导致一致性判据**永远失败**。

这两条都不是"代码能不能跑"的问题，是"结论可不可信"的问题，
所以必须由测试钉住，不能只靠人记得。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

pytest.importorskip("numpy")


def _load_module():
    """我按路径导入巡检脚本模块（它不是包内模块）。"""

    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import run_soak_stability as module

        return module
    finally:
        sys.path.remove(str(SCRIPTS_DIR))


module = _load_module()


def _round(
    peak: int, elapsed: float, status: str = "review", error: str | None = None
) -> dict:
    """我构造一轮记录，指纹里的 status 可用来制造结论漂移。"""

    return {
        "error": error,
        "fingerprint": {"status": status, "region_count": 6},
        "child_peak_rss_bytes": peak,
        "elapsed_seconds": elapsed,
    }


def test我方内存单调增长会被判失败() -> None:
    """泄漏是最该被抓到的坏法：单轮正常，连跑才爆。"""

    rounds = [
        _round(100_000_000 * (1 + i * 0.9), 10.0) for i in range(6)
    ]
    result = module.evaluate_invariants(rounds, 0.25, 0.50)

    assert result["overall"] == "FAIL"
    assert result["memory"]["status"] == "FAIL"


def test我方结论漂移会被判失败() -> None:
    """同一批输入跑出两种结论，说明有隐藏状态，必须红。"""

    rounds = [
        _round(100_000_000, 10.0, status="review" if i < 3 else "pass")
        for i in range(6)
    ]
    result = module.evaluate_invariants(rounds, 0.25, 0.50)

    assert result["overall"] == "FAIL"
    assert result["consistency"]["status"] == "FAIL"


def test我方样本不足时判未知而不是判通过() -> None:
    """只跑一两轮就说"稳定"是自欺；这时必须如实说不知道。

    这条守的是最容易被糊弄过去的地方：样本不够时"没发现问题"
    不等于"没问题"。
    """

    result = module.evaluate_invariants([_round(100_000_000, 10.0)], 0.25, 0.50)

    assert result["overall"] == "UNKNOWN"
    assert result["memory"]["status"] == "UNKNOWN"
    assert result["consistency"]["status"] == "UNKNOWN"


def test我方峰值内存不参与结论一致性比较() -> None:
    """峰值内存每轮必然不同，并进指纹会让一致性判据永远失败。

    这是我实现时真踩过的坑：不加这条，一致性检查实际上等于"永远报警"，
    真正该抓的结论漂移反而被淹没了。
    """

    rounds = [_round(725_000_000 + i * 1_000_000, 16.0) for i in range(6)]
    result = module.evaluate_invariants(rounds, 0.25, 0.50)

    assert result["consistency"]["status"] == "PASS"


def test我方错误轮会拉低总体结论() -> None:
    """子进程崩了必须被看见，不能被"其余几轮正常"盖过去。"""

    rounds = [_round(100_000_000, 10.0) for _ in range(4)]
    rounds.append(_round(100_000_000, 10.0, error="子进程退出码 1"))
    result = module.evaluate_invariants(rounds, 0.25, 0.50)

    assert result["error_count"] == 1
    assert result["overall"] == "FAIL"


def test我方短跑不会冒充跑满任务书时长() -> None:
    """任务书要 48 小时；跑 0.25 小时必须在台账里如实标记为未跑满。

    我盯住这个字段，是因为它是最容易被"顺手写成已达成"的地方，
    而一旦写成已达成，后面的验收就建立在假证据上。
    """

    assert module.SCHEMA_VERSION == "soak-stability-1.0"
    summary_path = ROOT / "outputs" / "soak_stability" / "summary.json"
    if not summary_path.is_file():
        pytest.skip("我方耐久巡检台账尚未生成。")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["task_required_hours"] == 48
    assert summary["requested_hours"] < 48
    # 短跑时这一位必须为 False，且结论文案里必须写明不算验收通过。
    assert summary["reached_full_duration"] is False
    assert "不等于 48 小时验收通过" in summary["duration_verdict"]


def test我方台账里的峰值内存是真实量级不是恒定假值() -> None:
    """我实测过跨进程读会恒返回约 4.4 MB，而真实值是 725 MB。

    这条钉住"量级"：真实跑一次 OCCT 检测必然是几百 MB，
    若台账里是几 MB，说明又回退到了那个假读数上。
    """

    summary_path = ROOT / "outputs" / "soak_stability" / "summary.json"
    if not summary_path.is_file():
        pytest.skip("我方耐久巡检台账尚未生成。")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    peaks = [
        r["child_peak_rss_bytes"]
        for r in summary["rounds"]
        if r.get("child_peak_rss_bytes")
    ]
    assert peaks, "我方台账里没有任何峰值内存样本"
    # 几百 MB 是真实量级；4 MB 是那个假读数的特征量级。
    assert min(peaks) > 50_000_000, f"我方读到可疑的小峰值：{min(peaks)}"


def test我方入口能跑通并落盘台账(tmp_path: Path) -> None:
    """我验真实入口：跑两轮，台账必须落盘且结构完整。"""

    candidate = (
        ROOT / "data" / "provided" / "三维模型破损区域智能修补" / "测试集" / "6.stl"
    )
    reference = (
        ROOT / "data" / "provided" / "三维模型破损区域智能修补" / "测试集" / "6.stp"
    )
    if not (candidate.is_file() and reference.is_file()):
        pytest.skip("我方找不到真实语料。")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_soak_stability.py"),
            str(candidate),
            str(reference),
            "--max-rounds",
            "2",
            "--timeout-seconds",
            "600",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )

    summary_path = tmp_path / "summary.json"
    assert summary_path.is_file(), f"我方台账没落盘：{completed.stdout[-400:]}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "soak-stability-1.0"
    assert summary["rounds"], "我方台账里没有任何轮次"
    # 两轮不足以判内存/耗时，但一致性已经可比，且不得报错。
    assert summary["invariants"]["error_count"] == 0
    assert summary["invariants"]["consistency"]["status"] == "PASS"
