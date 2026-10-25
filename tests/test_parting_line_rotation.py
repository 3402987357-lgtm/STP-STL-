"""我方网格分型线评分的旋转不变性回归。

背景：网格分型线候选此前用 ``abs(_signed_area_xy(points))`` 评分，
它是环在 XY 平面上的投影面积。模型整体倾斜摆放时，投影面积等于真实面积
乘 cos(环平面与 XY 的夹角)，环平面垂直于 XY 时会退化到 0，
于是真正的外轮廓会被判成"面积为零"的垃圾候选。我改成环自身拟合平面上的
真实围成面积后，本文件锁定"同一条环绕任意轴旋转，评分不变"这个性质。
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from cad_inspector.detection.parting_line import _loop_score


def test我方两条分型路线可以独立导入不产生循环依赖() -> None:
    """我证明两条分型路线先导入谁都不会炸。

    回归背景：网格路线为复用真实面积口径要导入 step_parting，而
    step_parting 又依赖网格路线。若把这个导入写在模块顶层就会形成
    循环导入，整个 cad_inspector 包不可用。我把它锁在子解释器里跑，
    因为同一个进程内模块缓存会掩盖循环导入问题。
    """

    import subprocess
    import sys
    from pathlib import Path

    # 子解释器默认不继承测试进程的 sys.path，我显式带上 src，
    # 否则无论有没有循环依赖都会先报 ModuleNotFoundError，测试就失去意义。
    src_dir = Path(__file__).resolve().parents[1] / "src"
    environment = {**os.environ, "PYTHONPATH": str(src_dir)}
    for first in ("cad_inspector.detection.parting_line", "cad_inspector.detection.step_parting"):
        completed = subprocess.run(
            [sys.executable, "-B", "-c", f"import {first}; print('ok')"],
            capture_output=True,
            text=True,
            env=environment,
        )
        assert completed.returncode == 0, (
            f"先导入 {first} 失败：{completed.stderr}"
        )
        assert "ok" in completed.stdout


def _rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    """我返回一个绕指定坐标轴旋转的 3x3 矩阵。"""

    angle = np.deg2rad(degrees)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    if axis == "x":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
        )
    if axis == "y":
        return np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
        )
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("degrees", [0.0, 30.0, 45.0, 90.0, 137.5, 180.0])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test我方闭环评分在任意旋转下保持不变(axis: str, degrees: float) -> None:
    """我证明同一条闭环绕任意轴旋转后评分不变。

    回归背景：修复前的 XY 投影口径在 90 度附近会把评分打到 0，
    导致"最大外轮廓"排序选错候选，这是本用例真正要防的失效模式。
    """

    square = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=np.float64,
    )
    rotated = square @ _rotation_matrix(axis, degrees).T
    baseline = _loop_score(square)
    assert baseline == pytest.approx(100.3, abs=1e-6)
    assert _loop_score(rotated) == pytest.approx(baseline, rel=1e-9)


def test我方旧口径在竖直环上会坍塌而新口径不会() -> None:
    """我直接对照新旧口径，锁定这个修复确实改变了结果。

    我把方形环绕 X 轴转 90 度变成竖直环：XY 投影面积归零，
    而环自身平面上的真实围成面积仍然是 100。
    """

    square = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=np.float64,
    )
    vertical = square @ _rotation_matrix("x", 90.0).T
    xy = vertical[:, :2]
    old_area = abs(
        float(0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
    )
    assert old_area == pytest.approx(0.0, abs=1e-9)
    # 新口径里面积项仍是 100；周长按 _polyline_length 的定义是相邻点距离之和
    # （不补首尾闭合段），四点方形环为 30，故评分为 100 + 0.01 x 30。
    assert _loop_score(vertical) == pytest.approx(100.0 + 0.01 * 30.0, abs=1e-6)


def test我方周长不自动闭合而面积自动闭合() -> None:
    """我锁定这两个助手函数的"是否闭合"语义差异。

    这两个函数对首尾的处理是**故意相反**的，混用会静默算错：
      - 周长不补首尾连线：调用方可能传开口链，自动补一段
        会把"开口"伪装成"闭合"并多出一段不存在的边；
      - 面积必须首尾相接：开口折线围不出面积，不补就无法计算。
    我直接用同一组四点验证两边，避免将来有人"统一"掉这个差异。
    """

    from cad_inspector.detection.step_parting import (
        _polyline_length,
        _signed_area_xy,
    )

    square = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    # 开口折线：长度为三条边之和 3，面积仍按闭合算出 1。
    assert _polyline_length(square) == pytest.approx(3.0, abs=1e-12)
    assert _signed_area_xy(square) == pytest.approx(1.0, abs=1e-12)

    closed = np.vstack([square, square[0]])
    # 显式闭合后：长度变成 4，面积不变（不会因首尾重合而多算）。
    assert _polyline_length(closed) == pytest.approx(4.0, abs=1e-12)
    assert _signed_area_xy(closed) == pytest.approx(1.0, abs=1e-12)
