"""我为 OCCT 三角化与 CAD 面 ID 映射写回归测试。

我要钉住的核心契约是**面 ID 映射逐行对齐**：检测在三角网格上算，
甲方看的工程语义在 B-Rep 面上，映射错位的话"缺失落在第 17 号面"
这句话就是错的，而错得悄无声息。

所以我用一个独立算出来的量（逐面面积之和 vs 网格总面积）做交叉校验：
这两个数由不同代码路径算出，相等才说明映射和几何都对。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("OCP")

from cad_inspector.geometry.occt_tessellation import (  # noqa: E402
    TessellationConfig,
    tessellate_step_file,
)

# 我为什么按仓库根定位：这里原本写相对路径 "data/..."，
# 只在"当前工作目录恰好是仓库根"时才找得到，别处执行 pytest 会静默 skip。
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


@pytest.fixture(scope="module")
def tessellated():
    """我只三角化一次：OCP 读 STEP 要十几秒，多个用例复用同一份。"""

    path = str(DATA_ROOT / "provided" / "三维模型破损区域智能修补" / "测试集" / "6.stp")

    if not os.path.exists(path):
        pytest.skip("甲方提供的 STEP 不存在。")
    return tessellate_step_file(path, TessellationConfig(linear_deflection=0.5))


def test我方映射长度与三角面数一一对应(tessellated) -> None:
    """映射少一行或多一行，后面的"属于哪个面"全部错位。"""

    assert len(tessellated.triangle_to_face_id) == len(tessellated.mesh.faces)


def test我方每个CAD面都有面积与三角计数(tessellated) -> None:
    tessellated_result = tessellated

    assert len(tessellated_result.face_areas_mm2) > 0
    assert set(tessellated_result.face_areas_mm2) == set(
        tessellated_result.face_triangle_counts
    )
    assert all(value > 0 for value in tessellated_result.face_areas_mm2.values())
    assert all(count > 0 for count in tessellated_result.face_triangle_counts.values())


def test我方逐面面积之和与网格总面积一致(tessellated) -> None:
    """交叉校验：两条独立代码路径算出的面积必须相等。

    face_areas 是我在建映射时逐三角累加的，
    mesh.total_area 是 TriangleMesh 自己算的。
    二者相等才说明映射没漏面、也没重复计面。
    """

    summed = sum(tessellated.face_areas_mm2.values())

    assert summed == pytest.approx(tessellated.mesh.total_area, rel=1e-9)


def test我方三角计数之和等于三角面总数(tessellated) -> None:
    assert sum(tessellated.face_triangle_counts.values()) == len(
        tessellated.mesh.faces
    )


def test我方CAD面ID从1连续编号(tessellated) -> None:
    """连续编号让"第 N 号面"在报告和复核时指向同一个面。

    我不用 id() 或 OCCT HashCode 当 ID，因为它们跨进程/跨版本会变，
    那样上次报的第 17 号面这次就对不上了。
    """

    unique = sorted(set(int(value) for value in tessellated.triangle_to_face_id))

    assert unique == list(range(1, len(unique) + 1))


def test我方同一文件两次三角化结果稳定() -> None:
    """面 ID 必须可复现，否则报告没法复核。"""

    import os

    path = str(DATA_ROOT / "provided" / "三维模型破损区域智能修补" / "测试集" / "6.stp")
    if not os.path.exists(path):
        pytest.skip("甲方提供的 STEP 不存在。")
    first = tessellate_step_file(path)
    second = tessellate_step_file(path)

    assert np.array_equal(first.triangle_to_face_id, second.triangle_to_face_id)
    assert np.allclose(first.mesh.vertices, second.mesh.vertices)


def test我方顶点全部有限(tessellated) -> None:
    """NaN 顶点会让下游距离全变 NaN，静默毁掉整份结论。"""

    assert np.all(np.isfinite(tessellated.mesh.vertices))


def test我方配置写进结果便于复现(tessellated) -> None:
    payload = tessellated.to_dict()

    assert payload["config"]["linear_deflection"] == 0.5
    assert payload["cad_face_count"] == len(tessellated.face_areas_mm2)
    assert isinstance(payload["warnings"], list)


def test我方报告可JSON序列化(tessellated) -> None:
    import json

    text = json.dumps(tessellated.to_dict(), ensure_ascii=False, allow_nan=False)

    assert "face_areas_mm2" in text
