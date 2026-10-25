"""我方配准区域的用例。

我为什么要补这个文件：我用 rg 全库查过，`registration/rigid.py`
**没有任何测试引用它**，但 pipeline、visualization、missing_regions 都在
调它。配准是后续一切偏差判定的地基——两个模型没对齐，再好的距离算法也
只会报出假缺陷，而 visualize 的副标题正是靠配准残差来告诉看图的人
"这个红框有多可信"。一个在关键链路上、却零覆盖的模块，风险不该留着。
"""

from __future__ import annotations

import numpy as np
import pytest

from cad_inspector.config import InspectionConfig
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.registration.rigid import align_mesh
from test_core import cube_mesh


def _shifted_cube(offset=(50.0, -20.0, 10.0)) -> TriangleMesh:
    """我造一个整体平移过的立方体，平移量已知。"""

    mesh = cube_mesh()
    return TriangleMesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64)
        + np.asarray(offset, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )


def _rotated_cube(angle_rad: float = 0.25) -> TriangleMesh:
    """我造一个**旋转过**的立方体，旋转量已知。

    我为什么不能用 `_shifted_cube` 来做 ICP 那条用例：
    纯平移的情形下 centroid 一步就完全解掉（残差降到 0），
    ICP 相对于它**没有任何额外改善**，于是那条
    "ICP 明显降低残差"的告警压根不会触发——
    测试会假通过，而真实风险没被钉住。
    加了旋转之后 centroid 解决不了（实测残差 1.76mm），
    ICP 能把它压到 0，正好复现"ICP 额外抹平了一大片"的实测场景。
    """

    mesh = cube_mesh()
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return TriangleMesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T
        + np.asarray([2.0, 1.0, 0.5], dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )


@pytest.mark.parametrize("mode", ["none", "centroid", "pca", "icp"])
def test我方四种配准模式都产出合法质量报告(mode) -> None:
    """我保证四种模式都不崩，且质量分级落在约定取值内。"""

    config = InspectionConfig(registration_mode=mode)
    aligned, result = align_mesh(_shifted_cube(), cube_mesh(), config)
    assert result.mode == mode, result.mode
    assert result.quality in {"good", "warning", "bad"}, result.quality
    assert result.mean_residual_mm >= 0.0, result.mean_residual_mm


def test我方ICP在明显降低残差时必须提示可能吃掉缺陷() -> None:
    """我钉住 ICP 那条"可能把缺陷对齐掉"的告警。

    我为什么专门钉这个（2026-10-15 实测发现的真风险）：
    ICP 把残差从 0.368 降到 0.153，看着是"配准更准了"，
    但**在破损检测里这可能是漏检**。ICP 最小化整体点集距离，
    候选件缺料时它会把模型挪向缺口来降低总残差，
    于是缺口被抹平。我实测确认过：
      测试集/2（B-Rep 有效性检查**失败**，确有破损）
        centroid -> missing=112.1 mm2，regions=3
        icp      -> missing=0，regions=0   ← **真缺陷被吃掉**
      测试集/1（台账实测真实孔洞 0.761872 mm2）
        centroid -> missing=153.9 mm2
        icp      -> missing=0.2 mm2
    没有这条告警，甲方会以为 ICP 是更好的默认选择。
    """

    config = InspectionConfig(registration_mode="icp")
    aligned, result = align_mesh(_rotated_cube(), cube_mesh(), config)

    assert result.mode == "icp", result.mode
    # 我先把"ICP 确实明显改善了残差"钉住：否则下面那条
    # 断言会通过得毫无意义——残差没降，告警本就不该出现。
    _, centroid_result = align_mesh(
        _rotated_cube(), cube_mesh(), InspectionConfig(registration_mode="centroid")
    )
    assert centroid_result.mean_residual_mm > 1.0, (
        f"这个用例里 centroid 残差只有 {centroid_result.mean_residual_mm}，"
        "不足以构成'ICP 明显改善'的前提，用例本身失效了。"
    )
    assert any("ICP" in warning for warning in result.warnings), (
        f"ICP 明显降低了残差却没有提示'可能吃掉缺陷'：{result.warnings}"
    )
    # 顺手钉住配准确实生效：对齐后质心应回到参考立方体附近。
    assert np.allclose(
        np.asarray(aligned.centroid), np.asarray(cube_mesh().centroid), atol=1.0
    ), aligned.centroid


def test我方质心配准不会挂上ICP那条告警() -> None:
    """我钉住那条告警**只**属于 ICP，不会误挂到别的模式上。

    我为什么加这条：告警本身是"ICP 特有风险"的提示，
    挂到 centroid 上就是**错误消息**——centroid 只平移不旋转，
    根本不会去迁就缺口。如果以后有人重构把它挪到了共用位置，
    这条会立刻红。
    """

    for mode in ("none", "centroid", "pca"):
        config = InspectionConfig(registration_mode=mode)
        aligned, result = align_mesh(_shifted_cube(), cube_mesh(), config)
        assert not any("ICP" in warning for warning in result.warnings), (
            f"{mode} 模式被误挂了 ICP 专属告警：{result.warnings}"
        )
    assert result.p95_residual_mm >= 0.0, result.p95_residual_mm
    assert aligned is not None


def test我方质心配准真的把已知平移纠正掉() -> None:
    """我保证配准是在干活，不是返回恒等变换糊弄过去。

    这是我刻意加的实质检查：只断言"四种模式不崩"的话，一个永远返回
    恒等变换的实现也能全部通过。所以这里给一个已知 50mm 级平移，
    质心配准后平均残差必须显著下降。
    """

    reference = cube_mesh()
    candidate = _shifted_cube()
    config_none = InspectionConfig(registration_mode="none")
    config_centroid = InspectionConfig(registration_mode="centroid")
    _, before = align_mesh(candidate, reference, config_none)
    _, after = align_mesh(candidate, reference, config_centroid)
    # 平移量 50/-20/10 的模长约 55mm，不配准时残差应当就是这个量级。
    assert before.mean_residual_mm > 10.0, before.mean_residual_mm
    assert after.mean_residual_mm < before.mean_residual_mm * 0.01, (
        before.mean_residual_mm,
        after.mean_residual_mm,
    )


def test我方拒绝空网格而不是静默返回恒等变换() -> None:
    """我保证空网格抛错。

    静默退回恒等变换会把"没配准"伪装成"配准成功"——这正是源码注释里
    明写要避免的边界。
    """

    empty = TriangleMesh(
        vertices=np.zeros((0, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    config = InspectionConfig(registration_mode="centroid")
    with pytest.raises(ValueError):
        align_mesh(empty, cube_mesh(), config)
    with pytest.raises(ValueError):
        align_mesh(cube_mesh(), empty, config)
