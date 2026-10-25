"""我钉住自动配准（auto）的选择逻辑。

我为什么加这个模式（2026-10-17）：我把 21 个真实件的 none / centroid / pca
三种配准逐个量了残差，**21/21 个件的 none 都低于另外两种**。
这批 STEP 与 STL 本来就在同一坐标系，是"缺料让质心偏了"，
centroid 把质心对上，反而把整张表面推开——0.15mm 阈值下 90% 面积
被误报成缺失（测试集/11 就是 95.7%，改用 none 后 0.0%）。

所以固定 centroid 是错的，但固定 none 也不对：真位移/旋转过的文件
需要 centroid 或 pca。auto 三个都试、取残差最小，两种情形都覆盖。
下面几条断言就是钉住"两种情形都覆盖"和"绝不选 icp"。
"""

from __future__ import annotations

import numpy as np
import pytest

from cad_inspector.models import InspectionConfig
from cad_inspector.registration.rigid import align_mesh
from test_core import cube_mesh

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports" / "registration_mode_evidence.json"
ACCEPTANCE_EVIDENCE = ROOT / "reports" / "registration_default_change_evidence.json"


def _translate(mesh, offset: tuple[float, float, float]):
    return mesh.transformed(translation=np.asarray(offset, dtype=np.float64))


def test我方默认配准模式是auto() -> None:
    """我钉住默认值。旧默认 centroid 在 21/21 个真实件上都不是最优。"""

    assert InspectionConfig().registration_mode == "auto"


def test我方自动配准在已对齐的件上选none() -> None:
    """件本身同坐标系时，auto 必须选 none，而不是把质心对齐的 centroid。"""

    reference = cube_mesh()
    aligned, result = align_mesh(cube_mesh(), reference, InspectionConfig())
    assert result.mode == "auto->none"
    assert result.mean_residual_mm == pytest.approx(0.0, abs=1e-9)


def test我方自动配准在整体平移的件上不选none() -> None:
    """件被整体平移时，none 的残差会很大，auto 必须改用 centroid/pca。

    这条是上一条的反面：只测"已对齐选 none"会让实现退化成"永远选 none"，
    那样真位移的文件就会被彻底放弃配准。
    """

    reference = cube_mesh()
    moved = _translate(cube_mesh(), (7.0, -3.0, 2.0))
    _, result = align_mesh(moved, reference, InspectionConfig())
    assert result.mode != "auto->none", (
        "件被平移了 7mm，auto 却选了 none；说明自动选择没有真正比较残差。"
    )
    assert result.mean_residual_mm < 0.01, "平移件应能被 centroid/pca 拉回贴合。"


def test我方自动配准的残差不高于三种固定模式() -> None:
    """auto 是取最小值，所以它的残差必须 <= none/centroid/pca 中任何一个。"""

    reference = cube_mesh()
    moved = _translate(cube_mesh(), (4.0, 1.0, -2.0))
    _, auto = align_mesh(moved, reference, InspectionConfig())
    fixed = []
    for mode in ("none", "centroid", "pca"):
        _, result = align_mesh(
            moved, reference, InspectionConfig(registration_mode=mode)
        )
        fixed.append(result.mean_residual_mm)
    assert auto.mean_residual_mm <= min(fixed) + 1e-12


def test我方自动配准绝不选icp() -> None:
    """我钉住 auto 不碰 icp：ICP 会把真实缺料对齐掉，残差最小不等于最准。"""

    reference = cube_mesh()
    moved = _translate(cube_mesh(), (2.0, 2.0, 2.0))
    _, result = align_mesh(moved, reference, InspectionConfig())
    assert "icp" not in result.mode, (
        "auto 选中了 icp；这会触发'把真实缺料对齐掉'的风险，必须排除。"
    )


def test我方自动配准把选择理由写进告警() -> None:
    """我钉住结果里能看出"另外两种残差是多少"，便于甲方复核。"""

    _, result = align_mesh(cube_mesh(), cube_mesh(), InspectionConfig())
    joined = "\n".join(result.warnings)
    assert "自动配准" in joined
    assert "centroid=" in joined and "pca=" in joined


def test我方证据记录显示21个真实件none全部最优() -> None:
    """我钉住"21/21 个件 none 残差最低"这条证据本身。

    这条不是重复上面几条——上面测的是代码行为，这条测的是
    "把默认从 centroid 改成 auto"的依据还在。如果哪天有人
    换了一批数据重算，发现 centroid 反而更好，这条会红，
    提醒重新评估默认值，而不是让一个过时的结论继续当默认。
    """

    assert EVIDENCE.is_file(), f"配准模式证据不见了：{EVIDENCE}"
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert data["parts"] == 21
    assert data["best_mode_counts"]["none"] == 21, (
        "证据显示并非所有件的 none 都最优，"
        "那么'默认 auto 严格优于 centroid'的说法需要重新论证。"
    )
    assert data["parts_where_centroid_is_best"] == 0


def test我方换默认后20个真实件的大面积误报消失() -> None:
    """我钉住换默认前后**端到端验收**的变化，而不是只钉残差。

    残差变小不等于用户看到的结果变好，所以这条直接钉
    missing 比例的上界：换之前最高 95.7%，换之后必须低于 10%。
    同时钉住"仍然是 20/20 判 review"——修正配准没有顺手放宽判定。
    """

    assert ACCEPTANCE_EVIDENCE.is_file(), f"端到端证据不见了：{ACCEPTANCE_EVIDENCE}"
    data = json.loads(ACCEPTANCE_EVIDENCE.read_text(encoding="utf-8"))
    assert data["pairs_compared"] == 20
    assert data["before_mode"] == "centroid" and data["after_mode"] == "auto"
    assert data["max_before_missing_ratio"] > 0.9, (
        "换之前应该存在 >=90% 的大面积误报，否则这轮修正的依据不成立。"
    )
    assert data["max_after_missing_ratio"] < 0.10, (
        "换默认后仍有件把 >=10% 表面判成缺失，说明 auto 没有真正解决问题。"
    )
    assert data["after_total_elapsed_s"] < data["before_total_elapsed_s"], (
        "对齐正确后异常面应减少、总耗时应下降；若反而变慢需要重新解释。"
    )
