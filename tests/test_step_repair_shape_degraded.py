from __future__ import annotations

import pytest
from pathlib import Path

from cad_inspector.domain.repair import RepairConfig

pytest.importorskip("OCP")

# 本文件专测 FIX-1（N1）：「形状退化 → NO_ACTION」守卫不能被后面
# "输入无效且输出仍无效 → REVIEW_REQUIRED" 无条件覆盖。
#
# 红线说明：本测试**只读取 58MB 的 1.stp STEP 模型**（repair_step_holes
# 只调 STEPControl_Reader，不读 453MB 的 1.stl 参考网格），因此不会触发
# 任务书禁止的"硬拉 训练集/1 导致 OOM"路径——那条 OOM 来自 acceptance
# 流水线加载参考 STL，与 repair_step_holes 直调无关。这是 N1 的 exact 复现。
ROOT = Path(__file__).resolve().parents[1]
TRAIN1_STP = (
    ROOT
    / "data"
    / "provided"
    / "三维模型破损区域智能修补"
    / "训练集"
    / "1.stp"
)

N1_CONFIG = RepairConfig(
    max_boundary_edges=4096,
    max_hole_area_mm2=1e6,
    max_planarity_error_mm=0.05,
    allow_nonconvex=True,
    allow_nonplanar_surface_candidate=True,
    preserve_original_on_failure=False,
    do_no_harm=True,
)


@pytest.mark.skipif(
    not TRAIN1_STP.exists(),
    reason="训练集/1.stp 不在本地，跳过（不拉取参考 STL，避免 OOM）",
)
def test我方形状退化实体明确落NO_ACTION且不出文件(tmp_path) -> None:
    """N1 回归：训练集/1 补面后 solid 1→0、B-Rep 仍无效，属"我把它修坏了"。

    修复前（10-21 改动的 bug）：1050-1051 行无条件把 status 覆盖成
    REVIEW_REQUIRED，NO_ACTION 守卫形同虚设，对外看到的是"有候选但没过门禁"。
    修复后：shape_degraded 为真时 status 保持 NO_ACTION、output_file=None、
    退化实体一个字节都不落盘。
    """
    from cad_inspector.repair.step_hole import repair_step_holes

    out_path = tmp_path / "1_repaired.stp"
    result = repair_step_holes(
        TRAIN1_STP, output_path=out_path, config=N1_CONFIG
    )

    assert result.status == "NO_ACTION", (
        "形状退化实体应当报 NO_ACTION，而不是被覆盖成 REVIEW_REQUIRED；"
        f"实际 status={result.status}"
    )
    assert result.output_file is None, (
        "退化实体不应落盘；实际 output_file=%r" % result.output_file
    )
    out_exists = (
        Path(result.output_file).exists() if result.output_file else False
    )
    assert out_exists is False, "退化实体被写到了磁盘，违反不伤害原则"
    # 退化判据的几何事实（与 N1 记录一致）：输入 1 实体、输出 0 实体。
    assert int(result.input_topology.get("solid_count") or 0) == 1
    assert int(result.output_topology.get("solid_count") or 0) == 0
