"""我定义 STEP 局部修补结果，明确区分成功修补和人工复核候选。

算法原理：修补质量不能只报"补上了"，必须按几何连续性分级：G0（位置连续，补面与原面
贴合无缝隙）、G1（切向连续，法向不跳变）、G2（曲率连续，肉眼无折痕）。三个等级分别
记录是否通过以及最大偏差量，任一项未达标即 ``continuity_verified=False`` 并进入人工
复核。这样甲方能一眼看出"补上了但补得糙"和"补得平顺"的区别，而不是只看一个布尔值。

``deviation_max_mm`` / ``deviation_mean_mm`` 允许是字符串，因为补面失败时写的是失败
原因而不是数字——用类型联合保住这个语义，不用 None 丢掉原因。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepRepairRegionResult:
    """我记录一个 STEP 开放边界区域的补面证据。

    我为什么把 continuity_level 与三个 g*_verified 分开存：
    continuity_level 是给甲方的**一句话结论**（当前达到的最好等级），
    而三个布尔是给追溯用的**分项证据**。只存等级会丢掉"G0 过了但 G1
    没过"到底是哪项没过；只存布尔又得让甲方自己去推结论。两者都留，
    报表和追溯各自取用。

    我为什么把面积键名在 to_dict 里改成 area_repaired：CSV 表头是甲方
    定的固定口径，而 dataclass 字段名要带毫米单位才不会误读。我在
    序列化这一层做键名对齐，让领域模型和对外契约互不污染。
    """

    region_id: int
    # damage_type 与 repair_method 我都存字符串而不是枚举：真实件上会
    # 出现我没预见到的类型（例如三角化补面这种后加的路径），用枚举
    # 会在遇到新值时直接反序列化失败，而字符串至少能把未知值原样留下来。
    damage_type: str
    status: str
    repair_method: str
    area_repaired_mm2: float
    continuity_level: str
    # 条数与顶点数分开存：一个环可能被回读拆成多条边（OCP 回读会拆开
    # 共享边），条数会凭空变多。复核者要判断"是不是越补越破"时必须
    # 同时看这两个数，只看条数会被回读产物误导。
    boundary_edge_count: int
    # **这两个数的口径不一样，别当同一个量比**（我在这上面栽过一次）：
    #   boundary_edge_count   = 这个环有**几条边**
    #   boundary_vertex_count = 边界**采样点的个数**（每条边约采 4 个点）
    # 所以真实语料上两者的比值**恒为约 4:1**——实测 (5,1) (10,2) (16,4)
    # (24,6)，**每个件每个环都这样**。我曾误以为"点 5 个边 1 条"是边被合并
    # 的 bug，据此把判据改成比点数，结果等于把判据废掉（详见 step_hole.py
    # 1368 附近的注释）。要判断"这个环够不够点建面"，得看点数；
    # 要判断"有几条边"，才看条数。
    boundary_vertex_count: int
    planarity_error_mm: float | None
    g0_verified: bool
    g1_verified: bool
    g2_verified: bool
    g0_max_deviation_mm: float | None
    g1_max_normal_angle_deg: float | None
    g2_max_curvature_delta_per_mm: float | None
    continuity_verified: bool
    deviation_max_mm: float | str
    deviation_mean_mm: float | str
    source_face_ids: list[int]
    review_required: bool
    # 我记录这个环自己量出来的缝总长（mm）。为什么要单独存：正式输出门禁
    # 要看"还有没有环带着实质孔洞"，只能用长度判；量不出来存 None，
    # 调用方按"有实质孔洞"处理，不会静默放过。
    loop_gap_mm: float | None = None
    # 我记录这张补面最终有没有被采纳进模型。
    # 为什么不能只看 status：补面被面积门禁拒掉的环与被采纳的环
    # 同样是 CANDIDATE——门禁是在 status 置成 CANDIDATE 之后才抛异常
    # 降级的，status 字段本身区分不出这两者。而正式输出门禁必须
    # 区分它们：被采纳的洞已经补上了，不该再拿它补面前的缝长去否决
    # 整份输出；被拒的洞是真没补上，必须照旧否决，否则
    # max_hole_area_mm2 这道安全门形同虚设。
    # 面积也不能替代它：面积下限门禁同样会把补面拒掉而留下 0，
    # 但"面积为 0"还可能是合法极小补面，语义不唯一。
    patch_accepted: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """我把区域结果转成字典，并把面积键名对齐 CSV 表头。"""
        payload = asdict(self)
        payload["area_repaired"] = payload.pop("area_repaired_mm2")
        return payload


@dataclass
class StepRepairResult:
    """我把 STEP 修补前后质量、输出路径和区域证据统一保存。"""

    status: str
    input_file: str
    output_file: str | None
    backend: str
    input_shape_is_valid: bool | None
    output_shape_is_valid: bool | None
    input_topology: dict[str, Any]
    output_topology: dict[str, Any]
    repaired_regions: list[StepRepairRegionResult]
    warnings: list[str]
    artifacts: dict[str, str]
    config: dict[str, Any]
    elapsed_seconds: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """我把完整修补结果转成带 schema 版本的可序列化字典。"""
        return {
            "schema_version": "step-repair-1.3",
            "status": self.status,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "backend": self.backend,
            "input_shape_is_valid": self.input_shape_is_valid,
            "output_shape_is_valid": self.output_shape_is_valid,
            "input_topology": dict(self.input_topology),
            "output_topology": dict(self.output_topology),
            "repaired_regions": [
                region.to_dict() for region in self.repaired_regions
            ],
            "warnings": list(self.warnings),
            "artifacts": dict(self.artifacts),
            "config": dict(self.config),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "diagnostics": dict(self.diagnostics),
        }
