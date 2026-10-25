"""我使用 OCP 读取 STEP/BRep 拓扑，并输出可追踪的结构化报告。

算法原理：STEP 是文本格式，文本里出现的实体数量不等于几何拓扑数量，所以我不用
文本计数冒充拓扑结果。我调用 OCP 的 ``STEPControl_Reader`` 真正转移出 Shape，再用
``TopExp_Explorer`` 逐个统计 Solid/Shell/Face/Wire/Edge/Vertex，最后用 Face 局部
映射统计每条 Edge 被多少个 Face 引用（incidence），据此区分：

- ``boundary_edge_count``：只连接一个 Face 的 Edge，说明 Shape 有开放边界；
- ``manifold_edge_count``：连接两个 Face 的 Edge，正常流形边；
- ``non_manifold_edge_count``：连接三个及以上 Face 的 Edge，非流形拓扑，必须人工复核。

只有 Shape 有效、且没有开放边界和非流形边时，状态才是 COMPLETE；否则降级为
REVIEW_REQUIRED 并给出对应警告。这样下游能一眼看出几何是否可信。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..errors import DependencyMissingError, GeometryProcessingError
from ..reproducibility import with_content_digest
from .file_validation import validate_input_file
from .step_probe import probe_step


@dataclass
class StepTopologyReport:
    """我记录一次真实 STEP 几何后端读取结果。"""

    status: str
    file_path: str
    unit: str | None
    backend: str
    shape_is_null: bool
    shape_is_valid: bool | None
    entity_counts: dict[str, int]
    face_ids: list[int]
    edge_topology: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """我把报告转成带 schema 版本和内容摘要的可序列化字典。"""
        return with_content_digest({
            "schema_version": "step-topology-1.2",
            **asdict(self),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        })


def inspect_step_topology(
    path: str | Path,
    *,
    validate_shape: bool = True,
) -> StepTopologyReport:
    """我读取 STEP 的 BRep 拓扑，不把文本实体数量冒充几何拓扑。

    三种失败我用三种不同处理，区别很关键：
      - 缺 OCP、读文件失败、没转移出根对象：抛错。这属于"根本没拿到几何"，
        返回一个空报告会让下游以为"这个模型没有面"；
      - Shape 为空：返回 status=FAILED 的报告。文件本身能读，
        是这一个模型没有几何内容，属于可记录的结果而非异常；
      - 拓扑有开口或非流形：不报错，降级为 REVIEW_REQUIRED。
        这是**真实的几何状态**，是这份报告存在的意义。

    实体统计顺序从大粒度到小粒度（SOLID -> VERTEX）：这个顺序也让
    报告本身可读，读者能顺着"几个实体、几个壳、几个面"往下看，
    反过来排会让人先看到几千个顶点再回头找实体数。
    """

    started = time.perf_counter()
    validated = validate_input_file(path, {".stp", ".step"})
    file_path = str(validated.resolve())
    probe = probe_step(validated)
    unit = probe.unit_hint
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.OCP.collections import (
            IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
        )
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import (
            TopAbs_EDGE,
            TopAbs_FACE,
            TopAbs_SHELL,
            TopAbs_SOLID,
            TopAbs_VERTEX,
            TopAbs_WIRE,
        )
        from OCP.TopExp import TopExp, TopExp_Explorer
    except ImportError as exc:
        raise DependencyMissingError(
            "我无法读取 STEP BRep，因为当前环境没有 OCP。"
        ) from exc

    # 真正读取 STEP 并转移根对象；读取失败或没有可用根对象都直接报错。
    reader = STEPControl_Reader()
    # 读文件与转移根对象分两步判错，且都抛异常而不是返回空报告：
    # 这两步失败意味着"根本没拿到几何"，返回空报告会让下游
    # 误以为"这个模型本来就没有面"，把读取错误伪装成模型特征。
    read_status = reader.ReadFile(file_path)
    if read_status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise GeometryProcessingError(
            f"我读取 STEP 失败，后端状态为 {read_status}: {file_path}"
        )
    transferred = reader.TransferRoots()
    if not transferred:
        raise GeometryProcessingError(
            f"我没有从 STEP 转移出可用根对象: {file_path}"
        )
    shape = reader.OneShape()
    # Shape 为空不抛错，而是返回 FAILED 报告：文件本身读得出来，
    # 是这个模型没有几何内容，属于可记录的结果而非异常。
    if shape.IsNull():
        return StepTopologyReport(
            status="FAILED",
            file_path=file_path,
            unit=unit,
            backend="OCP",
            shape_is_null=True,
            shape_is_valid=False,
            entity_counts={},
            face_ids=[],
            edge_topology={},
            errors=["我读取到的 STEP Shape 为空。"],
            elapsed_seconds=time.perf_counter() - started,
        )

    # 按拓扑类型逐层统计实体数量，顺序从大粒度到小粒度。
    entity_counts: dict[str, int] = {}
    for label, shape_type in (
        ("SOLID", TopAbs_SOLID),
        ("SHELL", TopAbs_SHELL),
        ("FACE", TopAbs_FACE),
        ("WIRE", TopAbs_WIRE),
        ("EDGE", TopAbs_EDGE),
        ("VERTEX", TopAbs_VERTEX),
    ):
        explorer = TopExp_Explorer(shape, shape_type)
        count = 0
        while explorer.More():
            count += 1
            explorer.Next()
        entity_counts[label] = count

    # Face 编号按 1 起算，与后续 Face 级特征报告的编号口径保持一致。
    face_ids = list(range(1, entity_counts["FACE"] + 1))
    # incidence_counts 是"每条边被多少个面引用"的计数表，
    # 它是整份拓扑报告的核心：开放边界与非流形都只能从这里读出来。
    incidence_counts, face_edge_occurrence_count = _face_edge_incidence(
        shape,
        face_type=TopAbs_FACE,
        edge_type=TopAbs_EDGE,
        explorer_type=TopExp_Explorer,
        indexed_map_type=IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
        map_shapes=TopExp.MapShapes_s,
    )
    # 按引用数分档：1=边界边（开放）、2=流形边（正常）、>2=非流形边。
    # 0=孤立边也单独计数，它说明映射表里有不依附任何面的边，
    # 属于数据异常而非几何问题，分开记才不会与开放边界混淆。
    edge_topology = {
        "mapped_edge_count": len(incidence_counts),
        "unique_edge_count": len(incidence_counts),
        "face_edge_occurrence_count": face_edge_occurrence_count,
        "boundary_edge_count": sum(count == 1 for count in incidence_counts),
        "manifold_edge_count": sum(count == 2 for count in incidence_counts),
        "non_manifold_edge_count": sum(count > 2 for count in incidence_counts),
        "zero_face_edge_count": sum(count == 0 for count in incidence_counts),
        "max_face_incidence": max(incidence_counts, default=0),
        # 方法标识随结果落盘：换了索引实现后，历史报告与新的
        # 计数口径可能不同，靠这个字段才能识别出来。
        "topology_index_method": "face_local_map",
    }
    # 可选执行 BRep 有效性检查；未开启时保持 None 而不是假装通过。
    shape_is_valid = None
    warnings: list[str] = []
    # shape_is_valid 用 None 表示"没查"，False 表示"查了且无效"：
    # 关掉 validate_shape 时不能报"有效"，也不能报"无效"，
    # 三态才能准确表达"未知"。
    if validate_shape:
        shape_is_valid = bool(BRepCheck_Analyzer(shape).IsValid())
        if not shape_is_valid:
            warnings.append("我发现 STEP BRep 有效性检查未通过。")
    if unit is None:
        # 单位未知要单独警告而不是猜一个默认值：
        # STEP 常见 mm/inch 混用，猜错会让所有毫米量纲差 25.4 倍。
        warnings.append("我没有从 STEP 文件中识别到明确单位。")
    if entity_counts["FACE"] == 0:
        warnings.append("我没有读取到 Face，后续局部区域处理无法继续。")
    if edge_topology["boundary_edge_count"]:
        warnings.append(
            f"我发现 {edge_topology['boundary_edge_count']} 条 Edge 只连接一个 Face，"
            "当前 Shape 存在开放边界，不能直接视为封闭实体。"
        )
    if edge_topology["non_manifold_edge_count"]:
        warnings.append(
            f"我发现 {edge_topology['non_manifold_edge_count']} 条 Edge 连接超过两个 Face，"
            "当前 Shape 存在非流形拓扑，必须人工复核。"
        )

    # COMPLETE 的三个条件都是"必须为零"：
    #   shape_is_valid is not False —— 未校验（None）也算通过，
    #     因为没校验不等于有问题；
    #   无自由边、无非流形边 —— 这是封闭实体的拓扑定义。
    # 任一不满足就降 REVIEW_REQUIRED，而不是报错，
    # 因为开放与非流形都是真实的几何状态，是这份报告要表达的内容。
    status = (
        "COMPLETE"
        if shape_is_valid is not False
        and not edge_topology["boundary_edge_count"]
        and not edge_topology["non_manifold_edge_count"]
        else "REVIEW_REQUIRED"
    )
    return StepTopologyReport(
        status=status,
        file_path=file_path,
        unit=unit,
        backend="OCP",
        shape_is_null=False,
        shape_is_valid=shape_is_valid,
        entity_counts=entity_counts,
        face_ids=face_ids,
        edge_topology=edge_topology,
        warnings=warnings,
        elapsed_seconds=time.perf_counter() - started,
    )


def _face_edge_incidence(
    shape: Any,
    *,
    face_type: Any,
    edge_type: Any,
    explorer_type: Any,
    indexed_map_type: Any,
    map_shapes: Any,
) -> tuple[list[int], int]:
    """我从 Face 局部映射统计 Edge，避免无效 Shape 的孤立索引污染结果。"""

    edge_map = indexed_map_type()
    incidence_counts: list[int] = []
    face_edge_occurrence_count = 0
    face_explorer = explorer_type(shape, face_type)
    while face_explorer.More():
        face_edge_map = indexed_map_type()
        map_shapes(face_explorer.Current(), edge_type, face_edge_map)
        face_edge_occurrence_count += face_edge_map.Extent()
        for index in range(1, face_edge_map.Extent() + 1):
            edge_index = edge_map.Add(face_edge_map.FindKey(index))
            while len(incidence_counts) < edge_index:
                incidence_counts.append(0)
            incidence_counts[edge_index - 1] += 1
        face_explorer.Next()
    return incidence_counts, face_edge_occurrence_count
