"""我使用 OCP 实体截交生成真实 STEP 的 B-Rep 分型线候选。

算法原理：分型线在工程上通常是模具沿开模方向的"最大外轮廓"。要拿到它，我用
B-Rep 级别的实体与平面求交，而不是拿三角网格近似，具体分四步：

1. 扫描平面高度。若配置固定 ``plane_z_mm`` 就直接用；否则在零件 Z 包围盒内按
   ``plane_scan_count`` 等距取多个高度，两侧各留 ``plane_margin_ratio`` 的边距，
   避免贴着顶底切出退化截面。
2. 逐平面截交。用 ``BRepAlgoAPI_Section`` 把实体与一张足够大的平面求交，得到
   截面 Edge 序列，再 ``ConnectEdgesToWires`` 连成 Wire 候选。
3. 选最大闭合线框。对每个 Wire 采样离散点、检查是否闭合、算周长与投影面积，
   在所有平面所有候选中挑出面积/周长最大的闭合 Wire 作为分型线。
4. 保留可导出几何。把选中的 Wire 和延伸平面 Face 一起返回，供后续 CAD 导出，
   而不是只返回一串点。

拓扑编号通过 ``_register_topology_shape`` / ``_find_topology_shape_id`` 用几何哈希
做稳定登记，保证 Edge/Face 编号在同一次运行内可追溯。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.parting import PartingConfig, PartingLineResult
from ..errors import DependencyMissingError, GeometryProcessingError
from ..io.file_validation import validate_input_file
from ..validation.parting_quality import (
    parting_line_diagnostics,
    parting_line_warnings,
)


@dataclass
class StepPartingGeometry:
    """我保存非序列化的 OCP 线框和延伸平面面候选。"""

    line: PartingLineResult
    wire: Any
    face: Any
    bbox_min_xyz_mm: list[float]
    bbox_max_xyz_mm: list[float]
    section_wire_count: int
    section_edge_count: int
    shape_is_valid: bool | None = None
    line_candidates: list[PartingLineResult] = field(default_factory=list)
    transition_surface: Any = None
    sewn_transition_surface: Any = None


def extract_step_parting_line(
    path: str | Path,
    config: PartingConfig | None = None,
) -> PartingLineResult:
    """我对 STEP 实体进行多个水平截交，并选择最大的闭合 B-Rep 线框。"""

    return extract_step_parting_geometry(path, config).line


def extract_step_parting_geometry(
    path: str | Path,
    config: PartingConfig | None = None,
) -> StepPartingGeometry:
    """我返回可序列化分型线和用于 CAD 导出的 OCP 线框/平面。

    我把 OCP 的所有导入都放在函数体内延迟执行，原因有两个：
      1. OCP 是重量级可选依赖，模块级导入会让"只做网格分析"的调用方
         也必须装 OCP，把本不需要的依赖变成硬门槛；
      2. 导入失败要转成 DependencyMissingError 并给出中文说明，
         让调用方知道"缺什么、为什么"，而不是看到一个裸的 ImportError
         去猜是环境坏了还是自己装错了。

    读取链路上的每一步都显式检查状态：ReadFile 返回码、TransferRoots
    是否转移成功、OneShape 是否为空。这三项任一失败都直接抛错，
    不做"先试试看能不能往下跑"——读都没读对的 Shape 得出的分型线
    没有任何意义，静默继续只会产出看起来正常实则错误的结论。
    """

    config = config or PartingConfig()
    file_path = validate_input_file(path, {".stp", ".step"}).resolve()
    try:
        # OCP 全部在函数内按需 import，而不是模块顶层 import：
        # 这个仓库要求"没装 OCP 也能 import 本模块"，
        # 只有在真的要做 B-Rep 分型线时才需要这些依赖。
        # 截交：拿平面去切实体，得到分型线的原始边。
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
        # 包围盒与有效性分析：前者取整体尺寸，后者判断形状是否成立。
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepBuilderAPI import (
            # 建面、缝合、变换：分别用于把截交结果变成面、
            # 把散面缝成壳、把整体挪到某个位姿。
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_Sewing,
            BRepBuilderAPI_Transform,
        )
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.BRepCheck import BRepCheck_Analyzer
        # 放样：从两条 Wire 生成过渡曲面候选。
        from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
        # Wire 遍历：按拓扑顺序走一条 Wire 上的边。
        from OCP.BRepTools import BRepTools_WireExplorer
        from OCP.Bnd import Bnd_Box
        # 几何基元：方向、平面、点、变换、向量。
        from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.OCP.collections import (
            HSequence_TopoDS_Shape,
            IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher,
        )
        # 自由边界：从一堆边里挑出真正开放的那几条。
        from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
        from OCP.STEPControl import STEPControl_Reader
        # 五类拓扑元素类型：遍历时指定"要找哪一类"。
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopAbs import TopAbs_SHELL
        from OCP.TopAbs import TopAbs_WIRE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Face
        from OCP.TopExp import TopExp
    # 缺 OCP 就转成 DependencyMissingError 而不是让 ImportError 直接冒泡：
    # 调用方能据此把"环境缺依赖"与"这个文件有问题"区分开。
    except ImportError as exc:
        raise DependencyMissingError(
            "我无法生成 STEP B-Rep 分型线，因为当前环境没有 OCP。"
        ) from exc

    reader = STEPControl_Reader()
    # 读文件与转移根对象两步都要判错，且都抛异常：
    # 这两步失败意味着根本没拿到几何，返回空结果会让下游
    # 把"读取失败"误当成"这个模型没有分型线"。
    read_status = reader.ReadFile(str(file_path))
    if read_status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise GeometryProcessingError(
            f"我读取 STEP 失败，后端状态为 {read_status}: {file_path}"
        )
    if not reader.TransferRoots():
        raise GeometryProcessingError(f"我没有转移出 STEP 根对象: {file_path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise GeometryProcessingError(f"我读取到的 STEP Shape 为空: {file_path}")
    # 有效性检查结果只记录不拦截：无效的 B-Rep 也可能截交出可用的线，
    # 把它当作候选并标记 REVIEW_REQUIRED，比直接拒绝更有价值。
    shape_is_valid = bool(BRepCheck_Analyzer(shape).IsValid())
    # 哈希桶按形状哈希分桶，用来把截交得到的 Edge/Face 反查回原模型的编号。
    # 不用线性查找是因为大模型上 Edge 有上万个，逐条比对会退化成 O(n²)。
    face_hash_buckets: dict[int, list[int]] = {}
    face_by_id: dict[int, Any] = {}
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    while face_explorer.More():
        face_id += 1
        face = TopoDS.Face(face_explorer.Current())
        _register_topology_shape(face, face_id, face_hash_buckets, face_by_id)
        face_explorer.Next()

    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    bounds = (
        float(bbox.GetXMin()),
        float(bbox.GetYMin()),
        float(bbox.GetZMin()),
        float(bbox.GetXMax()),
        float(bbox.GetYMax()),
        float(bbox.GetZMax()),
    )
    if (
        not np.all(np.isfinite(bounds))
        or bounds[3] <= bounds[0]
        or bounds[4] <= bounds[1]
        or bounds[5] <= bounds[2]
    ):
        raise GeometryProcessingError("我发现 STEP 包围盒无效，无法建立水平截面。")

    # 我先定开模方向。写死 Z 轴会在这批 STEP 上切错：甲方数据由不同 CAD 导出，
    # 坐标系朝向不统一，实测 Plastic Bag Clip Mould 的主惯量轴与 Z 夹角约 58 度。
    parting_axis, parting_axis_method, parting_axis_diagnostics = (
        _detect_parting_axis(shape, bounds, config)
    )
    best: tuple[float, PartingLineResult, Any, Any, int, int] | None = None
    line_candidates: list[PartingLineResult] = []
    # 扫描多个高度而不是只切一刀：分型线是"最大外轮廓"，
    # 只有扫过整个包围盒才知道哪个高度的截面最大。
    # 每个高度会产出多条候选 Wire，全部参与打分，最后只留最优的一条。
    for plane_z in _scan_plane_levels(bounds, config, parting_axis):
        # 沿分型轴逐个高度做截交：分型线是"沿拔模方向扫一遍"
        # 得到的每条截交线，逐个高度扫是最直接的做法。
        section = _build_section(
            shape,
            bounds,
            plane_z,
            config,
            BRepAlgoAPI_Section,
            BRepBuilderAPI_MakeFace,
            gp_Dir,
            gp_Pln,
            gp_Pnt,
            HSequence_TopoDS_Shape,
            ShapeAnalysis_FreeBounds,
            TopAbs_EDGE,
            TopExp_Explorer,
            TopoDS,
            parting_axis,
        )
        for candidate in section["candidates"]:
            # 一条截交 Wire 变一条候选分型线。
            # 哈希桶（edge_hash_buckets / face_hash_buckets）用于
            # 把 OCP 返回的无 ID 形状映射回稳定编号，
            # 否则同一条边重跑一次编号就变，无法跨批次比对。
            line, wire, face = _line_from_wire(
                candidate["wire"],
                plane_z,
                config,
                BRepAdaptor_Curve,
                BRepTools_WireExplorer,
                TopoDS,
                TopoDS_Face,
                section["section_api"],
                section["edge_hash_buckets"],
                section["edge_by_id"],
                face_hash_buckets,
                face_by_id,
            )
            if line is None:
                # 建不出线就跳过这一条，不中断整个扫描：
                # 某一个高度上截交失败不代表其他高度也失败。
                continue
            line.diagnostics.update(
                {
                    # 包围盒写进每条线的诊断：分型线的位置
                    # 只有相对于整体包围盒才有意义。
                    "bbox_min_xyz_mm": list(bounds[:3]),
                    "bbox_max_xyz_mm": list(bounds[3:]),
                    "section_wire_count": section["wire_count"],
                    "section_edge_count": section["edge_count"],
                    "shape_is_valid": shape_is_valid,
                    "topology_index_method": "hash_bucket_is_same",
                    **parting_axis_diagnostics,
                }
            )
            if not shape_is_valid:
                line.status = "REVIEW_REQUIRED"
                line.warnings.append(
                    "我发现 STEP BRep 有效性检查未通过，当前分型线只能作为 REVIEW_REQUIRED 候选。"
                )
            line_candidates.append(line)
            # 打分时给闭合线框加一个远大于正常分值的偏置，而不是单独判空后跳过：
            # 这样"闭合"是压倒性优先项，但环的其他质量分仍然参与比较，
            # 多个闭合候选之间还能按质量排序；若写成 if closed: continue
            # 就会丢掉"哪个闭合环更好"这一层信息。
            score = float(line.score + (1_000_000.0 if line.closed else 0.0))
            if best is None or score > best[0]:
                best = (
                    score,
                    line,
                    wire,
                    face,
                    section["wire_count"],
                    section["edge_count"],
                )

    if best is None:
        raise GeometryProcessingError(
            "我没有从 STEP 的水平截交中找到可用闭合线框；请人工确认分型方向。"
        )
    _, line, wire, face, wire_count, edge_count = best
    extended_face = _make_extended_planar_face(
        bounds=bounds,
        plane_z=line.plane_z_mm,
        margin_mm=config.surface_margin_mm,
        make_face_type=BRepBuilderAPI_MakeFace,
        gp_dir_type=gp_Dir,
        gp_pln_type=gp_Pln,
        gp_pnt_type=gp_Pnt,
        axis=parting_axis,
    )
    # 我按分型方向报告分型面的实际范围，供甲方复核时还原这个面到底在哪。
    surface_u, surface_v = _plane_frame(parting_axis)
    # 分型面的外扩范围取自整个模型的包围盒在平面内的投影：
    # 面必须比零件大，否则切不到全部边界，也就无法用于后续切割。
    surface_uv_bounds = _plane_extent(bounds, surface_u, surface_v)
    surface_normal = np.asarray(parting_axis, dtype=np.float64)
    # 法向归一化：后续所有点乘与偏移都假设单位向量，
    # 不归一化会让 plane_z_mm 的语义从"高度"变成"高度×模长"。
    surface_normal = surface_normal / float(np.linalg.norm(surface_normal))
    surface_origin = surface_normal * float(line.plane_z_mm)
    # 建不出外扩面时降级而不是抛错：分型线本身已经算出来了，
    # 它仍有价值。此时保留线、明确标记面无效，
    # 比因为面失败把整条分型线一起丢掉损失小。
    if extended_face is None:
        line.status = "REVIEW_REQUIRED"
        line.warnings.append(
            "我无法建立带外扩边界的 OCP 分型面，当前只保留截交线候选。"
        )
        line.diagnostics["surface_face_is_valid"] = False
    else:
        surface_face_is_valid = bool(BRepCheck_Analyzer(extended_face).IsValid())
        # stage6 诊断独立成块：它回答"这个面与源实体是什么关系"，
        # 与"面自身是否有效"是两个层面的问题，不能合并成一句话。
        stage6_diagnostics = _stage6_surface_diagnostics(
            shape,
            extended_face,
            shape_is_valid=shape_is_valid,
            analyzer_type=BRepCheck_Analyzer,
            explorer_type=TopExp_Explorer,
            top_abs_edge=TopAbs_EDGE,
            top_abs_solid=TopAbs_SOLID,
            top_abs_wire=TopAbs_WIRE,
            topods_type=TopoDS,
        )
        (
            transition_surface,
            sewn_transition_surface,
            transition_diagnostics,
        ) = (
            _build_transition_surface_candidate(
                # 所有 OCP 类型都是显式注入而不是在模块顶层 import，
                # 原因有两个：一是没装 OCP 的环境仍能 import 本模块
                # 并看到其余能力；二是测试可以直接传入替身，
                # 不必真的构造几何。代价是参数列表很长，
                # 这是为了让依赖边界可见而接受的。
                shape=shape,
                wire=wire,
                bounds=bounds,
                plane_z=line.plane_z_mm,
                config=config,
                # 变换与放样：过渡面是"把截交线沿分型方向
                # 平移一段再放样成面"，这两步分别对应
                # BRepBuilderAPI_Transform 与 BRepOffsetAPI_ThruSections。
                transform_type=BRepBuilderAPI_Transform,
                loft_type=BRepOffsetAPI_ThruSections,
                trsf_type=gp_Trsf,
                vec_type=gp_Vec,
                topo_type=TopoDS,
                analyzer_type=BRepCheck_Analyzer,
                explorer_type=TopExp_Explorer,
                # 截交与缝合：过渡面要与实体求交、再缝回去，
                # 否则它只是一片独立的面，不构成分型面。
                section_type=BRepAlgoAPI_Section,
                sewing_type=BRepBuilderAPI_Sewing,
                top_exp_type=TopExp,
                edge_map_type=(
                    IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher
                ),
                top_abs_edge=TopAbs_EDGE,
                top_abs_face=TopAbs_FACE,
                top_abs_shell=TopAbs_SHELL,
                top_abs_solid=TopAbs_SOLID,
                top_abs_wire=TopAbs_WIRE,
                axis=parting_axis,
            )
        )
        line.diagnostics.update(
            {
                "surface_face_is_valid": surface_face_is_valid,
                "surface_face_method": "ocp_extended_planar_face",
                "surface_margin_mm": float(config.surface_margin_mm),
                # 分型方向不一定是 Z，所以我按分型平面的两个基向量报告面范围，
                # 不再把 Z 分量写死成分型高度。
                "surface_plane_u_min_mm": float(surface_uv_bounds[0]),
                "surface_plane_u_max_mm": float(surface_uv_bounds[1]),
                "surface_plane_v_min_mm": float(surface_uv_bounds[2]),
                "surface_plane_v_max_mm": float(surface_uv_bounds[3]),
                "surface_plane_origin_xyz_mm": [
                    float(value) for value in surface_origin
                ],
                **stage6_diagnostics,
                **transition_diagnostics,
            }
        )
        line.diagnostics["sewing_status"] = transition_diagnostics.get(
            "transition_surface_sewing_status",
            "NOT_EXECUTED_REVIEW_REQUIRED",
        )
        line.diagnostics["stage6_limitations"] = [
            "我只确认了截交和候选平面面的结构证据。",
            "我已对过渡曲面候选执行基础缝合并记录自由边和多重边。",
            "我没有自动执行实体布尔切割、补面、G0/G1/G2 连续性或加工工艺验收。",
        ]
        face = extended_face
        if stage6_diagnostics["stage6_status"] != "BASELINE":
            line.status = "REVIEW_REQUIRED"
            line.warnings.append(
                "我发现阶段6分型面质量门禁未全部通过，当前结果需要人工复核。"
            )
    line.warnings = list(dict.fromkeys(line.warnings))
    line_candidates.sort(
        key=lambda candidate: (
            candidate.closed,
            candidate.score,
            # 排序口径与 score 保持一致，都用拟合平面真实面积，避免
            # "按真实面积选中、却按 XY 投影排序"造成前后矛盾。
            float(
                candidate.diagnostics.get(
                    "true_loop_area_mm2", candidate.area_xy_mm2
                )
                or 0.0
            ),
        ),
        reverse=True,
    )
    line.diagnostics["candidate_count"] = len(line_candidates)
    return StepPartingGeometry(
        line=line,
        wire=wire,
        face=face,
        bbox_min_xyz_mm=list(bounds[:3]),
        bbox_max_xyz_mm=list(bounds[3:]),
        section_wire_count=wire_count,
        section_edge_count=edge_count,
        shape_is_valid=shape_is_valid,
        line_candidates=line_candidates,
        transition_surface=transition_surface,
        sewn_transition_surface=sewn_transition_surface,
    )


def _stage6_surface_diagnostics(
    shape: Any,
    surface_face: Any,
    *,
    shape_is_valid: bool,
    analyzer_type: Any,
    explorer_type: Any,
    top_abs_edge: Any,
    top_abs_solid: Any,
    top_abs_wire: Any,
    topods_type: Any,
) -> dict[str, Any]:
    """我检查延伸平面面的结构和截交证据，不自动修改实体。"""

    solid_count = 0
    solid_explorer = explorer_type(shape, top_abs_solid)
    while solid_explorer.More():
        solid_count += 1
        solid_explorer.Next()

    wire_count = 0
    closed_wire_count = 0
    wire_explorer = explorer_type(surface_face, top_abs_wire)
    while wire_explorer.More():
        wire_count += 1
        try:
            wire = topods_type.Wire(wire_explorer.Current())
            if bool(wire.Closed()):
                closed_wire_count += 1
        except (AttributeError, TypeError, RuntimeError):
            pass
        wire_explorer.Next()

    surface_is_valid = bool(analyzer_type(surface_face).IsValid())
    boundary_is_closed = wire_count > 0 and closed_wire_count == wire_count
    # 我不再用"曾经截出过边"当作穿透证据。旧逻辑把选中分型线所在截面高度上的
    # 截交边数当成"分型面与实体相交"的证据，而这个面本身就是按零件 XY 包围盒
    # 外扩建出来的，只要那一刀切到了实体就必然有边——于是这个检查恒为真，
    # 永远证明不了"分型面穿过了实体"。我改成对选中的分型面重新做一次真实求交。
    intersects_source, penetration_evidence = _surface_penetration_evidence(
        shape,
        surface_face,
        explorer_type=explorer_type,
        top_abs_edge=top_abs_edge,
    )
    # BASELINE 五个条件全都要满足，缺一即降 REVIEW_REQUIRED：
    #   shape_is_valid —— 源实体本身有效；
    #   solid_count > 0 —— 真的切到了实体，不是空集；
    #   surface_is_valid —— 生成的面自身有效；
    #   boundary_is_closed —— 面的边界闭合，能围成实体面；
    #   intersects_source —— 面真的与实体相交，不是悬在空中的平面。
    # 最后一条常被误认为"越不相交越好"，其实分型面必须落在实体上，
    # 不相交说明这个面根本不在零件上。
    status = (
        "BASELINE"
        if (
            shape_is_valid
            and solid_count > 0
            and surface_is_valid
            and boundary_is_closed
            and intersects_source
        )
        else "REVIEW_REQUIRED"
    )
    return {
        "stage6_status": status,
        # 诊断版本号随结果落盘：stage6 的判据会演进，
        # 没有版本号就无法判断一份历史报告是按哪套判据得出的。
        "stage6_diagnostic_version": "parting-surface-stage6-ocp-1.0",
        "source_solid_count": solid_count,
        "surface_wire_count": wire_count,
        "surface_closed_wire_count": closed_wire_count,
        "surface_boundary_is_closed": boundary_is_closed,
        "surface_intersects_source_shape": intersects_source,
        "surface_penetration_evidence": penetration_evidence,
        # 未执行的阶段显式写 NOT_EXECUTED，而不是省略字段：
        # 省略会让下游分不清"没做"与"做了但无结果"。
        "sewing_status": "NOT_EXECUTED_REVIEW_REQUIRED",
        "transition_surface_status": "NOT_EXECUTED_REVIEW_REQUIRED",
        # 能力边界随每份报告一起交付：明确列出本阶段没做什么，
        # 防止甲方把"通过了 stage6"误读成"分型面已完成加工验收"。
        "stage6_limitations": [
            "我只确认了截交和候选平面面的结构证据。",
            "我没有自动执行实体布尔切割、曲面缝合、过渡曲面补面或加工工艺验收。",
        ],
    }


def _build_transition_surface_candidate(
    *,
    shape: Any,
    wire: Any,
    bounds: tuple[float, float, float, float, float, float],
    plane_z: float,
    config: PartingConfig,
    transform_type: Any,
    loft_type: Any,
    trsf_type: Any,
    vec_type: Any,
    topo_type: Any,
    analyzer_type: Any,
    explorer_type: Any,
    section_type: Any,
    sewing_type: Any,
    top_exp_type: Any,
    edge_map_type: Any,
    top_abs_edge: Any,
    top_abs_face: Any,
    top_abs_shell: Any,
    top_abs_solid: Any,
    top_abs_wire: Any,
    axis: list[float] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """我复制分型线并放样生成过渡曲面候选，只作为人工复核输入。"""

    if axis is None or not np.any(np.abs(axis) > 1e-12):
        axis = [0.0, 0.0, 1.0]
    normal = np.asarray(axis, dtype=np.float64)
    normal = normal / float(np.linalg.norm(normal))
    # 沿分型方向度量零件跨度，而不是固定 Z 跨度。
    corners = np.asarray(
        [
            [bounds[0], bounds[1], bounds[2]],
            [bounds[3], bounds[1], bounds[2]],
            [bounds[0], bounds[4], bounds[2]],
            [bounds[3], bounds[4], bounds[2]],
            [bounds[0], bounds[1], bounds[5]],
            [bounds[3], bounds[1], bounds[5]],
            [bounds[0], bounds[4], bounds[5]],
            [bounds[3], bounds[4], bounds[5]],
        ],
        dtype=np.float64,
    )
    projected = corners @ normal
    axis_span = max(float(projected.max() - projected.min()), 0.0)
    # 缺省偏移不写死，而是取"外扩边距、零件跨度的 5%、1mm"三者最大值：
    # 小件上 1mm 是下限保证放样不成退化面，大件上按比例才不会缩成一条缝。
    offset = (
        float(config.transition_offset_mm)
        # 未显式配置时按"边距、轴向跨度的 5%、1mm"三者取最大：
        # 跨度比例保证大件上过渡面不会小到没有意义，
        # 1mm 下限保证小件上不会退化成零。
        if config.transition_offset_mm is not None
        else max(float(config.surface_margin_mm), 0.05 * axis_span, 1.0)
    )
    # 偏移量沿法向分解到三个轴：后续平移用的是这个向量，
    # 存下来是为了复核"到底往哪个方向偏了多少"。
    offset_vector = [
        float(normal[0] * offset),
        float(normal[1] * offset),
        float(normal[2] * offset),
    ]
    base_diagnostics = {
        # 方法名写死：这份诊断只对"两条平移 Wire 之间放样"这一种做法成立。
        "transition_surface_method": "ocp_two_wire_ruled_loft_candidate",
        "transition_surface_offset_xyz_mm": offset_vector,
        "transition_surface_offset_mm": offset,
        # 初值一律按"未生成"填，而不是按"没问题"填：
        # 状态 REVIEW_REQUIRED、有效性 False、各计数为 0、
        # 边界是否开放留 None——任何一步没跑完就原样返回它，
        # 于是"没跑"和"跑出来确实没问题"能区分开。
        "transition_surface_status": "REVIEW_REQUIRED",
        "transition_surface_shape_is_valid": False,
        "transition_surface_face_count": 0,
        "transition_surface_edge_count": 0,
        # 裸边数单独记：它是"这片面有没有和邻面接上"的直接证据，
        # 与总边数是两个不同的量。
        "transition_surface_bare_edge_count": 0,
        # 边界是否开放留 None 而不是 False：
        # 没生成面时"不开放"是假结论，留 None 才准确。
        "transition_surface_boundary_is_open": None,
        "transition_surface_intersects_source_shape": False,
        "transition_surface_penetration_evidence": (
            "NOT_GENERATED_REVIEW_REQUIRED"
        ),
        # 缝合状态用 "NOT_EXECUTED_" 前缀而不是空串：
        # 空串会被读成"缝合完成且无异常"。
        "transition_surface_sewing_status": "NOT_EXECUTED_REVIEW_REQUIRED",
        "transition_surface_sewing_shape_is_valid": False,
        "transition_surface_sewing_free_edge_count": 0,
        "transition_surface_sewing_multiple_edge_count": 0,
        "transition_surface_sewing_face_count": 0,
        "transition_surface_sewing_edge_count": 0,
        "transition_surface_sewing_shell_count": 0,
        "transition_surface_sewing_solid_count": 0,
        # 能力边界三条都无条件列出：
        # 做的是什么、没判定什么、没验收什么，
        # 三者缺一条都会让人高估这份候选的可靠程度。
        "transition_surface_limitations": [
            "我只生成两条平移 Wire 之间的放样候选。",
            "我没有把候选曲面判定为最终加工分型面。",
            "我没有执行 G0/G1/G2 连续性、实体布尔切割或加工工艺验收。",
        ],
    }
    # 源 Wire 为空就记失败原因并直接返回：
    # 没有 Wire 就没有放样的输入，继续往下只会撞 OCP 异常。
    if wire is None or wire.IsNull():
        base_diagnostics["transition_surface_failure"] = "SOURCE_WIRE_IS_NULL"
        return None, None, base_diagnostics

    try:
        translation = trsf_type()
        # 我沿分型方向平移复制 Wire，不再固定沿 Z 平移。
        translation.SetTranslation(
            vec_type(
                float(offset_vector[0]),
                float(offset_vector[1]),
                float(offset_vector[2]),
            )
        )
        translated_wire = topo_type.Wire(
            transform_type(wire, translation, True).Shape()
        )
        if translated_wire.IsNull():
            base_diagnostics["transition_surface_failure"] = (
                "TRANSLATED_WIRE_IS_NULL"
            )
            return None, None, base_diagnostics

        loft = loft_type(False, True, float(config.section_tolerance_mm))
        # 放样只需要两条 Wire：原分型线与其沿分型方向的平移副本。
        # 构造两个参数分别是"是否做成实体"与"是否按参数对齐"——
        # 前者给 False 是因为我要的是面不是实体，后者给 True 是为了
        # 让两条线的参数一一对应，避免放样出扭曲的面。
        loft.AddWire(wire)
        loft.AddWire(translated_wire)
        loft.Build()
        if not loft.IsDone() or loft.Shape().IsNull():
            base_diagnostics["transition_surface_failure"] = "LOFT_NOT_DONE"
            return None, None, base_diagnostics

        candidate = loft.Shape()
        face_count = _count_subshapes(candidate, explorer_type, top_abs_face)
        edge_count = _count_subshapes(candidate, explorer_type, top_abs_edge)
        shape_is_valid = bool(analyzer_type(candidate).IsValid())
        edge_face_map = edge_map_type()
        # 自由边（裸边）计数：一条边只被一个面引用就是边界裸边。
        # 它是判断放样面是否完整闭合的关键——闭合的放样面不应有明显裸边。
        top_exp_type.MapShapesAndAncestors_s(
            candidate,
            top_abs_edge,
            top_abs_face,
            edge_face_map,
        )
        bare_edge_count = 0
        for index in range(1, edge_face_map.Extent() + 1):
            if edge_face_map.FindFromIndex(index).Size() == 1:
                bare_edge_count += 1

        # 与源实体求交：过渡面若插进零件内部，几何上是无效的过渡，
        # 必须检测出来。这一步可能抛异常（OCP 在非流形输入上不稳定），
        # 所以单独 try，失败只影响这一项诊断，不让整个曲面生成崩掉。
        intersects_source = False
        try:
            # 第三个参数 False 表示不做近似计算：
            # 近似会让"是否相交"这个判定带上额外容差，
            # 而这里需要的是确定答案。
            section = section_type(shape, candidate, False)
            # FuzzyValue 单独设置：不加容差时，面与实体恰好相切
            # 会因为浮点误差判成不相交，而相切正是分型面常见的情况。
            section.SetFuzzyValue(float(config.section_tolerance_mm))
            section.Build()
            if section.IsDone() and not section.Shape().IsNull():
                # 用"截交结果里有没有边"判断相交：
                # 有边说明真的切出了交线，比看 Shape 是否为空更可靠——
                # OCP 在无交时也可能返回一个非空的空集。
                intersects_source = (
                    _count_subshapes(section.Shape(), explorer_type, top_abs_edge)
                    > 0
                )
        # 截交失败按"不相交"处理：这只是少了一项诊断证据，
        # 后面的状态判定会因此降级为 REVIEW_REQUIRED，
        # 不会漏报也不会让整个分型面生成崩掉。
        except (AttributeError, TypeError, RuntimeError):
            intersects_source = False

        sewn_candidate, sewing_diagnostics = _sew_transition_surface_candidate(
            candidate,
            tolerance_mm=float(config.section_tolerance_mm),
            sewing_type=sewing_type,
            analyzer_type=analyzer_type,
            explorer_type=explorer_type,
            top_abs_edge=top_abs_edge,
            top_abs_face=top_abs_face,
            top_abs_shell=top_abs_shell,
            top_abs_solid=top_abs_solid,
        )
        base_diagnostics.update(
            {
                # CANDIDATE 三个条件：有面、有边、且形状有效。
                # 只有面和边而形状无效，说明放样出来的面自交或退化，
                # 不能作为可用候选交付。
                "transition_surface_status": (
                    "CANDIDATE"
                    if face_count > 0 and edge_count > 0 and shape_is_valid
                    else "REVIEW_REQUIRED"
                ),
                "transition_surface_shape_is_valid": shape_is_valid,
                "transition_surface_face_count": face_count,
                "transition_surface_edge_count": edge_count,
                # 裸边数大于 0 即认为边界开放：闭合的过渡面不该有裸边，
                # 裸边意味着这条边只属于一个面、面上有开口。
                "transition_surface_bare_edge_count": bare_edge_count,
                "transition_surface_boundary_is_open": bare_edge_count > 0,
                "transition_surface_intersects_source_shape": intersects_source,
                "transition_surface_penetration_evidence": (
                    "SECTION_EDGES_PRESENT"
                    if intersects_source
                    else "NO_SECTION_EDGES_OR_CHECK_FAILED"
                ),
                **sewing_diagnostics,
            }
        )
        return candidate, sewn_candidate, base_diagnostics
    except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
        base_diagnostics["transition_surface_failure"] = (
            f"{type(exc).__name__}:{exc}"
        )
        return None, None, base_diagnostics


def _sew_transition_surface_candidate(
    candidate: Any,
    *,
    tolerance_mm: float,
    sewing_type: Any,
    analyzer_type: Any,
    explorer_type: Any,
    top_abs_edge: Any,
    top_abs_face: Any,
    top_abs_shell: Any,
    top_abs_solid: Any,
) -> tuple[Any, dict[str, Any]]:
    """我缝合过渡面候选并记录自由边，不自动补齐开放边界。"""

    diagnostics: dict[str, Any] = {
        # 基线一律先给"未确认"：后面任一步失败都能直接返回这份诊断，
        # 下游不会看到"缝好了"的假状态。
        "transition_surface_sewing_status": "REVIEW_REQUIRED",
        "transition_surface_sewing_shape_is_valid": False,
        "transition_surface_sewing_free_edge_count": 0,
        "transition_surface_sewing_multiple_edge_count": 0,
        "transition_surface_sewing_face_count": 0,
        "transition_surface_sewing_edge_count": 0,
        "transition_surface_sewing_shell_count": 0,
        "transition_surface_sewing_solid_count": 0,
    }
    try:
        # 四个布尔依次是：允许非流形、按容差缝合、做曲面分析、允许退化面。
        # 最后一个给 False 是因为退化面会让"缝合成实体"的判定失去意义。
        sewing = sewing_type(float(tolerance_mm), True, True, True, False)
        sewing.Add(candidate)
        sewing.Perform()
        sewn = sewing.SewedShape()
        if sewn.IsNull():
            diagnostics["transition_surface_sewing_failure"] = (
                "SEWED_SHAPE_IS_NULL"
            )
            return None, diagnostics
        face_count = _count_subshapes(sewn, explorer_type, top_abs_face)
        edge_count = _count_subshapes(sewn, explorer_type, top_abs_edge)
        shell_count = _count_subshapes(sewn, explorer_type, top_abs_shell)
        solid_count = _count_subshapes(sewn, explorer_type, top_abs_solid)
        shape_is_valid = bool(analyzer_type(sewn).IsValid())
        # 自由边与多重边是缝合质量的两个硬指标：
        # 自由边非 0 说明还有开口，多重边非 0 说明有面被重复粘合。
        free_edge_count = int(sewing.NbFreeEdges())
        multiple_edge_count = int(sewing.NbMultipleEdges())
        # 三档结论：有效且无自由边且有实体 -> 缝成实体（最理想）；
        # 有效但只有面边 -> 至少缝成了完整壳；都不满足 -> 仍需人工复核。
        status = (
            "SEWN_CLOSED"
            if shape_is_valid and free_edge_count == 0 and solid_count > 0
            else "SEWN_OPEN_SHELL"
            if shape_is_valid and face_count > 0 and edge_count > 0
            else "REVIEW_REQUIRED"
        )
        diagnostics.update(
            {
                "transition_surface_sewing_status": status,
                "transition_surface_sewing_shape_is_valid": shape_is_valid,
                "transition_surface_sewing_free_edge_count": free_edge_count,
                "transition_surface_sewing_multiple_edge_count": (
                    multiple_edge_count
                ),
                "transition_surface_sewing_face_count": face_count,
                "transition_surface_sewing_edge_count": edge_count,
                "transition_surface_sewing_shell_count": shell_count,
                "transition_surface_sewing_solid_count": solid_count,
                "transition_surface_sewing_boundary_is_open": (
                    free_edge_count > 0
                ),
            }
        )
        return sewn, diagnostics
    except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
        diagnostics["transition_surface_sewing_failure"] = (
            f"{type(exc).__name__}:{exc}"
        )
        return None, diagnostics


def _count_subshapes(shape: Any, explorer_type: Any, shape_type: Any) -> int:
    """我统计 Shape 下某一拓扑类型的子形状数量。

    我为什么手动 ``while explorer.More()`` 计数而不用 ``TopExp.MapShapes``：
    MapShapes 要建一个 IndexedMap，对有几万个面的模具是实打实的内存
    与哈希开销；而这里只想知道"有多少个"，不需要拿到下标映射。
    手动遍历只读不写，代价是一次线性扫描。
    """
    count = 0
    explorer = explorer_type(shape, shape_type)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _detect_parting_axis(
    shape: Any,
    bounds: tuple[float, float, float, float, float, float],
    config: PartingConfig,
) -> tuple[list[float], str, dict[str, object]]:
    """我判断分型方向（开模方向），不再一律假设它是 Z 轴。

    算法原理：分型线是沿开模方向的最大外轮廓，所以开模方向必须先定对。
    我用两条互相独立的证据来选方向，并显式记录选了哪条、置信度如何：
      1. 主惯性轴：计算实体的惯量主轴，取第一主轴。模具件在开模方向上通常
         是"扁平"的，主惯量轴往往就是开模方向；
      2. 最小包围盒跨度：若主惯性轴不可信（惯量计算失败，或主轴与三个坐标
         轴都不接近、又没有明显扁平特征），我退回"最短包围盒跨度"方向——
         模具沿开模方向通常是厚度最小的方向。

    我为什么不能写死 Z 轴：甲方 STEP 由不同 CAD 系统导出，坐标系朝向不统一。
    实测 `Plastic Bag Clip Mould.stp` 的第一主惯量轴是 (0.847, -0.027, 0.531)，
    与 Z 轴夹角约 58 度；写死 Z 会切出完全错误的截面。

    返回值：单位方向向量、证据标记、可写入报告的诊断字典。
    方向只取正负号确定的轴；轴向正反向不影响"最大外轮廓"的几何结果，
    所以我统一把首个非零分量取正，保证同一次运行内可复现。
    """

    spans = (
        float(bounds[3] - bounds[0]),
        float(bounds[4] - bounds[1]),
        float(bounds[5] - bounds[2]),
    )
    diagnostics: dict[str, object] = {
        "parting_axis_method": "FALLBACK_MIN_BBOX_SPAN",
        "parting_axis_xyz": [0.0, 0.0, 1.0],
        "parting_axis_confidence": "LOW",
        "parting_axis_bbox_spans_mm": list(spans),
    }
    axis: list[float] | None = None
    try:
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps

        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        principal = props.PrincipalProperties()
        direction = principal.FirstAxisOfInertia()
        vector = [float(direction.X()), float(direction.Y()), float(direction.Z())]
        norm = float(np.linalg.norm(vector))
        if norm > 0.0 and np.all(np.isfinite(vector)):
            axis = [value / norm for value in vector]
            diagnostics["parting_axis_method"] = "PRINCIPAL_INERTIA_AXIS"
            diagnostics["parting_axis_confidence"] = "MEDIUM"
    except (ImportError, AttributeError, TypeError, RuntimeError, ValueError):
        axis = None

    if axis is None:
        # 惯量算不出来就退回最短包围盒跨度方向（模具厚度方向）。
        axis = [0.0, 0.0, 0.0]
        axis[int(np.argmin(spans))] = 1.0
        diagnostics["parting_axis_method"] = "FALLBACK_MIN_BBOX_SPAN"
        diagnostics["parting_axis_confidence"] = "LOW"
    else:
        # 主轴与最短跨度方向一致时置信度提升到 HIGH；否则保持 MEDIUM 交人工复核。
        min_span_axis = [0.0, 0.0, 0.0]
        min_span_axis[int(np.argmin(spans))] = 1.0
        if abs(float(np.dot(axis, min_span_axis))) > 0.9:
            diagnostics["parting_axis_confidence"] = "HIGH"

    # 符号归一：把首个绝对值大于阈值的分量取正，保证结果可复现。
    for index, value in enumerate(axis):
        if abs(value) > 1e-9:
            if value < 0.0:
                axis = [-component for component in axis]
            break
    axis = [float(value) for value in axis]
    diagnostics["parting_axis_xyz"] = axis
    return axis, str(diagnostics["parting_axis_method"]), diagnostics


def _surface_penetration_evidence(
    shape: Any,
    surface_face: Any,
    *,
    explorer_type: Any,
    top_abs_edge: Any,
) -> tuple[bool, str]:
    """我对分型面自身与实体重新求交，判断是否真的穿透。

    原理说明：判断"面是否穿过实体"只能拿这个面和实体做一次真实布尔求交，
    看结果里有没有交线（Edge）。我不能直接复用截面 Edge 数量，因为分型面是
    按零件 XY 包围盒外扩出来的，只要选中高度切到了实体就必然有截交边，
    那个数量与"面是否穿透"无关，恒为真会让门禁永远通过。

    返回值第二项区分三种情况：确实有交线、确认无交线、求交本身失败。
    求交失败时我返回 False 并标 CHECK_FAILED，让上层降级复核，而不是假装没穿透。
    """

    try:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    except ImportError:
        return False, "CHECK_FAILED_NO_OCP"
    try:
        section = BRepAlgoAPI_Section(shape, surface_face, False)
        section.Build()
        if not section.IsDone() or section.Shape().IsNull():
            return False, "CHECK_FAILED_SECTION_NULL"
        edge_count = _count_subshapes(section.Shape(), explorer_type, top_abs_edge)
    except (AttributeError, TypeError, RuntimeError, ValueError):
        return False, "CHECK_FAILED_EXCEPTION"
    return bool(edge_count > 0), (
        "SURFACE_SECTION_EDGES_PRESENT" if edge_count > 0 else "NO_SURFACE_SECTION_EDGES"
    )


def _scan_plane_levels(
    bounds: tuple[float, float, float, float, float, float],
    config: PartingConfig,
    axis: list[float] | None = None,
) -> list[float]:
    """我按配置返回沿分型方向的扫描位置列表（固定值或等距扫描）。

    我不再假设分型方向就是 Z 轴：把包围盒 8 个角点投影到分型轴上，
    得到该方向上的最小/最大位置，再在这个区间内等距扫描。分型轴恰为 Z 时，
    投影结果与原来的 Z 区间完全一致，历史行为不变。
    """

    if axis is None or not np.any(np.abs(axis) > 1e-12):
        axis = [0.0, 0.0, 1.0]
    vector = np.asarray(axis, dtype=np.float64)
    corners = np.asarray(
        [
            [bounds[0], bounds[1], bounds[2]],
            [bounds[3], bounds[1], bounds[2]],
            [bounds[0], bounds[4], bounds[2]],
            [bounds[3], bounds[4], bounds[2]],
            [bounds[0], bounds[1], bounds[5]],
            [bounds[3], bounds[1], bounds[5]],
            [bounds[0], bounds[4], bounds[5]],
            [bounds[3], bounds[4], bounds[5]],
        ],
        dtype=np.float64,
    )
    projected = corners @ vector
    z_min, z_max = float(projected.min()), float(projected.max())
    # 配置了固定高度就直接用单平面。
    if config.plane_z_mm is not None:
        return [float(config.plane_z_mm)]
    span = z_max - z_min
    # Z 跨度太小或只扫描一次时退化为中位平面，避免除零与退化截面。
    if span <= config.section_tolerance_mm or config.plane_scan_count == 1:
        return [float((z_min + z_max) / 2.0)]
    # 两侧留边距后等距取 plane_scan_count 个高度。
    margin = span * config.plane_margin_ratio
    return np.linspace(
        z_min + margin,
        z_max - margin,
        config.plane_scan_count,
        dtype=np.float64,
    ).tolist()


def _plane_frame(axis: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """我为分型平面构造一组与法向正交的单位基向量（U, V）。

    原理：给定法向后，我先取一个与它最不平行的坐标轴做叉积得到 U，
    再用 V = n × U 得到第二个基。选"最不平行"的轴是为了避免叉积退化成零向量。

    我为什么用 ``argmin(|normal|)`` 选参考轴而不是固定取 X 轴：
    当法向正好平行于 X 轴时，n × X 是零向量，归一化会除零、得到 nan。
    取法向上绝对值最小的那个坐标轴，等价于取与法向最接近垂直的轴，
    叉积模长最大，数值最稳。这是这道除零的唯一可靠规避方式。

    我为什么允许 norm <= 0 时退到 Z 轴：调用方可能传入未归一化的
    零向量（上游取不到方向时）。抛错会让整条分型线提取失败，
    而退到 Z 轴只是把这一份默认成"按 Z 开模"，仍可出结果并由告警注明。
    """

    normal = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / norm
    # 取与法向夹角最大的坐标轴，叉积模长最大，数值最稳。
    candidates = np.eye(3, dtype=np.float64)
    reference = candidates[int(np.argmin(np.abs(normal)))]
    u = np.cross(normal, reference)
    u = u / float(np.linalg.norm(u))
    v = np.cross(normal, u)
    v = v / float(np.linalg.norm(v))
    return u, v


def _plane_extent(
    bounds: tuple[float, float, float, float, float, float],
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[float, float, float, float]:
    """我算出包围盒在分型平面两个基方向上的投影范围，保证截面能盖住整个零件。

    我不再直接用 XY 包围盒：当分型方向不是 Z 轴时，XY 矩形投影到倾斜平面上
    会露出零件，截出来的轮廓会被平面边界截断，而不是真实的外轮廓。
    """

    corners = np.asarray(
        [
            [bounds[0], bounds[1], bounds[2]],
            [bounds[3], bounds[1], bounds[2]],
            [bounds[0], bounds[4], bounds[2]],
            [bounds[3], bounds[4], bounds[2]],
            [bounds[0], bounds[1], bounds[5]],
            [bounds[3], bounds[1], bounds[5]],
            [bounds[0], bounds[4], bounds[5]],
            [bounds[3], bounds[4], bounds[5]],
        ],
        dtype=np.float64,
    )
    u_values = corners @ u
    v_values = corners @ v
    u_span = max(float(u_values.max() - u_values.min()), 1.0)
    v_span = max(float(v_values.max() - v_values.min()), 1.0)
    # 外扩 padding 防止零件边界刚好压在截面边界上被切掉。
    u_pad = max(10.0, 0.1 * u_span)
    v_pad = max(10.0, 0.1 * v_span)
    return (
        float(u_values.min() - u_pad),
        float(u_values.max() + u_pad),
        float(v_values.min() - v_pad),
        float(v_values.max() + v_pad),
    )


def _build_section(
    shape: Any,
    bounds: tuple[float, float, float, float, float, float],
    plane_z: float,
    config: PartingConfig,
    section_type: Any,
    make_face_type: Any,
    gp_dir_type: Any,
    gp_pln_type: Any,
    gp_pnt_type: Any,
    sequence_type: Any,
    free_bounds_type: Any,
    top_abs_edge: Any,
    explorer_type: Any,
    topods_type: Any,
    axis: list[float] | None = None,
) -> dict[str, Any]:
    """我在给定分型位置上用平面截交实体，返回 Wire 候选与 Edge 拓扑登记表。

    我按分型方向建平面，而不是固定建 XY 平面：平面原点沿法向平移 plane_z，
    平面内两个基向量由 `_plane_frame` 给出，矩形范围由 `_plane_extent`
    按包围盒真实投影算出。分型轴为 Z 时这些结果与原来的实现完全等价。
    """

    if axis is None or not np.any(np.abs(axis) > 1e-12):
        axis = [0.0, 0.0, 1.0]
    normal = np.asarray(axis, dtype=np.float64)
    normal = normal / float(np.linalg.norm(normal))
    u, v = _plane_frame(axis)
    u_min, u_max, v_min, v_max = _plane_extent(bounds, u, v)
    origin = normal * float(plane_z)
    plane = gp_pln_type(
        gp_pnt_type(float(origin[0]), float(origin[1]), float(origin[2])),
        gp_dir_type(float(normal[0]), float(normal[1]), float(normal[2])),
    )
    face_builder = make_face_type(
        plane,
        u_min,
        u_max,
        v_min,
        v_max,
    )
    if not face_builder.IsDone():
        raise GeometryProcessingError(
            f"我无法建立 Z={plane_z:g} 的截面平面: {face_builder.Error()}"
        )
    # SetFuzzyValue 用容差吸收微小几何误差，减少近似共面导致的断裂。
    section = section_type(shape, face_builder.Face(), False)
    section.SetFuzzyValue(float(config.section_tolerance_mm))
    section.Build()
    if not section.IsDone() or section.Shape().IsNull():
        return {"wire_count": 0, "edge_count": 0, "candidates": [], "section_api": section}

    # 逐条登记截面 Edge，哈希桶用于后续反查拓扑编号。
    edges = sequence_type()
    edge_hash_buckets: dict[int, list[int]] = {}
    edge_by_id: dict[int, Any] = {}
    section_edge_ids: list[int] = []
    edge_explorer = explorer_type(section.Shape(), top_abs_edge)
    edge_count = 0
    while edge_explorer.More():
        edge = topods_type.Edge(edge_explorer.Current())
        edges.Append(edge)
        edge_id = _register_topology_shape(
            edge,
            len(edge_by_id) + 1,
            edge_hash_buckets,
            edge_by_id,
        )
        if edge_id > 0 and edge_id not in section_edge_ids:
            section_edge_ids.append(edge_id)
        edge_count += 1
        edge_explorer.Next()
    if edge_count == 0:
        return {
            "wire_count": 0,
            "edge_count": 0,
            "section_edge_ids": [],
            "edge_hash_buckets": {},
            "edge_by_id": {},
            "candidates": [],
            "section_api": section,
        }

    # 把散乱 Edge 按容差连成 Wire 候选，闭合的 Wire 才可能是分型线。
    wires = free_bounds_type.ConnectEdgesToWires_s(
        edges,
        float(config.section_tolerance_mm),
        True,
    )
    candidates: list[dict[str, Any]] = []
    for index in range(1, wires.Length() + 1):
        wire = topods_type.Wire(wires.Value(index))
        if not wire.IsNull():
            candidates.append({"wire": wire})
    return {
        "wire_count": len(candidates),
        "edge_count": edge_count,
        "section_edge_ids": section_edge_ids,
        "edge_hash_buckets": edge_hash_buckets,
        "edge_by_id": edge_by_id,
        "candidates": candidates,
        "section_api": section,
    }


def _line_from_wire(
    wire: Any,
    plane_z: float,
    config: PartingConfig,
    curve_type: Any,
    wire_explorer_type: Any,
    topods_type: Any,
    face_type: Any,
    section_api: Any,
    edge_hash_buckets: dict[int, list[int]],
    edge_by_id: dict[int, Any],
    face_hash_buckets: dict[int, list[int]],
    face_by_id: dict[int, Any],
) -> tuple[PartingLineResult | None, Any, Any]:
    """我把一条 Wire 采样成离散分型线，并回填来源 Edge/Face 编号。

    我沿 Wire 逐条边采样并同时记录两个来源编号：
      - section_edge_ids：截面边编号，回答"这条分型线由哪些边构成"；
      - source_face_ids：通过 HasAncestorFaceOn1 回溯每条边来自哪个面，
        回答"这条线切过了哪些面"。
    两者都回填，后续才能把分型线关联回具体几何面，而不是只留一串
    无来源的坐标点。

    编号查找用哈希桶加 IsSame，与 step_faces 同一套判同逻辑：
    OCP 的 TopoDS 对象没有稳定整数主键，必须自己发号并判重，
    否则同一条边在两次遍历中会被记成两个不同编号，来源关联就乱了。

    异常策略：祖先面回溯失败只跳过该条边的面关联（pass），
    不影响这条边本身的采样。面关联是增强信息，它缺失不该丢掉
    整条分型线的坐标。
    """
    explorer = wire_explorer_type(wire)
    points: list[np.ndarray] = []
    section_edge_ids: list[int] = []
    source_face_ids: list[int] = []
    edge_count = 0
    while explorer.More():
        edge = topods_type.Edge(explorer.Current())
        edge_id = _find_topology_shape_id(
            edge,
            edge_hash_buckets,
            edge_by_id,
        )
        if edge_id > 0 and edge_id not in section_edge_ids:
            section_edge_ids.append(edge_id)
        ancestor_face = face_type()
        try:
            if section_api.HasAncestorFaceOn1(edge, ancestor_face):
                source_face_id = _find_topology_shape_id(
                    ancestor_face,
                    face_hash_buckets,
                    face_by_id,
                )
                if source_face_id > 0 and source_face_id not in source_face_ids:
                    source_face_ids.append(source_face_id)
        except (AttributeError, TypeError, RuntimeError):
            pass
        curve = curve_type(edge)
        # 参数区间退化（first == last）或非有限说明这条边取不出曲线，
        # 跳过而不是让后面的采样算出除零或 NaN 坐标。
        first = float(curve.FirstParameter())
        last = float(curve.LastParameter())
        if not np.isfinite(first) or not np.isfinite(last) or first == last:
            explorer.Next()
            continue
        # 等参数采样整条边：分型线要的是曲线形状而不是端点连线，
        # 圆弧若只取两端会被压成弦，周长和面积都会偏小。
        edge_points: list[np.ndarray] = []
        for sample_id in range(config.curve_samples_per_edge):
            ratio = sample_id / (config.curve_samples_per_edge - 1)
            point = curve.Value(first + (last - first) * ratio)
            candidate = np.asarray([point.X(), point.Y(), point.Z()], dtype=np.float64)
            edge_points.append(candidate)
        # 采样点顺序取决于曲线参数方向，可能与上一條边的走向相反。
        # 我比较这条边首尾哪一端离上一段末尾更近，必要时整段翻转，
        # 否则拼出来的线会在每条边处来回跳，周长被放大成近乎两倍。
        if points and edge_points:
            distance_to_first = float(
                np.linalg.norm(edge_points[0] - points[-1])
            )
            distance_to_last = float(
                np.linalg.norm(edge_points[-1] - points[-1])
            )
            if distance_to_last < distance_to_first:
                edge_points.reverse()
        # 相邻采样点距离小于容差的视为重复点丢弃：重复点会让鞋带公式
        # 出现零长度边，也让闭合判据被"同一点"误判为已闭合。
        for candidate in edge_points:
            if (
                not points
                or float(np.linalg.norm(candidate - points[-1]))
                > config.section_tolerance_mm
            ):
                points.append(candidate)
        edge_count += 1
        explorer.Next()
    if len(points) < config.min_loop_points:
        return None, wire, None
    if float(np.linalg.norm(points[0] - points[-1])) > config.section_tolerance_mm:
        points.append(points[0].copy())
    values = np.asarray(points, dtype=np.float64)
    closed = bool(wire.Closed()) and float(
        np.linalg.norm(values[0] - values[-1])
    ) <= config.section_tolerance_mm
    warnings = [
        "我当前使用 OCP B-Rep 水平截交生成分型线候选。",
        "我仍需要执行枕型工程规则、台阶桥接、R 角语义和人工参考线验收。",
    ]
    warnings.extend(
        parting_line_warnings(
            values,
            closed=closed,
            tolerance_mm=config.section_tolerance_mm,
            min_area_mm2=config.minimum_line_area_mm2,
        )
    )
    quality_issue = any(
        keyword in warning
        for warning in warnings
        for keyword in ("自交", "投影面积接近零", "重复点", "零长度", "NaN")
    )
    area = abs(_signed_area_xy(values))
    # 我改用环自身拟合平面上的真实面积做排序主指标：XY 投影会在倾斜摆放和
    # 非凸环两种情况下失真，直接拿它选"最大外轮廓"会选错候选。
    true_area, true_area_status = _true_loop_area_mm2(
        values, tolerance_mm=config.section_tolerance_mm
    )
    length = _polyline_length(values)
    line = PartingLineResult(
        # 闭合且无质量问题才是 BASELINE：
        # 不闭合的分型线围不出分型面，
        # 即使其他质量项都好也不能给基线状态。
        status="BASELINE" if closed and not quality_issue else "REVIEW_REQUIRED",
        # 方法名固定：这条线来自 OCP 的 B-Rep 截交，
        # 与网格侧的水平截交不是同一套做法。
        method="ocp_brep_section",
        plane_z_mm=float(plane_z),
        points_xyz_mm=values.tolist(),
        closed=closed,
        length_mm=length,
        area_xy_mm2=area,
        # 打分 = 面积 + 0.01 × 周长：面积是主项，
        # 周长只作 0.01 权重的次级项，
        # 用于在面积相近时偏向更长的（更完整的）那条。
        score=true_area + 0.01 * length,
        # 警告去重：同一条警告可能由多个判据各触发一次。
        warnings=list(dict.fromkeys(warnings)),
        diagnostics={
            # 边数与边号一起记：编号用于定位这些边来自哪些面，
            # 计数用于判断这条线的复杂程度。
            "edge_count": edge_count,
            "section_edge_ids": section_edge_ids,
            "source_face_ids": source_face_ids,
            "source_face_count": len(source_face_ids),
            # 边的来源固定为截交结果：不同来源的边
            # 精度与可信度不同，必须能区分。
            "edge_source": "ocp_section_result",
            # 真实环面积与它的状态分开存：
            # 状态说明这个面积算得可不可信，
            # 只有数值的话无法判断该不该用它。
            "true_loop_area_mm2": true_area,
            "true_loop_area_status": true_area_status,
        },
    )
    # 通用诊断追加进来而不是替换：上面那组是"这条线怎么来的"，
    # 这组是"这条线本身有什么问题"，两类信息互补。
    line.diagnostics.update(
        parting_line_diagnostics(
            values,
            closed=closed,
            tolerance_mm=config.section_tolerance_mm,
        )
    )
    # 需要搭桥就警告：断段过长说明存在台阶缺口，
    # 桥接路径不能自动决定——自动连会把台阶连成斜面。
    if line.diagnostics["bridge_required"]:
        line.warnings.append(
            "我发现分型线存在相对主段过长的候选断段，阶段5要求人工确认台阶缺口桥接路径。"
        )
    # 尖角警告：尖角在加工上要么保留要么倒 R 角，
    # 这是工艺决策，不由几何判据决定。
    if line.diagnostics["sharp_corner_indices"]:
        line.warnings.append(
            "我发现分型线存在尖角候选，阶段5要求确认R角保留或工程圆角规则。"
        )
    # 追加之后再去重一次：新的警告可能与已有的重复。
    line.warnings = list(dict.fromkeys(line.warnings))
    return line, wire, _make_planar_face(wire)


def _topology_hash(shape: Any) -> int:
    """我取得 OCP 拓扑对象的快速哈希，失败时退回同一桶。"""

    try:
        return int(hash(shape))
    except Exception:
        return 0


def _find_topology_shape_id(
    shape: Any,
    hash_buckets: dict[int, list[int]],
    shapes_by_id: dict[int, Any],
) -> int:
    """我在当前报告内按哈希桶和 IsSame 找回拓扑对象编号。"""

    key = _topology_hash(shape)
    for shape_id in hash_buckets.get(key, []):
        try:
            if shape.IsSame(shapes_by_id[shape_id]):
                return shape_id
        except Exception:
            continue
    return 0


def _register_topology_shape(
    shape: Any,
    preferred_id: int,
    hash_buckets: dict[int, list[int]],
    shapes_by_id: dict[int, Any],
) -> int:
    """我为当前报告内的唯一拓扑对象分配连续编号。"""

    existing = _find_topology_shape_id(shape, hash_buckets, shapes_by_id)
    if existing > 0:
        return existing
    shape_id = int(preferred_id)
    while shape_id in shapes_by_id:
        shape_id += 1
    shapes_by_id[shape_id] = shape
    hash_buckets.setdefault(_topology_hash(shape), []).append(shape_id)
    return shape_id


def _make_planar_face(wire: Any) -> Any:
    """我用一条闭合 Wire 直接构造平面 Face，构造失败时返回 None。"""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    builder = BRepBuilderAPI_MakeFace(wire, True)
    return builder.Face() if builder.IsDone() else None


def _make_extended_planar_face(
    *,
    bounds: tuple[float, float, float, float, float, float],
    plane_z: float,
    margin_mm: float,
    make_face_type: Any,
    gp_dir_type: Any,
    gp_pln_type: Any,
    gp_pnt_type: Any,
    axis: list[float] | None = None,
) -> Any:
    """我沿分型平面外扩建立可导出的平面分型面候选。

    我按分型方向建面，范围取包围盒在该平面两个基方向上的真实投影再外扩，
    不再固定用 XY 包围盒——否则分型方向不是 Z 时，分型面会盖不住零件。
    """

    if axis is None or not np.any(np.abs(axis) > 1e-12):
        axis = [0.0, 0.0, 1.0]
    normal = np.asarray(axis, dtype=np.float64)
    normal = normal / float(np.linalg.norm(normal))
    u, v = _plane_frame(axis)
    u_min, u_max, v_min, v_max = _plane_extent(bounds, u, v)
    u_pad = max(float(margin_mm), 0.0) + max(u_max - u_min, 1.0) * 0.05
    v_pad = max(float(margin_mm), 0.0) + max(v_max - v_min, 1.0) * 0.05
    origin = normal * float(plane_z)
    plane = gp_pln_type(
        gp_pnt_type(float(origin[0]), float(origin[1]), float(origin[2])),
        gp_dir_type(float(normal[0]), float(normal[1]), float(normal[2])),
    )
    builder = make_face_type(
        plane,
        float(u_min - u_pad),
        float(u_max + u_pad),
        float(v_min - v_pad),
        float(v_max + v_pad),
    )
    return builder.Face() if builder.IsDone() else None


def _polyline_length(points: np.ndarray) -> float:
    """我返回折线相邻点欧氏距离之和，即折线总长。

    这里不自动闭合首尾：调用方传进来的可能是开口链，
    自动补一段首尾连线会把"开口"伪装成"闭合"，
    周长也会多出一段并不存在的边。是否闭合由调用方决定。
    """
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _signed_area_xy(points: np.ndarray) -> float:
    """我用鞋带公式算折线在 XY 平面的有符号投影面积（自动闭合）。

    与 _polyline_length 相反，这里**会**自动闭合：面积的定义本身就
    要求首尾相接，开口折线围不出面积。容差用 atol=1e-12 判断首尾
    是否已重合，避免重复补点导致末边长度为零。

    注意这是 XY 投影面积，不是真实围成面积。真实面积请走
    _true_loop_area_mm2（在环自身拟合平面上求），
    两者不可混用——斜置的环在这里会显著偏小。
    """
    xy = points[:, :2]
    if not np.allclose(xy[0], xy[-1], atol=1e-12, rtol=0.0):
        xy = np.vstack([xy, xy[0]])
    return float(
        0.5 * np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1])
    )


def _true_loop_area_mm2(
    points: np.ndarray,
    *,
    tolerance_mm: float,
) -> tuple[float, str]:
    """我返回闭合环沿自身最佳拟合平面围成的真实面积。

    算法原理：分型线是三维空间曲线，它围成的面积必须沿自己所在的平面计算。
    我先把点列平移到质心，再做 SVD：最小奇异值对应拟合法向，前两个奇异向量
    张成拟合平面。把点投影到这两个基上后用鞋带公式求面积，即为真实围成面积。

    我为什么不再直接用 XY 投影面积：
      分型环是在"垂直于开模方向"的平面上截出来的，而开模方向不一定是 Z 轴。
      环平面与 XY 夹角为 theta 时，XY 投影面积恰好等于真实面积乘 cos(theta)；
      环平面垂直于 XY 时会退化到接近 0，于是真正的外轮廓会被判成"面积为零的
      垃圾候选"，排序就会选错。我改成在环自身的拟合平面上求面积，与摆放角度无关。

    我顺带说明一个容易搞错的点：鞋带公式对简单非凸多边形同样是精确的
    （内凹部分不会被"减掉"），所以这里修的不是凸性问题，而是投影方向问题。
    真正会让面积失真的是环点排序错误（例如多段环被拼接成一条点列），
    那种情况由上游的 Wire 排序与闭合检查负责，本函数只如实给出 DEGENERATE/正常结果。

    返回的第二个值是证据标记，供甲方复核这个面积能不能信。
    """

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        return 0.0, "INSUFFICIENT_POINTS"
    if not np.all(np.isfinite(values)):
        return 0.0, "NON_FINITE_POINTS"
    # 闭合环首尾点重复，不应参与面积求和。
    if float(np.linalg.norm(values[0] - values[-1])) <= tolerance_mm:
        values = values[:-1]
    if len(values) < 3:
        return 0.0, "INSUFFICIENT_POINTS"
    centered = values - values.mean(axis=0)
    try:
        _, singular_values, basis = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0, "SVD_FAILED"
    if basis.shape[0] < 2:
        return 0.0, "DEGENERATE_POINTS"
    planar = np.column_stack([centered @ basis[0], centered @ basis[1]])
    area = abs(_signed_area_xy(planar))
    values_1d = np.asarray(singular_values, dtype=np.float64).ravel()
    if len(values_1d) < 2 or float(values_1d[1]) <= tolerance_mm:
        # 第二个奇异值约为 0 说明点几乎共线，围不出有效面积。
        return area, "DEGENERATE_POINTS"
    return area, "COMPUTED"
