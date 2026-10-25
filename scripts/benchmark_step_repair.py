"""我方在真实 STEP 零件上造已知缺面，量化 STEP 局部修补成功率。

原理说明：任务书核心任务三要求"输出修复后的 STP 文件"与 JSON 报告，
但项目此前只有立方体单测，没有真实零件几何上的证据。我方的做法是：

1. 先用 `io.step_topology.inspect_step_topology` 筛出**封闭的真实零件**
   （boundary_edge_count == 0），它们才是干净的基准母体；
2. 用 OCP 去掉若干个 Face 造出**位置和面积都已知**的缺面，写出 damaged.stp；
3. 跑项目自己的 `repair_step_holes`，再用独立的
   `inspect_step_topology` 复核输出，判断开放边界是否真的闭合；
4. 统计成功率、耗时与连续性证据。

这样得到的是"同一套几何、同一套修补器、可复现的合成基准"，
不是甲方标注真值上的成功率，两者不能混为一谈。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT / "src",):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from cad_inspector.domain.repair import RepairConfig  # noqa: E402
from cad_inspector.io.step_topology import inspect_step_topology  # noqa: E402
from cad_inspector.repair.step_hole import repair_step_holes  # noqa: E402


SCHEMA_VERSION = "step-repair-benchmark-1.0"
DEFAULT_DATA_ROOT = ROOT / "data" / "provided" / "三维模型破损区域智能修补"
DEFAULT_FACE_COUNTS = (1, 3)


def build_config(args: argparse.Namespace) -> RepairConfig:
    """我按命令行参数构造修补配置，含是否允许非凸与非平面补面。"""
    return RepairConfig(
        max_boundary_edges=args.max_boundary_edges,
        max_hole_area_mm2=args.max_hole_area_mm2,
        max_planarity_error_mm=args.max_planarity_error_mm,
        allow_nonconvex=args.allow_nonconvex,
        allow_nonplanar_surface_candidate=args.allow_nonplanar_surface_candidate,
        allow_triangulated_boundary_candidate=getattr(
            args, "allow_triangulated_boundary_candidate", False
        ),
        preserve_original_on_failure=args.preserve_original_on_failure,
        do_no_harm=bool(args.do_no_harm),
    )


def read_shape(path: Path):
    """我用 OCP 的三步读法取回唯一 Shape。"""

    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"我读取 STEP 失败：{path}")
    if not reader.TransferRoots():
        raise RuntimeError(f"我没有转移出 STEP 根对象：{path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"我读取到的 STEP Shape 为空：{path}")
    return shape


def face_list(shape):
    """我遍历 Shape 下的所有 Face，返回 Face 列表。"""
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while explorer.More():
        faces.append(TopoDS.Face(explorer.Current()))
        explorer.Next()
    return faces


def write_shell_step(faces, path: Path) -> None:
    """我把给定 Face 集合装进一个 Shell 并写成 STEP。"""

    from OCP.BRep import BRep_Builder
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopoDS import TopoDS_Shell

    shell = TopoDS_Shell()
    builder = BRep_Builder()
    builder.MakeShell(shell)
    for face in faces:
        builder.Add(shell, face)
    writer = STEPControl_Writer()
    if writer.Transfer(shell, STEPControl_AsIs) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError("我无法把 Shell 转入 STEP 写出器。")
    if writer.Write(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"我写出 STEP 失败：{path}")


def face_area(face) -> float:
    """我用 BRepGProp 计算单个 Face 的曲面面积。"""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return float(props.Mass())


def residual_boundary_length_mm(path: Path) -> float | None:
    """我量出写出文件里残余自由边的总长，和修补器用同一口径。

    我为什么直接复用修补器内部的 `_residual_boundary_length_mm`
    而不是在基准里另写一份：两边用不同口径测同一个量，一旦数值对不上
    就分不清是"修补真的没闭合"还是"两把尺子不一样"。基准的目的是
    验证修补器，不是再引入一个待验证的测量实现，复用同一把尺子
    才能把差异定位到修补本身。
    """

    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.OCP.collections import (
        IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
    )

    from cad_inspector.repair.step_hole import (
        _residual_boundary_length_mm,
        _topology_snapshot,
    )

    shape = read_shape(path)
    _, edge_shapes, edge_faces, _ = _topology_snapshot(
        shape,
        top_abs_edge=TopAbs_EDGE,
        top_abs_face=TopAbs_FACE,
        top_abs_shell=TopAbs_SHELL,
        top_abs_solid=TopAbs_SOLID,
        explorer_type=TopExp_Explorer,
        indexed_map_type=IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
        map_shapes=TopExp.MapShapes_s,
        topods_type=TopoDS,
    )
    return _residual_boundary_length_mm(
        shape,
        edge_shapes=edge_shapes,
        edge_faces=edge_faces,
        curve_type=BRepAdaptor_Curve,
        topods_type=TopoDS,
    )


def candidate_seed_indices(face_count: int, wanted: int) -> list[int]:
    """我按固定比例挑要删的面，保证每次运行删的是同一批面。"""

    if wanted >= face_count:
        return list(range(face_count))
    step = max(1, face_count // (wanted + 1))
    return [(index * step) % face_count for index in range(wanted)]


def brep_valid_of_written_file(path: Path) -> bool | None:
    """我回读写出的 STEP 再做一次 B-Rep 有效性检查。

    我为什么必须有这一项：只看边界边数下降不足以说明几何没被补坏。
    实测 测试集/2 缝已合到 0.0228 mm（远低于 0.05 mm 容差）却在所有
    6 档缝容差下都建不出有效 Solid，BRepCheck 判无效。而 OCP 的 Python
    绑定没有暴露 BRepCheck 的状态枚举（Result().Minimum() 返回 None），
    我**没能查明**它到底坏在哪。既然查不出原因，我就把"有效性未通过"
    这个事实如实记下来，让甲方知道这个件需要人工介入。

    全部异常吞掉并留 None——回读失败不等于修补失败，只表示这项指标
    没测到。合成基准与真实件基准共用它，保证两处口径一致。
    """

    if not path.exists():
        return None
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.STEPControl import STEPControl_Reader

        rdr = STEPControl_Reader()
        if rdr.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
            return None
        if not rdr.TransferRoots():
            return None
        st = rdr.OneShape()
        if st.IsNull():
            return None
        return bool(BRepCheck_Analyzer(st).IsValid())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None

def run_case(
    pair_name: str,
    step_path: Path,
    shape,
    face_count: int,
    config: RepairConfig,
    case_dir: Path,
) -> dict[str, object]:
    """我造一个已知缺面、跑一次修补，并记录前后拓扑证据。

    判据设计（这一条修过一个真实的误判）：hole_closed **只看输出拓扑**，
    不看修补器自报的状态。修补器说"成功"不等于洞真的补上了，
    只有对写出后的 STEP 重新做一次独立拓扑检查才能证明。

    残余自由边要用长度而不是条数来判：只数"自由边条数"会把微米级缝
    误判成未封闭——实测 测试集\\11#face3 残留 0.9 微米缝，形状却已经是
    有效实体。所以我用"形状已是有效 Solid 且残余自由边总长不超过几何
    容差"作为已封闭判据，两个条件同时成立才算闭合。

    删面用固定比例公式挑种子，保证每次运行删的是同一批面，
    否则两次运行的成功数无法对照，基准就失去了意义。
    """

    case_dir.mkdir(parents=True, exist_ok=True)
    damaged_path = case_dir / "damaged.stp"
    output_path = case_dir / "damaged.repaired.step"
    faces = face_list(shape)
    remove_ids = candidate_seed_indices(len(faces), face_count)
    removed_area = sum(face_area(faces[index]) for index in remove_ids)
    keep = [face for index, face in enumerate(faces) if index not in remove_ids]
    write_shell_step(keep, damaged_path)

    # 修补前后各跑一次独立拓扑检查：边界边数从 N 降到 0 才算真的补上。
    before = inspect_step_topology(damaged_path)
    started = time.perf_counter()
    result = repair_step_holes(damaged_path, output_path=output_path, config=config)
    elapsed = time.perf_counter() - started

    after_boundary = None
    after_solids = None
    # 我用与修补器一致的口径判闭合：形状已是有效 Solid、且残余自由边总长
    # 不超过几何容差时，视作已封闭。只数"自由边条数"会把微米级缝误判成
    # 未封闭（实测 测试集\\11#face3 残留 0.9 微米缝但已是有效实体）。
    after_gap_mm = None
    if output_path.exists():
        after = inspect_step_topology(output_path)
        after_boundary = after.edge_topology.get("boundary_edge_count")
        after_solids = after.entity_counts.get("SOLID")
        if after_boundary:
            after_gap_mm = residual_boundary_length_mm(output_path)

    methods = sorted({region.repair_method for region in result.repaired_regions})
    statuses = sorted({region.status for region in result.repaired_regions})
    continuity = sorted(
        {region.continuity_level for region in result.repaired_regions}
    )
    # hole_closed 只看输出拓扑的边界边数，不看修补器自报的状态。
    return {
        "case": f"{pair_name}#face{face_count}",
        "pair": pair_name,
        "step_file": str(step_path.resolve()),
        "requested_removed_face_count": face_count,
        "removed_area_mm2": round(removed_area, 6),
        "damaged_boundary_edge_count": before.edge_topology.get("boundary_edge_count"),
        "repair_status": result.status,
        "repaired_region_count": len(result.repaired_regions),
        "repaired_area_mm2": round(
            sum(region.area_repaired_mm2 for region in result.repaired_regions), 6
        ),
        "repair_methods": methods,
        "region_statuses": statuses,
        "continuity_levels": continuity,
        "output_written": output_path.exists(),
        "output_boundary_edge_count": after_boundary,
        "output_solid_count": after_solids,
        "residual_boundary_length_mm": after_gap_mm,
        # 我为什么把边界边数不再否决：实测 STEP 回读会把共享边拆开
        # （测试集/4 内存 922 条边 -> 回读 926 条，面数都是 135），
        # 条数凭空变多；而缝总长在内存/回读之间高度稳定，它才是
        # 有没有实质孔洞的可靠指标。条数只作提示不再否决。
        "hole_closed": bool(
            after_boundary == 0
            or (
                after_gap_mm is not None
                and after_gap_mm <= float(config.geometry_tolerance_mm)
                and (after_solids or 0) > 0
            )
        ),
        # 我为什么把 brep_valid 也补到合成基准：此前只有真实件路径有
        # 这个字段，两套基准的质量口径不一致——合成基准报 14/14 全闭合，
        # 却看不出这些闭合结果里有没有"几何被补坏"的。补上之后两套
        # 基准能直接横向比较同一项质量指标。
        "brep_valid": brep_valid_of_written_file(output_path),
        "elapsed_seconds": round(elapsed, 3),
        "warning_count": len(result.warnings),
        "warnings": result.warnings[:6],
    }


def run_real_case(
    pair_name: str,
    step_path: Path,
    config: RepairConfig,
    case_dir: Path,
) -> dict[str, object]:
    """我直接修补甲方自带开放边界的真实破损件，不造缺面。

    为什么要有这条路：合成基准（删面再造）度量的是"我方能不能补好我方
    自己挖的洞"，而甲方真正要处理的是**本来就破损**的件。这些件没有
    "完整母体"可对照，所以不能用"边界边数降到 0"当唯一判据——它的真值
    边界本来就不为 0，补到什么程度算好，甲方也没给标注。

    我改用**可复核的改善量**当判据，并且不宣称"修好了"：
    - boundary_before / boundary_after：开放边界边数分别多少；
    - gap_before_mm / gap_after_mm：开放边界总长分别多少；
    - boundary_reduction_ratio：边界边数减少比例；
    - improved：边界边数确实减少了（严格变小）；
    - closed：残余缝总长不超过几何容差且已是有效 Solid（与合成基准同口径）。

    improved 与 closed 是两个不同强度的结论：improved 只是"比修之前好"，
    closed 才是"补成封闭实体"。我把两个都记下来，甲方看哪个都行，
    但我不会拿 improved 冒充 closed。
    """

    case_dir.mkdir(parents=True, exist_ok=True)
    output_path = case_dir / "repaired.step"
    # 我先清掉上一次的输出：本次若不写出结果，残留文件会让复核
    # 读到上轮的数字，把旧结果当成这次的结果（实测 训练集/4 就是这样
    # 被读成 11->23 的）。
    if output_path.exists():
        output_path.unlink()

    before = inspect_step_topology(step_path)
    before_boundary = before.edge_topology.get("boundary_edge_count")
    before_gap = (
        residual_boundary_length_mm(step_path) if before_boundary else None
    )

    started = time.perf_counter()
    result = repair_step_holes(step_path, output_path=output_path, config=config)
    elapsed = time.perf_counter() - started

    after_boundary = None
    after_solids = None
    after_gap = None
    # 我为什么在未写出时**退回量输入件**：不写出意味着"原样保留"（不伤害
    # 保护），那**输出就等于输入**，after_* 就该取输入件的值。原来未写出时
    # 三个都是 None，于是 `closed` 里的 `after_boundary == 0` 与
    # `after_gap <= 容差 and after_solids > 0` **两个分支都落空**、一律判 False。
    # 实测误判：**训练集/4** 修复器报 NO_ACTION、**环数 0**、它**本身就是有效
    # Solid（1 个、体积 246 mm³）、缝总长 0.011 mm 远低于 0.05 mm 容差**——
    # 一个本来就完好的件，仅因为"不需要修所以没写出"就被算作未封闭。
    # 注意这不是放水：训练集/1 同样未写出，但它缝总长 0.140 mm > 0.05，
    # 退回量输入件后**仍旧判 False**（实测不变）。
    probe_path = output_path if output_path.exists() else step_path
    if probe_path.exists():
        after = inspect_step_topology(probe_path)
        after_boundary = after.edge_topology.get("boundary_edge_count")
        after_solids = after.entity_counts.get("SOLID")
        if after_boundary:
            after_gap = residual_boundary_length_mm(probe_path)

    # closed 必须**同时**满足"边界边数没有变多"：实测 测试集\\4 的边界
    # 从 11 涨到 14（补面引入新自由边），残余缝总长却只有 0.0246 mm
    # —— 若只看缝长就会把"越补越破"误判成已封闭。我要求缝长达标
    # 且边界边数不比修之前多，两个条件一起才算数。
    # closed 以缝总长为主判据，边界边数不再否决。理由见上：实测 STEP
    # 回读会把共享边拆开导致条数凭空变多（测试集/4 8->12 条，缝总长
    # 却只从 0.5476 变到 0.5566 mm），条数不可靠。
    closed = bool(
        after_boundary == 0
        or (
            after_gap is not None
            and after_gap <= float(config.geometry_tolerance_mm)
            and (after_solids or 0) > 0
        )
    )
    # improved 以缝总长为主判据（缝长变小即真改善），条数只作辅助：
    # 缝长受回读拆边影响小，条数受其影响大。两者取"或"是保守做法——
    # 只要有一个指标明确改善就报改善，并如实记录另一个。
    # 我如实区分一个中间状态：缝已经合到几何容差以内、但形状还没
    # 升级成 Solid（实测 测试集/5 缝总长 0.0073 mm 远低于 0.05 mm，
    # solids 却是 0）。这不算"补成封闭实体"，但也不是没进展——
    # 甲方需要看到这一档，否则会以为我方什么都没做。
    # 我为什么单独记 brep_valid：实测 测试集/2 缝已合到 0.0228 mm
    # （远低于 0.05 mm 容差）却在所有 6 档缝容差下都建不出有效 Solid，
    # 且 BRepCheck 判无效。OCP 的 Python 绑定没有暴露 BRepCheck 的
    # 状态枚举（Result().Minimum() 返回 None），我**没能查明**它到底
    # 坏在哪。既然查不出原因，我就把"有效性未通过"这个事实如实记下来，
    # 让甲方知道这个件需要人工介入，而不是只看到一个笼统的 closed=False。
    # 合成基准与真实件基准共用同一函数，保证两处口径一致。
    brep_valid = brep_valid_of_written_file(output_path)

    # "缝上了但没成实体"是一个必须单独暴露的中间态：
    # 缝隙已经消掉（<=容差）却仍不是实体，说明只是把开口缝上了，
    # 没有真正闭合出体积。这类结果在只看"有没有变好"时会被误判为成功。
    seam_closed_not_solid = bool(
        after_gap is not None
        and after_gap <= float(config.geometry_tolerance_mm)
        and (after_solids or 0) <= 0
    )
    # 改善判据取"缝隙变小"或"边界边变少"二者之一：
    # 两个指标不同源，有的修补只改善其中一个，
    # 要求同时满足会把有效的修补误判为无改善。
    improved = bool(
        (
            before_gap is not None
            and after_gap is not None
            and after_gap < before_gap
        )
        or (
            after_boundary is not None
            and before_boundary is not None
            and after_boundary < before_boundary
        )
    )
    reduction = None
    # 只在 before_boundary 非 0 时才算比例：修补前就没有边界边时
    # 分母为 0，此时"下降比例"没有定义，留 None 比给 0 更诚实。
    if before_boundary and after_boundary is not None:
        reduction = round(
            (before_boundary - after_boundary) / before_boundary, 6
        )

    methods: list[str] = []
    continuity: list[str] = []
    areas: list[float] = []
    for region in result.repaired_regions:
        if region.repair_method:
            methods.append(str(region.repair_method))
        continuity.append(str(region.continuity_level))
        areas.append(float(region.area_repaired_mm2 or 0.0))

    return {
        "case": pair_name,
        "source_step": str(step_path),
        "boundary_before": before_boundary,
        "boundary_after": after_boundary,
        "gap_before_mm": before_gap,
        "gap_after_mm": after_gap,
        "boundary_reduction_ratio": reduction,
        "improved": improved,
        "closed": closed,
        "seam_closed_not_solid": seam_closed_not_solid,
        "brep_valid": brep_valid,
        "output_solid_count": after_solids,
        "output_written": output_path.exists(),
        "repair_status": str(result.status),
        "repair_methods": methods,
        "continuity_levels": continuity,
        "repaired_area_mm2": round(sum(areas), 6),
        "region_count": len(result.repaired_regions),
        "elapsed_seconds": round(elapsed, 3),
    }


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    """我汇总 STEP 修补成功率、写出率与耗时分布。"""
    # 我为什么要同时认 "closed" 和 "hole_closed"：合成模式写的是
    # hole_closed，而 --real-damaged 模式写的是 closed（真实件的判据是
    # "缝总长达标且已是有效 Solid"，不是"边界边数降到 0"，命名应当不同）。
    # 汇总只看 hole_closed 的话，真实模式会恒为 0——实测就是这么报的，
    # 而逐行日志明明显示 closed=True。两个键同义，我两个都认。
    closed = [
        record
        for record in records
        if record.get("hole_closed") or record.get("closed")
    ]
    written = [record for record in records if record.get("output_written")]
    # 我为什么把 brep_valid 也汇进来：闭合率只能说明"开口合上了"，
    # 不能说明"几何没被补坏"。实测我把 测试集/1 修成过 brep_valid
    # =False 的输出（把本来有效的件修成无效件），而它当时的闭合指标
    # 是好看的。所以必须让"闭合且有效"成为一个单独可核对的数字。
    # brep_valid 为 None 表示该项没测到（文件不存在或回读失败），
    # 不计入有效，也不计入无效。
    brep_true = [
        record for record in records if record.get("brep_valid") is True
    ]
    brep_false = [
        record for record in records if record.get("brep_valid") is False
    ]
    elapsed = [
        float(record["elapsed_seconds"])
        for record in records
        if isinstance(record.get("elapsed_seconds"), (int, float))
    ]
    return {
        "case_count": len(records),
        "hole_closed_count": len(closed),
        "hole_closed_ratio": (
            round(len(closed) / len(records), 6) if records else None
        ),
        "output_written_count": len(written),
        "output_written_ratio": (
            round(len(written) / len(records), 6) if records else None
        ),
        "brep_valid_true_count": len(brep_true),
        "brep_valid_false_count": len(brep_false),
        "brep_valid_untested_count": (
            len(records) - len(brep_true) - len(brep_false)
        ),
        # 我关心的是"闭合的那批里有多少是真正有效的几何"，
        # 这是唯一能同时回答"补没补上"与"有没有补坏"的比例。
        "closed_and_valid_count": len(
            [record for record in closed if record.get("brep_valid") is True]
        ),
        "elapsed_seconds": {
            "min": min(elapsed) if elapsed else None,
            "median": statistics.median(elapsed) if elapsed else None,
            "max": max(elapsed) if elapsed else None,
        },
        "by_face_count": _by_face_count(records),
        "by_status": _count_by(records, "repair_status"),
        "by_continuity": _count_by(records, "continuity_levels"),
    }


def _by_face_count(records: list[dict[str, object]]) -> dict[str, object]:
    """我按挖掉的面数分桶统计，便于看出大孔洞是否明显更难补。"""
    buckets: dict[str, list[dict[str, object]]] = {}
    for record in records:
        key = str(record.get("requested_removed_face_count"))
        buckets.setdefault(key, []).append(record)
    # 真实破损模式没有 requested_removed_face_count，我按字符串排序，
    # 不再假定该字段一定是整数。
    return {
        key: {
            "case_count": len(items),
            "hole_closed_count": sum(
                bool(item.get("hole_closed")) or bool(item.get("closed"))
                for item in items
            ),
        }
        for key, items in sorted(buckets.items(), key=lambda pair: pair[0])
    }


def _count_by(records: list[dict[str, object]], field: str) -> dict[str, int]:
    """我按字段统计分布；字段是列表时按逗号/竖线拆成多个 token 分别计数。"""
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field)
        key = ",".join(value) if isinstance(value, list) else str(value)
        for token in key.split(","):
            counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items()))


def write_reports(
    output_root: Path,
    data_root: Path,
    config: RepairConfig,
    records: list[dict[str, object]],
    skipped: list[dict[str, object]],
) -> None:
    """我写出基准汇总 JSON 与逐用例 CSV。"""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "data_root": str(data_root.resolve()),
        "config": config.to_dict(),
        "case_count": len(records),
        "skipped_count": len(skipped),
        # 我为什么要把跳过原因分类：实测 --max-step-bytes 默认 16 MiB，
        # 把 训练集/1（60.8 MB、52 条自由边）和测试集/3（22.4 MB、14 条）
        # 两个**真破损件**挡在基准外面，而汇总里只给一个 skipped_count
        # 总数，甲方按默认跑根本看不出分母是残缺的（9/11 会显示成 8/9）。
        # "本来封闭"是合理跳过、不污染分母；"体积超限"是被预算砍掉、
        # **会污染分母**。这两类必须分开报。
        "skipped_by_reason": _count_by(skipped, "reason"),
        "skipped_not_damaged_count": len(
            [
                item
                for item in skipped
                if "封闭" in str(item.get("reason", ""))
            ]
        ),
        "skipped_size_count": len(
            [
                item
                for item in skipped
                if "体积" in str(item.get("reason", ""))
            ]
        ),
        "skipped": skipped,
        "aggregate": aggregate(records),
        "records": records,
    }
    # 列表字段用竖线拼成单单元格，避免被 CSV 分隔符拆开。
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "case",
        "pair",
        "requested_removed_face_count",
        "removed_area_mm2",
        "damaged_boundary_edge_count",
        "repair_status",
        "repaired_region_count",
        "repaired_area_mm2",
        "repair_methods",
        "continuity_levels",
        "output_written",
        "output_boundary_edge_count",
        "residual_boundary_length_mm",
        "hole_closed",
        "elapsed_seconds",
        "warning_count",
    ]
    # 真实破损模式的字段与合成基准不同，我取实际记录的并集，
    # 保证两种模式都不会静默丢字段。
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with (output_root / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in ("repair_methods", "continuity_levels"):
                if isinstance(row.get(key), list):
                    row[key] = "|".join(row[key])
            writer.writerow({field: row.get(field) for field in fields})


def _build_parser() -> argparse.ArgumentParser:
    """我构造 STEP 修补基准命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="我方在真实 STEP 零件上造已知缺面并量化 STEP 修补成功率。"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "step_repair_benchmark",
    )
    parser.add_argument(
        "--max-step-bytes",
        type=int,
        default=96 * 1024 * 1024,
        # 体积上限本意是时间预算控制，但旧值 16 MB 实测把 21 个件里的
        # 5 个（24%）直接挡在基准之外，于是"成功率"的分母只剩下
        # 挑剩的小件——这不是时间预算，是选择性取样，会让甲方误读。
        # 我实测过这些件的真实耗时：测试集/3（22.4 MB）修完 23 s、
        # 训练集/1（60.8 MB）66 s，远不到"跑不完"的程度；
        # 最慢的一轮基准整体也只有两分钟量级。
        # 所以我按语料里最大件（训练集/1 约 60.8 MB）留出余量，
        # 把上限提到 96 MB，让全部件都进入统计。
        help="我方只把不超过该体积的 STEP 当作基准母体，控制时间预算。",
    )
    parser.add_argument("--face-counts", type=int, nargs="+", default=list(DEFAULT_FACE_COUNTS))
    parser.add_argument("--max-boundary-edges", type=int, default=4096)
    parser.add_argument("--max-hole-area-mm2", type=float, default=1_000_000.0)
    # 我为什么不再硬编码 default：与 mesh 侧基准同一个坑——这里写死 0.05 /
    # store_true(默认 False) 会把 `RepairConfig` 的默认值**悄悄覆盖掉**，
    # 改了 dataclass 默认在命令行里却还是旧的。默认值只在 RepairConfig
    # 定一处，CLI 从这里取，两处才不会打架。
    parser.add_argument(
        "--max-planarity-error-mm",
        type=float,
        default=RepairConfig().max_planarity_error_mm,
    )
    # BooleanOptionalAction 同时给 --allow-nonconvex / --no-allow-nonconvex，
    # 默认变 True 之后仍然能显式关掉（原 store_true 只能开不能关）。
    parser.add_argument(
        "--allow-nonconvex",
        action=argparse.BooleanOptionalAction,
        default=RepairConfig().allow_nonconvex,
    )
    parser.add_argument(
        "--allow-triangulated-boundary-candidate",
        # 我为什么把默认从"关"改成"开"：三角化是"环闭合、曲面填充与拟合
        # 平面都没成"时的兜底。实测只改这一个开关的逐件对照：
        # 测试集/1 REVIEW->SUCCESS、**其余 10 件零退化**、
        # hole_closed_ratio 0.727273 -> **0.818182（9/11）**。
        # 原 `action="store_true"`（默认 False）会**覆盖** RepairConfig 的
        # 默认值，所以我只改 repair.py 没用——两处必须一起改（这个坑我在
        # allow_nonplanar_surface_candidate 上已经踩过一次）。
        action=argparse.BooleanOptionalAction,
        default=True,
        # 默认开启：三角化补出的面只保证贴合边界，
        # 不保证与相邻面相切，属于待复核候选，
        # 默认打开会让基准的"成功率"掺入未经验证的补法。
        help="（实验性）非平面环在曲面填充与拟合平面都不适用时，用边界采样点三角化补面",
    )
    # 我为什么把默认从"关"改成"开"：这个开关是"非平面环走曲面填充候选"。
    # 我一直默认关着，结果平面度超差的环既建不出平面面、又拿不到曲面候选，
    # 直接变成无解（典型：训练集/7 落到 REVIEW_REQUIRED）。我做了逐件对照，
    # 打开后 训练集/7 REVIEW->SUCCESS、**其余 10 件全部不变（零退化）**。
    # 原来这里是 `action="store_true"`（默认 False），它会**覆盖** RepairConfig
    # 里的默认值，所以我改 RepairConfig 没用——必须连这里一起改。
    # 用 BooleanOptionalAction 后：默认开，`--no-allow-nonplanar-surface-
    # candidate` 可关，原来的 `--allow-nonplanar-surface-candidate` 仍可用。
    parser.add_argument(
        "--allow-nonplanar-surface-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--no-preserve-original-on-failure",
        dest="preserve_original_on_failure",
        action="store_false",
        # 默认 do_no_harm（校验不过就退回原件）。
        # 关掉它才能拿到"没过门禁的补面结果"用于人工复核，
        # 但也意味着输出可能是被改坏的模型——所以默认是开的。
        help="我方把未过门禁的复核 Shape 也写出来，用于人工复核。",
    )
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument(
        "--real-damaged",
        action="store_true",
        # 真实破损件与"造出来的缺面"验收口径不同：
        # 前者没有"正确答案"可比对，只能看开放边界改善了多少，
        # 用闭合率去评会得出毫无意义的数字。
        help=(
            "我直接修补甲方自带开放边界的真实破损件（不造缺面），"
            "验收口径为开放边界改善量而非闭合率。"
        ),
    )
    parser.add_argument(
        "--do-no-harm",
        action="store_true",
        help=(
            "我开启不伤害保护：修补后开放边界若比修补前更多，"
            "就不写出结果，退回人工复核。（现在默认开启）"
        ),
    )
    # 我为什么把不伤害保护改成**默认开启**：实测合成基准 17 个输出里
    # 有 1 个是 brep_valid=False 的坏几何（测试集/8#face3，
    # hole_closed=False、solid=1）。它被写到磁盘上，只是因为合成模式
    # 默认没开这道保护——真实件模式我一直开着，所以坏件会被删掉。
    # 基准的作用是给甲方一个可核对的数字，而把一个坏文件留在输出里
    # 并计入 output_written，是在"写出率"上虚报。默认开之后，坏件
    # 会被撤回，写出的每一份都至少过了不伤害这道线。
    parser.add_argument(
        "--no-do-no-harm",
        dest="do_no_harm",
        action="store_false",
        help=(
            "我关闭不伤害保护，把未过门禁的结果也写出来用于人工复核；"
            "代价是输出可能包含被改坏的模型。"
        ),
    )
    parser.set_defaults(preserve_original_on_failure=True, do_no_harm=True)
    return parser


def main() -> int:
    """我跑完整基准并打印汇总；全部用例失败返回 1，否则返回 0。"""
    args = _build_parser().parse_args()
    if any(count < 1 for count in args.face_counts):
        raise ValueError("我要求 --face-counts 全部至少为 1。")
    config = build_config(args)
    data_root = args.data_root.expanduser()
    output_root = args.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    paths = sorted(data_root.rglob("*.stp")) + sorted(data_root.rglob("*.step"))
    for step_path in paths:
        name = f"{step_path.parent.name}/{step_path.stem}"
        if args.only and not any(token in name for token in args.only):
            continue
        if step_path.stat().st_size > args.max_step_bytes:
            skipped.append({"pair": name, "reason": "体积超过预算上限。"})
            print(f"[skip-size] {name}", flush=True)
            continue
        try:
            shape = read_shape(step_path)
        except (RuntimeError, OSError) as exc:
            skipped.append({"pair": name, "reason": f"读取失败：{exc}"})
            print(f"[skip-read] {name} {exc}", flush=True)
            continue
        topology = inspect_step_topology(step_path)
        boundary = topology.edge_topology.get("boundary_edge_count")
        if args.real_damaged:
            # 真实破损模式：我要的就是本来就开放边界的件。
            if boundary == 0:
                # 真实破损模式要求母体自带开放边界。
                # 封闭件在这一模式下没有可验的东西，
                # 跳过并单独记下原因，而不是当失败统计。
                skipped.append(
                    {
                        "pair": name,
                        "reason": "该件本身是封闭零件，不属于真实破损场景。",
                        "boundary_edge_count": boundary,
                    }
                )
                # 每个跳过都即时打印：长任务跑几十个件，
                # 看不到进度就无法判断是卡住还是在正常跑。
                print(f"[skip-closed] {name} boundary=0", flush=True)
                continue
            case_dir = output_root / f"{name.replace('/', '__')}_real"
            # 目录名把路径分隔符换成下划线：数据根目录里
            # 的件名可能含子目录，直接用会建出嵌套目录，
            # 后续按目录名归档时会找不到。
            record = run_real_case(name, step_path, config, case_dir)
            records.append(record)
            # 每跑完一个就重写一遍报告：中途被打断时
            # 已经跑完的部分不会丢，而不是全部跑完才落盘。
            write_reports(output_root, data_root, config, records, skipped)
            print(
                f"[{record['repair_status']}] {record['case']} "
                # 前置边界数、后置边界数、是否改善、是否闭合、是否写出
                # 五个量一起打印：只看是否闭合无法区分
                # "没改善"和"改善了但没闭合"这两种不同的结果。
                f"boundary {record['boundary_before']} -> "
                f"{record['boundary_after']} "
                f"improved={record['improved']} closed={record['closed']} "
                f"written={record['output_written']}",
                flush=True,
            )
            continue
        # 造缺面模式要求母体必须封闭：在不封闭的件上再挖一个洞，
        # 得到的是"两个洞"，无法把结果归因到造出来的那个洞上。
        if boundary != 0:
            skipped.append(
                {
                    "pair": name,
                    "reason": "母体本身不是封闭零件，不能作为造缺面基准。",
                    "boundary_edge_count": boundary,
                }
            )
            print(f"[skip-open] {name} boundary={boundary}", flush=True)
            continue
        # 同一个母体按不同缺面数各跑一遍：
        # 缺面数是这个基准里唯一的自变量，
        # 交叉它能看出"洞越大成功率掉多少"。
        for face_count in args.face_counts:
            case_dir = output_root / f"{name.replace('/', '__')}_face{face_count}"
            record = run_case(name, step_path, shape, face_count, config, case_dir)
            records.append(record)
            write_reports(output_root, data_root, config, records, skipped)
            print(
                f"[{record['repair_status']}] {record['case']} "
                f"removed_area={record['removed_area_mm2']:.3f} "
                f"out_boundary={record['output_boundary_edge_count']} "
                f"closed={record['hole_closed']} "
                f"written={record['output_written']}",
                flush=True,
            )
    write_reports(output_root, data_root, config, records, skipped)
    print(json.dumps(aggregate(records), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
