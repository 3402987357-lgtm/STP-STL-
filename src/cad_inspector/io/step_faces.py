"""我提取 STEP B-Rep 的 Face 级几何特征，供分型规则和人工复核使用。

关键算法原理（本模块是整个 STEP 链路的数据源头，读取一次供全链路复用）：

一、拓扑编号稳定化。OCP 的 TopoDS_Shape 不能直接当字典键，所以我用几何哈希做分桶
    （``_register_edge`` / ``_register_vertex`` / ``_find_edge_id``），让同一次运行内
    每个 Edge/Vertex 拿到连续且可复现的编号；Face 编号按遍历顺序从 1 递增。这样下游
    报告里的编号能跨模块对上，而不是各说各话。

二、边折线采样。闭合环只取两个端点会把圆弧压成弦——两段圆弧组成的圆环甚至会退化成
    两个点，投影面积算不出来。所以我沿曲线采样 16 段（``_EDGE_CURVE_SEGMENTS``）：
    直线只需要两个端点，其余曲线按固定段数细分，整圆折线面积误差约 2.5%。采样按需
    触发并缓存，避免为整模型所有 Edge 付出取样开销。

三、几何签名。把 Face 的曲面类型、面积、中心、法向、包围盒、边曲线类型与长度、
    半径、轴线等字段规范化后取 sha256，生成不依赖编号的签名。签名重复只作为追踪线索，
    不冒充唯一 CAD ID。

四、内外语义。用 ``BRepClass3d_SolidClassifier`` 对 Face 采样点做实体内外判定，
    采样点数受 ``semantic_sample_limit`` 限制，避免大模型上耗时失控。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DependencyMissingError, GeometryProcessingError
from ..reproducibility import with_content_digest
from .file_validation import validate_input_file
from .step_probe import probe_step


# 我方把每条曲线 Edge 采样成折线；直线只需两个端点，其余曲线按固定段数细分。
# 16 段把整圆折线的面积误差压到约 2.5%，同时不显著拖慢大模型的遍历。
_EDGE_CURVE_SEGMENTS = 16


@dataclass
class StepFaceFeature:
    """我记录一个 Face 的可追踪几何摘要。"""

    face_id: int
    surface_type: str
    area_mm2: float | None
    center_xyz_mm: list[float] | None
    normal_xyz: list[float] | None
    bbox_min_xyz_mm: list[float] | None
    bbox_max_xyz_mm: list[float] | None
    edge_count: int
    edge_ids: list[int]
    orientation: str
    edge_curve_types: list[str] = field(default_factory=list)
    edge_lengths_mm: list[float | None] = field(default_factory=list)
    edge_radii_mm: list[float | None] = field(default_factory=list)
    circular_edge_ids: list[int] = field(default_factory=list)
    surface_radius_mm: float | None = None
    surface_axis_origin_xyz_mm: list[float] | None = None
    surface_axis_direction_xyz: list[float] | None = None
    geometry_signature: str | None = None
    edge_endpoint_vertex_ids: dict[int, list[int]] = field(default_factory=dict)
    semantic_evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """我把这个 Face 特征转成字典，键名与 CSV 表头一致。"""

        return asdict(self)


@dataclass
class StepFaceFeatureReport:
    """我记录一次 STEP Face 级特征提取的完整结果。"""

    status: str
    file_path: str
    unit: str | None
    backend: str
    shape_is_valid: bool | None
    face_count: int
    surface_type_counts: dict[str, int]
    faces: list[StepFaceFeature]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    vertex_coordinates_mm: dict[int, list[float]] = field(default_factory=dict)
    # 我方把边折线采样做成按需回调：只有真正落在区域边界上的 Edge 才会被取样，
    # 既不写进 JSON，也不为整模型的每条边都付出取样开销。
    edge_polyline_sampler: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """我把整份 Face 特征报告转成带 schema 版本与内容摘要的字典。"""

        return with_content_digest({
            "schema_version": "step-face-features-1.8",
            "status": self.status,
            "file_path": self.file_path,
            "unit": self.unit,
            "backend": self.backend,
            "shape_is_valid": self.shape_is_valid,
            "face_count": self.face_count,
            "surface_type_counts": dict(sorted(self.surface_type_counts.items())),
            "faces": [face.to_dict() for face in self.faces],
            "vertex_coordinates_mm": {
                str(vertex_id): list(coordinates)
                for vertex_id, coordinates in sorted(
                    self.vertex_coordinates_mm.items()
                )
            },
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "diagnostics": dict(self.diagnostics),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        })


def inspect_step_face_features(
    path: str | Path,
    *,
    validate_shape: bool = True,
    classify_semantics: bool = False,
    semantic_sample_limit: int = 16,
) -> StepFaceFeatureReport:
    """我读取真实 STEP，并输出每个 Face 的面积、类型和方向摘要。"""

    started = time.perf_counter()
    # 采样上限必须是正整数，布尔值不算（避免 True 被当成 1 混进来）。
    if (
        isinstance(semantic_sample_limit, bool)
        or not isinstance(semantic_sample_limit, Integral)
        or semantic_sample_limit < 1
    ):
        raise ValueError("我要求 semantic_sample_limit 至少为 1。")
    file_path = validate_input_file(path, {".stp", ".step"}).resolve()
    probe = probe_step(file_path)
    try:
        # OCP 全部按需 import：仓库要求"没装 OCP 也能 import 本模块"，
        # 只有真的要读 STEP 面特征时才需要这些依赖。
        # 曲面与曲线的适配器：把拓扑对象变成可求值参数化几何。
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        # 取几何底层的工具：从面上拿出真正的曲面对象。
        from OCP.BRep import BRep_Tool
        # 实体内外分类器：判断一个点落在实体内部、外部还是面上。
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        # 包围盒与有效性：前者取面的范围，后者判断形状是否成立。
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepCheck import BRepCheck_Analyzer
        # 面积与体积等全局属性计算。
        from OCP.BRepGProp import BRepGProp
        from OCP.Bnd import Bnd_Box
        from OCP.GProp import GProp_GProps
        # 读取状态枚举与 STEP 读取器。
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.STEPControl import STEPControl_Reader
        # 拓扑类型：EDGE/FACE/VERTEX 用于遍历，
        # IN/ON/OUT 是分类结果，REVERSED 表示朝向被反转。
        from OCP.TopAbs import (
            TopAbs_EDGE,
            TopAbs_FACE,
            TopAbs_IN,
            TopAbs_ON,
            TopAbs_OUT,
            TopAbs_REVERSED,
            TopAbs_VERTEX,
        )
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        # 三种曲线/曲面类型：圆、圆柱、直线。
        # 这三类就是本模块能识别的全部几何形态。
        from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Cylinder, GeomAbs_Line
        from OCP.gp import gp_Pnt, gp_Vec
    # 缺 OCP 转成本项目的依赖错误：
    # 调用方能据此区分"环境缺依赖"与"这个文件有问题"。
    except ImportError as exc:
        raise DependencyMissingError(
            "我无法提取 STEP Face 特征，因为当前环境没有 OCP。"
        ) from exc

    # 读取与转移失败直接报错；空 Shape 同样报错，不返回空报告让下游误判"没有缺失"。
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(file_path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise GeometryProcessingError(
            f"我读取 STEP 失败，后端状态为 {status}: {file_path}"
        )
    if not reader.TransferRoots():
        raise GeometryProcessingError(f"我没有转移出 STEP 根对象: {file_path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise GeometryProcessingError(f"我读取到的 STEP Shape 为空: {file_path}")

    shape_is_valid = (
        bool(BRepCheck_Analyzer(shape).IsValid()) if validate_shape else None
    )
    warnings: list[str] = []
    if shape_is_valid is False:
        warnings.append("我发现 STEP BRep 有效性检查未通过。")
    edge_hash_buckets: dict[int, list[int]] = {}
    edge_by_id: dict[int, Any] = {}
    vertex_hash_buckets: dict[int, list[int]] = {}
    vertex_by_id: dict[int, Any] = {}
    edge_endpoint_vertex_ids: dict[int, list[int]] = {}
    # 第一遍先遍历全部 Edge，登记编号与端点，供 Face 遍历时反查。
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = TopoDS.Edge(edge_explorer.Current())
        edge_id = _register_edge(edge, edge_hash_buckets, edge_by_id)
        edge_endpoint_vertex_ids[edge_id] = _edge_vertex_ids(
            edge,
            explorer_type=TopExp_Explorer,
            top_abs_vertex=TopAbs_VERTEX,
            topods_type=TopoDS,
            vertex_hash_buckets=vertex_hash_buckets,
            vertex_by_id=vertex_by_id,
        )
        edge_explorer.Next()

    # 我方沿 Edge 曲线取有限采样点：闭合环只用两个端点会把圆弧压成弦，
    # 两段圆弧组成的圆环甚至会退化成两个点，无法计算投影面积。
    # 采样按需触发并缓存，避免为整模型所有 Edge 付出取样开销。
    polyline_cache: dict[int, list[list[float]] | None] = {}

    def _sample_edge_polyline(edge_id: int) -> list[list[float]] | None:
        """我按需采样单条 Edge 折线，并缓存结果避免重复取样。"""

        key = int(edge_id)
        if key in polyline_cache:
            return polyline_cache[key]
        points = _edge_polyline_points_mm(
            edge_by_id.get(key),
            BRepAdaptor_Curve,
            GeomAbs_Line,
            curve_segments=_EDGE_CURVE_SEGMENTS,
        )
        result = points if len(points) >= 2 else None
        polyline_cache[key] = result
        return result
    # 第二遍遍历 Face，逐面提取几何摘要；Face 编号按遍历顺序从 1 起算。
    faces: list[StepFaceFeature] = []
    type_counts: dict[str, int] = {}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    while explorer.More():
        # 面号按遍历顺序从 1 递增：这个编号只在本次读取内稳定，
        # 重跑一次就可能变。跨批次追踪必须靠几何签名而不是它。
        face_id += 1
        face = TopoDS.Face(explorer.Current())
        # OCP 类型全部显式传入：调用方在函数内 import 并注入，
        # 这样没装 OCP 的环境仍能 import 本模块。
        feature = _inspect_face(
            face,
            face_id,
            BRepAdaptor_Surface,
            BRepBndLib,
            Bnd_Box,
            BRepGProp,
            GProp_GProps,
            TopAbs_EDGE,
            TopExp_Explorer,
            edge_hash_buckets,
            edge_by_id,
            edge_endpoint_vertex_ids,
            BRepAdaptor_Curve,
            GeomAbs_Circle,
            TopAbs_REVERSED,
            gp_Pnt,
            gp_Vec,
            GeomAbs_Cylinder,
        )
        faces.append(feature)
        # 曲面类型计数：它是"这份模型由哪些类型的面构成"的概览，
        # 也是判断模型特征（例如回转体 vs 箱体）的第一手依据。
        type_counts[feature.surface_type] = (
            type_counts.get(feature.surface_type, 0) + 1
        )
        explorer.Next()

    vertex_coordinates_mm: dict[int, list[float]] = {}
    for vertex_id, vertex in vertex_by_id.items():
        try:
            point = BRep_Tool.Pnt_s(vertex)
            coordinates = [float(point.X()), float(point.Y()), float(point.Z())]
        # 取不到坐标的顶点跳过：个别退化顶点会在 OCP 层抛异常，
        # 中断整个遍历会丢掉其余所有顶点的坐标。
        except Exception:
            continue
        if all(isfinite(value) for value in coordinates):
            vertex_coordinates_mm[int(vertex_id)] = coordinates

    # 可选执行实体内外语义分类，采样点数受上限约束。
    semantic_diagnostics: dict[str, Any] = {}
    if classify_semantics:
        semantic_diagnostics = _classify_face_semantics(
            shape,
            faces,
            BRepClass3d_SolidClassifier,
            BRepBndLib,
            Bnd_Box,
            gp_Pnt,
            TopAbs_IN,
            TopAbs_ON,
            TopAbs_OUT,
            semantic_sample_limit,
        )
        warnings.extend(semantic_diagnostics.pop("warnings", []))

    report_status = "COMPLETE" if shape_is_valid is not False else "REVIEW_REQUIRED"
    if not faces:
        warnings.append("我没有从 STEP 中读取到 Face。")
    # 统计几何签名重复次数；重复只作追踪线索，不冒充唯一 CAD ID。
    signature_counts: dict[str, int] = {}
    for face in faces:
        # 空签名跳过：量不出签名的面不参与重复统计，
        # 否则多个空签名会被当成"大量重复"而误报。
        if face.geometry_signature:
            signature_counts[face.geometry_signature] = (
                signature_counts.get(face.geometry_signature, 0) + 1
            )
    # 重复数按"多出来的份数"算：一个签名出现 3 次贡献 2，
    # 而不是 3。这样它与"有多少个面需要人工区分"直接对应。
    duplicate_signature_count = sum(
        count - 1 for count in signature_counts.values() if count > 1
    )
    # 有重复就明确警告"签名不能当唯一 ID"：
    # 这是为了防止下游把签名当作稳定主键——
    # 同一个模型重导出一次签名就可能变，重复只是它不可靠的一个征兆。
    if duplicate_signature_count:
        warnings.append(
            f"我发现 {duplicate_signature_count} 个 Face 的几何签名重复，"
            "几何签名不能单独作为唯一 CAD ID。"
        )
    # 签名总数与重复数都进诊断：
    # 两者之比说明"签名这个判据在这份模型上的区分度"。
    semantic_diagnostics["geometry_signature_count"] = len(signature_counts)
    # 重复签名数单独记：同一个几何面在模型里出现多次
    # （例如阵列件的多个实例）是正常现象，
    # 但这个数异常高说明区域划分可能重复算了。
    semantic_diagnostics["duplicate_geometry_signature_count"] = (
        duplicate_signature_count
    )
    return StepFaceFeatureReport(
        status=report_status,
        file_path=str(file_path),
        # 单位取探针给出的提示而不是断言为毫米：
        # STEP 文件可能不标单位，猜错会让所有毫米量级的阈值失效。
        unit=probe.unit_hint,
        # 后端固定为 OCP：这份报告的每一个量都来自 OCP，
        # 换后端则所有精度口径都要重新标定。
        backend="OCP",
        shape_is_valid=shape_is_valid,
        face_count=len(faces),
        surface_type_counts=type_counts,
        faces=faces,
        # 警告去重：同一条警告可能由多个面各触发一次。
        warnings=list(dict.fromkeys(warnings)),
        # errors 留空列表：读取过程中的问题走的是 warnings 通道，
        # 真正致命的失败是在这里之前就抛异常了。
        errors=[],
        diagnostics=semantic_diagnostics,
        elapsed_seconds=time.perf_counter() - started,
        vertex_coordinates_mm=vertex_coordinates_mm,
        # 折线采样器作为可调用对象传出：下游要用它取边的折线点，
        # 但那些点只在需要时才算——提前全算会在大件上吃掉大量内存。
        edge_polyline_sampler=_sample_edge_polyline,
    )


def _inspect_face(
    face: Any,
    face_id: int,
    adaptor_type: Any,
    bbox_lib: Any,
    bbox_type: Any,
    prop_lib: Any,
    props_type: Any,
    top_abs_edge: Any,
    explorer_type: Any,
    edge_hash_buckets: dict[int, list[int]],
    edge_by_id: dict[int, Any],
    edge_endpoint_vertex_ids: dict[int, list[int]],
    curve_adaptor_type: Any,
    circle_type: Any,
    reversed_type: Any,
    point_type: Any,
    vector_type: Any,
    cylinder_type: Any,
) -> StepFaceFeature:
    """我提取单个 Face 的曲面类型、面积、重心、包围盒、法向与边特征。

    设计原则：这是全项目最靠近 OCP 的一层，真实 STEP 上什么怪事都会遇到——
    退化面、取不到参数曲线、面积算不出来、圆柱面轴退化。我的统一策略是
    **单项失败只降级该项，不中断整个 Face 提取**：面积失败就留 None 并记
    警告，包围盒失败同样留 None，由下游按"缺失"隔离这个面。

    为什么不能让异常往上抛：一个 4974 面的真实模具里只要有 1 个面触发异常，
    整件就全废了，甲方拿到的是"读取失败"而不是"4973 个面可用、1 个待复核"。
    局部降级 + 显式警告才是可复核的工程行为。

    法向要单独处理方向：B-Rep 面的朝向由 ORIENTED 标志决定，
    face 的 Reversed() 为真时解析曲面法向需要取反，否则内外判据会整体反向。
    """
    warnings: list[str] = []
    adaptor = adaptor_type(face, True)
    surface_type = getattr(adaptor.GetType(), "name", str(adaptor.GetType()))
    surface_radius_mm, axis_origin_xyz_mm, axis_direction_xyz = _cylinder_geometry(
        adaptor,
        cylinder_type,
        warnings,
        face_id,
    )

    # 面积与重心走 BRepGProp；失败只记警告并留 None，不中断整个 Face 提取。
    area: float | None = None
    center: list[float] | None = None
    try:
        props = props_type()
        prop_lib.SurfaceProperties_s(face, props)
        area = float(props.Mass())
        point = props.CentreOfMass()
        center = [float(point.X()), float(point.Y()), float(point.Z())]
    except Exception as exc:
        warnings.append(f"我无法计算 Face {face_id} 的面积或重心：{exc}")

    # 包围盒同理：失败留 None，由下游的方向分组自行隔离该面。
    bbox_min: list[float] | None = None
    bbox_max: list[float] | None = None
    try:
        bbox = bbox_type()
        bbox_lib.Add_s(face, bbox)
        bbox_min = [
            float(bbox.GetXMin()),
            float(bbox.GetYMin()),
            float(bbox.GetZMin()),
        ]
        bbox_max = [
            float(bbox.GetXMax()),
            float(bbox.GetYMax()),
            float(bbox.GetZMax()),
        ]
    except Exception as exc:
        warnings.append(f"我无法计算 Face {face_id} 的包围盒：{exc}")

    normal = _face_normal(
        adaptor,
        face,
        reversed_type,
        point_type,
        vector_type,
        warnings,
        face_id,
    )
    # 反查每条边在第一遍登记时拿到的稳定编号，去重后保持遍历顺序。
    edge_explorer = explorer_type(face, top_abs_edge)
    edge_ids: list[int] = []
    while edge_explorer.More():
        edge_id = _find_edge_id(
            edge_explorer.Current(),
            edge_hash_buckets,
            edge_by_id,
        )
        if edge_id > 0 and edge_id not in edge_ids:
            edge_ids.append(edge_id)
        edge_explorer.Next()
    edge_curve_types: list[str] = []
    edge_lengths_mm: list[float | None] = []
    edge_radii_mm: list[float | None] = []
    circular_edge_ids: list[int] = []
    # 逐边取曲线类型、长度、半径；有半径即为圆弧边，供孔壁与圆角判据使用。
    for edge_id in edge_ids:
        # 边的三个量按"类型/长度/半径"分别入三个列表，
        # 三者下标一一对应：后续按边号定位时靠的是这个对应关系，
        # 任何一项中途跳过都会让三个列表错位。
        curve_type, length, radius = _inspect_edge(
            edge_by_id.get(edge_id),
            edge_id,
            curve_adaptor_type,
            circle_type,
            prop_lib,
            props_type,
            warnings,
        )
        edge_curve_types.append(curve_type)
        edge_lengths_mm.append(length)
        edge_radii_mm.append(radius)
        # 只有半径量得出来的边才算圆边：直线没有半径，
        # 用"有没有半径"当判据比"曲线类型是否等于圆"更可靠，
        # 因为类型识别失败时半径也一定是 None。
        if radius is not None:
            circular_edge_ids.append(edge_id)
    orientation = str(face.Orientation())
    feature = StepFaceFeature(
        face_id=face_id,
        surface_type=surface_type,
        area_mm2=area,
        center_xyz_mm=center,
        normal_xyz=normal,
        bbox_min_xyz_mm=bbox_min,
        bbox_max_xyz_mm=bbox_max,
        # 边数与边号列表都存：列表是定位用的，
        # 计数是统计与门禁用的，缺一个就得每次重算。
        edge_count=len(edge_ids),
        edge_ids=edge_ids,
        # 朝向单独存：同一个几何面在实体里可能是正向也可能是反向，
        # 判断"这是内表面还是外表面"必须靠它。
        orientation=orientation,
        edge_curve_types=edge_curve_types,
        edge_lengths_mm=edge_lengths_mm,
        edge_radii_mm=edge_radii_mm,
        circular_edge_ids=circular_edge_ids,
        surface_radius_mm=surface_radius_mm,
        surface_axis_origin_xyz_mm=axis_origin_xyz_mm,
        surface_axis_direction_xyz=axis_direction_xyz,
        # 端点顶点映射只收有值的边：查不到端点的边留着
        # 只会让下游拿到一堆空列表，不如直接不收录。
        edge_endpoint_vertex_ids={
            edge_id: list(edge_endpoint_vertex_ids.get(edge_id, []))
            for edge_id in edge_ids
            if edge_endpoint_vertex_ids.get(edge_id)
        },
        warnings=warnings,
    )
    feature.geometry_signature = _geometry_signature(feature)
    return feature


def _geometry_signature(feature: StepFaceFeature) -> str | None:
    """我从有限几何摘要生成候选签名，不把它宣称为永久拓扑 ID。

    原理说明：甲方要的是"跨两次运行还能认出同一个面"的能力，但 STEP 的
    Face/Edge 编号会随导出顺序、CAD 版本和遍历顺序变化，不能当主键。
    所以我把**几何本身**编码成签名：曲面类型、面积、质心、包围盒、
    边数与每条边的（曲线类型、长度、半径）描述子，全部先按 6 位小数
    量化，再排序后做 SHA256 并截 24 位十六进制。

    两个设计细节：
      - 先量化再哈希。直接用原始双精度做哈希，两次导出间 1e-12 的
        浮点差异就会让签名完全不同，签名就失去了跨运行比对的意义；
      - 边描述子先排序。同一组边在不同遍历顺序下必须得到同一个签名，
        否则"面没变、签名变了"会制造大量假漂移。

    诚实边界：这是**几何摘要签名**，不是永久拓扑 ID。重复几何（两个
    完全一样的螺丝孔）会碰撞，单位换算、模型重建和真实几何变形也会
    让签名失配，所以它只用于同文件内的可解释证据与跨运行线索，
    不能用来宣称"这个面就是上次那个面"。
    """

    if (
        feature.area_mm2 is None
        or feature.center_xyz_mm is None
        or feature.bbox_min_xyz_mm is None
        or feature.bbox_max_xyz_mm is None
    ):
        # 四项基础量测缺任何一项就不生成签名：
        # 签名的作用是"识别同一个面"，缺项的签名
        # 无法唯一识别任何东西，生成它只会制造假重复。
        return None
    try:
        area = float(feature.area_mm2)
        center = [float(value) for value in feature.center_xyz_mm]
        bbox_min = [float(value) for value in feature.bbox_min_xyz_mm]
        bbox_max = [float(value) for value in feature.bbox_max_xyz_mm]
        # 非有限值一律放弃：签名里只要有一个 nan，
        # 整个字符串每次生成都可能不同，等于签名失效。
        if not all(
            np.isfinite(value)
            for value in [area, *center, *bbox_min, *bbox_max]
        ):
            return None
        # 边描述符排序后入签名：边的遍历顺序会随读取顺序变，
        # 不排序的话同一个面两次导出会得到两个不同的签名。
        edge_descriptors = sorted(
            (
                str(curve_type),
                _rounded_float(length),
                _rounded_float(radius),
            )
            for curve_type, length, radius in zip(
                feature.edge_curve_types,
                feature.edge_lengths_mm,
                feature.edge_radii_mm,
            )
        )
        payload = {
            "surface_type": feature.surface_type,
            # 数值统一先 round 再入签名：原始浮点有十几位小数，
            # 微小的重算差异就会让签名不同。
            # round 到 6 位在毫米量级上远小于任何工程容差。
            "area_mm2": round(area, 6),
            "center_xyz_mm": [_rounded_float(value) for value in center],
            "bbox_min_xyz_mm": [_rounded_float(value) for value in bbox_min],
            "bbox_max_xyz_mm": [_rounded_float(value) for value in bbox_max],
            "edge_count": int(feature.edge_count),
            "edge_descriptors": edge_descriptors,
            "surface_radius_mm": _rounded_float(feature.surface_radius_mm),
            "surface_axis_origin_xyz_mm": _rounded_vector(
                feature.surface_axis_origin_xyz_mm
            ),
            "surface_axis_direction_xyz": _canonical_vector(
                feature.surface_axis_direction_xyz
            ),
        }
    except (TypeError, ValueError, OverflowError):
        return None
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        # allow_nan=False：非有限值会让签名退化为非法 JSON 字面量，
        # 跨运行比对也就无从谈起，所以显式拒绝而不是静默写入。
        allow_nan=False,
    ).encode("utf-8")
    return f"facegeom-{hashlib.sha256(encoded).hexdigest()[:24]}"


def _rounded_float(value: float | None) -> float | None:
    """我把浮点值四舍五入到 6 位小数；非有限值统一转 None 以稳定签名。

    我为什么取 6 位小数：几何签名要用来判断"两个面是不是同一个面"。
    OCP 每次读取同一文件时，浮点末位会有 ULP 级差异，不取整就永远
    匹不上。而取到 6 位（1 微米以下的 1/1000）已经远小于本项目
    0.01 mm 级的容差，再粗就会把本来不同的面并成同一个。

    我为什么把非有限值统一转 None 而不是保留 NaN：NaN 不等于自己，
    带 NaN 的签名每次比较都是 False，等于永久失配；而转成 None 后
    这个分量按"缺失"处理，其他分量仍可参与比较。
    """
    if value is None:
        return None
    numeric = float(value)
    return round(numeric, 6) if np.isfinite(numeric) else None


def _rounded_vector(value: list[float] | None) -> list[float] | None:
    """我把三维向量逐分量四舍五入到 6 位小数。"""
    if value is None:
        return None
    return [_rounded_float(item) for item in value]


def _canonical_vector(value: list[float] | None) -> list[float] | None:
    """我把方向向量归一化成"首个显著分量为正"的规范形式。

    我为什么需要它：同一个平面被两个相邻面引用时，OCP 给两条法向
    可能是相反的（各自朝外）。它们描述的是同一个平面，但数值上差
    一个负号，直接比会被判成两个不同的面。我把方向统一翻成
    "首个超过 1e-12 的分量为正"，两条相反法向就收敛到同一表示。

    我为什么用 1e-12 当"显著"门槛而不是直接看是否为 0：取整到 6 位
    之后，本来是 0 的分量可能留下 1e-7 级别的残值，直接判 0 会
    把它当成有效分量去定符号，翻出的方向就不对了。
    """
    rounded = _rounded_vector(value)
    if rounded is None:
        return None
    for item in rounded:
        if item is not None and abs(item) > 1e-12:
            if item < 0:
                return [-value if value is not None else None for value in rounded]
            break
    return rounded


def _cylinder_geometry(
    adaptor: Any,
    cylinder_type: Any,
    warnings: list[str],
    face_id: int,
) -> tuple[float | None, list[float] | None, list[float] | None]:
    """我提取圆柱面的轴线和半径，失败时保留空证据并继续处理。

    轴线和半径是识别孔/轴类特征的直接证据：同轴阵列检测、孔壁判定
    都依赖它。我只对确认为圆柱类型的面提取，非圆柱直接返回空——
    强行把其他曲面拟合成圆柱会得到一堆没有意义的"半径"。

    三项有效性检查缺一不可：半径必须有限且为正（负半径无物理意义），
    轴向模长必须非零（零向量无法归一化，用它做投影会全零）。
    任一不满足就记 warning 并返回空，由下游按"圆柱证据缺失"处理。

    失败统一吞异常并记 warning：单个面提取失败不该中断整份
    面特征报告，那个面只是少一条证据，
    其余几十个面的信息仍然有效。
    """

    try:
        if adaptor.GetType() != cylinder_type:
            return None, None, None
        cylinder = adaptor.Cylinder()
        radius = float(cylinder.Radius())
        axis = cylinder.Axis()
        origin = axis.Location()
        direction = axis.Direction()
        vector = np.asarray(
            [float(direction.X()), float(direction.Y()), float(direction.Z())],
            dtype=np.float64,
        )
        length = float(np.linalg.norm(vector))
        if not isfinite(radius) or radius <= 0.0 or length <= 1e-12:
            warnings.append(f"我发现圆柱 Face {face_id} 的轴线或半径无效。")
            return None, None, None
        vector /= length
        return (
            radius,
            [float(origin.X()), float(origin.Y()), float(origin.Z())],
            vector.tolist(),
        )
    except Exception as exc:
        warnings.append(f"我无法提取圆柱 Face {face_id} 的轴线和半径：{exc}")
        return None, None, None


def _edge_hash(edge: Any) -> int:
    """我取得 OCP Edge 的快速哈希，失败时退回单独桶。

    我为什么自己包一层而不是直接用 ``hash(edge)``：OCP 的 Python
    绑定在某些 Edge 上不实现 ``__hash__``，直接调用会抛 TypeError，
    让整个拓扑遍历中断。这里吞掉异常退回桶 0，退化成"所有取不到
    哈希的 Edge 互相比较一遍"——慢，但结果仍然正确。
    """

    try:
        return int(hash(edge))
    except Exception:
        return 0


def _find_edge_id(
    edge: Any,
    edge_hash_buckets: dict[int, list[int]],
    edge_by_id: dict[int, Any],
) -> int:
    """我按哈希桶和 IsSame 找回同一个拓扑 Edge。"""

    key = _edge_hash(edge)
    for edge_id in edge_hash_buckets.get(key, []):
        try:
            if edge.IsSame(edge_by_id[edge_id]):
                return edge_id
        except Exception:
            continue
    return 0


def _register_edge(
    edge: Any,
    edge_hash_buckets: dict[int, list[int]],
    edge_by_id: dict[int, Any],
) -> int:
    """我给 Edge 分配当前报告内稳定且连续的编号。"""

    existing = _find_edge_id(edge, edge_hash_buckets, edge_by_id)
    if existing > 0:
        return existing
    edge_id = len(edge_by_id) + 1
    edge_by_id[edge_id] = edge
    edge_hash_buckets.setdefault(_edge_hash(edge), []).append(edge_id)
    return edge_id


def _edge_vertex_ids(
    edge: Any,
    *,
    explorer_type: Any,
    top_abs_vertex: Any,
    topods_type: Any,
    vertex_hash_buckets: dict[int, list[int]],
    vertex_by_id: dict[int, Any],
) -> list[int]:
    """我为当前报告建立 Edge 到 Vertex 的紧凑关联。"""

    vertex_explorer = explorer_type(edge, top_abs_vertex)
    vertex_ids: list[int] = []
    while vertex_explorer.More():
        vertex = topods_type.Vertex(vertex_explorer.Current())
        vertex_id = _register_vertex(
            vertex,
            vertex_hash_buckets,
            vertex_by_id,
        )
        if vertex_id > 0 and vertex_id not in vertex_ids:
            vertex_ids.append(vertex_id)
        vertex_explorer.Next()
    return sorted(vertex_ids)


def _find_vertex_id(
    vertex: Any,
    vertex_hash_buckets: dict[int, list[int]],
    vertex_by_id: dict[int, Any],
) -> int:
    """我按哈希桶和 IsSame 找回当前报告内的 Vertex 编号。"""

    key = _edge_hash(vertex)
    for vertex_id in vertex_hash_buckets.get(key, []):
        try:
            if vertex.IsSame(vertex_by_id[vertex_id]):
                return vertex_id
        except Exception:
            continue
    return 0


def _register_vertex(
    vertex: Any,
    vertex_hash_buckets: dict[int, list[int]],
    vertex_by_id: dict[int, Any],
) -> int:
    """我给 Vertex 分配当前报告内连续编号，不把它解释成永久 CAD ID。

    原理说明：OCP 的 TopoDS 对象没有稳定的整数主键，而下游的连通分量、
    环排序和边界追踪都要按 id 索引。所以我在**当前这一次报告内**自己
    发号，并保证"同一个拓扑顶点拿到同一个号"。

    判重用两步：先按快速哈希分桶缩小比较范围，再用 IsSame 做拓扑同一性
    判定。只用哈希会把不同顶点误并（哈希碰撞），只逐对 IsSame 在几万
    顶点上是 O(n^2)，两步结合才既正确又可负担。

    编号语义：这是运行内序号，重新导出 STEP 或换遍历顺序都会变，
    所以我把它叫 id 而不是"永久 CAD ID"，跨文件比对要用几何签名。
    """

    existing = _find_vertex_id(vertex, vertex_hash_buckets, vertex_by_id)
    if existing > 0:
        return existing
    vertex_id = len(vertex_by_id) + 1
    vertex_by_id[vertex_id] = vertex
    vertex_hash_buckets.setdefault(_edge_hash(vertex), []).append(vertex_id)
    return vertex_id


def _edge_polyline_points_mm(
    edge: Any,
    curve_adaptor_type: Any,
    line_type: Any,
    *,
    curve_segments: int,
) -> list[list[float]]:
    """我沿 Edge 曲线取有限采样点，供区域闭合环投影使用。

    我方只用端点会把圆弧压成弦，两段圆弧组成的圆环更会退化成两个点，
    投影面积算不出来。这里按曲线参数均匀取样，直线只保留两个端点。
    """

    if edge is None:
        # 空边返回空折线而不是抛异常：
        # 一条边采样失败不该中断整个面的读取。
        return []
    try:
        adaptor = curve_adaptor_type(edge)
        first = float(adaptor.FirstParameter())
        last = float(adaptor.LastParameter())
    except Exception:
        return []
    # 参数区间必须有限且正向：区间反了说明这条边的朝向被反向过，
    # 按它采样会得到倒序的点，下游按首尾配对时会全部错位。
    if not (isfinite(first) and isfinite(last)) or last < first:
        return []
    try:
        is_line = adaptor.GetType() == line_type
    except Exception:
        # 类型判不出来就按曲线处理：曲线会多采几个点，
        # 对直线来说只是多几个共线的点，不会丢精度。
        is_line = False
    # 直线只采两个端点：直线上的中间点没有信息量，
    # 采多了只是白白增大下游的折线数据量。
    segments = 1 if is_line else max(int(curve_segments), 1)
    points: list[list[float]] = []
    # 端点都要采到，所以是 segments + 1 个点。
    for step in range(segments + 1):
        # 按参数均匀采样：曲线不一定是直线，
        # 在参数域上均匀是最通用的取点方式。
        parameter = first + (last - first) * (step / segments)
        try:
            point = adaptor.Value(parameter)
            coordinates = [
                float(point.X()),
                float(point.Y()),
                float(point.Z()),
            ]
        # 任一个点取不出来就整条折线作废：
        # 给出半截折线会让下游以为这是完整的边。
        except Exception:
            return []
        # 坐标必须全部有限：带 nan 的点会让后面的
        # 包围盒、长度、面积计算全部变成 nan。
        if not all(isfinite(value) for value in coordinates):
            return []
        points.append(coordinates)
    return points


def _inspect_edge(
    edge: Any,
    edge_id: int,
    curve_adaptor_type: Any,
    circle_type: Any,
    prop_lib: Any,
    props_type: Any,
    warnings: list[str],
) -> tuple[str, float | None, float | None]:
    """我提取一个 Edge 的曲线类别、长度和圆弧半径。"""

    if edge is None:
        warnings.append(f"我找不到 Edge {edge_id} 的 B-Rep 对象。")
        # 三个量各自返回"未知/空"而不是抛异常：
        # 一条边量不出来不该让整个面的检测失败，
        # 其余边的结果仍然有用。
        return "UNKNOWN", None, None
    curve_type = "UNKNOWN"
    length: float | None = None
    radius: float | None = None
    adaptor = None
    # 曲线类型、长度、半径分别 try：三者互不相干，
    # 合并成一个 try 会让前一项失败就丢掉后两项。
    try:
        adaptor = curve_adaptor_type(edge)
        curve_value = adaptor.GetType()
        curve_type = getattr(curve_value, "name", str(curve_value))
    except Exception as exc:
        warnings.append(f"我无法识别 Edge {edge_id} 的曲线类型：{exc}")
    try:
        props = props_type()
        prop_lib.LinearProperties_s(edge, props)
        candidate = float(props.Mass())
        # 长度必须是有限非负：负数长度说明量测出错，
        # 直接采用会让后面的周长统计出现负贡献。
        if isfinite(candidate) and candidate >= 0.0:
            length = candidate
        else:
            warnings.append(f"我发现 Edge {edge_id} 的长度不是有限非负值。")
    except Exception as exc:
        warnings.append(f"我无法计算 Edge {edge_id} 的长度：{exc}")
    # 半径只对圆弧去取：直线没有半径，
    # 不问类型直接取会在直线上抛异常并污染警告列表。
    if adaptor is not None:
        try:
            if adaptor.GetType() == circle_type:
                candidate = float(adaptor.Circle().Radius())
                # 半径必须为正：半径 0 的"圆"是退化情形，
                # 它不该被当成真实圆弧参与孔径统计。
                if isfinite(candidate) and candidate > 0.0:
                    radius = candidate
                else:
                    warnings.append(f"我发现 Edge {edge_id} 的圆弧半径无效。")
        except Exception as exc:
            warnings.append(f"我无法计算 Edge {edge_id} 的圆弧半径：{exc}")
    return curve_type, length, radius


def _classify_face_semantics(
    shape: Any,
    faces: list[StepFaceFeature],
    classifier_type: Any,
    bbox_lib: Any,
    bbox_type: Any,
    point_type: Any,
    in_state: Any,
    on_state: Any,
    out_state: Any,
    semantic_sample_limit: int,
) -> dict[str, Any]:
    """我用法向两侧采样记录实体内外证据，不把它包装成最终语义。

    原理说明：判断一个面是"朝外的外表面"还是"朝内的型腔面"，本质是
    问"沿这个面的法向往外走，走到的是实体还是空气"。我用 OCP 的
    BRepClass3d_SolidClassifier 做点相对于实体的内外判定，沿法向
    取**五个递增的偏移距离**逐层探测：
      - 贴面处（span*1e-6）最容易受数值噪声干扰，单独一层；
      - 逐级放大到 span*2.0 的远场，确认不是"刚好穿过一个薄壁"的假结论。
    多层一致才算可信；层间结论矛盾时我只给低置信度和复核标记，
    不挑一个对自己有利的层。

    采样距离按整体包围盒跨度折算而不是写死毫米数：同一个 1e-6 的绝对
    偏移在大模具上等于没动、在微型件上等于飞出模型，按比例才能通用。

    面数超上限时我按索引**等距抽样**而不是取前 N 个：STEP 的面通常按
    导出顺序聚集（同一批孔连在一起），截断会把采样全压在同一区域，
    得到"整件都是外表面"的假结论。未采样的面一律标
    resource_guard 的 UNRESOLVED，绝不冒充"已判定"。
    """

    # 先拿到整体包围盒，采样距离全部按包围盒跨度折算，保证对不同尺寸零件都合理。
    bbox = bbox_type()
    try:
        bbox_lib.Add_s(shape, bbox)
        bounds = (
            float(bbox.GetXMin()),
            float(bbox.GetYMin()),
            float(bbox.GetZMin()),
            float(bbox.GetXMax()),
            float(bbox.GetYMax()),
            float(bbox.GetZMax()),
        )
    except Exception as exc:
        warning = f"我无法建立实体采样包围盒，语义证据保持未解析：{exc}"
        for face in faces:
            face.semantic_evidence = {
                "status": "UNRESOLVED",
                "method": "normal_offset_solid_classifier",
                "confidence": 0.0,
                "warnings": [warning],
            }
        return {"warnings": [warning]}

    # 跨度取三轴最大值并留 1e-9 下限，防止退化零件除零。
    span = max(
        bounds[3] - bounds[0],
        bounds[4] - bounds[1],
        bounds[5] - bounds[2],
        1e-9,
    )
    near_distance = max(span * 1e-6, 1e-7)
    # 五个递增的偏移距离：从贴面到远场，逐层确认"面外侧到底是实体还是空"。
    trace_distances = [
        near_distance,
        max(span * 0.01, near_distance * 10.0),
        max(span * 0.05, near_distance * 20.0),
        max(span * 0.2, near_distance * 40.0),
        max(span * 2.0, near_distance * 80.0),
    ]
    classifier = classifier_type(shape)
    # 面数超过上限时等距抽样，而不是截断前 N 个，避免只采样到集中在同一区域的面。
    sample_limit = min(semantic_sample_limit, len(faces))
    if len(faces) <= sample_limit:
        sampled_faces = faces
    else:
        indices = np.linspace(
            0,
            len(faces) - 1,
            num=sample_limit,
            dtype=np.int64,
        )
        sampled_faces = [faces[int(index)] for index in np.unique(indices)]
    sampled_ids = {id(face) for face in sampled_faces}
    status_counts: dict[str, int] = {}
    warnings: list[str] = []
    # 未被采样的面标 resource_guard 的 UNRESOLVED，绝不冒充"已判定为外部"。
    for face in faces:
        if id(face) not in sampled_ids:
            # 被资源保护跳过的 Face 显式标 UNRESOLVED，
            # 绝不留成空证据：空证据在下游会被当成"没有问题"，
            # 而这批面其实根本没检查过。confidence 给 0.0
            # 也是为了让任何按置信度筛选的逻辑都不会误采信它。
            face.semantic_evidence = {
                "status": "UNRESOLVED",
                "method": "resource_guard",
                "confidence": 0.0,
                "warnings": [
                    "我因资源保护跳过了该 Face 的实体采样，"
                    "需要人工复核或提高采样上限。"
                ],
            }
            status_counts["UNRESOLVED"] = status_counts.get("UNRESOLVED", 0) + 1
            continue
        evidence = _face_semantic_evidence(
            face,
            classifier,
            point_type,
            trace_distances,
            in_state,
            on_state,
            out_state,
        )
        face.semantic_evidence = evidence
        # 取不到 status 时按 UNRESOLVED 计：宁可算作未定，
        # 也不能让缺失字段悄悄变成某个具体结论。
        status = str(evidence.get("status", "UNRESOLVED"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        # 采样参数与规模一并落盘：只看 status_counts 无法判断
        # 这份结论是在"全部面都采了"还是"只采了 1%"的情况下得出的。
        "sample_span": span,
        "near_distance": near_distance,
        "trace_distances": trace_distances,
        "semantic_sample_limit": sample_limit,
        "semantic_sampled_face_count": len(sampled_faces),
        # 跳过数必须给：它是"这份报告覆盖不全"程度的直接度量。
        "semantic_skipped_face_count": len(faces) - len(sampled_faces),
        "status_counts": status_counts,
        "warnings": warnings,
    }


def _face_semantic_evidence(
    face: StepFaceFeature,
    classifier: Any,
    point_type: Any,
    trace_distances: list[float],
    in_state: Any,
    on_state: Any,
    out_state: Any,
) -> dict[str, Any]:
    """我沿 Face 法向两侧采样，判断这是外表面还是内腔壁面。

    原理：一个面的"哪一侧是材料"要靠实体分类器实测，不能靠法向猜。
    我沿法向正、负两侧各采样一串递增距离，拿到两条 IN/ON/OUT
    状态轨迹；近场必须一内一外，才说明采样点确实贴在实体边界上
    （两侧都是 IN 说明点在实体内部，两侧都 OUT 说明点悬空）。
    然后看"开放侧"那条轨迹：全程 OUT 是外表面；中途进实体、末端
    又出实体是内腔壁面。两种都不满足就保持 UNRESOLVED。

    我为什么不猜而是给 UNRESOLVED：真实件上常遇到薄壁、相切面，
    采样轨迹会给出含混的模式。硬判成外表面会把内腔判成外表面，
    而这种错在报表里表现为"少了一个孔"。UNRESOLVED 至少是诚实的。
    """
    evidence: dict[str, Any] = {
        "status": "UNRESOLVED",
        "method": "normal_offset_solid_classifier",
        "confidence": 0.0,
        "positive_trace": [],
        "negative_trace": [],
        "open_side": None,
        "material_side": None,
        "warnings": [],
    }
    if face.center_xyz_mm is None or face.normal_xyz is None:
        evidence["warnings"] = ["我缺少 Face 重心或法向，无法进行实体采样。"]
        return evidence
    center = np.asarray(face.center_xyz_mm, dtype=np.float64)
    normal = np.asarray(face.normal_xyz, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if center.shape != (3,) or normal.shape != (3,) or length <= 1e-12:
        evidence["warnings"] = ["我发现 Face 重心或法向不可归一化。"]
        return evidence
    normal /= length

    # 沿法向正、负两侧各采样五个递增距离，得到两条内外状态轨迹。
    positive = _classify_trace(
        center,
        normal,
        classifier,
        point_type,
        trace_distances,
        in_state,
        on_state,
        out_state,
    )
    negative = _classify_trace(
        center,
        -normal,
        classifier,
        point_type,
        trace_distances,
        in_state,
        on_state,
        out_state,
    )
    evidence["positive_trace"] = positive
    evidence["negative_trace"] = negative
    if not positive or not negative:
        evidence["warnings"] = ["我无法得到 Face 法向两侧的完整采样轨迹。"]
        return evidence

    # 近场必须一内一外，否则说明点落在边界或退化位置，判据不成立。
    positive_near = positive[0]
    negative_near = negative[0]
    if {positive_near, negative_near} != {"IN", "OUT"}:
        evidence["warnings"] = [
            "我发现法向两侧没有形成一内一外的稳定近场证据。"
        ]
        return evidence

    if positive_near == "IN":
        material_side, open_side = "positive", "negative"
        open_trace = negative
    else:
        material_side, open_side = "negative", "positive"
        open_trace = positive
    evidence["material_side"] = material_side
    evidence["open_side"] = open_side

    # 开放侧先外、中间进入实体、最后又回到实体外 -> 内部腔体候选。
    # 开放侧全程在实体外 -> 外部面候选。其余模式一律保持 UNRESOLVED。
    middle_states = open_trace[1:-1]
    enters_material = "IN" in middle_states
    exits_again = enters_material and open_trace[-1] == "OUT"
    if enters_material and exits_again:
        evidence["status"] = "INTERNAL_CAVITY_CANDIDATE"
        evidence["confidence"] = 0.9
        evidence["warnings"] = [
            "我发现开放侧沿法向先位于实体外，再进入实体后重新回到实体外，"
            "这只是内部腔体候选证据。"
        ]
    elif all(state == "OUT" for state in open_trace):
        evidence["status"] = "EXTERNAL_CANDIDATE"
        evidence["confidence"] = 0.75
        evidence["warnings"] = [
            "我发现开放侧沿法向持续位于实体外，这只是外部面候选证据。"
        ]
    else:
        evidence["warnings"] = [
            "我发现开放侧采样轨迹不满足外部面或内部腔体的稳定模式。"
        ]
    return evidence


def _classify_trace(
    center: np.ndarray,
    direction: np.ndarray,
    classifier: Any,
    point_type: Any,
    distances: list[float],
    in_state: Any,
    on_state: Any,
    out_state: Any,
) -> list[str]:
    """我沿指定方向在若干距离上采样，返回每点的实体内外状态序列。

    内外状态序列是判断"这是孔的内壁还是外表面"的核心证据：
    沿面的法向往外走若干步，状态序列 OUT,OUT,OUT 说明外侧是空的
    （这是外表面）；IN,IN,IN 说明外侧是实体（这是孔的内壁）。

    容差按距离比例取（distance * 1e-4）而不是固定值：采样距离可能
    跨越几个数量级（毫米级小孔到米级大件），固定容差在近距离处
    相对太大会把内外判混，在远距离处相对太小会让点恰好落在边界上
    时分类器抖动。比例容差能跟随尺度。

    单点分类失败我记 UNKNOWN 而不是中断：序列里有一个 UNKNOWN
    只让这条证据变弱（下游会因此要求复核），
    整条序列作废会让这个面彻底失去内外判断。
    """
    # 容差按距离比例取，避免采样点恰好落在边界上时分类器抖动。
    states: list[str] = []
    for distance in distances:
        point = center + direction * float(distance)
        try:
            classifier.Perform(
                point_type(float(point[0]), float(point[1]), float(point[2])),
                max(float(distance) * 1e-4, 1e-9),
            )
            state = classifier.State()
            if state == in_state:
                states.append("IN")
            elif state == out_state:
                states.append("OUT")
            elif state == on_state:
                states.append("ON")
            else:
                states.append("UNKNOWN")
        except Exception:
            states.append("UNKNOWN")
    return states


def _face_normal(
    adaptor: Any,
    face: Any,
    reversed_type: Any,
    point_type: Any,
    vector_type: Any,
    warnings: list[str],
    face_id: int,
) -> list[float] | None:
    """我取参数域中点的一阶偏导叉积作为 Face 法向，并按朝向决定是否反向。

    原理说明：曲面在参数 (u, v) 处的两个一阶偏导分别指向 u、v 增大的
    方向，它们的叉积就是该点的法向。我取参数域**中点**处的法向作为
    这个面的代表法向——对平面这精确成立，对曲面是近似，
    所以我在签名里同时保存曲面类型，让下游知道这是代表值。

    REVERSED 朝向必须反向：OCP 里 Face 的 Orientation 决定这个面
    在实体中是"正向引用"还是"反向引用"。拿偏导叉积得到的几何法向
    不携带这个信息，不修正就会得到一半面朝里的法向，
    后续拔模、内外判断会整体错一半。

    参数域非有限（无限平面）和法向长度为零（退化面）都视为失败，
    记进 warnings 并返回 None。我不猜一个默认值——
    错误的法向比没有法向危险得多，它会让结论看起来正常。
    """
    try:
        u_min = float(adaptor.FirstUParameter())
        u_max = float(adaptor.LastUParameter())
        v_min = float(adaptor.FirstVParameter())
        v_max = float(adaptor.LastVParameter())
        if not all(map(lambda value: abs(value) < 1e100, (u_min, u_max, v_min, v_max))):
            raise ValueError("参数域不是有限范围")
        u = (u_min + u_max) / 2.0
        v = (v_min + v_max) / 2.0
        point = point_type()
        derivative_u = vector_type()
        derivative_v = vector_type()
        adaptor.D1(u, v, point, derivative_u, derivative_v)
        normal = derivative_u.Crossed(derivative_v)
        length = float(normal.Magnitude())
        if length <= 1e-12:
            raise ValueError("中间参数处法向长度接近零")
        # REVERSED 朝向要把叉积法向反向，才能得到与实体一致的外法向。
        if face.Orientation() == reversed_type:
            normal.Reverse()
        return [
            float(normal.X() / length),
            float(normal.Y() / length),
            float(normal.Z() / length),
        ]
    except Exception as exc:
        warnings.append(f"我无法计算 Face {face_id} 的法向：{exc}")
        return None
