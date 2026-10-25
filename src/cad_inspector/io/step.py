"""我把 STEP/STP 读取隔离成适配层，避免核心算法绑定某一个 CAD Python 包。

算法原理：STEP 是精确 B-Rep，检测管线要的是三角网格。这一步必须做"离散化"，而
离散化会引入弦高误差，所以我显式暴露两个控制量：

- ``linear_deflection_mm``：弦高误差上限，越小网格越密、越贴近原曲面，代价是网格量
  与内存增长。默认 0.04 mm 对模具零件的毫米级检测足够。
- ``angular_deflection_deg``：角度误差上限，控制曲面弯曲处的细分程度。

三角化后我按"坐标四舍五入到 1e-12"做顶点去重，让相邻 Face 共享的顶点合并成同一个
索引，避免网格出现重复顶点导致的缝隙与法向断裂；同时记录每个三角形来自哪个 Face
（``face_ids_by_triangle``），让后续区域证据能回溯到原始 B-Rep 面。

后端选择：优先 pythonocc-core，找不到再回落 OCP，两者都缺才报依赖缺失。这样换环境
不会导致核心算法改写。
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from ..errors import DependencyMissingError, GeometryProcessingError
from ..geometry.mesh import TriangleMesh


def load_step_as_mesh(
    path: str | Path,
    linear_deflection_mm: float = 0.04,
    angular_deflection_deg: float = 10.0,
) -> TriangleMesh:
    """我优先使用 pythonocc-core，找不到时再尝试 OCP。

    我为什么接受两个后端：pythonocc-core 和 OCP 是同一套 OpenCASCADE
    的两套 Python 绑定，API 几乎一致但包名不同，不同环境里装的可能
    是任意一个。支持两者能让项目在不同机器上都能跑，而核心算法
    一行都不用改。

    两个后端都**用时导入**，不在模块顶层 import：缺后端时，
    "只处理 STL"的调用方不应该因为没装 OCP 就连包都导入不了。

    只有两个后端都 ImportError 我才抛 DependencyMissingError。
    这里只捕获 ImportError 而不捕获其他异常：后端存在但读文件失败
    （例如文件损坏）是另一类问题，被这里吞掉会让真正的错误
    伪装成"没装依赖"。
    """

    path = Path(path)
    # 两个后端都是用时导入，缺哪个都不会在 import 阶段就炸掉整个包。
    try:
        return _load_with_pythonocc(
            path,
            linear_deflection_mm=linear_deflection_mm,
            angular_deflection_deg=angular_deflection_deg,
        )
    except ImportError:
        pass
    try:
        return _load_with_ocp(
            path,
            linear_deflection_mm=linear_deflection_mm,
            angular_deflection_deg=angular_deflection_deg,
        )
    except ImportError as exc:
        raise DependencyMissingError(
            "我无法读取 STEP/STP，因为当前环境没有安装 pythonocc-core 或 OCP。"
            "我仍然可以运行 STL 基线；要启用 STEP，请安装 pythonocc-core 或 cadquery-ocp。"
        ) from exc


def _load_with_pythonocc(
    path: Path,
    linear_deflection_mm: float,
    angular_deflection_deg: float,
) -> TriangleMesh:
    """我用 pythonocc-core 把 STEP 三角化成网格，并保留三角形到 Face 的映射。"""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    # 读取与转移失败直接报几何处理错误，不静默返回空网格。
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise GeometryProcessingError(f"我读取 STEP 失败：{path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    # 增量网格化：弦高与角度两个误差上限共同决定网格密度。
    BRepMesh_IncrementalMesh(
        shape,
        float(linear_deflection_mm),
        False,
        np.deg2rad(float(angular_deflection_deg)),
        True,
    )
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    face_ids_by_triangle: list[int] = []
    index_by_key: dict[tuple[float, float, float], int] = {}
    # 逐 Face 取三角化结果；顶点按 1e-12 量化去重，实现跨 Face 顶点共享。
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    while explorer.More():
        face_id += 1
        face = topods.Face(explorer.Current())
        location = face.Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is not None:
            transformation = location.Transformation()
            local_indices: dict[int, int] = {}
            for node_id in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(node_id).Transformed(transformation)
                point_tuple = (float(point.X()), float(point.Y()), float(point.Z()))
                key = tuple(round(value, 12) for value in point_tuple)
                if key not in index_by_key:
                    index_by_key[key] = len(vertices)
                    vertices.append(point_tuple)
                local_indices[node_id] = index_by_key[key]
            for triangle_id in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(triangle_id)
                node_a, node_b, node_c = triangle.Get()
                faces.append(
                    (
                        local_indices[node_a],
                        local_indices[node_b],
                        local_indices[node_c],
                    )
                )
                face_ids_by_triangle.append(face_id)
        explorer.Next()
    # 一个三角面都没有就等于读取失败，不能返回空网格让下游误判"没有缺失"。
    if not faces:
        raise GeometryProcessingError(f"我没有从 STEP 中得到可用三角面：{path}")
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        name=path.name,
        metadata={
            "format": "step_via_pythonocc",
            **_step_unit_metadata(_detect_step_unit(path)),
            "linear_deflection_mm": linear_deflection_mm,
            "angular_deflection_deg": angular_deflection_deg,
            "face_count": face_id,
            "face_ids_by_triangle": face_ids_by_triangle,
        },
    )


def _load_with_ocp(
    path: Path,
    linear_deflection_mm: float,
    angular_deflection_deg: float,
) -> TriangleMesh:
    """我保留 OCP 入口；不同 OCP 版本的底层 API 可能需要小范围适配。"""

    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    # OCP 与 pythonocc 的读取流程一致，只有返回状态常量与静态调用名不同。
    reader = STEPControl_Reader()
    # 读文件失败抛错，TransferRoots 失败不抛：
    # 前者是"文件打不开"，必须中断；后者只是没转移出根对象，
    # 后面 OneShape 会拿到空 Shape，由调用方按空模型处理。
    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise GeometryProcessingError(f"我读取 STEP 失败：{path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    # 四个参数依次是：线性偏差、是否相对偏差、角偏差（弧度）、是否并行。
    # 角偏差必须转弧度——OCP 在这里只认弧度，传角度会得到极密的网格。
    BRepMesh_IncrementalMesh(
        shape,
        float(linear_deflection_mm),
        False,
        np.deg2rad(float(angular_deflection_deg)),
        True,
    )
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    # 逐三角形记录它来自哪个 Face：下游区域识别要按 Face 回溯，
    # 三角化之后唯一的溯源线索就是这个表。
    face_ids_by_triangle: list[int] = []
    # 顶点按坐标去重：相邻 Face 共享边上的顶点在 OCP 里是两份，
    # 不去重会得到一个顶点重复、边不连通的网格，
    # 后续任何依赖共享顶点的算法（如边界提取）都会失效。
    index_by_key: dict[tuple[float, float, float], int] = {}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    while explorer.More():
        face_id += 1
        face = TopoDS.Face(explorer.Current())
        location = face.Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            transformation = location.Transformation()
            # 节点坐标必须先施加 Face 的局部变换：
            # OCP 的三角化存的是面局部坐标，
            # 不做变换会让装配体里的零件全部叠在原点附近。
            local_indices: dict[int, int] = {}
            for node_id in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(node_id).Transformed(transformation)
                point_tuple = (float(point.X()), float(point.Y()), float(point.Z()))
                # 键取 round 到 12 位：浮点坐标直接当键时，
                # 数学上同一点的两份三角化可能有 1e-16 级差异，
                # 会被判成两个不同顶点，去重就失效了。
                key = tuple(round(value, 12) for value in point_tuple)
                if key not in index_by_key:
                    index_by_key[key] = len(vertices)
                    vertices.append(point_tuple)
                local_indices[node_id] = index_by_key[key]
            for triangle_id in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(triangle_id)
                node_a, node_b, node_c = triangle.Get()
                faces.append(
                    (
                        local_indices[node_a],
                        local_indices[node_b],
                        local_indices[node_c],
                    )
                )
                face_ids_by_triangle.append(face_id)
        explorer.Next()
    if not faces:
        raise GeometryProcessingError(f"我没有从 STEP 中得到可用三角面：{path}")
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        name=path.name,
        metadata={
            "format": "step_via_ocp",
            **_step_unit_metadata(_detect_step_unit(path)),
            "linear_deflection_mm": linear_deflection_mm,
            "angular_deflection_deg": angular_deflection_deg,
            "face_count": face_id,
            "face_ids_by_triangle": face_ids_by_triangle,
        },
    )


def _detect_step_unit(path: Path) -> str:
    """我从 STEP 文本中读取明确单位，不对未知文件进行猜测。"""

    try:
        text = path.read_text(encoding="utf-8", errors="ignore").upper()
    except OSError:
        return "unknown"
    # 单位只认 STEP 里显式写出的 SI_UNIT；认不出就返回 unknown，绝不猜。
    if re.search(r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)", text):
        return "mm"
    if re.search(r"SI_UNIT\s*\(\s*\.CENTI\.\s*,\s*\.METRE\.\s*\)", text):
        return "cm"
    if re.search(r"SI_UNIT\s*\(\s*\$\s*,\s*\.METRE\.\s*\)", text):
        return "m"
    if "INCH" in text or "INCHES" in text:
        return "inch"
    return "unknown"


def _step_unit_metadata(unit: str) -> dict[str, object]:
    """我记录源单位和读取器归一化结果，避免检测管线重复缩放。"""

    # OCCT 读取器已把坐标归一化到 mm，所以这里只记录源单位与换算系数，不再二次缩放。
    scales = {
        "mm": 1.0,
        "cm": 10.0,
        "m": 1000.0,
        "inch": 25.4,
    }
    scale = scales.get(unit)
    return {
        "unit": unit,
        "source_unit": unit,
        "coordinate_unit": "mm" if scale is not None else "unknown",
        "unit_scale_to_mm": scale,
        "unit_normalization": (
            "occt_step_reader_to_mm"
            if scale is not None
            else "unknown_source_unit"
        ),
    }
