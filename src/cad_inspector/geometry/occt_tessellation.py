"""我用 OCCT 把 STEP 的 B-Rep 面三角化，并保留"三角形 -> CAD 面 ID"的映射。

我为什么要显式做这件事：

几何检测的距离、聚类、区域合并全部在**三角网格**上算，但甲方看到的
工程语义（"这个面是哪个面""这个区域属于哪个 CAD 面"）在 **B-Rep 面**上。
没有这层映射，检测结果就只能说"这里有个 3.2 mm2 的缺失"，
说不出"它落在第 17 号 CAD 面上"——而后一句才是模具工程师能直接用的。

我守住的边界：

- **面 ID 按遍历顺序编号**，且**同一 shape 上稳定**。我不用 Python 的
  `id()`、也不用 OCCT 的 `HashCode` 当 ID：前者每次进程都变，
  后者在不同 OCCT 版本间可能变。稳定 ID 是"报告能复核"的前提。
- **三角化参数可配**。弦高（chord）越小三角越密、越贴曲面，但面数暴涨；
  我不猜一个"万能值"，由调用方按件大小给，并在结果里把实际用的
  参数写回去，便于复核时复现。
- **退化/空三角化要留痕**。某个 B-Rep 面三角化失败时，我记进警告
  而不是静默跳过——静默跳过会让"这个面没被检测"看起来像"这个面没问题"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import TriangleMesh


@dataclass
class TessellationConfig:
    """我配置 OCCT 三角化的精细度。"""

    # 弦高误差：三角边到真实曲面的最大偏离，单位与模型一致。
    #
    # 我为什么把默认从 0.5 改成 0.04：InspectionConfig.mesh_linear_deflection_mm
    # 就是 0.04，而 CAD 面映射要求"三角化参数必须与加载网格时完全相同"——
    # 参数不同会得到不同数量的三角形，下标对不上，映射出的 CAD 面号
    # 全是错的而且不报错。默认值不统一，等于给了一个默认就踩坑的入口。
    linear_deflection: float = 0.04
    # 角度偏差（弧度）：曲面弯曲处的细分依据。
    # 同样与 InspectionConfig.mesh_angular_deflection_deg(10度) 对齐，
    # 10 度 = 0.1745 弧度。
    angular_deflection: float = 0.1745
    relative: bool = True

    def to_dict(self) -> dict[str, object]:
        """我把配置写进结果，便于复核时复现同一份三角化。"""

        return {
            "linear_deflection": self.linear_deflection,
            "angular_deflection": self.angular_deflection,
            "relative": self.relative,
        }


@dataclass
class TessellationResult:
    """我承载三角化结果以及面 ID 映射。"""

    mesh: TriangleMesh
    # 每个三角面属于哪个 CAD 面：下标与 mesh.faces 行号一一对应。
    triangle_to_face_id: np.ndarray
    # CAD 面 ID -> 面积（mm2），便于直接回答"哪个面缺了多少"。
    face_areas_mm2: dict[int, float] = field(default_factory=dict)
    # CAD 面 ID -> 该面上的三角形数量。
    face_triangle_counts: dict[int, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    config: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """我把结果转成可序列化字典。"""

        return {
            "triangle_count": int(len(self.mesh.faces)),
            "vertex_count": int(len(self.mesh.vertices)),
            "cad_face_count": int(len(self.face_areas_mm2)),
            "face_areas_mm2": {
                str(key): round(float(value), 6)
                for key, value in sorted(self.face_areas_mm2.items())
            },
            "face_triangle_counts": {
                str(key): int(value)
                for key, value in sorted(self.face_triangle_counts.items())
            },
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


def tessellate_shape(
    shape,
    config: TessellationConfig | None = None,
) -> TessellationResult:
    """我把一个 OCP TopoDS_Shape 三角化，并建立三角形到 CAD 面 ID 的映射。

    我在函数内部导入 OCP：它是一个重量级依赖，只在这条路径上才需要，
    顶层导入会把它变成整个项目的必需依赖。
    """

    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        import OCP.TopoDS as TopoDS
    except ImportError as exc:
        raise RuntimeError(
            "我无法做 OCCT 三角化，因为当前环境没有安装 OCP（cadquery-ocp）。"
        ) from exc

    config = config or TessellationConfig()
    warnings: list[str] = []

    # 三角化必须先跑一次：OCCT 的三角化结果挂在 shape 上，
    # BRep_Tool.Triangulation 只是去取，不会触发计算。
    BRepMesh_IncrementalMesh(
        shape,
        float(config.linear_deflection),
        bool(config.relative),
        float(config.angular_deflection),
        True,
    )

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    triangle_to_face_id: list[int] = []
    face_areas_mm2: dict[int, float] = {}
    face_triangle_counts: dict[int, int] = {}

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    # 面 ID 按遍历顺序从 1 开始编号。为什么不用 id()/HashCode：
    # id() 每次进程都变，HashCode 跨 OCCT 版本可能变，
    # 两者都会让"上次报的第 17 号面"这次对不上，报告没法复核。
    cad_face_id = 0
    while explorer.More():
        cad_face_id += 1
        face = TopoDS.TopoDS.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None or triangulation.NbTriangles() == 0:
            # 三角化失败必须留痕：静默跳过会让"这个面没被检测"
            # 看起来像"这个面没问题"，这是最危险的沉默。
            warnings.append(
                f"我未能三角化第 {cad_face_id} 号 CAD 面，它不会进入网格检测。"
            )
            explorer.Next()
            continue

        transform = location.Transformation()
        # 顶点下标是**面内局部**的，我要先记下它在全局顶点表里的起点，
        # 才能把三角形的局部下标平移成全局下标。
        vertex_offset = len(vertices)
        for index in range(1, triangulation.NbNodes() + 1):
            node = triangulation.Node(index)
            # 必须应用面的定位变换：不应用的话装配体里
            # 每个实例的顶点都会堆在原点附近，网格整个是错的。
            node = node.Transformed(transform)
            vertices.append([float(node.X()), float(node.Y()), float(node.Z())])

        face_area = 0.0
        for index in range(1, triangulation.NbTriangles() + 1):
            triangle = triangulation.Triangle(index)
            # OCCT 的三角形顶点下标从 1 开始，转成从 0 开始的全局下标。
            a = vertex_offset + int(triangle.Value(1)) - 1
            b = vertex_offset + int(triangle.Value(2)) - 1
            c = vertex_offset + int(triangle.Value(3)) - 1
            faces.append([a, b, c])
            triangle_to_face_id.append(cad_face_id)
            p0 = np.asarray(vertices[a], dtype=np.float64)
            p1 = np.asarray(vertices[b], dtype=np.float64)
            p2 = np.asarray(vertices[c], dtype=np.float64)
            face_area += float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)) * 0.5)

        face_areas_mm2[cad_face_id] = face_area
        face_triangle_counts[cad_face_id] = int(triangulation.NbTriangles())
        explorer.Next()

    if not faces:
        raise ValueError("我没有从这个 shape 里三角化出任何三角形。")

    mesh = TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        name="occt_tessellation",
    )
    return TessellationResult(
        mesh=mesh,
        triangle_to_face_id=np.asarray(triangle_to_face_id, dtype=np.int64),
        face_areas_mm2=face_areas_mm2,
        face_triangle_counts=face_triangle_counts,
        warnings=warnings,
        config=config.to_dict(),
    )


def load_step_shape(path: str | Path):
    """我读取 STEP 文件并返回一个 OCP shape。"""

    try:
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.STEPControl import STEPControl_Reader
    except ImportError as exc:
        raise RuntimeError(
            "我无法读取 STEP，因为当前环境没有安装 OCP（cadquery-ocp）。"
        ) from exc

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError(f"我读取 STEP 失败：{path}")
    reader.TransferRoots()
    return reader.OneShape()


def tessellate_step_file(
    path: str | Path,
    config: TessellationConfig | None = None,
) -> TessellationResult:
    """我读一个 STEP 文件并直接三角化，返回带面 ID 映射的结果。"""

    return tessellate_shape(load_step_shape(path), config)
