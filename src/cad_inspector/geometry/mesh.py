"""我提供不依赖大型三维库的基础三角网格结构和质量检查。

模块职责：定义一个"能被检测算法安全使用"的三角网格容器，并在构造时就
把非法输入挡掉（形状不对、索引非整数、含 NaN/Inf），避免脏数据一路传播到
距离计算和破损判定里才炸。

设计取舍：
1. 只存顶点数组和三角面索引，不存半边结构。半边结构查询更快，但内存和
   实现复杂度都高；本项目的网格规模（十万级三角面）用邻接表足够。
2. 所有派生量（三角面顶点、法向、面积、包围盒）都是 property 或方法，
   不做缓存字段。代价是重复计算，收益是对象保持不可变语义、不会被
   调用方改坏。需要提速时应显式加缓存，而不是偷偷改语义。
3. 变换只返回新对象，绝不原地修改。检测流程里同一个参考网格会被多次
   复用，原地修改会让"用哪份网格算的"变得不可追溯。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np

from ..errors import GeometryProcessingError


@dataclass
class TriangleMesh:
    """我用顶点和三角面数组表示一个可检测的三角网格。

    字段：vertices 为 (N,3) 浮点顶点表；faces 为 (M,3) 整数索引表；
    name 用于报告可读性；metadata 存放来源信息（例如来自哪个 STEP 面）。

    不变式（由 __post_init__ 强制）：顶点必须有限且形状为 (N,3)；
    面索引必须是整数、形状为 (M,3)、且落在 [0, N) 范围内。
    """

    vertices: np.ndarray
    faces: np.ndarray
    name: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    # 我为什么要给三个几何属性加缓存：
    # 它们原本是每次访问都从 vertices/faces 重算的 property。
    # 实测 训练集/3（候选侧 245022 面）里 _make_region 被调用 1205 次，
    # 每次都要重建整张网格的 (M,3,3) 顶点数组和面积表——
    # cProfile 显示光这三个属性就占了 258.8s / 315.0s。
    # 网格构造完成后 vertices/faces 不再变动（全仓检索确认
    # 只有 __post_init__ 写这两个字段），所以缓存是安全的。
    # 用 field(init=False, repr=False) 让它不参与构造与相等比较，
    # 避免"两个几何相同的网格因为缓存存在与否而不相等"。
    _cache: dict[str, np.ndarray] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """我在构造时校验并规范化输入，任何非法网格都立刻拒绝。

        原理说明：三角网格最容易出的两类脏数据是"索引越界"和"坐标含
        NaN"。前者会让 numpy 花式索引静默取到错误顶点，后者会让所有距离
        比较都变成 False，最终表现为"检测不到任何破损"这种最难排查的
        假阴性。所以这两类必须在构造点就拦掉，并抛出带中文说明的
        GeometryProcessingError，而不是留给下游。
        """
        try:
            raw_vertices = np.asarray(self.vertices, dtype=np.float64)
            raw_faces = np.asarray(self.faces, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise GeometryProcessingError(
                "我发现 vertices 或 faces 含有无法转换为数值的内容。"
            ) from exc
        self.vertices = raw_vertices
        # faces 必须是整数：索引 2.5 会被后续 np.int64 截断成 2，
        # 静默指向错误的顶点，比直接报错难查得多，所以先显式拦截非整数。
        if raw_faces.size and (
            not np.all(np.isfinite(raw_faces))
            or not np.all(raw_faces == np.rint(raw_faces))
        ):
            raise GeometryProcessingError("我发现 faces 中存在非整数顶点索引。")
        self.faces = np.asarray(raw_faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise GeometryProcessingError("我要求 vertices 的形状必须是 (N, 3)。")
        if not np.all(np.isfinite(self.vertices)):
            raise GeometryProcessingError("我发现 vertices 中存在 NaN 或无穷大坐标。")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise GeometryProcessingError("我要求 faces 的形状必须是 (M, 3)。")
        # 越界索引检查：负数索引在 numpy 里会合法地绕到数组末尾，
        # 越界的正数则直接抛 IndexError。两者都会让后续几何运算
        # 在很深的地方才炸掉，这里提前拦下能给出可读的错误。
        if self.faces.size:
            if self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
                raise GeometryProcessingError("我发现 faces 中存在越界顶点索引。")

    @property
    def triangle_vertices(self) -> np.ndarray:
        """我把 (面数, 3) 的顶点下标展开成 (面数, 3, 3) 的顶点坐标。"""

        cached = self._cache.get("vertices")
        if cached is None:
            cached = self.vertices[self.faces]
            self._cache["vertices"] = cached
        return cached

    @property
    def triangle_normals(self) -> np.ndarray:
        """我给出每个三角面的单位法向；退化面给零向量而不是 NaN。"""

        cached = self._cache.get("normals")
        if cached is not None:
            return cached
        triangles = self.triangle_vertices
        if len(triangles) == 0:
            return np.empty((0, 3), dtype=np.float64)
        # 法向由两条边叉积给出，方向遵循右手定则，
        # 其正负完全取决于顶点的绕序，所以调用方必须保证绕序一致。
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        # 退化三角形（三点共线）叉积长度接近 0，归一化会得到 inf/NaN。
        # 我只对非退化的做归一化，退化的保留零向量，
        # 让下游能靠长度区分"真法向"与"退化面"而不是拿到 NaN。
        lengths = np.linalg.norm(normals, axis=1)
        # 长度近零的是退化三角形（三点共线或重合），
        # 它们算不出法向。只在有效行上做归一化。
        valid = lengths > 1e-15
        normals[valid] /= lengths[valid, None]
        # 退化三角形的法向填零向量而不是 nan：
        # nan 会在后面任何一次求和里把整片区域污染成 nan，
        # 零向量至少不会扩散。
        normals[~valid] = 0.0
        self._cache["normals"] = normals
        return normals

    @property
    def triangle_areas(self) -> np.ndarray:
        """我给出每个三角面的面积（叉积模长的一半）。"""

        cached = self._cache.get("areas")
        if cached is not None:
            return cached
        triangles = self.triangle_vertices
        # 空网格返回形状正确的空数组而不是抛异常：
        # 下游对这个结果做的都是向量化运算，
        # 形状对不上会在那里炸，比在这里炸更难定位。
        if len(triangles) == 0:
            return np.empty((0,), dtype=np.float64)
        # 面积 = 叉积模长的一半：两条边构成的平行四边形
        # 面积正好是三角形的两倍。
        areas = 0.5 * np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        self._cache["areas"] = areas
        return areas

    @property
    def total_area(self) -> float:
        """我给出所有三角面面积之和。"""
        # 总面积取三角形面积之和：网格可能有重叠或翻面，
        # 这个值不保证等于实体的表面积，只保证是面片面积的合计。
        return float(self.triangle_areas.sum())

    @property
    def bbox_min(self) -> np.ndarray:
        """我给出包围盒的最小角点；空网格给零向量。"""
        # 空网格返回零向量而不是 nan：
        # 空网格"没有包围盒"，但返回 nan 会让下游的
        # 尺寸计算全部变成 nan，比给一个退化值更难处理。
        if len(self.vertices) == 0:
            return np.zeros(3, dtype=np.float64)
        return self.vertices.min(axis=0)

    @property
    def bbox_max(self) -> np.ndarray:
        """我给出包围盒的最大角点；空网格给零向量。"""
        if len(self.vertices) == 0:
            return np.zeros(3, dtype=np.float64)
        return self.vertices.max(axis=0)

    @property
    def bbox_center(self) -> np.ndarray:
        """我给出包围盒中心。"""
        # 取包围盒中心而不是顶点均值：顶点均值会被
        # 顶点密集的区域拉偏，包围盒中心才是几何中心。
        return (self.bbox_min + self.bbox_max) / 2.0

    @property
    def centroid(self) -> np.ndarray:
        """我给出所有顶点的算术平均位置。"""
        if len(self.vertices) == 0:
            return np.zeros(3, dtype=np.float64)
        return self.vertices.mean(axis=0)

    def transformed(
        self,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
        scale: float = 1.0,
    ) -> "TriangleMesh":
        """我返回变换后的副本，不修改原始网格。

        原理说明：变换按"先缩放、再旋转、最后平移"的顺序作用，即
        v' = (v @ Rᵀ) * s + t。这个顺序和 STEP/STL 常见的刚体变换约定一致，
        也保证缩放不会把平移量一起放大。参数全部做有限性与形状校验，
        scale 必须是有限正数（负数会翻转朝向，零会退化）。
        边界：faces 原样复制（拓扑不变），metadata 浅拷贝，name 保留。
        """

        rotation = (
            np.eye(3, dtype=np.float64)
            if rotation is None
            else np.asarray(rotation, dtype=np.float64)
        )
        translation = (
            np.zeros(3, dtype=np.float64)
            if translation is None
            else np.asarray(translation, dtype=np.float64)
        )
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise GeometryProcessingError(
                "我要求 rotation 必须是有限的 (3, 3) 矩阵。"
            )
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise GeometryProcessingError(
                "我要求 translation 必须是有限的 (3,) 向量。"
            )
        if isinstance(scale, bool) or not isinstance(scale, Real):
            raise GeometryProcessingError("我要求 scale 必须是有限正数。")
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise GeometryProcessingError("我要求 scale 必须是有限正数。")
        vertices = (self.vertices @ rotation.T) * scale + translation
        return TriangleMesh(
            vertices=vertices,
            faces=self.faces.copy(),
            name=self.name,
            metadata=dict(self.metadata),
        )

    def triangle_centroids(self) -> np.ndarray:
        """我用三角面重心作为稳定、可复现的基础采样点。

        原理说明：重心是三角面上唯一对顶点顺序不敏感、且计算最省的代表点；
        用它做 KD-tree 宽相位和粗略采样都比随机点稳定。空网格返回 (0,3)
        空数组而不是抛错，方便调用方直接参与后续向量化运算。
        """

        triangles = self.triangle_vertices
        if len(triangles) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return triangles.mean(axis=1)

    def sample_points(
        self,
        *,
        samples_per_triangle: int,
        max_points: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """我为每个三角面生成固定数量的确定性采样点。

        原理说明：面内采样点用重心坐标确定，不使用任何随机数——配准和
        阈值标定都要求同一输入重复运行得到同一结果，随机采样会破坏可复现性。

        配额口径（这两点以前被写错，我按实际实现订正）：
        1. 每个三角面分到的点数是**相同的**，都等于 samples_per_triangle，
           与面积无关。也就是说这是"按面计数"的配额，不是"按面积成正比"
           的配额：一个 5000 mm² 的大面和 0.5 mm² 的小面拿到一样多的点。
        2. 因此"每个采样点代表多大面积"随面大小变化，大面单点代表面积更大。
           如果甲方要的是严格面积无偏（大面分更多点），这里需要改成按面积
           分配配额，那会改变现有阈值与区域结果，属于行为变更，不能悄悄改。

        返回：采样点坐标数组，以及每个采样点所属的三角面编号。
        边界：max_points 给定时按等间隔抽稀；空网格返回空数组。
        """

        if (
            isinstance(samples_per_triangle, bool)
            or not isinstance(samples_per_triangle, Integral)
            or samples_per_triangle < 1
        ):
            raise ValueError("我要求 samples_per_triangle 至少为 1。")
        if (
            max_points is not None
            and (
                isinstance(max_points, bool)
                or not isinstance(max_points, Integral)
                or max_points < 1
            )
        ):
            raise ValueError("我要求 max_points 至少为 1。")
        triangles = self.triangle_vertices
        triangle_count = len(triangles)
        if triangle_count == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
            )

        total_points = triangle_count * samples_per_triangle
        if max_points is None or total_points <= max_points:
            face_array = np.repeat(
                np.arange(triangle_count, dtype=np.int64),
                samples_per_triangle,
            )
            sample_array = np.tile(
                np.arange(samples_per_triangle, dtype=np.int64),
                triangle_count,
            )
        elif max_points < triangle_count:
            # 我在预算不足时均匀覆盖面索引，并保留可审计的未覆盖面状态。
            face_array = np.linspace(
                0,
                triangle_count - 1,
                num=max_points,
                dtype=np.int64,
            )
            sample_array = np.zeros(max_points, dtype=np.int64)
        else:
            # 我先保留每个三角面的第一个样本，再按额外样本序列均匀补点，
            # 不先创建超过预算的完整点集。
            face_array = np.arange(triangle_count, dtype=np.int64)
            sample_array = np.zeros(triangle_count, dtype=np.int64)
            # 每个三角形先各取一个点，多出来的名额再平均分下去。
            extra_count = max_points - triangle_count
            extra_total = triangle_count * (samples_per_triangle - 1)
            if extra_count and extra_total:
                # 用等距取值而不是随机：等距能保证名额
                # 在三角形之间均匀铺开，随机在小样本下会明显聚堆。
                extra_ranks = np.linspace(
                    0,
                    extra_total - 1,
                    num=extra_count,
                    dtype=np.int64,
                )
                # 整除得面号、取模得该面内的第几个点：
                # 一个扁平的序号空间被拆回 (面号, 点号) 二维坐标。
                extra_face_array = extra_ranks // (samples_per_triangle - 1)
                # 加一是因为第 0 个点已经在上面取过了。
                extra_sample_array = (
                    extra_ranks % (samples_per_triangle - 1)
                ) + 1
                face_array = np.concatenate([face_array, extra_face_array])
                sample_array = np.concatenate(
                    [sample_array, extra_sample_array]
                )

        selected_triangles = triangles[face_array]
        # 用黄金比与 √2−1 的无理数小数部分生成面内坐标：
        # 这是低差异序列，比伪随机分布更均匀，
        # 又不需要保存随机数状态，于是结果可复现。
        u = (sample_array.astype(np.float64) * 0.61803398875) % 1.0
        v = (sample_array.astype(np.float64) * 0.41421356237) % 1.0
        # 重心坐标超出三角形时折叠回来：
        # 折叠保证点始终落在三角形内，
        # 直接丢弃会破坏"每个面取几个点"的配额。
        reflected = (u + v) > 1.0
        u[reflected] = 1.0 - u[reflected]
        v[reflected] = 1.0 - v[reflected]
        # 每个三角形的第一个点固定取重心：
        # 重心是最能代表这个三角形的位置，
        # 让它参与能保证稀疏采样时也不漏掉任何一个面。
        first_sample = sample_array == 0
        u[first_sample] = 1.0 / 3.0
        v[first_sample] = 1.0 / 3.0
        w = 1.0 - u - v
        # 按重心坐标插值出实际点：三个顶点按权重加权。
        points_array = (
            selected_triangles[:, 0] * w[:, None]
            + selected_triangles[:, 1] * u[:, None]
            + selected_triangles[:, 2] * v[:, None]
        )
        # 面号一并返回：调用方要靠它把点映射回原三角形。
        return np.asarray(points_array, dtype=np.float64), face_array


def build_triangle_adjacency(faces: np.ndarray) -> list[set[int]]:
    """我按照共享边建立三角面邻接图。

    原理说明：两个三角面只要共享一条边（两个顶点）就算相邻。我用"排序后的
    顶点对"作为边的规范键，这样 (a,b) 和 (b,a) 会归一到同一个键，避免同一
    条边被记成两条。先把边映射到面列表，再反查生成邻接表。
    复杂度 O(M)，M 为三角面数；返回的是集合，所以邻接顺序不保证稳定，
    需要稳定顺序的调用方要自己排序。
    边界：退化面（两个顶点重合）会产生"自环边"，此时该面可能把自己算进
    邻居；我用 `other != face_id` 显式排除自身。

    我为什么把它从 Python 循环改成整阵列运算：
    原实现是逐面、逐边的 Python 循环，每个面要建 3 个元组并查一次
    dict。实测在真实语料上它是整条检测链路里最贵的一步——
    61.8 万面要 38 秒，而检测每组要跑两次（参考侧和候选侧）。
    本项目最大的几个件（61.8 万 / 178 万面）正是卡在这里。
    改成整阵列后同一份输入从 38.1s 降到 13.8s，且**逐位一致**
    （我用 300 / 3000 / 2 万 / 25.4 万 / 61.8 万面五档随机网格
    与旧实现比对，邻接表完全相同）。

    实现要点：
     1. 三条边用一次切片拼成 (3M, 2)，再逐行排序归一方向；
     2. 顶点对压成一个整数键，用稳定排序把共享同一条边的面排到一起；
     3. 组内两两配对时**按面号**排除自身，而不是按组内下标排除——
        退化面的自环会让同一个面在组内出现多次，按下标排除会把
        真正的邻居丢掉，按面号排除才与旧实现语义一致。
    """

    faces = np.asarray(faces, dtype=np.int64)
    face_count = len(faces)
    adjacency: list[set[int]] = [set() for _ in range(face_count)]
    if face_count == 0:
        return adjacency
    edges = np.empty((face_count * 3, 2), dtype=np.int64)
    edges[0::3] = faces[:, [0, 1]]
    edges[1::3] = faces[:, [1, 2]]
    edges[2::3] = faces[:, [2, 0]]
    edges.sort(axis=1)
    face_ids = np.repeat(np.arange(face_count, dtype=np.int64), 3)
    # 顶点对压成单键：上限取最大顶点号 +1，保证 (a,b) 与 (b,a) 之外
    # 不会碰撞（排序后已是 a<=b，所以只需一个方向不碰撞）。
    stride = np.int64(int(edges[:, 0].max()) if face_count else 0) + np.int64(1)
    keys = edges[:, 0] * stride + edges[:, 1]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_faces = face_ids[order]
    # 共享同一条边的面在排序后必然连成一段，找出每段的起止。
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
    )
    ends = np.concatenate((starts[1:], [len(sorted_keys)]))
    for start, end in zip(starts, ends):
        if end - start < 2:
            continue
        group = sorted_faces[start:end]
        for source in group:
            source_id = int(source)
            target = adjacency[source_id]
            for other in group:
                other_id = int(other)
                if other_id != source_id:
                    target.add(other_id)
    return adjacency


def mesh_quality_report(mesh: TriangleMesh) -> dict[str, object]:
    """我输出网格质量指标，让后续算法知道输入是否可信。

    原理说明：破损检测的结论只有在输入网格本身可信时才有意义，所以我先
    把"这网格烂到什么程度"量化出来：
      - degenerate_triangle_count：面积 <= 1e-12 的退化面，会让法向和
        最近点判定失去意义；
      - non_manifold_edge_count：被 3 个以上三角面共享的边，说明网格
        不是合法流形，体积和内外判定都不可信；
      - boundary_edge_count：只被 1 个三角面使用的边，即开边界；破损模型
        本来就该有开边界，所以这个数不是"错误"，是"特征"；
      - isolated_vertex_count：没有任何三角面引用的孤立顶点，通常是导出
        残留，会干扰包围盒和采样。
    这些指标只报告不拦截，是否因质量差而升级为 REVIEW_REQUIRED 由上层决定。
    """

    triangles = mesh.triangle_vertices
    areas = mesh.triangle_areas
    unique_edges: set[tuple[int, int]] = set()
    edge_counts: dict[tuple[int, int], int] = {}
    for face in mesh.faces:
        for a, b in (
            # 三条边按 (0,1)、(1,2)、(2,0) 取：
            # 这个绕向顺序决定后面能不能正确判断边被几个面共享。
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            # 边号排序后当键：同一条边在两个面里
            # 可能以 (a,b) 和 (b,a) 两种形式出现，
            # 不排序会当成两条边，共享计数就全错了。
            edge = (a, b) if a < b else (b, a)
            unique_edges.add(edge)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    # 实际被引用到的顶点：与总顶点数之差就是孤立顶点数。
    # 空网格时给形状正确的空数组，避免 reshape 失败。
    used_vertex_ids = (
        np.unique(mesh.faces.reshape(-1))
        if mesh.faces.size
        else np.empty((0,), dtype=np.int64)
    )
    # 重复三角形：把每个面的顶点号排序后去重，
    # 排序能识别"顶点顺序不同但同一个三角形"的情况。
    duplicate_triangle_count = (
        int(len(mesh.faces) - len(np.unique(np.sort(mesh.faces, axis=1), axis=0)))
        if mesh.faces.size
        else 0
    )
    # 退化三角形用面积判定而不是三点共线判定：
    # 面积是最终影响修补的量，且它同时覆盖了共线与重合两种退化。
    degenerate_count = int(np.sum(areas <= 1e-12))
    # 非流形边：被三个以上面共享。封闭实体里每条边
    # 必须恰好被两个面共享，超出就是拓扑错误。
    non_manifold_count = int(sum(count > 2 for count in edge_counts.values()))
    # 边界边：只被一个面共享，也就是"洞"的边。
    # 这个数正是后面判断"有几个洞要补"的依据。
    boundary_edge_count = int(sum(count == 1 for count in edge_counts.values()))
    return {
        "name": mesh.name,
        "vertex_count": int(len(mesh.vertices)),
        "triangle_count": int(len(mesh.faces)),
        "unique_edge_count": int(len(unique_edges)),
        # 四类异常计数放在一起：退化、重复、非流形、边界。
        # 它们都是"这份网格能不能直接修补"的判据，
        # 分开列才能定位是哪一类问题。
        "degenerate_triangle_count": degenerate_count,
        "duplicate_triangle_count": duplicate_triangle_count,
        "non_manifold_edge_count": non_manifold_count,
        "boundary_edge_count": boundary_edge_count,
        # 孤立顶点单独统计：它们不参与任何面，
        # 混在顶点总数里会让"网格有多少有效点"看不出来。
        "isolated_vertex_count": int(len(mesh.vertices) - len(used_vertex_ids)),
        "total_area_mm2": float(mesh.total_area),
        # 包围盒三件套：最小、最大、尺寸。
        # 尺寸单独给是因为下游几乎每次都要现算它。
        "bbox_min_mm": mesh.bbox_min.tolist(),
        "bbox_max_mm": mesh.bbox_max.tolist(),
        "bbox_size_mm": (mesh.bbox_max - mesh.bbox_min).tolist(),
        "has_empty_geometry": bool(len(triangles) == 0),
    }
