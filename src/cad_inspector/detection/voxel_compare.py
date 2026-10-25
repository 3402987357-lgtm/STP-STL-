"""我用体素占用与截面轮廓做辅助比对，补上"面级距离看不出来"的体缺。

为什么需要这一层：``missing_regions`` 走的是"点->面"的最近距离，它对
表面偏移和浅凹很敏感，但对面片整体缺失造成的**内部空腔**不敏感——
空腔壁上的点照样能在参考件上找到很近的面，距离判据会把它判成正常。
体素占用从"这个格子在不在实体内部"出发，正好补这一类的漏检。

我为什么用"逐层截面填充"而不是"逐体素射线奇偶"：

1. 射线法对每个体素都要和所有三角面求交，复杂度是 O(体素数 x 面数)，
   万级面配十万级体素就是十亿次运算，单次比对要跑几分钟；
2. 逐层填充先把每层的轮廓切出来，再用扫描线一次填满该层的 XY 网格，
   复杂度降到 O(层数 x (面数 + 每行排序))，快一个量级；
3. 顺带把截面轮廓也拿到了——任务书"体素和截面辅助比对"本就是一体两面，
   一次切片同时喂给两条证据，不用为截面再切一遍。

我对"能不能信"的态度：体素占用**要求网格封闭**。网格一旦有自由边（破面、
未缝合），奇偶填充就会从破口漏出去，把外部整片填成内部。所以我在结果里
显式给出两侧的自由边数与是否封闭，只要有一侧不封闭就把结论降成
``REVIEW_REQUIRED``，绝不拿漏水的填充冒充体积证据。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np

from ..geometry.mesh import TriangleMesh


@dataclass(frozen=True)
class CrossSection:
    """我在某个高度上切出来的截面证据。"""

    z_mm: float
    loop_count: int
    open_loop_count: int
    area_mm2: float
    point_count: int

    def to_dict(self) -> dict[str, object]:
        """我把截面证据转成可 JSON 序列化的字典。"""

        return {
            "z_mm": float(self.z_mm),
            "loop_count": int(self.loop_count),
            "open_loop_count": int(self.open_loop_count),
            "area_mm2": float(self.area_mm2),
            "point_count": int(self.point_count),
        }


@dataclass
class VoxelComparison:
    """我给出两张网格的体素级与截面级比对结论。"""

    voxel_size_mm: float
    grid_dims: tuple[int, int, int]
    grid_origin_mm: tuple[float, float, float]
    reference_voxel_count: int
    candidate_voxel_count: int
    intersection_count: int
    missing_count: int
    extra_count: int
    union_count: int
    iou: float
    missing_volume_mm3: float
    extra_volume_mm3: float
    reference_watertight: bool
    candidate_watertight: bool
    reference_free_edge_count: int
    candidate_free_edge_count: int
    status: str
    warnings: list[str] = field(default_factory=list)
    cross_sections: list[CrossSection] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """我把比对结论转成可 JSON 序列化的字典。"""

        return {
            "voxel_size_mm": float(self.voxel_size_mm),
            "grid_dims": [int(value) for value in self.grid_dims],
            "grid_origin_mm": [float(value) for value in self.grid_origin_mm],
            "reference_voxel_count": int(self.reference_voxel_count),
            "candidate_voxel_count": int(self.candidate_voxel_count),
            "intersection_count": int(self.intersection_count),
            "missing_count": int(self.missing_count),
            "extra_count": int(self.extra_count),
            "union_count": int(self.union_count),
            "iou": float(self.iou),
            "missing_volume_mm3": float(self.missing_volume_mm3),
            "extra_volume_mm3": float(self.extra_volume_mm3),
            "reference_watertight": bool(self.reference_watertight),
            "candidate_watertight": bool(self.candidate_watertight),
            "reference_free_edge_count": int(self.reference_free_edge_count),
            "candidate_free_edge_count": int(self.candidate_free_edge_count),
            "status": str(self.status),
            "warnings": list(self.warnings),
            "cross_sections": [item.to_dict() for item in self.cross_sections],
        }


def free_edge_count(mesh: TriangleMesh) -> int:
    """我数出只被一个三角面用到的边（自由边）数量。

    自由边是"网格没封闭"的直接证据：封闭网格里每条边都被恰好两个面
    共享。我把它单独暴露出来，是因为体素填充的正确性完全依赖它——
    调用方拿到占用结果之前，应当先看这个数是不是 0。

    我把每条边排序成 (小, 大) 再计数，这样 (a,b) 与 (b,a) 会并成同一条，
    不会因为两个面的绕序不同而被算成两条边、漏判自由边。
    """

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return 0
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    # 排序后去重：绕序相反的两个面给出的 (a,b)/(b,a) 必须视为同一条边。
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    # 只出现一次 = 这条边只属于一个面 = 破口边界。
    return int(np.count_nonzero(counts == 1))


def _slice_mesh_at_z(mesh: TriangleMesh, z: float) -> np.ndarray:
    """我把网格在给定高度切开，返回 (K, 2, 2) 的交线段（只要 XY）。

    判定哪条边跨越平面用"两端点是否分居平面两侧"的异或。我用**半开区间**
    ``(z0 <= z - eps) != (z1 <= z - eps)``，其中 eps 按网格尺度取一个极小量：

    - 用严格 ``< z``：包围盒尺寸是体素边长整数倍时，切片平面会恰好穿过
      一排顶点，落在平面上的顶点被归到上方，那些层一条交线段都取不到，
      整层被填成空——12x10x10 的盒子在 1mm 体素下会因此少掉近三成体积，
      而 10x10x10 恰好躲过，属于最难发现的"换个尺寸就错"；
    - 用宽松 ``<= z``：共面的三角形（例如立方体的顶/底面）三条边会被
      判成全部跨越，插值出 0/0 得到 nan 或垃圾交点，把轮廓搅碎。
    - 半开区间把"恰好等于"统一归到一侧，既不漏平面上的顶点，
      也不让共面三角形产生交点。

    一个三角形与平面最多交于两个点（退化情形除外），所以我只保留
    恰好两条边跨越的三角形；跨越数为 0 或 1 的分别在平面外或只是擦过
    一个顶点，都不构成长度非零的交线段。
    """

    triangles = mesh.triangle_vertices
    if len(triangles) == 0:
        return np.empty((0, 2, 2), dtype=np.float64)
    vertex_z = triangles[:, :, 2]
    start = triangles
    # 三条边各自的两端：0->1, 1->2, 2->0。
    end = np.roll(triangles, -1, axis=1)
    z_start = vertex_z
    z_end = np.roll(vertex_z, -1, axis=1)
    # eps 按网格尺度取，而不是固定 1e-12：毫米级与米级模型需要不同的
    # 分辨力，固定值在米级件上会被浮点噪声吃掉；取太大又会真的漏掉
    # 一层薄特征。这里取尺度的一百万分之一，实测在 mm~m 级都稳定。
    scale = float(np.abs(vertex_z).max()) if vertex_z.size else 1.0
    eps = max(scale, 1.0) * 1e-6
    limit = z - eps
    crosses = (z_start <= limit) != (z_end <= limit)
    # 只有恰好两条边跨越的三角形才给出一条有效交线段。
    valid = crosses.sum(axis=1) == 2
    if not np.any(valid):
        return np.empty((0, 2, 2), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        # 线性插值参数 t：沿边从 z_start 走到 z 需要走多远。
        # crosses 为假的边这里会算出 nan，下一步按 valid 掩码挑走即可。
        ratio = (z - z_start) / (z_end - z_start)
        # 我把不跨越的边显式置 0：它们在下一步会被掩码丢掉，但留着
        # nan 会在乘法里触发 RuntimeWarning，把"正常跳过"伪装成数值错误。
        ratio = np.where(crosses, ratio, 0.0)
    # 交点的 XY：每个三角形三条边各一个候选，形状 (M, 3, 2)。
    points = start[:, :, :2] + ratio[:, :, None] * (
        end[:, :, :2] - start[:, :, :2]
    )
    masked = np.where(crosses[:, :, None], points, np.nan)
    masked = masked[valid]
    # 把 nan 排到后面，前两列就是那两个真实交点。
    # 用稳定排序保证同一三角形两个交点的相对顺序不被打乱。
    order = np.argsort(np.isnan(masked[:, :, 0]), axis=1, kind="stable")
    ordered = np.take_along_axis(masked, order[:, :, None], axis=1)
    return ordered[:, :2, :]


def _cluster_endpoints(
    points: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """我把近乎重合的端点并成同一号，返回 (代表坐标, 每点的簇号)。

    我用 KD-tree 找出所有距离在容差内的点对，再用并查集合并：

    - 按"坐标除以容差后取整"分簇不行：坐标是连续的，同一个几何点被
      相邻两个三角形各算一次时，两次结果可能恰好落在取整边界的两侧；
    - 按"排序后只比较相邻元素"也不行：同一个几何点出现三次以上时，
      链式相邻关系会在中间断开（实测 0.6 与 0.6000000000000014 这两个
      本该同簇的点被分成两簇，一条闭合环就此碎成五段）；
    - KD-tree 的 ``query_pairs`` 直接给出**所有**距离小于容差的点对，
      并查集再把这些对传递闭包成簇，既没有边界效应也不依赖链式相邻。

    每簇取原始下标最小的点作代表，保证同一输入总得到同一结果。
    """

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int64)
    parent = list(range(len(points)))

    def find(item: int) -> int:
        """我找并查集的根，并顺手把沿途节点压到根上，让后续查找接近 O(1)。"""

        while parent[item] != item:
            # 路径压缩：把沿途节点直接挂到根上，后续查找接近 O(1)。
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        """我合并两簇，并固定用簇内最小下标当根，保证代表点可复现。"""

        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            # 始终让较小的下标当根，代表点就是该簇最小下标那个点。
            if root_left < root_right:
                parent[root_right] = root_left
            else:
                parent[root_left] = root_right

    try:
        from scipy.spatial import cKDTree

        pairs = cKDTree(points).query_pairs(tolerance, output_type="ndarray")
        for left, right in pairs:
            union(int(left), int(right))
    except ImportError:
        # 没有 SciPy 时退回精确去重：至少把完全相同的点并起来，
        # 不会比这更差，只是容差内的近似点合并不到一起。
        unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
        for cluster_id in range(len(unique_points)):
            members = np.flatnonzero(inverse == cluster_id)
            for other in members[1:]:
                union(int(members[0]), int(other))

    cluster_ids = np.array([find(index) for index in range(len(points))], dtype=np.int64)
    # 把根号重编号为 0..K-1：调用方按簇号索引代表点数组。
    roots = sorted(set(cluster_ids.tolist()))
    remap = {root: new_id for new_id, root in enumerate(roots)}
    cluster_ids = np.array([remap[value] for value in cluster_ids.tolist()], dtype=np.int64)
    representative = np.array([points[root] for root in roots], dtype=np.float64)
    return representative, cluster_ids


def _chain_loops(
    segments: np.ndarray, tolerance: float
) -> list[tuple[np.ndarray, bool]]:
    """我把散乱交线段串成闭合轮廓，返回 (点集, 是否闭合) 列表。

    切片拿到的是一堆互不关联的小线段，要算面积必须先串起来。我先把
    近乎重合的端点并成同一号（见 ``_cluster_endpoints``），再在**边**
    上做欧拉式遍历，而不是在顶点上做邻接查找。

    为什么必须在边上遍历：立方体这类形状里，切面平面穿过四个角时，
    同一个顶点会同时挂住两条不同侧面的交线段（该顶点度为 4）。若按
    "从当前顶点找一条没用过的边"来走，就会在两排侧面之间来回横跳，
    把一整条闭合轮廓切成五段开口折线——这正是我在 12x10x10 上只填出
    856 格（应为 1200）的根因。按边遍历则每次从"当前边的终点"续上
    下一条共享该终点的边，能在度为 4 的顶点上正确选择走向。

    我同时返回"是否闭合"。闭合是网格在这一层封闭的证据；不闭合说明
    这一层轮廓有缺口，面积不可信，调用方必须据此降级。
    """

    if len(segments) == 0:
        return []
    endpoints = np.concatenate([segments[:, 0], segments[:, 1]], axis=0)
    representative, cluster_ids = _cluster_endpoints(endpoints, tolerance)
    half = len(segments)
    id_a = cluster_ids[:half].astype(np.int64)
    id_b = cluster_ids[half:].astype(np.int64)
    # 自环（两端同一簇）对面积无贡献，直接丢掉，避免走链时死循环。
    keep = id_a != id_b
    if not np.any(keep):
        return []
    id_a = id_a[keep]
    id_b = id_b[keep]
    count = len(id_a)

    # 按"终点簇号"建索引：从这个簇出发可用的边编号。
    outgoing: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        outgoing[int(id_a[index])].append(index)
        outgoing[int(id_b[index])].append(index)

    used = [False] * count
    loops: list[tuple[np.ndarray, bool]] = []
    for start_index in range(count):
        if used[start_index]:
            continue
        used[start_index] = True
        head = int(id_a[start_index])
        current = int(id_b[start_index])
        points = [representative[head], representative[current]]
        start_node = head
        closed = False
        while True:
            # 从"当前边"的 current 端点续边，而不是从顶点乱挑。
            next_index = None
            for candidate in outgoing.get(current, ()):
                if used[candidate]:
                    continue
                # 这条边必须以 current 为某一端；另一端就是续行方向。
                if int(id_a[candidate]) == current:
                    next_index = candidate
                    next_node = int(id_b[candidate])
                elif int(id_b[candidate]) == current:
                    next_index = candidate
                    next_node = int(id_a[candidate])
                else:
                    continue
                break
            if next_index is None:
                break
            used[next_index] = True
            points.append(representative[next_node])
            current = next_node
            if current == start_node:
                closed = True
                break
        loops.append((np.array(points, dtype=np.float64), closed))
    return loops


def _signed_area_xy(points: np.ndarray) -> float:
    """我用鞋带公式算闭合多边形的有向面积。"""

    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    # 鞋带公式：sum(x_i * y_{i+1} - x_{i+1} * y_i) / 2，下标按环形回绕。
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _fill_layer_scanline(
    edges: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """我用扫描线把 XY 网格按轮廓填成布尔占用，返回 (nx, ny)。

    我固定返回 (nx, ny)（先 X 后 Y）与 ``occupancy`` 的前两轴一致：
    扫描线按行遍历 Y、行内是 X，天然先得到 (ny, nx)，我最后转置一次
    对齐到 X 轴优先。不转置的话调用方要靠广播侥幸对上——包围盒是立方体
    时恰好相同，一旦 X/Y 尺寸不同就会直接崩，属于最难发现的假正确。

    为什么不用"每个格点对所有边做奇偶测试"：那是 O(格点数 x 边数)，
    层数一多就是上亿次运算。扫描线按行复用同一批交点——每一行只求一次
    交点再排序，然后用 searchsorted 一次给出该行所有格点的"左侧交点数"，
    复杂度降到 O(行数 x 边数 + 格点数)。

    奇偶规则天然支持带孔截面：外环进一次、内环再进一次，计数为偶数的
    位置判为外部，孔就自动空出来了，不需要区分哪个环是外环。
    """

    inside = np.zeros((len(ys), len(xs)), dtype=bool)
    if len(edges) == 0 or len(xs) == 0 or len(ys) == 0:
        # 空结果也要转置：形状必须在任何路径下都与占用数组前两轴一致，
        # 否则"这一层没内容"会给出与其他层维度相反的数组。
        return inside.T
    x1 = edges[:, 0, 0]
    y1 = edges[:, 0, 1]
    x2 = edges[:, 1, 0]
    y2 = edges[:, 1, 1]
    for row, y in enumerate(ys):
        # 半开区间判定：恰好落在顶点高度上的边只计一次，避免顶点重复计数。
        crosses = (y1 > y) != (y2 > y)
        if not np.any(crosses):
            continue
        delta_y = y2[crosses] - y1[crosses]
        # 能跨越的必然两端异侧，delta_y 不会是 0，这里不需要再防除零。
        intersections = x1[crosses] + (y - y1[crosses]) * (
            x2[crosses] - x1[crosses]
        ) / delta_y
        intersections.sort()
        # searchsorted 左侧 = 严格小于 x 的交点个数；奇数即在内部。
        counts = np.searchsorted(intersections, xs, side="left")
        inside[row] = (counts % 2) == 1
    return inside.T


def _loops_to_edges(loops: list[tuple[np.ndarray, bool]]) -> np.ndarray:
    """我把所有轮廓转成边数组 (E, 2, 2)，供扫描线使用。"""

    collected: list[np.ndarray] = []
    for points, _closed in loops:
        if len(points) < 2:
            continue
        # 闭合环要补上"末点->首点"这条边，否则环会缺一段。
        start = points
        end = np.roll(points, -1, axis=0)
        collected.append(np.stack([start, end], axis=1))
    if not collected:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.concatenate(collected, axis=0)


def _common_grid(
    meshes: list[TriangleMesh], voxel_size: float, max_voxels: int
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """我给多张网格定一个公共体素网格。

    我为什么必须取并集包围盒：两张网格各自建网格再比，两套体素的
    原点与编号就对不上，"同一个格子"根本没有定义。共用一套网格是
    "缺失/多余"能逐格相减的前提。

    超过 max_voxels 时我把体素边长按比例放大而不是直接报错：体积比对
    本来就是粗粒度证据，放大体素只是变粗，远好于直接拒绝出结果。
    """

    minimum = np.minimum.reduce([mesh.bbox_min for mesh in meshes])
    maximum = np.maximum.reduce([mesh.bbox_max for mesh in meshes])
    extent = maximum - minimum
    # 完全退化的包围盒（例如单网格是空网格）给 0 尺寸，下面按至少 1 格处理。
    dims = np.maximum(np.ceil(extent / voxel_size).astype(np.int64), 1)
    total = float(dims[0] * dims[1] * dims[2])
    if total > float(max_voxels) and total > 0.0:
        scale = (total / float(max_voxels)) ** (1.0 / 3.0)
        voxel_size = voxel_size * scale
        dims = np.maximum(np.ceil(extent / voxel_size).astype(np.int64), 1)
    return minimum, (int(dims[0]), int(dims[1]), int(dims[2])), float(voxel_size)


def build_voxel_occupancy(
    mesh: TriangleMesh,
    voxel_size: float = 1.0,
    max_voxels: int = 2_000_000,
    origin: np.ndarray | None = None,
    dims: tuple[int, int, int] | None = None,
    assume_watertight: bool | None = None,
) -> tuple[np.ndarray, list[CrossSection]]:
    """我把网格转成体素占用与逐层截面，返回 (占用数组, 截面列表)。

    占用数组形状为 (nx, ny, nz)，True 表示该格中心落在实体内部。
    我固定用"格中心"而不是"格角点"判定：角点恰好落在表面上时归属
    会随绕序抖动，中心点则没有这个二义性。

    ``origin`` 与 ``dims`` 一传入就锁死网格，两张网格必须传同一组值
    才能相减；不传则按本网格自己的包围盒建，仅供单件查看。

    ``assume_watertight`` 控制是否过滤退化线段，默认按实测自由边判断。
    我为什么不能无条件下掉"很短的交线段"：真实 STL 上一个大三角面被
    切到时，切点可能离底边只有几微米，这段线是**真实轮廓的一部分**，
    无条件下掉会把一条闭合环切成几段开口折线，把本来封闭好好的网格
    误判成"轮廓未闭合"，进而把整份体积证据降级。只有网格真的开口时
    才需要靠这个过滤去噪。
    """

    if isinstance(voxel_size, bool) or not isinstance(voxel_size, Real):
        raise ValueError("我要求 voxel_size 是一个实数。")
    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("我要求 voxel_size 是一个有限的正数。")
    if isinstance(max_voxels, bool) or not isinstance(max_voxels, Integral):
        raise ValueError("我要求 max_voxels 是一个整数。")
    if max_voxels < 1:
        raise ValueError("我要求 max_voxels 至少为 1。")

    if origin is None or dims is None:
        origin, dims, voxel_size = _common_grid([mesh], voxel_size, max_voxels)
    origin = np.asarray(origin, dtype=np.float64)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    # 格中心而非格角点：中心不会恰好压在表面上，避免归属二义性。
    xs = origin[0] + (np.arange(nx, dtype=np.float64) + 0.5) * voxel_size
    ys = origin[1] + (np.arange(ny, dtype=np.float64) + 0.5) * voxel_size
    zs = origin[2] + (np.arange(nz, dtype=np.float64) + 0.5) * voxel_size
    occupancy = np.zeros((nx, ny, nz), dtype=bool)
    # 容差必须按**体素分辨率**取，不能按几何尺度取。两张件在空间中
    # 可能相隔很远（公共包围盒对角线很大），但切片平面是逐层独立的——
    # 用对角线百万分之一在大包围盒上会粗到把相邻的端点并成一个键，
    # 把一条闭合环切成开口折线。体素边长才是"这一层上两个端点算不算
    # 同一个点"的正确尺度，取它的一千分之一既接得上又不会误并。
    tolerance = max(voxel_size * 1e-3, 1e-12)
    if assume_watertight is None:
        assume_watertight = free_edge_count(mesh) == 0
    sections: list[CrossSection] = []
    for level, z in enumerate(zs):
        segments = _slice_mesh_at_z(mesh, float(z))
        if len(segments) == 0:
            sections.append(
                CrossSection(
                    z_mm=float(z),
                    loop_count=0,
                    open_loop_count=0,
                    area_mm2=0.0,
                    point_count=0,
                )
            )
            continue
        # 只在网格确实开口时才下掉退化线段。封闭网格上"很短"的线段
        # 往往是大三角面切出的真实微段，下掉会把环切碎（我会因此把
        # 一个好好的封闭网格误报成轮廓未闭合）。
        if not assume_watertight:
            lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
            segments = segments[lengths > tolerance]
        if len(segments) == 0:
            sections.append(
                CrossSection(
                    z_mm=float(z),
                    loop_count=0,
                    open_loop_count=0,
                    area_mm2=0.0,
                    point_count=0,
                )
            )
            continue
        loops = _chain_loops(segments, tolerance)
        edges = _loops_to_edges(loops)
        # 面积只对闭合环求和：开口折线围不出面积，硬算会给出一个
        # 看似合理、实际无意义的数。
        areas = [
            abs(_signed_area_xy(points))
            for points, closed in loops
            if closed and len(points) >= 3
        ]
        open_count = sum(1 for _points, closed in loops if not closed)
        sections.append(
            CrossSection(
                z_mm=float(z),
                loop_count=len(loops),
                open_loop_count=open_count,
                area_mm2=float(sum(areas)),
                point_count=int(sum(len(points) for points, _c in loops)),
            )
        )
        if len(edges) > 0:
            occupancy[:, :, level] = _fill_layer_scanline(edges, xs, ys)
    return occupancy, sections


def compare_meshes_voxel(
    reference: TriangleMesh,
    candidate: TriangleMesh,
    voxel_size: float = 1.0,
    max_voxels: int = 2_000_000,
) -> VoxelComparison:
    """我比对两张网格的体素占用，给出缺失体、多余体和截面差异。

    ``missing`` 是"参考件有、候选件没有"的格子，对应材料缺失；
    ``extra`` 是"候选件有、参考件没有"的格子，对应多出来的材料。
    两者都用**同一套公共网格**逐格相减得到，所以可以直接比体量。

    只要任一侧存在自由边，我就把状态降成 ``REVIEW_REQUIRED``：
    填充会从破口漏到外部，此时"占用"已经不代表实体内部，
    继续报体积等于给甲方一个看起来精确的错误数字。
    """

    if isinstance(voxel_size, bool) or not isinstance(voxel_size, Real):
        raise ValueError("我要求 voxel_size 是一个实数。")
    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("我要求 voxel_size 是一个有限的正数。")

    reference_free = free_edge_count(reference)
    candidate_free = free_edge_count(candidate)
    origin, dims, voxel_size = _common_grid(
        [reference, candidate], voxel_size, max_voxels
    )
    reference_occupancy, reference_sections = build_voxel_occupancy(
        reference,
        voxel_size=voxel_size,
        max_voxels=max_voxels,
        origin=origin,
        dims=dims,
        # 自由边数上面已经算过一次，这里传进去，避免逐层重复数。
        assume_watertight=reference_free == 0,
    )
    candidate_occupancy, candidate_sections = build_voxel_occupancy(
        candidate,
        voxel_size=voxel_size,
        max_voxels=max_voxels,
        origin=origin,
        dims=dims,
        assume_watertight=candidate_free == 0,
    )
    missing_mask = reference_occupancy & ~candidate_occupancy
    extra_mask = candidate_occupancy & ~reference_occupancy
    intersection = int(np.count_nonzero(reference_occupancy & candidate_occupancy))
    missing = int(np.count_nonzero(missing_mask))
    extra = int(np.count_nonzero(extra_mask))
    union = int(np.count_nonzero(reference_occupancy | candidate_occupancy))
    # 并集为 0 时 IoU 无定义；我给 0.0 而不是 nan，
    # 因为 nan 会顺着 JSON 传下去把下游的聚合整片污染成 nan。
    iou = float(intersection / union) if union > 0 else 0.0
    cell_volume = float(voxel_size) ** 3

    warnings: list[str] = []
    status = "OK"
    if reference_free > 0:
        status = "REVIEW_REQUIRED"
        warnings.append(
            f"我发现参考网格有 {reference_free} 条自由边，网格未封闭，"
            "体素填充会从破口漏到外部，体积证据不可信。"
        )
    if candidate_free > 0:
        status = "REVIEW_REQUIRED"
        warnings.append(
            f"我发现候选网格有 {candidate_free} 条自由边，网格未封闭，"
            "体素填充会从破口漏到外部，体积证据不可信。"
        )
    open_levels = sum(
        1 for item in reference_sections + candidate_sections if item.open_loop_count
    )
    if open_levels:
        status = "REVIEW_REQUIRED"
        warnings.append(
            f"我有 {open_levels} 个截面层出现未闭合轮廓，"
            "这些层的面积与占用不可信。"
        )

    # 截面层只保留参考件与候选件都有内容的层：两侧都是空层的高度
    # 对"哪里缺了"没有信息量，全列出来会把报告撑得没有重点。
    paired: list[CrossSection] = []
    for reference_item, candidate_item in zip(reference_sections, candidate_sections):
        if reference_item.area_mm2 == 0.0 and candidate_item.area_mm2 == 0.0:
            continue
        paired.append(reference_item)
    return VoxelComparison(
        voxel_size_mm=float(voxel_size),
        grid_dims=dims,
        grid_origin_mm=(
            float(origin[0]),
            float(origin[1]),
            float(origin[2]),
        ),
        reference_voxel_count=int(np.count_nonzero(reference_occupancy)),
        candidate_voxel_count=int(np.count_nonzero(candidate_occupancy)),
        intersection_count=intersection,
        missing_count=missing,
        extra_count=extra,
        union_count=union,
        iou=iou,
        missing_volume_mm3=float(missing) * cell_volume,
        extra_volume_mm3=float(extra) * cell_volume,
        reference_watertight=reference_free == 0,
        candidate_watertight=candidate_free == 0,
        reference_free_edge_count=reference_free,
        candidate_free_edge_count=candidate_free,
        status=status,
        warnings=warnings,
        cross_sections=paired,
    )
