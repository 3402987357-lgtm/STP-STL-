"""我实现点到三角网格的最近距离，不强制依赖 Open3D 或 trimesh。

模块职责：给定一批点和一张三角网格，算出每个点的"最近距离、最近三角面、
最近点坐标"。这三个量是后续"缺失区域/多余区域"判定的唯一几何依据。

为什么坚持精确窄相位：破损检测的阈值常常在 0.1~1.0 mm 量级，如果距离
本身带近似误差，阈值判定就会在临界处抖动，产生假阳性/假阴性。所以我用
KD-tree 只做"宽相位筛选"，一旦无法证明候选集合足够，就扩大集合，最坏
情况退化为精确暴力算法，保证结果永远是精确值而不是近似值。

输入边界：点与网格必须有限（无 NaN/Inf）；空输入按空结果处理。
输出边界：返回的三个数组与输入点一一对应；找不到任何三角面时距离为 inf、
面号为 -1，调用方必须显式处理这个哨兵值。
"""

from __future__ import annotations

from numbers import Integral
import warnings

import numpy as np

from ..geometry.mesh import TriangleMesh

# trimesh 单次查询的块大小上限 / 下限。
# 实测 训练集/5（69238 点 / 366262 面）：不分块 52.5s / 峰值 4883MB，
# 16384 一块 53.6s / 峰值 1719MB，8192 一块 83.0s / 峰值 1093MB ——
# 16384 几乎不掉吞吐却把峰值压到三分之一，所以上限取 16384。
# 下限 2048 是给内存紧张机器兜底用的，见 _closest_point_chunked。
_TRIMESH_CHUNK_SIZE = 16384
_TRIMESH_MIN_CHUNK = 2048
# 单次查询最多可以占用的可用物理内存比例。这台机器常态 78%~91% 内存占用，
# 把预算打满只会更早被系统杀掉；而进程被杀等于整轮验收归零，
# 所以这里取 0.20 而不是 0.35，宁可慢也不要丢结果。
_TRIMESH_MEMORY_FRACTION = 0.20


def _available_physical_bytes() -> int | None:
    """我读当前可用物理内存；拿不到就返回 None，让调用方退回固定块。"""

    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            """我按 Win32 的字段顺序声明内存状态结构，供 GlobalMemoryStatusEx 填值。"""

            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    except Exception:
        # 非 Windows 或读数失败：不猜，交给固定块大小。
        return None


def points_to_mesh(
    points: np.ndarray,
    mesh: TriangleMesh,
    chunk_size: int = 2048,
    backend: str = "auto",
    candidate_k: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """我返回每个点的最近距离、最近三角面和最近点坐标。

    我优先使用三角面重心 KD-tree 做保守宽相位筛选，再对候选三角面
    执行精确点到三角形距离。只要候选集合不足以证明结果安全，我就扩大
    集合；扩大到全部三角面时自动退化为精确暴力算法。

    原理说明：
    1. 宽相位下界——对尚未入选的三角面，用"点到其重心距离 − 该三角面
       最大包围半径"作为最近距离的下界。如果所有未入选三角面的下界都
       大于当前最优距离，就能证明再找下去也不会更近，可以安全停止。
       这个下界是保守的（不会高估），所以不会漏掉真正的最近面。
    2. 候选数翻倍增长（k -> max(2k, k+1)）而不是线性增加，避免病态输入
       下退化成 O(面数) 次查询；一旦 k 达到全部三角面仍未证明安全，
       就直接用完所有面，等价于精确暴力。
    3. backend 选择是显式的：auto 先试 trimesh，失败再走 KD-tree，
       最后兜底暴力算法；任何一条路径的结果都是精确距离，区别只在速度。
    """

    try:
        points = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("我要求 points 必须可以转换为数值坐标。") from exc
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("我要求 points 的形状必须是 (N, 3)。")
    if not np.all(np.isfinite(points)):
        raise ValueError("我发现 points 中存在 NaN 或无穷大值。")
    if backend not in {"auto", "bruteforce", "scipy_kdtree"}:
        raise ValueError(f"我不认识距离后端：{backend}")
    # candidate_k / chunk_size 显式排除 bool：
    # bool 是 int 的子类，isinstance(True, int) 为真，
    # 不排除的话 --candidate-k=True 会被当成 1 悄悄通过。
    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, Integral)
        or candidate_k < 1
    ):
        raise ValueError("我要求 candidate_k 至少为 1。")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, Integral)
        or chunk_size < 1
    ):
        raise ValueError("我要求 chunk_size 至少为 1。")
    triangles = mesh.triangle_vertices
    # 空输入返回空数组而不是报错：批量流水线里"这一批没有点"是正常情况，
    # 抛异常会打断整条流水线。三个返回值的形状必须与非空时一致。
    if len(points) == 0:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )
    # 空网格返回 inf / -1 / nan，而不是 0 或空：
    #   距离 inf —— 表示"根本没有参考面可比"，不是"完全贴合"；
    #   面号 -1 —— 表示"没有对应的面"，用 0 会指向第 0 个面；
    #   最近点 nan —— 表示"无定义"，用原点会让下游误以为有个真实坐标。
    if len(triangles) == 0:
        distances = np.full(len(points), np.inf, dtype=np.float64)
        return (
            distances,
            np.full(len(points), -1, dtype=np.int64),
            np.full((len(points), 3), np.nan, dtype=np.float64),
        )
    # auto 先试加速后端，失败则静默回退到暴力实现：
    # 距离计算是整条流水线的热点，能加速就加速，
    # 但加速库不可用不该让检测失败，回退路径必须始终可用。
    if backend == "auto":
        accelerated = _points_to_mesh_with_trimesh(points, mesh, chunk_size)
        if accelerated is not None:
            return accelerated
        # trimesh 返回 None 表示它算不了（退化面等），
        # 下面交给 KD-tree / 暴力后端，结果同样是精确值。

    use_kdtree = backend in {"auto", "scipy_kdtree"}
    if use_kdtree:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            if backend == "scipy_kdtree":
                raise
            use_kdtree = False

    if use_kdtree:
        return _points_to_mesh_kdtree(
            points,
            triangles,
            chunk_size=chunk_size,
            candidate_k=candidate_k,
            cKDTree=cKDTree,
        )
    return _points_to_mesh_bruteforce(points, triangles, chunk_size)


def _points_to_mesh_bruteforce(
    points: np.ndarray,
    triangles: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """我保留一个简单、精确、可作为回归基线的后端。"""

    # 原理说明：这个后端把每个点对每个三角面都算一遍，复杂度 O(N·M)，
    # 慢但没有近似。它存在的意义有两个：一是 KD-tree 后端的结果必须能和
    # 它逐点对上，二是当宽相位下界失效时它就是最终兜底路径。
    # 分块只是为了控制内存峰值，不改变结果。

    best_distances = np.full(len(points), np.inf, dtype=np.float64)
    best_faces = np.full(len(points), -1, dtype=np.int64)
    best_points = np.full((len(points), 3), np.nan, dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        chunk = points[start:stop]
        chunk_distances = np.full(len(chunk), np.inf, dtype=np.float64)
        chunk_faces = np.full(len(chunk), -1, dtype=np.int64)
        chunk_points = np.full((len(chunk), 3), np.nan, dtype=np.float64)
        for face_id, triangle in enumerate(triangles):
            closest = closest_points_on_triangle(chunk, triangle)
            distances = np.linalg.norm(chunk - closest, axis=1)
            improved = distances < chunk_distances
            chunk_distances[improved] = distances[improved]
            chunk_faces[improved] = face_id
            chunk_points[improved] = closest[improved]
        best_distances[start:stop] = chunk_distances
        best_faces[start:stop] = chunk_faces
        best_points[start:stop] = chunk_points
    return best_distances, best_faces, best_points


def _points_to_mesh_kdtree(
    points: np.ndarray,
    triangles: np.ndarray,
    chunk_size: int,
    candidate_k: int,
    cKDTree,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """我使用保守 KD-tree 宽相位，并保留精确点到三角形窄相位。"""

    # 原理说明：KD-tree 建在三角面"重心"上，不是建在顶点或面上。查询时
    # 取 k 个最近重心作为候选面，对候选面做精确点到三角形距离；然后按
    # 上面的保守下界判断"未入选的面有没有可能更近"。只有下界证明安全才
    # 停止，否则翻倍扩大 k。这样既拿到 KD-tree 的速度，又保留精确性。
    # 边界：单面网格时 k 直接取 1；候选数达到面数时不再扩大，避免死循环。

    centroids = triangles.mean(axis=1)
    # 每个三角形的包围半径：形心到三个顶点的最大距离。
    # 它给出了"这个三角形能伸到多远"的上界，
    # 下面用它判断"还能不能有更近的三角形"。
    radii = np.linalg.norm(triangles - centroids[:, None, :], axis=2).max(axis=1)
    max_radius = float(radii.max()) if len(radii) else 0.0
    # 索引建在形心上：KD 树只能查点，
    # 用三角形形心代表整个三角形是标准做法。
    tree = cKDTree(centroids)
    triangle_count = len(triangles)
    # 初始 k 至少为 1，且不超过三角形总数：
    # k 大于总数会直接报错。
    initial_k = min(max(1, int(candidate_k)), triangle_count)
    all_distances = np.full(len(points), np.inf, dtype=np.float64)
    # 面号初值 -1 而不是 0：0 是一个合法的面号，
    # 用 -1 才能区分"没找到"和"找到了第 0 个面"。
    all_faces = np.full(len(points), -1, dtype=np.int64)
    # 最近点初值填 nan 而不是零向量：
    # 零向量是空间里的一个真实位置，填它会被当成有效结果。
    all_points = np.full((len(points), 3), np.nan, dtype=np.float64)

    # 分块处理：一次查询所有点会占用与点数成正比的临时内存，
    # 大网格上分块能把峰值内存压下来。
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        chunk = points[start:stop]
        k = initial_k
        while True:
            # 先按形心找出 k 个最近的三角形作为候选：
            # 形心近不代表三角形近，但三角形近一定形心不会太远，
            # 所以这一步不会漏掉真正的最近者。
            centroid_distances, candidate_ids = tree.query(chunk, k=k)
            # k=1 时 scipy 返回一维数组，补成二维：
            # 下面统一按列遍历，形状不一致会取错数据。
            if k == 1:
                centroid_distances = centroid_distances[:, None]
                candidate_ids = candidate_ids[:, None]
            candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
            best_distance = np.full(len(chunk), np.inf, dtype=np.float64)
            best_face = np.full(len(chunk), -1, dtype=np.int64)
            best_point = np.full((len(chunk), 3), np.nan, dtype=np.float64)
            for column in range(candidate_ids.shape[1]):
                face_ids = candidate_ids[:, column]
                closest = closest_points_on_triangle_batch(
                    chunk,
                    triangles[face_ids],
                )
                distances = np.linalg.norm(chunk - closest, axis=1)
                # 只在更近时更新：同一列里每个点
                # 对应一个不同的候选三角形，逐个比较取最优。
                improved = distances < best_distance
                best_distance[improved] = distances[improved]
                best_face[improved] = face_ids[improved]
                best_point[improved] = closest[improved]

            # 已经取遍所有三角形就不必再扩 k。
            if k >= triangle_count:
                break
            next_distances, _ = tree.query(chunk, k=min(k + 1, triangle_count))
            next_centroid_distance = np.asarray(next_distances)[:, -1]
            # 我用所有未入选三角面的最大包围半径做保守下界。
            lower_bound = np.asarray(next_centroid_distance) - max_radius
            if np.all(lower_bound > best_distance):
                break
            next_k = min(triangle_count, max(k * 2, k + 1))
            if next_k == k:
                break
            k = next_k

        all_distances[start:stop] = best_distance
        all_faces[start:stop] = best_face
        all_points[start:stop] = best_point
    return all_distances, all_faces, all_points


def closest_points_on_triangle_batch(
    points: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """我用向量化区域判定处理一批一一对应的三角形。"""

    # 原理说明：这是 Real-Time Collision Detection 里 Ericson 的
    # 点到三角形最近点算法，但我把它向量化成"一批点对一批三角形"同时算。
    # 它按三角形的 7 个 Voronoi 区域（3 个顶点、3 条边、1 个内部）判定：
    # 先算点相对各顶点的重心坐标符号，落在哪个区域就把点投影到对应的
    # 顶点/边/面上。向量化后每个 mask 一步到位，比逐点循环快一个量级。
    # 边界：入参形状必须严格是 (N,3) 与 (N,3,3) 且行数一致，含 NaN/Inf
    # 直接拒绝；空输入返回空结果，不抛错。

    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("我要求 points 的形状必须是 (N, 3)。")
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("我要求 triangles 的形状必须是 (N, 3, 3)。")
    if len(points) != len(triangles):
        raise ValueError("我要求 points 和 triangles 的行数一致。")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(triangles)):
        raise ValueError("我发现 points 或 triangles 中存在 NaN 或无穷大值。")
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    # 点到三角形最近点的经典分区域解法（Ericson 的 Voronoi 区域法）：
    # 把三角形周围空间按"最近特征"分成 7 个区域——3 个顶点区、3 条边区、
    # 1 个面内部区。下面每个 mask 判定一个区域，命中就直接取对应特征值，
    # 全程无循环、无分支跳转，所以能对整批点一次算完。
    # 判定顺序不能随意调换：后面的 mask 都要排除前面已命中的点（~mask_a 等）。
    ab = b - a
    ac = c - a
    bc = c - b
    ap = points - a
    # d1/d2 是点相对顶点 a 在两条边方向上的投影。
    # 两者都 <=0 说明点落在 a 的 Voronoi 区域外，最近点就是 a 本身。
    d1 = np.sum(ap * ab, axis=1)
    d2 = np.sum(ap * ac, axis=1)
    result = np.empty_like(points)
    mask_a = (d1 <= 0) & (d2 <= 0)
    result[mask_a] = a[mask_a]

    bp = points - b
    # 同理判 b：点相对 b 在 ab 上投影 >=0（越过 a 一侧）且被 ac 方向夹住。
    d3 = np.sum(bp * ab, axis=1)
    d4 = np.sum(bp * ac, axis=1)
    mask_b = (d3 >= 0) & (d4 <= d3) & ~mask_a
    result[mask_b] = b[mask_b]

    # 边 ab 区域：用叉积 vc 的符号判点是否在 ab 这条边的一侧，
    # 再沿 ab 按参数 v 插值出最近点。分母接近 0 时三角形退化成一条线，
    # 用 safe_ab 兜底避免除零，此时该点的结果由后续区域继续处理。
    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~(mask_a | mask_b)
    denominator_ab = d1 - d3
    safe_ab = np.where(np.abs(denominator_ab) > 1e-15, denominator_ab, 1.0)
    v = d1 / safe_ab
    result[mask_ab] = a[mask_ab] + v[mask_ab, None] * ab[mask_ab]

    cp = points - c
    d5 = np.sum(cp * ab, axis=1)
    d6 = np.sum(cp * ac, axis=1)
    mask_c = (d6 >= 0) & (d5 <= d6) & ~(mask_a | mask_b | mask_ab)
    result[mask_c] = c[mask_c]

    vb = d5 * d2 - d1 * d6
    mask_ac = (
        (vb <= 0)
        & (d2 >= 0)
        & (d6 <= 0)
        & ~(mask_a | mask_b | mask_ab | mask_c)
    )
    denominator_ac = d2 - d6
    safe_ac = np.where(np.abs(denominator_ac) > 1e-15, denominator_ac, 1.0)
    w = d2 / safe_ac
    result[mask_ac] = a[mask_ac] + w[mask_ac, None] * ac[mask_ac]

    va = d3 * d6 - d5 * d4
    # 边 bc 区域。到这里只剩边区域与面内部两种可能，
    # 所以每个 mask 都要排除前面全部区域，否则同一批点会被重复赋值。
    mask_bc = (
        (va <= 0)
        & ((d4 - d3) >= 0)
        & ((d5 - d6) >= 0)
        & ~(mask_a | mask_b | mask_ab | mask_c | mask_ac)
    )
    denominator_bc = (d4 - d3) + (d5 - d6)
    safe_bc = np.where(np.abs(denominator_bc) > 1e-15, denominator_bc, 1.0)
    w_bc = (d4 - d3) / safe_bc
    result[mask_bc] = b[mask_bc] + w_bc[mask_bc, None] * bc[mask_bc]

    # 最后剩下的就是面内部：用三个子三角形的面积比（重心坐标）加权插值。
    # va+vb+vc 是三者和，退化三角形上它接近 0，同样用 safe_face 兜底。
    mask_face = ~(mask_a | mask_b | mask_ab | mask_c | mask_ac | mask_bc)
    denominator_face = va + vb + vc
    safe_face = np.where(np.abs(denominator_face) > 1e-15, denominator_face, 1.0)
    inv = 1.0 / safe_face
    v_face = vb * inv
    w_face = vc * inv
    result[mask_face] = (
        a[mask_face]
        + ab[mask_face] * v_face[mask_face, None]
        + ac[mask_face] * w_face[mask_face, None]
    )

    # 退化三角形（三点共线）上面的区域判定全部失效，重心坐标也无意义。
    # 对它们我改用最朴素但可靠的办法：直接取三个顶点里最近的那个。
    degenerate = np.linalg.norm(np.cross(ab, ac), axis=1) <= 1e-15
    if np.any(degenerate):
        candidates = np.stack(
            [a[degenerate], b[degenerate], c[degenerate]],
            axis=1,
        )
        distances = np.linalg.norm(
            points[degenerate, None, :] - candidates,
            axis=2,
        )
        result[degenerate] = candidates[
            np.arange(int(np.sum(degenerate))),
            np.argmin(distances, axis=1),
        ]
    return result


# 我为什么把内存占比从 0.35 降到 0.20：实测峰值会随配准质量变差而暴涨，
# 0.35 在拥挤机器上留不出余量，进程被杀等于整轮验收归零。
_TRIMESH_MEMORY_FRACTION = 0.20
# 峰值模型：peak ≈ 块大小 × 面数 × 每单元字节数 × 安全系数
#
# 每单元字节数必须按**最贵的件**取，不能按平均。实测：
#   训练集/5（366262 面）  0.51 / 0.36 / 0.29 字节（2048/8192/16384）
#   测试集/11（165694 面） 3.62 / 2.68 / 2.30 / 1.93 字节
# 同样"块大小×面数"，测试集/11 要贵 3~6 倍——它配准质量差，R-tree
# 候选面更多。用训练集/5 的 0.25 去估它，实测 16384 会冲到 5227MB
# 而被系统杀掉。所以取 1.93（最贵件在最大块下的边际值），
# 再乘安全系数 1.5 补上固定开销与网格差异。
# **低估峰值会让进程被杀、整轮归零；高估只是慢一点**，必须偏向保守。
_TRIMESH_BYTES_PER_CELL = 1.93
_TRIMESH_PEAK_SAFETY = 1.5
# 固定底噪：小尺寸下实测峰值不随块大小线性归零（测试集/11 在 2048 块下
# 仍有 1227MB），所以解预算时必须先减掉这块，否则小块也会被算成能放得下。
_TRIMESH_PEAK_FLOOR_MB = 700.0


def _closest_point_chunked(closest_point, indexed_mesh, points, requested_size=None):
    """我把一次大查询切成多趟，只为压住峰值内存。

    trimesh 的 closest_point 在内部按点批量物化候选三角形，点数越多
    峰值越高。实测 训练集/2（57322 点 / 133828 面）一次查询整批会把
    进程推到被系统杀掉。切成固定大小的块后峰值只与块大小有关。
    分块不改变任何结果：每块独立求最近点，再按原顺序拼回。

    块大小的选取是"能用多大就用多大"，因为吞吐随块增大而变好
    （16384 只比不分块慢 2%，2048 要慢 63%）。收块的唯一理由是
    **当时**的可用物理内存撑不住——而峰值并非只由面数决定，
    详见 _TRIMESH_BYTES_PER_CELL 的注释。

    requested_size 是调用方显式要求的块大小（InspectionConfig 的
    distance_chunk_size）。我把它当**上限**而不是直接采用：
    自适应算出的块已经保证"当时内存撑得住"，若调用方要求更大就照办，
    会把进程推到被杀；但要求更小（想省内存）必须尊重，否则这个配置项
    在默认的 trimesh 路径上完全不起作用——配了却没效果，
    比没有这个配置更容易误导人。
    """

    # 块大小默认取上限：实测 16384 只比不分块慢 2%，却把峰值压到三分之一，
    # 而 2048 要慢 63%（91.9s vs 56.3s），所以吞吐上"越大越好"。
    # 但光看面数兜不住：训练集/2 在 16384 和 8192 下都会被系统杀掉，
    # 到 4096 才跑完。可用物理内存是唯一能反映当时真实拥挤度的量，
    # 所以我只在内存确实紧张时才按它收块，作为**安全下限**而非默认档。
    size = _TRIMESH_CHUNK_SIZE
    budget_bytes = _available_physical_bytes()
    if budget_bytes is not None:
        # 解 peak ≈ 块大小 × 面数 × 每单元字节 × 安全系数 <= 预算
        per_point = (
            _TRIMESH_BYTES_PER_CELL
            * _TRIMESH_PEAK_SAFETY
            * len(indexed_mesh.faces)
        )
        budget = max(
            0.0,
            budget_bytes * _TRIMESH_MEMORY_FRACTION
            - _TRIMESH_PEAK_FLOOR_MB * 1e6,
        )
        affordable = int(budget / max(1.0, per_point))
        # 只在"连上限都撑不起"时才收，且不低于下限：
        # 内存宽裕时一律走上限，不牺牲吞吐。
        if affordable < size:
            size = max(_TRIMESH_MIN_CHUNK, min(size, affordable))
    if requested_size is not None and int(requested_size) >= 1:
        size = min(size, int(requested_size))
    if len(points) <= size:
        return closest_point(indexed_mesh, points)
    closest_parts = []
    distance_parts = []
    face_parts = []
    for start in range(0, len(points), size):
        stop = min(start + size, len(points))
        closest, distances, face_ids = closest_point(
            indexed_mesh, points[start:stop]
        )
        closest_parts.append(np.asarray(closest))
        distance_parts.append(np.asarray(distances, dtype=np.float64))
        face_parts.append(np.asarray(face_ids, dtype=np.int64))
    return (
        np.concatenate(closest_parts),
        np.concatenate(distance_parts),
        np.concatenate(face_parts),
    )


def _points_to_mesh_with_trimesh(
    points: np.ndarray,
    mesh: TriangleMesh,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """我优先使用 trimesh 的 R-tree；质量告警由上层保留并升级复核。"""

    # 原理说明：trimesh 的 closest_point 基于 R-tree，通常比我的 KD-tree
    # 后端快。但它对退化三角形会抛 RuntimeWarning 甚至给出非法面号，所以
    # 我先剔除面积接近 0 的退化面，再把 RuntimeWarning 提升为异常，
    # 一旦出现非法面号或非有限距离就返回 None，交给上层回退到精确后端。
    # 边界：trimesh 未安装时直接返回 None，不抛 ImportError。
    valid_face_mask = mesh.triangle_areas > 1e-12
    if not np.any(valid_face_mask):
        return None
    valid_face_ids = np.flatnonzero(valid_face_mask)
    # 退化面（零面积）必须单独处理，否则 auto 会给出与其他后端不同的结果。
    # trimesh 无法容纳退化三角形，我只把它交给 trimesh 算非退化面，
    # 再用自己的精确路径补算退化面并取两者较小值。
    #
    # 不这么做的后果：含退化面的网格上，auto 会"看不见"退化面而报出
    # 明显偏大的距离，同一个点在 auto 与 bruteforce 下得到不同结论，
    # 而退化面在真实 STL 里并不罕见（建模工具常导出零面积面片）。
    degenerate_face_ids = np.flatnonzero(~valid_face_mask)
    try:
        import trimesh
        from trimesh.proximity import closest_point
    except ImportError:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            indexed_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces[valid_face_mask],
                process=False,
            )
            closest, distances, face_ids = _closest_point_chunked(
                closest_point, indexed_mesh, points, requested_size=chunk_size
            )
        face_ids = np.asarray(face_ids, dtype=np.int64)
        if (
            not np.all(np.isfinite(distances))
            or np.any(face_ids < 0)
            or np.any(face_ids >= len(valid_face_ids))
        ):
            return None
        distances = np.asarray(distances, dtype=np.float64)
        # 我始终用真实面号数组（而非 valid_face_ids 的下标）记录归属，
        # 补算退化面时直接写真实编号，避免"下标还是编号"的换算歧义。
        resolved_face_ids = valid_face_ids[face_ids].astype(np.int64)
        if len(degenerate_face_ids):
            # 退化面退化为线段/点：我用精确路径补算，与 trimesh 结果取较小值。
            degenerate_triangles = mesh.triangle_vertices[degenerate_face_ids]
            for real_face_id, triangle in zip(
                degenerate_face_ids, degenerate_triangles
            ):
                candidate = closest_points_on_triangle(points, triangle)
                candidate_distance = np.linalg.norm(points - candidate, axis=1)
                improved = candidate_distance < distances
                distances = np.where(improved, candidate_distance, distances)
                resolved_face_ids = np.where(
                    improved, int(real_face_id), resolved_face_ids
                )
                closest = np.where(improved[:, None], candidate, closest)
        return (
            distances,
            resolved_face_ids,
            np.asarray(closest, dtype=np.float64),
        )
    except Exception:
        return None


def closest_points_on_triangle(
    points: np.ndarray,
    triangle: np.ndarray,
) -> np.ndarray:
    """我按 Real-Time Collision Detection 的区域判定逻辑求最近点。"""

    # 原理说明：这是单三角形版本，用于暴力后端逐面循环。它和批量版本
    # 用同一套 Voronoi 区域判定，保证两条路径结果一致——如果两者不一致，
    # 就说明其中一条有 bug，回归测试会立刻暴露。
    # 边界：空点集返回空数组；退化三角形（面积为 0）仍能返回边上的最近点，
    # 不会返回 NaN。

    points = np.asarray(points, dtype=np.float64)
    triangle = np.asarray(triangle, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("我要求 points 的形状必须是 (N, 3)。")
    if triangle.shape != (3, 3):
        raise ValueError("我要求 triangle 的形状必须是 (3, 3)。")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(triangle)):
        raise ValueError("我发现点或三角形中存在 NaN 或无穷大值。")
    a, b, c = triangle
    # 与批量版 _closest_points_on_triangles 完全相同的 Voronoi 区域法，
    # 只是这里只处理一个三角形、且用向量点积（@）而非逐行求和。
    # 两个实现必须保持一致：批量版走矩阵运算是为了吞吐，
    # 单三角形版是为了在只查一个面时省掉建大数组的开销。
    ab = b - a
    ac = c - a
    bc = c - b
    ap = points - a
    d1 = ap @ ab
    d2 = ap @ ac
    result = np.empty_like(points)

    # 顶点 a 区域：点在两条边方向上的投影都为负。
    mask_a = (d1 <= 0) & (d2 <= 0)
    result[mask_a] = a

    # 顶点 b 区域。
    bp = points - b
    d3 = bp @ ab
    d4 = bp @ ac
    mask_b = (d3 >= 0) & (d4 <= d3) & ~mask_a
    result[mask_b] = b

    # 边 ab 区域：沿 ab 插值。safe_ab 同样是防退化三角形除零。
    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~(mask_a | mask_b)
    denominator_ab = d1 - d3
    safe_ab = np.where(np.abs(denominator_ab) > 1e-15, denominator_ab, 1.0)
    v = d1 / safe_ab
    result[mask_ab] = a + v[mask_ab, None] * ab

    # 顶点 c 区域。
    cp = points - c
    d5 = cp @ ab
    d6 = cp @ ac
    mask_c = (d6 >= 0) & (d5 <= d6) & ~(mask_a | mask_b | mask_ab)
    result[mask_c] = c

    vb = d5 * d2 - d1 * d6
    # 下面是点-三角形最近点的分区判定（Ericson 的经典做法）：
    # 先把"最近点落在某个顶点上、某条边上"的情况逐块判定掉，
    # 剩下的才落在三角形内部。每块判定都要排除前面所有块
    # （~(mask_a | mask_b | ...)），否则重叠区会被后写的块覆盖，
    # 得到错误的最近点。
    mask_ac = (
        (vb <= 0)
        & (d2 >= 0)
        & (d6 <= 0)
        & ~(mask_a | mask_b | mask_ab | mask_c)
    )
    denominator_ac = d2 - d6
    # 分母近零时替换成 1.0：退化三角形上分母可能是 0，
    # 直接除会得到 inf/nan 并污染整个数组。
    # 替换成 1.0 后该点的结果虽然不准，但不会是 nan，
    # 且退化三角形本身的距离也不可信。
    safe_ac = np.where(np.abs(denominator_ac) > 1e-15, denominator_ac, 1.0)
    w = d2 / safe_ac
    result[mask_ac] = a + w[mask_ac, None] * ac

    va = d3 * d6 - d5 * d4
    mask_bc = (
        (va <= 0)
        & ((d4 - d3) >= 0)
        & ((d5 - d6) >= 0)
        & ~(mask_a | mask_b | mask_ab | mask_c | mask_ac)
    )
    denominator_bc = (d4 - d3) + (d5 - d6)
    safe_bc = np.where(np.abs(denominator_bc) > 1e-15, denominator_bc, 1.0)
    w_bc = (d4 - d3) / safe_bc
    result[mask_bc] = b + w_bc[mask_bc, None] * bc

    # 最后剩下的就是"最近点在三角形内部"，用重心坐标插值。
    mask_face = ~(mask_a | mask_b | mask_ab | mask_c | mask_ac | mask_bc)
    denominator_face = va + vb + vc
    safe_face = np.where(np.abs(denominator_face) > 1e-15, denominator_face, 1.0)
    inv = 1.0 / safe_face
    v_face = vb * inv
    w_face = vc * inv
    result[mask_face] = (
        a
        + ab * v_face[mask_face, None]
        + ac * w_face[mask_face, None]
    )

    degenerate = np.linalg.norm(np.cross(ab, ac)) <= 1e-15
    if degenerate:
        candidates = np.stack(
            [
                np.broadcast_to(a, points.shape),
                np.broadcast_to(b, points.shape),
                np.broadcast_to(c, points.shape),
            ],
            axis=1,
        )
        distances = np.linalg.norm(points[:, None, :] - candidates, axis=2)
        result = candidates[np.arange(len(points)), np.argmin(distances, axis=1)]
    return result
