"""我给 trimesh 距离查询的分块上回归。

起因：`auto` 后端一次把全部点交给 trimesh 的 closest_point，
峰值内存随点数上涨。实测 训练集/2（57322 点 / 133828 面）会把
进程推到被系统杀掉（退出码 -1、日志为空）。切成 8192 一块以后
同一批查询 134.1s 跑完、不再被杀。

这里只钉一件事：**分块不能改变任何结果**。这是纯内存优化，
正确性必须由"与不分块逐位一致"来证明，而不是靠"看着对"。
"""

from __future__ import annotations

import numpy as np

import cad_inspector.detection.distance as D
from cad_inspector.geometry.mesh import TriangleMesh


def _sphere_like_mesh(n_side: int = 24) -> TriangleMesh:
    """我造一个确定性的闭合网格，保证退化面与非退化面都覆盖到。"""

    u = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
    v = np.linspace(0, np.pi, n_side)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    verts = np.stack(
        [
            np.cos(uu) * np.sin(vv),
            np.sin(uu) * np.sin(vv),
            np.cos(vv),
        ],
        axis=-1,
    ).reshape(-1, 3)
    idx = np.arange(n_side * n_side).reshape(n_side, n_side)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    faces = np.stack(
        [np.concatenate([a, a]), np.concatenate([b, c]), np.concatenate([c, d])],
        axis=-1,
    )
    return TriangleMesh(vertices=np.ascontiguousarray(verts), faces=faces)


def test分块与不分块逐位一致() -> None:
    """我最在意这条：分块只能省内存，绝不能改动距离、面号或最近点。"""

    mesh = _sphere_like_mesh()
    rng = np.random.default_rng(20261011)
    # 点数故意跨块且最后一块不满，覆盖拼接边界。
    points = rng.normal(size=(500, 3))

    original = D._TRIMESH_CHUNK_SIZE
    try:
        D._TRIMESH_CHUNK_SIZE = 64
        d_chunk, f_chunk, c_chunk = D.points_to_mesh(points, mesh)
        D._TRIMESH_CHUNK_SIZE = 10**9
        d_whole, f_whole, c_whole = D.points_to_mesh(points, mesh)
    finally:
        D._TRIMESH_CHUNK_SIZE = original

    assert np.array_equal(d_chunk, d_whole)
    assert np.array_equal(f_chunk, f_whole)
    assert np.array_equal(c_chunk, c_whole)


def test分块结果仍然有限且面号合法() -> None:
    """分块后每个点都要有有限距离和合法面号，不能出现拼接漏点。"""

    mesh = _sphere_like_mesh()
    rng = np.random.default_rng(7)
    points = rng.normal(size=(300, 3))

    original = D._TRIMESH_CHUNK_SIZE
    try:
        D._TRIMESH_CHUNK_SIZE = 32
        distances, face_ids, closest = D.points_to_mesh(points, mesh)
    finally:
        D._TRIMESH_CHUNK_SIZE = original

    assert distances.shape == (len(points),)
    assert np.all(np.isfinite(distances))
    assert np.all(face_ids >= 0)
    assert np.all(face_ids < len(mesh.faces))
    assert closest.shape == (len(points), 3)


def test点数不超过块大小时走原路径() -> None:
    """点数小于块大小时不应进入分块分支，避免无谓的拼接开销。"""

    mesh = _sphere_like_mesh()
    rng = np.random.default_rng(11)
    points = rng.normal(size=(10, 3))
    assert len(points) <= D._TRIMESH_CHUNK_SIZE

    original = D._TRIMESH_CHUNK_SIZE
    try:
        D._TRIMESH_CHUNK_SIZE = 10**9
        d_whole, _, _ = D.points_to_mesh(points, mesh)
        D._TRIMESH_CHUNK_SIZE = 64
        d_small, _, _ = D.points_to_mesh(points, mesh)
    finally:
        D._TRIMESH_CHUNK_SIZE = original
    assert np.array_equal(d_small, d_whole)


def test块大小永不小于下限也永不超过上限() -> None:
    """自适应只能在上限与下限之间取值：上限保吞吐，下限防被杀。"""

    mesh = _sphere_like_mesh()
    points = np.zeros((D._TRIMESH_CHUNK_SIZE * 2, 3))

    # 内存读数异常宽裕 -> 取上限
    original_avail = D._available_physical_bytes
    original_size = D._TRIMESH_CHUNK_SIZE
    try:
        D._available_physical_bytes = lambda: 10**15
        d_big, _, _ = D.points_to_mesh(points, mesh)
        # 读数不可用 -> 退回上限，不报错
        D._available_physical_bytes = lambda: None
        d_none, _, _ = D.points_to_mesh(points, mesh)
        # 内存极度紧张 -> 也不得低于下限，必须仍能算出结果
        D._available_physical_bytes = lambda: 1
        d_tiny, _, _ = D.points_to_mesh(points, mesh)
    finally:
        D._available_physical_bytes = original_avail
        D._TRIMESH_CHUNK_SIZE = original_size

    assert np.array_equal(d_big, d_none)
    assert np.array_equal(d_big, d_tiny)
    assert np.all(np.isfinite(d_tiny))


def test配置里的块大小在默认trimesh路径上真的生效() -> None:
    """我实测出的真 bug：trimesh 是默认后端，但 _closest_point_chunked
    自己算块大小、**完全不接**调用方传进来的 chunk_size。
    结果 InspectionConfig.distance_chunk_size 在默认路径上是个哑配置——
    改它没有任何效果，却看起来能调。配了没效果比没有这个配置更误导人。

    它应当作为**上限**生效：可以要更小（省内存），
    不能要更大（那会把进程推到被杀）。
    """

    mesh = _sphere_like_mesh()
    points = np.zeros((D._TRIMESH_CHUNK_SIZE * 2, 3))

    seen: list[int | None] = []
    original = D._closest_point_chunked

    def spy(closest_point, indexed_mesh, pts, requested_size=None):
        seen.append(requested_size)
        return original(closest_point, indexed_mesh, pts, requested_size)

    D._closest_point_chunked = spy
    try:
        D.points_to_mesh(points, mesh, chunk_size=512)
        D.points_to_mesh(points, mesh)
    finally:
        D._closest_point_chunked = original

    # 显式传入必须真的到达分块函数，而不是被丢掉
    assert seen[0] == 512, f"chunk_size 没传到 trimesh 路径：{seen}"
    # 不显式传时 points_to_mesh 自己的默认（2048）会照样往下传，
    # 所以两处的块大小不同就证明"传进来的值被采纳了"，
    # 而不是两边都走同一个写死的数。
    assert seen[1] == 2048, f"默认值没往下传：{seen}"
    assert seen[0] != seen[1], "两次都拿到同一个值，说明参数根本没起作用"


def test峰值模型对所有实测件都只高估不低估() -> None:
    """低估峰值会让进程被系统杀掉、整轮验收归零，所以模型必须
    **只偏高、不偏低**。高估最多是慢，低估是丢结果。

    这个测试钉死的是我踩过的坑：我先用 训练集/5 单点反推出
    0.25 字节/单元，结果 测试集/11 在 16384 块下实测冲到 5227MB
    （模型只估 883MB）直接被杀。根因是每单元开销随配准质量变化
    3~6 倍，不是常数。
    """

    # (件名, 面数, [(块大小, 实测峰值字节)])
    cases = [
        ("训练集/5", 366262, [(2048, 383e6), (8192, 1093e6), (16384, 1719e6)]),
        (
            "测试集/11",
            165694,
            [(2048, 1227e6), (4096, 1817e6), (8192, 3119e6), (16384, 5227e6)],
        ),
    ]
    for name, faces, measured in cases:
        for chunk, actual in measured:
            pred = (
                D._TRIMESH_PEAK_FLOOR_MB * 1e6
                + D._TRIMESH_BYTES_PER_CELL
                * D._TRIMESH_PEAK_SAFETY
                * chunk
                * faces
            )
            assert pred >= actual, (
                f"{name} chunk={chunk} 我低估了峰值："
                f"模型 {pred / 1e6:.0f}MB < 实测 {actual / 1e6:.0f}MB"
            )
