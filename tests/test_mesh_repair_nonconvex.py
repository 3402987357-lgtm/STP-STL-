from __future__ import annotations

import numpy as np

from cad_inspector.domain.repair import RepairConfig
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.repair.mesh_hole import (
    _ear_clip_faces,
    _ear_quality,
    _loop_reference_area_mm2,
)
from cad_inspector.repair import repair_mesh_holes
from cad_inspector.validation.mesh_quality import boundary_edges


# 我用 L 形多边形（面积 = 7 mm2）拉伸出的棱柱做基准：它的顶面是非凸平面环，
# 拿掉顶面后就得到一个"边界环非凸但完全平面"的孔洞，正好覆盖耳切分支。
_L_POLYGON = ((0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0), (0.0, 4.0))
_TOP_TRIANGLES = ((0, 1, 2), (0, 2, 3), (0, 3, 5), (3, 4, 5))
_L_AREA_MM2 = 7.0


def _l_prism_without_top() -> TriangleMesh:
    vertices = [(x, y, 2.0) for x, y in _L_POLYGON] + [
        (x, y, 0.0) for x, y in _L_POLYGON
    ]
    faces: list[tuple[int, int, int]] = []
    for first, second, third in _TOP_TRIANGLES:
        faces.append((third + 6, second + 6, first + 6))
    for index in range(len(_L_POLYGON)):
        following = (index + 1) % len(_L_POLYGON)
        faces.append((index, following + 6, following))
        faces.append((index, index + 6, following + 6))
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        name="l_prism_without_top",
    )


def test_我方用耳切法修补非凸平面孔洞() -> None:
    mesh = _l_prism_without_top()
    assert len(boundary_edges(mesh)) == 6

    repaired, result = repair_mesh_holes(
        mesh, RepairConfig(allow_nonconvex=True)
    )

    assert result.status == "success"
    assert boundary_edges(repaired) == set()
    assert [region.method for region in result.repaired_regions] == [
        "planar_ear_clipping"
    ]
    assert result.repaired_regions[0].added_vertex_count == 0
    assert result.repaired_regions[0].added_triangle_count == 4
    comparison = result.output_quality["comparison"]
    assert comparison["added_area_error_mm2"] < 1e-9
    assert abs(comparison["expected_added_area_mm2"] - _L_AREA_MM2) < 1e-9
    assert abs(comparison["area_delta_mm2"] - _L_AREA_MM2) < 1e-9


def test_我方默认已开启非凸孔洞修补() -> None:
    """我钉住新默认：非凸孔洞**默认就能补**。

    我为什么改这条：它原来钉的是"默认拒绝非凸"（`RepairConfig()` 下
    status == "review"）。我把 `allow_nonconvex` 的默认值从 False 改成
    **True** 之后（依据见 `domain/repair.py` 里该字段的注释：关闭时实测
    只有 33.3% 能补，打开后含大洞合计 95.2%），这条旧断言就不成立且
    **与实现矛盾**，必须跟着改，否则测试永远红。

    **注意这条钉的是默认行为，不是"实现必须放行非凸"**——"显式关闭时
    仍拒绝"由下一条 `test_我方显式关闭非凸时仍拒绝` 单独钉住，两条合起来
    才完整。
    """

    mesh = _l_prism_without_top()

    repaired, result = repair_mesh_holes(mesh, RepairConfig())

    assert result.status == "success"
    # 6 条边界边全补掉才是真补上，不是判据放松放过去的。
    # boundary_edges 返回的是 **set**，所以要拿 len 比，不能直接 == 0
    # （我第一版写成 == 0，拿 set 和 int 比，直接红，这里改回 len）。
    assert len(boundary_edges(repaired)) == 0
    assert result.repaired_regions[0].status == "repaired"
    assert result.repaired_regions[0].method == "planar_ear_clipping"


def test_我方显式关闭非凸时仍拒绝() -> None:
    """我钉住：显式 `allow_nonconvex=False` 时仍按旧行为拒绝非凸环。

    这条是上一条的**反向保险**——默认打开了，但开关**必须仍然起作用**。
    如果哪天有人把开关接丢了（传 False 也不拒绝），这条会红。
    """

    mesh = _l_prism_without_top()

    repaired, result = repair_mesh_holes(
        mesh, RepairConfig(allow_nonconvex=False)
    )

    assert result.status == "review"
    assert boundary_edges(repaired) == boundary_edges(mesh)
    assert result.repaired_regions[0].status == "manual_review_required"
    assert result.repaired_regions[0].method == "none"


def test_我方凸孔洞仍走原来的中心扇面() -> None:
    # 去掉顶面的正方体：边界环是凸四边形，应该继续走中心扇面而不是耳切。
    square = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
    vertices = [(x, y, 0.0) for x, y in square]
    vertices += [(x, y, 2.0) for x, y in square]
    faces = [(0, 2, 1), (0, 3, 2)]
    for index in range(4):
        following = (index + 1) % 4
        faces.append((index, following + 4, following))
        faces.append((index, index + 4, following + 4))
    mesh = TriangleMesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        name="square_prism_without_top",
    )
    assert len(boundary_edges(mesh)) == 4

    repaired, result = repair_mesh_holes(mesh, RepairConfig())

    assert result.status == "success"
    assert boundary_edges(repaired) == set()
    assert [region.method for region in result.repaired_regions] == [
        "planar_triangle_fan"
    ]


def test_我方面积基准取真实多边形面积而不是外接凸包() -> None:
    # L 形多边形真实面积是 7；它的凸包面积是 11.5。如果面积门禁拿凸包或
    # 越界扇面当基准，就会把"多补了 64% 面积"当成合格。
    mesh = _l_prism_without_top()
    loop = [0, 1, 2, 3, 4, 5]
    reference = _loop_reference_area_mm2(
        loop, mesh.vertices, np.asarray([0.0, 0.0, 1.0])
    )
    assert abs(reference - _L_AREA_MM2) < 1e-9


def test我方耳切把凹环三角化成n减二个三角形() -> None:
    """我保证凹环能三角化到底，不中途卡住返回 None。

    我为什么钉这条：耳切原来是"遇到第一个可削耳朵就削"，实测
    ``测试集/9#face128#2`` 那个环本身不自相交、平面误差为 0，却因为削的
    顺序把剩余多边形削成了自交形状，削到第 35 轮时剩下 6 点、两个凸耳
    的三角形都含对面的顶点，于是再也削不动、整块区域退回人工复核。我
    改成"收集全部候选、按质量排序削最优"之后它才补上。这条锁住的是
    **凹环一定要三角化出结果**这个契约，换任何削耳顺序都不能破坏它。
    """

    mesh = _l_prism_without_top()
    loop = [0, 1, 2, 3, 4, 5]
    faces = _ear_clip_faces(loop, mesh.vertices, np.asarray([0.0, 0.0, 1.0]))
    assert faces is not None, "凹环不应卡住：返回 None 就是削耳顺序又出问题"
    # n 个点的多边形必须正好剖成 n-2 个三角形，少一个就说明有顶点被丢下。
    assert len(faces) == len(loop) - 2
    # 剖出来的三角形只能由环上原有的顶点组成，不许凭空造点。
    assert {vertex for face in faces for vertex in face} <= set(loop)


def test我方耳朵质量分只排序不否决() -> None:
    """我保证质量分的语义正确：正三角形最高、退化三角形为 0。

    我为什么钉这条：这个分是我新加的，它**只用来在通过判据的候选里
    排序**、不用来否决任何耳朵。如果它的量纲错了（比如退化三角形反而
    拿高分），削耳顺序就会被带偏，而两道否决判据看不出这种偏差。
    """

    # 正三角形：归一化面积定义下应当正好是 1.0。
    unit = _ear_quality(
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([0.5, float(np.sqrt(3.0)) * 0.5]),
    )
    assert abs(unit - 1.0) < 1e-9
    # 三点共线（退化）：面积为零，质量必须是 0，不能拿到正分。
    assert (
        _ear_quality(
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([2.0, 0.0]),
        )
        == 0.0
    )
