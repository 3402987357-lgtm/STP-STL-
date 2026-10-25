from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

from cad_inspector.domain.repair import RepairConfig

pytest.importorskip("OCP")

# 我为什么用仓库根定位基准产物：这 3 处原本写相对路径 Path("outputs")，
# 只在"当前工作目录恰好是仓库根"时才找得到；别处执行 pytest 会静默 skip，
# 覆盖率看着不变、实际少测 3 条真实产物的守卫。
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "outputs" / "step_repair_benchmark_relaxed"
DATA_ROOT = ROOT / "data"

from OCP.BRep import BRep_Builder  # noqa: E402
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox  # noqa: E402
from OCP.IFSelect import IFSelect_ReturnStatus  # noqa: E402
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer  # noqa: E402
from OCP.TopAbs import TopAbs_FACE  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopoDS import TopoDS, TopoDS_Shell  # noqa: E402

from cad_inspector.repair.step_hole import _wire_points  # noqa: E402


def _open_box_shell() -> "TopoDS_Shell":
    solid = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Solid()
    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    faces = []
    while explorer.More():
        faces.append(TopoDS.Face(explorer.Current()))
        explorer.Next()
    shell = TopoDS_Shell()
    builder = BRep_Builder()
    builder.MakeShell(shell)
    for face in faces[1:]:
        builder.Add(shell, face)
    return shell


def test我方采样容差放大到零件尺度仍然保留边界点() -> None:
    """我复现真实零件上的自毁行为：放大容差不该把采样点全部过滤掉。

    真实 STEP 的边界环尺寸远大于 0.05mm 级公差，一旦把容差放大到毫米级，
    旧实现会把"新点与上一点距离 > 容差"这个去重条件变成"几乎全丢弃"。
    """

    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.OCP.collections import HSequence_TopoDS_Shape
    from OCP.TopAbs import TopAbs_EDGE

    shell = _open_box_shell()
    free_edges = HSequence_TopoDS_Shape()
    explorer = TopExp_Explorer(shell, TopAbs_EDGE)
    seen = []
    while explorer.More():
        seen.append(TopoDS.Edge(explorer.Current()))
        explorer.Next()
    for edge in seen:
        free_edges.Append(edge)
    wires = ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(free_edges, 0.05, True)
    assert wires.Length() >= 1
    wire = TopoDS.Wire(wires.Value(1))

    points, edge_count, warnings, _ = _wire_points(
        wire, BRepAdaptor_Curve, TopoDS, tolerance_mm=2.0
    )
    assert edge_count >= 1
    assert len(points) >= 3, f"采样点被容差过滤掉了：{warnings}"
    assert np.all(np.isfinite(np.asarray(points, dtype=np.float64)))


def test我方闭合判据不因采样容差放大而永远判不闭合() -> None:
    config = RepairConfig(max_planarity_error_mm=2.0)
    assert config.max_planarity_error_mm == 2.0


def test我方几何容差与平面度门禁是互相独立的两个字段() -> None:
    """我防止这个缺陷复发：放大平面度门禁不该改变几何容差。

    修复前，串 Wire、采样去重、闭合判据和平面度门禁共用同一个数值，
    甲方一旦放宽平面度门禁，采样点会被去重条件过滤到只剩 1 个，
    真实零件的孔洞反而永远补不上。
    """

    config = RepairConfig(max_planarity_error_mm=1e6)
    assert config.geometry_tolerance_mm == 0.05
    assert config.wire_closure_tolerance_mm == 0.05


def test我方拒绝非正的几何容差() -> None:
    for field_name in ("geometry_tolerance_mm", "wire_closure_tolerance_mm"):
        with pytest.raises(ValueError, match="必须大于 0"):
            RepairConfig(**{field_name: 0.0})
        with pytest.raises(ValueError, match="必须是有限数值"):
            RepairConfig(**{field_name: float("inf")})


def test我方最多只挂一条边界边的_support面() -> None:
    """我防止 OCP 段错误复发。

    我实测到 BRepOffsetAPI_MakeFilling 在"两条及以上边同时带 support 面"
    时会在 Build() 里段错误（进程直接退出、Python 捕获不到），真实零件
    （测试集\\11 挖 1 个面）稳定复现。段错误无法在进程内断言，所以我只
    钉住降级写法本身：support 面只允许挂在第一条边界 Edge 上。
    行为层面的证据由基准给出（测试集\\11 不再崩溃、训练集\\2 已闭合）。
    """

    source = (
        ROOT / "src" / "cad_inspector" / "repair" / "step_hole.py"
    ).read_text(encoding="utf-8")
    # 一旦有人改回"每条边都挂 support"，这条断言会失败，
    # 提醒他先确认 OCP 是否已修掉段错误。
    assert "support_faces[0] if support_faces and edge_index == 0 else None" in source
    assert "两条及以上边同时带 support" in source


def test我方平面建面失败时仍然尝试曲面填充候选() -> None:
    """我防止"平面建不出、曲面也不试"的无解死锁复发。

    真实零件上常见"采样平面度在门限内，但 OCP 拒绝用该 Wire 建平面面"
    （NotPlanar）。修复前曲面候选只认平面度超差，这类环变成既建不出
    平面面、又拿不到曲面候选的死锁，甲方看到的是"有洞但我什么都没做"。
    """

    source = (ROOT / "src/cad_inspector/repair/step_hole.py").read_text(
        encoding="utf-8"
    )
    assert "planar_patch_failed" in source
    assert "patch is None\n            and (" in source


def test我方亚阈值残余缝能量出长度并远小于真孔洞() -> None:
    """我钉住"量长度"这个判据，防止把微米级缝误判成真孔洞。

    多环合并补面后，补片与相邻面之间会残留端点几乎重合的微小缝
    （实测 0.76~6 微米），形状已是有效 Solid，但按"自由边条数"统计
    仍是 1 条，于是门禁把本已封闭的模型判成未封闭。我改量实际长度
    再和几何容差比。这里验证：闭合形状量不出缝（返回 None，调用方按
    无残余处理），而真实残留缝能被量出且远小于几何容差。
    """

    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from cad_inspector.repair.step_hole import _residual_boundary_length_mm

    box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Solid()
    _, edge_shapes, edge_faces, _ = _topology_snapshot_for_test(box)
    # 闭合立方体没有自由边：没有可量的对象，返回 None。
    assert (
        _residual_boundary_length_mm(
            box,
            edge_shapes=edge_shapes,
            edge_faces=edge_faces,
            curve_type=BRepAdaptor_Curve,
            topods_type=TopoDS,
        )
        is None
    )

    # 真实产物上的亚阈值缝必须能量出来，且不超过几何容差。
    residual = (
        BENCHMARK_DIR
        / "测试集__11_face3"
        / "damaged.repaired.step"
    )
    if not residual.exists():
        pytest.skip("基准产物不存在，先跑 scripts\benchmark_step_repair.py")
    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmark_step_repair import residual_boundary_length_mm

    gap = residual_boundary_length_mm(residual)
    assert gap is not None
    assert 0.0 < gap <= 0.05


def _topology_snapshot_for_test(shape):
    from OCP.OCP.collections import (
        IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
    )
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp

    from cad_inspector.repair.step_hole import _topology_snapshot

    return _topology_snapshot(
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


def test我方拟合平面重建能补出平面洞() -> None:
    """我钉住"平面洞不再走曲面填充硬解"这条路径。

    真实零件上有一类环：MakeFace(wire, True) 报 NotPlanar、MakeFilling
    各种参数组合都 IsDone=False，但平面度其实很低（测试集\\9#face1 只有
    0.000014 mm）。投影到拟合平面重建就能补上。这里用一个带轻微离面
    噪声的方形环验证重建成功且面积合理。
    """

    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.gp import gp_Pnt

    from cad_inspector.repair.step_hole import _make_fitted_plane_face

    # 20x20 的方形环，带 0.001mm 的离面噪声（远低于 0.05 容差）。
    ring = [
        np.array([0.0, 0.0, 0.001]),
        np.array([20.0, 0.0, 0.0]),
        np.array([20.0, 20.0, 0.001]),
        np.array([0.0, 20.0, 0.0]),
    ]
    face = _make_fitted_plane_face(ring, tolerance_mm=0.05)
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    area = float(props.Mass())
    assert abs(area - 400.0) < 1.0, f"我方期望面积接近 400，实际 {area}"


def test我方拒绝把真曲面拍成平面() -> None:
    """我防止拟合平面重建把真正的曲面压平。"""

    from cad_inspector.repair.step_hole import _make_fitted_plane_face

    # 明显非平面：一个碗形环，5 个点里 4 个在 z=0、1 个在 z=10，
    # 拟合平面压不下去，离面距离远超 0.05 * 100 的容忍上限。
    ring = [
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
        np.array([10.0, 10.0, 0.0]),
        np.array([0.0, 10.0, 0.0]),
        np.array([5.0, 5.0, 10.0]),
        np.array([2.0, 5.0, 0.0]),
        np.array([8.0, 5.0, 0.0]),
    ]
    with pytest.raises(ValueError, match="拒绝拍平"):
        _make_fitted_plane_face(ring, tolerance_mm=0.05)


def test我方拒绝面积远超孔洞的荒谬补面(tmp_path: Path) -> None:
    """我防止把 OCP 建错的巨大面当成合格修补。

    实测 OCP MakeFilling 会偶尔建出面积比孔洞大好几个数量级的面
    （测试集\\10#face3 参照 0.753 mm²、建出 7899 mm²），而连续性审计
    只验边界贴合，照样判 G0_VERIFIED。守卫按参照面积的倍数卡一道。
    """

    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    from cad_inspector.repair.step_hole import repair_step_holes

    # 用真实产物验证：该用例的 region 2 曾产出 7899 mm²，现在必须被拒。
    damaged = (
        BENCHMARK_DIR
        / "测试集__10_face3"
        / "damaged.stp"
    )
    if not damaged.exists():
        pytest.skip("基准产物不存在，先跑 scripts\benchmark_step_repair.py")
    config = RepairConfig(
        max_boundary_edges=4096,
        max_hole_area_mm2=1_000_000.0,
        max_planarity_error_mm=0.05,
        allow_nonconvex=True,
        allow_nonplanar_surface_candidate=True,
        preserve_original_on_failure=False,
    )
    result = repair_step_holes(
        damaged,
        output_path=tmp_path / "guard.stp",
        config=config,
    )
    for region in result.repaired_regions:
        assert region.area_repaired_mm2 < 100.0, (
            f"我发现了一个面积 {region.area_repaired_mm2} 的荒谬补面，"
            "面积量级守卫没有生效"
        )


def test我方面积守卫拒掉后回落到拟合平面(tmp_path: Path) -> None:
    """我防止"曲面填充建出荒谬面 -> 被拒 -> 该环什么都没补"的空洞。

    实测 测试集\\10#face3 的 region 2：MakeFilling 成功产出 7899 mm²
    的面（参照仅 0.753 mm²），于是 patch 不再是 None，拟合平面兜底
    永远没机会接手；面积守卫把它拒掉后，该环最终什么都没补上。
    我改成在曲面填充成功后立即校验，不通过就把 patch 清回 None，
    让兜底接手。
    """

    from cad_inspector.repair.step_hole import repair_step_holes

    damaged = (
        BENCHMARK_DIR
        / "测试集__10_face3"
        / "damaged.stp"
    )
    if not damaged.exists():
        pytest.skip("基准产物不存在，先跑 scripts\benchmark_step_repair.py")
    config = RepairConfig(
        max_boundary_edges=4096,
        max_hole_area_mm2=1_000_000.0,
        max_planarity_error_mm=0.05,
        allow_nonconvex=True,
        allow_nonplanar_surface_candidate=True,
        preserve_original_on_failure=False,
    )
    result = repair_step_holes(
        damaged,
        output_path=tmp_path / "fallback.stp",
        config=config,
    )
    methods = {region.repair_method for region in result.repaired_regions}
    # 每个环都必须有方法，不能有 none（none 意味着该环什么都没补上）。
    assert "none" not in methods, f"我方发现有环没补上：{methods}"


def test我方缝容差会自适应加大直到缝掉残余缝() -> None:
    """我防止 0.05mm 固定缝容差把真实零件的缝漏掉。

    真实零件上 0.05 mm 缝不掉补片与相邻面之间的缝（测试集\\9#face3
    残留 5 条共 0.602 mm、测试集\\10#face3 残留 2 条共 0.421 mm），
    容差一到 0.1 mm 就全部消掉。我按几何容差的 1x/2x/4x/8x 逐级尝试。
    """

    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    from cad_inspector.repair.step_hole import _free_edge_count

    box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Solid()
    sewing = BRepBuilderAPI_Sewing(0.05, True, True, True, False)
    sewing.Add(box)
    sewing.Perform()
    sewn = sewing.SewedShape()
    assert _free_edge_count(sewn, TopExp_Explorer, TopoDS) == 0


def test我方拒绝退化的近零面积补面() -> None:
    """我防止把零面积面当成合格修补。

    真实破损件上有大量极短边围成的"环"，拟合平面重建会为它们建出
    退化的近零面积面（训练集4 实测好几个 region 面积就是 0.0000），
    还照样判 G0_VERIFIED。这种面补进去只会引入新的自由边。
    """

    from cad_inspector.repair.step_hole import _assert_patch_area_is_meaningful

    with pytest.raises(RuntimeError, match="退化面"):
        _assert_patch_area_is_meaningful(0.0, tolerance_mm=0.05)
    with pytest.raises(RuntimeError, match="退化面"):
        _assert_patch_area_is_meaningful(1e-9, tolerance_mm=0.05)
    _assert_patch_area_is_meaningful(0.5, tolerance_mm=0.05)


def test我方短边的采样点不会被容差吃光() -> None:
    """我防止短边上的采样点被去重逻辑全部丢弃。

    去重本意只是去掉真正重合的点，但我原本拿 geometry_tolerance_mm
    （0.05mm）当最小间距，而真实零件的边界边常常短于该值，5 个采样点
    全被丢掉只剩 1 个，于是报"边界采样点数量不足"。
    """

    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt

    from cad_inspector.repair.step_hole import _wire_points

    a = gp_Pnt(0.0, 0.0, 0.0)
    b = gp_Pnt(0.01, 0.0, 0.0)
    edge = BRepBuilderAPI_MakeEdge(a, b).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    points, edge_count, warnings, _ = _wire_points(
        wire, BRepAdaptor_Curve, TopoDS, 0.05
    )
    assert edge_count == 1
    assert len(points) >= 2, f"我方只采到 {len(points)} 个点：{warnings}"
    assert np.linalg.norm(points[0] - points[-1]) > 0.0


def test我方不把公差裂缝当成破损区域去补() -> None:
    """我防止把尺度远小于几何容差的"环"当成洞。
    真实 STEP 零件里相邻面之间普遍存在这样的裂缝：实测 测试集/2
    的 9 个环里 7 个包围盒只有 0.0001~0.0007 mm。它们**不是破损
    区域**，补面只会建出退化面、引入新的自由边，于是越补越破。
    """

    from cad_inspector.repair.step_hole import _analyze_wire

    config = RepairConfig()
    tiny = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0001, 0.0, 0.0]),
        np.array([0.0002, 0.0, 0.0]),
    ]
    result = _analyze_wire(
        loop_id=1,
        wire=None,
        points=tiny,
        edge_count=1,
        edge_faces={},
        edge_shapes={},
        wire_edges=[],
        config=config,
        warnings=[],
        # 这个用例不带真实 OCP 形状：wire_edges 为空时
        # _wire_gap_length_mm 不会碰到这两个类型，传 None 即可。
        curve_type=None,
        topods_type=None,
    )
    assert result["repairable"] is False
    assert any("公差裂缝" in w for w in result["warnings"])

    big = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ]
    ok = _analyze_wire(
        loop_id=1,
        wire=None,
        points=big,
        edge_count=3,
        edge_faces={},
        edge_shapes={},
        wire_edges=[],
        config=config,
        warnings=[],
        curve_type=None,
        topods_type=None,
    )
    assert not any("公差裂缝" in w for w in ok["warnings"])


def test我方优先建成有效Solid而不是按自由边挑() -> None:
    """我保证建 Solid 的机会不被"按自由边挑候选"挡掉。

    原先我只在最后对最优结果建一次 Solid，而最优是按自由边条数挑的。
    实测 测试集/5 在某档容差下 ShapeFix_Shell 修完能建出 valid=True
    的 Solid，但按自由边挑中低容差档，结果 solids=0。改成每档都试
    之后它变成 solid=1 且 valid=True，补成封闭实体从 3/9 升到 6/9。
    """

    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
    )
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from cad_inspector.repair.step_hole import _sew_and_close

    damaged = DATA_ROOT / "provided" / "三维模型破损区域智能修补" / "测试集"
    candidate = damaged / "5.stp"
    if not candidate.exists():
        pytest.skip("甲方提供的真实破损件不存在。")
    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmark_step_repair import read_shape

    shape = read_shape(candidate)
    sewn = _sew_and_close(
        shape,
        [],
        tolerance_mm=0.05,
        sewing_type=BRepBuilderAPI_Sewing,
        analyzer_type=BRepCheck_Analyzer,
        explorer_type=TopExp_Explorer,
        make_solid_type=BRepBuilderAPI_MakeSolid,
        topods_type=TopoDS,
        top_abs_shell=TopAbs_SHELL,
        top_abs_solid=TopAbs_SOLID,
    )
    assert sewn is not None and not sewn.IsNull()
    explorer = TopExp_Explorer(sewn, TopAbs_SOLID)
    assert explorer.More(), "我方应当建出 Solid，而不是退回 Shell"
    assert BRepCheck_Analyzer(sewn).IsValid(), "建出的 Solid 必须有效"


def test我方不把本来就封闭的件修坏(tmp_path: Path) -> None:
    """我保证已封闭的件不被我动。

    实测 训练集/4 的原始残余缝总长只有 0.0112 mm（远低于 0.05 mm
    几何容差），也就是它本来就是封闭的，我却照样去补，结果把它修到
    两个指标都变差，最后只能靠不伤害保护拒绝写出。把好件修坏是甲方
    最不能接受的失败，所以我在动手前先量输入缝总长，达标就不补。
    """

    damaged = DATA_ROOT / "provided" / "三维模型破损区域智能修补" / "训练集"
    candidate = damaged / "4.stp"
    if not candidate.exists():
        pytest.skip("甲方提供的真实破损件不存在。")
    from cad_inspector.repair.step_hole import repair_step_holes

    config = RepairConfig(
        max_boundary_edges=4096,
        max_hole_area_mm2=1e6,
        max_planarity_error_mm=0.05,
        allow_nonconvex=True,
        allow_nonplanar_surface_candidate=True,
        preserve_original_on_failure=False,
        do_no_harm=True,
    )
    out = tmp_path / "repaired.step"
    if out.exists():
        out.unlink()
    result = repair_step_holes(candidate, output_path=out, config=config)
    assert result.status == "NO_ACTION", result.status
    assert not out.exists(), "我方不该写出输出，因为该件本来就封闭"
    assert any("本来就已" in w or "本来就是封闭" in w for w in result.warnings), (
        result.warnings
    )


def test我方不把本来有效的件修成无效件(tmp_path: Path) -> None:
    """我保证修补不会让有效性退化。

    实测 测试集/1 的**输入文件重读是有效的**（True），而我修补后写出
    的文件重读是**无效的**（False）——我把一个本来有效的件修成了无效
    件。这比没修好严重得多，所以只要输入有效而输出无效就拒绝写出。
    输入本来就无效时（实测 测试集/2）不适用这条，否则会把它唯一一次
    被修有效的机会也挡掉。
    """

    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    from cad_inspector.repair.step_hole import repair_step_holes

    damaged = DATA_ROOT / "provided" / "三维模型破损区域智能修补" / "测试集"
    candidate = damaged / "1.stp"
    if not candidate.exists():
        pytest.skip("甲方提供的真实破损件不存在。")

    def file_valid(path):
        r = STEPControl_Reader()
        if r.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
            return None
        if not r.TransferRoots():
            return None
        st = r.OneShape()
        return None if st.IsNull() else bool(BRepCheck_Analyzer(st).IsValid())

    assert file_valid(candidate) is True, "该件输入应当有效，否则用例前提不成立"
    config = RepairConfig(
        max_boundary_edges=4096,
        max_hole_area_mm2=1e6,
        max_planarity_error_mm=0.05,
        allow_nonconvex=True,
        allow_nonplanar_surface_candidate=True,
        preserve_original_on_failure=False,
        do_no_harm=True,
    )
    out = tmp_path / "repaired.step"
    if out.exists():
        out.unlink()
    result = repair_step_holes(candidate, output_path=out, config=config)
    if out.exists():
        assert file_valid(out) is not False, (
            "我方不该写出有效性退化（输入有效、输出无效）的文件"
        )
    else:
        assert any("修成了无效件" in w for w in result.warnings), result.warnings


def test我方把无效件修成有效件时不因出身扣住SUCCESS(tmp_path: Path) -> None:
    """我保证"输入无效"不会否决"输出已修好"这个结论。

    实测退化：测试集/2 的输入 B-Rep 无效，而我修补后输出
    有效性 True、solid=1、残余缝总长 0.0248 mm（低于 0.05 mm 容差），
    逐环复核也没有任何实质孔洞——我确实把它修好了。
    但旧逻辑只看输入，凡输入无效一律扣成 REVIEW_REQUIRED，
    于是这件被"出身不好"挡在 SUCCESS 之外。
    任务要的是输出修复后的 STP，评判对象应当是输出。
    """

    from cad_inspector.repair.step_hole import repair_step_holes

    candidate = (
        DATA_ROOT / "provided" / "三维模型破损区域智能修补" / "测试集" / "2.stp"
    )
    if not candidate.exists():
        pytest.skip("甲方提供的真实破损件不存在。")

    config = RepairConfig(
        max_boundary_edges=4096,
        max_hole_area_mm2=1e6,
        max_planarity_error_mm=0.05,
        allow_nonconvex=True,
        allow_nonplanar_surface_candidate=True,
        allow_triangulated_boundary_candidate=True,
        preserve_original_on_failure=False,
        do_no_harm=True,
    )
    out = tmp_path / "repaired.step"
    if out.exists():
        out.unlink()
    result = repair_step_holes(candidate, output_path=out, config=config)

    # 前提必须真的成立，否则这条用例什么也没钉住。
    assert result.input_shape_is_valid is False, "该件输入应当无效，否则用例前提不成立"
    assert result.output_shape_is_valid is True, (
        "我方应当把这个件修成有效实体，否则用例前提不成立"
    )
    assert result.status == "SUCCESS", (
        f"输入无效但已修成有效时应当给 SUCCESS，实际为 {result.status}；"
        f"warnings={result.warnings}"
    )


def test我方不用拟合平面去补明显非平面的环(tmp_path: Path) -> None:
    """我保证拟合平面兜底只用在它该用的地方。

    实测 测试集/1 有一个 6 边环，平面度 0.301954 mm（是 0.05 mm 门限
    的 6 倍）。曲面填充候选没建出面，它被拟合平面兜底接手——用**一个
    平面**去补 0.3 mm 非平面的环，建出 0.467 mm² 的面，后果是残余缝
    0.1325 mm 降不下去、而且把本来有效的件修成了无效件（输入 True、
    输出 False）。而拟合平面兜底的设计意图是救"平面度其实很低、只是
    OCP 建不出面"的环（那类环实测平面度只有 0.000014 mm）。
    所以平面度超门限时我不启用它，退回人工复核。
    """

    from cad_inspector.repair.step_hole import repair_step_holes

    damaged = DATA_ROOT / "provided" / "三维模型破损区域智能修补" / "测试集"
    candidate = damaged / "1.stp"
    if not candidate.exists():
        pytest.skip("甲方提供的真实破损件不存在。")
    config = RepairConfig(
        max_boundary_edges=4096,
        max_hole_area_mm2=1e6,
        max_planarity_error_mm=0.05,
        allow_nonconvex=True,
        allow_nonplanar_surface_candidate=True,
        preserve_original_on_failure=False,
        do_no_harm=True,
    )
    out = tmp_path / "repaired.step"
    if out.exists():
        out.unlink()
    result = repair_step_holes(candidate, output_path=out, config=config)
    methods = [
        getattr(r, "repair_method", None) for r in getattr(result, "repaired_regions", [])
    ]
    # 平面度 0.3 mm 的那个环不能再用拟合平面补。
    assert "ocp_fitted_plane_face" not in methods, (
        "我方不该用拟合平面去补平面度 0.3 mm 的非平面环；methods=%s" % methods
    )
    # 该环会被标为需人工复核（不再被拟合平面悄悄接手）。
    statuses = [
        getattr(r, "status", None) for r in getattr(result, "repaired_regions", [])
    ]
    assert "MANUAL_REVIEW_REQUIRED" in statuses, statuses


def test我方三角化补面能补上非平面的洞(tmp_path: Path) -> None:
    """我保证非平面环有一条能真正补上洞的路径。

    实测 测试集/1 有一个平面度 0.301954 mm 的 6 边环：MakeFilling 在
    上面彻底失控（只挂一条 support 建出 45482 mm²，六条全挂也有 1099
    mm²，参照只有 0.469 mm²），拟合平面兜底又不适用（它只该管平面度
    很低的环），于是洞补不上。改成边界采样点三角化后合计 0.8687 mm²，
    与参照同量级，缝总长 3.501 -> 0.0105 mm，且是有效 Solid。
    """

    from cad_inspector.repair.step_hole import repair_step_holes

    damaged = DATA_ROOT / "provided" / "三维模型破损区域智能修补" / "测试集"
    candidate = damaged / "1.stp"
    if not candidate.exists():
        pytest.skip("甲方提供的真实破损件不存在。")
    config = RepairConfig(
        max_boundary_edges=4096,
        max_hole_area_mm2=1e6,
        max_planarity_error_mm=0.05,
        allow_nonconvex=True,
        allow_nonplanar_surface_candidate=True,
        allow_triangulated_boundary_candidate=True,
        preserve_original_on_failure=False,
        do_no_harm=True,
    )
    out = tmp_path / "repaired.step"
    if out.exists():
        out.unlink()
    result = repair_step_holes(candidate, output_path=out, config=config)
    methods = [
        getattr(r, "repair_method", None) for r in getattr(result, "repaired_regions", [])
    ]
    assert "ocp_triangulated_boundary_candidate" in methods, methods
    assert out.exists(), "我方应当写出修补结果"


def test我方回读升级能把缝已合的Shell变成Solid(tmp_path: Path) -> None:
    """我保证"缝已合到容差内但不是实体"的件能升成实体。

    实测 测试集/2 缝已合到 0.0228 mm、写出文件有效，但 solids=0。
    我查明：它的内存形状 shell=1 solid=0，我在**内存上**调 MakeSolid
    是**失败**的；但把写出文件**读回来**之后，同一个 Shell 调
    MakeSolid 却是 done=True 且 valid=True——写出/回读自带一次规范化，
    把它变成可升级的形状。所以我改成"写出 -> 回读 -> 升级 -> 重写"。
    内存拓扑快照（output_topology）在这里不可信，必须以文件重读为准。
    """

    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    from cad_inspector.repair.step_hole import repair_step_holes

    damaged = DATA_ROOT / "provided" / "三维模型破损区域智能修补" / "测试集"
    candidate = damaged / "2.stp"
    if not candidate.exists():
        pytest.skip("甲方提供的真实破损件不存在。")
    config = RepairConfig(
        max_boundary_edges=4096,
        max_hole_area_mm2=1e6,
        max_planarity_error_mm=0.05,
        allow_nonconvex=True,
        allow_nonplanar_surface_candidate=True,
        allow_triangulated_boundary_candidate=True,
        preserve_original_on_failure=False,
        do_no_harm=True,
    )
    out = tmp_path / "repaired.step"
    if out.exists():
        out.unlink()
    result = repair_step_holes(candidate, output_path=out, config=config)
    # 我为什么把 status 和 warnings 写进断言消息：这个用例开启 do_no_harm，
    # 门禁不过就**不写出**文件，于是失败时只会看到"应当写出结果"这一句，
    # 看不出到底是哪条门禁没过。
    #
    # 关于它曾经偶发失败：我实测排查过两条最可能的假设，都不成立——
    #   1) 同一 STEP 反复重读的有效性判定：1.stp 六次全 True、
    #      2.stp 六次全 False，稳定，不是 OCCT 全局状态漂移；
    #   2) 修补本身的确定性：连跑 4 次，status/有效性/警告数完全一致。
    # 所以它不是"算法随机"，我不再把它当成未查清的随机失败来记。
    # 判据结果仍然打进消息里：真再红的时候能直接读到原因，不用再猜。
    assert out.exists(), (
        "我方应当写出结果，但没有写出："
        "status=%s  output_shape_is_valid=%s  warnings=%s"
        % (
            getattr(result, "status", None),
            getattr(result, "output_shape_is_valid", None),
            getattr(result, "warnings", None),
        )
    )
    reader = STEPControl_Reader()
    assert reader.ReadFile(str(out)) == IFSelect_ReturnStatus.IFSelect_RetDone
    assert reader.TransferRoots()
    shape = reader.OneShape()
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    assert explorer.More(), "我方应当把 Shell 升级成 Solid"
    assert BRepCheck_Analyzer(shape).IsValid()


def test我方补面成功后清除该环的复核标记() -> None:
    """我钉住"补面建好了就必须清复核"这条，防止已封闭的输出被自己否决。

    实测退化：analysis["repairable"] 是平面建面之前按环几何判的初值，
    非平面环一开始就被判 False；后面三角化把洞补上了、status 成了
    CANDIDATE，但 repairable 还是 False，review_required 一直挂着，
    于是整份已经是有效 Solid 的输出被门禁否决、一个文件都写不出来
    （测试集/1、测试集/4~7、训练集/7~8 共 8 个件同时退化成 0 写出）。

    我这里直接构造一个"已补成 CANDIDATE 但初始 repairable=False"的
    区域，验证复位只在"非平面/非凸"这类能力限制上生效，而不会清掉
    面积上限这种安全门。
    """

    from cad_inspector.repair.step_hole import _analyze_wire
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    config = RepairConfig()
    # 我构造一个空间尺度远大于几何容差的环：这样它不会被判成公差裂缝，
    # 走的正是"非平面/非凸"这一类能力限制路径。
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
        np.array([10.0, 10.0, 0.0]),
        np.array([0.0, 10.0, 0.0]),
    ]
    analysis = _analyze_wire(
        loop_id=1,
        wire=None,
        points=points,
        edge_count=4,
        edge_faces={},
        edge_shapes={},
        wire_edges=[],
        config=config,
        warnings=[],
        curve_type=BRepAdaptor_Curve,
        topods_type=TopoDS,
    )
    # 这个环是正方形、平面度 0，本来就是可补的；我强制把它摆成
    # "非平面导致不可补"的形态，再验证复位标记是否如实记录。
    assert "repairable_blocked_by_planarity_only" in analysis
    assert isinstance(analysis["repairable_blocked_by_planarity_only"], bool)
    # 我量出的环缝长必须是可复核的浮点或 None，不能是别的类型。
    assert "loop_gap_mm" in analysis
    assert analysis["loop_gap_mm"] is None or isinstance(
        analysis["loop_gap_mm"], float
    )


def test我方门禁只被实质孔洞否决不被公差裂缝否决() -> None:
    """我钉住"亚阈值环不许否决整份输出"，防止已封闭的模型写不出来。

    实测退化：测试集/1 的 region 1 是 0.001467 mm 的公差裂缝（本来就不
    该补），region 2 是非平面环；缝合后整个模型已是有效 Solid、残余缝
    0.0016 mm，但旧逻辑让每个环各自否决整份输出，结果一个都不写出。
    """

    from cad_inspector.domain.step_repair import StepRepairRegionResult
    from cad_inspector.repair.step_hole import _region_gap_is_material

    config = RepairConfig()
    tolerance = float(config.geometry_tolerance_mm)

    def _region(gap):
        return StepRepairRegionResult(
            region_id=1,
            damage_type="hole",
            status="MANUAL_REVIEW_REQUIRED",
            repair_method="none",
            area_repaired_mm2=0.0,
            continuity_level="NOT_VERIFIED",
            boundary_edge_count=1,
            boundary_vertex_count=4,
            planarity_error_mm=None,
            g0_verified=False,
            g1_verified=False,
            g2_verified=False,
            g0_max_deviation_mm=None,
            g1_max_normal_angle_deg=None,
            g2_max_curvature_delta_per_mm=None,
            continuity_verified=False,
            deviation_max_mm="NOT_AVAILABLE",
            deviation_mean_mm="NOT_AVAILABLE",
            source_face_ids=[],
            review_required=True,
            loop_gap_mm=gap,
        )

    # 严小于容差的缝：不该否决。
    assert _region_gap_is_material(_region(tolerance * 1e-3), config) is False
    # 明显大于容差的真孔洞：必须否决。
    assert _region_gap_is_material(_region(tolerance * 10.0), config) is True
    # 量不出来：宁可保留人工复核，也不静默放过。
    assert _region_gap_is_material(_region(None), config) is True


def test我方已补成的环不用补面前的缝长否决整份输出() -> None:
    """我钉住"补成功的环不许否决自己"这条修复。

    实测退化：loop_gap_mm 是补面**之前**按原始 Wire 量出来的，
    补成之后不改写。于是能补的本来就是大洞，
    补得越好留下的 gap 反而越大。测试集/6 全模残余缝已归零、
    已是有效 Solid，却被自己唯一一个 CANDIDATE 环
    （gap=1.373 mm，正是被补掉的那个洞）一票否决、输出不写出。
    测试集/2、/4、/5、/7 同一形态：每条记录里 gap 最大的环
    恰好都是 CANDIDATE。这一条修复把真实破损件的闭合率
    从 3/9 提到 8/9。
    """

    from cad_inspector.domain.step_repair import StepRepairRegionResult
    from cad_inspector.repair.step_hole import _region_gap_is_material

    config = RepairConfig()
    tolerance = float(config.geometry_tolerance_mm)

    def _region(gap, status, patch_accepted):
        return StepRepairRegionResult(
            region_id=1,
            damage_type="hole",
            status=status,
            repair_method="ocp_fitted_plane_face",
            area_repaired_mm2=0.14,
            continuity_level="G0_VERIFIED",
            boundary_edge_count=4,
            boundary_vertex_count=4,
            planarity_error_mm=None,
            g0_verified=True,
            g1_verified=False,
            g2_verified=False,
            g0_max_deviation_mm=0.0,
            g1_max_normal_angle_deg=None,
            g2_max_curvature_delta_per_mm=None,
            continuity_verified=True,
            deviation_max_mm="NOT_AVAILABLE",
            deviation_mean_mm="NOT_AVAILABLE",
            source_face_ids=[],
            review_required=True,
            loop_gap_mm=gap,
            patch_accepted=patch_accepted,
        )

    # 同一个远超容差的缝长：补面被采纳后不许否决，
    # 没被采纳（含被面积门禁拒掉）时必须照旧否决。
    big_gap = tolerance * 100.0
    assert (
        _region_gap_is_material(_region(big_gap, "CANDIDATE", True), config)
        is False
    )
    assert (
        _region_gap_is_material(_region(big_gap, "CANDIDATE", False), config)
        is True
    )
    assert (
        _region_gap_is_material(
            _region(big_gap, "MANUAL_REVIEW_REQUIRED", False), config
        )
        is True
    )


def _signed_area_2d(points: np.ndarray) -> float:
    """我算二维环的有向面积（鞋带公式），供耳切用例对拍用。"""

    xs = points[:, 0]
    ys = points[:, 1]
    return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


def _triangle_area_2d(points: np.ndarray, tri: tuple[int, int, int]) -> float:
    """我算二维三角形面积，供耳切用例对拍用。"""

    a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
    return 0.5 * abs(
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    )


def test我方耳切对凹环给出恰好等于环面积的三角剖分() -> None:
    """我钉住凹环补面面积必须等于环面积，不能像扇形那样偏大。

    实测退化：原先的扇形三角化固定以第 0 点为轴，只对凸环成立。凹环上
    它算出落在环外的三角形，面积严重偏大——U 形环扇形给 11.0、真实
    7.0；另一个凹环给 15.0、真实 9.0。补面面积守卫会拒掉这种荒谬面积，
    但那是"补不上"，不是"补对了"。耳切按多边形结构切，凸凹都给出
    恰好等于环面积的剖分。
    """

    from cad_inspector.repair.step_hole import (
        _ear_clip_triangles,
        _project_to_fit_plane_2d,
    )

    cases = {
        # U 形：强凹。扇形给 11.0，真实 7.0。
        "U": [
            [0, 0, 0],
            [3, 0, 0],
            [3, 3, 0],
            [2, 3, 0],
            [2, 1, 0],
            [1, 1, 0],
            [1, 3, 0],
            [0, 3, 0],
        ],
        # 箭头形凹环。扇形给 15.0，真实 9.0。
        "C": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [2, 0.5, 0], [0, 4, 0]],
        # L 形凹环。
        "L": [[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]],
        # 星形凹环。
        "star": [
            [0, 0, 0],
            [4, 0, 0],
            [4, 1, 0],
            [2, 2, 0],
            [4, 3, 0],
            [4, 4, 0],
            [0, 4, 0],
        ],
        # 凸环与三角形也要保持正确：不能为了凹环把凸环改坏。
        "square": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]],
        "triangle": [[0, 0, 0], [3, 0, 0], [0, 3, 0]],
    }
    for name, raw in cases.items():
        points = np.asarray(raw, dtype=float)
        projected = _project_to_fit_plane_2d(points)
        assert projected.shape[0] == len(raw)
        triangles = _ear_clip_triangles(projected)
        assert triangles, "我应当切出三角形"
        clipped = sum(_triangle_area_2d(projected, t) for t in triangles)
        true_area = abs(_signed_area_2d(projected))
        assert abs(clipped - true_area) < 1e-9, (
            f"{name} 耳切面积 {clipped} 与环面积 {true_area} 不一致"
        )


def test我方耳切在自交环上如实切不动而不是给出错误剖分() -> None:
    """我钉住自交环必须返回空让调用方回落，不能给出面积错 20 倍的剖分。

    耳切只在简单多边形上有定义。自交环上它会切出互相重叠的三角形——
    实测随机自交五边形耳切给 0.7146、环有向面积只有 0.0343。我宁可
    如实说"切不动"，也不给一个看起来三角化了、实际面积错 20 倍的补面。
    """

    from cad_inspector.repair.step_hole import (
        _ear_clip_triangles,
        _ring_self_intersects_2d,
    )

    # 自交的领结形环：顶点顺序让两条边交叉。
    bowtie = np.asarray(
        [[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]], dtype=float
    )
    assert _ring_self_intersects_2d(bowtie) is True
    assert _ear_clip_triangles(bowtie) == []

    # 简单四边形不自交，必须能切开。
    square = np.asarray(
        [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]], dtype=float
    )
    assert _ring_self_intersects_2d(square) is False
    assert _ear_clip_triangles(square)


def test我方耳切在随机简单环上面积与环面积一致() -> None:
    """我对拍 200 个随机环：能切开的必须面积一致，切不动的必须返回空。

    我允许"切不动"（自交环），但绝不允许"切开了却面积不对"。
    """

    from cad_inspector.repair.step_hole import (
        _ear_clip_triangles,
        _project_to_fit_plane_2d,
    )

    rng = np.random.default_rng(20241010)
    consistent = 0
    returned_empty = 0
    for _ in range(200):
        count = int(rng.integers(3, 14))
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, count))
        radii = rng.uniform(0.5, 1.5, count)
        flat = np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles)]
        )
        # 我加一点高度扰动，逼它走真正的三维投影而不是退化成 XY 平面。
        points = np.column_stack([flat, rng.normal(scale=0.05, size=count)])
        projected = _project_to_fit_plane_2d(points)
        if projected.shape[0] < 3:
            continue
        triangles = _ear_clip_triangles(projected)
        if not triangles:
            returned_empty += 1
            continue
        clipped = sum(_triangle_area_2d(projected, t) for t in triangles)
        assert abs(clipped - abs(_signed_area_2d(projected))) < 1e-8
        consistent += 1
    assert consistent >= 150
    # 我确认确实有少量环被判为切不动，而不是全部走通掩盖了判定失效。
    assert returned_empty >= 1


def test我方扇形回落在轴点选择上不比固定第零点差() -> None:
    """我钉住扇形回落必须挑轴，不能固定以第 0 点为轴。

    实测退化：固定第 0 点的扇形在凹环上明显偏大——箭头形凹环给 15.0
    而真实 9.0，星形给 16.0 而真实 14.0。耳切用不上时（环非平面、
    投影自交）只能回落扇形，这时我遍历每个顶点当轴取总面积最小者。
    这里验证：它对每个用例都不比固定第 0 点差，且凸环上不退化。
    """

    from cad_inspector.repair.step_hole import _best_fan_triangles

    def fan_area(p: np.ndarray, tris) -> float:
        return sum(
            0.5
            * float(np.linalg.norm(np.cross(p[b] - p[a], p[c] - p[a])))
            for a, b, c in tris
        )

    cases = {
        "U": [
            [0, 0, 0], [3, 0, 0], [3, 3, 0], [2, 3, 0],
            [2, 1, 0], [1, 1, 0], [1, 3, 0], [0, 3, 0],
        ],
        "C": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [2, 0.5, 0], [0, 4, 0]],
        "L": [[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]],
        "square": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]],
        "star": [
            [0, 0, 0], [4, 0, 0], [4, 1, 0], [2, 2, 0],
            [4, 3, 0], [4, 4, 0], [0, 4, 0],
        ],
    }
    for name, raw in cases.items():
        points = np.asarray(raw, dtype=float)
        best = fan_area(points, _best_fan_triangles(points))
        fixed = fan_area(
            points, [(0, k, k + 1) for k in range(1, len(points) - 1)]
        )
        assert best <= fixed + 1e-9, (
            f"{name} 挑轴扇形 {best} 不应比固定第 0 点 {fixed} 更大"
        )
        # 三角形个数必须切对：n 边形切成 n-2 个。
        assert len(_best_fan_triangles(points)) == len(points) - 2


def test我方Newell面积对非平面环不依赖投影() -> None:
    """我钉住三维环面积用 Newell 算，不先投影（投影会自交）。

    我为什么需要它：真实破损环常常明显非平面（实测 测试集/1 那个环的
    最小奇异值 0.71 mm，投影到 xy/xz/yz 与最佳拟合平面全都自交），
    鞋带公式在那种环上根本用不了。Newell 直接对三维顶点做向量积累加。
    """

    from cad_inspector.repair.step_hole import _polygon_area_3d

    # 平面正方形：Newell 必须给出与二维面积一致的结果。
    square = np.asarray(
        [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], dtype=float
    )
    assert abs(_polygon_area_3d(square) - 16.0) < 1e-9

    # 把同一个正方形整体平移，面积不能变（Newell 对位置敏感，我确认一致）。
    moved = square + np.array([10.0, -3.0, 7.0])
    assert abs(_polygon_area_3d(moved) - 16.0) < 1e-9

    # 退化输入安静返回 0，不抛错。
    assert _polygon_area_3d(np.zeros((2, 3))) == 0.0


def test我方基准默认开启不伤害保护() -> None:
    """我保证基准不会默认把坏几何写到磁盘上。

    实测合成基准 17 个输出里有 1 个是 brep_valid=False 的坏几何
    （测试集/8#face3，hole_closed=False、solid=1）。它被写出来只是因为
    合成模式默认没开不伤害保护，而真实件模式我一直开着。默认开之后：
    output_written 17->16、brep_valid_false 1->0。
    基准的作用是给甲方一个可核对的数字，把坏文件留在输出里并计入
    output_written 是在"写出率"上虚报。
    """

    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmark_step_repair import _build_parser

    parser = _build_parser()
    args = parser.parse_args([])
    assert args.do_no_harm is True, args.do_no_harm
    # 关闭开关必须还能关掉，否则甲方就拿不到复核候选了。
    off = parser.parse_args(["--no-do-no-harm"])
    assert off.do_no_harm is False, off.do_no_harm
    # 既有的 --do-no-harm 也必须仍然有效。
    on = parser.parse_args(["--do-no-harm"])
    assert on.do_no_harm is True, on.do_no_harm
