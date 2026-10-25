from __future__ import annotations

import numpy as np

from cad_inspector.detection.missing_regions import detect_missing_regions
from cad_inspector.learning.evaluation import threshold_reached_face_ids
from cad_inspector.learning.synthetic_defects import remove_face_region
from cad_inspector.models import InspectionConfig, RegistrationResult

from test_core import cube_mesh


def _identity_registration() -> RegistrationResult:
    """我构造恒等配准：合成案例天然对齐，配准不该参与这份基线。"""

    return RegistrationResult(
        mode="none",
        rotation_matrix=np.eye(3).tolist(),
        translation_mm=[0.0, 0.0, 0.0],
        scale=1.0,
        mean_residual_mm=0.0,
        p95_residual_mm=0.0,
        quality="good",
    )


def _spaced_planar_grid(
    divisions: int = 16,
    spacing: float = 0.2,
) -> "object":
    """我造一块"三角形边长与 0.15mm 阈值同量级"的平面网格。

    为什么必须造这个尺度：面级判据默认只在**重心**取一个点。当一个三角形
    的重心离缺口边界不足 0.15mm、而它对面的顶点超过 0.15mm 时，
    "这个面算不算缺失"就完全取决于采样点落在面的哪里。0.2mm 边长正好
    把重心（约 0.1mm）与顶点（可达 0.28mm）分在阈值两侧。
    """

    side = divisions + 1
    axis = np.arange(side, dtype=np.float64) * spacing
    vx, vy = np.meshgrid(axis, axis)
    vertices = np.column_stack([vx.ravel(), vy.ravel(), np.zeros(vx.size)])
    faces: list[list[int]] = []
    for row in range(divisions):
        for col in range(divisions):
            a = row * side + col
            b, c = a + 1, a + side
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    from cad_inspector.geometry.mesh import TriangleMesh

    return TriangleMesh(vertices, np.asarray(faces, dtype=np.int64))


def test我方面级判据补上顶点采样后不再漏掉轮廓带() -> None:
    """我钉住两件事：轮廓带是真存在，以及收口它的开关真的生效。

    实测背景（2026-10-25，0.2mm 平面网格 / 阈值 0.15mm / 删 24 面）：
    关闭开关时检测器只报出"重心已达阈值"的那 8 个面；按任务书口径
    （面上存在 ≥0.15mm 的误差）真值是 15 个面。差的 7 个就在轮廓带上——
    它们的重心距离最大 0.149071mm，**低于**阈值，但顶点超过阈值。
    这些面按任务书口径属于缺失区域，漏掉它们不是"业务可接受"，
    而是**判据采样不足**。

    我把最小区域面积设为 0，让这条用例只测"面级判据"这一件事：
    否则 0.16mm² 的簇会被 0.5mm² 门限整块丢掉，开关的增量就看不见了。
    """

    mesh = _spaced_planar_grid()
    defect = remove_face_region(mesh, seed_face_id=7 * 17 + 7, face_count=24)
    base = dict(
        registration_mode="none",
        unit_policy="assume_mm",
        min_region_area_mm2=0.0,
        min_triangle_count=1,
    )
    config_off = InspectionConfig(**base)
    config_on = InspectionConfig(face_corner_check=True, **base)
    registration = _identity_registration()

    truth = threshold_reached_face_ids(
        mesh,
        defect.candidate_mesh,
        config=config_off,
        samples_per_triangle=9,
        face_ids=defect.removed_face_ids,
    )
    assert truth.reached_face_ids, "本用例的构造已失效：一个达阈值面都没有。"

    off_regions, _, _, _ = detect_missing_regions(
        mesh, defect.candidate_mesh, config_off, registration
    )
    on_regions, _, _, _ = detect_missing_regions(
        mesh, defect.candidate_mesh, config_on, registration
    )
    reported_off = {
        face_id for region in off_regions for face_id in region.reference_face_ids
    }
    reported_on = {
        face_id for region in on_regions for face_id in region.reference_face_ids
    }

    # 1) 关着的时候确实漏：任务书口径真值里有面没被报出。
    assert set(truth.reached_face_ids) - reported_off, (
        "本用例需要「关着开关就会漏」这个前提；当前没漏，构造已失效。"
    )
    # 2) 打开后严格变多（不是"换了一批"，而是"补上了"）。
    assert reported_off < reported_on, (
        f"开关打开后报出的面没有严格变多：关={len(reported_off)} "
        f"开={len(reported_on)}。"
    )
    # 3) 打开后，任务书口径判定的达阈值面必须全部被报出。
    missing = sorted(set(truth.reached_face_ids) - reported_on)
    assert not missing, (
        f"打开面级顶点采样后仍有 {len(missing)} 个达阈值面没被报出：{missing[:20]}。"
        "这说明开关没有覆盖到全部面，或顶点距离没有并入面级最大值。"
    )


def test我方面级顶点采样开关默认关闭且能从配置里打开() -> None:
    """我钉住这个开关的默认值与配置解析，防止它悄悄变成默认打开。

    为什么这条值得单独钉：它一旦默认打开，历史结果会全部变化
    （召回上升、耗时约翻倍），而"哪一次交付用的是哪套判据"就说不清了。
    开关是给甲方决定用的，默认必须是关闭。
    """

    assert InspectionConfig().face_corner_check is False
    assert InspectionConfig.from_dict({}).face_corner_check is False
    assert InspectionConfig.from_dict({"face_corner_check": True}).face_corner_check is True
    # JSON/YAML 里写成字符串 "false" 时必须是 False：直接取值会得到真值。
    assert InspectionConfig.from_dict({"face_corner_check": "false"}).face_corner_check is False


def test我方采样不完整时必须进入复核() -> None:
    mesh = cube_mesh()
    registration = RegistrationResult(
        mode="none",
        rotation_matrix=np.eye(3).tolist(),
        translation_mm=[0.0, 0.0, 0.0],
        scale=1.0,
        mean_residual_mm=0.0,
        p95_residual_mm=0.0,
        quality="good",
    )

    _, metrics, warnings, _ = detect_missing_regions(
        mesh,
        mesh,
        InspectionConfig(
            registration_mode="none",
            max_query_points=2,
            bidirectional_check=True,
            distance_backend="bruteforce",
        ),
        registration,
    )

    assert metrics["status"] == "review"
    assert metrics["sampling_review_required"] is True
    assert metrics["reference_unsampled_face_count"] > 0
    assert metrics["candidate_unsampled_face_count"] > 0
    assert any("没有覆盖全部三角面" in warning for warning in warnings)


def test我方采样预算先截断再生成点() -> None:
    mesh = cube_mesh()

    points, face_ids = mesh.sample_points(
        samples_per_triangle=100_000,
        max_points=2,
    )

    assert points.shape == (2, 3)
    assert face_ids.tolist() == [0, len(mesh.faces) - 1]
