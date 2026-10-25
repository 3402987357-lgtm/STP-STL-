"""我方基于网格的基础几何特征识别。

我先用面三角形法向、面积和重心建立可解释的上下平面候选。这个实现不能替代
OCC 的真实面类型和拓扑识别，但它可以先验证坐标方向、面积排序和后续分型参数
是否合理。

算法原理：按法向的 Z 分量绝对值把三角面分成水平、垂直、其他三类（取绝对值是因为法向
符号不代表上下）。水平面再按高度分成顶部带、底部带、中间带三个候选——中间带必须单独
归类而不是丢弃，否则模具内部的水平台阶面会凭空消失，导致区域覆盖不完整。

顶/底平面取各带里平均高度最高/最低的那个候选。两处告警是这套启发式的真实边界：没有
可靠的顶底候选说明坐标方向可能不对；顶底高度几乎相同说明这个模型不适合按 Z 轴分型。
"""

from __future__ import annotations

from math import cos, radians, sin

import numpy as np

from ..domain.features import MeshFeatureResult, PlaneCandidate
from ..geometry.mesh import TriangleMesh


def recognize_mesh_features(
    mesh: TriangleMesh,
    horizontal_tolerance_deg: float = 15.0,
) -> MeshFeatureResult:
    """我按法向方向和面积识别网格中的平面候选。

    容差上界为什么是 45°（不含）：水平判据是法向与 Z 轴夹角
    <= 容差，垂直判据是夹角 >= 90° - 容差。容差一旦到 45°，
    两个区间就互相碰上，同一个面会同时被算作水平和垂直，
    面积统计直接翻倍。上界取开区间就是为了让两者始终分离。

    为什么用 cos/sin 双阈值而不是只用一个角度：把法向 Z 分量
    取绝对值后，水平是 |nz| >= cos(容差)，垂直是 |nz| <= sin(容差)。
    这样两次判断都只是一次数组比较，不需要逐面算 arccos。

    零三角面返回 FAILED 而不是"零个平面"的成功结果：
    空网格是输入问题，不是"这个零件没有平面"。
    返回成功会让下游以为识别跑通了，
    把"没读到数据"误当成"分析结论"。
    """

    # 容差必须落在 (0, 45°)：超过 45° 后"水平"和"垂直"的判定会互相重叠。
    if not 0.0 < horizontal_tolerance_deg < 45.0:
        raise ValueError("我要求 horizontal_tolerance_deg 位于 (0, 45) 区间。")

    normals = mesh.triangle_normals
    areas = mesh.triangle_areas
    centers = mesh.triangle_centroids()
    # 没有任何三角面时返回 FAILED 并给出明确原因，不返回"零个平面"的成功结果。
    if len(normals) == 0:
        return MeshFeatureResult(
            status="FAILED",
            backend=str(mesh.metadata.get("format", "mesh")),
            bbox_min_xyz_mm=mesh.bbox_min.tolist(),
            bbox_max_xyz_mm=mesh.bbox_max.tolist(),
            horizontal_face_count=0,
            vertical_face_count=0,
            other_face_count=0,
            horizontal_area_mm2=0.0,
            vertical_area_mm2=0.0,
            top_plane=None,
            bottom_plane=None,
            warnings=["我没有找到可识别的三角面。"],
        )

    # 法向 Z 分量取绝对值后与 cos/sin 双阈值比较，符号不参与判断。
    cos_limit = cos(radians(horizontal_tolerance_deg))
    sin_limit = sin(radians(horizontal_tolerance_deg))
    z_normal = np.abs(normals[:, 2])
    horizontal_mask = z_normal >= cos_limit
    vertical_mask = z_normal <= sin_limit
    other_mask = ~(horizontal_mask | vertical_mask)

    horizontal = _group_horizontal_faces(
        centers,
        normals,
        areas,
        horizontal_mask,
        mesh.bbox_min[2],
        mesh.bbox_max[2],
    )
    # 顶/底取平均高度最高/最低的候选；没有候选时留 None 而不是编造。
    top = max(horizontal, key=lambda candidate: candidate.mean_z_mm, default=None)
    bottom = min(horizontal, key=lambda candidate: candidate.mean_z_mm, default=None)
    warnings = [
        "我当前按三角网格法向识别平面候选，不代表最终 CAD 面类型。",
        "我当前未识别台阶、R 角、孔洞和真实 B-Rep 邻接拓扑。",
    ]
    if top is None or bottom is None:
        warnings.append("我没有得到可靠的上下平面候选，需要人工确认坐标方向。")
    # 顶底高度没有差异，说明这个模型不能按 Z 轴建立分型方向。
    elif abs(top.mean_z_mm - bottom.mean_z_mm) <= 1e-9:
        warnings.append("我发现上下平面高度没有明显差异，不能直接建立分型方向。")

    return MeshFeatureResult(
        status="BASELINE",
        backend=str(mesh.metadata.get("format", "mesh")),
        bbox_min_xyz_mm=mesh.bbox_min.tolist(),
        bbox_max_xyz_mm=mesh.bbox_max.tolist(),
        horizontal_face_count=int(horizontal_mask.sum()),
        vertical_face_count=int(vertical_mask.sum()),
        other_face_count=int(other_mask.sum()),
        # 面积按掩码求和而不是按"面数 × 平均面积"：
        # 三角面大小差异很大，用面数估面积会在疏密不均的网格上显著失真。
        horizontal_area_mm2=float(areas[horizontal_mask].sum()),
        vertical_area_mm2=float(areas[vertical_mask].sum()),
        top_plane=top,
        bottom_plane=bottom,
        warnings=warnings,
        # 阈值与三角面数一并留档：面数决定了这些统计量的可信度，
        # 只有几十个面的模型上，面积占比的误差会大得多。
        diagnostics={
            "horizontal_tolerance_deg": horizontal_tolerance_deg,
            "triangle_count": int(len(mesh.faces)),
        },
    )


def _group_horizontal_faces(
    centers: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    horizontal_mask: np.ndarray,
    z_min: float,
    z_max: float,
) -> list[PlaneCandidate]:
    """我先以顶部、底部和中间层三个高度带形成可解释候选。

    为什么按高度带分组而不做聚类：模具分型面关心的是
    "顶 / 底 / 中间层"这几个可解释的位置，
    聚类出来的第 7 簇没人能说清它对应什么结构。
    三个固定带让每个候选都带 role，报告里能直接读。

    平均法向按面积加权而不是直接平均：大面片的法向
    比碎小面片更能代表这个带的真实朝向。
    直接平均会让一堆碎三角把整体法向带偏。

    法向归一化前检查长度：一个带里的法向若正好两两抵消，
    加权平均会接近零向量，除零会产出 nan 污染后续所有计算。
    """

    if not np.any(horizontal_mask):
        return []
    # 三个高度带的边界都用 span 的比例（各 10%）而不是绝对毫米值：
    # 这样大小件共用一套默认值，不必为每个模型单独配参数。
    # span 取 1e-12 下限，防止扁平模型上除零。
    span = max(float(z_max - z_min), 1e-12)
    bands: list[tuple[str, np.ndarray]] = [
        ("bottom", centers[:, 2] <= z_min + span * 0.1),
        ("top", centers[:, 2] >= z_max - span * 0.1),
        (
            "middle",
            (centers[:, 2] > z_min + span * 0.1)
            & (centers[:, 2] < z_max - span * 0.1),
        ),
    ]
    candidates: list[PlaneCandidate] = []
    # 每个带都要再与 horizontal_mask 取交集：只有水平面才可能是平面候选，
    # 单独按高度筛选会把倾斜面也圈进来。
    for role, band_mask in bands:
        mask = horizontal_mask & band_mask
        if not np.any(mask):
            continue
        weighted_area = float(areas[mask].sum())
        normal = np.average(normals[mask], axis=0, weights=areas[mask])
        normal_length = float(np.linalg.norm(normal))
        if normal_length > 1e-12:
            normal /= normal_length
        candidates.append(
            PlaneCandidate(
                face_count=int(mask.sum()),
                area_mm2=weighted_area,
                mean_z_mm=float(np.average(centers[mask, 2], weights=areas[mask])),
                min_z_mm=float(centers[mask, 2].min()),
                max_z_mm=float(centers[mask, 2].max()),
                normal_xyz=normal.tolist(),
                role=role,
            )
        )
    return candidates
