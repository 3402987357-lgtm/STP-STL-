"""我把几何检测结果整理成点级或面级模型可以直接使用的特征。

算法原理：原始坐标不能直接喂给模型——同一个零件换个摆放位置、缩个尺寸，坐标全变，
模型学到的就变成"记住这个零件"而不是"识别缺失区域"。所以我只喂**归一化后的量**：

- 重心按包围盒归一化到 [0,1]，与零件绝对位置和尺寸无关；
- 法向本身已是单位向量，天然尺度无关；
- 面积按总面积归一成占比；
- 距离既给绝对值，也给"距离 / 阈值"的比值，让模型同时看到绝对偏差和相对超标程度。

标签取 ``distances >= threshold_mm``，与检测口径完全一致，保证训练目标和线上判定
用的是同一把尺子。输入距离必须是有限值：NaN 混进来会让整批标签静默变错。
"""

from __future__ import annotations

import numpy as np

from ..geometry.mesh import TriangleMesh


def build_face_features(
    reference: TriangleMesh,
    face_distances_mm: np.ndarray,
    threshold_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """我为每个参考三角面生成特征和可用于监督训练的标签。"""

    threshold_mm = float(threshold_mm)
    # 阈值必须先校验：非正数或 NaN 会让所有标签退化成全 0 或全 1。
    if not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError("我要求 threshold_mm 是大于 0 的有限数。")
    centroids = reference.triangle_centroids()
    normals = reference.triangle_normals
    areas = reference.triangle_areas
    bbox_size = np.maximum(reference.bbox_max - reference.bbox_min, 1e-12)
    # 重心与面积都做归一化，保证特征与零件绝对位置和尺寸无关。
    normalized_centroids = (centroids - reference.bbox_min) / bbox_size
    normalized_areas = areas / max(reference.total_area, 1e-12)
    distances = np.asarray(face_distances_mm, dtype=np.float64)
    if distances.ndim != 1 or len(distances) != len(reference.faces):
        raise ValueError("我要求 face_distances_mm 与参考三角面数量一致。")
    # 距离含 NaN/Inf 直接报错，避免静默生成错误标签。
    if not np.all(np.isfinite(distances)):
        raise ValueError("我要求 face_distances_mm 全部是有限数，不接受 NaN 或 Infinity。")
    features = np.column_stack(
        [
            normalized_centroids,
            normals,
            normalized_areas,
            distances,
            distances / threshold_mm,
        ]
    ).astype(np.float32)
    labels = (distances >= threshold_mm).astype(np.int64)
    return features, labels
