"""我用可追踪的合成标签评估缺失区域基线，不伪造工业准确率。

原理说明：没有甲方标注集时，我不能凭空宣称准确率。能做的是"合成受控缺陷"：
拿一份完整的参考网格，按 Face 编号故意删掉一块连续区域，得到一个人造的
"破损件"，因为删了哪些面是我自己决定的，所以我有 100% 可信的标签。

用这份标签我可以算面级和面积级指标：
    precision = TP / (TP + FP)   报出来的区域里有多少是真的
    recall    = TP / (TP + FN)   真缺陷里有多少被报出来了
    IoU       = TP / (TP + FP + FN)
以及 case_recall（有多少个案例被检出）、area_* （按面积加权）、
boundary_iou（区域边界的重合度）和质心距离。

我刻意做了三件事防止把合成基线吹成工业指标：
1. 报告里写死 evaluation_scope = "synthetic_face_deletion_only"；
2. limitations 里明确写"不能代表甲方真实标注集"；
3. 达标判定要求 precision/recall/case_recall/area_*/boundary_iou 六项
   同时 >= target，任何一项没过就是 TARGET_NOT_PROVEN，而不是部分达标。

2026-10-25 接管续写：真值口径一分为二，两个都报，谁都不许顶替谁。
我为什么必须动这一块：任务书把缺失区域定义成"几何误差 >= 0.15mm 的区域"，
而原先的真值是"所有被删掉的面"。删一块连续面时，紧贴缺口边界的那一圈面
的误差天然低于阈值，它们按任务书口径**不属于缺失区域**，用它们当分母，
召回永远到不了 1。更要紧的是：我一度用**同一份生产距离数组**同时当预测
和真值，那样召回必然 1.0——那只能证明检测器是确定性的，不能证明它准。

所以现在：
- ``aggregate`` / ``status``：旧口径（全部被删面），保持原样，不动；
- ``spec_conformant`` / ``status_spec_conformant``：任务书口径，真值由
  :func:`threshold_reached_face_ids` 用**独立密度**（面内低差异点 + 三个顶点）
  重新量出来，不复用生产那份距离；
- ``contour_band_*``：把"密采样达阈值、生产采样没达阈值"的面单独量出来。
  这些面按定义在生产口径下误差低于阈值，是采样分辨率差异，不是漏检。
验收用哪一套口径必须由甲方书面确认，工程侧只负责把两套都算准、都摆出来。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np

from ..detection.missing_regions import detect_missing_regions
from ..detection.distance import points_to_mesh
from ..geometry.mesh import TriangleMesh
from ..models import InspectionConfig, RegistrationResult
from .synthetic_defects import add_face_region, remove_face_region


@dataclass(frozen=True)
class SpecTruth:
    """我记录"几何误差达阈值"这一口径的真值，以及它是怎么量出来的。

    我为什么把采样依据一起存下来：同一个面在不同采样密度下可能一边算
    "达阈值"、一边算"没达阈值"。只给一个到达与否的布尔值，复核者无法
    判断这个数是"几何事实"还是"采样密度的产物"。把密度、点数、阈值
    一起冻结在对象里，任何一个数字都能被复算。
    """

    # 被测量的面（真值候选集，通常是本次被删掉的那批面）。
    face_ids: list[int]
    # 每面在**独立密采样**下的最大误差（mm），下标与 face_ids 对齐。
    per_face_max_distance_mm: list[float]
    # 其中最大误差 >= 阈值的面：这些才按任务书口径属于缺失区域。
    reached_face_ids: list[int]
    samples_per_triangle: int
    sample_point_count: int
    threshold_mm: float

    def to_dict(self) -> dict[str, object]:
        """我把真值口径的采样依据转成字典，便于写进报告。"""

        return {
            "truth_scope": "faces_with_geometric_error_at_or_above_threshold",
            "truth_samples_per_triangle": self.samples_per_triangle,
            "truth_sample_point_count": self.sample_point_count,
            "truth_threshold_mm": self.threshold_mm,
            "truth_face_count": len(self.reached_face_ids),
            "measured_face_count": len(self.face_ids),
        }


def _per_face_max_distance(
    face_indices: np.ndarray,
    distances: np.ndarray,
    face_count: int,
) -> np.ndarray:
    """我把点级距离按面归约成"该面所有采样点的最大距离"。

    "最大"而不是"平均"是任务书判据决定的：缺失区域的定义是
    "误差 >= 阈值的区域"，只要这个面上有一处对不上，这个面就落在区域里。
    取平均会把信号摊薄，一块大面积正常、小面积缺失的网格会被摊成"没超阈值"。
    非有限距离（NaN/Inf）直接跳过，不参与 max：它们代表度量失败，
    不是"误差无限大"。
    """

    reduced = np.zeros(face_count, dtype=np.float64)
    finite = np.isfinite(distances)
    if np.any(finite):
        np.maximum.at(reduced, face_indices[finite], distances[finite])
    return reduced


def threshold_reached_face_ids(
    source_mesh: TriangleMesh,
    other_mesh: TriangleMesh,
    *,
    config: InspectionConfig,
    samples_per_triangle: int = 9,
    face_ids: Iterable[int] | None = None,
) -> SpecTruth:
    """我按任务书口径量"几何误差 >= 阈值 的面"，且用独立于生产采样的密度。

    原理说明：任务书把缺失区域定义成"几何误差 >= 0.15mm 的区域"。
    因此"某个面属不属于这个区域"是一个**该面上是否存在误差达阈值的点**
    的问题，而不是"该面的重心有没有达阈值"。生产默认 1 点/面且那一点
    恒取重心，于是紧贴缺口边界、只有对面顶点超阈值的那些面会被判成
    "没有误差"——这不是算法错，是判据的采样不足。

    所以这里做两件生产采样不做的事：
    1. 按调用方给的密度在面内取低差异点（默认 9 点/面，含重心）；
    2. **额外把每个面的三个顶点也当采样点**。顶点是面内离边界最远的
       极值点之一，缺了它，"面上的最大误差"会被系统性低估。

    我为什么不复用 `detect_missing_regions` 那份距离数组（2026-10-25 修正）：
    那是循环论证。预测是"生产距离 >= 阈值"，真值若也用同一份生产距离，
    召回必然 1.0，它证明的只是"检测器是确定性的"。要判准不准，
    真值必须来自一条**独立**的测量路径——这就是这个函数存在的理由。

    边界：面集合为空时返回空真值（而不是抛错），因为"没有真值面"
    本身就是评估需要如实上报的状态；越界的 face_id 由调用方先过滤。
    """

    if samples_per_triangle < 1:
        raise ValueError("我要求 samples_per_triangle 至少为 1。")
    if face_ids is None:
        target = np.arange(len(source_mesh.faces), dtype=np.int64)
    else:
        target = np.asarray(sorted({int(f) for f in face_ids}), dtype=np.int64)
    threshold = float(config.threshold_mm)
    if target.size == 0:
        return SpecTruth(
            face_ids=[],
            per_face_max_distance_mm=[],
            reached_face_ids=[],
            samples_per_triangle=int(samples_per_triangle),
            sample_point_count=0,
            threshold_mm=threshold,
        )
    # 面内点用网格自己的确定性采样（第 0 个样本恒为重心），
    # 这里**不设 max_points 上限**：真值测量被预算截断会静默改变真值集合，
    # 那是把"我没量全"伪装成"这些面没达阈值"。
    points, sampled_faces = source_mesh.sample_points(
        samples_per_triangle=int(samples_per_triangle),
        max_points=None,
    )
    keep = np.isin(sampled_faces, target)
    interior_points = points[keep]
    interior_faces = sampled_faces[keep]
    # 顶点样本：面内点到边界最远的候选位置之一，生产采样永远取不到它们。
    triangles = source_mesh.triangle_vertices[target]
    vertex_points = triangles.reshape(-1, 3)
    vertex_faces = np.repeat(target, 3)
    if interior_points.size:
        all_points = np.concatenate([interior_points, vertex_points], axis=0)
        all_faces = np.concatenate([interior_faces, vertex_faces], axis=0)
    else:
        all_points = vertex_points
        all_faces = vertex_faces
    distances, _, _ = points_to_mesh(
        all_points,
        other_mesh,
        chunk_size=config.distance_chunk_size,
        backend=config.distance_backend,
        candidate_k=config.distance_candidate_k,
    )
    # 把全局面号映射成 target 内的位置下标，再做面级最大值归约。
    positions = np.searchsorted(target, all_faces)
    reduced = _per_face_max_distance(positions, distances, int(target.size))
    per_face = [float(value) for value in reduced]
    reached = [
        int(target[index])
        for index in range(int(target.size))
        if per_face[index] >= threshold
    ]
    return SpecTruth(
        face_ids=[int(value) for value in target],
        per_face_max_distance_mm=per_face,
        reached_face_ids=reached,
        samples_per_triangle=int(samples_per_triangle),
        sample_point_count=int(all_points.shape[0]),
        threshold_mm=threshold,
    )


@dataclass(frozen=True)
class DetectionMetrics:
    """我记录一个合成案例的面级检测指标。"""

    # 案例标识、真值面数与预测面数，用来核对两个集合的规模。
    case_id: str
    ground_truth_face_count: int
    predicted_face_count: int
    # 混淆矩阵的四个基本量（TN 对面级指标没有意义，所以不记）。
    true_positive: int
    false_positive: int
    false_negative: int
    # 面级指标。
    precision: float
    recall: float
    f1: float
    iou: float
    # 只要真值集合和预测集合有交集就算这个案例被检出。
    region_detected: bool
    # PASS 要求零误报、零漏报，且真值标签本身没有非法面号。
    status: str
    # 面积级指标：面积加权能降低"碎片面多但面积小"对结论的影响。
    ground_truth_area_mm2: float = 0.0
    predicted_area_mm2: float = 0.0
    true_positive_area_mm2: float = 0.0
    area_precision: float = 1.0
    area_recall: float = 1.0
    area_iou: float = 1.0
    # 质心距离和边界 IoU：回答"报出来的位置对不对"，而不只是"报没报"。
    centroid_distance_mm: float | None = None
    boundary_iou: float = 1.0
    # 真值里越界或负数的面号数量；只要不为 0，这个案例就不能算 PASS。
    invalid_ground_truth_face_count: int = 0
    # 真值区域相对于阈值的**最大可检出深度**（mm）。
    # 我为什么要单独记它：如果删掉的那片区域最深处都不足 threshold_mm，
    # 那么"没检出"是物理上必然的，不是检测器的失误——
    # 把这种案例算进召回率，会把"我造的缺陷太小"误报成"检测器漏检"。
    max_achievable_depth_mm: float | None = None
    # 该案例是否**在物理上可检出**（最大深度达到阈值）。
    # False 表示这是一条"阈值不可达"案例，应当从准确率统计里排除，
    # 而不是当成漏检测掉。
    detectable: bool = True
    # 缺陷类型必须落到每个案例上：汇总层的 evaluation_scope 只说明
    # 这一批整体是哪类，而案例被单独取出来看时（比如导出 CSV 逐行核对）
    # 就没有上下文了，缺料/多料的真值口径完全不同，看不出会被误读。
    defect_type: str = "missing"
    # 检出后又被 min_region_area_mm2 滤掉的面积（mm²）。
    # 我为什么把它单列（2026-10-16 实测）：它是"漏检"的第三种成因——
    # 不是检测器没检出，而是**检出了、又被小区域阈值丢掉**。
    # 实测例子：训练集/8、seed=1200、删 400 面（真值 0.898 mm²），
    # 检测器报出 0.3804 mm²，但因不足 0.5 mm² 被丢，最终表现为漏检；
    # 把 min_region_area_mm2 设为 0 后同一案例立刻被检出（IoU 0.445）。
    # 不区分这两者，甲方看到的就只是"漏了一个"，无法判断该调阈值还是改算法。
    filtered_area_mm2: float = 0.0
    # ---- 2026-10-25 接管续写：任务书口径（误差达阈值的面）与轮廓带 ----
    # 真值口径的采样密度与点数：没有它们，"达阈值面数"无法复算。
    spec_truth_samples_per_triangle: int = 0
    spec_truth_sample_point_count: int = 0
    # 任务书口径的真值面数与混淆矩阵三件套。
    spec_truth_face_count: int = 0
    spec_true_positive: int = 0
    spec_false_positive: int = 0
    spec_false_negative: int = 0
    spec_precision: float = 1.0
    spec_recall: float = 1.0
    spec_iou: float = 1.0
    spec_region_detected: bool = False
    # 轮廓带：被独立密采样判为达阈值、但生产采样下每面误差 < 阈值的面。
    # 按定义它们不可能被生产判据报出，所以它是"采样分辨率差异"，
    # 不是"漏检"。把这条单列，是为了不让分辨率问题被算进算法误差。
    contour_band_face_count: int = 0
    contour_band_production_max_distance_mm: float | None = None
    # 任务书口径的漏检里，有多少是"检出了但被 min_region_area_mm2 丢掉"的。
    # 把它和轮廓带分开，甲方才知道该调阈值还是该提采样密度。
    spec_false_negative_area_filtered_face_count: int = 0

    def to_dict(self) -> dict[str, object]:
        """我把这份评估结果转成字典，键名与 CSV 表头一致。"""

        return asdict(self)


def evaluate_synthetic_case(
    reference: TriangleMesh,
    candidate: TriangleMesh,
    removed_face_ids: Iterable[int],
    *,
    case_id: str = "synthetic",
    config: InspectionConfig | None = None,
    defect_type: str = "missing",
    truth_samples_per_triangle: int = 9,
) -> DetectionMetrics:
    """我按已知删除面标签评估一次缺失检测。

    我默认关闭配准并假定单位是 mm：合成案例是我自己删面生成的，
    参考件和破损件天然对齐，任何配准误差都会污染这份"应该完全对齐"的基线。

    defect_type 决定我评估哪一类，两类真值的**坐标系不同**，不能混用：

    - ``missing``：真值是**参考网格**上的面下标（哪些面被删了）；
    - ``extra``：真值是**候选网格**上的面下标（哪些面是多出来的）。

    我为什么必须支持 extra：生产管线对每件真实件都会报 extra 区域，
    而原先只评估 missing，"多余材料"这一类结论的准确率一直是零证据。
    """

    if defect_type not in ("missing", "extra"):
        raise ValueError(
            f"我只支持 missing 与 extra 两类缺陷，当前收到 {defect_type!r}。"
        )
    config = config or InspectionConfig(
        registration_mode="none",
        unit_policy="assume_mm",
    )
    # 真值所在的网格：missing 看参考网格，extra 看候选网格。
    # 上界与"取哪个网格"绑在一起，写错就会把越界判据用错网格的面数去比。
    truth_mesh = reference if defect_type == "missing" else candidate
    # 真值先转成 int 再过滤越界值。我把越界的数量单独记下来，
    # 而不是直接报错——因为"标签里有脏数据"本身就是要暴露的信息。
    raw_ground_truth = [int(face_id) for face_id in removed_face_ids]
    ground_truth = {
        face_id
        for face_id in raw_ground_truth
        if 0 <= face_id < len(truth_mesh.faces)
    }
    invalid_ground_truth_count = sum(
        face_id < 0 or face_id >= len(truth_mesh.faces)
        for face_id in raw_ground_truth
    )
    registration = _identity_registration()
    # 直接复用生产用的检测函数，不另写一套评估专用检测器：
    # 如果评估用另一套逻辑，它测出来的分数就不能代表真实流水线。
    # 第四个返回值是参考面的生产面级距离数组——旧口径只用它当预测依据，
    # 新口径还要用它区分"轮廓带"与"被面积门限丢掉"，所以必须接住。
    regions, global_metrics, _, production_face_distances = detect_missing_regions(
        reference=reference,
        candidate=candidate,
        config=config,
        registration=registration,
    )
    # 我按 defect_type 取对应的那一侧：缺失看被滤掉的缺失面积，
    # 多料看被滤掉的多料面积。取错侧会得到一个恒为 0 的字段，
    # 看起来"没有被滤掉"，实际是把成因藏起来了。
    filtered_area_mm2 = float(
        global_metrics.get(
            "ignored_missing_area_mm2"
            if defect_type == "missing"
            else "ignored_extra_area_mm2",
            0.0,
        )
    )
    predicted = {
        int(face_id)
        for region in regions
        if region.defect_type == defect_type
        for face_id in (
            region.reference_face_ids
            if defect_type == "missing"
            else region.candidate_face_ids
        )
    }
    true_positive = len(ground_truth & predicted)
    false_positive = len(predicted - ground_truth)
    false_negative = len(ground_truth - predicted)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2.0 * precision * recall, precision + recall)
    iou = _ratio(true_positive, true_positive + false_positive + false_negative)
    # 面积与质心都必须用**真值所在网格**的面表来算：
    # extra 的真值是候选网格上的面号，若仍拿参考网格的面积表去索引，
    # 要么越界要么静默取到错误的面，面积指标会变成无意义的数。
    truth_areas = truth_mesh.triangle_areas
    ground_truth_area = float(truth_areas[list(ground_truth)].sum())
    predicted_area = float(truth_areas[list(predicted)].sum())
    true_positive_area = float(
        truth_areas[list(ground_truth & predicted)].sum()
    )
    area_precision = _ratio(true_positive_area, predicted_area)
    area_recall = _ratio(true_positive_area, ground_truth_area)
    area_iou = _ratio(
        true_positive_area,
        ground_truth_area + predicted_area - true_positive_area,
    )
    centroid_distance = _centroid_distance(
        truth_mesh,
        ground_truth,
        predicted,
    )
    boundary_iou = _boundary_iou(
        truth_mesh,
        ground_truth,
        predicted,
    )
    # 我为什么在这里重算一次"真值区域到底有多深"：
    # 检测器的判据是"采样点到候选网格的距离 >= threshold_mm"，
    # 所以一个案例能不能被检出，取决于删掉那片区域后
    # 真实产生的**几何深度**，而不是取决于我删了多少个面。
    # 精细三角化的小件（实测 训练集/8：4.8x5.9x2.5mm 却切了 19228 面，
    # 平均边长 0.079mm）删 60 个面只有 0.1355mm 深——
    # 低于 0.15mm 阈值，此时 recall=0 是**物理必然**，不是漏检。
    # 不把这种案例区分出来，评估结论就会把"我造的缺陷太小"
    # 错记成"检测器不行"，两种完全不同的问题混成一个数字。
    max_achievable_depth_mm = _max_achievable_depth(
        reference=reference,
        candidate=candidate,
        ground_truth=ground_truth,
        config=config,
        defect_type=defect_type,
    )
    detectable = (
        max_achievable_depth_mm is not None
        and max_achievable_depth_mm >= config.threshold_mm
    )
    # ------------------------------------------------------------------
    # 任务书口径：几何误差 >= 阈值的面。真值走独立密采样（面内点 + 顶点），
    # **不复用上面那份生产距离**，否则就是拿预测当真值的循环论证。
    # ------------------------------------------------------------------
    spec_truth = threshold_reached_face_ids(
        truth_mesh,
        candidate if defect_type == "missing" else reference,
        config=config,
        samples_per_triangle=int(truth_samples_per_triangle),
        face_ids=ground_truth,
    )
    spec_reached = {int(face_id) for face_id in spec_truth.reached_face_ids}
    # 预测到没有被删的面，在两种口径下都是错，所以口径间的唯一差别
    # 在于"达阈值的面"这个分母，而不是在于怎么算假阳性。
    spec_true_positive = len(spec_reached & predicted)
    spec_false_positive = len(predicted - ground_truth)
    spec_false_negative = len(spec_reached - predicted)
    spec_precision = _ratio(
        spec_true_positive, spec_true_positive + spec_false_positive
    )
    spec_recall = _ratio(
        spec_true_positive, spec_true_positive + spec_false_negative
    )
    spec_iou = _ratio(
        spec_true_positive,
        spec_true_positive + spec_false_positive + spec_false_negative,
    )
    # 轮廓带只在缺失方向上量：多料的生产距离来自反向那一遍，
    # detect_missing_regions 没有把它返回出来，我不去编一个数补上，
    # 而是让多料口径下轮廓带留 0/None（spec_* 指标仍然照算）。
    contour_band_face_count = 0
    contour_band_max_distance: float | None = None
    area_filtered_face_count = 0
    if defect_type == "missing":
        production_reached = {
            face_id
            for face_id in spec_reached
            if float(production_face_distances[face_id]) >= config.threshold_mm
        }
        contour_band = spec_reached - production_reached
        contour_band_face_count = len(contour_band)
        if contour_band:
            contour_band_max_distance = max(
                float(production_face_distances[face_id])
                for face_id in contour_band
            )
        # 生产判据本来会报出、却被 min_region_area_mm2 丢掉的面。
        # 这一类的行动是"调阈值"，与轮廓带的"提采样密度"完全不同，
        # 所以必须在口径内部就分开计数。
        production_abnormal = {
            int(face_id) for face_id in np.nonzero(
                production_face_distances >= config.threshold_mm
            )[0]
        }
        area_filtered_face_count = len(
            (production_abnormal - predicted) & spec_reached
        )
    # 我把"指标"和"状态"分开：指标永远如实计算并输出，
    # 状态才表达"这个案例是否干净通过"。
    return DetectionMetrics(
        case_id=case_id,
        ground_truth_face_count=len(ground_truth),
        predicted_face_count=len(predicted),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou,
        region_detected=bool(ground_truth & predicted),
        status=(
            "PASS"
            if (
                bool(ground_truth)
                and false_negative == 0
                and false_positive == 0
                and invalid_ground_truth_count == 0
            )
            else "REVIEW"
        ),
        ground_truth_area_mm2=ground_truth_area,
        predicted_area_mm2=predicted_area,
        true_positive_area_mm2=true_positive_area,
        area_precision=area_precision,
        area_recall=area_recall,
        area_iou=area_iou,
        centroid_distance_mm=centroid_distance,
        boundary_iou=boundary_iou,
        invalid_ground_truth_face_count=invalid_ground_truth_count,
        max_achievable_depth_mm=max_achievable_depth_mm,
        detectable=detectable,
        defect_type=defect_type,
        filtered_area_mm2=filtered_area_mm2,
        spec_truth_samples_per_triangle=int(truth_samples_per_triangle),
        spec_truth_sample_point_count=spec_truth.sample_point_count,
        spec_truth_face_count=len(spec_reached),
        spec_true_positive=spec_true_positive,
        spec_false_positive=spec_false_positive,
        spec_false_negative=spec_false_negative,
        spec_precision=spec_precision,
        spec_recall=spec_recall,
        spec_iou=spec_iou,
        spec_region_detected=bool(spec_reached & predicted),
        contour_band_face_count=contour_band_face_count,
        contour_band_production_max_distance_mm=contour_band_max_distance,
        spec_false_negative_area_filtered_face_count=area_filtered_face_count,
    )


def evaluate_synthetic_suite(
    reference: TriangleMesh,
    *,
    seed_face_ids: Iterable[int],
    face_counts: Iterable[int],
    config: InspectionConfig | None = None,
    target: float = 0.99,
    thresholds: Iterable[float] | None = None,
    defect_type: str = "missing",
    offset_mm: float = 1.0,
    truth_samples_per_triangle: int = 9,
) -> dict[str, object]:
    """我批量评估合成区域，并把目标是否达到写成明确门禁。

    我遍历 seed_face_ids × face_counts 的笛卡尔积，每个组合删一块连续面、
    跑一次检测，然后把所有案例的混淆矩阵加总成全局指标。
    这样报告里既有每个案例的明细，也有整体口径。

    defect_type 选 missing 时删面造缺料，选 extra 时复制外推造多料。
    offset_mm 只在 extra 下有意义，必须大于检测阈值才检得到。
    """

    if defect_type not in ("missing", "extra"):
        raise ValueError(
            f"我只支持 missing 与 extra 两类缺陷，当前收到 {defect_type!r}。"
        )
    # 先做参数校验：目标必须落在 (0, 1]，种子非空非负，删除面数至少为 1。
    if not np.isfinite(float(target)) or not 0.0 < float(target) <= 1.0:
        raise ValueError("我要求 target 位于 (0, 1] 区间。")
    seeds = [int(seed) for seed in seed_face_ids]
    counts = [int(count) for count in face_counts]
    if not seeds or not counts or any(seed < 0 for seed in seeds):
        raise ValueError("我要求 seed_face_ids 和 face_counts 都是非空正向配置。")
    if any(count < 1 for count in counts):
        raise ValueError("我要求 face_counts 中每个值至少为 1。")
    # config 缺省时必须在这里就地补默认值，而不是让它以 None 往下传：
    # 下面报告里的"缺陷深度是否达到阈值"要用 config.threshold_mm，
    # 而 evaluate_synthetic_case 内部的缺省只作用在那一个函数里，
    # 套件这一层拿不到。两处缺省必须一致，否则"用什么阈值判的可检出"
    # 在套件和单案例里会是两个数。
    if config is None:
        config = InspectionConfig(
            registration_mode="none",
            unit_policy="assume_mm",
        )

    threshold_values = _validate_thresholds(thresholds)
    cases: list[DetectionMetrics] = []
    # 笛卡尔积：每个种子位置 × 每种删除规模，构成一组受控案例。
    for seed in seeds:
        for count in counts:
            # 每个 (种子, 缺面数) 组合都造一个独立样本：
            # 种子决定缺哪块、数量决定缺多大，
            # 两者交叉才能覆盖"位置"和"尺度"两个变量。
            if defect_type == "missing":
                defect = remove_face_region(reference, seed, count)
                truth = defect.removed_face_ids
            else:
                defect = add_face_region(reference, seed, count, offset_mm=offset_mm)
                truth = defect.added_face_ids
            cases.append(
                evaluate_synthetic_case(
                    reference,
                    defect.candidate_mesh,
                    truth,
                    # case_id 把两个自变量写进名字：
                    # 结果表里看到哪一行差，直接就知道是哪个组合。
                    case_id=f"{defect_type};seed={seed};faces={count}",
                    config=config,
                    defect_type=defect_type,
                    truth_samples_per_triangle=int(truth_samples_per_triangle),
                )
            )
    # 四项计数先跨样本求和再算比率，而不是对每个样本的比率取平均：
    # 后者会让每个样本等权，小样本的一次误检
    # 就能把整体指标拉低一大截，与实际工作量不成比例。
    #
    # 我为什么只统计"物理上可检出"的案例：
    # 精细三角化的小件上，删少量面造成的凹陷深度可能根本达不到阈值
    # （实测 训练集/8 删 60 面只有 0.1355mm，阈值 0.15mm）。
    # 这种案例"没检出"是物理必然，把它算进召回率分母，
    # 等于用"我造的缺陷太小"去惩罚检测器——两个完全不同的问题
    # 会被混成同一个 recall 数字，谁也解释不清。
    # 所以我把它单列成 undetectable_case_count，并在下面如实上报。
    scored = [case for case in cases if case.detectable]
    undetectable_case_count = len(cases) - len(scored)
    # 我单独数出"检出了、又被小区域阈值滤掉、于是表现为漏检"的案例。
    # 判据是"没被算作检出"且"该案例确实有面积被滤掉"——
    # 两者同时成立才说明这一次漏检的成因是阈值，而不是算法没找到。
    # 我把这个数放进报告，是为了让"漏检"这件事有归因：
    # 甲方看到 11/12 时，需要知道少的那一个该调 min_region_area_mm2
    # 还是该改检测算法，否则只能笼统地认为"检测器不行"。
    filtered_after_detection = [
        case
        for case in cases
        if not case.region_detected and case.filtered_area_mm2 > 0.0
    ]
    total_tp = sum(case.true_positive for case in scored)
    total_fp = sum(case.false_positive for case in scored)
    total_fn = sum(case.false_negative for case in scored)
    precision = _ratio(total_tp, total_tp + total_fp)
    recall = _ratio(total_tp, total_tp + total_fn)
    f1 = _ratio(2.0 * precision * recall, precision + recall)
    # IoU 的分母是并集：TP 被算了两次（分子一次、GT 与预测各一次），
    # 减掉一次才是并集面积。
    iou = _ratio(total_tp, total_tp + total_fp + total_fn)
    # 区域级召回：只要"这块缺面被检出"就算命中，
    # 不管检出的面积准不准。它与面积召回回答的是不同问题：
    # 前者是"漏不漏"，后者是"圈得准不准"。
    case_recall = _ratio(
        sum(case.region_detected for case in scored),
        len(scored),
    )
    # 面积级三件套：把"面积"当成量而不是"区域"当成单位。
    # 甲方关心的是"漏检的面积占多大比例"，
    # 而不是"漏检了几块"——一大块和一小块权重不该相同。
    area_precision = _ratio(
        sum(case.true_positive_area_mm2 for case in cases),
        sum(case.predicted_area_mm2 for case in cases),
    )
    area_recall = _ratio(
        sum(case.true_positive_area_mm2 for case in scored),
        sum(case.ground_truth_area_mm2 for case in scored),
    )
    area_iou = _ratio(
        sum(case.true_positive_area_mm2 for case in scored),
        sum(
            case.ground_truth_area_mm2
            + case.predicted_area_mm2
            - case.true_positive_area_mm2
            for case in scored
        ),
    )
    boundary_iou = _ratio(
        sum(case.boundary_iou for case in scored),
        len(scored),
    )
    centroid_distances = [
        case.centroid_distance_mm
        for case in scored
        if case.centroid_distance_mm is not None
    ]
    target_met = (
        # 一个案例都没进入打分时，**绝不能**判达标。
        # 我为什么把这条放在最前面（2026-10-16 实测）：上面六个指标在
        # "分母为 0"时都会被 _ratio 取成缺省值 1.0，于是
        # "我一个案例都没评"会伪装成"六项全 1.0，目标达成"。
        # 我在 训练集/8 上实测过这个假绿：9 个案例全部因缺陷深度不足阈值
        # 被排除（scored=0、detected=0），报告却给出
        # status=BASELINE_TARGET_MET、case_recall=1.0。
        # 这是"把什么都没验证当成通过"这一类最危险的假绿，
        # 而它恰好长在对应甲方 ≥99% 验收的那条指标上，所以必须堵死。
        bool(scored)
        # 六个条件必须同时满足，缺一不可。我不用加权平均，
        # 因为"precision 0.99 但 recall 0.5"被平均后看着像达标，实际是漏检。
        and precision >= float(target)
        and recall >= float(target)
        and case_recall >= float(target)
        and area_precision >= float(target)
        and area_recall >= float(target)
        and boundary_iou >= float(target)
    )
    # ------------------------------------------------------------------
    # 任务书口径的汇总（独立密采样真值）。它与旧口径**打分集合也不同**：
    # 旧口径按"生产采样能不能达到阈值"判定可检出；任务书口径按
    # "独立密采样能不能达到阈值"判定。两边各算各的分母，互不借用。
    # ------------------------------------------------------------------
    spec_scored = [case for case in cases if case.spec_truth_face_count > 0]
    spec_tp = sum(case.spec_true_positive for case in spec_scored)
    spec_fp = sum(case.spec_false_positive for case in spec_scored)
    spec_fn = sum(case.spec_false_negative for case in spec_scored)
    spec_precision = _ratio(spec_tp, spec_tp + spec_fp)
    spec_recall = _ratio(spec_tp, spec_tp + spec_fn)
    spec_iou = _ratio(spec_tp, spec_tp + spec_fp + spec_fn)
    spec_case_recall = _ratio(
        sum(case.spec_region_detected for case in spec_scored),
        len(spec_scored),
    )
    # 任务书口径目前只有面级与区域级三个指标，所以门禁只卡这三项。
    # 我**没有**把旧口径的 area_*/boundary_iou 借过来凑成六项：
    # 那两项算在"全部被删面"这个分母上，借过来会让门禁看起来更严，
    # 实际是两个口径混用，谁也解释不清失败在哪一项。
    spec_target_met = (
        bool(spec_scored)
        and spec_precision >= float(target)
        and spec_recall >= float(target)
        and spec_case_recall >= float(target)
    )
    contour_band_total = sum(case.contour_band_face_count for case in cases)
    area_filtered_total = sum(
        case.spec_false_negative_area_filtered_face_count for case in cases
    )
    contour_band_max = max(
        (
            case.contour_band_production_max_distance_mm
            for case in cases
            if case.contour_band_production_max_distance_mm is not None
        ),
        default=None,
    )
    report: dict[str, object] = {
        # 1.1 -> 1.2（2026-10-16）：新增
        # filtered_after_detection_case_count / filtered_area_total_mm2，
        # 以及每个案例的 filtered_area_mm2。
        # 我按本仓既有惯例升版本（BOM 加一列时也从 1.7 升到 1.8）：
        # 加了字段就换版本号，让读报告的人能一眼判断该按哪套字段解析。
        # 1.2 -> 1.3（2026-10-25 接管续写）：新增 spec_conformant 口径块、
        # status_spec_conformant，以及每个案例的 spec_* / contour_band_* 字段。
        "schema_version": "synthetic-evaluation-1.3",
        # 达标状态写 "TARGET_NOT_PROVEN" 而不是 "FAILED"：
        # 没达标可能是真的差，也可能是样本量不够还没证明，
        # 用"未证明"比"失败"更准确，也避免被读成回归。
        "status": "BASELINE_TARGET_MET" if target_met else "TARGET_NOT_PROVEN",
        # 任务书口径的状态**另开一个字段**，不替换 status：
        # 换口径是甲方的书面决定，工程侧只能把两套都算出来。
        "status_spec_conformant": (
            "BASELINE_TARGET_MET" if spec_target_met else "TARGET_NOT_PROVEN"
        ),
        "target": float(target),
        # 参考面数与样本数都记：指标的含义依赖于它们，
        # 只看精度无法知道这个精度是在多大的面上、多少个样本上得到的。
        "reference_face_count": int(len(reference.faces)),
        "case_count": len(cases),
        "detected_case_count": sum(case.region_detected for case in cases),
        # 我如实上报"有多少案例因缺陷深度不足阈值而不可检出"，
        # 以及参与打分的案例数。这两个数必须都暴露出来：
        # 只报 scored 会让统计口径看着变小却说不清为什么；
        # 只报总数又会把不可检出的案例当成漏检混进准确率。
        "undetectable_case_count": undetectable_case_count,
        "scored_case_count": len(scored),
        # 这两种"漏检"成因必须分开报：前者要改几何/阈值口径，
        # 后者只要调 min_region_area_mm2，行动完全不同。
        "filtered_after_detection_case_count": len(filtered_after_detection),
        "filtered_area_total_mm2": float(
            sum(case.filtered_area_mm2 for case in cases)
        ),
        "aggregate": {
            # 三个原始计数排在比率之前：比率都能从它们推出来，
            # 反过来不行。先给计数能让复核者自己验算。
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "case_recall": case_recall,
            "area_precision": area_precision,
            "area_recall": area_recall,
            "area_iou": area_iou,
            "boundary_iou": boundary_iou,
            # 质心距离均值在无样本时留 None 而不是 0.0：
            # 0.0 会被读成"预测与真值完全重合"，
            # 而真实情况是"没有可比对的样本"。
            "mean_centroid_distance_mm": (
                float(np.mean(centroid_distances))
                if centroid_distances
                else None
            ),
            # 无效真值面数单独统计：它说明标签本身有问题，
            # 混进总数会让"指标差"被错误地归因到算法上。
            "invalid_ground_truth_face_count": sum(
                case.invalid_ground_truth_face_count for case in cases
            ),
        },
        "cases": [case.to_dict() for case in cases],
        # 任务书口径（几何误差达阈值的面）单独成块，并自带采样依据。
        # 我为什么坚持两套都留在同一份报告里：只报一套，读的人无法判断
        # 结论是"检测器变了"还是"口径变了"；两套并排，差异一目了然。
        "spec_conformant": {
            "truth_scope": "faces_with_geometric_error_at_or_above_threshold",
            "truth_samples_per_triangle": int(truth_samples_per_triangle),
            "truth_face_count": sum(case.spec_truth_face_count for case in cases),
            "scored_case_count": len(spec_scored),
            "scored_truth_face_count": sum(
                case.spec_truth_face_count for case in spec_scored
            ),
            "true_positive": spec_tp,
            "false_positive": spec_fp,
            "false_negative": spec_fn,
            "precision": spec_precision,
            "recall": spec_recall,
            "iou": spec_iou,
            "case_recall": spec_case_recall,
            # 轮廓带：密采样达阈值、生产采样却低于阈值的面（采样分辨率差异）。
            "contour_band_face_count": contour_band_total,
            "contour_band_production_max_distance_mm": contour_band_max,
            # 另一类口径内漏检：检出了但被 min_region_area_mm2 丢掉。
            "false_negative_area_filtered_face_count": area_filtered_total,
            # 哪些指标进了这个口径的门禁，写出来避免被误读成六项全卡。
            "gated_metrics": ["precision", "recall", "case_recall"],
            "target_met": spec_target_met,
        },
        # 评估范围必须随 defect_type 变，不能写死成"删面"：
        # 跑 --defect-type extra 时写死会给出假的标签
        #（实际造的是多料，报告却说只评估删面），
        # 这比范围写得含糊更糟——它是**错**的，不只是不精确。
        # 但范围仍然必须限定在"合成"内，
        # 否则会被拿去评价真实破损件的效果。
        "evaluation_scope": (
            "synthetic_face_deletion_only"
            if defect_type == "missing"
            else "synthetic_face_extrusion_only"
        ),
        "limitations": [
            # 这句话也必须跟着 defect_type 变：跑 extra 时说"删除面标签"
            # 会和报告里的真值口径自相矛盾。
            (
                "我只使用合成删除面标签，不能代表甲方真实标注集。"
                if defect_type == "missing"
                else "我只使用合成外推多料标签，不能代表甲方真实标注集。"
            ),
            "我没有把合成基线指标写成工业准确率或最终验收结论。",
            "我没有用合成数据替代真实训练、验证、测试集隔离。",
            (
                f"本轮 {undetectable_case_count}/{len(cases)} 个案例的缺陷深度"
                f"不足阈值 {config.threshold_mm}mm，物理上不可检出，"
                "已从准确率统计中排除；它们不代表检测器漏检。"
            ),
            # 这条必须单独出现，不能只靠上面那句"排除"来暗示。
            # 全部案例都被排除时，报告里的 precision/recall/case_recall
            # 全是 0/0 的缺省值 1.0，**看上去像满分**。不把"一个案例都
            # 没打分"这句话明写出来，读报告的人会把缺省值当成真实成绩。
            *(
                [
                    "本轮没有任何案例进入打分，因此**无法证明**目标是否达成，"
                    "状态记为 TARGET_NOT_PROVEN；报告里的各项比率是"
                    "“无样本”的缺省值 1.0，不代表检测成绩。"
                ]
                if not scored
                else []
            ),
            # 漏检归因：把"被阈值滤掉"和"没检出"分开说清楚。
            *(
                [
                    f"本轮有 {len(filtered_after_detection)} 个案例"
                    f"（共 {sum(case.filtered_area_mm2 for case in filtered_after_detection):.6f} mm2）"
                    "是**检出后被 min_region_area_mm2 滤掉**才表现为漏检的，"
                    "不是检测器没找到；要收口应调整该阈值而不是改检测算法。"
                ]
                if filtered_after_detection
                else []
            ),
            # 2026-10-25 接管续写：两套口径必须同时写进限制说明，
            # 否则读报告的人只会看到 `aggregate.recall` 或
            # `spec_conformant.recall` 其中一个，把两件事当成一件事。
            (
                "本报告同时给出两套真值口径：`aggregate` / `status` 是"
                "旧口径（把全部被删面当真值）；`spec_conformant` / "
                "`status_spec_conformant` 是任务书口径（几何误差 >= "
                f"{config.threshold_mm}mm 的面），真值用独立密度 "
                f"{int(truth_samples_per_triangle)} 点/面 + 三个顶点重算，"
                "不复用生产那份距离。验收用哪一套必须由甲方书面确认，"
                "我不替换 `status`。"
            ),
            (
                "把生产距离数组同时当预测与真值得到的那个 1.0 召回是"
                "**循环**口径：预测与真值同源，只能证明检测器是确定性的，"
                "不能当准确率；要看「找得准不准」，只认 "
                "`spec_conformant` 那一块。"
            ),
            *(
                [
                    f"轮廓带：本轮 {contour_band_total} 个面被独立密采样判为"
                    "误差达阈值、而生产采样（"
                    f"{int(config.samples_per_triangle)} 点/面且第 0 点恒取重心）"
                    "没判到；它们在生产口径下的面级误差最大 "
                    f"{float(contour_band_max):.6f}mm，"
                    f"低于阈值 {config.threshold_mm}mm —— 这是**采样分辨率**"
                    "差异，不是漏检。要收口应提高 samples_per_triangle"
                    " 或补采顶点，而不是改聚类算法。"
                ]
                if contour_band_total and contour_band_max is not None
                else [
                    "轮廓带：本轮没有被独立密采样判为达阈值、"
                    "而生产采样没判到的面（轮廓带为空），"
                    "所以这一批的漏检不能用采样分辨率解释。"
                ]
            ),
            *(
                [
                    f"任务书口径的漏检里另有 {area_filtered_total} 个面是"
                    "**检出后被 min_region_area_mm2 滤掉**的：这一类的行动是"
                    "调该阈值，与轮廓带的「提采样密度」完全不同，所以分开计数。"
                ]
                if area_filtered_total
                else []
            ),
            (
                "任务书口径目前只卡面级 precision/recall 与区域级 case_recall"
                " 三项；面积与边界指标仍只在旧口径下计算。我没有把旧口径的"
                " area_*/boundary_iou 借过来凑成六项——那会是两个口径混用。"
            ),
        ],
    }
    if threshold_values:
        # 阈值扫描：同一批案例在不同 threshold_mm 下各跑一遍，
        # 让甲方能看到"阈值取多少时指标开始崩"，而不是只看一个点。
        report["threshold_sweep"] = [
            _evaluate_threshold_point(
                reference,
                seeds,
                counts,
                config,
                threshold,
                target,
                truth_samples_per_triangle=int(truth_samples_per_triangle),
            )
            for threshold in threshold_values
        ]
    return report


def _evaluate_threshold_point(
    reference: TriangleMesh,
    seeds: list[int],
    counts: list[int],
    config: InspectionConfig | None,
    threshold: float,
    target: float,
    truth_samples_per_triangle: int = 9,
) -> dict[str, object]:
    """我在不同阈值下重复同一合成案例集合，避免只报告单点结果。"""

    threshold_config = replace(
        config or InspectionConfig(
            registration_mode="none",
            unit_policy="assume_mm",
        ),
        threshold_mm=threshold,
    )
    nested = evaluate_synthetic_suite(
        reference,
        seed_face_ids=seeds,
        face_counts=counts,
        config=threshold_config,
        target=target,
        thresholds=None,
        # 阈值扫描必须带上同一个真值密度：扫描点在阈值轴上变，
        # 真值口径不能在背后偷偷跟着变，否则各点的差异说不清是阈值还是采样。
        truth_samples_per_triangle=int(truth_samples_per_triangle),
    )
    aggregate = nested["aggregate"]
    return {
        "threshold_mm": float(threshold),
        "status": nested["status"],
        "aggregate": aggregate,
        # 每个扫描点也带上任务书口径，免得读报告的人只看旧口径。
        "status_spec_conformant": nested["status_spec_conformant"],
        "spec_conformant": nested["spec_conformant"],
    }


def _identity_registration() -> RegistrationResult:
    """我构造一个"恒等配准"结果：不旋转、不平移、尺度 1、残差 0、质量 good。

    合成案例本身完全对齐，用恒等配准才能让指标反映检测能力，
    而不是被配准算法的不确定性带偏。
    """

    return RegistrationResult(
        mode="none",
        rotation_matrix=np.eye(3).tolist(),
        translation_mm=[0.0, 0.0, 0.0],
        scale=1.0,
        mean_residual_mm=0.0,
        p95_residual_mm=0.0,
        quality="good",
    )


def _ratio(numerator: float, denominator: float) -> float:
    """我算比值，分母为 0 时返回 1.0。

    返回 1.0 而不是 0.0 或抛异常，是因为"没有可比的样本"在指标语义上
    等于"没有观察到错误"：例如真值集合为空、预测也为空时，precision 和
    recall 都不该被判成 0 分。这是我方明确的约定，不是默认值巧合。

    但这条约定**只适用于单个指标**，不能直接拿去做"达标"判定：
    六个指标在无样本时全是 1.0，于是"一个样本都没评"会同时满足
    六项 >= target 的判据。所以调用方 `evaluate_synthetic_suite`
    在判达标前必须先确认 `scored` 非空（2026-10-16 实测到的假绿，
    详见那里的注释与 `test我方全部案例都不可检出时绝不判达标`）。
    """

    return float(numerator / denominator) if denominator else 1.0


def _validate_thresholds(
    thresholds: Iterable[float] | None,
) -> list[float]:
    """我把阈值列表去重、排序并校验为有限正数，None 表示不做扫描。"""

    if thresholds is None:
        return []
    # 用 set 去重，避免同一阈值被扫两遍；排序让报告里的顺序稳定。
    values = sorted({float(value) for value in thresholds})
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("我要求 thresholds 中每个值必须是有限正数。")
    return values


def _centroid_distance(
    mesh: TriangleMesh,
    ground_truth: set[int],
    predicted: set[int],
) -> float | None:
    """我算真值区域与预测区域面积加权质心之间的欧氏距离，单位 mm。

    任一侧为空时返回 None：没有可比对象就不编造一个 0 距离。
    """

    if not ground_truth or not predicted:
        return None
    areas = mesh.triangle_areas
    centroids = mesh.triangle_centroids()
    gt_ids = np.asarray(sorted(ground_truth), dtype=np.int64)
    pred_ids = np.asarray(sorted(predicted), dtype=np.int64)
    gt_area = max(float(areas[gt_ids].sum()), 1e-12)
    pred_area = max(float(areas[pred_ids].sum()), 1e-12)
    gt_center = (centroids[gt_ids] * areas[gt_ids, None]).sum(axis=0) / gt_area
    pred_center = (
        (centroids[pred_ids] * areas[pred_ids, None]).sum(axis=0) / pred_area
    )
    return float(np.linalg.norm(gt_center - pred_center))


def _boundary_edges(mesh: TriangleMesh, face_ids: set[int]) -> set[tuple[int, int]]:
    """我求一组面的边界边集合（只被其中一个面用到的无向边）。

    我把每条边的小端点放前面归一化，这样 (a,b) 和 (b,a) 会被算成同一条边；
    出现次数恰好为 1 的边就是这组面的外边界。
    """

    counts: Counter[tuple[int, int]] = Counter()
    for face_id in face_ids:
        face = mesh.faces[face_id]
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = (first, second) if first < second else (second, first)
            counts[edge] += 1
    return {edge for edge, count in counts.items() if count == 1}


def _boundary_iou(
    mesh: TriangleMesh,
    ground_truth: set[int],
    predicted: set[int],
) -> float:
    """我算真值边界边集合与预测边界边集合的 IoU（交并比）。"""

    ground_truth_edges = _boundary_edges(mesh, ground_truth)
    predicted_edges = _boundary_edges(mesh, predicted)
    return _ratio(
        float(len(ground_truth_edges & predicted_edges)),
        float(len(ground_truth_edges | predicted_edges)),
    )


def _max_achievable_depth(
    *,
    reference: TriangleMesh,
    candidate: TriangleMesh,
    ground_truth: set[int],
    config: InspectionConfig,
    defect_type: str,
) -> float | None:
    """我量出真值区域**实际造成的最大几何深度**（mm）。

    这是"该案例在给定阈值下物理上能不能被检出"的判据。
    检测器判定缺失的依据是"参考面上的采样点到候选网格的距离 >= 阈值"，
    所以可达深度就是：**真值面上的采样点到候选网格的最大距离**。

    - ``missing``：真值在参考网格上，量"参考真值面 -> 候选网格"；
    - ``extra``：真值在候选网格上，量"候选真值面 -> 参考网格"。
      两类真值所在网格不同，方向也必须反过来，否则量的就是另一件事。

    我为什么要按真值面取样而不是全网格取样：全网格的最大距离可能来自
    别处的噪声，而这个值回答的问题是"**这一处**缺陷有多深"。
    真值为空时返回 None，由调用方按"不可判定"处理。
    """

    if not ground_truth:
        return None
    # 真值所在网格与被测网格：缺失看参考、多料看候选。
    truth_mesh = reference if defect_type == "missing" else candidate
    other_mesh = candidate if defect_type == "missing" else reference
    if len(other_mesh.faces) == 0:
        return None
    face_ids = np.asarray(sorted(ground_truth), dtype=np.int64)
    # 采样参数沿用生产配置：评估必须与生产用同一套采样密度，
    # 否则"能检出"的判据和生产实际看到的不是同一批点。
    points, sampled_faces = truth_mesh.sample_points(
        samples_per_triangle=config.samples_per_triangle,
        max_points=config.max_query_points,
    )
    keep = np.isin(sampled_faces, face_ids)
    if not np.any(keep):
        return None
    distances, _, _ = points_to_mesh(
        points[keep],
        other_mesh,
        chunk_size=config.distance_chunk_size,
        backend=config.distance_backend,
        candidate_k=config.distance_candidate_k,
    )
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return None
    return float(finite.max())
