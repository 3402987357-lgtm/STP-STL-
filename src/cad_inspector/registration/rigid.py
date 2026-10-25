"""我用 NumPy 实现可解释的质心、主轴和 ICP 配准基线。

模块职责：把待检网格（candidate）对齐到参考网格（reference）的坐标系，
并给出这次对齐的质量报告（残差均值/P95、缩放因子、是否需要复核）。

为什么不用 Open3D 当主路径：本机没有安装 Open3D，而且甲方要求"本地、
可复现、可解释"。纯 NumPy 实现虽然慢，但每一步（质心、主轴、Kabsch、
最近邻）都能单独验证，也能在没有第三方配准库时照常运行。

输入边界：两个网格都必须非空；单位由上层 probe 统一到毫米，本模块不做
单位换算，只报出"整体缩放是否明显偏离 1"并要求复核。
输出边界：RegistrationResult 只描述"对齐得怎么样"，不描述"模型有没有
破损"；破损判定在 detection 层完成。
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np

from ..geometry.mesh import TriangleMesh
from ..models import InspectionConfig, RegistrationResult


def align_mesh(
    candidate: TriangleMesh,
    reference: TriangleMesh,
    config: InspectionConfig,
) -> tuple[TriangleMesh, RegistrationResult]:
    """我将 candidate 变换到 reference 坐标系，并返回质量报告。"""

    # 原理说明（甲方要求"关键算法附原理说明"）：配准是后续一切偏差判定的
    # 地基——如果两个模型的位姿没对齐，再好的距离算法也只会报出假缺陷。
    # 我按"由弱到强、由快到慢"提供四种模式，让调用方按需要选：
    #   none     —— 输入本身已经同坐标系，只做恒等变换；
    #   centroid —— 只对质心，最快，适合位姿已知只差平移的场景；
    #   pca      —— 主轴对齐 + 24 种符号/轴序组合择优，能纠正旋转；
    #   icp      —— 在 PCA 初值上迭代最近点，精度最高但最慢。
    # 边界：空网格直接拒绝；未知模式抛错而不是静默退回恒等变换，
    # 否则会把"没配准"伪装成"配准成功"。

    if not len(candidate.vertices) or not len(reference.vertices):
        raise ValueError("我无法对空网格执行配准。")
    if config.registration_mode == "auto":
        return _auto_align(candidate, reference, config)
    if config.registration_mode == "none":
        # 我即使不做变换也要实测残差：none 的语义是"输入已同坐标系"，
        # 不是"输入一定对齐"。写死零残差会把严重错位（例如整体缩放 2.5 倍）
        # 伪造成 quality=good、平均残差 0，后续所有距离判定都建立在谎上。
        residuals = _registration_residuals(candidate, reference)
        # 不做配准时也要算一遍残差：
        # 残差是"这两个模型本来就差多少"的基线，
        # 没有它就无法判断配准到底改善了多少。
        result = _make_result(
            mode="none",
            # 单位旋转、零平移、单位缩放：
            # 明确记下"没动过"，而不是留空让下游去猜。
            rotation=np.eye(3),
            translation=np.zeros(3),
            scale=1.0,
            residuals=residuals,
        )
        return candidate, result
    if config.registration_mode == "centroid":
        return _centroid_align(candidate, reference, config)
    if config.registration_mode == "pca":
        # PCA 配准：按两个模型的主轴方向对齐。
        # 它能处理整体朝向不同的情形，
        # 但主轴方向存在符号不确定性，结果可能差 180 度。
        rotation, translation, scale = _pca_transform(
            candidate, reference, allow_scale=config.allow_scale
        )
        aligned = candidate.transformed(rotation, translation, scale)
        residuals = _registration_residuals(aligned, reference)
        return aligned, _make_result("pca", rotation, translation, scale, residuals)
    if config.registration_mode == "icp":
        # ICP 迭代最近点：精度最高，但需要 PCA 或质心
        # 给出一个像样的初值，否则会收敛到局部最优。
        rotation, translation, scale, residuals = _icp_transform(
            candidate,
            reference,
            allow_scale=config.allow_scale,
        )
        aligned = candidate.transformed(rotation, translation, scale)
        result = _make_result("icp", rotation, translation, scale, residuals)
        # 我为什么给 ICP 单独挂一条"可能吃掉缺陷"的告警（2026-10-15 实测）：
        # 残差变小在多数语境下是好事，但在**破损检测**里可能是灾难。
        # ICP 最小化的是"整体点集距离"，候选件缺一块料时，
        # 它会主动把模型往缺口方向挪来降低整体残差——
        # 于是缺口被"对齐掉了"，检测器什么也报不出。
        # 我实测确认过这不是猜测：测试集/2 的 B-Rep 有效性检查**失败**、
        # 确实有破损，centroid 报 missing=112.1 mm2，
        # 换成 icp 后 missing=0、regions=0——**真缺陷被吃掉了**。
        # 测试集/1 有台账实测的 0.761872 mm2 真实孔洞，
        # icp 也把它从 153.9 压到 0.2。
        # 所以残差降低不等于检测更准，我必须让这条风险显式出现在结果里。
        #
        # 我拿 centroid 当参照系而不是"未配准的原始输入"：
        # 这批语料两侧本来就在同一坐标系（实测原始残差比 centroid 还小），
        # 拿原始输入比会得出"ICP 反而变差 49%"的荒谬结论。
        # 真正该问的是"相对基准配准，ICP 额外抹平了多少"，
        # 实测这批件上是 37%~58%。
        _, centroid_result = _centroid_align(candidate, reference, config)
        if centroid_result.mean_residual_mm > 0:
            reduced = 1.0 - (
                result.mean_residual_mm / centroid_result.mean_residual_mm
            )
            if reduced >= 0.2:
                result.warnings.append(
                    "我用的是 ICP 配准，它把平均残差降低了 "
                    f"{reduced * 100:.0f}%；残差变小在破损检测里不一定是好事——"
                    "ICP 会为了整体距离最小而把模型挪向缺口，"
                    "可能把真实缺料区域'对齐掉'而报不出缺陷。"
                    "请把 ICP 结果与 centroid 结果对比后再下通过结论。"
                )
        return aligned, result
    # 未知模式直接抛异常：静默退回"不配准"会让后续
    # 所有距离计算都在错误的位姿下进行，
    # 而结果看起来完全正常。
    raise ValueError(f"我不认识注册模式：{config.registration_mode}")


def _auto_align(
    candidate: TriangleMesh,
    reference: TriangleMesh,
    config: InspectionConfig,
) -> tuple[TriangleMesh, RegistrationResult]:
    """我在 none / centroid / pca 里挑平均残差最小的一种，绝不选 icp。

    我为什么加这个模式（2026-10-17 实测）：我把 21 个真实件的三种配准
    逐个量了一遍，**21/21 个件的 none 残差都低于 centroid 和 pca**。
    原因是这批件的 STEP 与 STL 本来就在同一坐标系，而候选件缺了料，
    质心跟着偏了；centroid 把两个质心对齐，等于把整张表面推开，
    于是 0.15mm 阈值下 90% 的表面积被误报成"缺失"。

    所以固定用 centroid 是不对的，但反过来固定用 none 也不对——
    真有位姿差的文件（旋转过、平移到别处）需要 centroid 或 pca。
    唯一在两种情况下都说得通的做法是"三个都试，选残差最小的"。
    因为 auto 的残差恒 <= 固定 centroid 的残差，它严格优于旧默认值。

    为什么把 icp 排除在外：ICP 会为了压低整体残差把模型挪向缺口，
    把真实缺料"对齐掉"（见 icp 分支里那条告警）。残差最小在这里
    不等于配准最准，所以自动选择只在安全的三种里挑。
    """

    trials: list[tuple[float, str, TriangleMesh, RegistrationResult]] = []
    for mode in ("none", "centroid", "pca"):
        # 用 dataclasses.replace 复制配置，只改模式，其余参数原样带过去；
        # 逐个调 align_mesh 复用已有实现，不另写一份变换逻辑。
        aligned, result = align_mesh(
            candidate,
            reference,
            replace(config, registration_mode=mode),
        )
        trials.append((result.mean_residual_mm, mode, aligned, result))
    best_mean, best_mode, best_aligned, best_result = min(
        trials, key=lambda item: item[0]
    )
    # 我保留被选中那一种的真实变换矩阵，只把 mode 改写成 auto->xxx，
    # 这样下游既知道"是自动选的"，也知道"最后选的是谁"。
    best_result.mode = f"auto->{best_mode}"
    best_result.warnings.append(
        f"我用了自动配准：在 none/centroid/pca 里选平均残差最小的 {best_mode}"
        f"（{best_mean:.6f} mm）；另外两种分别是 "
        + "、".join(
            f"{mode}={mean:.6f} mm"
            for mean, mode, _, _ in trials
            if mode != best_mode
        )
        + "。"
    )
    return best_aligned, best_result


def apply_registration(
    mesh: TriangleMesh,
    registration: RegistrationResult,
) -> TriangleMesh:
    """我根据已保存的结果重新变换任意同源网格。"""

    return mesh.transformed(
        rotation=np.asarray(registration.rotation_matrix, dtype=np.float64),
        translation=np.asarray(registration.translation_mm, dtype=np.float64),
        scale=registration.scale,
    )


def _estimate_scale(candidate: TriangleMesh, reference: TriangleMesh) -> float:
    """我用"质心 RMS 半径之比"估计整体缩放。

    原理说明：不取包围盒对角线之比，是因为对角线对个别离群顶点极敏感；
    RMS 半径把所有顶点都算进去，抗噪更稳。分母接近 0（退化为一点）时
    返回 1.0，避免除零并把尺度判成无穷大。
    """
    source_centered = candidate.vertices - candidate.vertices.mean(axis=0)
    target_centered = reference.vertices - reference.vertices.mean(axis=0)
    source_size = float(
        np.sqrt(np.mean(np.sum(source_centered * source_centered, axis=1)))
    )
    target_size = float(
        np.sqrt(np.mean(np.sum(target_centered * target_centered, axis=1)))
    )
    if source_size <= 1e-12:
        return 1.0
    return float(target_size / source_size)


def _pca_transform(
    candidate: TriangleMesh,
    reference: TriangleMesh,
    allow_scale: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """我用主轴对齐并枚举 24 种轴向组合，选双向残差最小的一种。

    原理说明：PCA 只能给出三条轴的方向，无法区分正负和轴的对应关系，
    所以有 3! x 2^3 = 48 种候选，其中行列式为负的 24 种是镜像（把模型
    翻面），必须剔除，剩下 24 种逐一打分。打分不用半径分布而用
    `_symmetric_nearest_score` 的双向最近邻残差，能真正区分"轴对了但
    头尾反了"的情况。为了让 24 次打分可控，打分点集先降到 512 点。
    """
    source_center = candidate.vertices.mean(axis=0)
    target_center = reference.vertices.mean(axis=0)
    source_centered = candidate.vertices - source_center
    target_centered = reference.vertices - target_center
    source_axes = _principal_axes(source_centered)
    target_axes = _principal_axes(target_centered)
    scale = _estimate_scale(candidate, reference) if allow_scale else 1.0
    source_score_points = _limit_points(source_centered * scale, 512)
    target_score_points = _limit_points(target_centered, 512)
    best_rotation = np.eye(3)
    best_score = float("inf")
    for permutation in itertools.permutations(range(3)):
        permuted = target_axes[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate_axes = permuted * np.asarray(signs)
            rotation = candidate_axes @ source_axes.T
            if np.linalg.det(rotation) < 0:
                continue
            transformed = source_score_points @ rotation.T
            score = _symmetric_nearest_score(
                transformed,
                target_score_points,
            )
            if score < best_score:
                best_score = score
                best_rotation = rotation
    translation = target_center - (source_center * scale) @ best_rotation.T
    return best_rotation, translation, scale


def _principal_axes(points: np.ndarray) -> np.ndarray:
    """我对协方差矩阵做对称特征分解，得到从大到小排列的主轴。

    原理说明：协方差矩阵对称半正定，用 eigh 而不是 svd，数值上更稳且
    天然给实特征值。特征值降序即主轴重要性排序。最后一行把行列式修正为
    +1，保证主轴构成右手系，否则后续旋转会夹带镜像。
    """
    covariance = np.cov(points.T) if len(points) > 1 else np.eye(3)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes


def _icp_transform(
    candidate: TriangleMesh,
    reference: TriangleMesh,
    allow_scale: bool,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """我实现小规模 CPU ICP，作为没有 Open3D 时的可运行基线。"""

    # 原理说明：ICP 只保证收敛到局部最优，所以初值决定成败。我不用单位
    # 矩阵起步，而是先跑 `_pca_transform` 得到粗对齐，再在粗对齐上迭代：
    #   1) 对每个源点找目标点集里的最近点（对应关系）；
    #   2) 用 Kabsch 求使对应点对残差最小的刚体变换；
    #   3) 累乘到总变换上，直到平均残差小于 1e-7 或达到迭代上限。
    # 边界：源/目标都先降到 2000 点，保证单次迭代的 O(N·M) 可控；
    # 这是可运行的 CPU 基线，不是工业级多分辨率 ICP。

    rotation, translation, scale = _pca_transform(
        candidate,
        reference,
        allow_scale=allow_scale,
    )
    transformed = candidate.transformed(rotation, translation, scale)
    source = _limit_points(transformed.vertices, 2000)
    target = _limit_points(reference.vertices, 2000)
    for _ in range(iterations):
        nearest_indices, nearest_distances = _nearest_points(source, target)
        target_matches = target[nearest_indices]
        delta_rotation, delta_translation = _kabsch(source, target_matches)
        source = source @ delta_rotation.T + delta_translation
        rotation = delta_rotation @ rotation
        translation = (translation @ delta_rotation.T) + delta_translation
        if len(nearest_distances) and float(np.mean(nearest_distances)) < 1e-7:
            break
    _, residuals = _nearest_points(source, target)
    return rotation, translation, scale, residuals


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """我用 SVD 求两组对应点的最优刚体变换（Kabsch 算法）。

    原理说明：先各自去质心，再对互协方差做 SVD；最优旋转为 V·Uᵀ。
    若 det(V·Uᵀ) < 0，说明解里混进了镜像，此时翻转 V 的最后一行再重算，
    保证结果一定是纯旋转。平移由"目标质心 − 旋转后的源质心"直接给出。
    """
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _nearest_points(
    source: np.ndarray,
    target: np.ndarray,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """我按块计算最近邻，返回（最近点下标，最近距离）。

    原理说明：直接算 N x M x 3 的差分会爆内存——5000×5000 时那个中间
    数组约 600MB。我按源点分块，峰值只与 chunk_size 有关，结果与一次性
    计算完全一致。边界：目标为空时抛错；源为空时返回空数组而不是报错，
    因为空源在采样后是合法输入。
    """
    if len(source) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )
    if len(target) == 0:
        raise ValueError("我无法把点配准到空目标点集。")
    # 我按 source 分块计算最近邻，避免一次分配 (N, M, 3) 的稠密数组；
    # 5000×5000 时那个数组要 600MB，分块后峰值只和 chunk_size 有关。
    indices = np.empty(len(source), dtype=np.int64)
    distances = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), chunk_size):
        block = source[start : start + chunk_size]
        differences = block[:, None, :] - target[None, :, :]
        squared = np.sum(differences * differences, axis=2)
        nearest = np.argmin(squared, axis=1)
        indices[start : start + len(block)] = nearest
        distances[start : start + len(block)] = np.sqrt(
            squared[np.arange(len(block)), nearest]
        )
    return indices, distances


def _limit_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """我按等间隔抽稀点集，把复杂度压到可控范围。

    原理说明：用 linspace 取等间隔下标而不是随机采样，是为了让同一输入
    每次得到同一条点集——配准结果必须可复现，随机采样会破坏这一点。
    注意：这是抽稀不是体素降采样，不会改变点位置。
    """
    if len(points) <= max_points:
        return points.copy()
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def _registration_residuals(
    candidate: TriangleMesh,
    reference: TriangleMesh,
) -> np.ndarray:
    """我算候选顶点到参考**曲面**的最近距离，作为配准残差分布。

    为什么从"到顶点"改成"到曲面"（2026-10-18 实测）：甲方的 STP 与 STL 是
    同一零件的**两套不同三角化**，顶点位置天然对不上。按"到最近顶点"量，
    量到的其实是参考点云的采样间距，不是错位。同一批 5000 点两种口径实测：

      6.stl/6.stp  记录 0.1334 → 到顶点 0.1666 → **到曲面 0.0042** mm
                   参考点间距中位数 0.1018 mm
      1.stl/2.stp  记录 0.7598 → 到顶点 0.7564 → **到曲面 0.0054** mm
                   参考点间距中位数 0.5001 mm

    0.13~0.76 mm 全部是采样口径造出来的。所以新增素材 5 对全被压到 review，
    真正原因不是配准不准，而是这个指标量错了对象。

    仍截到 5000 点：曲面距离的统计特征在这个量级已经收敛，再多只是更慢。
    没有三角面的纯点云退回"到顶点"——那种输入本来就没有曲面可量。
    """

    sample = _limit_points(candidate.vertices, 5000)
    if len(reference.faces):
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray(reference.vertices, dtype=np.float64),
            faces=np.asarray(reference.faces, dtype=np.int64),
            process=False,
        )
        _, distances, _ = trimesh.proximity.closest_point(mesh, sample)
        return np.asarray(distances, dtype=np.float64)
    _, distances = _nearest_points(
        sample,
        _limit_points(reference.vertices, 5000),
    )
    return distances


def _symmetric_nearest_score(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> float:
    """我用双向最近邻残差区分 PCA 候选姿态，而不是只比较半径。"""

    # 原理说明：单向最近邻会漏掉"源覆盖不到目标某些区域"的情况，所以我
    # 正反各算一次再合并。合并后取"均值 + 0.25 × P95"，均值反映整体贴合，
    # P95 惩罚局部大偏差，避免某个轴对了但整体错位却被均值掩盖。

    if len(source_points) == 0 or len(target_points) == 0:
        return float("inf")
    _, source_distances = _nearest_points(source_points, target_points)
    _, target_distances = _nearest_points(target_points, source_points)
    combined = np.concatenate([source_distances, target_distances])
    return float(np.mean(combined) + 0.25 * np.percentile(combined, 95))


def _centroid_align(
    candidate: TriangleMesh,
    reference: TriangleMesh,
    config: InspectionConfig,
) -> tuple[TriangleMesh, RegistrationResult]:
    """质心配准只平移不旋转：两个模型只是整体错位时这一步就够了。

    我为什么把它抽成一个函数、并且让 `centroid` 分支也走它：
    ICP 那条"可能吃掉缺陷"的告警需要一个**可比参照系**来判断
    "ICP 额外抹平了多少"。如果 `centroid` 分支自己内联一份，
    两条路径的残差算法就可能不一致，而那条告警会跟着开始说谎。
    """

    rotation = np.eye(3)
    scale = _estimate_scale(candidate, reference) if config.allow_scale else 1.0
    # 平移量 = 参考质心 - 候选质心 × 缩放：
    # 先缩放再对齐质心，顺序反了会得到错误的平移量。
    translation = reference.centroid - candidate.centroid * scale
    aligned = candidate.transformed(rotation, translation, scale)
    residuals = _registration_residuals(aligned, reference)
    return aligned, _make_result("centroid", rotation, translation, scale, residuals)


def _make_result(
    mode: str,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
    residuals: np.ndarray,
) -> RegistrationResult:
    """我把配准参数和残差汇总成质量报告，并给出分级告警。

    原理说明：质量分级用 P95 而不是最大残差——最大残差对单个离群点过于
    敏感，会把整体对齐良好的结果判成坏。分级阈值：
       P95 <= 0.10 mm  -> good
       0.10 < P95 <= 1.0 -> warning
       P95 > 1.0 mm    -> bad（不足以支撑 0.15mm 级判定）
    另外只要整体缩放明显偏离 1（相对误差 > 2%），就单独挂一条"单位差异
    可能被缩放掩盖"的告警，要求人工确认，而不是默默接受。
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    mean_residual = float(residuals.mean()) if len(residuals) else 0.0
    p95_residual = float(np.percentile(residuals, 95)) if len(residuals) else 0.0
    quality = "good"
    warnings: list[str] = []
    scale_review_required = bool(
        not np.isclose(float(scale), 1.0, rtol=0.02, atol=1e-6)
    )
    if scale_review_required:
        warnings.append(
            "我发现配准使用了明显非 1 的整体缩放因子 "
            f"{float(scale):.9g}；请确认单位差异没有被缩放掩盖。"
        )
    if p95_residual > 0.10:
        quality = "warning"
        warnings.append(
            "我发现配准 P95 残差大于 0.10 mm，请确认单位、姿态和参考模型。"
        )
    if p95_residual > 1.0:
        quality = "bad"
        warnings.append("我认为当前配准质量不足以直接进行 0.15 mm 级别判定。")
    return RegistrationResult(
        mode=mode,
        rotation_matrix=np.asarray(rotation).tolist(),
        translation_mm=np.asarray(translation).tolist(),
        scale=float(scale),
        mean_residual_mm=mean_residual,
        p95_residual_mm=p95_residual,
        quality=quality,
        warnings=warnings,
        scale_review_required=scale_review_required,
    )
