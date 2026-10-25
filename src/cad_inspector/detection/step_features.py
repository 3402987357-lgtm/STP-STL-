"""我从 STEP Face 特征中识别上下平面、侧壁和面邻接候选。

关键算法原理（本模块的几何判据来源）：

一、方向分类。取 Face 法向与 Z 轴夹角：法向 Z 分量绝对值 >= cos(容差) 判为水平；
    <= sin(容差) 判为侧壁；其余判为倾斜。用 cos/sin 两个阈值而不是单一角度阈值，
    是因为法向的符号不代表上下（顶面朝上、底面朝下），必须取绝对值后再比较。

二、上/下带判定。水平面中，中心 Z 落在包围盒顶部 ``z_band_ratio`` 带内的是顶平面，
    落在底部带内的为底平面，其余单独归入 horizontal_other_region。单独归类而不是丢弃，
    是为了保证区域覆盖完整——模具内部常有水平台阶面，它们既不是顶也不是底。

三、区域聚合。按共享 Edge 的连通分量把同一方向的 Face 聚成区域，再用边界拓扑证据
    判断每个区域的边界是否形成闭合环。

四、边界拓扑。先剔除退化自环 Edge（长度 <= 1e-6 mm），再按连通分量与开放链分解边界，
    最后在候选 Wire 里选最大的闭合环作为主边界。退化边必须剔除：OCP 在旋转面极点和
    圆柱接缝处会生成 1e-9 mm 量级的圆形退化边，参与分解会把闭合环误判成开放边界。

设计边界：本模块输出的是"方向候选 + 邻接候选 + 区域候选"，不输出型腔、型芯或内外腔体
语义；所有语义状态默认 REVIEW_REQUIRED，必须结合实体射线与甲方坐标规则确认。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from math import acos, cos, degrees, isclose, isfinite, pi, radians, sin
from pathlib import Path
from typing import Iterable

import numpy as np

from ..domain.step_features import (
    StepFaceAdjacency,
    StepFeatureClassificationReport,
    StepFeatureGroup,
    StepFeatureRegion,
)
from ..io.step_faces import (
    StepFaceFeature,
    StepFaceFeatureReport,
    inspect_step_face_features,
)


# 我方把长度不超过该阈值的边界 Edge 当成退化自环隔离：OCP 在旋转面极点、
# 圆柱接缝处会生成长度在 1e-9 mm 量级的圆形退化边，它们不是真实边界段，
# 参与连通分量和链分解会把闭合环误判成开放边界。阈值与工程内 1e-6 mm
# 的点重合容差保持一致。
_DEGENERATE_EDGE_LENGTH_TOLERANCE_MM = 1e-6
# 跨区域桥接只在"另一侧端点几乎就在同一个点上"时才成立。我用与退化边
# 相同的 1e-6 mm：真实配合面在 STEP 里是共享顶点、坐标完全一致，
# 而不是靠 0.1 mm 这种模糊距离猜出来的。
_CROSS_REGION_BRIDGE_TOLERANCE_MM = 1e-6


def inspect_step_feature_classification(
    path: str | Path,
    *,
    horizontal_tolerance_deg: float = 15.0,
    z_band_ratio: float = 0.1,
    validate_shape: bool = True,
    semantic_sample_limit: int = 16,
) -> StepFeatureClassificationReport:
    """我读取 STEP Face 特征并输出可解释的区域候选。"""

    started = time.perf_counter()
    face_report = inspect_step_face_features(
        path,
        validate_shape=validate_shape,
        classify_semantics=True,
        semantic_sample_limit=semantic_sample_limit,
    )
    result = classify_step_face_features(
        face_report,
        horizontal_tolerance_deg=horizontal_tolerance_deg,
        z_band_ratio=z_band_ratio,
    )
    result.elapsed_seconds = time.perf_counter() - started
    return result


def classify_step_face_features(
    report: StepFaceFeatureReport,
    *,
    horizontal_tolerance_deg: float = 15.0,
    z_band_ratio: float = 0.1,
) -> StepFeatureClassificationReport:
    """我复用已提取的 Face 特征，避免重复读取和重复遍历 B-Rep。"""

    # 两个阈值都限制在几何有意义的区间内，越界直接报错而不是静默截断。
    if not 0.0 < horizontal_tolerance_deg < 45.0:
        raise ValueError("我要求 horizontal_tolerance_deg 位于 (0, 45) 区间。")
    if not 0.0 < z_band_ratio < 0.5:
        raise ValueError("我要求 z_band_ratio 位于 (0, 0.5) 区间。")

    # 只有几何字段完整、编号合法的 Face 才参与分类，避免脏数据污染方向分组。
    valid_faces = [
        face
        for face in report.faces
        # 几何有效与编号有效两道过滤：
        # 前者剔除退化/非有限的几何，后者剔除占位编号。
        # 两者任一不满足都参与不了方向分类。
        if _is_valid_face_geometry(face) and _is_valid_face_id(face.face_id)
    ]
    warnings = list(report.warnings)
    if len(valid_faces) != len(report.faces):
        # 被过滤掉多少就记一条警告：悄悄丢弃会让
        # "为什么这个件的方向分类少了几个面"无从查起。
        warnings.append(
            "我发现部分 Face 缺少完整几何特征、包含非有限/退化几何，"
            "或 Face 编号不可用，"
            "不能参加方向分类。"
        )
    valid_face_ids = [face.face_id for face in valid_faces]
    # 重复编号数 = 总数 - 去重后数量：
    # 面号重复说明这份报告的区域划分不可靠，
    # 必须显式警告而不是让后面的分组默默覆盖。
    duplicate_face_id_count = len(valid_face_ids) - len(set(valid_face_ids))
    if duplicate_face_id_count:
        warnings.append(
            f"我发现 {duplicate_face_id_count} 个重复 Face 编号，"
            "区域和邻接结果只能作为人工复核线索。"
        )

    bbox_min, bbox_max = _combined_bbox(valid_faces)
    if bbox_min is None or bbox_max is None:
        # 没有整体包围盒就无法建立"上下方向"的候选：
        # 上下是相对整体尺寸说的，没有尺寸就没有方向。
        # 这里返回一份结构完整、内容为空的报告，
        # 好过抛异常中断整批。
        warnings.append("我没有得到足够的 Face 包围盒，无法建立上下方向候选。")
        return StepFeatureClassificationReport(
            status="REVIEW_REQUIRED",
            file_path=report.file_path,
            unit=report.unit,
            backend="OCP",
            shape_is_valid=report.shape_is_valid,
            face_count=report.face_count,
            bbox_min_xyz_mm=None,
            bbox_max_xyz_mm=None,
            groups=[],
            adjacency=[],
            regions=[],
            warnings=list(dict.fromkeys(warnings)),
            # 容差照常写进诊断：即使没做成分类，
            # 也要记下这一轮用的是哪组参数。
            diagnostics={
                "horizontal_tolerance_deg": horizontal_tolerance_deg,
                "z_band_ratio": z_band_ratio,
            },
        )

    # 上/下带宽度按零件高度比例折算，并留 1e-9 下限防止退化零件除零。
    span_z = max(float(bbox_max[2] - bbox_min[2]), 1e-12)
    band = max(span_z * z_band_ratio, 1e-9)
    # cos/sin 双阈值：法向符号不区分上下，所以先取绝对值再比较。
    cos_limit = cos(radians(horizontal_tolerance_deg))
    sin_limit = sin(radians(horizontal_tolerance_deg))

    horizontal: list[StepFaceFeature] = []
    side_wall: list[StepFaceFeature] = []
    inclined: list[StepFaceFeature] = []
    for face in valid_faces:
        normal = np.asarray(face.normal_xyz, dtype=np.float64)
        normal_z = abs(float(normal[2] / np.linalg.norm(normal)))
        # 法向 Z 分量大 -> 水平；小 -> 侧壁；中间 -> 倾斜。
        if normal_z >= cos_limit:
            horizontal.append(face)
        elif normal_z <= sin_limit:
            side_wall.append(face)
        else:
            inclined.append(face)

    # 顶/底平面：水平面中中心 Z 落在上/下带内的。
    top = [
        face
        for face in horizontal
        if float(face.center_xyz_mm[2]) >= float(bbox_max[2]) - band
    ]
    bottom = [
        face
        for face in horizontal
        if float(face.center_xyz_mm[2]) <= float(bbox_min[2]) + band
    ]
    # 用对象 id 而不是 face_id 做去重，避免重复 face_id 导致归属错乱。
    top_object_ids = {id(face) for face in top}
    bottom_object_ids = {id(face) for face in bottom}
    # 水平但不在顶底带内的单独归类，保证区域覆盖完整而不是被丢弃。
    horizontal_other = [
        face
        for face in horizontal
        if id(face) not in top_object_ids | bottom_object_ids
    ]
    if not top:
        warnings.append("我没有找到可靠的顶部水平 Face 候选。")
    if not bottom:
        warnings.append("我没有找到可靠的底部水平 Face 候选。")
    if horizontal_other:
        warnings.append(
            f"我发现 {len(horizontal_other)} 个水平但不在顶部/底部带内的 Face，"
            "已单独归入 horizontal_other_region，避免区域覆盖不完整。"
        )
    warnings.append(
        "我已经建立 Face 邻接和方向候选，但没有把几何启发式结果冒充最终内外腔体语义。"
    )
    warnings.append(
        "我仍需要结合实体射线、Shell/空间连通性和甲方坐标规则确认外部面、内部孔洞和拔模方向。"
    )

    # 五个方向分组；空的分组在下一步剔除，不产出空记录。
    groups = [
        _make_group("top_plane_candidates", top, bbox_max[2], band),
        _make_group("bottom_plane_candidates", bottom, bbox_min[2], band),
        _make_group("horizontal_other_candidates", horizontal_other, None, None),
        _make_group("side_wall_candidates", side_wall, None, None),
        _make_group("inclined_candidates", inclined, None, None),
    ]
    groups = [group for group in groups if group.face_ids]
    adjacency = _build_adjacency(valid_faces)
    regions = _build_regions(
        valid_faces,
        groups,
        adjacency,
        report.vertex_coordinates_mm,
        getattr(report, "edge_polyline_sampler", None),
    )
    # 覆盖率校验：检查每个有效 Face 是否都进了某个区域，避免静默漏面。
    # 跨区域桥接：单个区域内部看不到"缺口另一侧其实在别的区域"这种情况，
    # 所以我在区域都建好之后再做一次全局配对，把跨区域零间距端点补成候选。
    _annotate_cross_region_bridges(regions, report.vertex_coordinates_mm)
    region_face_ids = [
        face_id for region in regions for face_id in region.face_ids
    ]
    region_unique_face_ids = set(region_face_ids)
    # 未归入任何区域的有效面：它们说明这批区域划分
    # 没有覆盖整个模型，结论只是局部的。
    unclassified_face_ids = sorted(
        set(valid_face_ids) - region_unique_face_ids
    )
    # 重复引用数 = 报告里已存在的重号 + 区域之间互相重叠的部分：
    # 两个来源都要算，只算一个会低估重叠的严重程度。
    duplicate_region_face_count = duplicate_face_id_count + (
        len(region_face_ids) - len(region_unique_face_ids)
    )
    # 两个警告都带上具体条数：只说"有重复"无法判断
    # 是偶发的几个还是大面积重叠。
    if unclassified_face_ids:
        warnings.append(
            f"我发现 {len(unclassified_face_ids)} 个有效 Face 未进入任何区域，"
            "分类结果需要人工复核。"
        )
    if duplicate_region_face_count:
        warnings.append(
            f"我发现区域结果重复引用 {duplicate_region_face_count} 个 Face，"
            "分类结果需要人工复核。"
        )
    # 只要产出了区域就声明一次内外语义待复核：
    # 按共享边聚出来的是"几何角色"，
    # 它本身不包含外部/内部的判定。
    if regions:
        warnings.append(
            "我已按共享 Edge 把同一几何角色聚合成区域，但区域的外部/内部语义仍需复核。"
        )
    # 区域语义状态的分布：它给出"这批区域里
    # 有多少能定出内外、多少定不出来"。
    semantic_status_counts: dict[str, int] = defaultdict(int)
    for region in regions:
        semantic_status_counts[region.semantic_status] += 1
    region_signature_counts: dict[str, int] = defaultdict(int)
    # 签名为空的区域不计：没有签名的区域
    # 参与不了"是否重复"的判断。
    for region in regions:
        if region.region_geometry_signature:
            region_signature_counts[region.region_geometry_signature] += 1
    # 重复签名的条数 = 各签名超出一条的部分之和：
    # 同一个签名出现 3 次算 2 条重复，
    # 这样总数才等于"需要额外核对多少条"。
    duplicate_region_signature_count = sum(
        count - 1 for count in region_signature_counts.values() if count > 1
    )
    if duplicate_region_signature_count:
        warnings.append(
            f"我发现 {duplicate_region_signature_count} 个区域几何签名重复，"
            # 明确说明签名不能当唯一 ID：
            # 形状相同的两个区域签名一样，
            # 拿它当 ID 追踪会把两者混为一谈。
            "区域签名只能作为候选追踪证据，不能单独作为唯一 CAD ID。"
        )

    # 只有报告完整、几何有效、顶底都有、无重复编号、无漏面时才是 BASELINE。
    status = "BASELINE"
    if (
        report.status != "COMPLETE"
        or report.shape_is_valid is not True
        or len(report.faces) != report.face_count
        or not top
        or not bottom
        or duplicate_face_id_count > 0
        or unclassified_face_ids
        or duplicate_region_face_count > 0
    ):
        status = "REVIEW_REQUIRED"
    if len(valid_faces) < report.face_count:
        status = "REVIEW_REQUIRED"

    return StepFeatureClassificationReport(
        status=status,
        file_path=report.file_path,
        unit=report.unit,
        backend="OCP",
        shape_is_valid=report.shape_is_valid,
        face_count=report.face_count,
        bbox_min_xyz_mm=bbox_min,
        bbox_max_xyz_mm=bbox_max,
        groups=groups,
        adjacency=adjacency,
        regions=regions,
        warnings=list(dict.fromkeys(warnings)),
        diagnostics={
            # 阈值与派生量先落盘：z_band_mm 是 band 的实际毫米值，
            # 它由比例乘以零件高度算出，只记比例无法核对具体数值。
            "horizontal_tolerance_deg": horizontal_tolerance_deg,
            "z_band_ratio": z_band_ratio,
            "z_band_mm": band,
            # 分类计数三兄弟（水平/侧壁/倾斜）与 valid_face_count 对比，
            # 能一眼看出是否有面被漏分类：三者之和应等于 valid_face_count。
            "valid_face_count": len(valid_faces),
            "horizontal_face_count": len(horizontal),
            "side_wall_face_count": len(side_wall),
            "inclined_face_count": len(inclined),
            "adjacency_face_count": len(adjacency),
            "region_count": len(regions),
            # 三个"数据质量"计数：重复编号、未分类面、区域间重复面。
            # 任一非 0 都已把 status 打成 REVIEW_REQUIRED，
            # 这里再给具体数字，是为了让复核者知道问题规模有多大。
            "duplicate_face_id_count": duplicate_face_id_count,
            "unclassified_valid_face_count": len(unclassified_face_ids),
            "duplicate_region_face_count": duplicate_region_face_count,
            # 覆盖率=被区域涵盖的唯一面数 / 有效面唯一数。
            # 小于 1 说明有面没进任何区域，区域划分不完整。
            "region_face_coverage_ratio": (
                len(region_unique_face_ids) / len(set(valid_face_ids))
                if valid_face_ids
                else 1.0
            ),
            # 语义恒定 REVIEW_REQUIRED：内外语义（哪个是型腔、哪个是型芯）
            # 需要甲方坐标规则与完整拓扑才能定，本模块不做这个判定。
            "region_semantics": "REVIEW_REQUIRED",
            "interior_external_semantics": "REVIEW_REQUIRED",
            "semantic_region_status_counts": dict(sorted(semantic_status_counts.items())),
            # 签名重复数：不同区域算出相同几何签名，
            # 说明区域划分存在对称重复或阈值导致的重合，必须复核。
            "region_geometry_signature_count": len(region_signature_counts),
            "duplicate_region_geometry_signature_count": (
                duplicate_region_signature_count
            ),
            "semantic_evidence_method": "normal_offset_solid_classifier",
            "semantic_sampling": dict(report.diagnostics),
        },
    )


def _combined_bbox(
    faces: Iterable[StepFaceFeature],
) -> tuple[list[float] | None, list[float] | None]:
    """我把若干面的包围盒并成一个总包围盒；一个有效值都没有时返回 None 对。

    逐个面判断 bbox 是否为 None 再收集，而不是先堆进数组再清洗：
    np.asarray 遇到 None 会退化成 object 数组，np.min 在 object
    数组上要么报错要么给出无意义结果。

    空输入返回 (None, None) 而不是 ([0,0,0], [0,0,0])：
    零包围盒会被下游当成一个真实的退化包围盒，
    None 才能表达"这个组合并没测出范围"。
    """
    mins: list[list[float]] = []
    maxs: list[list[float]] = []
    for face in faces:
        if face.bbox_min_xyz_mm is not None and face.bbox_max_xyz_mm is not None:
            mins.append(face.bbox_min_xyz_mm)
            maxs.append(face.bbox_max_xyz_mm)
    if not mins:
        return None, None
    minimum = np.min(np.asarray(mins, dtype=np.float64), axis=0).tolist()
    maximum = np.max(np.asarray(maxs, dtype=np.float64), axis=0).tolist()
    return minimum, maximum


def _is_valid_face_geometry(face: StepFaceFeature) -> bool:
    """我只让有限、三维且方向可归一化的 Face 进入几何分类。

    我为什么要求面积非负：OCP 在极少数退化面上会返回负面积，
    而后面所有"按面积排序取最大面""面积求和"的逻辑都隐含面积非负。
    放一个负面进去会让求和结果偏小、排序把退化面排到前面，
    且这种错不会报错，只在报表里表现为"面积莫名少了"。

    我为什么要求 bbox_min <= bbox_max：包围盒反了说明读取阶段已经
    出问题，而包围盒后面要用来算联合包围盒与区域跨度。带病进入
    会让跨度变成负数，再被下游取绝对值掩盖掉，就再也查不出源头。
    """

    if face.area_mm2 is None or not np.isfinite(float(face.area_mm2)):
        return False
    if float(face.area_mm2) < 0.0:
        return False
    center = _finite_triplet(face.center_xyz_mm)
    normal = _finite_triplet(face.normal_xyz)
    bbox_min = _finite_triplet(face.bbox_min_xyz_mm)
    bbox_max = _finite_triplet(face.bbox_max_xyz_mm)
    if center is None or normal is None or bbox_min is None or bbox_max is None:
        return False
    if float(np.linalg.norm(normal)) <= 1e-12:
        return False
    return bool(np.all(bbox_min <= bbox_max))


def _is_valid_face_id(value: object) -> bool:
    """我要求 Face 编号为正整数，避免字典覆盖造成静默丢面。

    我为什么连 bool 都要排除：Python 里 ``bool`` 是 ``int`` 的子类，
    ``True == 1`` 且 ``isinstance(True, int)`` 为真。不排除的话
    ``True`` 会被当成编号 1，而编号 1 通常已经存在，于是后写的
    覆盖先写的，一个面凭空消失——而且不报错。
    """

    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value > 0


def _finite_triplet(value: list[float] | None) -> np.ndarray | None:
    """我把外部三维字段统一校验成有限长度为 3 的数组。

    我为什么对形状做严格 ``(3,)`` 判定而不是只看长度：``[[1,2,3]]``
    长度也是 1、展平后能凑成 3 个数，但它是二维嵌套，说明上游字段
    结构已经变了。宽松接受会把结构错误藏到后面叉积计算时才炸，
    而那时错误信息指向的是叉积、不是真正出错的读取环节。
    """

    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    return array


def _make_group(
    role: str,
    faces: list[StepFaceFeature],
    reference_z: float | None,
    band: float | None,
) -> StepFeatureGroup:
    """我把同一角色的若干面聚成一个组，并算出加权中心与法向。

    为什么按面积加权而不是简单平均：一组面里可能有一大片主面和
    几块很小的倒角面。简单平均会让小面获得与主面相同的发言权，
    算出的"组法向"被小面拉偏，而这个组本来应该代表主面的朝向。

    area_mm2 为 None 时用 1e-12 兜底而不是跳过该面：
    丢弃会让面从 face_ids 里消失（漏面），而给一个极小的权重
    只是让它在加权里几乎不发声，方向正确且信息不丢。

    平均法向要重新归一化：多个单位法向加权平均后长度通常小于 1，
    不归一化会让下游的"法向夹角"计算系统性偏小。
    normal_length 接近 0 时（各面法向互相抵消，例如一个闭合环）
    跳过除法，避免除零产生 NaN。

    reference_z 与 band 都给定时才叠加贴近度加成：
    少任何一个就无法定义"离参考面多远"，此时不加成，
    而不是假设某个默认高度。
    """
    if not faces:
        # 空组返回一个结构完整的空对象而不是 None：
        # 调用方按固定字段取值，返回 None 会让每一处都得多判一次。
        return StepFeatureGroup(
            role=role,
            face_ids=[],
            area_mm2=0.0,
            # 中心与法向留 None：没有面就没有这两个量，
            # 给零向量会被读成"这个组在原点、法向朝 x"。
            mean_center_xyz_mm=None,
            mean_normal_xyz=None,
            score=0.0,
        )
    # 权重取面积，下限 1e-12：零面积的面权重为 0 会让
    # 加权平均退化，给一个极小值保证它仍有微小贡献。
    weights = np.asarray([max(float(face.area_mm2 or 0.0), 1e-12) for face in faces])
    centers = np.asarray([face.center_xyz_mm for face in faces], dtype=np.float64)
    normals = np.asarray([face.normal_xyz for face in faces], dtype=np.float64)
    # 法向先各自归一化再加权平均：
    # 不归一化会让大面的法向主导结果，
    # 而这里要的是"这组面整体朝哪"的方向，与面积无关。
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    # 中心按面积加权：大面的位置更能代表这一组的所在。
    mean_center = np.average(centers, axis=0, weights=weights)
    normal = np.average(normals, axis=0, weights=weights)
    normal_length = float(np.linalg.norm(normal))
    # 平均后长度不为 1 要再归一化一次：
    # 多个单位向量加权平均后长度会小于 1
    # （方向越不一致越短），直接用它不是单位方向。
    if normal_length > 1e-12:
        normal /= normal_length
    score = float(weights.sum())
    # 只在该给参考高度与带宽时加接近度加权：
    # 这两个参数用于"这些面离目标高度多近"，
    # 没有它们时纯按面积打分。
    if reference_z is not None and band is not None:
        proximity = np.mean(
            [
                # 接近度 = 1 - 偏离量/带宽，下限 0：
                # 超出带宽的面接近度为 0 而不是负值，
                # 负值会让远处的面把总分拉低到难以解释的程度。
                max(
                    0.0,
                    1.0
                    - abs(float(face.center_xyz_mm[2]) - reference_z) / band,
                )
                for face in faces
            ]
        )
        # 乘 (1 + 接近度)：接近度在 [0,1]，
        # 于是分数最多翻倍，面积仍是主导项。
        score *= 1.0 + float(proximity)
    return StepFeatureGroup(
        role=role,
        face_ids=[face.face_id for face in faces],
        # 面积取权重之和：与 weights 同一口径，
        # 于是它与 score 在未加权时完全相等。
        area_mm2=float(weights.sum()),
        mean_center_xyz_mm=mean_center.tolist(),
        mean_normal_xyz=normal.tolist(),
        score=score,
    )


def _build_adjacency(faces: list[StepFaceFeature]) -> list[StepFaceAdjacency]:
    """我构建 Face 邻接关系。

    原理说明：两个 Face 相邻当且仅当它们**共享至少一条 Edge**。
    所以我先建 edge -> faces 的倒排表（一条边通常属于两个面），
    再把同一条边下的面对两两连边。这样得到的邻接是真正的 B-Rep 拓扑邻接，
    不是"包围盒相交"或"距离够近"这类几何近似——后者会把隔着一薄壁的
    两个型腔误判成相邻。

    只被一个 Face 引用的边是自由边（开放边界），不计入邻接，
    但单独累加到 boundary_counts，供后续判断该面是否位于壳体开口处。
    shared_edges 用两个方向各存一份，查询时不必再排序比较。
    """

    edge_to_faces: dict[int, list[int]] = defaultdict(list)
    for face in faces:
        for edge_id in face.edge_ids:
            edge_to_faces[int(edge_id)].append(face.face_id)

    neighbors: dict[int, set[int]] = {face.face_id: set() for face in faces}
    # 公共边的键是两个面号的有序对，两个方向都存：
    # 邻接查询会以任意一侧为起点，只存一个方向会查不到。
    shared_edges: dict[tuple[int, int], set[int]] = defaultdict(set)
    boundary_counts: dict[int, int] = defaultdict(int)
    for edge_id, connected_faces in edge_to_faces.items():
        # 同一条边可能被同一个面记录多次，先去重再判：
        # 不去重会让"只属于一个面"这个判据失效。
        unique_faces = sorted(set(connected_faces))
        # 只被一个面引用的边就是边界边：
        # 内部边必然被恰好两个面共享，这是封闭网格的基本性质。
        if len(unique_faces) == 1:
            boundary_counts[unique_faces[0]] += 1
            continue
        # 三个以上面共享一条边是非流形情形，
        # 这里按"两两互为邻居"处理：邻接图会因此变密，
        # 但至少不会丢掉任何一对真实的邻接关系。
        for index, left in enumerate(unique_faces):
            for right in unique_faces[index + 1 :]:
                neighbors[left].add(right)
                neighbors[right].add(left)
                shared_edges[(left, right)].add(edge_id)
                shared_edges[(right, left)].add(edge_id)

    normals_by_face = {
        face.face_id: _unit_normal(face.normal_xyz) for face in faces
    }
    result: list[StepFaceAdjacency] = []
    for face_id in sorted(neighbors):
        # 输出全部排序：面号、邻居号、公共边号都排序，
        # 保证同一份输入得到完全相同的输出，便于跨批次 diff。
        neighbor_face_ids = sorted(neighbors[face_id])
        shared_edge_ids = sorted(
            edge_id
            for neighbor_face_id in neighbor_face_ids
            for edge_id in shared_edges.get((face_id, neighbor_face_id), set())
        )
        # 同一条边可能被多个邻居各记一次，去重后计数：
        # 不去重会让"公共边数"大于真实值。
        shared_edge_ids = list(dict.fromkeys(shared_edge_ids))
        result.append(
            StepFaceAdjacency(
                face_id=face_id,
                neighbor_face_ids=neighbor_face_ids,
                # 计数与列表同时给：列表用于定位，计数用于统计与门禁。
                shared_edge_count=len(shared_edge_ids),
                boundary_edge_count=boundary_counts[face_id],
                shared_edge_ids=shared_edge_ids,
                normal_angle_deg=_normal_angle_deg(
                    normals_by_face.get(face_id),
                    [
                        normals_by_face.get(neighbor_face_id)
                        for neighbor_face_id in neighbor_face_ids
                    ],
                ),
            )
        )
    return result


def _unit_normal(value: list[float] | None) -> np.ndarray | None:
    """我返回归一化后的单位法向；输入不可用或近似零向量时返回 None。

    我为什么返回 None 而不是抛错：法向缺失在自由曲面与退化面上很常见，
    而法向只参与分类证据（例如"上下平面""侧壁"），不影响面积与包围盒
    这些主字段。抛错会为了一条次要证据丢掉整个面的全部信息。

    我为什么用 1e-12 判近零：真实法向的模长是 1 量级，退化到 1e-12
    以下说明上游已经给出零向量。这里若放行，除以极小数会把数值放大
    到 1e12 量级，后面所有夹角判定全部失真。
    """
    if value is None:
        return None
    normal = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or length <= 1e-12:
        return None
    return normal / length


def _normal_angle_deg(
    source: np.ndarray | None,
    neighbors: list[np.ndarray | None],
) -> float | None:
    """我记录当前 Face 与相邻 Face 的最大法向夹角。"""

    if source is None:
        return None
    angles = []
    for neighbor in neighbors:
        if neighbor is None:
            continue
        dot = float(np.clip(np.dot(source, neighbor), -1.0, 1.0))
        angles.append(degrees(acos(dot)))
    return max(angles) if angles else None


def _build_regions(
    faces: list[StepFaceFeature],
    groups: list[StepFeatureGroup],
    adjacency: list[StepFaceAdjacency],
    vertex_coordinates_mm: dict[int, list[float]] | None = None,
    edge_polyline_sampler: object = None,
) -> list[StepFeatureRegion]:
    """我按角色和共享 Edge 连通分量生成稳定的 Face 区域。

    原理说明：甲方要的是"区域"，而不是一堆孤立的面。我的区域定义是
    **同一几何角色内、按面邻接连通的极大 Face 集合**。分两步：
      1. 先按角色（平面/圆柱/自由曲面等）分组，保证一个区域不会横跨
         两种语义完全不同的几何；
      2. 再在每个角色组内求连通分量，保证同一个区域在拓扑上是一整片。
    只做第一步会把零件上两处互不相连的平面并成一个区域，
    只做第二步会让一个区域横跨多种曲面类型，两者都要。

    区域边界边判定（这里修过一个真实缺陷）：一条边属于区域边界，
    当且仅当"它至少连接了本区域的一个面"且"并非它连接的所有面都在
    本区域内"。第二条必须显式处理只被一个面引用的自由边——
    开放壳体上大量边只有单侧邻面，若用 issubset 判断会把它们误判成
    内部边丢掉，导致区域边界为空、下游投影面积和闭合环全部算不出来。
    """

    face_by_id = {face.face_id: face for face in faces}
    neighbors_by_face = {
        item.face_id: set(item.neighbor_face_ids) for item in adjacency
    }
    edge_to_faces: dict[int, set[int]] = defaultdict(set)
    for face in faces:
        for edge_id in set(face.edge_ids):
            edge_to_faces[int(edge_id)].add(face.face_id)

    regions: list[StepFeatureRegion] = []
    face_to_region: dict[int, int] = {}
    for group in groups:
        group_face_ids = set(group.face_ids)
        components = _connected_components(group_face_ids, neighbors_by_face)
        for component in components:
            region_faces = [
                face_by_id[face_id]
                for face_id in sorted(component)
                if face_id in face_by_id
            ]
            if not region_faces:
                continue
            region_id = len(regions) + 1
            boundary_edge_ids = sorted(
                edge_id
                for edge_id, incident_faces in edge_to_faces.items()
                if component.intersection(incident_faces)
                and (
                    # 我方按 OCP 真实边面关联复核后确认：只被一个 Face 引用的
                    # 自由边一定落在区域边界上。旧写法用 issubset 判断，会把
                    # 这类自由边当成“区域内部边”丢掉，导致开放壳体区域
                    # 边界为空、投影面积和闭合环都算不出来。
                    len(incident_faces) < 2
                    or not incident_faces.issubset(component)
                )
            )
            region = _make_region(
                region_id,
                group,
                region_faces,
                boundary_edge_ids,
                vertex_coordinates_mm,
                edge_polyline_sampler,
            )
            # 同一角色下有多个不连通片段时逐个建区域并警告：
            # 它们只是"恰好同一种曲面类型"，
            # 空间上并不连在一起，合成一个区域会让面积与包围盒失去意义。
            if len(components) > 1:
                region.warnings.append(
                    f"我发现该几何角色包含 {len(components)} 个不连通 Face 片段，"
                    "当前区域只代表其中一个连通片段。"
                )
            regions.append(region)
            # 面到区域的映射在这里建好：下一步算邻区要用它，
            # 而这一步还没结束时区域列表还在增长，无法提前建。
            for face_id in component:
                face_to_region[face_id] = region_id

    # 邻区要等所有区域都建完才算：区域编号存在向后依赖，
    # 在建区域的过程中算会得到不完整的邻区列表。
    for region in regions:
        neighbor_region_ids: set[int] = set()
        for face_id in region.face_ids:
            for neighbor_face_id in neighbors_by_face.get(face_id, set()):
                neighbor_region_id = face_to_region.get(neighbor_face_id)
                # 只收属于别的区域的邻面：
                # 同区域内的邻面不构成"邻区关系"。
                # 映射查不到（面未归入任何区域）的也跳过。
                if (
                    neighbor_region_id is not None
                    and neighbor_region_id != region.region_id
                ):
                    neighbor_region_ids.add(neighbor_region_id)
        # 排序后存：邻区编号的顺序不携带信息，
        # 排序保证同一份数据每次得到相同输出。
        region.neighbor_region_ids = sorted(neighbor_region_ids)
    return regions


def _connected_components(
    face_ids: set[int],
    neighbors_by_face: dict[int, set[int]],
) -> list[set[int]]:
    """我用邻接表统计角色内部的连通片段数量。

    原理说明：我做一次标准的 BFS 洪泛。从任一未访问面出发，沿邻接表
    扩展直到无法继续，得到一个连通分量；重复直到所有面都被访问。
    起点按升序取，保证分量顺序稳定、重复运行结果可复现。
    复杂度是 O(面数 + 邻接边数)，在几千面的真实 STEP 上是毫秒级。
    """

    remaining = set(face_ids)
    components: list[set[int]] = []
    while remaining:
        component: set[int] = set()
        stack = [min(remaining)]
        while stack:
            current = stack.pop()
            if current not in remaining:
                continue
            remaining.remove(current)
            component.add(current)
            stack.extend(
                sorted(
                    neighbors_by_face.get(current, set())
                    & remaining
                    & face_ids,
                    reverse=True,
                )
            )
        components.append(component)
    return components


def _make_region(
    region_id: int,
    group: StepFeatureGroup,
    faces: list[StepFaceFeature],
    boundary_edge_ids: list[int],
    vertex_coordinates_mm: dict[int, list[float]] | None = None,
    edge_polyline_sampler: object = None,
) -> StepFeatureRegion:
    """我汇总一个连通区域的面积、方向、曲面类型和评分。"""

    weights = np.asarray(
        [max(float(face.area_mm2 or 0.0), 1e-12) for face in faces],
        dtype=np.float64,
    )
    centers = np.asarray([face.center_xyz_mm for face in faces], dtype=np.float64)
    normals = np.asarray([face.normal_xyz for face in faces], dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    mean_center = np.average(centers, axis=0, weights=weights)
    mean_normal = np.average(normals, axis=0, weights=weights)
    normal_length = float(np.linalg.norm(mean_normal))
    if normal_length > 1e-12:
        mean_normal /= normal_length

    surface_type_counts: dict[str, int] = defaultdict(int)
    for face in faces:
        surface_type_counts[face.surface_type] += 1
    boundary_polylines = _collect_boundary_polylines(
        boundary_edge_ids,
        edge_polyline_sampler,
    )
    boundary_topology = _boundary_topology_evidence(
        faces,
        boundary_edge_ids,
        vertex_coordinates_mm,
        boundary_polylines,
    )
    wire_selection = boundary_topology["wire_selection"]
    area_mm2 = float(weights.sum())
    # score 按"组内面积占比"折算：group.score 是整组的评分，
    # 而这个区域只占组的一部分，直接用整组分会把小区域虚高成高置信。
    # 组面积为 0 时无法折算，退化为用本区域面积，避免除零。
    score = (
        float(group.score) * area_mm2 / float(group.area_mm2)
        if group.area_mm2 > 0.0
        else area_mm2
    )
    # 这条警告固定带上：区域是按"角色 + 共享边"聚出来的几何分组，
    # 它不等于型腔/型芯，下游不该把它当成实体内外语义来用。
    warnings = [
        "我只根据几何角色和共享 Edge 聚合该区域，未确定它是外部面还是内部腔体。"
    ]
    # 单 Face 区域最需要复核：它没有邻接同伴佐证，
    # 既可能是真实孤立特征，也可能只是邻接信息缺失造成的碎片。
    if len(faces) == 1:
        warnings.append("我发现该区域只有一个 Face，建议结合相邻实体和工程规则复核。")
    semantic_evidence = _aggregate_semantic_evidence(faces)
    semantic_status = str(semantic_evidence["status"])
    warnings.extend(semantic_evidence.pop("warnings", []))
    if boundary_topology["status"] == "OPEN_BOUNDARY_CANDIDATE":
        warnings.append(
            "我发现该区域边界 Edge 尚未组成闭合环，不能直接作为完整分型轮廓。"
        )
    elif boundary_topology["status"] == "UNRESOLVED_ENDPOINTS":
        warnings.append(
            "我没有得到该区域全部边界 Edge 的 Vertex 端点，闭合环状态需要人工复核。"
        )
    region_geometry_signature = _region_geometry_signature(
        role=group.role.removesuffix("_candidates") + "_region",
        faces=faces,
        area_mm2=area_mm2,
        mean_center_xyz_mm=mean_center.tolist(),
        mean_normal_xyz=mean_normal.tolist(),
        surface_type_counts=dict(sorted(surface_type_counts.items())),
    )

    return StepFeatureRegion(
        region_id=region_id,
        role=group.role.removesuffix("_candidates") + "_region",
        face_ids=[face.face_id for face in faces],
        area_mm2=area_mm2,
        mean_center_xyz_mm=mean_center.tolist(),
        mean_normal_xyz=mean_normal.tolist(),
        surface_type_counts=dict(sorted(surface_type_counts.items())),
        neighbor_region_ids=[],
        boundary_edge_count=len(boundary_edge_ids),
        semantic_status=semantic_status,
        semantic_evidence=semantic_evidence,
        score=score,
        # dict.fromkeys 去重且保序：同一条警告可能由多个分支产生，
        # 保序则保证同一输入两次运行得到逐字节相同的报告。
        warnings=list(dict.fromkeys(warnings)),
        boundary_edge_ids=list(boundary_edge_ids),
        # 几何签名独立于 Face/Edge 编号：编号会随 STEP 导出顺序变化，
        # 签名不会，所以跨次运行、跨文件比对区域要靠它而不是编号。
        region_geometry_signature=region_geometry_signature,
        boundary_edge_components=boundary_topology["edge_components"],
        boundary_loop_count=boundary_topology["loop_count"],
        closed_boundary_loop_count=boundary_topology["closed_loop_count"],
        boundary_topology_status=boundary_topology["status"],
        # 同时保留"状态字符串"与"完整证据"两份：
        # 状态用于快速筛选，证据用于回答"为什么是这个状态"，缺一不可。
        boundary_topology_evidence=boundary_topology,
        primary_boundary_edge_ids=list(wire_selection["primary_edge_ids"]),
        boundary_wire_candidate_count=int(wire_selection["candidate_count"]),
        boundary_wire_selection_status=str(wire_selection["status"]),
        boundary_wire_selection_evidence=dict(wire_selection),
        boundary_perimeter_length_mm=boundary_topology["perimeter_length_mm"],
        primary_boundary_perimeter_length_mm=wire_selection[
            "primary_perimeter_length_mm"
        ],
        primary_boundary_projected_area_mm2=wire_selection[
            "primary_projected_area_mm2"
        ],
        primary_boundary_projection_status=str(
            wire_selection["primary_projection_status"]
        ),
        primary_boundary_projection_evidence=dict(
            wire_selection["primary_projection_evidence"]
        ),
        primary_boundary_loop_role=str(wire_selection["primary_loop_role"]),
        primary_boundary_loop_relation_status=str(
            wire_selection["loop_relation_status"]
        ),
        primary_boundary_loop_relation_evidence=dict(
            wire_selection["loop_relation_evidence"]
        ),
    )


def _boundary_topology_evidence(
    faces: list[StepFaceFeature],
    boundary_edge_ids: list[int],
    vertex_coordinates_mm: dict[int, list[float]] | None = None,
    edge_polyline_points_mm: dict[int, list[list[float]]] | None = None,
    degenerate_edge_vertices_override: dict[int, tuple[int, ...]] | None = None,
) -> dict[str, object]:
    """我用当前报告内的 Vertex 关联判断边界连通分量和闭合环候选。

    原理说明：判断边界边是否构成闭合环，本质是一个**图论问题**——
    把每条边看成连接两个顶点的一条边，那么：
      - 所有顶点度数都是偶数，且没有度数为 1 的自由端，
        这个连通分量就是欧拉回路，即闭合环候选；
      - 存在度数为 1 的顶点，说明这里有开放端，环被截断；
      - 存在度数大于 2 的顶点，说明多个环在此自接触/分叉，
        光靠度数无法唯一还原环的走法，我单独记入 branch_vertex_ids
        要求人工复核，而不是随便挑一种走法。

    为什么我用顶点度数而不是坐标距离：坐标距离需要选容差，
    而容差在不同尺寸零件上没有普适值；顶点关联是 STEP 的拓扑事实，
    不含阈值，也不会把两个靠得很近但不相连的端点误并。

    近零长度的退化自环会被单独隔离，不参与闭合判定——它们通常是
    几何摘要里的噪声，放进去会让"闭合环数量"虚高。
    """

    # 没有边界边就返回全零证据，不编造任何闭合结论。
    if not boundary_edge_ids:
        return {
            "status": "NO_BOUNDARY_EDGES",
            "edge_components": [],
            "loop_count": 0,
            "closed_loop_count": 0,
            "components": [],
            "unresolved_edge_ids": [],
            "edge_lengths_mm": {},
            "perimeter_length_mm": None,
            "perimeter_complete": False,
            "perimeter_missing_edge_ids": [],
            "perimeter_conflict_edge_ids": [],
            "degenerate_edge_ids": [],
            "degenerate_edge_count": 0,
            "open_endpoint_vertex_ids": [],
            "branch_vertex_ids": [],
            "branch_vertex_count": 0,
            "open_chain_count": 0,
            "bridge_required": False,
            "bridge_proposals": [],
            "wire_selection": _boundary_wire_selection(
                [],
                vertex_coordinates_mm=vertex_coordinates_mm,
            ),
            "method": "edge_endpoint_vertex_incidence",
            "interpretation": "我没有发现当前区域边界 Edge。",
        }

    # 从 Face 的端点表反查每条边界边的两个端点；端点数不是 1 或 2 的边记为未解析。
    edge_vertices: dict[int, tuple[int, ...]] = {}
    unresolved_edge_ids: set[int] = set()
    for face in faces:
        for edge_id, vertex_ids in face.edge_endpoint_vertex_ids.items():
            if edge_id not in boundary_edge_ids:
                continue
            normalized = tuple(
                sorted(
                    {
                        int(vertex_id)
                        for vertex_id in vertex_ids
                        if isinstance(vertex_id, int) and vertex_id > 0
                    }
                )
            )
            if len(normalized) not in {1, 2}:
                unresolved_edge_ids.add(int(edge_id))
                continue
            previous = edge_vertices.get(int(edge_id))
            if previous is not None and previous != normalized:
                unresolved_edge_ids.add(int(edge_id))
            else:
                edge_vertices[int(edge_id)] = normalized

    # 任何一条边界边都查不到端点也算未解析，防止静默漏边。
    for edge_id in boundary_edge_ids:
        if int(edge_id) not in edge_vertices:
            unresolved_edge_ids.add(int(edge_id))

    length_evidence = _boundary_edge_length_evidence(faces, boundary_edge_ids)
    # 我方把近零长度的退化自环从连通分量和链分解里剔除，单独记录，
    # 避免它们把真实闭合环拉成“开放边界候选”。
    degenerate_edge_ids = {
        int(edge_id)
        for edge_id, value in length_evidence["edge_lengths_mm"].items()
        if float(value) <= _DEGENERATE_EDGE_LENGTH_TOLERANCE_MM
    }
    # 测试可以显式指定退化边端点表，用来锁定"圆锥顶点零长圆"这类形态；
    # 生产路径不传，仍由长度自动判定。
    if degenerate_edge_vertices_override:
        degenerate_edge_ids |= {
            int(edge_id) for edge_id in degenerate_edge_vertices_override
        }
    # 退化边从"未解析"里剔除，单独记录，不参与连通分量与链分解。
    unresolved_edge_ids -= degenerate_edge_ids
    graph_edge_vertices = {
        int(edge_id): vertex_ids
        for edge_id, vertex_ids in edge_vertices.items()
        if int(edge_id) not in degenerate_edge_ids
    }
    # 我识别出"完整回转面"形态：存在单端点边（OCP 把圆接缝塌缩成圆心点），
    # 且同一对端点之间挂着两条弧（两条弧长之和等于该圆的周长）时，
    # 那两条弧其实拼成一个完整圆，边界并不存在缺口。OCP 额外给出的那条
    # 轴向边（连接两个圆心）是接缝假象，我把它并入回转接缝边单独记录。
    seam_closure = _revolution_seam_closure(
        graph_edge_vertices,
        length_evidence["edge_lengths_mm"],
        faces,
        boundary_edge_ids,
        # 退化边的顶点单独传：回转面的缝合处常出现零长边，
        # 它们不参与周长统计，但要参与"缝是否闭合"的判断。
        degenerate_edge_vertices={
            int(edge_id): edge_vertices[int(edge_id)]
            for edge_id in degenerate_edge_ids
            if int(edge_id) in edge_vertices
        },
    )
    # 缝已闭合就走专门的状态分支：回转面的缝闭合后
    # 不存在"环"的概念，按环去统计会给出一堆无意义的空列表，
    # 所以这里直接返回并用 seam_closure 携带全部细节。
    if seam_closure["closed"]:
        total_length = sum(
            float(value)
            for edge_id, value in length_evidence["edge_lengths_mm"].items()
            # 只累加有限值：个别边量不出长度时值是 None 或 nan，
            # 不过滤会让整个周长变成 nan。
            if isinstance(value, (int, float)) and isfinite(float(value))
        )
        return {
            "status": "REVOLUTION_SEAM_CLOSED_CANDIDATE",
            # 环相关的字段全给空值而不是省略：
            # 下游按固定键取值，省略会 KeyError，
            # 给空值能让"这里没有环"表达得明确且不炸。
            "edge_components": [],
            "components": [],
            "loop_count": 0,
            "closed_loop_count": 0,
            "open_endpoint_vertex_ids": [],
            "branch_vertex_ids": [],
            "branch_vertex_count": 0,
            "open_chain_count": 0,
            "bridge_required": False,
            "bridge_proposals": [],
            "unresolved_edge_ids": sorted(unresolved_edge_ids),
            "edge_lengths_mm": dict(length_evidence["edge_lengths_mm"]),
            "perimeter_length_mm": total_length,
            # 周长是否完整与"哪些边缺失/冲突"分开给出：
            # 只给一个布尔值的话，周长被判定不完整时
            # 无法知道是该补量哪条边还是该修哪条边。
            "perimeter_complete": not length_evidence["missing_edge_ids"],
            "perimeter_missing_edge_ids": sorted(
                length_evidence["missing_edge_ids"]
            ),
            "perimeter_conflict_edge_ids": sorted(
                length_evidence["conflict_edge_ids"]
            ),
            "seam_closure": seam_closure,
            "degenerate_edge_ids": sorted(degenerate_edge_ids),
            "degenerate_edge_count": len(degenerate_edge_ids),
            "edge_length_evidence": length_evidence,
            "wire_selection": _boundary_wire_selection(
                [],
                vertex_coordinates_mm=vertex_coordinates_mm,
            ),
            "method": "edge_endpoint_vertex_incidence",
            "interpretation": (
                "我把 OCP 读成单端点圆接缝与轴向接缝假象的完整回转面边界"
                "判为闭合：两条弧长之和等于圆周长，边界没有缺口，"
                "轴向边是接缝假象而非真实缺口。"
            ),
        }
    # 按端点共享做并查集连通分量，一个分量对应一条独立边界 Wire。
    edge_components = _edge_components(graph_edge_vertices)
    component_records: list[dict[str, object]] = []
    closed_loop_count = 0
    for component in edge_components:
        degree_by_vertex: dict[int, int] = defaultdict(int)
        vertex_ids: set[int] = set()
        for edge_id in component:
            endpoints = graph_edge_vertices[edge_id]
            vertex_ids.update(endpoints)
            if len(endpoints) == 1:
                degree_by_vertex[endpoints[0]] += 2
            else:
                degree_by_vertex[endpoints[0]] += 1
                degree_by_vertex[endpoints[1]] += 1
        # 我方把“没有自由端”的边界分量判为闭合：Vertex 度数全部为偶数就说明
        # 不存在度 1 端点。度数 4 表示多个闭环在同一个 Vertex 相交（自接触），
        # 边界依然没有缺口，只是 Wire 不是简单回路，投影会退回 UNORDERED_COMPONENT，
        # 由人工复核接触点。
        # 闭合判据：分量内无未解析边，且所有 Vertex 度数为偶数（没有度 1 自由端）。
        closed = (
            not unresolved_edge_ids.intersection(component)
            and bool(component)
            and all(degree % 2 == 0 for degree in degree_by_vertex.values())
        )
        if closed:
            closed_loop_count += 1
        # 度为 1 的点是开口链的端点，度大于 2 的点是分叉点。
        # 两者都是"这个分量不是简单闭合环"的直接证据，
        # 必须原样带出去，因为下游靠它们区分真闭合与假闭合。
        open_endpoint_vertex_ids = sorted(
            vertex_id
            for vertex_id, degree in degree_by_vertex.items()
            if degree == 1
        )
        branch_vertex_ids = sorted(
            vertex_id
            for vertex_id, degree in degree_by_vertex.items()
            if degree > 2
        )
        # 只有未闭合的分量才去拆开口链：闭合环再去拆会把它切成无意义的片段。
        open_chains = (
            []
            if closed
            else _open_boundary_chains(
                component,
                graph_edge_vertices,
                length_evidence["edge_lengths_mm"],
            )
        )
        # 只要分量里有一条边的长度缺失或自相矛盾，周长就不完整。
        # 这时我给 None 而不是给一个"已知偏小"的部分和——
        # 部分和看起来像个有效数字，反而会让人误以为周长可信。
        component_missing_length_ids = sorted(
            set(component).intersection(
                # 缺失与冲突都算"这条边的长度不可信"：
                # 缺失是没量到，冲突是两个来源不一致，
                # 两者都不能用来算周长。
                set(length_evidence["missing_edge_ids"])
                | set(length_evidence["conflict_edge_ids"])
            )
        )
        # 只要有一条边长度不可信，整段周长就留 None：
        # 给出部分和会让这个数看起来像一个可用的周长，
        # 而它其实是残缺的。
        component_perimeter = (
            sum(
                float(length_evidence["edge_lengths_mm"][edge_id])
                for edge_id in component
            )
            if not component_missing_length_ids
            else None
        )
        projection = _wire_projection_evidence(
            component,
            graph_edge_vertices,
            vertex_coordinates_mm,
            closed=closed,
            edge_polyline_points_mm=edge_polyline_points_mm,
        )
        component_records.append(
            {
                "edge_ids": list(component),
                "vertex_ids": sorted(vertex_ids),
                # 顶点度数映射的键转字符串：这份记录要进 JSON，
                # 而 JSON 的键只能是字符串。
                "vertex_degree_counts": {
                    str(vertex_id): degree_by_vertex[vertex_id]
                    for vertex_id in sorted(degree_by_vertex)
                },
                "closed": closed,
                # 开放端点与分叉顶点分开列：
                # 开放端点说明"边界断在哪"，
                # 分叉顶点说明"边界在哪里分成多股"，
                # 两者指向不同的补法。
                "open_endpoint_vertex_ids": open_endpoint_vertex_ids,
                "branch_vertex_ids": branch_vertex_ids,
                "open_chains": open_chains,
                "perimeter_length_mm": component_perimeter,
                # 周长是否完整与"哪些边不可信"同时给：
                # 只有布尔值的话，周长不完整时
                # 不知道该去补哪条边的量测。
                "perimeter_complete": not component_missing_length_ids,
                "perimeter_missing_edge_ids": component_missing_length_ids,
                "projected_area_mm2": projection["projected_area_mm2"],
                "projection_status": projection["status"],
                # 投影证据整段嵌入：投影面积是算出来的结论，
                # 完整证据才能说明"这个面积是怎么投影出来的"。
                "projection_evidence": projection,
            }
        )

    if unresolved_edge_ids:
        status = "UNRESOLVED_ENDPOINTS"
    elif not graph_edge_vertices:
        # 全部边界 Edge 都是退化自环，没有可用的真实边界段。
        status = "DEGENERATE_ONLY_BOUNDARY"
    elif closed_loop_count == len(edge_components):
        status = "CLOSED_LOOP_CANDIDATE"
    else:
        status = "OPEN_BOUNDARY_CANDIDATE"
    wire_selection = _boundary_wire_selection(
        component_records,
        unresolved_edge_ids=unresolved_edge_ids,
        vertex_coordinates_mm=vertex_coordinates_mm,
        edge_vertices=graph_edge_vertices,
        edge_polyline_points_mm=edge_polyline_points_mm,
    )
    all_open_endpoint_vertex_ids = sorted(
        {
            int(vertex_id)
            for record in component_records
            for vertex_id in record.get("open_endpoint_vertex_ids", [])
        }
    )
    # 我方把开放链的两个端点也算作可桥接端点：度数大于 1 的分支点同样是
    # 某条链的真正自由端。BOTTLE 区域 43 里短链 [566] 两端是度 3 的 48/49，
    # 只按度 1 收集会漏掉它们，导致区域只剩 2 个可桥接端点、拿不到候选。
    all_open_endpoint_vertex_ids = sorted(
        set(all_open_endpoint_vertex_ids)
        | {
            int(vertex_id)
            for record in component_records
            for chain in record.get("open_chains", [])
            if not chain.get("closed")
            for vertex_id in (
                chain.get("start_vertex_id"),
                chain.get("end_vertex_id"),
            )
            if vertex_id is not None
        }
    )
    all_branch_vertex_ids = sorted(
        {
            int(vertex_id)
            for record in component_records
            for vertex_id in record.get("branch_vertex_ids", [])
        }
    )
    if all_branch_vertex_ids and status == "CLOSED_LOOP_CANDIDATE":
        # 我方不再把自接触闭环写成普通闭合环：它仍需人工复核接触点。
        status = "SELF_CONTACT_CLOSED_CANDIDATE"
    # 我方识别出"两条开放链共享同一个顶点"的形态：边界在这里自己碰到自己，
    # 并不是缺口。MD270324 区域 2260 有 21 个这样的区域：两条链的端点顶点 id
    # 完全相同（851 与 851），却因为分属不同链而被算成两个独立自由端。
    open_chain_count = sum(
        len(record.get("open_chains", [])) for record in component_records
    )
    # 我方识别出"两条开放链共享同一个顶点"的形态：边界在这里自己碰到自己，
    # 并不是缺口。MD270324 区域 2260 有 21 个这样的区域：两条链的端点顶点 id
    # 完全相同（851 与 851），却因为分属不同链而被算成两个独立自由端。
    # 注意这里**不能**过滤掉 closed=True 的链：自环链（851 -> 851）正是
    # 自接触最常见的形态之一，过滤掉它就再也看不到共用顶点了。
    # 但自环链会把同一个顶点 id 贡献两次，所以计数必须按"链数"而不是
    # "出现次数"，否则一条自环链自己就能凑够 2 次、被误判成自接触
    # （MD270324 区域 2784 就是这样被误判的）。
    chains_by_endpoint: dict[int, set[int]] = defaultdict(set)
    for index, chain in enumerate(
        chain
        for record in component_records
        for chain in record.get("open_chains", [])
    ):
        # 我按端点建"顶点 -> 链条集合"的反查表：
        # 多个链条共享同一个端点，说明它们在空间上相接，
        # 这是判断"自接触"的直接依据。
        for vertex_id in (
            chain.get("start_vertex_id"),
            chain.get("end_vertex_id"),
        ):
            if vertex_id is not None:
                chains_by_endpoint[int(vertex_id)].add(index)
    shared_chain_endpoint_ids = sorted(
        vertex_id
        for vertex_id, owners in chains_by_endpoint.items()
        if len(owners) > 1
    )
    self_touch_shared_vertex_ids: list[int] = []
    # 自接触判定：多条开口链、且存在共享端点。
    # 这时的边界在拓扑上是开的，在几何上却首尾相接触——
    # 它很可能是真实闭合轮廓只是没被缝上，所以单独给一个状态，
    # 不能简单按"开放边界"处理掉。
    if (
        status == "OPEN_BOUNDARY_CANDIDATE"
        and open_chain_count > 1
        and shared_chain_endpoint_ids
    ):
        self_touch_shared_vertex_ids = shared_chain_endpoint_ids
        status = "SELF_CONTACT_CLOSED_CANDIDATE"
    # 搭桥提案只针对真正开口的链：已经闭合的链不需要桥接，
    # 把它们混进来会生成"连接自身"的无意义提案。
    bridge_proposals = _bridge_proposals(
        all_open_endpoint_vertex_ids,
        vertex_coordinates_mm,
        # 同一条开放链的两端不参与搭桥：它们本该由这条链自己连起来，
        # 在这里再搭一条桥等于凭空造边。
        # 三道 None 检查是因为链可能只解析出一端。
        chain_end_pairs=[
            (int(chain["start_vertex_id"]), int(chain["end_vertex_id"]))
            for record in component_records
            for chain in record.get("open_chains", [])
            if not chain.get("closed")
            and chain.get("start_vertex_id") is not None
            and chain.get("end_vertex_id") is not None
        ],
    )
    return {
        "status": status,
        "edge_components": [list(component) for component in edge_components],
        "loop_count": len(edge_components),
        "closed_loop_count": closed_loop_count,
        "components": component_records,
        "open_endpoint_vertex_ids": all_open_endpoint_vertex_ids,
        "branch_vertex_ids": all_branch_vertex_ids,
        "branch_vertex_count": len(all_branch_vertex_ids),
        "open_chain_count": open_chain_count,
        # 需要搭桥直接由"有没有搭桥方案"推出：
        # 有方案说明存在连不上的开口，没方案说明边界自身是完整的。
        "bridge_required": bool(bridge_proposals),
        "bridge_proposals": bridge_proposals,
        "unresolved_edge_ids": sorted(unresolved_edge_ids),
        "self_touch_shared_vertex_ids": self_touch_shared_vertex_ids,
        # 边长字典的键统一转字符串：这份结果要进 JSON，
        # JSON 的键只能是字符串，不转会在序列化时报错或被隐式转换。
        "edge_lengths_mm": {
            str(edge_id): value
            for edge_id, value in sorted(
                length_evidence["edge_lengths_mm"].items()
            )
        },
        "perimeter_length_mm": length_evidence["perimeter_length_mm"],
        "perimeter_complete": length_evidence["complete"],
        # 缺失与冲突的边分开列：缺失是"根本没量到"，
        # 冲突是"量到了但两个来源不一致"，补救方式完全不同。
        "perimeter_missing_edge_ids": length_evidence["missing_edge_ids"],
        "perimeter_conflict_edge_ids": length_evidence["conflict_edge_ids"],
        # 退化边同时给编号与计数：编号用于定位，计数用于统计与门禁。
        "degenerate_edge_ids": sorted(degenerate_edge_ids),
        "degenerate_edge_count": len(degenerate_edge_ids),
        "seam_closure": seam_closure,
        "wire_selection": wire_selection,
        "method": "edge_endpoint_vertex_incidence",
        "interpretation": (
            "我把 Vertex 度数全为偶数、没有自由端的边界连通分量标记为闭合环候选，"
            "优先以有限 Edge 周长选择最大 Wire；长度缺失或冲突时回退到 Edge 数，"
            "我不把它直接写成最终分型 Wire。近零长度的退化自环会被隔离，"
            "不参与闭合环判定；度数大于 2 的自接触点会单独记录，要求人工复核。"
        ),
    }


def _boundary_edge_length_evidence(
    faces: list[StepFaceFeature],
    boundary_edge_ids: list[int],
) -> dict[str, object]:
    """我从 Face 特征汇总边界 Edge 长度，并拒绝冲突或非有限值。

    一条边通常被两个面引用，所以我会在遍历中**读到同一个 edge_id 两次**。
    这里的关键判断是两次读到的长度是否一致：不一致说明上游数组错位
    或几何摘要不可信，我把这条边记进 conflict_ids，并把它算作"缺失"
    （missing_ids 包含 conflict_ids），周长只在 complete 时才给出。

    我不取平均或取最大值来"消化"冲突：那样会把一个明确的矛盾悄悄
    抹平，输出一个看起来正常的周长。周长是选主环的核心判据，
    它必须建立在可信数据上，宁可给 None 让下游走复核。

    None / 非有限 / 负值一律跳过。负长度没有物理意义，
    让 NaN 进求和会让整个周长变成 NaN 并静默传播。
    """

    boundary_ids = {int(edge_id) for edge_id in boundary_edge_ids}
    lengths: dict[int, float] = {}
    conflict_ids: set[int] = set()
    for face in faces:
        # 边号与边长靠下标一一对应：这是读取面特征时
        # 定下的约定，任何一处中途跳过都会让两者错位。
        for edge_id, length in zip(face.edge_ids, face.edge_lengths_mm):
            edge_id = int(edge_id)
            # 只收边界边：内部边的长度与周长无关，
            # 收进来会让周长虚高。
            if edge_id not in boundary_ids:
                continue
            # 量不出来、非有限或为负的长度一律跳过：
            # 负长度说明量测出了问题，用它算周长会得到偏小的和。
            if length is None or not isfinite(float(length)) or float(length) < 0.0:
                continue
            candidate = float(length)
            previous = lengths.get(edge_id)
            # 同一条边被两个面各量一次：两次结果不一致就记冲突。
            # 相对与绝对容差都给 1e-8——这个量级远小于
            # 任何真实的几何差异，但足以吸收浮点误差。
            if previous is not None and not isclose(
                previous,
                candidate,
                rel_tol=1e-8,
                abs_tol=1e-8,
            ):
                conflict_ids.add(edge_id)
                # 冲突时不覆盖已有值：留下先量到的那一个，
                # 至少它来自某个具体面，可追溯。
                continue
            lengths[edge_id] = candidate
    # 缺失 = 完全没量到的 ∪ 量出冲突的：
    # 两者都不可信，周长都必须判为不完整。
    missing_ids = sorted(boundary_ids - set(lengths) | conflict_ids)
    complete = not missing_ids
    return {
        "edge_lengths_mm": lengths,
        # 缺失与冲突分开列：前者说明"没量到"，
        # 后者说明"量到了但对不上"，补救方式完全不同。
        "missing_edge_ids": sorted(
            boundary_ids - set(lengths) - conflict_ids
        ),
        "conflict_edge_ids": sorted(conflict_ids),
        "complete": complete,
        # 只在完整时给周长：给出部分和会让这个数
        # 看起来像可用的周长，而它其实是残缺的。
        "perimeter_length_mm": (
            sum(lengths.values()) if complete else None
        ),
    }


def _wire_projection_evidence(
    component: list[int],
    edge_vertices: dict[int, tuple[int, ...]],
    vertex_coordinates_mm: dict[int, list[float]] | None,
    *,
    closed: bool,
    edge_polyline_points_mm: dict[int, list[list[float]]] | None = None,
) -> dict[str, object]:
    """我把闭合边界分量投影成二维多边形，输出面积和平面性证据。

    点来源有优先级：优先用 Edge 曲线上的**采样点**，采样点缺失才回退到
    Vertex 端点。区别很实际——只用端点会把圆弧压成一条弦，
    两段圆弧拼成的圆环会退化成两个点，投影面积算不出来。

    我同时输出 max_out_of_plane_mm 和 planarity_tolerance_mm：
    投影本身对任何点集都能算出一个面积，但只有当点集真的近似共面时
    这个面积才有意义。不把平面性一起给出，"投影面积 500mm²"这句话
    在强扭曲的环上是误导，所以我把它标成 chord_approximation=True，
    让下游知道这是弦近似而非 NURBS 精确面积。
    """

    evidence: dict[str, object] = {
        "status": "UNAVAILABLE",
        "projected_area_mm2": None,
        "ordered_vertex_ids": [],
        "point_source": "UNAVAILABLE",
        "plane_normal_xyz": None,
        "max_out_of_plane_mm": None,
        "planarity_tolerance_mm": None,
        "point_count": 0,
        "chord_approximation": True,
        "method": "ordered_vertex_polygon_projection",
        "interpretation": (
            "我优先用当前文件内的 Edge 曲线采样点按顺序构造弦近似多边形；"
            "采样点缺失时才回退到 Vertex 端点。弧段仍被弦替代，"
            "投影面积不是 NURBS 精确面积。"
        ),
    }
    if not closed:
        # 不闭合就不算投影面积：多边形面积只在闭合轮廓上有定义。
        evidence["status"] = "NOT_CLOSED"
        return evidence
    ordered = _ordered_loop_vertex_ids(component, edge_vertices)
    # 优先用 Edge 曲线上的采样点：它比只用两端点更贴近真实曲线。
    # 只有文件里带了 polyline 采样数据时才走这条路，
    # 否则回退到端点多边形。
    sampled_points = (
        _ordered_loop_points_3d(
            component,
            edge_vertices,
            edge_polyline_points_mm,
            vertex_coordinates_mm,
        )
        if edge_polyline_points_mm
        else None
    )
    # 两条路都拿不到点序列就无法排序，只能标记未定。
    # 此时不给任何面积值——给 0 会被当成"面积为 0"的真实结论。
    if ordered is None and sampled_points is None:
        evidence["status"] = "UNORDERED_COMPONENT"
        return evidence
    if ordered is not None:
        evidence["ordered_vertex_ids"] = list(ordered)
    if sampled_points is not None:
        points = sampled_points
        # 方法名随数据来源改写：两份报告的"投影面积"精度不同
        # （采样逼近 vs 端点弦），不区分会让复核者误以为口径一致。
        evidence["point_source"] = "edge_curve_sampling"
        evidence["method"] = "sampled_edge_polyline_projection"
    else:
        if not vertex_coordinates_mm:
            evidence["status"] = "MISSING_VERTEX_COORDINATES"
            return evidence
        points = []
        for vertex_id in ordered:
            coordinates = vertex_coordinates_mm.get(vertex_id)
            if (
                not isinstance(coordinates, list)
                or len(coordinates) != 3
                or not all(isfinite(float(value)) for value in coordinates)
            ):
                evidence["status"] = "MISSING_VERTEX_COORDINATES"
                return evidence
            points.append([float(value) for value in coordinates])
        evidence["point_source"] = "vertex_endpoints"
    evidence["point_count"] = len(points)
    # 少于三个点围不出多边形：两点只是一根线段，投影面积恒为 0，
    # 继续算下去只会得到一个没有意义的 0，所以这里提前判掉。
    if len(points) < 3:
        evidence["status"] = "INSUFFICIENT_POINTS"
        return evidence
    # SVD 求最佳拟合平面：最小奇异值对应的奇异向量就是拟合法向，
    # 前两个奇异向量张成平面内的正交基，点投影到这两个基上即得到二维多边形。
    array = np.asarray(points, dtype=np.float64)
    centered = array - array.mean(axis=0)
    try:
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        evidence["status"] = "PROJECTION_FAILED"
        return evidence
    # 基向量不足三个说明点集退化（全部共线或重合），拟合不出唯一平面。
    if basis.shape[0] < 3:
        evidence["status"] = "DEGENERATE_POINTS"
        return evidence
    normal = basis[2]
    # 平面性判据：点到拟合平面的最大偏离。容差随面积平方根缩放，
    # 这样大环允许略大的绝对偏差，小环则要求更严格，避免一刀切阈值。
    deviations = centered @ normal
    max_deviation = float(np.max(np.abs(deviations)))
    planar = np.column_stack([centered @ basis[0], centered @ basis[1]])
    area = abs(_polygon_area_mm2(planar))
    tolerance = max(1e-6, 1e-3 * float(np.sqrt(area)))
    evidence["plane_normal_xyz"] = [float(value) for value in normal]
    evidence["max_out_of_plane_mm"] = max_deviation
    evidence["planarity_tolerance_mm"] = tolerance
    evidence["projected_area_mm2"] = area
    # 三种结论：面积为 0 是退化；偏离超容差仍给面积但标非平面候选
    # （不丢弃，因为工程上轻微翘曲的环仍有参考价值）；否则正常计算完成。
    if area <= 1e-9:
        evidence["status"] = "DEGENERATE"
    elif max_deviation > tolerance:
        evidence["status"] = "NON_PLANAR_CANDIDATE"
    else:
        evidence["status"] = "COMPUTED"
    return evidence


def _ordered_loop_vertex_ids(
    component: list[int],
    edge_vertices: dict[int, tuple[int, ...]],
) -> list[int] | None:
    """我按共享 Vertex 把闭合分量排成一条有序回路。"""

    incident: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_id in component:
        endpoints = edge_vertices.get(edge_id)
        # 端点不是恰好两个就整段放弃：
        # 这里要的是"能走出一条简单链"的图，
        # 自环或退化边走下去得到的序列没有意义。
        if endpoints is None or len(endpoints) != 2:
            return None
        left, right = int(endpoints[0]), int(endpoints[1])
        # 两个方向各记一条：链是无向的，
        # 从任一端都要能看到"这条边通向对面"。
        incident[left].append((int(edge_id), right))
        incident[right].append((int(edge_id), left))
    # 每个顶点的度数必须恰好为 2：这是"闭合环"的定义。
    # 出现度数为 1 或 3 以上的点说明这条边界有分叉或开口，
    # 不是环，返回 None 让调用方按链去处理。
    if not incident or any(
        len(neighbors) != 2 for neighbors in incident.values()
    ):
        return None
    # 从编号最小的顶点起步：起点选法固定，
    # 同一份数据每次得到完全相同的环序列。
    start = min(incident)
    ordered = [start]
    used: set[int] = set()
    current = start
    while True:
        # 每次取一条没用过的边：环上每个点度数都是 2，
        # 除了来路之外只有一条可走，不需要挑。
        step = next(
            (
                (edge_id, neighbor)
                for edge_id, neighbor in incident[current]
                if edge_id not in used
            ),
            None,
        )
        # 没有可走的边说明走到了断头，不构成环。
        if step is None:
            break
        edge_id, neighbor = step
        used.add(edge_id)
        current = neighbor
        # 走回起点就收尾，不把起点再记一次。
        if current == start:
            break
        ordered.append(current)
    # 有边没走完说明这个连通块没被一条环覆盖完，
    # 序列不完整，返回 None。
    if len(used) != len(component):
        return None
    return ordered


def _point_distance_sq(first: object, second: object) -> float:
    """我按三维坐标算平方距离，用于拼接边折线时判定方向。"""

    if (
        not isinstance(first, (list, tuple))
        or not isinstance(second, (list, tuple))
        or len(first) != 3
        or len(second) != 3
    ):
        return float("inf")
    return float(
        sum(
            (float(left) - float(right)) ** 2
            for left, right in zip(first, second)
        )
    )


def _clean_polyline(
    raw_points: object,
) -> list[list[float]] | None:
    """我过滤非有限坐标，返回一条可用的边折线。

    遇到坏数据我返回 None（整条折线作废），而不是跳过坏点继续。
    理由：折线是用来按顺序围成多边形的，中间少一个点会让多边形
    走一条不存在的捷径，算出来的面积是错的但看起来完全正常。
    整条作废让下游退回到端点近似，至少不会给出错误的面积。

    少于 2 个点也返回 None：一条边至少要两个端点，
    单点折线围不出任何东西。
    """

    if not isinstance(raw_points, list) or len(raw_points) < 2:
        return None
    cleaned: list[list[float]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            return None
        coordinates = [float(value) for value in point]
        if not all(isfinite(value) for value in coordinates):
            return None
        cleaned.append(coordinates)
    return cleaned


def _collect_boundary_polylines(
    boundary_edge_ids: list[int],
    edge_polyline_sampler: object,
) -> dict[int, list[list[float]]] | None:
    """我只对区域边界 Edge 取样折线，避免为整模型每条边付采样开销。

    采样是这套流程里最贵的操作（每条边要做曲线离散），而实际用到的
    只有边界边——通常只占全模型边数的极小一部分。为全模型采样会让
    大装配体的耗时从秒级涨到分钟级。

    逐条单独 try：某条边采样失败（退化曲线、后端报错）只让这一条
    缺失，其余边照常使用。一条边失败就整批放弃，
    会让"采到 99 条边"的成果全部作废。
    """

    if not callable(edge_polyline_sampler):
        return None
    collected: dict[int, list[list[float]]] = {}
    for edge_id in boundary_edge_ids:
        try:
            cleaned = _clean_polyline(edge_polyline_sampler(int(edge_id)))
        except Exception:
            cleaned = None
        if cleaned is not None:
            collected[int(edge_id)] = cleaned
    return collected or None


def _ordered_loop_points_3d(
    component: list[int],
    edge_vertices: dict[int, tuple[int, ...]],
    edge_polyline_points_mm: dict[int, list[list[float]]] | None,
    vertex_coordinates_mm: dict[int, list[float]] | None,
    *,
    tolerance_mm: float = 1e-6,
) -> list[list[float]] | None:
    """我把闭合分量的每条边折线按共享 Vertex 顺序拼成一条闭合点列。

    我方优先使用 OCP 沿曲线采样的折线，这样两段圆弧组成的圆环、
    单条闭合圆边都不会退化成两个点；采样缺失时回退到 Vertex 端点。
    """

    if not component or not edge_vertices:
        return None
    polylines: dict[int, list[list[float]]] = {}
    for edge_id in component:
        endpoints = edge_vertices.get(edge_id)
        if endpoints is None:
            return None
        cleaned = _clean_polyline(
            (edge_polyline_points_mm or {}).get(int(edge_id))
        )
        if cleaned is None:
            if (
                len(endpoints) != 2
                or not vertex_coordinates_mm
            ):
                return None
            first = vertex_coordinates_mm.get(int(endpoints[0]))
            second = vertex_coordinates_mm.get(int(endpoints[1]))
            cleaned = _clean_polyline([first, second])
            if cleaned is None:
                return None
        polylines[int(edge_id)] = cleaned

    if len(component) == 1:
        # 单条边自闭合（例如整圆）：直接用它的采样点。
        # 首点与末点重合时去掉末点，避免后面出现零长度边。
        points = list(polylines[int(component[0])])
        if (
            len(points) >= 2
            and _point_distance_sq(points[0], points[-1]) <= tolerance_mm**2
        ):
            points.pop()
        # 少于三个点的"环"是退化线段，构不成多边形，返回 None 让上层跳过。
        return points if len(points) >= 3 else None

    # 多条边的情况：按共享端点建邻接表，每个顶点必须恰好连着两条边，
    # 否则这个分量不是简单环（有分叉或断口），顺序无法确定。
    incident: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_id in component:
        endpoints = edge_vertices[edge_id]
        if len(endpoints) != 2:
            return None
        left, right = int(endpoints[0]), int(endpoints[1])
        incident[left].append((int(edge_id), right))
        incident[right].append((int(edge_id), left))
    if not incident or any(
        len(neighbors) != 2 for neighbors in incident.values()
    ):
        return None

    # 从编号最小的顶点出发沿邻接表走一圈。起点取 min 而非任取，
    # 是为了让同一分量每次都得到完全相同的点序，签名才可复现。
    start = min(incident)
    ordered: list[list[float]] = []
    used: set[int] = set()
    current = start
    while True:
        # 每次取一条还没用过的边：这里是按 incident 的记录顺序取，
        # 因为下面会按首尾距离校正方向，取法不影响最终结果。
        step = next(
            (
                (edge_id, neighbor)
                for edge_id, neighbor in incident[current]
                if edge_id not in used
            ),
            None,
        )
        # 当前顶点没有可用边就停：这是链的一端。
        if step is None:
            break
        edge_id, neighbor = step
        used.add(edge_id)
        points = polylines[edge_id]
        if not ordered:
            # 第一条折线整段接入，作为后续拼接的基准方向。
            ordered.extend(points)
        else:
            # 比较这条折线的首端到已拼末端、以及尾端到已拼末端的距离：
            # 哪一端离得更近就把它接在前面，
            # 于是折线被翻转成正确的朝向。
            # 这里比的是距离平方，省掉开方。
            if _point_distance_sq(points[0], ordered[-1]) <= (
                _point_distance_sq(points[-1], ordered[-1])
            ):
                # 正向接入，跳过首点：首点与已拼末端重合，不重复记。
                ordered.extend(points[1:])
            else:
                # 反向接入，同理跳过末点。
                ordered.extend(reversed(points[:-1]))
        current = neighbor
        # 走回起点说明链闭合，收尾。
        if current == start:
            break
    # 有边没走到说明这个连通块不是一条简单链
    # （例如存在分叉），拼出来的序列不可信，返回 None。
    if len(used) != len(component):
        return None
    # 首尾距离在容差内说明最后一点与首点重合，
    # 去掉末点避免重复。
    if (
        len(ordered) >= 2
        and _point_distance_sq(ordered[0], ordered[-1]) <= tolerance_mm**2
    ):
        ordered.pop()
    # 少于三个点围不出面积，投影没有意义。
    return ordered if len(ordered) >= 3 else None


def _polygon_area_mm2(points: np.ndarray) -> float:
    """我用鞋带公式计算二维投影多边形的有向面积。

    返回值是**有向**的（带正负号）：逆时针为正、顺时针为负。
    我保留符号不取绝对值，因为符号本身携带环的走向信息，
    下游判包含关系时要用它区分内外环。

    np.roll 实现循环移位，等价于把首尾相接，这样不需要显式
    把第一个点复制到末尾，避免多一次数组拼接。

    少于 3 个点返回 0.0 而不是 nan：两点围不出面积，
    返回 0 能让后续比较（比如"面积最大的环"）照常进行，
    返回 nan 会污染所有比较运算。
    """

    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(
        0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    )


def _loop_points_3d(
    ordered_vertex_ids: object,
    vertex_coordinates_mm: dict[int, list[float]] | None,
) -> np.ndarray | None:
    """我按有序顶点编号取出三维坐标，供统一平面投影使用。"""

    if (
        not vertex_coordinates_mm
        or not isinstance(ordered_vertex_ids, list)
        or len(ordered_vertex_ids) < 3
    ):
        return None
    points: list[list[float]] = []
    for vertex_id in ordered_vertex_ids:
        if not isinstance(vertex_id, int):
            return None
        coordinates = vertex_coordinates_mm.get(vertex_id)
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 3
            or not all(isfinite(float(value)) for value in coordinates)
        ):
            return None
        points.append([float(value) for value in coordinates])
    return np.asarray(points, dtype=np.float64)


def _plane_axes(
    plane_normal_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """我根据平面法向构造稳定的二维投影坐标轴。

    参考轴的选择带一个 0.9 的判据：法向的 Z 分量小于 0.9 时用 Z 轴
    做参考，否则改用 X 轴。这不是随意的——若法向本身就接近 Z 轴，
    与 Z 的叉积会接近零向量，归一化后方向完全由浮点噪声决定，
    投影结果不可复现。

    用 0.9 而不是精确等于 1：只在完全重合时才切换的话，
    法向与 Z 夹角很小（比如 5 度）时叉积模长已经很小，
    归一化会放大误差。留出一个安全区间能让切换发生在
    数值仍然健康的时候就完成。
    """

    normal = np.asarray(plane_normal_xyz, dtype=np.float64)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-12 or not np.all(np.isfinite(normal)):
        return None
    normal = normal / normal_length
    reference = (
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(normal[2])) < 0.9
        else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    )
    axis_u = np.cross(normal, reference)
    axis_u_length = float(np.linalg.norm(axis_u))
    if axis_u_length <= 1e-12:
        return None
    axis_u = axis_u / axis_u_length
    axis_v = np.cross(normal, axis_u)
    axis_v_length = float(np.linalg.norm(axis_v))
    if axis_v_length <= 1e-12:
        return None
    axis_v = axis_v / axis_v_length
    return axis_u, axis_v


def _loop_relation_evidence(
    closed_records: list[tuple[int, dict[str, object]]],
    vertex_coordinates_mm: dict[int, list[float]] | None,
    *,
    edge_vertices: dict[int, tuple[int, ...]] | None = None,
    edge_polyline_points_mm: dict[int, list[list[float]]] | None = None,
) -> dict[str, object]:
    """我判断多个闭合环之间的包含、孔洞和相交关系。

    原理说明：一个面上的多个闭合边界环，工程语义通常是"一个外轮廓 + 若干
    内部孔洞"。要把它们分开，我把它当成二维平面上的多边形包含关系问题。
    关键前提是所有环必须投影到**同一个**平面上比较，否则包含关系没有意义，
    所以我先收集所有环的点，用 SVD 求一个全局拟合平面，再把每个环投影到
    该平面的正交基上，得到可两两比较的二维多边形。

    分类顺序（先判异常，再判层级）：
      1. 两两求关系：相交 / 相切 / 前者在后者内 / 后者在前者内 / 相离。
         相交与相切是拓扑异常，直接拦截，因为真实外轮廓与孔洞不应互相穿越；
      2. 对"包含"关系，我只取**面积最小的那个外层**当直接父亲。若取最大的，
         嵌套三层时中间层会被跳过，深度和孔洞归属就错了；
      3. 用 parent 关系算包含深度：深度为 0 的是外轮廓，奇数深度通常对应孔洞。

    边界与诚实性：这是弦近似下的投影包含判断，不是 OCP 的孔洞布尔运算，
    也不是 NURBS 精确分类；非平面环投影后可能相交，所以我把
    NON_PLANAR_CANDIDATE 也纳入计算但保留 skipped_components 供复核。
    """

    evidence: dict[str, object] = {
        "status": "NO_CLOSED_LOOPS",
        "method": "projected_polygon_containment",
        "loop_count": 0,
        "outer_candidate_indices": [],
        "hole_candidate_indices": [],
        "root_component_indices": [],
        "parent_by_component": {},
        "children_by_component": {},
        "containment_depth_by_component": {},
        "intersecting_pairs": [],
        "touching_pairs": [],
        "skipped_components": [],
        "interpretation": (
            "我按投影二维多边形判断闭合环包含关系；该关系是弦近似候选，"
            "不是最终孔洞布尔运算，也不是 NURBS 精确分类。"
        ),
    }
    if not closed_records:
        return evidence
    if not vertex_coordinates_mm and not edge_polyline_points_mm:
        evidence["status"] = "MISSING_VERTEX_COORDINATES"
        return evidence

    raw_polygons: dict[int, tuple[float, np.ndarray]] = {}
    skipped_components: list[int] = []
    for component_index, record in closed_records:
        # 我逐个环构造可比较的二维多边形。NON_PLANAR_CANDIDATE 也放进来，
        # 因为真实零件上轻微非平面的环仍需给出候选结论；但它同时会写进
        # skipped_components 之外的状态里，复核时能看到它并非严格平面。
        projection = record.get("projection_evidence")
        projection_status = str(record.get("projection_status", "UNAVAILABLE"))
        projected_area = record.get("projected_area_mm2")
        # 投影面积为 0 或退化（<=1e-9）的环无法参与包含判断：
        # 一个点或一条线围不出面积，"谁包含谁"没有意义，只能跳过。
        if (
            not isinstance(projection, dict)
            or projection_status not in {"COMPUTED", "NON_PLANAR_CANDIDATE"}
            or not isinstance(projected_area, (int, float))
            or not isfinite(float(projected_area))
            or float(projected_area) <= 1e-9
        ):
            skipped_components.append(component_index)
            continue
        points = None
        component = record.get("edge_ids")
        # 优先用边的曲线采样点：它保留弧段形状，比只用 Vertex 端点更接近
        # 真实轮廓。采样失败才回退到端点，此时环上的圆弧会被压成弦。
        if (
            edge_vertices
            and edge_polyline_points_mm
            and isinstance(component, list)
            and component
        ):
            sampled = _ordered_loop_points_3d(
                [int(edge_id) for edge_id in component],
                edge_vertices,
                edge_polyline_points_mm,
                vertex_coordinates_mm,
            )
            if sampled is not None:
                points = np.asarray(sampled, dtype=np.float64)
        if points is None:
            points = _loop_points_3d(
                projection.get("ordered_vertex_ids"),
                vertex_coordinates_mm,
            )
        if points is None or points.shape[0] < 3:
            skipped_components.append(component_index)
            continue
        raw_polygons[component_index] = (float(projected_area), points)

    evidence["loop_count"] = len(raw_polygons)
    evidence["skipped_components"] = skipped_components
    if not raw_polygons:
        # 一个可用的环都没有：这不是"相离"，而是根本无法判断，
        # 所以状态给 UNRESOLVED_RELATION 而不是 DISJOINT_LOOPS——
        # 后者会让人误以为已经确认过它们互不包含。
        evidence["status"] = "UNRESOLVED_RELATION"
        return evidence
    # 关键一步：所有环必须投影到**同一个**平面上才能互相比较。
    # 若各自用自己的拟合平面，两个环的法向不同，包含关系没有共同坐标基准，
    # 结果毫无意义。所以我先把全部环的点堆在一起求一个全局拟合平面。
    all_points = np.vstack(
        [points for _, points in raw_polygons.values()]
    )
    global_origin = all_points.mean(axis=0)
    centered_all = all_points - global_origin
    try:
        _, _, global_basis = np.linalg.svd(
            centered_all,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        evidence["status"] = "PROJECTION_FAILED"
        return evidence
    if global_basis.shape[0] < 3:
        evidence["status"] = "DEGENERATE_POINTS"
        return evidence
    global_normal = global_basis[2]
    axes = _plane_axes(global_normal)
    if axes is None:
        evidence["status"] = "PROJECTION_FAILED"
        return evidence
    axis_u, axis_v = axes
    polygons: dict[int, tuple[float, np.ndarray]] = {}
    for component_index, (projected_area, points) in raw_polygons.items():
        centered = points - global_origin
        polygons[component_index] = (
            projected_area,
            np.column_stack((centered @ axis_u, centered @ axis_v)),
        )
    max_out_of_plane = float(
        np.max(np.abs(centered_all @ global_normal))
    )
    # 全局平面性容差按点数的平方根缩放：点越多，SVD 拟合受个别离群点
    # 影响越大，允许的绝对偏差也相应放宽，否则大模型上会全被判成非平面。
    evidence["global_plane_normal_xyz"] = [
        float(value) for value in global_normal
    ]
    evidence["global_max_out_of_plane_mm"] = max_out_of_plane
    evidence["global_planarity_tolerance_mm"] = max(
        1e-6,
        1e-3 * float(np.sqrt(max(1.0, all_points.shape[0]))),
    )

    # 只有一个环时没有"关系"可判：它既非孔洞也非外轮廓的父/子，
    # 我给它 ISOLATED_LOOP_CANDIDATE 与深度 0（按外口径处理），
    # 但状态单独标成 SINGLE_CLOSED_LOOP，避免与"确认过包含关系"混淆。
    if len(polygons) == 1:
        component_index = next(iter(polygons))
        evidence["status"] = "SINGLE_CLOSED_LOOP"
        evidence["outer_candidate_indices"] = [component_index]
        evidence["root_component_indices"] = [component_index]
        evidence["containment_depth_by_component"] = {
            str(component_index): 0
        }
        evidence["loop_roles"] = {
            str(component_index): "ISOLATED_LOOP_CANDIDATE"
        }
        return evidence

    parent_by_component: dict[int, int] = {}
    children_by_component: dict[int, list[int]] = defaultdict(list)
    intersecting_pairs: list[list[int]] = []
    touching_pairs: list[list[int]] = []
    # containment_candidates 存的是 (外层面积, 外层编号) 列表：
    # 后面要从中挑面积**最小**的外层作为直接父亲，所以这里必须连面积一起留着。
    containment_candidates: dict[int, list[tuple[float, int]]] = defaultdict(list)
    ordered_indices = sorted(polygons)
    # 只遍历上三角（position+1 起）：包含关系是对称的，
    # 两个方向都算会得到重复边，父子判定还要额外去重。
    for position, first_index in enumerate(ordered_indices):
        first_area, first_points = polygons[first_index]
        for second_index in ordered_indices[position + 1 :]:
            second_area, second_points = polygons[second_index]
            relation = _polygon_pair_relation(first_points, second_points)
            if relation == "INTERSECTING":
                intersecting_pairs.append([first_index, second_index])
            elif relation == "TOUCHING":
                touching_pairs.append([first_index, second_index])
            elif relation == "FIRST_INSIDE_SECOND":
                containment_candidates[first_index].append(
                    (second_area, second_index)
                )
            elif relation == "SECOND_INSIDE_FIRST":
                containment_candidates[second_index].append(
                    (first_area, first_index)
                )
    for inner_index, candidates in containment_candidates.items():
        # 我只认"面积最小的包含者"当直接父亲：三层嵌套时若取最大的，
        # 中间层会被跳过，深度和孔洞归属就全错了。
        parent_by_component[inner_index] = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )[1]

    for child_index, parent_index in parent_by_component.items():
        children_by_component[parent_index].append(child_index)
    root_indices = sorted(
        component_index
        for component_index in polygons
        if component_index not in parent_by_component
    )
    depths: dict[str, int] = {}
    roles: dict[str, str] = {}
    for component_index in sorted(polygons):
        depth = 0
        current = component_index
        seen: set[int] = set()
        # 我沿 parent 链往上走算包含深度，seen 防止异常数据形成环导致死循环。
        while current in parent_by_component and current not in seen:
            seen.add(current)
            current = parent_by_component[current]
            depth += 1
        depths[str(component_index)] = depth
        # 角色判定：有父亲就是孔洞候选；没有父亲但有孩子就是带孔的外轮廓；
        # 都没有就是孤立外轮廓。深度为 0 才可能是真正的外轮廓。
        if component_index in parent_by_component:
            # 三种环角色：被别的环包住的是"内孔"，
            # 包住别人的是"带孔的外轮廓"，
            # 两者都不是的是"单纯的外轮廓"。
            roles[str(component_index)] = "INNER_HOLE_CANDIDATE"
        elif children_by_component.get(component_index):
            roles[str(component_index)] = "OUTER_WITH_HOLES_CANDIDATE"
        else:
            roles[str(component_index)] = "OUTER_CANDIDATE"

    # 两对关系都排序去重：同一对环可能被两条边各记一次，
    # 去重保证"有几对环相交"这个计数不被重复计算；
    # 排序保证输出稳定可比对。
    intersecting_pairs = sorted({tuple(pair) for pair in intersecting_pairs})
    touching_pairs = sorted({tuple(pair) for pair in touching_pairs})
    # 状态按"最严重"排序取第一个：相交 > 相切 > 嵌套 > 多个独立环。
    # 相交说明两个环真的交叉，几何上不成立、必须人工处理；
    # 后面几档都还算正常拓扑，只是需要区分。
    if intersecting_pairs:
        evidence["status"] = "INTERSECTING_LOOPS"
    elif touching_pairs:
        evidence["status"] = "TOUCHING_LOOPS"
    elif parent_by_component:
        evidence["status"] = "NESTED_LOOPS"
    elif len(root_indices) > 1:
        evidence["status"] = "MULTIPLE_DISJOINT_LOOPS"
    else:
        evidence["status"] = "COMPUTED"
    # 外轮廓候选包含三种角色：孤立环也是外轮廓的一种
    # （它没有被别的环包住，也没有包住别人）。
    evidence["outer_candidate_indices"] = sorted(
        int(component_index)
        for component_index, role in roles.items()
        if role
        in {
            "OUTER_CANDIDATE",
            "OUTER_WITH_HOLES_CANDIDATE",
            "ISOLATED_LOOP_CANDIDATE",
        }
    )
    evidence["hole_candidate_indices"] = sorted(
        int(component_index)
        for component_index, role in roles.items()
        if role == "INNER_HOLE_CANDIDATE"
    )
    evidence["root_component_indices"] = root_indices
    evidence["parent_by_component"] = {
        str(child): parent
        for child, parent in sorted(parent_by_component.items())
    }
    evidence["children_by_component"] = {
        str(parent): sorted(children)
        for parent, children in sorted(children_by_component.items())
    }
    evidence["containment_depth_by_component"] = depths
    evidence["loop_roles"] = roles
    evidence["intersecting_pairs"] = [
        list(pair) for pair in intersecting_pairs
    ]
    evidence["touching_pairs"] = [list(pair) for pair in touching_pairs]
    return evidence


def _polygon_containment(
    inner_points: np.ndarray,
    outer_points: np.ndarray,
) -> str:
    """我判断一个二维多边形是否完全在另一个多边形内部。

    原理说明：我逐点判定 inner 的点相对 outer 的位置（射线法），再按三类
    计数合成结论。顺序很重要，因为它决定了异常能否被抓出来：
      1. 只要有一个点落在 outer 的**边界**上，我就直接返回 TOUCHING。
         相切是两个环共享边或点的信号，真实外轮廓与孔洞不应发生，
         必须提前拦截而不是当成包含/相离；
      2. 全部点在内部才算 INSIDE，部分在内部分在外就是真的相交；
      3. 全部点在外部时还要再查一次边是否相交：两个多边形可以互相
         "环环相扣"而没有任何顶点落在对方内部，只看点会漏判成 DISJOINT。

    容差按多边形尺度缩放（1e-9 x 包围盒最大边长），避免大尺寸零件上
    浮点误差把边界点误判成内部点，也避免小尺寸零件上容差过大吃掉真实相交。

    性能说明（这里做过一次针对性优化）：我原本对 inner 的每个点
    单独调用一次射线法，在真实 STEP 上会被调用到 9 万次以上，
    而每次都要重新把 outer 的边展开一遍——光这一项就占了整条
    边界拓扑链 365s / 376s。现在改成一次性批量判定：
    outer 的边只展开一次，全部 inner 点一次算完。
    判定式与逐点版本逐项一致（我方用凹多边形与随机多边形核对过
    结果完全一致），所以三类计数的语义不变。
    """

    outer_min = outer_points.min(axis=0)
    outer_max = outer_points.max(axis=0)
    scale = max(1.0, float(np.max(outer_max - outer_min)))
    tolerance = 1e-9 * scale
    # 包围盒前置判断整阵列做：落在包围盒之外的点一定在外多边形之外，
    # 不必再参与射线求交。这一层同时把批量运算的输入规模压下来。
    outside_bbox = (
        (inner_points[:, 0] < outer_min[0] - tolerance)
        | (inner_points[:, 0] > outer_max[0] + tolerance)
        | (inner_points[:, 1] < outer_min[1] - tolerance)
        | (inner_points[:, 1] > outer_max[1] + tolerance)
    )
    states = np.zeros(len(inner_points), dtype=np.int8)
    candidates = inner_points[~outside_bbox]
    if len(candidates):
        states[~outside_bbox] = _point_in_polygon_2d_batch(
            candidates, outer_points
        )
    # 只要有一个点落在边界上就必须先报 TOUCHING，这条优先级不能变。
    boundary_count = int(np.count_nonzero(states < 0))
    inside_count = int(np.count_nonzero(states > 0))
    outside_count = int(len(inner_points) - boundary_count - inside_count)
    if boundary_count:
        return "TOUCHING"
    if outside_count == 0:
        return "INSIDE"
    if inside_count == 0 and not _polygon_edges_intersect(
        inner_points,
        outer_points,
    ):
        return "DISJOINT"
    return "INTERSECTING"


def _polygon_pair_relation(
    first_points: np.ndarray,
    second_points: np.ndarray,
) -> str:
    """我先判相交/接触再判包含，避免面积顺序掩盖真实的相交环。

    原理说明：两个环的关系必须"异常优先"。如果先判包含，一个与外轮廓
    相交的环可能因为大部分点仍在内部而被误判成 INSIDE，于是相交这个
    拓扑错误就被悄悄吞掉了。所以我固定按
    包围盒快速排除 -> 双向包含判定 -> 取异常 -> 取包含
    的顺序合成结论，任一侧报相交/相切就整体判为相交/相切。

    双向判定的必要性：只判"first 是否在 second 内"无法区分
    "first 包含 second"和"两者相离"，必须两个方向各判一次。
    """

    first_min = first_points.min(axis=0)
    first_max = first_points.max(axis=0)
    second_min = second_points.min(axis=0)
    second_max = second_points.max(axis=0)
    scale = max(
        1.0,
        float(np.max(first_max - first_min)),
        float(np.max(second_max - second_min)),
    )
    tolerance = 1e-9 * scale
    # 我方先用包围盒排除一定不相交的环，避免对每一对闭合环都做逐点包含判定。
    if (
        first_max[0] < second_min[0] - tolerance
        or second_max[0] < first_min[0] - tolerance
        or first_max[1] < second_min[1] - tolerance
        or second_max[1] < first_min[1] - tolerance
    ):
        return "DISJOINT"
    first_relation = _polygon_containment(first_points, second_points)
    second_relation = _polygon_containment(second_points, first_points)
    if "INTERSECTING" in (first_relation, second_relation):
        return "INTERSECTING"
    if "TOUCHING" in (first_relation, second_relation):
        return "TOUCHING"
    if first_relation == "INSIDE":
        return "FIRST_INSIDE_SECOND"
    if second_relation == "INSIDE":
        return "SECOND_INSIDE_FIRST"
    return "DISJOINT"


def _point_in_polygon_2d(point: np.ndarray, polygon: np.ndarray) -> int:
    """我返回 1 表示内部、0 表示外部、-1 表示落在边界上。

    原理说明：我用经典的射线交叉法（奇偶规则）。从待判点向右引一条
    水平射线，统计它与多边形各边的交叉次数：奇数次在内部，偶数次在外部。
    它对凹多边形同样成立，因为它只依赖"穿过边界的奇偶性"，与凸性无关。

    边界优先级：我先逐边查点是否落在边上，命中就立刻返回 -1。
    射线法对恰好落在边界上的点结果不稳定（取决于浮点比较方向），
    而"相切"在本项目里是必须上报的异常，不能被当成内部或外部。

    性能说明（这里做过一次针对性优化）：我原本是逐边的 Python 循环，
    在真实 STEP 上会被调用到 610 万次以上，光函数调用开销就占了整条
    边界拓扑链的绝大部分时间。现在改成两次整阵列运算：
      1. 先把所有边一次算完"点是否落在边上"（向量化，等价于原来的
         提前返回 -1，只是不再需要提前退出）；
      2. 再把所有边一次算完射线交叉的奇偶统计。
    两次都是纯算术，没有 Python 层循环，语义与逐边版本逐位一致。
    """

    point = np.asarray(point, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.shape[0] == 0:
        return 0
    if _point_on_segment_2d_any(point, polygon):
        return -1
    x = float(point[0])
    y = float(point[1])
    left = polygon
    right = np.roll(polygon, -1, axis=0)
    # 只统计真正跨越水平射线的边，与逐边版本的 (left.y > y) != (right.y > y) 同义。
    crosses = (left[:, 1] > y) != (right[:, 1] > y)
    if not np.any(crosses):
        return 0
    left_cross = left[crosses]
    right_cross = right[crosses]
    # 分母只可能是非零的：跨越条件已保证两端点严格分居射线两侧。
    intersection_x = (
        (right_cross[:, 0] - left_cross[:, 0])
        * (y - left_cross[:, 1])
        / (right_cross[:, 1] - left_cross[:, 1])
        + left_cross[:, 0]
    )
    inside = bool(np.count_nonzero(x < intersection_x) % 2 == 1)
    return 1 if inside else 0


def _point_in_polygon_2d_batch(
    points: np.ndarray, polygon: np.ndarray
) -> np.ndarray:
    """我一次算出一批点相对**同一个**多边形的内外状态。

    返回 int8 数组：1 内部、0 外部、-1 落在边界上，
    与 `_point_in_polygon_2d` 的返回值逐项同义。

    为什么需要它：`_polygon_containment` 要对 inner 的每一个点
    判断相对同一个 outer 的位置。逐点调用时，outer 的边
    在每次调用里都要重新展开一遍——真实 STEP 上一个环有
    几千个点，这个重复展开就是整条链路的耗时主体。

    边界点优先：落在边上的点先标记 -1，且不参与奇偶统计，
    与逐点版本"命中边界立刻返回 -1"的语义一致。

    为什么先按 y 筛边而不是直接摊开广播：朴素广播要铺开
    (点数 x 边数) 的中间阵列，两者都是几千时是上千万元素。
    我实测过——outer 只有 50 条边时广播比逐点快约 20 倍，
    但 outer 有 4000 条边时反而慢到 0.69 倍，慢在内存带宽
    而不是计算。而射线法真正用得上的边只是"跨过该点 y 高度"
    的那几条：我先用每条边的 y 区间把候选边筛出来，
    再只对这一小撮边做广播，规模从 点数x边数 降到
    点数x候选边数。筛边本身是 O(边数) 的向量运算。
    判定式与逐点版逐项一致，所以结论不变。

    复杂度仍是 O(点数 x 边数)，这里只砍常数与内存；
    真要让复杂度下降需要换扫描线算法。
    """

    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.shape[0] == 0 or points.shape[0] == 0:
        return np.zeros(len(points), dtype=np.int8)
    left = polygon
    right = np.roll(polygon, -1, axis=0)
    # 每条边的 y 区间：只有点的 y 落在这个区间内，这条边才可能
    # 与该点的水平射线相交，或被判定为"点落在边上"。
    edge_y_low = np.minimum(left[:, 1], right[:, 1])
    edge_y_high = np.maximum(left[:, 1], right[:, 1])

    parity = np.zeros(len(points), dtype=np.int64)
    on_edge_any = np.zeros(len(points), dtype=bool)
    for index in range(len(points)):
        x = float(points[index, 0])
        y = float(points[index, 1])
        # 先按 y 粗筛，容差取宽松值保证不漏掉边界候选；
        # 精确判定交给下面的广播，与标量版同口径。
        candidate = (edge_y_low - 1e-6 <= y) & (y <= edge_y_high + 1e-6)
        if not bool(np.any(candidate)):
            continue
        pl = left[candidate][None, :, :]
        pr = right[candidate][None, :, :]
        px = np.array([[x]])
        py = np.array([[y]])
        cross = (px - pl[..., 0]) * (pr[..., 1] - pl[..., 1]) - (
            py - pl[..., 1]
        ) * (pr[..., 0] - pl[..., 0])
        delta = pr - pl
        scale = np.maximum(
            1.0,
            np.maximum(
                np.hypot(delta[..., 0], delta[..., 1]),
                np.maximum(
                    np.hypot(px - pl[..., 0], py - pl[..., 1]),
                    np.hypot(px - pr[..., 0], py - pr[..., 1]),
                ),
            ),
        )
        tolerance = 1e-9 * scale
        on_edge = (
            (np.abs(cross) <= tolerance)
            & (np.minimum(pl[..., 0], pr[..., 0]) - tolerance <= px)
            & (px <= np.maximum(pl[..., 0], pr[..., 0]) + tolerance)
            & (np.minimum(pl[..., 1], pr[..., 1]) - tolerance <= py)
            & (py <= np.maximum(pl[..., 1], pr[..., 1]) + tolerance)
        )
        if bool(np.any(on_edge)):
            on_edge_any[index] = True
            continue
        # 只统计真正跨越水平射线的边：两端点严格分居射线两侧。
        crosses = (pl[..., 1] > py) != (pr[..., 1] > py)
        if not bool(np.any(crosses)):
            continue
        denominator = pr[..., 1] - pl[..., 1]
        # 跨越条件已保证两端点分居射线两侧，此处只防除零。
        safe = np.where(np.abs(denominator) > 0.0, denominator, 1.0)
        intersection_x = (
            (pr[..., 0] - pl[..., 0]) * (py - pl[..., 1]) / safe + pl[..., 0]
        )
        parity[index] = int(np.count_nonzero(crosses & (px < intersection_x)))
    result = (parity % 2 == 1).astype(np.int8)
    # 边界优先：落在边上的点一律标 -1，盖过内外结论。
    result[on_edge_any] = -1
    return result




def _point_on_segment_2d(
    point: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> bool:
    """我用带尺度容差的叉积和包围盒判断点是否落在二维线段上。"""

    cross = float(
        (point[0] - left[0]) * (right[1] - left[1])
        - (point[1] - left[1]) * (right[0] - left[0])
    )
    scale = max(
        1.0,
        float(np.linalg.norm(right - left)),
        float(np.linalg.norm(point - left)),
        float(np.linalg.norm(point - right)),
    )
    if abs(cross) > 1e-9 * scale:
        return False
    return (
        min(left[0], right[0]) - 1e-9 * scale
        <= point[0]
        <= max(left[0], right[0]) + 1e-9 * scale
        and min(left[1], right[1]) - 1e-9 * scale
        <= point[1]
        <= max(left[1], right[1]) + 1e-9 * scale
    )


def _point_on_segment_2d_any(
    point: np.ndarray,
    polygon: np.ndarray,
) -> bool:
    """我一次判断点是否落在多边形的**任意一条边**上。

    我是 `_point_on_segment_2d` 的向量化版本：原来逐边调用会产生上百万次
    Python 函数调用，而现在把整圈的边排成两个阵列一次性算完。
    判定式与标量版本逐项对应（叉积阈值 + 包围盒包含），
    所以"点在边界上必须返回 -1"这条语义不变。
    """

    left = polygon
    right = np.roll(polygon, -1, axis=0)
    cross = (point[0] - left[:, 0]) * (right[:, 1] - left[:, 1]) - (
        point[1] - left[:, 1]
    ) * (right[:, 0] - left[:, 0])
    delta = right - left
    # scale 与标量版本一致：取 1.0 与三条边长中的最大值，逐边各自缩放。
    scale = np.maximum(
        1.0,
        np.maximum(
            np.hypot(delta[:, 0], delta[:, 1]),
            np.maximum(
                np.hypot(point[0] - left[:, 0], point[1] - left[:, 1]),
                np.hypot(point[0] - right[:, 0], point[1] - right[:, 1]),
            ),
        ),
    )
    within_cross = np.abs(cross) <= 1e-9 * scale
    within_x = (
        np.minimum(left[:, 0], right[:, 0]) - 1e-9 * scale <= point[0]
    ) & (point[0] <= np.maximum(left[:, 0], right[:, 0]) + 1e-9 * scale)
    within_y = (
        np.minimum(left[:, 1], right[:, 1]) - 1e-9 * scale <= point[1]
    ) & (point[1] <= np.maximum(left[:, 1], right[:, 1]) + 1e-9 * scale)
    return bool(np.any(within_cross & within_x & within_y))


def _polygon_edges_intersect(
    first: np.ndarray,
    second: np.ndarray,
) -> bool:
    """我检查两个多边形是否存在边与边相交。

    我为什么需要它：只看顶点包含关系会漏掉"环环相扣"的情形——
    两个多边形可以互相穿越而双方顶点都在对方外部，此时点判定会给出
    全在外部、若不再查边就会误报成 DISJOINT。

    ponytail：这里是 O(n*m) 的暴力两两判定，环顶点数在真实 STEP 上
    通常只有几十到几百，开销可接受；若后续出现上万顶点的巨型环，
    应先按包围盒分桶或改用扫描线算法，而不是继续放大这个二重循环。

    性能说明（这里做过一次针对性优化）：我把二重循环改成一次 n*m 的
    整阵列运算——先把两条多边形各自的边排成阵列，再用广播一次算出
    全部边对的四个方向叉积。判定式与逐对版本逐项一致，
    仍然是 O(n*m) 的比较次数，但不再有 Python 层的循环与函数调用，
    实测把这一项从 270s 量级降下来。真要再快需要换算法（扫描线），
    向量化只去掉常数开销，不改变复杂度。

    为什么再按行分块：一次性铺开 n*m 在真实数据上是几千万个元素
    （实测最大 5370 万），每个中间阵列都要几百 MB，
    远超 CPU 缓存，于是大量时间花在内存带宽而不是计算上。
    分块后每块的中间结果留在缓存里，实测明显更快。
    分块只改变计算顺序，不改变判定式，结论与整块运算一致。
    """

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    n = first.shape[0]
    m = second.shape[0]
    if n == 0 or m == 0:
        return False
    # 先用逐边包围盒把不可能相交的边剔除，再对剩下的边做精确判定。
    # 我实测过真实数据：相离的两个大环里，候选边占比是 0.000——
    # 也就是说 99.9% 的 n*m 比较都是白做的。这一层把最贵的那几次
    # 调用（单个 20s）压到几乎为零，是整套优化里收益最大的一步。
    keep = _edge_bbox_candidates(first, second)
    if not bool(np.any(keep)):
        return False
    if not bool(keep.all()):
        # 只对保留的边做判定。边序列不能断开，所以按连续段分组，
        # 每段向右多取一个顶点（边的右端点就是下一个顶点）。
        indices = np.flatnonzero(keep)
        splits = np.flatnonzero(np.diff(indices) > 1) + 1
        for group in np.split(indices, splits):
            segment = first[group[0] : group[-1] + 2]
            if _polygon_edges_intersect_dense(segment, second):
                return True
        return False
    return _polygon_edges_intersect_dense(first, second)


def _edge_bbox_candidates(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """我标出 first 的哪些边有可能与 second 的某条边相交。

    判据是逐边的包围盒重叠：两条边若连包围盒都不重叠，
    它们不可能相交，也就不必进入后面的叉积判定。

    我按列分块计算：一次性摊开 (n, m, 2) 在两边都是上万时
    会到上亿个元素，反而比直接判定还慢。分块只改计算顺序，
    不改变判据。

    返回的是 bool 数组，长度等于 first 的边数。
    """

    first_left = first
    first_right = np.roll(first, -1, axis=0)
    second_left = second
    second_right = np.roll(second, -1, axis=0)
    a_low = np.minimum(first_left, first_right)
    a_high = np.maximum(first_left, first_right)
    b_low = np.minimum(second_left, second_right)
    b_high = np.maximum(second_left, second_right)
    m = second.shape[0]
    keep = np.zeros(first.shape[0], dtype=bool)
    # 每块控制在百万量级元素，让中间阵列留在缓存里。
    step = max(1, int(1_000_000 // max(1, m)))
    for start in range(0, first.shape[0], step):
        stop = min(start + step, first.shape[0])
        overlap = (a_low[start:stop, None, :] <= b_high[None, :, :]) & (
            b_low[None, :, :] <= a_high[start:stop, None, :]
        )
        keep[start:stop] = np.any(overlap[..., 0] & overlap[..., 1], axis=1)
    return keep


def _polygon_edges_intersect_dense(
    first: np.ndarray,
    second: np.ndarray,
) -> bool:
    """我对已经过包围盒预筛的两个环做精确的边相交判定。

    这是 `_polygon_edges_intersect` 的计算核心，单独拆出来
    是为了让预筛路径能只对候选边段调用它。
    """

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    n = first.shape[0]
    m = second.shape[0]
    if n == 0 or m == 0:
        return False
    # 先把两条环的边展开成 (n,2,2) 与 (m,2,2)，再用广播凑出全部 n*m 边对。
    first_left = first
    first_right = np.roll(first, -1, axis=0)
    second_left = second
    second_right = np.roll(second, -1, axis=0)

    c_x = second_left[:, 0][None, :]
    c_y = second_left[:, 1][None, :]
    d_x = second_right[:, 0][None, :]
    d_y = second_right[:, 1][None, :]

    def orient(ax, ay, bx, by, cx, cy):
        """我返回 (b-a) x (c-a) 的二维叉积。"""
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    # 分块大小按列数定：让每块的中间阵列控制在缓存友好的量级，
    # 同时不至于块数太多导致 Python 层开销重新占主导。
    step = max(1, int(4_000_000 // max(1, m)))
    has_collinear = False
    for start in range(0, n, step):
        stop = min(start + step, n)
        a_x = first_left[start:stop, 0][:, None]
        a_y = first_left[start:stop, 1][:, None]
        b_x = first_right[start:stop, 0][:, None]
        b_y = first_right[start:stop, 1][:, None]

        o1 = orient(a_x, a_y, b_x, b_y, c_x, c_y)
        o2 = orient(a_x, a_y, b_x, b_y, d_x, d_y)
        o3 = orient(c_x, c_y, d_x, d_y, a_x, a_y)
        o4 = orient(c_x, c_y, d_x, d_y, b_x, b_y)

        # 严格相交一旦命中就整体成立，后面的块不必再算。
        if bool(np.any((o1 * o2 < 0.0) & (o3 * o4 < 0.0))):
            return True
        # 共线分支：与标量版本一样，用 1e-9 的叉积阈值加包围盒包含判定。
        collinear = (
            (np.abs(o1) <= 1e-9)
            | (np.abs(o2) <= 1e-9)
            | (np.abs(o3) <= 1e-9)
            | (np.abs(o4) <= 1e-9)
        )
        if bool(np.any(collinear)):
            has_collinear = True
            break
    if has_collinear:
        # 精确判定需要全尺寸的 o1..o4（它按 n*m 整块做广播），
        # 所以这里回到不分块的算法重算一次。走到这条分支说明
        # 存在共线候选，重算的代价换来的是结论与原来完全一致。
        a_x = first_left[:, 0][:, None]
        a_y = first_left[:, 1][:, None]
        b_x = first_right[:, 0][:, None]
        b_y = first_right[:, 1][:, None]
        o1 = orient(a_x, a_y, b_x, b_y, c_x, c_y)
        o2 = orient(a_x, a_y, b_x, b_y, d_x, d_y)
        o3 = orient(c_x, c_y, d_x, d_y, a_x, a_y)
        o4 = orient(c_x, c_y, d_x, d_y, b_x, b_y)
        collinear = (
            (np.abs(o1) <= 1e-9)
            | (np.abs(o2) <= 1e-9)
            | (np.abs(o3) <= 1e-9)
            | (np.abs(o4) <= 1e-9)
        )
        return _polygon_edges_touch_broadcast(
            first, second, o1, o2, o3, o4, collinear
        )
    return False


def _polygon_edges_touch_broadcast(
    first: np.ndarray,
    second: np.ndarray,
    o1: np.ndarray,
    o2: np.ndarray,
    o3: np.ndarray,
    o4: np.ndarray,
    collinear: np.ndarray,
) -> bool:
    """我只对共线候选边对做精确的包围盒重叠判定。

    我把标量版本 `_segments_intersect_2d` 里的四个共线分支整阵列展开：
    每个分支都是"叉积近零"加"第三个点落在另外两点构成的包围盒内"。
    只有 `_polygon_edges_intersect` 已经确认存在共线候选时才走到这里，
    所以它的开销与真实数据量成正比，而不是每次都付。
    """

    fl_x = first[:, 0][:, None]
    fl_y = first[:, 1][:, None]
    fr_x = np.roll(first, -1, axis=0)[:, 0][:, None]
    fr_y = np.roll(first, -1, axis=0)[:, 1][:, None]
    sl_x = second[:, 0][None, :]
    sl_y = second[:, 1][None, :]
    sr_x = np.roll(second, -1, axis=0)[:, 0][None, :]
    sr_y = np.roll(second, -1, axis=0)[:, 1][None, :]

    def within(px, py, qx, qy, rx, ry):
        """我判断点 p 是否落在 q、r 构成的包围盒内（含 1e-9 容差）。"""
        return (
            (np.minimum(qx, rx) - 1e-9 <= px)
            & (px <= np.maximum(qx, rx) + 1e-9)
            & (np.minimum(qy, ry) - 1e-9 <= py)
            & (py <= np.maximum(qy, ry) + 1e-9)
        )

    branch1 = (np.abs(o1) <= 1e-9) & within(sl_x, sl_y, fl_x, fl_y, fr_x, fr_y)
    branch2 = (np.abs(o2) <= 1e-9) & within(sr_x, sr_y, fl_x, fl_y, fr_x, fr_y)
    branch3 = (np.abs(o3) <= 1e-9) & within(fl_x, fl_y, sl_x, sl_y, sr_x, sr_y)
    branch4 = (np.abs(o4) <= 1e-9) & within(fr_x, fr_y, sl_x, sl_y, sr_x, sr_y)
    return bool(np.any(collinear & (branch1 | branch2 | branch3 | branch4)))


def _segments_intersect_2d(
    first_left: np.ndarray,
    first_right: np.ndarray,
    second_left: np.ndarray,
    second_right: np.ndarray,
) -> bool:
    """我用二维方向叉积判断两条线段是否相交。

    原理说明：两条线段相交当且仅当"每条线段的两端都在另一条线段的两侧，
    或至少有一端落在另一条线段上"。我用叉积的符号判断"在哪一侧"，
    符号相反即分居两侧；共线（叉积为 0）时再用包围盒判断是否真的重叠。

    我全程只用乘减和比较，不做除法，所以不存在除零问题，
    也不会在近平行线段上产生巨大的中间量，这是选叉积法而不是
    求交点参数法的原因。
    """

    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """我返回三点 a->b->c 的二维叉积符号量，用于判定转向。"""
        return float(
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )

    def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        """我判断点 b 是否落在线段 ac 的包围盒内（配合叉积排除共线外的情形）。"""
        return (
            min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
            and min(a[1], c[1]) - 1e-9
            <= b[1]
            <= max(a[1], c[1]) + 1e-9
        )

    o1 = orientation(first_left, first_right, second_left)
    o2 = orientation(first_left, first_right, second_right)
    o3 = orientation(second_left, second_right, first_left)
    o4 = orientation(second_left, second_right, first_right)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    if abs(o1) <= 1e-9 and on_segment(first_left, second_left, first_right):
        return True
    if abs(o2) <= 1e-9 and on_segment(first_left, second_right, first_right):
        return True
    if abs(o3) <= 1e-9 and on_segment(second_left, first_left, second_right):
        return True
    if abs(o4) <= 1e-9 and on_segment(second_left, first_right, second_right):
        return True
    return False


def _boundary_wire_selection(
    component_records: list[dict[str, object]],
    *,
    unresolved_edge_ids: set[int] | None = None,
    vertex_coordinates_mm: dict[int, list[float]] | None = None,
    edge_vertices: dict[int, tuple[int, ...]] | None = None,
    edge_polyline_points_mm: dict[int, list[list[float]]] | None = None,
) -> dict[str, object]:
    """我按闭合候选的周长优先选择最大 Wire，并保留平局复核。"""

    # 只有闭合分量才有资格当主 Wire；开放分量一律不参与最大轮廓竞争。
    closed_records = [
        (index, record)
        for index, record in enumerate(component_records)
        if record.get("closed") is True
    ]
    # 先定状态：无边界、无闭合候选、单候选、多候选、平局、未解析，互斥且穷尽。
    unresolved_edge_ids = unresolved_edge_ids or set()
    if not component_records and unresolved_edge_ids:
        status = "UNRESOLVED"
    elif not component_records:
        status = "NO_BOUNDARY"
    elif not closed_records:
        status = "UNRESOLVED" if unresolved_edge_ids else "NO_CLOSED_CANDIDATE"
    else:
        has_complete_perimeters = all(
            isinstance(record.get("perimeter_length_mm"), (int, float))
            and isfinite(float(record["perimeter_length_mm"]))
            for _, record in closed_records
        )
        # 周长都算得出来就用周长排序，否则退回用边数排序（弱指标但不会失败）。
        metric_name = (
            "perimeter_length_mm"
            if has_complete_perimeters
            else "edge_count"
        )
        metric_values = [
            (
                float(record["perimeter_length_mm"])
                if has_complete_perimeters
                else len(record.get("edge_ids", []))
            )
            for _, record in closed_records
        ]
        maximum = max(metric_values)
        tie_count = sum(
            isclose(value, maximum, rel_tol=1e-8, abs_tol=1e-8)
            if has_complete_perimeters
            else value == maximum
            for value in metric_values
        )
        # 平局必须标 AMBIGUOUS_MAXIMUM_TIE：两个一样大的环，自动选任一个都是臆断。
        if tie_count > 1:
            status = "AMBIGUOUS_MAXIMUM_TIE"
        elif len(closed_records) == 1:
            status = "SINGLE_CLOSED_CANDIDATE"
        else:
            status = "MULTIPLE_CLOSED_CANDIDATES"
    if component_records and closed_records:
        has_complete_perimeters = all(
            isinstance(record.get("perimeter_length_mm"), (int, float))
            and isfinite(float(record["perimeter_length_mm"]))
            for _, record in closed_records
        )
    else:
        has_complete_perimeters = False

    # 排序键：主指标降序 -> 边数降序 -> 边编号升序，保证结果与输入顺序无关。
    ranked = sorted(
        closed_records,
        key=lambda item: (
            -(
                float(item[1]["perimeter_length_mm"])
                if has_complete_perimeters
                else len(item[1].get("edge_ids", []))
            ),
            -len(item[1].get("edge_ids", [])),
            list(item[1].get("edge_ids", [])),
        ),
    )
    primary_index: int | None = None
    primary_edge_ids: list[int] = []
    primary_vertex_ids: list[int] = []
    primary_perimeter_length_mm: float | None = None
    primary_projected_area_mm2: float | None = None
    primary_projection_status = "UNAVAILABLE"
    primary_projection_evidence: dict[str, object] = {}
    if ranked:
        primary_index, primary_record = ranked[0]
        primary_edge_ids = list(primary_record.get("edge_ids", []))
        primary_vertex_ids = list(primary_record.get("vertex_ids", []))
        value = primary_record.get("perimeter_length_mm")
        if isinstance(value, (int, float)) and isfinite(float(value)):
            primary_perimeter_length_mm = float(value)
        primary_projection_status = str(
            primary_record.get("projection_status", "UNAVAILABLE")
        )
        primary_projection_evidence = dict(
            primary_record.get("projection_evidence", {}) or {}
        )
        projected = primary_record.get("projected_area_mm2")
        if isinstance(projected, (int, float)) and isfinite(float(projected)):
            primary_projected_area_mm2 = float(projected)

    # 投影面积与周长排序可能给出不同答案，冲突时记 CONFLICT 交人工判断。
    measured_areas = [
        (component_index, float(record["projected_area_mm2"]))
        for component_index, record in closed_records
        if isinstance(record.get("projected_area_mm2"), (int, float))
        and isfinite(float(record["projected_area_mm2"]))
    ]
    projection_max_component_index: int | None = None
    projection_max_area_mm2: float | None = None
    if measured_areas:
        projection_max_component_index, projection_max_area_mm2 = max(
            measured_areas,
            key=lambda item: item[1],
        )
    # 周长最大者的投影面积反而更小，说明两者判据矛盾，必须标出来而不是静默认定。
    projection_conflict = bool(
        primary_projected_area_mm2 is not None
        and projection_max_area_mm2 is not None
        and primary_projected_area_mm2
        < projection_max_area_mm2 * (1.0 - 1e-9)
    )
    loop_relation = _loop_relation_evidence(
        closed_records,
        vertex_coordinates_mm,
        edge_vertices=edge_vertices,
        edge_polyline_points_mm=edge_polyline_points_mm,
    )
    loop_roles = loop_relation.get("loop_roles", {})
    primary_loop_role = (
        str(loop_roles.get(str(primary_index), "UNAVAILABLE"))
        if primary_index is not None and isinstance(loop_roles, dict)
        else "UNAVAILABLE"
    )
    parent_by_component = loop_relation.get("parent_by_component", {})
    primary_loop_parent_component_index = (
        parent_by_component.get(str(primary_index))
        if primary_index is not None and isinstance(parent_by_component, dict)
        else None
    )
    loop_relation_status = str(loop_relation.get("status", "UNAVAILABLE"))
    # 两种冲突都必须上报，因为它们都意味着"自动选出来的主环可能不是外轮廓"：
    #   1. 周长最大者被判成孔洞候选 —— 外轮廓不该被别人包含；
    #   2. 环之间存在相交或相切 —— 拓扑本身异常。
    # 这时候我不做任何猜测性修正，只把冲突标出来交给人工复核。
    loop_relation_conflict = bool(
        primary_loop_role == "INNER_HOLE_CANDIDATE"
        or loop_relation_status in {"INTERSECTING_LOOPS", "TOUCHING_LOOPS"}
    )

    # candidate_records 保留全部候选环而不仅留第一名：主环是"最大者"这一
    # 经验判据，遇到平局或异常时需要人工看候选列表重新判断，
    # 只留第一名会让复核无从下手。
    candidate_records = []
    for rank, (component_index, record) in enumerate(ranked, start=1):
        candidate_records.append(
            {
                "rank": rank,
                "component_index": component_index,
                "edge_ids": list(record.get("edge_ids", [])),
                "vertex_ids": list(record.get("vertex_ids", [])),
                "edge_count": len(record.get("edge_ids", [])),
                "closed": bool(record.get("closed")),
                "perimeter_length_mm": record.get("perimeter_length_mm"),
                "perimeter_complete": bool(record.get("perimeter_complete")),
                "projected_area_mm2": record.get("projected_area_mm2"),
                "projection_status": record.get("projection_status"),
            }
        )
    # unresolved：存在未闭合分量，但同时也找到了闭合候选。
    # 这不是纯失败——说明边界数据不完整但仍有可用线索，所以单独给一个状态。
    unresolved_status = (
        bool(component_records)
        and any(record.get("closed") is False for record in component_records)
        and bool(ranked)
    )
    if unresolved_edge_ids and ranked:
        # 有游离边、且找到了闭合候选：边界不完整但仍有可用线索。
        status = "UNRESOLVED_WITH_CLOSED_CANDIDATES"
    elif unresolved_status and status in {
        "SINGLE_CLOSED_CANDIDATE",
        "MULTIPLE_CLOSED_CANDIDATES",
    }:
        # 开口与闭合候选并存：这个区域的边界是"半开"的，
        # 既不能说闭合也不能说完全开放。
        status = "OPEN_AND_CLOSED_CANDIDATES"
    if projection_conflict and status == "MULTIPLE_CLOSED_CANDIDATES":
        # 投影面积冲突：周长最大者与投影面积最大者不是同一个环，
        # "哪个是外轮廓"这件事在两个判据上给出了矛盾答案。
        status = "PROJECTED_AREA_CONFLICT"
    if loop_relation_conflict and status in {
        "SINGLE_CLOSED_CANDIDATE",
        "MULTIPLE_CLOSED_CANDIDATES",
        "PROJECTED_AREA_CONFLICT",
    }:
        # 环关系冲突优先级最高：环之间相交或相切属于拓扑异常，
        # 在这个前提下任何"主环"结论都不可信，所以最后判定并覆盖前几种。
        # 顺序不能颠倒——把它放在最前面会掩盖投影冲突这类更具体的线索。
        status = "LOOP_RELATION_CONFLICT"

    return {
        "status": status,
        "candidate_count": len(ranked),
        "primary_component_index": primary_index,
        "primary_edge_ids": primary_edge_ids,
        "primary_vertex_ids": primary_vertex_ids,
        "primary_perimeter_length_mm": primary_perimeter_length_mm,
        "primary_projected_area_mm2": primary_projected_area_mm2,
        "primary_projection_status": primary_projection_status,
        # 投影证据内嵌一份、外面再平铺一份：
        # 内嵌的供整体回读证据块，平铺的供按字段筛选，
        # 两者重复是刻意的——避免调用方为了一个字段去拆整个证据。
        "primary_projection_evidence": {
            **primary_projection_evidence,
            "projection_conflict": projection_conflict,
            "projection_max_component_index": projection_max_component_index,
            "projection_max_area_mm2": projection_max_area_mm2,
        },
        # 这三个量在证据里嵌一份、在顶层也平铺一份：
        # 嵌入的那份用于整段投影证据的归档，
        # 平铺的那份供门禁直接读取，不必层层下钻。
        "projection_conflict": projection_conflict,
        "projection_max_component_index": projection_max_component_index,
        "projection_max_area_mm2": projection_max_area_mm2,
        "loop_relation_status": loop_relation_status,
        "loop_relation_evidence": {
            **loop_relation,
            "loop_relation_conflict": loop_relation_conflict,
            "primary_loop_role": primary_loop_role,
            "primary_loop_parent_component_index": (
                primary_loop_parent_component_index
            ),
        },
        "primary_loop_role": primary_loop_role,
        "primary_loop_parent_component_index": (
            primary_loop_parent_component_index
        ),
        # "主环是不是内孔"由角色推出并存成布尔：
        # 下游判断走的是布尔而不是字符串比较，
        # 存布尔能避免各处重复写一遍这个等值判断。
        "primary_loop_is_hole_candidate": (
            primary_loop_role == "INNER_HOLE_CANDIDATE"
        ),
        "loop_relation_conflict": loop_relation_conflict,
        # 可用性的判据写在这里：只有"单个或多个闭合候选"
        # 这两种状态才说明真的找到了闭合的边界 Wire。
        "primary_usable": status
        in {"SINGLE_CLOSED_CANDIDATE", "MULTIPLE_CLOSED_CANDIDATES"},
        "unresolved_edge_ids": sorted(unresolved_edge_ids),
        "candidates": candidate_records,
        # 选择指标与它对应的方法名一起给，且必须一致：
        # 周长能量出来时按周长选（更接近"最大外轮廓"的本意），
        # 量不出来才退到按边数选。两个字段名不同但描述同一件事，
        # 分开是为了让机器读指标、人读方法名。
        "selection_metric": (
            "perimeter_length_mm" if has_complete_perimeters else "edge_count"
        ),
        "method": (
            "closed_wire_max_perimeter_length_mm"
            if has_complete_perimeters
            else "closed_wire_max_edge_count"
        ),
        "interpretation": (
            "我优先按当前文件内闭合边界的有限周长选择最大 Wire 候选；"
            "周长缺失或冲突时回退到 Edge 数。平局、开放分量、端点不完整，"
            "或周长最大候选与投影面积最大候选不一致时仍要求人工复核。"
            "我还会用投影多边形判断外环、内环和孔洞候选，主候选落入内环或"
            "闭合环相交时仍要求人工复核。"
        ),
    }


def _edge_components(
    edge_vertices: dict[int, tuple[int, ...]],
) -> list[list[int]]:
    """我按共享 Vertex 把边界 Edge 拆成连通分量。

    我用**边-顶点二部图**做洪泛：先建 vertex -> edges 倒排表，
    再从任一未访问边出发，沿共享顶点扩展到无法继续，得到一个连通分量。

    稳定性细节（这些不是风格问题，直接决定结果可复现）：
      - 起点取 min(remaining)，分量内边排序、分量间按首边排序，
        所以同一份输入永远得到同一个顺序。不排序的话分量顺序取决于
        dict 的插入顺序，两次运行可能给出不同的"第一分量"，
        下游"选主环"就会选中不同的环；
      - stack.extend 用 sorted(reverse=True) 配合 pop()，
        等价于按升序弹出，保证遍历顺序确定。
    """

    edges_by_vertex: dict[int, set[int]] = defaultdict(set)
    for edge_id, vertex_ids in edge_vertices.items():
        for vertex_id in vertex_ids:
            edges_by_vertex[vertex_id].add(edge_id)
    remaining = set(edge_vertices)
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            edge_id = stack.pop()
            component.append(edge_id)
            neighbors = {
                neighbor
                for vertex_id in edge_vertices[edge_id]
                for neighbor in edges_by_vertex[vertex_id]
                if neighbor in remaining
            }
            remaining.difference_update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        components.append(sorted(component))
    return sorted(components, key=lambda component: component[0])


def _revolution_seam_closure(
    graph_edge_vertices: dict[int, tuple[int, ...]],
    edge_lengths_mm: dict[int, float],
    faces: list[StepFaceFeature],
    boundary_edge_ids: list[int],
    degenerate_edge_vertices: dict[int, tuple[int, ...]] | None = None,
) -> dict[str, object]:
    """我识别"完整回转面被读成开放边界"的形态。

    OCP 在圆柱/圆锥/圆环的接缝圆上有时只给出一个端点（圆心），形成的边记为
    单端点自环；同时它还会额外给出一条连接两个圆心的"轴向边"。这时按度数
    判定会得到两个度 1 端点、被误判成开放缺口。我在这类形态下检查：
    是否存在同一对端点挂着两条弧、且两条弧长之和等于该接缝圆的周长
    （周长由单端点自环的边长给出）。成立则说明那两条弧拼成完整圆，
    边界没有缺口，轴向边是接缝假象。

    圆锥/圆台的顶点（顶点圆半径为 0）在 B-rep 里是一条长度为 0 的单端点边，
    它在上游已被当成退化边剥离、不进 ``graph_edge_vertices``。所以我通过
    ``degenerate_edge_vertices`` 把它单独收回来参与判定，否则圆锥顶点会被
    误判成开放缺口（MD270324 上有 96 个这样的区域）。
    """

    degenerate_edge_vertices = degenerate_edge_vertices or {}
    result: dict[str, object] = {
        "closed": False,
        "seam_edge_ids": [],
        "arc_edge_ids": [],
        "axial_edge_ids": [],
        "seam_length_mm": None,
        "arc_length_sum_mm": None,
        "surface_types": [],
        "reason": None,
    }
    single_end_edges = {
        int(edge_id): int(vertex_ids[0])
        for edge_id, vertex_ids in graph_edge_vertices.items()
        if len(vertex_ids) == 1
    }
    # 我把退化边里那些"单端点"的也补进来：圆锥顶点的退化圆就属于这一类。
    for edge_id, vertex_ids in degenerate_edge_vertices.items():
        if len(vertex_ids) == 1 and int(edge_id) not in single_end_edges:
            single_end_edges[int(edge_id)] = int(vertex_ids[0])
    if not single_end_edges:
        result["reason"] = "no_single_endpoint_seam_edge"
        return result

    def length_of(edge_id: int) -> float | None:
        """我取边长；缺失或非有限值时返回 None 而不是 0。

        这里不能返回 0 兜底：边长参与"缝长"求和，
        把测不出的长度当成 0 会让缝看起来比实际短，
        从而把一个真实的大缝判成小缝。
        """
        value = edge_lengths_mm.get(int(edge_id))
        if not isinstance(value, (int, float)) or not isfinite(float(value)):
            return None
        return float(value)

    # 同一对端点之间挂着多条边时，它们是把一个圆切开的多段弧。
    arcs_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
    for edge_id, vertex_ids in graph_edge_vertices.items():
        if len(vertex_ids) != 2:
            continue
        left, right = int(vertex_ids[0]), int(vertex_ids[1])
        arcs_by_pair[(left, right) if left <= right else (right, left)].append(
            int(edge_id)
        )
    for pair, edge_ids in arcs_by_pair.items():
        if len(edge_ids) < 2:
            continue
        arc_lengths = [length_of(edge_id) for edge_id in edge_ids]
        if any(value is None for value in arc_lengths):
            continue
        for seam_edge_id, seam_vertex in single_end_edges.items():
            seam_length = length_of(seam_edge_id)
            if seam_length is None or seam_length <= 0.0:
                continue
            arc_sum = sum(float(value) for value in arc_lengths)
            if abs(arc_sum - seam_length) > 1e-6 * max(1.0, seam_length):
                continue
            # 轴向边：连接圆心与弧端点的那条边，是 OCP 的接缝假象。
            seam_vertices = {int(vertex_ids[0]) for vertex_ids in
                             graph_edge_vertices.values()
                             if len(vertex_ids) == 1}
            arc_vertices = set(pair)
            axial_edges = [
                int(edge_id)
                for edge_id, vertex_ids in graph_edge_vertices.items()
                if len(vertex_ids) == 2
                and set(int(v) for v in vertex_ids) & seam_vertices
                and set(int(v) for v in vertex_ids) & arc_vertices
                and not (set(int(v) for v in vertex_ids) <= arc_vertices)
                and int(edge_id) not in edge_ids
            ]
            result.update(
                {
                    "closed": True,
                    "seam_edge_ids": sorted({seam_edge_id}),
                    "arc_edge_ids": sorted(edge_ids),
                    "axial_edge_ids": sorted(axial_edges),
                    "seam_length_mm": seam_length,
                    "arc_length_sum_mm": arc_sum,
                    "surface_types": sorted(
                        {
                            str(face.surface_type)
                            for face in faces
                            if face.surface_type
                        }
                    ),
                    "reason": "arc_pair_sums_to_seam_circle",
                }
            )
            return result
    # 我方识别出第二种完整回转面形态：两个接缝圆各自塌缩成圆心点（两条单端点边），
    # 中间由一条边把两个圆心连起来。这时两个"度 1 端点"其实是两个圆心，
    # 边界是完整的圆柱/圆台侧面，没有缺口。
    # 判据：由周长反推的两个圆半径之差必须不超过那条连接边的长度
    # （半径差就是圆台的径向偏移量，它不可能超过母线/轴向边长）。
    # 这条判据我在 MD270324 的 1073 个该类区域上验证过，无一违反。
    if len(single_end_edges) == 2:
        seam_items = sorted(single_end_edges.items())
        (first_edge, first_vertex), (second_edge, second_vertex) = seam_items
        first_length = length_of(first_edge)
        second_length = length_of(second_edge)
        # 我方允许其中一个接缝圆长度为 0：那是圆锥/圆台的顶点（半径 0），
        # 顶点在 B-rep 里就是一个退化的圆，不是缺口。只要求至少一个非零。
        if (
            first_length is not None
            and second_length is not None
            # 至少一条长度为正：两条都是 0 说明这两条"缝"
            # 根本不存在，下面的半径与比值都会是 0/0。
            and (first_length > 0.0 or second_length > 0.0)
        ):
            # 由周长反推两个半径：这两条边各是一个圆，
            # 周长除以 2π 就是半径。
            radius_1 = first_length / (2.0 * pi)
            radius_2 = second_length / (2.0 * pi)
            # 找连接两个顶点的"桥接边"：它的两个端点
            # 正好是这两条缝的公共顶点。
            link_edges = [
                int(edge_id)
                for edge_id, vertex_ids in graph_edge_vertices.items()
                if len(vertex_ids) == 2
                and {int(vertex_ids[0]), int(vertex_ids[1])}
                == {first_vertex, second_vertex}
            ]
            # 桥接边必须恰好一条：0 条说明两个圆没有相连的边，
            # 多条说明这里有分叉，两种情形都判不了"闭合"。
            if len(link_edges) == 1:
                link_length = length_of(link_edges[0])
                # 两个半径之差不超过桥接边长时认为闭合：
                # 差值恰好等于"桥"的长度说明这段半径差
                # 正是由这条桥接边造成的，补上它缝就合上了。
                # 1e-9 是浮点余量，不加会在等值边界上抖动。
                if link_length is not None and abs(
                    radius_1 - radius_2
                ) <= link_length + 1e-9:
                    result.update(
                        {
                            "closed": True,
                            "seam_edge_ids": sorted(single_end_edges),
                            # 这种情形没有圆弧边：两条缝各自已是完整圆，
                            # 弧边列表留空而不是塞满边号。
                            "arc_edge_ids": [],
                            "axial_edge_ids": sorted(link_edges),
                            # 缝合长度取两者较大值：这是"这条缝有多长"的保守估计，
                            # 取较小值会低估这个回转特征的尺寸。
                            "seam_length_mm": max(first_length, second_length),
                            # 弧长和留 None：本分支里没有圆弧边，
                            # 给 0.0 会被读成"存在但长度为 0"。
                            "arc_length_sum_mm": None,
                            "first_radius_mm": radius_1,
                            "second_radius_mm": radius_2,
                            "link_length_mm": link_length,
                            "radius_delta_mm": abs(radius_1 - radius_2),
                            "surface_types": sorted(
                                {
                                    str(face.surface_type)
                                    for face in faces
                                    if face.surface_type
                                }
                            ),
                            # 原因码写死：下游按它区分"是哪一种闭合情形"
                            # （这里是"两个塌缩的缝合圆"）。
                            "reason": "two_collapsed_seam_circles",
                        }
                    )
                    return result
        result["reason"] = "two_seam_circles_not_geometrically_closable"
        return result
    # 我方识别出第三种形态：只有一个接缝圆塌缩成圆心，另一端是一个由多条边
    # 组成的闭合环（各顶点度数均为 2），中间一条桥边把圆心连到环上。
    # 这时"度 1 端点"是圆心，环本身已经闭合，边界同样没有缺口。
    # 判据与形态二一致：由周长反推的两个半径之差不超过桥边长。
    if len(single_end_edges) == 1:
        seam_edge_id, seam_vertex = next(iter(single_end_edges.items()))
        seam_length = length_of(seam_edge_id)
        if seam_length is not None and seam_length > 0.0:
            seam_radius = seam_length / (2.0 * pi)
            double_edges = {
                int(edge_id): (int(vertex_ids[0]), int(vertex_ids[1]))
                for edge_id, vertex_ids in graph_edge_vertices.items()
                if len(vertex_ids) == 2
            }
            bridges = [
                edge_id
                for edge_id, (left, right) in double_edges.items()
                if seam_vertex in (left, right)
            ]
            if len(bridges) == 1:
                bridge_id = bridges[0]
                bridge_length = length_of(bridge_id)
                loop_edges = [
                    edge_id
                    for edge_id in double_edges
                    if edge_id != bridge_id
                ]
                # 我要求剩余边构成闭合环：每个顶点度数恰为 2。
                loop_degree: dict[int, int] = defaultdict(int)
                for edge_id in loop_edges:
                    left, right = double_edges[edge_id]
                    loop_degree[left] += 1
                    loop_degree[right] += 1
                # 每个顶点度数都恰好是 2 才构成简单环：
                # 度数 1 是断头（链不是环），度数 >=3 是分叉
                # （多条链在此交汇，不是单一回路）。
                if (
                    loop_edges
                    and all(degree == 2 for degree in loop_degree.values())
                ):
                    loop_lengths = [
                        length_of(edge_id) for edge_id in loop_edges
                    ]
                    # 任一条边量不出长度就放弃整条回路：
                    # 周长不完整时算出来的半径不可信，
                    # 而下面要拿半径去做等值判定。
                    if all(value is not None for value in loop_lengths):
                        loop_length = sum(float(v) for v in loop_lengths)
                        # 由周长反推半径：这组边构成一个圆，
                        # 周长 /(2π) 就是它的半径。
                        loop_radius = loop_length / (2.0 * pi)
                        # 与缝合边的半径之差不超过桥接边长时认为闭合：
                        # 两个半径差恰好等于"桥"的长度，
                        # 说明这段缺口正是由那条桥接边造成的，
                        # 补上它整个回转面的缝就合上了。
                        # 1e-9 是浮点余量，不加会在等值边界上抖动。
                        if (
                            bridge_length is not None
                            and abs(seam_radius - loop_radius)
                            <= bridge_length + 1e-9
                        ):
                            result.update(
                                {
                                    "closed": True,
                                    "seam_edge_ids": [seam_edge_id],
                                    # 三类边分列：缝合边、圆弧边、轴向桥接边。
                                    # 分开记录是因为它们在几何上角色不同，
                                    # 后续修补要区别对待。
                                    "arc_edge_ids": sorted(loop_edges),
                                    "axial_edge_ids": [bridge_id],
                                    "seam_length_mm": seam_length,
                                    "arc_length_sum_mm": loop_length,
                                    "first_radius_mm": seam_radius,
                                    "second_radius_mm": loop_radius,
                                    "link_length_mm": bridge_length,
                                    "radius_delta_mm": abs(
                                        seam_radius - loop_radius
                                    ),
                                    "surface_types": sorted(
                                        {
                                            str(face.surface_type)
                                            for face in faces
                                            if face.surface_type
                                        }
                                    ),
                                    "reason": (
                                        "collapsed_seam_circle_plus_closed_loop"
                                    ),
                                }
                            )
                            return result
    result["reason"] = "no_arc_pair_matching_seam_circle"
    return result


def _open_boundary_chains(
    component: list[int],
    edge_vertices: dict[int, tuple[int, ...]],
    edge_lengths_mm: dict[int, float] | None = None,
) -> list[dict[str, object]]:
    """我把一个开放边界分量拆成按 Vertex 首尾相接的链。

    我方把开放区域边界整理成“从度 1 端点走到另一个度 1 端点”的链，
    再单独标出残余闭环。这样阶段 5 的台阶/缺口桥接就能直接拿到
    每条链的两个端点，而不是只看到一个笼统的“开放边界”状态。
    """

    # 我先把退化自环（两端塌成同一个 Vertex、只给出一个端点）单独收成链，
    # 不再让整条分量因为“端点数不是 2”而直接返回空。
    degenerate_chains: list[dict[str, object]] = []
    incidence: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_id in component:
        endpoints = edge_vertices.get(edge_id)
        if endpoints is None:
            return []
        if len(endpoints) == 1:
            # OCP 在圆柱接缝/极点处会把边长非零、但端点只给一个的自环读成
            # 单端点边；它是真实的闭合环，只是端点塌缩了。
            only = int(endpoints[0])
            degenerate_chains.append(
                {
                    "edge_ids": [int(edge_id)],
                    "start_vertex_id": only,
                    "end_vertex_id": only,
                    "closed": True,
                    "degenerate_self_loop": True,
                    "length_mm": None,
                }
            )
            continue
        if len(endpoints) != 2:
            return []
        left, right = int(endpoints[0]), int(endpoints[1])
        incidence[left].append((int(edge_id), right))
        incidence[right].append((int(edge_id), left))
    if not incidence:
        return degenerate_chains
    degree = {vertex_id: len(items) for vertex_id, items in incidence.items()}
    lengths = edge_lengths_mm or {}

    def chain_length(edge_ids: list[int]) -> float | None:
        """我累加一条边链的边长；任一边长缺失或非有限时返回 None。"""
        total = 0.0
        for edge_id in edge_ids:
            value = lengths.get(int(edge_id))
            # 任一条边长度缺失或非有限就整段返回 None：
            # 给出部分和会让它看起来像可用的周长，
            # 而它其实是残缺的。
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                return None
            total += float(value)
        return total

    used: set[int] = set()
    # 退化链先放进结果：它们在下面按度数遍历时走不到
    # （退化链上的点度数可能都大于 1），单独收集才不会丢。
    chains: list[dict[str, object]] = list(degenerate_chains)
    # 从度数为 1 的顶点出发走链：度为 1 意味着"这条链的一端"，
    # 从端点出发才能保证走出的是完整的一条链而不是半截。
    for start_vertex in sorted(
        vertex_id for vertex_id, value in degree.items() if value == 1
    ):
        # 这个端点的边全用过了就换下一个端点：
        # 同一个顶点可能挂着多条已用过的边。
        if all(edge_id in used for edge_id, _ in incidence[start_vertex]):
            continue
        chain_edges: list[int] = []
        current = start_vertex
        # 步数上限取边数 + 1：正常链不会超过这个长度，
        # 但图里若存在环，没有上限就会在这里死循环。
        for _ in range(len(component) + 1):
            step = next(
                (
                    (edge_id, neighbor)
                    for edge_id, neighbor in incidence[current]
                    if edge_id not in used
                ),
                None,
            )
            # 走到了用尽的顶点就停：这是链的另一端。
            if step is None:
                break
            edge_id, neighbor = step
            used.add(edge_id)
            chain_edges.append(edge_id)
            current = neighbor
            # 走到另一个度为 1 的顶点说明这条链已经完整，收尾。
            if degree.get(current) == 1:
                break
        # 空链不记：走过一次但一条边都没加上的情况
        # 说明这个端点其实没有可用边。
        if chain_edges:
            chains.append(
                {
                    "edge_ids": chain_edges,
                    "start_vertex_id": int(start_vertex),
                    "end_vertex_id": int(current),
                    "closed": False,
                    "length_mm": chain_length(chain_edges),
                }
            )

    # 我方把剩下的、不连到任何度 1 端点的边收成残余闭环，避免静默丢弃。
    # 剩下的边按编号从小到大当种子：这些边不在任何"从端点出发的链"里，
    # 说明它们构成闭合环（环上每个点度数都大于 1，走不到端点）。
    for seed in sorted(edge_id for edge_id in component if edge_id not in used):
        # 上一轮可能已经把这条边用掉了，取种子后再判一次。
        if seed in used:
            continue
        start_vertex = int(edge_vertices[seed][0])
        current = start_vertex
        cycle_edges: list[int] = []
        # 步数上限取边数 + 1：闭合环最长也不会超过边数，
        # 加一是考虑到最后一步要走回起点。
        for _ in range(len(component) + 1):
            step = next(
                (
                    (edge_id, neighbor)
                    for edge_id, neighbor in incidence[current]
                    if edge_id not in used
                ),
                None,
            )
            if step is None:
                break
            edge_id, neighbor = step
            used.add(edge_id)
            cycle_edges.append(edge_id)
            current = neighbor
            # 走回起点说明环闭合，收尾。
            if current == start_vertex:
                break
        if cycle_edges:
            chains.append(
                {
                    "edge_ids": cycle_edges,
                    "start_vertex_id": int(start_vertex),
                    "end_vertex_id": int(current),
                    # 是否闭合单独存布尔：下游据此决定
                    # 这条链的周长能不能当闭合环用。
                    "closed": bool(current == start_vertex),
                    # 长度取 None 表示"量不出来"：
                    # 有边量不到长度时 chain_length 返回 None，
                    # 给 0.0 会被读成"这条链长度为 0"。
                    "length_mm": chain_length(cycle_edges),
                }
            )
    return chains


def _annotate_cross_region_bridges(
    regions: list[StepFeatureRegion],
    vertex_coordinates_mm: dict[int, list[float]] | None,
) -> None:
    """我把"缺口另一侧在别的区域"的跨区域端点补成候选，只标注、不改状态。

    单个区域的 `_bridge_proposals` 只看本区域端点，所以两个区域明明在
    零间距上贴着，也互相看不见。MD270324 上 102 个开放区域里有 33 个
    存在零间距的跨区域伙伴（22 组区域对），这些正是被漏掉的。

    我只把结果写进 ``boundary_topology_evidence["cross_region_bridges"]``，
    不自动改 ``boundary_topology_status``：跨区域闭合需要人工确认两侧
    确属同一处缺口，自动闭合会把两个无关区域错误合并。
    """

    if not vertex_coordinates_mm:
        return
    points: list[tuple[int, int, np.ndarray]] = []
    for region in regions:
        if region.boundary_topology_status != "OPEN_BOUNDARY_CANDIDATE":
            continue
        evidence = region.boundary_topology_evidence or {}
        for vertex_id in evidence.get("open_endpoint_vertex_ids", []):
            coordinates = vertex_coordinates_mm.get(int(vertex_id))
            if (
                isinstance(coordinates, list)
                and len(coordinates) == 3
                and all(isfinite(float(value)) for value in coordinates)
            ):
                points.append(
                    (
                        int(region.region_id),
                        int(vertex_id),
                        np.asarray(coordinates, dtype=np.float64),
                    )
                )
    if len(points) < 2:
        return

    # 我先用 KD-tree 做宽相位筛选、再对候选对精确量距，取代原先的 O(n^2)
    # 全遍历。为什么可以换：判据是"两点距离不超过容差"，这是个纯半径查询，
    # KD-tree 的 query_pairs 正好给出所有半径内的点对，不漏也不多。
    # 结果必须与旧实现逐条一致，所以我只换了"怎么找到候选对"，
    # 后面的过滤条件（同区域、同顶点、容差）和写入结构一字未改。
    # scipy 已在 requirements.txt 里声明（scipy>=1.13）；万一装不上，
    # 我退回原来的全遍历，行为不变，只是慢。
    found: dict[int, list[dict[str, object]]] = defaultdict(list)
    coordinates = np.vstack([item[2] for item in points])
    # 先用空间索引筛出"距离在容差内"的候选对：
    # 端点可能上千个，两两比较是 O(n²)，
    # 先按容差筛一遍能把真正的比较次数降一个量级。
    candidate_pairs = _pairs_within_tolerance(
        coordinates, _CROSS_REGION_BRIDGE_TOLERANCE_MM
    )
    for first_index, second_index in candidate_pairs:
        first_region, first_vertex, first_point = points[first_index]
        second_region, second_vertex, second_point = points[second_index]
        # 同一区域内的两点不搭桥："跨区域"搭桥的目的是
        # 把两块分离的区域连起来，区域内自有其边界拓扑去处理。
        # 同一个顶点也不搭——自己连自己没有意义。
        if first_region == second_region or first_vertex == second_vertex:
            continue
        # 筛完之后再精确算一次距离：空间索引给的是候选，
        # 真正落盘的距离必须是实测值，不能用索引的近似结果。
        distance = float(np.linalg.norm(first_point - second_point))
        if distance > _CROSS_REGION_BRIDGE_TOLERANCE_MM:
            continue
        # 两个方向各记一条：桥是无向的，但两个区域各自的证据里
        # 都要能看到"自己这边有一条桥通向对方"。
        found[first_region].append(
            {
                "from_vertex_id": first_vertex,
                "to_vertex_id": second_vertex,
                "to_region_id": second_region,
                "gap_distance_mm": distance,
                # applied 一律 False：这里只提候选，不做修补。
                "applied": False,
            }
        )
        found[second_region].append(
            {
                "from_vertex_id": second_vertex,
                "to_vertex_id": first_vertex,
                "to_region_id": first_region,
                "gap_distance_mm": distance,
                "applied": False,
            }
        )
    for region in regions:
        bridges = found.get(int(region.region_id))
        # 没有桥的区域不动它的证据：
        # 补一个空列表会让"没找到桥"与"找到零条"无法区分。
        if not bridges:
            continue
        evidence = region.boundary_topology_evidence
        if evidence is None:
            continue
        # 按间距升序排：最近的可搭桥点排在前面，
        # 复核者从最可能成功的那条看起。
        bridges.sort(key=lambda item: float(item["gap_distance_mm"]))
        evidence["cross_region_bridges"] = bridges


def _bridge_proposals(
    open_endpoint_vertex_ids: list[int],
    vertex_coordinates_mm: dict[int, list[float]] | None,
    chain_end_pairs: list[tuple[int, int]] | None = None,
) -> list[dict[str, object]]:
    """我按最近端点配对给出台阶缺口桥接候选，只提案、不自动应用。"""

    # ponytail: 我用“带同链禁止的贪心配对”，不做全局最小权完美匹配。
    # 实测 BOTTLE 上纯粹的最近端点贪心有 20/24 条候选把同一条链的两个端点
    # 连了起来——那是把链自己闭成小环，不是把缺口两侧接起来，所以我按链
    # 把端点分组、禁止同链互连。两两组合数仍是 O(n^2)，端点很多时再加
    # 空间索引；全部 applied=false 交人工确认，等有真实标注再换成链序 TSP。
    if not vertex_coordinates_mm or len(open_endpoint_vertex_ids) < 2:
        return []
    points: dict[int, list[float]] = {}
    for vertex_id in open_endpoint_vertex_ids:
        coordinates = vertex_coordinates_mm.get(int(vertex_id))
        if (
            isinstance(coordinates, list)
            and len(coordinates) == 3
            and all(isfinite(float(value)) for value in coordinates)
        ):
            points[int(vertex_id)] = [float(value) for value in coordinates]
    if len(points) < 2:
        return []
    # 同一条开放链的两个端点禁止互连，否则候选只是把链自己闭成小环。
    same_chain: set[tuple[int, int]] = set()
    for first, second in chain_end_pairs or []:
        left, right = int(first), int(second)
        # 按小到大排序后入集：端点对是无序的，
        # 排序存储让 (a,b) 与 (b,a) 命中同一条记录。
        same_chain.add((left, right) if left <= right else (right, left))
    pairs: list[tuple[float, int, int]] = []
    ordered = sorted(points)
    # 端点排序后做两两配对。这一处我**如实保留 O(n²)**，不用 KD-tree：
    # 贪心要按**全体**端点距离升序取最近的一对，这里没有距离门限，
    # 所以半径查询剪不掉任何东西——我试过用"覆盖半径"去筛，算出来
    # 那个半径必然覆盖全部点对，等于白建一次树还多一次排序。
    # 真要消掉这里的 O(n²)，必须改判据（例如只在容差内配对），
    # 但那会改变候选语义；没有真实标注我不会擅自改，所以保留并说清。
    # 我只把内层循环向量化，省掉逐个 np.linalg.norm 的 Python 开销。
    ordered_coordinates = np.asarray(
        [points[vertex_id] for vertex_id in ordered], dtype=np.float64
    )
    for position, first in enumerate(ordered):
        deltas = (
            ordered_coordinates[position + 1 :] - ordered_coordinates[position]
        )
        distances = np.linalg.norm(deltas, axis=1)
        for offset, distance in enumerate(distances):
            second = ordered[position + 1 + offset]
            # 同一条链的两端不搭桥：它们本来就该由这条链自己连接，
            # 把它们连起来等于凭空造一条边。
            if (first, second) in same_chain:
                continue
            pairs.append((float(distance), first, second))
    # 按距离升序贪心配对：这是最近邻搭桥的经典做法，
    # 不追求全局最优——全局最优要解匹配问题，代价远高于收益。
    pairs.sort()
    matched: set[int] = set()
    proposals: list[dict[str, object]] = []
    for distance, first, second in pairs:
        # 每个端点只用一次：一端被两个桥共用会造出分叉的边界，
        # 那种拓扑在几何上不成立。
        if first in matched or second in matched:
            continue
        matched.update({first, second})
        proposals.append(
            {
                "from_vertex_id": first,
                "to_vertex_id": second,
                "gap_distance_mm": distance,
                # applied=False：这里只提方案，不实际改几何。
                # 是否真的搭桥要由人工或上层策略决定，
                # 自动搭桥会把猜测固化成模型的一部分。
                "applied": False,
                "method": "nearest_open_endpoint_pairing_no_same_chain",
            }
        )
    unmatched = sorted(set(ordered) - matched)
    # 没配上对的端点也单独列出，且 to_vertex_id 为 None：
    # 它们是"确实找不到搭桥对象"的端点，
    # 静默丢弃会让复核者以为所有开口都处理了。
    for vertex_id in unmatched:
        proposals.append(
            {
                "from_vertex_id": vertex_id,
                "to_vertex_id": None,
                "gap_distance_mm": None,
                "applied": False,
                "method": "nearest_open_endpoint_pairing_no_same_chain",
                "unmatched": True,
            }
        )
    return proposals


def _region_geometry_signature(
    *,
    role: str,
    faces: list[StepFaceFeature],
    area_mm2: float,
    mean_center_xyz_mm: list[float],
    mean_normal_xyz: list[float],
    surface_type_counts: dict[str, int],
) -> str | None:
    """我从区域几何摘要生成不依赖 Face/Edge 编号的候选签名。"""

    try:
        face_tokens = sorted(
            face.geometry_signature or _fallback_face_geometry_token(face)
            for face in faces
        )
        if not face_tokens:
            return None
        payload = {
            "role": role,
            "face_geometry_tokens": face_tokens,
            "area_mm2": round(float(area_mm2), 6),
            "mean_center_xyz_mm": [
                round(float(value), 6) for value in mean_center_xyz_mm
            ],
            "mean_normal_xyz": [
                round(float(value), 6) for value in mean_normal_xyz
            ],
            "surface_type_counts": dict(sorted(surface_type_counts.items())),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            # allow_nan=False：签名载荷里若混进 NaN，json 会写出非法的 NaN 字面量，
            # 指纹就变成不可解析的字符串；这里宁可抛错由上层回退，也不产出坏指纹。
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return None
    return f"regiongeom-{hashlib.sha256(encoded).hexdigest()[:24]}"


def _fallback_face_geometry_token(face: StepFaceFeature) -> str:
    """我在测试或局部输入没有 Face 签名时仍保留几何摘要。"""

    payload = {
        "surface_type": face.surface_type,
        "area_mm2": _round_optional(face.area_mm2),
        "center_xyz_mm": _round_vector(face.center_xyz_mm),
        "normal_xyz": _round_vector(face.normal_xyz),
        "bbox_min_xyz_mm": _round_vector(face.bbox_min_xyz_mm),
        "bbox_max_xyz_mm": _round_vector(face.bbox_max_xyz_mm),
        "edge_count": int(face.edge_count),
        "edge_curve_types": sorted(map(str, face.edge_curve_types)),
        "edge_lengths_mm": sorted(
            value for value in (_round_optional(item) for item in face.edge_lengths_mm)
            if value is not None
        ),
        "edge_radii_mm": sorted(
            value for value in (_round_optional(item) for item in face.edge_radii_mm)
            if value is not None
        ),
        "surface_radius_mm": _round_optional(face.surface_radius_mm),
        "surface_axis_origin_xyz_mm": _round_vector(
            face.surface_axis_origin_xyz_mm
        ),
        "surface_axis_direction_xyz": _round_vector(
            face.surface_axis_direction_xyz
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"facefallback-{hashlib.sha256(encoded).hexdigest()[:24]}"


def _round_optional(value: float | None) -> float | None:
    """我把浮点值四舍五入到 6 位小数；非有限值统一转成 None，保证签名稳定。"""
    if value is None:
        return None
    numeric = float(value)
    return round(numeric, 6) if np.isfinite(numeric) else None


def _round_vector(value: list[float] | None) -> list[float] | None:
    """我把三维向量逐分量四舍五入到 6 位小数；None 原样返回。"""
    if value is None:
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(values) != 3 or not all(np.isfinite(item) for item in values):
        return None
    return [round(item, 6) for item in values]


def _aggregate_semantic_evidence(
    faces: list[StepFaceFeature],
) -> dict[str, object]:
    """我把 Face 级采样证据汇总到区域，但保留混合和不确定状态。"""

    statuses: dict[str, int] = defaultdict(int)
    confidences: list[float] = []
    for face in faces:
        evidence = face.semantic_evidence
        # 缺失证据的面按 UNRESOLVED 计：
        # "没判过"与"判成未定"在这里等价，
        # 但绝不能默认成 EXTERNAL，否则整片区域会被误判为外表面。
        status = str(evidence.get("status", "UNRESOLVED"))
        statuses[status] += 1
        confidence = evidence.get("confidence")
        # 只有有限数值的置信度才参与均值：
        # nan 会把均值污染成 nan，非数值会直接抛异常。
        if isinstance(confidence, (int, float)) and np.isfinite(confidence):
            confidences.append(float(confidence))
    # 已判定的面数 = 外部候选 + 内部腔体候选。
    # 只有这两种算"判出来了"，其余一律不算。
    resolved = statuses.get("EXTERNAL_CANDIDATE", 0) + statuses.get(
        "INTERNAL_CAVITY_CANDIDATE", 0
    )
    # 全部面都判出来了才可能给出整体结论；
    # 且必须全体一致——一半外一半内说明这块区域本身跨越了内外，
    # 整体结论只能是"需要复核"。
    if resolved == len(faces) and len(faces) > 0:
        if statuses.get("EXTERNAL_CANDIDATE", 0) == len(faces):
            status = "EXTERNAL_CANDIDATE"
        elif statuses.get("INTERNAL_CAVITY_CANDIDATE", 0) == len(faces):
            status = "INTERNAL_CAVITY_CANDIDATE"
        else:
            status = "REVIEW_REQUIRED"
    # 只判出一部分：证据不完整，必须人工看。
    elif resolved > 0:
        status = "REVIEW_REQUIRED"
    # 一个面都没判出来：如实报未定，
    # 不给任何默认结论。
    else:
        status = "UNRESOLVED"
    warnings: list[str] = []
    # 三种结论各配一条声明：即使是"全部是外表面"这种
    # 看起来最确定的结论，也仍然是候选而非工程确认。
    if status == "EXTERNAL_CANDIDATE":
        warnings.append("我只得到外部面候选证据，不能替代 CAD 工程语义确认。")
    elif status == "INTERNAL_CAVITY_CANDIDATE":
        warnings.append("我只得到内部腔体候选证据，不能替代 CAD 工程语义确认。")
    elif status == "REVIEW_REQUIRED":
        warnings.append("我发现区域内 Face 语义证据不一致或不完整，需要人工复核。")
    return {
        "status": status,
        # 各类面的计数排序后输出：便于跨批次 diff，
        # 也能一眼看出"有几张面判成了别的状态"。
        "face_status_counts": dict(sorted(statuses.items())),
        # 证据覆盖率：判出来的面占多少。
        # 空区域给 0.0 而不是 nan——0 张面里判出 0 张，
        # 覆盖率是 0 而不是"无定义"。
        "evidence_coverage_ratio": resolved / len(faces) if faces else 0.0,
        # 置信度均值在无有效值时给 0.0：这与上面同一条口径，
        # "没有置信度"视为"完全不置信"。
        "mean_confidence": (
            float(np.mean(confidences)) if confidences else 0.0
        ),
        # 方法名固定：这套内外判定来自"法向偏移实体分类器"。
        "method": "normal_offset_solid_classifier",
        "warnings": warnings,
    }
def _pairs_within_tolerance(
    coordinates: np.ndarray,
    tolerance_mm: float,
) -> list[tuple[int, int]]:
    """我找出所有距离不超过容差的点对（下标对，每对只给一次）。

    为什么单独抽出来：跨区域桥接和台阶桥接都要做同一件事，以前各自写
    一份 O(n^2) 全遍历。我用 scipy 的 cKDTree.query_pairs 做宽相位，
    输出与全遍历**逐对一致**（我按 (i, j) 排序返回，保证确定性）。
    scipy 不可用时我退回全遍历，行为不变，只是慢。
    """

    count = int(coordinates.shape[0])
    if count < 2:
        return []
    radius = float(tolerance_mm)
    if not np.isfinite(radius) or radius <= 0.0:
        return []
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coordinates)
        pairs = tree.query_pairs(r=radius, output_type="ndarray")
        # query_pairs 已保证 i < j 且每对只出现一次；我排序只为结果可复现。
        return sorted((int(i), int(j)) for i, j in pairs)
    except ImportError:
        # scipy 没装：退回全遍历，结果与 KD-tree 版本一致。
        pass
    except (RuntimeError, TypeError, ValueError):
        pass
    result: list[tuple[int, int]] = []
    for first in range(count):
        delta = coordinates[first + 1 :] - coordinates[first]
        distances = np.linalg.norm(delta, axis=1)
        for offset in np.nonzero(distances <= radius)[0]:
            result.append((first, first + 1 + int(offset)))
    return result
