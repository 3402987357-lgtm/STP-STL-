"""我方的可视化与对比分析区域（任务书核心任务四）。

我按甲方要求把三件事做成可复现的交付文件：
1. 破损区域高亮显示；
2. 修复前后对比（原模型 vs 修复模型，可附 STL 参考）；
3. 修补区域标记与分类显示。

我不引入新的三维渲染依赖。原理说明：我用 matplotlib 的 Agg 后端做
正交投影散点渲染——把每个三角面的重心投影到两轴平面，破损区域按包围盒
命中后换成分类颜色。这样在没有独立显卡、没有显示器的 Windows 机器上也能
稳定出图，而且输出是确定性的 PNG，便于甲方归档和比对。
"""

from __future__ import annotations

import base64
import html
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..geometry.mesh import TriangleMesh
from .atomic import atomic_path


# 我方固定分类配色，保证同一分类在不同批次出图颜色一致，便于人工比对。
_KNOWN_CATEGORY_COLORS: dict[str, str] = {
    "hole": "#d62728",
    "missing_face": "#ff7f0e",
    "boundary_break": "#9467bd",
    "discontinuity": "#8c564b",
    "topology_damage": "#e377c2",
    "gap": "#17becf",
    "open_boundary": "#bcbd22",
    "repaired": "#2ca02c",
    "review_required": "#7f7f7f",
}

# 我方的三个正交视图：俯视 XY、前视 XZ、侧视 YZ。
_VIEWS: tuple[tuple[str, int, int], ...] = (
    ("XY 俯视", 0, 1),
    ("XZ 前视", 0, 2),
    ("YZ 侧视", 1, 2),
)

# 我方用包围盒判定命中；这个容差吸收 STEP 面重心落在包围盒边界上的浮点误差。
_BBOX_TOLERANCE_MM = 1e-9

# 我方给散点渲染设一个上限，避免超大网格把出图时间拖爆。
_MAX_PLOT_POINTS = 400_000
# 图上最多标多少个误差数值；超过就只保留高亮（见 _draw_panel 里的说明）。
_MAX_ERROR_LABELS = 20


@dataclass(frozen=True)
class HighlightRegion:
    """我方要在模型上高亮的一个区域。

    精确高亮的原理说明：检测结果里，`missing` 区域保存的是完整参考模型上的
    三角面编号，`extra` 区域保存的是待检模型上的三角面编号。我用这两个编号
    直接命中三角面，比只按包围盒高亮精确得多；只有拿不到编号时，我才退回
    包围盒判定。
    """

    region_id: int
    category: str
    status: str = "DETECTED"
    area_mm2: float | None = None
    # 我为什么把误差一起带进可视化对象（2026-10-17）：
    # 甲方任务书里那张"示例图"画的就是缺口处一条红色测量线 + "2.7847 mm"
    # 数值标签。也就是说他们要看的不是"这里有个洞"，而是"这个洞差多少"。
    # 误差本来就在 regions.csv / result.json 里，但可视化层一直只读 area_mm2，
    # 于是图上只有颜色没有数字，和甲方给的示意对不上。
    max_error_mm: float | None = None
    review_required: bool = False
    center_xyz_mm: tuple[float, float, float] | None = None
    bbox_min_xyz_mm: tuple[float, float, float] | None = None
    bbox_max_xyz_mm: tuple[float, float, float] | None = None
    triangle_indices: tuple[int, ...] = ()
    triangle_space: str = "unknown"

    @property
    def has_geometry(self) -> bool:
        """我只要有精确三角面编号或完整包围盒，就能把它画到图上。"""

        return bool(self.triangle_indices) or (
            self.bbox_min_xyz_mm is not None
            and self.bbox_max_xyz_mm is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """我把区域序列化成报告用字典。

        triangle_space 必须一起输出：三角面编号只在特定网格空间
        （参考件或待检件）里有效。丢了它，下游拿着一堆编号
        无法判断该往哪个模型上画，只能靠猜。

        输出 triangle_count 而不是完整编号列表：一个区域可能有
        上万个三角面，全量写进报告会把 JSON 撑大好几倍，而对人工
        复核来说"这个区域覆盖了多少个面"已经足够定位问题。
        """
        return {
            "region_id": self.region_id,
            "category": self.category,
            "status": self.status,
            "area_mm2": self.area_mm2,
            "max_error_mm": self.max_error_mm,
            "review_required": self.review_required,
            "center_xyz_mm": (
                list(self.center_xyz_mm)
                if self.center_xyz_mm is not None
                else None
            ),
            "bbox_min_xyz_mm": (
                list(self.bbox_min_xyz_mm)
                if self.bbox_min_xyz_mm is not None
                else None
            ),
            "bbox_max_xyz_mm": (
                list(self.bbox_max_xyz_mm)
                if self.bbox_max_xyz_mm is not None
                else None
            ),
            "triangle_count": len(self.triangle_indices),
            "triangle_space": self.triangle_space,
        }


def category_color(category: str) -> str:
    """我按分类名取固定颜色；未知分类用 CRC32 生成稳定颜色。

    已知分类用人工选定的固定色，保证甲方在不同报告里看到"成型零件"
    永远是同一个颜色，不用每次对照图例。

    未知分类为什么用 CRC32 而不是随机色：颜色必须**对同一分类名稳定**。
    随机色会让同一个分类在两张图里变成两种颜色，图就不具备对照价值。
    CRC32 是确定性的，同一名字永远得到同一颜色；我取低 24 位当 RGB，
    虽然不保证色觉可分辨，但配合图例已经足够区分，且零依赖。
    """

    known = _KNOWN_CATEGORY_COLORS.get(category)
    if known is not None:
        return known
    digest = zlib.crc32(category.encode("utf-8")) & 0xFFFFFF
    return f"#{digest:06x}"


def _finite_triplet(values: Any, name: str) -> tuple[float, float, float]:
    """我把三元组坐标收敛成三个有限浮点数，否则直接报错。

    我为什么先 ``reshape(-1)`` 再看形状：上游 JSON 里的坐标有时是
    ``[[x, y, z]]`` 这种嵌套一层的形式。直接判 ``shape != (3,)``
    会把形状正确、只是多包了一层的合法数据拒掉；先展平再判则两种
    写法都能接受，而展平后长度不是 3 才真的数据有问题。
    """

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"我要求 {name} 是三个有限数值。")
    return (float(array[0]), float(array[1]), float(array[2]))


def _optional_triplet(values: Any) -> tuple[float, float, float] | None:
    """我把可缺失的三元组坐标解析成三元组或 None。"""

    if values is None:
        return None
    try:
        return _finite_triplet(values, "坐标")
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    """我把可缺失的数值解析成有限浮点数或 None。

    我为什么连 ``bool`` 都排除：``float(True)`` 是 1.0，而上游 JSON
    里布尔字段写成数值的情况偶有发生。不排除就会把"是否复核"的
    True 当成 1.0 毫米画到图上，出现一个位置莫名其妙的高亮点，
    而且查起来极费劲。
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def highlight_regions_from_inspection(
    payload: Mapping[str, Any],
) -> list[HighlightRegion]:
    """我从 missing-region-detection JSON 读出要高亮的缺失区域。

    我按缺陷类型决定精确编号落在哪个模型上：
    - `missing`：编号属于完整参考模型，画在“修复后/参考”那一侧；
    - `extra`：编号属于待检模型，画在“修复前/待检”那一侧。
    """

    regions: list[HighlightRegion] = []
    for record in payload.get("regions") or ():
        review_required = bool(record.get("review_required"))
        category = str(record.get("defect_type") or "unknown")
        # 缺失与多出取的面号来自不同的模型：
        # 缺失要标在参考模型上（那里才有这块料），
        # 多出要标在候选模型上（多出来的料只存在于候选上）。
        # space 字段决定这批面号去哪个模型上找，
        # 取错的话高亮会落在错误的模型甚至查不到面。
        if category == "missing":
            indices = tuple(
                int(value) for value in (record.get("reference_face_ids") or ())
            )
            space = "reference"
        elif category == "extra":
            indices = tuple(
                int(value) for value in (record.get("candidate_face_ids") or ())
            )
            space = "candidate"
        # 未知类型留空索引而不是默认去参考模型上找：
        # 猜一个模型会让高亮出现在没有依据的位置，
        # 空索引至少是"没画"而不是"画错"。
        else:
            indices = ()
            space = "unknown"
        regions.append(
            HighlightRegion(
                region_id=int(record["region_id"]),
                category=category,
                # 状态由 review_required 推出：
                # 报告里没有独立的状态字段，
                # "要不要复核"就是这里唯一能区分的维度。
                status="REVIEW_REQUIRED" if review_required else "DETECTED",
                area_mm2=_optional_float(record.get("area_mm2")),
                max_error_mm=_optional_float(record.get("max_error_mm")),
                review_required=review_required,
                # 中心与包围盒都过一遍可选解析：
                # 报告里的这些字段可能缺失或格式不对，
                # 统一走 _optional_* 能拿到 None 而不是抛异常。
                center_xyz_mm=_optional_triplet(record.get("center_xyz_mm")),
                bbox_min_xyz_mm=_optional_triplet(
                    record.get("bbox_min_xyz_mm")
                ),
                bbox_max_xyz_mm=_optional_triplet(
                    record.get("bbox_max_xyz_mm")
                ),
                triangle_indices=indices,
                triangle_space=space,
            )
        )
    return regions


def face_bboxes_from_step_face_features(
    payload: Mapping[str, Any],
) -> dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """我从 step-face-features JSON 建 Face 编号到包围盒的映射。"""

    mapping: dict[
        int, tuple[tuple[float, float, float], tuple[float, float, float]]
    ] = {}
    for face in payload.get("faces") or ():
        low = _optional_triplet(face.get("bbox_min_xyz_mm"))
        high = _optional_triplet(face.get("bbox_max_xyz_mm"))
        if low is None or high is None:
            continue
        mapping[int(face["face_id"])] = (low, high)
    return mapping


def highlight_regions_from_step_repair(
    payload: Mapping[str, Any],
    face_bboxes: Mapping[
        int, tuple[tuple[float, float, float], tuple[float, float, float]]
    ]
    | None = None,
) -> list[HighlightRegion]:
    """我从 step-repair JSON 读出补面区域，并按来源 Face 还原包围盒。"""

    regions: list[HighlightRegion] = []
    for record in payload.get("repaired_regions") or ():
        bbox_min: tuple[float, float, float] | None = None
        bbox_max: tuple[float, float, float] | None = None
        # 修补报告里只有面号没有坐标，要靠面特征文件
        # 把面号映射回空间位置；没有那份文件就只能画不定位的区域。
        if face_bboxes:
            lows: list[tuple[float, float, float]] = []
            highs: list[tuple[float, float, float]] = []
            for face_id in record.get("source_face_ids") or ():
                pair = face_bboxes.get(int(face_id))
                # 查不到的面号跳过：区域引用的面可能不在特征文件里，
                # 用一个面都对不上就放弃定位，也不能用默认值编一个位置。
                if pair is not None:
                    lows.append(pair[0])
                    highs.append(pair[1])
            # 区域包围盒取所有源面包围盒的外包：
            # 一块修补区域通常由多个面构成，
            # 任何一个面的包围盒都不足以代表整块区域。
            if lows and highs:
                low_array = np.asarray(lows, dtype=np.float64)
                high_array = np.asarray(highs, dtype=np.float64)
                bbox_min = tuple(low_array.min(axis=0).tolist())  # type: ignore[assignment]
                bbox_max = tuple(high_array.max(axis=0).tolist())  # type: ignore[assignment]
        # 标注点取包围盒中心而不是质心：中心一定落在包围盒内，
        # 凹形区域的质心可能跑到区域外面，标出去会指错位置。
        center = (
            tuple(
                (np.asarray(bbox_min) + np.asarray(bbox_max)) / 2.0
            )
            if bbox_min is not None and bbox_max is not None
            else None
        )
        review_required = bool(record.get("review_required"))
        # 状态缺失时给 "UNKNOWN" 而不是空串：
        # 空串在图上什么都看不见，看的人无法区分
        # "没有状态"和"状态是空"。
        status = str(record.get("status") or "UNKNOWN")
        category = str(record.get("damage_type") or "hole")
        # 已修复的区域统一着"repaired"色：这一类的共同点是
        # "已经处理过"，按 damage_type 细分反而看不出哪些处理过了。
        if status.upper().startswith("REPAIRED"):
            category = "repaired"
        regions.append(
            HighlightRegion(
                region_id=int(record["region_id"]),
                category=category,
                status=status,
                area_mm2=_optional_float(record.get("area_repaired")),
                review_required=review_required,
                center_xyz_mm=(
                    (float(center[0]), float(center[1]), float(center[2]))
                    if center is not None
                    else None
                ),
                bbox_min_xyz_mm=bbox_min,
                bbox_max_xyz_mm=bbox_max,
            )
        )
    return regions


def _plot_points(mesh: TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    """我取三角面重心作为渲染采样点，并按上限做等距抽样。

    我同时返回被保留的三角面编号，保证抽样后仍能和精确高亮编号对齐。

    为什么用等距抽样而不是取前 N 个：三角面在网格里通常按空间局部性
    排列（一个面上的三角形连续存放），截断会把采样全压在模型的一角，
    画出来的图看起来像"只有一块区域有点"。等距抽样能在整件上均匀铺开，
    视觉上才反映真实形貌。

    为什么必须把保留下来的编号一并返回：高亮是按三角面编号着色的，
    如果抽样后丢掉了编号映射，就无法把区域信息对齐到被画出来的点上，
    图上的颜色就会张冠李戴。
    """

    centroids = mesh.triangle_centroids()
    if len(centroids) <= _MAX_PLOT_POINTS:
        return centroids, np.arange(len(centroids), dtype=np.int64)
    step = int(np.ceil(len(centroids) / _MAX_PLOT_POINTS))
    indices = np.arange(0, len(centroids), step, dtype=np.int64)
    return centroids[indices], indices


def _triangle_codes(
    mesh: TriangleMesh,
    regions: Sequence[HighlightRegion],
    space: str,
) -> np.ndarray:
    """我给每个三角面分配它命中的第一个区域编号，0 表示完整区域。

    优先用检测结果里的精确三角面编号；没有编号时退回包围盒判定。
    `space` 说明当前渲染的是 `candidate`（待检/修复前）还是
    `reference`（参考/修复后）模型。

    两级判定的理由：检测结果里带的是**精确三角面编号**，是权威口径；
    只有当区域没有几何信息（例如只有 STEP 面摘要、没落到网格面上）时
    才退回包围盒判定。包围盒判定会把盒内的完整面也算进去，是近似，
    所以只能当兜底，不能反过来覆盖精确编号。

    `space` 必须显式匹配：修复前后是两个不同的网格，三角面编号体系
    完全不同。把 reference 的编号打到 candidate 的图上会整体错位，
    所以我只在 region.triangle_space == space 时才使用精确编号。

    重叠处理：我只给尚未被占用的面着色（codes[valid] == 0），
    所以区域重叠时先到先得，后一个区域不会被前一个盖掉，
    也不会出现一个面同时属于两个区域导致图例对不上。
    """

    codes = np.zeros(len(mesh.faces), dtype=np.int64)
    centroids: np.ndarray | None = None
    for index, region in enumerate(regions, start=1):
        if region.triangle_indices and region.triangle_space == space:
            identifiers = np.fromiter(
                region.triangle_indices,
                dtype=np.int64,
                count=len(region.triangle_indices),
            )
            valid = identifiers[
                (identifiers >= 0) & (identifiers < len(codes))
            ]
            free = valid[codes[valid] == 0]
            codes[free] = index
            continue
        if not region.has_geometry or region.bbox_min_xyz_mm is None:
            continue
        if centroids is None:
            centroids = mesh.triangle_centroids()
        low = np.asarray(region.bbox_min_xyz_mm, dtype=np.float64)
        high = np.asarray(region.bbox_max_xyz_mm, dtype=np.float64)
        inside = np.all(
            (centroids >= low - _BBOX_TOLERANCE_MM)
            & (centroids <= high + _BBOX_TOLERANCE_MM),
            axis=1,
        )
        codes[inside & (codes == 0)] = index
    return codes


def _draw_panel(
    axis: Any,
    points: np.ndarray,
    codes: np.ndarray,
    regions: Sequence[HighlightRegion],
    *,
    horizontal: int,
    vertical: int,
    title: str,
    legend: dict[str, Any],
    space: str,
) -> None:
    """我把一个正交视图画到给定子图上，并按分类着色。"""

    axis.scatter(
        points[:, horizontal],
        points[:, vertical],
        s=1.0,
        c="#c9c9c9",
        linewidths=0,
        rasterized=True,
    )
    for category in sorted({region.category for region in regions}):
        indices = [
            index
            for index, region in enumerate(regions, start=1)
            if region.category == category
        ]
        mask = np.isin(codes, indices)
        if not mask.any():
            continue
        handle = axis.scatter(
            points[mask, horizontal],
            points[mask, vertical],
            s=5.0,
            c=category_color(category),
            linewidths=0,
            rasterized=True,
        )
        legend.setdefault(category, handle)
    # 我为什么给数值标注设上限：真实噪声件在低阈值下能出几千个区域，
    # 逐区域标数字会把整张图糊成一片，比不标更难读。
    # 区域数在上限内才标（正常验收件通常就几个到十几个），
    # 超过上限就只保留高亮，数值请到 HTML 明细表/CSV 里看。
    if len(regions) <= _MAX_ERROR_LABELS:
        for index, region in enumerate(regions, start=1):
            center = region.center_xyz_mm
            # 只标本行这个网格空间里的区域：
            # missing 区域属于参考件、extra 属于待检件，
            # 不过滤的话同一个数字会在"修复前/修复后"两行各出现一次，
            # 看图的人会以为两侧都有这块缺陷。
            if region.triangle_space != space:
                continue
            # 没有中心或没有误差的就不标：标一个 0.000 或空值会误导。
            if center is None or region.max_error_mm is None:
                continue
            axis.annotate(
                f"#{index} {region.max_error_mm:.3f}mm",
                xy=(center[horizontal], center[vertical]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
                color=category_color(region.category),
                # 加半透明白底：数值常常压在深色模型上，
                # 没有底色时字和填充糊在一起，等于没标。
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )
    axis.set_title(title, fontsize=9)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _configure_chinese_font(matplotlib_module: Any) -> None:
    """我优先使用 Windows 自带中文字体，避免标题出现方框。"""

    matplotlib_module.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib_module.rcParams["axes.unicode_minus"] = False


def _render_figure(
    before: TriangleMesh,
    after: TriangleMesh | None,
    regions: Sequence[HighlightRegion],
    *,
    title: str,
    subtitle: str | None = None,
):
    """我构造“修复前 / 修复后”对比图，返回 matplotlib Figure。"""

    import matplotlib

    matplotlib.use("Agg")
    _configure_chinese_font(matplotlib)
    from matplotlib import pyplot as plt

    # 修复前一侧是待检模型（candidate），修复后一侧是参考/修复模型（reference）。
    rows = [(f"修复前：{before.name or '输入模型'}", before, "candidate")]
    if after is not None:
        # 三元组第三项是"这份网格在判据里扮演哪个角色"：
        # 修复前是 candidate、修复后是 reference，
        # 区域高亮要按角色去找对应的面号，不能混用。
        rows.append((f"修复后：{after.name or '修复模型'}", after, "reference"))

    # 每行一个模型、每列一个视角：
    # 同一视角上下对照才能看出"修好了没有"，
    # 这是对比图唯一有意义的排布方式。
    figure, axes = plt.subplots(
        len(rows),
        len(_VIEWS),
        # 尺寸随行列数增长：固定尺寸在只有一行时会留大片空白，
        # 在两行时又会把面板压扁。
        figsize=(4.0 * len(_VIEWS), 3.6 * len(rows)),
        # squeeze=False：单行单列时 matplotlib 会返回非数组对象，
        # 关掉挤压才能保证 axes[row][col] 的取法始终成立。
        squeeze=False,
    )
    legend: dict[str, Any] = {}
    for row_index, (row_title, mesh, space) in enumerate(rows):
        # 三角形着色码按面算一次，再按绘图用的顶点索引展开：
        # 绘图用的是去重组后的点面结构，
        # 必须先算到三角形粒度才能正确取子集。
        all_codes = _triangle_codes(mesh, regions, space)
        points, indices = _plot_points(mesh)
        codes = all_codes[indices]
        for column_index, (view_name, horizontal, vertical) in enumerate(_VIEWS):
            _draw_panel(
                axes[row_index][column_index],
                points,
                codes,
                # legend 用同一个字典在多个面板间传递：
                # 各面板可能只出现部分类别，共享字典才能保证
                # "同一类别在全部面板里颜色一致"。
                regions,
                horizontal=horizontal,
                vertical=vertical,
                title=f"{row_title} · {view_name}",
                legend=legend,
                space=space,
            )
    if legend:
        # 图例放整幅图底部而不是每个面板各放一份：
        # 每格都放会把面板挤小，而所有面板共用同一套配色。
        figure.legend(
            legend.values(),
            legend.keys(),
            loc="lower center",
            # 最多 6 列：类别多时分列换行，
            # 不限制会拉成一条超宽的长条。
            ncol=min(len(legend), 6),
            fontsize=8,
            # 无边框：图例是辅助信息，加框会让它看起来像一个独立控件。
            frameon=False,
        )
    figure.suptitle(title, fontsize=12, y=0.985)
    if subtitle:
        # 副标题用 text 而不是第二个 suptitle：
        # 两个 suptitle 会重叠，用固定 y 坐标的 text 才能稳定排版。
        figure.text(0.5, 0.945, subtitle, ha="center", fontsize=9, color="#444")
    # 底部留白随有无图例变化：有图例时留出空间，
    # 否则图例会压住最后一行面板。
    figure.tight_layout(rect=(0.0, 0.04 if legend else 0.0, 1.0, 0.93))
    return figure, plt


def _render_category_figure(
    mesh: TriangleMesh,
    regions: Sequence[HighlightRegion],
    *,
    title: str,
    space: str = "candidate",
):
    """我按分类各画一张图，做“修补区域标记与分类显示”。"""

    import matplotlib

    matplotlib.use("Agg")
    _configure_chinese_font(matplotlib)
    from matplotlib import pyplot as plt

    categories = sorted({region.category for region in regions}) or ["none"]
    columns = min(len(categories), 3)
    rows = int(np.ceil(len(categories) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.4 * rows),
        squeeze=False,
    )
    points, indices = _plot_points(mesh)
    for index, category in enumerate(categories):
        axis = axes[index // columns][index % columns]
        subset = [
            region for region in regions if region.category == category
        ]
        codes = _triangle_codes(mesh, subset, space)[indices]
        _draw_panel(
            axis,
            points,
            codes,
            subset,
            horizontal=0,
            vertical=1,
            title=f"{category}（{len(subset)} 个区域）",
            legend={},
            space=space,
        )
    for index in range(len(categories), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return figure, plt


def _save_png(figure: Any, plt: Any, path: Path, dpi: int) -> None:
    """我用原子写入保存 PNG，避免中断留下半个图片。

    我显式传 format="png"：原子写入的临时文件名以 .tmp 结尾，
    matplotlib 无法从扩展名推断格式，必须由我指定。
    """

    with atomic_path(path) as temporary:
        figure.savefig(
            temporary,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)


def _counts_by_category(
    regions: Iterable[HighlightRegion],
) -> dict[str, int]:
    """我统计每个分类的区域数量，写入清单和 HTML。

    我为什么返回时排了序：这个字典要写进 HTML 表格和图例。不排序的话
    顺序取决于遍历次序，同一份数据两次渲染可能给出不同行序，
    两份报告无法逐行对照。按分类名排序后行序完全确定。
    """

    counts: dict[str, int] = {}
    for region in regions:
        counts[region.category] = counts.get(region.category, 0) + 1
    return dict(sorted(counts.items()))


def _render_html(
    title: str,
    images: Sequence[tuple[str, Path]],
    regions: Sequence[HighlightRegion],
    counts: Mapping[str, int],
) -> str:
    """我拼出自包含 HTML：PNG 以 base64 内嵌，不依赖外部文件或网络。"""

    blocks = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        # 标题与分类名都要 escape：它们来自模型或配置，
        # 可能含 < 或引号，不转义会破坏 HTML 结构。
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;"
        "margin:24px;color:#222}h1{font-size:20px}h2{font-size:15px}"
        "img{max-width:100%;border:1px solid #ddd;border-radius:4px}"
        "table{border-collapse:collapse;font-size:13px}"
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}"
        ".sw{display:inline-block;width:12px;height:12px;margin-right:6px;"
        "border-radius:2px;vertical-align:middle}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    blocks.append("<h2>分类统计</h2><ul>")
    for category, count in counts.items():
        # 色块与分类图例同色：图例的颜色必须和图上区域的颜色
        # 来自同一个 category_color，否则读者会对不上号。
        color = category_color(category)
        blocks.append(
            f'<li><span class="sw" style="background:{color}"></span>'
            f"{html.escape(category)}：{count} 个区域</li>"
        )
    blocks.append("</ul>")
    for caption, path in images:
        # 图片内联成 base64：这份 HTML 要能单文件分发，
        # 用外链 <img src="xxx.png"> 在单独拷走 HTML 时会全部裂图。
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        blocks.append(f"<h2>{html.escape(caption)}</h2>")
        blocks.append(
            f'<img alt="{html.escape(caption)}" src="data:image/png;base64,{payload}">'
        )
    blocks.append("<h2>区域明细</h2>")
    blocks.append(
        # 表头固定五列：与 CSV 的筛选视角互补，
        # HTML 只放"定位一个区域"所需的最少字段，
        # 完整字段在 CSV 里，避免表格宽到没法看。
        "<table><tr><th>区域</th><th>分类</th><th>状态</th>"
        "<th>面积 mm2</th><th>需要复核</th></tr>"
    )
    for region in regions:
        # 面积为空时输出空串而不是 "None" 或 "nan"：
        # 表格里出现这两种字符串会被读成真的有这个值。
        area = "" if region.area_mm2 is None else f"{region.area_mm2:.3f}"
        blocks.append(
            "<tr>"
            f"<td>{region.region_id}</td>"
            f"<td>{html.escape(region.category)}</td>"
            f"<td>{html.escape(region.status)}</td>"
            f"<td>{area}</td>"
            f"<td>{'是' if region.review_required else '否'}</td>"
            "</tr>"
        )
    blocks.append("</table></body></html>")
    return "\n".join(blocks) + "\n"


def write_comparison_report(
    before: TriangleMesh,
    after: TriangleMesh | None,
    regions: Sequence[HighlightRegion],
    output_dir: str | Path,
    *,
    title: str = "我方破损区域检测与修补对比",
    subtitle: str | None = None,
    dpi: int = 120,
) -> dict[str, Path]:
    """我写出修复前后对比图、分类图和自包含 HTML 报告。

    分类图我按区域所在的模型分开渲染：待检模型上的 `extra` 区域画在
    “修复前”网格上，参考模型上的 `missing` 区域画在“修复后”网格上，
    避免把两侧的区域混在一张图上造成误读。
    """

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    counts = _counts_by_category(regions)

    comparison_path = target / "comparison_before_after.png"
    # 三张图各有分工，缺一不可：
    #   对比图给"改了哪里"的直观印象；
    #   候选图按待检模型的三角面空间着色，看漏检/误判分布；
    #   参考图用参考模型自身空间着色，看参考侧的分类是否合理。
    figure, plt = _render_figure(
        before,
        after,
        regions,
        title=title,
        subtitle=subtitle,
    )
    _save_png(figure, plt, comparison_path, dpi)

    # "unknown" 归入候选图一并展示：它表示空间归属未定，
    # 和确定的 candidate 一样需要人工关注，不该被静默略过。
    candidate_regions = [
        region
        for region in regions
        if region.triangle_space in {"candidate", "unknown"}
    ]
    reference_regions = [
        region for region in regions if region.triangle_space == "reference"
    ]
    candidate_path = target / "region_classification_candidate.png"
    figure, plt = _render_category_figure(
        before,
        candidate_regions,
        title=f"{title} · 待检模型分类标记",
        space="candidate",
    )
    _save_png(figure, plt, candidate_path, dpi)

    # 只有存在参考模型且有参考区域时才画参考图：
    # 没有参考模型的单模型检查场景下，这张图会是一张空图，
    # 画出来反而让人误以为"参考侧无问题"。
    classification_path = candidate_path
    if after is not None and reference_regions:
        classification_path = target / "region_classification_reference.png"
        figure, plt = _render_category_figure(
            after,
            reference_regions,
            title=f"{title} · 参考模型分类标记",
            space="reference",
        )
        _save_png(figure, plt, classification_path, dpi)

    images: list[tuple[str, Path]] = [("修复前后对比", comparison_path)]
    images.append(("待检模型分类标记", candidate_path))
    # 只有真的画了参考图才把它列进清单：
    # 清单里列一个不存在的文件会让 HTML 出现裂图，
    # 而"没画"这件事本身已经由上面的条件判断表达清楚了。
    if classification_path != candidate_path:
        images.append(("参考模型分类标记", classification_path))
    html_path = target / "visualization.html"
    with atomic_path(html_path) as temporary:
        temporary.write_text(
            _render_html(title, images, regions, counts),
            encoding="utf-8",
        )

    manifest_path = target / "visualization.json"
    manifest = {
        "schema_version": "visualization-1.0",
        "status": "COMPLETE",
        "title": title,
        "subtitle": subtitle,
        # 视角清单写进清单：HTML 里的图是按固定视角序列渲染的，
        # 记下序列才能让程序化消费方知道每张图对应哪个视角。
        "views": [name for name, _, _ in _VIEWS],
        "before_triangle_count": int(len(before.faces)),
        # 参考模型的三角面数允许 None：单模型检查时没有"after"，
        # 用 0 会被误读成"参考模型是空的"。
        "after_triangle_count": (
            int(len(after.faces)) if after is not None else None
        ),
        "region_count": len(regions),
        "regions_by_category": counts,
        # 有几何的区域单独计数：有些区域只有分类、没有可画的三角面，
        # 这个差值是"报告里列了但图上没有"的数量，必须可见。
        "regions_with_geometry": sum(
            1 for region in regions if region.has_geometry
        ),
        "artifacts": {
            "comparison_png": str(comparison_path),
            "classification_png": str(classification_path),
            "classification_candidate_png": str(candidate_path),
            "html": str(html_path),
        },
        "regions": [region.to_dict() for region in regions],
    }
    with atomic_path(manifest_path) as temporary:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    return {
        "comparison_png": comparison_path,
        "classification_png": classification_path,
        "classification_candidate_png": candidate_path,
        "html": html_path,
        "manifest": manifest_path,
    }
