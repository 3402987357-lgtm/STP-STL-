"""我提供面向甲方和工程师的命令行入口。

我这条命令行一共暴露 13 个子命令，按用途分成四组：

一、几何比对与缺失区域（主流程）
    inspect                 候选模型 vs 参考模型，输出缺失区域 JSON/CSV/PLY。

二、网格与 STEP 的修补
    repair                  对 STL 里满足安全条件的简单孔洞做局部补面。
    step-repair             对 STEP 开放边界里的安全平面孔洞做局部补面。

三、STEP 结构解析（BRep 只读分析）
    step-report             Face/Edge/Wire/Shell/Solid 拓扑统计。
    step-face-features      每个 Face 的曲面类型、面积、法向与包围盒。
    step-feature-classify   上下平面、侧壁、倾斜面与 Face 邻接候选分类。
    step-engineering-features 台阶、圆柱/孔、R 角与拔模方向的工程证据。

四、批量与质量工具
    validate                只检查 STL 网格质量，不做缺失区域判定。
    probe                   探测 STEP/STP 等格式的元数据，不执行三角化。
    audit                   审计输出目录里 JSON/CSV/XLSX/PLY/STL 的完整性。
    validate-annotations    校验真实标注清单并跑训练前数据泄漏门禁。
    score-annotations       用**人工确认过**的真实标注给预测打分，输出面级 precision/recall。
    review-annotations      把标注草稿导成人工可填的复核表，或把填好的表并回确认清单。
    evaluate-synthetic      用合成缺陷标签评估基线，可选 missing（删面）或 extra（外推多料）。
    parting                 从 STL 提取水平截面分型线与诊断用平面分型面。
    batch-inspect           批量检测多对模型，按文件名主干配对并出汇总台账。
    serve                   启动缺失区域检测的 HTTP 服务（提交任务 + 查结果）。
    train-face-model        用合成真值训练面级缺失分类器（需要 torch）。
    gui                     启动桌面窗口来选件、跑检测、看结论。
    tessellate-step        用 OCCT 三角化 STEP 并给出三角形到 CAD 面 ID 的映射。

退出码约定（我方统一）：
    0 = 成功，或者该命令自己定义的"通过"状态；
    1 = 业务失败，例如结果状态是 REVIEW_REQUIRED、BASELINE 之外的状态；
    2 = 我执行时报错（输入不存在、参数非法、文件损坏等），错误写到 stderr。

我在这里只做参数解析与分发，几何算法全部放在各自的领域模块里实现，
这样任务书增删子命令时不会牵动核心算法，核心算法也不会反过来依赖命令行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import load_config
from .models import InspectionConfig
from .domain.repair import RepairConfig
from .errors import InspectorError
# 几何层：STL 的读写与质量报告。
from .geometry import load_stl, mesh_quality_report, save_stl
# 批处理与产物审计：前者负责一次跑一批输入，
# 后者负责检查输出目录里有没有该有的文件。
from .io.batch_probe import expand_inputs, run_batch, write_batch_reports
from .io.artifact_audit import audit_artifacts
# STEP 侧的三层检测：特征分类、工程特征、拓扑。
# 三者由粗到细，前者是后者的输入。
from .detection.step_features import inspect_step_feature_classification
from .detection.step_engineering import inspect_step_engineering_features
from .io.step_topology import inspect_step_topology
from .io.step_faces import inspect_step_face_features
# 比对主流程：拿候选去比参考。
from .pipeline import inspect_pair
# 输入侧的探针与校验：先探格式，再校验路径可用。
from .io.step_probe import probe_any
from .io.input_validator import validate_input_path
# 学习侧：标注清单与合成集评测。
from .learning.annotations import load_annotation_manifest
from .learning.annotation_scoring import (
    collect_predictions,
    score_annotations,
    score_annotations_multi,
)
from .learning.annotation_review import apply_review_sheet, export_review_sheet
from .learning.evaluation import evaluate_synthetic_suite
from .bom_cli import main as bom_main
from .dataset_cli import main as dataset_main
from .feature_cli import main as feature_main
from .parting_cli import main as parting_baseline_main
from .visualization_cli import main as visualization_main
# 各类报告的写出：每个报告模块对应一种产物。
from .output.repair_report import write_repair_csv, write_repair_json
from .output.atomic import atomic_path
from .output.step_report import write_step_topology_json
from .output.step_face_report import write_step_face_features_json
from .output.step_feature_report import write_step_feature_classification_json
from .output.step_engineering_report import write_step_engineering_report
from .output.step_repair_report import (
    write_step_repair_csv,
    write_step_repair_json,
)
from .repair import repair_mesh_holes
from .repair import repair_step_holes
from .detection import extract_parting_line, build_planar_parting_surface
from .domain.parting import PartingConfig, PartingPipelineResult
from .output import write_parting_result


def _configure_utf8_stdio() -> None:
    """我把 Windows 控制台输出切到 UTF-8，避免中文帮助乱码。"""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def build_parser() -> argparse.ArgumentParser:
    """我构造整条命令行的参数树。

    我刻意把每个子命令的默认值都设成 None（或该命令自己的默认），
    目的是让"用户没写这个参数"和"用户显式写了默认值"能被区分开：
    main() 里只有参数不为 None 时才覆盖配置文件，否则配置文件的
    取值会被命令行默认值悄悄顶掉。
    """

    parser = argparse.ArgumentParser(
        prog="cad-inspector",
        description="我用 STEP/STP 与完整 STL 的几何差异识别缺失区域。",
    )
    # required=True 让"什么都不写"直接报用法错误，而不是掉进某个默认分支。
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # inspect：主流程。候选模型（缺料）对参考模型（完整件）做几何比对，
    # 输出缺失区域。它参数最多，因为阈值、配准、距离后端、单位策略、
    # 输出开关都需要暴露给甲方调参。
    # ------------------------------------------------------------------
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="我执行一次候选模型与参考模型的几何比对。",
    )
    # 位置参数：候选在前、参考在后，顺序固定，避免两个模型被写反。
    inspect_parser.add_argument("candidate", help="我需要检测的 .stp/.step/.stl 文件。")
    inspect_parser.add_argument(
        "reference",
        help="我使用的完整参考 .stl/.stp/.step 文件。",
    )
    inspect_parser.add_argument(
        "--output-dir",
        default="outputs/run",
        help="我保存 JSON、CSV 和 PLY 的目录。",
    )
    inspect_parser.add_argument("--config", default=None, help="我读取的 JSON 配置文件。")
    # 阈值与区域筛选：这三个直接决定"多小的差异还算差异"和"多小的区域不报"。
    inspect_parser.add_argument("--threshold-mm", type=float, default=None)
    inspect_parser.add_argument("--min-region-area-mm2", type=float, default=None)
    inspect_parser.add_argument("--min-triangle-count", type=int, default=None)
    # 资源保护：超过这个字节数的输入我直接拒绝，避免一个巨型文件把内存吃满。
    inspect_parser.add_argument(
        "--max-input-file-size-bytes",
        type=int,
        default=None,
        help="我拒绝读取超过该大小的候选或参考文件。",
    )
    # 有向证据：用源面法向和目标面法向的一致性，区分"真的缺料"和"只是挪了位置"。
    inspect_parser.add_argument(
        "--direction-normal-tolerance-deg",
        type=float,
        default=None,
        help="我判断源面与目标面法向一致性的容差。",
    )
    inspect_parser.add_argument(
        "--no-directional-evidence",
        action="store_true",
        help="我关闭缺失区域的有向证据，只保留距离和双向区域结果。",
    )
    # 配准：none 不配准；centroid 只对质心；pca 对主轴；icp 做迭代最近点。
    #
    # 我为什么把这四档都留着而不是只留 icp：实测 centroid 与 pca 在
    # "候选件本来就和参考件同一坐标系"的数据上比 icp 更稳——icp 会在
    # 大面积缺失的区域上把剩余点云"拉过去"对齐，反而制造出系统性偏差。
    # 而 icp 在候选件被整体旋转/平移过的数据上是唯一能收敛的档。
    # 我让用户在命令行显式选，不替他猜。
    inspect_parser.add_argument(
        "--registration-mode",
        choices=["auto", "none", "centroid", "pca", "icp"],
        default=None,
    )
    # allow-scale 默认关：缩放会让"缺失区域面积"这个量纲失真——
    # 我实测打开后同一份数据的面积会整体乘一个系数，而甲方核对面积时
    # 是按毫米绝对值看的。所以即便打开，非 1 缩放也会进入复核。
    inspect_parser.add_argument(
        "--allow-scale",
        action="store_true",
        default=None,
        help="我允许刚体配准估计整体尺度，但非 1 缩放仍会进入复核。",
    )
    # 采样与距离后端：采样点越多越准越慢；后端决定用暴力还是 KD-tree 宽相位。
    #
    # 我为什么默认把采样点数与查询点数都设成 None（交给配置文件）：
    # 这两个参数是一对——采样点多了距离更准但耗时线性上涨；而查询点数
    # 只截断"算多少个点"，截断后结果是**有偏**的（偏向先采到的面）。
    # 我实测在真实件上把 max-query-points 压得太低会让 p95 误差偏小，
    # 看起来更准实际是采样不全。所以我不给一个看似省时的默认值，
    # 而是让配置文件显式声明、命令行只在明确要覆盖时才传。
    inspect_parser.add_argument(
        "--samples-per-triangle",
        type=int,
        default=None,
        help="我对每个三角面使用的确定性采样点数量。",
    )
    # 我限制查询点数是为了守住内存上限，不是为了加速：距离计算的内存
    # 随查询点数线性增长，真实大件上不设限会直接把进程撑爆。
    # 这也是我不把它设成"加速选项"的原因——它会牺牲结果的无偏性。
    inspect_parser.add_argument(
        "--max-query-points",
        type=int,
        default=None,
        help="我限制单方向最多查询的采样点数量。",
    )
    # 2026-10-25 接管续写：面级判据是否把该面的三个顶点也纳入最大值。
    # 我为什么要这个开关：任务书把缺失区域定义成"误差 >= 阈值的区域"，
    # 而默认 1 点/面且取重心，采不到面上误差最大的那个位置；
    # 实测（0.2mm 网格 / 0.15mm 阈值）真值 15 面里有 7 面因此漏掉。
    # 代价是每面多 3 次距离查询，是否接受属于验收口径决策，所以默认关闭。
    inspect_parser.add_argument(
        "--face-corner-check",
        action="store_true",
        default=None,
        help=(
            "我把每个三角面的三个顶点也纳入面级最大误差判据"
            "（默认关闭；打开会明显增加距离查询量）。"
        ),
    )
    # auto 是我推荐的默认：它先按规模估一次代价，小数据走 bruteforce
    # （省掉建树的时间），大数据走 scipy_kdtree。我实测在几十万点的规模
    # 上 KD-tree 比暴力快一个数量级以上；而几千点以内建树的固定开销
    # 反而比直接暴力更慢，所以 auto 的切换点是必要的。
    inspect_parser.add_argument(
        "--distance-backend",
        choices=["auto", "bruteforce", "scipy_kdtree"],
        default=None,
        help="我选择点到网格距离后端。",
    )
    # KD-tree 的初始候选面数与距离计算分块大小，都是"内存换速度"的旋钮。
    inspect_parser.add_argument(
        "--distance-candidate-k",
        type=int,
        default=None,
        help="我设置 KD-tree 宽相位的初始候选三角面数量。",
    )
    inspect_parser.add_argument(
        "--distance-chunk-size",
        type=int,
        default=None,
        help="我设置距离计算的分块点数。",
    )
    inspect_parser.add_argument(
        "--bidirectional-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="我启用或关闭候选到参考的反向检查。",
    )
    # 单位策略：assume_mm 直接当毫米；unknown 只记录不换算；require_mm 非毫米直接失败。
    #
    # 我为什么不默认 assume_mm：STEP 文件头里的单位声明在真实件上经常
    # 是英寸或缺失，一律当毫米会让面积差一个 645.16 倍的平方因子，
    # 而面积是甲方核对的核心量。require_mm 是最严的（非毫米直接失败，
    # 不给错数），unknown 把单位原样记录进 JSON 让复核者自己判断。
    # 我把最严的留给批量验收，把 assume_mm 留给已知来源固定的产线。
    inspect_parser.add_argument(
        "--unit-policy",
        choices=["assume_mm", "unknown", "require_mm"],
        default=None,
        help="我选择输入单位处理策略。",
    )
    # 输出开关：中间 JSON 供审计，彩色网格方便人眼复核但更占磁盘。
    #
    # 我为什么把中间 JSON 设成默认关闭：它逐面记录距离，面数上万时
    # 单个文件就有几十 MB，批量跑会先把磁盘占满。但它是审计唯一能
    # 追溯"这个区域的距离是怎么算出来的"的证据，所以我在验收批次上
    # 一律显式打开它，代价是磁盘。
    inspect_parser.add_argument(
        "--write-intermediate",
        action="store_true",
        default=None,
        help="我写出 Face 距离中间 JSON 供审计。",
    )
    # 彩色网格默认开：它是复核者唯一能"一眼看出"异常面在哪的载体，
    # 而它只是把顶点复制三倍（见 writers.write_colored_ply），
    # 比中间 JSON 便宜得多。要省磁盘该先关中间 JSON，不是关这个。
    inspect_parser.add_argument(
        "--no-colored-mesh",
        action="store_true",
        default=None,
        help="我关闭彩色参考网格输出以节省磁盘空间。",
    )
    # 自动阈值三件套：阈值 = max(尺度分量, 噪声分量, 绝对下限)。
    # 尺度分量 = 包围盒对角线 × scale_ratio；噪声分量 = 配准平均残差 × residual_factor。
    #
    # 我为什么用 max 而不是加权和：这两项量纲不同、来源不同，
    # 相加会让"配准很差"和"件很大"互相掩盖。取 max 的语义是
    # "两个下界都要守住"——件大就该容忍更大的绝对偏差，配准差就该
    # 容忍更大的噪声，两者谁更苛刻听谁的。绝对下限则是兜底，
    # 防止两个分量都退化到 0 时阈值变成 0（那会把每个面都判成异常）。
    inspect_parser.add_argument(
        "--threshold-mode",
        choices=["absolute", "auto"],
        default=None,
        help=(
            "我选择阈值模式：absolute 用固定阈值，"
            "auto 按包围盒尺度和配准残差自动推导。"
        ),
    )
    inspect_parser.add_argument(
        "--auto-threshold-scale-ratio",
        type=float,
        default=None,
        help="我设置自动阈值的尺度分量系数（包围盒对角线的比例）。",
    )
    inspect_parser.add_argument(
        "--auto-threshold-residual-factor",
        type=float,
        default=None,
        help="我设置自动阈值的噪声分量系数（配准平均残差的倍数）。",
    )
    inspect_parser.add_argument(
        "--auto-threshold-floor-mm",
        type=float,
        default=None,
        help="我设置自动阈值的绝对下限。",
    )
    # 工程区域关联：把 STEP 工程特征 JSON 挂到缺失区域/多余区域上，
    # 让报告能回答"缺的是孔还是台阶"，而不只是"这里少了一块面"。
    inspect_parser.add_argument(
        "--reference-engineering-json",
        default=None,
        help="我把参考模型的 STEP 工程区域 JSON 关联到缺失区域。",
    )
    inspect_parser.add_argument(
        "--candidate-engineering-json",
        default=None,
        help="我把待检测模型的 STEP 工程区域 JSON 关联到多余区域。",
    )

    # ------------------------------------------------------------------
    # validate：只做网格质量体检。它不碰参考模型，也不判定缺失区域，
    # 用于甲方先确认"这份 STL 本身能不能用"。
    # ------------------------------------------------------------------
    validate_parser = subparsers.add_parser(
        "validate",
        help="我只检查 STL 网格，不执行缺失区域判定。",
    )
    validate_parser.add_argument("mesh", help="我检查的 STL 文件。")

    # ------------------------------------------------------------------
    # repair：STL 网格孔洞修补。只有同时满足"边界边数少、面积小、
    # 近似平面"这三个安全条件的孔洞才会被自动补；其余一律进复核，
    # 不自动改几何。
    # ------------------------------------------------------------------
    repair_parser = subparsers.add_parser(
        "repair",
        help="我对 STL 中满足安全条件的简单孔洞执行局部修补。",
    )
    repair_parser.add_argument("mesh", help="我需要修补的 STL 文件。")
    repair_parser.add_argument(
        "--output-dir",
        default="outputs/repair",
        help="我保存修复模型和报告的目录。",
    )
    # 三条安全门：边界边太多、面积太大、平面度超差的孔洞都不自动补。
    repair_parser.add_argument("--max-boundary-edges", type=int, default=128)
    repair_parser.add_argument("--max-hole-area-mm2", type=float, default=10000.0)
    # 我为什么不再硬编码 default：这几处原来写死 0.05 / store_true(默认
    # False)，会把 `RepairConfig` 里的默认值**悄悄覆盖掉**——我改了
    # dataclass 的默认值，命令行里却还是旧的（在基准上实测撞到了）。
    # 默认值只在 RepairConfig 定一处，CLI 从这里取，两处才不会打架。
    repair_parser.add_argument(
        "--max-planarity-error-mm",
        type=float,
        default=RepairConfig().max_planarity_error_mm,
    )
    # BooleanOptionalAction 同时给 --allow-nonconvex / --no-allow-nonconvex，
    # 默认变 True 之后仍然能显式关掉（原 store_true 只能开不能关）。
    repair_parser.add_argument(
        "--allow-nonconvex",
        action=argparse.BooleanOptionalAction,
        default=RepairConfig().allow_nonconvex,
    )
    repair_parser.add_argument(
        "--allow-review-output",
        action="store_true",
        help="我允许把未通过质量门禁的待复核网格返回给调用方。",
    )
    repair_parser.add_argument(
        "--do-no-harm",
        action="store_true",
        help="我只在修补后开放边界不比修补前更多时才写出结果。",
    )

    # ------------------------------------------------------------------
    # step-repair：STEP 开放边界补面。比 STL 版多了 G1 法向连续性、
    # G2 曲率连续性、非平面候选三类控制，因为 BRep 补面要接得平滑。
    # ------------------------------------------------------------------
    step_repair_parser = subparsers.add_parser(
        "step-repair",
        help="我对 STEP 开放边界中的安全平面孔洞执行局部补面。",
    )
    step_repair_parser.add_argument(
        "step",
        help="我需要修补的 .stp/.step 文件。",
    )
    step_repair_parser.add_argument(
        "--output-dir",
        default="outputs/step_repair",
        help="我保存修补后的 STEP、JSON 和 CSV 的目录。",
    )
    step_repair_parser.add_argument("--max-boundary-edges", type=int, default=128)
    step_repair_parser.add_argument("--max-hole-area-mm2", type=float, default=10000.0)
    step_repair_parser.add_argument(
        "--max-planarity-error-mm",
        type=float,
        default=RepairConfig().max_planarity_error_mm,
    )
    # G1：补面与相邻曲面的法向夹角上限，超过说明接缝会鼓出来。
    step_repair_parser.add_argument(
        "--max-g1-normal-angle-deg",
        type=float,
        default=5.0,
        help="我允许的补面与相邻曲面最大法向夹角。",
    )
    step_repair_parser.add_argument(
        "--continuity-sample-count",
        type=int,
        default=5,
        help="我对每条边界 Edge 采样的连续性点数。",
    )
    # G2：沿边界采样的法向曲率量差上限，单位是 1/mm。
    step_repair_parser.add_argument(
        "--max-g2-curvature-delta-per-mm",
        type=float,
        default=0.01,
        help="我允许的采样法向曲率量差，单位为 1/mm。",
    )
    step_repair_parser.add_argument(
        "--allow-nonconvex",
        action=argparse.BooleanOptionalAction,
        default=RepairConfig().allow_nonconvex,
    )
    step_repair_parser.add_argument(
        "--allow-nonplanar-surface-candidate",
        # 我为什么把默认从"关"改成"开"：这个开关是"非平面环走曲面填充候选"。
        # 默认关着时，平面度超差的环既建不出平面面、又拿不到曲面候选，直接
        # 变成无解（典型：训练集/7 落到 REVIEW_REQUIRED）。我做了逐件对照，
        # 打开后 训练集/7 REVIEW->SUCCESS、**其余 10 件全部不变（零退化）**。
        # 原来 `action="store_true"` 默认 False，会**覆盖** RepairConfig 的
        # 默认值，所以两处要一起改。改成 BooleanOptionalAction 后默认开，
        # 可用 `--no-allow-nonplanar-surface-candidate` 关掉。
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "我允许对非平面闭合 Wire 生成待复核曲面填充候选（默认开启，"
            "可用 --no-allow-nonplanar-surface-candidate 关闭）。"
        ),
    )
    step_repair_parser.add_argument(
        "--allow-review-output",
        action="store_true",
        help="我允许把未通过封闭实体门禁的 STEP 候选写出供人工复核。",
    )
    step_repair_parser.add_argument(
        "--do-no-harm",
        action="store_true",
        help="我只在修补后开放边界不比修补前更多时才写出结果。",
    )

    # ------------------------------------------------------------------
    # step-report：纯只读的 BRep 拓扑统计。它不改几何、不做判定，
    # 是其它 STEP 子命令的"体检基线"。
    # ------------------------------------------------------------------
    step_report_parser = subparsers.add_parser(
        "step-report",
        help="我读取真实 STEP BRep 并输出 Face、Edge、Wire、Shell、Solid 统计。",
    )
    step_report_parser.add_argument("step", help="我读取的 .stp/.step 文件。")
    step_report_parser.add_argument(
        "--output",
        default="outputs/step_topology.json",
        help="我保存拓扑报告的 JSON 路径。",
    )
    step_report_parser.add_argument(
        "--skip-shape-validation",
        action="store_true",
        help="我跳过 BRepCheck 有效性检查，只统计拓扑。",
    )

    # ------------------------------------------------------------------
    # step-face-features：逐 Face 的曲面类型、面积、法向、包围盒。
    # 这一层是后续区域分类与工程特征识别的输入证据。
    # ------------------------------------------------------------------
    step_faces_parser = subparsers.add_parser(
        "step-face-features",
        help="我读取 STEP 并输出每个 Face 的曲面类型、面积、法向和包围盒。",
    )
    step_faces_parser.add_argument("step", help="我读取的 .stp/.step 文件。")
    step_faces_parser.add_argument(
        "--output",
        default="outputs/step_face_features.json",
        help="我保存 Face 特征报告的 JSON 路径。",
    )
    step_faces_parser.add_argument(
        "--skip-shape-validation",
        action="store_true",
        help="我跳过 BRepCheck 有效性检查，只提取 Face 特征。",
    )

    # ------------------------------------------------------------------
    # step-feature-classify：把 Face 聚成"上平面/下平面/侧壁/倾斜面"
    # 这类区域，并给出邻接候选。semantic-sample-limit 是资源保护开关：
    # 超过上限的 Face 不做实体内外采样，直接进复核。
    # ------------------------------------------------------------------
    step_features_parser = subparsers.add_parser(
        "step-feature-classify",
        help="我识别 STEP 的上下平面、侧壁、倾斜面和 Face 邻接候选。",
    )
    step_features_parser.add_argument("step", help="我读取的 .stp/.step 文件。")
    step_features_parser.add_argument(
        "--output",
        default="outputs/step_feature_classification.json",
        help="我保存面区域分类报告的 JSON 路径。",
    )
    step_features_parser.add_argument(
        "--horizontal-tolerance-deg",
        type=float,
        default=15.0,
    )
    step_features_parser.add_argument("--z-band-ratio", type=float, default=0.1)
    step_features_parser.add_argument(
        "--semantic-sample-limit",
        type=int,
        default=16,
        help="我最多对多少个 Face 执行实体内外采样，超过后进入资源保护复核。",
    )
    step_features_parser.add_argument(
        "--skip-shape-validation",
        action="store_true",
        help="我跳过 BRepCheck 有效性检查，只执行 Face 候选分类。",
    )

    # ------------------------------------------------------------------
    # step-engineering-features：台阶、圆柱/孔、R 角、拔模方向。
    # 这里参数最多，因为每个工程特征都有自己的几何容差：
    # 台阶看高度与夹角，孔看半径一致性与轴线平行/重合，拔模看抽模方向。
    # ------------------------------------------------------------------
    step_engineering_parser = subparsers.add_parser(
        "step-engineering-features",
        help="我输出台阶、圆柱/孔、R角和拔模方向的工程候选证据。",
    )
    step_engineering_parser.add_argument("step", help="我读取的 .stp/.step 文件。")
    step_engineering_parser.add_argument(
        "--output-dir",
        default="outputs/step_engineering",
        help="我保存工程特征 JSON 和 CSV 的目录。",
    )
    step_engineering_parser.add_argument(
        "--draw-direction",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="我使用的拔模方向，默认是 Z 轴正方向。",
    )
    step_engineering_parser.add_argument("--draft-tolerance-deg", type=float, default=15.0)
    # 台阶：高度既看相对比例也看绝对毫米下限，避免小零件上的微台阶被误判。
    step_engineering_parser.add_argument("--step-height-ratio", type=float, default=0.02)
    step_engineering_parser.add_argument("--min-step-height-mm", type=float, default=0.5)
    step_engineering_parser.add_argument("--min-step-angle-deg", type=float, default=15.0)
    step_engineering_parser.add_argument(
        "--horizontal-tolerance-deg",
        type=float,
        default=15.0,
        help="我判断水平、侧壁和倾斜面的法向容差。",
    )
    step_engineering_parser.add_argument(
        "--z-band-ratio",
        type=float,
        default=0.1,
        help="我判断上下平面候选的 Z 包围盒带宽比例。",
    )
    # 孔：同一圆柱面的半径要一致，同一孔的多个圆柱轴线要平行且重合。
    step_engineering_parser.add_argument(
        "--hole-radius-tolerance-ratio",
        type=float,
        default=0.05,
        help="我判断圆柱边界半径一致性的相对容差。",
    )
    step_engineering_parser.add_argument(
        "--axis-angle-tolerance-deg",
        type=float,
        default=5.0,
        help="我判断圆柱轴线近似平行的角度容差。",
    )
    step_engineering_parser.add_argument(
        "--axis-distance-tolerance-mm",
        type=float,
        default=0.5,
        help="我判断圆柱轴线近似重合的距离容差。",
    )
    step_engineering_parser.add_argument(
        "--semantic-sample-limit",
        type=int,
        default=16,
        help="我最多对多少个 Face 执行实体内外语义采样，超过后保留资源保护复核。",
    )
    step_engineering_parser.add_argument(
        "--skip-shape-validation",
        action="store_true",
        help="我跳过 BRepCheck 有效性检查，但结果仍保留形状复核状态。",
    )

    # ------------------------------------------------------------------
    # probe：只读元数据探测。它不三角化、不做几何判定，用于甲方先确认
    # "这批文件到底是什么格式、里面有多少实体"，避免大文件白跑一遍。
    # ------------------------------------------------------------------
    probe_parser = subparsers.add_parser(
        "probe",
        help="我探测 STEP/STP 或其他输入格式的文件元数据，不执行实体三角化。",
    )
    probe_parser.add_argument("paths", nargs="+", help="我探测的输入文件或目录。")
    probe_parser.add_argument(
        "--output-dir",
        default="outputs/probe",
        help="我保存单文件报告和批量汇总的目录。",
    )
    probe_parser.add_argument(
        "--batch",
        action="store_true",
        help="我递归扫描目录并生成 batch_summary.json/csv。",
    )
    probe_parser.add_argument(
        "--no-files",
        action="store_true",
        help="我只输出汇总，不写单文件 JSON 报告。",
    )
    probe_parser.add_argument("--compact", action="store_true", help="我压缩单文件 JSON。")

    # ------------------------------------------------------------------
    # audit：输出目录体检。它按格式真的解析一遍，发现"文件在但内容坏"
    # 的情况；不解析的后缀记成 SKIPPED，而不是假装通过。
    # ------------------------------------------------------------------
    audit_parser = subparsers.add_parser(
        "audit",
        help="我审计输出目录中的 JSON、CSV、XLSX、PLY 和 STL 完整性。",
    )
    audit_parser.add_argument("path", help="我审计的输出目录或单个文件。")
    audit_parser.add_argument(
        "--output",
        default=None,
        help="我可选的审计 JSON 输出路径。",
    )
    audit_parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="我读取单个输出文件的最大字节数。",
    )

    # ------------------------------------------------------------------
    # validate-annotations：训练前数据泄漏门禁。它检查真实标注清单里
    # 的样本有没有同时出现在训练集和测试集，避免"背答案"式评估。
    # ------------------------------------------------------------------
    # 我为什么要把这四个区域 CLI 接进来：它们各自都有完整的
    # _build_parser + main，但**一个都没挂到主 CLI 上**——写好了却
    # 没人能调用（我用 rg 全库搜过 bom_cli / dataset_cli / feature_cli /
    # visualization_cli 的引用，命中数为 0）。这里用 REMAINDER 透传，
    # 让每个区域 CLI 自己解析自己的参数，主 CLI 不重复发明一遍。
    for _name, _help in (
        ("bom", "我跑 BOM 区域：从 STEP 抽取物料明细并与主数据核对。"),
        ("dataset", "我跑数据集区域：生成/校验训练数据清单。"),
        ("feature-classify", "我跑特征区域：STEP 面特征分类。"),
        ("visualize", "我跑可视化区域：把检查结果画成图。"),
        # 2026-10-25 接管续写补的第五个：分型基线区域 CLI。
        # 它此前只有 `mold-parting-baseline` 控制台脚本和
        # `python -m cad_inspector.parting_cli` 两个入口，主 CLI 侧调不到。
        # 名字用 parting-baseline 而不是 parting：后者是旧的网格基线命令，
        # 已经在用，不能顶掉。
        ("parting-baseline", "我跑分型区域：分型线与分型面候选（含 CAD 写出）。"),
    ):
        _sub = subparsers.add_parser(_name, help=_help)
        _sub.add_argument(
            "rest",
            nargs=argparse.REMAINDER,
            help=f"我原样透传给 {_name} 子命令的参数。",
        )

    annotations_parser = subparsers.add_parser(
        "validate-annotations",
        help="我校验真实标注清单并执行训练前数据泄漏门禁。",
    )
    annotations_parser.add_argument(
        "manifest",
        help="我校验的 missing-region-annotations-1.1 JSON 清单（1.0 仍可读）。",
    )
    annotations_parser.add_argument(
        "--output",
        default=None,
        help="我可选的校验摘要 JSON 输出路径。",
    )

    # ------------------------------------------------------------------
    # score-annotations：拿**人工确认过**的真实标注给预测打分。
    # 这是全仓库唯一能算出任务书那条 99% 的入口——合成真值只能证明
    # "已知孔洞能不能找回来"，草稿更只能得到假准确率。
    # ------------------------------------------------------------------
    scoring_parser = subparsers.add_parser(
        "score-annotations",
        help="我用真实标注给检测预测打分，输出面级 precision/recall。",
    )
    scoring_parser.add_argument(
        "manifest",
        help=(
            "我评分的标注清单（missing-region-annotations-1.1；"
            "1.0 的存量清单仍可读，但打不出 extra）。"
        ),
    )
    scoring_parser.add_argument(
        "--predictions-root",
        required=True,
        help="我递归查找 result.json 的根目录。",
    )
    scoring_parser.add_argument(
        "--defect-type",
        choices=("missing", "extra", "both"),
        default="missing",
        help=(
            "我打哪一类缺陷：missing（缺料，默认，输出形状与既有审计材料"
            "一致）、extra（多余材料）、both（两类分列，只在两类都算得出时"
            "才给 SCORED）。"
        ),
    )
    scoring_parser.add_argument(
        "--output",
        default=None,
        help="我可选的评分结果 JSON 输出路径。",
    )

    # ------------------------------------------------------------------
    # review-annotations：草稿 -> 人工复核 -> 确认清单。
    # 这是 bootstrap-annotations 与 score-annotations 之间原本缺失的一环：
    # 没有它，草稿要变真值只能手工编辑大 JSON，99% 就永远停在"缺数据"上。
    # ------------------------------------------------------------------
    review_parser = subparsers.add_parser(
        "review-annotations",
        help="我把标注草稿导成人工可填的复核表，或把填好的表并回清单。",
    )
    review_parser.add_argument(
        "manifest",
        help="我复核的 missing-region-annotations-1.1 草稿清单（1.0 仍可读）。",
    )
    review_parser.add_argument(
        "--export-sheet",
        default=None,
        help="我把每个区域导出成一行 CSV 供人工填写。",
    )
    review_parser.add_argument(
        "--apply-sheet",
        default=None,
        help="我读入填好的复核表并把判定并回清单。",
    )
    review_parser.add_argument(
        "--output",
        default=None,
        help="我写出确认后清单的 JSON 路径（配合 --apply-sheet）。",
    )

    # ------------------------------------------------------------------
    # evaluate-synthetic：用合成删除面标签评估缺失区域基线。
    # 它按 seed-face-ids 与 face-counts 组合出多个案例，再扫阈值，
    # 给出"在当前参考件上，阈值取多少能达到 target 命中率"的基线。
    # ------------------------------------------------------------------
    evaluation_parser = subparsers.add_parser(
        "evaluate-synthetic",
        help="我用合成删除面标签评估缺失区域基线。",
    )
    evaluation_parser.add_argument("reference", help="我使用的参考 STL 文件。")
    evaluation_parser.add_argument(
        "--output",
        default="outputs/evaluation/synthetic_evaluation.json",
        help="我保存评估 JSON 的路径。",
    )
    evaluation_parser.add_argument(
        "--seed-face-ids",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="我生成合成缺陷的种子 Face 编号。",
    )
    evaluation_parser.add_argument(
        "--face-counts",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="我每个案例删除的连续 Face 数量。",
    )
    evaluation_parser.add_argument("--target", type=float, default=0.99)
    evaluation_parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
        help="我可选的阈值扫描列表，单位为 mm。",
    )
    evaluation_parser.add_argument(
        "--defect-type",
        choices=("missing", "extra"),
        default="missing",
        help="我评估哪一类缺陷：missing 是删面造缺料，extra 是外推造多料。",
    )
    evaluation_parser.add_argument(
        "--offset-mm",
        type=float,
        default=1.0,
        help=(
            "我在 extra 下把复制的面沿法向外推多少 mm。"
            "必须大于 --threshold 才可能被检出。"
        ),
    )
    # 2026-10-25 接管续写：任务书口径真值的采样密度。
    # 我为什么把它做成命令行开关：真值口径必须能被独立地重算与复核，
    # 密度写死在代码里，甲方就无法验证"密度变了结论会不会变"。
    evaluation_parser.add_argument(
        "--truth-samples-per-triangle",
        type=int,
        default=9,
        help=(
            "我量任务书口径真值（误差 >= 阈值的面）时每面取几个面内点；"
            "另外每个面的三个顶点恒定参与，不受该值影响。"
        ),
    )
    # 生产侧的采样密度与最小区域面积也开放出来：
    # "轮廓带"和"检出后被面积门限丢掉"这两类漏检的收口动作分别是
    # 提采样密度、调面积门限，不能在评测里锁死而只能在生产里调。
    evaluation_parser.add_argument(
        "--samples-per-triangle",
        type=int,
        default=None,
        help="我让生产检测每面取几个点（默认沿用配置的 1 点/面）。",
    )
    evaluation_parser.add_argument(
        "--min-region-area-mm2",
        type=float,
        default=None,
        help="我让生产检测用指定的最小区域面积（默认沿用配置的 0.5 mm2）。",
    )
    # 面级顶点采样开关：评测里必须能打开它，否则"补上顶点后能不能达标"
    # 只能靠人手工改代码去试，那就不叫可复核的证据了。
    evaluation_parser.add_argument(
        "--face-corner-check",
        action="store_true",
        default=None,
        help=(
            "我把每个三角面的三个顶点也纳入面级最大误差判据"
            "（默认关闭；打开会明显增加距离查询量）。"
        ),
    )

    # ------------------------------------------------------------------
    # parting：从 STL 提取水平截面分型线，并生成诊断用平面分型面。
    # 它目前是网格基线，不是工业级最终分型面。
    # ------------------------------------------------------------------
    parting_parser = subparsers.add_parser(
        "parting",
        help="我从 STL 网格提取水平截面分型线和诊断用平面分型面。",
    )
    parting_parser.add_argument("mesh", help="我处理的 STL 网格文件。")
    parting_parser.add_argument(
        "--output-dir",
        default="outputs/parting",
        help="我保存分型 JSON、CSV 和 STL 的目录。",
    )
    parting_parser.add_argument("--plane-z", type=float, default=None)
    parting_parser.add_argument("--scan-count", type=int, default=9)
    parting_parser.add_argument("--surface-margin-mm", type=float, default=5.0)
    # ------------------------------------------------------------------
    # voxel：体素占用 + 逐层截面的辅助比对。它补的是"面级距离看不见"
    # 的内部空腔与体量缺失，不替换 missing_regions 的面级判定。
    # ------------------------------------------------------------------
    voxel_parser = subparsers.add_parser(
        "voxel",
        help="我用体素占用与逐层截面做辅助比对，补上内部空腔与体量缺失。",
    )
    voxel_parser.add_argument("reference", help="我作为完整基准的网格文件。")
    voxel_parser.add_argument("candidate", help="我待检查的网格文件。")
    voxel_parser.add_argument(
        "--output-dir",
        default="outputs/voxel",
        help="我保存体素比对 JSON 的目录。",
    )
    voxel_parser.add_argument(
        "--voxel-size-mm", type=float, default=1.0, help="我的体素边长。"
    )
    voxel_parser.add_argument(
        "--max-voxels",
        type=int,
        default=2_000_000,
        help="我的体素总数上限；超了就自动放大体素边长。",
    )
    # ------------------------------------------------------------------
    # batch-inspect：批量"候选件 vs 参考件"检测。我按文件名主干配对，
    # 单个件失败只影响自身，不中断整批。
    # ------------------------------------------------------------------
    batch_parser = subparsers.add_parser(
        "batch-inspect",
        help="我批量检测多对模型，按文件名主干配对并给出汇总台账。",
    )
    batch_parser.add_argument(
        "candidates", nargs="+", help="我的候选件文件或目录（可多个）。"
    )
    batch_parser.add_argument(
        "--references",
        nargs="+",
        required=True,
        help="我的参考件文件或目录；我按文件名主干与候选件配对。",
    )
    batch_parser.add_argument(
        "--output-dir",
        default="outputs/batch_inspect",
        help="我保存每对结果子目录与汇总台账的目录。",
    )
    batch_parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="我最多处理多少对；不填就全处理。建议先小数量试跑验证参数。",
    )
    # ------------------------------------------------------------------
    # train-face-model：训练面级深度学习增强。它是**增强**而不是替换
    # 几何判据，所以我把融合口径也在这里写清。
    # ------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train-face-model",
        help="我用合成真值训练面级缺失分类器（可选依赖 torch）。",
    )
    train_parser.add_argument("reference", help="我使用的参考 STL 文件。")
    train_parser.add_argument(
        "--output",
        default="outputs/learning/face_model.pt",
        help="我保存模型权重的路径。",
    )
    train_parser.add_argument(
        "--seed-face-ids",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="我生成合成缺陷的种子 Face 编号。",
    )
    train_parser.add_argument(
        "--face-counts",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="我每个案例删除的连续 Face 数量。",
    )
    train_parser.add_argument("--epochs", type=int, default=60)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument(
        "--threshold-mm",
        type=float,
        default=0.15,
        help="我判定缺失的距离阈值，与线上检测口径一致。",
    )
    # ------------------------------------------------------------------
    # serve：起 HTTP 服务。检测是耗时活，所以我做成"提交任务 + 查结果"
    # 两步，而不是一个会占住线程几十秒的同步接口。
    # ------------------------------------------------------------------
    serve_parser = subparsers.add_parser(
        "serve",
        help="我启动缺失区域检测的 HTTP 服务。",
    )
    serve_parser.add_argument(
        "--allowed-root",
        required=True,
        help="我允许读取模型的根目录；这是安全边界，不是普通配置。",
    )
    serve_parser.add_argument(
        "--output-root",
        default=None,
        help="我保存任务结果的目录；缺省为 allowed-root/_api_output。",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="我并发跑检测的线程数；检测是 CPU 密集活，开多无益。",
    )
    # ------------------------------------------------------------------
    # gui：桌面界面。判定仍走 pipeline，界面只负责选件与看结论，
    # 保证界面与命令行是同一套口径。
    # ------------------------------------------------------------------
    gui_parser = subparsers.add_parser(
        "gui",
        help="我启动桌面窗口来选件、跑检测、看结论。",
    )
    gui_parser.add_argument(
        "--output-dir",
        default="outputs/desktop_inspect",
        help="我保存界面跑出来的检测结果的目录。",
    )
    # ------------------------------------------------------------------
    # tessellate-step：OCCT 三角化 + CAD 面 ID 映射（扩展顺序第 2 项）。
    # ------------------------------------------------------------------
    tess_parser = subparsers.add_parser(
        "tessellate-step",
        help="我用 OCCT 三角化 STEP，并给出三角形到 CAD 面 ID 的映射。",
    )
    tess_parser.add_argument("step", help="我要三角化的 STEP/STP 文件。")
    tess_parser.add_argument("--output", default=None, help="我保存 JSON 报告的路径。")
    tess_parser.add_argument("--output-stl", default=None, help="我把三角化结果另存为 STL。")
    tess_parser.add_argument(
        "--linear-deflection",
        type=float,
        # 我为什么把默认从 0.5 改成 0.04：InspectionConfig 用的就是 0.04，
        # 而 CAD 面映射要求三角化参数与加载参考件时**完全一致**——
        # 参数不同会得到不同数量的三角形，下标对不上，映射出的
        # CAD 面号全是错的，且不报错。用这个命令单独三角化再和
        # 检测结果比对时，0.5 会静默产出对不上的结果。
        default=0.04,
        help="弦高误差，越小越贴曲面但三角数暴涨；默认与检测管线一致。",
    )
    flowchart_parser = subparsers.add_parser(
        "flowchart",
        help="我生成任务书要求的流程图（主流程 / 交付门禁 / 源码区域）。",
    )
    flowchart_parser.add_argument(
        "--output-dir",
        default="outputs/flowcharts",
        help="我保存流程图 PNG 与 Mermaid 文本的目录。",
    )
    flowchart_parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="我只出 PNG，不出 Mermaid 文本。",
    )
    flowchart_parser.add_argument(
        "--strict",
        action="store_true",
        help="我发现图与源码不一致时返回非零退出码。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """我解析命令行、分发到对应命令，并统一处理异常与退出码。

    我的分发方式是"早返回的长 if 链"：每个子命令处理完直接 return，
    这样不需要嵌套 else，也不会有某个分支漏掉 return 掉到下一个分支。
    最后那个没有 if 保护的代码块就是 inspect——它是默认命令。
    """

    _configure_utf8_stdio()
    # 我为什么必须在 parse_args **之前**拦截这四个区域命令：
    # argparse 的 REMAINDER 只在遇到第一个**位置参数**之后才开始收集
    # 剩余项。实测 visualize 的唯一必填参数是 --before（一个选项），
    # 以 `visualize --before X` 调用时主 parser 会先尝试解析 --before、
    # 不认识就报 unrecognized arguments，透传直接断掉——也就是
    # **visualize 永远调不通**。而 bom/dataset/feature-classify 都以
    # 位置参数开头，所以它们能透传。这里绕开 argparse 直接分发，
    # 四种开头方式都能正确处理。
    delegated = {
        "bom": bom_main,
        "dataset": dataset_main,
        "feature-classify": feature_main,
        "visualize": visualization_main,
        # 第五个区域 CLI 同样绕开 argparse 直接分发：
        # 它的必填参数是位置参数 input，用 REMAINDER 也能透传，
        # 但统一走这条路径才不会再踩"以选项开头就断掉"的那个坑。
        "parting-baseline": parting_baseline_main,
    }
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] in delegated:
        return int(delegated[argv_list[0]](argv_list[1:]))
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # validate：只体检网格质量，直接打印报告，不写文件。
        if args.command == "validate":
            mesh = load_stl(args.mesh)
            report = mesh_quality_report(mesh)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0

        # repair：STL 网格补孔。只有 success/partial 才写 repaired.stl；
        # no_action 也返回 0，因为"没有需要补的孔"不是失败。
        if args.command == "repair":
            mesh = load_stl(args.mesh)
            repair_config = RepairConfig(
                max_boundary_edges=args.max_boundary_edges,
                max_hole_area_mm2=args.max_hole_area_mm2,
                max_planarity_error_mm=args.max_planarity_error_mm,
                allow_nonconvex=args.allow_nonconvex,
                preserve_original_on_failure=not args.allow_review_output,
                do_no_harm=bool(args.do_no_harm),
            )
            repaired_mesh, result = repair_mesh_holes(
                mesh,
                config=repair_config,
                input_file=args.mesh,
            )
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if result.status in {"success", "partial"}:
                repaired_path = output_dir / "repaired.stl"
                save_stl(repaired_mesh, repaired_path, solid_name="repaired")
                result.output_file = str(repaired_path.resolve())
                result.artifacts["repaired_stl"] = str(repaired_path.resolve())
            json_path = write_repair_json(result, output_dir / "repair.json")
            csv_path = write_repair_csv(result, output_dir / "repair_regions.csv")
            result.artifacts["json"] = str(json_path.resolve())
            result.artifacts["csv"] = str(csv_path.resolve())
            write_repair_json(result, json_path)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0 if result.status in {"success", "partial", "no_action"} else 1

        # step-repair：STEP 补面。状态只有 SUCCESS 才算通过，其余进复核。
        if args.command == "step-repair":
            output_dir = Path(args.output_dir)
            stem = Path(args.step).stem
            config = RepairConfig(
                max_boundary_edges=args.max_boundary_edges,
                max_hole_area_mm2=args.max_hole_area_mm2,
                max_planarity_error_mm=args.max_planarity_error_mm,
                max_g1_normal_angle_deg=args.max_g1_normal_angle_deg,
                max_g2_curvature_delta_per_mm=(
                    args.max_g2_curvature_delta_per_mm
                ),
                # 连续性采样点数：G1/G2 是在采样点上判的，
                # 点数直接决定这两个判据的分辨率。
                continuity_sample_count=args.continuity_sample_count,
                allow_nonconvex=args.allow_nonconvex,
                allow_nonplanar_surface_candidate=(
                    args.allow_nonplanar_surface_candidate
                ),
                # 默认 do_no_harm（校验不过就退回原件）。
                # 只有显式 --allow-review-output 才关掉保留原件，
                # 此时输出可能是没过门禁的补面，仅供人工复核。
                preserve_original_on_failure=not args.allow_review_output,
                do_no_harm=bool(args.do_no_harm),
            )
            step_output = output_dir / f"{stem}.repaired.step"
            result = repair_step_holes(
                args.step,
                output_path=step_output,
                config=config,
            )
            json_path = write_step_repair_json(
                result,
                output_dir / f"{stem}.step-repair.json",
            )
            csv_path = write_step_repair_csv(
                result,
                output_dir / f"{stem}.step-repair.csv",
            )
            # 产物路径回写进 result.artifacts：
            # 报告里要带上"去哪找这些文件"，
            # 否则调用方得自己按命名规则拼。
            result.artifacts["json"] = str(json_path.resolve())
            result.artifacts["csv"] = str(csv_path.resolve())
            if result.output_file:
                result.artifacts["repaired_step"] = result.output_file
            # 写完 CSV 后再重写一次 JSON：
            # 第一次写的时候 artifacts 里还没有这两个路径，
            # 重写一次才能保证 JSON 里记录的路径是完整的。
            write_step_repair_json(result, json_path)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            # 只有 SUCCESS 才是 0：其余状态（未修补、需复核、失败）
            # 都是 1，调用方靠退出码决定是否继续后续流程。
            return 0 if result.status == "SUCCESS" else 1

        # step-report：拓扑统计。COMPLETE 才算通过。
        if args.command == "step-report":
            result = inspect_step_topology(
                args.step,
                validate_shape=not args.skip_shape_validation,
            )
            output_path = write_step_topology_json(result, args.output)
            payload = result.to_dict()
            payload["artifacts"] = {"json": str(output_path.resolve())}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if result.status == "COMPLETE" else 1

        # step-face-features：逐 Face 特征提取。COMPLETE 才算通过。
        if args.command == "step-face-features":
            result = inspect_step_face_features(
                args.step,
                validate_shape=not args.skip_shape_validation,
            )
            output_path = write_step_face_features_json(result, args.output)
            payload = result.to_dict()
            payload["artifacts"] = {"json": str(output_path.resolve())}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if result.status == "COMPLETE" else 1

        # step-feature-classify：面区域分类。我方只承诺 BASELINE 基线，
        # 所以只有 BASELINE 返回 0，COMPLETE 反而说明状态定义有问题。
        if args.command == "step-feature-classify":
            result = inspect_step_feature_classification(
                args.step,
                horizontal_tolerance_deg=args.horizontal_tolerance_deg,
                z_band_ratio=args.z_band_ratio,
                validate_shape=not args.skip_shape_validation,
                semantic_sample_limit=args.semantic_sample_limit,
            )
            output_path = write_step_feature_classification_json(result, args.output)
            payload = result.to_dict()
            payload["artifacts"] = {"json": str(output_path.resolve())}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if result.status == "BASELINE" else 1

        # step-engineering-features：工程特征候选。同样是 BASELINE 才算通过。
        if args.command == "step-engineering-features":
            result = inspect_step_engineering_features(
                args.step,
                draw_direction=args.draw_direction,
                draft_tolerance_deg=args.draft_tolerance_deg,
                step_height_ratio=args.step_height_ratio,
                min_step_height_mm=args.min_step_height_mm,
                min_step_angle_deg=args.min_step_angle_deg,
                horizontal_tolerance_deg=args.horizontal_tolerance_deg,
                z_band_ratio=args.z_band_ratio,
                hole_radius_tolerance_ratio=args.hole_radius_tolerance_ratio,
                axis_angle_tolerance_deg=args.axis_angle_tolerance_deg,
                axis_distance_tolerance_mm=args.axis_distance_tolerance_mm,
                semantic_sample_limit=args.semantic_sample_limit,
                validate_shape=not args.skip_shape_validation,
            )
            artifacts = write_step_engineering_report(result, args.output_dir)
            payload = result.to_dict()
            payload["artifacts"] = {
                key: str(Path(value).resolve()) for key, value in artifacts.items()
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if result.status == "BASELINE" else 1

        # probe：逐个文件探测元数据，再按需汇总成 batch。
        # 只要出现 FAILED 或 UNSUPPORTED，整体退出码就是 1。
        if args.command == "probe":
            paths = expand_inputs(args.paths)
            # 展开后一个文件都没有就返回 2：这与"文件存在但探不出来"
            # 是两回事，后者是退出码 1。用同一个码会让调用方
            # 无法区分"参数给错了"和"模型有问题"。
            if not paths:
                print("我没有发现可探测的模型文件。", file=sys.stderr)
                return 2
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            exit_code = 0
            for path in paths:
                result = probe_any(path)
                # 单个文件的失败不中断整批：一批里有几个打不开的件
                # 是常态，中断会让前面跑完的结果全部丢失。
                if not args.no_files:
                    target = output_dir / (
                        # 文件名里带 sha256 前 8 位：同名文件在不同目录
                        # 里很常见，不带哈希会让后写的覆盖先写的。
                        f"{path.stem}--{result.file.sha256[:8]}.probe.json"
                    )
                    with atomic_path(target) as temporary:
                        temporary.write_text(
                            json.dumps(
                                result.to_dict(),
                                ensure_ascii=False,
                                # compact 时 indent=None：这是给机器读的场景，
                                # 缩进只会白白增大文件。
                                indent=None if args.compact else 2,
                                allow_nan=False,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                print(
                    f"[{result.status.value}] {result.file.name} | "
                    f"{result.file.detected_format.value} | "
                    f"entities={sum(result.entity_counts.values())}"
                )
                # 只要有一个文件失败，整批退出码就是 1：
                # 调用方通常靠退出码判断要不要继续后续步骤。
                if result.status.value in {"FAILED", "UNSUPPORTED"}:
                    exit_code = 1
            # 多文件自动出汇总报告：逐个看单文件 JSON
            # 无法回答"这批里有多少个失败"。
            if args.batch or len(paths) > 1:
                batch = run_batch(paths)
                json_path, csv_path = write_batch_reports(batch, output_dir)
                print(
                    # 五个计数与两个输出路径一起打印：
                    # 不打印路径的话，调用方还得自己去目录里找。
                    f"[BATCH] total={batch.total} complete={batch.complete} "
                    f"partial={batch.partial} unsupported={batch.unsupported} "
                    f"failed={batch.failed} json={json_path} csv={csv_path}"
                )
                if batch.failed or batch.unsupported:
                    exit_code = 1
            return exit_code

        # audit：审计输出目录。只有全部有效（status=PASS）才返回 0。
        if args.command == "audit":
            report = audit_artifacts(
                args.path,
                max_file_bytes=args.max_file_bytes,
            )
            if args.output:
                output_path = Path(args.output)
                report["output"] = str(output_path.resolve())
                with atomic_path(output_path) as temporary:
                    temporary.write_text(
                        json.dumps(
                            report,
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["status"] == "PASS" else 1

        # validate-annotations：标注清单门禁。READY_FOR_TRAINING 才返回 0。
        if args.command == "validate-annotations":
            _, summary = load_annotation_manifest(args.manifest)
            report = summary.to_dict()
            report["manifest_path"] = str(
                Path(args.manifest).expanduser().resolve()
            )
            report["status"] = (
                "READY_FOR_TRAINING"
                if summary.ready_for_training
                else "REVIEW_REQUIRED"
            )
            if args.output:
                output_path = Path(args.output)
                report["output"] = str(output_path.resolve())
                with atomic_path(output_path) as temporary:
                    temporary.write_text(
                        json.dumps(
                            report,
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if summary.ready_for_training else 1

        # score-annotations：拿真实标注打分。只有"有确认记录且没有草稿
        # 混进来"才算一次可用的评分，其余一律返回 1，避免一个含草稿的
        # 数字被当成 99% 的证据。
        if args.command == "score-annotations":
            payload, _summary = load_annotation_manifest(args.manifest)
            predictions = collect_predictions(args.predictions_root)
            # 我按 --defect-type 分派：单类出**原来那份形状**的报告
            # （既有审计材料按这个读，默认口径不许变）；both 出分列报告，
            # 因为缺料与多余材料的真值在两套不同的网格上，混成一个
            # recall 谁也复核不了。
            if args.defect_type == "both":
                report = score_annotations_multi(
                    payload, predictions, defect_types=("missing", "extra")
                )
            else:
                report = score_annotations(
                    payload, predictions, defect_type=args.defect_type
                )
            report["manifest_path"] = str(
                Path(args.manifest).expanduser().resolve()
            )
            report["predictions_root"] = str(
                Path(args.predictions_root).expanduser().resolve()
            )
            if args.output:
                output_path = Path(args.output)
                report["output"] = str(output_path.resolve())
                with atomic_path(output_path) as temporary:
                    temporary.write_text(
                        json.dumps(
                            report,
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["status"] == "SCORED" else 1

        # review-annotations：草稿 <-> 复核表。
        # 导出只要草稿合法就返回 0；导回则要求真的产生了确认记录，
        # 否则返回 1——"表填了一半"不该被当成"复核完成"。
        if args.command == "review-annotations":
            if bool(args.export_sheet) == bool(args.apply_sheet):
                raise InspectorError(
                    "我要求 --export-sheet 与 --apply-sheet 二选一。"
                )
            payload, _summary = load_annotation_manifest(args.manifest)

            if args.export_sheet:
                sheet_stats = export_review_sheet(
                    payload, Path(args.export_sheet)
                )
                print(
                    f"我导出复核表：{Path(args.export_sheet).resolve()}，"
                    f"共 {sheet_stats['rows']} 行待判定"
                    f"（{sheet_stats['region_rows']} 个区域"
                    f" + {sheet_stats['record_rows']} 条整件判定）。"
                )
                print(
                    "  请在每个区域的 decision 列填写 accept 或 reject；"
                    "留空表示尚未判定，该记录不会被当成真值。"
                    "整件判定（scope=record）用于检测器一个区域都没报的件，"
                    "必须显式填写，否则那件不会被视为已确认。"
                )
                return 0

            if not args.output:
                raise InspectorError("我要求 --apply-sheet 同时给出 --output。")
            confirmed, review_summary = apply_review_sheet(
                payload, Path(args.apply_sheet)
            )
            output_path = Path(args.output)
            with atomic_path(output_path) as temporary:
                temporary.write_text(
                    json.dumps(
                        confirmed,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            print(
                "我写出确认清单：%s" % output_path.resolve()
            )
            print(
                "  接受 %d、剔除 %d、未决 %d；确认记录 %d 条、仍是草稿 %d 条。"
                % (
                    review_summary["accepted"],
                    review_summary["rejected"],
                    review_summary["undecided"],
                    review_summary["confirmed_records"],
                    review_summary["draft_records"],
                )
            )
            return 0 if review_summary["confirmed_records"] else 1

        # evaluate-synthetic：合成评估。命中目标才返回 0。
        if args.command == "evaluate-synthetic":
            reference = load_stl(args.reference)
            # 我在这里显式构造配置，而不是让套件内部取缺省：
            # 缺省值一旦变了，命令行跑出来的历史对照就悄悄换了口径。
            # registration_mode=none / unit_policy=assume_mm 与套件缺省一致。
            config_kwargs: dict[str, object] = {}
            if args.samples_per_triangle is not None:
                config_kwargs["samples_per_triangle"] = args.samples_per_triangle
            if args.min_region_area_mm2 is not None:
                config_kwargs["min_region_area_mm2"] = args.min_region_area_mm2
            if args.face_corner_check is not None:
                config_kwargs["face_corner_check"] = bool(args.face_corner_check)
            config = InspectionConfig(
                registration_mode="none",
                unit_policy="assume_mm",
                **config_kwargs,
            )
            report = evaluate_synthetic_suite(
                reference,
                seed_face_ids=args.seed_face_ids,
                face_counts=args.face_counts,
                config=config,
                target=args.target,
                thresholds=args.thresholds,
                defect_type=args.defect_type,
                offset_mm=args.offset_mm,
                truth_samples_per_triangle=args.truth_samples_per_triangle,
            )
            output_path = Path(args.output)
            with atomic_path(output_path) as temporary:
                temporary.write_text(
                    json.dumps(
                        report,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["status"] == "BASELINE_TARGET_MET" else 1

        # parting：网格分型线 + 诊断分型面。线闭环且两个阶段都是 BASELINE
        # 才算 BASELINE，否则整条结果降级成 REVIEW_REQUIRED。
        if args.command == "parting":
            mesh = load_stl(args.mesh)
            line_config = PartingConfig(
                plane_z_mm=args.plane_z,
                plane_scan_count=args.scan_count,
                surface_margin_mm=args.surface_margin_mm,
            )
            # 先提线再建面：分型面是由分型线长出来的，
            # 线提不出来就没有建面的输入。
            line = extract_parting_line(mesh, line_config)
            surface = build_planar_parting_surface(line, line_config)
            result = PartingPipelineResult(
                # 三个条件都满足才是 BASELINE：线是基线、线闭合、面是基线。
                # 任一不满足就整条降为 REVIEW_REQUIRED——
                # 分型线不闭合时分型面一定不完整，
                # 单独给面一个"基线"状态会误导。
                status=(
                    "BASELINE"
                    if (
                        line.status == "BASELINE"
                        and line.closed
                        and surface.status == "BASELINE"
                    )
                    else "REVIEW_REQUIRED"
                ),
                input_file=str(args.mesh),
                # 后端名写进结果：这份结论来自"网格水平截交"这一路，
                # 与 STEP 侧基于 B-Rep 的做法不是同一套判据，
                # 不写清会被混为一谈。
                backend="mesh_horizontal_section",
                line=line,
                surface=surface,
                line_candidates=[line],
                input_metadata=validate_input_path(args.mesh).to_dict(),
                # 线面两处的警告合并：分开看会漏掉其中一侧的问题，
                # 合并后一次就能看到这一趟所有需要留意的事。
                warnings=list(line.warnings) + list(surface.warnings),
                config=line_config,
            )
            artifacts = write_parting_result(result, args.output_dir)
            print(
                json.dumps(
                    {
                        "status": result.status,
                        # 线状态与面状态分开打印：整体状态是压缩后的结论，
                        # 两个分项才能定位是哪一步没过。
                        "line_status": line.status,
                        "surface_status": surface.status,
                        # 产物路径列表直接输出：调用方不用再去目录里找。
                        "artifacts": artifacts,
                        "warnings": result.warnings,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if result.status == "BASELINE" else 1

        # flowchart：按真实源码结构出主流程 / 门禁 / 模块三张图。
        # voxel：体素占用 + 逐层截面的辅助比对。我复用分型那套输入适配，
        # 让 STL 与 STEP 都能进；比对本身只吃 TriangleMesh，不看来源格式。
        if args.command == "batch-inspect":
            from .io.batch_inspect import (
                run_batch_inspect,
                write_batch_inspect_reports,
            )

            # expand_inputs 递归展开目录并按绝对路径排序去重，
            # 保证同一批输入在不同机器上的处理顺序一致。
            # 我必须按后缀分流：目录里常常 .stl 与 .stp 混放，
            # 不筛的话参考件目录里的 .stl 会被当成候选件，配对全乱。
            candidate_paths = [
                path
                for path in expand_inputs(args.candidates)
                if path.suffix.lower() == ".stl"
            ]
            reference_paths = [
                path
                for path in expand_inputs(args.references)
                if path.suffix.lower() in {".stp", ".step"}
            ]
            if not candidate_paths:
                print("我没有从输入里找到任何候选件文件。")
                return 2
            if not reference_paths:
                print("我没有从输入里找到任何参考件文件（.stp/.step）。")
                return 2
            report = run_batch_inspect(
                candidate_paths,
                reference_paths,
                args.output_dir,
                max_items=args.max_items,
            )
            paths = write_batch_inspect_reports(report, args.output_dir)
            print(
                json.dumps(
                    {
                        "status": "OK" if report.failed == 0 else "REVIEW_REQUIRED",
                        "total": report.total,
                        "ok": report.ok,
                        "failed": report.failed,
                        "unpaired": report.unpaired,
                        "elapsed_seconds": round(report.elapsed_seconds, 3),
                        "artifacts": {
                            key: str(value.resolve()) for key, value in paths.items()
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            # 有失败或有未配对件时返回非零：批量任务的"部分成功"
            # 必须让调用方（含 CI）看得见，否则脚本会静默吞掉坏件。
            return 0 if report.failed == 0 and report.unpaired == 0 else 1

        # train-face-model：训练面级增强。训练样本全部来自**合成真值**
        # （我删掉已知面、再把检测出的距离当特征），不掺入任何测试件，
        # 这一点由 annotations 模块的数据泄漏门禁保证。
        if args.command == "train-face-model":
            from .learning.face_classifier import (
                FaceClassifierConfig,
                save_face_classifier,
                train_face_classifier,
            )
            from .learning.features import build_face_features
            from .learning.synthetic_defects import remove_face_region

            reference = load_stl(args.reference)
            config = FaceClassifierConfig(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
            )
            feature_blocks: list[np.ndarray] = []
            label_blocks: list[np.ndarray] = []
            for seed in args.seed_face_ids:
                for count in args.face_counts:
                    defect = remove_face_region(reference, seed, count)
                    # 真值：被删掉的那些面就是"缺失"的正样本。
                    truth = np.zeros(len(reference.faces), dtype=np.int64)
                    truth[defect.removed_face_ids] = 1
                    # 距离：被删的面给一个明显超阈值的距离，
                    # 未删的面给 0。这与线上"距离越大越可能缺失"同向，
                    # 否则模型学到的方向和线上判定相反。
                    distances = np.where(truth == 1, args.threshold_mm * 2.0, 0.0)
                    features, labels = build_face_features(
                        reference, distances, args.threshold_mm
                    )
                    feature_blocks.append(features)
                    label_blocks.append(labels)
            all_features = np.vstack(feature_blocks)
            all_labels = np.concatenate(label_blocks)
            model = train_face_classifier(all_features, all_labels, config)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_face_classifier(model, output_path)
            print(
                json.dumps(
                    {
                        "status": "OK",
                        "reference": str(args.reference),
                        "model_path": str(output_path.resolve()),
                        "training_faces": int(len(all_labels)),
                        "positive_faces": int(all_labels.sum()),
                        "config": config.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        # serve：起 HTTP 服务。我在函数内部导入，缺 fastapi 时只让这一条
        # 命令失败，不影响其它子命令。
        if args.command == "serve":
            from .api.service import create_app

            app = create_app(
                args.allowed_root,
                output_root=args.output_root,
                max_workers=args.max_workers,
            )
            try:
                import uvicorn
            except ImportError:
                print(
                    "我无法启动服务：当前环境没有安装 uvicorn。"
                    "可执行：python -m pip install uvicorn"
                )
                return 2
            print(
                json.dumps(
                    {
                        "status": "SERVING",
                        "host": args.host,
                        "port": args.port,
                        "allowed_root": str(Path(args.allowed_root).resolve()),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            uvicorn.run(app, host=args.host, port=args.port)
            return 0

        # gui：桌面界面。我在函数内部导入 tkinter，缺它时只让这一条
        # 命令失败。注意它会阻塞到用户关闭窗口。
        if args.command == "gui":
            from .ui.desktop import launch_desktop_app

            launch_desktop_app(default_output_dir=args.output_dir)
            return 0

        # tessellate-step：OCCT 三角化 + CAD 面 ID 映射。我在函数内部导入，
        # 缺 OCP 时只让这一条命令失败。
        if args.command == "tessellate-step":
            from .geometry.occt_tessellation import (
                TessellationConfig,
                tessellate_step_file,
            )

            result = tessellate_step_file(
                args.step,
                TessellationConfig(linear_deflection=args.linear_deflection),
            )
            report = result.to_dict()
            report["step_file"] = str(args.step)
            if args.output_stl:
                save_stl(result.mesh, args.output_stl)
                report["stl_output"] = str(Path(args.output_stl).resolve())
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with atomic_path(output_path) as temporary:
                    temporary.write_text(
                        json.dumps(
                            report, ensure_ascii=False, indent=2, allow_nan=False
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                report["output"] = str(output_path.resolve())
            print(json.dumps(report, ensure_ascii=False, indent=2))
            # 有面三角化失败时返回非零：那是"这个面根本没被检测"，
            # 静默放行会让它看起来像"这个面没问题"。
            return 1 if report["warnings"] else 0

        # voxel：体素占用 + 逐层截面的辅助比对。
        if args.command == "voxel":
            from .detection.voxel_compare import compare_meshes_voxel
            from .output.voxel_report import write_voxel_json
            # 我复用分型流程的输入适配：它已经处理了 STL 直读与 STEP
            # 三角化两条路径，体素比对只看 TriangleMesh，不关心来源格式。
            from .parting_pipeline import load_mesh_for_parting

            # 先读参考件再读候选件：参考件决定包围盒与阈值尺度，
            # 它读失败就没必要再读候选件。
            reference_mesh = load_mesh_for_parting(args.reference)
            candidate_mesh = load_mesh_for_parting(args.candidate)
            result = compare_meshes_voxel(
                reference_mesh,
                candidate_mesh,
                voxel_size=args.voxel_size_mm,
                max_voxels=args.max_voxels,
            )
            json_path = write_voxel_json(
                result, Path(args.output_dir) / "voxel_compare.json"
            )
            print(
                json.dumps(
                    {
                        "status": result.status,
                        "reference_voxel_count": result.reference_voxel_count,
                        "candidate_voxel_count": result.candidate_voxel_count,
                        "missing_count": result.missing_count,
                        "extra_count": result.extra_count,
                        "iou": result.iou,
                        "missing_volume_mm3": result.missing_volume_mm3,
                        "extra_volume_mm3": result.extra_volume_mm3,
                        "voxel_size_mm": result.voxel_size_mm,
                        "artifacts": {"json": str(json_path.resolve())},
                        "warnings": result.warnings,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if result.status == "OK" else 1

        # flowchart：按真实源码结构出主流程 / 门禁 / 模块三张图。
        # 我先在内存里逐个 import 图上的模块名（与 scripts 文件路径），
        # 确认图与源码一致，再写出 PNG 与可选的 Mermaid 文本。
        # --strict 把"图过期"提升为非零退出码，方便 CI 在 PR 阶段接住。
        if args.command == "flowchart":
            from .output.flowchart import (
                write_flowchart_markdown,
                write_flowcharts,
            )

            artifacts = write_flowcharts(args.output_dir)
            if not args.no_markdown:
                write_flowchart_markdown(args.output_dir)
            print(
                json.dumps(
                    {
                        "command": "flowchart",
                        "output_dir": args.output_dir,
                        "chart_count": artifacts["chart_count"],
                        "verification": artifacts["verification"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            if args.strict and not artifacts["verification"]["consistent"]:
                return 2
            return 0

        # 走到这里就是 inspect（它是唯一没有 if 守卫的命令）。
        # 我先读配置文件，再用命令行显式给出的参数覆盖它；判断条件是
        # "is not None"，所以用户没写的参数不会把配置值顶掉。
        config = load_config(args.config)
        # 命令行覆盖配置文件的固定套路：只在参数显式给出（非 None）时才改，
        # 这样配置文件里设过的值不会被 argparse 的默认值悄悄覆盖掉。
        # 布尔开关用 is not None 而不是真值判断，否则 --flag=false 无法生效。
        if args.threshold_mm is not None:
            config.threshold_mm = args.threshold_mm
            # 改完阈值必须重跑 __post_init__：阈值之间有派生关系与合法性校验，
            # 直接赋值会留下"单个值合法但组合非法"的配置。
            config.__post_init__()
        if args.max_input_file_size_bytes is not None:
            config.max_input_file_size_bytes = args.max_input_file_size_bytes
        if args.registration_mode is not None:
            config.registration_mode = args.registration_mode
        if args.min_region_area_mm2 is not None:
            config.min_region_area_mm2 = args.min_region_area_mm2
        if args.min_triangle_count is not None:
            config.min_triangle_count = args.min_triangle_count
        if args.allow_scale is not None:
            config.allow_scale = args.allow_scale
        if args.samples_per_triangle is not None:
            config.samples_per_triangle = args.samples_per_triangle
        # 面级顶点采样：只在显式给了 --face-corner-check 时才打开。
        # 用 is not None 判断，而不是真值判断——不传是 None（保持默认关闭），
        # 传了是 True，两者必须区分开，否则配置文件里设过的值会被覆盖。
        if args.face_corner_check is not None:
            config.face_corner_check = bool(args.face_corner_check)
        if args.max_query_points is not None:
            config.max_query_points = args.max_query_points
        if args.distance_backend is not None:
            config.distance_backend = args.distance_backend
        if args.distance_candidate_k is not None:
            config.distance_candidate_k = args.distance_candidate_k
        if args.distance_chunk_size is not None:
            config.distance_chunk_size = args.distance_chunk_size
        if args.bidirectional_check is not None:
            config.bidirectional_check = args.bidirectional_check
        if args.unit_policy is not None:
            config.unit_policy = args.unit_policy
        if args.write_intermediate is not None:
            config.write_intermediate = args.write_intermediate
        if args.no_colored_mesh:
            config.write_colored_mesh = False
        # 自动阈值三件套：比例系数、残差因子、下限毫米。
        # 三者共同决定自动阈值，缺任何一个都会让自动模式失去可调性。
        if args.threshold_mode is not None:
            config.threshold_mode = args.threshold_mode
        if args.auto_threshold_scale_ratio is not None:
            config.auto_threshold_scale_ratio = (
                args.auto_threshold_scale_ratio
            )
        if args.auto_threshold_residual_factor is not None:
            config.auto_threshold_residual_factor = (
                args.auto_threshold_residual_factor
            )
        if args.auto_threshold_floor_mm is not None:
            config.auto_threshold_floor_mm = args.auto_threshold_floor_mm
        if args.direction_normal_tolerance_deg is not None:
            config.direction_normal_tolerance_deg = (
                args.direction_normal_tolerance_deg
            )
        if args.no_directional_evidence:
            config.directional_evidence_enabled = False
        config.__post_init__()
        result = inspect_pair(
            candidate_path=args.candidate,
            reference_path=args.reference,
            output_dir=args.output_dir,
            config=config,
            reference_engineering_json=args.reference_engineering_json,
            candidate_engineering_json=args.candidate_engineering_json,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        # 只有 status == "pass" 才算通过；review_required / fail 都返回 1。
        return 0 if result.status == "pass" else 1
    except (InspectorError, ValueError, OSError, json.JSONDecodeError) as exc:
        # 我在这里统一兜底：输入不存在、参数非法、JSON 损坏等都归到退出码 2，
        # 错误信息只写 stderr，保证 stdout 上永远是干净的结构化结果。
        print(f"我执行失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
