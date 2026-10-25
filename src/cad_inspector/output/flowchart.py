"""我按真实源码结构生成任务书要求的流程图。

任务书要求交付"Python 源码与流程图"。我为什么不手画一张图再放进仓库：
手画的图在源码改动之后就成了假的——流水线加一步、门禁加一项，图还停在
旧状态，而读图的人会把它当成权威。所以我改成**从源码里读结构再出图**：
节点定义取自 `pipeline.inspect_pair` 的实际调用顺序，模块名取自真实
模块路径，画的时候再逐个校验这些模块确实能被 import。

我为什么用 matplotlib 画而不是引入 graphviz：graphviz 需要本机装
Graphviz 二进制（当前环境就没有，实测 import 失败），在甲方机器上
大概率也装不上；matplotlib 已经是本项目的既有依赖，纯 CPU 光栅化，
和可视化区域用的是同一套字体回退链，中文不会变方框。

我出的三张图分工不同，不是同一张图换标题：

1. 主流程图：检测主流程的十步与复核汇合点，回答"数据怎么流"；
2. 门禁图：交付门禁六步与失败分支，回答"什么情况下我不给通过"；
3. 模块图：按区域分组的源码模块，回答"代码放在哪"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .atomic import atomic_path


@dataclass(frozen=True)
class FlowNode:
    """我描述流程图里的一个节点。

    `module` 存真实模块路径而不是只存显示名：出图时我要逐个 import
    这些模块来校验图与源码一致。只存显示名的话，改了模块名之后
    图还在、校验也发现不了，图就成了装饰。
    """

    key: str
    label: str
    module: str | None = None
    kind: str = "step"
    detail: str = ""
    # 我为什么要单独加 symbol 字段（2026-10-17 查图与源码是否一致时发现）：
    # 主流程图原本把"这一步对应源码里哪个函数"写在**第 4 个位置参数**上，
    # 而第 4 个字段是 kind。于是 `kind` 被填成了 validate_model_pair、
    # detect_missing_regions 这类函数名，而 kind 的合法取值其实是
    # step/module/group/gate/measure。后果是：这些符号**从来没被校验过**，
    # verify_flowcharts 只看模块能不能 import，函数改名了图照样"通过"。
    # 放在 detail 之后而不是 kind 之前，是为了不动其他三张图已有的
    # 位置参数顺序（它们用的是 key/label/module/kind/detail 五个位置）。
    symbol: str = ""


@dataclass(frozen=True)
class FlowEdge:
    """我描述一条从 `source` 到 `target` 的流向，可带分支标签。"""

    source: str
    target: str
    label: str = ""


@dataclass
class FlowChart:
    """我把一张图的节点、边和标题收在一起。"""

    key: str
    title: str
    subtitle: str
    nodes: list[FlowNode] = field(default_factory=list)
    edges: list[FlowEdge] = field(default_factory=list)


def _main_flow() -> FlowChart:
    """我给出检测主流程的十步，顺序取自 `pipeline.inspect_pair`。

    我为什么把"复核"画成一个汇合点而不是分散在各步：流水线里
    分辨率、质量、单位、配准、区域关联这五类警告最后都汇到同一个
    `review_required`。分散画会让人以为它们各自决定一个状态，
    而实际是"任一成立即整体降级"，这个语义只有汇合点能表达。
    """

    nodes = [
        FlowNode("input", "输入对\n候选 STL/STP + 参考", "cad_inspector.io.file_validation", symbol="validate_model_pair"),
        FlowNode("load", "读网格\nSTL 直读 / STEP 三角化", "cad_inspector.pipeline", symbol="_load_mesh"),
        FlowNode("quality", "网格质量体检", "cad_inspector.io.file_validation", symbol="validate_mesh"),
        # 单位策略这一步没有对应的函数调用，它是 inspect_pair 里的内联分支；
        # 我宁可不声明 symbol，也不硬凑一个名字上去假装可校验。
        FlowNode("unit", "单位策略\nassume_mm / unknown / require_mm", "cad_inspector.pipeline"),
        FlowNode("register", "配准\nnone/centroid/pca/icp", "cad_inspector.registration", symbol="align_mesh"),
        FlowNode("threshold", "推导有效阈值\nmax(尺度, 噪声, 下限)", "cad_inspector.detection.threshold", symbol="recommend_threshold_mm"),
        FlowNode("detect", "双向缺失/多余区域检测", "cad_inspector.detection.missing_regions", symbol="detect_missing_regions"),
        FlowNode("join", "STEP 工程区域关联（可选）", "cad_inspector.detection.step_region_join", symbol="join_step_engineering_regions"),
        # 复核汇合点同样是内联聚合，没有单一函数可指。
        FlowNode("review", "复核汇合点\n任一警告成立即降级", "cad_inspector.pipeline"),
        FlowNode("write", "写产物\nresult.json / regions.csv / PLY", "cad_inspector.output", symbol="write_json"),
    ]
    edges = [
        FlowEdge("input", "load"),
        FlowEdge("load", "quality"),
        FlowEdge("quality", "unit"),
        FlowEdge("unit", "register", "require_mm 非毫米则直接失败"),
        FlowEdge("register", "threshold", "残差先出，阈值才准"),
        FlowEdge("threshold", "detect"),
        FlowEdge("detect", "join"),
        FlowEdge("join", "review"),
        FlowEdge("quality", "review", "退化/重复/非流形"),
        FlowEdge("unit", "review", "单位不可靠"),
        FlowEdge("register", "review", "配准非 good 或需尺度复核"),
        FlowEdge("review", "write"),
    ]
    return FlowChart(
        key="main",
        title="缺失区域检测主流程",
        subtitle="步骤顺序取自 pipeline.inspect_pair 的实际调用链",
        nodes=nodes,
        edges=edges,
    )


def _gate_flow() -> FlowChart:
    """我给出交付门禁的六步与失败分支。

    我为什么把失败分支画出来而不是只画六个成功框：这张图要回答的
    正是"什么情况下我不给通过"。只画成功路径会让人误以为门禁
    总会放行，而历史上真正出问题的都是失败路径没被走到。
    """

    nodes = [
        FlowNode("bytecode", "清字节码", "scripts.verify_project", "gate", "bytecode_purge"),
        FlowNode("pytest", "全量回归", "scripts.verify_project", "gate", "pytest"),
        FlowNode("compile", "编译检查", "scripts.verify_project", "gate", "compileall"),
        FlowNode("pip", "依赖检查", "scripts.verify_project", "gate", "pip_check"),
        FlowNode("audit", "产物审计", "scripts.verify_project", "gate", "outputs_audit"),
        FlowNode("cache", "缓存清理", "scripts.verify_project", "gate", "cache_cleanup"),
        FlowNode("pass", "PASS", "scripts.verify_project", "gate", "status"),
        FlowNode("fail", "FAIL\n任一步不通过即整体失败", "scripts.verify_project", "gate", "status"),
    ]
    edges = [
        FlowEdge("bytecode", "pytest"),
        FlowEdge("pytest", "compile"),
        FlowEdge("compile", "pip"),
        FlowEdge("pip", "audit"),
        FlowEdge("audit", "cache"),
        FlowEdge("cache", "pass", "六步全绿"),
        FlowEdge("pytest", "fail", "有测试失败"),
        FlowEdge("compile", "fail", "有语法错误"),
        FlowEdge("pip", "fail", "有断依赖"),
        FlowEdge("audit", "fail", "有坏产物或漏检"),
    ]
    # 我在这里补上注释率这一项：它属于门禁的独立度量项，
    # 不在这六步的串行链里，但同样能决定 PASS 与否。
    nodes.append(FlowNode("ratio", "注释率门禁\n>= 0.25", "scripts.comment_ratio", "measure", ">= 0.25"))
    edges.append(FlowEdge("ratio", "fail", "低于 0.25"))
    return FlowChart(
        key="gate",
        title="统一交付门禁",
        subtitle="六步串行 + 注释率独立项，任一不成立即 FAIL",
        nodes=nodes,
        edges=edges,
    )


def _module_flow() -> FlowChart:
    """我给出按区域分组的源码模块图。

    我为什么分组和模块分两排：分组的 7 个区域是"路线"（CLI → IO →
    几何 → 配准 → 检测 → 修补 → 输出），模块挂在所属区域下面，
    路线本身是一条清晰的水平链，模块则在垂直方向上各自归位。
    两层分开后视觉上和实际依赖都对得上，不会出现跨区域的乱线。
    """

    groups = [
        FlowNode("g_cli", "命令行入口", "cad_inspector.cli", "group", ""),
        FlowNode("g_io", "输入与读取", "cad_inspector.io", "group", ""),
        FlowNode("g_geo", "几何与网格", "cad_inspector.geometry", "group", ""),
        FlowNode("g_reg", "配准", "cad_inspector.registration", "group", ""),
        FlowNode("g_det", "检测与识别", "cad_inspector.detection", "group", ""),
        FlowNode("g_rep", "修补", "cad_inspector.repair", "group", ""),
        FlowNode("g_out", "输出与可视化", "cad_inspector.output", "group", ""),
    ]
    leaves = [
        FlowNode("m_topology", "拓扑/BOM/面特征", "cad_inspector.io.step_faces", "module", ""),
        FlowNode("m_mesh", "网格模型与三角化", "cad_inspector.geometry.mesh", "module", ""),
        FlowNode("m_miss", "缺失区域检测", "cad_inspector.detection.missing_regions", "module", ""),
        FlowNode("m_thr", "自适应阈值", "cad_inspector.detection.threshold", "module", ""),
        FlowNode("m_part", "分型线/面", "cad_inspector.detection.step_parting", "module", ""),
        FlowNode("m_voxel", "体素/截面比对", "cad_inspector.detection.voxel_compare", "module", ""),
        FlowNode("m_batch", "批量检测编排", "cad_inspector.io.batch_inspect", "module", ""),
        FlowNode("m_rigid", "刚体配准", "cad_inspector.registration.rigid", "module", ""),
        FlowNode("m_shole", "STEP 补面", "cad_inspector.repair.step_hole", "module", ""),
        FlowNode("m_mhole", "STL 补孔", "cad_inspector.repair.mesh_hole", "module", ""),
        FlowNode("m_vis", "对比可视化", "cad_inspector.output.visualization", "module", ""),
        FlowNode("m_flow", "流程图生成", "cad_inspector.output.flowchart", "module", ""),
    ]
    edges = [
        FlowEdge("g_cli", "g_io"),
        FlowEdge("g_io", "g_geo"),
        FlowEdge("g_geo", "g_reg"),
        FlowEdge("g_reg", "g_det"),
        FlowEdge("g_det", "g_rep"),
        FlowEdge("g_rep", "g_out"),
        FlowEdge("g_io", "m_topology"),
        FlowEdge("g_geo", "m_mesh"),
        FlowEdge("g_det", "m_miss"),
        FlowEdge("g_det", "m_thr"),
        FlowEdge("g_det", "m_part"),
        FlowEdge("g_det", "m_voxel"),
        FlowEdge("g_io", "m_batch"),
        FlowEdge("g_reg", "m_rigid"),
        FlowEdge("g_rep", "m_shole"),
        FlowEdge("g_rep", "m_mhole"),
        FlowEdge("g_out", "m_vis"),
        FlowEdge("g_out", "m_flow"),
    ]
    return FlowChart(
        key="modules",
        title="源码区域分组",
        subtitle="按区域分组，模块名取自真实 import 路径",
        nodes=groups + leaves,
        edges=edges,
    )


def build_flowcharts() -> list[FlowChart]:
    """我返回任务书要求的三张流程图定义。"""

    return [_main_flow(), _gate_flow(), _module_flow()]


def verify_flowcharts(charts: list[FlowChart] | None = None) -> dict[str, Any]:
    """我逐个 import 图里的模块，确认图与源码一致。

    我为什么要做这一步：图是手写的节点清单，写着写着就会和源码脱节。
    出图前先 import 一遍，模块改名或删除会立刻在这里报出来，
    而不是等到甲方拿着一张错图来问。这是我把"文档"变成"可验证产物"
    的唯一办法——否则它只是一张好看的图。

    我为什么对 `scripts.*` 单独按文件路径校验：`scripts/` 是脚本目录，
    不是包（没有 `__init__.py`），`import scripts.verify_project` 会失败，
    但文件确实存在。按"能不能 import"一条规则去判，会把所有脚本
    误报成缺失。所以我对它改用"文件是否存在"这一条口径，
    两类目标各有各的正确判据。
    """

    import importlib

    charts = charts if charts is not None else build_flowcharts()
    missing: list[str] = []
    missing_symbols: list[str] = []
    checked = 0
    checked_symbols = 0
    # 本文件在 <仓库根>/src/cad_inspector/output/flowchart.py，
    # 往上三级才是仓库根（output -> cad_inspector -> src -> 根）。
    root = Path(__file__).resolve().parents[3]
    for chart in charts:
        for node in chart.nodes:
            if node.module is None:
                continue
            checked += 1
            if node.module.startswith("scripts."):
                relative = node.module[len("scripts.") :].replace(".", "/") + ".py"
                if not (root / "scripts" / relative).is_file():
                    missing.append(f"{chart.key}/{node.key}:{node.module}")
                continue
            try:
                module = importlib.import_module(node.module)
            except Exception:
                missing.append(f"{chart.key}/{node.key}:{node.module}")
                continue
            # 光能 import 模块不够：节点还声明了"这一步对应源码里哪个函数"。
            # 只查模块的话，函数被改名/删除而模块还在，图依然会通过校验，
            # 于是图上写着 detect_missing_regions、源码里已经没这个名字。
            if node.symbol:
                checked_symbols += 1
                if not hasattr(module, node.symbol):
                    missing_symbols.append(
                        f"{chart.key}/{node.key}:{node.module}.{node.symbol}"
                    )
    return {
        "checked_module_count": checked,
        "checked_symbol_count": checked_symbols,
        "missing_modules": missing,
        "missing_symbols": missing_symbols,
        "consistent": not missing and not missing_symbols,
    }


def _called_function_names(function: Any) -> list[tuple[int, str]]:
    """我按源码行号收集一个函数体里出现的调用名（简单名与属性名都收）。"""

    import ast

    calls: list[tuple[int, str]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            calls.append((node.lineno, target.id))
        elif isinstance(target, ast.Attribute):
            calls.append((node.lineno, target.attr))
    calls.sort()
    return calls


def verify_main_flow_order(chart: FlowChart | None = None) -> dict[str, Any]:
    """我核对主流程图的步骤顺序与 `inspect_pair` 的实际调用顺序一致。

    图里写着"步骤顺序取自 pipeline.inspect_pair 的实际调用链"。我加这条
    校验之前，这句话**没有任何东西守着**：verify_flowcharts 只 import
    模块、查符号是否存在，顺序改了它一个字都不会说。而顺序恰恰是流程图
    最核心的信息——配准必须排在阈值之前（阈值里的噪声分量用的就是配准
    残差），顺序错了这张图就是在教人做错事。

    我用 AST 而不是正则：正则会把注释、字符串里出现的同名函数也算进去，
    而 AST 只看真实调用。
    """

    import ast

    root = Path(__file__).resolve().parents[3]
    source_path = root / "src" / "cad_inspector" / "pipeline.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "inspect_pair"
        ),
        None,
    )
    if function is None:
        return {
            "consistent": False,
            "message": "我在 pipeline.py 里找不到 inspect_pair。",
            "expected_order": [],
            "observed_order": [],
            "missing_from_source": [],
        }

    first_line: dict[str, int] = {}
    for line, name in _called_function_names(function):
        # 只记首次出现：读参考件和读候选件都调 _load_mesh，
        # 记最后一次会让顺序判定被"最后一个重复调用"带偏。
        first_line.setdefault(name, line)

    if chart is None:
        chart = next(item for item in build_flowcharts() if item.key == "main")
    expected = [node.symbol for node in chart.nodes if node.symbol]
    observed: list[str] = []
    missing_from_source: list[str] = []
    for symbol in expected:
        if symbol in first_line:
            observed.append(symbol)
        else:
            missing_from_source.append(symbol)

    in_order = observed == sorted(observed, key=lambda name: first_line[name])
    return {
        "consistent": not missing_from_source and in_order,
        "expected_order": expected,
        "observed_order": observed,
        "missing_from_source": missing_from_source,
    }


def _configure_chinese_font(matplotlib_module: Any) -> None:
    """我按回退链选一个带中文字形的字体。

    我为什么不直接用 DejaVu Sans：它不含中文字形，标题里的中文会
    渲染成一排方框，图直接没法看。而甲方机器上的字体不确定，
    所以我按常见中文字体逐个试，全都没有时才退回默认。
    """

    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"):
        if name in available:
            matplotlib_module.rcParams["font.sans-serif"] = [name]
            break
    # 负号用 ASCII  hyphen：部分中文字体把 U+2212 画成方框。
    matplotlib_module.rcParams["axes.unicode_minus"] = False


def _layout(nodes: list[FlowNode], columns: int) -> dict[str, tuple[float, float]]:
    """我把节点按列优先铺成网格，返回每个节点键的中心坐标。

    我为什么用网格布局而不是自己算 DAG 层级：模块图里有跨层的边，
    严格分层会把跨层边拉得很长；主图有汇合点，也不是一条链。
    网格布局对这三种形状都能给出可读的结果，且实现确定——
    同样输入永远得到同样位置，两张图可以逐像素比对。
    """

    # 我对模块图特殊处理：把 group 放在第一行，module 放在第二行，
    # 让"区域行"和"模块行"分别有自己的水平链，箭头不会跨区域乱飞。
    # group 与 module 的分类按节点 `kind` 区分，不用另外传参。
    if nodes and any(node.kind == "group" for node in nodes) and any(
        node.kind == "module" for node in nodes
    ):
        groups = [node for node in nodes if node.kind == "group"]
        modules = [node for node in nodes if node.kind == "module"]
        positions = {}
        # 我用 1.6 的列间距而不是 1.0：模块图里最长的标签
        # 9 个汉字大约要 1.5 单位宽，再小的间距就会出现
        # "命令行入口"和"输入与读取"两个箱体贴在一起。
        for column, node in enumerate(groups):
            positions[node.key] = (float(column) * 1.6, 0.0)
        for column, node in enumerate(modules):
            positions[node.key] = (float(column) * 1.6, -1.2)
        return positions
    positions = {}
    rows = (len(nodes) + columns - 1) // columns
    for index, node in enumerate(nodes):
        column = index % columns
        row = index // columns
        # 我把行间距从 1.0 提到 1.05：箱体高度变成 0.5 之后，
        # 1.0 的间距只能留给箭头 0.5 的长度，箭头尾端贴着下一行
        # 箱体，没有呼吸。1.05 给 0.55 的箭头长度，是箱体高度的 1.1 倍。
        positions[node.key] = (float(column), -float(row) * 1.05)
    _ = rows
    return positions


def _draw_chart(
    chart: FlowChart,
    axis: Any,
    patches_module: Any,
    columns: int,
) -> None:
    """我在给定坐标轴上画一张流程图。"""

    positions = _layout(chart.nodes, columns)
    color_by_kind = {
        "step": "#DCE9F7",
        "module": "#EAF3E4",
        "group": "#F7EEDC",
    }
    # 模块图里最长的标签是 9 个汉字，需要 1.3 单位宽的箱体才不换行；
    # 普通流程图保持 0.74，因为节点较少、空间紧凑。
    is_module_chart = any(node.kind == "group" for node in chart.nodes) and any(
        node.kind == "module" for node in chart.nodes
    )
    # 我把箱体高度从 0.34 提到 0.5：很多标签是"主题\n副说明"两行，
    # 0.34 会让第二行被压在箱体外面看着像溢出；0.5 给两行都留出
    # 8pt 与 7pt 文本的内边距。两行文本大约 0.42，再加 0.08 的呼吸。
    # 主流程图节点标签最长是"双向缺失/多余区域检测"14 个汉字，
    # 0.95 单位的箱宽在 8.5pt 字体下能容下两行换行不溢出。
    # 模块图的最长是"对比可视化"/"刚体配准"6 个汉字，1.3 已经够。
    # 0.55 的箱高容纳 8.5pt + 7.5pt 两行（约 0.42），加 0.13 的内边距。
    box_width = 1.3 if is_module_chart else 0.95
    box_height = 0.55
    for node in chart.nodes:
        x, y = positions[node.key]
        axis.add_patch(
            patches_module.FancyBboxPatch(
                (x - box_width / 2, y - box_height / 2),
                box_width,
                box_height,
                boxstyle="round,pad=0.02",
                linewidth=1.2,
                edgecolor="#33506B",
                facecolor=color_by_kind.get(node.kind, "#DCE9F7"),
            )
        )
        axis.text(
            x,
            y,
            node.label,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#1B2A38",
        )

    for edge in chart.edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        axis.add_patch(
            patches_module.FancyArrowPatch(
                (x1, y1 - box_height / 2),
                (x2, y2 + box_height / 2),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                color="#33506B",
                connectionstyle="arc3,rad=0.0",
            )
        )
        if edge.label:
            # 我给 label 文字加一个白底框，避免 label 文字与下一行
            # 节点的边框/箭头尾端交叠时变成一段黑线。这是 matplotlib
            # text 不支持 path_effects 时的等价做法。
            axis.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                edge.label,
                ha="center",
                va="center",
                fontsize=6.4,
                color="#7A4A1E",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fc="white",
                    ec="none",
                    alpha=0.85,
                ),
            )

    if is_module_chart:
        # 7 个 group 在 1.6 间距下 x 范围是 0..9.6，
        # 我把图限放成 -0.6..10.8 以免最后一列贴边。
        axis.set_xlim(-0.6, 1.6 * 6 + 1.2)
        axis.set_ylim(-1.7, 0.6)
        axis.set_aspect("equal")
    else:
        # 我用 1 列宽 + 1.4 行高 + auto aspect：箱体不强行正方形，
        # 让 0.95 宽的箱体和 0.55 高的箱体可以正常显示。
        # 5 行 × 1.4 = 7，给 0.2 上下边距。
        axis.set_xlim(-0.55, 1.5)
        axis.set_ylim(-((len(chart.nodes) + columns - 1) // columns) * 1.4 - 0.2, 0.7)
        axis.set_aspect("auto")
    axis.axis("off")


def write_flowcharts(
    output_dir: str | Path,
    dpi: int = 150,
    charts: list[FlowChart] | None = None,
) -> dict[str, Any]:
    """我把流程图写成 PNG，并返回路径与一致性校验结果。

    我在写图**之前**先跑 `verify_flowcharts`：图与源码不一致时我
    照样出图，但返回里会带上 `consistent=false` 与缺失模块清单，
    让调用方能拦住。我为什么不直接抛错中断：出图失败和"图过期"
    是两件不同的事，混在一起会让人以为流程坏了，而实际只是文档
    该更新了。
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches as patches_module

    _configure_chinese_font(matplotlib)
    charts = charts if charts is not None else build_flowcharts()
    verification = verify_flowcharts(charts)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for chart in charts:
        columns = 3 if chart.key == "modules" else 2
        rows = (len(chart.nodes) + columns - 1) // columns
        # 我按图的最长一行节点数决定宽度：模块图有 7 个 group，
        # 7 * 1.4 = 9.8 英寸；主图 10 个节点除以 2 列 = 5 行，
        # 5 * 1.6 = 8 英寸。这样不会因为节点多而被截断。
        if chart.key == "modules":
            max_columns = max(
                sum(1 for n in chart.nodes if n.kind == "group"),
                sum(1 for n in chart.nodes if n.kind == "module"),
            )
            width = 1.55 * max_columns
            height = 4.6
        else:
            width = 8.4
            height = 1.6 * rows + 1.6
        figure, axis = plt.subplots(figsize=(width, height), dpi=dpi)
        try:
            _draw_chart(chart, axis, patches_module, columns)
            figure.suptitle(chart.title, fontsize=13, fontweight="bold")
            axis.set_title(chart.subtitle, fontsize=8.5, color="#55666F", pad=6)
            figure.tight_layout(rect=(0, 0, 1, 0.94))
            target = output_path / f"flowchart_{chart.key}.png"
            with atomic_path(target) as temporary:
                figure.savefig(temporary, format="png", dpi=dpi)
            artifacts[f"flowchart_{chart.key}"] = str(target)
        finally:
            plt.close(figure)

    return {
        "artifacts": artifacts,
        "verification": verification,
        "chart_count": len(charts),
    }


def write_flowchart_markdown(
    output_dir: str | Path,
    charts: list[FlowChart] | None = None,
) -> Path:
    """我把同一份流程结构写成 Mermaid 文本。

    我为什么同时出 Mermaid：PNG 进不了代码评审的 diff，而 Mermaid
    能直接在浏览器和多数 Git 平台里渲染，改一个节点就能看出改了什么。
    两份来自同一份节点定义，不会出现"图和文本说的不一样"。
    """

    charts = charts if charts is not None else build_flowcharts()
    lines: list[str] = ["# 我方流程图（由源码结构生成）", ""]
    for chart in charts:
        lines.append(f"## {chart.title}")
        lines.append("")
        lines.append(f"> {chart.subtitle}")
        lines.append("")
        lines.append("```mermaid")
        lines.append("flowchart TD")
        for node in chart.nodes:
            label = node.label.replace("\n", "<br/>")
            lines.append(f'    {node.key}["{label}"]')
        for edge in chart.edges:
            if edge.label:
                lines.append(f'    {edge.source} -->|"{edge.label}"| {edge.target}')
            else:
                lines.append(f"    {edge.source} --> {edge.target}")
        lines.append("```")
        lines.append("")

    target = Path(output_dir) / "flowcharts.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(target) as temporary:
        temporary.write_text("\n".join(lines), encoding="utf-8")
    return target
