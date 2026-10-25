"""我把 STEP 模具工程特征候选写成 JSON 和 Excel 兼容 CSV。

算法原理：JSON 是完整证据的权威载体，CSV 是给人扫读和 Excel 打开的摘要视图。
两者容量与用途不同，所以我对 CSV 单元格做递归压缩（限深度、限条目、限字符串长、
超长则替换为"截断说明 + 完整证据在 JSON"），保证 CSV 不会因为单个证据字段过大而
撑爆 Excel 单元格上限（32767 字符）。JSON 侧不做截断，永远保留完整证据。

写出统一走 ``atomic_path`` 原子替换，避免中断留下半个文件；CSV 用 utf-8-sig 带 BOM，
确保中文列名在 Excel 里不乱码。
"""

from __future__ import annotations

import csv
import json
from math import isfinite
from pathlib import Path

from ..domain.step_engineering import StepEngineeringReport
from .atomic import atomic_path


def _limited_list(value: object, limit: int = 64) -> list:
    """我把 CSV 摘要列表限制在可扫描长度，完整内容仍保留在 JSON。

    为什么 CSV 要截断而 JSON 不截断：两者的用途不同。JSON 是归档
    与程序再处理的载体，必须完整；CSV 是给人扫的，一个单元格里
    塞进上万个编号只会让人放弃阅读。

    非列表输入返回空列表而不是抛错：证据字段在部分路径下可能缺失
    或类型不同，CSV 摘要的取值本来就是"尽力而为"，
    不该因为一个字段缺失让整份报告写不出来。
    """

    if not isinstance(value, list):
        return []
    return value[:limit]


def _compact_evidence_for_csv(
    value: object,
    *,
    depth: int = 0,
    max_depth: int = 4,
    max_items: int = 32,
    max_string: int = 512,
) -> object:
    """我递归限制 CSV 证据长度，完整证据仍由 JSON 保存。

    我为什么不直接截断成字符串：证据是嵌套结构，粗暴地截前 N 个字符会
    得到一个不完整的 JSON 片段，既不能解析也无法判断"到底截了多少"。
    所以我做**结构化裁剪**——按深度和元素数逐层限制，并在被截掉的
    位置显式写入 `<truncated N items>`、`_truncated_key_count`
    这类标记，让人一眼看出"这里被截了、截了多少"。

    两个细节：非有限浮点（inf/nan）必须转成字符串，因为 json.dumps
    默认会写出裸的 Infinity/NaN，那不是合法 JSON，下游严格解析会直接报错；
    完整证据始终由 JSON 文件保存，CSV 裁掉的内容不会真的丢失。
    """

    # 到达深度上限时，容器直接替换为占位串，标量只做字符串截断。
    if depth >= max_depth:
        if isinstance(value, (dict, list, tuple, set)):
            return "<evidence depth limit>"
        if isinstance(value, str):
            return (
                value[:max_string] + "..."
                if len(value) > max_string
                else value
            )
        return value
    # 字典：保留前 max_items 个键，超出部分只记数量，避免 CSV 无限膨胀。
    if isinstance(value, dict):
        compact: dict[str, object] = {}
        items = list(value.items())
        for key, item in items[:max_items]:
            compact[str(key)] = _compact_evidence_for_csv(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
            )
        if len(items) > max_items:
            compact["_truncated_key_count"] = len(items) - max_items
        return compact
    # 列表/元组：保留前 max_items 个元素，超出部分追加一行截断说明。
    if isinstance(value, (list, tuple)):
        items = list(value)
        compact_list = [
            _compact_evidence_for_csv(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
            )
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            compact_list.append(
                f"<truncated {len(items) - max_items} items>"
            )
        return compact_list
    if isinstance(value, str):
        return value[:max_string] + "..." if len(value) > max_string else value
    if isinstance(value, float) and not isfinite(value):
        return str(value)
    return value


def _json_for_csv(value: object) -> str:
    """我把证据压缩成不超过 30000 字符的 CSV 单元格。

    30000 这个数不是随手取的：Excel 单单元格上限是 32767 字符，
    超过会静默截断甚至让整行打不开。我留出约 2700 字符余量，
    覆盖后续可能追加的标记字段。

    两级降级而不是一步切平：先做**结构化裁剪**（按深度和条目数保留
    结构），只有结构化裁剪后仍超长，才退化成"截断说明 + 顶层键名"。
    直接切平会让所有大证据都变成没有信息量的占位符，
    而多数情况下结构化裁剪后是能放进单元格的。

    降级结果里保留 truncated_from_characters 和 top_level_key_count：
    让人能判断"原来有多大、有多少键"，而不是只看到一个
    "已截断"就把证据当作不存在。
    """

    compact = _compact_evidence_for_csv(value)
    text = json.dumps(
        compact,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    # 30000 字符上限低于 Excel 的 32767，留出安全余量。
    if len(text) <= 30000:
        return text
    # 仍超长则退化为"截断说明 + 顶层键名"，并指向 JSON 里的完整证据。
    fallback: dict[str, object] = {
        "_truncated": True,
        "truncated_from_characters": len(text),
        "full_evidence_location": "JSON",
    }
    if isinstance(value, dict):
        fallback["top_level_keys"] = [str(key) for key in list(value)[:32]]
        fallback["top_level_key_count"] = len(value)
    elif isinstance(value, list):
        fallback["item_count"] = len(value)
    return json.dumps(
        fallback,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )


def _compact_wire_selection_evidence(
    region: object,
) -> dict[str, object]:
    """我把 Wire 选择证据压缩成 CSV 可读摘要。

    我只保留"甲方复核时真正要看的字段"：状态、选中了哪个分量、周长与
    投影面积、以及两个冲突标记（projection_conflict、
    loop_relation_conflict）。前者回答"选了谁"，后者回答"这次选择能不能信"。

    候选列表最多保留 64 条：CSV 单元格要能一眼扫完，几百条候选会把
    表格撑爆；完整候选仍在 JSON 里。我同时保留 candidate_count，
    所以"截掉了多少条"在 CSV 里也是可见的，不会让人误以为只有 64 条。
    """

    evidence = getattr(region, "boundary_wire_selection_evidence", {}) or {}
    candidates = evidence.get("candidates", [])
    compact_candidates = []
    if isinstance(candidates, list):
        # 候选最多取 64 条：真实零件上边界 Wire 候选可能有几百条，
        # 全量嵌进 CSV 会让单元格大到打不开。
        # 64 条足够看出"选得对不对"，总量由下面的 candidate_count 给出。
        for candidate in candidates[:64]:
            # 非字典的条目跳过：证据里的候选格式可能被外部改动过，
            # 不校验会让 .get 在非字典上抛异常。
            if not isinstance(candidate, dict):
                continue
            compact_candidates.append(
                {
                    # 每条只留这八个字段：完整候选还带边号列表等重字段，
                    # 在 CSV 单元格里全量展开没有可读性可言。
                    "rank": candidate.get("rank"),
                    "component_index": candidate.get("component_index"),
                    "edge_count": candidate.get("edge_count"),
                    # 闭合与周长完整是两个独立字段：
                    # 闭合是拓扑性质，周长完整是量测性质，
                    # 一条 Wire 可以闭合但周长量不全。
                    "closed": candidate.get("closed"),
                    "perimeter_length_mm": candidate.get("perimeter_length_mm"),
                    "perimeter_complete": candidate.get("perimeter_complete"),
                    "projected_area_mm2": candidate.get("projected_area_mm2"),
                    "projection_status": candidate.get("projection_status"),
                }
            )
    return {
        "status": evidence.get("status"),
        # 候选总数单独给：上面只留了 64 条明细，
        # 没有总数就无法知道被截断了多少。
        "candidate_count": evidence.get("candidate_count"),
        "primary_component_index": evidence.get("primary_component_index"),
        "primary_edge_ids": _limited_list(evidence.get("primary_edge_ids")),
        "primary_perimeter_length_mm": evidence.get(
            "primary_perimeter_length_mm"
        ),
        "primary_projected_area_mm2": evidence.get(
            "primary_projected_area_mm2"
        ),
        # 投影状态与冲突标记一起给：状态说明"投影成没成"，
        # 冲突说明"多个来源给的结果是否不一致"。
        # 只有状态的话，冲突的那次会被当成一次正常的成功投影。
        "primary_projection_status": evidence.get(
            "primary_projection_status"
        ),
        "projection_conflict": evidence.get("projection_conflict"),
        "loop_relation_status": evidence.get("loop_relation_status"),
        "primary_loop_role": evidence.get("primary_loop_role"),
        "loop_relation_conflict": evidence.get("loop_relation_conflict"),
        # 选择指标保留原始值：它决定"凭什么选了这条 Wire"，
        # 丢掉它就无法复核选择逻辑。
        "selection_metric": evidence.get("selection_metric"),
        "unresolved_edge_ids": _limited_list(
            evidence.get("unresolved_edge_ids")
        ),
        "candidates": compact_candidates,
        "evidence_truncated_in_csv": True,
        "full_evidence_location": "JSON boundary_wire_selection_evidence",
    }


def _compact_boundary_topology_evidence(
    region: object,
) -> dict[str, object]:
    """我把边界拓扑证据压缩成 CSV 可读摘要。

    我先 dict() 拷贝再 pop，而不是就地改原对象：证据字典是
    region 上的共享引用，就地删除 wire_selection 会连 JSON 报告
    里的数据一起改掉。CSV 与 JSON 必须各走各的副本。

    wire_selection 被摘出去单独压缩：它本身是嵌套很深的结构，
    直接留在 components 旁边会让一个 CSV 单元格深不见底；
    单独压平成 wire_selection_summary 后既能扫读又保留关键信息。

    evidence_truncated_in_csv 与 full_evidence_location 是显式标记，
    不是可选装饰：看 CSV 的人必须知道"这里被截断了、
    完整版在哪"，否则会把截断后的数量当成真实数量去汇报。
    """

    evidence = dict(getattr(region, "boundary_topology_evidence", {}) or {})
    evidence.pop("wire_selection", None)
    components = evidence.get("components", [])
    compact_components = []
    if isinstance(components, list):
        for component in components[:64]:
            if not isinstance(component, dict):
                continue
            compact_components.append(
                {
                    "edge_ids": _limited_list(component.get("edge_ids")),
                    "vertex_ids": _limited_list(component.get("vertex_ids")),
                    "closed": component.get("closed"),
                    "perimeter_length_mm": component.get("perimeter_length_mm"),
                    "perimeter_complete": component.get("perimeter_complete"),
                    "projected_area_mm2": component.get("projected_area_mm2"),
                    "projection_status": component.get("projection_status"),
                }
            )
    evidence["components"] = compact_components
    evidence["wire_selection_summary"] = _compact_wire_selection_evidence(
        region
    )
    evidence["evidence_truncated_in_csv"] = True
    evidence["full_evidence_location"] = (
        "JSON boundary_topology_evidence and boundary_wire_selection_evidence"
    )
    return evidence


def _compact_loop_relation_evidence(
    region: object,
) -> dict[str, object]:
    """我把环关系证据压缩成 CSV 可读摘要。

    用 getattr 取值而不是直接访问属性：这个函数同时服务于
    StepEngineeringRegion 和上游的 StepFeatureRegion，
    两者的字段集并不完全一致。用 getattr 带默认值可以让
    缺字段时优雅退化，而不用为每个类型各写一份。

    每个列表都同时输出 count 和（截断后的）indices：
    count 是真实的完整数量，用于判断是否超限；
    indices 供人查看前若干个。只给截断后的列表会让人误以为
    "总共就这几个"。

    status 优先取 region 上的属性，取不到才退回 evidence 里的值：
    region 上是最终判定，evidence 里是计算中间量，
    两者可能因后处理而不一致，最终判定优先。
    """

    evidence = (
        getattr(region, "primary_boundary_loop_relation_evidence", {}) or {}
    )
    outer_indices = evidence.get("outer_candidate_indices", [])
    hole_indices = evidence.get("hole_candidate_indices", [])
    intersecting_pairs = evidence.get("intersecting_pairs", [])
    touching_pairs = evidence.get("touching_pairs", [])
    return {
        # 状态优先取区域上的字段，取不到才回退到证据里：
        # 区域字段是这一轮判定后的结论，证据里的是原始计算值，
        # 两者在复核流程里可能已经被人工修正过。
        "status": getattr(
            region,
            "primary_boundary_loop_relation_status",
            evidence.get("status"),
        ),
        "loop_count": evidence.get("loop_count"),
        # 每个列表字段都配一个计数：列表本身会被 _limited_list 截断，
        # 只留列表的话"总共有多少"这个信息就丢了。
        # 计数才是做统计和门禁判断要用的数。
        "outer_candidate_count": (
            len(outer_indices) if isinstance(outer_indices, list) else 0
        ),
        "outer_candidate_indices": _limited_list(outer_indices),
        "hole_candidate_count": (
            len(hole_indices) if isinstance(hole_indices, list) else 0
        ),
        "hole_candidate_indices": _limited_list(hole_indices),
        "root_component_indices": _limited_list(
            evidence.get("root_component_indices")
        ),
        # 相交与相切分开统计：相交是几何上不成立的坏拓扑，
        # 相切是正常但需要注意的接触，混在一起无法判断严重性。
        "intersecting_pair_count": (
            len(intersecting_pairs)
            if isinstance(intersecting_pairs, list)
            else 0
        ),
        "intersecting_pairs": _limited_list(intersecting_pairs),
        "touching_pair_count": (
            len(touching_pairs) if isinstance(touching_pairs, list) else 0
        ),
        "touching_pairs": _limited_list(touching_pairs),
        "loop_relation_conflict": evidence.get("loop_relation_conflict"),
        "primary_loop_role": getattr(
            region,
            "primary_boundary_loop_role",
            evidence.get("primary_loop_role"),
        ),
        "primary_loop_parent_component_index": evidence.get(
            "primary_loop_parent_component_index"
        ),
        "evidence_truncated_in_csv": True,
        "full_evidence_location": (
            "JSON primary_boundary_loop_relation_evidence"
        ),
    }


def write_step_engineering_report(
    report: StepEngineeringReport,
    output_dir: str | Path,
) -> dict[str, str]:
    """我输出候选证据和拔模方向明细。

    我为什么同时写 JSON 和 CSV：两类读者的需求不同，不能互相替代。
      - JSON 是**完整证据载体**，不做任何截断，下游程序能逐条回溯
        每个候选的来源面、边、评分和证据字典；
      - CSV 是给人看的复核表，一列一项、一行一条候选，甲方能直接用
        Excel 打开、筛选和排序，也能直接对照"哪几条需要人工确认"。

    CSV 用 utf-8-sig 而不是 utf-8：Excel 默认按本地编码打开 CSV，
    不带 BOM 会让中文列名（"候选类型""人工复核警告"）变成乱码，
    BOM 是让 Excel 正确识别 UTF-8 的最小代价。

    四个文件共用同一零件文件名主干，便于按零件归档；全部走 atomic_path，
    避免中断留下半截文件骗过下游的"文件存在即有效"检查。
    """

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    # 四个输出文件共用同一零件文件名主干，便于按零件归档与检索。
    stem = Path(report.file_path).stem
    json_path = target_dir / f"{stem}.engineering.json"
    csv_path = target_dir / f"{stem}.engineering.csv"
    draft_csv_path = target_dir / f"{stem}.draft.csv"
    regions_csv_path = target_dir / f"{stem}.regions.csv"
    # JSON 是完整证据载体，不做任何截断，确保下游可追溯每条候选的来源。
    with atomic_path(json_path) as temporary:
        temporary.write_text(
            json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    # 候选 CSV：一行一个工程候选，证据字段走 CSV 压缩。
    with atomic_path(csv_path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "序号",
                    "候选类型",
                    "Face编号",
                    "Edge编号",
                    "评分",
                    "证据",
                    "人工复核警告",
                ],
            )
            writer.writeheader()
            for candidate in report.candidates:
                writer.writerow(
                    {
                        "序号": candidate.candidate_id,
                        "候选类型": candidate.kind,
                        "Face编号": " | ".join(map(str, candidate.face_ids)),
                        "Edge编号": " | ".join(map(str, candidate.edge_ids)),
                        "评分": candidate.score,
                        "证据": _json_for_csv(candidate.evidence),
                        "人工复核警告": " | ".join(candidate.warnings),
                    }
                )
    # 拔模 CSV：一行一个 Face 的拔模方向证据。
    with atomic_path(draft_csv_path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "Face编号",
                    "拔模方向",
                    "法向与拔模方向夹角_deg",
                    "面与拔模方向夹角_deg",
                    "分类",
                    "评分",
                    "人工复核警告",
                ],
            )
            writer.writeheader()
            for item in report.draft_evidence:
                writer.writerow(
                    {
                        "Face编号": item.face_id,
                        "拔模方向": " | ".join(map(str, item.direction_xyz)),
                        "法向与拔模方向夹角_deg": item.normal_vs_direction_angle_deg,
                        "面与拔模方向夹角_deg": item.plane_vs_direction_angle_deg,
                        "分类": item.classification,
                        "评分": item.score,
                        "人工复核警告": " | ".join(item.warnings),
                    }
                )
    # 区域 CSV：一行一个区域的角色、类别、分型、边界拓扑和复核队列证据。
    with atomic_path(regions_csv_path) as temporary:
        # 列顺序不是随意排的，按"先身份、再量测、再边界、再判定、再证据"
        # 五段组织，让甲方在 Excel 里从左到右读就是一条完整的判断链：
        # 编号与签名定位区域 -> 面积包围盒给量纲 -> 边界系列回答轮廓是否闭合
        # -> 状态与置信度给结论 -> 证据与警告解释结论从哪来。
        # 列名一旦发布就会进入甲方的筛选器和统计脚本，所以只增不改名。
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                # 列顺序是刻意按复核流程排的，不是随手列的：
                #   标识区（编号、签名、角色）→ 定位用；
                #   几何区（面积、包围盒、法向）→ 判断尺度与朝向；
                #   边界区（边、环、Wire、投影）→ 判断边界是否可靠；
                #   结论区（语义、类别、复核原因、优先级）→ 下结论。
                # 固定顺序保证不同批次产出的 CSV 能直接横向比对。
                fieldnames=[
                    "区域编号",
                    # 签名紧跟编号：编号在同一次运行内唯一但跨运行不稳定，
                    # 签名才是跨批次追踪同一个区域的键，两者必须相邻。
                    "区域几何签名",
                    "源区域角色",
                    "区域工程状态",
                    "区域角色候选",
                    "Face编号",
                    "邻接区域编号",
                    # 从"面积"到"包围盒尺寸"这一段是几何量测区：
                    # 它们描述这块区域本身有多大、朝哪、在哪，
                    # 与下面的边界区和结论区都是独立的维度。
                    "面积_mm2",
                    "边界Edge数",
                    "边界Edge编号",
                    # 连通分量排在环数量之前：先看边界分成几块，
                    # 再看每块里有没有环，这个顺序与判断顺序一致。
                    "边界Edge连通分量",
                    "边界环数量",
                    "闭合边界环候选数",
                    "边界拓扑状态",
                    "边界拓扑证据",
                    # Wire 区三段（主 Wire 的编号、候选数、选择状态与证据）：
                    # 一块区域可能解析出多条候选 Wire，
                    # 这一段回答"选了哪条、凭什么选它"。
                    "主边界Wire Edge",
                    "边界Wire候选数",
                    "边界Wire选择状态",
                    "边界Wire选择证据",
                    # 周长与投影面积：Wire 自身的尺度。
                    # 周长是长度量、投影面积是面积量，
                    # 两者一起才能判断"这条 Wire 围出的东西有多大"。
                    "边界周长_mm",
                    "主边界Wire周长_mm",
                    "主边界Wire投影面积_mm2",
                    "主边界Wire投影状态",
                    "主边界Wire投影证据",
                    # 环角色与环关系：回答"这条 Wire 是外轮廓还是内孔"
                    # 以及"它和其他环是什么关系"。
                    "主边界Wire环角色",
                    "主边界Wire环关系状态",
                    "主边界Wire环关系证据",
                    "中心_xyz_mm",
                    "平均法向_xyz",
                    "包围盒最小_xyz_mm",
                    "包围盒最大_xyz_mm",
                    "包围盒尺寸_xyz_mm",
                    # 从"语义状态"往下是结论区：
                    # 候选逐条展开，一条区域×候选一行，
                    # 所以这里的字段对每个候选都会重复一遍。
                    "语义状态",
                    "候选编号",
                    "候选类型",
                    "类别候选",
                    "类别置信度",
                    "类别是否需要复核",
                    # 复核优先级紧跟原因代码之前：先看这条有多急，
                    # 再看为什么急，比反过来更符合处理顺序。
                    "复核优先级",
                    "复核原因代码",
                    "分型区域候选",
                    "分型区域置信度",
                    "分型区域是否需要复核",
                    "分型区域证据",
                    "类别证据",
                    "证据",
                    "人工复核警告",
                ],
            )
            writer.writeheader()
            # 写行时把列表类字段统一转成文本：
            #   编号列表用 " | " 拼接 —— 甲方要在 Excel 里肉眼扫编号，
            #     用 json 数组会带引号括号，不好读；
            #   嵌套结构用 _json_for_csv —— 证据字典是给程序回读的，
            #     必须保持可解析，拼接会丢结构。
            # 这个区分是刻意的：人看的字段求可读，机器读的字段求可解析。
            for region in report.regions:
                writer.writerow(
                    {
                        "区域编号": region.region_id,
                        # 签名为 None 时写空串而不是 "None"：
                        # 后者在 Excel 里看起来像个有效签名值。
                        "区域几何签名": region.region_geometry_signature or "",
                        "源区域角色": region.source_role,
                        "区域工程状态": region.engineering_status,
                        "区域角色候选": " | ".join(region.role_candidates),
                        "Face编号": " | ".join(map(str, region.face_ids)),
                        "邻接区域编号": " | ".join(
                            map(str, region.neighbor_region_ids)
                        ),
                        "面积_mm2": region.area_mm2,
                        # 边界相关字段成组紧随面积：
                        # 甲方的判断顺序是"这块多大 -> 它的边界什么样"，
                        # 列序跟着这个思路走，读表时不用左右跳列。
                        "边界Edge数": region.boundary_edge_count,
                        "边界Edge编号": " | ".join(
                            map(str, region.boundary_edge_ids)
                        ),
                        # 连通分量用 json：它是"列表的列表"，
                        # 拼接会让层级丢失，无法再还原出每个分量有哪些边。
                        "边界Edge连通分量": _json_for_csv(
                            region.boundary_edge_components
                        ),
                        "边界环数量": region.boundary_loop_count,
                        # 环总数与闭合数相邻：两者之差就是"未闭合环数"，
                        # 放在相邻列甲方一眼就能做减法。
                        "闭合边界环候选数": region.closed_boundary_loop_count,
                        "边界拓扑状态": region.boundary_topology_status,
                        # 证据先压缩再写入：原始证据含大量嵌套结构，
                        # 直接写会让 CSV 的行高到无法浏览，
                        # 压缩保留关键字段、丢弃重复信息。
                        "边界拓扑证据": _json_for_csv(
                            _compact_boundary_topology_evidence(region)
                        ),
                        "主边界Wire Edge": " | ".join(
                            map(str, region.primary_boundary_edge_ids)
                        ),
                        "边界Wire候选数": region.boundary_wire_candidate_count,
                        "边界Wire选择状态": region.boundary_wire_selection_status,
                        "边界Wire选择证据": _json_for_csv(
                            _compact_wire_selection_evidence(region)
                        ),
                        "边界周长_mm": region.boundary_perimeter_length_mm,
                        "主边界Wire周长_mm": (
                            region.primary_boundary_perimeter_length_mm
                        ),
                        "主边界Wire投影面积_mm2": (
                            region.primary_boundary_projected_area_mm2
                        ),
                        "主边界Wire投影状态": (
                            region.primary_boundary_projection_status
                        ),
                        "主边界Wire投影证据": _json_for_csv(
                            region.primary_boundary_projection_evidence
                        ),
                        "主边界Wire环角色": region.primary_boundary_loop_role,
                        "主边界Wire环关系状态": (
                            region.primary_boundary_loop_relation_status
                        ),
                        # 环关系证据与角色、状态三者成套给出：
                        # 角色回答"这是外轮廓还是孔洞"，状态回答"结论可不可信"，
                        # 证据回答"凭什么这么判"。缺任何一个都无法复核。
                        "主边界Wire环关系证据": _json_for_csv(
                            _compact_loop_relation_evidence(region)
                        ),
                        # 中心、平均法向、包围盒是一组定位信息：
                        # 前两者用于在模型里找到这块区域，
                        # 包围盒用于判断它的尺度与形状，三者互补。
                        # 空列表写空串而不是"None"。
                        "中心_xyz_mm": " | ".join(
                            map(str, region.mean_center_xyz_mm or [])
                        ),
                        # 三个 xyz 列都用 " | " 拼接而不是 JSON 数组：
                        # 这个 CSV 是给甲方直接打开看的，
                        # 数组形式的字符串在 Excel 里读起来很费劲。
                        # 空值一律转空串而不是 "None"——
                        # 写 "None" 会被当成有值。
                        "平均法向_xyz": " | ".join(
                            map(str, region.mean_normal_xyz or [])
                        ),
                        "包围盒最小_xyz_mm": " | ".join(
                            map(str, region.bbox_min_xyz_mm or [])
                        ),
                        "包围盒最大_xyz_mm": " | ".join(
                            map(str, region.bbox_max_xyz_mm or [])
                        ),
                        # 尺寸不存成独立字段，埋在 evidence.measurements 里：
                        # 它可由最小最大推出来，不占一列。
                        # 取不到就给空列表，于是这一格为空。
                        "包围盒尺寸_xyz_mm": " | ".join(
                            map(
                                str,
                                region.evidence.get("measurements", {}).get(
                                    "bbox_size_xyz_mm",
                                    [],
                                ),
                            )
                        ),
                        "语义状态": region.semantic_status,
                        # 候选编号、类型、类别候选都是清单，
                        # 全部用 " | " 连接成一行内的多值列。
                        "候选编号": " | ".join(map(str, region.candidate_ids)),
                        "候选类型": " | ".join(region.candidate_kinds),
                        "类别候选": " | ".join(region.category_candidates),
                        "类别置信度": region.category_confidence,
                        # 复核标记与优先级相邻：
                        # 先看要不要复核，再看有多急。
                        "类别是否需要复核": region.category_review_required,
                        "复核优先级": region.review_priority,
                        # 原因码用原因码而不是自由文本：
                        # 原因码才能在整张表上按类别统计
                        # "这批主要卡在哪"。
                        "复核原因代码": " | ".join(region.review_reason_codes),
                        "分型区域候选": " | ".join(
                            region.parting_role_candidates
                        ),
                        "分型区域置信度": region.parting_confidence,
                        "分型区域是否需要复核": region.parting_review_required,
                        # 三类证据整段序列化成 JSON 字符串：
                        # 它们是嵌套结构，在 CSV 里无法展开成列，
                        # 保留原始 JSON 才能让复核者完整取回。
                        "分型区域证据": _json_for_csv(
                            region.parting_evidence
                        ),
                        "类别证据": _json_for_csv(region.category_evidence),
                        "证据": _json_for_csv(region.evidence),
                        # 警告也拼成一行：它是给人看的提示，
                        # 拆成多列会让列数随数据变化。
                        "人工复核警告": " | ".join(region.warnings),
                    }
                )
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "draft_csv": str(draft_csv_path),
        "regions_csv": str(regions_csv_path),
    }
