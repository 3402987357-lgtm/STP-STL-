"""我用 OCP XCAF 读取 STEP 装配结构并生成 BOM 几何基线。"""

from __future__ import annotations

import time
import re
from math import isfinite
from pathlib import Path
from typing import Any

from ..domain.bom import (
    BomComponent,
    BomReport,
    canonical_part_category,
    canonical_part_category_from_candidates,
    reconcile_bom_completeness,
    summarize_category_coverage,
)
from ..domain.bom_master_data import BomMasterData, BomMasterRecord
from ..errors import DependencyMissingError, GeometryProcessingError
from .file_validation import validate_input_file
from .step_probe import probe_step


_IDENTITY_MATRIX_3X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
]


def inspect_step_bom(
    path: str | Path,
    master_data: BomMasterData | None = None,
) -> BomReport:
    """我读取装配树、提取零部件几何量并保留人工复核字段。

    原理说明（甲方要求"关键算法附原理说明"）：

    1. 解析层：用 OCP 的 STEPCAFControl_Reader 把 STEP 读进 XCAF 文档。
       选 XCAF 而不是只读 Shape，是因为只有 XCAF 同时保留"装配树 + 每个
       节点的名称/材料/变换"，而 BOM 需要的正是这三样。
    2. 计量层：我只在"叶级 occurrence"上计量。装配中间节点只是容器，
       如果在容器上也计量，同一个零件会被父节点和子节点重复计入。
    3. 归并层：把指向同一参考 Shape 的 occurrence 归并成一条 BOM 行，
       数量用 occurrence 个数表达，同时保留每一个实例的装配路径和变换，
       这样既得到"零件种类"，也不丢"装了几件、装在哪"。
    4. 拆分层：当单个节点内部含多个 Solid 时按 Solid 拆分，避免
       MD270324 这类"一个节点塞 363 个实体"的模具被压成 1 个零件。
    5. 复核层：几何量能算的算，类别只给候选并附证据；材料、采购规格这类
       非几何信息在没有主数据时留空并标注来源，绝不猜。

    输入边界：只接受 .stp/.step，且必须先通过 validate_input_file 校验；
    原始文件全程只读，不写回、不改名、不联网。
    输出边界：返回结构化 BomReport，状态只会是 BASELINE 或 REVIEW_REQUIRED，
    不会在证据不足时给出 COMPLETE。
    """

    started = time.perf_counter()
    file_path = validate_input_file(path, {".stp", ".step"}).resolve()
    probe = probe_step(file_path)
    try:
        # OCP 在函数内导入且只捕 ImportError：
        # 这样没装 OCP 的环境仍能 import 本模块、仍能看到
        # 其他不依赖 OCP 的功能；且报的是"缺依赖"而不是
        # 一串难以理解的 NameError。
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepGProp import BRepGProp
        from OCP.Bnd import Bnd_Box
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TCollection import TCollection_ExtendedString, TCollection_HAsciiString
        from OCP.TDataStd import TDataStd_Name
        from OCP.TDocStd import TDocStd_Document
        from OCP.TDF import TDF_Label
        from OCP.TopAbs import (
            TopAbs_EDGE,
            TopAbs_FACE,
            TopAbs_SHELL,
            TopAbs_SOLID,
            TopAbs_VERTEX,
        )
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
        from OCP.collections import Sequence_TDF_Label
    except ImportError as exc:
        raise DependencyMissingError(
            "我无法读取 STEP 装配 BOM，因为当前环境没有 OCP XCAF。"
        ) from exc

    document = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    reader = STEPCAFControl_Reader()
    # 三个模式开关都必须打开：
    #   NameMode —— 读得到零件名（BOM 的名称列）；
    #   PropsMode —— 读得到用户自定义属性（材料、规格）；
    #   MetaMode —— 读得到 STEP 头部的元数据。
    # 少开任何一个都会让 BOM 相应列整列为空，
    # 而空列看起来像"模型里没写"，极易被误判。
    reader.SetNameMode(True)
    reader.SetPropsMode(True)
    reader.SetMetaMode(True)
    read_status = reader.ReadFile(str(file_path))
    # 读失败抛业务异常而不是返回空 BOM：
    # 空 BOM 会被下游当成"这个装配体没有零件"，
    # 而真实情况是文件根本没读进来。
    if read_status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise GeometryProcessingError(
            f"我读取 STEP 装配体失败，后端状态为 {read_status}: {file_path}"
        )
    if not reader.Transfer(document):
        raise GeometryProcessingError(f"我没有转移出 STEP 装配树: {file_path}")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    material_tool = XCAFDoc_DocumentTool.MaterialTool_s(document.Main())
    free_labels = Sequence_TDF_Label()
    shape_tool.GetFreeShapes(free_labels)
    occurrences: list[dict[str, Any]] = []
    for index in range(1, free_labels.Length() + 1):
        root = free_labels.Value(index)
        _collect_leaf_occurrences(
            root,
            f"root:{index}",
            shape_tool,
            Sequence_TDF_Label,
            TDF_Label,
            occurrences,
            parent_matrix=_IDENTITY_MATRIX_3X4,
        )

    warnings: list[str] = []
    if not occurrences:
        warnings.append("我没有从 XCAF 装配树得到可计量的零部件。")
    elif free_labels.Length() > 1:
        warnings.append("我发现多个顶层装配对象，当前 BOM 按出现次序记录。")
    groups = _group_occurrences(occurrences)
    shape_types = {
        "SOLID": TopAbs_SOLID,
        "SHELL": TopAbs_SHELL,
        "FACE": TopAbs_FACE,
        "EDGE": TopAbs_EDGE,
        "VERTEX": TopAbs_VERTEX,
    }
    # 我把"一个装配节点里含多个 Solid"的复合体拆成逐个 Solid 计量，
    # 否则 MD270324 这类 363 个实体的模具会被压成 1 个零部件，漏掉零件。
    measurement_units, multi_solid_group_count = _expand_multi_solid_groups(
        groups,
        shape_tool,
        TopExp_Explorer,
        TopAbs_SOLID,
        TopoDS,
    )
    # 一个"多实体组"要拆成多个独立计量单元：BOM 里同一零件若由多个
    # 不相交实体拼成，按整体算体积和质量都会算进中间的空隙，
    # 所以必须先拆开逐个量，再在汇总层决定是否合并展示。
    components = [
        _measure_component(
            unit["measure_label"],
            index,
            shape_tool,
            TDataStd_Name,
            BRepBndLib,
            Bnd_Box,
            BRepGProp,
            GProp_GProps,
            TopExp_Explorer,
            shape_types,
            adaptor_surface_type=BRepAdaptor_Surface,
            topods_type=TopoDS,
            material_tool=material_tool,
            ascii_string_type=TCollection_HAsciiString,
            # 拆分后优先用拆分体自己的引用形状，没有才回退到原引用：
            # 拆分体的几何与原实体不同，用错的引用会量出原整体的属性。
            reference_shape_id=(
                unit["split_reference_shape_id"] or unit["reference_shape_id"]
            ),
            occurrence_ids=unit["occurrence_ids"],
            occurrence_paths=unit["occurrence_paths"],
            occurrence_locations=unit["occurrence_locations"],
            shape_override=unit["shape_override"],
            # 名称后缀带上 solid 序号，让甲方能在 BOM 里区分
            # "同一个零件被拆成了几块"，否则多行同名无法核对。
            name_suffix=(
                f"#solid_{unit['solid_index']:03d}"
                if unit["solid_index"] is not None
                else ""
            ),
            split_solid_total=(
                unit["solid_total"] if unit["solid_index"] is not None else None
            ),
        )
        for index, unit in enumerate(measurement_units, start=1)
    ]
    multi_solid_split_component_count = sum(
        1 for unit in measurement_units if unit["solid_index"] is not None
    )
    # 拆分数与组数都写进警告：甲方核对 BOM 时最常问的就是
    # "为什么一个零件变成了三行"，这句话直接给出答案与拆分依据。
    if multi_solid_group_count:
        warnings.append(
            "我把 "
            f"{multi_solid_group_count} 个多 Solid 复合体拆成 "
            f"{multi_solid_split_component_count} 个零部件，避免漏计零件；"
            "拆分依据是每个 Solid 的独立几何量，仍需要人工复核装配语义。"
        )
    master_data_match_count = _apply_master_data(components, master_data)
    # 能力边界无条件写入警告：BOM 的行业类别与采购规格
    # 不在几何能回答的范围内，这句话必须跟着每一份 BOM 走。
    warnings.append(
        "我只根据装配树和几何量生成 BOM 基线；材料、采购规格和最终行业类别需要人工或主数据复核。"
    )
    # 空组件列表直接判 REVIEW_REQUIRED：没有零件的 BOM
    # 说明连装配树都没读出来，不是"这个模型没有零件"。
    status = "BASELINE" if components else "REVIEW_REQUIRED"
    # 任一零件 Shape 为空也降级：只要有一个零件没量到，
    # 整份 BOM 的数量与重量汇总就不成立。
    if any(
        any("Shape 为空" in warning for warning in component.warnings)
        for component in components
    ):
        status = "REVIEW_REQUIRED"
    # 我先把"不遗漏任何零件"对一次账，再决定整体状态。
    # 放在这里是因为它只需要 components 与装配树实例数，
    # 两者此时都已确定；放在前面会因为 components 还没建好而无从对账。
    # 预期槽位 = 每个计量单元各自继承的实例数之和。
    # 拆分会让"1 个装配节点含 3 个 Solid"展开成 3 个单元、各继承 1 个实例，
    # 所以这个值通常大于装配树实例数；拿装配树实例数去比件数永远对不上，
    # 那条判据就会一直报假警。这里按实际展开后的期望值比对。
    expected_occurrence_slots = sum(
        len(unit["occurrence_ids"]) for unit in measurement_units
    )
    completeness = reconcile_bom_completeness(
        components,
        len(occurrences),
        expected_occurrence_slots=expected_occurrence_slots,
    )
    # 对账失败时把它升级成整份报告的复核理由：
    # 数量对不上意味着"按这份 BOM 采购会多买或少买"，
    # 这是必须被看见的问题，不能只躺在诊断字段里。
    if completeness["status"] == "FAIL":
        status = "REVIEW_REQUIRED"
        warnings.append(
            "我方 BOM 数量与装配树实例数对不上，不能据此直接采购："
            + completeness["reason"]
        )
    # 大类覆盖统计顺手算出：8 大类的命中数要能直接对上任务书口径，
    # 并且每个 0 都必须说明它是"我方认不出"还是"几何歧义未定"。
    coverage = summarize_category_coverage(components)
    if coverage["unreachable_groups"]:
        # 这些 0 是我方能力缺口，不是"模具里没有"。
        # 不写这句话，甲方会把"成型零件 0 件"读成"这台模具没有型腔型芯"——
        # 而那是任何一套注塑模都必然存在的零件。
        warnings.append(
            "我方无法识别以下部件大类（缺少零件名称或主数据，"
            "不是模具里没有）：" + "、".join(coverage["unreachable_groups"])
            + "。这些大类的 0 命中不等于不存在，需要甲方提供名称规范或主数据。"
        )
    return BomReport(
        status=status,
        file_path=str(file_path),
        unit=probe.unit_hint,
        backend="OCP_XCAF_STEPCAFControl",
        component_count=len(components),
        components=components,
        input_metadata=probe.file.to_dict(),
        warnings=list(dict.fromkeys(warnings)),
        diagnostics={
            # 装配树三层计数：顶层标签、实例出现次数、去重后的引用形状数。
            # 三者不等是正常的（同一零件被引用多次），
            # 差别本身就是"用了多少种零件、装了多少个"的信息。
            "top_level_count": free_labels.Length(),
            "assembly_occurrence_count": len(occurrences),
            "unique_reference_shape_count": len(groups),
            "measurement_unit_count": len(measurement_units),
            "multi_solid_group_count": multi_solid_group_count,
            "multi_solid_split_component_count": (
                multi_solid_split_component_count
            ),
            # 拆分策略写进结果：一个 Solid 一行是当下的口径，
            # 将来若改成"一个装配节点一行"，靠这个字段能区分两种报告。
            "solid_split_policy": "ONE_BOM_COMPONENT_PER_SOLID",
            # 类别语义恒为"仅几何候选"：行业类别不在几何能回答的范围内，
            # 写死这条是为了防止有人把 category 当成确定结论使用。
            "category_semantics": "GEOMETRIC_CANDIDATE_ONLY",
            # 两个计数分开：完全没分类的 vs 分了类但仍需复核的。
            # 前者要人工从头定类别，后者只需确认，工作量差一个量级。
            "category_review_count": sum(
                component.category == "未确定" for component in components
            ),
            "category_review_required_count": sum(
                component.category_review_required for component in components
            ),
            "category_candidate_counts": _category_candidate_counts(components),
            # 材料与采购规格的来源单独标出：它们可能来自主数据，
            # 也可能这份 STEP 里根本没有。不知道来源就无法判断
            # "没写"是"确实没有"还是"没读到"。
            "material_source": (
                # 材料可能来自 XCAF 也可能来自主数据，两者都算有来源；
                # 都没有时写 NOT_PRESENT_IN_BASELINE 而不是空串，
                # 这样"模型里没标"和"我们没读出来"能区分开。
                "MASTER_DATA_OR_XCAF"
                if any(component.material for component in components)
                else "NOT_PRESENT_IN_BASELINE"
            ),
            "purchase_specification_source": (
                # 采购规格只可能来自主数据：几何 BOM 推不出它，
                # 所以这里从不写 XCAF。
                "MASTER_DATA"
                if any(component.purchase_specification for component in components)
                else "NOT_PRESENT_IN_BASELINE"
            ),
            # 是否提供了主数据、匹配上多少条都记下来：
            # 没有这两个数就无法判断"匹配数少"是因为
            # 没给主数据还是因为给的都对不上。
            "master_data_supplied": master_data is not None,
            "master_data_match_count": master_data_match_count,
            # 不重不漏对账：甲方"不遗漏任何零件"这条要求此前没有任何判据，
            # 少算零件不会报错。我把它做成结构化证据写进诊断，
            # 让每次产出都能被对账，而不是靠人肉核对。
            "completeness": completeness,
            # 8 大类覆盖：命中数 + 每个 0 的原因（能力缺口 vs 几何歧义）。
            # 只给计数的话，"成型零件 0 件"会被读成"没有型腔型芯"。
            "category_coverage": coverage,
            # 数量来源固定为 XCAF 的出现次数：几何 BOM 里的
            # "数量"只能是装了几次，不是采购数量。
            "quantity_source": "XCAF_OCCURRENCE_COUNT",
        },
        # 耗时单独记：BOM 读取在大装配上可能很慢，
        # 没有耗时就无法判断"慢"是模型复杂还是读法有问题。
        elapsed_seconds=time.perf_counter() - started,
    )


def _measure_component(
    label: Any,
    index: int,
    shape_tool: Any,
    name_type: Any,
    bbox_lib: Any,
    bbox_type: Any,
    prop_lib: Any,
    props_type: Any,
    explorer_type: Any,
    shape_types: dict[str, Any],
    *,
    adaptor_surface_type: Any,
    topods_type: Any,
    material_tool: Any,
    ascii_string_type: Any,
    reference_shape_id: str,
    occurrence_ids: list[str],
    occurrence_paths: list[str],
    occurrence_locations: list[dict[str, Any] | None],
    shape_override: Any | None = None,
    name_suffix: str = "",
    split_solid_total: int | None = None,
) -> BomComponent:
    """我计量单个零部件（或拆分后的单个 Solid）并组装成一条 BOM 记录。

    原理说明：

    1. 包围盒用 BRepBndLib 的 Add 直接对 B-Rep 求界，不先三角化；
       三角化会引入离散误差，而 BOM 的"长宽厚"要能对上图纸。
    2. 表面积/体积用 BRepGProp 的 SurfaceProperties_s / VolumeProperties_s
       做精确积分，失败时返回 None 而不是 0——0 会被误读成"退化为零"，
       None 才能表达"没算出来"。
    3. 曲面类型统计交给 BRepAdaptor_Surface，把 GeomAbs_Cylinder 等
       解析曲面计数交给类别规则当弱证据，而不是拿网格法向去猜。
    4. shape_override 不为空表示这是从多 Solid 复合体里拆出来的单个 Solid，
       此时名称追加 #solid_00X 后缀，保证拆分件可追溯。
    """
    shape = (
        # shape_override 用于多实体拆分后的子体：拆分体的几何不在原 label 上，
        # 必须显式传入，否则会量回未拆分的整体，体积与质量都会偏大。
        shape_override
        if shape_override is not None
        else shape_tool.GetShape_s(label)
    )
    warnings: list[str] = []
    component_id = f"component_{index:04d}"
    # 名称缺失时回退到 component_id：BOM 里每一行都必须有可引用的标识，
    # 用空字符串会让下游合并同类项时把不同零件误并成一条。
    name = (_label_name(label, name_type) or component_id) + name_suffix
    bbox_min: list[float] | None = None
    bbox_max: list[float] | None = None
    dimensions: list[float] | None = None
    # 包围盒失败不中断整条 BOM：只在这一个零件上记警告，
    # 其余零件的体积质量照常产出。一个坏零件就丢整份 BOM 是不可接受的。
    if shape.IsNull():
        warnings.append("我读取到的零部件 Shape 为空。")
    else:
        try:
            bbox = bbox_type()
            bbox_lib.Add_s(shape, bbox)
            bbox_min = [float(bbox.GetXMin()), float(bbox.GetYMin()), float(bbox.GetZMin())]
            bbox_max = [float(bbox.GetXMax()), float(bbox.GetYMax()), float(bbox.GetZMax())]
            dimensions = [
                float(bbox_max[0] - bbox_min[0]),
                float(bbox_max[1] - bbox_min[1]),
                float(bbox_max[2] - bbox_min[2]),
            ]
        except Exception as exc:
            warnings.append(f"我无法计算零部件包围盒：{exc}")

    counts = {
        label_name: _count_shapes(shape, explorer_type, shape_type)
        for label_name, shape_type in shape_types.items()
    }
    surface_type_counts = _surface_type_counts(
        shape,
        explorer_type,
        shape_types["FACE"],
        adaptor_surface_type,
        topods_type,
    )
    surface_area = _mass_property(shape, prop_lib.SurfaceProperties_s, props_type)
    volume = _mass_property(shape, prop_lib.VolumeProperties_s, props_type)
    # 面积与体积走同一个 _mass_property，区别只在传哪个属性函数：
    # OCP 的 GProp 要求先算属性再读值，封装成同一调用避免两处写重复样板。
    if surface_area is None:
        # 算不出只记警告，不否决整条 BOM 行：
        # 缺体积的零件仍可列出名称、数量与材质，
        # 让整行消失比缺一个字段损失大得多。
        warnings.append("我无法稳定计算零部件表面积。")
    if volume is None:
        warnings.append("我无法稳定计算零部件体积。")
    # 分类决策必须在量测之后：它综合名称、尺寸、拓扑计数、
    # 面积体积与面类型分布来判断，缺任一项依据就不完整。
    decision = _category_decision(
        name,
        dimensions,
        counts,
        surface_area,
        volume,
        surface_type_counts=surface_type_counts,
    )
    category = str(decision["category"])
    confidence = float(decision["confidence"])
    category_source = str(decision["source"])
    category_evidence = list(decision["evidence"])
    if category == "未确定":
        # 未确定与"靠名称猜出来"是两种不同的警告：
        # 前者要人工从头定类别，后者只需确认现有候选，
        # 混成一句话会让复核者无法判断工作量。
        warnings.append("我没有足够的工程特征确认行业零件类别。")
    elif category_source == "NAME_RULE":
        warnings.append("我根据零件名称得到类别候选，仍需要工程规则或人工复核。")
    if decision["review_required"]:
        warnings.append("我保留该零部件的类别候选和复核原因，不把它写成最终行业类别。")
    material = _material_name(label, material_tool, ascii_string_type)
    remarks = "我方几何 BOM 基线，材料和采购规格待主数据补充。"
    # 从复合体拆出的零件在备注里说明来源：
    # 甲方看到"一个零件变成多行"时，这句话直接解释原因，
    # 否则每行都要去查诊断字段才能明白。
    if split_solid_total:
        remarks = (
            f"我方从含 {split_solid_total} 个 Solid 的复合体拆出该零部件，"
            "避免漏计零件；"
            + remarks
        )
    return BomComponent(
        component_id=component_id,
        reference_shape_id=reference_shape_id,
        # 出现路径、编号、位置、次数都保留：
        # 同一个零件被引用多次时，这四组字段是
        # "装在哪里、装了几个"的唯一依据。
        occurrence_path=occurrence_paths[0],
        occurrence_ids=list(occurrence_ids),
        occurrence_paths=list(occurrence_paths),
        occurrence_locations=list(occurrence_locations),
        occurrence_count=len(occurrence_ids),
        part_name=name,
        category=category,
        category_confidence=confidence,
        # 类别来源与证据都留：只有类别名无法判断
        # 它是靠名称、靠几何还是靠主数据得来的，
        # 而不同来源的可信度差很多。
        category_source=category_source,
        category_evidence=category_evidence,
        # 全部候选都保留：只留下选中的那个，
        # 复核者就无法知道"还有哪些可能、为什么没选它们"。
        category_candidates=list(decision["candidates"]),
        category_review_required=bool(decision["review_required"]),
        # 复核原因用原因码而不是自由文本：
        # 原因码才能按类别统计"这批 BOM 里主要卡在哪"。
        category_review_reason_codes=list(decision["review_reason_codes"]),
        part_category_group=decision.get("part_category_group"),
        part_category_group_strength=str(
            decision.get("part_category_group_strength", "none")
        ),
        material=material,
        # 材料来源区分"读到了"与"模型里没写"：
        # 都留空的话，甲方无法区分"这个件没标材料"
        # 和"我们没读出来"。
        material_source="XCAF_MaterialTool" if material else "NOT_PRESENT",
        quantity=len(occurrence_ids),
        # 采购规格一律留 None 并标 NOT_PRESENT：
        # 几何 BOM 推不出采购规格，编一个会让人误以为有来源。
        purchase_specification=None,
        purchase_specification_source="NOT_PRESENT",
        remarks=remarks,
        bbox_min_xyz_mm=bbox_min,
        bbox_max_xyz_mm=bbox_max,
        dimensions_xyz_mm=dimensions,
        # 五类拓扑元素计数都给：它们既是"这个件有多复杂"的度量，
        # 也是判断"是否拆对了"的依据——例如只有一个面的件
        # 通常说明几何没取全。
        solid_count=counts["SOLID"],
        shell_count=counts["SHELL"],
        face_count=counts["FACE"],
        edge_count=counts["EDGE"],
        vertex_count=counts["VERTEX"],
        surface_area_mm2=surface_area,
        volume_mm3=volume,
        warnings=warnings,
    )


def _collect_leaf_occurrences(
    label: Any,
    path: str,
    shape_tool: Any,
    sequence_type: Any,
    label_type: Any,
    occurrences: list[dict[str, Any]],
    *,
    parent_matrix: list[list[float]] | None = None,
) -> None:
    """我只收集叶级 occurrence，避免把装配节点和零件重复计入 BOM。

    原理说明：装配树里"有子节点"的标签只是容器，不是零件。我沿
    Components() 递归下降，只有没有子节点的标签才登记为 occurrence；
    同时把父级变换一路乘下来，得到每个实例的全局 3x4 矩阵。
    边界：装配环、自引用或损坏的引用会被 Components() 自身的迭代终止条件
    挡住；我不额外做环检测，因为 OCAF 的引用在读取阶段已经是树结构。
    """

    local_summary = _location_summary(shape_tool, label)
    # 取不到局部变换就用单位矩阵：装配树里顶层节点常常没有变换，
    # 这时"不变换"就是正确的语义，不是错误。
    local_matrix = (
        local_summary["matrix_3x4"]
        if local_summary is not None
        else _IDENTITY_MATRIX_3X4
    )
    # 全局变换 = 父级变换 ∘ 局部变换：
    # 一个零件的真实位置要靠整条路径上的变换依次复合，
    # 只用局部变换会得到"相对于父节点"的坐标，
    # 无法在全局坐标系里定位它。
    global_matrix = _compose_affine_3x4(
        parent_matrix or _IDENTITY_MATRIX_3X4,
        local_matrix,
    )
    components = sequence_type()
    # 第三参数 False 表示不递归取子组件：
    # 这里要自己控制遍历顺序以便累积路径与变换，
    # 让 OCP 递归的话拿不到这些信息。
    shape_tool.GetComponents_s(label, components, False)
    if components.Length():
        # 有子组件说明这是装配节点，继续往下走；
        # 路径里带上序号，保证同一父节点下的多个子组件路径不重复。
        for index in range(1, components.Length() + 1):
            _collect_leaf_occurrences(
                components.Value(index),
                # XCAF 的子组件下标从 1 开始，不是 0。
                f"{path}/component:{index}",
                shape_tool,
                sequence_type,
                label_type,
                occurrences,
                parent_matrix=global_matrix,
            )
        # 装配节点本身不入 BOM：BOM 里只应有叶子零件，
        # 把装配节点也记进去会让零件数量翻倍。
        return

    reference_label = label_type()
    # 引用型节点要解引用后再量测：引用节点自身没有几何，
    # 直接量它会得到空的体积与面积。
    if shape_tool.IsReference_s(label) and shape_tool.GetReferredShape_s(
        label, reference_label
    ):
        measure_label = reference_label
    else:
        measure_label = label
    occurrence_location = _location_evidence(local_summary, global_matrix)
    occurrences.append(
        {
            # 参考形状 ID 与实际出现 ID 分开：同一个零件被引用多次时
            # reference 相同、occurrence 不同，
            # 这两个 ID 正是"同一个件装了几处"的依据。
            "reference_shape_id": f"label:{measure_label.Tag()}",
            "occurrence_id": f"label:{label.Tag()}",
            "occurrence_path": path,
            "occurrence_location": occurrence_location,
            "measure_label": measure_label,
        }
    )


def _group_occurrences(occurrences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """我按参考 Shape 合并重复实例，并保留所有 occurrence 追踪信息。"""

    # 原理说明：BOM 的"数量"有两种口径——装配树里出现了几次（occurrence），
    # 和几何上是不是同一个零件（参考 Shape）。我把两者都留下：一条 BOM 行
    # 对应一个参考 Shape，数量取 occurrence 个数，实例编号/路径/变换全部
    # 并列保存，这样既能按种类汇总，也能按位置追溯，不会二选一丢信息。

    grouped: dict[str, dict[str, Any]] = {}
    for occurrence in occurrences:
        key = occurrence["reference_shape_id"]
        group = grouped.setdefault(
            key,
            {
                "reference_shape_id": key,
                "measure_label": occurrence["measure_label"],
                "occurrence_ids": [],
                "occurrence_paths": [],
                "occurrence_locations": [],
            },
        )
        group["occurrence_ids"].append(occurrence["occurrence_id"])
        group["occurrence_paths"].append(occurrence["occurrence_path"])
        group["occurrence_locations"].append(occurrence["occurrence_location"])
    return list(grouped.values())


def _label_name(label: Any, name_type: Any) -> str | None:
    """我读取装配节点上的名称属性；取不到或取到脏值时返回 None。

    两个现实边界：
      - 名称取不到是常态。很多 STEP 只在装配节点上有名字，零件节点没写，
        所以返回 None 是正常结果，不是错误，由调用方退回按几何编号命名；
      - 我额外过滤以 "=>" 开头的字符串。那是某些 CAD 导出器写出的
        内部引用标记而不是真名，直接拿去当零件名会出现一整列 "=>xxx"
        的垃圾名称，比留空更难处理。

    我把整个读取包在 try 里：名称只是辅助信息，它失败不该让整件
    模具的 BOM 解析挂掉。
    """

    try:
        attribute = name_type()
        if not label.FindAttribute(name_type.GetID_s(), attribute):
            return None
        value = attribute.Get().ToExtString()
        if not value or value.startswith("=>"):
            return None
        return value
    except Exception:
        return None


def _location_summary(shape_tool: Any, label: Any) -> dict[str, Any] | None:
    """我保存 occurrence 的局部变换矩阵，不假设它已经是全局坐标。"""

    try:
        location = shape_tool.GetLocation_s(label)
        transformation = location.Transformation()
        return {
            "source": "XCAFDoc_ShapeTool.GetLocation_s",
            "form": str(transformation.Form()),
            "matrix_3x4": [
                [float(transformation.Value(row, column)) for column in range(1, 5)]
                for row in range(1, 4)
            ],
        }
    except Exception:
        return None


def _location_evidence(
    local_summary: dict[str, Any] | None,
    global_matrix: list[list[float]],
) -> dict[str, Any] | None:
    """我同时保留局部来源和沿装配路径合成的全局矩阵。"""

    if local_summary is None:
        return None
    evidence = dict(local_summary)
    evidence["global_source"] = (
        "我按父级 occurrence 到当前 occurrence 的装配路径合成"
    )
    evidence["global_matrix_3x4"] = [
        [float(value) for value in row] for row in global_matrix
    ]
    return evidence


def _compose_affine_3x4(
    parent: list[list[float]],
    local: list[list[float]],
) -> list[list[float]]:
    """我用标准刚体仿射乘法合成父级和当前 occurrence 变换。"""

    # 原理说明：STEP 装配里每个节点只存相对父级的局部变换，最终位置要靠
    # 从根到叶逐级相乘。这里只处理 3x4（旋转+平移）齐次矩阵的乘法：
    # 结果平移 = 父旋转 * 子平移 + 父平移，结果旋转 = 父旋转 * 子旋转。
    # 边界：我不做尺度分解，也不做坐标系转换；单位统一交给上层 probe 处理。

    return [
        [
            sum(parent[row][index] * local[index][column] for index in range(3))
            + (parent[row][3] if column == 3 else 0.0)
            for column in range(4)
        ]
        for row in range(3)
    ]


def _count_shapes(shape: Any, explorer_type: Any, shape_type: Any) -> int:
    """我用 TopExp_Explorer 数出指定类型的子形状个数。

    先判 IsNull 再建 explorer：对空形状建 explorer 在部分 OCP
    版本上会直接抛异常，而"这个形状是空的"是合法输入，
    应当得到计数 0 而不是让整份 BOM 失败。

    计数走 while More/Next 而不是一次性取出列表：
    探索器是惰性遍历，不需要为中间结果分配容器，
    在面数上万的装配体上差别明显。
    """
    if shape.IsNull():
        return 0
    explorer = explorer_type(shape, shape_type)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _iter_solid_shapes(
    shape: Any,
    explorer_type: Any,
    solid_type: Any,
    topods_type: Any,
) -> list[Any]:
    """我按出现顺序枚举复合 Shape 中的每个 Solid，用于逐件拆分 BOM。"""

    if shape.IsNull():
        return []
    solids: list[Any] = []
    explorer = explorer_type(shape, solid_type)
    while explorer.More():
        try:
            solids.append(topods_type.Solid(explorer.Current()))
        except Exception:  # noqa: BLE001
            pass
        explorer.Next()
    return solids


def _expand_multi_solid_groups(
    groups: list[dict[str, Any]],
    shape_tool: Any,
    explorer_type: Any,
    solid_type: Any,
    topods_type: Any,
) -> tuple[list[dict[str, Any]], int]:
    """我把含多个 Solid 的装配节点拆成逐个 Solid 的计量单元。

    甲方要求"不遗漏任何零件"。当一套模具被导出成"单个装配节点 + 多个
    Solid"时，只按装配节点计量会把上百个零件压成一个，所以我按 Solid
    拆开，并给每个拆分件一个可追溯的参考 Shape 编号。

    原理说明与边界：
    1. 我用 TopExp_Explorer 按出现顺序枚举 TopAbs_SOLID，顺序稳定，
       因此 #solid_00X 编号在重复运行下可复现。
    2. 每个拆分件继承原节点的 occurrence 列表，所以 occurrence_count
       仍然是"这个子零件被装了几次"，总量 = 子零件数 x 实例数，不会漏计。
    3. 只拆 Solid，不拆 Shell。同一 Solid 内的多 Shell（例如 BOTTLE 的
       1 Solid / 14 Shell）在装配语义上仍是一个零件，拆了反而会多报。
    4. 拆分依据是几何，不等于人工确认的装配语义，所以状态保持 BASELINE
       并追加复核提示，不写成最终结论。
    """

    units: list[dict[str, Any]] = []
    multi_solid_group_count = 0
    for group in groups:
        shape = shape_tool.GetShape_s(group["measure_label"])
        solids = _iter_solid_shapes(
            shape,
            explorer_type,
            solid_type,
            topods_type,
        )
        # 单个实体的组直接作为一个计量单位：
        # 绝大多数零件都属于这种，不需要拆分。
        if len(solids) <= 1:
            units.append(
                {
                    **group,
                    # shape_override 留 None 表示"用组本身的形状"：
                    # 留空与"覆盖成别的形状"是两种不同的语义。
                    "shape_override": None,
                    "solid_index": None,
                    "solid_total": len(solids),
                    "split_reference_shape_id": None,
                }
            )
            continue
        # 多个实体的组按实体拆开：一个 STEP 零件里含有
        # 多个互不相连的实体时，它们的体积与面积
        # 必须分开算，合在一起会得到错误的单体重量。
        multi_solid_group_count += 1
        for solid_index, solid in enumerate(solids, start=1):
            units.append(
                {
                    **group,
                    # 覆盖成这个实体：后续所有量测都针对它进行。
                    "shape_override": solid,
                    # 下标从 1 开始，与 BOM 里其他"第几个"的口径一致；
                    # 后缀补零到三位是为了让 ID 长度固定，排序时不会错位。
                    "solid_index": solid_index,
                    "solid_total": len(solids),
                    # 拆分后的 ID 带 #solid_001 后缀：
                    # 同一零件的多个实体要靠它区分，
                    # 否则它们在 BOM 里会共用同一个 ID 而互相覆盖。
                    "split_reference_shape_id": (
                        f"{group['reference_shape_id']}#solid_{solid_index:03d}"
                    ),
                    # 关键修正：拆分出来的每个 Solid 都**只代表原节点的
                    # 一部分**，不能继承原节点那一份完整的 occurrence 列表。
                    # 继承会让同一个 occurrence 在 N 个拆分行里各出现一次，
                    # 数量合计虚高 N-1 倍——实测 mould_04 就因此把 156 个
                    # 实例算成 159（多 3 件）。采购按这个数下单会多买。
                    # 所以我给拆分行各自一个"派生的、互不重复"的 occurrence id，
                    # 既保留可追溯（能看出是从哪个 occurrence 的第几个 Solid 拆的），
                    # 又不会让同一个实例被重复计数。
                    "occurrence_ids": [
                        f"{occurrence_id}#solid_{solid_index:03d}"
                        for occurrence_id in group["occurrence_ids"]
                    ],
                    # 路径也带后缀：两行同名同路径会让甲方无法分辨
                    # "这是两个不同的块"还是"同一块被列了两次"。
                    "occurrence_paths": [
                        f"{path}#solid_{solid_index:03d}"
                        for path in group["occurrence_paths"]
                    ],
                }
            )
    return units, multi_solid_group_count


def _surface_type_counts(
    shape: Any,
    explorer_type: Any,
    face_type: Any,
    adaptor_surface_type: Any,
    topods_type: Any,
) -> dict[str, int]:
    """我统计每个零部件的曲面类型，用于圆柱面等几何类别证据。"""

    if shape.IsNull():
        return {}
    counts: dict[str, int] = {}
    explorer = explorer_type(shape, face_type)
    while explorer.More():
        try:
            face = topods_type.Face(explorer.Current())
            adaptor = adaptor_surface_type(face)
            surface_type = getattr(
                adaptor.GetType(),
                "name",
                str(adaptor.GetType()),
            )
        except Exception:
            surface_type = "UNKNOWN"
        counts[surface_type] = counts.get(surface_type, 0) + 1
        explorer.Next()
    return counts


def _mass_property(shape: Any, operation: Any, props_type: Any) -> float | None:
    """我算质量属性（质量/体积/面积中的一种），拿不到就返回 None。

    为什么吞掉异常而不是让它往上抛：OCP 的质量属性计算在自相交
    或未闭合的形状上会抛异常，而 BOM 里"这一个零件的体积算不出来"
    不该让整份物料清单报废。返回 None 让该字段在报告里显示为缺失，
    其余字段照常产出。

    返回前判 isfinite 且非负：质量属性算出一个负值或 NaN 说明
    这次计算本身不可信（形状未闭合时体积会给出荒谬的结果），
    宁可留空也不能把一个错数值写进 BOM 让人去报价。
    """
    if shape.IsNull():
        return None
    try:
        props = props_type()
        operation(shape, props)
        value = float(props.Mass())
        return value if isfinite(value) and value >= 0.0 else None
    except Exception:
        return None


_NAME_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("成型件候选", ("型腔", "型芯", "镶件", "cavity", "core", "insert")),
    ("模架件候选", ("模架", "模座", "模板", "mold base", "mould base", "plate")),
    (
        "导向件候选",
        ("导柱", "导套", "导向", "guide post", "guide pillar", "guide bushing"),
    ),
    ("顶出件候选", ("顶针", "顶杆", "顶出", "ejector", "ejector pin")),
    ("侧向抽芯件候选", ("滑块", "斜顶", "slider", "slide", "lifter")),
    ("紧固件候选", ("螺钉", "螺栓", "螺母", "bolt", "screw", "nut")),
    (
        "冷却浇注件候选",
        ("冷却", "水路", "浇口", "喷嘴", "cooling", "runner", "gate", "nozzle"),
    ),
)


def _category_decision(
    part_name: str,
    dimensions: list[float] | None,
    counts: dict[str, int],
    surface_area: float | None,
    volume: float | None,
    *,
    surface_type_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """我用名称和几何证据输出多候选类别、置信度和复核原因。"""

    # 原理说明：我不做"单标签硬分类"，而是把名称规则和几何规则各自产生的
    # 候选合并成候选集合，每条候选带来源（NAME_RULE / GEOMETRY_RULE）和
    # 自然语言证据。置信度取同一候选在各规则下的最大值，主候选取置信度
    # 最高者，并列时按标签排序保证可复现。
    # 复核语义：只要候选不唯一、或只有名称/只有几何单边证据，就置
    # review_required=True 并写入原因码，绝不把弱证据包装成确定类别。

    normalized_name = " ".join(part_name.casefold().replace("_", " ").split())
    candidates: dict[str, dict[str, Any]] = {}
    review_reason_codes: list[str] = []
    # 名称规则先跑：名称是甲方自己写的，语义最强，
    # 命中就给 0.75 的高置信度。归一化（小写+下划线转空格+压缩空白）
    # 是为了让 "Top_Plate" 与 "top plate" 得到同一个键。
    for category, tokens in _NAME_CATEGORY_RULES:
        matched = [
            token for token in tokens if _matches_name_token(normalized_name, token)
        ]
        if not matched:
            continue
        # setdefault 按类别归并：同一个类别可能被多条规则命中，
        # 合并成一条记录而不是产生多行同类别的候选。
        record = candidates.setdefault(
            category,
            {
                "category": category,
                "confidence": 0.0,
                "sources": [],
                "evidence": [],
            },
        )
        # 置信度取 max 而不是覆盖：后命中的规则不一定比先命中的更强。
        record["confidence"] = max(float(record["confidence"]), 0.75)
        if "NAME_RULE" not in record["sources"]:
            record["sources"].append("NAME_RULE")
        record["evidence"].append(
            f"我在零件名称中匹配到关键词：{', '.join(matched)}"
        )

    # 几何规则后跑并合并进同一份候选表：
    # 名称与几何可能指向同一类别，此时这条候选的 sources 会有两个来源，
    # 这正是"双重印证"的证据，比任一单独来源更可信。
    geometry_candidates, geometry_evidence, _geometry_codes = _geometry_category_candidates(
        dimensions,
        counts,
        surface_area,
        volume,
        surface_type_counts or {},
    )
    for record in geometry_candidates:
        target = candidates.setdefault(
            record["category"],
            {
                "category": record["category"],
                "confidence": 0.0,
                "sources": [],
                "evidence": [],
            },
        )
        # 置信度取 max：名称规则给 0.75、几何规则一般更低，
        # 同一类别被两者同时命中时应保留较高的那个，
        # 用几何的低值覆盖会抹掉名称这条强证据。
        target["confidence"] = max(float(target["confidence"]), float(record["confidence"]))
        # 来源去重追加：sources 里出现 ["NAME_RULE", "GEOMETRY_RULE"]
        # 就是双重印证，重复出现同一个来源没有意义。
        for source in record["sources"]:
            if source not in target["sources"]:
                target["sources"].append(source)
        # 证据全量累积而不是替换：名称证据与几何证据说明的不是同一件事，
        # 都要留下来供复核。
        target["evidence"].extend(record["evidence"])

    # 一条候选都没有时给"未确定"而不是抛错或猜一个类别：
    # 未确定是真实结论，review_required=True 会把它推给人工，
    # 猜一个类别反而会让错误的分类混进 BOM。
    if not candidates:
        reason_codes = ["NO_CATEGORY_EVIDENCE"]
        # 两个具体原因分别标出：没有 Solid 与没有包围盒，
        # 对应完全不同的补救方式（检查模型 / 检查量测）。
        if counts.get("SOLID", 0) == 0:
            reason_codes.append("NO_SOLID_GEOMETRY")
        if not dimensions or max(dimensions) <= 0.0:
            reason_codes.append("MISSING_BOUNDING_BOX")
        return {
            "category": "未确定",
            "confidence": 0.0,
            "source": "INSUFFICIENT_GEOMETRY",
            # 即使没有候选，几何证据也照常带出：
            # 复核者需要知道"为什么判不出来"，空证据等于无从下手。
            "evidence": list(geometry_evidence),
            "candidates": [],
            "part_category_group": None,
            "review_required": True,
            # dict.fromkeys 去重保序：原因码可能多处重复产生，
            # 保序保证同一输入得到稳定的输出。
            "review_reason_codes": list(
                dict.fromkeys([*reason_codes, *review_reason_codes])
            ),
        }

    ordered = sorted(
        candidates.values(),
        # 按置信度降序、同分按类别名升序：
        # 名称作次级键保证排序稳定，
        # 否则字典顺序变化会让"主类别"在两次运行间抖动。
        key=lambda item: (-float(item["confidence"]), str(item["category"])),
    )
    primary = ordered[0]
    sources = list(primary["sources"])
    # 多个来源说明名称规则与几何规则都指向了同一个类别，
    # 这比只有单一来源更可信，来源字段要如实反映。
    if len(sources) > 1:
        source = "NAME_AND_GEOMETRY_RULE"
    else:
        source = sources[0] if sources else "GEOMETRY_RULE"
    # 三个原因码三选一：同时有两方证据、只有名称、
    # 只有几何。它们对应完全不同的可信度，
    # 复核者据此决定要不要自己再确认一遍。
    if "NAME_RULE" in sources and "GEOMETRY_RULE" in sources:
        review_reason_codes.append("NAME_GEOMETRY_AGREEMENT")
    elif "NAME_RULE" in sources:
        review_reason_codes.append("NAME_RULE_ONLY")
    else:
        review_reason_codes.append("GEOMETRY_ONLY")
    if len(ordered) > 1:
        review_reason_codes.append("MULTIPLE_CATEGORY_CANDIDATES")
    # 语义未定无条件追加：即便选出的是唯一且高置信的类别，
    # 这套类别仍不是工程确认，必须始终留这一条。
    review_reason_codes.append("SEMANTICS_UNRESOLVED")
    serialized = [
        {
            "category": str(item["category"]),
            # 置信度统一保留 6 位：所有候选同一精度，
            # 便于在表上直接比较排名相邻的两者。
            "confidence": round(float(item["confidence"]), 6),
            "sources": list(item["sources"]),
            # 证据去重：同一个类别可能由多条规则各追加一条说明。
            "evidence": list(dict.fromkeys(item["evidence"])),
        }
        for item in ordered
    ]
    # 先算大类与强度，再进返回字典：ordered 已按置信度排好序，
    # 第 0 项就是主类别，强度判定以它的来源为准。
    (
        part_category_group,
        part_category_group_strength,
    ) = canonical_part_category_from_candidates(serialized)
    return {
        "category": str(primary["category"]),
        "confidence": round(float(primary["confidence"]), 6),
        "source": source,
        # 主类别没有证据时退回几何证据：
        # 至少保留"这个结论有几何上的依据"，
        # 留空会让人以为它没有任何来源。
        "evidence": list(dict.fromkeys(primary["evidence"])) or list(geometry_evidence),
        # 全部候选都输出：只给主类别，
        # 复核者就无法判断"第二名差多少"。
        "candidates": serialized,
        # 我按证据强度选部件大类。这里不能用"所有候选求交集"的老做法：
        # 名称规则命中时几何规则会无条件补三个弱候选，交集恒为空，
        # 实测会让唯一有强证据的 11 件（Cap Screw / Ejector / Runner
        # System）全部掉成 None，而 158 个纯几何兜底件反倒能归上——
        # 恰好把可信度搞反。
        "part_category_group": part_category_group,
        # 强度一并返回：让复核者能区分"名称命中"与"几何兜底"，
        # 两者不该在表上看起来一样可信。
        "part_category_group_strength": part_category_group_strength,
        "review_required": True,
        "review_reason_codes": list(dict.fromkeys(review_reason_codes)),
    }


def _geometry_category_candidates(
    dimensions: list[float] | None,
    counts: dict[str, int],
    surface_area: float | None,
    volume: float | None,
    surface_type_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """我只用有限几何量给出弱类别候选，不把比例直接当成最终类别。"""

    # 原理说明：几何规则只用三类可稳定计算的量——长宽厚比例、圆柱面占
    # Face 的比例、以及 Solid 是否存在：
    #   细长回转（比例 >= 8 且圆柱面占比 >= 0.4）-> 导向/顶出/紧固弱候选；
    #   板状块状（比例 <= 3）-> 模架/辅助结构弱候选；
    #   其余但有表面积体积 -> 辅助结构兜底弱候选。
    # 这些阈值是工程经验，不是标注集标定结果，所以一律给弱置信度并要求复核。
    # 边界：没有 Solid、没有包围盒、或尺寸退化时直接返回空候选，不硬猜。

    evidence: list[str] = []
    # 尺寸缺失或非正直接返回空候选：长宽厚是下面所有几何规则的输入，
    # 没有它任何长径比判断都无从谈起，此时给"未分类"而非猜一个类别。
    if not dimensions or max(dimensions) <= 0.0:
        return [], evidence, []
    # 降序排列后，ordered[0] 是最大边、ordered[-1] 是最小边，
    # 长径比与"板/轴/块"的判定全靠这两个值。
    ordered = sorted(dimensions, reverse=True)
    if counts.get("SOLID", 0) == 0:
        return [], ["我没有读取到 Solid。"], []
    # 分母兜 1e-12：厚度趋近 0 的退化件会得到天文数字的长径比，
    # 兜底能让它落到一个有限的大值而不是 inf。
    aspect = ordered[0] / max(ordered[-1], 1e-12)
    face_count = max(int(counts.get("FACE", 0)), 0)
    # 圆柱面占比用于区分"轴"与"板"：同样细长的零件，
    # 圆柱面占多数才可能是回转体（导向件/顶出件），否则多是板材或镶件。
    cylinder_count = int(surface_type_counts.get("GeomAbs_Cylinder", 0))
    cylinder_ratio = cylinder_count / face_count if face_count else 0.0
    evidence.append(f"我计算到尺寸长宽厚比例为 {aspect:.3f}。")
    if cylinder_count:
        evidence.append(
            f"我统计到 {cylinder_count} 个圆柱面，占 Face 比例 {cylinder_ratio:.3f}。"
        )
    candidates: list[dict[str, Any]] = []
    codes: list[str] = []
    # 细长判据：长径比 >= 8 且最小边为正。
    # 置信度刻意压在 0.35 —— 几何规则只能给出弱候选，
    # 最终归类必须靠名称或主数据确认，不能让甲方以为这是确定结论。
    if aspect >= 8.0 and ordered[-1] > 0.0:
        evidence.append("我得到细长轴类几何，可能是导向件、顶出件或紧固件。")
        candidates.append(
            {
                "category": "细长轴类候选",
                "confidence": 0.35,
                "sources": ["GEOMETRY_RULE"],
                "evidence": list(evidence),
            }
        )
        if cylinder_ratio >= 0.4:
            # 三个候选按置信度从高到低给出，且不互相排斥：
            # 导向件、顶出件、紧固件在几何上高度相似，
            # 用互斥分类会强行抹掉真实存在的歧义。
            for category, confidence in (
                ("导向件候选", 0.45),
                ("顶出件候选", 0.4),
                ("紧固件候选", 0.3),
            ):
                candidates.append(
                    {
                        "category": category,
                        "confidence": confidence,
                        "sources": ["GEOMETRY_RULE"],
                        "evidence": [
                            "我根据细长回转几何给出该类别弱候选，仍需名称或主数据确认。"
                        ],
                    }
                )
        codes.append("GEOMETRY_ONLY")
        return candidates, evidence, codes
    # 长径比 <= 3 判为板状/块状：这是与"细长"相对的另一档，
    # 两个分支之间（3~8 之间）不给候选——那个区间的几何既不像轴
    # 也不像板，硬给一个类别只是制造噪声。
    if aspect <= 3.0:
        evidence.append("我得到板状或块状几何，可能是模架件或辅助结构件。")
        candidates.append(
            {
                "category": "板件/结构件候选",
                # 三个候选置信度都压在 0.25~0.3：包围盒只能区分"扁/长"，
                # 区分不了模板、垫板、支撑板这些具体件，
                # 给高置信度等于误导甲方直接采信。
                "confidence": 0.3,
                "sources": ["GEOMETRY_RULE"],
                # 这条用累积的 evidence（含前面算出的长径比），
                # 让复核者能看到判定所依据的具体数值。
                "evidence": list(evidence),
            }
        )
        candidates.append(
            {
                "category": "模架件候选",
                "confidence": 0.3,
                "sources": ["GEOMETRY_RULE"],
                # 这条只给一句固定的弱候选说明：模架件在几何上没有
                # 区别于其他板件的特征，列出具体数值反而暗示它有依据。
                "evidence": ["我根据板状包围盒给出模架件弱候选，仍需名称或主数据确认。"],
            }
        )
        candidates.append(
            {
                "category": "辅助结构件候选",
                "confidence": 0.25,
                # 来源标记 GEOMETRY_RULE：这两个候选都只靠几何规则得出，
                # 与靠名称或主数据得出的候选可信度不同，
                # 必须能区分。
                "sources": ["GEOMETRY_RULE"],
                "evidence": ["我根据板状包围盒给出辅助结构件弱候选。"],
            }
        )
        # GEOMETRY_ONLY 原因码：整条结论只靠几何，
        # 需要名称或主数据来确认。
        codes.append("GEOMETRY_ONLY")
        return candidates, evidence, codes
    # 有表面积与体积时给出更低置信度（0.2）的兜底：
    # 这两个量只说明"这是一个实体零件"，
    # 连"是不是板状"都看不出来，所以置信度比上面的 0.25 更低。
    if surface_area is not None and volume is not None and volume > 0.0:
        # 体积必须为正：零体积说明几何退化，
        # 用它推"这是一个零件"没有依据。
        candidates.append(
            {
                "category": "通用模具零件候选",
                "confidence": 0.2,
                "sources": ["GEOMETRY_RULE"],
                "evidence": ["我根据有效表面积和体积保留通用模具零件候选。"],
            }
        )
        candidates.append(
            {
                "category": "辅助结构件候选",
                "confidence": 0.2,
                "sources": ["GEOMETRY_RULE"],
                "evidence": ["我保留辅助结构件兜底候选，等待名称或主数据确认。"],
            }
        )
        codes.append("GEOMETRY_ONLY")
        return candidates, evidence, codes
    # 连表面积与体积都没有就一个候选都不给：
    # 没有任何几何依据时不编造类别，
    # 给一个"默认类别"会让人以为它有来源。
    return [], evidence, []


def _category_candidate(
    part_name: str,
    dimensions: list[float] | None,
    counts: dict[str, int],
    surface_area: float | None,
    volume: float | None,
) -> tuple[str, float, str, list[str]]:
    """我保留单值入口，内部委托给多候选类别决策。"""

    decision = _category_decision(
        part_name,
        dimensions,
        counts,
        surface_area,
        volume,
    )
    return (
        str(decision["category"]),
        float(decision["confidence"]),
        str(decision["source"]),
        list(decision["evidence"]),
    )


def _matches_name_token(name: str, token: str) -> bool:
    """我避免英文关键词的子串误报，同时保留中文工程词的连续匹配。"""

    # 原理说明：英文关键词必须按词边界匹配，否则 "core" 会命中
    # "scoreboard"、"screw" 会命中 "screwdriver_holder" 这类无关词；
    # 中文没有天然词边界，工程词（导柱、顶针、滑块）本身就是连续子串，
    # 所以中文走连续包含匹配。这个差异是刻意的，不是漏写边界。

    if any("\u4e00" <= character <= "\u9fff" for character in token):
        return token in name
    return re.search(
        rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])",
        name,
        flags=re.IGNORECASE,
    ) is not None


def _material_name(label: Any, material_tool: Any, ascii_string_type: Any) -> str | None:
    """我只读取 XCAF 已保存的材料名称，不从几何外观猜测材料。

    "不从外观猜"是这条规则的实质：颜色、光泽这类外观属性与材料
    之间没有可靠映射，灰色的件可能是钢也可能是喷了漆的铝。
    猜出来的材料一旦进 BOM 就会被拿去询价，报错的代价远高于留空。

    先取 name 再退回 description：XCAF 里 name 常是材料代号
    （例如 S45C），description 才是可读名。代号信息密度更高，
    所以优先取它；两者都为空才返回 None。

    整段包在 try 里：GetMaterial_s 在没有材料的 label 上会抛异常，
    "这个零件没标材料"是正常情况而不是错误。
    """

    try:
        name = ascii_string_type()
        description = ascii_string_type()
        density_name = ascii_string_type()
        density_type = ascii_string_type()
        result = material_tool.GetMaterial_s(
            label,
            name,
            description,
            0.0,
            density_name,
            density_type,
        )
        if not result:
            return None
        value = name.ToCString().strip() or description.ToCString().strip()
        return value or None
    except Exception:
        return None


def _apply_master_data(
    components: list[BomComponent],
    master_data: BomMasterData | None,
) -> int:
    """我只应用精确主数据匹配，并为每个覆盖字段保留来源。

    为什么只做**精确**匹配：模糊匹配会把"M6 螺钉"和"M8 螺钉"配到
    同一条主数据上，写错采购规格比不写更糟——不写的地方
    在 BOM 里是空白（明显要人工补），写错的地方看起来是完整的，
    会一路错到采购。

    匹配不上就跳过（continue）而不是记 warning：主数据本来就
    只覆盖标准件，非标成型件匹配不上是常态。若每个不匹配的
    零件都记一条 warning，警告列表会被几百条正常情况淹没，
    真正需要看的复核提示反而找不到了。
    """

    if master_data is None:
        return 0
    match_count = 0
    for component in components:
        record = master_data.match(component.reference_shape_id, component.part_name)
        if record is None:
            continue
        _apply_master_record(component, record)
        match_count += 1
    return match_count


def _apply_master_record(component: BomComponent, record: BomMasterRecord) -> None:
    """我把一条主数据记录的字段覆盖到组件上，并为每个覆盖留下来源证据。

    逐字段判空后覆盖，而不是整体替换：主数据通常只提供
    "名称/类别/材料/采购规格"中的一部分，整体替换会用空值
    抹掉几何解析已经得到的有效信息。

    主数据确认类别时我同时做三件事，缺一就会自相矛盾：
      - confidence 置 1.0、source 标 MASTER_DATA；
      - category_candidates 替换成单元素列表。保留原来的多候选
        会与"已确认"冲突，下游会看到"确定但仍有候选"；
      - review_required 置 False，并把原因码换成
        MASTER_DATA_CONFIRMED，让"为什么不用复核"可追溯。

    part_category_group 必须跟着重算：类别变了但大类还是旧值，
    输出就会出现"类别属于成型件、大类却是模架件"的矛盾。
    """
    if record.part_name:
        component.part_name = record.part_name
    if record.category:
        component.category = record.category
        component.category_confidence = 1.0
        component.category_source = "MASTER_DATA"
        component.category_evidence.append("我通过精确 BOM 主数据匹配确认了零件类别。")
        component.category_candidates = [
            {
                "category": record.category,
                "confidence": 1.0,
                "sources": ["MASTER_DATA"],
                "evidence": ["我通过精确 BOM 主数据匹配确认了零件类别。"],
            }
        ]
        component.category_review_required = False
        component.category_review_reason_codes = ["MASTER_DATA_CONFIRMED"]
        # 主数据确认类别后，部件大类也必须跟着收敛到任务书的 8 个大类。
        component.part_category_group = canonical_part_category([record.category])
        # 主数据是最高档证据：只要映射得上就标 named，
        # 与"几何兜底"的 geometry 档区分开。
        component.part_category_group_strength = (
            "named" if component.part_category_group else "none"
        )
    if record.material:
        component.material = record.material
        component.material_source = "MASTER_DATA"
    if record.purchase_specification:
        component.purchase_specification = record.purchase_specification
        component.purchase_specification_source = "MASTER_DATA"
    component.remarks += " 我方已应用精确匹配的 BOM 主数据；未提供的字段仍保持原始状态。"


def _category_candidate_counts(components: list[BomComponent]) -> dict[str, int]:
    """我汇总所有零部件的类别候选，便于后续统计和人工复核排序。"""

    counts: dict[str, int] = {}
    for component in components:
        seen: set[str] = set()
        for candidate in component.category_candidates:
            category = candidate.get("category")
            if not isinstance(category, str) or category in seen:
                continue
            seen.add(category)
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))
