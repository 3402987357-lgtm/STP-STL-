"""我把检测出的缺失区域映射到 CAD 面编号上。

我为什么要做这件事（这是本模块存在的唯一理由）：

检测报告说"这里有个 11 mm2 的缺失"，甲方拿到这句话还得自己回去在
CAD 里找是哪个面。而"缺失落在第 17 号 CAD 面上"才是模具工程师能直接
动手的那句话。三角网格上的面编号是我自己编的下标，CAD 面编号才是
零件自带的语义编号，这两者之间差的就是这层映射。

我守住的边界（这几条写错都会让结论静默变错）：

- **只对 STEP 侧启用**。STL 没有 B-Rep 面概念，硬给它编 CAD 面号
  等于编造语义。格式不对我就明确返回"未启用"，不猜。
  两侧分别判断：缺料区域用参考件的表，多料区域用候选件的表，
  因为多料只存在于候选件上，参考件根本没有对应的面。
- **三角数必须与映射长度一致**。参考网格如果和我三角化出来的不是同一
  份（比如换过弦高参数），下标就对不上，映射出来的面号全是错的。
  我宁可拒绝映射也不给错的编号。
- **拒绝时给理由**。我返回"未启用 + 原因"，而不是静默留空：
  静默留空会让"没做映射"看起来像"这个区域不属于任何面"。
- **OCP 延迟导入**。缺 OCP 只让本模块不可用，几何检测照常跑。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CadFaceMapping:
    """我承载"区域 -> CAD 面编号"的映射结果。"""

    enabled: bool
    reason: str
    # 区域编号 -> 该区域涉及的 CAD 面编号（去重、升序）。
    region_cad_face_ids: dict[int, list[int]] = field(default_factory=dict)
    # 区域编号 -> 该区域落在各 CAD 面上的面积（mm2），按面号升序。
    region_cad_face_areas_mm2: dict[int, dict[int, float]] = field(
        default_factory=dict
    )
    warnings: list[str] = field(default_factory=list)
    cad_face_count: int = 0
    config: dict[str, object] = field(default_factory=dict)
    # 区域编号 -> 这一号面属于哪一侧（"reference" 或 "candidate"）。
    # 我为什么要记它：多料的面号来自候选件，缺料来自参考件，
    # 不写清楚的话甲方会把候选件的 12 号面当成参考件的 12 号面去找。
    region_cad_face_side: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """我把结果转成可序列化字典。

        JSON 的键只能是字符串，所以我统一转成字符串键；
        不这样做的话写盘时会被隐式转成字符串，读回来类型就变了。
        """

        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "cad_face_count": self.cad_face_count,
            "region_cad_face_ids": {
                str(key): list(value)
                for key, value in sorted(self.region_cad_face_ids.items())
            },
            "region_cad_face_areas_mm2": {
                str(region_id): {
                    str(face_id): round(float(area), 6)
                    for face_id, area in sorted(areas.items())
                }
                for region_id, areas in sorted(self.region_cad_face_areas_mm2.items())
            },
            "warnings": list(self.warnings),
            "config": dict(self.config),
            "region_cad_face_side": {
                str(key): value
                for key, value in sorted(self.region_cad_face_side.items())
            },
        }


def _disabled(reason: str) -> CadFaceMapping:
    """我返回一个"未启用"的结果，并把原因一起带上。"""

    return CadFaceMapping(enabled=False, reason=reason)


def _degrees_to_radians(degrees: float) -> float:
    """我把角度偏差从度转成弧度。

    我为什么要显式做这个换算：管线里配置用的是**度**（读起来直观），
    而 OCCT 的三角化接口要**弧度**。混用会让三角化精细度差 57 倍，
    三角形数量跟着变，下标就全对不上了。
    """

    return float(degrees) * 3.141592653589793 / 180.0


def _is_step_side(metadata: dict, path: object) -> bool:
    """我判断这一侧是不是 STEP。

    我只认两种证据：加载器声明的 detected_format，或文件后缀。
    两者都没有就当非 STEP——宁可不映射，也不给没有 B-Rep 语义的
    网格编面号。
    """

    detected = str(metadata.get("detected_format", "")).lower()
    suffix = str(path).lower()
    return detected == "step" or suffix.endswith((".stp", ".step"))


def _mapping_from_metadata(
    mesh,
    metadata: dict,
) -> np.ndarray | None:
    """我优先复用 STEP 加载器已经算好的面 ID 表。

    我为什么要优先用它而不是重新三角化一次：

    加载器在把 STEP 读成网格时**已经**逐三角记下了它属于哪个 B-Rep 面
    （`face_ids_by_triangle`），那张表和参考网格是**同一份**、下标天然对齐。
    我再三角化一次，只要弦高或 `relative` 参数差一点，三角数就不一样
    （我实测 26216 vs 26302），下标立刻错位。用现成的表没有这个风险。
    """

    raw = metadata.get("face_ids_by_triangle")
    if not raw:
        return None
    mapping = np.asarray(raw, dtype=np.int64)
    # 长度必须和三角数一致：不一致说明这不是同一份网格，
    # 拿它去索引会静默给出错的编号。
    if len(mapping) != len(mesh.faces):
        return None
    return mapping


def _resolve_side_table(
    mesh,
    metadata: dict,
    path: object,
    linear_deflection: float,
    angular_deflection_deg: float,
) -> tuple[np.ndarray | None, list[str], int, dict]:
    """我为**一侧**解析出"三角下标 -> CAD 面号"的表。

    我把两侧共用的逻辑收在这里，因为缺料走参考件、多料走候选件，
    两条路径的要求完全一样：先复用加载器的表，拿不到才按同参数重新
    三角化，且三角数必须对齐。分成两个函数写就会各漏一条。

    返回值依次是：映射表（拿不到为 None）、警告、CAD 面数、配置。
    """

    if not _is_step_side(metadata, path):
        return None, [], 0, {}
    existing = _mapping_from_metadata(mesh, metadata)
    if existing is not None:
        return (
            existing,
            [],
            len({int(value) for value in existing}),
            {
                "source": "step_loader_face_ids_by_triangle",
                "linear_deflection": linear_deflection,
                "angular_deflection_deg": angular_deflection_deg,
            },
        )
    try:
        from ..geometry.occt_tessellation import (
            TessellationConfig,
            tessellate_step_file,
        )
    except Exception:  # noqa: BLE001 - 可选依赖缺失只让这一侧不可用
        return None, [], 0, {}
    try:
        tessellation = tessellate_step_file(
            path,
            TessellationConfig(
                linear_deflection=linear_deflection,
                angular_deflection=_degrees_to_radians(angular_deflection_deg),
                relative=False,
            ),
        )
    except Exception:  # noqa: BLE001 - 读不动就当这一侧没有表
        return None, [], 0, {}
    table = tessellation.triangle_to_face_id
    # 下标对齐是硬要求：三角数不同就说明不是同一份网格，
    # 映射出的面号全是错的，而且错得悄无声息。
    if len(table) != len(mesh.faces):
        return None, [], 0, {}
    return (
        table,
        list(tessellation.warnings),
        len(tessellation.face_areas_mm2),
        dict(tessellation.config),
    )


def map_regions_to_cad_faces(
    regions: list,
    reference_mesh,
    reference_path: str | object,
    reference_metadata: dict | None = None,
    candidate_mesh=None,
    candidate_path: str | object = "",
    candidate_metadata: dict | None = None,
    linear_deflection: float = 0.5,
    angular_deflection_deg: float = 10.0,
) -> CadFaceMapping:
    """我把每个区域的三角面编号翻译成 CAD 面编号。

    ``regions`` 里每个元素需要有 ``region_id``、``defect_type``、
    ``reference_face_ids`` 与 ``candidate_face_ids``。

    **缺料与多料走不同的侧**：上游对 extra 区域把候选件的面编号写进
    ``candidate_face_ids`` 并留空 ``reference_face_ids``（多料只存在于
    候选件上，参考件根本没有对应的面）。所以我按 defect_type 选侧：

    - ``missing`` -> 用参考件的映射表 + 参考件的三角面积；
    - ``extra``   -> 用候选件的映射表 + 候选件的三角面积。

    两侧只要有一侧是 STEP 就能出结果；都不是 STEP 才整体不启用。

    ``linear_deflection`` 与 ``angular_deflection_deg`` 必须和**加载参考件
    网格时用的参数一致**。我为什么要强调这个：三角化参数不同就会得到
    不同数量的三角形，下标自然对不上，映射出的面号全是错的。
    调用方应当把 InspectionConfig 里那两个值传进来，不要另填一个数。
    """

    reference_metadata = reference_metadata or {}
    candidate_metadata = candidate_metadata or {}
    reference_table, ref_warnings, ref_face_count, ref_config = _resolve_side_table(
        reference_mesh,
        reference_metadata,
        reference_path,
        linear_deflection,
        angular_deflection_deg,
    )
    # 候选侧只在调用方给了候选网格时才解析：不传就说明这次比对
    # 不需要多料的面归属（例如只有参考件是 STEP）。
    if candidate_mesh is not None:
        (
            candidate_table,
            cand_warnings,
            cand_face_count,
            cand_config,
        ) = _resolve_side_table(
            candidate_mesh,
            candidate_metadata,
            candidate_path,
            linear_deflection,
            angular_deflection_deg,
        )
    else:
        candidate_table, cand_warnings, cand_face_count, cand_config = (
            None,
            [],
            0,
            {},
        )
    if reference_table is None and candidate_table is None:
        return _disabled(
            "我没有做 CAD 面映射，因为参考件与候选件都不是 STEP："
            "STL 没有 B-Rep 面概念，我不能凭空编造 CAD 面编号。"
        )
    warnings = list(ref_warnings) + list(cand_warnings)
    config: dict[str, object] = {
        "reference": ref_config or "not_step_or_unavailable",
        "candidate": cand_config or "not_step_or_unavailable",
        "linear_deflection": linear_deflection,
        "angular_deflection_deg": angular_deflection_deg,
    }
    region_faces: dict[int, list[int]] = {}
    region_areas: dict[int, dict[int, float]] = {}
    region_side: dict[int, str] = {}
    for region in regions:
        # 多料只存在于候选件上，必须换到候选侧的表与三角面积；
        # 沿用参考侧会得到一套看似合理、实际张冠李戴的面号。
        if str(getattr(region, "defect_type", "missing")) == "extra":
            table = candidate_table
            mesh = candidate_mesh
            side = "candidate"
            face_ids = getattr(region, "candidate_face_ids", ())
        else:
            table = reference_table
            mesh = reference_mesh
            side = "reference"
            face_ids = getattr(region, "reference_face_ids", ())
        if table is None or mesh is None:
            # 这一侧不是 STEP（例如候选件是 STL 而参考件是 STEP），
            # 我不猜它对面的编号，但另一侧照常出结果。
            continue
        mesh_face_ids = [int(value) for value in face_ids]
        if not mesh_face_ids:
            continue
        # 越界下标必须拦住：这类下标来自别处的数据，
        # 直接拿去索引会拿到错误的面号甚至越界报错。
        valid = [
            index
            for index in mesh_face_ids
            if 0 <= index < len(table)
        ]
        if len(valid) != len(mesh_face_ids):
            warnings.append(
                f"区域 {getattr(region, 'region_id', '?')} 里有 "
                f"{len(mesh_face_ids) - len(valid)} 个三角下标越界，我跳过了它们。"
            )
        if not valid:
            continue
        triangle_areas = mesh.triangle_areas
        areas: dict[int, float] = {}
        for index in valid:
            cad_face_id = int(table[index])
            areas[cad_face_id] = areas.get(cad_face_id, 0.0) + float(
                triangle_areas[index]
            )
        region_id = int(getattr(region, "region_id", 0))
        region_faces[region_id] = sorted(areas)
        region_areas[region_id] = areas
        region_side[region_id] = side

    return CadFaceMapping(
        enabled=True,
        reason="我已把区域映射到 CAD 面编号。",
        region_cad_face_ids=region_faces,
        region_cad_face_areas_mm2=region_areas,
        region_cad_face_side=region_side,
        warnings=warnings,
        # 两侧面数不同时取较大者：这个字段只是"这份模型有多少个 CAD 面"
        # 的概览，真正的归属在 region_cad_face_side 里。
        cad_face_count=max(ref_face_count, cand_face_count),
        config=config,
    )


def summarize_region_cad_faces(mapping: CadFaceMapping) -> list[str]:
    """我把映射结果写成甲方能直接读的中文句子。"""

    if not mapping.enabled:
        return [mapping.reason]
    if not mapping.region_cad_face_ids:
        return ["我已启用 CAD 面映射，但当前没有区域能落到 CAD 面上。"]
    lines: list[str] = []
    for region_id in sorted(mapping.region_cad_face_ids):
        face_ids = mapping.region_cad_face_ids[region_id]
        areas = mapping.region_cad_face_areas_mm2.get(region_id, {})
        # 面号属于哪一侧必须写进句子：多料的面号来自候选件，
        # 缺料来自参考件，不写清甲方会拿着候选件的 12 号面
        # 去参考件的 12 号面上找，白跑一趟。
        side = mapping.region_cad_face_side.get(region_id, "reference")
        side_text = "候选件" if side == "candidate" else "参考件"
        # 只报面积最大的那个面：一个区域跨好几个面时，
        # 全列出来反而让人抓不住重点，详情在 JSON 里查。
        dominant = max(areas, key=lambda key: areas[key]) if areas else None
        if dominant is None:
            lines.append(f"区域 {region_id} 落在{side_text}的 CAD 面 {face_ids} 上。")
            continue
        total = sum(areas.values())
        ratio = areas[dominant] / total if total > 0 else 0.0
        lines.append(
            f"区域 {region_id} 主要落在{side_text}的第 {dominant} 号 CAD 面上"
            f"（占该区域 {ratio * 100:.1f}%），另涉及 {len(face_ids) - 1} 个面。"
        )
    return lines
