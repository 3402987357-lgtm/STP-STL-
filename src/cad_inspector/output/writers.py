"""我把检测结果写成便于程序读取和人工查看的文件。

三种载体各管一件事：JSON 是权威结果（含完整配置与区域明细，供机器追溯）、CSV 是
表格视图（供甲方用 Excel 扫读核对）、彩色 PLY 是可视化（红蓝着色直接看异常面）。

两个容易踩的坑在这里显式处理：CSV 里的列表字段（坐标、编号集合、类别候选）一律用
JSON 序列化成单元格文本，避免 Python 的列表 repr 混进 CSV 后被 Excel 当分隔符拆列；
彩色 PLY 为每个三角面复制一份顶点，这样同一顶点在不同面上可以着不同颜色，不会因为
共享顶点导致红蓝串色。JSON 与 CSV 都禁止 NaN（``allow_nan=False``），非有限值直接
报错而不是写出打不开的文件。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..geometry.mesh import TriangleMesh
from ..models import InspectionResult
from .atomic import atomic_path


def write_json(result: InspectionResult, path: str | Path) -> Path:
    """我输出完整的、可复现的 JSON 结果。"""

    path = Path(path)
    # allow_nan=False：非有限值直接报错，绝不写出违反 JSON 规范的文件。
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return path


def write_csv(result: InspectionResult, path: str | Path) -> Path:
    """我把区域级结果写成表格，方便甲方用 Excel 检查。

    列顺序写死而不是从数据里推导：不同批次、不同区域集合推导出的列序
    可能不同（某个字段只在部分区域出现时就会被排到不同位置），
    两份 CSV 就无法横向比对。固定列序让"这一列永远是 area_mm2"
    成为可靠约定。

    列表字段（坐标、编号集合、类别候选）一律用 JSON 序列化成单元格文本。
    用 Python 的 list repr 会带逗号，CSV 解析时会被当成列分隔符，
    一行数据被劈成好几列，Excel 里看到的是错位的表。
    """

    path = Path(path)
    # 列顺序固定，保证不同批次产出的 CSV 可以直接横向比对。
    fields = [
        "region_id",
        "region_geometry_signature",
        "defect_type",
        "center_xyz_mm",
        "bbox_min_xyz_mm",
        "bbox_max_xyz_mm",
        "area_mm2",
        "estimated_volume_mm3",
        "max_error_mm",
        "mean_error_mm",
        "p95_error_mm",
        "triangle_count",
        "confidence",
        "reference_face_ids",
        "candidate_face_ids",
        "candidate_step_face_ids",
        "reference_step_face_ids",
        "candidate_step_region_ids",
        "reference_step_region_ids",
        "candidate_step_category_candidates",
        "reference_step_category_candidates",
        "review_required",
        "direction_evidence",
        "signed_offset_mean_mm",
        "normal_alignment_mean",
        "direction_valid_sample_count",
        "direction_aligned_sample_count",
        "direction_boundary_sample_count",
    ]
    with atomic_path(path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for region in result.regions:
                row = region.to_dict()
                # 列表字段统一 JSON 序列化，避免 Python repr 里的逗号被当成 CSV 分隔符。
                for key in (
                    "center_xyz_mm",
                    "bbox_min_xyz_mm",
                    "bbox_max_xyz_mm",
                    "reference_face_ids",
                    "candidate_face_ids",
                    "candidate_step_face_ids",
                    "reference_step_face_ids",
                    "candidate_step_region_ids",
                    "reference_step_region_ids",
                    "candidate_step_category_candidates",
                    "reference_step_category_candidates",
                ):
                    row[key] = json.dumps(
                        row[key],
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                row.pop("warnings", None)
                writer.writerow(row)
    return path


def write_colored_ply(
    mesh: TriangleMesh,
    face_distances: np.ndarray,
    threshold_mm: float,
    path: str | Path,
) -> Path:
    """我为每个参考三角面复制顶点并写出红色异常、蓝色正常的 PLY。

    我为什么每个面都复制一份顶点、而不是共享顶点：PLY 的顶点色是挂在
    顶点上的，共享顶点就意味着"一个点只能有一种颜色"。而异常面与正常
    面往往共边，共享顶点会让红色沿着公共边渗到相邻的正常面上，看的人
    会把着色边界误读成缺陷边界。复制顶点后同一个空间点在不同面上可以
    带不同颜色，红蓝边界严格落在三角面边界上。代价是文件里顶点数是
    面数的三倍，对这个规模完全可以接受。

    我为什么写 ASCII 而不是二进制 PLY：甲方要在 MeshLab 之外也能直接
    用文本编辑器抽查，二进制虽然小一半但不可读；而 PLY 只在可视化环节
    用一次，体积不是瓶颈。
    """

    path = Path(path)
    face_distances = np.asarray(face_distances, dtype=np.float64)
    # 我按"下标是否落在有效距离数组内"取用，而不是直接索引：面数与距离
    # 数组长度在个别退化网格上会不一致（三角化失败的面没有距离），
    # 直接索引会抛 IndexError 让整份可视化写不出来。
    # 缺距离的面我按"未超过阈值"处理，即标蓝不标红——宁可漏标，
    # 也不把没有证据的面标成异常。
    vertices: list[tuple[float, float, float, int, int, int]] = []
    faces: list[tuple[int, int, int]] = []
    for face_id, face in enumerate(mesh.faces):
        abnormal = (
            face_id < len(face_distances)
            and float(face_distances[face_id]) >= threshold_mm
        )
        # 超过阈值的面标红，其余标蓝；阈值即检测阈值，与 JSON 结果口径一致。
        if abnormal:
            color = (220, 35, 35)
        else:
            color = (45, 120, 220)
        # 每面复制三个顶点，让相邻面可以独立着色而不串色。
        base_index = len(vertices)
        for vertex_id in face:
            point = mesh.vertices[int(vertex_id)]
            vertices.append((float(point[0]), float(point[1]), float(point[2]), *color))
        faces.append((base_index, base_index + 1, base_index + 2))
    with atomic_path(path) as temporary:
        with temporary.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(vertices)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            handle.write(f"element face {len(faces)}\n")
            handle.write("property list uchar int vertex_indices\nend_header\n")
            for vertex in vertices:
                handle.write(
                    f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                    f"{vertex[3]} {vertex[4]} {vertex[5]}\n"
                )
            for face in faces:
                handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    return path
