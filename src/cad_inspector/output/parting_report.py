"""我方分型结果的 JSON、CSV 和 STL 输出。

算法原理：一次分型任务的结果分三类载体，各管一件事：

- JSON 是权威结果，含完整诊断与警告，供机器读取和后续追溯；
- CSV 是给人扫读和 Excel 打开的表格视图，分型线点和尖角点逐行落盘，方便甲方直接用 Excel 核对；
- STL 是给可视化和第三方软件看的网格，只描述形状、不承载语义。

所有可选产物（分型线 CSV、候选 JSON、分型面 STL）都先判断对应字段是否存在再写，
缺什么就不产出什么，绝不写一个内容为空的文件去冒充成果。写出统一走 atomic_path
原子替换，CSV 用 utf-8-sig 带 BOM 保证中文不乱码。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.dataset import DatasetManifest
from ..domain.parting import PartingPipelineResult
from ..geometry.mesh import TriangleMesh
from ..geometry.stl import save_stl
from .atomic import atomic_path


def write_parting_result(
    result: PartingPipelineResult,
    output_dir: str | Path,
    additional_artifacts: dict[str, str] | None = None,
) -> dict[str, str]:
    """我把一次分型基线任务写成可追溯的多个输出文件。

    文件名用输入文件的 stem 加固定后缀，而不是序号或时间戳：
    同一份输入重复运行得到的是同一个路径，覆盖式更新符合
    "重跑应当更新结论"的预期；用时间戳会堆出一堆同名不同时的文件，
    反而让人分不清哪个是最新结论。

    allow_nan=False 在这里尤其重要：分型线坐标里出现 NaN 通常意味着
    上游求交失败。让它静默写进 JSON 会产出一个"能打开但坐标是空"
    的文件，下游可视化会画出乱线；宁可写出失败并暴露问题。

    返回值是"逻辑名 -> 路径"的字典，调用方据此拿到产物清单，
    不必去猜文件名，将来改命名规则也不影响调用方。
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = Path(result.input_file).stem
    json_path = output_path / f"{stem}.parting.json"
    extra_artifacts = dict(additional_artifacts or {})
    payload = result.to_dict()
    if extra_artifacts:
        payload["artifacts"] = extra_artifacts
    with atomic_path(json_path) as temporary:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    # 有分型线才写点集 CSV，没有就不产出空文件。
    artifacts = {"json": str(json_path), **extra_artifacts}
    if result.line is not None:
        line_path = output_path / f"{stem}.parting_line.csv"
        _write_points_csv(line_path, result.line.points_xyz_mm)
        artifacts["parting_line_csv"] = str(line_path)
        sharp_path = output_path / f"{stem}.sharp_corners.csv"
        sharp_corners = result.line.diagnostics.get("sharp_corners", [])
        _write_sharp_corners_csv(
            sharp_path,
            sharp_corners if isinstance(sharp_corners, list) else [],
        )
        artifacts["sharp_corners_csv"] = str(sharp_path)

    # 有多条候选时才单独落一份候选 JSON，便于人工比对为什么选了当前这条。
    if result.line_candidates:
        candidates_path = output_path / f"{stem}.parting_candidates.json"
        with atomic_path(candidates_path) as temporary:
            temporary.write_text(
                json.dumps(
                    [
                        candidate.to_dict()
                        for candidate in result.line_candidates
                    ],
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        artifacts["parting_candidates_json"] = str(candidates_path)

    # 有分型面网格才写 STL，供第三方软件可视化核对。
    if result.surface is not None and result.surface.vertices_xyz_mm:
        mesh = TriangleMesh(
            vertices=np.asarray(result.surface.vertices_xyz_mm, dtype=np.float64),
            faces=np.asarray(result.surface.faces, dtype=np.int64),
            name=f"{stem}_parting_surface",
            metadata={"format": "mesh_baseline", "unit": "mm"},
        )
        surface_path = output_path / f"{stem}.parting_surface.stl"
        save_stl(mesh, surface_path, solid_name="parting_surface_baseline")
        artifacts["parting_surface_stl"] = str(surface_path)
    return artifacts


def write_manifest(
    manifest: DatasetManifest,
    output_path: str | Path,
) -> Path:
    """我把数据清单保存成 JSON，同时生成适合表格查看的 CSV。"""

    target = Path(output_path)
    with atomic_path(target) as temporary:
        temporary.write_text(
            json.dumps(
                manifest.to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    csv_path = target.with_suffix(".csv")
    fieldnames = [
        "sample_id",
        "path",
        "name",
        "extension",
        "size_bytes",
        "sha256",
        "detected_format",
        "dataset_split",
        "label_status",
        "probe_status",
        "notes",
        "entity_counts",
    ]
    # notes 是列表，CSV 单元格里用竖线拼成单行文本。
    with atomic_path(csv_path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in manifest.records:
                row: dict[str, Any] = record.to_dict()
                row["notes"] = " | ".join(record.notes)
                # entity_counts 是 dict，直接进 CSV 会炸
                # （实测 dict contains fields not in fieldnames）。
                # 我把它摊平成 k=v 文本，保证 CSV 可读又不丢信息。
                row["entity_counts"] = " ".join(
                    f"{k}={v}"
                    for k, v in sorted(dict(record.entity_counts).items())
                )
                writer.writerow(row)
    return target


def _write_points_csv(path: Path, points: list[list[float]]) -> None:
    """我把三维点集写成带 x/y/z 毫米表头的一行一点 CSV。"""
    with atomic_path(path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x_mm", "y_mm", "z_mm"])
            writer.writerows(points)


def _write_sharp_corners_csv(
    path: Path,
    corners: list[dict[str, Any]],
) -> None:
    """我把尖角索引、转角和毫米坐标写成独立 CSV。"""

    with atomic_path(path) as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["index", "turn_angle_deg", "x_mm", "y_mm", "z_mm"]
            )
            for corner in corners:
                point = corner.get("point_xyz_mm")
                if not isinstance(point, list) or len(point) != 3:
                    continue
                writer.writerow(
                    [
                        corner.get("index"),
                        corner.get("turn_angle_deg"),
                        *point,
                    ]
                )
