"""我方分型设计基线流程编排。

算法原理：同一个"分型"需求有两条实现路线，本模块按输入格式分派，并把结果包装成同一
个 ``PartingRun``，让上层用一套代码处理两种精度：

- STEP/STP：走 OCP B-Rep 实体截交（``ocp_brep_section``），拿到真实拓扑级分型线，
  同时保留 OCP 线框与平面几何供 CAD 导出；
- STL：走网格水平截面基线（``mesh_horizontal_section``），在没有 OCC 的环境也能跑通。

两条路线的状态判定口径一致：**分型线闭合 + 分型线 BASELINE + 分型面 BASELINE** 三者
同时成立才算整体 BASELINE，否则降级 REVIEW_REQUIRED。任一路线的警告都会汇总上去，
不会因为是"基线"就把警告吞掉。
"""

from __future__ import annotations

from pathlib import Path

from .detection.parting_line import extract_parting_line
from .detection.parting_surface import build_planar_parting_surface
from .detection.step_parting import (
    StepPartingGeometry,
    extract_step_parting_geometry,
)
from .domain.parting import PartingConfig, PartingPipelineResult
from .errors import DependencyMissingError
from .geometry.mesh import TriangleMesh
from .geometry.stl import load_stl
from .io.input_validator import validate_input_path
from .io.step import load_step_as_mesh


def load_mesh_for_parting(
    path: str | Path,
    config: PartingConfig | None = None,
) -> TriangleMesh:
    """我根据扩展名选择当前可用的网格输入适配器。"""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return load_stl(path)
    if suffix in {".stp", ".step"}:
        return load_step_as_mesh(
            path,
            linear_deflection_mm=0.04,
            angular_deflection_deg=10.0,
        )
    raise DependencyMissingError(
        f"我当前只对 STL 提供无额外后端的分型基线，"
        f"{path.suffix or '<无扩展名>'} 需要专用适配器。"
    )


def run_mesh_parting(
    path: str | Path,
    config: PartingConfig | None = None,
) -> PartingPipelineResult:
    """我执行分型线和分型面的最小可验证闭环。"""

    return run_parting(path, config).result


class PartingRun:
    """我保留结构化结果和可选 OCP 几何，供命令行输出 CAD 文件。"""

    def __init__(
        self,
        result: PartingPipelineResult,
        step_geometry: StepPartingGeometry | None = None,
    ) -> None:
        """我保存结构化结果与可选 OCP 几何（网格路线下为 None）。"""
        self.result = result
        self.step_geometry = step_geometry


def run_parting(
    path: str | Path,
    config: PartingConfig | None = None,
) -> PartingRun:
    """我按输入格式选择网格基线或 OCP B-Rep 分型线流程。"""

    config = config or PartingConfig()
    input_path = Path(path).expanduser().resolve()
    input_metadata = validate_input_path(input_path).to_dict()
    # STEP 走 B-Rep 截交，并保留 OCP 几何供后续导出 IGS/STEP。
    if input_path.suffix.lower() in {".stp", ".step"}:
        # STEP 走 B-Rep 截交：直接对实体求交，精度由解析几何保证，
        # 不受三角化密度影响。这是精度最高的路径。
        geometry = extract_step_parting_geometry(input_path, config)
        surface = build_planar_parting_surface(geometry.line, config)
        # BASELINE 要三个条件同时成立：线本身可信、线闭合、面可信。
        # 尤其"闭合"这条——未闭合的线围不出分型面，不能当作可用结果。
        status = (
            "BASELINE"
            if (
                geometry.line.status == "BASELINE"
                and geometry.line.closed
                and surface.status == "BASELINE"
            )
            else "REVIEW_REQUIRED"
        )
        warnings = list(geometry.line.warnings) + list(surface.warnings)
        # 线与面的警告合并后去重：同一条可能两处都产生，
        # 重复出现会让甲方以为存在两个不同的问题。
        return PartingRun(
            result=PartingPipelineResult(
                status=status,
                input_file=str(input_path),
                backend="ocp_brep_section",
                line=geometry.line,
                surface=surface,
                line_candidates=geometry.line_candidates,
                input_metadata=input_metadata,
                warnings=list(dict.fromkeys(warnings)),
                config=config,
            ),
            step_geometry=geometry,
        )

    # 其余格式走网格水平截面基线。
    # 精度受三角化密度限制，所以 backend 如实记成网格格式，
    # 让甲方知道这份结果的来源与精度上限。
    mesh = load_mesh_for_parting(input_path, config)
    line = extract_parting_line(mesh, config)
    surface = build_planar_parting_surface(line, config)
    status = (
        "BASELINE"
        if line.status == "BASELINE"
        and line.closed
        and surface.status == "BASELINE"
        else "REVIEW_REQUIRED"
    )
    warnings = list(line.warnings) + list(surface.warnings)
    return PartingRun(
        result=PartingPipelineResult(
            status=status,
            input_file=str(input_path),
            backend=str(mesh.metadata.get("format", "mesh")),
            line=line,
            surface=surface,
            line_candidates=[line],
            input_metadata=input_metadata,
            warnings=warnings,
            config=config,
        )
    )
