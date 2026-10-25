"""我把 OCP 分型线和分型面写成可交换的 IGS/STEP 文件。

算法原理：CAD 交换格式写出比文本写出更容易留下半截文件——OCC 写出器要分两步
（Transfer 到内部会话，再 Write 到磁盘），中断就会写出一个能骗过"文件存在"检查、
但打不开的坏文件。所以我统一走 ``_atomic_write_cad``：真正写到同目录的临时文件，
写完再原子替换到正式路径。

另外一个坑是写出器的返回值语义不一致：IGES 的 ``Write`` 返回布尔，STEP 的返回
``IFSelect_ReturnStatus``。我没有统一它们的类型，而是在各自的 lambda 里按自己的
返回值判断成功与否，避免把"RetDone"当成 Python 真值这种假通过。

每种几何都同时输出 STEP 与 IGS 两份，方便甲方用不同 CAD 软件打开核对；过渡曲面
额外输出缝合前后两个版本，让甲方能看出缝合这一步到底改变了什么。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..detection.step_parting import StepPartingGeometry
from ..errors import GeometryProcessingError
from .atomic import atomic_path


def _atomic_write_cad(path: Path, write: Callable[[Path], bool]) -> str:
    """我用同目录临时文件替换正式 CAD 输出，避免中断留下半个 STEP/IGS。"""

    with atomic_path(path) as temporary:
        if not write(temporary):
            raise GeometryProcessingError(f"我写出 CAD 文件失败: {path}")
    return str(path)


def write_step_parting_cad_outputs(
    geometry: StepPartingGeometry,
    output_dir: str | Path,
    stem: str,
) -> dict[str, str]:
    """我输出真实 B-Rep 线框和截交平面，并保留清晰的失败边界。"""

    # OCP 是硬依赖：没有就明确报几何处理错误，不降级成"什么都不输出"。
    try:
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.IGESControl import IGESControl_Writer
        from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    except ImportError as exc:
        raise GeometryProcessingError(
            "我无法写出 IGS/STEP，因为当前环境没有 OCP。"
        ) from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    # 分型线先写 IGS（线框在 IGS 里兼容性最好）。
    line_iges = output_path / f"{stem}.parting_line.igs"
    iges_line_writer = IGESControl_Writer("MM", 0)
    if not iges_line_writer.AddShape(geometry.wire):
        raise GeometryProcessingError("我向 IGS 写入分型线线框失败。")
    artifacts["parting_line_iges"] = _atomic_write_cad(
        line_iges,
        lambda temporary: iges_line_writer.Write(str(temporary)),
    )

    # 没有分型面几何就到此为止，只返回线框产物。
    if geometry.face is None:
        return artifacts

    surface_step = output_path / f"{stem}.parting_surface.step"
    step_writer = STEPControl_Writer()
    # AsIs 模式原样转移拓扑，不做形状修改：
    # 分型面是计算结果，任何自动"修复"都会让它与报告里的数值对不上。
    transfer_status = step_writer.Transfer(geometry.face, STEPControl_AsIs)
    if transfer_status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise GeometryProcessingError("我向 STEP 写入分型面失败。")
    artifacts["parting_surface_step"] = _atomic_write_cad(
        surface_step,
        lambda temporary: step_writer.Write(str(temporary))
        == IFSelect_ReturnStatus.IFSelect_RetDone,
    )

    surface_iges = output_path / f"{stem}.parting_surface.igs"
    # IGES 显式指定 MM 单位：IGES 的单位字段可被读取方忽略，
    # 不写死毫米会让甲方在英制环境的 CAD 里拿到放大 25.4 倍的面。
    iges_surface_writer = IGESControl_Writer("MM", 0)
    if not iges_surface_writer.AddShape(geometry.face):
        raise GeometryProcessingError("我向 IGS 写入分型面失败。")
    artifacts["parting_surface_iges"] = _atomic_write_cad(
        surface_iges,
        lambda temporary: iges_surface_writer.Write(str(temporary)),
    )

    # 过渡曲面：STEP 与 IGS 各一份，便于甲方用不同软件交叉核对。
    if geometry.transition_surface is not None:
        transition_step = output_path / f"{stem}.transition_surface.step"
        transition_writer = STEPControl_Writer()
        transfer_status = transition_writer.Transfer(
            geometry.transition_surface,
            STEPControl_AsIs,
        )
        if transfer_status != IFSelect_ReturnStatus.IFSelect_RetDone:
            raise GeometryProcessingError("我向 STEP 写入过渡曲面候选失败。")
        artifacts["transition_surface_step"] = _atomic_write_cad(
            transition_step,
            lambda temporary: transition_writer.Write(str(temporary))
            == IFSelect_ReturnStatus.IFSelect_RetDone,
        )

        transition_iges = output_path / f"{stem}.transition_surface.igs"
        transition_iges_writer = IGESControl_Writer("MM", 0)
        if not transition_iges_writer.AddShape(geometry.transition_surface):
            raise GeometryProcessingError("我向 IGS 写入过渡曲面候选失败。")
        artifacts["transition_surface_iges"] = _atomic_write_cad(
            transition_iges,
            lambda temporary: transition_iges_writer.Write(str(temporary)),
        )

    # 缝合后的过渡曲面：单独输出，让甲方能对比缝合前后的差异。
    if geometry.sewn_transition_surface is not None:
        sewn_step = output_path / f"{stem}.transition_surface_sewn.step"
        sewn_step_writer = STEPControl_Writer()
        transfer_status = sewn_step_writer.Transfer(
            geometry.sewn_transition_surface,
            STEPControl_AsIs,
        )
        if transfer_status != IFSelect_ReturnStatus.IFSelect_RetDone:
            raise GeometryProcessingError("我向 STEP 写入缝合过渡曲面候选失败。")
        artifacts["transition_surface_sewn_step"] = _atomic_write_cad(
            sewn_step,
            lambda temporary: sewn_step_writer.Write(str(temporary))
            == IFSelect_ReturnStatus.IFSelect_RetDone,
        )

        sewn_iges = output_path / f"{stem}.transition_surface_sewn.igs"
        sewn_iges_writer = IGESControl_Writer("MM", 0)
        if not sewn_iges_writer.AddShape(geometry.sewn_transition_surface):
            raise GeometryProcessingError("我向 IGS 写入缝合过渡曲面候选失败。")
        artifacts["transition_surface_sewn_iges"] = _atomic_write_cad(
            sewn_iges,
            lambda temporary: sewn_iges_writer.Write(str(temporary)),
        )
    return artifacts
