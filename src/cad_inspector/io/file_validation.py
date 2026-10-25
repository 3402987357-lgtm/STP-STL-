"""我在几何计算之前先检查文件，避免错误数据产生看似正常的结果。

算法原理：这类校验的全部价值在于"拦住那些能跑完但结果毫无意义"的输入。三类典型
陷阱必须挡在所有几何计算之前：

1. 空文件或不存在的文件：读出来是空网格，检测会输出"没有缺失区域"——一个完全错误的
   成功结论；
2. 后缀不受支持：拿 DWG 当 STEP 解析，要么崩要么静默出乱码几何；
3. **候选文件和参考文件是同一个文件**：这是最凶险的一条。自己和自己比，距离处处为 0，
   检测必然输出"没有任何缺失区域"，看起来是一次漂亮的通过，实际什么都没测。所以我用
   ``os.path.samefile`` 显式拒绝，而不是靠比较路径字符串（硬链接和符号链接会绕过
   字符串比较）。

网格级校验同理：没有三角面或全是退化三角面的网格直接拒绝，不让它进入后续计算。
"""

from __future__ import annotations

import os
from numbers import Integral
from pathlib import Path

from ..errors import InputValidationError
from ..geometry.mesh import TriangleMesh, mesh_quality_report
from .input_validator import validate_input_path


def validate_input_file(
    path: str | Path,
    allowed_suffixes: set[str],
    *,
    max_size_bytes: int | None = None,
) -> Path:
    """我只允许存在且扩展名明确的输入文件进入算法。"""

    # 上限参数先自查：必须是正整数，布尔值不算。
    if (
        max_size_bytes is not None
        and (
            isinstance(max_size_bytes, bool)
            or not isinstance(max_size_bytes, Integral)
            or max_size_bytes < 1
        )
    ):
        raise InputValidationError("我要求 max_size_bytes 必须是正整数。")
    path = Path(path)
    if not path.exists():
        raise InputValidationError(f"我找不到输入文件：{path}")
    if not path.is_file():
        raise InputValidationError(f"我要求输入路径必须是文件：{path}")
    if path.suffix.lower() not in allowed_suffixes:
        suffixes = ", ".join(sorted(allowed_suffixes))
        raise InputValidationError(f"我只接受 {suffixes} 文件：{path}")
    size_bytes = path.stat().st_size
    # 空文件必须拦住：空网格会让检测输出"没有缺失区域"这种假通过。
    if size_bytes == 0:
        raise InputValidationError(f"我发现输入文件为空：{path}")
    if max_size_bytes is not None and size_bytes > max_size_bytes:
        raise InputValidationError(
            f"文件大小超过限制: {size_bytes} > {max_size_bytes} bytes"
        )
    return path


def validate_model_pair(
    candidate_path: str | Path,
    reference_path: str | Path,
    *,
    max_size_bytes: int | None = None,
) -> dict[str, object]:
    """我先校验候选和参考文件的基本边界，不在这里读取复杂几何。"""

    candidate = validate_input_file(
        candidate_path,
        {".stp", ".step", ".stl"},
        max_size_bytes=max_size_bytes,
    )
    reference = validate_input_file(
        reference_path,
        {".stl", ".stp", ".step"},
        max_size_bytes=max_size_bytes,
    )
    # 用 samefile 而不是字符串比较：硬链接与符号链接也能认出是同一个文件。
    if os.path.samefile(candidate, reference):
        raise InputValidationError(
            "我要求候选文件和参考文件不是同一个文件："
            f"{candidate.resolve()}。"
            "把同一个文件同时当作候选和参考只会得到“没有缺失区域”的假通过结果，"
            "我拒绝静默给出这种结论。"
        )
    candidate_metadata = validate_input_path(
        candidate,
        max_size_bytes=max_size_bytes,
    )
    reference_metadata = validate_input_path(
        reference,
        max_size_bytes=max_size_bytes,
    )
    return {
        "candidate": str(candidate.resolve()),
        "reference": str(reference.resolve()),
        "candidate_suffix": candidate.suffix.lower(),
        "reference_suffix": reference.suffix.lower(),
        "candidate_metadata": candidate_metadata.to_dict(),
        "reference_metadata": reference_metadata.to_dict(),
    }


def validate_mesh(mesh: TriangleMesh, role: str) -> dict[str, object]:
    """我拒绝没有三角面的网格，并把质量问题交给上层决定。"""

    report = mesh_quality_report(mesh)
    # 完全没有有效三角面，或全是退化三角面 -> 直接拒绝，不进后续计算。
    if report["has_empty_geometry"]:
        raise InputValidationError(f"我发现 {role} 网格没有有效三角面。")
    if report["degenerate_triangle_count"] == report["triangle_count"]:
        raise InputValidationError(f"我发现 {role} 网格全部是退化三角面。")
    return report
