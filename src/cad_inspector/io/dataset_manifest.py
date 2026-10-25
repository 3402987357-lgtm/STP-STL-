"""我方离线数据集登记工具。

我只登记文件事实、格式探测状态和标签状态，不自动把样本猜成训练集或测试集。
这满足我方先冻结数据基线、再开展算法开发的要求。

算法原理：数据基线必须先冻结再开发，否则"训练了哪些文件"这件事本身就不确定。所以我
只登记三件事实：文件标识（路径/大小/sha256）、探测出来的格式状态、标签状态。
**不自动猜测数据集划分**——训练/测试划分一旦靠猜，训练集和测试集就会混入同一份文件，
准确率必然虚高，而且事后无法发现。

样本 ID 直接取 sha256 前 12 位：内容不变 ID 就不变，内容改一个字节 ID 就变，天然具备
"改了数据就会被发现"的性质。单个文件探测失败只记进 warnings 和 notes，不中断整批登记。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from cad_inspector.domain.dataset import DatasetManifest, DatasetRecord
from cad_inspector.io.input_validator import InputValidationError, validate_input_path
from cad_inspector.io.step_probe import probe_any


DEFAULT_EXTENSIONS = {".step", ".stp", ".xt", ".prt", ".stl"}


def build_manifest(
    root: str | Path,
    extensions: set[str] | None = None,
) -> DatasetManifest:
    """我递归扫描数据目录，并为每个文件生成稳定记录。"""

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise InputValidationError(f"我找不到数据目录：{root_path}")
    if not root_path.is_dir():
        raise InputValidationError(f"我要求数据清单根路径必须是目录：{root_path}")

    allowed = {suffix.lower() for suffix in (extensions or DEFAULT_EXTENSIONS)}
    records: list[DatasetRecord] = []
    warnings: list[str] = []
    # 按路径排序遍历，保证同一目录每次得到的记录顺序一致（结果可复现）。
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        # 单文件不可读只记警告，不中断整批登记。
        try:
            metadata = validate_input_path(path)
        except InputValidationError as exc:
            warnings.append(str(exc))
            continue

        probe_status = "unknown"
        notes: list[str] = []
        entity_counts: dict[str, int] = {}
        # 探测失败记成 probe_failed 并附上原因，不冒充"未知格式"。
        try:
            probe = probe_any(path)
            probe_status = probe.status.value
            # 我把探测出的实体计数一并登记：只有 status 的话，清单里
            # 看不出这个样本到底有多少顶点/面、网格质量如何，
            # 而这两项正是判断样本能不能用于训练的依据。
            entity_counts = {
                str(k): int(v)
                for k, v in dict(probe.entity_counts or {}).items()
                if isinstance(v, (int, float))
            }
        except Exception as exc:  # noqa: BLE001
            probe_status = "probe_failed"
            notes.append(f"我登记时无法完成格式探测：{type(exc).__name__}: {exc}")

        records.append(
            DatasetRecord(
                sample_id=_sample_id(metadata.sha256),
                path=metadata.path,
                name=metadata.name,
                extension=metadata.extension,
                size_bytes=metadata.size_bytes,
                sha256=metadata.sha256,
                detected_format=metadata.detected_format.value,
                probe_status=probe_status,
                notes=notes,
                entity_counts=entity_counts,
            )
        )

    return DatasetManifest(
        schema_version="dataset-manifest-1.0",
        root=str(root_path),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        records=records,
        warnings=warnings,
    )


def _sample_id(sha256: str) -> str:
    """我用 sha256 前 12 位生成样本 ID：内容变则 ID 变。"""
    return f"sample-{sha256[:12]}"
