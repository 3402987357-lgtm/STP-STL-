from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "calibrate_real_thresholds.py"
    spec = importlib.util.spec_from_file_location("calibrate_real_thresholds", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(module, **overrides):
    parser = module._build_parser()
    args = parser.parse_args(["--worker", "--step", "a.stp", "--stl", "a.stl",
                              "--output", "out"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test我方真实件上没有任何一个支持015mm这个默认值() -> None:
    """我钉住一条**反直觉但已被实测确认**的结论。

    README 里写"20 个真实件里没有任何一个支持 0.15 mm"，
    这是靠一次性跑脚本得出的结论，写死在文档里。
    我把这句文档里的话变成门禁每次都会重跑的断言：
    真实件上算出的**推荐阈值最小也要 0.1859 mm**，
    中位数 0.6008 mm——全部**大于** 0.15 mm 这个默认配置。

    我为什么钉这个而不是只钉"算法能算"：
    这正是"系统在这批件上普遍判 review"的根因——
    不是模型差，是阈值取了一个真实件上不存在的精度。
    如果哪天有人把 0.15 调大让它"看起来通过率变高"，
    而没有重新确认这个关系，这条就会红。

    我还钉住 residual_dominates：20/20 全部为 True，
    说明限制来自**配准残差**而不是网格尺度，
    因此改进方向是配准精度，而不是把网格切得更细。
    """

    root = Path(__file__).resolve().parents[1]
    results = sorted(
        (root / "outputs" / "real_threshold_calibration_full").rglob("result.json")
    )
    if not results:
        pytest.skip("我找不到真实阈值标定产物，无法确认这条结论")

    configured = set()
    recommended: list[float] = []
    not_residual_dominated: list[str] = []
    for path in results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit = (payload.get("global_metrics") or {}).get("threshold_audit") or {}
        if not audit:
            continue
        configured.add(audit.get("configured_mm"))
        value = audit.get("recommended_mm")
        if value is not None:
            recommended.append(float(value))
        if not audit.get("residual_dominates"):
            not_residual_dominated.append(path.parent.name)

    assert len(recommended) >= 20, (
        f"真实标定样本只剩 {len(recommended)} 个，不足以支撑这条结论"
    )
    assert configured == {0.15}, configured
    assert min(recommended) > 0.15, (
        f"最小推荐阈值 {min(recommended):.4f} mm 已经不大于配置的 0.15 mm，"
        "说明 0.15 这个默认值在真实件上变得可用了——"
        "请重新确认 README 里的阈值结论并同步更新这条测试。"
    )
    assert not not_residual_dominated, (
        f"这些件的主导因素变成了尺度而非配准残差：{not_residual_dominated[:5]}，"
        "改进方向需要重新判断。"
    )


def test我方标定配置强制使用auto阈值并透传系数() -> None:
    module = _load_module()
    config = module.calibration_config(
        _args(module, scale_ratio=0.005, residual_factor=2.5, floor_mm=0.05)
    )
    assert config.threshold_mode == "auto"
    assert config.auto_threshold_scale_ratio == 0.005
    assert config.auto_threshold_residual_factor == 2.5
    assert config.auto_threshold_floor_mm == 0.05
    # 标定轮不写大体积中间产物，避免把 D 盘堆满。
    assert config.write_intermediate is False
    assert config.write_colored_mesh is False


def test我方标定指纹随系数变化() -> None:
    module = _load_module()
    base = module.config_fingerprint(module.calibration_config(_args(module)))
    changed = module.config_fingerprint(
        module.calibration_config(_args(module, residual_factor=1.6))
    )
    assert base != changed


def test我方汇总排除异常记录并统计八秒达标数() -> None:
    module = _load_module()
    records = [
        {"pair": "a", "status": "review", "elapsed_seconds": 3.0,
         "threshold_audit": {"recommended_mm": 0.2},
         "registration": {"mean_residual_mm": 0.1}},
        {"pair": "b", "status": "timeout", "elapsed_seconds": 240.0,
         "threshold_audit": {"recommended_mm": 0.4},
         "registration": {"mean_residual_mm": 0.2}},
        {"pair": "c", "status": "exception", "elapsed_seconds": None,
         "threshold_audit": {}, "registration": {}},
    ]
    aggregate = module._aggregate(records)
    assert aggregate["measured_pair_count"] == 2
    assert aggregate["skipped_pair_count"] == 1
    assert aggregate["within_8s_count"] == 1
    assert aggregate["elapsed_seconds"]["max"] == 240.0
    assert aggregate["recommended_threshold_mm"]["median"] == pytest.approx(0.3)


def test我方汇总不把跳过的样本算成八秒达标() -> None:
    module = _load_module()
    records = [
        {"pair": "guard", "status": "resource_guard", "elapsed_seconds": 0.0,
         "threshold_audit": {}, "registration": {}},
        {"pair": "real", "status": "review", "elapsed_seconds": 12.0,
         "threshold_audit": {"recommended_mm": 0.3},
         "registration": {"mean_residual_mm": 0.2}},
    ]
    aggregate = module._aggregate(records)
    assert aggregate["measured_pair_count"] == 1
    assert aggregate["within_8s_count"] == 0
    assert aggregate["elapsed_seconds"]["min"] == 12.0


def test我方标定汇总写出失败时保留旧文件(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "output"
    output_root.mkdir()
    summary_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    summary_path.write_text("old-json", encoding="utf-8")
    csv_path.write_text("old-csv", encoding="utf-8")

    args = _args(module, timeout_seconds=180.0, max_stl_bytes=100)
    with pytest.raises(ValueError):
        module._write_reports(
            output_root=output_root,
            data_root=tmp_path / "data",
            args=args,
            records=[{"pair": "bad", "value": float("nan")}],
            pair_count_total=1,
        )

    assert summary_path.read_text(encoding="utf-8") == "old-json"
    assert csv_path.read_text(encoding="utf-8") == "old-csv"


def test我方标定汇总可序列化(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "output"
    args = _args(module, timeout_seconds=180.0, max_stl_bytes=100)
    module._write_reports(
        output_root=output_root,
        data_root=tmp_path / "data",
        args=args,
        records=[
            {
                "pair": "样本/1",
                "status": "review",
                "elapsed_seconds": 4.5,
                "threshold_mm": 0.3,
                "threshold_audit": {
                    "bbox_diagonal_mm": 10.0,
                    "scale_component_mm": 0.02,
                    "residual_component_mm": 0.3,
                    "recommended_mm": 0.3,
                },
                "metrics": {"missing_area_ratio": 0.0, "extra_area_ratio": 0.01},
                "region_count": 1,
                "error": None,
            }
        ],
        pair_count_total=1,
    )
    payload = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == module.SCHEMA_VERSION
    assert payload["aggregate"]["within_8s_count"] == 1
    assert "样本/1" in (output_root / "summary.csv").read_text(encoding="utf-8-sig")


@dataclass
class _FakeRegion:
    region_id: int
    defect_type: str
    area_mm2: float
    review_required: bool = True


@dataclass
class _FakeResult:
    status: str = "review"
    unit: str = "mm"
    warnings: list[str] = field(default_factory=list)
    threshold_mm: float = 0.3
    regions: list = field(default_factory=list)
    global_metrics: dict = field(default_factory=dict)
    registration: object = field(
        default_factory=lambda: SimpleNamespace(
            mode="centroid", quality="ok", mean_residual_mm=0.2,
            p95_residual_mm=0.4, scale=1.0,
        )
    )


def test我方worker记录区域明细而不是不存在的damage_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    step = tmp_path / "a.stp"
    stl = tmp_path / "a.stl"
    step.write_text("step", encoding="ascii")
    stl.write_text("stl", encoding="ascii")
    result = _FakeResult(
        regions=[_FakeRegion(region_id=1, defect_type="missing", area_mm2=1.5)],
        global_metrics={
            "missing_area_ratio": 0.01,
            "extra_area_ratio": 0.0,
            "threshold_audit": {"recommended_mm": 0.3},
        },
    )
    monkeypatch.setattr(module, "inspect_pair", lambda **_: result)

    output_dir = tmp_path / "out"
    module.run_worker(step, stl, output_dir, _args(module))
    summary = json.loads(
        (output_dir / "worker_summary.json").read_text(encoding="utf-8")
    )
    assert summary["region_defect_types"] == ["missing"]
    assert summary["region_detail"][0]["area_mm2"] == 1.5
    assert summary["metrics"]["missing_area_ratio"] == 0.01
