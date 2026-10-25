from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_stress_module():
    path = Path(__file__).parents[1] / "scripts" / "run_real_pair_stress.py"
    spec = importlib.util.spec_from_file_location("real_pair_stress", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test我方压力轮只复用当前输入对应的worker缓存(tmp_path: Path) -> None:
    stress = _load_stress_module()
    step = tmp_path / "model.step"
    stl = tmp_path / "model.stl"
    step.write_text("step-v1", encoding="ascii")
    stl.write_text("stl-v1", encoding="ascii")

    summary = {
        "input_fingerprints": {
            "step": stress._input_fingerprint(step),
            "stl": stress._input_fingerprint(stl),
        }
    }
    assert stress._worker_summary_matches_inputs(summary, step, stl)

    stl.write_text("stl-v2", encoding="ascii")
    assert not stress._worker_summary_matches_inputs(summary, step, stl)


def test我方压力轮拒绝不同检测配置的worker缓存(tmp_path: Path) -> None:
    stress = _load_stress_module()
    step = tmp_path / "model.step"
    stl = tmp_path / "model.stl"
    step.write_text("step-v1", encoding="ascii")
    stl.write_text("stl-v1", encoding="ascii")

    summary = {
        "input_fingerprints": {
            "step": stress._input_fingerprint(step),
            "stl": stress._input_fingerprint(stl),
        },
        "inspection_config_fingerprint": "old",
    }
    assert not stress._worker_summary_matches_inputs(
        summary,
        step,
        stl,
        "new",
    )


def test我方压力轮只恢复相同输入和运行配置(tmp_path: Path) -> None:
    stress = _load_stress_module()
    data_root = tmp_path / "data"
    pair_dir = data_root / "样本"
    pair_dir.mkdir(parents=True)
    step = pair_dir / "1.step"
    stl = pair_dir / "1.stl"
    step.write_text("step-v1", encoding="ascii")
    stl.write_text("stl-v1", encoding="ascii")
    pairs = [("样本/1", step, stl)]
    config = stress._stress_config()
    run_fingerprint = stress._run_config_fingerprint(config, 180.0, 100)
    record = {
        "pair": "样本/1",
        "input_fingerprints": {
            "step": stress._input_fingerprint(step),
            "stl": stress._input_fingerprint(stl),
        },
        "run_config_fingerprint": run_fingerprint,
        "status": "REVIEW_REQUIRED",
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": stress.SCHEMA_VERSION,
                "data_root": str(data_root.resolve()),
                "run_config_fingerprint": run_fingerprint,
                "records": [record],
            }
        ),
        encoding="utf-8",
    )

    resumed = stress._load_resume_records(
        summary_path,
        pairs,
        data_root,
        run_fingerprint,
    )
    assert resumed["样本/1"]["status"] == "REVIEW_REQUIRED"

    stl.write_text("stl-v2", encoding="ascii")
    assert not stress._load_resume_records(
        summary_path,
        pairs,
        data_root,
        run_fingerprint,
    )


def test我方压力汇总写出失败时保留旧文件(tmp_path: Path) -> None:
    stress = _load_stress_module()
    output_root = tmp_path / "output"
    output_root.mkdir()
    summary_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    summary_path.write_text("old-json", encoding="utf-8")
    csv_path.write_text("old-csv", encoding="utf-8")

    with pytest.raises(ValueError):
        stress._write_reports(
            output_root=output_root,
            data_root=tmp_path / "data",
            config=stress._stress_config(),
            records=[{"pair": "bad", "value": float("nan")}],
            pair_count_total=1,
            timeout_seconds=180.0,
            resource_guard_bytes=100,
            run_config_fingerprint="fingerprint",
        )

    assert summary_path.read_text(encoding="utf-8") == "old-json"
    assert csv_path.read_text(encoding="utf-8") == "old-csv"


def test我方续跑只接受输入与配置都完全一致的记录(tmp_path: Path) -> None:
    """我锁定续跑的"宁可重跑，不可错复用"这条契约。

    压力结论是要拿去向甲方交代"这批数据跑得动"的。续跑时若悄悄混进
    上一轮不同阈值下的结果，这份汇总就不再是单一口径的证据，
    出了问题无法回溯是哪一轮、哪个配置。

    七个场景里最关键的是第 4 个：文件内容变了但字节数没变。
    只比路径或只比大小都会复用旧结论，必须靠内容哈希挡住。
    """

    import tempfile

    stress = _load_stress_module()
    step = tmp_path / "a.stp"
    stl = tmp_path / "b.stl"
    step.write_text("AAA", encoding="ascii")
    stl.write_text("BBB", encoding="ascii")
    fingerprint = "FP123"
    pairs = [("x/a", step, stl)]
    summary_path = tmp_path / "summary.json"

    def write_summary(*, schema=None, config=None) -> None:
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": schema or stress.SCHEMA_VERSION,
                    "data_root": str(tmp_path.resolve()),
                    "run_config_fingerprint": fingerprint,
                    "records": [
                        {
                            "pair": "x/a",
                            "run_config_fingerprint": fingerprint,
                            "input_fingerprints": {
                                "step": stress._input_fingerprint(step),
                                "stl": stress._input_fingerprint(stl),
                            },
                            "inspection_config_fingerprint": (
                                config or fingerprint
                            ),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    # 1) 完全一致 -> 复用
    write_summary()
    assert list(
        stress._load_resume_records(
            summary_path, pairs, tmp_path, fingerprint
        ).keys()
    ) == ["x/a"]

    # 2) 运行配置变了 -> 整份不复用
    assert (
        stress._load_resume_records(summary_path, pairs, tmp_path, "OTHER")
        == {}
    )

    # 3) schema 版本变了 -> 不复用
    write_summary(schema="v-old")
    assert stress._load_resume_records(summary_path, pairs, tmp_path, fingerprint) == {}

    # 4) 内容变了但字节数相同 -> 必须不复用
    step.write_text("AAA", encoding="ascii")
    write_summary()
    assert list(
        stress._load_resume_records(
            summary_path, pairs, tmp_path, fingerprint
        ).keys()
    ) == ["x/a"]
    step.write_text("AAB", encoding="ascii")
    assert stress._load_resume_records(summary_path, pairs, tmp_path, fingerprint) == {}

    # 5) 损坏 JSON -> 不复用且不抛错
    summary_path.write_text("{broken", encoding="utf-8")
    assert stress._load_resume_records(summary_path, pairs, tmp_path, fingerprint) == {}

    # 6) 未知样本名 -> 跳过
    step.write_text("AAA", encoding="ascii")
    write_summary()
    assert (
        stress._load_resume_records(
            summary_path, [("y/z", step, stl)], tmp_path, fingerprint
        )
        == {}
    )

    # 7) 数据根目录变了 -> 不复用
    assert (
        stress._load_resume_records(
            summary_path, pairs, Path(tempfile.mkdtemp()), fingerprint
        )
        == {}
    )
