from __future__ import annotations

import json
from pathlib import Path

import pytest

from cad_inspector.errors import InspectorError
from cad_inspector.geometry.mesh import TriangleMesh
from cad_inspector.geometry.stl import save_stl
from cad_inspector.models import InspectionConfig, RegistrationResult
from cad_inspector.pipeline import _mesh_resolution_warnings, inspect_pair

from test_core import cube_mesh


def _real_step_reference() -> Path | None:
    """我找甲方真实语料里的一个 STEP，用来构造"一侧 mm、一侧未知"。

    我为什么需要真实的 STEP 而不是自己造一个（2026-10-14）：
    STL 读取器**永远**报 unit=unknown，所以我没法用两个 STL
    造出"一侧明确 mm"的输入。而真实 STEP 经 OCCT 读取后
    会明确声明 unit=mm，这正是触发那个 bug 的输入形态。
    """

    candidates = sorted(
        Path(__file__).resolve().parents[1].glob(
            "data/provided/三维模型破损区域智能修补/测试集/*.stp"
        )
    )
    return candidates[0] if candidates else None


def _write_same_cube(tmp_path: Path) -> tuple[Path, Path]:
    candidate = tmp_path / "candidate.stl"
    reference = tmp_path / "reference.stl"
    mesh = cube_mesh()
    save_stl(mesh, candidate)
    save_stl(mesh, reference)
    return candidate, reference


def test我方assume_mm记录配置假定单位(tmp_path: Path) -> None:
    candidate, reference = _write_same_cube(tmp_path)
    result = inspect_pair(
        candidate,
        reference,
        tmp_path / "assume",
        InspectionConfig(
            registration_mode="none",
            unit_policy="assume_mm",
            distance_backend="bruteforce",
        ),
    )

    assert result.unit == "mm"
    assert result.unit_source == "config_assumption"
    assert len(result.global_metrics["input_metadata"]["candidate"]["sha256"]) == 64
    assert result.global_metrics["input_metadata"]["reference"]["size_bytes"] > 0
    assert any("assume_mm" in warning for warning in result.warnings)


def test我方磁盘产物里不再残留自相矛盾的单位警告() -> None:
    """我钉住**已经写出去的产物**里也不许再有那句话。

    我为什么单独加这条：修了代码只保证**以后**不再产出这句，
    但磁盘上已经存了 128 份带着它的 result.json——
    而甲方看的是**产物**，不是源码。只改代码不改产物等于没改。

    我不在这里自动重写历史产物：那些是不同版本跑出来的留档，
    擅自改写会把"当时跑出来是什么样"这个信息抹掉。
    这条测试的作用是把残留**暴露出来**，逼人显式去重新生成
    当前基线，而不是让它悄悄留在交付物里。
    """

    root = Path(__file__).resolve().parents[1]
    stale: list[str] = []
    for path in root.glob("outputs/**/result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        warnings = payload.get("warnings") or []
        if any("非 mm 单位 mm" in str(warning) for warning in warnings):
            stale.append(str(path.relative_to(root)))

    assert not stale, (
        f"磁盘上还有 {len(stale)} 份产物带着自相矛盾的"
        '"我检测到输入文件声明了非 mm 单位 mm"，'
        "请重新生成这些基线：\n" + "\n".join(stale[:10])
    )


def test我方unknown保留未知单位并提示不可直接验收(tmp_path: Path) -> None:
    candidate, reference = _write_same_cube(tmp_path)
    result = inspect_pair(
        candidate,
        reference,
        tmp_path / "unknown",
        InspectionConfig(
            registration_mode="none",
            unit_policy="unknown",
            distance_backend="bruteforce",
        ),
    )

    assert result.unit == "unknown"
    assert result.unit_source == "policy_unknown"
    assert any("不能直接用于毫米级验收" in warning for warning in result.warnings)


def test我方一侧声明mm一侧未知时不说自相矛盾的话(tmp_path: Path) -> None:
    """我钉住一句**自相矛盾**的警告不再出现（2026-10-14 修的真 bug）。

    原来的过滤只排除了空值和 unknown，**没有排除 mm 本身**，
    于是"参考 STEP 明确声明 mm、待检测 STL 未知"时，
    mm 被当成"非 mm 单位"报出去，警告原文是：

        我检测到输入文件声明了非 mm 单位 mm；……

    "非 mm 单位 mm" 是一句自己打自己的话。甲方读到只会更困惑：
    到底是不是 mm？这类文案缺陷比功能缺陷更隐蔽——
    它不影响数值，但直接损害"复核结论可信"这件事。

    我真正要钉的：**声明过的 mm 绝不能出现在"非 mm 单位"清单里**。
    """

    step_reference = _real_step_reference()
    if step_reference is None:
        pytest.skip("我找不到真实 STEP 语料来构造混合单位输入")

    candidate = tmp_path / "candidate.stl"
    save_stl(cube_mesh(), candidate)

    result = inspect_pair(
        candidate,
        step_reference,
        tmp_path / "mixed",
        InspectionConfig(
            registration_mode="none",
            unit_policy="assume_mm",
            distance_backend="bruteforce",
        ),
    )

    # 先确认这个输入确实进到了"混合单位"分支，否则这条等于没验。
    assert result.unit_source == "config_assumption", result.unit_source

    for warning in result.warnings:
        assert "非 mm 单位 mm" not in warning, (
            f"我报出了自相矛盾的单位警告：{warning}"
        )
        # 更一般地：清单里不许出现 mm（大小写都已经统一成小写）。
        assert "非 mm 单位 mm," not in warning and not warning.endswith(
            "非 mm 单位 mm"
        ), warning

    # 我顺手钉住另一件事：这条归一化分支的警告**声称**"比例已记录在
    # unit_audit"，那 unit_audit 就必须真的存在，不能是一句空承诺。
    if any("unit_audit" in warning for warning in result.warnings):
        audit = (result.global_metrics or {}).get("unit_audit")
        assert isinstance(audit, dict) and audit, (
            "我在警告里叫甲方去看 unit_audit，但结果里根本没写 unit_audit。"
        )
        assert audit.get("candidate_source_unit") == "mm", audit


def test我方require_mm拒绝未声明单位的stl(tmp_path: Path) -> None:
    candidate, reference = _write_same_cube(tmp_path)

    with pytest.raises(InspectorError, match="明确声明 mm"):
        inspect_pair(
            candidate,
            reference,
            tmp_path / "require",
            InspectionConfig(
                registration_mode="none",
                unit_policy="require_mm",
                distance_backend="bruteforce",
            ),
        )


def test我方会提示网格分辨率不足或参数不一致() -> None:
    mesh = TriangleMesh(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        faces=[[0, 1, 2]],
        metadata={
            "linear_deflection_mm": 1.0,
            "angular_deflection_deg": 10.0,
        },
    )
    reference = TriangleMesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        metadata={
            "linear_deflection_mm": 0.04,
            "angular_deflection_deg": 15.0,
        },
    )

    warnings = _mesh_resolution_warnings(mesh, reference, threshold_mm=0.15)

    assert any("线性三角化偏差" in warning for warning in warnings)
    assert any("不同的角度三角化偏差" in warning for warning in warnings)


def test我方压力模式可以关闭大体积彩色网格输出(tmp_path: Path) -> None:
    candidate, reference = _write_same_cube(tmp_path)
    output_dir = tmp_path / "compact"
    result = inspect_pair(
        candidate,
        reference,
        output_dir,
        InspectionConfig(
            registration_mode="none",
            distance_backend="bruteforce",
            write_colored_mesh=False,
        ),
    )

    assert result.status == "review"
    assert "colored_mesh" not in result.artifacts
    assert not (output_dir / "colored_reference.ply").exists()


def test我方缺失区域签名写入_json和_csv(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.stl"
    reference = tmp_path / "reference.stl"
    save_stl(cube_mesh(omit_top=True), candidate)
    save_stl(cube_mesh(), reference)
    output_dir = tmp_path / "region-signature"

    result = inspect_pair(
        candidate,
        reference,
        output_dir,
        InspectionConfig(
            registration_mode="none",
            distance_backend="bruteforce",
            min_region_area_mm2=0.1,
            write_colored_mesh=False,
        ),
    )

    assert result.regions
    signature = result.regions[0].region_geometry_signature
    assert signature is not None
    assert signature.startswith("meshregion-")
    payload = json.loads(
        (output_dir / "result.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "missing-region-detection-1.3"
    assert payload["regions"][0]["region_geometry_signature"] == signature
    assert result.artifacts["json"] == str((output_dir / "result.json").resolve())
    assert result.artifacts["csv"] == str((output_dir / "regions.csv").resolve())
    assert payload["artifacts"]["json"] == result.artifacts["json"]
    assert payload["artifacts"]["csv"] == result.artifacts["csv"]
    header = (output_dir / "regions.csv").read_text(
        encoding="utf-8-sig"
    ).splitlines()[0]
    assert "region_geometry_signature" in header


def test我方配准质量不足时即使无缺失也必须复核(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate, reference = _write_same_cube(tmp_path)
    bad_registration = RegistrationResult(
        mode="test",
        rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_mm=[0.0, 0.0, 0.0],
        scale=1.0,
        mean_residual_mm=2.0,
        p95_residual_mm=2.0,
        quality="bad",
        warnings=["测试配准质量不足。"],
    )
    monkeypatch.setattr(
        "cad_inspector.pipeline.align_mesh",
        lambda mesh, ref, config: (mesh, bad_registration),
    )

    result = inspect_pair(
        candidate,
        reference,
        tmp_path / "bad-registration",
        InspectionConfig(
            registration_mode="none",
            unit_policy="unknown",
            distance_backend="bruteforce",
        ),
    )

    assert result.status == "review"
    assert result.global_metrics["registration_review_required"] is True


def test我方配置字符串布尔值按字面解析() -> None:
    config = InspectionConfig.from_dict(
        {
            "allow_scale": "false",
            "bidirectional_check": "0",
            "write_intermediate": "true",
            "write_colored_mesh": "off",
        }
    )

    assert config.allow_scale is False
    assert config.bidirectional_check is False
    assert config.write_intermediate is True
    assert config.write_colored_mesh is False

    with pytest.raises(ValueError, match="必须是布尔值"):
        InspectionConfig.from_dict({"allow_scale": "maybe"})


def test我方有向证据角度容差拒绝无效值() -> None:
    with pytest.raises(ValueError, match="direction_normal_tolerance_deg"):
        InspectionConfig(direction_normal_tolerance_deg=0.0)
    with pytest.raises(ValueError, match="direction_normal_tolerance_deg"):
        InspectionConfig(direction_normal_tolerance_deg=90.1)
