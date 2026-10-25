"""我给"验收结果转标注草稿"这个脚本写回归。

为什么值得钉：这条链路产出的是**人工复核的输入**。草稿一旦缺来源、
或者把机器预测写成看起来像真值的东西，后面的人工确认就建立在
错误的对照物上。我实测踩到的具体问题是——仓库里那份草稿没有任何
来源字段，我要判断它是否已经被新管线跑陈旧时，只能把 outputs 下
57 份 result.json 全扫一遍逐条比对面号，才认出它出自
real_pair_acceptance_v3。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def _load_module():
    """我按路径导入脚本模块（它不是包内模块）。"""

    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import bootstrap_annotations as module

        return module
    finally:
        sys.path.remove(str(SCRIPTS_DIR))


module = _load_module()


def _write_pair(
    root: Path,
    name: str,
    faces: list[int],
    *,
    deflection: float = 0.04,
) -> None:
    pair_dir = root / name
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "review",
                "config": {"mesh_linear_deflection_mm": deflection},
                # 参考件必须每对都不同：标注校验会拒绝同一文件跨 split
                # 复用（那正是训练数据泄漏）。
                "reference_file": f"D:/data/ref-{name}.stl",
                "candidate_file": f"D:/data/{name}.stp",
                "regions": [
                    {
                        "defect_type": "missing",
                        "reference_face_ids": faces,
                        "center_xyz_mm": [1.0, 2.0, 3.0],
                        "bbox_min_xyz_mm": [0.0, 0.0, 0.0],
                        "bbox_max_xyz_mm": [1.0, 1.0, 1.0],
                        "area_mm2": 2.0,
                        "boundary_edge_ids": [],
                        "confidence": 0.8,
                        "max_error_mm": 0.4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test我方仓库里的草稿来源目录现在仍然存在() -> None:
    """我钉住仓库里那份草稿的来源字段不是死路径。

    我为什么加这条：项目 2026-10-18 整体搬进了大赛目录，
    而 reports/draft_annotations.json 里记的还是搬家前的绝对路径。
    来源字段存在的意义就是"能顺藤摸瓜找到这批预测出自哪次检测"；
    路径失效等于来源字段失效，草稿又变回不可复核。
    """

    path = ROOT / "reports" / "draft_annotations.json"
    if not path.is_file():
        pytest.skip("仓库里没有草稿清单")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = Path(manifest["source_acceptance_root"])
    assert root.is_dir(), (
        f"草稿来源目录已不存在：{root}；"
        "请用当前 outputs 目录重新生成 reports/draft_annotations.json。"
    )


def test我方草稿带来源与时间戳并且明确标成未确认(tmp_path: Path) -> None:
    """来源字段是我这一轮补的：没有它，草稿不可复核。"""

    _write_pair(tmp_path, "训练集__0", [1, 2, 3])
    _write_pair(tmp_path, "测试集__7", [9, 10])

    manifest = module.build_draft_manifest(
        tmp_path,
        lambda pair: "train" if pair.startswith("训练集") else "test",
    )

    # 来源必须是**解析后的绝对路径**，相对路径换个工作目录就指不到了。
    assert manifest["source_acceptance_root"] == str(tmp_path.resolve())
    # 时间戳必须能被解析回来，不能是个占位串。
    parsed = datetime.fromisoformat(manifest["generated_at"])
    assert parsed.tzinfo is not None
    # 顶层与每条记录都要标明这是未确认的机器预测。
    assert manifest["provenance"] == "draft"
    assert all(r["provenance"] == "draft" for r in manifest["records"])
    assert all(r["review_state"] == "unconfirmed" for r in manifest["records"])


def test我方草稿按语料目录名划分并保留机器证据(tmp_path: Path) -> None:
    """split 跟语料走，置信度/误差量要带出来供人工排序。"""

    _write_pair(tmp_path, "训练集__0", [1, 2, 3])
    _write_pair(tmp_path, "测试集__7", [9, 10])

    manifest = module.build_draft_manifest(
        tmp_path,
        lambda pair: "train" if pair.startswith("训练集") else "test",
    )

    by_id = {r["sample_id"]: r for r in manifest["records"]}
    assert by_id["训练集_0"]["split"] == "train"
    assert by_id["测试集_7"]["split"] == "test"
    region = by_id["训练集_0"]["regions"][0]
    assert region["reference_face_ids"] == [1, 2, 3]
    assert region["confidence"] == 0.8
    assert region["max_error_mm"] == 0.4
    # 草稿本身要能通过标注校验（结构上合法），否则人工复核前就先崩了。
    from cad_inspector.learning.annotations import validate_annotation_manifest

    summary = validate_annotation_manifest(manifest)
    # 但它绝不能被判成"可训练"：草稿当真值就是假准确率的直通车。
    assert summary.ready_for_training is False
    assert summary.draft_record_count == 2


def test我方把降精度重试的件单独标出来(tmp_path: Path) -> None:
    """我钉住这条实测教训：同一对件，0.2mm 精度报 26519 个缺料面，
    0.04mm 精度报 48810 个，差 84%。

    验收脚本超时后会自动降精度重试，而精度只记在 config 里。
    人工复核时若不知道手上这条来自粗网格，就会把三角化误差
    当成真实缺料来确认——所以草稿必须把精度亮出来。
    """

    _write_pair(tmp_path, "训练集__0", [1, 2, 3], deflection=0.04)
    _write_pair(tmp_path, "训练集__5", [4, 5], deflection=0.2)
    _write_pair(tmp_path, "测试集__8", [6], deflection=0.5)

    manifest = module.build_draft_manifest(
        tmp_path,
        lambda pair: "train" if pair.startswith("训练集") else "test",
    )

    # 只有比默认精度更粗的才进这个名单。
    assert manifest["coarsened_sample_ids"] == ["测试集_8", "训练集_5"]
    assert manifest["source_default_linear_deflection_mm"] == 0.04

    by_id = {r["sample_id"]: r for r in manifest["records"]}
    assert by_id["训练集_0"]["source_linear_deflection_mm"] == 0.04
    assert by_id["训练集_5"]["source_linear_deflection_mm"] == 0.2
    assert by_id["测试集_8"]["source_linear_deflection_mm"] == 0.5


def test我方对没记精度的旧结果不误报为降精度(tmp_path: Path) -> None:
    """缺字段是"不知道"，不是"被降过精度"，两者不能混。"""

    pair_dir = tmp_path / "训练集__1"
    pair_dir.mkdir(parents=True)
    (pair_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "review",
                "reference_file": "D:/data/ref.stl",
                "candidate_file": "D:/data/c1.stp",
                "regions": [],
            }
        ),
        encoding="utf-8",
    )

    manifest = module.build_draft_manifest(tmp_path, lambda pair: "train")

    assert manifest["coarsened_sample_ids"] == []
    assert manifest["records"][0]["source_linear_deflection_mm"] is None
