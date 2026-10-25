"""我把验收脚本跑出的真实结果转成**待人工复核的标注草稿**。

算法原理：这个项目卡在"算不出准确率"上，根因不是代码不会算，而是
**没有甲方真实标注**——没有真值就无法算召回率和 IoU。而标注集不可能凭空
出现，最现实的路径是"机器先出候选、人来确认/删除"。

本脚本做的正是这条路的起点：读验收结果目录里的 result.json，
按标注 schema 拼成清单，让每一条都带上 `provenance=draft` 标记。

我为什么坚持打 draft 标记而不是直接当真值：
这是**机器预测**，不是人工确认过的真值。拿预测当真值去评估同一个
预测器，会得到 100% 的假准确率——自己给自己打分。草稿必须能一眼
被认出来是未确认的，且不能悄悄流进训练/评估。

2026-10-25 我补上了多余材料（extra）的真值面号：检测器每件都会报
extra 区域，但草稿此前只抄 reference_face_ids，而 extra 的面号在
**候选网格**上（candidate_face_ids），于是"人工确认 -> 打分"这条链
对多料永远闭合不了。现在 extra 区域写 candidate_face_ids、
missing 区域写 reference_face_ids，两边不混。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 我把项目根的 src 插进 sys.path：本脚本作为 CLI 直接运行时（而非经
# pytest 的 conftest 导入时）不会自动带上 src，import cad_inspector
# 会 ModuleNotFoundError。参照 scripts/run_real_pair_acceptance.py:27 的写法。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 我故意不 import annotations 模块里的 SCHEMA_VERSION：
# 本脚本是**生成**清单，生成后再用 validate-annotations 校验，
# 两边都读同一个常量才不会漂移。
from cad_inspector.learning.annotations import SCHEMA_VERSION
from cad_inspector.models import InspectionConfig


def _pair_name(directory: Path) -> str:
    """我把输出目录名还原回"训练集/0"这种语料名。"""

    return directory.name.replace("__", "/")


def _region_to_draft(region: dict[str, Any]) -> dict[str, Any]:
    """我把一个检出区域转成标注区域，只保留标注 schema 认得的字段。

    面号我按缺陷类型分开写（2026-10-25 补）：
    - 缺料的真值在**参考网格**上，写进 reference_face_ids；
    - 多余材料的真值在**候选网格**上，写进 candidate_face_ids。

    检测结果里 missing 区域也会带 candidate_face_ids，但那是"采样落点"
    （每个参考面的采样点在候选件上的最近三角面），**不是真值**。
    我把它清空而不是照抄：留着它会被 extra 口径的打分器当成多料的真值，
    凭空造出一批假命中。
    """

    defect_type = str(region.get("defect_type", "missing"))
    draft: dict[str, Any] = {
        "defect_type": defect_type,
        "center_xyz_mm": region.get("center_xyz_mm"),
        "bbox_min_xyz_mm": region.get("bbox_min_xyz_mm"),
        "bbox_max_xyz_mm": region.get("bbox_max_xyz_mm"),
        "area_mm2": region.get("area_mm2"),
        "boundary_edge_ids": list(region.get("boundary_edge_ids") or []),
    }
    if defect_type == "extra":
        # 多出来的料在参考件上并不存在，参考面号必须留空；
        # 本地校验（annotations.py，1.1）也会因此拒绝"两侧都填"的区域。
        draft["reference_face_ids"] = []
        draft["candidate_face_ids"] = list(region.get("candidate_face_ids") or [])
    else:
        draft["reference_face_ids"] = list(region.get("reference_face_ids") or [])
        draft["candidate_face_ids"] = []
    # 我把机器侧的置信度和误差量一起带出来：人工复核时
    # 最需要的排序依据就是"哪些区域最可疑"。
    for key in ("confidence", "max_error_mm", "mean_error_mm", "p95_error_mm"):
        if region.get(key) is not None:
            draft[key] = region[key]
    return draft


def build_draft_manifest(
    acceptance_root: Path,
    split_of_pair: callable,
) -> dict[str, Any]:
    """我遍历验收输出目录，拼出一份待复核的标注草稿清单。"""

    # 我为什么要盯三角化精度（2026-10-16 实测查出来的）：
    # 验收脚本在单组超时后会自动**降精度重试**（0.2mm → 0.5mm），
    # 而结果里只记在 config.mesh_linear_deflection_mm 里。
    # 实测对照：同一对 训练集/0，0.2mm 精度下报 26519 个缺料面、
    # 0.04mm 精度下报 48810 个——差 84%，区域数也从 27 变 23。
    # 也就是说"哪几件被降过精度"直接决定了草稿里的区域长什么样。
    # 人工复核时若不知道手上这条来自粗网格，就会把三角化误差
    # 当成真实缺料来确认。所以我把精度写进每条记录，并单独列出
    # 被降过精度的 sample_id。
    default_deflection = float(InspectionConfig().mesh_linear_deflection_mm)
    coarsened_sample_ids: list[str] = []
    records: list[dict[str, Any]] = []
    for result_path in sorted(acceptance_root.glob("*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        pair = _pair_name(result_path.parent)
        regions = payload.get("regions") or []
        sample_id = pair.replace("/", "_")
        deflection = (payload.get("config") or {}).get(
            "mesh_linear_deflection_mm"
        )
        if deflection is not None and float(deflection) > default_deflection:
            coarsened_sample_ids.append(sample_id)
        records.append(
            {
                "sample_id": sample_id,
                "split": split_of_pair(pair),
                "reference_file": payload.get("reference_file"),
                "candidate_file": payload.get("candidate_file"),
                # 我为什么把 provenance 写进每一条而不是只写在顶层：
                # 人工复核是**逐条**增删的，顶层标记会在复制粘贴中丢掉，
                # 写在记录里才能跟着记录走。
                "provenance": "draft",
                "review_state": "unconfirmed",
                "source_status": payload.get("status"),
                # 这条草稿是在什么三角化精度下生成的；数值越大越粗。
                "source_linear_deflection_mm": deflection,
                "regions": [_region_to_draft(r) for r in regions],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        # 顶层也标一次，方便拿到文件的人第一眼就知道这是草稿。
        "provenance": "draft",
        "note": (
            "本清单由机器预测自动生成，未经人工确认，"
            "不能直接当作真值用于准确率评估。"
        ),
        # 我为什么必须记来源（2026-10-16 实测踩到）：
        # 仓库里那份 reports/draft_annotations.json 没有任何来源字段，
        # 我要判断它"是不是已经被新管线跑陈旧了"时，只能把 outputs 下
        # 57 份 result.json 全扫一遍、逐条比对面号，才认出它出自
        # real_pair_acceptance_v3。没有来源的草稿是不可复核的：
        # 既说不清它对应哪次检测，也就无法判断人工确认的对象是不是
        # 当前管线真正会输出的东西——确认一份旧预测等于把结论建在
        # 已经不存在的输出上。
        "source_acceptance_root": str(
            Path(acceptance_root).expanduser().resolve()
        ),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "source_default_linear_deflection_mm": default_deflection,
        # 被降过精度重试的件：这些记录的区域形状受三角化精度影响，
        # 复核时要格外当心，不能当成和全精度件同等可信。
        "coarsened_sample_ids": sorted(coarsened_sample_ids),
        "records": records,
    }


def main() -> int:
    """我按命令行参数生成标注草稿清单并写出 JSON。"""

    parser = argparse.ArgumentParser(
        description="我把验收结果转成待人工复核的标注草稿清单。"
    )
    parser.add_argument(
        "--acceptance-root",
        type=Path,
        required=True,
        help="验收脚本的 --output-root 目录，其下每个子目录含 result.json。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="我写出的标注草稿 JSON 路径。",
    )
    args = parser.parse_args()

    def split_of_pair(pair: str) -> str:
        """我按语料自带的目录名定 split：训练集进 train，其余进 test。"""

        # 训练集/测试集的划分直接用语料自带的目录名，
        # 我不自己发明划分规则——语料怎么分的就怎么记。
        return "train" if pair.startswith("训练集") else "test"

    manifest = build_draft_manifest(args.acceptance_root, split_of_pair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total_regions = sum(len(r["regions"]) for r in manifest["records"])
    print(f"我写出标注草稿：{args.output}")
    print(f"  记录 {len(manifest['records'])} 条，区域 {total_regions} 个")
    print("  注意：这是机器预测，未经人工确认，不能当真值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
