"""我从一个参考 STL 生成一个带已知缺失面标签的训练样例。

用途：在没有甲方真实标注集时，造出"标签 100% 可信"的受控样本，
用它们评估缺失区域检测的召回和精度。因为删掉哪些面是我自己指定的，
所以 labels.json 里的 removed_face_ids 就是标准答案。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cad_inspector.geometry.stl import load_stl, save_stl
from cad_inspector.learning.synthetic_defects import remove_face_region


def main() -> int:
    """我读取参考 STL、删掉一块连续面、写出破损 STL 和标签 JSON。"""

    parser = argparse.ArgumentParser(description="我生成可复现合成缺失区域。")
    parser.add_argument("reference_stl")
    parser.add_argument("output_stl")
    parser.add_argument("--seed-face-id", type=int, default=0)
    parser.add_argument("--face-count", type=int, default=1)
    parser.add_argument("--label-output", default=None)
    args = parser.parse_args()
    reference = load_stl(args.reference_stl)
    # remove_face_region 按 Face 邻接向外扩，删掉的是"连续区域"而不是随机散点，
    # 这样造出来的缺陷更接近真实破损失效。
    synthetic = remove_face_region(
        reference,
        seed_face_id=args.seed_face_id,
        face_count=args.face_count,
    )
    save_stl(synthetic.candidate_mesh, args.output_stl, "synthetic_missing")
    # 标签默认跟破损 STL 同目录同名，只把扩展名换成 .labels.json。
    label_path = (
        Path(args.label_output)
        if args.label_output
        else Path(args.output_stl).with_suffix(".labels.json")
    )
    label_path.write_text(
        json.dumps(
            {
                # 我把参考件、破损件的绝对路径都写进标签，
                # 这样后续评估能核对"标签是不是配错了模型"。
                "reference_file": str(Path(args.reference_stl).resolve()),
                "candidate_file": str(Path(args.output_stl).resolve()),
                "removed_face_ids": synthetic.removed_face_ids,
                "defect_center_xyz_mm": synthetic.defect_center_xyz_mm,
                "defect_area_mm2": synthetic.defect_area_mm2,
                "defect_type": synthetic.defect_type,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"我已经生成合成缺陷：{args.output_stl}")
    print(f"我已经写出标签：{label_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
