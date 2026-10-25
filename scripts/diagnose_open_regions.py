"""我按形态给 STEP 模型的“开放边界区域”做聚类诊断。

我方在排查 MD270324 这类大模型的“开放边界候选”时发现，几千个区域里
绝大多数是同一两种 OCP 读取假象的重复，靠肉眼一个个看不现实。
这个脚本把每个开放区域压成一个“形态签名”（面类型 + 单端点自环数 +
非退化链数 + 端点数 + 是否有平行边），再按签名聚类，让我能一眼看出
该优先修哪一种形态，而不是被总数吓住。

用法：
    python -B scripts\\diagnose_open_regions.py <step 文件路径> [--top N]
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cad_inspector.detection.step_features import (  # noqa: E402
    classify_step_face_features,
    inspect_step_feature_classification,
)
from cad_inspector.io.step_faces import inspect_step_face_features  # noqa: E402


def _signature(region, faces_by_id) -> tuple[object, ...]:
    """我把一个区域压成形态签名，相同签名的区域按同一类问题处理。"""

    evidence = region.boundary_topology_evidence or {}
    self_loop_count = 0
    open_chain_count = 0
    for component in evidence.get("components") or []:
        for chain in component.get("open_chains") or []:
            if chain.get("degenerate_self_loop"):
                self_loop_count += 1
            else:
                open_chain_count += 1
    endpoints = evidence.get("open_endpoint_vertex_ids") or []
    face = faces_by_id.get(region.face_ids[0]) if region.face_ids else None

    # 我统计同一对端点之间挂着几条边：多于一条说明一个圆被切成了多段弧。
    pair_counts: collections.Counter[tuple[int, int]] = collections.Counter()
    if face is not None:
        for edge_id in region.boundary_edge_ids or []:
            vertices = face.edge_endpoint_vertex_ids.get(edge_id) or []
            if len(vertices) == 2:
                pair_counts[tuple(sorted(int(v) for v in vertices))] += 1
    max_parallel = max(pair_counts.values(), default=0)

    return (
        str(face.surface_type) if face is not None else None,
        self_loop_count,
        open_chain_count,
        len(endpoints),
        max_parallel,
    )


def main() -> int:
    """我诊断一个 STEP 的开放区域形态并按规模打印前若干个。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_path", help="STEP 文件路径")
    parser.add_argument("--top", type=int, default=15, help="打印前 N 个形态")
    args = parser.parse_args()

    # 我分开跑两步，为的是拿到 face 报告：区域只带 face_id，
    # 面类型要看 face 报告才能知道，形态签名里必须有它。
    face_report = inspect_step_face_features(args.step_path)
    report = classify_step_face_features(face_report)
    faces_by_id = {face.face_id: face for face in face_report.faces}
    status_counts: collections.Counter[str] = collections.Counter(
        region.boundary_topology_status for region in report.regions
    )
    # 先打总体分布再深入细节：如果开放区域本来就只有几个，
    # 后面的形态分析就没有规模意义，这一步能立刻判断要不要继续查。
    print(f"模型：{args.step_path}")
    print(f"区域总数：{len(report.regions)}")
    # most_common 按数量降序：开放区域诊断最关心"哪类问题最多"。
    for status, count in status_counts.most_common():
        print(f"  {status}: {count}")
    print()

    open_regions = [
        region
        for region in report.regions
        if region.boundary_topology_status == "OPEN_BOUNDARY_CANDIDATE"
    ]
    print(f"开放边界区域：{len(open_regions)}")

    # 按"形态签名"分桶而不是逐区域打印：开放边界在大模型上常有几百个，
    # 逐条打印无法阅读；而它们往往只有几种成因，
    # 按签名归并后修一种形态就能解决一批问题。
    signatures: collections.Counter[tuple[object, ...]] = collections.Counter()
    examples: dict[tuple[object, ...], object] = {}
    for region in open_regions:
        key = _signature(region, faces_by_id)
        signatures[key] += 1
        # setdefault 保留每类的第一个例子，作为人工打开模型查看的入口。
        examples.setdefault(key, region)

    print()
    print("形态签名（面类型, 单端点自环数, 开放链数, 端点数, 最大平行边数）-> 区域数")
    for key, count in signatures.most_common(args.top):
        print(f"  {key} -> {count}")

    print()
    print("每类形态的第一个例子：")
    # 只打印前 5 类的例子：这个脚本是给人看的定位工具，
    # 例子太多就失去了"快速定位"的意义。
    for key, count in signatures.most_common(min(5, args.top)):
        region = examples[key]
        evidence = region.boundary_topology_evidence or {}
        print(f"  region {region.region_id}  同形态 {count} 个")
        print(f"    face_ids {region.face_ids}")
        print(f"    boundary_edge_ids {region.boundary_edge_ids}")
        print(f"    endpoints {evidence.get('open_endpoint_vertex_ids')}")
        face = faces_by_id.get(region.face_ids[0]) if region.face_ids else None
        if face is not None:
            print(f"    surface_type {face.surface_type}")
            # 逐边打印端点与长度：长度能立刻区分"极短公差裂缝"
            # 与"真实缺失面"，这是判断要不要补面的关键依据。
            for edge_id in region.boundary_edge_ids or []:
                vertices = face.edge_endpoint_vertex_ids.get(edge_id) or []
                length = None
                # 长度按 face.edge_ids 的下标取：长度数组与边编号数组
                # 是一一对应的，必须先定位下标再取值。
                if edge_id in face.edge_ids:
                    length = face.edge_lengths_mm[
                        face.edge_ids.index(edge_id)
                    ]
                print(
                    f"      edge {edge_id} verts {vertices}"
                    f" len {round(length, 4) if length is not None else None}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
