"""我方在真实水密网格上挖已知孔洞，量化局部修补成功率。

原理说明：任务书要求"修补成功率 ≥ 92%"，但甲方给的真实 STL 全部是水密
闭合网格（`boundary_edge_count = 0`），没有现成开放孔洞可直接统计。
我方的做法是：拿这些真实零件网格当"完整模型"，用
`learning\\synthetic_defects.remove_face_region` 按共享边邻接删掉一块
连续三角面，造出**面积和位置都已知**的孔洞，再跑项目自己的
`repair_mesh_holes`，最后用输出网格的边界边数判断孔洞是否真的被补上。

这样得到的成功率是"同一套几何、同一套修补器、可复现的合成基准"，
不是甲方标注真值上的成功率；两者不能混为一谈，我在报告里分开写。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT / "src",):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from cad_inspector.domain.repair import RepairConfig  # noqa: E402
from cad_inspector.geometry.mesh import mesh_quality_report  # noqa: E402
from cad_inspector.geometry.stl import load_stl  # noqa: E402
from cad_inspector.learning.synthetic_defects import remove_face_region  # noqa: E402
from cad_inspector.output.atomic import atomic_path  # noqa: E402
from cad_inspector.output.repair_report import (  # noqa: E402
    write_repair_csv,
    write_repair_json,
)
from cad_inspector.repair import repair_mesh_holes  # noqa: E402


SCHEMA_VERSION = "mesh-repair-benchmark-1.0"
DEFAULT_DATA_ROOT = ROOT / "data" / "provided" / "三维模型破损区域智能修补"
DEFAULT_FACE_COUNTS = (1, 8, 64)


def build_config(args: argparse.Namespace) -> RepairConfig:
    """我按命令行参数构造修补配置，门禁值完全由调用方显式给出。"""
    return RepairConfig(
        max_boundary_edges=args.max_boundary_edges,
        max_hole_area_mm2=args.max_hole_area_mm2,
        max_planarity_error_mm=args.max_planarity_error_mm,
        allow_nonconvex=args.allow_nonconvex,
    )


def case_seed(face_total: int, case_index: int) -> int:
    """我用固定公式挑种子面，保证每次运行挖的是同一块区域。

    我为什么用 7919 这个步长：7919 是素数，与常见的面数（往往是
    2 的幂或整百）互素，于是 case_index 递增时种子面会在整件上
    均匀散开，而不是每隔几个就落回同一片区域。用整百步长的话
    不同 case 会反复挖同一处，基准就失去了覆盖意义。
    """

    return (case_index * 7919 + 13) % face_total


def find_meshes(
    data_root: Path, max_bytes: int, only: list[str] | None
) -> list[tuple[str, Path]]:
    """我按路径排序收集基准网格，按体积上限与子串过滤。

    我为什么显式按路径字符串排序：``rglob`` 的返回次序取决于文件系统，
    在不同机器上可能不同。基准要能横向比对就必须有确定的执行顺序，
    否则两次跑出来的"第 3 个 case"根本不是同一个件。
    """
    meshes: list[tuple[str, Path]] = []
    for path in sorted(data_root.rglob("*.stl"), key=lambda item: str(item)):
        name = f"{path.parent.name}/{path.stem}"
        if only and not any(token in name for token in only):
            continue
        if path.stat().st_size > max_bytes:
            continue
        meshes.append((name, path))
    return meshes


def run_case(
    pair_name: str,
    stl_path: Path,
    reference,
    face_count: int,
    case_index: int,
    config: RepairConfig,
    case_dir: Path,
) -> dict[str, object]:
    """我挖一个已知孔洞、跑一次修补，并记录前后边界证据。

    基准的可信度取决于三件事，缺一个这份汇总就没有说服力：
      1. **缺陷是受控生成的**。我用固定公式挑种子面，同一
         face_total 与 case_index 每次挖到同一块区域，所以两次运行的
         结果可以直接对照，不会出现"这次成功下次失败却说不清为什么"；
      2. **判据是硬的**。我用输出网格的边界边数是否降到 0 作为
         hole_closed——这是拓扑事实，不受阈值和评分影响。
         只看"区域状态是 repaired"会被自欺：区域报成功但网格仍有开口
         是可能发生的，只有边界边数不会说谎；
      3. **同时记录被挖掉和补回来的量**。removed_area_mm2 与
         repaired_area_mm2 并列，能直接看出补片是偏大还是偏小，
         而不只是"补没补上"这个二值结论。

    一个容易读错的字段：区域面积挂在 region.boundary_loop.area_mm2 上，
    不是 region.area_mm2，后者在本修复实现里并不承载补片面积。
    """

    # 固定公式选种子面：同一 face_total 与 case_index 每次挖到同一块区域，结果可复现。
    seed = case_seed(len(reference.faces), case_index)
    defect = remove_face_region(reference, seed_face_id=seed, face_count=face_count)
    damaged = defect.candidate_mesh
    before = mesh_quality_report(damaged)
    started = time.perf_counter()
    repaired, result = repair_mesh_holes(
        damaged, config, input_file=str(stl_path)
    )
    elapsed = time.perf_counter() - started
    after = mesh_quality_report(repaired)
    case_dir.mkdir(parents=True, exist_ok=True)
    write_repair_json(result, case_dir / "repair.json")
    write_repair_csv(result, case_dir / "repair_regions.csv")
    # 区域面积挂在 region.boundary_loop.area_mm2 上，不是 region.area_mm2。
    repaired_area = sum(
        region.boundary_loop.area_mm2 for region in result.repaired_regions
    )
    added_triangles = sum(
        region.added_triangle_count for region in result.repaired_regions
    )
    # hole_closed 是最硬的判据：输出网格边界边数降到 0 才算真的补上。
    return {
        "case": f"{pair_name}#face{face_count}#{case_index}",
        "pair": pair_name,
        "stl_file": str(stl_path.resolve()),
        "seed_face_id": seed,
        "requested_face_count": face_count,
        "removed_face_count": len(defect.removed_face_ids),
        "removed_area_mm2": defect.defect_area_mm2,
        "damaged_boundary_edge_count": before["boundary_edge_count"],
        "damaged_non_manifold_edge_count": before["non_manifold_edge_count"],
        "repair_status": result.status,
        "repaired_region_count": len(result.repaired_regions),
        "repaired_area_mm2": repaired_area,
        "added_triangle_count": added_triangles,
        "region_methods": sorted(
            {region.method for region in result.repaired_regions}
        ),
        "region_statuses": sorted(
            {region.status for region in result.repaired_regions}
        ),
        "output_boundary_edge_count": after["boundary_edge_count"],
        "output_non_manifold_edge_count": after["non_manifold_edge_count"],
        "hole_closed": after["boundary_edge_count"] == 0,
        "elapsed_seconds": round(elapsed, 3),
        "warning_count": len(result.warnings),
        "warnings": result.warnings,
    }


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    """我汇总修补成功率；closed 与 closed_with_region 分开统计。

    我为什么要分开统计这两个数：边界边数降到 0 说明"洞没了"，
    但如果同时**没有任何修补区域记录**，那洞多半不是"补上的"，
    而是被别的路径（例如退化网格重建）消掉了。这种情况下报
    "100% 修补成功"是自欺——甲方拿去验收会发现补片几何根本不存在。
    所以 hole_closed_ratio 是乐观口径，
    hole_closed_with_region_ratio 才是能拿去汇报的可信口径。

    两个比例的分母都是**全部用例数**（含失败），不是成功数，
    否则失败越多比例反而越高。
    """
    closed = [record for record in records if record.get("hole_closed")]
    # 补上了但没有任何修补区域记录，说明结果可疑，单独统计不与真成功混在一起。
    closed_with_region = [
        record
        for record in closed
        if isinstance(record.get("repaired_region_count"), int)
        and record["repaired_region_count"] > 0
    ]
    elapsed = [
        float(record["elapsed_seconds"])
        for record in records
        if isinstance(record.get("elapsed_seconds"), (int, float))
    ]
    return {
        "case_count": len(records),
        "hole_closed_count": len(closed),
        "hole_closed_ratio": (
            round(len(closed) / len(records), 6) if records else None
        ),
        "hole_closed_with_region_count": len(closed_with_region),
        "hole_closed_with_region_ratio": (
            round(len(closed_with_region) / len(records), 6) if records else None
        ),
        "elapsed_seconds": {
            "min": min(elapsed) if elapsed else None,
            "median": statistics.median(elapsed) if elapsed else None,
            "max": max(elapsed) if elapsed else None,
        },
        "by_face_count": _by_face_count(records),
        "by_status": _count_by(records, "repair_status"),
    }


def _by_face_count(records: list[dict[str, object]]) -> dict[str, object]:
    """我按挖孔面数分桶统计，便于看出大孔洞是否明显更难补。"""
    buckets: dict[str, list[dict[str, object]]] = {}
    for record in records:
        buckets.setdefault(str(record.get("requested_face_count")), []).append(record)
    return {
        key: {
            "case_count": len(items),
            "hole_closed_count": sum(bool(item.get("hole_closed")) for item in items),
        }
        for key, items in sorted(buckets.items(), key=lambda pair: int(pair[0]))
    }


def _count_by(records: list[dict[str, object]], field: str) -> dict[str, int]:
    """我按指定字段统计分布，用于查看状态与方法的分布。

    我为什么把键统一转成字符串：同一个字段在不同记录里可能是 int
    也可能是 str（例如某些 case 没算出值就是 None），混用类型会让
    字典里出现 "1" 与 1 两个键，统计被劈成两半。统一转字符串后
    缺失值也会归到 "None" 这一类，不会被静默丢掉。
    """
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get(field))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_reports(
    output_root: Path,
    data_root: Path,
    config: RepairConfig,
    records: list[dict[str, object]],
    skipped: list[dict[str, object]],
) -> None:
    """我写出基准汇总 JSON 与逐用例 CSV，两份都走原子替换。"""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "data_root": str(data_root.resolve()),
        "config": config.to_dict(),
        "case_count": len(records),
        "skipped_mesh_count": len(skipped),
        "skipped_meshes": skipped,
        "aggregate": aggregate(records),
        "records": records,
    }
    with atomic_path(output_root / "summary.json") as temporary:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    fields = [
        "case",
        "pair",
        "seed_face_id",
        "requested_face_count",
        "removed_face_count",
        "removed_area_mm2",
        "damaged_boundary_edge_count",
        "repair_status",
        "repaired_region_count",
        "repaired_area_mm2",
        "output_boundary_edge_count",
        "hole_closed",
        "elapsed_seconds",
        "warning_count",
    ]
    with atomic_path(output_root / "summary.csv") as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({field: record.get(field) for field in fields})


def _build_parser() -> argparse.ArgumentParser:
    """我构造修补基准命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="我方在真实水密网格上挖已知孔洞并量化修补成功率。"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "repair_benchmark",
    )
    parser.add_argument(
        "--max-stl-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="我方只把不超过该体积的 STL 当作基准网格，控制时间预算。",
    )
    parser.add_argument("--face-counts", type=int, nargs="+", default=list(DEFAULT_FACE_COUNTS))
    parser.add_argument("--max-boundary-edges", type=int, default=128)
    parser.add_argument("--max-hole-area-mm2", type=float, default=10_000.0)
    # 我为什么不再硬编码 default：这里原来写死 0.05、`store_true`(默认
    # False)，会把 `RepairConfig` 里的默认值**悄悄覆盖掉**——我改了
    # dataclass 的默认值，基准里却一点没生效（实测跑出来的还是旧配置）。
    # 默认值只在 RepairConfig 里定一处，CLI 从这里取，才不会两处打架。
    parser.add_argument(
        "--max-planarity-error-mm",
        type=float,
        default=RepairConfig().max_planarity_error_mm,
    )
    # BooleanOptionalAction 同时给 --allow-nonconvex 和 --no-allow-nonconvex，
    # 这样默认值变成 True 之后仍然能显式关掉（原来的 store_true 只能开不能关）。
    parser.add_argument(
        "--allow-nonconvex",
        action=argparse.BooleanOptionalAction,
        default=RepairConfig().allow_nonconvex,
    )
    parser.add_argument("--only", action="append", default=None)
    return parser


def main() -> int:
    """我跑完整基准并打印汇总；全部用例失败返回 1，否则返回 0。"""
    args = _build_parser().parse_args()
    if args.max_stl_bytes <= 0:
        raise ValueError("我要求 --max-stl-bytes 大于 0。")
    if any(count < 1 for count in args.face_counts):
        raise ValueError("我要求 --face-counts 全部至少为 1。")
    config = build_config(args)
    data_root = args.data_root.expanduser()
    output_root = args.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    # 非水密网格不能当挖洞基准：它本身就有洞，补完之后无法判断补的是不是新挖的。
    for pair_name, stl_path in find_meshes(data_root, args.max_stl_bytes, args.only):
        reference = load_stl(stl_path)
        quality = mesh_quality_report(reference)
        # 非水密的网格不能当挖洞基准：它本来就带洞，
        # 再挖一个洞得到的"修复后边界数"无法归因到挖的那个洞上。
        if quality["boundary_edge_count"] != 0:
            skipped.append(
                {
                    "pair": pair_name,
                    "stl_file": str(stl_path.resolve()),
                    "reason": "参考网格本身不是水密闭合网格，不能作为挖洞基准。",
                    # 把实际边界边数一并记下：它能说明这个件
                    # 是"差一点点"还是"到处都是洞"。
                    "boundary_edge_count": quality["boundary_edge_count"],
                }
            )
            # 跳过原因即时打印：跑几十个件时看不到进度
            # 就无法区分"在跑"和"卡住了"。
            print(f"[skip] {pair_name} 不是水密网格", flush=True)
            continue
        # 同一个件按不同挖洞面数各跑一遍：
        # 面数是这个基准里唯一的自变量，
        # 交叉它能看出"洞越大修补成功率掉多少"。
        for case_index, face_count in enumerate(args.face_counts):
            case_dir = output_root / f"{pair_name.replace('/', '__')}_face{face_count}"
            record = run_case(
                pair_name,
                stl_path,
                # 参考网格传进去而不是每个用例各读一次：
                # 同一个件的多个用例共用同一份网格，
                # 重复读盘只是浪费时间。
                reference,
                face_count,
                case_index,
                config,
                case_dir,
            )
            records.append(record)
            # 每跑完一个用例就重写报告：中途被掐断时
            # 已跑完的部分不会丢。
            write_reports(output_root, data_root, config, records, skipped)
            print(
                f"[{record['repair_status']}] {record['case']} "
                # 挖掉的面积、输出边界数、是否闭合三个量一起打印：
                # 只看是否闭合无法区分"没修上"和"修上了但还有别的洞"。
                f"removed_area={record['removed_area_mm2']:.4f} "
                f"out_boundary={record['output_boundary_edge_count']} "
                f"closed={record['hole_closed']}",
                flush=True,
            )
    # 收尾再写一次并打印汇总：逐条明细已经写过，
    # 这一步给出的是整批的聚合数字。
    write_reports(output_root, data_root, config, records, skipped)
    print(json.dumps(aggregate(records), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
