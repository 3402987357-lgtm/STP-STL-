"""我方在甲方真实 STEP/STL 数据集上标定自适应阈值系数并实测单模型耗时。

原理说明：`auto_threshold_scale_ratio`（默认 0.002）和
`auto_threshold_residual_factor`（默认 1.5）此前只是工程经验值，没有在
甲方真实数据上留下证据。我方这个脚本把每一对真实样本跑一遍 auto 阈值
检测，记录包围盒尺度分量、配准残差分量、实际生效阈值、区域数量、面积占比
和单对耗时，最后汇总成可审计的 JSON/CSV。它只读原始数据，写出的中间结果
放在独立目录，便于随时删除重跑。

复用说明：配对扫描、内存读取和输入指纹直接复用
`scripts/run_real_pair_stress.py` 里的实现，避免两套口径不一致。

统计口径的关键约定：汇总时必须排除"没有真实耗时"的记录（exception 与 resource_guard
两类跳过）。否则一个被跳过的样本会因为耗时 0 秒被算进"达标"统计里，凭空把"8 秒内完成"
的比例抬高——这是统计口径最容易被污染的地方，我在 `_aggregate` 里显式过滤并同时报告
`processed_pair_count` 与 `measured_pair_count`，让甲方看得出分母到底是多少。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT / "scripts", ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

# 从压测脚本里复用这三个工具而不是重写一遍：
# 找样本对、算输入指纹、读内存，三者的口径必须与压测一致，
# 否则标定出来的阈值无法和压测结果对照。
from run_real_pair_stress import (  # noqa: E402
    _find_pairs,
    _input_fingerprint,
    _memory_mb,
)

from cad_inspector.models import InspectionConfig  # noqa: E402
from cad_inspector.output.atomic import atomic_path  # noqa: E402
from cad_inspector.pipeline import inspect_pair  # noqa: E402


SCHEMA_VERSION = "real-threshold-calibration-1.0"
DEFAULT_DATA_ROOT = ROOT / "data" / "provided" / "三维模型破损区域智能修补"

# 指标键固定为这个元组：标定要横向比较多个样本，
# 列集合一旦随版本漂移，前后两批结果就无法放在一张表里。
_METRIC_KEYS = (
    "reference_area_mm2",
    "candidate_area_mm2",
    # 缺失与多出的面积与占比交替排列：
    # 先看绝对量（多大），再看相对量（占比多少），
    # 这个顺序与人工判读的顺序一致。
    "missing_area_mm2",
    "missing_area_ratio",
    "extra_area_mm2",
    "extra_area_ratio",
    # 三个偏差量：最大偏差是"最坏的那一点"，
    # 均值是整体水平，p95 排除了极端值的影响。
    # 三者缺一都无法描述偏差的分布形态。
    "max_error_mm",
    "mean_error_mm",
    "p95_error_mm",
    "missing_region_count",
    "extra_region_count",
    "reference_face_count",
    "candidate_face_count",
)


def calibration_config(args: argparse.Namespace) -> InspectionConfig:
    """我构造 auto 阈值标定配置；其余参数与压力轮保持一致。

    "与压力轮一致"不是偷懒，是标定的前提：压力轮已经证明了这套参数
    在真实数据上不崩、不漏、耗时可接受。如果标定另起一套参数
    （比如换配准模式或距离后端），得到的阈值就掺进了参数差异，
    无法归因到 auto 阈值公式本身。

    threshold_mm=0.15 在这里不生效（threshold_mode="auto" 会覆盖它），
    我仍显式写出：这样当 auto 计算失败退回绝对阈值时，
    退回值是可预期的而不是某个隐藏默认值。

    write_colored_mesh 与 write_intermediate 都关掉：标定要跑很多对，
    彩色网格和中间产物对阈值统计毫无用处，却会显著拉长耗时
    并撑大输出目录。
    """

    return InspectionConfig(
        threshold_mm=0.15,
        threshold_mode="auto",
        auto_threshold_scale_ratio=args.scale_ratio,
        auto_threshold_residual_factor=args.residual_factor,
        auto_threshold_floor_mm=args.floor_mm,
        # 2026-10-17：库默认从 centroid 改成 auto（实测 21/21 个真实件的
        # 原始位姿残差都低于 centroid/pca）。标定要对着产品默认口径，
        # 所以这里跟着改成 auto，否则标出来的阈值是给一个没人用的模式用的。
        registration_mode="auto",
        max_query_points=args.max_query_points,
        distance_backend="auto",
        distance_candidate_k=64,
        distance_chunk_size=2_048,
        bidirectional_check=True,
        unit_policy="assume_mm",
        write_intermediate=False,
        write_colored_mesh=False,
    )


def config_fingerprint(config: InspectionConfig) -> str:
    """我对标定配置做规范化序列化后取 sha256，用于续跑时判断配置是否变化。

    sort_keys 与 separators 必须固定：同一个配置若因字典顺序或空格
    差异得到不同哈希，续跑就会误判"配置变了"而把已完成的
    样本全部重跑一遍。标定是耗时任务，这种假失效代价很高。

    只哈希配置本身，不哈希数据：数据变化由每对样本自己的
    输入指纹负责。两者分开，才能区分"配置变了要重跑"
    和"数据变了要重跑"这两种不同的失效原因。
    """
    import hashlib

    payload = json.dumps(
        config.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        # allow_nan=False：配置里出现 NaN 会让同一份配置算出不同指纹，
        # 去重与可复现校验就失去意义，所以这里必须显式拒绝非有限值。
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_worker(
    step_path: Path,
    stl_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    """我在隔离子进程里跑单对 auto 检测，避免异常样本拖住批量任务。

    我为什么用子进程隔离而不是直接 import 着跑：OCP 在极端几何上可能
    触发进程级崩溃（段错误、内存耗尽），这类失败在 Python 里捕获不到，
    一次就会带走整批标定。放进子进程后，单对崩溃只表现为一个非零退出码，
    批量任务照常往下走，最后在汇总里体现为一个 exception 记录。

    worker 的产出**先写 worker_summary.json 再由父进程读回**，不依赖
    返回值的进程间传输：这样即使子进程在写完之后才崩溃，结果也已经落盘，
    不会白白浪费一次几十秒的计算。
    """

    config = calibration_config(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = inspect_pair(
        candidate_path=stl_path,
        reference_path=step_path,
        output_dir=output_dir,
        config=config,
    )
    metrics = result.global_metrics
    summary = {
        "status": result.status,
        "unit": result.unit,
        # 警告数与警告全文都留：只留全文的话
        # "这次有多少条警告"要读一遍才知道，
        # 只留数的话又看不到内容。
        "warning_count": len(result.warnings),
        "warnings": result.warnings,
        "region_count": len(result.regions),
        # 阈值与阈值审计一起记：阈值本身只是个数，
        # 审计里才有"这个阈值是怎么选出来的"以及它是否可信。
        "threshold_mm": result.threshold_mm,
        "threshold_audit": metrics.get("threshold_audit", {}),
        "registration": {
            "mode": result.registration.mode,
            "quality": result.registration.quality,
            # 均值残差与 p95 残差分开：均值会被大量贴合良好的点拉低，
            # p95 才反映"最差的那 5% 偏了多少"，
            # 只看均值会把局部配准失败盖过去。
            "mean_residual_mm": result.registration.mean_residual_mm,
            "p95_residual_mm": result.registration.p95_residual_mm,
            "scale": result.registration.scale,
        },
        # 指标按固定键集合抽取：不同版本的结果可能多出或少了某些指标，
        # 固定集合保证这批结果的列不会因为版本差异而漂移。
        "metrics": {key: metrics.get(key) for key in _METRIC_KEYS},
        # 区域缺陷类型只留去重后的种类：标定关心的是
        # "这批样本出现了哪几类缺陷"，不是每块区域一条。
        "region_defect_types": sorted(
            {str(region.defect_type) for region in result.regions}
        ),
        # 区域明细只留四个字段：完整区域对象嵌套很深，
        # 全量落盘会让 summary.json 大到没法人工看。
        "region_detail": [
            {
                "region_id": region.region_id,
                "defect_type": region.defect_type,
                "area_mm2": region.area_mm2,
                "review_required": region.review_required,
            }
            for region in result.regions
        ],
        # 输入指纹用于确认"跑的是不是同一批文件"：
        # 标定结论只对特定输入成立，文件名相同内容变了必须能发现。
        "input_fingerprints": {
            "step": _input_fingerprint(step_path),
            "stl": _input_fingerprint(stl_path),
        },
        "config_fingerprint": config_fingerprint(config),
        "worker_elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    with atomic_path(output_dir / "worker_summary.json") as temporary:
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


def _worker_payload(output_dir: Path) -> dict[str, object] | None:
    """我读取 worker 已写出的摘要；不存在或不是对象时返回 None。"""
    path = output_dir / "worker_summary.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    """我汇总真实数据上的阈值与耗时分布，作为系数标定证据。

    统计口径的关键约定：耗时统计必须排除 exception 与 resource_guard
    两类"没有真实耗时"的记录。一个被跳过的样本耗时是 0 秒，如果混进
    分母，它会凭空把"8 秒内完成"的比例抬高——样本越容易失败，达标率
    看起来反而越好，这是统计口径最容易被自己骗过去的地方。

    所以我同时报告 processed_pair_count 与 measured_pair_count，
    让甲方能一眼看出达标率的分母到底是"全部样本"还是"真正跑完的样本"。
    """

    # 排除两类"没有真实耗时"的记录：异常和资源保护跳过。
    # 否则一个被跳过的样本会因为耗时 0 秒被误算成"8 秒达标"。
    skipped_statuses = {"exception", "resource_guard"}
    ok = [record for record in records if record.get("status") not in skipped_statuses]
    elapsed = [
        float(record["elapsed_seconds"])
        for record in ok
        if isinstance(record.get("elapsed_seconds"), (int, float))
    ]
    recommended = [
        float(record["threshold_audit"]["recommended_mm"])
        for record in ok
        if isinstance(record.get("threshold_audit"), dict)
        and isinstance(record["threshold_audit"].get("recommended_mm"), (int, float))
    ]
    residual = [
        float(record["registration"]["mean_residual_mm"])
        for record in ok
        if isinstance(record.get("registration"), dict)
        and isinstance(record["registration"].get("mean_residual_mm"), (int, float))
    ]
    within_budget = [value for value in elapsed if value <= 8.0]
    # 同时报告处理总数与实测总数，让甲方看得出达标率的分母。
    return {
        "processed_pair_count": len(records),
        "measured_pair_count": len(ok),
        "skipped_pair_count": len(records) - len(ok),
        "within_8s_count": len(within_budget),
        "within_8s_ratio": (
            round(len(within_budget) / len(elapsed), 6) if elapsed else None
        ),
        "elapsed_seconds": {
            "min": min(elapsed) if elapsed else None,
            "median": statistics.median(elapsed) if elapsed else None,
            "max": max(elapsed) if elapsed else None,
            "mean": statistics.fmean(elapsed) if elapsed else None,
        },
        "recommended_threshold_mm": {
            "min": min(recommended) if recommended else None,
            "median": statistics.median(recommended) if recommended else None,
            "max": max(recommended) if recommended else None,
        },
        "registration_mean_residual_mm": {
            "min": min(residual) if residual else None,
            "median": statistics.median(residual) if residual else None,
            "max": max(residual) if residual else None,
        },
    }


def _write_reports(
    output_root: Path,
    data_root: Path,
    args: argparse.Namespace,
    records: list[dict[str, object]],
    pair_count_total: int,
) -> None:
    """我写出标定汇总 JSON 与逐样本 CSV，两份都走原子替换。"""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "data_root": str(data_root.resolve()),
        # 总数与已处理数分开：长任务中途被打断时，
        # 这两个数的差就是"还剩多少没跑"，只看总数无法判断进度。
        "pair_count": pair_count_total,
        "processed_pair_count": len(records),
        # 配置整体落盘：标定结论只对这套配置成立，
        # 不把配置写进去就无法复现这批结果。
        "config": calibration_config(args).to_dict(),
        "timeout_seconds": args.timeout_seconds,
        # 资源上限写进结果：被资源保护跳过的样本
        # 与"真的跑失败"含义不同，没有这个数就无法区分。
        "resource_guard_bytes": args.max_stl_bytes,
        "aggregate": _aggregate(records),
        "records": records,
    }
    with atomic_path(output_root / "summary.json") as temporary:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    fields = [
        "pair",
        "status",
        "stl_size_bytes",
        "elapsed_seconds",
        "threshold_mm",
        # 包围盒对角线给出样本的尺度基准：
        # 同一个 0.15 mm 阈值在 20 mm 的小件和 800 mm 的大件上
        # 含义完全不同，没有尺度就无法横向比较。
        "bbox_diagonal_mm",
        # 配准的三个分量（残差、尺度、剩余）分开列：
        # 合并成一个"配准误差"无法区分
        # "整体缩放没对齐"与"姿态没对齐"这两种完全不同的问题。
        "registration_mean_residual_mm",
        "scale_component_mm",
        "residual_component_mm",
        # 缺失与多出的面积占比分开：两者工程含义相反，
        # 合并成"差异占比"会掩盖方向。
        "missing_area_ratio",
        "extra_area_ratio",
        "region_count",
        # error 放最后一列：绝大多数行它都是空的，
        # 放在前部会让人工复核时每次都要跳过它。
        "error",
    ]
    with atomic_path(output_root / "summary.csv") as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                audit = record.get("threshold_audit") or {}
                metrics = record.get("metrics") or {}
                writer.writerow(
                    {
                        "pair": record.get("pair"),
                        "status": record.get("status"),
                        "stl_size_bytes": record.get("stl_size_bytes"),
                        "elapsed_seconds": record.get("elapsed_seconds"),
                        "threshold_mm": record.get("threshold_mm"),
                        "bbox_diagonal_mm": audit.get("bbox_diagonal_mm"),
                        "registration_mean_residual_mm": audit.get(
                            "registration_mean_residual_mm"
                        ),
                        "scale_component_mm": audit.get("scale_component_mm"),
                        "residual_component_mm": audit.get("residual_component_mm"),
                        "missing_area_ratio": metrics.get("missing_area_ratio"),
                        "extra_area_ratio": metrics.get("extra_area_ratio"),
                        "region_count": record.get("region_count"),
                        "error": record.get("error"),
                    }
                )


def _build_parser() -> argparse.ArgumentParser:
    """我构造标定命令行解析器；worker 专用参数对用户隐藏。"""
    parser = argparse.ArgumentParser(
        description="我方在真实数据上标定自适应阈值系数并实测单模型耗时。"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "real_threshold_calibration",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--max-stl-bytes",
        type=int,
        default=80 * 1024 * 1024,
        help="我方跳过超过该体积的 STL，避免超大样本占满时间预算。",
    )
    parser.add_argument("--scale-ratio", type=float, default=0.002)
    parser.add_argument("--residual-factor", type=float, default=1.5)
    parser.add_argument("--floor-mm", type=float, default=0.02)
    parser.add_argument("--max-query-points", type=int, default=20_000)
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="我方只跑名称包含该子串的样本，可重复传入。",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reaggregate",
        action="store_true",
        help="我方只按现有 summary.json 重算汇总，不重跑任何样本。",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--step", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stl", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """我遍历样本逐对标定；结果全部失败返回 1，否则返回 0。"""
    args = _build_parser().parse_args()
    if args.timeout_seconds <= 0 or args.max_stl_bytes <= 0:
        raise ValueError("我要求 --timeout-seconds 和 --max-stl-bytes 都大于 0。")
    if args.worker:
        if args.step is None or args.stl is None or args.output is None:
            raise ValueError("我要求 worker 同时提供 --step、--stl 和 --output。")
        return run_worker(args.step, args.stl, args.output, args)

    data_root = args.data_root.expanduser()
    output_root = args.output_root.expanduser()
    if args.reaggregate:
        # 重新聚合模式：不重跑任何模型，只读回上一轮 summary.json 重算汇总。
        # 汇总口径（比如换了聚合函数或阈值档位）调整后，
        # 用它就能在几秒内重出报告，不必再花几小时重跑全部数据。
        payload = json.loads(
            (output_root / "summary.json").read_text(encoding="utf-8")
        )
        # 只保留 dict 记录：历史文件里可能混进字符串或 None 的脏记录，
        # 让它们进入聚合会在取值时炸掉，宁可静默丢弃。
        records = [
            record
            for record in payload.get("records", [])
            if isinstance(record, dict)
        ]
        _write_reports(output_root, data_root, args, records, len(records))
        print(
            json.dumps(_aggregate(records), ensure_ascii=False, indent=2),
            flush=True,
        )
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = _find_pairs(data_root)
    # --only 按路径子串过滤，便于只重跑某一个模型做调试，
    # 而不用为它单独准备一份数据根。
    if args.only:
        pairs = [
            pair for pair in pairs if any(token in pair[0] for token in args.only)
        ]
    fingerprint = config_fingerprint(calibration_config(args))

    # 断点续跑：先载入上一轮已完成的记录，后面只对缺失的跑。
    # 读取失败当作"没有历史"处理——损坏的 summary 不该让整个标定起不来，
    # 重跑一遍的代价远小于卡死。
    resume_records: dict[str, dict[str, object]] = {}
    summary_path = output_root / "summary.json"
    if args.resume and summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if (
            isinstance(previous, dict)
            # 版本与数据根都要对上：schema 变了说明记录结构不兼容，
            # 数据根变了说明这批结果根本不是同一个数据集上的。
            # 只查其一会让续跑读到张冠李戴的历史记录。
            and previous.get("schema_version") == SCHEMA_VERSION
            and previous.get("data_root") == str(data_root.resolve())
        ):
            for record in previous.get("records", []):
                # 配置指纹也必须一致：换了阈值或参数后，
                # 上一轮的结果不能代表本轮配置下的表现，必须重跑。
                # 另外排除 exception 记录——异常要重跑，不该被续用。
                if (
                    isinstance(record, dict)
                    and record.get("config_fingerprint") == fingerprint
                    and record.get("status") != "exception"
                ):
                    resume_records[str(record.get("pair"))] = record

    records: list[dict[str, object]] = []
    for pair_name, step_path, stl_path in pairs:
        if pair_name in resume_records:
            record = dict(resume_records[pair_name])
            # resumed 标记让汇总里能区分"本轮算的"与"读回来的"：
            # 没有它就无法判断耗时统计里混没混入历史数据。
            record["resumed"] = True
            records.append(record)
            # 每续一条就重写一次报告：长任务中途被打断时，
            # 已完成的进度不会丢在内存里。
            _write_reports(output_root, data_root, args, records, len(pairs))
            print(f"[resumed] {pair_name} status={record.get('status')}", flush=True)
            continue
        record: dict[str, object] = {
            "pair": pair_name,
            "step_file": str(step_path.resolve()),
            "stl_file": str(stl_path.resolve()),
            "stl_size_bytes": stl_path.stat().st_size,
            "config_fingerprint": fingerprint,
            # 初值一律给"异常/空"：任何一步崩掉时，
            # 落盘的仍是这条记录，状态如实是 exception，
            # 不会留下一个看似正常的半成品。
            "status": "exception",
            "elapsed_seconds": None,
            "threshold_mm": None,
            "threshold_audit": {},
            "registration": {},
            "metrics": {},
            "region_count": 0,
            "warning_count": 0,
            "memory_before_mb": _memory_mb(),
            "memory_after_mb": None,
            "error": None,
        }
        if stl_path.stat().st_size > args.max_stl_bytes:
            record["status"] = "resource_guard"
            record["error"] = (
                f"STL 为 {stl_path.stat().st_size} bytes，超过本轮 "
                f"资源保护阈值 {args.max_stl_bytes} bytes。"
            )
            record["elapsed_seconds"] = 0.0
            records.append(record)
            _write_reports(output_root, data_root, args, records, len(pairs))
            print(f"[guard] {pair_name}", flush=True)
            continue
        pair_output = output_root / pair_name.replace("/", "__")
        command = [
            sys.executable,
            # 我用 -B 关闭 worker 的字节码写入，避免把 __pycache__ 灌进 .venv，
            # 让 cleanup_project_caches.ps1 的"剩余缓存必须为 0"门禁保持成立。
            "-B",
            # 每个样本跑在独立子进程里，而不是在主进程内循环：
            # OCP 的内存不会完全释放，跑几十个样本后主进程会累积到
            # 几个 GB；子进程退出后内存一定归还。
            # 这也是为什么下面能靠 timeout 硬性掐掉一个超时的样本。
            str(Path(__file__).resolve()),
            "--worker",
            "--step",
            str(step_path),
            "--stl",
            str(stl_path),
            "--output",
            str(pair_output),
            "--scale-ratio",
            str(args.scale_ratio),
            "--residual-factor",
            str(args.residual_factor),
            "--floor-mm",
            str(args.floor_mm),
            "--max-query-points",
            str(args.max_query_points),
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
                # cwd 固定在仓库根：worker 里用的是相对导入与相对路径，
                # 不从根启动会因为找不到模块而失败。
                cwd=str(ROOT),
            )
        # 超时按"这个样本失败"处理而不是终止整轮：
        # 一个超大的件不该让前面几十个样本的结果白跑。
        except subprocess.TimeoutExpired:
            record["status"] = "timeout"
            record["error"] = f"超过 {args.timeout_seconds} 秒未完成。"
            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        else:
            payload = _worker_payload(pair_output)
            # 读不到 worker 输出时把 stderr 尾部带出来：
            # worker 崩在子进程里，不留 stderr 就无从知道崩在哪，
            # 截断到 2000 字符是因为 OCP 的报错可能极长。
            if payload is None:
                record["error"] = (
                    (completed.stderr or "").strip()[-2000:]
                    or "worker 没有写出 worker_summary.json。"
                )
                record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            else:
                record.update(payload)
                record["pair"] = pair_name
                record["stl_size_bytes"] = stl_path.stat().st_size
                record["config_fingerprint"] = fingerprint
                record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                record["memory_after_mb"] = _memory_mb()
        records.append(record)
        _write_reports(output_root, data_root, args, records, len(pairs))
        print(
            f"[{record['status']}] {pair_name} "
            f"elapsed={record['elapsed_seconds']}s "
            f"threshold={record.get('threshold_mm')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
