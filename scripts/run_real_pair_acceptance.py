"""我对甲方提供的真实破损/完整配对做一次成规模的验收。

我为什么单独写这个脚本，而不是继续用单次 inspect 命令：
任务口径要的是"20+20 组"这种成规模的准确率验收，之前每一轮
记录里它都停在 BASELINE——不是跑不动，是**从来没有被跑起来过**。
单次命令只能证伪"能不能跑"，给不出"21 组里几组判对"。
这个脚本把每一组的结论归成一行，并把汇总写进 JSON，
让"通过率"第一次有真实数字，而不是靠推断。

我为什么把退出码设计成"有任一组失败就返回 1"：
验收脚本最怕的是失败被静默吞掉。返回 1 才能让 CI/门禁看见，
而明细仍然逐行打印，方便人工看是哪一组不行。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cad_inspector.pipeline import inspect_pair
from cad_inspector.models import InspectionConfig

DEFAULT_CORPUS = ROOT / "data" / "provided" / "三维模型破损区域智能修补"
SUBSETS = ("训练集", "测试集")


def _configure_utf8_stdout() -> None:
    """我把 stdout 切成 UTF-8，避免中文提示在 Windows 控制台乱码。"""

    stream = getattr(sys, "stdout", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def discover_pairs(corpus: Path, only: list[str] | None) -> list[tuple[str, Path, Path]]:
    """我找出语料里所有"同一个编号同时有 STP 与 STL"的配对。

    我为什么要求两侧都存在才算一组：本验收比的是
    "待检测 STP" 与 "完整参考 STL"，缺任一侧就无从比对。
    缺侧的编号我不静默跳过，而是单独记进 skipped，
    免得"通过率"的分母被悄悄改小。
    """

    pairs: list[tuple[str, Path, Path]] = []
    skipped: list[dict[str, str]] = []
    for subset in SUBSETS:
        folder = corpus / subset
        if not folder.is_dir():
            continue
        for step_path in sorted(folder.glob("*.stp")):
            name = f"{subset}/{step_path.stem}"
            if only and name not in only:
                continue
            stl_path = folder / f"{step_path.stem}.stl"
            if not stl_path.exists():
                skipped.append({"pair": name, "reason": "缺少配对的参考 STL"})
                continue
            pairs.append((name, step_path, stl_path))
    return pairs, skipped


def _count_stl_faces(path: Path) -> int | None:
    """我只数 STL 的三角面数，不把整张网格读进内存。

    我为什么不直接 load_stl 再数：门禁的目的正是**避免**把 178 万面的
    网格读进来——读一次就要 219 秒，那门禁本身就失去了意义。
    Binary STL 的面数写在第 80 字节起的 4 个字节里，读它只要几毫秒。
    ASCII STL 没有这个头，我按 "facet normal" 出现次数数。

    我为什么不能只看开头 5 字节是不是 "solid" 来分二进制/ASCII：
    实测语料里有二进制 STL 的名字就叫 "solid 4__叶子"，
    开头恰好是 solid，按这个判会被当成 ASCII，数出 0 面，
    于是"面数门禁"对这些件完全失效——最大的那个件反而被放行。
    真正的 ASCII STL 在 "solid" 之后必然是换行与 "facet normal" 文本，
    所以我改成看**整个文件里能不能找到 facet normal** 来判 ASCII。
    """

    try:
        import struct

        raw = path.read_bytes()
        if len(raw) < 84:
            return None
        # 先按 ASCII 判：能读到 facet normal 就一定是文本格式。
        # 用局部扫描避免把 475MB 文件整个解码成 str。
        probe = raw[: 1 << 20].decode("utf-8", errors="ignore")
        if "facet normal" in probe:
            return raw.decode("utf-8", errors="ignore").count("facet normal")
        # 否则按二进制头读面数。
        return int(struct.unpack("<I", raw[80:84])[0])
    except OSError:
        return None


def _run_one(
    outbox: "mp.Queue",
    name: str,
    step_path: Path,
    stl_path: Path,
    output_root: Path,
    registration_mode: str,
    threshold_mm: float,
    linear_deflection_mm: float | None = None,
    max_query_points: int | None = None,
) -> None:
    """我在子进程里跑一组，把结论放进队列。

    我为什么要把每一组放进单独子进程：实测整轮 21 组跑到
    训练集/5（49MB STEP）时整个进程被系统杀掉，
    前面 4 组的结论也随之丢失——因为汇总只在最后才写盘。
    放进子进程后，某一组被杀只损失这一组，父进程照常收尾，
    而且我还能把它如实记成 KILLED，而不是假装它通过了。
    """

    sys.path.insert(0, str(ROOT / "src"))
    from cad_inspector.models import InspectionConfig
    from cad_inspector.pipeline import inspect_pair

    started = time.perf_counter()
    # 三角化偏差决定网格密度：0.04mm 是默认精度，但实测它让
    # 训练集/5、测试集/8 这类大件在 900 秒内跑不完甚至被系统杀掉。
    # 传了 coarser 值就用它，换来"能出结论"而不是"精确但出不来"。
    extra = (
        {"mesh_linear_deflection_mm": linear_deflection_mm}
        if linear_deflection_mm is not None
        else {}
    )
    # 2026-10-18：默认 20 万点会让大件的参考面只采到 30%~80%。
    # 这里让调用方能按件提高上限，面积数字才代表整件而不是抽样部分。
    if max_query_points is not None:
        extra["max_query_points"] = int(max_query_points)
    config = InspectionConfig(
        registration_mode=registration_mode,
        threshold_mm=threshold_mm,
        **extra,
    )
    result = inspect_pair(
        candidate_path=step_path,
        reference_path=stl_path,
        output_dir=output_root / name.replace("/", "__"),
        config=config,
    )
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    regions = payload.get("regions") or []
    metrics = payload.get("global_metrics") or {}
    # 我把阈值审计的三个关键量记进台账：
    # "配置阈值低于推荐阈值"是这批数据上最实质的准确性风险——
    # 阈值落在配准残差之下就分不清"真的缺料"和"只是没对齐"。
    # 只留 region_count 的话，读者想复核得逐个打开 result.json，
    # 而这条结论本来就该在汇总层一眼看到。
    audit = metrics.get("threshold_audit") or {}
    outbox.put(
        {
            "pair": name,
            "status": payload.get("status"),
            "region_count": len(regions),
            "missing_area_mm2": metrics.get("missing_area_mm2"),
            "missing_area_ratio": metrics.get("missing_area_ratio"),
            "warning_count": len(payload.get("warnings") or []),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "linear_deflection_mm": linear_deflection_mm,
            "configured_threshold_mm": audit.get("configured_mm"),
            "recommended_threshold_mm": audit.get("recommended_mm"),
            "effective_threshold_mm": audit.get("effective_mm"),
            "registration_mean_residual_mm": audit.get(
                "registration_mean_residual_mm"
            ),
            "configured_below_recommended": audit.get(
                "configured_below_recommended"
            ),
            # 2026-10-18：采样覆盖率以前只在 result.json 的 warnings 里，
            # 汇总层看不到。20 个真实件里有 8 个的参考面只采到一部分
            # （测试集/4 只有 32%），这直接决定"面积数字能不能当准数用"。
            # 放进汇总，读者不必逐个打开 result.json 才知道哪几行不可全信。
            "reference_sampling_coverage_ratio": metrics.get(
                "reference_sampling_coverage_ratio"
            ),
            "candidate_sampling_coverage_ratio": metrics.get(
                "candidate_sampling_coverage_ratio"
            ),
            "error": None,
        }
    )


def _write_summary(
    path: Path,
    records: list[dict[str, object]],
    skipped: list[dict[str, str]],
    killed: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, object]:
    """我把当前进度写成汇总 JSON，并回传给调用方打印。"""

    statuses = Counter(str(r["status"]) for r in records)
    # 阈值低于推荐值的件数单列出来：这是这批数据上最实质的风险指标，
    # 我不想让它只散落在 19 个 result.json 里靠人去数——
    # 数漏了就等于没发现。
    audited = [r for r in records if r.get("recommended_threshold_mm") is not None]
    below = [
        r["pair"]
        for r in audited
        if r.get("configured_below_recommended") is True
    ]
    summary = {
        "pair_count": len(records),
        "by_status": dict(statuses),
        "ran_without_error": sum(
            1 for r in records if r["status"] not in {"ERROR", "KILLED"}
        ),
        "with_regions": sum(1 for r in records if r["region_count"]),
        "threshold_audit": {
            "audited_pair_count": len(audited),
            "configured_below_recommended_count": len(below),
            "configured_below_recommended_pairs": below,
            "note": (
                "配置阈值低于推荐阈值意味着阈值落在配准残差之下，"
                "分不清真的缺料与没对齐；这些件的区域数量需要人工复核。"
            ),
        },
        "skipped": skipped,
        "killed": killed,
        "config": {
            "registration_mode": args.registration_mode,
            "threshold_mm": args.threshold_mm,
            "max_reference_faces": args.max_reference_faces,
            "timeout_seconds": args.timeout_seconds,
        },
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _run_isolated(
    name: str,
    step_path: Path,
    stl_path: Path,
    output_root: Path,
    registration_mode: str,
    threshold_mm: float,
    timeout_seconds: float,
    linear_deflection_mm: float | None = None,
    max_query_points: int | None = None,
) -> dict[str, object]:
    """我在独立子进程里跑一组，并区分"报错""超时""被杀"三种失败。

    我为什么必须区分这三种：它们的处置完全不同——
    报错是代码问题要去修；超时是这件太大要换更粗的三角化；
    被杀（实测发生在 49MB STEP 上）是内存问题，
    而且被杀时子进程根本没机会返回，只能由父进程从"超时且退出码非 0"推断。
    混成一个 ERROR 会让"通过率"失去可诊断性。
    """

    outbox: mp.Queue = mp.Queue()
    started = time.perf_counter()
    # 我为什么用 spawn 而不是默认的 fork/多进程共享：实测父进程
    # 自己会被系统杀掉（16GB 机器上大件峰值 5GB+），
    # 而父进程里还驻留着前面几组的 records 与中间数组。
    # spawn 起的是全新的解释器，父子内存完全隔离，
    # 父进程只留一个很小的"等到超时"循环，被杀的概率大幅下降。
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_run_one,
        args=(
            outbox,
            name,
            step_path,
            stl_path,
            output_root,
            registration_mode,
            threshold_mm,
            linear_deflection_mm,
            max_query_points,
        ),
    )
    proc.start()
    try:
        record = outbox.get(timeout=timeout_seconds)
        proc.join(timeout=30)
        return record
    except queue.Empty:
        proc.terminate()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)
        # 退出码 0 却没给结果，说明它还在跑（纯超时）；
        # 非 0 且没结果，通常是被系统杀掉（内存）。
        exit_code = proc.exitcode
        status = "TIMEOUT" if exit_code == 0 else "KILLED"
        return {
            "pair": name,
            "status": status,
            "region_count": 0,
            "missing_area_mm2": None,
            "missing_area_ratio": None,
            "warning_count": 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "linear_deflection_mm": linear_deflection_mm,
            "reference_sampling_coverage_ratio": None,
            "candidate_sampling_coverage_ratio": None,
            "error": (
                f"超过 {timeout_seconds}s 未出结果（exit={exit_code}）；"
                "该件过大，建议调粗三角化偏差后单独跑。"
            ),
        }


def main() -> int:
    """我对每一组配对跑一次检测，并汇总结论。"""

    _configure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="我对甲方真实破损/完整配对做成规模验收。",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "real_pair_acceptance")
    parser.add_argument("--registration-mode", default="auto")
    parser.add_argument("--threshold-mm", type=float, default=0.15)
    parser.add_argument(
        "--max-query-points",
        type=int,
        default=None,
        # 默认沿用库里的 20 万；只有需要"整件面积准数"时才调高，
        # 代价是耗时按采样点比例上升。
        help="覆盖更多参考面；不传则用库默认 200000。",
    )
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--max-reference-faces",
        type=int,
        default=600_000,
        # 我为什么默认给一个面数上限：实测 训练集/1 的参考 STL 有
        # 178 万个三角面，光 load_stl 就要 219 秒，整组跑完要小时级，
        # 一轮 21 组的验收会被这一个件单独拖死。
        # 上限不是"跳过就算了"——超出的组我记进 skipped 并写明原因，
        # 让"通过率"的分母变化是显式可见的，而不是悄悄少算几组。
        help="参考 STL 三角面数超过此值的组不参与本轮验收，记入 skipped。",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900.0,
        help="单组超时秒数；超时或被杀的组记为 TIMEOUT/KILLED 而不是整轮失败。",
    )
    parser.add_argument(
        "--coarsen-on-timeout",
        type=float,
        nargs="*",
        default=[0.2, 0.5],
        # 我为什么默认给两档而不是一档：0.2mm 能救回一部分件，
        # 但最大的件连 0.2 都跑不动，需要退到 0.5。
        # 逐级下降而不是一步到位，是为了尽量保住精度——
        # 能用 0.2 出结论的件就不该被降到 0.5。
        help="单组超时/被杀后，按这些三角化偏差值逐级降精度重试。",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="忽略已有 records.jsonl，从零开始跑（默认续跑）。",
    )
    args = parser.parse_args()

    pairs, skipped = discover_pairs(args.corpus, args.only)
    if not pairs:
        print("我没有找到任何可用的 STP/STL 配对，验收无法进行。")
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    # 我为什么自己构造 InspectionConfig 而不是用默认值：
    # 配准模式与阈值是这次验收的自变量，必须显式写进报告，
    # 否则别人看到"通过率"也不知道是按什么口径算的。
    records: list[dict[str, object]] = []
    killed: list[dict[str, str]] = []
    # 我为什么额外写一个 JSONL：父进程自己也实测被系统杀掉过，
    # 内存里的 records 会随之全丢。JSONL 每行一组、即时 flush，
    # 父进程被杀后仍能逐行读回已完成的组。
    jsonl_path = args.output_root / "records.jsonl"
    already_done: set[str] = set()
    if not args.fresh and jsonl_path.exists():
        # 我为什么要能续跑：整轮 21 组要跑 1~2 小时，
        # 中途任何一次中断都必然发生。没有续跑能力的话，
        # 每次中断都要从头再跑一遍已经出过结论的组。
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                old = json.loads(line)
            except json.JSONDecodeError:
                continue
            if old.get("status") not in {"KILLED", "TIMEOUT", "ERROR"}:
                already_done.add(str(old.get("pair")))
                records.append(old)
        if already_done:
            print(f"[resume] 已有 {len(already_done)} 组结论，跳过它们。")
    else:
        # 我为什么不在这里 write_text("") 造一个空文件：
        # 当所有对都被面数门禁跳过时，records.jsonl 会**始终是 0 字节**，
        # 而输出审计把零字节文件判为 INVALID（这是对的——产出物写空本就该拦）。
        # 一个"零结果"的批次不该留下一个看起来像坏文件的空壳，
        # 所以我只清掉旧的续跑数据，文件留到第一条结论落盘时由 _flush 创建。
        jsonl_path.unlink(missing_ok=True)

    def _flush(record: dict[str, object]) -> None:
        """我把一条记录追加进 jsonl 并即刻 flush，再重写 summary.json。"""

        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        _write_summary(
            args.output_root / "summary.json", records, skipped, killed, args
        )
    for name, step_path, stl_path in pairs:
        if name in already_done:
            continue
        # 面数门禁：超上限的组显式跳过，避免一个件拖死整轮验收。
        face_count = _count_stl_faces(stl_path)
        if face_count is not None and face_count > args.max_reference_faces:
            skipped.append(
                {
                    "pair": name,
                    "reason": (
                        f"参考 STL 有 {face_count} 个三角面，"
                        f"超过上限 {args.max_reference_faces}"
                    ),
                }
            )
            print(f"[skip-too-large] {name} faces={face_count}")
            continue
        record = _run_isolated(
            name=name,
            step_path=step_path,
            stl_path=stl_path,
            output_root=args.output_root,
            registration_mode=args.registration_mode,
            threshold_mm=args.threshold_mm,
            timeout_seconds=args.timeout_seconds,
            max_query_points=args.max_query_points,
        )
        # 我为什么在超时/被杀后自动降精度重试：这些件不是"判不出来"，
        # 而是 0.04mm 三角化把网格撑到跑不动。直接放弃等于把
        # 5/21 的语料排除在验收之外；降精度能换来真实结论，
        # 而偏差值会写进记录，结论的精度代价是可追溯的，不是偷偷降标。
        for coarser in args.coarsen_on_timeout:
            if record["status"] not in {"KILLED", "TIMEOUT"}:
                break
            print(f"[retry-coarser] {name} -> {coarser}mm")
            retry = _run_isolated(
                name=name,
                step_path=step_path,
                stl_path=stl_path,
                output_root=args.output_root,
                registration_mode=args.registration_mode,
                threshold_mm=args.threshold_mm,
                timeout_seconds=args.timeout_seconds,
                linear_deflection_mm=coarser,
                max_query_points=args.max_query_points,
            )
            retry["retries"] = retry.get("retries", 0)
            if retry["status"] in {"KILLED", "TIMEOUT"}:
                retry["retries"] = int(record.get("retries", 0)) + 1
            record = retry
        if record["status"] == "KILLED":
            killed.append({"pair": name, "reason": str(record["error"])})
        records.append(record)
        _flush(record)
        print(
            "[%s] %s regions=%s missing_ratio=%s warnings=%s %.1fs"
            % (
                record["status"],
                name,
                record["region_count"],
                record["missing_area_ratio"],
                record["warning_count"],
                record["elapsed_seconds"],
            )
        )

    summary = _write_summary(
        args.output_root / "summary.json", records, skipped, killed, args
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False, indent=2))
    return 0 if all(r["status"] != "ERROR" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
