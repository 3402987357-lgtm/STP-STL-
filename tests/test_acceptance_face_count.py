"""我守住验收脚本的面数门禁不会失效。

我踩到的真实缺陷：语料里有**二进制 STL 的文件名以 "solid" 开头**
（例如 测试集/1 的 "solid 4__叶子"）。按"开头 5 字节是不是 solid"
判格式会把它们误判成 ASCII，于是面数数出 0——门禁形同虚设，
最大的那几个件反而被放行。
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    """我按路径加载验收脚本，避免为它单独建包。"""

    path = ROOT / "scripts" / "run_real_pair_acceptance.py"
    spec = importlib.util.spec_from_file_location("acceptance_mod", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_binary_stl(path: Path, name: str, face_count: int) -> None:
    """我写一个名字以 solid 开头的**二进制** STL。"""

    header = (name.encode("utf-8")[:80]).ljust(80, b"\0")
    body = bytearray()
    for _ in range(face_count):
        body.extend(struct.pack("<3f", 0.0, 0.0, 1.0))
        body.extend(struct.pack("<3f", 0.0, 0.0, 0.0))
        body.extend(struct.pack("<3f", 1.0, 0.0, 0.0))
        body.extend(struct.pack("<3f", 0.0, 1.0, 0.0))
        body.extend(struct.pack("<H", 0))
    path.write_bytes(header + struct.pack("<I", face_count) + bytes(body))


def _write_ascii_stl(path: Path, face_count: int) -> None:
    """我写一个普通的 ASCII STL。"""

    lines = ["solid demo"]
    for _ in range(face_count):
        lines += [
            " facet normal 0 0 1",
            "  outer loop",
            "   vertex 0 0 0",
            "   vertex 1 0 0",
            "   vertex 0 1 0",
            "  endloop",
            " endfacet",
        ]
    lines += ["endsolid demo"]
    path.write_text("\n".join(lines), encoding="utf-8")


def test我方数出名字以solid开头的二进制STL的面数(tmp_path) -> None:
    """二进制 STL 即使名字以 solid 开头，也必须读到真实面数。"""

    module = _load_module()
    target = tmp_path / "binary.stl"
    _write_binary_stl(target, "solid 4__叶子", 37)
    assert module._count_stl_faces(target) == 37


def test我方数出ASCII_STL的面数(tmp_path) -> None:
    """ASCII STL 按 facet normal 计数。"""

    module = _load_module()
    target = tmp_path / "ascii.stl"
    _write_ascii_stl(target, 5)
    assert module._count_stl_faces(target) == 5


def test我方对语料里所有STL都能数出正数() -> None:
    """我确认真实语料上没有一件被数成 0 面。"""

    module = _load_module()
    corpus = ROOT / "data" / "provided" / "三维模型破损区域智能修补"
    if not corpus.is_dir():
        pytest.skip("甲方语料不存在。")
    files = sorted(corpus.rglob("*.stl"))
    assert files, "语料里没有任何 STL，用例前提不成立。"
    zero: list[str] = []
    for path in files:
        count = module._count_stl_faces(path)
        if count is None or count <= 0:
            zero.append(f"{path.name}->{count}")
    assert not zero, f"以下 STL 面数没数出来，门禁会失效：{zero}"


def test我方面数门禁确实会拦住超大件() -> None:
    """178 万面那个件必须被默认上限拦下，否则整轮验收会被拖死。"""

    module = _load_module()
    corpus = ROOT / "data" / "provided" / "三维模型破损区域智能修补"
    big = corpus / "训练集" / "1.stl"
    if not big.exists():
        pytest.skip("最大件不存在。")
    count = module._count_stl_faces(big)
    assert count is not None and count > 600_000, (
        f"该件应当超过默认上限，实际面数 {count}"
    )


def test我方续跑只跳过已出结论的组(tmp_path) -> None:
    """我验证中断后重跑不会重复跑已完成的组，但会重跑被杀的组。

    这条守的是实测踩到的坑：整轮 18 组要跑一两小时，父进程还会被
    系统杀掉。若续跑逻辑把 KILLED 也当成"已完成"跳过，
    那么最慢、最容易失败的几组会**永久留空**——
    表面上"续跑成功"，实际那几组再也不会有人跑。
    """

    jsonl = tmp_path / "records.jsonl"
    done = {"pair": "训练集/0", "status": "review", "region_count": 27}
    killed = {"pair": "测试集/8", "status": "KILLED", "region_count": 0}
    timeout = {"pair": "训练集/3", "status": "TIMEOUT", "region_count": 0}
    jsonl.write_text(
        "\n".join(json.dumps(r) for r in (done, killed, timeout)) + "\n",
        encoding="utf-8",
    )

    # 我直接复现 main 里的续跑判定逻辑：
    # 只有非 KILLED/TIMEOUT/ERROR 的组才算"已有结论"。
    already_done: set[str] = set()
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        old = json.loads(line)
        if old.get("status") not in {"KILLED", "TIMEOUT", "ERROR"}:
            already_done.add(str(old.get("pair")))

    assert already_done == {"训练集/0"}
    # 被杀和超时的两组必须留待重跑。
    assert "测试集/8" not in already_done
    assert "训练集/3" not in already_done


def test我方台账把低于推荐阈值的件数列成汇总指标() -> None:
    """"配置阈值低于推荐阈值"是这批数据上最实质的准确性风险——
    阈值落在配准残差之下就分不清"真的缺料"和"只是没对齐"。
    它必须能在汇总层一眼看到，不能散落在一堆 result.json 里靠人数，
    数漏了就等于没发现。

    这里钉住真实台账：审计件数、低于推荐值的件数、以及每条记录
    都带上了阈值审计的三个关键量。
    """

    path = ROOT / "outputs" / "acceptance_v10" / "summary.json"
    if not path.exists():
        pytest.skip("我还没有验收台账")

    summary = json.loads(path.read_text(encoding="utf-8"))
    audit = summary.get("threshold_audit")
    assert audit, "台账里没有阈值审计汇总"

    audited = [
        r
        for r in summary["records"]
        if r.get("recommended_threshold_mm") is not None
    ]
    assert audit["audited_pair_count"] == len(audited)
    # 低于推荐值的件数必须与逐条记录数出来的一致，不能是写死的数。
    below = [
        r["pair"]
        for r in audited
        if r.get("configured_below_recommended") is True
    ]
    assert audit["configured_below_recommended_count"] == len(below)
    assert sorted(audit["configured_below_recommended_pairs"]) == sorted(below)
    # 推荐值必须真的是 mm 量级的正数，否则说明抓错了字段。
    for r in audited:
        assert r["recommended_threshold_mm"] > 0
        assert r["effective_threshold_mm"] > 0


def test我方台账把采样覆盖率列进每条记录() -> None:
    """我钉住采样覆盖率能出现在汇总层，而不是只埋在 result.json 的 warnings 里。

    我为什么钉这个：20 个真实件里有 5 个参考面没采全，最低只有 30.7%
    （测试集/3、测试集/4 都在 30% 出头）。这些行的 missing_area_ratio
    只代表"被采到的那部分表面"，不能当整件面积的准数用。
    以前汇总层看不到覆盖率，读者不逐个打开 result.json 就发现不了。
    """

    # 主批次 18 件 + 补跑批次 2 件（测试集/3、测试集/4），两批合起来才是 20 件。
    paths = [
        ROOT / "outputs" / "acceptance_auto" / "summary.json",
        ROOT / "outputs" / "acceptance_auto_extra" / "summary.json",
    ]
    if not all(p.exists() for p in paths):
        pytest.skip("我还没有完整的 auto 配准验收台账")
    rows = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(
            r
            for r in summary["records"]
            if isinstance(r, dict) and r.get("error") is None
        )
    assert rows, "台账里没有任何有效记录"
    missing = [r["pair"] for r in rows if "reference_sampling_coverage_ratio" not in r]
    assert not missing, f"这些记录缺采样覆盖率，汇总层会看不出来：{missing}"
    ratios = [
        r["reference_sampling_coverage_ratio"]
        for r in rows
        if r.get("reference_sampling_coverage_ratio") is not None
    ]
    assert ratios, "所有记录的覆盖率都是空的，说明字段没填进去"
    assert all(0.0 < x <= 1.0001 for x in ratios)
    # 这条钉住"覆盖率确实有低到必须提示"的事实；若将来全采满，这条该改成
    # 断言"没有低覆盖件"，而不是悄悄删掉。
    assert min(ratios) < 0.5, (
        "台账里已没有低覆盖件；若这是提高采样上限后的结果，"
        "应同步更新这条断言与 README 里的覆盖率说明。"
    )


def test我方全采样复跑没有翻转任何结论() -> None:
    """我钉住"提高采样覆盖只改面积精度、不改结论"这条实测。

    上一轮我把"低覆盖件提满采样会不会翻转结论"留成了未验证的风险。
    这一轮把 4 个低覆盖件用 70 万点重跑：区域数、状态全部不变，
    missing 比例最大只差 0.024 个百分点（测试集/1）。
    所以默认 20 万点是"面积略保守"而不是"会漏判"。
    若将来有人改了采样或判定逻辑导致翻转，这条会红，提醒重新评估默认值。
    """

    path = ROOT / "reports" / "full_coverage_check.json"
    if not path.exists():
        pytest.skip("我还没有全采样复跑的证据")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["parts"] == 4
    assert data["conclusion_flips"] == [], (
        f"提高采样覆盖后结论发生变化：{data['conclusion_flips']}；"
        "需要重新评估默认 max_query_points。"
    )
    assert data["max_missing_delta"] < 0.005, (
        "missing 比例变化超过 0.5 个百分点，说明抽样误差比预期大得多。"
    )
    for row in data["rows"]:
        assert row["coverage_after"] >= 0.999, (
            f"{row['pair']} 复跑后覆盖率仍不到 100%，这轮证据不成立。"
        )


def test我方零结果批次不留下空records文件(tmp_path: Path, monkeypatch) -> None:
    """我钉住"全部被门禁跳过"的批次不会留下 0 字节的 records.jsonl。

    我为什么专门钉这个：main 原先在 fresh 分支里无条件
    ``jsonl_path.write_text("")``，于是只要这一轮**所有对都被面数门禁
    跳过**，records.jsonl 就会始终是 0 字节。而输出审计把零字节文件
    判为 INVALID——这是对的，产出物静默写空本就该拦——结果就是
    "一个什么都没跑成的批次"把整道门禁搞红。

    我修的是**生产端**：零结果时不创建这个文件（留到第一条结论落盘
    时由 _flush 创建），而不是去放宽审计。审计的严格性必须保住。
    """

    module = _load_module()
    corpus = tmp_path / "corpus"
    subset = corpus / "训练集"
    subset.mkdir(parents=True)
    (subset / "1.stp").write_text("ISO-10303-21;", encoding="ascii")
    _write_binary_stl(subset / "1.stl", "solid demo", 5)
    out = tmp_path / "out"

    # --max-reference-faces 0：任何正面数的参考 STL 都会被门禁跳过，
    # 于是这一轮必然零结果。
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_pair_acceptance.py",
            "--corpus",
            str(corpus),
            "--output-root",
            str(out),
            "--max-reference-faces",
            "0",
            "--fresh",
        ],
    )
    assert module.main() == 0, "全跳过不是失败，应当是正常收口"

    records = out / "records.jsonl"
    assert not records.exists(), (
        "零结果批次不该留下 records.jsonl；"
        "留一个 0 字节文件会被输出审计判成 INVALID，"
        f"现有大小={records.stat().st_size if records.exists() else '不存在'}"
    )
    # summary.json 仍要写：批次没结果，但"这一轮跑了什么、跳过了谁"必须留证。
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["skipped"]) == 1
    assert summary["pair_count"] == 0
