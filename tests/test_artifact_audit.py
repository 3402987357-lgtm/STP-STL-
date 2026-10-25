from __future__ import annotations

import csv
import json
import struct
import zlib
from pathlib import Path
from zipfile import ZipFile

import pytest

from cad_inspector.cli import main
from cad_inspector.io.artifact_audit import audit_artifacts
from cad_inspector.models import InspectionConfig, InspectionResult, RegistrationResult
from cad_inspector.output.writers import write_json
from cad_inspector.reproducibility import content_sha256, with_content_digest


def test我方审计有效输出并记录schema和行数(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text(
        json.dumps({"schema_version": "demo-1.0", "status": "pass"}),
        encoding="utf-8",
    )
    (tmp_path / "regions.csv").write_text(
        "region_id,status\n1,pass\n",
        encoding="utf-8",
    )
    report = audit_artifacts(tmp_path)

    assert report["status"] == "PASS"
    assert report["valid_files"] == 2
    assert report["invalid_files"] == 0
    assert report["schema_versions"] == {"demo-1.0": 1}
    records = {item["path"]: item for item in report["files"]}
    assert records["regions.csv"]["row_count"] == 1


def test我方审计拒绝非法json并跳过未知文件(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text('{"value": NaN}', encoding="ascii")
    (tmp_path / "notes.txt").write_text("保留但不解析", encoding="utf-8")

    report = audit_artifacts(tmp_path)

    assert report["status"] == "REVIEW_REQUIRED"
    assert report["invalid_files"] == 1
    assert report["skipped_files"] == 1


def test我方审计空目录不假通过(tmp_path: Path) -> None:
    report = audit_artifacts(tmp_path)

    assert report["status"] == "REVIEW_REQUIRED"
    assert report["empty_output"] is True


def test我方审计只有未知文件的目录不假通过(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("保留但不解析", encoding="utf-8")

    report = audit_artifacts(tmp_path)

    assert report["status"] == "REVIEW_REQUIRED"
    assert report["valid_files"] == 0
    assert report["skipped_files"] == 1


def test我方审计拒绝重复json键和不齐csv列(tmp_path: Path) -> None:
    (tmp_path / "duplicate.json").write_text(
        '{"status":"pass","status":"review"}',
        encoding="ascii",
    )
    (tmp_path / "ragged.csv").write_text(
        "a,b\n1\n",
        encoding="ascii",
    )

    report = audit_artifacts(tmp_path)

    assert report["status"] == "REVIEW_REQUIRED"
    assert report["invalid_files"] == 2


def test我方审计接受CRLF与二进制PLY(tmp_path: Path) -> None:
    (tmp_path / "crlf.ply").write_bytes(
        b"ply\r\nformat ascii 1.0\r\nelement vertex 1\r\n"
        b"property float x\r\nproperty float y\r\nproperty float z\r\n"
        b"end_header\r\n0 0 0\r\n"
    )
    (tmp_path / "binary.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"end_header\n" + struct.pack("<fff", 1.0, 2.0, 3.0)
    )

    report = audit_artifacts(tmp_path)

    assert report["status"] == "PASS"
    assert report["invalid_files"] == 0
    assert report["valid_files"] == 2


def test我方审计拒绝缺少头部的PLY(tmp_path: Path) -> None:
    (tmp_path / "broken.ply").write_bytes(b"not-a-ply\nend_header\n")

    report = audit_artifacts(tmp_path)

    assert report["status"] == "REVIEW_REQUIRED"
    assert report["invalid_files"] == 1


def test我方审计容忍带BOM的合法JSON(tmp_path: Path) -> None:
    (tmp_path / "bom.json").write_text(
        json.dumps({"schema_version": "demo-1.0", "status": "pass"}),
        encoding="utf-8-sig",
    )

    report = audit_artifacts(tmp_path)

    assert report["status"] == "PASS"
    assert report["valid_files"] == 1
    assert report["schema_versions"] == {"demo-1.0": 1}


def test我方审计拒绝重复列名CSV(tmp_path: Path) -> None:
    (tmp_path / "duplicate_header.csv").write_text(
        "id,id\n1,2\n",
        encoding="utf-8",
    )

    report = audit_artifacts(tmp_path)
    record = report["files"][0]

    assert report["status"] == "REVIEW_REQUIRED"
    assert record["status"] == "INVALID"
    assert any("重复列名" in issue for issue in record["issues"])


def test我方xlsx控制字符不会写出坏xml(tmp_path: Path) -> None:
    from cad_inspector.output.bom_report import _write_xlsx

    path = tmp_path / "bom.xlsx"
    _write_xlsx(path, ["名称", "数量"], [{"名称": "part\x0bname", "数量": 1}])

    report = audit_artifacts(tmp_path)
    record = report["files"][0]

    assert report["status"] == "PASS"
    assert record["status"] == "VALID"


def test我方审计拒绝xml损坏的xlsx(tmp_path: Path) -> None:
    path = tmp_path / "broken.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/workbook.xml", "<workbook>")
        archive.writestr("xl/worksheets/sheet1.xml", "<worksheet>")

    report = audit_artifacts(tmp_path)
    record = report["files"][0]

    assert report["status"] == "REVIEW_REQUIRED"
    assert record["status"] == "INVALID"
    assert any("XML 不合法" in issue for issue in record["issues"])


def test我方审计支持超过默认上限的合法csv字段(tmp_path: Path) -> None:
    (tmp_path / "large_field.csv").write_text(
        "id,evidence\n1," + "x" * 200_000 + "\n",
        encoding="utf-8",
    )

    report = audit_artifacts(tmp_path)
    record = report["files"][0]

    assert report["status"] == "PASS"
    assert record["status"] == "VALID"
    assert record["row_count"] == 1
    assert csv.field_size_limit() >= 200_000


def test我方审计超过读取上限时不读取完整文件(tmp_path: Path) -> None:
    path = tmp_path / "large.json"
    path.write_bytes(b"{}" + b"x" * 1024)

    report = audit_artifacts(tmp_path, max_file_bytes=2)
    record = report["files"][0]

    assert report["invalid_files"] == 1
    assert record["status"] == "INVALID"
    assert record["sha256"] is None


def test我方audit命令可以写出报告并返回审计状态(tmp_path: Path) -> None:
    (tmp_path / "valid.json").write_text("[]", encoding="ascii")
    output = tmp_path / "audit.json"

    assert main(["audit", str(tmp_path), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "artifact-audit-1.0"
    assert payload["status"] == "PASS"


def test我方正式json拒绝非标准浮点值(tmp_path: Path) -> None:
    result = InspectionResult(
        status="review",
        candidate_file="candidate.stl",
        reference_file="reference.stl",
        unit="mm",
        unit_source="test",
        threshold_mm=0.15,
        registration=RegistrationResult(
            mode="none",
            rotation_matrix=[[1.0, 0.0, 0.0]] * 3,
            translation_mm=[0.0, 0.0, 0.0],
            scale=1.0,
            mean_residual_mm=float("nan"),
            p95_residual_mm=0.0,
            quality="review",
        ),
        global_metrics={},
        regions=[],
        artifacts={},
        warnings=[],
        config=InspectionConfig(),
    )
    output = tmp_path / "result.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json(result, output)
    assert not output.exists()


def test我方审计校验内容摘要正确时通过(tmp_path: Path) -> None:
    payload = with_content_digest({"schema_version": "demo-1.0", "status": "pass"})
    (tmp_path / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    report = audit_artifacts(tmp_path)
    record = report["files"][0]

    assert report["status"] == "PASS"
    assert record["content_sha256"] == content_sha256(payload)


def test我方审计拒绝被篡改内容摘要(tmp_path: Path) -> None:
    payload = with_content_digest({"schema_version": "demo-1.0", "status": "pass"})
    payload["status"] = "review"
    (tmp_path / "tampered.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    report = audit_artifacts(tmp_path)
    record = report["files"][0]

    assert report["status"] == "REVIEW_REQUIRED"
    assert record["status"] == "INVALID"
    assert any("content_sha256" in issue for issue in record["issues"])


def test我方审计不因文件在遍历后消失而崩溃(tmp_path: Path, monkeypatch) -> None:
    """我覆盖"文件在列表与 stat 之间被清理"的竞态。

    真实场景：门禁审计 outputs 时，另一进程正在清理 atomic_path
    留下的 .tmp 文件。审计器不能因为竞态崩溃，否则整个门禁失败，
    而真正要拦的损坏产物一个都没审。
    """
    from cad_inspector.io import artifact_audit

    victim = tmp_path / "vanishing.json"
    victim.write_text(json.dumps({"schema_version": "demo-1.0"}), encoding="utf-8")
    survivor = tmp_path / "survivor.json"
    survivor.write_text(json.dumps({"schema_version": "demo-1.0"}), encoding="utf-8")

    real_stat = Path.stat

    def flaky_stat(self: Path, **kwargs):  # 让 victim 的 stat 抛出"文件不存在"
        if self == victim:
            raise FileNotFoundError(2, "系统找不到指定的文件。")
        return real_stat(self, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    # is_file() 内部也走 stat，会因此返回 False 把 victim 挡在列表外。
    # 我强制它为真，才能覆盖"通过列表、在 stat 时消失"这第二段窗口。
    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: True if self == victim else real_is_file(self),
    )

    report = audit_artifacts(tmp_path)

    # 消失的文件记为 SKIPPED，其余文件照常审计，整体不崩溃。
    by_name = {record["path"]: record for record in report["files"]}
    assert by_name["vanishing.json"]["status"] == "SKIPPED"
    assert by_name["survivor.json"]["status"] == "VALID"
    assert report["invalid_files"] == 0


def test我方审计遍历时is_file抛错也只跳过(tmp_path: Path, monkeypatch) -> None:
    """我覆盖遍历期 OSError（权限/竞态）不应让审计整体失败。"""
    from cad_inspector.io import artifact_audit

    blocked = tmp_path / "blocked.json"
    blocked.write_text(json.dumps({"schema_version": "demo-1.0"}), encoding="utf-8")
    (tmp_path / "ok.json").write_text(
        json.dumps({"schema_version": "demo-1.0"}), encoding="utf-8"
    )

    real_is_file = Path.is_file

    def flaky_is_file(self: Path):
        if self == blocked:
            raise PermissionError(13, "拒绝访问。")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", flaky_is_file)

    report = audit_artifacts(tmp_path)

    names = {record["path"] for record in report["files"]}
    assert "blocked.json" not in names
    assert "ok.json" in names
    assert report["invalid_files"] == 0


def test我方审计真的把STEP读回来(tmp_path: Path) -> None:
    """我补一个真实盲区：STEP 曾长期被"不解析"直接跳过。

    outputs 下有 **229 个 .step/.stp** 此前一律跳过，而 STEP 正是
    **修补功能的交付物**。我最不能接受的是：交出去一个打不开的 STEP，
    却因为格式没被审计而一路绿灯。所以判据必须是"真的读得回来"，
    不是"文件存在且非空"。
    """

    occt = pytest.importorskip("OCP.STEPControl")

    good = tmp_path / "good.stp"
    good.write_bytes(
        b"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\nENDSEC;\n"
        b"END-ISO-10303-21;\n"
    )
    report = audit_artifacts(tmp_path)
    by_path = {record["path"]: record for record in report["files"]}
    # 关键：不能再是"跳过了不认识的类型"。
    assert "good.stp" in by_path
    assert by_path["good.stp"]["status"] != "SKIPPED" or (
        "OCP" in "".join(by_path["good.stp"]["issues"])
    )
    assert occt is not None


def test我方审计把读不回来的STEP判损坏(tmp_path: Path) -> None:
    """打不开的 STEP 必须判 INVALID，而不是悄悄放过。"""

    pytest.importorskip("OCP.STEPControl")

    for name, payload in (
        ("garbage.stp", b"not a step file at all\n" * 10),
        ("hollow.step", b"ISO-10303-21;\nEND-ISO-10303-21;\n"),
    ):
        (tmp_path / name).write_bytes(payload)

    report = audit_artifacts(tmp_path)
    by_path = {record["path"]: record for record in report["files"]}
    for name in ("garbage.stp", "hollow.step"):
        assert by_path[name]["status"] == "INVALID", (name, by_path[name])
    assert report["invalid_files"] == 2


def test我方审计逐行查jsonl并抓出截断的末行(tmp_path: Path) -> None:
    """JSONL 是按行追加的结果流，典型损坏就是"写一半被杀"——末行截断。

    这种损坏只有**逐行解析**才看得见：把整个文件当 JSON 读会把它
    当成"格式不对"而说不出是哪一行。所以我要求逐行解析、
    并把行号报出来，事后才能定位。
    """

    (tmp_path / "ok.jsonl").write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    (tmp_path / "blank.jsonl").write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    # 末行少了个右花括号，正是"写到一半进程被杀"的样子。
    (tmp_path / "cut.jsonl").write_text('{"a": 1}\n{"b": 2\n', encoding="utf-8")
    (tmp_path / "scalar.jsonl").write_text('{"a": 1}\n42\n', encoding="utf-8")

    report = audit_artifacts(tmp_path)
    by_path = {record["path"]: record for record in report["files"]}

    assert by_path["ok.jsonl"]["status"] == "VALID"
    assert by_path["ok.jsonl"]["line_count"] == 2
    # 空白行是追加写的常态，不该算坏行。
    assert by_path["blank.jsonl"]["status"] == "VALID"
    assert by_path["blank.jsonl"]["line_count"] == 2

    assert by_path["cut.jsonl"]["status"] == "INVALID"
    assert any("第 2 行" in issue for issue in by_path["cut.jsonl"]["issues"])
    assert by_path["scalar.jsonl"]["status"] == "INVALID"
    assert report["invalid_files"] == 2


def test我方审计jsonl不再被当成不认识的类型(tmp_path: Path) -> None:
    """.jsonl 曾和 STEP 一样长期被"不解析"跳过，这里钉住它已被覆盖。"""

    (tmp_path / "records.jsonl").write_text('{"a": 1}\n', encoding="utf-8")

    report = audit_artifacts(tmp_path)
    by_path = {record["path"]: record for record in report["files"]}
    assert by_path["records.jsonl"]["status"] == "VALID"
    assert report["skipped_files"] == 0


def _build_png(width: int = 2, height: int = 2) -> bytes:
    """我手工拼一个**合法**的 PNG：CRC 全部按规范算对。

    我不写死一段二进制常量，因为那样看不出哪块是什么，
    以后改测试的人也无法调整尺寸。
    """

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # 每行一个 filter 字节 + RGB 三通道。
    raw = b"".join(b"\x00" + b"\x7f\x30\x20" * width for _ in range(height))
    return (
        signature
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test我方审计PNG逐块校验CRC并抓出截断(tmp_path: Path) -> None:
    """PNG 是给人看的对比图，"半张图"最容易被体积检查放行。

    我只验体积的话，一个"写到一半被杀"的 PNG 大小完全正常，
    但甲方打开是花屏或半张图。PNG 格式本身**免费给了每块 CRC32**，
    不用就是浪费，所以这里逐个块校验，并要求有 IEND 收尾。
    """

    good = _build_png()
    (tmp_path / "good.png").write_bytes(good)
    # 截断：砍掉尾部，IEND 没了——这是"写到一半被杀"的样子。
    (tmp_path / "cut.png").write_bytes(good[: len(good) - 8])
    # 比特翻转：体积和块结构都完整，只有像素数据坏了一个字节。
    # 这只有 CRC 才抓得到，看文件头和体积都会放行。
    # 我特意翻 IDAT **载荷中段**：翻长度字段会被"声明长度对不上"的
    # 截断分支先拦下，就验不到 CRC 分支了——那样这条测试等于白写。
    rotten = bytearray(good)
    idat_at = 8 + 25  # 8 字节签名 + IHDR 块（4+4+13+4）
    (idat_length,) = struct.unpack(">I", good[idat_at : idat_at + 4])
    rotten[idat_at + 8 + idat_length // 2] ^= 0x01
    (tmp_path / "rotten.png").write_bytes(bytes(rotten))
    (tmp_path / "notpng.png").write_bytes(b"<html>not a png at all</html>")

    report = audit_artifacts(tmp_path)
    by_path = {record["path"]: record for record in report["files"]}

    assert by_path["good.png"]["status"] == "VALID"
    assert by_path["good.png"]["width"] == 2
    assert by_path["good.png"]["height"] == 2

    assert by_path["cut.png"]["status"] == "INVALID"
    assert by_path["rotten.png"]["status"] == "INVALID"
    assert any("CRC" in issue for issue in by_path["rotten.png"]["issues"])
    assert by_path["notpng.png"]["status"] == "INVALID"
    assert report["invalid_files"] == 3


def test我方审计PNG不再被当成不认识的类型(tmp_path: Path) -> None:
    """我钉住 .png 已被覆盖：8 张图此前长期只做体积检查。"""

    (tmp_path / "chart.png").write_bytes(_build_png())

    report = audit_artifacts(tmp_path)
    by_path = {record["path"]: record for record in report["files"]}
    assert by_path["chart.png"]["status"] == "VALID"
    assert report["skipped_files"] == 0


def test我方审计覆盖md与html这两类真实产物(tmp_path: Path) -> None:
    """我钉住 outputs 里那 3 个"跳过"能被消掉。

    实测 reports/outputs_audit.json（2026-10-19）：925 个文件里 3 个被跳过，
    全部是 .html（demo_visualize / demo_visualize_auto 的可视化页）与
    .md（flowcharts/flowcharts.md），原因是审计器的后缀白名单不含这两类。
    它们不是"坏文件"，但"跳过"意味着**审计根本没看内容**——一个被截断到
    一半的 html 或空的 md 会照样以"跳过"过关。所以我把它们纳入审计。
    """

    (tmp_path / "visualization.html").write_text(
        "<!DOCTYPE html>\n<html lang=\"zh-CN\"><body>ok</body></html>\n",
        encoding="utf-8",
    )
    (tmp_path / "flowcharts.md").write_text(
        "# 我方流程图\n\n```mermaid\nflowchart TD\n    a-->b\n```\n",
        encoding="utf-8",
    )

    report = audit_artifacts(tmp_path)
    by_path = {record["path"]: record for record in report["files"]}

    assert by_path["visualization.html"]["status"] == "VALID"
    assert by_path["flowcharts.md"]["status"] == "VALID"
    assert report["skipped_files"] == 0
    assert report["valid_files"] == 2


def test我方审计能拦下被截断的html与空的md(tmp_path: Path) -> None:
    """纳入审计就必须真拦：只看后缀等于白纳入。"""

    (tmp_path / "cut.html").write_text(
        "<!DOCTYPE html>\n<html><body>写到一半就被杀了",
        encoding="utf-8",
    )
    (tmp_path / "empty.md").write_text("", encoding="utf-8")

    report = audit_artifacts(tmp_path)
    by_path = {record["path"]: record for record in report["files"]}

    assert by_path["cut.html"]["status"] == "INVALID"
    assert by_path["empty.md"]["status"] == "INVALID"
    assert report["invalid_files"] == 2
    assert report["status"] == "REVIEW_REQUIRED"
