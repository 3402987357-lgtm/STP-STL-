"""我把打包好的 onedir 结果压成一个可分发的 zip。

我为什么要有这个脚本：PyInstaller 的 onedir 输出是一个**目录**
（几百 MB、上千个文件），没法直接发给甲方——
拖进微信会散架，邮件附件也过不去。

我为什么选 onedir + zip 而不是 --onefile：
onefile 每次启动都要把几百 MB 解包到临时目录才能跑，
而甲方的使用场景是**反复跑检测**，启动慢会被感知成"卡"。
onedir 启动快；分发不便这个代价我用本脚本压 zip 来消掉。

我只压真正需要的文件，并且**先算校验和再压**，
让甲方解压后能核实到手的东西没被改过。
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = ROOT / "dist_pyi" / "cad-inspector"


def _sha256(path: Path) -> str:
    """我按 1MiB 分块算 sha256，避免把大文件整份读进内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(dist_dir: Path, output: Path) -> int:
    """我把 onedir 打包目录压成交付 zip，并在 zip 外写一份逐文件校验清单。"""

    if not (dist_dir / "cad-inspector.exe").is_file():
        print(
            f"我在 {dist_dir} 里找不到 cad-inspector.exe。"
            "请先跑：python -m PyInstaller cad_inspector.spec"
        )
        return 2

    files = sorted(p for p in dist_dir.rglob("*") if p.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            arcname = path.relative_to(dist_dir).as_posix()
            zf.write(path, arcname)
            manifest_lines.append(f"{_sha256(path)}  {arcname}")

    # 校验清单我单独放在 zip 外面：放进 zip 里的话，
    # 甲方得先解压才能核对，而"解压前先核实"才是它的用处。
    manifest = output.with_suffix(".sha256.txt")
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in files)
    print(f"我已打包 {len(files)} 个文件（共 {total / 1e6:.1f} MB）-> {output}")
    print(f"校验清单 -> {manifest}")
    return 0


def main() -> int:
    """我解析 --dist-dir / --output，把打包工作交给 build 并回传退出码。"""

    parser = argparse.ArgumentParser(description="我把 onedir 结果压成分发包。")
    parser.add_argument("--dist-dir", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "cad-inspector.zip")
    args = parser.parse_args()
    return build(args.dist_dir, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
