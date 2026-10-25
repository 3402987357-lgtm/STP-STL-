from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cad_inspector.geometry.stl import save_stl
from cad_inspector.output.visualization import (
    HighlightRegion,
    _triangle_codes,
    _render_html,
    category_color,
    face_bboxes_from_step_face_features,
    highlight_regions_from_inspection,
    highlight_regions_from_step_repair,
    write_comparison_report,
)
from cad_inspector.visualization_cli import main as visualize_main

from test_core import cube_mesh


def _inspection_payload() -> dict:
    """我方构造一个最小缺失区域检测 JSON，覆盖精确编号和退回包围盒两种区域。"""

    return {
        "schema_version": "missing-region-detection-1.3",
        "regions": [
            {
                "region_id": 1,
                "defect_type": "missing",
                "center_xyz_mm": [5.0, 5.0, 10.0],
                "bbox_min_xyz_mm": [4.0, 4.0, 9.5],
                "bbox_max_xyz_mm": [6.0, 6.0, 10.5],
                "area_mm2": 4.0,
                # 这个数不是我编的：甲方任务书示例图里那条红色测量线
                # 标的就是"最小距离 2.7847 mm"。我拿它当测试值，
                # 是为了让用例和甲方给的示意图对得上。
                "max_error_mm": 2.7847,
                "reference_face_ids": [2, 3],
                "candidate_face_ids": [],
                "review_required": False,
            },
            {
                "region_id": 2,
                "defect_type": "extra",
                "center_xyz_mm": [0.0, 0.0, 0.0],
                "bbox_min_xyz_mm": [0.0, 0.0, 0.0],
                "bbox_max_xyz_mm": [1.0, 1.0, 1.0],
                "area_mm2": 1.0,
                "reference_face_ids": [],
                "candidate_face_ids": [4, 5, 6],
                "review_required": True,
            },
            {
                "region_id": 3,
                "defect_type": "unknown_kind",
                "center_xyz_mm": [0.0, 0.0, 0.0],
                "bbox_min_xyz_mm": None,
                "bbox_max_xyz_mm": None,
                "area_mm2": None,
                "review_required": True,
            },
        ],
    }


def test我方从缺失区域JSON解析精确三角面编号() -> None:
    regions = highlight_regions_from_inspection(_inspection_payload())
    assert [region.region_id for region in regions] == [1, 2, 3]

    missing = regions[0]
    assert missing.category == "missing"
    assert missing.triangle_space == "reference"
    assert missing.triangle_indices == (2, 3)
    assert missing.has_geometry is True
    assert missing.status == "DETECTED"

    extra = regions[1]
    assert extra.category == "extra"
    assert extra.triangle_space == "candidate"
    assert extra.triangle_indices == (4, 5, 6)
    assert extra.review_required is True
    assert extra.status == "REVIEW_REQUIRED"

    unknown = regions[2]
    assert unknown.triangle_indices == ()
    assert unknown.triangle_space == "unknown"
    assert unknown.has_geometry is False


def test我方从STEP修补JSON按来源Face还原包围盒() -> None:
    face_payload = {
        "faces": [
            {
                "face_id": 3,
                "bbox_min_xyz_mm": [0.0, 0.0, 0.0],
                "bbox_max_xyz_mm": [2.0, 2.0, 2.0],
            },
            {
                "face_id": 4,
                "bbox_min_xyz_mm": [8.0, 8.0, 8.0],
                "bbox_max_xyz_mm": [10.0, 10.0, 10.0],
            },
            {"face_id": 5, "bbox_min_xyz_mm": None, "bbox_max_xyz_mm": None},
        ]
    }
    face_bboxes = face_bboxes_from_step_face_features(face_payload)
    assert set(face_bboxes) == {3, 4}

    repair_payload = {
        "repaired_regions": [
            {
                "region_id": 7,
                "damage_type": "hole",
                "status": "REPAIRED",
                "area_repaired": 12.5,
                "continuity_level": "G1",
                "source_face_ids": [3, 4],
                "review_required": False,
            },
            {
                "region_id": 8,
                "damage_type": "hole",
                "status": "REVIEW_REQUIRED",
                "area_repaired": 0.0,
                "continuity_level": "NOT_VERIFIED",
                "source_face_ids": [99],
                "review_required": True,
            },
        ]
    }
    regions = highlight_regions_from_step_repair(repair_payload, face_bboxes)
    assert regions[0].category == "repaired"
    assert regions[0].has_geometry is True
    assert regions[0].bbox_min_xyz_mm == (0.0, 0.0, 0.0)
    assert regions[0].bbox_max_xyz_mm == (10.0, 10.0, 10.0)
    assert regions[0].center_xyz_mm == (5.0, 5.0, 5.0)
    assert regions[1].has_geometry is False
    assert regions[1].status == "REVIEW_REQUIRED"


def test我方分类颜色对已知分类固定且未知分类稳定() -> None:
    assert category_color("hole") == "#d62728"
    assert category_color("missing_face") == "#ff7f0e"
    unknown_first = category_color("我方自定义分类")
    unknown_second = category_color("我方自定义分类")
    assert unknown_first == unknown_second
    assert unknown_first.startswith("#") and len(unknown_first) == 7


def test我方精确编号只命中目标三角面且不跨模型串色() -> None:
    mesh = cube_mesh()
    candidate_region = HighlightRegion(
        region_id=1,
        category="extra",
        triangle_indices=(1, 3, 5),
        triangle_space="candidate",
    )
    reference_region = HighlightRegion(
        region_id=2,
        category="missing",
        triangle_indices=(0, 2),
        triangle_space="reference",
    )
    candidate_codes = _triangle_codes(
        mesh, [candidate_region, reference_region], "candidate"
    )
    assert np.flatnonzero(candidate_codes).tolist() == [1, 3, 5]
    reference_codes = _triangle_codes(
        mesh, [candidate_region, reference_region], "reference"
    )
    assert np.flatnonzero(reference_codes).tolist() == [0, 2]


def test我方包围盒区域在拿不到编号时仍能命中() -> None:
    mesh = cube_mesh()
    region = HighlightRegion(
        region_id=1,
        category="missing_face",
        bbox_min_xyz_mm=(0.0, 0.0, 9.9),
        bbox_max_xyz_mm=(10.0, 10.0, 10.1),
    )
    codes = _triangle_codes(mesh, [region], "candidate")
    assert codes.sum() > 0
    # 只有顶面两个三角面的重心 z=10 落在包围盒内。
    assert int((codes > 0).sum()) == 2


def _count_red_pixels(path: Path) -> int:
    """我方统计明显偏红的像素，用来证明高亮真的画到了图上。"""

    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    mask = (red > 150) & (red - green > 60) & (red - blue > 60)
    return int(mask.sum())


def test我方可视化输出对比图分类图和自包含HTML(tmp_path: Path) -> None:
    before = cube_mesh(omit_top=True)
    after = cube_mesh()
    regions = [
        HighlightRegion(
            region_id=1,
            category="hole",
            status="REPAIRED",
            area_mm2=4.0,
            triangle_indices=(0, 1, 2, 3),
            triangle_space="candidate",
        ),
        HighlightRegion(
            region_id=2,
            category="missing",
            status="DETECTED",
            triangle_indices=(0, 1),
            triangle_space="reference",
        ),
        HighlightRegion(region_id=3, category="topology_damage"),
    ]
    artifacts = write_comparison_report(
        before,
        after,
        regions,
        tmp_path,
        title="我方单元测试对比图",
    )
    for key in (
        "comparison_png",
        "classification_png",
        "classification_candidate_png",
        "html",
        "manifest",
    ):
        assert artifacts[key].exists(), key
    assert artifacts["comparison_png"].stat().st_size > 5000
    assert _count_red_pixels(artifacts["comparison_png"]) > 0
    assert _count_red_pixels(artifacts["classification_candidate_png"]) > 0

    html_text = artifacts["html"].read_text(encoding="utf-8")
    assert "data:image/png;base64," in html_text
    assert "hole" in html_text
    assert "missing" in html_text
    assert "我方单元测试对比图" in html_text

    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "visualization-1.0"
    assert manifest["status"] == "COMPLETE"
    assert manifest["region_count"] == 3
    assert manifest["regions_with_geometry"] == 2
    assert manifest["regions_by_category"] == {
        "hole": 1,
        "missing": 1,
        "topology_damage": 1,
    }
    assert manifest["before_triangle_count"] == 10
    assert manifest["after_triangle_count"] == 12


def test我方可视化在只有修复前模型时也能出图(tmp_path: Path) -> None:
    artifacts = write_comparison_report(
        cube_mesh(omit_top=True),
        None,
        [],
        tmp_path,
    )
    assert artifacts["comparison_png"].exists()
    assert artifacts["classification_candidate_png"].exists()
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert manifest["after_triangle_count"] is None
    assert manifest["region_count"] == 0


def test我方可视化命令行端到端(tmp_path: Path) -> None:
    stl_path = tmp_path / "cube.stl"
    save_stl(cube_mesh(), stl_path)
    regions_path = tmp_path / "regions.json"
    regions_path.write_text(
        json.dumps(_inspection_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    code = visualize_main(
        [
            "--before",
            str(stl_path),
            "--after",
            str(stl_path),
            "--regions",
            str(regions_path),
            "--output-dir",
            str(output_dir),
            "--dpi",
            "60",
        ]
    )
    assert code == 0
    assert (output_dir / "comparison_before_after.png").exists()
    assert (output_dir / "visualization.html").exists()
    manifest = json.loads(
        (output_dir / "visualization.json").read_text(encoding="utf-8")
    )
    assert manifest["region_count"] == 3
    # 我方在没显式给副标题时，必须自动带上真实检测状态说明。
    assert manifest["subtitle"]
    assert "检测状态" in manifest["subtitle"]


def test我方可视化命令行拒绝坏输入(tmp_path: Path) -> None:
    missing = tmp_path / "nope.stl"
    assert (
        visualize_main(
            ["--before", str(missing), "--output-dir", str(tmp_path / "out")]
        )
        == 2
    )
    stl_path = tmp_path / "cube.stl"
    save_stl(cube_mesh(), stl_path)
    assert (
        visualize_main(
            [
                "--before",
                str(stl_path),
                "--output-dir",
                str(tmp_path / "out2"),
                "--dpi",
                "1",
            ]
        )
        == 2
    )
    bad_regions = tmp_path / "bad.json"
    bad_regions.write_text("[1, 2, 3]", encoding="utf-8")
    assert (
        visualize_main(
            [
                "--before",
                str(stl_path),
                "--regions",
                str(bad_regions),
                "--output-dir",
                str(tmp_path / "out3"),
            ]
        )
        == 2
    )


def test我方HTML报告对注入内容做转义() -> None:
    """我锁定 HTML 报告的转义契约。

    报告里会写入分类名这类来自数据的内容。分类名在模具工程里常由
    上游系统给出，不能假定它是干净的文本。若不转义，一个含
    `<script>` 或 `onerror=` 的分类名就会在甲方打开报告时执行。

    我用一个同时含闭合标签、属性逃逸和事件处理器的样本验证：
    输出里不留任何可执行形式。
    """

    hostile = '<img src=x onerror="alert(1)"></h2><script>alert(2)</script>'
    region = HighlightRegion(
        region_id=1,
        category=hostile,
        area_mm2=1.0,
        review_required=True,
        bbox_min_xyz_mm=(0.0, 0.0, 0.0),
        bbox_max_xyz_mm=(1.0, 1.0, 1.0),
    )

    html = _render_html(
        title=hostile,
        images=[],
        regions=[region],
        counts={hostile: 1},
    )

    assert "<script>" not in html
    assert 'onerror="alert' not in html
    assert hostile not in html
    assert "&lt;" in html


def test我方区域序列化带出三角面所在空间() -> None:
    """我锁定 triangle_space 必须随区域一起输出。

    三角面编号只在特定网格空间（参考件或待检件）里有效。
    丢了它，下游拿着一堆编号无法判断该往哪个模型上画，只能靠猜。

    注意这里序列化出去的是 triangle_count 而不是完整的编号列表：
    一个区域可能有上万个三角面，全量写进报告会把 JSON 撑大好几倍，
    而对人工复核来说"这个区域覆盖了多少个面"已经足够。
    """

    region = HighlightRegion(
        region_id=7,
        category="missing",
        triangle_indices=(1, 2, 3),
        triangle_space="reference",
    )

    payload = region.to_dict()
    assert region.to_dict()["triangle_space"] == "reference"
    assert payload["triangle_count"] == 3


def test我方把区域误差带进可视化对象并序列化() -> None:
    """我钉住"误差数值必须能走到图上"这条链路的入口。

    我为什么加（2026-10-17）：甲方任务书的示例图画的是缺口处一条红色
    测量线 + "2.7847 mm" 数值标签——他们要看的是"这里差多少"，
    不只是"这里有个洞"。误差本来就在 regions.csv / result.json 里，
    但可视化层一直只读 area_mm2，于是图上只有颜色没有数字。
    这个用例钉的是第一段：解析时别把 max_error_mm 丢了。
    """

    regions = highlight_regions_from_inspection(_inspection_payload())
    assert regions[0].max_error_mm == pytest.approx(2.7847)
    assert regions[0].to_dict()["max_error_mm"] == pytest.approx(2.7847)
    # 没写误差的区域保持 None，不能默认成 0——
    # 图上标一个 0.000mm 会被读成"这里没有误差"，与"没测到"是两回事。
    assert regions[1].max_error_mm is None


def test我方可视化在区域不多时把误差标到图上() -> None:
    """我钉住第二段：区域数在上限内时，误差数值真的画上去了。

    我顺带钉住上限行为：噪声件能出几千个区域，逐区域标数字会把图糊成
    一片。超过上限时只保留高亮、不标数字，数值去 HTML 明细表里看。
    """

    from cad_inspector.output.visualization import (
        _MAX_ERROR_LABELS,
        _render_figure,
    )

    mesh = cube_mesh()
    few = highlight_regions_from_inspection(_inspection_payload())
    # _render_figure 返回 (figure, plt)，不是单个 figure。
    figure, _ = _render_figure(mesh, mesh, few, title="标注测试")
    # 两行各三个视角：前 3 个轴是修复前（candidate），后 3 个是修复后（reference）。
    candidate_labels = [
        text.get_text() for axis in figure.axes[:3] for text in axis.texts
    ]
    reference_labels = [
        text.get_text() for axis in figure.axes[3:] for text in axis.texts
    ]
    assert any("2.785mm" in label for label in reference_labels), (
        f"missing 区域属于参考件，应当标在修复后那一行，实际：{reference_labels}"
    )
    # 同一个数字不能两行都出现：那会让人以为两侧都有这块缺陷。
    assert not any("2.785mm" in label for label in candidate_labels), (
        f"missing 区域不该标到修复前那一行，实际：{candidate_labels}"
    )

    many = [
        HighlightRegion(
            region_id=index,
            category="missing",
            center_xyz_mm=(1.0, 1.0, 1.0),
            max_error_mm=0.2,
            # 必须给空间，否则无论标不标都匹配不上任何一行，
            # 这个用例就测不到"上限"这件事了。
            triangle_space="reference",
        )
        for index in range(1, _MAX_ERROR_LABELS + 2)
    ]
    figure, _ = _render_figure(mesh, mesh, many, title="超上限")
    labels = [text.get_text() for axis in figure.axes for text in axis.texts]
    assert not any("0.200mm" in label for label in labels), (
        f"区域数超过 {_MAX_ERROR_LABELS} 时不应逐个标数字，实际：{labels}"
    )
