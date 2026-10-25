"""我方为刚落地的**坏面检测**写单元测试。

我为什么要补这一轮：上一轮我把坏面定位能力做成了产品功能（诊断里带
`invalid_face_count` / `first_invalid_face_id`，并在 warnings 里报出来），
但当时**没有为它新增单元测试**——现有的 36 个测试只证明了"没破坏现有
行为"，**没有断言"坏面真的被报出来"**。这是个测试缺口，这一轮补上。

我怎么测（这是关键设计）：`_topology_snapshot` 是**注入式**的——所有 OCP
类型都由参数传入，本来就是为了方便测试。所以我用：
    **真的拓扑**（一个真 Box，让遍历/索引的行为都是真的）+
    **假的有效性判定**（我自己控制哪几个面"无效"）
这样既不依赖真实语料（跑得快、结果确定），又能精确验证统计逻辑。

ponytail: 我没有直接用真实语料里的 训练集/1，因为它有 730 个面、读取要
几十秒，放进单元测试会让测试套件变慢；而用真 Box + 假判定可以在毫秒级
验证同样的逻辑。
"""

from pathlib import Path

# 我为什么按仓库根定位：这里原本写相对路径 Path("data")，
# 只在"当前工作目录恰好是仓库根"时才找得到，别处执行 pytest 会静默 skip。
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


import pytest

from cad_inspector.repair.step_hole import _topology_snapshot


def _corpus_file(relative: str) -> Path:
    """我取出真实语料里的一个件，不存在就让 pytest 跳过。"""

    path = DATA_ROOT / "provided" / "三维模型破损区域智能修补" / relative
    if not path.exists():
        pytest.skip("甲方提供的真实破损件不存在：%s" % relative)
    return path


def _read_shape(path: Path) -> object:
    """我读出一个 STEP 形状。"""

    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    reader.ReadFile(str(path))
    reader.TransferRoots()
    return reader.OneShape()


def _real_box() -> object:
    """我造一个真的 Box 当测试用形状。"""

    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    return BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()


def _real_types() -> dict:
    """我取出真的 OCP 类型，保证遍历与索引行为都是真的。"""

    from OCP.OCP.collections import (
        IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
    )
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer, TopExp
    from OCP.TopoDS import TopoDS

    return {
        "top_abs_edge": TopAbs_EDGE,
        "top_abs_face": TopAbs_FACE,
        "top_abs_shell": TopAbs_SHELL,
        "top_abs_solid": TopAbs_SOLID,
        "explorer_type": TopExp_Explorer,
        "indexed_map_type": IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
        "map_shapes": TopExp.MapShapes_s,
        "topods_type": TopoDS,
    }


class _FakeAnalyzer:
    """我假的有效性判定：我指定哪几个**面序号**无效，其余都算有效。

    ponytail: 我只替换"判定"，不替换拓扑——因为要验证的是**统计逻辑**，
    不是 OCP 的遍历行为。这样测试既快又不会因 OCP 版本差异而脆弱。
    """

    def __init__(self, invalid_ids: set[int]) -> None:
        self._invalid_ids = set(invalid_ids)
        self._seen = 0

    def __call__(self, face: object) -> "_FakeAnalyzer":
        self._seen += 1
        self._is_valid = self._seen not in self._invalid_ids
        return self

    def IsValid(self) -> bool:
        return self._is_valid


def test我方不传analyzer时坏面字段完全不出现(tmp_path: Path) -> None:
    """我不传 analyzer_type 时，诊断**不应**多出坏面字段。

    我为什么专门测这一条：坏面统计是**逐面**跑有效性检查的额外开销，我把它
    做成可选能力，承诺"不传就零开销、行为不变"。这条测试就是守住这个承诺，
    防止以后有人不小心把它改成默认全开。
    """

    counts, _, _, _ = _topology_snapshot(_real_box(), **_real_types())

    assert "invalid_face_count" not in counts
    assert "first_invalid_face_id" not in counts
    # 常规统计仍然要正常工作，不能因为没传判定就少算。
    assert counts["face_count"] == 6


def test我方传入analyzer时能报出坏面数量与首个编号(tmp_path: Path) -> None:
    """我传入 analyzer_type 时，应正确统计坏面数**和**第一个坏面的编号。

    我为什么两个都断言：只断言数量不够——如果统计对了但编号错了，甲方按
    编号跳到那个面去看，会看错地方。所以数量和编号都要验。
    """

    counts, _, _, _ = _topology_snapshot(
        _real_box(), analyzer_type=_FakeAnalyzer({2, 4}), **_real_types()
    )

    assert counts["invalid_face_count"] == 2
    assert counts["first_invalid_face_id"] == 2


def test我方没有坏面时报零而不是漏字段(tmp_path: Path) -> None:
    """我没有坏面时，应**明确报 0**，而不是把字段去掉。

    我为什么区分这一条和上一条：调用方（诊断输出）会直接读这个字段。如果
    "没有坏面"和"没有开启"都表现为字段不存在，调用方就分不清是"查过了没
    问题"还是"压根没查"——这两种情况对甲方的含义完全不同。
    """

    counts, _, _, _ = _topology_snapshot(
        _real_box(), analyzer_type=_FakeAnalyzer(set()), **_real_types()
    )

    assert counts["invalid_face_count"] == 0
    assert counts["first_invalid_face_id"] == 0


def test我方单个面判定抛异常不影响整体统计(tmp_path: Path) -> None:
    """我某个面判定报错时，不应让整个统计失败。

    我为什么测这个：真实 STEP 里有各种畸形面，逐个面跑检查时**个别面报错
    是常态**。我在实现里对单面异常做了跳过处理，这条测试守住它——否则一个
    坏面就会让整件的拓扑快照全部失败，连常规统计都拿不到。
    """

    class _BoomAnalyzer:
        """我在第 3 个面上抛异常，其余面都判有效。"""

        def __init__(self) -> None:
            self._seen = 0

        def __call__(self, face: object) -> "_BoomAnalyzer":
            self._seen += 1
            if self._seen == 3:
                raise RuntimeError("我模拟这个面判定失败")
            return self

        def IsValid(self) -> bool:
            return True

    counts, _, _, _ = _topology_snapshot(
        _real_box(), analyzer_type=_BoomAnalyzer(), **_real_types()
    )

    # 常规统计不受影响，坏面数为 0（那个报错的面被跳过，不算坏面）。
    assert counts["face_count"] == 6
    assert counts["invalid_face_count"] == 0


@pytest.mark.parametrize("invalid_ids", [set(), {1}, {1, 6}, {2, 3, 4}])
def test我方坏面数量随指定集合正确变化(invalid_ids: set) -> None:
    """我用参数化覆盖多种坏面分布，确认数量与首个编号都对。

    我为什么用参数化：手工写四遍同样的断言容易抄错，参数化能让"数量=集合
    大小、首个编号=最小编号"这个规律在多组输入上都得到验证。
    """

    counts, _, _, _ = _topology_snapshot(
        _real_box(), analyzer_type=_FakeAnalyzer(invalid_ids), **_real_types()
    )

    assert counts["invalid_face_count"] == len(invalid_ids)
    if invalid_ids:
        assert counts["first_invalid_face_id"] == min(invalid_ids)
    else:
        assert counts["first_invalid_face_id"] == 0


def test我方真实语料能报出已知的坏面() -> None:
    """我在**真实语料**上验证坏面被如实报出来（不再只用假判定）。

    我为什么要补这一条：上面几个用例用的是假 analyzer，只能证明**统计逻辑**
    对。但甲方真正关心的是"我这个真实件上到底报得准不准"。所以我在全量普查
    已知答案的件上再做一次：
        训练集/1 -> 普查已知 **1 个坏面，编号 633**
        测试集/1 -> 普查已知 **0 个坏面**
    带对照，避免"见谁都报"被当成"报对了"。

    ponytail: 我只做**拓扑快照**这一层、不跑完整修补，因为完整修补要几十秒，
    放进单元测试太慢；而坏面统计本身只需要读到形状即可验证。
    """

    from OCP.BRepCheck import BRepCheck_Analyzer

    # 有坏面的件。
    counts_bad, _, _, _ = _topology_snapshot(
        _read_shape(_corpus_file("训练集/1.stp")),
        analyzer_type=BRepCheck_Analyzer,
        **_real_types(),
    )
    assert counts_bad["invalid_face_count"] == 1
    assert counts_bad["first_invalid_face_id"] == 633

    # 对照：已知没有坏面的件，必须报 0，且不能出现"有坏面"的误报。
    counts_ok, _, _, _ = _topology_snapshot(
        _read_shape(_corpus_file("测试集/1.stp")),
        analyzer_type=BRepCheck_Analyzer,
        **_real_types(),
    )
    assert counts_ok["invalid_face_count"] == 0
    assert counts_ok["first_invalid_face_id"] == 0


def test我方真实语料端到端把坏面写进诊断提示() -> None:
    """我验证坏面信息真的出现在**产品输出**的 warnings 里。

    我为什么要补这一条：上面只验证了 `_topology_snapshot` 这一层的数字。但
    甲方看到的是**最终报告的 warnings**，不是内部字典。所以必须在完整修补
    流程上确认那句提示真的出现——否则数字对了、甲方却看不到，等于没做。

    我选用 `测试集/2`：它 141 个面（比训练集/1 的 730 个面小很多），有 1 个
    已知坏面（#133），跑完整修补不至于让单元测试太慢。
    """

    from cad_inspector.domain.repair import RepairConfig
    from cad_inspector.repair.step_hole import repair_step_holes

    path = _corpus_file("测试集/2.stp")
    result = repair_step_holes(
        str(path),
        config=RepairConfig(
            max_boundary_edges=4096,
            max_hole_area_mm2=1_000_000.0,
            max_planarity_error_mm=0.25,
            do_no_harm=True,
            preserve_original_on_failure=True,
        ),
    )

    assert result.input_topology.get("invalid_face_count") == 1
    joined = " ".join(result.warnings or [])
    assert "坏面" in joined
    # 我同时确认它是"只报不修"：提示里要明确说出不动它。
    assert "不动它" in joined
