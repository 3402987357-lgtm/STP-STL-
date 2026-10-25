"""我钉住任务书交付物"算法流程图"与源码的一致性。

任务书要求交付"算法源码"与"算法流程图"两样东西。图是手写节点清单，
写着写着就会和源码脱节，而读图的人会把它当权威。这个文件把我加的两道
校验钉死，并且**证明它们不是摆设**——用变异用例让校验真的报错。
"""

from __future__ import annotations

from cad_inspector.output.flowchart import (
    FlowChart,
    FlowNode,
    build_flowcharts,
    verify_flowcharts,
    verify_main_flow_order,
)


def test我方主流程图顺序与源码调用顺序一致() -> None:
    """图里写着"顺序取自 inspect_pair 的实际调用链"，这条要真的被守住。

    顺序不是装饰：配准必须排在阈值之前，因为阈值里的噪声分量用的就是
    配准残差。顺序错了，这张图就是在教人做错事。
    """

    result = verify_main_flow_order()

    assert result["consistent"] is True
    assert result["missing_from_source"] == []
    # 至少要有配准、阈值、检测这三步，否则"一致"是空的。
    assert "align_mesh" in result["observed_order"]
    assert "recommend_threshold_mm" in result["observed_order"]
    assert "detect_missing_regions" in result["observed_order"]
    # 关键顺序：配准在阈值之前。
    order = result["observed_order"]
    assert order.index("align_mesh") < order.index("recommend_threshold_mm")


def test我方把顺序调换后校验必须报不一致() -> None:
    """变异用例：证明上面的"一致"不是恒真。"""

    original = next(c for c in build_flowcharts() if c.key == "main")
    # 把"配准"和"阈值"对调，模拟有人改了流水线顺序却忘了改图。
    nodes = list(original.nodes)
    positions = {node.key: i for i, node in enumerate(nodes)}
    register = nodes[positions["register"]]
    threshold = nodes[positions["threshold"]]
    nodes[positions["register"]] = threshold
    nodes[positions["threshold"]] = register

    tampered = FlowChart(
        key="main",
        title=original.title,
        subtitle=original.subtitle,
        nodes=nodes,
        edges=original.edges,
    )

    assert verify_main_flow_order(tampered)["consistent"] is False


def test我方符号校验能抓到改了名的函数() -> None:
    """模块还在、函数改了名，旧校验会放过；现在必须报出来。"""

    original = next(c for c in build_flowcharts() if c.key == "main")
    nodes = [
        FlowNode(
            node.key,
            node.label,
            node.module,
            symbol="this_symbol_does_not_exist",
        )
        for node in original.nodes
    ]
    tampered = FlowChart(
        key="main",
        title=original.title,
        subtitle=original.subtitle,
        nodes=nodes,
    )

    result = verify_flowcharts([tampered])

    assert result["consistent"] is False
    assert result["missing_symbols"]
    # 模块本身是好的，所以模块这一项不该被牵连报错。
    assert result["missing_modules"] == []


def test我方真实三张图全部通过模块与符号校验() -> None:
    result = verify_flowcharts()

    assert result["consistent"] is True
    assert result["missing_modules"] == []
    assert result["missing_symbols"] == []
    # 主流程图上确实声明了可校验的符号，不是空跑一遍。
    assert result["checked_symbol_count"] >= 8
