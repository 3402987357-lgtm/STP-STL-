"""我提供一个可选的 PyTorch 面级分类器，不让缺少 torch 阻塞几何基线。

算法原理：深度学习在这个项目里是**可选增强**，不是必需依赖。所以 torch 的导入被延迟到
真正构造模型时（``build_pointwise_mlp`` 内部），缺少 torch 只让这一个函数失败，几何
基线、BOM、分型、修补全部照常可用。项目门禁里 torch 也不是必需依赖。

模型本身是逐面 MLP：输入是 ``learning/features.py`` 产出的归一化面特征（重心、法向、
面积占比、距离、距离/阈值），对每个参考面独立输出一个缺失概率 logits。逐面独立而不是
逐点联合，是为了让它的输出能和几何规则的结果按面对齐融合，而不是另起一套判定口径。
"""

from __future__ import annotations


def build_pointwise_mlp(input_dim: int = 9, hidden_dim: int = 64):
    """我构造一个逐面缺失概率模型，实际导入 torch 延迟到使用时。"""

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "我无法构造深度学习模型，因为当前环境没有安装 torch。"
        ) from exc

    class PointwiseMissingMLP(nn.Module):
        """我对每个参考面独立输出缺失概率，方便和几何规则融合。"""

        def __init__(self) -> None:
            """我搭一个两层隐藏层的 MLP，输出逐面的缺失对数几率。"""
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, features):
            """我把逐面特征前向成逐面分数，末维压掉便于和规则分数相加。"""
            return self.network(features).squeeze(-1)

    return PointwiseMissingMLP()
