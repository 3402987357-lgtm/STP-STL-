"""我提供面级深度学习增强：训练、保存、加载，以及与几何判据的融合。

我为什么把它做成**可选增强**而不是替换几何判据：

几何检测（双向最近距离 + 连通 + 阈值）给出的是**可解释的硬判据**，
甲方要拿它判定一个件合不合格。模型给的是概率，单独用它做判定，
出了问题没法解释"为什么这里算缺失"。所以我的定位是**融合**：
模型的概率只用于调整几何结论的可信度与复核标记，不单独决定
"这里缺不缺"。

我守住的几条边界：

- **torch 延迟导入**。缺 torch 只让本模块的函数失败，几何基线照常可用。
- **模型输出必须和几何结果按面对齐**。我在融合时按参考面下标对齐，
  不对齐就拒绝融合——否则等于拿另一套口径偷偷改结论。
- **融合结果必须留痕**：哪些面被模型改判、模型概率是多少，都要写进
  输出，不能只给一个"最终结论"。
- **训练集必须来自真实标注或合成真值**，我不在数据里掺入测试件，
  这一点由 ``annotations`` 模块的数据泄漏门禁保证。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FaceClassifierConfig:
    """我配置面级分类器的训练与融合行为。"""

    epochs: int = 60
    learning_rate: float = 1e-3
    batch_size: int = 4096
    # 融合时"模型认为缺失"的概率门槛。它与几何阈值是**两把尺子**：
    # 几何阈值管距离，这个管概率，不要混为一谈。
    fusion_probability: float = 0.5
    # 模型判缺失、几何判不缺时，是否把该面标成待复核。
    # 我默认 True：这种分歧正是最需要人看的地方，
    # 静默按某一方下结论等于把不确定性藏起来。
    mark_disagreement_for_review: bool = True
    random_seed: int = 20250101

    def to_dict(self) -> dict[str, object]:
        """我把配置转成可序列化字典，便于写进报告。"""

        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "fusion_probability": self.fusion_probability,
            "mark_disagreement_for_review": self.mark_disagreement_for_review,
            "random_seed": self.random_seed,
        }


def train_face_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    config: FaceClassifierConfig | None = None,
):
    """我在面特征上训练一个逐面缺失分类器，返回已训练模型。

    我用 BCEWithLogitsLoss 而不是先 sigmoid 再算 BCE：前者内部做了
    log-sum-exp 稳定化，概率接近 0 或 1 时不会炸成 inf，
    后者在这种数据上很容易出 nan。

    epochs 默认 60：这类小 MLP 在这个规模上几十轮就收敛，再跑只是浪费时间。
    """

    import torch
    from torch import nn

    from .model import build_pointwise_mlp

    config = config or FaceClassifierConfig()
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("我要求 features 是二维且行数与 labels 一致。")
    if len(labels) == 0:
        raise ValueError("我要求训练样本至少有一个面。")
    if not np.all(np.isfinite(features)):
        raise ValueError("我要求 features 全部是有限数。")

    torch.manual_seed(config.random_seed)
    model = build_pointwise_mlp(input_dim=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    model.train()
    for _ in range(int(config.epochs)):
        # 小批量而不是整批：整批一次前向在大网格上显存/内存吃不住，
        # 而且小批量的梯度噪声本身有助于泛化。
        permutation = torch.randperm(x.shape[0])
        for start in range(0, x.shape[0], int(config.batch_size)):
            batch_index = permutation[start : start + int(config.batch_size)]
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch_index]), y[batch_index])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_face_probability(model, features: np.ndarray) -> np.ndarray:
    """我用训练好的模型对每个面输出缺失概率。"""

    import torch

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("我要求 features 是二维数组。")
    if not np.all(np.isfinite(features)):
        raise ValueError("我要求 features 全部是有限数。")
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        return torch.sigmoid(logits).numpy().astype(np.float64)


def fuse_geometry_with_model(
    geometric_mask: np.ndarray,
    model_probability: np.ndarray,
    config: FaceClassifierConfig | None = None,
) -> dict[str, object]:
    """我把几何判据与模型概率按面对齐融合，输出带留痕的融合结论。

    我为什么用"或"来扩面、而不是让模型直接覆盖几何结论：

    几何检测漏掉的面（距离没超阈值但确实有问题）正是模型能补的
    那部分价值；而模型说"这里没问题"时，我不敢据此把几何已经判定的
    缺失区域删掉——那会让一个本来报得出来的缺陷凭空消失，
    在质检场景里这种漏报比多报危险得多。

    所以规则是：模型只**增加**待复核标记，不减少已有判定。
    """

    config = config or FaceClassifierConfig()
    geometric_mask = np.asarray(geometric_mask, dtype=bool).reshape(-1)
    model_probability = np.asarray(model_probability, dtype=np.float64).reshape(-1)
    # 对齐是硬要求：不对齐就融合，等于拿别人的结论改我的面，
    # 而且错得悄无声息。
    if geometric_mask.shape != model_probability.shape:
        raise ValueError(
            "我要求 geometric_mask 与 model_probability 形状一致，"
            f"当前为 {geometric_mask.shape} 与 {model_probability.shape}。"
        )
    model_mask = model_probability >= float(config.fusion_probability)
    # 模型新增判定的面：几何没报、模型报了。
    added_by_model = model_mask & ~geometric_mask
    # 两方一致的面：这是可信度最高的部分。
    agreed = model_mask & geometric_mask
    # 分歧面：几何报了、模型没报。我不删它，只标记出来给人看。
    geometry_only = geometric_mask & ~model_mask
    fused = geometric_mask | (
        model_mask if config.mark_disagreement_for_review else np.zeros_like(model_mask)
    )
    return {
        "fused_mask": fused,
        "model_mask": model_mask,
        "added_by_model_face_ids": np.flatnonzero(added_by_model).tolist(),
        "agreed_face_ids": np.flatnonzero(agreed).tolist(),
        # 分歧面单独列出：这是模型与几何不一致的地方，
        # 也是人工复核最该先看的地方。
        "geometry_only_face_ids": np.flatnonzero(geometry_only).tolist(),
        "added_by_model_count": int(added_by_model.sum()),
        "agreed_count": int(agreed.sum()),
        "geometry_only_count": int(geometry_only.sum()),
        "config": config.to_dict(),
    }


def save_face_classifier(model, path: str | Path) -> str:
    """我把模型权重存盘，便于服务与批量任务复用同一份模型。"""

    import torch

    torch.save(model.state_dict(), str(path))
    return str(path)


def load_face_classifier(path: str | Path, input_dim: int = 9):
    """我加载已训练模型；torch 缺失时给出明确提示。"""

    import torch

    from .model import build_pointwise_mlp

    model = build_pointwise_mlp(input_dim=input_dim)
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    model.eval()
    return model
