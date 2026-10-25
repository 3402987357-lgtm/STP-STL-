"""我为面级深度学习增强区域写回归测试。

我钉住三件事：**能学得会**、**学不会时不装懂**、**融合不删已有判定**。
第三条最关键——模型把几何已报出的缺失区域删掉，在质检场景里
就是凭空制造漏报，比多报危险得多。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from cad_inspector.learning.face_classifier import (  # noqa: E402
    FaceClassifierConfig,
    fuse_geometry_with_model,
    load_face_classifier,
    predict_face_probability,
    save_face_classifier,
    train_face_classifier,
)
from cad_inspector.learning.features import build_face_features  # noqa: E402
from test_core import cube_mesh  # noqa: E402


def _separable_case() -> tuple[np.ndarray, np.ndarray]:
    """我造一份特征与标签强相关的数据：距离大的面判缺失。"""

    mesh = cube_mesh()
    distances = np.linspace(0.0, 0.4, len(mesh.faces))
    return build_face_features(mesh, distances, 0.15)


def _many_face_case(count: int = 3000) -> np.ndarray:
    """我造一份面数足够的特征表，用于"不该学会"的用例。

    为什么不能拿 12 面的立方体验这件事：样本太少时模型靠**记忆**就能
    拟合随机标签（我实测 n=12 时随机标签准确率能到 0.67），
    那是过拟合而不是模型有 bug，会把这条用例变成假警报。
    面数上千后记忆不再可行，准确率才会回落到多数类占比附近。
    """

    mesh = cube_mesh()
    base, _ = build_face_features(mesh, np.linspace(0.0, 0.4, len(mesh.faces)), 0.15)
    rng = np.random.default_rng(20250101)
    # 叠加噪声扰动行数，避免大量逐位重复的行让"记忆"重新变得容易。
    tiled = np.tile(base, (count // len(base) + 1, 1))[:count]
    return tiled + rng.normal(0.0, 0.01, size=tiled.shape).astype(np.float32)


def test我方分类器能学会可分的数据() -> None:
    features, labels = _separable_case()

    model = train_face_classifier(
        features, labels, FaceClassifierConfig(epochs=80)
    )
    probability = predict_face_probability(model, features)

    assert probability.shape == (len(labels),)
    assert np.all(np.isfinite(probability))
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    # 可分的线性结构，训练后应当基本全对。
    assert ((probability >= 0.5).astype(int) == labels).mean() >= 0.9


def test我方分类器在没有信号时不会装懂() -> None:
    """随机标签不该被"学会"：学到就是数据泄漏或实现错误。

    我为什么钉这条：一个在噪声上也能拿高分的分类器，
    上线后给出的概率是假的，比没有模型更危险。
    """

    features = _many_face_case()
    rng = np.random.default_rng(20250101)
    # 与特征完全独立的随机标签。
    random_labels = (rng.random(len(features)) < 0.5).astype(np.int64)

    model = train_face_classifier(
        features, random_labels, FaceClassifierConfig(epochs=20)
    )
    probability = predict_face_probability(model, features)

    accuracy = ((probability >= 0.5).astype(int) == random_labels).mean()
    # 面数上千后记忆不可行，准确率必须回落到随机水平附近。
    assert accuracy <= 0.65


def test我方融合不会删除几何已判定的面() -> None:
    """模型只增加待复核标记，不减少已有判定——漏报比多报危险。"""

    config = FaceClassifierConfig(fusion_probability=0.5)
    geometric = np.array([True, True, False, False])
    # 模型认为只有第 1、3 个面缺失。
    probability = np.array([0.9, 0.1, 0.8, 0.2])

    fused = fuse_geometry_with_model(geometric, probability, config)

    # 几何判定的两个面必须都还在融合结果里。
    assert fused["fused_mask"].tolist() == [True, True, True, False]
    assert fused["agreed_face_ids"] == [0]
    # 几何报了、模型没报：这是分歧面，要标出来给人看，但不能删。
    assert fused["geometry_only_face_ids"] == [1]
    # 模型新增：几何没报、模型报了。
    assert fused["added_by_model_face_ids"] == [2]
    assert fused["added_by_model_count"] == 1


def test我方融合拒绝形状不一致的入参() -> None:
    """不对齐就融合等于拿别人的结论改我的面，而且错得悄无声息。"""

    with pytest.raises(ValueError, match="形状一致"):
        fuse_geometry_with_model(
            np.array([True, False]), np.array([0.1, 0.2, 0.3])
        )


def test我方训练拒绝空样本与非法特征() -> None:
    features, labels = _separable_case()

    with pytest.raises(ValueError, match="训练样本至少有一个面"):
        train_face_classifier(features[:0], labels[:0])
    bad = features.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="有限数"):
        train_face_classifier(bad, labels)
    with pytest.raises(ValueError, match="行数与 labels 一致"):
        train_face_classifier(features, labels[:2])


def test我方模型存盘后能原样加载(tmp_path) -> None:
    """服务与批量任务要复用同一份模型，加载后输出必须逐位一致。"""

    features, labels = _separable_case()
    model = train_face_classifier(
        features, labels, FaceClassifierConfig(epochs=20)
    )
    path = tmp_path / "model.pt"

    save_face_classifier(model, path)
    reloaded = load_face_classifier(path, input_dim=features.shape[1])

    assert path.exists()
    assert np.allclose(
        predict_face_probability(model, features),
        predict_face_probability(reloaded, features),
    )


def test我方配置可序列化() -> None:
    payload = FaceClassifierConfig().to_dict()

    assert payload["epochs"] == 60
    assert payload["fusion_probability"] == 0.5
    assert payload["mark_disagreement_for_review"] is True
