"""我把合成缺陷、训练特征和可选深度学习模型放在独立区域。

这个包解决的是"没有真值怎么评估"的问题，三个能力互相咬合：

- ``synthetic_defects``：自造真值。拿真实水密网格按共享边邻接删掉已知面积的连续面，
  造出的孔洞真值已知，可直接当标签用；也支持复制一块面沿法向外推，
  造出"多出的料"真值——生产管线对每件真实件都会报 extra 区域，
  没有 extra 真值就等于这一类结论零证据；
- ``features``：把检测结果转成归一化特征与标签，保证训练目标和线上判定同口径；
- ``evaluation``：在合成用例上算指标；
- ``annotations``：真实标注的严格 JSON 契约，在训练前拦住数据泄漏。
- ``face_classifier``：面级深度学习增强的**训练与融合**（可选依赖 torch）。
  它是增强而不是替换：几何判据是甲方判定合格与否的硬判据，模型只用来
  补上几何漏掉的面并标出待复核处，**不删除几何已判定的区域**——
  在质检场景里漏报比多报危险得多。

必须分清两类"真值"：合成真值能证明算法在已知孔洞上能不能找回来，真实标注才能证明
在甲方实际零件上准不准。两者不能互相替代，报告里也必须分开写。
"""

from .features import build_face_features
from .evaluation import (
    DetectionMetrics,
    SpecTruth,
    evaluate_synthetic_case,
    evaluate_synthetic_suite,
    threshold_reached_face_ids,
)
from .annotations import load_annotation_manifest, validate_annotation_manifest
from .annotation_scoring import collect_predictions, score_annotations
from .annotation_review import (
    apply_review_sheet,
    export_review_sheet,
    manifest_fingerprint,
)
from .synthetic_defects import add_face_region, remove_face_region
from .face_classifier import (
    FaceClassifierConfig,
    fuse_geometry_with_model,
    load_face_classifier,
    predict_face_probability,
    save_face_classifier,
    train_face_classifier,
)

__all__ = [
    "add_face_region",
    "apply_review_sheet",
    "collect_predictions",
    "build_face_features",
    "DetectionMetrics",
    "SpecTruth",
    "evaluate_synthetic_case",
    "evaluate_synthetic_suite",
    "threshold_reached_face_ids",
    "export_review_sheet",
    "FaceClassifierConfig",
    "fuse_geometry_with_model",
    "load_annotation_manifest",
    "load_face_classifier",
    "manifest_fingerprint",
    "predict_face_probability",
    "remove_face_region",
    "save_face_classifier",
    "score_annotations",
    "train_face_classifier",
    "validate_annotation_manifest",
]
