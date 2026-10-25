"""我集中管理候选模型与参考模型之间的刚体配准。"""

from .rigid import align_mesh, apply_registration

__all__ = ["align_mesh", "apply_registration"]
