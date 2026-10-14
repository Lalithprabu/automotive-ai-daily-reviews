"""CoCam4D (arXiv:2610.11577) — DOP-level reconstruction. See SOURCING.md."""
from .model import MotionUncertaintyNet, ModelConfig
from .fusion import (DOP, compensate_latency, associate, information_fuse,
                     fuse_agents, pack_dop, unpack_dop, DOP_BYTES, CHI2_3_95)
__all__ = ["MotionUncertaintyNet", "ModelConfig", "DOP", "compensate_latency",
           "associate", "information_fuse", "fuse_agents", "pack_dop",
           "unpack_dop", "DOP_BYTES", "CHI2_3_95"]
