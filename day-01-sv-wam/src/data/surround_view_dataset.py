"""
SurroundViewDataset -- adapter template for 6-camera driving clips.

This is intentionally a thin, well-documented adapter rather than a full
NAVSIMv2/nuScenes data loader: those datasets have their own devkits with
licensing and download steps outside this repo's scope. Point `clip_index`
at your own manifest (one entry per clip) and this class handles batching,
tensor shapes, and the synthetic fallback used by the tests.

Expected manifest entry (dict), one per clip:
    {
        "camera_paths": [str] * n_cams,   # image paths, one per camera, same timestep
        "trajectory": [[x, y, heading], ...] * n_action_steps,  # ground-truth waypoints
        "future_frames": Optional[[str] * n_cams] * n_future_steps,  # only needed for
                                                                       # training (video branch)
        "sdf_path": Optional[str],        # precomputed BEV signed-distance field, .npy
        "map_resolution": Optional[float],
        "map_origin": Optional[[float, float]],
    }
"""
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class SurroundViewDataset(Dataset):
    def __init__(
        self,
        clip_index: Optional[List[Dict]] = None,
        n_cams: int = 6,
        img_size=(224, 400),
        n_action_steps: int = 6,
        n_video_tokens: int = 64,
        embed_dim: int = 512,
        synthetic: bool = False,
        synthetic_length: int = 64,
    ):
        """
        clip_index: list of manifest entries (see module docstring). If None
                    and synthetic=True, the dataset generates random tensors
                    of the right shape -- useful for smoke-testing the
                    training loop before real data is wired up.

        n_video_tokens / embed_dim: must match the SVWAM model's config so the
                    synthetic "future_latents_target" lines up with what
                    model.video_head actually outputs (only used when
                    synthetic=True).
        """
        self.n_cams = n_cams
        self.img_size = img_size
        self.n_action_steps = n_action_steps
        self.n_video_tokens = n_video_tokens
        self.embed_dim = embed_dim
        self.synthetic = synthetic or clip_index is None
        self.clip_index = clip_index or []
        self.synthetic_length = synthetic_length

    def __len__(self) -> int:
        return self.synthetic_length if self.synthetic else len(self.clip_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.synthetic:
            return self._synthetic_sample()
        return self._load_sample(self.clip_index[idx])

    def _synthetic_sample(self) -> Dict[str, torch.Tensor]:
        h, w = self.img_size
        return {
            "images": torch.rand(self.n_cams, 3, h, w),
            "trajectory": torch.randn(self.n_action_steps, 3),
            # A flat random "future latent" target standing in for whatever
            # your video-tokenizer/VAE would encode the future frames into.
            # Shape must match model.video_head's output: [n_video_tokens, embed_dim].
            "future_latents_target": torch.randn(self.n_video_tokens, self.embed_dim),
        }

    def _load_sample(self, entry: Dict) -> Dict[str, torch.Tensor]:
        # TODO: replace with real image decoding (PIL/cv2) once camera_paths
        # point at your actual dataset. Left unimplemented deliberately --
        # do not depend on a specific dataset's directory layout here.
        raise NotImplementedError(
            "Wire this up to your own image-loading pipeline; "
            "the manifest format is documented in this file's module docstring."
        )
