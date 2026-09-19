"""
Synthetic paired ego/collaborator BEV scene generator.

This is a reconstruction-only synthetic proxy task (the real LDE paper's
dataset was not recoverable -- see README "Sourcing note"). It is
deliberately simple but keeps the structural properties that matter for
testing the architecture end to end:

  - a shared BEV grid (H x W cells), ~6 true objects per scene out of
    H*W cells (e.g. 6 / 1024 = 0.6% positive, ~99.4% background) --
    matching the class-imbalance regime bug #2 addresses;
  - the ego only has real sensor signal for objects within its own
    sensing range (occlusion proxy) -- objects further out are pure noise
    in the ego's own BEV tensor, i.e. genuinely undetectable without help;
  - the collaborator sits at a random integer-cell offset (dx, dy) from
    the ego and has signal for objects within ITS OWN range, in ITS OWN
    local frame -- FoVAligner must warp this into the ego frame before
    any comparison is meaningful.
"""

import torch


def _stamp(bev: torch.Tensor, y: int, x: int, scale: float, g) -> None:
    """Write a small graded 'object' signal centered at (y, x): full scale
    at the center cell, decaying outward over a 5x5 block. Neighboring
    cells are NOT given positive labels (only the exact center is a true
    object) -- they exist so the teacher's confidence has a graded
    penumbra around real objects, which is what the curriculum threshold
    progressively reveals as it relaxes."""
    C, H, W = bev.shape
    center_pattern = scale * (0.5 + torch.rand(C, generator=g))
    rings = {1: 0.55, 2: 0.22}
    for oy in range(y - 2, y + 3):
        for ox in range(x - 2, x + 3):
            if not (0 <= oy < H and 0 <= ox < W):
                continue
            dist = max(abs(oy - y), abs(ox - x))  # chebyshev ring
            if dist == 0:
                bev[:, oy, ox] += center_pattern
            elif dist in rings:
                bev[:, oy, ox] += rings[dist] * center_pattern


def generate_scene(
    H: int = 32,
    W: int = 32,
    C_in: int = 8,
    num_objects: int = 6,
    ego_range: float = 10.0,
    collab_range: float = 15.0,
    collab_offset_range=(4, 8),
    noise_std: float = 0.15,
    signal_scale: float = 3.0,
    generator: torch.Generator = None,
) -> dict:
    g = generator
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    obj_y = torch.randint(2, H - 2, (num_objects,), generator=g)
    obj_x = torch.randint(2, W - 2, (num_objects,), generator=g)

    dx = float(torch.randint(collab_offset_range[0], collab_offset_range[1] + 1, (1,), generator=g).item())
    dy = float(torch.randint(collab_offset_range[0], collab_offset_range[1] + 1, (1,), generator=g).item())
    if torch.rand(1, generator=g).item() < 0.5:
        dx = -dx
    if torch.rand(1, generator=g).item() < 0.5:
        dy = -dy

    ego_bev = noise_std * torch.randn(C_in, H, W, generator=g)
    collab_bev = noise_std * torch.randn(C_in, H, W, generator=g)
    ego_labels = torch.zeros(H, W)
    collab_labels = torch.zeros(H, W)
    world_labels = torch.zeros(H, W)

    for oy, ox in zip(obj_y.tolist(), obj_x.tolist()):
        world_labels[oy, ox] = 1.0

        if (oy - cy) ** 2 + (ox - cx) ** 2 <= ego_range ** 2:
            ego_labels[oy, ox] = 1.0
            _stamp(ego_bev, oy, ox, signal_scale, g)

        # collaborator's own local frame: world coords minus its offset
        cly, clx = int(round(oy - dy)), int(round(ox - dx))
        if 0 <= cly < H and 0 <= clx < W:
            if (cly - cy) ** 2 + (clx - cx) ** 2 <= collab_range ** 2:
                collab_labels[cly, clx] = 1.0
                _stamp(collab_bev, cly, clx, signal_scale, g)

    return {
        "ego_bev": ego_bev,
        "collab_bev": collab_bev,
        "ego_labels": ego_labels,
        "collab_labels": collab_labels,
        "world_labels": world_labels,
        "dx": dx,
        "dy": dy,
    }


def generate_batch(batch_size: int, generator: torch.Generator = None, **kwargs) -> dict:
    scenes = [generate_scene(generator=generator, **kwargs) for _ in range(batch_size)]
    out = {}
    for key in ["ego_bev", "collab_bev", "ego_labels", "collab_labels", "world_labels"]:
        out[key] = torch.stack([s[key] for s in scenes], dim=0)
    out["dx"] = torch.tensor([s["dx"] for s in scenes], dtype=torch.float32)
    out["dy"] = torch.tensor([s["dy"] for s in scenes], dtype=torch.float32)
    return out
