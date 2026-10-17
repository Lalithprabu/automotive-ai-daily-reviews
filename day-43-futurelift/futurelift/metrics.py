import torch


def forecast_metrics(traj, logits, gt, miss=2.0):
    """minADE_K, minFDE_K, MissRate_K (min endpoint error > 2 m), top-1 FDE (highest-score mode)."""
    err = (traj - gt[:, None]).norm(dim=-1)                  # (B,K,T)
    ade, fde = err.mean(-1), err[..., -1]
    top = logits.argmax(1)
    return dict(minADE=ade.min(1).values.mean().item(), minFDE=fde.min(1).values.mean().item(),
                MR=(fde.min(1).values > miss).float().mean().item(),
                top1FDE=fde[torch.arange(len(top)), top].mean().item())


def per_scene_minfde(traj, gt):
    return (traj - gt[:, None]).norm(dim=-1)[..., -1].min(1).values
