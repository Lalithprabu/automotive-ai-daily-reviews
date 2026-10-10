import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def voxel_iou(probs, gt, thr=0.5):
    """Per-horizon voxel IoU averaged over scenes. probs/gt [B,T1,V] -> [T1]."""
    pr = probs > thr
    g = gt > 0.5
    inter = (pr & g).sum(-1).float()
    union = (pr | g).sum(-1).float().clamp(min=1)
    return (inter / union).mean(0)


def _match(pred, gt, radius):
    """pred [P,2], gt [G,2] -> (#matched, sum of matched distances)."""
    if len(pred) == 0 or len(gt) == 0:
        return 0, 0.0
    d = np.linalg.norm(pred[:, None] - gt[None], axis=-1)
    r, c = linear_sum_assignment(d)
    ok = d[r, c] <= radius
    return int(ok.sum()), float(d[r, c][ok].sum())


def dense_instances(probs_t, cfg, thr=0.5):
    """Extract instance centers from a dense occupancy map [V] (connected components in BEV)."""
    w = cfg["world"]
    p = probs_t.reshape(w["grid_x"], w["grid_y"], w["grid_z"]).max(-1).values.numpy() > thr
    lab, n = ndimage.label(p)
    cs = []
    for i in range(1, n + 1):
        xs, ys = np.nonzero(lab == i)
        if len(xs) < 3:
            continue
        cs.append([w["x_min"] + (xs.mean() + 0.5) * w["cell"], w["y_min"] + (ys.mean() + 0.5) * w["cell"]])
    return np.array(cs).reshape(-1, 2)


def instance_metrics(pred_sets, batch, cfg, radius=2.0):
    """pred_sets[b][t] -> [P,2] predicted centers. Returns per-horizon (F1, ADE of matched) arrays [T1]."""
    w = cfg["world"]
    T1 = w["t_future"] + 1
    tp = np.zeros(T1); fp = np.zeros(T1); fn = np.zeros(T1); dist = np.zeros(T1)
    for b, sets in enumerate(pred_sets):
        for t in range(T1):
            g = batch["cen"][b, :, t].numpy()[batch["valid"][b].numpy()]
            inside = (g[:, 0] > w["x_min"]) & (g[:, 0] < w["x_min"] + w["grid_x"]) & (np.abs(g[:, 1]) < 16)
            g = g[inside]
            m, d = _match(sets[t], g, radius)
            tp[t] += m; fp[t] += len(sets[t]) - m; fn[t] += len(g) - m; dist[t] += d
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    ade = dist / np.maximum(tp, 1)
    return f1, ade


def lighTROcc_instances(out, thr=0.5):
    ex = out["exist"].detach().numpy() > thr
    cen = out["centers"].detach().numpy()[..., :2]                      # [B,T1,Q,2]
    return [[cen[b, t][ex[b]] for t in range(cen.shape[1])] for b in range(cen.shape[0])]
