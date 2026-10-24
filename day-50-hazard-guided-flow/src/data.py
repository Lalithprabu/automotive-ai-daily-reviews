import numpy as np, torch
from torch.utils.data import TensorDataset
from src.scene import demo_trajectory, sample_scene, make_context, encode

def build_dataset(n, hazard_frac, seed):
    rng = np.random.default_rng(seed)
    trajs, vh_l, d_l, p_l = [], [], [], []
    for _ in range(n):
        hz = rng.random() < hazard_frac
        v0, vh, d = sample_scene(rng, hz)
        trajs.append(demo_trajectory(v0, d, hz, rng)); vh_l.append(vh); d_l.append(d); p_l.append(hz)
    traj = torch.from_numpy(np.stack(trajs)); vh = torch.from_numpy(np.stack(vh_l))
    d = torch.tensor(d_l, dtype=torch.float32); p = torch.tensor(p_l)
    return TensorDataset(encode(traj), make_context(vh, d, p)), dict(vh=vh, d=d, present=p)
