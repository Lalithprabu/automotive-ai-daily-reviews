"""Synthetic driving scenes: BEV raster + expert waypoints. (This repo's own toy world, NOT NAVSIM.)"""
import numpy as np, torch
H = W = 32          # raster cells; x forward 0..40 m (rows), y lateral -10..10 m (cols)
X_MAX, Y_MAX = 40.0, 10.0
T = 8               # waypoints, dt = 0.5 s
DT = 0.5

def _xy_to_rc(x, y):
    r = np.clip((x / X_MAX * H).astype(int), 0, H - 1)
    c = np.clip(((y + Y_MAX) / (2 * Y_MAX) * W).astype(int), 0, W - 1)
    return r, c

def make_scene(rng):
    """Return raster (2,H,W): ch0 road band, ch1 lead vehicle; params; future raster; expert waypoints (T,2)."""
    k = rng.uniform(-0.02, 0.02)                 # road curvature: y_c(x) = k x^2 / 2 * 10
    lead = rng.random() < 0.6
    d = rng.uniform(8, 32)                        # lead gap (m)
    vl = rng.uniform(0, 6)                        # lead speed (m/s)
    xs = np.linspace(0, X_MAX, 200)
    yc = 5.0 * k * xs ** 2 / 10.0 * 10.0 / 10.0   # gentle curve
    road = np.zeros((H, W), np.float32)
    for off in np.linspace(-1.8, 1.8, 7):
        r, c = _xy_to_rc(xs, yc + off); road[r, c] = 1
    veh = np.zeros((H, W), np.float32); veh_f = np.zeros((H, W), np.float32)
    if lead:
        yl = 5.0 * k * d ** 2 / 10.0
        r, c = _xy_to_rc(np.array([d, d + 3.0]), np.array([yl, yl])); veh[r[0]:r[1] + 1, c[0]] = 1
        d2 = d + vl * 4.0
        yl2 = 5.0 * k * d2 ** 2 / 10.0
        r, c = _xy_to_rc(np.array([d2, d2 + 3.0]), np.array([yl2, yl2])); veh_f[r[0]:r[1] + 1, c[0]] = 1
    # expert: follow centreline; speed set by gap controller (IDM-like), cruise 10 m/s
    v, x = 6.0, 0.0
    wp = []
    gap = d if lead else 1e3
    xl = d
    for t in range(T):
        target = 10.0
        if lead:
            gap = xl - x - 4.0
            target = float(np.clip(vl + 0.5 * (gap - 6.0), 0.0, 10.0))
        v += np.clip(target - v, -3.0 * DT, 2.0 * DT)
        x += v * DT; xl += vl * DT
        wp.append([x, 5.0 * k * x ** 2 / 10.0])
    wp = np.array(wp, np.float32)
    return np.stack([road, veh]), np.stack([road, veh_f]), wp, (lead, d, k)

class Scenes(torch.utils.data.Dataset):
    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        rows = [make_scene(rng) for _ in range(n)]
        self.x = torch.tensor(np.stack([r[0] for r in rows]))
        self.xf = torch.tensor(np.stack([r[1] for r in rows]))
        self.wp = torch.tensor(np.stack([r[2] for r in rows]))
        self.meta = [r[3] for r in rows]
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.xf[i], self.wp[i]

WP_SCALE = torch.tensor([30.0, 5.0])   # normalisation (m)
def norm_wp(w): return w / WP_SCALE.to(w.device)
def denorm_wp(w): return w * WP_SCALE.to(w.device)
