"""
Synthetic forward-looking mmWave radar world that produces raw Range-Azimuth-Doppler (RAD) cubes.

Why synthetic?  BRIDGE-Liv (the paper's dataset) is not something this repo ships.  The toy world keeps the
three physical effects RaMOD is built around:

  1. Ego motion makes STATIC structure light up in Doppler:   v_r(static) = -(v_ego + w x p) . p_hat
  2. Doppler WRAPPING: a radar measures v_r only modulo 2*v_max   v_meas = ((v_r + v_max) mod 2 v_max) - v_max
  3. Movers (pedestrian / cyclist / car) are sparse, weak returns mixed into that static clutter.

Conventions
-----------
ego / radar frame : x forward, y left (metres); the radar sits at the ego origin looking along +x.
azimuth bins      : uniform in sin(theta) over [-S_MAX, S_MAX]  (what an angle-FFT produces)
range bins        : uniform in [0, R_MAX)
Doppler bins      : uniform in [-V_MAX, V_MAX)  (radial velocity, positive = moving AWAY from the radar)
cube              : (R_BINS, A_BINS, D_BINS) log-power, roughly zero-mean / unit-std after normalisation
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ radar parameters (toy values)
R_BINS, A_BINS, D_BINS = 40, 40, 32
R_MAX = 28.0                     # metres
S_MAX = 0.866                    # sin(60 deg)
V_MAX = 4.0                      # max unambiguous radial velocity (m/s) -> 0.25 m/s per Doppler bin
DT = 0.1                         # frame period (s), 10 Hz
R_RES = R_MAX / R_BINS
S_RES = 2 * S_MAX / A_BINS
V_RES = 2 * V_MAX / D_BINS
R_CENTERS = (np.arange(R_BINS) + 0.5) * R_RES
S_CENTERS = -S_MAX + (np.arange(A_BINS) + 0.5) * S_RES
V_CENTERS = -V_MAX + (np.arange(D_BINS) + 0.5) * V_RES
N_WRAP = 5                       # wrap index k in {-2,-1,0,1,2}:  v_true = v_meas + 2 V_MAX k

CLASSES = ("pedestrian", "cyclist", "car")
# length, width (m), speed range (m/s), RCS-ish amplitude, micro-Doppler spread (m/s)
CLASS_SPEC = {
    0: dict(L=0.6, W=0.6, v=(0.6, 1.8), amp=(2.0, 3.5), micro=0.6),
    1: dict(L=1.8, W=0.6, v=(2.0, 6.0), amp=(3.5, 6.0), micro=0.35),
    2: dict(L=4.2, W=1.8, v=(2.0, 9.0), amp=(8.0, 14.0), micro=0.1),
}


def wrap_v(v: np.ndarray) -> np.ndarray:
    """Fold a radial velocity into the radar's unambiguous interval [-V_MAX, V_MAX)."""
    return np.mod(v + V_MAX, 2 * V_MAX) - V_MAX


# ------------------------------------------------------------------ episode simulation
def simulate_episode(rng: np.random.Generator, steps: int = 40, waves: int = 1) -> dict:
    """Ego drives down a street lined with walls, poles and parked cars; 1-4 movers share the street.

    waves > 1 (used by simulate.py for long demo clips) spawns extra groups of movers ahead of the ego
    every steps/waves frames, and extends the street to cover the distance driven.  The default
    (steps <= 40, waves = 1) path draws exactly the same random numbers as the cached training data.

    Returns world-frame trajectories (ego pose, mover states) and static scatterers.
    """
    # ---- ego: speed profile 0..7.5 m/s (wrapping kicks in above V_MAX), gentle yaw rate
    v0 = rng.uniform(0.0, 7.5)
    acc = rng.uniform(-0.8, 0.8)
    w0 = rng.uniform(-0.12, 0.12)
    ego = np.zeros((steps, 5))                     # x, y, yaw, v, yaw_rate
    x = y = yaw = 0.0
    v = v0
    for k in range(steps):
        w = w0 * np.cos(0.05 * k)
        ego[k] = (x, y, yaw, v, w)
        x += v * np.cos(yaw) * DT
        y += v * np.sin(yaw) * DT
        yaw += w * DT
        v = float(np.clip(v + acc * DT, 0.0, 8.0))

    # ---- static scatterers in the world frame: walls, poles, parked cars, ground speckle
    half = rng.uniform(4.5, 8.0)
    x_end = 70.0 if steps <= 40 else float(ego[-1, 0]) + 45.0
    xs = np.arange(-5, x_end, 0.5)
    pts, amps = [], []
    for side in (-1, 1):
        if rng.random() < 0.9:
            wall_y = side * (half + rng.uniform(0.0, 1.5))
            keep = rng.random(len(xs)) < 0.7
            pts.append(np.stack([xs[keep], np.full(keep.sum(), wall_y) + rng.normal(0, 0.1, keep.sum())], 1))
            amps.append(rng.uniform(1.5, 4.0, keep.sum()))
        for px in np.arange(rng.uniform(0, 6), x_end, rng.uniform(5, 9)):          # poles
            pts.append(np.array([[px, side * (half - 0.5)]]))
            amps.append(np.array([rng.uniform(5, 10)]))
        for px in np.arange(rng.uniform(0, 10), x_end, rng.uniform(9, 16)):        # parked cars (static!)
            if rng.random() < 0.5:
                cx, cy = px, side * (half - 1.6)
                g = np.stack(np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-0.8, 0.8, 3)), -1).reshape(-1, 2)
                pts.append(g + [cx, cy])
                amps.append(rng.uniform(3.0, 7.0, len(g)))
    n_sp = 120                                                                  # road-surface speckle
    n_sp = int(n_sp * (x_end + 5) / 75)
    pts.append(np.stack([rng.uniform(-5, x_end, n_sp), rng.uniform(-half, half, n_sp)], 1))
    amps.append(rng.uniform(0.3, 1.0, n_sp))
    static_xy, static_amp = np.concatenate(pts), np.concatenate(amps)

    # ---- movers: straight-line motion, along / against / across the street
    movers = []
    for _ in range(rng.integers(1, 5)):
        c = int(rng.choice(3, p=[0.4, 0.3, 0.3]))
        spec = CLASS_SPEC[c]
        speed = rng.uniform(*spec["v"])
        mode = rng.choice(["along", "against", "across"], p=[0.35, 0.35, 0.3] if c else [0.3, 0.3, 0.4])
        x0 = rng.uniform(4, 26) + (ego[0, 0])
        if mode == "across":
            y0 = rng.uniform(-half, half)
            hdg = np.pi / 2 * rng.choice([-1, 1])
        else:
            y0 = rng.uniform(-half + 1, half - 1)
            hdg = 0.0 if mode == "along" else np.pi
            if mode == "along":
                x0 = rng.uniform(4, 20)
        tr = np.zeros((steps, 4))                  # x, y, heading, speed
        for k in range(steps):
            tr[k] = (x0 + speed * np.cos(hdg) * k * DT, y0 + speed * np.sin(hdg) * k * DT, hdg, speed)
            if mode == "across" and abs(tr[k, 1]) > half:     # bounce at the kerb
                pass
        movers.append(dict(cls=c, traj=tr, amp=rng.uniform(*spec["amp"]),
                           pts=_box_points(spec["L"], spec["W"], rng), active=np.ones(steps, bool)))
    for wv in range(1, waves):                                                 # demo-only extra movers
        t0 = int(wv * steps / waves)
        ex = ego[t0, 0]
        for _ in range(rng.integers(2, 4)):
            c = int(rng.choice(3, p=[0.4, 0.3, 0.3]))
            spec = CLASS_SPEC[c]
            speed = rng.uniform(*spec["v"])
            mode = rng.choice(["along", "against", "across"])
            if mode == "across":
                x0, y0, hdg = ex + rng.uniform(8, 22), rng.uniform(-half, half), np.pi / 2 * rng.choice([-1, 1])
            else:
                x0, y0 = ex + rng.uniform(6, 26), rng.uniform(-half + 1, half - 1)
                hdg = 0.0 if mode == "along" else np.pi
            kk = np.arange(steps) - t0
            tr = np.stack([x0 + speed * np.cos(hdg) * kk * DT, y0 + speed * np.sin(hdg) * kk * DT,
                           np.full(steps, hdg), np.full(steps, speed)], 1)
            movers.append(dict(cls=c, traj=tr, amp=rng.uniform(*spec["amp"]),
                               pts=_box_points(spec["L"], spec["W"], rng), active=kk >= 0))
    return dict(ego=ego, static_xy=static_xy, static_amp=static_amp, movers=movers, steps=steps)


def _box_points(L: float, W: float, rng: np.random.Generator) -> np.ndarray:
    """Scattering centres on a box outline (object frame, x along heading)."""
    nl, nw = max(2, int(L / 0.4)), max(2, int(W / 0.4))
    a = np.linspace(-L / 2, L / 2, nl)
    b = np.linspace(-W / 2, W / 2, nw)
    p = np.concatenate([np.stack([a, np.full(nl, W / 2)], 1), np.stack([a, np.full(nl, -W / 2)], 1),
                        np.stack([np.full(nw, L / 2), b], 1), np.stack([np.full(nw, -L / 2), b], 1)])
    return p + rng.normal(0, 0.03, p.shape)


def to_ego(xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """World (N,2) -> ego frame of pose (x, y, yaw, ...)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = xy - pose[:2]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def frame_state(sim: dict, k: int) -> dict:
    """Everything the radar sees at step k, in the ego frame (positions + relative velocities)."""
    ego = sim["ego"][k]
    v_e, w_e, yaw = ego[3], ego[4], ego[2]
    # static points: velocity relative to the (moving, rotating) radar = -v_ego_vec - w x p
    sp = to_ego(sim["static_xy"], ego)
    s_vel = np.stack([-v_e + w_e * sp[:, 1], -w_e * sp[:, 0]], 1)
    scat_p, scat_v, scat_a, scat_micro = [sp], [s_vel], [sim["static_amp"]], [np.zeros(len(sp))]
    objs = []
    for m in sim["movers"]:
        if not m.get("active", np.ones(k + 1, bool))[k]:
            continue
        x, y, h, spd = m["traj"][k]
        c, s = np.cos(h), np.sin(h)
        wp = m["pts"] @ np.array([[c, s], [-s, c]]) + [x, y]                  # object points, world
        ep = to_ego(wp, ego)
        vw = np.array([spd * c, spd * s])                                       # ground velocity, world
        cy, sy = np.cos(yaw), np.sin(yaw)
        v_ego_frame = np.array([cy * vw[0] + sy * vw[1], -sy * vw[0] + cy * vw[1]])   # rotate into ego axes
        rel = v_ego_frame[None] + np.stack([-v_e + w_e * ep[:, 1], -w_e * ep[:, 0]], 1)
        scat_p.append(ep); scat_v.append(rel); scat_a.append(np.full(len(ep), m["amp"] / len(ep) * 4))
        scat_micro.append(np.full(len(ep), CLASS_SPEC[m["cls"]]["micro"]))
        ctr = to_ego(np.array([[x, y]]), ego)[0]
        rel_c = v_ego_frame + np.array([-v_e + w_e * ctr[1], -w_e * ctr[0]])
        r = np.linalg.norm(ctr)
        objs.append(dict(cls=m["cls"], center=ctr, pts=ep, r=r, s=ctr[1] / max(r, 1e-6),
                         v_rad=float(rel_c @ ctr / max(r, 1e-6)),            # true (unwrapped) radial vel.
                         v_fwd=float(v_ego_frame[0])))                         # ground velocity along ego x
    return dict(p=np.concatenate(scat_p), v=np.concatenate(scat_v), a=np.concatenate(scat_a),
                micro=np.concatenate(scat_micro), objs=objs, v_ego=v_e, w_ego=w_e)


# ------------------------------------------------------------------ cube rendering
def render_cube(st: dict, rng: np.random.Generator) -> np.ndarray:
    """Separable point-spread rendering of all scatterers into a log-power RAD cube (R, A, D) float32."""
    p, v, a = st["p"], st["v"], st["a"]
    r = np.linalg.norm(p, axis=1)
    s = p[:, 1] / np.maximum(r, 1e-6)
    vis = (r > 0.8) & (r < R_MAX) & (np.abs(s) < S_MAX) & (p[:, 0] > 0)
    p, v, a, r, s, micro = p[vis], v[vis], a[vis], r[vis], s[vis], st["micro"][vis]
    v_rad = (v * p).sum(1) / r
    v_meas = wrap_v(v_rad)
    amp = a * (8.0 / np.maximum(r, 2.0)) ** 2                                  # ~1/r^2 path loss (toy)
    # 1-D kernels (Gaussian main lobe + a weak sinc^2 side-lobe floor in azimuth)
    kr = np.exp(-0.5 * ((R_CENTERS[None] - r[:, None]) / (0.6 * R_RES)) ** 2)
    ds = (S_CENTERS[None] - s[:, None]) / S_RES
    ka = np.exp(-0.5 * (ds / 0.7) ** 2) + 0.02 * np.sinc(ds / 2.0) ** 2
    dv = np.mod(V_CENTERS[None] - v_meas[:, None] + V_MAX, 2 * V_MAX) - V_MAX  # circular Doppler distance
    sig = np.sqrt((0.6 * V_RES) ** 2 + micro[:, None] ** 2)
    kd = np.exp(-0.5 * (dv / sig) ** 2) * (0.6 * V_RES / sig)
    cube = np.einsum("n,nr,na,nd->rad", amp, kr, ka, kd, optimize=True)
    noise = rng.exponential(0.05, cube.shape)                                  # thermal-noise floor
    return np.log1p(cube / 0.05 + noise / 0.05).astype(np.float32)


def ra_box(o: dict) -> np.ndarray:
    """Axis-aligned box of an object's scattering points in continuous (range-bin, azimuth-bin) units."""
    r = np.linalg.norm(o["pts"], axis=1)
    s = o["pts"][:, 1] / np.maximum(r, 1e-6)
    rb, ab = r / R_RES, (s + S_MAX) / S_RES
    pad = 0.5
    return np.array([rb.min() - pad, ab.min() - pad, rb.max() + pad, ab.max() + pad], np.float32)


def labels(st: dict) -> list[dict]:
    """Visible movers -> detection / velocity labels."""
    out = []
    for o in st["objs"]:
        if not (1.5 < o["r"] < R_MAX - 1.0 and abs(o["s"]) < S_MAX - 0.03 and o["center"][0] > 0.5):
            continue
        v_meas = float(wrap_v(np.array(o["v_rad"])))
        k = int(np.round((o["v_rad"] - v_meas) / (2 * V_MAX)))
        out.append(dict(cls=o["cls"], box=ra_box(o), rb=o["r"] / R_RES, ab=(o["s"] + S_MAX) / S_RES,
                        v_rad=o["v_rad"], v_meas=v_meas, k=int(np.clip(k, -2, 2)), v_fwd=o["v_fwd"]))
    return out
