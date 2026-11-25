"""Synthetic forward-camera driving clips with three ALIGNED sensors (RGB, thermal, LiDAR depth) in six weathers.

Stand-in for DSERT-RoLL, the dataset used by arXiv 2611.22669 ("Beyond the Camera: A Tri-Modal Driving World Model
for Future RGB, Thermal, and LiDAR Forecasting in Adverse Weather").  Nothing here comes from the paper beyond the idea:
three co-registered sensors, LiDAR projected into the RGB camera as a depth video, six weather conditions.

    camera      pinhole, 48 x 64 px, f = 40 px, principal point (32, 18), mounted 1.5 m above a flat road
    world       ego drives +Z at 4-10 m/s; ego lane X in [-1.75, 1.75], a same-direction lane at X = 3.5, two
                oncoming lanes at X = -3.5 / -7; dashed lane paint scrolls with ego travel
    agents      1-2 lead cars (constant accel), 1-3 oncoming cars, 0-2 pedestrians (some cross), 4-6 roadside poles
    sensors     RGB (3 ch)      colour, weather-dependent lighting, fog/snow veiling, rain streaks, snow flakes, noise
                thermal (1 ch)  temperature-like: hot engines / people, cold sky; barely affected by light or fog
                LiDAR (2 ch)    16 beams (image rows) x 64 columns: measured depth [m] + validity; weather cuts the max
                                range, drops returns and (snow) adds spurious near hits from flakes
    timing      dt = 0.25 s, 6 context frames (1.5 s) + 8 future frames (2.0 s)

Weathers (index -> name): 0 clear, 1 dusk, 2 night, 3 rain, 4 fog, 5 snow.
"""
from __future__ import annotations

import numpy as np

H, W = 48, 64
F_PX, CX, CY, CAM_H = 40.0, 32.0, 18.0, 1.5
DT = 0.25
T_CTX, T_FUT = 6, 8
T_ALL = T_CTX + T_FUT
N_BEAMS = 16
BEAM_ROWS = np.round(np.linspace(14, 47, N_BEAMS)).astype(int)
D_MAX = 60.0                                   # depth normalisation [m]
WEATHERS = ["clear", "dusk", "night", "rain", "fog", "snow"]
MAX_OBJ = 12
CAR, PED, POLE = 0, 1, 2

# LiDAR behaviour per weather: max range [m], random dropout, spurious near-return rate
LIDAR = {0: (60, 0.02, 0.0), 1: (60, 0.02, 0.0), 2: (60, 0.02, 0.0), 3: (45, 0.15, 0.0), 4: (25, 0.10, 0.0),
         5: (35, 0.20, 0.04)}

_v = np.arange(H)[:, None].astype(float)
_u = np.arange(W)[None, :].astype(float)
GROUND = np.broadcast_to(_v > CY + 0.5, (H, W)).copy()                             # (H, W) below the horizon
Z_GROUND = np.where(GROUND, F_PX * CAM_H / np.maximum(_v - CY, 0.5), np.inf)        # (H, W) ground depth [m]
X_GROUND = np.where(GROUND, (_u - CX) * np.where(GROUND, Z_GROUND, 0.0) / F_PX, 0.0)


def spawn(rng: np.random.Generator) -> tuple[float, list[dict]]:
    ve = rng.uniform(4, 10)
    obj = []
    for _ in range(int(rng.integers(1, 3))):                                         # lead / same-direction cars
        obj.append(dict(c=CAR, X=float(rng.choice([0.0, 3.5])) + rng.normal(0, 0.15), Z=rng.uniform(9, 40),
                        vz=rng.uniform(-4, 3), az=rng.uniform(-2.0, 1.0), vx=0.0, w=1.8, h=1.5, rear=True,
                        col=rng.uniform(0.15, 0.95, 3)))
    for _ in range(int(rng.integers(1, 4))):                                         # oncoming cars
        obj.append(dict(c=CAR, X=float(rng.choice([-3.5, -7.0])) + rng.normal(0, 0.15), Z=rng.uniform(28, 75),
                        vz=-(ve + rng.uniform(6, 12)), az=0.0, vx=0.0, w=1.8, h=1.5, rear=False,
                        col=rng.uniform(0.15, 0.95, 3)))
    for _ in range(int(rng.integers(0, 3))):                                         # pedestrians
        side = rng.choice([-1, 1])
        cross = rng.random() < 0.6
        obj.append(dict(c=PED, X=side * rng.uniform(6.0, 9.5), Z=rng.uniform(10, 32), vz=-ve, az=0.0,
                        vx=-side * rng.uniform(1.0, 1.7) if cross else 0.0, w=0.55, h=1.7, rear=False,
                        col=rng.uniform(0.05, 0.4, 3)))
    z0 = rng.uniform(2, 12)
    for k in range(int(rng.integers(4, 7))):                                         # roadside poles
        obj.append(dict(c=POLE, X=float(rng.choice([-1, 1])) * rng.uniform(7, 9), Z=z0 + k * rng.uniform(9, 14),
                        vz=-ve, az=0.0, vx=0.0, w=0.25, h=4.0, rear=False, col=np.array([0.55, 0.55, 0.5])))
    return ve, obj[:MAX_OBJ]


def project(o: dict) -> tuple[int, int, int, int] | None:
    """World box -> pixel box (u0, v0, u1, v1), inclusive-exclusive; None if behind / outside."""
    if o["Z"] < 1.5:
        return None
    u0 = CX + F_PX * (o["X"] - o["w"] / 2) / o["Z"]
    u1 = CX + F_PX * (o["X"] + o["w"] / 2) / o["Z"]
    vb = CY + F_PX * CAM_H / o["Z"]
    vt = CY + F_PX * (CAM_H - o["h"]) / o["Z"]
    b = (int(np.floor(u0)), int(np.floor(vt)), int(np.ceil(u1)), int(np.ceil(vb)))
    if b[2] <= 0 or b[0] >= W or b[3] <= 0 or b[1] >= H:
        return None
    return max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])


def render(objs: list[dict], s_ego: float, wx: int, rng: np.random.Generator):
    """One time step -> rgb (H, W, 3) [0,1], thermal (H, W) [0,1], depth (H, W) [m], boxes list."""
    Z = Z_GROUND.copy()
    X = X_GROUND
    # ---------------- ground classes
    road = GROUND & (X > -8.75) & (X < 5.25)
    zw = np.where(GROUND, Z_GROUND, 0.0) + s_ego
    dash = (np.mod(zw, 6.0) < 3.0)
    lw = 0.13
    paint = road & ((np.abs(X + 1.75) < lw) | (np.abs(X + 8.6) < lw) | (np.abs(X - 5.1) < lw)
                    | (dash & ((np.abs(X - 1.75) < lw) | (np.abs(X + 5.25) < lw))))
    yellow = road & (np.abs(X + 1.75) < lw)
    rgb = np.zeros((H, W, 3))
    th = np.zeros((H, W))
    sky_rgb = {0: (0.55, 0.72, 0.92), 1: (0.85, 0.55, 0.35), 2: (0.03, 0.03, 0.08), 3: (0.45, 0.48, 0.52),
               4: (0.7, 0.7, 0.72), 5: (0.78, 0.8, 0.84)}[wx]
    grass_rgb = (0.92, 0.93, 0.95) if wx == 5 else (0.25, 0.45, 0.2)
    road_rgb = (0.75, 0.76, 0.78) if wx == 5 else (0.33, 0.33, 0.35)
    rgb[:] = sky_rgb
    rgb[GROUND] = grass_rgb
    rgb[road] = road_rgb
    rgb[paint] = (0.95, 0.95, 0.95) if wx != 5 else (0.85, 0.85, 0.85)
    rgb[yellow] = (0.9, 0.75, 0.1)
    amb = {0: 0.0, 1: -0.03, 2: -0.08, 3: -0.06, 4: -0.04, 5: -0.15}[wx]
    th[:] = 0.08
    th[GROUND] = 0.32 + amb
    th[road] = 0.42 + amb
    th[paint] = 0.40 + amb
    # ---------------- objects, painter's algorithm far -> near
    boxes = []
    lights = []
    for o in sorted(objs, key=lambda q: -q["Z"]):
        b = project(o)
        if b is None:
            boxes.append(None)
            continue
        u0, v0, u1, v1 = b
        sub = Z[v0:v1, u0:u1]
        m = o["Z"] < sub
        sub[m] = o["Z"]
        rgb[v0:v1, u0:u1][m] = o["col"]
        if o["c"] == CAR:
            th[v0:v1, u0:u1][m] = 0.62 + amb
            hh = max(1, (v1 - v0) // 3)                                                 # engine / exhaust band
            th[v1 - hh:v1, u0:u1][m[-hh:]] = 0.92 + amb
            lights.append((u0, u1, v1 - hh, v1, o["rear"]))
        elif o["c"] == PED:
            th[v0:v1, u0:u1][m] = 0.86 + amb
        else:
            th[v0:v1, u0:u1][m] = 0.36 + amb
        boxes.append((o, b))
    # ---------------- lighting / weather on RGB
    if wx == 1:
        rgb = rgb * np.array([1.0, 0.75, 0.6]) * 0.75
    elif wx == 2:
        rgb = rgb * 0.12
        near = GROUND & (Z < 18) & (np.abs(X) < 3.5)                                    # ego headlight pool
        rgb[near] += 0.25 * np.exp(-Z[near] / 9)[:, None]
    elif wx == 3:
        rgb = rgb * 0.7
    for (u0, u1, va, vb, rear) in lights:                                               # tail / head lights
        col = np.array([0.9, 0.1, 0.1]) if rear else np.array([1.0, 1.0, 0.85])
        k = 1.0 if wx in (2, 1, 3, 4, 5) else 0.4
        for uu in (u0, max(u0, u1 - 1)):
            rgb[va:vb, uu:uu + 1] = np.clip(rgb[va:vb, uu:uu + 1] * (1 - k) + col * k, 0, 1)
    Zf = np.where(np.isfinite(Z), Z, 200.0)
    if wx == 4:
        a = 1 - np.exp(-Zf / 14.0)
        rgb = rgb * (1 - a[..., None]) + 0.72 * a[..., None]
        th = th * np.exp(-Zf / 90.0) + 0.3 * (1 - np.exp(-Zf / 90.0))
    elif wx == 5:
        a = 1 - np.exp(-Zf / 35.0)
        rgb = rgb * (1 - a[..., None]) + 0.82 * a[..., None]
        fl = rng.random((H, W)) < 0.05
        rgb[fl] = 0.97
    elif wx == 3:
        a = 1 - np.exp(-Zf / 45.0)
        rgb = rgb * (1 - a[..., None]) + 0.45 * a[..., None]
        for _ in range(14):                                                             # rain streaks
            uu, vv = rng.integers(0, W), rng.integers(0, H - 4)
            rgb[vv:vv + 4, uu] = rgb[vv:vv + 4, uu] * 0.5 + 0.35
        th = 0.85 * th + 0.15 * 0.3
    rgb = np.clip(rgb + rng.normal(0, 0.02 if wx != 2 else 0.03, rgb.shape), 0, 1)
    th = np.clip(th + rng.normal(0, 0.015, th.shape), 0, 1)
    # thermal camera is lower resolution: 2x2 box blur
    th = 0.25 * (th + np.roll(th, 1, 0) + np.roll(th, 1, 1) + np.roll(np.roll(th, 1, 0), 1, 1))
    return rgb, th, Z, boxes


def lidar(Z: np.ndarray, wx: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Dense depth -> sparse measured depth (H, W) [m] and validity (H, W) on the 16 beam rows."""
    rmax, drop, spur = LIDAR[wx]
    d = np.zeros((H, W))
    v = np.zeros((H, W), bool)
    zr = Z[BEAM_ROWS]
    ok = np.isfinite(zr) & (zr < rmax) & (rng.random(zr.shape) > drop)
    meas = zr + rng.normal(0, 0.05, zr.shape)
    if spur > 0:
        s = rng.random(zr.shape) < spur
        meas = np.where(s, rng.uniform(1.0, 8.0, zr.shape), meas)
        ok = ok | s
    d[BEAM_ROWS] = np.where(ok, meas, 0.0)
    v[BEAM_ROWS] = ok
    return d, v


def make_clip(rng: np.random.Generator, wx: int) -> dict:
    ve, objs = spawn(rng)
    s = 0.0
    out = dict(rgb=np.zeros((T_ALL, H, W, 3), np.uint8), th=np.zeros((T_ALL, H, W), np.uint8),
               depth=np.zeros((T_ALL, H, W), np.uint16), valid=np.zeros((T_ALL, H, W), bool),
               dense=np.zeros((T_ALL, H, W), np.uint16), boxes=np.full((T_ALL, MAX_OBJ, 4), -1, np.int16),
               cls=np.full(MAX_OBJ, -1, np.int8))
    for i, o in enumerate(objs):
        o["id"] = i
        out["cls"][i] = o["c"]
    for t in range(T_ALL):
        rgb, th, Z, boxes = render(objs, s, wx, rng)
        d, v = lidar(Z, wx, rng)
        out["rgb"][t] = np.round(rgb * 255)
        out["th"][t] = np.round(th * 255)
        out["depth"][t] = np.round(np.clip(d, 0, 650) * 100)
        out["valid"][t] = v
        out["dense"][t] = np.round(np.clip(np.where(np.isfinite(Z), Z, 0), 0, 650) * 100)
        for item in boxes:
            if item is not None:
                out["boxes"][t, item[0]["id"]] = item[1]
        for o in objs:                                                                   # advance the world
            o["Z"] += o["vz"] * DT + 0.5 * o["az"] * DT ** 2
            o["vz"] += o["az"] * DT
            o["X"] += o["vx"] * DT
            if o["c"] == POLE and o["Z"] < 1.0:
                o["Z"] += 62.0
        s += ve * DT
    out["ve"] = np.float32(ve)
    return out


def make_dataset(n_per_weather: int, seed: int, keep_boxes: bool = False) -> dict:
    """Returns uint8/uint16 arrays: rgb (N, T, H, W, 3), th (N, T, H, W), depth cm (N, T, H, W), valid, weather (N,)."""
    rng = np.random.default_rng(seed)
    clips = []
    wxs = []
    for wx in range(len(WEATHERS)):
        for _ in range(n_per_weather):
            clips.append(make_clip(rng, wx))
            wxs.append(wx)
    keys = ["rgb", "th", "depth", "valid"] + (["boxes", "cls", "dense", "ve"] if keep_boxes else [])
    out = {k: np.stack([c[k] for c in clips]) for k in keys}
    out["weather"] = np.array(wxs, np.int64)
    return out


# ------------------------------------------------------------------------------------------------ tensor packing
def to_tiles(rgb: np.ndarray, th: np.ndarray, depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Raw uint arrays (..., H, W[, 3]) -> three 3-channel tiles (..., 3, 3, H, W) float32 in [0, 1].

    tile 0 RGB; tile 1 thermal replicated x3 (a grey video); tile 2 LiDAR depth video: [d/60, d/60, valid].
    Every sensor becomes a 3-channel video so one video model can consume any of them, or all three side by side.
    """
    r = rgb.astype(np.float32) / 255.0
    r = np.moveaxis(r, -1, -3)
    t = (th.astype(np.float32) / 255.0)[..., None, :, :].repeat(3, -3)
    d = np.clip(depth.astype(np.float32) / 100.0 / D_MAX, 0, 1)
    v = valid.astype(np.float32)
    l = np.stack([d, d, v], -3)
    return np.stack([r, t, l], -4)


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = make_dataset(10, 0, keep_boxes=True)
    print("60 clips in %.2fs" % (time.time() - t0), {k: v.shape for k, v in d.items()})
    for wx, nm in enumerate(WEATHERS):
        m = d["weather"] == wx
        print(f"{nm:6s} lidar returns/frame {d['valid'][m].sum((2, 3)).mean():6.1f}  rgb mean {d['rgb'][m].mean():5.1f}"
              f"  thermal mean {d['th'][m].mean():5.1f}")
