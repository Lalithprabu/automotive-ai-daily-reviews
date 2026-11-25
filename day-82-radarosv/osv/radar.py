"""Two simulated mmWave radars with complementary strengths (my construction, following the RadarOSV abstract).

    CASCADED IMAGING RADAR   range 64 bins x 0.4 m, azimuth 64 bins over +-60 deg (1.875 deg), Doppler 16 bins
                             over [-2.5, 2.5) m/s  -> fine geometry, but a SMALL unambiguous velocity range:
                             anything faster than +-2.5 m/s relative to the radar folds back (aliases).
    SINGLE-CHIP RADAR        range 32 bins x 0.8 m, azimuth 16 bins over +-60 deg (7.5 deg, extra blur), Doppler 32
                             bins over [-24, 24) m/s (1.5 m/s) -> coarse and sparse (40% of returns dropped), but a
                             WIDE unambiguous range: no aliasing for anything in these scenes.

Both produce raw power spectra in POLAR coordinates (Doppler, range, azimuth). Nothing is converted to Cartesian here:
that is the job of the network's polar-to-Cartesian geometry (PCG) block.

Returns are sampled on the radar-facing edges of each box. Pedestrians, cyclists and car wheels add micro-Doppler
returns (limb / wheel motion spread around the body velocity), which is the class evidence the wide-range radar sees.
No occlusion or multipath is modelled.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .scene import DYN, FOV, R_MAX, box_corners, make_scene, rasterize

# imaging radar
RI, AI, DI, VI = 64, 64, 16, 2.5          # bins, bins, bins, unambiguous half-range (m/s)
# single-chip radar
RS, AS, DS, VS = 32, 16, 32, 24.0
RCS = {1: 0.6, 2: 1.0, 3: 1.4, 4: 0.35, 5: 0.2}
MICRO = {2: (4, 0.4), 4: (10, 1.0), 5: (14, 1.5)}   # class -> (n extra returns, micro-Doppler sigma m/s)


def wrap(v, vmax=VI):
    """Fold a velocity into the unambiguous interval [-vmax, vmax)."""
    return (np.asarray(v) + vmax) % (2 * vmax) - vmax


def sample_returns(scene: dict, rng: np.random.Generator) -> np.ndarray:
    """-> (P, 4) array of [range, azimuth, relative radial velocity, amplitude]."""
    pts, vel, amp = [], [], []
    for o in scene["objs"]:
        c = box_corners(o)
        ctr = np.array([o["cx"], o["cy"]])
        if np.hypot(*ctr) - max(o["L"], o["W"]) > R_MAX + 2:
            continue
        for k in range(4):
            a, b = c[k], c[(k + 1) % 4]
            mid = (a + b) / 2
            nrm = mid - ctr                                                   # outward normal (box is convex)
            if nrm @ mid >= 0:                                                # edge faces away from the radar
                continue
            m = max(int(np.linalg.norm(b - a) / 0.25), 1)
            t = (np.arange(m) + rng.random(m)) / m
            pts.append(a + (b - a) * t[:, None])
            vel.append(np.tile([o["vx"], o["vy"]], (m, 1)))
            amp.append(RCS[o["cls"]] * rng.exponential(1.0, m))
        if o["cls"] in MICRO:                                                 # limbs / pedals / wheels
            n, sd = MICRO[o["cls"]]
            h = np.array([np.cos(o["yaw"]), np.sin(o["yaw"])])
            p = ctr + rng.uniform(-0.5, 0.5, (n, 2)) * [o["L"], o["W"]] @ np.array([[h[0], h[1]], [-h[1], h[0]]])
            pts.append(p)
            vel.append(np.array([o["vx"], o["vy"]]) + rng.normal(0, sd, (n, 1)) * h)
            amp.append(RCS[o["cls"]] * 0.6 * rng.exponential(1.0, n))
    p, v, a = np.concatenate(pts), np.concatenate(vel), np.concatenate(amp)
    r, th = np.hypot(p[:, 0], p[:, 1]), np.arctan2(p[:, 1], p[:, 0])
    u = p / np.maximum(r[:, None], 1e-6)
    vrel = ((v - [scene["v_ego"], 0.0]) * u).sum(1)
    keep = (r < R_MAX) & (np.abs(th) < FOV) & (r > 0.5)
    return np.stack([r, th, vrel, a], 1)[keep]


def _cube(ret, nr, na, nd, dr, vmax, sig, noise, rng, ang_noise, rng_noise, fold):
    r = ret[:, 0] + rng.normal(0, rng_noise, len(ret))
    th = ret[:, 1] + rng.normal(0, ang_noise, len(ret))
    v = wrap(ret[:, 2], vmax) if fold else ret[:, 2]
    ri = np.clip((r / dr).astype(int), 0, nr - 1)
    ai = np.clip(((th + FOV) / (2 * FOV) * na).astype(int), 0, na - 1)
    di = np.clip(((v + vmax) / (2 * vmax) * nd).astype(int), 0, nd - 1)
    cube = np.zeros((nd, nr, na), np.float32)
    np.add.at(cube, (di, ri, ai), ret[:, 3])
    cube = gaussian_filter(cube, sig, mode=("wrap" if fold else "nearest", "nearest", "nearest"))
    cube += rng.exponential(noise, cube.shape).astype(np.float32)
    return (np.log10(cube + 0.01) + 2.0).astype(np.float16)                   # log power, ~[0, 2.5]


def render(scene: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """-> imaging cube (16, 64, 64) and single-chip cube (32, 32, 16), both (Doppler, range, azimuth)."""
    ret = sample_returns(scene, rng)
    img = _cube(ret, RI, AI, DI, R_MAX / RI, VI, (0.6, 0.6, 0.7), 0.01, rng, np.deg2rad(0.4), 0.05, fold=True)
    sc_ret = ret[rng.random(len(ret)) > 0.4]
    sc = _cube(sc_ret, RS, AS, DS, R_MAX / RS, VS, (0.7, 0.7, 1.2), 0.03, rng, np.deg2rad(3.0), 0.15, fold=False)
    return img, sc


def make_sample(scene, rng):
    img, sc = render(scene, rng)
    gt = rasterize(scene)
    return dict(img=img, sc=sc, sem=gt["sem"], vgr=gt["vgr"].astype(np.float16),
                vrel=gt["vrel"].astype(np.float16), vego=np.float32(scene["v_ego"]))


def _worker(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    out = [make_sample(make_scene(rng), rng) for _ in range(n)]
    return {k: np.stack([o[k] for o in out]) for k in out[0]}


def make_dataset(n: int, seed: int, workers: int = 2, chunk: int = 250) -> dict:
    """Generated in small chunks written into preallocated arrays (keeps peak memory near the final size)."""
    from multiprocessing import Pool
    sizes = [min(chunk, n - i) for i in range(0, n, chunk)]
    out, pos = None, 0
    with Pool(workers) as p:
        for part in p.imap(_worker, [(s, seed * 100000 + i) for i, s in enumerate(sizes)]):
            if out is None:
                out = {k: np.empty((n,) + v.shape[1:], v.dtype) for k, v in part.items()}
            m = len(part["sem"])
            for k, v in part.items():
                out[k][pos:pos + m] = v
            pos += m
    return out


if __name__ == "__main__":
    import time
    t = time.time()
    d = _worker((50, 0))
    print({k: (v.shape, v.dtype) for k, v in d.items()}, f"{(time.time() - t) / 50 * 1000:.1f} ms/scene")
    dyn = np.isin(d["sem"], DYN)
    v = d["vrel"][dyn].astype(np.float32)
    print("dynamic cells: share aliased in the imaging radar (|v_rel| > 2.5):", round(float((np.abs(v) > VI).mean()), 3))
