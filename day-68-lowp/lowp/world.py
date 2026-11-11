"""
Synthetic driving world + multi-camera ray-cast renderer for the LOWP toy re-implementation.

SYNTHETIC: this is NOT nuScenes / NAVSIM / Bench2Drive and NOT the authors' code.
It exists so the planner has to *look* at camera pixels to know (a) how the road bends
and (b) where the lead vehicle is, which ego status alone cannot tell it.

World
-----
* A global road = polyline built from constant-curvature segments (straights, left/right arcs).
* Two lanes, each 3.6 m; the ego drives the right lane centre (lateral offset -1.8 m).
* Lead vehicles drive the ego lane at slower speeds; the ego follows them with an
  Intelligent Driver Model (IDM), so ground-truth futures contain braking / stopping.
* Ego frame: x forward, y left, z up (metres). Cameras: x right, y down, z forward (OpenCV).

Renderer
--------
Every pixel's viewing ray is intersected with (1) the obstacle boxes (slab method) and
(2) the ground plane z = 0. Ground colour depends on lateral distance to the road
centre-line (asphalt / edge lines / dashed centre line / grass); boxes are shaded by face.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

DT = 0.5            # waypoint spacing (s)
T_FUT = 6           # 6 waypoints -> 3.0 s horizon (nuScenes-style 1/2/3 s evaluation points)
SIM_DT = 0.1        # simulator step (s)
LANE_W = 3.6
EGO_LANE_OFF = -LANE_W / 2          # right-hand traffic: ego in the right lane
CAR_L, CAR_W, CAR_H = 4.5, 1.9, 1.5
CMD_STRAIGHT, CMD_LEFT, CMD_RIGHT = 0, 1, 2


# ----------------------------------------------------------------------------- cameras
@dataclass
class Camera:
    name: str
    K: np.ndarray          # (3,3) intrinsics, pixel units for an image of size (H, W)
    R: np.ndarray          # (3,3) camera->ego rotation
    t: np.ndarray          # (3,)  camera centre in ego frame


def make_rig(H: int = 40, W: int = 80) -> list[Camera]:
    """Four pinhole cameras at 1.6 m height: front (90 deg), front-left, front-right, back."""
    def rot(yaw_deg: float) -> np.ndarray:
        # base: camera z (optical axis) -> ego x, camera x (right) -> ego -y, camera y (down) -> ego -z
        base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
        a = math.radians(yaw_deg)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        # small downward pitch (5 deg) so the near road is visible
        p = math.radians(5.0)
        Rp = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        return Rz @ base @ Rp.T     # columns = camera axes expressed in ego frame

    def K_of(fov_deg: float) -> np.ndarray:
        f = (W / 2) / math.tan(math.radians(fov_deg) / 2)
        return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)

    t = np.array([1.5, 0.0, 1.6])
    return [
        Camera("front", K_of(90), rot(0), t.copy()),
        Camera("front_left", K_of(90), rot(70), t.copy()),
        Camera("front_right", K_of(90), rot(-70), t.copy()),
        Camera("back", K_of(110), rot(180), np.array([-1.0, 0.0, 1.6])),
    ]


# ----------------------------------------------------------------------------- road
def build_road(rng: np.random.Generator, n_seg: int = 14, ds: float = 0.5) -> np.ndarray:
    """Global centre-line polyline (N, 3) with columns (x, y, heading)."""
    x, y, h = 0.0, 0.0, 0.0
    pts = [(x, y, h)]
    for k in range(n_seg):
        length = rng.uniform(25, 60)
        kind = rng.choice(3, p=[0.4, 0.3, 0.3]) if k > 0 else 0
        kappa = 0.0 if kind == 0 else (1 if kind == 1 else -1) * rng.uniform(1 / 60, 1 / 18)
        for _ in range(int(length / ds)):
            h += kappa * ds
            x += ds * math.cos(h)
            y += ds * math.sin(h)
            pts.append((x, y, h))
    return np.asarray(pts)


@dataclass
class Episode:
    road: np.ndarray                 # (N, 3) centre-line
    tree: cKDTree = field(init=False)
    s_of: np.ndarray = field(init=False)

    def __post_init__(self):
        self.tree = cKDTree(self.road[:, :2])
        d = np.linalg.norm(np.diff(self.road[:, :2], axis=0), axis=1)
        self.s_of = np.concatenate([[0.0], np.cumsum(d)])

    def pose_at(self, s: float, lat: float = EGO_LANE_OFF) -> tuple[float, float, float]:
        """Global pose at arc length s, lateral offset lat (left positive)."""
        i = int(np.clip(np.searchsorted(self.s_of, s), 1, len(self.road) - 1))
        a = (s - self.s_of[i - 1]) / max(self.s_of[i] - self.s_of[i - 1], 1e-6)
        x, y, h = (1 - a) * self.road[i - 1] + a * self.road[i]
        return x - lat * math.sin(h), y + lat * math.cos(h), h


def simulate_episode(rng: np.random.Generator, steps: int = 400):
    """Roll out ego (IDM) + lead vehicles along a random road.

    Returns dict with ego arc-length/speed/accel per step and lead-vehicle arc lengths per step.
    """
    ep = Episode(build_road(rng))
    s_max = ep.s_of[-1] - 80
    n_lead = rng.integers(1, 4)
    lead_s = np.sort(rng.uniform(25, min(400, s_max), n_lead))
    lead_v = rng.uniform(0, 9, n_lead)                   # some are stopped (queue / stalled car)
    lead_stop_at = rng.uniform(0, steps * SIM_DT, n_lead)  # some brake to a stop mid-episode
    v0 = rng.uniform(8, 15)                              # desired speed
    s, v, a_prev = 0.0, rng.uniform(4, 12), 0.0
    ego_s, ego_v, ego_a, leads = [], [], [], []
    for k in range(steps):
        tsec = k * SIM_DT
        cur_lead_v = np.where(tsec > lead_stop_at, np.maximum(lead_v - 2.0 * (tsec - lead_stop_at), 0), lead_v)
        # IDM w.r.t. nearest lead ahead
        ahead = lead_s - s
        mask = ahead > 0
        acc_free = 1.5 * (1 - (v / v0) ** 4)
        if mask.any():
            j = np.argmin(np.where(mask, ahead, np.inf))
            gap = max(ahead[j] - CAR_L, 0.1)
            dv = v - cur_lead_v[j]
            s_star = 2.0 + v * 1.2 + v * dv / (2 * math.sqrt(1.5 * 3.0))
            acc = 1.5 * (1 - (v / v0) ** 4 - (max(s_star, 0) / gap) ** 2)
        else:
            acc = acc_free
        acc = float(np.clip(acc, -6.0, 2.0))
        ego_s.append(s); ego_v.append(v); ego_a.append(a_prev); leads.append(lead_s.copy())
        v = max(v + acc * SIM_DT, 0.0)
        s = min(s + v * SIM_DT, s_max)
        a_prev = acc
        lead_s = lead_s + cur_lead_v * SIM_DT
    return dict(ep=ep, ego_s=np.array(ego_s), ego_v=np.array(ego_v), ego_a=np.array(ego_a),
                lead_s=np.array(leads))


# ----------------------------------------------------------------------------- frames
def to_ego(ep: Episode, s_ego: float, xy_global: np.ndarray, h_global: np.ndarray | None = None):
    ex, ey, eh = ep.pose_at(s_ego)
    c, sn = math.cos(eh), math.sin(eh)
    d = xy_global - np.array([ex, ey])
    out = np.stack([c * d[..., 0] + sn * d[..., 1], -sn * d[..., 0] + c * d[..., 1]], -1)
    if h_global is None:
        return out
    return out, h_global - eh


def sample_at(sim: dict, k: int):
    """Ego-frame supervision for step k: GT future waypoints, command, ego status, obstacles."""
    ep, step = sim["ep"], int(DT / SIM_DT)
    fut_idx = [k + step * (i + 1) for i in range(T_FUT)]
    fut_g = np.array([ep.pose_at(sim["ego_s"][j])[:2] for j in fut_idx])
    fut = to_ego(ep, sim["ego_s"][k], fut_g)                               # (T, 2)
    # command from heading change over the horizon (like nuScenes' left/right/straight)
    h_now = ep.pose_at(sim["ego_s"][k])[2]
    h_end = ep.pose_at(sim["ego_s"][fut_idx[-1]] + 5.0)[2]
    dh = (h_end - h_now + math.pi) % (2 * math.pi) - math.pi
    cmd = CMD_LEFT if dh > math.radians(12) else CMD_RIGHT if dh < -math.radians(12) else CMD_STRAIGHT
    status = np.array([sim["ego_v"][k] / 10.0, sim["ego_a"][k] / 3.0], dtype=np.float32)
    # obstacles now (for rendering) and in the future (for collision checks): (n, 3) x, y, yaw
    def obs_at(j):
        rows = []
        for s_l in sim["lead_s"][j]:
            x, y, h = ep.pose_at(s_l)
            xy, hh = to_ego(ep, sim["ego_s"][k], np.array([x, y]), np.array(h))
            rows.append([xy[0], xy[1], float(hh)])
        return np.array(rows, dtype=np.float32).reshape(-1, 3)
    obs_now = obs_at(k)
    obs_fut = np.stack([obs_at(j) for j in fut_idx])                      # (T, n, 3)
    return dict(fut=fut.astype(np.float32), cmd=cmd, status=status, obs_now=obs_now, obs_fut=obs_fut)


# ----------------------------------------------------------------------------- renderer
def _pixel_rays(cam: Camera, H: int, W: int) -> np.ndarray:
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    pix = np.stack([u, v, np.ones_like(u)], -1)                              # (H, W, 3)
    d_cam = pix @ np.linalg.inv(cam.K).T
    d = d_cam @ cam.R.T                                                     # ego-frame directions
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def render(sim: dict, k: int, rig: list[Camera], H: int = 40, W: int = 80,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """Render all cameras at step k -> uint8 array (Ncam, H, W, 3)."""
    ep = sim["ep"]
    ex, ey, eh = ep.pose_at(sim["ego_s"][k])
    c, sn = math.cos(eh), math.sin(eh)
    obs = sample_at_obstacles(sim, k)
    imgs = []
    for cam in rig:
        d = _pixel_rays(cam, H, W).reshape(-1, 3)
        o = cam.t
        col = np.tile(np.array([150, 190, 235], np.float64), (len(d), 1))   # sky
        # --- ground
        hit_g = d[:, 2] < -1e-3
        lam = np.where(hit_g, -o[2] / np.where(hit_g, d[:, 2], -1), np.inf)
        lam = np.where(lam < 120, lam, np.inf)
        gp = o[None, :2] + lam[:, None] * d[:, :2]
        ok = np.isfinite(lam)
        if ok.any():
            gx = ex + c * gp[ok, 0] - sn * gp[ok, 1]
            gy = ey + sn * gp[ok, 0] + c * gp[ok, 1]
            dist, idx = ep.tree.query(np.stack([gx, gy], -1))
            rh = ep.road[idx, 2]
            lat = -(gx - ep.road[idx, 0]) * np.sin(rh) + (gy - ep.road[idx, 1]) * np.cos(rh)
            s_arc = ep.s_of[idx]
            g = np.tile(np.array([70, 130, 60], np.float64), (ok.sum(), 1))      # grass
            road = np.abs(lat) < LANE_W
            g[road] = [95, 95, 100]
            edge = np.abs(np.abs(lat) - LANE_W) < 0.2
            g[edge] = [235, 235, 235]
            centre = (np.abs(lat) < 0.12) & ((s_arc % 6.0) < 3.0)
            g[centre] = [230, 200, 40]
            fog = np.clip(lam[ok] / 120, 0, 1)[:, None]
            col[ok] = (1 - fog * 0.5) * g + fog * 0.5 * np.array([150, 190, 235])
        # --- boxes (slab test in each box frame); nearer than ground wins
        best = np.where(ok, lam, np.inf)
        for (bx, by, byaw) in obs:
            cb, sb = math.cos(byaw), math.sin(byaw)
            Rb = np.array([[cb, sb, 0], [-sb, cb, 0], [0, 0, 1]])   # ego -> box frame
            ob = Rb @ (o - np.array([bx, by, 0.0]))
            db = d @ Rb.T
            half = np.array([CAR_L / 2, CAR_W / 2, CAR_H / 2])
            ctr = np.array([0, 0, CAR_H / 2])
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (ctr - half - ob) / db
                t2 = (ctr + half - ob) / db
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= tmin) & (tmin > 0) & (tmin < best)
            if hit.any():
                face = np.argmax(np.minimum(t1, t2)[hit], axis=1)           # which slab hit first
                shade = np.array([1.0, 0.75, 0.55])[face]
                col[hit] = np.array([200, 40, 40])[None] * shade[:, None]
                best[hit] = tmin[hit]
        img = col.reshape(H, W, 3)
        if rng is not None:
            img = img * rng.uniform(0.85, 1.15) + rng.normal(0, 4, img.shape)
        imgs.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.stack(imgs)


def sample_at_obstacles(sim: dict, k: int) -> np.ndarray:
    ep = sim["ep"]
    rows = []
    for s_l in sim["lead_s"][k]:
        x, y, h = ep.pose_at(s_l)
        xy, hh = to_ego(ep, sim["ego_s"][k], np.array([x, y]), np.array(h))
        rows.append([xy[0], xy[1], float(hh)])
    return np.array(rows).reshape(-1, 3)


def road_in_ego(sim: dict, k: int, lat: float = 0.0, ahead: float = 70, behind: float = 15) -> np.ndarray:
    """Road line (centre or offset) in ego frame for plotting."""
    ep = sim["ep"]
    s0 = sim["ego_s"][k]
    ss = np.linspace(max(s0 - behind, 0), min(s0 + ahead, ep.s_of[-1]), 120)
    g = np.array([ep.pose_at(s, lat)[:2] for s in ss])
    return to_ego(ep, s0, g)


def collides(traj: np.ndarray, obs_fut: np.ndarray, margin: float = 0.3) -> bool:
    """Ego footprint (point + margin, half-width) vs obstacle boxes at each future time."""
    for t in range(traj.shape[0]):
        for (bx, by, byaw) in obs_fut[t]:
            dx, dy = traj[t, 0] - bx, traj[t, 1] - by
            lx = math.cos(byaw) * dx + math.sin(byaw) * dy
            ly = -math.sin(byaw) * dx + math.cos(byaw) * dy
            if abs(lx) < CAR_L + margin and abs(ly) < CAR_W + margin:   # sum of half-sizes of two cars
                return True
    return False
