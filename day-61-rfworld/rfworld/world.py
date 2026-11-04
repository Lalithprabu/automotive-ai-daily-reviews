"""Toy driving world + analytic PDMS-style scorer (NOT NAVSIM).

Global frame: X forward, Y left. Ego frame: x forward, y left, origin at ego start pose.
BEV raster: 64x64 cells of 0.5 m, rows = forward (row 0 = x 32 m), cols = lateral (col 0 = y +16 m).
"""
import numpy as np

G, RES, XMAX, YMAX = 64, 0.5, 32.0, 16.0
T, DT = 6, 0.5                    # 6 waypoints, 0.5 s apart -> 3 s horizon
R_OBJ = 1.2                       # circle radius used for ego and objects (m)
NK, NS = 16, 16                   # anchor grid: curvatures x path lengths -> N = 256 fixed anchors
N_ANCH = NK * NS


def make_anchors():
    """Fixed trajectory vocabulary [N,T,2] in the ego frame (constant-curvature arcs, uniform speed)."""
    ks = np.linspace(-0.05, 0.05, NK)
    Ss = np.linspace(4.0, 28.0, NS)
    out = np.zeros((NK * NS, T, 2), np.float32)
    meta = np.zeros((NK * NS, 2), np.float32)
    i = 0
    for k in ks:
        for S in Ss:
            s = S * np.arange(1, T + 1) / T
            if abs(k) < 1e-6:
                x, y = s, 0 * s
            else:
                x, y = np.sin(k * s) / k, (1 - np.cos(k * s)) / k
            out[i, :, 0], out[i, :, 1] = x, y
            meta[i] = (k, S)
            i += 1
    return out, meta


ANCHORS, ANCHOR_META = make_anchors()


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


class World:
    """Road centreline Y=yc(X) (sinusoid), moving vehicles (constant velocity), ego pose + speed."""

    def __init__(self, rng, n_obj=None):
        self.rng = rng
        self.a = rng.uniform(2.0, 6.0); self.lam = rng.uniform(60, 120); self.ph = rng.uniform(0, 6.28)
        self.width = rng.uniform(8.0, 11.0)
        X0 = 0.0
        self.ego = np.array([X0, self.yc(X0) + rng.uniform(-1.5, 1.5)])
        self.th = self.road_heading(X0) + rng.uniform(-0.1, 0.1)
        self.v = rng.uniform(3.0, 9.0)
        self.t = 0.0
        n = n_obj if n_obj is not None else rng.integers(1, 5)
        self.obj_p = np.zeros((0, 2)); self.obj_v = np.zeros((0, 2))
        for _ in range(n): self.spawn(first=True)

    # ---- road ------------------------------------------------------------
    def yc(self, X): return self.a * np.sin(2 * np.pi * X / self.lam + self.ph)
    def dyc(self, X): return self.a * 2 * np.pi / self.lam * np.cos(2 * np.pi * X / self.lam + self.ph)
    def road_heading(self, X): return float(np.arctan(self.dyc(X)))
    def on_road(self, P): return np.abs(P[..., 1] - self.yc(P[..., 0])) < self.width / 2 - 0.3

    # ---- objects ---------------------------------------------------------
    def spawn(self, first=False):
        r = self.rng
        X = self.ego[0] + (r.uniform(8, 35) if first else r.uniform(30, 40))
        P = np.array([X, self.yc(X) + r.uniform(-3.0, 3.0)])
        V = np.array([r.uniform(0, 7), 0.0])
        if r.random() < 0.3: V[1] = r.choice([-1.5, 1.5])      # lateral drift ~ cut-in
        self.obj_p = np.vstack([self.obj_p, P]); self.obj_v = np.vstack([self.obj_v, V])

    def obj_at(self, t): return self.obj_p + self.obj_v * t

    # ---- frames ----------------------------------------------------------
    def to_ego(self, P): return (np.asarray(P) - self.ego) @ rot(self.th)         # = R(-th)(P-ego)
    def to_glob(self, p): return np.asarray(p) @ rot(self.th).T + self.ego

    # ---- observation -----------------------------------------------------
    def _cells(self):
        rows, cols = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        x = XMAX - (rows + .5) * RES; y = YMAX - (cols + .5) * RES
        return x, y

    def occupancy(self, dt):
        """Object occupancy raster at time offset dt (s), ego frame (static ego start pose)."""
        x, y = self._cells(); occ = np.zeros((G, G), np.float32)
        for p in self.to_ego(self.obj_at(dt)):
            occ = np.maximum(occ, ((x - p[0]) ** 2 + (y - p[1]) ** 2 < (R_OBJ + .3) ** 2).astype(np.float32))
        return occ

    def drivable(self):
        x, y = self._cells(); Pg = self.to_glob(np.stack([x, y], -1))
        return self.on_road(Pg).astype(np.float32)

    def observe(self):
        r = self.rng
        road, occ0, occm = self.drivable(), self.occupancy(0.0), self.occupancy(-1.0)
        cam = np.stack([occ0, 0.6 * road, 0.3 * road + 0.7 * (np.abs(self._cells()[1]) < .3)], 0)
        cam = np.clip(cam + r.normal(0, 0.05, cam.shape), 0, 1).astype(np.float32)          # "camera-like" BEV
        keep = (r.random((G, G)) > 0.1).astype(np.float32)                                    # lidar dropout
        lidar = np.stack([occ0 * keep, occm * keep], 0)
        ego = np.array([self.v / 10, 0.0, 0.0], np.float32)
        ahead = self.to_ego(np.array([self.ego[0] + 20, self.yc(self.ego[0] + 20)]))[1]
        nav = np.eye(3, dtype=np.float32)[0 if ahead > 2.5 else (2 if ahead < -2.5 else 1)]   # left/straight/right
        fut = np.stack([self.occupancy(1.5), self.occupancy(3.0)], 0)
        fut = fut.reshape(2, 16, 4, 16, 4).mean((2, 4)).astype(np.float32)                    # 16x16 targets
        return dict(cam=cam, lidar=lidar, ego=ego, nav=nav, fut=fut)

    # ---- scoring ---------------------------------------------------------
    def score(self, anchors=ANCHORS):
        """PDMS-style: NC*DAC*(5*EP + 5*TTC + 2*COMF)/12 for all anchors. Returns dict of [N] arrays."""
        N = anchors.shape[0]
        wp = anchors                                                                # [N,T,2] ego frame
        wg = self.to_glob(wp.reshape(-1, 2)).reshape(N, T, 2)
        dac = self.on_road(wg).all(1).astype(np.float32)
        d = np.full((N, T), 1e3)
        for k in range(T):
            op = self.to_ego(self.obj_at(DT * (k + 1)))                              # [M,2]
            if len(op): d[:, k] = np.linalg.norm(wp[:, k, None] - op[None], axis=-1).min(1)
        dmin = d.min(1)
        nc = (dmin > 2 * R_OBJ).astype(np.float32)
        ttc = (dmin > 2 * R_OBJ + 1.2).astype(np.float32)
        ep = np.clip(wp[:, -1, 0] / 28.0, 0, 1).astype(np.float32)
        acc = 2 * (np.linalg.norm(wp[:, -1], axis=-1) / (T * DT) - self.v) / (T * DT)
        comf = (np.abs(acc) < 2.5).astype(np.float32)
        pdms = nc * dac * (5 * ep + 5 * ttc + 2 * comf) / 12
        return dict(nc=nc, dac=dac, ttc=ttc, ep=ep, comf=comf, pdms=pdms.astype(np.float32), dmin=dmin)

    # ---- closed loop -----------------------------------------------------
    def step(self, traj_ego):
        """Execute the first waypoint of a chosen ego-frame trajectory (0.5 s), advance traffic."""
        p1 = traj_ego[0]; newg = self.to_glob(p1)
        self.v = float(np.linalg.norm(p1) / DT)
        self.th = self.th + float(np.arctan2(p1[1], max(p1[0], 1e-3)))
        self.ego = newg; self.obj_p = self.obj_p + self.obj_v * DT; self.t += DT
        keep = (self.obj_p[:, 0] > self.ego[0] - 12) & (self.obj_p[:, 0] < self.ego[0] + 60)
        self.obj_p, self.obj_v = self.obj_p[keep], self.obj_v[keep]
        while len(self.obj_p) < 3: self.spawn()
        return bool((np.linalg.norm(self.to_ego(self.obj_p), axis=1) < 2 * R_OBJ).any())


def make_batch(rng, n):
    obs, sc = [], []
    for _ in range(n):
        w = World(rng); o = w.observe(); s = w.score()
        obs.append(o); sc.append(s)
    B = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
    B.update({"t_" + k: np.stack([s[k] for s in sc]) for k in ("nc", "dac", "ttc", "ep", "comf", "pdms")})
    return B
