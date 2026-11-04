"""Synthetic ego-centric BEV 'driving video' world with ego-reactive traffic (my own toy, NOT NAVSIM / nuScenes).

Scene : 3 lanes, ego + 5 other road users (cars / trucks). Frames are 48x48 BEV images (ego-centric, forward = up).
Ego   : picks one of 5 action prototypes every 0.5 s (brake / ease / keep / accel / swerve-left).
Others: IDM car-following (ego is a leader for agents behind it) + hidden "aggressiveness" -> aggressive cars in an
        adjacent lane CUT IN when the ego is not accelerating. So the ego's action changes how the traffic evolves
        (this is the 'interaction' the world model has to learn from video alone).
Oracle: privileged simulator rollout cost per prototype (collision / near miss / progress) used ONLY to distil the planner.
"""
import numpy as np

DT = 0.5
LANES = np.array([-3.5, 0.0, 3.5])
IMG = 48
XR = (-6.0, 42.0)          # forward extent in metres, 1 m / row
YR = (-12.0, 12.0)         # lateral extent in metres, 0.5 m / col
PROTOS = np.array([[-3.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.5, 0.0], [0.0, 1.5]], dtype=np.float32)  # (accel, lat vel)
PROTO_NAMES = ["brake", "ease", "keep", "accel", "swerve_L"]
NA = len(PROTOS)
NAG = 5
EGO_L, EGO_W = 4.5, 1.8
KEEP = 2


class World:
    def __init__(self, rng=None, state=None):
        if state is not None:
            for k, v in state.items():
                setattr(self, k, v.copy() if isinstance(v, np.ndarray) else v)
            return
        self.xe, self.ve, self.ye = 0.0, float(rng.uniform(8, 16)), float(rng.choice(LANES))
        xs, ys = [], []
        while len(xs) < NAG:
            y, x = float(rng.choice(LANES)), float(rng.uniform(-8, 40))
            if abs(y - self.ye) < 2.2 and abs(x) < 10:
                continue
            if any(abs(y - yy) < 2.2 and abs(x - xx) < 10 for xx, yy in zip(xs, ys)):
                continue
            xs.append(x); ys.append(y)
        self.x, self.y = np.array(xs), np.array(ys)
        self.v = rng.uniform(5, 15, NAG)
        self.vdes = self.v + rng.uniform(0, 3, NAG)
        truck = rng.random(NAG) < 0.25
        self.L = np.where(truck, 7.0, 4.5)
        self.W = np.where(truck, 2.4, 1.8)
        self.g = rng.random(NAG)                       # hidden aggressiveness
        self.ty = self.y.copy()
        self.merging = np.zeros(NAG, bool)
        self.cutin_now = False

    def state(self):
        return {k: v for k, v in self.__dict__.items()}

    def clone(self):
        return World(state=self.state())

    def step(self, m):
        """Advance 0.5 s with ego prototype m. Returns True if a collision happened in this step."""
        a, vy = PROTOS[m]
        ve = float(np.clip(self.ve + a * DT, 0, 22))
        self.xe += 0.5 * (self.ve + ve) * DT
        self.ve = ve
        self.ye = float(np.clip(self.ye + vy * DT, -5.25, 5.25))
        n = NAG
        acc = np.zeros(n)
        self.cutin_now = False
        for i in range(n):
            gap, vl = 1e9, self.vdes[i]
            for j in range(n):
                if j != i and self.x[j] > self.x[i] and abs(self.y[j] - self.y[i]) < 2.2:
                    gj = self.x[j] - self.x[i] - 0.5 * (self.L[i] + self.L[j])
                    if gj < gap: gap, vl = gj, self.v[j]
            if self.xe > self.x[i] and abs(self.ye - self.y[i]) < 2.2:      # ego is a leader for agents behind it
                ge = self.xe - self.x[i] - 0.5 * (self.L[i] + EGO_L)
                if ge < gap: gap, vl = ge, self.ve
            gap = max(gap, 0.3)
            sstar = 2.0 + max(0.0, self.v[i] + self.v[i] * (self.v[i] - vl) / (2 * np.sqrt(3.0)))
            acc[i] = np.clip(1.5 * (1 - (self.v[i] / self.vdes[i]) ** 4 - (sstar / gap) ** 2), -6, 2)
            rel = self.x[i] - self.xe
            if (not self.merging[i]) and self.g[i] > 0.4 and 6 < rel < 20 and 2.2 <= abs(self.y[i] - self.ye) < 5.0 \
                    and a <= 0.0 and self.ve < self.v[i] + 2:
                self.merging[i], self.ty[i], self.cutin_now = True, self.ye, True       # cut-in: reaction to the ego's action
            if self.merging[i]:
                d = self.ty[i] - self.y[i]
                self.y[i] += np.clip(d, -1.5 * DT, 1.5 * DT)
                if abs(self.ty[i] - self.y[i]) < 1e-3: self.merging[i] = False
        vn = np.clip(self.v + acc * DT, 0, 25)
        self.x = self.x + 0.5 * (self.v + vn) * DT
        self.v = vn
        return self.collided()

    def collided(self):
        return bool(np.any((np.abs(self.x - self.xe) < 0.5 * (self.L + EGO_L)) & (np.abs(self.y - self.ye) < 0.5 * (self.W + EGO_W) + 0.1)))

    def min_gap(self):
        dx = np.maximum(np.abs(self.x - self.xe) - 0.5 * (self.L + EGO_L), 0)
        dy = np.maximum(np.abs(self.y - self.ye) - 0.5 * (self.W + EGO_W), 0)
        return float(np.sqrt(dx ** 2 + dy ** 2).min())

    def render(self):
        """uint8 [3,48,48]: ch0 cars, ch1 trucks, ch2 ego (ego-centric BEV, forward up)."""
        im = np.zeros((3, IMG, IMG), np.uint8)
        def box(ch, x, y, L, W):
            r0, r1 = int(round(XR[1] - (x + L / 2 - 0.0))), int(round(XR[1] - (x - L / 2)))
            c0, c1 = int(round((y - W / 2 - YR[0]) / 0.5)), int(round((y + W / 2 - YR[0]) / 0.5))
            r0, r1, c0, c1 = max(r0, 0), min(r1, IMG), max(c0, 0), min(c1, IMG)
            if r1 > r0 and c1 > c0: im[ch, r0:r1, c0:c1] = 255
        for i in range(NAG):
            box(1 if self.L[i] > 5 else 0, self.x[i] - self.xe, self.y[i], self.L[i], self.W[i])
        box(2, 0.0, self.ye, EGO_L, EGO_W)
        return im

    def agent_pixels(self, rel_dx=0.0):
        """pixel centres (row, col) of ego (index 0) + the 5 agents."""
        xs = np.concatenate([[0.0], self.x - self.xe]); ys = np.concatenate([[self.ye], self.y])
        return np.stack([XR[1] - xs - 0.5, (ys - YR[0]) / 0.5 - 0.5], 1)


def oracle_costs(w, H=6, hold=2):
    """Privileged cost of each prototype: hold it `hold` steps, then keep speed, over H steps (3 s)."""
    c = np.zeros(NA, np.float32)
    for m in range(NA):
        w2 = w.clone(); x0 = w2.xe; col = False; mg = 1e9
        for h in range(H):
            col |= w2.step(m if h < hold else KEEP)
            mg = min(mg, w2.min_gap())
        c[m] = 10.0 * col + 2.0 * np.exp(-mg / 4.0) - 0.3 * (w2.xe - x0) / 10.0 + 0.1 * abs(PROTOS[m, 0])
    return c


def make_dataset(n, seed, T=10):
    """Random-action episodes. frames [n,T+1,3,48,48] uint8, act [n,T], costs [n,T,5], cutin [n,T] (cut-in during that step)."""
    rng = np.random.default_rng(seed)
    F = np.zeros((n, T + 1, 3, IMG, IMG), np.uint8); A = np.zeros((n, T), np.int64)
    C = np.zeros((n, T, NA), np.float32); CI = np.zeros((n, T), bool)
    for e in range(n):
        w = World(rng); a = int(rng.integers(NA))
        F[e, 0] = w.render()
        for t in range(T):
            if t % 2 == 0: a = int(rng.integers(NA))
            C[e, t] = oracle_costs(w); A[e, t] = a
            w.step(a); CI[e, t] = w.cutin_now; F[e, t + 1] = w.render()
    return dict(frames=F, act=A, costs=C, cutin=CI)
