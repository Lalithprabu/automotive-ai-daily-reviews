"""Synthetic 1:10-scale race car + closed track with friction sectors (toy world, my construction).

Vehicle: dynamic single-track ("bicycle") model with a simplified Pacejka lateral tyre
    F_y = mu * F_z * sin(C * atan(B * alpha))
so BOTH the peak force and the cornering stiffness (B*C*mu*F_z) scale with the friction coefficient mu.
States (X, Y, psi, vx, vy, r); inputs: steering delta (rate-limited) and a longitudinal speed set-point
tracked by a first-order drive loop.  RK4 at 100 Hz, control at 20 Hz.

Track: smooth closed curve (perturbed ellipse) sampled at 1 cm, width 1.0 m.  Each sample carries a friction
value mu(s), so a sector can be "wet" (low mu).  The racing line is the centre line (tracking task).
"""
from __future__ import annotations

import numpy as np

G = 9.81
P = dict(m=3.74, Iz=0.04712, lf=0.15875, lr=0.17145, B=6.0, C=1.6, delta_max=0.40, ddelta_max=4.0)
DT_SIM, DT = 0.01, 0.05
HALF_W = 0.5


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Track:
    """Closed centre line with arc length s, heading, curvature and per-sample friction mu(s)."""

    def __init__(self, a=4.0, b=2.4, ds=0.01, mu_high=1.0, mu_low=0.5, wet=()):
        th = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
        # perturbed ellipse: two tighter corners + a gentle S on one straight
        r = 1.0 + 0.12 * np.cos(3 * th) + 0.06 * np.sin(5 * th)
        x, y = a * r * np.cos(th), b * r * np.sin(th)
        seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
        s = np.concatenate([[0], np.cumsum(seg)[:-1]])
        self.L = float(seg.sum())
        self.s = np.arange(0, self.L, ds)
        self.x = np.interp(self.s, s, x, period=self.L)
        self.y = np.interp(self.s, s, y, period=self.L)
        dx = np.gradient(np.concatenate([self.x[-3:], self.x, self.x[:3]]))[3:-3]
        dy = np.gradient(np.concatenate([self.y[-3:], self.y, self.y[:3]]))[3:-3]
        self.psi = np.unwrap(np.arctan2(dy, dx))
        k = np.gradient(np.concatenate([self.psi[-50:] - 2 * np.pi, self.psi, self.psi[:50] + 2 * np.pi]), ds)[50:-50]
        # smooth curvature (numerical noise from 1 cm resampling)
        ker = np.ones(41) / 41
        self.kappa = np.convolve(np.concatenate([k[-20:], k, k[:20]]), ker, "valid")
        self.mu = np.full(len(self.s), mu_high)
        for s0, s1 in wet:                                       # wet sectors as fractions of the lap
            m = (self.s >= s0 * self.L) & (self.s < s1 * self.L)
            self.mu[m] = mu_low
        self.mu_high, self.mu_low = mu_high, mu_low

    def idx(self, s):
        return (np.round(np.asarray(s) / (self.s[1] - self.s[0])).astype(int)) % len(self.s)

    def frenet(self, X, Y, psi, i_hint=None, win=300):
        """Nearest centre-line sample (local search around i_hint) -> (i, e_y [+left], e_psi)."""
        n = len(self.s)
        if i_hint is None:
            cand = np.arange(n)
        else:
            cand = (i_hint + np.arange(-win // 3, win)) % n
        d2 = (self.x[cand] - X) ** 2 + (self.y[cand] - Y) ** 2
        i = int(cand[np.argmin(d2)])
        nx, ny = -np.sin(self.psi[i]), np.cos(self.psi[i])
        ey = (X - self.x[i]) * nx + (Y - self.y[i]) * ny
        return i, float(ey), float(wrap(psi - self.psi[i]))

    def xy_from_frenet(self, s, ey):
        i = self.idx(s)
        return np.stack([self.x[i] - np.sin(self.psi[i]) * ey, self.y[i] + np.cos(self.psi[i]) * ey], -1)

    def speed_profile(self, v_max=3.0, a_lat=2.9, a_long=2.0):
        """v_ref(s) = min(v_max, sqrt(a_lat/|kappa|)), forward/backward passes for accel/brake limits."""
        v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(self.kappa), 1e-3)))
        ds = self.s[1] - self.s[0]
        for _ in range(2):
            for i in range(1, len(v)):
                v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_long * ds))
            for i in range(len(v) - 2, -1, -1):
                v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_long * ds))
        return v


def tyre_forces(state, delta, mu_f, mu_r, p=P):
    X, Y, psi, vx, vy, r = state
    L = p["lf"] + p["lr"]
    Fzf, Fzr = p["m"] * G * p["lr"] / L, p["m"] * G * p["lf"] / L
    vxs = max(vx, 0.5)
    af = delta - np.arctan2(vy + p["lf"] * r, vxs)
    ar = -np.arctan2(vy - p["lr"] * r, vxs)
    Ff = mu_f * Fzf * np.sin(p["C"] * np.arctan(p["B"] * af))
    Fr = mu_r * Fzr * np.sin(p["C"] * np.arctan(p["B"] * ar))
    return Ff, Fr, af, ar


def deriv(state, delta, ax, mu_f, mu_r, p=P):
    X, Y, psi, vx, vy, r = state
    Ff, Fr, _, _ = tyre_forces(state, delta, mu_f, mu_r, p)
    vxs = max(vx, 0.5)
    return np.array([
        vx * np.cos(psi) - vy * np.sin(psi),
        vx * np.sin(psi) + vy * np.cos(psi),
        r,
        ax + vy * r - Ff * np.sin(delta) / p["m"],
        (Ff * np.cos(delta) + Fr) / p["m"] - vxs * r,
        (p["lf"] * Ff * np.cos(delta) - p["lr"] * Fr) / p["Iz"],
    ])


def cornering_stiffness(mu, p=P):
    """Linearised axle stiffness dF/dalpha at alpha = 0: B*C*mu*F_z."""
    L = p["lf"] + p["lr"]
    Fzf, Fzr = p["m"] * G * p["lr"] / L, p["m"] * G * p["lf"] / L
    return p["B"] * p["C"] * mu * Fzf, p["B"] * p["C"] * mu * Fzr


class Car:
    """Plant: integrates the single-track model; friction read from the track at the car's position."""

    def __init__(self, track: Track, v0=1.5, ey0=0.0, seed=0, meas_noise=(0.004, 0.004)):
        self.tr = track
        i = 0
        self.state = np.array([track.x[i] - np.sin(track.psi[i]) * ey0, track.y[i] + np.cos(track.psi[i]) * ey0,
                               track.psi[i], v0, 0.0, 0.0])
        self.delta = 0.0
        self.i = 0
        self.s_tot = 0.0
        self.rng = np.random.default_rng(seed)
        self.noise = meas_noise

    def mu_here(self):
        return float(self.tr.mu[self.i])

    def measure(self):
        """Frenet errors with additive sensor noise (e_y in m, e_psi in rad)."""
        i, ey, ep = self.tr.frenet(self.state[0], self.state[1], self.state[2], self.i)
        ds = (self.tr.s[i] - self.tr.s[self.i]) % self.tr.L
        if ds > self.tr.L / 2:
            ds -= self.tr.L
        self.s_tot += ds
        self.i = i
        return np.array([ey + self.rng.normal(0, self.noise[0]), ep + self.rng.normal(0, self.noise[1])]), (ey, ep)

    def step(self, delta_cmd, v_ref):
        """Apply steering command for one control period DT (rate limit + RK4 at DT_SIM)."""
        p = P
        for _ in range(int(round(DT / DT_SIM))):
            dd = np.clip(delta_cmd - self.delta, -p["ddelta_max"] * DT_SIM, p["ddelta_max"] * DT_SIM)
            self.delta = float(np.clip(self.delta + dd, -p["delta_max"], p["delta_max"]))
            ax = np.clip(3.0 * (v_ref - self.state[3]), -4.0, 3.0)
            mu = self.mu_here()
            s = self.state
            k1 = deriv(s, self.delta, ax, mu, mu)
            k2 = deriv(s + 0.5 * DT_SIM * k1, self.delta, ax, mu, mu)
            k3 = deriv(s + 0.5 * DT_SIM * k2, self.delta, ax, mu, mu)
            k4 = deriv(s + DT_SIM * k3, self.delta, ax, mu, mu)
            self.state = s + DT_SIM / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.state[3] = max(self.state[3], 0.5)
