"""Synthetic corridor world for PET-MPC (toy re-creation, NOT the authors' simulator).

A unicycle robot drives from (0, 0) to (GOAL_X, 0) along a corridor |y| <= HALF_W.
Two scenario families:

  * "obstacle": ONE moving generic obstacle (a cart / robot, radius 0.40 m) that drives toward or across
    the robot's lane with random speed changes and turns - mirrors the paper's "moving generic obstacle"
    paired-seed study.
  * "crowd":    N pedestrians (radius 0.30 m) walking with- and against the flow, with stochastic turn
    rates, stop-and-go episodes and occasional abrupt heading changes ("texting drift").

Agents are NON-reactive (they do not avoid the robot). That is a worst case and keeps the paired comparison
exact: the same seed gives the same agent motion for every controller.

Agent state per step: position p (2,), heading theta, speed s, turn rate omega.  dt = DT.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DT = 0.1                 # s, simulation and MPC sampling time
GOAL_X = 20.0            # m
HALF_W = 3.0             # m, corridor half width
R_ROBOT = 0.35           # m, robot body radius
R_PED = 0.30             # m
R_OBS = 0.40             # m, generic obstacle
PS_PED = 0.90            # m, pedestrian personal-space radius (centre to boundary) - my choice
T_MAX = 45.0             # s, episode timeout


@dataclass
class Agent:
    kind: str                        # "ped" | "obs"
    p: np.ndarray                    # (2,)
    theta: float
    s: float
    omega: float
    s_nom: float
    radius: float
    stop_timer: float = 0.0
    hist: list = field(default_factory=list)

    def obs(self) -> np.ndarray:
        """7-D Koopman observable  z = [x, y, vx, vy, theta, omega, speed]."""
        return np.array([self.p[0], self.p[1], self.s * np.cos(self.theta), self.s * np.sin(self.theta),
                         self.theta, self.omega, self.s])


class World:
    """Holds the agents and advances them with a seeded RNG (independent of the robot -> paired seeds)."""

    def __init__(self, scenario: str, seed: int, n_ped: int = 10):
        self.rng = np.random.default_rng(seed)
        self.scenario = scenario
        self.t = 0.0
        self.agents: list[Agent] = []
        r = self.rng
        if scenario == "obstacle":
            # oncoming or crossing cart that will meet the robot around x in [6, 12] m
            mode = r.choice(["oncoming", "crossing"])
            if mode == "oncoming":
                p = np.array([r.uniform(14, 18), r.uniform(-1.5, 1.5)])
                th = np.pi + r.uniform(-0.25, 0.25)
                s = r.uniform(0.6, 1.2)
            else:
                side = r.choice([-1, 1])
                p = np.array([r.uniform(6, 11), side * r.uniform(2.6, 3.2)])
                th = -side * np.pi / 2 + r.uniform(-0.5, 0.5)
                s = r.uniform(0.4, 0.9)
            self.agents.append(Agent("obs", p, th, s, 0.0, s, R_OBS))
            self.p_jump, self.jump_scale, self.sig_w, self.sig_s, self.k_w = 0.03, 1.0, 0.12, 0.12, 0.35
        elif scenario == "crowd":
            for _ in range(n_ped):
                with_flow = r.random() < 0.4
                p = np.array([r.uniform(3, 22), r.uniform(-2.6, 2.6)])
                th = (0.0 if with_flow else np.pi) + r.uniform(-0.4, 0.4)
                s = r.uniform(0.5, 1.3) * (0.6 if with_flow else 1.0)
                self.agents.append(Agent("ped", p, th, s, 0.0, s, R_PED))
            self.p_jump, self.jump_scale, self.sig_w, self.sig_s, self.k_w = 0.015, 0.9, 0.12, 0.1, 0.35
        else:
            raise ValueError(scenario)
        for a in self.agents:
            a.hist.append(a.obs())

    def step(self):
        r = self.rng
        for a in self.agents:
            # turn rate: Ornstein-Uhlenbeck + rare abrupt heading changes; speed: OU around s_nom + stop-and-go
            a.omega += -self.k_w * a.omega * DT + self.sig_w * np.sqrt(DT) * r.normal()
            if r.random() < self.p_jump:
                a.omega = r.choice([-1, 1]) * r.uniform(0.6, 1.0) * self.jump_scale
            if a.stop_timer > 0:
                a.stop_timer -= DT
                target = 0.0
            else:
                target = a.s_nom
                if r.random() < self.p_jump * 0.6:
                    a.stop_timer = r.uniform(0.5, 1.5)
                if r.random() < self.p_jump * 0.6:
                    a.s_nom = float(np.clip(a.s_nom * r.uniform(1.3, 1.9), 0.3, 1.8))
            a.s += 1.5 * (target - a.s) * DT + self.sig_s * np.sqrt(DT) * r.normal()
            a.s = float(np.clip(a.s, 0.0, 2.0))
            a.theta += a.omega * DT
            a.p = a.p + DT * a.s * np.array([np.cos(a.theta), np.sin(a.theta)])
            # soft corridor walls: steer back toward the inside
            if abs(a.p[1]) > HALF_W - a.radius:
                a.p[1] = np.sign(a.p[1]) * (HALF_W - a.radius)
                a.theta = -a.theta * 0.5
                a.omega = 0.0
            a.hist.append(a.obs())
        self.t += DT

    def positions(self) -> np.ndarray:
        return np.array([a.p for a in self.agents])
