"""Closed-loop episode: world -> predictor (+margins) -> MPC -> unicycle, with safety metrics and an optional log."""
from __future__ import annotations

import numpy as np
import torch

from . import world as Wd
from .planner import MPC
from .predictor import PETPredictor

METHODS = {
    # name: predictor mode, margin type, fixed margin (m)
    "cv_nominal":      dict(mode="cv", margin="none"),
    "koopman_nominal": dict(mode="koopman", margin="none"),
    "koopman_fixed":   dict(mode="koopman", margin="fixed", fixed=0.30),
    "pet_mpc":         dict(mode="koopman", margin="markov"),
}


def run_episode(scenario: str, seed: int, method: str, cfg: dict, aware: bool = True, log: bool = False,
                n_ped: int = 10) -> dict:
    m = METHODS[method]
    pc, mc = cfg["predictor"], cfg["mpc"]
    world = Wd.World(scenario, seed, n_ped=n_ped)
    pred_mod = PETPredictor(N=mc["N"], dt=Wd.DT, window=pc["window"], lam=pc["lam"], delta=pc["delta"],
                            beta=pc["beta"], cap=pc["cap"], mode=m["mode"])
    mpc = MPC(N=mc["N"], v_max=mc["v_max"], w_max=mc["w_max"], a_max=mc["a_max"], v_ref=mc["v_ref"], aware=aware)
    state = np.array([0.0, 0.0, 0.0])
    goal = np.array([Wd.GOAL_X, 0.0])
    u_applied = np.zeros(2)
    kinds = [a.kind for a in world.agents]
    radii = np.array([a.radius for a in world.agents])
    contact_steps = ps_steps = 0
    min_clear = np.inf
    steps, reached, t_idx = 0, False, 0
    frames = []
    e2_log, mg_log = [], []
    n_max = int(Wd.T_MAX / Wd.DT)
    while t_idx < n_max:
        hists = [a.hist for a in world.agents]
        pos, marg = pred_mod.predict(hists, t_idx)
        pos_np = pos.numpy()
        if m["margin"] == "none":
            marg_np = np.zeros_like(marg.numpy())
        elif m["margin"] == "fixed":
            marg_np = np.full_like(marg.numpy(), m["fixed"])
        else:
            marg_np = marg.numpy()
        # coverage bookkeeping: realised squared errors vs the margin that was promised for them
        v, w, u_applied, plan = mpc.solve(state, goal, pos_np, marg_np, kinds, radii, u_applied)
        if log:
            frames.append(dict(t=world.t, state=state.copy(), agents=world.positions().copy(),
                               theta=np.array([a.theta for a in world.agents]), pred=pos_np.copy(),
                               marg=marg_np.copy(), markov=marg.numpy().copy(), plan=plan.copy(),
                               rows=[(i, h.copy(), ell) for i, h, ell, _ in mpc.last["rows"]], v=v, w=w))
        # unicycle update
        state = state + Wd.DT * np.array([v * np.cos(state[2]), v * np.sin(state[2]), w])
        world.step()
        t_idx += 1
        # safety metrics on the robot BODY centre
        if len(world.agents):
            d = np.linalg.norm(world.positions() - state[:2], axis=1)
            clear = d - Wd.R_ROBOT - radii
            min_clear = min(min_clear, float(clear.min()))
            contact_steps += int(np.any(clear < 0))
            peds = np.array([k == "ped" for k in kinds])
            if peds.any():
                ps_steps += int(np.any(d[peds] < Wd.R_ROBOT + Wd.PS_PED))
        steps += 1
        if np.linalg.norm(state[:2] - goal) < 0.4 or state[0] >= Wd.GOAL_X:
            reached = True
            break
    # prediction-error coverage of the Markov margins (computed from the predictor's own residual logs)
    out = dict(scenario=scenario, seed=seed, method=method, aware=aware, contact=int(contact_steps > 0),
               contact_frac=contact_steps / steps, ps_frac=ps_steps / steps, time=steps * Wd.DT,
               reached=int(reached), min_clear=min_clear)
    if log:
        out["frames"] = frames
        out["final_agents"] = world.positions().copy()
    return out


@torch.no_grad()
def prediction_study(cfg: dict, scenario: str, seeds, T: int = 300, n_ped: int = 10):
    """Open-loop prediction quality: ADE/FDE (CV vs Koopman) and empirical coverage of the Markov margins."""
    N = cfg["mpc"]["N"]
    pc = cfg["predictor"]
    res = {k: dict(se=np.zeros(N), n=0, cover=np.zeros(N), ncov=np.zeros(N), marg=np.zeros(N)) for k in ("cv", "koopman")}
    for seed in seeds:
        world = Wd.World(scenario, seed, n_ped=n_ped)
        preds = {k: PETPredictor(N=N, window=pc["window"], lam=pc["lam"], delta=pc["delta"], beta=pc["beta"],
                                 cap=pc["cap"], mode=k) for k in res}
        store = {k: [] for k in res}
        traj = []
        for t in range(T):
            hists = [a.hist for a in world.agents]
            for k, p in preds.items():
                pos, mg = p.predict(hists, t)
                store[k].append((pos.numpy(), mg.numpy()))
            traj.append(world.positions().copy())
            world.step()
        traj.append(world.positions().copy())
        traj = np.array(traj)                                                  # (T+1, B, 2)
        for k in res:
            for t in range(T - N):
                if t < 15:                                                     # skip warm-up
                    continue
                pos, mg = store[k][t]
                fut = traj[t + 1:t + 1 + N].transpose(1, 0, 2)                 # (B, N, 2)
                e2 = ((fut - pos) ** 2).sum(-1)                                # (B, N)
                res[k]["se"] += np.sqrt(e2).sum(0)
                res[k]["n"] += e2.shape[0]
                res[k]["cover"] += (np.sqrt(e2) <= mg).sum(0)
                res[k]["marg"] += mg.sum(0)
    out = {}
    for k, r in res.items():
        ade = r["se"] / max(r["n"], 1)
        out[k] = dict(ADE=float(ade.mean()), FDE=float(ade[-1]), err_by_step=ade.tolist(),
                      coverage_by_step=(r["cover"] / max(r["n"], 1)).tolist(),
                      coverage=float((r["cover"] / max(r["n"], 1)).mean()),
                      margin_by_step=(r["marg"] / max(r["n"], 1)).tolist())
    return out
