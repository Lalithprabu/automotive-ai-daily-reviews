"""Evaluation helpers: grounding, open-loop future prediction, closed-loop driving in the synthetic world."""
import numpy as np, torch, torch.nn.functional as F
from .synthetic import World, oracle_costs, NA, KEEP
from .model import candidate_futures


@torch.no_grad()
def encode_all(wm, d, aemb, bs=64):
    """slots for every frame [N,T+1,K,d] and predicted futures for every (episode, step, action) [N,T,A,2,K,d] (steps +1 and +3, fp16)."""
    Ss, Fs = [], []
    for i in range(0, len(d["frames"]), bs):
        x = torch.as_tensor(d["frames"][i:i + bs]).float() / 255.0
        S = wm.enc(x); Ss.append(S)
        B, T1, K, dd = S.shape
        fut = candidate_futures(wm, S[:, :-1].reshape(B * (T1 - 1), K, dd), aemb, 3)         # [B*T,A,3,K,d]
        Fs.append(fut[:, :, [0, 2]].reshape(B, T1 - 1, *fut[:, :, [0, 2]].shape[1:]).half())
    return torch.cat(Ss), torch.cat(Fs)


@torch.no_grad()
def ground_actions(wm, d, n=300):
    """Embed each action prototype as the mean latent-action code the inverse model assigns when that prototype was executed.
    (Needs a small action-labelled set; INTERACT's latent actions themselves are learned WITHOUT labels.)"""
    x = torch.as_tensor(d["frames"][:n]).float() / 255.0
    S = wm.enc(x); B, T1, K, dd = S.shape
    code, _, y = wm.lat(S[:, :-1].reshape(-1, K, dd), S[:, 1:].reshape(-1, K, dd))
    act = torch.as_tensor(d["act"][:n]).reshape(-1)
    aemb = torch.stack([code[act == m].mean(0) if (act == m).any() else code.mean(0) for m in range(NA)])
    idx = y.argmax(-1); pur = sum(np.bincount(idx[act == m].numpy(), minlength=y.shape[1]).max() for m in range(NA) if (act == m).any()) / len(act)
    return aemb, float(pur)


@torch.no_grad()
def open_loop_eval(wm, d, aemb, H=3):
    """MSE (x1e3) of decoded future frames +1..+3: copy-last / action-blind / grounded true action / inferred latent (teacher)."""
    x = torch.as_tensor(d["frames"]).float() / 255.0
    act, ci = torch.as_tensor(d["act"]), torch.as_tensor(d["cutin"])
    S = wm.enc(x); N, T1, K, dd = S.shape
    out = {k: [] for k in ["copy_last", "action_blind", "grounded", "inferred"]}
    sub = {k: [] for k in ["action_blind", "grounded"]}; sub_flag = []
    blind = aemb.mean(0)
    for t0 in range(0, T1 - H - 1, 2):
        s0 = S[:, t0]; tgt = x[:, t0 + 1:t0 + 1 + H]
        flag = ci[:, t0:t0 + H].any(1)
        res = {}
        res["copy_last"] = ((x[:, t0:t0 + 1] - tgt) ** 2).mean((1, 2, 3, 4))
        for name, getc in [("action_blind", lambda t: blind[None].expand(N, -1)), ("grounded", lambda t: aemb[act[:, t0 + t]]),
                           ("inferred", lambda t: wm.lat(S[:, t0 + t], S[:, t0 + t + 1])[0])]:
            cur, errs = s0, []
            for t in range(H):
                cur = wm.pred(cur, getc(t)); img, _ = wm.dec(cur); errs.append((img - tgt[:, t]) ** 2)
            res[name] = torch.stack(errs, 1).mean((1, 2, 3, 4))
        for k in out: out[k].append(res[k])
        sub_flag.append(flag)
        for k in sub: sub[k].append(res[k])
    cat = lambda v: torch.cat(v)
    r = {k: round(cat(v).mean().item() * 1e3, 3) for k, v in out.items()}
    f = cat(sub_flag)
    r["cutin_windows"] = int(f.sum()); r["other_windows"] = int((~f).sum())
    for k in sub:
        r[f"{k}_cutin"] = round(cat(sub[k])[f].mean().item() * 1e3, 3) if f.any() else None
        r[f"{k}_nocutin"] = round(cat(sub[k])[~f].mean().item() * 1e3, 3)
    with torch.no_grad():
        rec, m = wm.dec(S[:, 5]); r["recon_mse_x1e3"] = round(F.mse_loss(rec, x[:, 5]).item() * 1e3, 3)
        active = ((m > 0.5).flatten(2).sum(-1) >= 4).sum(1).float() - 1          # minus the background slot
        true_n = (x[:, 5].amax((1)) > 0.5).float().flatten(1).sum(-1)
        r["active_slots_mean"] = round(active.mean().item(), 2)
    return r


def policy_actions(name, wm, planners, aemb, frames, prev_frames, S):
    """frames [B,3,H,W] float; S current slots [B,K,d] -> action indices [B]."""
    if name == "slot_full" or name == "slot_nofuture":
        fut = candidate_futures(wm, S, aemb, 3)[:, :, [0, 2]]
        return planners[name](S, fut, aemb).argmin(-1)
    if name == "dense":
        return planners["dense"](prev_frames, frames).argmin(-1)


@torch.no_grad()
def closed_loop_eval(wm, planners, aemb, n=300, T=10, seed=424242):
    """Same n initial scenes for every policy; the world's cars REACT to the ego's actions."""
    pols = ["keep", "random", "dense", "slot_nofuture", "slot_full", "oracle"]
    res = {}
    for p in pols:
        rng = np.random.default_rng(seed); prng = np.random.default_rng(seed + 1)
        worlds = [World(rng) for _ in range(n)]
        col = np.zeros(n, bool); mg = np.full(n, 1e9); prev_f, S = None, None
        for t in range(T):
            f = torch.as_tensor(np.stack([w.render() for w in worlds])).float() / 255.0
            if p in ("slot_full", "slot_nofuture"):
                S = wm.enc.step(f, S)
            if p in ("dense", "slot_full", "slot_nofuture"):
                acts = policy_actions(p, wm, planners, aemb, f, f if prev_f is None else prev_f, S).numpy()
            elif p == "keep": acts = np.full(n, KEEP)
            elif p == "random": acts = prng.integers(0, NA, n)
            else: acts = np.array([int(np.argmin(oracle_costs(w))) for w in worlds])
            prev_f = f
            for i, w in enumerate(worlds):
                col[i] |= w.step(int(acts[i])); mg[i] = min(mg[i], w.min_gap())
        res[p] = dict(collision_rate=round(float(col.mean()), 3), progress_m=round(float(np.mean([w.xe for w in worlds])), 1), min_gap_m=round(float(mg.mean()), 2))
    return res
