"""Open-loop + closed-loop + 'binding-gap' evaluation. Same weights, guidance toggled."""
import json, sys, yaml, numpy as np, torch
from models.flow_model import FlowNet, sample
from src.scene import make_context, decode, box_sdf, inflated_box, stopping_distance
from src.closed_loop import run_episode

def load(cfg):
    net = FlowNet(cfg["horizon"], 7, cfg["hidden"], cfg["depth"]); net.load_state_dict(torch.load(cfg["checkpoint"])); return net.eval()

def intrusion(traj_m, d):
    sdf = box_sdf(traj_m[..., 0], traj_m[..., 1], inflated_box(d[:, None]))
    return (sdf < 0).any(1)

@torch.no_grad()
def _nograd(f, *a, **k): return f(*a, **k)

def open_loop(net, cfg, n=400, seed=7):
    rng = np.random.default_rng(seed)
    v0 = torch.tensor(rng.uniform(8, 14, n), dtype=torch.float32)
    # braking-relevant & feasible: enough stopping distance at 5 m/s^2 plus 8 m
    d = torch.tensor([float(stopping_distance(v, 5.0) + EG + rng.uniform(0, 14)) for v, EG in zip(v0.tolist(), [8.0] * n)])
    c = make_context(v0[:, None].repeat(1, 4), d, torch.ones(n, dtype=torch.bool))
    out = {}
    for name, lam in [("baseline", 0.0), ("guided", cfg["guide_lambda"])]:
        g = torch.Generator().manual_seed(seed)
        tr = decode(sample(net, c, d, torch.ones(n, dtype=torch.bool), cfg["sampler_steps"], lam, cfg["guide_margin"], generator=g))
        out[name] = dict(intrusion=float(intrusion(tr, d).float().mean()),
                         mean_final_s=float(tr[:, -1, 0].mean()), lateral_abs=float(tr[..., 1].abs().mean()))
    return out

@torch.no_grad()
def far_hazard_progress(net, cfg, n=200, seed=5):
    """Over-conservatism check: hazard is FAR (>= 60 m) so no braking is needed; guided plan should not lose progress."""
    rng = np.random.default_rng(seed); v0 = torch.tensor(rng.uniform(8, 13, n), dtype=torch.float32)
    d = torch.tensor(rng.uniform(60, 90, n), dtype=torch.float32); pr = torch.ones(n, dtype=torch.bool)
    c = make_context(v0[:, None].repeat(1, 4), d, pr); out = {}
    for name, lam in [("baseline", 0.0), ("guided", cfg["guide_lambda"])]:
        g = torch.Generator().manual_seed(seed)
        out[name] = float(decode(sample(net, c, d, pr, cfg["sampler_steps"], lam, cfg["guide_margin"], generator=g))[:, -1, 0].mean())
    return out

def binding_gap(net, cfg, n=300, seed=11):
    """Finite-difference sensitivity of plan length (final s) to hazard distance vs ego speed, in comparable relative units."""
    rng = np.random.default_rng(seed); res = {}
    v0 = torch.tensor(rng.uniform(8, 13, n), dtype=torch.float32); d = torch.tensor(rng.uniform(25, 45, n), dtype=torch.float32)
    pr = torch.ones(n, dtype=torch.bool)
    for name, lam in [("baseline", 0.0), ("guided", cfg["guide_lambda"])]:
        def fs(v, dd):
            g = torch.Generator().manual_seed(seed)
            c = make_context(v[:, None].repeat(1, 4), dd, pr)
            return decode(sample(net, c, dd, pr, cfg["sampler_steps"], lam, cfg["guide_margin"], generator=g))[:, -1, 0]
        base = fs(v0, d)
        dv = (fs(v0 * 1.1, d) - base).abs().mean()              # +10% ego speed
        dd = (fs(v0, d * 0.8) - base).abs().mean()             # hazard 20% closer
        res[name] = dict(d_plan_per_speed_pct10=float(dv), d_plan_per_hazard_20pct_closer=float(dd),
                         hazard_to_speed_ratio=float(dd / (dv + 1e-6)))
    return res

def closed_loop(net, cfg, n=40, seed=3):
    rng = np.random.default_rng(seed); res = {"baseline": [], "guided": []}
    for i in range(n):
        v0 = float(rng.uniform(8, 13)); D = float(stopping_distance(v0, 5.0) + 5.0 + rng.uniform(4, 18))
        for name, lam in [("baseline", 0.0), ("guided", cfg["guide_lambda"])]:
            r = run_episode(net, v0, D, lam, seed=1000 + i, steps=cfg["sampler_steps"], margin=cfg["guide_margin"])
            res[name].append((r["collided"], r["final_gap"]))
    return {k: dict(collision_rate=float(np.mean([c for c, _ in v])), mean_final_gap_m=float(np.mean([g for _, g in v])), n=len(v)) for k, v in res.items()}

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml")); torch.manual_seed(0); net = load(cfg)
    r = dict(open_loop=open_loop(net, cfg), binding_gap=binding_gap(net, cfg), far_hazard_mean_final_s=far_hazard_progress(net, cfg), closed_loop=closed_loop(net, cfg, n=int(sys.argv[1]) if len(sys.argv) > 1 else 40))
    print(json.dumps(r, indent=2)); json.dump(r, open("assets/results.json", "w"), indent=2)
