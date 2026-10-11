"""End-to-end: data -> train diffusion backbone -> distill SCOPE head -> calibrate -> evaluate -> save."""
import argparse, json, time, math
import numpy as np, torch, yaml
from scope import *
from scope.evaluate import fit_alpha, metrics

torch.set_num_threads(2)


def anchors_for(model, split, n, cfg):
    s = cfg["sampler"]
    return sample(model, split["cond"][:n], split["mode_t"][:n], None, s["steps"], s["sigma_max"], s["sigma_min"])


def main(cfg_path="config.yaml", out="checkpoint.pt", quick=False):
    cfg = yaml.safe_load(open(cfg_path))
    if quick:
        cfg["backbone"]["steps"], cfg["head"]["steps"], cfg["head"]["anchors"] = 150, 60, 512
        cfg["data"].update(n_train=1500, n_val=300, n_test=300)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    data = make_data(cfg, cfg["seed"])
    tr, va, te = data["train"], data["val"], data["test"]
    b = cfg["backbone"]
    model = Denoiser(b["hidden"], b["depth"], b["sigma_data"])
    opt = torch.optim.Adam(model.parameters(), lr=b["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, b["steps"], eta_min=b["lr"] * 0.02)
    N = tr["cond"].shape[0]; t0 = time.time()
    for it in range(b["steps"]):
        idx = torch.randint(0, N, (b["batch"],))
        loss = edm_loss(model, tr["resid"][idx], tr["cond"][idx], tr["mode_t"][idx], b["p_mean"], b["p_std"])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % max(1, b["steps"] // 10) == 0:
            print(f"[backbone] {it:5d} loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    # ---- distil SCOPE head ----
    h = cfg["head"]; s0 = h["sigma0"]
    head = PrecisionHead(s0, h["hidden"], h["bandwidth"], h["rank"])
    na = min(h["anchors"], N)
    anc = anchors_for(model, tr, na, cfg)
    c_a, m_a = tr["cond"][:na], tr["mode_t"][:na]
    with torch.no_grad() if False else torch.enable_grad():
        Pt = dense_target(model, anc, c_a, m_a, s0)
    Sig_t, ld_P = psd_inverse(Pt)
    ld_Sig_t = -ld_P
    opt = torch.optim.Adam(head.parameters(), lr=h["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, h["steps"], eta_min=h["lr"] * 0.02)
    for it in range(h["steps"]):
        idx = torch.randint(0, na, (h["batch"] if not quick else 64,))
        loss, kl = distill_loss(head, model, anc[idx], c_a[idx], m_a[idx], h["probes"], Sig_t[idx], ld_Sig_t[idx], h["kl_weight"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0); opt.step(); sched.step()
        if it % max(1, h["steps"] // 10) == 0:
            print(f"[head] {it:5d} loss {loss.item():.4f}  KL {kl.item():.3f}")
    head.eval()

    # ---- calibration (val) + evaluation (test) ----
    def prep(split):
        a = anchors_for(model, split, split["cond"].shape[0], cfg)
        return a, split["resid"] - a
    a_va, e_va = prep(va); a_te, e_te = prep(te)
    with torch.no_grad():
        S_va = head.covariance(va["cond"], va["mode_t"], a_va, calibrated=False)
        alpha = fit_alpha(S_va, e_va)
        head.log_scale.fill_(math.log(alpha))
        S_te = head.covariance(te["cond"], te["mode_t"], a_te)
        e_tr = tr["resid"][:na] - anc
        S_fixed = torch.cov(e_tr.T)
        fix = lambda n: S_fixed.expand(n, D, D)
        al_fix = fit_alpha(fix(len(e_va)), e_va)
        K = cfg["mc_baseline"]["samples"]
        def mc_cov(split, anchor, n):
            out = []
            for i in range(n):
                c = split["cond"][i:i+1].expand(K, -1); m = split["mode_t"][i:i+1].expand(K, -1)
                x = sample(model, c, m, torch.randn(K, D), cfg["sampler"]["steps"], cfg["sampler"]["sigma_max"], cfg["sampler"]["sigma_min"])
                out.append(torch.cov(x.T) + 1e-4 * torch.eye(D))
            return torch.stack(out)
        n_mc = min(500, len(e_te))
        S_mc_va = mc_cov(va, a_va, n_mc); al_mc = fit_alpha(S_mc_va, e_va[:n_mc])
        S_mc_te = mc_cov(te, a_te, n_mc)
        S_or_te = torch.tensor(te["cov"], dtype=torch.float32)
        S_or_va = torch.tensor(va["cov"], dtype=torch.float32); al_or = fit_alpha(S_or_va, e_va)
        # curvature fidelity: distilled-from-diffusion target vs known truth (diagnostic)
        Pt_te = dense_target(model, a_te[:300], te["cond"][:300], te["mode_t"][:300], s0)
        S_tgt, _ = psd_inverse(Pt_te); S_tgt = S_tgt * s0 ** 2
    res = {
        "SCOPE (calibrated)": metrics(S_te, e_te),
        "Fixed covariance": metrics(fix(len(e_te)), e_te, al_fix),
        f"MC sampling (K={K})": metrics(S_mc_te, e_te[:n_mc], al_mc),
        "Oracle (true noise cov)": metrics(S_or_te, e_te, al_or),
        "Undistilled curvature (dense Hessian target)": metrics(S_tgt, e_te[:300], fit_alpha(S_tgt, e_te[:300])),
        "SCOPE (uncalibrated, alpha=1)": metrics(head.covariance(te["cond"], te["mode_t"], a_te, calibrated=False), e_te),
    }
    # latency (batch of 1, CPU, ms)
    def timeit(fn, n=30):
        fn(); t = time.perf_counter()
        for _ in range(n): fn()
        return (time.perf_counter() - t) / n * 1e3
    c1, m1, a1 = te["cond"][:1], te["mode_t"][:1], a_te[:1]
    with torch.no_grad():
        lat = {"scope_head_plus_solve_ms": timeit(lambda: head.covariance(c1, m1, a1)),
               "mc_sampling_ms": timeit(lambda: mc_cov(te, a_te, 1), 10),
               "nominal_decode_ms (shared by all)": timeit(lambda: sample(model, c1, m1, None, cfg["sampler"]["steps"]), 10)}
    out_json = {"alpha": {"scope": alpha, "fixed": al_fix, "mc": al_mc, "oracle": al_or}, "test_metrics": res, "latency_ms": lat,
                "params": {"backbone": sum(p.numel() for p in model.parameters()), "scope_head": sum(p.numel() for p in head.parameters())},
                "quick": quick}
    print(json.dumps(out_json, indent=1))
    json.dump(out_json, open("results.json" if not quick else "results_quick.json", "w"), indent=1)
    torch.save({"cfg": cfg, "model": model.state_dict(), "head": head.state_dict(), "S_fixed": S_fixed, "al_fix": al_fix, "al_mc": al_mc}, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="checkpoint.pt"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); main(a.config, a.out, a.quick)
