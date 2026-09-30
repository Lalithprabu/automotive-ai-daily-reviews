"""Train: (1) frozen world model, (2) trajectory AE (with/without REPA alignment), (3) new-way latent planner vs old-way direct planner."""
import json, yaml, torch, torch.nn.functional as F, time, sys
from torch.utils.data import DataLoader
from walt.data import Scenes, norm_wp, denorm_wp
from walt.model import FrozenWorldModel, TrajAutoencoder, WALTPlanner, DirectPlanner

def fit(params, loss_fn, loader, epochs, lr, tag):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(loader))
    for ep in range(epochs):
        tot = 0
        for b in loader:
            loss = loss_fn(*b); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if ep in (0, epochs - 1): print(f"[{tag}] epoch {ep+1}/{epochs} loss {tot/len(loader):.4f}", flush=True)

def wm_loss(pred, tgt):
    # sparse rasters: plain MSE lets the world model ignore the thin lead-vehicle channel (found empirically:
    # tokens carried ~no vehicle info, planners stuck at mean-baseline). Up-weight occupied cells.
    w = 1.0 + 30.0 * tgt
    return (w * (pred - tgt) ** 2).mean() * 10

def build_ae(cfg, wm, tl, semantic=True, repa=True):
    ae = TrajAutoencoder(use_semantic=semantic)
    def loss(x, xf, wp):
        with torch.no_grad(): ft = wm.future_tokens(wm.tokens(x))
        w = norm_wp(wp); z = ae.encode(w, ft)
        l = F.smooth_l1_loss(ae.decode(z), w, beta=0.05)
        if repa: l = l + cfg["repa_weight"] * ae.align_loss(z, ft)
        return l
    fit(ae.parameters(), loss, tl, cfg["ae_epochs"], cfg["lr"], f"AE sem={semantic} repa={repa}")
    for p in ae.parameters(): p.requires_grad_(False)
    return ae.eval()

def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path)); torch.manual_seed(cfg["seed"]); torch.set_num_threads(2)
    tr, va = Scenes(cfg["n_train"], 1), Scenes(cfg["n_val"], 2)
    tl = DataLoader(tr, cfg["batch_size"], shuffle=True)
    t0 = time.time()
    wm = FrozenWorldModel()
    fit(wm.parameters(), lambda x, xf, wp: wm_loss(wm(x), xf), tl, cfg["wm_epochs"], cfg["lr"], "WorldModel")
    for p in wm.parameters(): p.requires_grad_(False)
    wm.eval()
    ae = build_ae(cfg, wm, tl, True, True)          # WALT latent space (aligned)
    ae_plain = build_ae(cfg, wm, tl, False, False)  # ablation: geometry-only latent, no alignment

    def train_planner(model, is_latent, tag):
        def loss(x, xf, wp):
            w = norm_wp(wp); pred, *rest = model(x) if is_latent else (model(x),)
            l = F.smooth_l1_loss(pred, w, beta=0.05)
            return l
        fit(model.planner.parameters(), loss, tl, cfg["planner_epochs"], cfg["lr"], tag)
        return model.eval()
    new = train_planner(WALTPlanner(wm, ae), True, "planner NEW (WALT latent)")
    abl = train_planner(WALTPlanner(wm, ae_plain), True, "planner ABL (geo-only latent)")
    old = train_planner(DirectPlanner(wm), False, "planner OLD (raw waypoints)")

    @torch.no_grad()
    def evaluate(model, is_latent):
        ades, fdes, per = [], [], []
        for x, xf, wp in DataLoader(va, 256):
            p = model(x); p = p[0] if is_latent else p
            e = (denorm_wp(p) - wp).norm(dim=-1); ades.append(e.mean(1)); fdes.append(e[:, -1])
        return torch.cat(ades), torch.cat(fdes)
    res = {}
    for name, m, il in [("old_direct", old, False), ("new_walt", new, True), ("ablation_geo_only_latent", abl, True)]:
        a, f = evaluate(m, il); res[name] = {"ADE": a.mean().item(), "FDE": f.mean().item()}
    # bucket: scenes with a lead vehicle (interaction-heavy) vs free road
    lead = torch.tensor([m[0] for m in va.meta])
    for name, m, il in [("old_direct", old, False), ("new_walt", new, True)]:
        a, _ = evaluate(m, il)
        res[name]["ADE_lead"] = a[lead].mean().item(); res[name]["ADE_free"] = a[~lead].mean().item()
    res["params_planner"] = {"old": sum(p.numel() for p in old.planner.parameters()),
                             "new": sum(p.numel() for p in new.planner.parameters())}
    res["train_seconds"] = time.time() - t0
    json.dump(res, open("results.json", "w"), indent=2); print(json.dumps(res, indent=2))
    torch.save({"wm": wm.state_dict(), "ae": ae.state_dict(), "new": new.planner.state_dict(),
                "old": old.planner.state_dict()}, "checkpoint.pt")

if __name__ == "__main__": main(*sys.argv[1:])
