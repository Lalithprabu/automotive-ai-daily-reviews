import argparse, json, time, yaml, copy, torch, torch.nn.functional as F
from emplan.data import make_dataset
from emplan.model import EMPlan, RegressionPlanner, kmeans_anchors
from emplan.reward import rule_reward, rule_terms


def evaluate(plan_fn, scene, expert, dt):
    traj = plan_fn(scene)
    r = rule_terms(scene, traj[:, None], dt)
    ade = (traj - expert).norm(dim=-1).mean().item()
    return dict(reward=rule_reward(scene, traj[:, None], dt).mean().item(),
                collision_rate=r["collision"].mean().item(), offroad_rate=r["offroad"].mean().item(),
                progress=r["progress"].mean().item(), ade_vs_expert=ade)


def stage1(model, scene, expert, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"]); n = len(scene)
    for ep in range(cfg["stage1_epochs"]):
        perm = torch.randperm(n); tot = 0
        for i in range(0, n, cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]; s, e = scene[b], expert[b]
            logits, trajs = model(s)
            near = (model.anchors[None] - e[:, None]).flatten(2).norm(dim=-1).argmin(1)
            pick = trajs[torch.arange(len(b)), near]
            loss = F.cross_entropy(logits, near) + F.l1_loss(pick, e)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        if ep % 10 == 0 or ep == cfg["stage1_epochs"] - 1: print(f"[s1] ep{ep} loss {tot/n:.4f}")


def stage2(model, scene, expert, cfg, dt):
    """Reward-guided UNPAIRED preference optimisation (KTO-flavoured; this repo's own formulation).
    Each candidate k gets an independent good(+1)/bad(-1) label from the rule reward - no (winner, loser) pairs.
    Loss: -log sigmoid(beta * y_k * (logp_theta(k) - logp_ref(k))) with a frozen stage-1 reference."""
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["stage2_lr"]); n = len(scene)
    for ep in range(cfg["stage2_epochs"]):
        perm = torch.randperm(n); tot = 0; fg = 0
        for i in range(0, n, cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]; s, e = scene[b], expert[b]
            logits, trajs = model(s)
            with torch.no_grad():
                _, rtraj = ref(s)
                r = rule_reward(s, trajs.detach(), dt)                       # reward of the candidates actually produced
                y = torch.where(r > r.max(1, keepdim=True).values - 3.0 + cfg["good_margin"], 1.0, -1.0)
                y = torch.where(rule_terms(s, trajs.detach(), dt)["collision"] > 0, -1.0, y)  # collisions always 'bad'
                ref_lp = F.log_softmax(ref(s)[0], -1)
            lp = F.log_softmax(logits, -1)
            pref = -F.logsigmoid(cfg["beta"] * y * (lp - ref_lp)).mean()
            near = (model.anchors[None] - e[:, None]).flatten(2).norm(dim=-1).argmin(1)
            bc = F.l1_loss(trajs[torch.arange(len(b)), near], e)             # keep offsets anchored to demos
            loss = pref + cfg["bc_weight"] * bc
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b); fg += (y > 0).float().mean().item() * len(b)
        if ep % 5 == 0 or ep == cfg["stage2_epochs"] - 1: print(f"[s2] ep{ep} loss {tot/n:.4f} good-label-frac {fg/n:.2f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); torch.manual_seed(cfg["seed"])
    T, dt = cfg["data"]["horizon"], cfg["data"]["dt"]
    scene, expert = make_dataset(cfg["data"]["n_train"], cfg["seed"], T, dt)
    vs, ve = make_dataset(cfg["data"]["n_val"], cfg["seed"] + 1, T, dt)
    t0 = time.time(); res = {}
    reg = RegressionPlanner(T, cfg["model"]["hidden"]); opt = torch.optim.AdamW(reg.parameters(), lr=cfg["train"]["lr"])
    for ep in range(cfg["train"]["stage1_epochs"]):
        perm = torch.randperm(len(scene))
        for i in range(0, len(scene), cfg["train"]["batch_size"]):
            b = perm[i:i + cfg["train"]["batch_size"]]
            loss = F.l1_loss(reg(scene[b]), expert[b]); opt.zero_grad(); loss.backward(); opt.step()
    res["old_regression"] = evaluate(reg.plan, vs, ve, dt); print("old", res["old_regression"])
    anchors = kmeans_anchors(expert, cfg["model"]["num_anchors"], seed=cfg["seed"])
    m = EMPlan(anchors, cfg["model"]["hidden"]); stage1(m, scene, expert, cfg["train"])
    res["emplan_stage1_only"] = evaluate(lambda s: m.plan(s)[0], vs, ve, dt); print("stage1", res["emplan_stage1_only"])
    stage2(m, scene, expert, cfg["train"], dt)
    res["emplan_stage2"] = evaluate(lambda s: m.plan(s)[0], vs, ve, dt); print("stage2", res["emplan_stage2"])
    res["expert_reference"] = evaluate(lambda s: ve, vs, ve, dt)
    res["params"] = dict(regression=sum(p.numel() for p in reg.parameters()), emplan=sum(p.numel() for p in m.parameters()))
    res["train_seconds"] = time.time() - t0
    torch.save(dict(model=m.state_dict(), anchors=anchors, cfg=cfg), "emplan_ckpt.pt")
    torch.save(dict(model=reg.state_dict(), cfg=cfg), "regression_ckpt.pt")
    json.dump(res, open("results.json", "w"), indent=2); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
