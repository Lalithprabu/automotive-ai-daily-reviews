"""'AI in action' synthetic simulation:  python simulate.py --seed 11 --out assets/interact_demo
Needs checkpoints/{wm,planners}.pt from train.py. Writes .gif + .mp4 + .json.

Left  : the BEV 'camera' feed the model actually sees (48x48, upscaled). INPUT = the feed.
        Coloured boxes + ids = what the slot encoder has bound to each object (bounding box of each slot's decoded mask).
        PREDICTION  = red dashed path of each slot's centroid over the next 1.5 s, rolled out by the world model
                      conditioned on the *chosen* latent ego action.
        GROUND TRUTH = green solid path: where that object really goes when the simulator executes the same action.
Right : candidate-action costs (planner vs privileged oracle), ego telemetry, centroid prediction error.
"""
import argparse, json, warnings
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle
from interact.synthetic import World, oracle_costs, PROTO_NAMES, NA, XR, EGO_L, EGO_W, NAG
from interact.model import WorldModel, SlotPlanner, candidate_futures

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=11); ap.add_argument("--out", default="assets/interact_demo")
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--mock", action="store_true", help="ILLUSTRATIVE mode: simulator boxes + constant-velocity predictor + oracle action (no learned model; use this if checkpoints/ is untrained)")
a = ap.parse_args()
torch.set_num_threads(2)
wm = WorldModel(); wm.load_state_dict(torch.load("checkpoints/wm.pt")); wm.eval()
ck = torch.load("checkpoints/planners.pt"); aemb = ck["aemb"]
pl = SlotPlanner(use_future=True); pl.load_state_dict(ck["slot_full"]); pl.eval()


def centroids(masks):
    """masks [K,H,W] -> list of (row,col,area,bbox) or None per slot; background slot (largest area) -> None."""
    out, areas = [], [(m > 0.5).sum().item() for m in masks]
    bg = int(np.argmax(areas))
    for k, m in enumerate(masks):
        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        if k == bg or len(ys) < 3 or len(ys) > 500: out.append(None); continue
        out.append((ys.float().mean().item(), xs.float().mean().item(), len(ys), (xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item())))
    return out


# choose a scenario with a cut-in in the first 8 steps (so interaction is visible); NOT chosen for model success
s = a.seed
while True:
    probe = World(np.random.default_rng(s)); ci = False
    for t in range(8): probe.step(2); ci |= probe.cutin_now
    if ci: break
    s += 1
w = World(np.random.default_rng(s)); S = None; rec = []; prev_ag = None
with torch.no_grad():
    for t in range(a.steps):
        f = torch.as_tensor(w.render()).float()[None] / 255.0
        S = wm.enc.step(f, S)
        _, m0 = wm.dec(S); cur = centroids(m0[0])
        fut = candidate_futures(wm, S, aemb, 3)                                              # [1,A,3,K,d]
        cost = pl(S, fut[:, :, [0, 2]], aemb)[0]; act = int(cost.argmin()); oc = oracle_costs(w)
        pm = []
        for n in range(3):
            _, mm = wm.dec(fut[0, act, n][None]); pm.append(centroids(mm[0]))              # predicted masks at +1,+2,+3
        ag0 = w.agent_pixels(); g = w.clone(); gt = []
        for n in range(3): g.step(act); gt.append(g.agent_pixels())                        # ground truth under the same action
        tracks = []
        for k, c in enumerate(cur):
            if c is None: continue
            j = int(np.argmin(((ag0 - np.array([c[0], c[1]])) ** 2).sum(1)))               # associate slot -> true agent (eval only)
            pp = [(c[0], c[1])] + [(p[k][0], p[k][1]) if p[k] is not None else None for p in pm]
            gp = [(ag0[j][0], ag0[j][1])] + [(q[j][0], q[j][1]) for q in gt]
            tracks.append(dict(k=k, bbox=c[3], pred=pp, gt=gp))
        errs = [np.hypot((p[n][0] - q[n][0]), (p[n][1] - q[n][1]) * 0.5) for tr in tracks for p, q in [(tr["pred"], tr["gt"])] for n in (3,) if p[n] is not None]
        if a.mock:                                              # ILLUSTRATIVE ONLY: ground-truth boxes, constant-velocity 'prediction', oracle action
            act = int(np.argmin(oc)); cost = torch.as_tensor(oc, dtype=torch.float32)
            g = w.clone(); gt = []
            for n in range(3): g.step(act); gt.append(g.agent_pixels())
            sizes = [(EGO_L, EGO_W)] + [(w.L[i], w.W[i]) for i in range(NAG)]; tracks = []
            for j in range(NAG + 1):
                p0 = ag0[j]; d = (ag0[j] - prev_ag[j]) if prev_ag is not None else np.zeros(2); L_, W_ = sizes[j]
                tracks.append(dict(k=j, bbox=(p0[1] - W_ / 1.0, p0[0] - L_ / 2, p0[1] + W_ / 1.0, p0[0] + L_ / 2),
                                   pred=[tuple(p0)] + [tuple(p0 + d * (n + 1)) for n in range(3)], gt=[tuple(p0)] + [tuple(q[j]) for q in gt]))
            errs = [np.hypot(tr["pred"][3][0] - tr["gt"][3][0], (tr["pred"][3][1] - tr["gt"][3][1]) * 0.5) for tr in tracks]
        prev_ag = ag0
        rec.append(dict(img=w.render().transpose(1, 2, 0), cost=cost.numpy(), oc=oc, act=act, tracks=tracks, ve=w.ve, gap=w.min_gap(),
                        err=float(np.mean(errs)) if errs else np.nan, cutin=False))
        col = w.step(act); rec[-1]["cutin"] = w.cutin_now; rec[-1]["col"] = col
T = len(rec); cols = plt.cm.tab10(np.arange(7))
fig = plt.figure(figsize=(12.4, 6.4), dpi=100)
gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], left=.04, right=.985, top=.84, bottom=.08, hspace=.8, wspace=.18)
axm, axc, axt, axe = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])
fig.text(.04, .94, ("ILLUSTRATIVE MOCK-UP (no learned model): simulator boxes, constant-velocity prediction, oracle action" if a.mock else "AI in action: object-centric world model sees slots, imagines the next 1.5 s, picks the ego action (synthetic)"), fontsize=11.5, weight="bold")
sub = fig.text(.04, .895, "", fontsize=9, family="monospace")
fig.text(.985, .002, f"Synthetic demo - INTERACT-style re-implementation on my toy BEV world (seed {s}), NOT NAVSIM / nuScenes, NOT the paper's results" + (" - MOCK MODE" if a.mock else ""), ha="right", fontsize=7.5, color="#555")


def draw(fi):
    t = min(fi, T - 1); r = rec[t]
    axm.cla(); axm.imshow(np.clip(r["img"] * 1.0, 0, 255).astype(np.uint8), interpolation="nearest"); axm.set_xticks([]); axm.set_yticks([])
    for yl in (-5.25, -1.75, 1.75, 5.25): axm.axvline((yl + 12) / .5 - .5, color="w", lw=.6, ls=":", alpha=.5)
    for tr in r["tracks"]:
        x0, y0, x1, y1 = tr["bbox"]; c = cols[tr["k"] % 7]
        axm.add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0 + 1, y1 - y0 + 1, fill=False, ec=c, lw=1.6)); axm.text(x0 - .5, y0 - 1, f"s{tr['k']}", color=c, fontsize=7, weight="bold")
        g = np.array(tr["gt"]); axm.plot(g[:, 1], g[:, 0], color="#2ecc71", lw=2.2)
        p = [q for q in tr["pred"] if q is not None]; p = np.array(p); axm.plot(p[:, 1], p[:, 0], color="#ff4d4d", lw=1.8, ls="--", marker="o", ms=3)
    axm.plot([], [], c="#ff4d4d", ls="--", label=("PREDICTION (MOCK: constant velocity)" if a.mock else "PREDICTION (world model, chosen action)")); axm.plot([], [], c="#2ecc71", lw=2.2, label="GROUND TRUTH (same action)")
    axm.plot([], [], c="w", lw=0, label=("boxes = simulator ground truth (MOCK)" if a.mock else "boxes = slot masks (INPUT -> objects)"))
    axm.legend(fontsize=6.5, loc="upper left", framealpha=.85); axm.set_title("INPUT feed (red car / green truck / blue ego) + slots + prediction vs truth", fontsize=8.5)
    axc.cla(); x = np.arange(NA); axc.bar(x - .18, r["cost"], .36, color=["#1f5fd1" if i == r["act"] else "#9bb7ea" for i in x], label=("oracle cost (MOCK: no planner)" if a.mock else "planner cost (lower = better)"))
    axc.bar(x + .18, r["oc"], .36, color="#2ecc71", alpha=.6, label="oracle cost (privileged)"); axc.set_xticks(x); axc.set_xticklabels(PROTO_NAMES, fontsize=7)
    axc.tick_params(labelsize=7); axc.legend(fontsize=6.5); axc.set_title(f"candidate ego actions -> chose '{PROTO_NAMES[r['act']]}'", fontsize=8.5)
    ts = np.arange(t + 1); axt.cla(); axt.plot(ts, [q["ve"] for q in rec[:t + 1]], c="#1f5fd1", lw=2, label="ego speed (m/s)")
    axt.plot(ts, [min(q["gap"], 30) / 3 for q in rec[:t + 1]], c="#e67e22", lw=2, label="nearest gap (m) / 3")
    for i in range(t + 1):
        if rec[i]["cutin"]: axt.axvline(i, c="#c0392b", alpha=.4, lw=3)
        if rec[i]["col"]: axt.axvline(i, c="k", lw=2)
    axt.set_xlim(0, T); axt.set_ylim(0, 18); axt.legend(fontsize=6.5, loc="upper right"); axt.tick_params(labelsize=7); axt.grid(alpha=.3)
    axt.set_title("telemetry (red band = a car cut in; black = collision)", fontsize=8.5)
    axe.cla(); axe.plot(ts, [q["err"] for q in rec[:t + 1]], c="#c0392b", lw=2, marker="o", ms=3); axe.set_xlim(0, T); axe.set_ylim(0, 8); axe.grid(alpha=.3); axe.tick_params(labelsize=7)
    axe.set_xlabel("frame (0.5 s)", fontsize=8); axe.set_title("mean centroid error of the +1.5 s prediction (m)", fontsize=8.5)
    sub.set_text(f"frame {t:02d}  slots tracked {len(r['tracks'])}  ego {r['ve']:.1f} m/s  action '{PROTO_NAMES[r['act']]}'  oracle best '{PROTO_NAMES[int(np.argmin(r['oc']))]}'  pred err {r['err']:.2f} m")
    return []


anim = FuncAnimation(fig, draw, frames=T + 3, interval=300, blit=False)
anim.save(a.out + ".gif", writer=PillowWriter(fps=3)); anim.save(a.out + ".mp4", writer=FFMpegWriter(fps=3, bitrate=2200))
summ = dict(seed=s, frames=T, collisions=int(sum(q["col"] for q in rec)), cutins=int(sum(q["cutin"] for q in rec)),
            agree_with_oracle=round(float(np.mean([q["act"] == int(np.argmin(q["oc"])) for q in rec])), 3),
            mean_pred_err_m=round(float(np.nanmean([q["err"] for q in rec])), 2))
json.dump(summ, open(a.out + ".json", "w"), indent=1); print(summ)
