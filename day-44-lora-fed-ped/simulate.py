"""Renders assets/simulation.gif: input history vs ground truth vs zero-shot base ("old") vs LoRA-FL model ("new"),
plus live ADE telemetry and the communication-cost panel (full FedAvg vs LoRA FedAvg).
Usage: python simulate.py --run runs --scenes 6 --out assets/simulation.gif"""
import argparse, json, io, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from src.model import TrajTransformer
from src.lora import inject_lora
from src.data import make_target_val

BG, FG = "#0e1117", "#e6e6e6"

def load(run):
    ck = torch.load(f"{run}/lora_fl.pt", weights_only=False); c = ck["cfg"]
    mk = lambda: TrajTransformer(c["obs_len"], c["pred_len"], c["d_model"], c["n_heads"], c["n_layers"])
    base = mk(); base.load_state_dict(ck["base"]); base.eval()
    lora = mk(); inject_lora(lora, set(c["lora_targets"]), c["lora_rank"], c["lora_alpha"])
    lora.load_state_dict(ck["lora"]); lora.eval()
    return base, lora, c

def frame(obs, fut, pb, pl, k, ade_b, ade_l, res, scene_i, n_scenes):
    O = len(obs); fig = plt.figure(figsize=(15, 5.2), facecolor=BG)
    ax = fig.add_axes([0.03, 0.1, 0.36, 0.78]); ax2 = fig.add_axes([0.45, 0.14, 0.24, 0.7]); ax3 = fig.add_axes([0.75, 0.14, 0.23, 0.7])
    for a in (ax, ax2, ax3):
        a.set_facecolor("#161b22"); a.tick_params(colors=FG, labelsize=8)
        for s in a.spines.values(): s.set_color("#444")
    allp = np.concatenate([obs, fut, pb, pl]); cx, cy = allp.mean(0); r = max(4.0, np.abs(allp - [cx, cy]).max() + 0.8)
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r); ax.set_aspect("equal")
    for i in range(-6, 7):  # zebra crosswalk stripes for context
        ax.axhspan(cy - r + 0.0, cy - r + 0.0, alpha=0)
    ax.plot(obs[:, 0], obs[:, 1], "o-", c="#dddddd", ms=4, label="Input: observed history")
    ax.plot(fut[:k + 1, 0], fut[:k + 1, 1], "o-", c="#3fb950", ms=4, label="Ground truth")
    ax.plot(pb[:k + 1, 0], pb[:k + 1, 1], "--", c="#58a6ff", lw=2, label=f"Old: frozen base  ADE {ade_b:.2f}m")
    ax.plot(pl[:k + 1, 0], pl[:k + 1, 1], "-", c="#f85149", lw=2.5, label=f"New: LoRA-FL  ADE {ade_l:.2f}m")
    ax.plot(*obs[-1], "*", c="#f2cc60", ms=14)
    ax.set_title(f"Pedestrian curb-crossing forecast  (scene {scene_i}/{n_scenes}, t+{(k+1)*0.4:.1f}s)", color=FG, fontsize=10)
    ax.legend(loc="lower left", fontsize=7, facecolor=BG, labelcolor=FG, framealpha=0.8)
    # telemetry: accuracy per round
    fr, lr_ = res["full_fl"], res["lora_fl"]
    ax2.plot([x["round"] for x in fr], [x["ade"] for x in fr], "o-", c="#d29922", label="Full-model FedAvg")
    ax2.plot([x["round"] for x in lr_], [x["ade"] for x in lr_], "s-", c="#f85149", label="LoRA FedAvg")
    ax2.axhline(res["zero_shot"]["ade"], c="#58a6ff", ls="--", label="Frozen base (no FL)")
    ax2.set_title("Held-out ADE (m) per comm. round", color=FG, fontsize=10); ax2.set_xlabel("round", color=FG)
    ax2.legend(fontsize=7, facecolor=BG, labelcolor=FG)
    # telemetry: cumulative radio cost
    ax3.plot([x["round"] for x in fr], [x["resource_blocks"] for x in fr], "o-", c="#d29922", label="Full-model")
    ax3.plot([x["round"] for x in lr_], [x["resource_blocks"] for x in lr_], "s-", c="#f85149", label="LoRA")
    ax3.set_yscale("log"); ax3.set_title("Cumulative radio resource blocks (V2I)", color=FG, fontsize=10); ax3.set_xlabel("round", color=FG)
    ax3.legend(fontsize=7, facecolor=BG, labelcolor=FG)
    ax3.text(0.02, 0.04, f"payload/round: {res['full_payload_params']:,} -> {res['lora_payload_params']:,} params", transform=ax3.transAxes, color=FG, fontsize=7)
    fig.text(0.5, 0.965, "SYNTHETIC DATA - reconstruction of arXiv:2610.18640 idea, not the paper's results", color="#f2cc60", ha="center", fontsize=9)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=70, facecolor=BG); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runs"); ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--out", default="assets/simulation.gif"); a = ap.parse_args()
    base, lora, c = load(a.run); res = json.load(open(f"{a.run}/results.json"))
    O, P = c["obs_len"], c["pred_len"]
    vo, vf, _ = make_target_val(200, O, P, c["n_clients"], c["seed"])
    with torch.no_grad(): pb, pl = base(vo), lora(vo)
    eb, el = (pb - vf).norm(dim=-1).mean(1), (pl - vf).norm(dim=-1).mean(1)
    # show scenes where the adaptation matters most, spread across the set
    order = torch.argsort(eb - el, descending=True)[: a.scenes * 3][:: 3][: a.scenes]
    frames = []
    for si, i in enumerate(order.tolist(), 1):
        for k in range(1, P, 2):
            frames.append(frame(vo[i].numpy(), vf[i].numpy(), pb[i].numpy(), pl[i].numpy(), k, eb[i].item(), el[i].item(), res, si, len(order)))
        frames += [frames[-1]] * 2
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=350, loop=0)
    print(f"wrote {a.out}: {len(frames)} frames; shown-scene ADE old {eb[order].mean():.2f} vs new {el[order].mean():.2f}")

if __name__ == "__main__":
    main()
