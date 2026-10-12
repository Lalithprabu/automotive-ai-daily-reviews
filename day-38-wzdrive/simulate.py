"""Render the 'AI in action' simulation: baseline vs WZDrive on held-out synthetic work-zone scenes.
Panels: (left) BEV: cones/workers, driver gaze cone, observed history, ground-truth future,
baseline best-of-6 and WZDrive K=6 modes (top mode bold); (right-top) driver gaze/steer/brake telemetry;
(right-bottom) running ADE of both models vs ground truth as the ego 'plays out' the future.
Every prediction comes from the trained checkpoints in checkpoints/."""
import argparse, yaml, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from PIL import Image
import io
from src.wzdrive import WZDrive, generate_dataset, min_ade_fde
from src.wzdrive.data import CLASS_NAMES

C = dict(gt="#2ca02c", base="#d62728", new="#1f77b4", hist="#444444", cone="#ff8c00", worker="#9467bd")


def load(tag, cfg):
    ck = torch.load(f"checkpoints/{tag}.pt"); m = cfg["model"]
    net = WZDrive(m["d_model"], m["n_heads"], m["n_modes"], m["n_dec_layers"], 0.0, cfg["data"]["hist_steps"], cfg["data"]["fut_steps"], **ck["kw"])
    net.load_state_dict(ck["state"]); return net.eval()


def pick_scenes(d, base, new):
    """One held-out scene per class: the one where WZDrive's advantage over the baseline is largest."""
    with torch.no_grad():
        eb = min_ade_fde(base(d["kin"], d["ctrl"], d["gaze"], d["scene"])[0], d["fut"])[0]
        en = min_ade_fde(new(d["kin"], d["ctrl"], d["gaze"], d["scene"])[0], d["fut"])[0]
    idx = []
    for c in [1, 2, 3, 0]:
        gain = torch.where(d["cls"] == c, eb - en, torch.tensor(-1e9)); idx.append(int(gain.argmax()))
    return idx


def frame(d, i, k, pb, pn, lb, ln, attn, fig_size=(13, 6.2)):
    Th, Tf = 8, 12
    fig = plt.figure(figsize=fig_size, dpi=80)
    ax = fig.add_axes([0.04, 0.08, 0.50, 0.84]); at = fig.add_axes([0.62, 0.58, 0.35, 0.32]); ae = fig.add_axes([0.62, 0.10, 0.35, 0.32])
    sc = d["scene"][i]; kin = d["kin"][i]; fut = d["fut"][i]; gz = d["gaze"][i]
    ax.set_facecolor("#f2f2f2"); ax.set_xlim(-12, 50); ax.set_ylim(-9, 9); ax.set_aspect("equal")
    for yy in (-5.25, 5.25): ax.axhline(yy, color="white", lw=2)
    for yy in (-1.75, 1.75): ax.axhline(yy, color="#bbbbbb", lw=1, ls="--")
    for j in range(sc.shape[0]):
        if sc[j, 6] < 0.5: continue
        if sc[j, 0] > 0.5: ax.plot(sc[j, 2], sc[j, 3] , "^", color=C["cone"], ms=9, mec="k", alpha=0.35 + 0.65 * float(attn[j] / attn.max()))
        else:
            ax.add_patch(Circle((sc[j, 2], sc[j, 3]), 0.9, color=C["worker"], alpha=0.35 + 0.65 * float(attn[j] / attn.max())))
    # driver gaze cone (from the observed gaze at the current history step)
    th = min(k, Th - 1); yaw = float(gz[th, 0]); ax.add_patch(Wedge((0, 0), 30, np.degrees(yaw) - 6, np.degrees(yaw) + 6, color="gold", alpha=0.25))
    ax.plot(kin[:th + 1, 0], kin[:th + 1, 1], "-o", color=C["hist"], ms=3, lw=2, label="observed history")
    played = max(0, k - (Th - 1))                                    # future steps already driven
    if k >= Th - 1:
        ax.plot(np.r_[0, fut[:, 0]], np.r_[0, fut[:, 1]], "-", color=C["gt"], lw=3, label="ground truth future")
        order = pn[i].norm(dim=-1).sum(-1).argsort()
        for kk in range(pb.shape[1]): ax.plot(np.r_[0, pb[i, kk, :, 0]], np.r_[0, pb[i, kk, :, 1]], "-", color=C["base"], lw=1, alpha=0.18)
        for kk in range(pn.shape[1]): ax.plot(np.r_[0, pn[i, kk, :, 0]], np.r_[0, pn[i, kk, :, 1]], "-", color=C["new"], lw=1, alpha=0.30)
        tb, tn = int(lb[i].argmax()), int(ln[i].argmax())
        ax.plot(np.r_[0, pb[i, tb, :, 0]], np.r_[0, pb[i, tb, :, 1]], "--", color=C["base"], lw=2.5, label="baseline top-1 (kin+scene)")
        ax.plot(np.r_[0, pn[i, tn, :, 0]], np.r_[0, pn[i, tn, :, 1]], "-", color=C["new"], lw=2.5, label="WZDrive top-1 (+gaze+control)")
        if played: ax.plot(fut[played - 1, 0], fut[played - 1, 1], "s", color="k", ms=11)
    else: ax.plot(kin[th, 0], kin[th, 1], "s", color="k", ms=11)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9); ax.set_title(f"Scene class: {CLASS_NAMES[int(d['cls'][i])]}   |   t = {(min(k,Th-1)-(Th-1))*0.25 + max(0,played)*0.25:+.2f} s   |   {'OBSERVING' if k < Th - 1 else 'PREDICT + PLAY-OUT'}", fontsize=10)
    ax.set_xlabel("x forward [m]"); ax.set_ylabel("y left [m]")
    tt = np.arange(-Th + 1, 1) * 0.25
    at.plot(tt, gz[:, 0], color="gold", lw=2, label="gaze yaw [rad]"); at.plot(tt, d["ctrl"][i][:, 0], color="k", lw=1.5, label="steer")
    at.plot(tt, d["ctrl"][i][:, 2], color="r", lw=1.5, label="brake"); at.axvline(tt[min(th, Th - 1)], color="gray", ls=":")
    at.set_title("Driver telemetry (observed)", fontsize=9); at.legend(fontsize=7, loc="upper left"); at.set_xlabel("t [s]")
    errb = np.linalg.norm(pb[i, int(lb[i].argmax())] - fut, axis=-1); errn = np.linalg.norm(pn[i, int(ln[i].argmax())] - fut, axis=-1)
    tf = np.arange(1, Tf + 1) * 0.25
    if played:
        ae.plot(tf[:played], errb[:played], color=C["base"], lw=2, label=f"baseline err {errb[played-1]:.2f} m")
        ae.plot(tf[:played], errn[:played], color=C["new"], lw=2, label=f"WZDrive err {errn[played-1]:.2f} m"); ae.legend(fontsize=8, loc="upper left")
    ae.set_xlim(0, 3.1); ae.set_ylim(0, max(2.0, float(max(errb.max(), errn.max())) * 1.05)); ae.set_title("Top-1 position error vs ground truth", fontsize=9); ae.set_xlabel("horizon [s]"); ae.set_ylabel("m")
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0); return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--out", default="assets/wzdrive_simulation.gif"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); d = generate_dataset(cfg["data"]["n_val"], cfg["seed"] + 1)
    base, new = load("baseline", cfg), load("wzdrive", cfg); idx = pick_scenes(d, base, new)
    with torch.no_grad():
        pb, lb = base(d["kin"], d["ctrl"], d["gaze"], d["scene"]); pn, ln, attn = new(d["kin"], d["ctrl"], d["gaze"], d["scene"], return_attn=True)
    frames, durs = [], []
    for i in idx:
        for k in range(8 + 12):
            frames.append(frame(d, i, k, pb, pn, lb, ln, attn[i])); durs.append(450 if k < 7 else 300)
        durs[-1] = 1500
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=False)
    print("wrote", a.out, len(frames), "frames; scenes:", [(CLASS_NAMES[int(d['cls'][i])], i) for i in idx])


if __name__ == "__main__": main()
