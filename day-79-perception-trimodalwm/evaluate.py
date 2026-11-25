"""Held-out evaluation: 600 clips (100 per weather), 6 observed frames -> 8 forecast frames (2 s).

Questions (from the abstract of arXiv 2611.22669):
    Q1  do post-trained RGB / thermal forecasts beat persistence?
    Q2  does joint (tri-sensor) training lower LiDAR error at 2 s, and at what cost in camera fidelity?
    Q3  is the tri-sensor model the only one whose LiDAR forecast beats persistence on mean error?
    Q4  does blanking one sensor hurt the others more in snow than in fog?

    python evaluate.py      -> results/summary.json, results/RESULTS.md, results/per_clip.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from train import RUNS, batch_tiles, load_split
from twm.data import BEAM_ROWS, D_MAX, T_CTX, T_FUT, WEATHERS, Z_GROUND
from twm.model import TILE_W, VideoWorldModel

ROOT = Path(__file__).resolve().parent
NAMES = ["rgb", "thermal", "lidar"]


def load_model(run: str, seed: int = 0) -> tuple[VideoWorldModel, list[int]]:
    ck = torch.load(ROOT / f"results/ckpt/{run}_s{seed}.pt", map_location="cpu")
    m = VideoWorldModel(d=ck["model"]["pos"].shape[1])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["sensors"]


@torch.no_grad()
def forecast(m, d, sensors, idx, blank_tile: int | None = None) -> np.ndarray:
    """-> future forecasts (B, 8, 3, 48, 64 n) as numpy."""
    x = batch_tiles(d, idx, sensors)
    blank = None
    if blank_tile is not None:
        blank = torch.ones(1, 1, 1, TILE_W * len(sensors))
        blank[..., blank_tile * TILE_W:(blank_tile + 1) * TILE_W] = 0
    y = m(x, T_CTX, blank=blank)
    return y[:, T_CTX - 1:].numpy()


def sensor_errors(pred_tile: np.ndarray, d: dict, idx: np.ndarray, s: int, obj_mask: np.ndarray) -> dict:
    """pred_tile (B, 8, 3, 48, 64) for sensor s -> per clip, per future frame: sum |err| and counts.

    rgb / thermal: mean abs error in 0-255 units over all pixels (thermal: over the grey channel mean).
    lidar: abs depth error [m] on target-measured returns ('all'), on returns that were ALSO measured in the last
    observed frame ('common', where persistence is defined) and on target returns that hit an object ('obj').
    """
    if s == 0:
        g = np.moveaxis(d["rgb"][idx, T_CTX:].astype(np.float32), -1, 2) / 255.0
        e = np.abs(np.clip(pred_tile, 0, 1) - g).mean((2, 3, 4)) * 255
        mse = ((np.clip(pred_tile, 0, 1) - g) ** 2).mean((2, 3, 4))
        return dict(mae=e, psnr=10 * np.log10(1.0 / np.maximum(mse, 1e-8)))
    if s == 1:
        g = d["th"][idx, T_CTX:].astype(np.float32) / 255.0
        return dict(mae=np.abs(np.clip(pred_tile.mean(2), 0, 1) - g).mean((2, 3)) * 255)
    g = d["depth"][idx, T_CTX:].astype(np.float32) / 100.0
    v = d["valid"][idx, T_CTX:]
    v_last = d["valid"][idx, T_CTX - 1][:, None]
    p = np.clip(pred_tile[:, :, :2].mean(2), 0, 1) * D_MAX
    err = np.abs(p - g)
    out = {}
    for k, m in (("all", v), ("common", v & v_last), ("obj", v & obj_mask)):
        out[f"{k}_sum"] = (err * m).sum((2, 3))
        out[f"{k}_n"] = m.sum((2, 3))
    return out


def persistence_tile(d, idx, s) -> np.ndarray:
    x = batch_tiles(d, idx, [s]).numpy()
    return np.repeat(x[:, T_CTX - 1:T_CTX], T_FUT, 1)


def object_mask(d, idx) -> np.ndarray:
    """Pixels whose true (dense) depth is closer than the empty road -> an object was hit."""
    dense = d["dense"][idx, T_CTX:].astype(np.float32) / 100.0
    zg = np.where(np.isfinite(Z_GROUND), Z_GROUND, 1e9)
    return (dense > 0) & (dense < zg - 0.5)


def run_all(d, model_specs, chunk=50):
    """model_specs: name -> (callable idx -> pred canvas (B, 8, 3, 48, 64 n), sensors).  Returns per-clip arrays."""
    N = len(d["weather"])
    res = {}
    for name, (fn, sensors) in model_specs.items():
        acc: dict = {}
        for i0 in range(0, N, chunk):
            idx = np.arange(i0, min(N, i0 + chunk))
            pred = fn(idx)
            om = object_mask(d, idx)
            for j, s in enumerate(sensors):
                e = sensor_errors(pred[..., j * TILE_W:(j + 1) * TILE_W], d, idx, s, om)
                for k, v in e.items():
                    acc.setdefault(f"{NAMES[s]}/{k}", []).append(v)
        res[name] = {k: np.concatenate(v) for k, v in acc.items()}
        print("done", name, flush=True)
    return res


def pooled(r: dict, key: str, sel: np.ndarray, frame: int | None = None) -> float:
    """LiDAR: pooled mean over returns; rgb/thermal: mean over clips.  frame None = mean over the 8 future frames."""
    if key.startswith("lidar/"):
        s, n = r[key + "_sum"][sel], r[key + "_n"][sel]
        if frame is not None:
            s, n = s[:, frame], n[:, frame]
        return float(s.sum() / max(1, n.sum()))
    v = r[key][sel]
    return float((v[:, frame] if frame is not None else v).mean())


def boot_diff(ra, rb, key, sel, frame, n_boot=1000, seed=0):
    """95% CI of pooled(ra) - pooled(rb) by clip bootstrap."""
    rng = np.random.default_rng(seed)
    ids = np.nonzero(sel)[0]
    out = []
    for _ in range(n_boot):
        b = rng.choice(ids, len(ids))
        m = np.zeros(len(sel), bool)
        cnt = np.bincount(b, minlength=len(sel))
        # weighted pooled mean with multiplicity
        def pw(r):
            if key.startswith("lidar/"):
                s, n = r[key + "_sum"][:, frame], r[key + "_n"][:, frame]
                return (s * cnt).sum() / max(1, (n * cnt).sum())
            return (r[key][:, frame] * cnt).sum() / cnt.sum()
        out.append(pw(ra) - pw(rb))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(pooled(ra, key, sel, frame) - pooled(rb, key, sel, frame)), float(lo), float(hi)


def main():
    torch.set_num_threads(2)
    d = load_split("test")
    N = len(d["weather"])
    models = {r: load_model(r) for r in ("rgb", "thermal", "lidar", "tri")}
    if (ROOT / "results/ckpt/base_s0.pt").exists():
        models["base"] = load_model("base")
    specs = {name: ((lambda idx, m=m, s=s: forecast(m, d, s, idx)), s) for name, (m, s) in models.items()}
    for s in range(3):
        specs[f"persist_{NAMES[s]}"] = ((lambda idx, s=s: persistence_tile(d, idx, s)), [s])
    mt, st = models["tri"]
    for b in range(3):
        specs[f"tri_blank_{NAMES[b]}"] = ((lambda idx, b=b: forecast(mt, d, st, idx, blank_tile=b)), st)
    R = run_all(d, specs)
    np.savez_compressed(ROOT / "results/per_clip.npz", **{f"{m}|{k}": v for m, r in R.items() for k, v in r.items()},
                        weather=d["weather"])
    F2 = T_FUT - 1                                                                    # frame index of +2 s
    allc = np.ones(N, bool)
    wsel = {w: d["weather"] == i for i, w in enumerate(WEATHERS)}
    S: dict = dict(n_clips=int(N), n_per_weather=int(N // 6), horizon_s=2.0, metrics={}, by_weather={}, blank={})
    rows = [("rgb", "rgb"), ("thermal", "thermal"), ("lidar", "lidar"), ("base", "rgb"), ("tri", "rgb"), ("tri", "thermal"),
            ("tri", "lidar"), ("persist_rgb", "rgb"), ("persist_thermal", "thermal"), ("persist_lidar", "lidar")]
    for name, sn in rows:
        if name not in R:
            continue
        r = R[name]
        rec = S["metrics"].setdefault(name, {})
        if sn == "lidar":
            for k in ("all", "common", "obj"):
                rec[f"lidar_{k}_2s"] = pooled(r, f"lidar/{k}", allc, F2)
                rec[f"lidar_{k}_mean"] = pooled(r, f"lidar/{k}", allc)
        else:
            rec[f"{sn}_mae_2s"] = pooled(r, f"{sn}/mae", allc, F2)
            rec[f"{sn}_mae_mean"] = pooled(r, f"{sn}/mae", allc)
            if sn == "rgb":
                rec["rgb_psnr_mean"] = pooled(r, "rgb/psnr", allc)
        for w, sel in wsel.items():
            bw = S["by_weather"].setdefault(name, {}).setdefault(w, {})
            if sn == "lidar":
                bw["lidar_common_2s"] = pooled(r, "lidar/common", sel, F2)
                bw["lidar_all_2s"] = pooled(r, "lidar/all", sel, F2)
                bw["lidar_obj_2s"] = pooled(r, "lidar/obj", sel, F2)
            else:
                bw[f"{sn}_mae_2s"] = pooled(r, f"{sn}/mae", sel, F2)
    # paired comparisons with bootstrap CIs
    S["ci"] = {
        "tri_minus_lidar/lidar_all_2s": boot_diff(R["tri"], R["lidar"], "lidar/all", allc, F2),
        "tri_minus_lidar/lidar_obj_2s": boot_diff(R["tri"], R["lidar"], "lidar/obj", allc, F2),
        "tri_minus_persist/lidar_common_2s": boot_diff(R["tri"], R["persist_lidar"], "lidar/common", allc, F2),
        "lidar_minus_persist/lidar_common_2s": boot_diff(R["lidar"], R["persist_lidar"], "lidar/common", allc, F2),
        "tri_minus_rgb/rgb_mae_2s": boot_diff(R["tri"], R["rgb"], "rgb/mae", allc, F2),
        "tri_minus_thermal/thermal_mae_2s": boot_diff(R["tri"], R["thermal"], "thermal/mae", allc, F2),
        "rgb_minus_persist/rgb_mae_2s": boot_diff(R["rgb"], R["persist_rgb"], "rgb/mae", allc, F2),
        "thermal_minus_persist/thermal_mae_2s": boot_diff(R["thermal"], R["persist_thermal"], "thermal/mae", allc, F2),
    }
    # Q4 blanking: relative change in the OTHER sensors' +2 s error, per weather
    for b in range(3):
        nm = f"tri_blank_{NAMES[b]}"
        for w, sel in wsel.items():
            for o in range(3):
                if o == b:
                    continue
                key = "lidar/all" if o == 2 else f"{NAMES[o]}/mae"
                base = pooled(R["tri"], key, sel, F2)
                bl = pooled(R[nm], key, sel, F2)
                S["blank"].setdefault(NAMES[b], {}).setdefault(w, {})[NAMES[o]] = 100 * (bl - base) / base
    # clip choices for the simulation: the median tri-LiDAR clip in fog and in snow
    e = R["tri"]["lidar/all_sum"][:, F2] / np.maximum(1, R["tri"]["lidar/all_n"][:, F2])
    S["clips"] = {}
    for w in ("fog", "snow", "night"):
        ids = np.nonzero(wsel[w])[0]
        S["clips"][w] = int(ids[np.argsort(e[ids])[len(ids) // 2]])
    S["params"] = int(sum(p.numel() for p in mt.parameters()))
    S["train_minutes"] = {r: round(torch.load(ROOT / f"results/ckpt/{r}_s0.pt")["minutes"], 1)
                          for r in list(models)}
    json.dump(S, open(ROOT / "results/summary.json", "w"), indent=1)
    write_md(S)
    print(json.dumps(S["metrics"], indent=1))
    print(json.dumps(S["ci"], indent=1))


def write_md(S):
    M = S["metrics"]
    L = [f"# Full results ({S['n_clips']} held-out clips, {S['n_per_weather']} per weather, 6 observed -> 8 forecast frames)\n",
         "Errors at +2 s unless marked mean (= mean over the 8 forecast frames). RGB / thermal MAE in 0-255 grey levels; "
         "LiDAR MAE in metres against measured returns.\n",
         "## Cameras\n", "| Model | RGB MAE +2 s | RGB MAE mean | RGB PSNR mean | thermal MAE +2 s | thermal MAE mean |",
         "|---|---|---|---|---|---|"]
    for n in ("rgb", "thermal", "base", "tri", "persist_rgb", "persist_thermal"):
        if n not in M:
            continue
        r = M[n]
        f = lambda k, fmt="{:.2f}": fmt.format(r[k]) if k in r else "-"
        L.append(f"| {n} | {f('rgb_mae_2s')} | {f('rgb_mae_mean')} | {f('rgb_psnr_mean')} | {f('thermal_mae_2s')} | {f('thermal_mae_mean')} |")
    L += ["\n## LiDAR (metres)\n", "| Model | all returns +2 s | all mean | common returns +2 s | object returns +2 s | object mean |",
          "|---|---|---|---|---|---|"]
    for n in ("lidar", "tri", "persist_lidar"):
        r = M[n]
        L.append(f"| {n} | {r['lidar_all_2s']:.2f} | {r['lidar_all_mean']:.2f} | {r['lidar_common_2s']:.2f} | "
                 f"{r['lidar_obj_2s']:.2f} | {r['lidar_obj_mean']:.2f} |")
    L.append("\n'common' = returns measured both in the target frame and in the last observed frame (where persistence is defined).\n")
    L += ["## Paired differences, 95% CI by clip bootstrap\n", "| Comparison | diff | CI |", "|---|---|---|"]
    for k, (dd, lo, hi) in S["ci"].items():
        L.append(f"| {k} | {dd:+.3f} | {lo:+.3f} .. {hi:+.3f} |")
    L += ["\n## By weather (+2 s)\n", "| Model / metric | " + " | ".join(WEATHERS) + " |", "|---|" + "---|" * 6]
    for n, key in (("rgb", "rgb_mae_2s"), ("tri", "rgb_mae_2s"), ("persist_rgb", "rgb_mae_2s"), ("thermal", "thermal_mae_2s"),
                   ("tri", "thermal_mae_2s"), ("persist_thermal", "thermal_mae_2s"), ("lidar", "lidar_all_2s"),
                   ("tri", "lidar_all_2s"), ("lidar", "lidar_common_2s"), ("tri", "lidar_common_2s"),
                   ("persist_lidar", "lidar_common_2s"), ("lidar", "lidar_obj_2s"), ("tri", "lidar_obj_2s"),
                   ("persist_lidar", "lidar_obj_2s")):
        bw = S["by_weather"][n]
        L.append(f"| {n} {key} | " + " | ".join(f"{bw[w][key]:.2f}" for w in WEATHERS) + " |")
    L += ["\n## Blanking one sensor in the tri model: % change of the other sensors' +2 s error\n",
          "| blanked | other | " + " | ".join(WEATHERS) + " |", "|---|---|" + "---|" * 6]
    for b, ws in S["blank"].items():
        for o in [x for x in NAMES if x != b]:
            L.append(f"| {b} | {o} | " + " | ".join(f"{ws[w][o]:+.1f}%" for w in WEATHERS) + " |")
    (ROOT / "results/RESULTS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
