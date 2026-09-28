#!/usr/bin/env python3
"""
simulate.py -- runs a single interactive merge scenario end-to-end using the
trained checkpoints and renders a real GIF (matplotlib -> PIL, no browser).

Pipeline demonstrated, per anchor:
  1. AnchorGenerator produces K=5 anchors from scenario geometry.
  2. conditioned_model (New Way) is queried ONCE per anchor -> cached prediction.
  3. unconditional_model (Old Way A) is queried ONCE total -> one fixed, non-reactive
     forecast reused (unchanged) for every anchor.
  4. TrustRegionCEM refines each anchor into a concrete ego trajectory using the
     cached prediction (New Way / Old Way A both use mode="cached"; they differ
     only in WHICH cached prediction was supplied).
  5. Old Way B (naive) is also run once, re-querying conditioned_model per CEM
     candidate, to measure the real predictor-call / wall-clock cost difference.
  6. The oracle synthetic simulator gives the TRUE other-agent reaction under each
     anchor (since we control the ground-truth generator), so forecast error
     (ADE) can be measured per anchor for both old-way and new-way predictions.
  7. The anchor with lowest CEM cost is "committed to" and played back frame-by-frame.

Also prints a real efficiency comparison (predictor forward-pass counts, wall time)
used verbatim in the LinkedIn post's Key Improvements section.

Usage:
    python3 simulate.py --config config.yaml
"""
from __future__ import annotations

import argparse
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from src.data.synthetic_interactive_scenario import generate_episode
from src.models.anchor_conditioned_predictor import AnchorConditionedPredictor
from src.models.anchor_generator import AnchorGenerator
from src.models.trust_region_cem import TrustRegionCEM
from src.utils.kinematics import assertiveness, build_history

ANCHOR_COLORS = {
    "aggressive_merge": "#d64545",
    "accelerate_past": "#e0983c",
    "hold_lane_delay": "#8a8a8a",
    "cautious_yield": "#3c8ee0",
    "decelerate_follow": "#3c56b0",
}
COLOR_EGO = "#1f9d55"
COLOR_OTHER_GT = "#111111"
COLOR_OLD_WAY = "#b0446a"
COLOR_NEW_WAY = "#1f9d55"


def load_models(cfg):
    sc, mc = cfg["scenario"], cfg["model"]
    conditioned = AnchorConditionedPredictor(state_dim=sc["state_dim"], cond_dim=sc["cond_dim"],
                                              hidden_dim=mc["hidden_dim"], future_len=sc["future_len"],
                                              future_dim=sc["future_dim"])
    unconditional = AnchorConditionedPredictor(state_dim=sc["state_dim"], cond_dim=sc["cond_dim"],
                                                hidden_dim=mc["hidden_dim"], future_len=sc["future_len"],
                                                future_dim=sc["future_dim"])
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    conditioned.load_state_dict(torch.load(os.path.join(ckpt_dir, "conditioned_model.pt")))
    unconditional.load_state_dict(torch.load(os.path.join(ckpt_dir, "unconditional_model.pt")))
    conditioned.eval()
    unconditional.eval()
    return conditioned, unconditional


def oracle_other_future(cond, other_lat0, other_spd0, future_len, k_speed, k_lateral, noise_std, seed):
    """Ground-truth other-agent reaction under a hypothetical committed `cond`,
    using the SAME rule as the data simulator (this is our oracle since we author
    the synthetic world) -- lets us measure real forecast error per anchor."""
    g = torch.Generator().manual_seed(seed)
    a = assertiveness(cond)
    t = torch.linspace(1.0 / future_len, 1.0, future_len)
    spd = other_spd0 - k_speed * a * t + noise_std * torch.randn(future_len, generator=g)
    lat = other_lat0 - k_lateral * a * t.pow(1.2) + noise_std * torch.randn(future_len, generator=g)
    return torch.stack([lat, spd], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--out", type=str, default="assets/simulation.gif")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sc, mc, cc, ac = cfg["scenario"], cfg["model"], cfg["cem"], cfg["anchors"]
    torch.manual_seed(args.seed)

    conditioned_model, unconditional_model = load_models(cfg)

    # --- scenario setup: one fixed "current state" the ego must plan from ---
    ego_lat0, ego_spd0 = 0.0, sc["base_speed"]
    other_lat0, other_spd0 = 0.12, sc["base_speed"] * 0.95
    ego_hist = build_history(ego_lat0, ego_spd0, sc["history_len"])
    other_hist = build_history(other_lat0, other_spd0, sc["history_len"])

    anchor_gen = AnchorGenerator()
    anchors = anchor_gen.generate(gap_to_merge=ac["default_gap_to_merge"], lane_width=ac["default_lane_width"])
    anchor_bank = torch.stack([a.cond for a in anchors])   # [K,2]

    cem = TrustRegionCEM(future_len=sc["future_len"], weights=cc["cost_weights"], n_iter=cc["n_iter"],
                          n_samples=cc["n_samples"], elite_frac=cc["elite_frac"], init_std=cc["init_std"],
                          bounds=tuple(cc["bounds"]), collision_margin=cc["collision_margin"],
                          trust_region_radius=cc["trust_region_radius"], base_speed=sc["base_speed"])

    # === NEW WAY (INTERACT): query conditioned_model ONCE per anchor ===
    new_way_predictor_calls = 0
    per_anchor = []
    with torch.no_grad():
        for anchor_idx, anchor in enumerate(anchors):
            cached_pred = conditioned_model(ego_hist.unsqueeze(0), other_hist.unsqueeze(0),
                                             anchor.cond.unsqueeze(0)).squeeze(0)   # [T,2]
            new_way_predictor_calls += 1
            cem_result = cem.refine(anchor.cond, mode="cached", cached_pred=cached_pred)
            # NOTE: seed derived from anchor_idx (not Python's randomized string hash()) so this
            # scenario is exactly reproducible across runs given the same --seed.
            true_future = oracle_other_future(anchor.cond, other_lat0, other_spd0, sc["future_len"],
                                               sc["k_speed_reaction"], sc["k_lateral_reaction"],
                                               sc["noise_std"], seed=args.seed + anchor_idx)
            new_way_ade = torch.norm(cached_pred - true_future, dim=-1).mean().item()
            per_anchor.append(dict(anchor=anchor, cached_pred=cached_pred, cem_result=cem_result,
                                    true_future=true_future, new_way_ade=new_way_ade))

    # === OLD WAY A (non-interactive): query unconditional_model ONCE total, reuse for all anchors ===
    with torch.no_grad():
        old_way_a_pred = unconditional_model(ego_hist.unsqueeze(0), other_hist.unsqueeze(0),
                                              torch.zeros(1, sc["cond_dim"])).squeeze(0)   # [T,2]
    old_way_a_predictor_calls = 1
    for rec in per_anchor:
        rec["old_way_ade"] = torch.norm(old_way_a_pred - rec["true_future"], dim=-1).mean().item()
        rec["old_way_pred"] = old_way_a_pred

    # === OLD WAY B (naive per-candidate re-query): run once, on the winning anchor, to measure cost ===
    best_rec = min(per_anchor, key=lambda r: r["cem_result"].best_cost)

    def predict_fn(ego_h, other_h, cond_batch):
        return conditioned_model(ego_h, other_h, cond_batch)

    naive_result = cem.refine(best_rec["anchor"].cond, mode="naive", predict_fn=predict_fn,
                               ego_hist=ego_hist, other_hist=other_hist)

    # === Print the real efficiency comparison ===
    k = len(anchors)
    naive_total_calls_all_anchors = naive_result.predictor_calls * k   # if run for every anchor
    print("=== Efficiency comparison (real, measured) ===")
    print(f"New Way (INTERACT):         {new_way_predictor_calls} predictor forward passes total "
          f"(1 per anchor x {k} anchors)")
    print(f"Old Way A (non-interactive): {old_way_a_predictor_calls} predictor forward pass total "
          f"(reused, non-reactive)")
    print(f"Old Way B (naive re-query), single anchor: {naive_result.predictor_calls} predictor calls "
          f"({cc['n_iter']} CEM iters x {cc['n_samples']} candidates), wall_time={naive_result.wall_time_s:.4f}s")
    print(f"Old Way B (naive re-query), extrapolated to all {k} anchors: {naive_total_calls_all_anchors} "
          f"predictor calls "
          f"({naive_total_calls_all_anchors / new_way_predictor_calls:.1f}x more than New Way)")

    print("\n=== Forecast error per anchor (New Way vs Old Way A) ===")
    for rec in per_anchor:
        print(f"{rec['anchor'].name:20s} a={rec['anchor'].assertiveness:+.2f} | "
              f"new_way_ADE={rec['new_way_ade']:.4f}  old_way_ADE={rec['old_way_ade']:.4f}  "
              f"cem_best_cost={rec['cem_result'].best_cost:.4f}")

    print(f"\nCommitted anchor (lowest CEM cost): {best_rec['anchor'].name}  "
          f"(cost={best_rec['cem_result'].best_cost:.4f})")

    mean_new_ade = float(np.mean([r["new_way_ade"] for r in per_anchor]))
    mean_old_ade = float(np.mean([r["old_way_ade"] for r in per_anchor]))
    sim_summary = {
        "anchor_names": [a.name for a in anchors],
        "new_way_predictor_calls": new_way_predictor_calls,
        "old_way_a_predictor_calls": old_way_a_predictor_calls,
        "old_way_b_naive_calls_single_anchor": naive_result.predictor_calls,
        "old_way_b_naive_calls_all_anchors": naive_total_calls_all_anchors,
        "old_way_b_wall_time_s": naive_result.wall_time_s,
        "committed_anchor": best_rec["anchor"].name,
        "per_anchor_forecast_error": {r["anchor"].name: {"new_way_ade": r["new_way_ade"],
                                                           "old_way_ade": r["old_way_ade"]}
                                       for r in per_anchor},
        "mean_new_way_ade": mean_new_ade,
        "mean_old_way_ade": mean_old_ade,
        "mean_ade_reduction_pct": 100.0 * (mean_old_ade - mean_new_ade) / max(mean_old_ade, 1e-9),
    }
    os.makedirs("assets", exist_ok=True)
    with open("assets/simulation_summary.json", "w") as f:
        json.dump(sim_summary, f, indent=2)
    print(f"\nMean forecast ADE across all anchors: new_way={mean_new_ade:.4f} old_way={mean_old_ade:.4f} "
          f"(reduction={sim_summary['mean_ade_reduction_pct']:.1f}%)")

    # ================= RENDER GIF =================
    render_gif(per_anchor, best_rec, anchor_bank, sc["future_len"], args.out)
    print(f"\nSaved simulation GIF to {args.out}")


def _fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.load()
    return img


def render_gif(per_anchor, best_rec, anchor_bank, future_len, out_path):
    frames = []
    other_hist_lat = 0.12  # matches scenario setup, used for BEV origin

    # ---- Phase 1: reveal each anchor's branch one at a time (K frames) ----
    for reveal_n in range(1, len(per_anchor) + 1):
        fig, (ax_bev, ax_tel) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
        ax_bev.set_title("Top-down BEV: anchors fanning out", fontsize=10)
        ax_bev.set_xlabel("longitudinal progress (t)")
        ax_bev.set_ylabel("lateral offset")
        ax_bev.set_xlim(-0.5, future_len + 0.5)
        ax_bev.set_ylim(-1.6, 1.6)
        ax_bev.axhline(0, color="#cccccc", lw=1, ls="--")

        t_axis = np.arange(1, future_len + 1)
        for rec in per_anchor[:reveal_n]:
            name = rec["anchor"].name
            color = ANCHOR_COLORS[name]
            is_best = rec is best_rec
            lw = 3.0 if is_best else 1.4
            alpha = 1.0 if is_best else 0.55
            ax_bev.plot(t_axis, rec["cem_result"].best_traj[:, 0].numpy(), color=color, lw=lw, alpha=alpha,
                        label=f"{name}{'  (committed)' if is_best else ''}")
            ax_bev.plot(t_axis, rec["cached_pred"][:, 0].numpy(), color=color, lw=1.0, ls=":", alpha=alpha * 0.8)

        ax_bev.scatter([0], [other_hist_lat], color=COLOR_OTHER_GT, marker="s", s=40, zorder=5,
                        label="other agent (now)")
        ax_bev.legend(fontsize=6.5, loc="upper left", ncol=1)

        ax_tel.set_title("Predictor queries used (real, measured)", fontsize=10)
        ax_tel.bar(["New Way\n(per anchor)", "Old Way A\n(non-reactive)", "Old Way B\n(naive, 1 anchor)"],
                   [reveal_n, 1, 0], color=[COLOR_NEW_WAY, COLOR_OLD_WAY, "#cccccc"])
        ax_tel.set_ylabel("predictor forward passes")
        fig.tight_layout()
        frames.append(_fig_to_pil(fig))
        plt.close(fig)

    # ---- Phase 2: playback of the committed anchor's trajectory + telemetry ----
    name = best_rec["anchor"].name
    color = ANCHOR_COLORS[name]
    ego_traj = best_rec["cem_result"].best_traj.numpy()          # [T,2]
    pred_other = best_rec["cached_pred"].numpy()                  # [T,2] (new way, reactive)
    true_other = best_rec["true_future"].numpy()                  # [T,2] ground truth
    old_way_pred = best_rec["old_way_pred"].numpy()                # [T,2] (old way A, non-reactive, fixed)

    new_err_cum, old_err_cum = [], []
    for t in range(future_len):
        fig, (ax_bev, ax_tel) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
        ax_bev.set_title(f"Committed plan: {name}  (t={t+1}/{future_len})", fontsize=10)
        ax_bev.set_xlabel("longitudinal progress (t)")
        ax_bev.set_ylabel("lateral offset")
        ax_bev.set_xlim(-0.5, future_len + 0.5)
        ax_bev.set_ylim(-1.6, 1.6)
        ax_bev.axhline(0, color="#cccccc", lw=1, ls="--")

        t_axis = np.arange(1, t + 2)
        ax_bev.plot(t_axis, ego_traj[:t + 1, 0], color=COLOR_EGO, lw=3, marker="o", ms=4, label="ego (CEM-refined)")
        ax_bev.plot(t_axis, true_other[:t + 1, 0], color=COLOR_OTHER_GT, lw=2.5, marker="s", ms=4,
                    label="other agent (ground truth)")
        ax_bev.plot(t_axis, pred_other[:t + 1, 0], color=COLOR_NEW_WAY, lw=1.6, ls="--", marker="^", ms=3,
                    label="new-way prediction (reactive)")
        ax_bev.plot(t_axis, old_way_pred[:t + 1, 0], color=COLOR_OLD_WAY, lw=1.6, ls="--", marker="v", ms=3,
                    label="old-way-A prediction (non-reactive)")
        ax_bev.legend(fontsize=6.5, loc="upper left")

        new_err_cum.append(float(np.linalg.norm(pred_other[t] - true_other[t])))
        old_err_cum.append(float(np.linalg.norm(old_way_pred[t] - true_other[t])))

        ax_tel.set_title("Forecast error over time: old-way vs new-way", fontsize=10)
        ax_tel.plot(range(1, len(new_err_cum) + 1), new_err_cum, color=COLOR_NEW_WAY, lw=2.5,
                    marker="o", label="new way (anchor-conditioned)")
        ax_tel.plot(range(1, len(old_err_cum) + 1), old_err_cum, color=COLOR_OLD_WAY, lw=2.5,
                    marker="o", label="old way A (non-reactive)")
        ax_tel.set_xlim(0.5, future_len + 0.5)
        ax_tel.set_ylim(0, max(max(new_err_cum + old_err_cum) * 1.2, 0.1))
        ax_tel.set_xlabel("future timestep")
        ax_tel.set_ylabel("displacement error")
        ax_tel.legend(fontsize=7, loc="upper left")

        fig.tight_layout()
        frames.append(_fig_to_pil(fig))
        plt.close(fig)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=450, loop=0)


if __name__ == "__main__":
    main()
