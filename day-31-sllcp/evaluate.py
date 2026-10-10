"""Shared evaluation: baseline argmax VLM vs. global label-conditional CP vs. SLLCP."""
import torch
from sllcp.conformal import SplitLabelConditionalCP, SplitLabelLocalizedCP
from sllcp.data import REGIMES


def evaluate(model, cal, test, alpha, bandwidth):
    pc, pt = model.probs(cal["x"]), model.probs(test["x"])
    glob = SplitLabelConditionalCP(alpha).fit(pc, cal["y"])
    loc = SplitLabelLocalizedCP(alpha, bandwidth).fit(pc, cal["emb"], cal["y"])
    flags = {
        "argmax (old way)": pt.argmax(1) == 1,
        "global CP": glob.predict_sets(pt)[:, 1],
        "SLLCP (new way)": loc.predict_sets(pt, test["emb"])[:, 1],
    }
    sets_loc = loc.predict_sets(pt, test["emb"])
    y, reg = test["y"], test["regime"]
    res = {}
    for name, f in flags.items():
        unsafe, safe = y == 1, y == 0
        r = {"unsafe_recall": f[unsafe].float().mean().item(),
             "false_alarm": f[safe].float().mean().item()}
        for i, rn in enumerate(REGIMES):
            m = unsafe & (reg == i)
            r[f"recall_{rn}"] = f[m].float().mean().item() if m.any() else float("nan")
            ms = safe & (reg == i)
            r[f"fa_{rn}"] = f[ms].float().mean().item() if ms.any() else float("nan")
        res[name] = r
    res["_set_size_sllcp"] = sets_loc.float().sum(1).mean().item()
    res["_base_rate"] = (y == 1).float().mean().item()
    return res, flags, sets_loc, (glob, loc)


def format_table(res, alpha):
    rows = [f"target unsafe-recall >= {1-alpha:.0%}   (test unsafe base rate {res['_base_rate']:.1%})",
            f"{'method':<20}{'recall':>8}{'clear':>8}{'dense':>8}{'occl.':>8}{'false-alarm':>13}{'FA clear':>10}{'FA dense':>10}{'FA occl.':>10}"]
    for k, r in res.items():
        if k.startswith("_"):
            continue
        rows.append(f"{k:<20}{r['unsafe_recall']:>8.3f}{r['recall_clear']:>8.3f}{r['recall_dense']:>8.3f}"
                    f"{r['recall_occluded']:>8.3f}{r['false_alarm']:>13.3f}{r['fa_clear']:>10.3f}{r['fa_dense']:>10.3f}{r['fa_occluded']:>10.3f}")
    return "\n".join(rows)
