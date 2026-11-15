"""Regenerate README.md and daily-series-log.md from catalog.json (the single source of truth).

Adding a new day:
  1. drop the folder in the repo root as  day-NN-<name>/  (with its README.md and an assets/*.gif)
  2. add one entry to catalog.json -> "days"  (day, name, title, arxiv, domain, summary, folder, gif)
  3. run:  python tools/build_readme.py
Weekly recap: add one entry to catalog.json -> "legs" (optional "highlights" make it the featured leg).
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Lalithprabu/automotive-ai-daily-reviews"

DOMAINS = {   # key: (emoji, label, one-line description)
    "perception": ("👁️", "Perception", "Cameras, radar, LiDAR, depth, HD maps and 3D occupancy"),
    "prediction": ("🔮", "Prediction & forecasting", "Trajectories, intentions and future occupancy of other road users"),
    "planning":   ("🧭", "Planning & end-to-end driving", "World models and planners that decide where the car goes"),
    "v2x":        ("📡", "V2X & cooperative driving", "Cars and roadside units sharing what they see, despite delay and loss"),
    "battery":    ("🔋", "EV battery & vehicle health", "Degradation, remaining useful life and health monitoring"),
    "safety":     ("🛡️", "Safety, risk & uncertainty", "Risk fields, calibrated uncertainty and safety-critical scenarios"),
}
VERDICT = {"held": "✅ Held up", "partly": "🟡 Partly", "no": "❌ Not at toy scale"}


def load():
    return json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))


def date_of(cat, day):
    return dt.date.fromisoformat(cat["start_date"]) + dt.timedelta(days=day - 1)


def arxiv_link(a):
    return f"[{a}](https://arxiv.org/abs/{a})" if a else "—"


def anchor(label):
    return re.sub(r"[^a-z0-9 -]", "", label.lower()).replace(" ", "-")


def build_readme(cat):
    days = cat["days"]
    by_day = {d["day"]: d for d in days}
    latest = max(by_day)
    n_gif = sum(1 for d in days if d.get("gif"))
    featured = next(l for l in reversed(cat["legs"]) if l.get("highlights"))
    L = []
    w = L.append

    # ---------------------------------------------------------------- header
    w('<p align="center"><img src="assets/banner.png" alt="Automotive AI Daily Reviews" width="100%"></p>\n')
    w('<p align="center">')
    w(f'  <img alt="papers" src="https://img.shields.io/badge/papers%20rebuilt-{len(days)}-ffb020?style=for-the-badge">')
    w(f'  <img alt="latest" src="https://img.shields.io/badge/latest-Day%20{latest}-4cc9f0?style=for-the-badge">')
    w(f'  <img alt="simulations" src="https://img.shields.io/badge/simulations-{n_gif}-3ddc84?style=for-the-badge">')
    w('  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">')
    w('  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey?style=for-the-badge">')
    w('</p>\n')
    w('<p align="center">')
    w('  <a href="#-latest-road-trip-leg">Latest leg</a> ·')
    w('  <a href="#-latest-simulations">Simulations</a> ·')
    w('  <a href="#-explore-by-domain">Explore by domain</a> ·')
    w('  <a href="#-the-ai-road-trip-weekly-recaps">Weekly recaps</a> ·')
    w('  <a href="#-run-any-day-in-60-seconds">Quick start</a> ·')
    w('  <a href="#-suggest-a-use-case">Suggest a use case</a>')
    w('</p>\n')
    w("Every day I pick a fresh automotive machine-learning paper (ADAS perception, motion forecasting, planning, "
      "V2X, EV batteries), **rebuild its core idea from scratch in PyTorch**, test it, run it in a synthetic "
      "simulation, and report honestly whether the idea held up at toy scale. Each day ships as one self-contained "
      "folder, plus a LinkedIn post that explains it in plain language.\n")

    # ---------------------------------------------------------------- latest leg
    a, b = featured["days"]
    w(f"## 🔥 Latest road-trip leg\n")
    w(f"**Leg {featured['leg']} · Days {a}–{b}: {featured.get('title', '')}**\n")
    w(f'<p align="center"><img src="{featured["file"]}" alt="Leg {featured["leg"]} overview" width="520"></p>\n')
    w("| Real-world moment | Paper | Day | What my rebuild showed | Verdict |")
    w("|---|---|---|---|---|")
    for h in featured["highlights"]:
        d = by_day[h["day"]]
        w(f"| {h['moment']} | [**{d['name']}**]({d['folder']}/) | {d['day']} | {h['result']} | {VERDICT[h['verdict']]} |")
    w("\n<sub>All numbers come from my own small synthetic rebuilds, not the papers' results.</sub>\n")

    # ---------------------------------------------------------------- gallery
    w("## 🎬 Latest simulations\n")
    recent = [d for d in sorted(days, key=lambda d: -d["day"]) if d.get("gif")][:6]
    w("<table>")
    for i in range(0, len(recent), 3):
        w("  <tr>")
        for d in recent[i:i + 3]:
            w(f'    <td align="center" width="33%"><a href="{d["folder"]}/"><img src="{d["gif"]}" width="100%" '
              f'alt="{d["name"]} simulation"></a><br><b>Day {d["day"]} · {d["name"]}</b><br>'
              f'<sub>{d["summary"]}</sub></td>')
        w("  </tr>")
    w("</table>\n")

    # ---------------------------------------------------------------- domains
    w("## 🧭 Explore by domain\n")
    w("| Domain | What it covers | Papers |")
    w("|---|---|---|")
    for k, (emo, label, desc) in DOMAINS.items():
        n = sum(1 for d in days if d["domain"] == k)
        w(f"| {emo} [{label}](#{anchor(emo + ' ' + label)}) | {desc} | {n} |")
    w("")
    for k, (emo, label, desc) in DOMAINS.items():
        sub = sorted((d for d in days if d["domain"] == k), key=lambda d: -d["day"])
        w(f"### {emo} {label}\n")
        w(f"<details{' open' if k == 'prediction' else ''}><summary><b>{len(sub)} papers</b>: {desc}</summary>\n")
        w("| Day | Paper | In one line | arXiv | Demo |")
        w("|---|---|---|---|---|")
        for d in sub:
            demo = f'[▶ GIF]({d["gif"]})' if d.get("gif") else "—"
            w(f"| {d['day']} | [**{d['name']}**]({d['folder']}/) · {d['title']} | {d['summary']} | {arxiv_link(d['arxiv'])} | {demo} |")
        w("\n</details>\n")

    # ---------------------------------------------------------------- legs
    w("## 🛣️ The AI Road Trip (weekly recaps)\n")
    w("Every week the daily rebuilds are bundled into one recap: a carousel, a narrated video, or a one-image overview.\n")
    w("| Leg | Days | Papers | Format |")
    w("|---|---|---|---|")
    for l in reversed(cat["legs"]):
        a, b = l["days"]
        names = ", ".join(by_day[x]["name"] for x in range(a, b + 1) if x in by_day)
        w(f"| [Leg {l['leg']}]({l['file']}) | {a}–{b} | {names} | {l['format']} |")
    w("")

    # ---------------------------------------------------------------- quick start
    w("## 🚀 Run any day in 60 seconds\n")
    w("Every `day-NN-*/` folder is self-contained:\n")
    w("```bash")
    w(f"git clone {REPO}.git")
    w(f"cd automotive-ai-daily-reviews/{by_day[latest]['folder']}")
    w("pip install -r requirements.txt")
    w("pytest tests/ -v        # unit tests")
    w("python train.py         # trains on synthetic data (see the folder README for flags)")
    w("python simulate.py      # renders the simulation GIF")
    w("```\n")
    w("Each folder README covers the architecture, an old-vs-new comparison, results, limitations and a "
      "sourcing note (`SOURCING.md` in recent days) that separates what came from the paper from what is my own choice.\n")

    # ---------------------------------------------------------------- layout
    w("## 📁 Repository layout\n")
    w("```text")
    w("automotive-ai-daily-reviews/")
    w("├── README.md              ← you are here (generated by tools/build_readme.py)")
    w("├── catalog.json           ← one entry per day: the single source of truth")
    w("├── daily-series-log.md    ← full chronological log (generated)")
    w(f"├── day-01-sv-wam/ … {by_day[latest]['folder']}/   ← one self-contained folder per paper")
    w("├── road-trip/             ← weekly recap carousels, videos and overview images")
    w("├── assets/                ← banner")
    w("├── docs/PROVENANCE.md     ← how earlier days were sourced and rebuilt")
    w("└── tools/                 ← README generator + banner source")
    w("```\n")

    # ---------------------------------------------------------------- CTA
    w("## 🙋 Suggest a use case\n")
    w("Is there a driving problem you wish AI would solve? "
      f"[Open a use-case request]({REPO}/issues/new?template=use-case.yml) or comment on the LinkedIn post. "
      "I pick one suggestion for an upcoming day and credit you in that day's post.\n")

    # ---------------------------------------------------------------- honesty
    w("## ⚖️ Honesty note\n")
    w("This is an independent, educational series of **toy-scale reference re-implementations**. It is *not* "
      "the authors' official code, and results come from synthetic data rather than the papers' benchmarks. "
      "When an idea did not reproduce, the folder says so. Sourcing details are in [docs/PROVENANCE.md](docs/PROVENANCE.md) "
      "and each folder's README / `SOURCING.md`. "
      f"Day {', '.join(map(str, cat['rest_days']))} was a rest stop. Licensed under [MIT](LICENSE).\n")
    return "\n".join(L)


def build_log(cat):
    old = {}
    p = ROOT / "daily-series-log.md"
    if p.exists():                     # keep the detailed category text already written for a day
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\|\s*(\d+)\s*\|", line)
            if m:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 6:            # current format: Day|Date|Paper|arXiv|Domain|Notes
                    old[int(m.group(1))] = cells[5]
                elif len(cells) == 5:          # original format: Day|Date|Paper|arXiv|Category
                    old[int(m.group(1))] = cells[4]
    L = ["# Daily Tech Review Series: Log", "",
         "Every day of the series in order (generated from `catalog.json` by `tools/build_readme.py`).", "",
         "| Day | Date | Paper | arXiv | Domain | Notes |", "|---|---|---|---|---|---|"]
    for d in sorted(cat["days"], key=lambda d: d["day"]):
        dom = DOMAINS[d["domain"]][1]
        L.append(f"| {d['day']} | {date_of(cat, d['day'])} | [{d['name']}: {d['title']}]({d['folder']}/) | "
                 f"{arxiv_link(d['arxiv'])} | {dom} | {old.get(d['day'], d['summary'])} |")
    for r in cat["rest_days"]:
        L.append(f"| {r} | {date_of(cat, r)} | Rest stop | — | — | — |")
    L.sort(key=lambda s: int(re.match(r"\|\s*(\d+)", s).group(1)) if re.match(r"\|\s*\d", s) else -1)
    return "\n".join(L) + "\n"


def check(cat):
    problems = []
    for d in cat["days"]:
        if not (ROOT / d["folder"]).is_dir():
            problems.append(f"Day {d['day']}: folder {d['folder']} missing")
        if d.get("gif") and not (ROOT / d["gif"]).is_file():
            problems.append(f"Day {d['day']}: gif {d['gif']} missing")
        if d["domain"] not in DOMAINS:
            problems.append(f"Day {d['day']}: unknown domain {d['domain']}")
    for l in cat["legs"]:
        if not (ROOT / l["file"]).is_file():
            problems.append(f"Leg {l['leg']}: {l['file']} missing")
    return problems


if __name__ == "__main__":
    cat = load()
    probs = check(cat)
    for p in probs:
        print("WARNING:", p)
    (ROOT / "daily-series-log.md").write_text(build_log(cat), encoding="utf-8")
    (ROOT / "README.md").write_text(build_readme(cat), encoding="utf-8")
    print(f"README.md + daily-series-log.md rebuilt: {len(cat['days'])} days, {len(cat['legs'])} legs, "
          f"{len(probs)} warnings")
