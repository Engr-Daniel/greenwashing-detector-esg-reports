"""README figures.  Run: python src/greenwash/figures.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenwash.paths import FIGURES, METRICS, PROCESSED  # noqa: E402
from greenwash.scoring import load_signals  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#ecebe7"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b7b0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#d6d4ce", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "figure.dpi": 150})
NOTE = "Daniel Oyewale · Petgrave.io · github.com/Engr-Daniel/greenwashing-detector-esg-reports"


def _save(fig, name, title, sub, foot_y=-0.03):
    fig.subplots_adjust(top=0.80)
    fig.text(0.01, 0.97, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.885, sub, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.01, foot_y, NOTE, ha="left", va="top", fontsize=8, color="#8a8984")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_benchmark(m):
    cd = m["claim_detector"]
    rows = [(k, v["macro_f1"]) for k, v in cd["paper_test"].items()] + [("This project\n(TF-IDF + LR)", cd["macro_f1"])]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 4.0))
    cols = [ORANGE if "This project" in r[0] else GREY for r in rows]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=cols, height=0.6)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0.6, 0.92)
    ax.set_xlabel("Macro-F1 on test split")
    ax.grid(axis="y", visible=False)
    _save(fig, "01_claim_detector_benchmark.png", "Environmental-claim detection: lightweight model vs transformers",
          "Macro-F1, test split. Grey: Stammbach et al. (2023) Table 2. The public test release has 265 of the 300 sentences.")


def signal_lift(m):
    prev = pd.DataFrame(m["greenwashing_risk"]["signal_prevalence"]).T
    lab = {s.name: s.label for s in load_signals()}
    kind = {s.name: s.kind for s in load_signals()}
    prev = prev[(prev.sum(axis=1) > 0.02)]
    prev["ratio"] = (prev.greenwashing + 0.005) / (prev.not_greenwashing + 0.005)
    prev = prev.sort_values("ratio")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh([lab[i] for i in prev.index], prev.ratio, color=[ORANGE if kind[i] == "red" else AQUA for i in prev.index],
            height=0.62)
    fig.subplots_adjust(left=0.32)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    for i, v in enumerate(prev.ratio):
        ax.text(v * 1.04, i, f"{v:.1f}×", va="center", fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("How much more often the signal appears in greenwashing vs non-greenwashing claims (log scale)")
    ax.grid(axis="y", visible=False)
    _save(fig, "02_signal_lift.png", "Red flags are over-represented in labelled greenwashing claims",
          f"{m['greenwashing_risk']['n']} claims labelled greenwashing / not (EmeraldMind 2026). Orange = red flag, green = evidence.")


def case_studies(cs):
    rep = cs["reports"]
    names = [f'{r["company"].replace(" Plc", "")}\n{r["year"]}' for r in rep.values()]
    metrics = [("share_vague", "Vague wording", ORANGE), ("share_quantified", "Quantified", BLUE),
               ("share_verified", "Externally verified", AQUA)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    w = 0.26
    for j, (k, lab, col) in enumerate(metrics):
        vals = [r["scorecard"][k] for r in rep.values()]
        xs = [i + (j - 1) * w for i in range(len(vals))]
        ax.bar(xs, vals, width=w - 0.02, color=col, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.008, f"{v:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f'{n}\nGrade {r["scorecard"]["report_grade"]} · {r["scorecard"]["claims"]} claims'
                        for n, r in zip(names, rep.values())])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_ylabel("Share of environmental claims")
    ax.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.02))
    ax.grid(axis="x", visible=False)
    _save(fig, "03_nigerian_case_studies.png", "Nigerian sustainability reports: vague words outnumber verifiable numbers",
          "Same model and rules applied to three public reports. Disclosure quality, not a finding of greenwashing.",
          foot_y=-0.10)


def main():
    m = json.loads(METRICS.read_text())
    cs = json.loads((PROCESSED / "case_studies.json").read_text())
    model_benchmark(m)
    signal_lift(m)
    case_studies(cs)
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
