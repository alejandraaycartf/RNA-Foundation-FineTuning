"""Line plots of the top-K sweep: IC and FA vs K for each case.

Reads results/alejandra_genes/summary_sweep_IC_FA.csv and draws, per metric,
one panel per (dataset, category) with one line per case
(Baseline - Lasso, Baseline - RF, Phase 1, Phase 2) over K. The IC plot uses
K = 25/50/100/150/200; the FA plot additionally uses K = 250/300.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("/home/joseadrian/foundation_model_scrnaseq/results/alejandra_genes")
PROBLEMS = ["PPMI", "frontal_cortex_vs_blood", "colon_sigmoid_vs_colon_transverse"]
DATASET_NAMES = {
    "PPMI": "PPMI",
    "frontal_cortex_vs_blood": "Frontal Cortex vs Blood",
    "colon_sigmoid_vs_colon_transverse": "Colon Sigmoid vs Colon Transverse",
}
CASES = ["baseline-lasso", "baseline-rf", "phase1", "phase2"]
CASE_LABELS = {"baseline-lasso": "Baseline - Lasso", "baseline-rf": "Baseline - RF",
               "phase1": "Phase 1", "phase2": "Phase 2"}
CASE_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
# (metric column, display name, K values shown)
METRICS = [
    ("information_content", "Information Content", [25, 50, 100, 150, 200]),
    ("functional_abundance", "Functional Abundance", [25, 50, 100, 150, 200, 250, 300]),
]


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    data_type = np.where(df["source_file"].str.contains("Synthetic"),
                         "Real_Synthetic", "Real")
    df["variant"] = df["feature_type"] + "_" + data_type
    df["case"] = np.where(df["phase"] == "baseline",
                          "baseline-" + df["method"], df["phase"])
    return df


def category_label(variant: str) -> str:
    """'DEG_Real_Synthetic' -> 'DEG - Real + Synthetic'."""
    feature, _, data = variant.partition("_")
    return f"{feature} - {data.replace('_', ' + ')}"


def plot_metric_problem(df: pd.DataFrame, problem: str, metric: str, name: str,
                        ks: list[int], out_png: Path) -> None:
    """One figure for a single (metric, dataset): a row of category panels."""
    sub = df[(df["problem"] == problem) & (df["top_k"].isin(ks))]
    variants = sorted(sub["variant"].unique())
    xpos = np.arange(len(ks))
    fig = plt.figure(figsize=(4 * len(variants), 4.2))
    # Dedicated top strip for the legend so it never overlaps the panels.
    legend_sf, row_sf = fig.subfigures(2, 1, height_ratios=[0.35, 3.6])
    handles = [plt.Line2D([], [], color=col, marker="o", ms=4,
                          label=CASE_LABELS[case])
               for case, col in zip(CASES, CASE_COLORS)]
    legend_sf.legend(handles=handles, ncol=len(CASES), frameon=False,
                     loc="center", fontsize=11)
    row_sf.suptitle(DATASET_NAMES[problem], fontsize=13, fontweight="bold")
    row_sf.subplots_adjust(top=0.82)
    axes = row_sf.subplots(1, len(variants), squeeze=False)[0]
    for ax, variant in zip(axes, variants):
        vsub = sub[sub["variant"] == variant]
        for case, color in zip(CASES, CASE_COLORS):
            line = vsub[vsub["case"] == case].set_index("top_k")[metric]
            if line.empty:
                continue
            ax.plot(xpos, [line.get(k, np.nan) for k in ks], marker="o",
                    ms=4, color=color, label=CASE_LABELS[case])
        ax.set_xticks(xpos)
        ax.set_xticklabels(ks)
        ax.set_title(category_label(variant), fontsize=10)
        ax.set_xlabel("Top-K genes")
        ax.grid(ls=":", alpha=0.5)
    axes[0].set_ylabel(name)
    fig.suptitle(f"{name} through Top-K Genes - Baseline vs Phase 1 vs Phase 2",
                 fontsize=14, y=1.06)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def main() -> None:
    df = annotate(pd.read_csv(RESULTS / "summary_sweep_IC_FA.csv"))
    for metric, name, ks in METRICS:
        for problem in PROBLEMS:
            plot_metric_problem(df, problem, metric, name, ks,
                                RESULTS / f"sweep_{metric}_{problem}.png")


if __name__ == "__main__":
    main()
