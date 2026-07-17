"""g:Profiler enrichment for baseline / phase1 / phase2 gene rankings.

For every result file we build a query gene set and run a g:Profiler
enrichment against a problem-specific background, then summarise each result
with two metrics (Information Content and Functional Abundance) so that
baseline, phase1 and phase2 can be compared.

Query gene selection (uniform across methods, for a fair comparison)
  - Every file -> top-K genes by score (TOP_K=100).
    LASSO ranks by |coef|, RF by importance (zero-importance genes excluded),
    phase1/phase2 by attention score.

Background (custom, passed to g:Profiler): one fixed gene universe per problem
  - PPMI    -> all genes in the omic table (column headers).
  - frontal -> intersected_genes_gtex_archs4_brain_vs_blood.txt
  - colon   -> intersected_genes_gtex_archs4_colon.txt

Enrichment options: no_iea=True, custom background, and terms whose query
overlap is 1 or 2 genes are removed (intersection_size >= 3 kept).

Metrics (same definitions as code/pipeline_scgpt/steps/step7_explainability.py)
  - Information Content  = mean over kept terms of -log2(term_size / n_background)
  - Functional Abundance = sum(-log10(p_value)) / n_query_genes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gprofiler import GProfiler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(
    "/home/joseadrian/foundation_model_scrnaseq/data/alejandra_genes/"
    "resultados_3_problemas"
)
PROBLEMS = ["PPMI", "frontal_cortex_vs_blood", "colon_sigmoid_vs_colon_transverse"]
GPROFILER_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "HP",
                     "HPA", "WP", "CORUM"]
MIN_INTERSECTION = 3  # drop terms whose query overlap is 1 or 2 genes
TOP_K = 100  # genes kept per result file (uniform across methods for a fair comparison)

# Each problem uses a single fixed background gene universe, applied to all its
# result files. "csv_cols" = gene symbols are the column headers (sample x gene
# table); "lines" = one gene symbol per line.
PROBLEM_BACKGROUND = {
    "PPMI": ("csv_cols", Path(
        "/home/joseadrian/foundation_model_scrnaseq/data/PPMI/"
        "omic_data_ppmi_sc_pc_nm_ids.csv")),
    "frontal_cortex_vs_blood": ("lines", ROOT / "frontal_cortex_vs_blood" /
        "intersected_genes_gtex_archs4_brain_vs_blood.txt"),
    "colon_sigmoid_vs_colon_transverse": ("lines", ROOT /
        "colon_sigmoid_vs_colon_transverse" /
        "intersected_genes_gtex_archs4_colon.txt"),
}


# --------------------------------------------------------------------------- #
# Loading queries and backgrounds
# --------------------------------------------------------------------------- #
def file_kind(name: str) -> tuple[str, str]:
    """Return (method, feature_type) inferred from a result file name."""
    low = name.lower()
    if low.startswith("lasso"):
        method = "lasso"
    elif low.startswith("rf"):
        method = "rf"
    else:
        method = "attention"
    feature = "HVG" if "hvg" in low else "DEG"
    return method, feature


def ranked_query(path: Path, method: str) -> tuple[list[str], np.ndarray]:
    """Return (genes ranked by score desc, their scores).

    LASSO ranks by |coef|, RF by importance (zero-importance genes excluded),
    phase1/phase2 by attention score.
    """
    df = pd.read_csv(path)
    if df.empty:
        return [], np.array([])
    gene_col = "gene_name" if "gene_name" in df.columns else "gene"
    if method == "lasso":
        score_col = "abs_coef"
    elif "importance" in df.columns:
        score_col = "importance"
    else:
        score_col = "Attention_Score"
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    if method == "rf":  # ignore zero-importance genes
        df = df[df[score_col] > 0].reset_index(drop=True)
    return df[gene_col].astype(str).tolist(), df[score_col].to_numpy(dtype=float)


def load_background(problem: str) -> list[str]:
    """Single fixed background gene universe for a problem."""
    kind, path = PROBLEM_BACKGROUND[problem]
    if kind == "csv_cols":  # sample x gene table; genes are the column headers
        genes = pd.read_csv(path, nrows=0).columns[1:].astype(str)
    else:  # one gene symbol per line
        genes = pd.read_csv(path, header=None).iloc[:, 0].astype(str)
    return pd.Index(genes).str.strip().dropna().unique().tolist()


# --------------------------------------------------------------------------- #
# Enrichment + metrics
# --------------------------------------------------------------------------- #
def run_gprofiler(gp: GProfiler, genes: list[str], background: list[str]):
    df = gp.profile(
        organism="hsapiens",
        query=genes,
        sources=GPROFILER_SOURCES,
        no_iea=True,
        no_evidences=False,
        domain_scope="custom",
        background=list(background),
    )
    if df is None or len(df) == 0:
        return df
    sig = df[df["significant"]].copy()
    return sig[sig["intersection_size"] >= MIN_INTERSECTION].reset_index(drop=True)


def compute_metrics(sig_df, n_query: int, n_background: int) -> dict:
    """Information Content and Functional Abundance for one enrichment result."""
    if sig_df is None or len(sig_df) == 0:
        return {"information_content": 0.0, "functional_abundance": 0.0,
                "n_significant_terms": 0}
    log_p = -np.log10(sig_df["p_value"].clip(lower=1e-300))
    fa = float(log_p.sum() / max(n_query, 1))
    ic = -np.log2(sig_df["term_size"].clip(lower=1) / max(n_background, 1))
    return {"information_content": float(ic.mean()),
            "functional_abundance": fa,
            "n_significant_terms": int(len(sig_df))}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def iter_result_files(problem_dir: Path):
    """Yield (phase, path) for every file to enrich in a problem."""
    for path in sorted((problem_dir / "baseline").glob("*.csv")):
        if path.name.lower().startswith(("rf", "lasso")):
            yield "baseline", path
    for phase in ("phase1", "phase2"):
        for path in sorted((problem_dir / phase).glob("*.csv")):
            yield phase, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/joseadrian/foundation_model_scrnaseq/"
                    "results/alejandra_genes")
    args = ap.parse_args()
    out_root = Path(args.out)

    gp = GProfiler(return_dataframe=True)
    summary: list[dict] = []

    for problem in PROBLEMS:
        problem_dir = ROOT / problem
        background = load_background(problem)  # single fixed background per problem
        for phase, path in iter_result_files(problem_dir):
            method, feature = file_kind(path.name)
            ranked, _ = ranked_query(path, method)
            n_total = len(ranked)
            genes = ranked[:TOP_K]

            print(f"[{problem}/{phase}] {path.name}: method={method} "
                  f"feature={feature} query={len(genes)}/{n_total} "
                  f"bg={len(background)}")

            sig = run_gprofiler(gp, genes, background) if genes else None
            metrics = compute_metrics(sig, len(genes), len(background))
            summary.append({
                "problem": problem, "phase": phase, "source_file": path.name,
                "method": method, "feature_type": feature,
                "n_total_genes": n_total, "n_query_genes": len(genes),
                "n_background": len(background), **metrics,
            })
            print(f"    -> IC={metrics['information_content']:.3f} "
                  f"FA={metrics['functional_abundance']:.3f} "
                  f"terms={metrics['n_significant_terms']}")

    out_root.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_root / "summary_IC_FA.csv", index=False)
    print(f"\nSaved summary with {len(summary_df)} rows to "
          f"{out_root / 'summary_IC_FA.csv'}")


if __name__ == "__main__":
    main()
