"""Top-K sweep of enrichment metrics over K = 25, 50, 100, 200.

For every result file, runs the same enrichment as run_enrichment.py but at
several gene-set sizes, so the IC/FA trajectories of baseline / phase1 / phase2
can be compared across K. Writes results/alejandra_genes/summary_sweep_IC_FA.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from gprofiler import GProfiler

from run_enrichment import (ROOT, PROBLEMS, compute_metrics, file_kind,
                            iter_result_files, load_background, ranked_query,
                            run_gprofiler)

KS = [25, 50, 100, 150, 200, 250, 300]
OUT = Path("/home/joseadrian/foundation_model_scrnaseq/results/alejandra_genes")


def main() -> None:
    gp = GProfiler(return_dataframe=True)
    rows: list[dict] = []
    for problem in PROBLEMS:
        background = load_background(problem)
        for phase, path in iter_result_files(ROOT / problem):
            method, feature = file_kind(path.name)
            genes, _ = ranked_query(path, method)
            n_total = len(genes)
            for k in KS:
                q = genes[:k]
                sig = run_gprofiler(gp, q, background) if q else None
                metrics = compute_metrics(sig, len(q), len(background))
                rows.append({
                    "problem": problem, "phase": phase, "source_file": path.name,
                    "method": method, "feature_type": feature, "top_k": k,
                    "n_query_genes": len(q), "n_candidate_genes": n_total,
                    "n_background": len(background), **metrics,
                })
                print(f"[{problem}/{phase}] {path.name} K={k:3d} "
                      f"q={len(q):3d} IC={metrics['information_content']:.3f} "
                      f"FA={metrics['functional_abundance']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "summary_sweep_IC_FA.csv", index=False)
    print(f"\nSaved {len(rows)} rows to {OUT / 'summary_sweep_IC_FA.csv'}")


if __name__ == "__main__":
    main()
