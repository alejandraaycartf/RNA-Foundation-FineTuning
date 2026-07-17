
import matplotlib
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
matplotlib.use('Agg')
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import umap
import torch
from sklearn.decomposition import PCA
from scipy.stats import levene, chi2
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample
from pathlib import Path
from calm_data_generator.generators.tabular import RealGenerator, QualityReporter


# =========================
# Configuration
# =========================
def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config.json").exists() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root (expected config.json + src/)")

REPO_ROOT = find_repo_root()

PROBLEM_NAME = "ppmi"
CLINICAL_INPUT_CSV = str(REPO_ROOT / "data" / "clinic_data_ppmi_no_mutation.csv")
OMIC_INPUT_CSV = str(REPO_ROOT / "data" / "omic_data_ppmi_no_mutation_protein_coding.csv")
OUTPUT_SYNTHETIC_CSV = str(REPO_ROOT / "data" / "processed" / PROBLEM_NAME / "synthetic" / "combined" / "synthetic_combined.csv")
OUTPUT_DIR = str(REPO_ROOT / "data" / "processed" / PROBLEM_NAME / "synthetic" / "combined")

RANDOM_STATE = 42
N_TOP_GENES = 1000
N_NEIGHBORS_UMAP = 15
MIN_DIST_UMAP = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generar y evaluar datos sinteticos de RNA-seq con TVAE/scVI."
    )
    parser.add_argument(
        "--pca-filter-pct",
        type=float,
        default=0.6,
        help="Porcentaje de muestras a conservar en el filtrado PCA (ej: 0.7 para 70%).",
    )
    parser.add_argument(
        "--pca-alpha",
        type=float,
        default=0.7,
        help="Transparencia (alpha) para los puntos del scatter plot de PCA (0-1).",
    )
    parser.add_argument(
        "--clinical-csv",
        default=CLINICAL_INPUT_CSV,
        help="Ruta del CSV clinico (debe contener la columna objetivo).",
    )
    parser.add_argument(
        "--omic-csv",
        default=OMIC_INPUT_CSV,
        help="Ruta del CSV omico (expresion genica).",
    )
    parser.add_argument(
        "--output-csv",
        default=OUTPUT_SYNTHETIC_CSV,
        help="Nombre del CSV de salida para datos sinteticos (se guarda en figures/).",
    )
    parser.add_argument(
        "--label-col",
        default="diagnosis",
        help="Nombre de la columna objetivo en el CSV clinico.",
    )
    parser.add_argument(
        "--sample-id-col",
        default=None,
        help="Columna de ID para alinear clinico y omico. Si no se define, se alinea por orden de filas.",
    )
    parser.add_argument(
        "--method",
        default="tvae",
        choices=["tvae", "scvi"],
        help="Metodo de generacion (por defecto: tvae).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=300,
        help="Numero de muestras sinteticas. Si no se indica, usa len(train_df).",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Columna objetivo opcional para aumentar separabilidad.",
    )
    parser.add_argument(
        "--differentiation-factor",
        type=float,
        default=None,
        help="Factor de separabilidad si se usa target_col.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=N_TOP_GENES,
        help="Numero de genes altamente variables (HVG).",
    )
    parser.add_argument(
        "--missing-strategy",
        default="median",
        choices=["error", "median", "mean", "zero"],
        help=(
            "Estrategia para valores faltantes tras convertir a numerico: "
            "error, median, mean o zero."
        ),
    )
    parser.add_argument(
        "--drop-nonnumeric-cols",
        action="store_true",
        default=True,
        help="Eliminar columnas que queden completamente no numericas tras conversion.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=10,
        help="Dimensionalidad del espacio latente del TVAE.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de batch para entrenamiento.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Numero de epocas de entrenamiento.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje (learning rate) del optimizador.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Coeficiente de dropout para regularizacion.",
    )
    parser.add_argument(
        "--pca-extreme-filter",
        action="store_true",
        default=True,
        help="Si se activa, selecciona el 70%% de muestras mas extremas en PCA (por defecto: True).",
    )
    parser.add_argument(
        "--no-pca-extreme-filter",
        dest="pca_extreme_filter",
        action="store_false",
        help="Desactiva el filtrado de muestras extremas tras PCA (usa todas las muestras).",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        default=True,
        help="Balancea las clases del train set por oversampling de la minoria antes de generar datos sinteticos (por defecto: True).",
    )
    parser.add_argument(
        "--no-balance-classes",
        dest="balance_classes",
        action="store_false",
        help="Desactiva el balanceo de clases en el train set.",
    )
    parser.add_argument(
        "--separate-groups",
        action="store_true",
        default=False,
        help=(
            "Entrena y evalua el TVAE por separado para controls y PD. "
            "El orden es: train/test split → filtrado PCA extremo en train → "
            "entrenamiento y generacion independiente por grupo."
        ),
    )
    parser.add_argument(
        "--variance-scale",
        type=float,
        default=None,
        help=(
            "Escala post-hoc la varianza de los datos sinteticos para que coincida con la "
            "varianza real multiplicada por este factor. 1.0 = iguala exactamente la varianza "
            "real; 1.5 = 50%% mas varianza que la real. Si no se indica, no se aplica escalado."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directorio de salida para figuras, CSVs y metricas (por defecto: OUTPUT_DIR).",
    )
    parser.add_argument(
        "--gene-set",
        default="hvg",
        choices=["hvg", "deg"],
        help=(
            "Gene set to use for training: 'hvg' loads genes/top1000_hvg.txt, "
            "'deg' loads genes/degs_filtered.txt. "
            "Both files are generated by split.py."
        ),
    )
    parser.add_argument(
        "--splits-dir",
        default="splits",
        help="Directorio con los splits pre-computados por split.py (train/val/test CSVs).",
    )
    parser.add_argument(
        "--genes-dir",
        default="genes",
        help="Directorio con las listas de genes pre-computadas por split.py.",
    )
    return parser.parse_args()


# =========================
# 1. Dataset preprocessing
# =========================
def preprocess_dataset(args):
    """Load pre-split train data from splits/ and filter to the selected gene set from genes/.

    Requires split.py to have been run first to produce:
      splits/train.csv          — log10-normalized full gene matrix for training samples
      genes/top1000_hvg.txt     — top HVG gene names (one per line)
      genes/degs_filtered.txt   — DEG gene names with padj < 0.05 (one per line)

    Returns
    -------
    df_top : pd.DataFrame  — training samples x (selected genes + label_col)
    """
    train_path = os.path.join(args.splits_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Train split not found: {train_path}. Run split.py first."
        )

    train_df = pd.read_csv(train_path)
    print(f"Train data loaded: {len(train_df)} samples, {train_df.shape[1] - 1} genes + label")

    # Load the appropriate gene list
    if args.gene_set == "deg":
        gene_file = os.path.join(args.genes_dir, "degs_filtered.txt")
        gene_type = "DEGs"
    else:
        gene_file = os.path.join(args.genes_dir, "top1000_hvg.txt")
        gene_type = "top HVGs"

    if not os.path.exists(gene_file):
        raise FileNotFoundError(
            f"Gene list not found: {gene_file}. Run split.py first."
        )

    top_genes = pd.read_csv(gene_file, header=None)[0].tolist()
    print(f"Gene list loaded: {len(top_genes)} {gene_type} from {gene_file}")

    available = [g for g in top_genes if g in train_df.columns]
    missing_count = len(top_genes) - len(available)
    if missing_count > 0:
        print(f"Warning: {missing_count} gene(s) from gene list not found in train split and will be skipped.")

    df_top = train_df[available + [args.label_col]].copy()
    print(f"Gene expression matrix filtered to {len(available)} {gene_type}")

    return df_top


# =========================
# 2. Individual selection (PCA filtering)
# =========================
def select_individuals(args, df_top, output_dir):
    """Apply PCA-based sample filtering.

    PD cases: keep the pct% with highest PC1 (RIGHT side).
    Controls: keep the pct% with lowest PC1 (LEFT side).
    If --no-pca-extreme-filter is passed, all samples are kept.

    Returns
    -------
    df_final : pd.DataFrame  — filtered samples x (top genes + label_col), no PC columns
    """
    n_genes_to_keep = df_top.shape[1] - 1  # exclude label col

    X_pca = df_top.drop(columns=[args.label_col]).values
    pca = PCA(n_components=2, random_state=args.random_state)
    coords = pca.fit_transform(X_pca)

    # --- Plot PCA before filtering ---
    plt.figure(figsize=(8, 6))
    colors_all = ['#3b4cc0' if lbl == 0 else '#b40426' for lbl in df_top[args.label_col].values]
    plt.scatter(coords[:, 0], coords[:, 1],
                c=colors_all, alpha=args.pca_alpha, edgecolor='k', s=40)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA of Top {n_genes_to_keep} Genes (before filtering)')
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b4cc0',
                   markeredgecolor='k', markersize=8, label='Control'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#b40426',
                   markeredgecolor='k', markersize=8, label='PD'),
    ]
    plt.legend(handles=handles, title='Muestras')
    plt.tight_layout()
    pca_before_fig = os.path.join(output_dir, f'00_pca_diagnosis_{n_genes_to_keep}genes_before_filtering.png')
    plt.savefig(pca_before_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figura guardada: {pca_before_fig}')

    if not args.pca_extreme_filter:
        print(f"PCA-based filtering skipped: using ALL samples ({len(df_top)})")
        return df_top.copy()

    # --- PCA-based filtering ---
    df_pca = df_top.copy()
    df_pca['PC1'] = coords[:, 0]
    df_pca['PC2'] = coords[:, 1]

    controls = df_pca[df_pca[args.label_col] == 0].copy()
    pd_cases = df_pca[df_pca[args.label_col] == 1].copy()
    pct = args.pca_filter_pct

    # PD: RIGHT side = highest PC1 (sort descending, take first pct%)
    pd_sorted = pd_cases.sort_values('PC1', ascending=False)
    n_keep_pd = int(np.ceil(pct * len(pd_sorted)))
    pd_selected = pd_sorted.head(n_keep_pd)
    pd_discarded = pd_cases.drop(pd_selected.index)
    print(f"PD: keeping {len(pd_selected)}/{len(pd_cases)} ({pct*100:.0f}%) with highest PC1 (RIGHT side)")

    # Controls: LEFT side = lowest PC1 (sort ascending, take first pct%)
    controls_sorted = controls.sort_values('PC1', ascending=True)
    n_keep_controls = int(np.ceil(pct * len(controls_sorted)))
    controls_selected = controls_sorted.head(n_keep_controls)
    controls_discarded = controls.drop(controls_selected.index)
    print(f"Controls: keeping {len(controls_selected)}/{len(controls)} ({pct*100:.0f}%) with lowest PC1 (LEFT side)")

    df_final = pd.concat([controls_selected, pd_selected], axis=0).sort_index()
    print(f"Total samples after PCA-based filtering: {len(df_final)} "
          f"(controls: {len(controls_selected)}, PD: {len(pd_selected)})")

    selected_indices = list(controls_selected.index) + list(pd_selected.index)
    discarded_indices = list(controls_discarded.index) + list(pd_discarded.index)

    # --- Plot PCA after filtering ---
    plt.figure(figsize=(8, 6))
    if len(discarded_indices) > 0:
        discarded_labels = df_top.loc[discarded_indices, args.label_col].values
        discarded_colors = ['#3b4cc0' if lbl == 0 else '#b40426' for lbl in discarded_labels]
        plt.scatter(coords[discarded_indices, 0], coords[discarded_indices, 1],
                    c=discarded_colors, alpha=0.1, edgecolor='none', s=40)
    selected_labels = df_top.loc[selected_indices, args.label_col].values
    selected_colors = ['#3b4cc0' if lbl == 0 else '#b40426' for lbl in selected_labels]
    plt.scatter(coords[selected_indices, 0], coords[selected_indices, 1],
                c=selected_colors, alpha=args.pca_alpha, edgecolor='k', s=40)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA of Top {n_genes_to_keep} Genes (after filtering)')
    plt.legend(handles=handles, title='Muestras')
    plt.tight_layout()
    pca_after_fig = os.path.join(output_dir, f'01_pca_diagnosis_{n_genes_to_keep}genes_after_filtering.png')
    plt.savefig(pca_after_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figura guardada: {pca_after_fig}')

    print(f"PCA filtering: total={len(selected_indices) + len(discarded_indices)}, "
          f"selected={len(selected_indices)}, discarded={len(discarded_indices)}")

    # Drop PC columns before returning
    df_final = df_final.drop(columns=['PC1', 'PC2'])
    return df_final


# =========================
# 3. Class balancing
# =========================
def balance_train(args, train_df):
    """Upsample the minority class in train_df to match the majority class size.

    Returns
    -------
    train_balanced : pd.DataFrame  — balanced training set (features + label_col)
    """
    if not args.balance_classes:
        return train_df

    counts = train_df[args.label_col].value_counts()
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    n_majority = counts[majority_class]

    majority_df = train_df[train_df[args.label_col] == majority_class]
    minority_df = train_df[train_df[args.label_col] == minority_class]

    minority_upsampled = resample(
        minority_df,
        replace=True,
        n_samples=n_majority,
        random_state=args.random_state,
    )

    train_balanced = pd.concat([majority_df, minority_upsampled]).sample(
        frac=1, random_state=args.random_state
    ).reset_index(drop=True)

    print(f"Class balancing: minority class {minority_class} upsampled "
          f"{len(minority_df)} → {n_majority} samples")
    print(f"Balanced train size: {len(train_balanced)} "
          f"(class 0: {(train_balanced[args.label_col]==0).sum()}, "
          f"class 1: {(train_balanced[args.label_col]==1).sum()})")

    return train_balanced


# =========================
# 5. Synthetic data generation
# =========================
def generate_synthetic(args, train_df, output_csv_path, n_samples_override=None):
    """Train the model and generate synthetic data. Saves the result to output_csv_path.

    Returns
    -------
    synthetic_diff : pd.DataFrame  — synthetic samples with label_col + gene columns
    """
    print(f"Generating synthetic data with {args.method.upper()}...")
    generator = RealGenerator(random_state=args.random_state)

    n_samples = n_samples_override if n_samples_override is not None else (
        args.n_samples if args.n_samples is not None else len(train_df)
    )

    generate_kwargs = {
        "data": train_df,
        "method": args.method,
        "n_samples": n_samples,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
    }
    if args.target_col is not None:
        generate_kwargs["target_col"] = args.target_col
    if args.differentiation_factor is not None:
        generate_kwargs["differentiation_factor"] = args.differentiation_factor

    # TVAE (synthcity) does not support learning_rate or dropout
    if args.method == "tvae":
        generate_kwargs.pop("learning_rate", None)
        generate_kwargs.pop("dropout", None)

    synthetic_diff = generator.generate(**generate_kwargs)

    # Clip to 0: log10(count+1) cannot be negative; TVAE decoder is unbounded.
    gene_cols_gen = [c for c in synthetic_diff.columns if c != args.label_col]
    synthetic_diff[gene_cols_gen] = synthetic_diff[gene_cols_gen].clip(lower=0)

    # --- Post-hoc variance scaling ---
    if args.variance_scale is not None:
        gene_cols = [c for c in synthetic_diff.columns if c != args.label_col]
        real_gene_cols = [c for c in train_df.columns if c != args.label_col]
        common_genes = [c for c in gene_cols if c in real_gene_cols]

        real_std  = train_df[common_genes].std()
        synth_std = synthetic_diff[common_genes].std().replace(0, np.nan)
        synth_mean = synthetic_diff[common_genes].mean()

        scaled = synth_mean + (synthetic_diff[common_genes] - synth_mean) * (
            args.variance_scale * real_std / synth_std
        )
        # Clip again after scaling: stretching around the mean can re-introduce negatives
        # when real_std > synth_std (the common case) and values sit close to 0.
        synthetic_diff[common_genes] = scaled.clip(lower=0).fillna(synth_mean)
        print(f"Variance scaling applied (factor={args.variance_scale}): "
              f"synthetic std now targets {args.variance_scale}x real std.")

    # Save with an ID column for traceability
    synthetic_meta = pd.DataFrame({
        "ID": [f"synthetic_{i}" for i in range(len(synthetic_diff))],
        args.label_col: synthetic_diff[args.label_col].values,
    })
    synthetic_to_save = pd.concat(
        [synthetic_meta.reset_index(drop=True),
         synthetic_diff.drop(columns=[args.label_col]).reset_index(drop=True)],
        axis=1,
    )
    synthetic_to_save.to_csv(output_csv_path, index=False)
    print(f"Synthetic data saved to: {output_csv_path}")

    return synthetic_diff


# =========================
# 6. Quality evaluation
# =========================
def evaluate_quality(args, df_final, synthetic_diff, output_dir, prefix=""):
    """Run SDMetrics quality report and gene variance comparison.

    Outputs (all in output_dir, optionally prefixed):
    - {prefix}sdmetrics_property_scores.csv
    - {prefix}sdmetrics_<property>_details.csv
    - {prefix}sdmetrics_full_report.json
    - {prefix}gene_variance_comparison.csv
    - {prefix}gene_variance_real_vs_synthetic.png
    - {prefix}gene_variance_ratio_distribution.png
    """
    if prefix and not prefix.endswith("_"):
        prefix = prefix + "_"
    from sdmetrics.reports.single_table import QualityReport
    import json

    # -------------------------
    # Align columns — gene columns only (exclude label and any metadata)
    # -------------------------
    all_common = set(df_final.columns) & set(synthetic_diff.columns)
    gene_cols = sorted(all_common - {args.label_col})
    real_aligned = df_final[gene_cols].copy()
    synth_aligned = synthetic_diff[gene_cols].copy()
    print(f"Aligning on {len(gene_cols)} gene columns for quality evaluation (label excluded).")

    # -------------------------
    # Build SDMetrics metadata
    # -------------------------
    metadata = {"columns": {}}
    for col in gene_cols:
        if pd.api.types.is_numeric_dtype(real_aligned[col]):
            metadata["columns"][col] = {"sdtype": "numerical"}
        elif pd.api.types.is_datetime64_any_dtype(real_aligned[col]):
            metadata["columns"][col] = {"sdtype": "datetime"}
        else:
            metadata["columns"][col] = {"sdtype": "categorical"}

    # -------------------------
    # Generate QualityReport
    # -------------------------
    report = QualityReport()
    report.generate(real_aligned, synth_aligned, metadata)

    overall_score = report.get_score()
    print(f"\nOverall quality score: {overall_score:.4f}")

    # -------------------------
    # Per-property scores
    # -------------------------
    properties_df = report.get_properties()
    print("\nPer-property scores:")
    print(properties_df.to_string(index=False))

    prop_summary_path = os.path.join(output_dir, f"{prefix}sdmetrics_property_scores.csv")
    properties_df_out = properties_df.copy()
    properties_df_out.insert(0, 'overall_score', overall_score)
    properties_df_out.to_csv(prop_summary_path, index=False)
    print(f"Property scores saved to: {prop_summary_path}")

    # -------------------------
    # Per-column / per-pair details (the individual metrics)
    # -------------------------
    all_details = {}
    for prop in properties_df['Property'].tolist():
        details_df = report.get_details(prop)
        all_details[prop] = details_df
        safe_name = prop.lower().replace(" ", "_")
        csv_path = os.path.join(output_dir, f"{prefix}sdmetrics_{safe_name}_details.csv")
        details_df.to_csv(csv_path, index=False)
        print(f"Details for '{prop}' saved to: {csv_path}")

    # -------------------------
    # Full JSON report
    # -------------------------
    summary = {
        "overall_quality_score": overall_score,
        "property_scores": (
            properties_df[['Property', 'Score']]
            .set_index('Property')['Score']
            .to_dict()
        ),
        "column_details": {
            prop: df.to_dict(orient='records')
            for prop, df in all_details.items()
        },
    }
    json_path = os.path.join(output_dir, f"{prefix}sdmetrics_full_report.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Full SDMetrics report saved to: {json_path}")

    # -------------------------
    # Gene variance comparison
    # -------------------------
    real_var = real_aligned.var()
    synth_var = synth_aligned.var()

    var_df = pd.DataFrame({
        'gene': gene_cols,
        'real_variance': real_var.values,
        'synthetic_variance': synth_var.values,
        'variance_ratio': (synth_var / real_var.replace(0, np.nan)).values,
        'variance_diff': (synth_var - real_var).values,
        'abs_variance_diff': (synth_var - real_var).abs().values,
    })
    var_df = var_df.sort_values('abs_variance_diff', ascending=False).reset_index(drop=True)


    var_csv_path = os.path.join(output_dir, f"{prefix}gene_variance_comparison.csv")
    var_df.to_csv(var_csv_path, index=False)
    print(f"\nGene variance comparison saved to: {var_csv_path}")

    print("\nVariance summary statistics:")
    print(f"  Mean real variance:       {real_var.mean():.4f}")
    print(f"  Mean synthetic variance:  {synth_var.mean():.4f}")
    print(f"  Mean variance ratio:      {var_df['variance_ratio'].mean():.4f}")
    print(f"  Median variance ratio:    {var_df['variance_ratio'].median():.4f}")
    print(f"  Genes with ratio > 1.5:   {(var_df['variance_ratio'] > 1.5).sum()}")
    print(f"  Genes with ratio < 0.67:  {(var_df['variance_ratio'] < 0.67).sum()}")
    print("\nTop 10 genes with largest absolute variance difference:")
    print(var_df[['gene', 'real_variance', 'synthetic_variance', 'variance_ratio']].head(10).to_string(index=False))

    # Scatter plot: real vs synthetic variance
    plt.figure(figsize=(7, 6))
    plt.scatter(real_var.values, synth_var.values, alpha=0.4, s=10, color='steelblue')
    max_val = max(real_var.max(), synth_var.max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='y = x')
    plt.xlabel('Real gene variance')
    plt.ylabel('Synthetic gene variance')
    plt.title('Gene Variance: Real vs Synthetic')
    plt.legend()
    plt.tight_layout()
    var_fig_path = os.path.join(output_dir, f"{prefix}gene_variance_real_vs_synthetic.png")
    plt.savefig(var_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Variance scatter plot saved to: {var_fig_path}")

    # Distribution of variance ratio
    plt.figure(figsize=(7, 5))
    plt.hist(var_df['variance_ratio'].dropna(), bins=50, color='steelblue', edgecolor='white')
    plt.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Ratio = 1')
    plt.xlabel('Synthetic / Real variance ratio')
    plt.ylabel('Number of genes')
    plt.title('Distribution of Gene Variance Ratio (Synthetic / Real)')
    plt.legend()
    plt.tight_layout()
    ratio_fig_path = os.path.join(output_dir, f"{prefix}gene_variance_ratio_distribution.png")
    plt.savefig(ratio_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Variance ratio distribution saved to: {ratio_fig_path}")


# =========================
# Main
# =========================
def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Canonical path for the synthetic CSV (always inside output_dir)
    csv_name = os.path.basename(args.output_csv)
    output_csv_path = os.path.join(output_dir, csv_name)

    # Also accept a legacy location (e.g. file created before this convention)
    legacy_path = args.output_csv
    if os.path.exists(output_csv_path):
        synthetic_path = output_csv_path
    elif os.path.exists(legacy_path):
        synthetic_path = legacy_path
    else:
        synthetic_path = None

    stats_path = os.path.join(output_dir, "sdmetrics_full_report.json")

    # ------------------------------------------------------------------
    # Fast path: synthetic data already exists
    # ------------------------------------------------------------------
    if synthetic_path is not None:
        print(f"Synthetic file found: {synthetic_path}")
        synthetic_diff = pd.read_csv(synthetic_path)
        if 'ID' in synthetic_diff.columns:
            synthetic_diff = synthetic_diff.drop(columns=['ID'])

        print("Synthetic class distribution:")
        print(synthetic_diff[args.label_col].value_counts())

        if os.path.exists(stats_path):
            print(f"Statistics already exist at {stats_path}. Nothing to do.")
            return

        # Load train split as the real reference for evaluation
        df_real = preprocess_dataset(args)

        evaluate_quality(args, df_real, synthetic_diff, output_dir)
        return

    # ------------------------------------------------------------------
    # Separate-groups pipeline: train/evaluate TVAE per diagnosis group
    # ------------------------------------------------------------------
    if args.separate_groups:
        df_top = preprocess_dataset(args)
        # PCA extreme filtering on train (reset_index so positional coords match DataFrame indices)
        train_filtered = select_individuals(args, df_top.reset_index(drop=True), output_dir)
        # 3. Per-group: train TVAE → generate → evaluate
        group_map = {0: "controls", 1: "pd"}
        n_syn = args.n_samples if args.n_samples is not None else 200
        for label_val, group_name in group_map.items():
            group_df = train_filtered[
                train_filtered[args.label_col] == label_val
            ].reset_index(drop=True)
            print(f"\n{'='*55}")
            print(f"Group: {group_name.upper()}  ({len(group_df)} training samples)")
            print(f"{'='*55}")
            output_csv_group = os.path.join(output_dir, f"synthetic_{group_name}.csv")
            synthetic_group = generate_synthetic(
                args, group_df, output_csv_group, n_samples_override=n_syn
            )
            evaluate_quality(args, group_df, synthetic_group, output_dir, prefix=group_name)
        return

    # ------------------------------------------------------------------
    # Full pipeline: no synthetic data yet
    # ------------------------------------------------------------------
    df_top = preprocess_dataset(args)
    df_final = select_individuals(args, df_top, output_dir)
    train_df = balance_train(args, df_final)
    synthetic_diff = generate_synthetic(args, train_df, output_csv_path)
    evaluate_quality(args, train_df, synthetic_diff, output_dir)


if __name__ == "__main__":
    main()
