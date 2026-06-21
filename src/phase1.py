import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import umap
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from IPython.display import display

# ==========================================
# 1. Global configuration and paths
# ==========================================
TEJIDO = "colon_sigmoid_vs_colon_transverse"  
PROJ_PATH   = "/home/alejandrayf/GeneRAIN"
SPLITS_DIR  = f"{PROJ_PATH}/data/generar_sinteticos/splits/tejidos/{TEJIDO}"
GENES_DIR   = f"{PROJ_PATH}/data/generar_sinteticos/genes/tejidos/{TEJIDO}"
# Phase 2 will use the same directory structure as Phase 1 for consistency:
PHASE1_DIR  = Path(PROJ_PATH) / "results" /"tejidos" / "phase1" / TEJIDO
RESULTS_CSV = PHASE1_DIR / "best_config_generain.csv"

PARAM_JSON  = f"{PROJ_PATH}/jsons/exp3_BERT_Pred_Genes_Binning_By_Gene.param_config.json"
CHECKPOINT  = f"{PROJ_PATH}/data/models/GeneRAIN.BERT_Pred_Genes_Binning_By_Gene.pth"
LABEL_COL   = "tissue"

os.environ['PARAM_JSON_FILE'] = PARAM_JSON
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append(f"{PROJ_PATH}/src")
sys.path.append(PROJ_PATH)

from utils.config_loader import Config
import anal_utils as au
from data.adata import Adata
from utils.utils import get_device
from utils.checkpoint_utils import load_checkpoint
from data.GetBinsByGeneForNewSamples import get_bins_by_gene_for_new_samples
from train.common import initiate_model

PHASE1_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.max_colwidth", None)
config = Config()

def _load_real_split(split_path: Path, prefix: str):
    df = pd.read_csv(split_path, index_col=0)
    df.index = [f"{prefix}_{i}" for i in range(len(df))]
    return df

# ==========================================
# 2. Loading data splits and preparing binned datasets for HVG1000 and DEG
# ==========================================
print("--- Loading data splits ---")
df_train_real = _load_real_split(Path(SPLITS_DIR) / "train.csv", "RealTrain")
df_val_real   = _load_real_split(Path(SPLITS_DIR) / "val.csv", "RealVal")
df_test_real  = _load_real_split(Path(SPLITS_DIR) / "test.csv", "RealTest")
df_merged_all = pd.concat([df_train_real, df_val_real, df_test_real], axis=0)

# Loading gene lists for HVG1000 and DEG datasets
with open(Path(GENES_DIR) / "top1000_hvg.txt", "r") as f:
    hvg_1000_genes = [line.strip() for line in f if line.strip()]
with open(Path(GENES_DIR) / "degs_filtered.txt", "r") as f:
    deg_genes = [line.strip() for line in f if line.strip()]

samples_subsampled_file = config.proj_path + "/data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_gene2vec_0.005_subsampled.npy"
symbol_file = config.proj_path + "/data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_gene2vec_0.005_subsampled.gene_symbols.txt"

def prepare_binned_adata(df_merged, gene_list):
    df_sub = df_merged[[LABEL_COL, *gene_list]].copy()
    mat_expr = df_sub[gene_list].values.astype(np.float32)
    
    mat_binned, mask_samples, mask_genes, _ = get_bins_by_gene_for_new_samples(
        samples_subsampled_file, symbol_file,
        gene_by_sample_expr_mat=mat_expr.T,
        quantified_gene_list=gene_list,
        output_prefix=None, num_gene_bin=2000, min_total_count=-1,
    )
    
    adata = Adata(
        np.array(df_sub.index)[mask_samples],
        np.array(gene_list)[mask_genes],
        mat_binned.T[:, mask_genes],
    )
    return df_sub, adata, mask_samples

print("Binning HVG1000 data...")
df_hvg, adata_hvg, mask_hvg = prepare_binned_adata(df_merged_all, hvg_1000_genes)

print("Binning DEG data...")
df_deg, adata_deg, mask_deg = prepare_binned_adata(df_merged_all, deg_genes)

# ==========================================
# 3. Loading the model and extracting embeddings for HVG1000 and DEG datasets
# ==========================================
device = get_device()
print(f"Loading GeneRAIN model to {device}...")
model = initiate_model()
model, _, _ = load_checkpoint(model, None, CHECKPOINT, None)
model = model.to(device).eval()

print("Extracting HVG1000 Embeddings...")
emb_hvg = au.get_generain_embeddings(adata_hvg.X, adata_hvg.var_names.tolist(), model, device, batch_size=8)

print("Extracting DEG Embeddings...")
emb_deg = au.get_generain_embeddings(adata_deg.X, adata_deg.var_names.tolist(), model, device, batch_size=8)

datasets_emb = {
    "HVG1000": au.splits(df_hvg, mask_hvg, emb_hvg, label_col=LABEL_COL),
    "DEG":     au.splits(df_deg, mask_deg, emb_deg, label_col=LABEL_COL)
}

# ==========================================
# 4.Training MLP classifier on embeddings and saving results
# ==========================================

experiment_grid = [
    ("HVG1000", "Real"),
    ("DEG", "Real"),
]
mlp_plot_payload = []
results_rows = []

for ds_name, train_setup in experiment_grid:
    print(f"\n================ Running MLP on {ds_name} ({train_setup}) ================")
    ds = datasets_emb[ds_name]
    au.set_seed(42)

    X_train_real_raw = ds["X_train_real"]
    y_train_real_enc = ds["y_train_real_enc"]
    X_val_raw = ds["X_val"]
    y_val_enc = ds["y_val_enc"]
    X_test_raw = ds["X_test"]
    y_test_enc = ds["y_test_enc"]

    # balance the training set using balanced_real_indices
    idx_train_real_bal = au.balanced_real_indices(
        y_train_real_enc, ds["pd_label"], ds["control_label"], seed=42,
    )
    X_train_bal = X_train_real_raw[idx_train_real_bal]
    y_train_bal = y_train_real_enc[idx_train_real_bal]

    # scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    raw_mlp_param_grid = {
        "dropout": [0.3, 0.4, 0.5], "batch_size": [32], "max_epochs": [20],
        "patience": [20], "min_delta": [0.005], "schedule_ratio": [0.9],
    }

    train_ba, val_ba, test_ba, preds, best_params, mlp_details = au.run_mlp_optuna_grid(
        X_train=X_train_scaled, y_train=y_train_bal, X_test=X_test_scaled, y_test=y_test_enc,
        n_classes=len(ds["le"].classes_), device=device, seed=42,
        X_val=X_val_scaled, y_val=y_val_enc, optimize_metric="val_balanced_accuracy",
        return_details=True, lr_grid=[5e-4, 1e-4], param_grid=raw_mlp_param_grid,
        final_save_all_epochs=False, final_early_stopping=False
    )
    mlp_plot_payload.append({
        "dataset": ds_name,
        "train_setup": train_setup,
        "best_params": best_params,
        "final_history": mlp_details["final_history"],
        "final_diagnostics": mlp_details.get("final_diagnostics"),
    })
    tn, fp = confusion_matrix(y_test_enc, preds, labels=[0, 1]).ravel()[:2]
    results_rows.append({
        "model": "MLP",
        "dataset": ds_name,
        "train_setup": train_setup,
        "train_ba": float(train_ba),
        "val_ba": float(val_ba),
        "test_ba": float(test_ba),
        "test_f1": float(f1_score(y_test_enc, preds, average="macro", zero_division=0)),
        "test_sensitivity": float(recall_score(y_test_enc, preds, average="macro", zero_division=0)),
        "test_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "best_params": str(best_params),
    })
    # -------------------------------------------------------------
    # Save artifacts for Phase 1 (numpy arrays, best_params.json, best_model.pt)
    # -------------------------------------------------------------
    condition = "original" if train_setup == "Real" else "augmented"
    out_dir = PHASE1_DIR / ds_name / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving numpy arrays to {out_dir} ...")
    np.save(out_dir / "emb_train.npy", ds["X_train_real"])
    np.save(out_dir / "labels_train.npy", ds["y_train_real_enc"])
    np.save(out_dir / "sources_train.npy", np.zeros(len(ds["y_train_real_enc"]), dtype=np.int64))

    np.save(out_dir / "emb_val.npy", ds["X_val"])
    np.save(out_dir / "labels_val.npy", ds["y_val_enc"])

    np.save(out_dir / "emb_test.npy", ds["X_test"])
    np.save(out_dir / "labels_test.npy", ds["y_test_enc"])
    
    saved_artifacts = au.save_phase1_artifacts(
        phase1_root=PHASE1_DIR,
        dataset=ds_name,
        train_setup=train_setup,
        best_params=best_params,
        state_dict=mlp_details["best_model_state_dict"],
        checkpoint_src=CHECKPOINT,
        test_preds=preds,
        test_labels=y_test_enc
    )
    
    print(f"Phase 1 artifacts saved successfully for {ds_name} ({condition})")
if results_rows:
    df_results = pd.DataFrame(results_rows).sort_values(["dataset", "model", "train_setup"]).reset_index(drop=True)
    print("\nExperiment results - FASE 1:")
    display(df_results)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"Results saved in {RESULTS_CSV}")

def plot_umap_by_class(X_test, y_test, le, dataset_name, save_dir):
    """Genera un UMAP del set de test coloreado por la clase real."""
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_test)
    labels_text = le.inverse_transform(y_test)

    df_umap = pd.DataFrame({
        'UMAP 1': embedding[:, 0], 'UMAP 2': embedding[:, 1], 'Tissue': labels_text
    })

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_umap, x='UMAP 1', y='UMAP 2', hue='Tissue', palette='viridis', alpha=0.7, s=60)
    plt.title(f"UMAP Projection - Test Set: {dataset_name}", fontsize=15)
    plt.grid(alpha=0.3)
    
    save_path = Path(save_dir) / f"umap_{dataset_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"UMAP saved to: {save_path}")
    plt.close()

def plot_best_mlp_diagnostics_grid(payloads, section_title, dataset_order ,results_dir=PHASE1_DIR):
    if not payloads:
        print(f"No history available for {section_title}")
        return

    payload_map = {(p["dataset"], p["train_setup"]): p for p in payloads}
    n_items = len(dataset_order)

    fig, axes = plt.subplots(n_items, 2, figsize=(14, 4.5 * n_items), squeeze=False)
    fig.suptitle(f"{section_title}\nTraining diagnostics", fontsize=16, fontweight="bold", y=1.01)

    for row_idx, key in enumerate(dataset_order):
        payload = payload_map.get(key)
        ax_loss, ax_ba = axes[row_idx, 0], axes[row_idx, 1]
        dataset_name, train_setup = key

        if payload is None or not payload.get("final_history"):
            ax_loss.axis("off"); ax_ba.axis("off"); continue

        history = payload["final_history"]
        df_hist = pd.DataFrame(history).sort_values("epoch").reset_index(drop=True)
        
        required = {"epoch", "train_loss", "val_loss", "s_val_loss", "train_balanced_accuracy", "val_balanced_accuracy", "s_val_ba", "score"}
        if not required.issubset(df_hist.columns):
            ax_loss.axis("off"); ax_ba.axis("off"); continue

        best_idx = int(df_hist["score"].idxmax())
        best_epoch = int(df_hist.loc[best_idx, "epoch"])

        # LOSS PLOT
        ax_loss.plot(df_hist["epoch"], df_hist["train_loss"], label="Train loss", linewidth=2)
        ax_loss.plot(df_hist["epoch"], df_hist["val_loss"], label="Validation loss", linewidth=2)
        ax_loss.plot(df_hist["epoch"], df_hist["s_val_loss"], linestyle="--", linewidth=2, label="Smoothed val loss")
        ax_loss.scatter(best_epoch, df_hist.loc[best_idx, "s_val_loss"], s=60, color="red", zorder=5, label=f"Best epoch ({best_epoch})")
        ax_loss.axvline(best_epoch, linestyle=":", alpha=0.7, color="red")
        ax_loss.set_title(f"{dataset_name} | {train_setup}\nLoss")
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.25); ax_loss.legend(fontsize=8)

        # BA PLOT
        ax_ba.plot(df_hist["epoch"], df_hist["train_balanced_accuracy"], label="Train BA", linewidth=2)
        ax_ba.plot(df_hist["epoch"], df_hist["val_balanced_accuracy"], label="Validation BA", linewidth=2)
        ax_ba.plot(df_hist["epoch"], df_hist["s_val_ba"], linestyle="--", linewidth=2, label="Smoothed val BA")
        ax_ba.scatter(best_epoch, df_hist.loc[best_idx, "s_val_ba"], s=60, color="red", zorder=5, label=f"Best epoch ({best_epoch})")
        ax_ba.axvline(best_epoch, linestyle=":", alpha=0.7, color="red")
        ax_ba.set_ylim(0, 1)
        ax_ba.set_title(f"{dataset_name} | {train_setup}\nBalanced Accuracy")
        ax_ba.set_xlabel("Epoch"); ax_ba.set_ylabel("BA")
        ax_ba.grid(alpha=0.25); ax_ba.legend(fontsize=8)

    plt.tight_layout()
    save_path = Path(results_dir) / "mlp_diagnostics_grid.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"MLP curves saved to: {save_path}")
    plt.close()

def plot_model_ba_barplots(df_results, section_title, dataset_order=None, model_order=None, load_saved_baseline=False, results_dir=None):
    if load_saved_baseline:
        if results_dir is None: return
        baseline_path = Path(results_dir) / "best_config_generain.csv" 
        if not baseline_path.exists():
            print(f"No se encontró el archivo: {baseline_path}")
            return
        df_results = pd.read_csv(baseline_path)

    if df_results is None or df_results.empty: return

    dataset_order = dataset_order or sorted(df_results["dataset"].unique().tolist())
    model_order = model_order or ["MLP", "LASSO", "RF"]

    fig, axes = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
    fig.suptitle(f"{section_title}\nValidation and test BA by model", fontsize=18, fontweight="bold")

    setup_df = df_results[df_results["train_setup"] == "Real"].copy()
    setup_df["dataset_model"] = setup_df["dataset"] + " | " + setup_df["model"]
    x_order = [f"{ds} | {mdl}" for ds in dataset_order for mdl in model_order if f"{ds} | {mdl}" in set(setup_df["dataset_model"])]

    melted = setup_df.melt(id_vars=["dataset", "model", "dataset_model"], value_vars=["val_ba", "test_ba"], var_name="metric", value_name="ba")
    melted["metric"] = melted["metric"].map({"val_ba": "Val BA", "test_ba": "Test BA"})

    sns.barplot(data=melted, x="dataset_model", y="ba", hue="metric", order=x_order, hue_order=["Val BA", "Test BA"], palette={"Val BA": "#1f77b4", "Test BA": "#ff7f0e"}, ax=axes)

    axes.set_title("Real", fontsize=16)
    axes.set_xlabel("Dataset | Model", fontsize=14); axes.set_ylabel("Balanced Accuracy", fontsize=14)
    axes.set_ylim(0, 1.18); axes.grid(axis="y", alpha=0.25)
    axes.tick_params(axis="x", labelsize=14, rotation=35)
    
    for container in axes.containers:
        axes.bar_label(container, fmt="%.3f", fontsize=12, padding=4, rotation=45)
    axes.legend(title="Metric", fontsize=12, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = Path(results_dir) / "model_ba_barplots.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"BA barplots saved to: {save_path}")
    plt.close()

plot_best_mlp_diagnostics_grid(
    mlp_plot_payload,
    "Best-config metrics - PHASE 1",
    [("HVG1000", "Real"), ("DEG", "Real")],
    results_dir=PHASE1_DIR
)

plot_model_ba_barplots(
    df_results=None,
    section_title="Best-config BA summary - PHASE 1",
    dataset_order=["HVG1000", "DEG"],
    model_order=["MLP", "LASSO", "RF"],
    load_saved_baseline=True,
    results_dir=PHASE1_DIR,
)

ds_deg = datasets_emb["DEG"]
plot_umap_by_class(ds_deg["X_test"], ds_deg["y_test_enc"], ds_deg["le"], "DEG", PHASE1_DIR)
ds_hvg = datasets_emb["HVG1000"]
plot_umap_by_class(ds_hvg["X_test"], ds_hvg["y_test_enc"], ds_hvg["le"], "HVG1000", PHASE1_DIR)