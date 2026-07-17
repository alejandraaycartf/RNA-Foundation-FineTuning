"""
===============================================================================
PHASE 1 SCRIPT (ZERO-SHOT LATENT EXTRACTION & MLP EVALUATION)
===============================================================================
Extracts latent representations (embeddings) from the frozen GeneRAIN 
foundation model and evaluates them using a Multi-Layer Perceptron (MLP) 
classifier on both HVG1000 and DEG datasets.

WORKFLOW:
---------
  1. Data Loading & Binning: Reads pre-processed train/val/test splits and 
                             applies 'Binning-By-Gene' normalization using 
                             the ARCHS4 reference data.
  2. Latent Extraction     : Passes the normalized expression data through the 
                             frozen GeneRAIN model to extract 200D embeddings.
  3. Modeling              : Trains and optimizes an MLP classifier on the 
                             extracted embeddings.
  4. Output                : Saves classification metrics ('best_config_generain.csv') 
                             and generates UMAP scatter plots for latent analysis.

MANUAL CONFIGURATION VARIABLES (Must be updated by the user):
-------------------------------------------------------------
Modify the following variables in the "Global configuration" section:

  * PROBLEM_NAME   : Folder name of the specific problem/dataset being evaluated.
                     (e.g., "colon_sigmoid_vs_colon_transverse", "ppmi").
  * PROJ_PATH      : Absolute path to the root of the GeneRAIN repository.
  * LABEL_COL      : The target column name to predict in your CSVs.
                     (Note: this must be "diagnosis" if PROBLEM_NAME is "ppmi", 
                     or "tissue" if evaluating tissue datasets).
  * SPLITS_DIR     : Path to the directory containing train/val/test CSV splits.
  * GENES_DIR      : Path to the directory containing gene lists (HVG1000 and DEG).
  * PHASE1_DIR     : Path to the directory where Phase 1 results will be saved.
  * PROCESS_DEG    : Set to True if you want to process the DEG dataset as well.
  * CHECKPOINT     : Path to the downloaded GeneRAIN model weights (.pth).
  * PARAM_JSON     : Path to the GeneRAIN model configuration file (.json).
  * CUDA_VISIBLE_DEVICES : GPU ID allocated for execution (e.g., "0" or "1").
===============================================================================
"""

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
# ========================================
def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config.json").exists() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root (expected config.json + src/)")

PROBLEM_NAME = "colon_sigmoid_vs_colon_transverse"  
PROJ_PATH   = find_repo_root()
SPLITS_DIR  = f"{PROJ_PATH}/data/processed/{PROBLEM_NAME}/splits"  
GENES_DIR   = f"{PROJ_PATH}/data/processed/{PROBLEM_NAME}/genes"
# Phase 2 will use the same directory structure as Phase 1 for consistency:
PHASE1_DIR  = f"{PROJ_PATH}/results/{PROBLEM_NAME}/phase1"
RESULTS_CSV = f"{PHASE1_DIR}/best_config_generain.csv"

PARAM_JSON  = f"{PROJ_PATH}/jsons/exp3_BERT_Pred_Genes_Binning_By_Gene.param_config.json"
CHECKPOINT  = f"{PROJ_PATH}/data/models/GeneRAIN.BERT_Pred_Genes_Binning_By_Gene.pth"
LABEL_COL   = "tissue"
PROCESS_DEG = False 

os.environ['PARAM_JSON_FILE'] = PARAM_JSON
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append(f"{PROJ_PATH}/src")
sys.path.append(str(PROJ_PATH))

from utils.config_loader import Config
import anal_utils as au
from data.adata import Adata
from utils.utils import get_device
from utils.checkpoint_utils import load_checkpoint
from data.GetBinsByGeneForNewSamples import get_bins_by_gene_for_new_samples
from train.common import initiate_model

Path(PHASE1_DIR).mkdir(parents=True, exist_ok=True)
pd.set_option("display.max_colwidth", None)
config = Config()

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

# ==========================================
# 2. Loading data splits and preparing binned datasets for HVG1000 and DEG
# ==========================================
samples_subsampled_file = config.proj_path + "/data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_gene2vec_0.005_subsampled.npy"
symbol_file = config.proj_path + "/data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_gene2vec_0.005_subsampled.gene_symbols.txt"

print("--- Loading data splits ---")
df_train_real = au.load_real_split(Path(SPLITS_DIR) / "train.csv", "RealTrain", LABEL_COL)
df_val_real   = au.load_real_split(Path(SPLITS_DIR) / "val.csv", "RealVal", LABEL_COL)
df_test_real  = au.load_real_split(Path(SPLITS_DIR) / "test.csv", "RealTest", LABEL_COL)
df_merged_all = pd.concat([df_train_real, df_val_real, df_test_real], axis=0)

# Loading gene lists for HVG1000 and DEG datasets
with open(Path(GENES_DIR) / "top1000_hvg.txt", "r") as f:
    hvg_1000_genes = [line.strip() for line in f if line.strip()]

print("Binning HVG1000 data...")
df_hvg, adata_hvg, mask_hvg = prepare_binned_adata(df_merged_all, hvg_1000_genes)

if PROCESS_DEG:
    with open(Path(GENES_DIR) / "degs_filtered.txt", "r") as f:
        deg_genes = [line.strip() for line in f if line.strip()]

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

datasets_emb = {}
datasets_emb["HVG1000"] = au.splits(df_hvg, mask_hvg, emb_hvg, label_col=LABEL_COL)
if PROCESS_DEG:
    print("Extracting DEG Embeddings...")
    emb_deg = au.get_generain_embeddings(adata_deg.X, adata_deg.var_names.tolist(), model, device, batch_size=8)
    datasets_emb["DEG"]     = au.splits(df_deg, mask_deg, emb_deg, label_col=LABEL_COL)
    experiment_grid = [
    ("HVG1000", "Real"),
    ("DEG", "Real"),
    ]
else:
    experiment_grid = [
    ("HVG1000", "Real"),
    ]

# ==========================================
# 4.Training MLP classifier on embeddings and saving results
# ==========================================

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
    out_dir = Path(f"{PHASE1_DIR}/{ds_name}/{condition}")
    os.makedirs(out_dir, exist_ok=True)

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


if PROCESS_DEG:
    grid = [("HVG1000", "Real"), ("DEG", "Real")]
    dataset_order = ["HVG1000", "DEG"]
    ds_deg = datasets_emb["DEG"]
    au.plot_umap_by_class(ds_deg["X_test"], ds_deg["y_test_enc"], ds_deg["le"], "DEG", PHASE1_DIR)

else:
    grid = [("HVG1000", "Real")]
    dataset_order = ["HVG1000"]

au.plot_best_mlp_diagnostics_grid(
    mlp_plot_payload,
    "Best-config metrics - PHASE 1",
    grid,
    results_dir=PHASE1_DIR
)

au.plot_model_ba_barplots(
    df_results=None,
    section_title="Best-config BA summary - PHASE 1",
    dataset_order=dataset_order,
    model_order=["MLP", "LASSO", "RF"],
    load_saved=True,
    results_dir=PHASE1_DIR,
    csv="best_config_generain.csv"
)

ds_hvg = datasets_emb["HVG1000"]
au.plot_umap_by_class(ds_hvg["X_test"], ds_hvg["y_test_enc"], ds_hvg["le"], "HVG1000", PHASE1_DIR)