import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from IPython.display import display
from sklearn.preprocessing import StandardScaler
import umap

# ==========================================
# 1. Global configuration and paths
# ==========================================
PROJ_PATH   = "/home/alejandrayf/GeneRAIN"
TEJIDO = "colon_sigmoid_vs_colon_transverse"
SPLITS_DIR  = f"{PROJ_PATH}/data/generar_sinteticos/splits/tejidos/{TEJIDO}"  
GENES_DIR   = f"{PROJ_PATH}/data/generar_sinteticos/genes/tejidos/{TEJIDO}"
RESULTS_DIR = f"{PROJ_PATH}/results/tejidos/baseline/{TEJIDO}"

PARAM_JSON  = f"{PROJ_PATH}/jsons/exp3_BERT_Pred_Genes_Binning_By_Gene.param_config.json"
CHECKPOINT  = f"{PROJ_PATH}/data/models/GeneRAIN.BERT_Pred_Genes_Binning_By_Gene.pth"
LABEL_COL   = "tissue"

RESULT_SUFFIX = "bb"
RUN_REDUCED_RAW_MLP_GRID = True
MLP_EPOCH_SELECTION_STRATEGY = "score"
MLP_EPOCH_WINDOW = None
SAVE_ALL_MLP_EPOCHS = True

os.environ['PARAM_JSON_FILE'] = PARAM_JSON
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append(f"{PROJ_PATH}/src")
sys.path.append(PROJ_PATH)

from utils.config_loader import Config
import anal_utils as au
from utils.utils import get_device
from utils.checkpoint_utils import load_checkpoint
from train.common import initiate_model

# Out files
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
OPTUNA_TRIALS_CSV = Path(RESULTS_DIR) / f"optuna_mlp_bb_trials_{RESULT_SUFFIX}.csv"
OPTUNA_TRIALS_CSV.unlink(missing_ok=True)
RESULTS_CSV = Path(RESULTS_DIR) / "best_config_baseline_bb.csv"

pd.set_option("display.max_colwidth", None)
config = Config()

# ==========================================
# 2. Auxiliary functions
# ==========================================
def append_optuna_trials_block(df_trials, section_title, dataset_name, train_setup):
    df_to_write = df_trials.copy()
    df_to_write.insert(0, "section", section_title)
    df_to_write.insert(1, "dataset", dataset_name)
    df_to_write.insert(2, "train_setup", train_setup)

    write_header = not OPTUNA_TRIALS_CSV.exists()
    with OPTUNA_TRIALS_CSV.open("a", encoding="utf-8") as handle:
        df_to_write.to_csv(handle, index=False, header=write_header)
    return OPTUNA_TRIALS_CSV

def summarize_test_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()[:2]
    return {
        "test_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }

def _load_real_split(split_path: Path, prefix: str):
    df = pd.read_csv(split_path, index_col=0)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing '{LABEL_COL}' column in {split_path}")
    df = df.copy()
    df.index = [f"{prefix}_{i}" for i in range(len(df))]
    return df

def _scale_stats(X):
    X = np.asarray(X, dtype=np.float64)
    flat = X.ravel()
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "mean": float(np.mean(flat)),
        "var": float(np.var(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p05": float(np.percentile(flat, 5)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
    }

# ==========================================
# 3. Loading data: splits and gene lists
# ==========================================
print("--- Loading data splits ---")
df_train_real = _load_real_split(Path(SPLITS_DIR) / "train.csv", "RealTrain")
df_val_real   = _load_real_split(Path(SPLITS_DIR) / "val.csv", "RealVal")
df_test_real  = _load_real_split(Path(SPLITS_DIR) / "test.csv", "RealTest")

df_merged_all = pd.concat([df_train_real, df_val_real, df_test_real], axis=0)

hvg_file = Path(GENES_DIR) / "top1000_hvg.txt"
deg_file = Path(GENES_DIR) / "degs_filtered.txt"

if not hvg_file.exists() or not deg_file.exists():
    raise FileNotFoundError("No se encuentran los archivos .txt de genes en GENES_DIR. ¡Ejecuta split.py primero!")

with open(hvg_file, "r") as f:
    hvg_1000_genes = [line.strip() for line in f if line.strip()]

with open(deg_file, "r") as f:
    deg_genes = [line.strip() for line in f if line.strip()]

df_hvg_1000 = df_merged_all[[LABEL_COL, *hvg_1000_genes]].copy()
df_deg      = df_merged_all[[LABEL_COL, *deg_genes]].copy()

mat_hvg_1000 = df_hvg_1000[hvg_1000_genes].values.astype(np.float32)
mat_deg      = df_deg[deg_genes].values.astype(np.float32)

bool_mask_hvg = np.ones(len(df_hvg_1000), dtype=bool)
bool_mask_deg = np.ones(len(df_deg), dtype=bool)

print(f"Loaded HVG1000 dataset: {len(hvg_1000_genes)} genes, {len(df_hvg_1000)} samples")
print(f"Loaded DEG dataset: {len(deg_genes)} genes, {len(df_deg)} samples")

# ==========================================
# 4. Load model and prepare datasets
# ==========================================
device = get_device()
model = initiate_model()
model, optimizer, scheduler = load_checkpoint(model, None, CHECKPOINT, None)
model = model.to(device).eval()

datasets_raw = {}
datasets_raw["HVG1000"] = au.splits(df_hvg_1000, bool_mask_hvg, mat_hvg_1000, label_col=LABEL_COL)
datasets_raw["DEG"]     = au.splits(df_deg, bool_mask_deg, mat_deg, label_col=LABEL_COL)

rows = []
for ds_name in sorted(datasets_raw.keys()):
    ds_raw = datasets_raw[ds_name]
    rows.append({"space": "raw", "dataset": ds_name, "train_setup": "Real", **_scale_stats(ds_raw["X_train_real"])})
    
df_scale_diagnostics = pd.DataFrame(rows).sort_values(["space", "dataset", "train_setup"]).reset_index(drop=True)
print("\nScale diagnostics (before scaling):")
display(df_scale_diagnostics)

# ==========================================
# 5. Run baseline experiments: MLP, LASSO, RF
# ==========================================
results_rows = []
mlp_plot_payload = []

experiment_grid = [
    ("HVG1000", "MLP", "Real"),
    ("HVG1000", "LASSO", "Real"),
    ("HVG1000", "RF", "Real"),
    ("DEG", "MLP", "Real"),
    ("DEG", "LASSO", "Real"),
    ("DEG", "RF", "Real"),
]

for ds_name, model_name, train_setup in experiment_grid:
    ds = datasets_raw[ds_name]
    au.set_seed(42)

    X_train_real_raw = ds["X_train_real"]
    y_train_real_enc = ds["y_train_real_enc"]
    X_val_raw = ds["X_val"]
    y_val_enc = ds["y_val_enc"]
    X_test_raw = ds["X_test"]
    y_test_enc = ds["y_test_enc"]

    pd_label = ds["pd_label"]
    control_label = ds["control_label"]
    le = ds["le"]

    idx_train_real_bal = au.balanced_real_indices(
        y_train_real_enc, pd_label, control_label, seed=42,
    )

    X_train = X_train_real_raw[idx_train_real_bal]
    y_train = y_train_real_enc[idx_train_real_bal]

    # Scale data for baseline
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val_raw_scaled = scaler.transform(X_val_raw)
    X_test_raw_scaled = scaler.transform(X_test_raw)

    best_params = None
    preds = None
    
    if model_name == "MLP":
        raw_mlp_param_grid = {
            "dropout": [0.3, 0.4, 0.5],
            "batch_size": [32],
            "max_epochs": [10],
            "patience": [10],
            "min_delta": [0.005],
            "schedule_ratio": [0.9],
        }

        train_ba, val_ba, test_ba, preds, best_params, mlp_details = au.run_mlp_optuna_grid(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test_raw_scaled,
            y_test=y_test_enc,
            n_classes=len(le.classes_),
            device=device,
            seed=42,
            X_val=X_val_raw_scaled,
            y_val=y_val_enc,
            optimize_metric="val_balanced_accuracy",
            return_details=True,
            lr_grid=[5e-4, 1e-4],
            param_grid=raw_mlp_param_grid,
            final_save_all_epochs=True,
            final_checkpoint_dir=Path(RESULTS_DIR) / "raw_mlp_epoch_checkpoints" / f"{ds_name}_{train_setup}",
            final_checkpoint_prefix=f"{ds_name}_{train_setup}",
            final_early_stopping=False,
        )
        
        mlp_plot_payload.append({
            "dataset": ds_name,
            "train_setup": train_setup,
            "best_params": best_params,
            "final_history": mlp_details["final_history"],
            "final_diagnostics": mlp_details.get("final_diagnostics"),
        })
        trials_display = au.prepare_optuna_trials_display(mlp_details["trials_table"], mlp_details)
        print(f"\nMLP baseline - {ds_name} - {train_setup}")
        display(trials_display.head(1))
        append_optuna_trials_block(trials_display, "baseline", ds_name, train_setup)

    elif model_name == "LASSO":
        train_ba, val_ba, test_ba, preds, best_params = au.run_lasso(
            X_train=X_train, y_train=y_train, X_test=X_test_raw_scaled, y_test=y_test_enc, seed=42
        )

    elif model_name == "RF":
        train_ba, val_ba, test_ba, preds, best_params = au.run_rf(
            X_train=X_train, y_train=y_train, X_test=X_test_raw_scaled, y_test=y_test_enc, seed=42
        )

    test_metrics = summarize_test_metrics(y_test_enc, preds)

    results_rows.append({
        "model": model_name,
        "dataset": ds_name,
        "train_setup": train_setup,
        "train_ba": float(train_ba),
        "val_ba": float(val_ba),
        "test_ba": float(test_ba),
        "test_f1": test_metrics["test_f1"],
        "test_sensitivity": test_metrics["test_sensitivity"],
        "test_specificity": test_metrics["test_specificity"],
        "best_params": str(best_params) if best_params is not None else None,
    })

# ==========================================
# 6. Save results
# ==========================================
df_results = pd.DataFrame(results_rows).sort_values(["dataset", "model", "train_setup"]).reset_index(drop=True)

print("\nExperiment results - BASELINES:")
print(f"Total experiments run: {len(df_results)}")
display(df_results)

df_results.to_csv(RESULTS_CSV, index=False)
print(f"Results saved in {RESULTS_CSV}")

# ==========================================
# 7. Plotting and visualization functions
# ==========================================
def plot_best_mlp_diagnostics_grid(payloads, section_title, dataset_order, results_dir=RESULTS_DIR):
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
            ax_loss.axis("off")
            ax_ba.axis("off")
            continue

        history = payload["final_history"]
        df_hist = pd.DataFrame(history)
        
        required = {"epoch", "train_loss", "val_loss", "s_val_loss", "train_balanced_accuracy", "val_balanced_accuracy", "s_val_ba", "score"}
        if not required.issubset(df_hist.columns):
            ax_loss.axis("off")
            ax_ba.axis("off")
            continue

        # Ensure rows are ordered by epoch and use the score argmax
        # as the definitive best-epoch marker (matches select_best_epoch).
        df_hist = df_hist.sort_values("epoch").reset_index(drop=True)
        # index of the row with max score (first occurrence)
        best_idx = int(df_hist["score"].idxmax())
        # epoch number corresponding to the score argmax
        best_epoch = int(df_hist.loc[best_idx, "epoch"])

        # LOSS PLOT
        ax_loss.plot(df_hist["epoch"], df_hist["train_loss"], label="Train loss", linewidth=2)
        ax_loss.plot(df_hist["epoch"], df_hist["val_loss"], label="Validation loss", linewidth=2)
        ax_loss.plot(df_hist["epoch"], df_hist["s_val_loss"], linestyle="--", linewidth=2, label="Smoothed val loss")
        ax_loss.scatter(best_epoch, df_hist.loc[best_idx, "s_val_loss"], s=60, color="red", zorder=5, label=f"Best epoch ({best_epoch})")
        ax_loss.axvline(best_epoch, linestyle=":", alpha=0.7, color="red")
        ax_loss.set_title(f"{dataset_name} | {train_setup}\nLoss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.25)
        ax_loss.legend(fontsize=8)

        # BALANCED ACCURACY PLOT
        ax_ba.plot(df_hist["epoch"], df_hist["train_balanced_accuracy"], label="Train BA", linewidth=2)
        ax_ba.plot(df_hist["epoch"], df_hist["val_balanced_accuracy"], label="Validation BA", linewidth=2)
        ax_ba.plot(df_hist["epoch"], df_hist["s_val_ba"], linestyle="--", linewidth=2, label="Smoothed val BA")
        ax_ba.scatter(best_epoch, df_hist.loc[best_idx, "s_val_ba"], s=60, color="red", zorder=5, label=f"Best epoch ({best_epoch})")
        ax_ba.axvline(best_epoch, linestyle=":", alpha=0.7, color="red")
        ax_ba.set_ylim(0, 1)
        ax_ba.set_title(f"{dataset_name} | {train_setup}\nBalanced Accuracy")
        ax_ba.set_xlabel("Epoch")
        ax_ba.set_ylabel("BA")
        ax_ba.grid(alpha=0.25)
        ax_ba.legend(fontsize=8)

    plt.tight_layout()
    save_path = Path(results_dir) / "mlp_diagnostics_grid.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"MLP curves saved to: {save_path}")
    plt.show()

def _load_saved_baseline_results_for_barplot(results_dir):
    results_dir = Path(results_dir)
    baseline_path = results_dir / "best_config_baseline_bb.csv"
    return pd.read_csv(baseline_path)

def plot_model_ba_barplots(df_results, section_title, dataset_order=None, model_order=None, load_saved_baseline=False, results_dir=None):
    if load_saved_baseline:
        if results_dir is None:
            print("results_dir is required when load_saved_baseline=True")
            return
        df_results = _load_saved_baseline_results_for_barplot(results_dir)

    if df_results is None or df_results.empty:
        print(f"No results available for {section_title}")
        return

    required_cols = {"dataset", "model", "train_setup", "val_ba", "test_ba"}
    if not required_cols.issubset(df_results.columns):
        print("Missing columns to plot BA barplots")
        return

    dataset_order = dataset_order or sorted(df_results["dataset"].unique().tolist())
    model_order = model_order or ["MLP", "LASSO", "RF"]

    fig, axes = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
    fig.suptitle(f"{section_title}\nValidation and test BA by model", fontsize=18, fontweight="bold")

    setup_df = df_results[df_results["train_setup"] == "Real"].copy()
    setup_df["dataset_model"] = setup_df["dataset"] + " | " + setup_df["model"]

    x_order = [f"{ds} | {mdl}" for ds in dataset_order for mdl in model_order if f"{ds} | {mdl}" in set(setup_df["dataset_model"])]

    melted = setup_df.melt(
        id_vars=["dataset", "model", "dataset_model"],
        value_vars=["val_ba", "test_ba"],
        var_name="metric",
        value_name="ba",
    )
    melted["metric"] = melted["metric"].map({"val_ba": "Val BA", "test_ba": "Test BA"})

    sns.barplot(
        data=melted, x="dataset_model", y="ba", hue="metric",
        order=x_order, hue_order=["Val BA", "Test BA"],
        palette={"Val BA": "#1f77b4", "Test BA": "#ff7f0e"}, ax=axes,
    )

    axes.set_title("Real", fontsize=16)
    axes.set_xlabel("Dataset | Model", fontsize=14)
    axes.set_ylabel("Balanced Accuracy", fontsize=14)
    axes.set_ylim(0, 1.18)
    axes.grid(axis="y", alpha=0.25)
    axes.tick_params(axis="x", labelsize=14, rotation=35)
    axes.tick_params(axis="y", labelsize=12)

    for container in axes.containers:
        axes.bar_label(container, fmt="%.3f", fontsize=12, padding=4, rotation=45)

    axes.legend(title="Metric", fontsize=12, title_fontsize=13, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_dir = Path(results_dir) if results_dir else Path(RESULTS_DIR)
    save_path = save_dir / "model_ba_barplots.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"BA barplots saved to: {save_path}")
    plt.show()
def plot_umap_by_class(X_test, y_test, le, dataset_name):
    """
    Genera un UMAP del set de test coloreado por la clase real (tissue).
    """
    # 1. Configurar UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_test)

    # 2. Convertir etiquetas numéricas a texto original
    labels_text = le.inverse_transform(y_test)

    # 3. Crear DataFrame para Seaborn
    df_umap = pd.DataFrame({
        'UMAP 1': embedding[:, 0],
        'UMAP 2': embedding[:, 1],
        'Tissue': labels_text
    })

    # 4. Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_umap, 
        x='UMAP 1', y='UMAP 2', 
        hue='Tissue', 
        palette='viridis', 
        alpha=0.7, 
        s=60
    )
    
    plt.title(f"UMAP Projection - Test Set: {dataset_name}", fontsize=15)
    plt.grid(alpha=0.3)
    
    # Guardar
    save_path = Path(RESULTS_DIR) / f"umap_{dataset_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"UMAP saved to: {save_path}")
    plt.show()

# ==========================================
# 8. Run plotting functions
# ==========================================
plot_best_mlp_diagnostics_grid(
    mlp_plot_payload,
    "Best-config metrics - BASELINES",
    [("HVG1000", "Real"), ("DEG", "Real")]
)

plot_model_ba_barplots(
    df_results=None,
    section_title="Best-config BA summary - BASELINES",
    dataset_order=["HVG1000", "DEG"],
    model_order=["MLP", "LASSO", "RF"],
    load_saved_baseline=True,
    results_dir=RESULTS_DIR,
)

ds_deg = datasets_raw["DEG"]
plot_umap_by_class(ds_deg["X_test"], ds_deg["y_test_enc"], ds_deg["le"], "DEG")
ds_hvg = datasets_raw["HVG1000"]
plot_umap_by_class(ds_hvg["X_test"], ds_hvg["y_test_enc"], ds_hvg["le"], "HVG1000")