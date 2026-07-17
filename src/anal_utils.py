import itertools
import pickle
import random
from datetime import datetime
from pathlib import Path
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
try:
    import umap
except ImportError:
    umap = None
try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
import os
import gc
from pathlib import Path
import importlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
from data.GN_Dataset import GN_Dataset
from train.common_params_funs import extract_hidden_states, get_pred_using_model_and_input, get_gene2idx



# 1. ARCHITECTURE 
class CustomMLP(nn.Module):
    """MLP with two hidden layers of 64 nodes each."""
    def __init__(self, input_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), # 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64), # 64,64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes), # 64
        )

    def forward(self, x):
        return self.net(x)


def select_best_epoch(train_loss, val_loss, train_ba, val_ba, window=None):
    train_loss = np.array(train_loss, dtype=float)
    val_loss = np.array(val_loss, dtype=float)
    train_ba = np.array(train_ba, dtype=float)
    val_ba = np.array(val_ba, dtype=float)

    n = len(val_loss)
    if window is None:
        window = max(3, n // 10)
        if window % 2 == 0:
            window += 1

    s_val_loss = uniform_filter1d(val_loss, size=window)
    s_val_ba = uniform_filter1d(val_ba, size=window)
    s_train_loss = uniform_filter1d(train_loss, size=window)
    s_train_ba = uniform_filter1d(train_ba, size=window)

    gap_penalty = 1 + np.clip(s_val_loss - s_train_loss, 0, None)
    score = s_val_ba / (s_val_loss * gap_penalty)

    best_epoch = int(np.argmax(score))
    diagnostics = {
        "window": int(window),
        "score": score,
        "s_train_loss": s_train_loss,
        "s_val_loss": s_val_loss,
        "s_train_ba": s_train_ba,
        "s_val_ba": s_val_ba,
        "gap_penalty": gap_penalty,
        "best_epoch_by_score": best_epoch,
        "best_epoch_by_val_loss": int(np.argmin(s_val_loss)),
        "best_epoch_by_val_ba": int(np.argmax(s_val_ba)),
        "best_epoch_by_train_loss": int(np.argmin(s_train_loss)),
        "best_epoch_by_train_ba": int(np.argmax(s_train_ba)),
        "best_epoch_by_s_train_loss": int(np.argmin(s_train_loss)),
        "best_epoch_by_s_val_loss": int(np.argmin(s_val_loss)),
        "best_epoch_by_s_train_ba": int(np.argmax(s_train_ba)),
        "best_epoch_by_s_val_ba": int(np.argmax(s_val_ba)),
    }
    return best_epoch, diagnostics


# 2. TRAINING WITH EARLY STOPPING
def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    lr,
    batch_size,
    dropout,
    max_epochs=100,
    patience=5,
    min_delta=0.001,
    schedule_ratio=0.9,
    early_stopping=True,
    best_epoch_window=None,
    save_all_epochs=False,
    checkpoint_dir=None,
    checkpoint_prefix=None,
):
    """Train custom MLP with StepLR scheduler and early stopping.

    Scheduler: StepLR with step_size=1 and gamma=schedule_ratio (as in
    Tutorial_Annotation.ipynb).
    The best epoch is selected after training with `select_best_epoch`, and the
    full diagnostics from that function are returned alongside the epoch
    history.
    """
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomMLP(X_train.shape[1], num_classes, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=schedule_ratio
    )

    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_ba_for_early_stop = -float("inf")
    patience_counter = 0
    best_epoch = -1
    epoch_history = []
    saved_epoch_states = []
    # Evaluate overfitting every 5% of the total epochs (at least once per epoch)
    eval_every = max(1, int(max_epochs * 0.05))

    def _eval_metrics(loader):
        """Compute loss, macro-F1, accuracy, and balanced accuracy on a DataLoader."""
        total_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                out = model(x_b)
                total_loss += criterion(out, y_b).item()
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(y_b.cpu().numpy())
        avg_loss = total_loss / len(loader)
        f1 = f1_score(targets, preds, average="macro")
        acc = float(np.mean(np.array(preds) == np.array(targets)))
        ba = float(balanced_accuracy_score(targets, preds))
        return avg_loss, f1, acc, ba

    for epoch in range(max_epochs):
        # ── Train ──
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()

        # ── Evaluate epoch ──
        model.eval()
        train_loss, train_f1_ep, train_acc_ep, train_ba_ep = _eval_metrics(
            DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        )
        val_loss, val_f1_ep, val_acc_ep, val_ba_ep = _eval_metrics(val_loader)

        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_f1_macro": train_f1_ep,
                "train_accuracy": train_acc_ep,
                "train_balanced_accuracy": train_ba_ep,
                "val_loss": val_loss,
                "val_f1_macro": val_f1_ep,
                "val_accuracy": val_acc_ep,
                "val_balanced_accuracy": val_ba_ep,
            }
        )

        # Compute overfitting gap (train BA - val BA) and store it
        gap = float(train_ba_ep) - float(val_ba_ep)
        epoch_history[-1]["overfit_gap"] = gap

        epoch_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        saved_epoch_states.append(epoch_state)
        if save_all_epochs and checkpoint_path is not None:
            epoch_file = checkpoint_path / f"{checkpoint_prefix or 'mlp'}_epoch_{epoch + 1:03d}.pt"
            torch.save(epoch_state, epoch_file)

        # Print epoch and gap at every 5% of progress (and at the last epoch)
        if ((epoch + 1) % eval_every == 0) or (epoch == max_epochs - 1):
            gap_pct = gap * 100.0
            print(f"[Overfit Eval] Epoch {epoch+1}/{max_epochs} - Gap (train BA - val BA): {gap:.4f} ({gap_pct:.2f}%) - val BA: {val_ba_ep:.4f}")

        # ── Early stopping on validation BA ──
        if val_ba_ep > best_val_ba_for_early_stop + min_delta:
            best_val_ba_for_early_stop = val_ba_ep
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stopping and patience_counter >= patience:
                break

        scheduler.step()

    diagnostics = {}
    if epoch_history:
        train_loss = [row["train_loss"] for row in epoch_history]
        val_loss = [row["val_loss"] for row in epoch_history]
        train_ba = [row["train_balanced_accuracy"] for row in epoch_history]
        val_ba = [row["val_balanced_accuracy"] for row in epoch_history]
        best_epoch, diagnostics = select_best_epoch(
            train_loss,
            val_loss,
            train_ba,
            val_ba,
            window=best_epoch_window,
        )

        for idx, row in enumerate(epoch_history):
            row["s_train_loss"] = float(diagnostics["s_train_loss"][idx])
            row["s_val_loss"] = float(diagnostics["s_val_loss"][idx])
            row["s_train_ba"] = float(diagnostics["s_train_ba"][idx])
            row["s_val_ba"] = float(diagnostics["s_val_ba"][idx])
            row["score"] = float(diagnostics["score"][idx])
            row["gap_penalty"] = float(diagnostics["gap_penalty"][idx])

        best_epoch = int(diagnostics["best_epoch_by_score"])
        best_epoch = int(np.clip(best_epoch, 0, len(epoch_history) - 1))
        if best_epoch < len(saved_epoch_states):
            model.load_state_dict(saved_epoch_states[best_epoch])

    best_row = epoch_history[best_epoch] if best_epoch >= 0 else epoch_history[-1]
    train_f1 = best_row["train_f1_macro"]
    val_f1 = best_row["val_f1_macro"]
    val_ba = best_row["val_balanced_accuracy"]
    train_ba = best_row["train_balanced_accuracy"]

    if best_epoch >= 0:
        for row in epoch_history:
            row["selected_best_epoch"] = False
        epoch_history[best_epoch]["selected_best_epoch"] = True

    return model, train_f1, val_f1, val_ba, train_ba, epoch_history, diagnostics


def set_seed(seed=42):
    np.random.seed(seed)           # Fix Numpy seed (for downsampling/shuffling)
    random.seed(seed)              # Fix Python native random seed
    torch.manual_seed(seed)        # Fix PyTorch seed (CPU weights and dropout)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)       # Fix PyTorch seed (GPU)
        torch.cuda.manual_seed_all(seed)   # Fix PyTorch seed (Multi-GPU)
        # Force deterministic behavior in cuDNN to avoid random optimizations:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_data(ppmi_root):
 # 1) Load real data
    df_train_real = pd.read_csv(os.path.join(ppmi_root,"PPMI_train_matrix.csv"), index_col=0)
    df_test_real = pd.read_csv(os.path.join(ppmi_root, "PPMI_test_matrix.csv"), index_col=0)

    # Track origin in index
    df_train_real.index = [f"RealTrain_{i}" for i in range(len(df_train_real))]
    df_test_real.index = [f"RealTest_{i}" for i in range(len(df_test_real))]
    return df_train_real, df_test_real


def _find_processed_root(ppmi_root):
    """Locate the repository-level processed folder."""
    ppmi_path = Path(ppmi_root).resolve()
    for candidate in [ppmi_path, *ppmi_path.parents]:
        processed_root = candidate / "data" / "processed"
        if processed_root.exists():
            return processed_root
    return None


def _read_split_frame(csv_path, prefix):
    df = pd.read_csv(csv_path)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df = df.copy()
    df.index = [f"{prefix}_{i}" for i in range(len(df))]
    return df


def _align_split_columns(real_frames, synthetic_frame):
    """Keep the synthetic gene order and restrict real frames to shared genes."""
    synthetic_gene_cols = [col for col in synthetic_frame.columns if col not in {"diagnosis", "ID"}]
    real_gene_cols = set(real_frames[0].columns) - {"diagnosis"}
    shared_gene_cols = [gene for gene in synthetic_gene_cols if gene in real_gene_cols]

    if not shared_gene_cols:
        raise ValueError("No shared gene columns found between real and synthetic data.")

    aligned_frames = []
    for frame in real_frames:
        aligned_frames.append(frame[["diagnosis", *shared_gene_cols]].copy())

    synthetic_aligned = synthetic_frame[["diagnosis", *shared_gene_cols]].copy()
    return aligned_frames, synthetic_aligned, shared_gene_cols

def load_real_split(split_path: Path, prefix: str, label_col: str):
    df = pd.read_csv(split_path, index_col=0)
    if label_col not in df.columns:
        raise ValueError(f"Missing '{label_col}' column in {split_path}")
    df = df.copy()
    df.index = [f"{prefix}_{i}" for i in range(len(df))]
    return df

def _load_processed_merged(processed_root, dataset_name):
    print(f"Loading train/val/test splits and synthetic data {processed_root} repository...")
    split_dir = processed_root / "splits"
    synthetic_root = processed_root / "synthetic"
    if not split_dir.exists() or not synthetic_root.exists():
        return None

    synthetic_dir = "combined_deg" if "deg" in dataset_name else "combined"
    synthetic_file = synthetic_root / synthetic_dir / f"synthetic_{synthetic_dir}.csv"

    if not synthetic_file.exists():
        return None

    df_train_real = _read_split_frame(split_dir / "train.csv", "RealTrain")
    df_val_real = _read_split_frame(split_dir / "val.csv", "RealVal")
    df_test_real = _read_split_frame(split_dir / "test.csv", "RealTest")
    df_syn = pd.read_csv(synthetic_file)
    if "ID" in df_syn.columns:
        df_syn = df_syn.drop(columns=["ID"])
    df_syn = df_syn.copy()
    df_syn.index = [f"SynTrain_{i}" for i in range(len(df_syn))]

    aligned_frames, df_syn_aligned, gene_names = _align_split_columns(
        [df_train_real, df_val_real, df_test_real],
        df_syn,
    )

    df_merged = pd.concat([*aligned_frames, df_syn_aligned], axis=0)
    cell_group_names = df_merged.index.values
    sample_by_gene_expr_matrix = df_merged.drop(columns=["diagnosis"]).values
    sample_by_gene_expr_matrix = np.clip(sample_by_gene_expr_matrix, a_min=0, a_max=None)

    return df_merged, cell_group_names, gene_names, sample_by_gene_expr_matrix

def prepare_ppmi_data(ppmi_root, dataset_name):
    """Load the processed real train/val/test split and synthetic rows.

    Falls back to the older PPMI layout if the new folder structure is not
    available in the workspace.
    """
    ppmi_root = Path(ppmi_root)
    merged = _load_processed_merged(ppmi_root, dataset_name)
    if merged is not None:
        return merged

    df_train_real, df_test_real = extract_data(ppmi_root)

    # 2) Load synthetic data
    df_syn_co = pd.read_csv(os.path.join(ppmi_root, "synthetic_controls.csv"), index_col=0)
    df_syn_pd = pd.read_csv(os.path.join(ppmi_root, "synthetic_pd.csv"), index_col=0)
    df_train_syn = pd.concat([df_syn_co, df_syn_pd], axis=0)
    df_train_syn.index = [f"SynTrain_{i}" for i in range(len(df_train_syn))]

    # 3) Merge all
    df_merged = pd.concat([df_train_real, df_test_real, df_train_syn], axis=0)

    # cell group names are the index, gene names are the columns (except diagnosis), and the expression matrix is the values
    cell_group_names = df_merged.index.values
    gene_names = df_merged.drop(columns=["diagnosis"]).columns.tolist()
    sample_by_gene_expr_matrix = df_merged.drop(columns=["diagnosis"]).values
    # Filtering those whose expression is negative (if any) to 0, since the model cannot handle negatives and they are likely after normalization
    sample_by_gene_expr_matrix = np.clip(sample_by_gene_expr_matrix, a_min=0, a_max=None)

    return df_merged, cell_group_names, gene_names, sample_by_gene_expr_matrix

def get_generain_embeddings(data_mat, symbols, model, device, batch_size=8, attention = False):
    """Get GeneRAIN embeddings for all samples in the provided expression matrix. If attention=True, also returns a dataframe with the average attention scores for each gene across all samples."""
    gn_dataset = GN_Dataset(sample_by_gene_expr_mat=data_mat, gene_symbols=symbols, num_of_genes=2048)
    dataloader = DataLoader(gn_dataset, batch_size=batch_size, shuffle=False)
    all_embeddings = []
    if attention:
        _, idx2gene = get_gene2idx()
        gene_attention_sum = {sym: 0.0 for sym in symbols}
        gene_counts = {sym: 0 for sym in symbols}
    
    with torch.no_grad():
        for batch in dataloader:
            gene_indices = batch["gene_indices"].to(device)
            input_expression = batch["true_expression"].to(device)
            zero_expression_genes = batch["zero_expression_genes"].to(device).bool()
            
            out = get_pred_using_model_and_input(
                model, gene_indices=gene_indices, input_expression=input_expression, 
                zero_expression_genes=zero_expression_genes, transformer_model_name="Bert_pred_tokens",
                output_hidden_states=True, output_attentions=attention
            )
            hs = extract_hidden_states(out, layer=-1)
            mask = (~zero_expression_genes).unsqueeze(-1)
            cell_emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            all_embeddings.append(cell_emb.cpu().numpy())

            if attention:
                # Grouping attention: (batch, 6_layers, 4_heads, seq_len, seq_len)
                stacked_attentions = torch.stack(out.attentions, dim=1)
                # Avaraging layers and heads: (batch, seq_len, seq_len)
                avg_layers_heads = stacked_attentions.mean(dim=(1, 2))
                # Averaging attention: (batch, seq_len)
                avg_attention = avg_layers_heads.mean(dim=1)
                
                # Mapping symbols
                for b_idx in range(gene_indices.shape[0]):
                    sample_indices = gene_indices[b_idx].cpu().numpy()
                    sample_scores = avg_attention[b_idx].cpu().numpy()
                    
                    for seq_idx, gene_id in enumerate(sample_indices):
                        gene_sym = idx2gene.get(gene_id, None)
                        if gene_sym in gene_attention_sum:
                            gene_attention_sum[gene_sym] += sample_scores[seq_idx]
                            gene_counts[gene_sym] += 1
            del out; torch.cuda.empty_cache(); gc.collect()
            
    embeddings_out = np.vstack(all_embeddings)
    if attention:
        results = []
        for sym in symbols:
            score = gene_attention_sum[sym] / gene_counts[sym] if gene_counts[sym] > 0 else 0.0
            results.append({"gene": sym, "Attention_Score": score})
            
        df_attention = pd.DataFrame(results).sort_values(by="Attention_Score", ascending=False).reset_index(drop=True)
        return embeddings_out, df_attention     
            
    return embeddings_out


def splits(df_merged, bool_for_if_samples_included_in_returned_mat, matrix, label_col="diagnosis"):
    """Labels + train/validation/test/combined splits.

    If the dataframe already carries RealTrain/RealVal/RealTest prefixes, use
    them directly. Otherwise fall back to the older 50/50 split of the real
    test subset.
    """
    valid_indices = df_merged.index[bool_for_if_samples_included_in_returned_mat]
    valid_labels = df_merged[label_col].values[bool_for_if_samples_included_in_returned_mat]

    le = LabelEncoder()
    y_encoded = le.fit_transform(valid_labels)

    classes_list = list(le.classes_)
    control_label = classes_list.index(0)
    pd_label = classes_list.index(1)

    train_real_mask = valid_indices.str.startswith("RealTrain")
    val_real_mask = valid_indices.str.startswith("RealVal")
    test_real_mask = valid_indices.str.startswith("RealTest")
    train_syn_mask = valid_indices.str.startswith("SynTrain")
    train_comb_mask = train_real_mask | train_syn_mask

    if val_real_mask.any():
        val_real_positions = np.flatnonzero(val_real_mask)
        test_real_positions = np.flatnonzero(test_real_mask)
    else:
        test_real_positions = np.flatnonzero(test_real_mask)
        test_real_labels = y_encoded[test_real_positions]
        split_stratify = (
            test_real_labels
            if len(np.unique(test_real_labels)) > 1 and np.min(np.bincount(test_real_labels)) > 1
            else None
        )

        if len(test_real_positions) > 0:
            val_real_positions, test_real_positions = train_test_split(
                test_real_positions,
                test_size=0.5,
                random_state=42,
                stratify=split_stratify,
            )
        else:
            val_real_positions = np.array([], dtype=int)
            test_real_positions = np.array([], dtype=int)

    out = {
        "X_train_real": matrix[train_real_mask],
        "y_train_real_enc": y_encoded[train_real_mask],
        "X_val": matrix[val_real_positions],
        "y_val_enc": y_encoded[val_real_positions],
        "X_test": matrix[test_real_positions],
        "y_test_enc": y_encoded[test_real_positions],
        "X_train_comb": matrix[train_comb_mask],
        "y_train_comb_enc": y_encoded[train_comb_mask],
        "source_comb": np.where(valid_indices[train_comb_mask].str.startswith("RealTrain"), "Real", "Synthetic"),
        "pd_label": pd_label,
        "control_label": control_label,
        "le": le,
    }
    return out

def balanced_real_indices(y_train_real_enc, pd_label, control_label, seed=42):
    set_seed(seed)
    idx_pd = np.where(y_train_real_enc == pd_label)[0]
    idx_co = np.where(y_train_real_enc == control_label)[0]
    m = min(len(idx_pd), len(idx_co))
    idx = np.concatenate([
        np.random.choice(idx_pd, m, replace=False),
        np.random.choice(idx_co, m, replace=False)
    ])
    np.random.shuffle(idx)
    return idx

def balanced_combined_indices(y_train_comb_enc, source_comb, idx_train_real_bal, pd_label, control_label, seed=42):
    set_seed(seed)
    idx_pd_syn = np.where((y_train_comb_enc == pd_label) & (source_comb == "Synthetic"))[0]
    idx_co_syn = np.where((y_train_comb_enc == control_label) & (source_comb == "Synthetic"))[0]
    m = min(len(idx_pd_syn), len(idx_co_syn))
    idx_pd_syn_down = np.random.choice(idx_pd_syn, m, replace=False)
    idx_co_syn_down = np.random.choice(idx_co_syn, m, replace=False)

    idx = np.concatenate([idx_train_real_bal, idx_pd_syn_down, idx_co_syn_down])
    np.random.shuffle(idx)
    return idx


def run_mlp_optuna_grid(
    X_train,
    y_train,
    X_test,
    y_test,
    n_classes,
    device,
    lr_grid=None,
    seed=42,
    param_grid=None,
    X_val=None,
    y_val=None,
    optimize_metric="val_balanced_accuracy",
    cv_folds=5,
    return_details=False,
    best_epoch_window=None,
    final_save_all_epochs=False,
    final_checkpoint_dir=None,
    final_checkpoint_prefix=None,
    final_early_stopping=True,
):
    """Optuna GridSampler search for MLP with CV inside train only.

    The hyperparameter search uses StratifiedKFold over X_train/y_train.
    Once the best configuration is selected, the final model is trained on the
    full train split and uses X_val/y_val only for early stopping. Test is used
    only once at the end for the final evaluation.

    Returns:
        (train_ba_final, val_ba_final, test_ba, preds, best_params)
        If return_details=True, appends a details dict with the CV table and the
        final training history.
    """
    import optuna

    set_seed(seed)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    if X_val is not None:
        X_val = np.asarray(X_val)
    if y_val is not None:
        y_val = np.asarray(y_val)

    if lr_grid is None:
        lr_grid = [5e-4, 1e-4, 5e-5, 1e-5, 1e-6, 1e-7]

    if param_grid is None:
        param_grid = {
            "lr": list(lr_grid),
            "dropout": [0.3, 0.4, 0.5],
            "batch_size": [8],
            "max_epochs": [10],
            "patience": [20],
            "min_delta": [0.005],
            "schedule_ratio": [0.9],
        }
    else:
        param_grid = {k: list(v) for k, v in param_grid.items()}
        param_grid["lr"] = list(lr_grid)

    param_names = sorted(param_grid.keys())
    search_space = {name: list(param_grid[name]) for name in param_names}
    n_trials = int(np.prod([len(search_space[k]) for k in search_space]))

    metric_lookup = {
        "val_f1_final": "mean_val_f1",
        "val_balanced_accuracy": "mean_val_balanced_accuracy",
        "cv_val_f1": "mean_val_f1",
        "cv_val_balanced_accuracy": "mean_val_balanced_accuracy",
    }
    if optimize_metric not in metric_lookup:
        raise ValueError(
            "optimize_metric must be one of ['val_f1_final', 'val_balanced_accuracy', 'cv_val_f1', 'cv_val_balanced_accuracy']"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_cv_results = {}

    cv_splits = list(
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(
            X_train, y_train
        )
    )


    def _cv_eval_for_params(params, trial=None):
        # Use the same base seed for all trials for reproducibility across trials
        trial_seed = seed
        fold_train_f1_vals = []
        fold_val_f1_vals = []
        fold_train_ba_vals = []
        fold_val_ba_vals = []
        fold_histories = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            set_seed(trial_seed + fold_idx)

            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]

            _, train_f1_fold, val_f1_fold, val_ba_fold, train_ba_fold, history_fold, fold_diagnostics = train_mlp(
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                num_classes=n_classes,
                lr=params["lr"],
                batch_size=params["batch_size"],
                dropout=params["dropout"],
                max_epochs=params["max_epochs"],
                patience=params["patience"],
                min_delta=params["min_delta"],
                schedule_ratio=params["schedule_ratio"],
            )

            fold_train_f1_vals.append(float(train_f1_fold))
            fold_val_f1_vals.append(float(val_f1_fold))
            fold_train_ba_vals.append(float(train_ba_fold))
            fold_val_ba_vals.append(float(val_ba_fold))
            fold_histories.append(history_fold)

        return {
            "params": params.copy(),
            "mean_train_f1": float(np.mean(fold_train_f1_vals)),
            "mean_val_f1": float(np.mean(fold_val_f1_vals)),
            "std_val_f1": float(np.std(fold_val_f1_vals)),
            "mean_train_balanced_accuracy": float(np.mean(fold_train_ba_vals)),
            "mean_val_balanced_accuracy": float(np.mean(fold_val_ba_vals)),
            "std_val_balanced_accuracy": float(np.std(fold_val_ba_vals)),
            "fold_histories": fold_histories,
        }

    def _objective(trial):
        params = {
            name: trial.suggest_categorical(name, search_space[name])
            for name in param_names
        }
        # Inform which trial is starting and with which parameters
        try:
            print(f"[Optuna Trial {int(trial.number)+1}/{n_trials}] Running with params: {params}")
        except Exception:
            print(f"[Optuna Trial] Running with params: {params}")
        cv_result = _cv_eval_for_params(params, trial)
        trial_cv_results[int(trial.number)] = cv_result

        trial.set_user_attr("mean_train_f1", cv_result["mean_train_f1"])
        trial.set_user_attr("mean_val_f1", cv_result["mean_val_f1"])
        trial.set_user_attr("std_val_f1", cv_result["std_val_f1"])
        trial.set_user_attr(
            "mean_train_balanced_accuracy",
            cv_result["mean_train_balanced_accuracy"],
        )
        trial.set_user_attr(
            "mean_val_balanced_accuracy",
            cv_result["mean_val_balanced_accuracy"],
        )
        trial.set_user_attr(
            "std_val_balanced_accuracy",
            cv_result["std_val_balanced_accuracy"],
        )

        return cv_result[metric_lookup[optimize_metric]]

    study.optimize(_objective, n_trials=n_trials)

    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed_trials) == 0:
        raise RuntimeError("No completed Optuna trials.")

    rows = []
    for t in completed_trials:
        cv_result = trial_cv_results.get(int(t.number), {})
        row = {
            "trial": int(t.number),
            "params": dict(t.params),
            **t.params,
            "mean_train_f1": float(t.user_attrs.get("mean_train_f1", np.nan)),
            "mean_val_f1": float(t.user_attrs.get("mean_val_f1", np.nan)),
            "mean_train_balanced_accuracy": float(t.user_attrs.get("mean_train_balanced_accuracy", np.nan)),
            "train_balanced_accuracy": float(t.user_attrs.get("mean_train_balanced_accuracy", np.nan)),
            "val_balanced_accuracy": float(t.user_attrs.get("mean_val_balanced_accuracy", np.nan)),
            "objective_value": float(t.value),
        }
        rows.append(row)

    df_trials = pd.DataFrame(rows).sort_values(
        by="objective_value", ascending=False
    ).reset_index(drop=True)

    best_params = study.best_trial.params.copy()

    set_seed(seed)
    final_history = []

    if X_val is not None and y_val is not None:
        model_final, train_f1_final, val_f1_final, val_ba_final, train_ba_final, final_history, final_diagnostics = train_mlp(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_classes=n_classes,
            lr=best_params["lr"],
            batch_size=best_params["batch_size"],
            dropout=best_params["dropout"],
            max_epochs=best_params["max_epochs"],
            patience=best_params["patience"],
            min_delta=best_params["min_delta"],
            schedule_ratio=best_params["schedule_ratio"],
            early_stopping=final_early_stopping,
            best_epoch_window=best_epoch_window,
            save_all_epochs=final_save_all_epochs,
            checkpoint_dir=final_checkpoint_dir,
            checkpoint_prefix=final_checkpoint_prefix,
        )
    else:
        val_size = max(1, len(X_train) // 5)
        model_final, train_f1_final, val_f1_final, val_ba_final, train_ba_final, final_history, final_diagnostics = train_mlp(
            X_train=X_train,
            y_train=y_train,
            X_val=X_train[:val_size],
            y_val=y_train[:val_size],
            num_classes=n_classes,
            lr=best_params["lr"],
            batch_size=best_params["batch_size"],
            dropout=best_params["dropout"],
            max_epochs=best_params["max_epochs"],
            patience=best_params["patience"],
            min_delta=best_params["min_delta"],
            schedule_ratio=best_params["schedule_ratio"],
            early_stopping=final_early_stopping,
            best_epoch_window=best_epoch_window,
            save_all_epochs=final_save_all_epochs,
            checkpoint_dir=final_checkpoint_dir,
            checkpoint_prefix=final_checkpoint_prefix,
        )

    model_final.eval()
    with torch.no_grad():
        preds_test = torch.argmax(
            model_final(torch.FloatTensor(X_test).to(device)), dim=1
        ).cpu().numpy()

    test_f1 = float(f1_score(y_test, preds_test, average="macro"))
    test_ba = float(balanced_accuracy_score(y_test, preds_test))
    
    # Calculate test metrics: f1, sensitivity, specificity
    test_sensitivity = float(recall_score(y_test, preds_test, average="macro"))
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, preds_test, labels=[0, 1]).ravel()
    test_specificity = float(tn_test / (tn_test + fp_test)) if (tn_test + fp_test) > 0 else 0.0

    # Add test_ba to trials table for notebook compatibility
    df_trials["test_balanced_accuracy"] = test_ba
    df_trials["test_f1"] = test_f1
    df_trials["test_sensitivity"] = test_sensitivity
    df_trials["test_specificity"] = test_specificity

    base_return = (float(train_ba_final), float(val_ba_final), float(test_ba), preds_test, best_params)
    if return_details:
        cv_results_table = df_trials.copy()
        cv_results = [
            {
                "params": {name: row[name] for name in param_names},
                "mean_train_f1": float(row["mean_train_f1"]),
                "mean_val_f1": float(row["mean_val_f1"]),
                "mean_train_balanced_accuracy": float(row["mean_train_balanced_accuracy"]),
                "mean_val_balanced_accuracy": float(row["val_balanced_accuracy"]),
            }
            for _, row in df_trials.iterrows()
        ]
        details = {
            "optimize_metric": optimize_metric,
            "cv_results": cv_results,
            "cv_results_table": cv_results_table,
            "trials_table": df_trials,
            "n_trials": n_trials,
            "search_space": search_space,
            "final_history": final_history,
            "final_diagnostics": final_diagnostics,
            "train_balanced_accuracy": float(train_ba_final),
            "val_balanced_accuracy": float(val_ba_final),
            "test_balanced_accuracy": float(test_ba),
            "train_f1_final": float(train_f1_final),
            "val_f1_final": float(val_f1_final),
            "test_f1": float(test_f1),
            "best_params_metrics": {
                "test_ba": float(test_ba),
                "test_f1": test_f1,
                "test_sensitivity": test_sensitivity,
                "test_specificity": test_specificity,
            },
            "best_model_state_dict": model_final.state_dict(),
        }
        return base_return + (details,)
    return base_return

def _compute_metrics(y_true_local, y_pred_local, pos_label_local, individual_ids_local=None):
    """Compute f1/sensitivity/specificity at sample level or individual level.

    If individual_ids_local is provided, predictions are aggregated by majority vote.
    """
    y_true_local = np.asarray(y_true_local)
    y_pred_local = np.asarray(y_pred_local)

    n_individuals = None
    if individual_ids_local is not None:
        individual_ids_local = np.asarray(individual_ids_local)
        unique_inds = np.unique(individual_ids_local)
        agg_true = []
        agg_pred = []
        for ind in unique_inds:
            mask = individual_ids_local == ind
            agg_true.append(y_true_local[mask][0])
            u, counts = np.unique(y_pred_local[mask], return_counts=True)
            agg_pred.append(u[np.argmax(counts)])
        y_true_local = np.asarray(agg_true)
        y_pred_local = np.asarray(agg_pred)
        n_individuals = int(len(unique_inds))

    out = {
        "f1_macro": float(f1_score(y_true_local, y_pred_local, average="macro", zero_division=0)),
        "sensitivity": None,
        "specificity": None,
    }
    if n_individuals is not None:
        out["n_individuals"] = n_individuals

    labels = np.unique(y_true_local)
    if len(labels) == 2:
        neg_label_local = [x for x in labels if x != pos_label_local][0]
        out["sensitivity"] = float(
            recall_score(y_true_local, y_pred_local, pos_label=pos_label_local, zero_division=0)
        )
        out["specificity"] = float(
            recall_score(y_true_local, y_pred_local, pos_label=neg_label_local, zero_division=0)
        )
    return out


def compute_full_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    specificity = float(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

def save_phase1_artifacts(
    phase1_root,
    dataset,
    train_setup,
    best_params,
    model=None,
    state_dict=None,
    checkpoint_src=None,
    test_preds=None,
    test_labels=None,
):
    """Save Phase1 artifacts in a reproducible layout.

    Parameters
    ----------
    phase1_root : str or Path
        Root folder where to create the phase1 outputs (e.g. results/phase1).
    dataset : str
        Dataset name (e.g. 'DEG' or 'HVG1000').
    train_setup : str
        'Real' or 'Real+Synthetic' (used to derive condition 'original'/'augmented').
    best_params : dict
        Best hyperparameters to save as best_params.json.
    model : torch.nn.Module, optional
        Trained model object; if provided its `state_dict()` will be saved.
    state_dict : dict, optional
        State dict for the model. If both `model` and `state_dict` provided, `state_dict` takes precedence.
    checkpoint_src : str or Path, optional
        Optional path or identifier of the base checkpoint used (saved into checkpoint.json).
    test_preds, test_labels : array-like, optional
        If provided, compute and save basic metrics (f1, balanced_accuracy, precision, recall) in metrics.json.

    Returns
    -------
    dict
        Dictionary with saved file paths.
    """
    import json
    import time

    phase1_root = Path(phase1_root)
    condition = "original" if train_setup == "Real" else "augmented"
    out_dir = phase1_root / dataset / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save best params
    params_path = out_dir / "best_params.json"
    with params_path.open("w") as fh:
        json.dump(best_params, fh, indent=2)

    # Determine state dict
    sd = None
    if state_dict is not None:
        sd = state_dict
    elif model is not None:
        try:
            sd = model.state_dict()
        except Exception:
            sd = None

    model_path = None
    if sd is not None:
        try:
            import torch

            model_path = out_dir / "best_model.pt"
            torch.save({"model_state": sd, "params": best_params}, model_path)
        except Exception:
            model_path = None

    # Save checkpoint metadata
    checkpoint_path = None
    if checkpoint_src is not None:
        checkpoint_path = out_dir / "checkpoint.json"
        with checkpoint_path.open("w") as fh:
            json.dump({"checkpoint": str(checkpoint_src), "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=2)

    # Compute and save simple metrics if predictions provided
    metrics_path = None
    if test_preds is not None and test_labels is not None:
        try:
            import numpy as _np
            from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score

            test_preds = _np.asarray(test_preds)
            test_labels = _np.asarray(test_labels)
            
            metrics = compute_full_metrics(test_labels, test_preds)
            metrics_path = out_dir / "metrics.json"
            with metrics_path.open("w") as fh:
                json.dump(metrics, fh, indent=2)
        except Exception:
            metrics_path = None

    saved = {
        "params": str(params_path),
        "model": str(model_path) if model_path is not None else None,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "metrics": str(metrics_path) if metrics_path is not None else None,
        "out_dir": str(out_dir),
    }
    return saved

def prepare_optuna_trials_display(df_trials: pd.DataFrame, mlp_details: dict) -> pd.DataFrame:
    """Return a cleaned view of Optuna trials for display.

    Removes verbose items like `fold_histories` and `std_*` columns and ensures the
    final test metrics (BA, f1, sensitivity, specificity) are present so notebooks
    can display a concise table of trial params + requested metrics.
    """
    df_display = df_trials.copy()
    # Ensure test metrics from mlp_details are present
    metric_defaults = {
        "test_balanced_accuracy": mlp_details.get("test_balanced_accuracy"),
        "test_f1": mlp_details.get("test_f1"),
        "test_sensitivity": mlp_details.get("best_params_metrics", {}).get("test_sensitivity"),
        "test_specificity": mlp_details.get("best_params_metrics", {}).get("test_specificity"),
    }
    for col, val in metric_defaults.items():
        if col not in df_display.columns:
            df_display[col] = val

    # Drop any fold histories or std columns if present
    drop_cols = [c for c in df_display.columns if c.startswith("fold_histories") or c.startswith("std_")]
    if drop_cols:
        df_display = df_display.drop(columns=drop_cols)

    # Select ordered columns for display when available
    preferred = [
        "trial",
        "params",
        "batch_size",
        "dropout",
        "lr",
        "max_epochs",
        "min_delta",
        "patience",
        "schedule_ratio",
        "train_balanced_accuracy",
        "val_balanced_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_specificity",
        "test_sensitivity",
    ]
    cols = [c for c in preferred if c in df_display.columns]
    # Append any remaining columns (but avoid huge objects like history)
    remaining = [c for c in df_display.columns if c not in cols and c != "params"]
    cols = cols + remaining
    return df_display[cols]

def run_lasso(
    X_train,
    y_train,
    X_test,
    y_test,
    seed=42,
    cv_splits=None,
    handle_imbalance=False,
    individual_ids_test=None,
    pos_label=None,
    n_jobs=1,
    return_details=False,
):
    """Run LASSO.

    Returns:
    (train_cv_ba, val_cv_ba, test_ba, preds, best_params)

    If return_details=True, returns a 6th element with extra metrics.
    """


    if cv_splits is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        cv = cv_splits

    param_values = {"C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}

    if handle_imbalance:
        try:
            imblearn_pipeline = importlib.import_module("imblearn.pipeline")
            imblearn_sampling = importlib.import_module("imblearn.under_sampling")
            ImbPipeline = getattr(imblearn_pipeline, "Pipeline")
            RandomUnderSampler = getattr(imblearn_sampling, "RandomUnderSampler")

            model = ImbPipeline([
                ("sampler", RandomUnderSampler(random_state=seed)),
                ("clf", LogisticRegression(penalty="l1", solver="liblinear", random_state=seed, max_iter=1000)),
            ])
            param_grid = {f"clf__{k}": v for k, v in param_values.items()}
        except ImportError:
            model = Pipeline([
                ("clf", LogisticRegression(
                    penalty="l1", solver="liblinear", random_state=seed, max_iter=1000
                )),
            ])
            param_grid = {f"clf__{k}": v for k, v in param_values.items()}
    else:
        model = Pipeline([
            ("clf", LogisticRegression(
                penalty="l1", solver="liblinear", random_state=seed, max_iter=1000
            )),
        ])
        param_grid = {f"clf__{k}": v for k, v in param_values.items()}

    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=n_jobs,
        return_train_score=True,
    )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        grid.fit(X_train, y_train)

    best = grid.best_estimator_
    preds = best.predict(X_test)

    best_idx = grid.best_index_
    train_cv = float(grid.cv_results_["mean_train_score"][best_idx])
    val_cv = float(grid.cv_results_["mean_test_score"][best_idx])
    test_ba = float(balanced_accuracy_score(y_test, preds))

    if pos_label is None:
        pos_label = np.unique(y_train)[-1]
    sample_metrics = _compute_metrics(y_test, preds, pos_label)

    details = {
        "test_sensitivity": sample_metrics["sensitivity"],
        "test_specificity": sample_metrics["specificity"],
        "test_individual_level": None,
    }
    if individual_ids_test is not None:
        details["test_individual_level"] = _compute_metrics(
            y_test, preds, pos_label, individual_ids_local=individual_ids_test
        )

    best_params = {
        k.replace("clf__", ""): v for k, v in grid.best_params_.items()
    }

    base_return = (train_cv, val_cv, test_ba, preds, best_params)
    if return_details:
        return base_return + (details,)
    return base_return


def get_lasso_nonzero_genes(
    X_train,
    y_train,
    C=0.01,
    gene_names=None,
    seed=42,
    handle_imbalance=False,
):
    """
    Train LASSO (L1-penalized LogisticRegression) and return non-zero coefficient genes.

    Parameters:
    -----------
    X_train : array-like, shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like, shape (n_samples,)
        Training target labels.
    C : float, default=0.01
        Inverse of regularization strength (lower C = more regularization).
    gene_names : list of str, optional
        Names of genes/features. If None, features are numbered (0, 1, 2, ...).
    seed : int, default=42
        Random seed for reproducibility.
    handle_imbalance : bool, default=False
        If True, use RandomUnderSampler in pipeline.

    Returns:
    --------
    df_nonzero : pandas.DataFrame
        DataFrame with columns: gene_name, coef, abs_coef
        Sorted by absolute coefficient value (descending).
        Only includes genes with non-zero coefficients.
    model : sklearn estimator
        The fitted Pipeline model.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    n_features = X_train.shape[1]
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_features)]
    else:
        gene_names = list(gene_names)
        if len(gene_names) != n_features:
            raise ValueError(
                f"Length of gene_names ({len(gene_names)}) does not match "
                f"number of features ({n_features})."
            )

    if handle_imbalance:
        try:
            imblearn_pipeline = importlib.import_module("imblearn.pipeline")
            imblearn_sampling = importlib.import_module("imblearn.under_sampling")
            ImbPipeline = getattr(imblearn_pipeline, "Pipeline")
            RandomUnderSampler = getattr(imblearn_sampling, "RandomUnderSampler")

            model = ImbPipeline([
                ("sampler", RandomUnderSampler(random_state=seed)),
                ("clf", LogisticRegression(
                    penalty="l1", solver="liblinear", C=C, random_state=seed, max_iter=1000
                )),
            ])
        except ImportError:
            model = Pipeline([
                ("clf", LogisticRegression(
                    penalty="l1", solver="liblinear", C=C, random_state=seed, max_iter=1000
                )),
            ])
    else:
        model = Pipeline([
            ("clf", LogisticRegression(
                penalty="l1", solver="liblinear", C=C, random_state=seed, max_iter=1000
            )),
        ])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(X_train, y_train)

    coef = model.named_steps["clf"].coef_
    if coef.ndim > 1:
        coef = coef[0]

    df_coef = pd.DataFrame({
        "gene_name": gene_names,
        "coef": coef,
    })
    df_coef["abs_coef"] = np.abs(df_coef["coef"])
    df_nonzero = df_coef[df_coef["coef"] != 0].sort_values(
        by="abs_coef", ascending=False
    ).reset_index(drop=True)

    return df_nonzero, model


def run_rf(
    X_train,
    y_train,
    X_test,
    y_test,
    seed=42,
    cv_splits=None,
    handle_imbalance=False,
    individual_ids_test=None,
    pos_label=None,
    n_jobs=1,
    rf_jobs=2,
    return_details=False,
):
    """Run RandomForest baseline.

    Backward compatible return by default:

    If return_details=True, returns a 6th element with extra metrics.
    """

    if cv_splits is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        cv = cv_splits

    n_features = X_train.shape[1]
    f_sqrt = max(1, int(np.sqrt(n_features)))
    f_half = max(1, int(f_sqrt / 2))
    f_quarter = max(1, int(f_sqrt / 4))

    param_values = {
        "n_estimators": [100, 250, 500],
        "max_features": [f_sqrt, f_half, f_quarter],
        "min_samples_leaf": [50, 100, 200, 300, 400, 500],
    }

    if handle_imbalance:
        try:
            imblearn_pipeline = importlib.import_module("imblearn.pipeline")
            imblearn_sampling = importlib.import_module("imblearn.under_sampling")
            ImbPipeline = getattr(imblearn_pipeline, "Pipeline")
            RandomUnderSampler = getattr(imblearn_sampling, "RandomUnderSampler")

            model = ImbPipeline([
                ("sampler", RandomUnderSampler(random_state=seed)),
                ("clf", RandomForestClassifier(random_state=seed, n_jobs=rf_jobs)),
            ])
            param_grid = {f"clf__{k}": v for k, v in param_values.items()}
        except ImportError:
            model = RandomForestClassifier(random_state=seed, n_jobs=rf_jobs)
            param_grid = param_values
    else:
        model = RandomForestClassifier(random_state=seed, n_jobs=rf_jobs)
        param_grid = param_values

    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=n_jobs,
        return_train_score=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        grid.fit(X_train, y_train)

    best = grid.best_estimator_
    preds = best.predict(X_test)

    best_idx = grid.best_index_
    train_cv = float(grid.cv_results_["mean_train_score"][best_idx])
    val_cv = float(grid.cv_results_["mean_test_score"][best_idx])
    test_ba = float(balanced_accuracy_score(y_test, preds))

    if pos_label is None:
        pos_label = np.unique(y_train)[-1]
    sample_metrics = _compute_metrics(y_test, preds, pos_label)

    details = {
        "test_sensitivity": sample_metrics["sensitivity"],
        "test_specificity": sample_metrics["specificity"],
        "test_individual_level": None,
    }
    if individual_ids_test is not None:
        details["test_individual_level"] = _compute_metrics(
            y_test, preds, pos_label, individual_ids_local=individual_ids_test
        )

    best_params = {
        k.replace("clf__", ""): v for k, v in grid.best_params_.items()
    }

    base_return = (train_cv, val_cv, test_ba, preds, best_params)
    if return_details:
        return base_return + (details,)
    return base_return


def get_rf_variable_importance(
    X_train,
    y_train,
    n_estimators=100,
    max_features="sqrt",
    min_samples_leaf=1,
    gene_names=None,
    seed=42,
    handle_imbalance=False,
    n_jobs=1,
    rf_jobs=2,
):
    """
    Train Random Forest and return variable importance scores.

    Parameters:
    -----------
    X_train : array-like, shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like, shape (n_samples,)
        Training target labels.
    n_estimators : int, default=100
        Number of trees in the forest.
    max_features : int, float, or {"sqrt", "log2"}, default="sqrt"
        Number of features to consider at each split.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    gene_names : list of str, optional
        Names of genes/features. If None, features are numbered (0, 1, 2, ...).
    seed : int, default=42
        Random seed for reproducibility.
    handle_imbalance : bool, default=False
        If True, use RandomUnderSampler in pipeline.
    n_jobs : int, default=1
        Number of jobs for GridSearchCV parallelization.
    rf_jobs : int, default=2
        Number of jobs for RandomForestClassifier parallelization.

    Returns:
    --------
    df_importance : pandas.DataFrame
        DataFrame with columns: gene_name, importance
        Sorted by importance value (descending).
    model : sklearn estimator (RandomForestClassifier or Pipeline)
        The fitted model.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    n_features = X_train.shape[1]
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_features)]
    else:
        gene_names = list(gene_names)
        if len(gene_names) != n_features:
            raise ValueError(
                f"Length of gene_names ({len(gene_names)}) does not match "
                f"number of features ({n_features})."
            )

    if handle_imbalance:
        try:
            imblearn_pipeline = importlib.import_module("imblearn.pipeline")
            imblearn_sampling = importlib.import_module("imblearn.under_sampling")
            ImbPipeline = getattr(imblearn_pipeline, "Pipeline")
            RandomUnderSampler = getattr(imblearn_sampling, "RandomUnderSampler")

            model = ImbPipeline([
                ("sampler", RandomUnderSampler(random_state=seed)),
                ("clf", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed,
                    n_jobs=rf_jobs,
                )),
            ])
        except ImportError:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                random_state=seed,
                n_jobs=rf_jobs,
            )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=rf_jobs,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(X_train, y_train)

    if isinstance(model, Pipeline):
        rf_model = model.named_steps["clf"]
    else:
        rf_model = model

    importances = rf_model.feature_importances_

    df_importance = pd.DataFrame({
        "gene_name": gene_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_importance, model


def plot_best_mlp_diagnostics_grid(payloads, section_title, dataset_order, results_dir):
    """Plot a grid of MLP training diagnostics (loss and balanced accuracy) for each dataset and training setup."""
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

def plot_model_ba_barplots(df_results, section_title, dataset_order=None, model_order=None, load_saved=False, results_dir=None, csv = None):
    """Generates barplots of validation and test balanced accuracy (BA) for each model and dataset."""
    if load_saved:
        if results_dir is None: return
        path = Path(results_dir) / csv
        if not path.exists():
            print(f"No se encontró el archivo: {path}")
            return
        df_results = pd.read_csv(path)

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

def plot_umap_by_class(X_test, y_test, le, dataset_name, out):
    """
    Generates a UMAP projection of the test set colored by the true class.
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
    save_path = Path(out) / f"umap_{dataset_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"UMAP saved to: {save_path}")
    plt.show()
