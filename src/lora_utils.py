"""Shared utilities for Phase 2 LoRA training."""
import math
import os
import random
import sys
from pathlib import Path

# --- Third-party library imports ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import umap
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

# ==========================================
# 1. Global configuration and paths
# ==========================================

# Find the repository root 
def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config.json").exists() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root (expected config.json + src/)")
REPO_ROOT = find_repo_root()

# GeneRAIN model configuration
PARAM_JSON = str(REPO_ROOT / "jsons" / "exp3_BERT_Pred_Genes_Binning_By_Gene.param_config.json")
os.environ.setdefault("PARAM_JSON_FILE", PARAM_JSON)

# Module paths
GENERAIN_SRC = str(REPO_ROOT / "src")

# Add src and code directories to sys.path
if GENERAIN_SRC not in sys.path:
    sys.path.insert(0, GENERAIN_SRC)

# Add repository root to sys.path to access anal_utils.py
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ==========================================
# 2. Local module imports
# ==========================================
# (These must be imported AFTER modifying sys.path)
from anal_utils import CustomMLP, set_seed
from data.GN_Dataset import GN_Dataset
from train.common import initiate_model
from train.common_params_funs import extract_hidden_states, get_pred_using_model_and_input
from utils.checkpoint_utils import load_checkpoint
from utils.utils import get_device

# Activation functions dictionary
_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


"""def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
"""

class LabeledGNDataset(Dataset):
    """Wraps GN_Dataset and attaches a class label to each item."""
    def __init__(self, gn_dataset: GN_Dataset, labels: np.ndarray, sources: np.ndarray | None = None):
        self.gn = gn_dataset
        self.labels = labels.astype(np.int64)
        if sources is None:
            self.sources = np.zeros(len(self.labels), dtype=np.int64)
        else:
            self.sources = sources.astype(np.int64)

    def __len__(self):
        return len(self.gn)

    def __getitem__(self, idx):
        item = self.gn[idx]
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["source"] = torch.tensor(self.sources[idx], dtype=torch.long)
        return item


class CustomMLP(nn.Module):
    """MLP used in the analysis notebook (two hidden layers of 64)."""
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.counter = 0

    def step(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def reset(self):
        self.best = -math.inf
        self.counter = 0


def build_lora_model(base_checkpoint: str, lora_rank: int, alpha_mult: int,
                     lora_dropout: float, device: torch.device):
    """Load pre-trained GeneRain and wrap with LoRA adapters."""
    model = initiate_model()
    model, _, _ = load_checkpoint(model, None, base_checkpoint, None)
    model = model.to(device)
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * alpha_mult,
        lora_dropout=lora_dropout,
        target_modules=["key", "query", "value","dense"],
        bias="none",
    )
    return get_peft_model(model, lora_cfg)


def get_embedding(lora_model, batch: dict, device: torch.device) -> torch.Tensor:
    """Forward pass through LoRA GeneRain → pooled sample embedding."""
    gene_indices = batch["gene_indices"].to(device)
    input_expression = batch["true_expression"].to(device)
    zero_expr = batch["zero_expression_genes"].to(device).bool()
    out = get_pred_using_model_and_input(
        lora_model, gene_indices=gene_indices, input_expression=input_expression,
        zero_expression_genes=zero_expr, transformer_model_name="Bert_pred_tokens",
        output_hidden_states=True, output_attentions=False,
    )
    hs = extract_hidden_states(out, layer=-1)
    mask = (~zero_expr).unsqueeze(-1)
    return (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def make_labeled_dataset(df: pd.DataFrame, label_col: str, source: int = 0) -> LabeledGNDataset:
    gene_cols = [c for c in df.columns if c != label_col]
    expr_mat = df[gene_cols].values.astype(np.float32)
    labels = df[label_col].values
    gn = GN_Dataset(sample_by_gene_expr_mat=expr_mat, gene_symbols=gene_cols, num_of_genes=2048)
    sources = np.full(len(labels), source, dtype=np.int64)
    return LabeledGNDataset(gn, labels, sources=sources)




def make_train_loader(dataset: LabeledGNDataset, batch_size: int,
                      extra_dataset: LabeledGNDataset = None) -> DataLoader:
    """Build a training DataLoader with RandomOverSampler for class balance.

    If extra_dataset is provided (augmented condition), it is concatenated
    before oversampling. Synthetic samples are never used as validation.
    """
    if extra_dataset is not None:
        combined = ConcatDataset([dataset, extra_dataset])
        labels = np.concatenate([dataset.labels, extra_dataset.labels])
    else:
        combined = dataset
        labels = dataset.labels

    indices = np.arange(len(combined)).reshape(-1, 1)
    ros = RandomUnderSampler(random_state=42)
    resampled_idx, _ = ros.fit_resample(indices, labels)
    resampled_idx = resampled_idx.flatten()

    resampled_subset = Subset(combined, resampled_idx.tolist())
    return DataLoader(resampled_subset, batch_size=batch_size, shuffle=True)


def make_eval_loader(dataset: LabeledGNDataset, batch_size: int = 16) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_dataframes(dataset, condition: str, data_root: Path, splits_dir: Path, genes_dir: Path, label_col: str):
    """Load train/val/test splits and optional synthetic data from
    the `data/processed` structure.

    If `condition == "augmented"` the function also expects a synthetic
    combined file under `data/processed/synthetic/` (one of a small
    set of well-known names). This function does NOT fall back to legacy
    CSVs; it raises a clear error if required files are missing.
    """

    gs_root = data_root or (REPO_ROOT / "data" / "processed")

    def _load_gene_list(path: Path) -> list[str]:
        return pd.read_csv(path, header=None)[0].astype(str).tolist()

    def _subset_to_genes(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
        return df[[label_col, *genes]].copy()

    if dataset == "HVG1000":
        target_gene_list = _load_gene_list(genes_dir / "top1000_hvg.txt")
    elif dataset == "hvg_2048":
        target_gene_list = _load_gene_list(genes_dir / "top2048_hvg.txt")

    else:
        target_gene_list = _load_gene_list(genes_dir / "degs_filtered.txt")

    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    missing = [p for p in (train_path, val_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required split files for dataset '{dataset}' in {splits_dir}: {missing}")

    df_train = pd.read_csv(train_path, index_col=0)
    df_val = pd.read_csv(val_path, index_col=0)
    df_test = pd.read_csv(test_path, index_col=0)

    real_gene_cols = set(df_train.columns) - {label_col}
    selected_gene_cols = [gene for gene in target_gene_list if gene in real_gene_cols]
    if not selected_gene_cols:
        raise ValueError(f"No target gene columns found for '{dataset}' in the real splits.")

    df_train = _subset_to_genes(df_train, selected_gene_cols)
    df_val = _subset_to_genes(df_val, selected_gene_cols)
    df_test = _subset_to_genes(df_test, selected_gene_cols)

    df_syn = None
    if condition == "augmented":
        syn_dir = gs_root / "synthetic"
        if dataset == "HVG1000":
            syn_path = syn_dir / "combined" / "synthetic_combined.csv"
        else:
            syn_path = syn_dir / "combined_deg" / "synthetic_combined_deg.csv"
        if not syn_path.exists():
            raise FileNotFoundError(f"Missing synthetic file for '{dataset}': {syn_path}")
        df_syn = pd.read_csv(syn_path, index_col=0)

        # Keep only the genes that are shared with the synthetic combined set.
        synthetic_gene_cols = {col for col in df_syn.columns if col != label_col}
        shared_gene_cols = [gene for gene in selected_gene_cols if gene in synthetic_gene_cols]

        if not shared_gene_cols:
            raise ValueError("No shared gene columns found between real and synthetic data.")

        df_train = _subset_to_genes(df_train, shared_gene_cols)
        df_val = _subset_to_genes(df_val, shared_gene_cols)
        df_test = _subset_to_genes(df_test, shared_gene_cols)
        df_syn = _subset_to_genes(df_syn, shared_gene_cols)
    return df_train, df_val, df_test, df_syn


def train_epoch_lora(lora_model, mlp_head: CustomMLP, loader: DataLoader,
                     criterion, optimizer, device: torch.device,
                     debug_hook=None) -> float:
    lora_model.train()
    mlp_head.train()
    total_loss, n = 0.0, 0
    for batch_idx, batch in enumerate(loader):
        if debug_hook is not None and batch_idx == 0:
            debug_hook("train_epoch:first_batch:before_get_embedding")
        y = batch["label"].to(device)
        optimizer.zero_grad()
        emb = get_embedding(lora_model, batch, device)
        if debug_hook is not None and batch_idx == 0:
            debug_hook("train_epoch:first_batch:after_get_embedding")
        logits = mlp_head(emb)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


def eval_epoch_lora(lora_model, mlp_head: CustomMLP, loader: DataLoader,
                    criterion, device: torch.device):
    """Returns (loss, balanced_accuracy, prob_class1, labels_array)."""
    lora_model.eval()
    mlp_head.eval()
    total_loss, n = 0.0, 0
    all_probs, all_labels, all_embeddings = [], [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["label"].to(device)
            emb = get_embedding(lora_model, batch, device)
            all_embeddings.append(emb.cpu().numpy())
            logits = mlp_head(emb)
            total_loss += criterion(logits, y).item() * len(y)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().tolist())
            all_labels.extend(y.cpu().long().tolist())
            n += len(y)
    preds = [int(p > 0.5) for p in all_probs]
    bal_acc = balanced_accuracy_score(all_labels, preds)
    return total_loss / n, bal_acc, np.array(all_probs), np.array(all_labels), np.vstack(all_embeddings)


def compute_full_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs > 0.5).astype(int)
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
        "auc_roc": float(roc_auc_score(labels, probs)),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def plot_umap_embeddings(embeddings, labels, title, save_path="umap_results.png"):
    """Plot single UMAP visualization."""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=labels,
        palette="coolwarm",
        s=50,
        alpha=0.6,
        edgecolor='none'
    )

    plt.title("Phase 2 embeddings - " + title, fontsize=14, pad=15)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(title=title, labels=[' (0)', ' (1)'], loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_umap_embeddings_subplots(embeddings_dict, labels_dict, title, save_path="umap_subplots.png", sources_dict=None):
    """Plot UMAP visualizations for train, val, and test in subplots.
    
    Args:
        embeddings_dict: Dict with keys 'train', 'val', 'test' containing embeddings arrays/tensors
        labels_dict: Dict with keys 'train', 'val', 'test' containing labels arrays/tensors
        sources_dict: Optional dict with keys 'train', 'val', 'test' containing origin arrays/tensors
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_style("whitegrid")
    
    splits = ['train', 'val', 'test']
    plt.suptitle(title, fontsize=16)
    for idx, split in enumerate(splits):
        embeddings = embeddings_dict[split]
        labels = labels_dict[split]
        sources = sources_dict[split] if sources_dict is not None else None
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if sources is not None and isinstance(sources, torch.Tensor):
            sources = sources.detach().cpu().numpy()
        
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42
        )
        embedding_2d = reducer.fit_transform(embeddings)
        
        label_colors = {0: '#1f77b4', 1: '#ff7f0e'}
        source_markers = {0: 'o', 1: '^'}

        if sources is None:
            axes[idx].scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=[label_colors[int(label)] for label in labels],
                s=50,
                alpha=0.6,
                edgecolors='none'
            )
        else:
            for source_value in sorted(np.unique(sources)):
                source_mask = sources == source_value
                for label_value in sorted(np.unique(labels[source_mask])):
                    mask = source_mask & (labels == label_value)
                    axes[idx].scatter(
                        embedding_2d[mask, 0],
                        embedding_2d[mask, 1],
                        c=label_colors[int(label_value)],
                        marker=source_markers.get(int(source_value), 'o'),
                        s=50,
                        alpha=0.6,
                        edgecolors='none'
                    )

        legend_elements = [
            Patch(facecolor=label_colors[0], label='(0)'),
            Patch(facecolor=label_colors[1], label='(1)')
        ]
        if sources is None:
            axes[idx].legend(handles=legend_elements, loc='best', title='Group', fontsize=9, title_fontsize=9)
        else:
            source_elements = [
                Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=8, label='Real'),
                Line2D([0], [0], marker='^', color='black', linestyle='None', markersize=8, label='Synthetic'),
            ]
            legend_group = axes[idx].legend(handles=legend_elements, loc='upper right', title='Group', fontsize=9, title_fontsize=9)
            axes[idx].add_artist(legend_group)
            axes[idx].legend(handles=source_elements, loc='lower right', title='Origin', fontsize=9, title_fontsize=9)
        
        axes[idx].set_title(f"Phase 2 embeddings - {split.capitalize()}", fontsize=14, pad=10)
        axes[idx].set_xlabel("UMAP Dimension 1", fontsize=12)
        axes[idx].set_ylabel("UMAP Dimension 2", fontsize=12)
        axes[idx].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.show()