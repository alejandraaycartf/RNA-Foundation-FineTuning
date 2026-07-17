#!/usr/bin/env python3
"""
===============================================================================
PHASE 2 SCRIPT (LORA FINE-TUNING)
===============================================================================
Run Phase 2 LoRA pipeline for the requested dataset/condition combinations.
It loads Phase 1 artifacts, builds dataloaders, performs an optional LoRA
grid-search (or single-run), saves best artifacts and plots, and repeats
for the requested combinations.

WORKFLOW:
---------
  1. Data Loading   : Loads the pre-processed splits.
  2. Initialization : Loads the best MLP hyperparameters from Phase 1 artifacts.
  3. Fine-Tuning    : Injects LoRA adapters into GeneRAIN's attention matrices 
                      and trains the entire pipeline (LoRA + MLP) end-to-end.
  4. Output         : Saves training curves/metrics to 'results/phase2/' and 
                      serialized model weights (.pt) to 'artifacts/phase2/'.

USAGE:
------
    python phase2.py [--quick]

OPTIONS:
--------
    --quick        Run a reduced grid for a fast smoke test.

MANUAL CONFIGURATION VARIABLES (Must be updated by the user):
-------------------------------------------------------------
Modify the following variables in the Global variables section:

  * PROBLEM_NAME       : Folder name of the specific problem/dataset being evaluated.
                         (e.g., "colon_sigmoid_vs_colon_transverse", "ppmi").
  * LABEL_COL          : The target column name to predict in your CSVs.
                         (Note: must be "diagnosis" for PPMI, or "tissue" for tissue datasets).

OPTIONAL CONFIGURATION:
-----------------------
  * ENABLE_MEMORY_LOGS : Boolean flag (True/False). Set to True to print detailed CUDA 
                         memory consumption logs during training (useful to avoid OOM).
  * LORA_RANK          : Rank (dimension) of the injected LoRA adapter matrices.
  * ALPHA_MULT         : Scaling multiplier for the LoRA adapters.
===============================================================================
"""



from __future__ import annotations
import argparse
import gc
import json
import resource
import sys
from datetime import datetime
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import uniform_filter1d

# Find repo root (expects config.json + src/)
def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config.json").exists() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root (expected config.json + src/)")


REPO_ROOT = find_repo_root()
PROBLEM_NAME = "ppmi"
LABEL_COL = "diagnosis" if PROBLEM_NAME == "ppmi" else "tissue"
PHASE1_DIR = REPO_ROOT / "results" / PROBLEM_NAME / "phase1"
PHASE2_DIR = REPO_ROOT / "results" / PROBLEM_NAME / "phase2"
SPLITS_DIR = REPO_ROOT / "data" / "processed" / PROBLEM_NAME / "splits"
GENES_DIR = REPO_ROOT / "data" / "processed" / PROBLEM_NAME / "genes"
PROCESS_DEG = True
PROCESS_SYNTHETIC = True

from lora_utils import (
    build_lora_model,
    get_embedding,
    make_labeled_dataset,
    make_eval_loader,
    make_train_loader,
    plot_umap_embeddings_subplots,
    CustomMLP,
    load_dataframes,
    train_epoch_lora,
    eval_epoch_lora,
    compute_full_metrics,
    EarlyStopping,
    set_seed,
)

EMB_DIM = 200
MAX_EPOCHS = 200

# Default hyperparams
LORA_RANK = 8
ALPHA_MULT = 2
LORA_DROPOUT = 0.00
MLP_LR = 1e-4
LORA_LR_MULT = 4.0
ENABLE_MEMORY_LOGS = False



def get_embeddings_from_loader(model, loader, device):
    all_embeddings = []
    all_labels = []
    all_sources = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            emb = get_embedding(model, batch, device)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
            if "source" in batch:
                all_sources.append(batch["source"].cpu().numpy())
            else:
                all_sources.append(np.zeros(len(batch["label"]), dtype=np.int64))
    return np.vstack(all_embeddings), np.concatenate(all_labels), np.concatenate(all_sources)


def state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def add_date_suffix(path: Path, date_tag: str) -> Path:
    """Append a date tag before the file extension."""
    return path.with_name(f"{path.stem}_{date_tag}{path.suffix}")


def moving_average(values, window: int) -> np.ndarray:
    """Compute a trailing moving average with a minimum window of 1."""
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    window = max(1, int(window))
    averaged = np.empty_like(array, dtype=np.float32)
    for idx in range(array.size):
        start = max(0, idx - window + 1)
        averaged[idx] = float(np.mean(array[start : idx + 1]))
    return averaged


def select_best_epoch(train_loss, val_loss, train_ba, val_ba, window=None):
    """Mirror the epoch selection logic used in notebooks/anal/anal_utils.py."""
    train_loss = np.array(train_loss, dtype=float)
    val_loss = np.array(val_loss, dtype=float)
    train_ba = np.array(train_ba, dtype=float)
    val_ba = np.array(val_ba, dtype=float)

    n = len(val_loss)
    if n == 0:
        raise ValueError("Cannot select best epoch from an empty history.")

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


def log_memory_snapshot(stage: str) -> None:
    if not ENABLE_MEMORY_LOGS:
        return
    parts = [f"[Mem] {stage}"]
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            parts.append(
                f"cuda_alloc={allocated:.1f}MiB reserved={reserved:.1f}MiB "
                f"peak_alloc={peak_allocated:.1f}MiB peak_reserved={peak_reserved:.1f}MiB "
                f"free={free_bytes / (1024 ** 2):.1f}MiB total={total_bytes / (1024 ** 2):.1f}MiB"
            )
        except Exception as exc:
            parts.append(f"cuda_snapshot_error={exc}")
    else:
        parts.append("cuda=unavailable")

    try:
        rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        parts.append(f"rss={rss_mib:.1f}MiB")
    except Exception:
        pass

    print(" | ".join(parts))
    sys.stdout.flush()


def select_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return torch.device("cuda:0")

    best_idx = 0
    best_free_bytes = -1
    for idx in range(device_count):
        torch.cuda.set_device(idx)
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes > best_free_bytes:
            best_free_bytes = free_bytes
            best_idx = idx

    torch.cuda.set_device(best_idx)
    print(f"Using cuda:{best_idx} (free={best_free_bytes / (1024 ** 2):.1f} MiB)")
    return torch.device(f"cuda:{best_idx}")


def save_training_curves(history, plots_dir: Path, save_name: str = "training_curves.png", selected_epoch: int | None = None) -> Path:
    """Save loss and balanced-accuracy curves with the best validation BA marked.

    If `selected_epoch` is provided, mark that epoch on both loss and BA plots.
    Otherwise, use the same score-based criterion as `select_best_epoch(...)`.
    """
    epochs = np.arange(1, len(history["train_losses"]) + 1)
    val_ba_avg = history.get("val_ba_avg")
    if val_ba_avg is None:
        avg_window = int(history.get("val_ba_avg_window", max(1, int(MAX_EPOCHS * 0.05))))
        val_ba_avg = moving_average(history["val_bas"], avg_window)
    if selected_epoch is None:
        best_epoch_idx, _ = select_best_epoch(
            history["train_losses"],
            history["val_losses"],
            history["train_bas"],
            history["val_bas"],
        )
        best_epoch = int(epochs[best_epoch_idx]) if len(epochs) > 0 else 1
    else:
        best_epoch_idx = int(selected_epoch)
        best_epoch = int(epochs[best_epoch_idx]) if 0 <= best_epoch_idx < len(epochs) else 1
    best_val_ba = float(history["val_bas"][best_epoch_idx]) if len(history["val_bas"]) > 0 else 0.0

    plots_dir.mkdir(parents=True, exist_ok=True)
    curves_path = plots_dir / save_name

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_losses"], label="Train")
    ax1.plot(epochs, history["val_losses"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss per Epoch")
    # Mark selected epoch on loss plot
    if 1 <= best_epoch <= len(epochs):
        val_loss_at_selected = history["val_losses"][best_epoch_idx]
        ax1.axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
        ax1.scatter([best_epoch], [val_loss_at_selected], color="red", zorder=6)
        ax1.annotate(f"epoch {best_epoch}\nloss={val_loss_at_selected:.4f}", (best_epoch, val_loss_at_selected), textcoords="offset points", xytext=(8, -18), fontsize=8, color="red")
    ax1.legend()

    ax2.plot(epochs, history["train_bas"], label="Train")
    ax2.plot(epochs, history["val_bas"], label="Val")
    if len(val_ba_avg) > 0:
        avg_window = int(history.get("val_ba_avg_window", max(1, int(MAX_EPOCHS * 0.05))))
        ax2.plot(epochs, val_ba_avg, label=f"Val BA prom. ({avg_window} ep.)", linestyle="--", linewidth=2)
    ax2.scatter(best_epoch, best_val_ba, color="red", s=65, zorder=5, label=f"Best Val BA (epoch {best_epoch})")
    ax2.annotate(f"{best_val_ba:.4f}", (best_epoch, best_val_ba), textcoords="offset points", xytext=(8, 6), fontsize=9, color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_title("Balanced Accuracy per Epoch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f" Training curves saved to {curves_path}")
    return curves_path


def run_lora_pipeline(train_loader, val_loader, test_loader, mlp_params, lora_params, checkpoint_path, phase1_dir, save_checkpoint=False):
    device = select_cuda_device()
    set_seed(lora_params.get("seed", 42))

    def debug_memory_hook(stage: str) -> None:
        log_memory_snapshot(stage)

    log_memory_snapshot("before build_lora_model")
    lora_model = build_lora_model(
        base_checkpoint=str(checkpoint_path),
        lora_rank=lora_params["lora_rank"],
        alpha_mult=lora_params["alpha_mult"],
        lora_dropout=lora_params["lora_dropout"],
        device=device,
    )
    log_memory_snapshot("after build_lora_model")

    log_memory_snapshot("before load_mlp_head")
    mlp_head = CustomMLP(input_dim=EMB_DIM, num_classes=2, dropout=mlp_params.get("dropout", 0.2)).to(device)
    ckpt = torch.load(phase1_dir / "best_model.pt", map_location=device)
    mlp_head.load_state_dict(ckpt["model_state"])
    log_memory_snapshot("after load_mlp_head")

    lora_lr = lora_params.get("lora_lr", mlp_params.get("mlp_lr", 1e-4))

    optimizer = torch.optim.AdamW([
        {"params": mlp_head.parameters(), "lr": lora_params["mlp_lr"], "weight_decay": mlp_params.get("weight_decay", 0.0)},
        {"params": [p for n, p in lora_model.named_parameters() if "lora_" in n], "lr": lora_lr},
    ])

    criterion = torch.nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=10, min_delta=0.01)

    train_losses, val_losses, train_bas, val_bas = [], [], [], []
    eval_every = max(1, int(MAX_EPOCHS * 0.05))
    overfit_checks = []
    epoch_states = []

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train_epoch_lora(
            lora_model,
            mlp_head,
            train_loader,
            criterion,
            optimizer,
            device,
            debug_hook=debug_memory_hook if epoch == 1 else None,
        )
        _, tr_ba, _, _, _ = eval_epoch_lora(lora_model, mlp_head, train_loader, criterion, device)
        v_loss, v_ba, _, _, _ = eval_epoch_lora(lora_model, mlp_head, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        train_bas.append(tr_ba)
        val_bas.append(v_ba)
        epoch_states.append((state_dict_to_cpu(lora_model.state_dict()), state_dict_to_cpu(mlp_head.state_dict())))

        # Overfitting gap: positive values mean train BA is above val BA.
        gap = float(tr_ba) - float(v_ba)
        overfit_checks.append({"epoch": epoch, "gap": gap})

        # Check overfitting every 5% of total epochs (and always at the last epoch).
        if (epoch % eval_every == 0) or (epoch == MAX_EPOCHS):
            gap_pct = gap * 100.0
            print(f"[Overfit Eval] Epoch {epoch}/{MAX_EPOCHS} - Gap (train BA - val BA): {gap:.4f} ({gap_pct:.2f}%)")

            window_vals = val_bas[-eval_every:]
            v_ba_avg = float(np.mean(window_vals)) if len(window_vals) > 0 else float(v_ba)
            print(f"[Selection Eval] Epoch {epoch}/{MAX_EPOCHS} - Val BA avg(last {len(window_vals)}): {v_ba_avg:.4f}")

        if early_stop.step(v_ba):
            break

        # Print progress per epoch (match notebook verbosity)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train_loss={tr_loss:.4f} val_loss={v_loss:.4f} | train_ba={tr_ba:.4f} val_ba={v_ba:.4f}")

    best_epoch, diagnostics = select_best_epoch(train_losses, val_losses, train_bas, val_bas)
    best_epoch = int(np.clip(best_epoch, 0, len(epoch_states) - 1))
    best_lora_state, best_mlp_state = epoch_states[best_epoch]

    best_val_ba = float(val_bas[best_epoch])
    best_train_ba = float(train_bas[best_epoch])
    best_val_loss = float(val_losses[best_epoch])
    best_val_ba_avg = float(diagnostics["s_val_ba"][best_epoch])
    selection_score = float(diagnostics["score"][best_epoch])

    # Load best and evaluate on test
    lora_model.load_state_dict(best_lora_state)
    mlp_head.load_state_dict(best_mlp_state)
    _, _, test_probs, test_labels, test_embeddings = eval_epoch_lora(lora_model, mlp_head, test_loader, criterion, device)
    metrics = compute_full_metrics(test_labels.astype(int), test_probs)
    metrics["best_val_ba"] = float(best_val_ba)
    metrics["train_ba"] = float(best_train_ba)
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["selection_score"] = float(selection_score)
    metrics["best_val_epoch"] = int(best_epoch)
    metrics["best_epoch_window"] = int(diagnostics["window"])
    metrics["best_epoch_by_score"] = int(diagnostics["best_epoch_by_score"])
    metrics["best_epoch_by_val_ba"] = int(diagnostics["best_epoch_by_val_ba"])
    print(f" Test metrics: {json.dumps(metrics, indent=2)}")

    # If requested, save training curves (useful when running a single run)
    if save_checkpoint:
        try:
            plots_dir = REPO_ROOT / "results" / "phase2" / phase1_dir.parent.name / phase1_dir.name / "plots"
            save_training_curves(
                {"train_losses": train_losses, "val_losses": val_losses, "train_bas": train_bas, "val_bas": val_bas},
                plots_dir,
                save_name=f"training_curves.png",
                selected_epoch=best_epoch,
            )
        except Exception as e:
            print("Warning: could not save training curves:", e)

    val_ba_avg_window = max(1, int(MAX_EPOCHS * 0.05))
    val_ba_avg = moving_average(val_bas, val_ba_avg_window)
    return metrics, best_lora_state, best_mlp_state, {"train_losses": train_losses, "val_losses": val_losses, "train_bas": train_bas, "val_bas": val_bas, "val_ba_avg": val_ba_avg.tolist(), "val_ba_avg_window": val_ba_avg_window, "overfit_checks": overfit_checks, "epoch_diagnostics": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in diagnostics.items()}}


def run_lora_grid_search(train_loader, val_loader, test_loader, mlp_params, lora_grid, checkpoint_path, phase1_dir, out_base, seed, quick: bool = False):
    grid_items = [(r, a, lr) for r in lora_grid["lora_rank"] for a in lora_grid["alpha_mult"] for lr in lora_grid["lora_lr"]]
    results = []
    best = None
    best_score = -1.0
    for trial_id, (lora_rank, alpha_mult, lora_lr) in enumerate(grid_items):
        print(f"\n[Trial {trial_id+1}/{len(grid_items)}] Testing: lora_rank={lora_rank}, alpha_mult={alpha_mult}, lora_lr={lora_lr}")
        mlp_lr_aux = MLP_LR
        lora_params = {
            "lora_rank": lora_rank,
            "alpha_mult": alpha_mult,
            "lora_dropout": LORA_DROPOUT,
            "mlp_lr": mlp_lr_aux,
            "lora_lr": lora_lr,
            "seed": seed,
        }
        log_memory_snapshot(f"before trial {trial_id+1}/{len(grid_items)}")
        try:
            metrics, lora_state, mlp_state, history = run_lora_pipeline(train_loader, val_loader, test_loader, mlp_params, lora_params, checkpoint_path, phase1_dir, save_checkpoint=False)
        except Exception as exc:
            log_memory_snapshot(f"trial failed {trial_id+1}/{len(grid_items)} -> {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        log_memory_snapshot(f"after trial {trial_id+1}/{len(grid_items)}")
        selection_score = metrics.get("selection_score", -1.0)
        avg_selection_score = metrics.get("avg_selection_score", -1.0)
        val_ba = metrics.get("best_val_ba", -1.0)
        val_loss = metrics.get("best_val_loss", float("inf"))
        results.append({
            "trial": trial_id,
            "lora_rank": lora_rank,
            "alpha_mult": alpha_mult,
            "lora_lr": lora_lr,
            "train_ba": metrics.get("train_ba", -1.0),
            "val_ba": val_ba,
            "best_val_loss": val_loss,
            "selection_score": selection_score,
            "avg_selection_score": avg_selection_score,
            "best_val_epoch": metrics.get("best_val_epoch", -1),
            "test_ba": metrics.get("balanced_accuracy", -1.0),
            "test_f1": metrics.get("f1", -1.0),
            "test_sensitivity": metrics.get("sensitivity", -1.0),
            "test_specificity": metrics.get("specificity", -1.0),
            "metrics": metrics,
        })
        print(f"  -> selection_score={selection_score:.4f} | avg_selection_score={avg_selection_score:.4f}")

        if selection_score > best_score:
            best_score = selection_score
            best = (lora_params, metrics, lora_state, mlp_state, history)
            print(f"  - New best by select_best_epoch score: {selection_score:.4f}\n")

    out_base.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{k: r[k] for k in ["trial", "best_val_epoch", "lora_rank", "alpha_mult", "lora_lr", "train_ba", "val_ba", "best_val_loss", "selection_score", "avg_selection_score", "test_ba", "test_f1", "test_sensitivity", "test_specificity"]} for r in results])
    df = df.sort_values(["selection_score", "val_ba"], ascending=False).reset_index(drop=True)
    if not quick:
        df.to_csv(out_base / f"grid_search_results.csv", index=False)
        with open(out_base / f"best_config.json", "w") as f:
            json.dump(best[0], f, indent=2)
    return best


def run_phase2_combo(dataset: str, condition: str, checkpoint_path: Path, quick: bool = False, seed: int = 42):
    print(f"Running combo: {dataset} | {condition}")
    log_memory_snapshot(f"enter combo {dataset} | {condition}")
    date_tag = datetime.now().strftime("%Y-%m-%d")
    phase1_dir = PHASE1_DIR / dataset / condition
    out_base = PHASE2_DIR / dataset / condition
    with open(phase1_dir / "best_params.json") as f:
        mlp_params = json.load(f)

    df_train, df_val, df_test, df_syn = load_dataframes(
        problem_name=PROBLEM_NAME,
        dataset=dataset,
        condition=condition,
        proj_path=REPO_ROOT / "data" / "processed"
    )
    train_dataset = make_labeled_dataset(df_train, label_col=LABEL_COL, source=0)
    val_dataset = make_labeled_dataset(df_val, label_col=LABEL_COL, source=0)
    test_dataset = make_labeled_dataset(df_test, label_col=LABEL_COL, source=0)
    syn_dataset = make_labeled_dataset(df_syn, label_col=LABEL_COL, source=1) if df_syn is not None else None

    train_loader = make_train_loader(train_dataset, batch_size=mlp_params["batch_size"], extra_dataset=syn_dataset if condition == "augmented" else None)
    val_loader = make_eval_loader(val_dataset, batch_size=16)
    test_loader = make_eval_loader(test_dataset, batch_size=16)
    log_memory_snapshot(f"loaded loaders {dataset} | {condition}")

    emb_dir = phase1_dir
    def _first_existing(cands):
        for p in cands:
            if p.exists():
                return p
        return None

    def load_phase1_embeddings(split):
        emb_candidates = [
            emb_dir / f"emb_{split}_real.npy",
            emb_dir / f"emb_{split}.npy",
            emb_dir / f"emb_{split}_before.npy",
            emb_dir / f"emb_{split}_synthetic.npy",
        ]
        lab_candidates = [
            emb_dir / f"labels_{split}_real.npy",
            emb_dir / f"labels_{split}.npy",
            emb_dir / f"labels_{split}_synthetic.npy",
        ]
        src_candidates = [
            emb_dir / f"sources_{split}_comb.npy",
            emb_dir / f"sources_{split}.npy",
        ]
        emb_path = _first_existing(emb_candidates)
        lab_path = _first_existing(lab_candidates)
        src_path = _first_existing(src_candidates)
        if not emb_path or not lab_path:
            raise FileNotFoundError(
                f"Missing phase1 embeddings/labels for split '{split}' in {emb_dir}. "
                f"Expected one of {emb_candidates} and one of {lab_candidates}."
            )
        emb = np.load(emb_path)
        labels = np.load(lab_path)
        sources = np.load(src_path) if src_path is not None else np.zeros(len(labels), dtype=np.int64)
        return emb, labels, sources

    if condition == "augmented":
        train_emb_before, train_labels_before, train_sources_before = load_phase1_embeddings("train_comb")
    else:
        train_emb_before, train_labels_before, train_sources_before = load_phase1_embeddings("train")
    val_emb_before, val_labels_before, _ = load_phase1_embeddings("val")
    test_emb_before, test_labels_before, _ = load_phase1_embeddings("test")

    # Plot UMAP BEFORE LoRA (match notebook)
    if not quick:
        try:
            plots_dir = out_base / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            umap_before_path = plots_dir / f"umap_before_lora_all_splits.png"
            plot_umap_embeddings_subplots(
                embeddings_dict={"train": train_emb_before, "val": val_emb_before, "test": test_emb_before},
                labels_dict={"train": train_labels_before, "val": val_labels_before, "test": test_labels_before},
                sources_dict={"train": train_sources_before, "val": None, "test": None},
                title=f"Phase 1 - {dataset} - {condition}",
                save_path=umap_before_path,
            )
            print(f" UMAP before LoRA saved to: {umap_before_path}")
        except Exception as e:
            print("Warning: could not plot UMAP before LoRA:", e)

    # Quick grid for testing
    if quick:
        lora_grid = {"lora_rank": [4], "alpha_mult": [4], "lora_lr": [1e-4]}
    else:
        lora_grid = {"lora_rank": [16], "alpha_mult": [1], "lora_lr": [5e-4]}
        #lora_grid = {"lora_rank": [4, 8, 16], "alpha_mult": [1, 2, 4], "lora_lr": [1e-4, 5e-4, 1e-3, 5e-3]}

    log_memory_snapshot(f"before grid search {dataset} | {condition}")
    try:
        best = run_lora_grid_search(train_loader, val_loader, test_loader, mlp_params, lora_grid, checkpoint_path, phase1_dir, out_base, seed, quick=quick)
    except Exception as exc:
        log_memory_snapshot(f"combo failed during grid {dataset} | {condition} -> {exc}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    log_memory_snapshot(f"after grid search {dataset} | {condition}")
    best_lora_params, best_metrics, best_lora_state, best_mlp_state, best_history = best

    # Save best artifacts
    out_base.mkdir(parents=True, exist_ok=True)
    torch.save({"lora_state": best_lora_state, "mlp_state": best_mlp_state, "mlp_params": mlp_params, "lora_params": best_lora_params}, out_base / f"best_model.pt")
    if not quick:
        with open(out_base / f"metrics.json", "w") as f:
            json.dump(best_metrics, f, indent=2)
        with open(out_base / f"best_params.json", "w") as f:
            json.dump(best_lora_params, f, indent=2)

    save_training_curves(best_history, out_base / "plots", save_name=f"training_curves.png", selected_epoch=best_metrics.get("best_val_epoch"))

    # Generate UMAP after best model
    if not quick:
        try:
            # Rebuild models from states and plot
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lora_model = build_lora_model(str(checkpoint_path), best_lora_params["lora_rank"], best_lora_params["alpha_mult"], best_lora_params.get("lora_dropout", 0.0), device)
            mlp_head = CustomMLP(input_dim=EMB_DIM, num_classes=2, dropout=mlp_params.get("dropout", 0.2)).to(device)
            lora_model.load_state_dict(best_lora_state)
            mlp_head.load_state_dict(best_mlp_state)
            train_emb_after, train_labels_after, train_sources_after = get_embeddings_from_loader(lora_model, train_loader, device)
            val_emb_after, val_labels_after, val_sources_after = get_embeddings_from_loader(lora_model, val_loader, device)
            test_emb_after, test_labels_after, test_sources_after = get_embeddings_from_loader(lora_model, test_loader, device)
            plots_dir = out_base / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            umap_path = plots_dir / f"umap_after_lora_all_splits.png"
            plot_umap_embeddings_subplots(
                {"train": train_emb_after, "val": val_emb_after, "test": test_emb_after},
                {"train": train_labels_after, "val": val_labels_after, "test": test_labels_after},
                title=f"Phase 2 - {dataset} - {condition}",
                save_path=umap_path,
                sources_dict={"train": train_sources_after, "val": val_sources_after, "test": test_sources_after},
            )
            print(f" UMAP after LoRA saved to: {umap_path}")
        except Exception as e:
            print("Warning: could not generate UMAP after LoRA:", e)

    if "lora_model" in locals():
        del lora_model
    if "mlp_head" in locals():
        del mlp_head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Finished combo: {dataset} | {condition}. Results saved to {out_base}")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run reduced grid for quick smoke test")
    args = parser.parse_args()

    print(f"--- Starting Phase 2 for {PROBLEM_NAME.upper()} ---")
    
    # 1. Dynamic construction of the Phase 2 Grid
    dataset_order = ["HVG1000"]
    if PROCESS_DEG:
        dataset_order.append("DEG")

    train_setups = ["original"]
    if PROBLEM_NAME == "ppmi" and PROCESS_SYNTHETIC:
        train_setups.append("augmented")

    experiment_grid = []
    for ds in dataset_order:
        for setup in train_setups:
            experiment_grid.append((ds, setup))

    print(f"[INFO] Grid configured with {len(experiment_grid)} combinations:")
    for exp in experiment_grid:
        print(f"  -> {exp}")

    # 2. Checkpoint for the base GeneRAIN model
    checkpoint_path = REPO_ROOT / "data" / "models" / "GeneRAIN.BERT_Pred_Genes_Binning_By_Gene.pth"

    # 3. Run all combos sequentially
    for ds_name, condition in experiment_grid:
        run_phase2_combo(
            dataset=ds_name,
            condition=condition,
            checkpoint_path=checkpoint_path,
            quick=args.quick,
            seed=42
        )

if __name__ == "__main__":
    main()
