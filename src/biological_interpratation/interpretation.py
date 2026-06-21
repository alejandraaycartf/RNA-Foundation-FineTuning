import os
import ast
import sys
from pathlib import Path

PROJ_PATH = "/home/alejandrayf/GeneRAIN"
CHECKPOINT  = f"{PROJ_PATH}/data/models/GeneRAIN.BERT_Pred_Genes_Binning_By_Gene.pth"
PARAM_JSON  = f"{PROJ_PATH}/jsons/exp3_BERT_Pred_Genes_Binning_By_Gene.param_config.json"

os.environ["PARAM_JSON_FILE"] = PARAM_JSON

sys.path.append(f"{PROJ_PATH}/src")
sys.path.append(PROJ_PATH)
from utils.config_loader import Config
import anal_utils as au
from data.adata import Adata
from utils.utils import get_device
from utils.checkpoint_utils import load_checkpoint
from data.GetBinsByGeneForNewSamples import get_bins_by_gene_for_new_samples
from train.common import initiate_model
import torch
from utils.config_loader import Config
# -----------------------------
import numpy as np
import pandas as pd
import anal_utils as au

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


# ---> CHANGE BASED ON THE PROBLEM <---
PROBLEM_NAME = "ppmi" # Options: "ppmi", "colon_sigmoid_vs_colon_transverse", "frontal_cortex_vs_blood"
LABEL_COL = "diagnosis" # Change to "tissue" if switching to tissues
ETAPA = "phase2" # Options: "baseline", "phase1", "phase2"


if PROBLEM_NAME == "ppmi":
    OUTPUT_DIR = Path(PROJ_PATH) / "results" / "ppmi" / ETAPA
    SPLITS_DIR = Path(PROJ_PATH) / "data" / "generar_sinteticos" / "splits"  
    GENES_DIR  = Path(PROJ_PATH) / "data" / "generar_sinteticos" / "genes" 
    
    experiment_grid = [
        ("HVG1000", "LASSO", "Real"), ("HVG1000", "LASSO", "Real+Synthetic"),
        ("HVG1000", "RF", "Real"), ("HVG1000", "RF", "Real+Synthetic"),
        ("DEG", "LASSO", "Real"), ("DEG", "LASSO", "Real+Synthetic"),
        ("DEG", "RF", "Real"), ("DEG", "RF", "Real+Synthetic")
    ]
    filepath = f"{PROJ_PATH}/notebooks/anal/best_config_Baselines.csv"
else:
    OUTPUT_DIR = Path(PROJ_PATH) / "results" / "tejidos" / ETAPA / PROBLEM_NAME
    SPLITS_DIR = Path(PROJ_PATH) / "data" / "generar_sinteticos" / "splits" / "tejidos" / PROBLEM_NAME
    GENES_DIR  = Path(PROJ_PATH) / "data" / "generar_sinteticos" / "genes" / "tejidos" / PROBLEM_NAME
    
    experiment_grid = [
        ("HVG1000", "LASSO", "Real"), ("HVG1000", "RF", "Real"),
        ("DEG", "LASSO", "Real"), ("DEG", "RF", "Real")
    ]
    filepath = f"{PROJ_PATH}/results/tejidos/{ETAPA}/{PROBLEM_NAME}/best_config_baseline_bb.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Directorio de salida configurado en: {OUTPUT_DIR}")
config = Config()

def _load_real_split(split_path: Path, prefix: str):
    df = pd.read_csv(split_path, index_col=0)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Falta la columna '{LABEL_COL}' en {split_path}")
    df = df.copy()
    df.index = [f"{prefix}_{i}" for i in range(len(df))]
    return df

import ast
def get_params_from_csv(filepath, model_name, ds_name, train_setup):
    
    df = pd.read_csv(filepath)
    
    query = (
        (df['model'] == model_name) & 
        (df['dataset'] == ds_name) & 
        (df['train_setup'] == train_setup)
    )
    
    result = df[query]
    
    if result.empty:
        print("No params for this combination.")
        return None

    params = result.iloc[0]['best_params']
    
    try:
        return ast.literal_eval(params)
    except (ValueError, SyntaxError):
        print("Error parsing params.")
        return None
    

hvg_file = Path(GENES_DIR) / "top1000_hvg.txt"
deg_file = Path(GENES_DIR) / "degs_filtered.txt"

if not hvg_file.exists() or not deg_file.exists():
    raise FileNotFoundError(f"¡Archivos no encontrados en {GENES_DIR}!")

with open(hvg_file, "r") as f:
    gene_names_hvg1000 = [line.strip() for line in f if line.strip()]

with open(deg_file, "r") as f:
    gene_names_deg = [line.strip() for line in f if line.strip()]

if PROBLEM_NAME == "ppmi":
    print("Cargando datos de PPMI (Reales y Sintéticos) usando prepare_ppmi_data...")
    df_hvg_1000, _, _, mat_hvg_1000 = au.prepare_ppmi_data(
        os.path.join(config.proj_path, "data/generar_sinteticos"), "hvg"
    )
    bool_mask_hvg = np.ones(len(df_hvg_1000), dtype=bool)
    
    df_deg, _, _, mat_deg = au.prepare_ppmi_data(
        os.path.join(config.proj_path, "data/generar_sinteticos"), "deg"
    )
    bool_mask_deg = np.ones(len(df_deg), dtype=bool)
else:
    print(f"Cargando datos de tejidos ({PROBLEM_NAME}) de forma manual...")
    df_train_real = _load_real_split(Path(SPLITS_DIR) / "train.csv", "RealTrain")
    df_val_real   = _load_real_split(Path(SPLITS_DIR) / "val.csv", "RealVal")
    df_test_real  = _load_real_split(Path(SPLITS_DIR) / "test.csv", "RealTest")
    
    df_merged_all = pd.concat([df_train_real, df_val_real, df_test_real], axis=0)
    
    df_hvg_1000 = df_merged_all[[LABEL_COL, *gene_names_hvg1000]].copy()
    df_deg      = df_merged_all[[LABEL_COL, *gene_names_deg]].copy()

    mat_hvg_1000 = df_hvg_1000[gene_names_hvg1000].values.astype(np.float32)
    mat_deg      = df_deg[gene_names_deg].values.astype(np.float32)

    bool_mask_hvg = np.ones(len(df_hvg_1000), dtype=bool)
    bool_mask_deg = np.ones(len(df_deg), dtype=bool)

mean_expr = df_hvg_1000[gene_names_hvg1000].mean()
variance_expr = df_hvg_1000[gene_names_hvg1000].var()

df_hvg_stats = pd.DataFrame({
    "Gene": gene_names_hvg1000,
    "Mean_Expression": mean_expr.values,
    "Variance": variance_expr.values
})

df_hvg_stats = df_hvg_stats.sort_values(by="Variance", ascending=False).reset_index(drop=True)

hvg_stats_file = Path(GENES_DIR) / "top1000_hvg_con_varianza.csv"
df_hvg_stats.to_csv(hvg_stats_file, index=False)
print(f"HVG file saved to: {hvg_stats_file}")

if ETAPA == "baseline":
    print("----------- BASELINE -----------")
    datasets_raw = {}
    datasets_raw["HVG1000"] = au.splits(df_hvg_1000, bool_mask_hvg, mat_hvg_1000, label_col=LABEL_COL)
    datasets_raw["DEG"]     = au.splits(df_deg, bool_mask_deg, mat_deg, label_col=LABEL_COL)

    results = {}

    for ds_name, model_name, train_setup in experiment_grid:
        best_params = get_params_from_csv(filepath, model_name, ds_name, train_setup)
        ds = datasets_raw[ds_name]
        au.set_seed(42)
        
        gene_list = gene_names_hvg1000 if ds_name == "HVG1000" else gene_names_deg

        X_train_real_raw = ds["X_train_real"]
        y_train_real_enc = ds["y_train_real_enc"]
        pd_label = ds["pd_label"]
        control_label = ds["control_label"]

        idx_train_real_bal = au.balanced_real_indices(
            y_train_real_enc, pd_label, control_label, seed=42
        )

        if train_setup == "Real":
            X_train = X_train_real_raw[idx_train_real_bal]
            y_train = y_train_real_enc[idx_train_real_bal]
        else: # "Real+Synthetic"
            X_train_comb_raw = ds["X_train_comb"]
            y_train_comb_enc = ds["y_train_comb_enc"]
            source_comb = ds["source_comb"]
            idx_train_comb_bal = au.balanced_combined_indices(
                y_train_comb_enc, source_comb, idx_train_real_bal, pd_label, control_label, seed=42
            )
            X_train = X_train_comb_raw[idx_train_comb_bal]
            y_train = y_train_comb_enc[idx_train_comb_bal]

        safe_setup = train_setup.replace("+", "_")

        if model_name == "LASSO": 
            df_nonzero, model = au.get_lasso_nonzero_genes(
                X_train=X_train, y_train=y_train, C=best_params.get("C", 0.01),
                gene_names=gene_list, seed=42, handle_imbalance=False
            )
            results[(ds_name, train_setup, model_name)] = df_nonzero
            print(f"Dataset: {ds_name}, Train setup: {train_setup}, Model: {model_name}")
            
            if "abs_coef" not in df_nonzero.columns and "coef" in df_nonzero.columns:
                df_nonzero["abs_coef"] = df_nonzero["coef"].abs()
            
            df_export_lasso = df_nonzero.sort_values(by="abs_coef", ascending=False).reset_index(drop=True)
            file_path = OUTPUT_DIR / f"lasso_coefficients_{ds_name}_{safe_setup}.csv"
            print(f"Exportando resultados de LASSO a: {file_path}")
            df_export_lasso[["gene_name", "abs_coef", "coef"]].to_csv(file_path, index=False)

        if model_name == "RF": 
            df_importance, model = au.get_rf_variable_importance(
                X_train=X_train, y_train=y_train, 
                n_estimators=best_params.get("n_estimators", 100),
                max_features=best_params.get("max_features", "sqrt"),
                min_samples_leaf=best_params.get("min_samples_leaf", 1),
                gene_names=gene_list, seed=42, handle_imbalance=False
            )
            results[(ds_name, train_setup, model_name)] = df_importance
            print(f"Dataset: {ds_name}, Train setup: {train_setup}, Model: {model_name}")
            
            df_export_rf = df_importance.sort_values(by="importance", ascending=False).reset_index(drop=True)
            file_path = OUTPUT_DIR / f"rf_importances_{ds_name}_{safe_setup}.csv"
            print(f"Exportando resultados de RF a: {file_path}")
            df_export_rf[["gene_name", "importance"]].to_csv(file_path, index=False)

elif ETAPA == "phase1":
    print("----------- PHASE 1 -----------")
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
        
        obs_names = np.array(df_sub.index)[mask_samples]
        var_names = np.array(gene_list)[mask_genes]
        adata = Adata(obs_names, var_names, mat_binned.T[:, mask_genes])
        
        adata.obs_names = obs_names
        adata.var_names = var_names
        
        return df_sub, adata, mask_samples

    print("Binning HVG1000 data (Full dataset)...")
    df_hvg_full, adata_hvg_full, mask_hvg_full = prepare_binned_adata(df_hvg_1000, gene_names_hvg1000)

    print("Binning DEG data (Full dataset)...")
    df_deg_full, adata_deg_full, mask_deg_full = prepare_binned_adata(df_deg, gene_names_deg)

    device = get_device()
    print(f"Loading GeneRAIN model to {device}...")
    model = initiate_model()
    model, _, _ = load_checkpoint(model, None, CHECKPOINT, None)
    model = model.to(device).eval()

    if PROBLEM_NAME == "ppmi":
        phase1_experiments = [
            ("HVG1000", "Real"), ("HVG1000", "Real+Synthetic"),
            ("DEG", "Real"), ("DEG", "Real+Synthetic")
        ]
    else:
        phase1_experiments = [
            ("HVG1000", "Real"), ("DEG", "Real")
        ]

    for ds_name, train_setup in phase1_experiments:
        safe_setup = train_setup.replace("+", "_")
        
        df_base = df_hvg_full if ds_name == "HVG1000" else df_deg_full
        adata_base = adata_hvg_full if ds_name == "HVG1000" else adata_deg_full
        mask_base = mask_hvg_full if ds_name == "HVG1000" else mask_deg_full
        
        if train_setup == "Real" and PROBLEM_NAME == "ppmi":
            mask_real_surviving = ~pd.Series(adata_base.obs_names).str.contains("Synthetic", case=False).values
            X_current = adata_base.X[mask_real_surviving]
        else:
            X_current = adata_base.X

        print(f"\nExtracting {ds_name} ({train_setup}) Embeddings and Attention Scores...")
        emb_out, df_att_out = au.get_generain_embeddings(
            X_current, adata_base.var_names.tolist(), model, device, batch_size=8, attention=True
        )

        out_file = OUTPUT_DIR / f"generain_base_attention_{ds_name}_{safe_setup}.csv"
        df_att_out.to_csv(out_file, index=False)
        print(f"Saving {ds_name} ({train_setup}) attention scores to {out_file}")

elif ETAPA == "phase2":
    print(f"----------- PHASE 2 (LoRA) - {PROBLEM_NAME.upper()} -----------")
    
    device = get_device()
    sys.path.append(f"{PROJ_PATH}/src/LoRA/phase2")
    
    from lora_utils import build_lora_model, CustomMLP 
    from sklearn.preprocessing import LabelEncoder
    import torch

    OUTPUT_TAG = "10_06"
    
    if PROBLEM_NAME == "ppmi":
        lora_experiments = [
            ("HVG1000", "Real"), ("HVG1000", "Real+Synthetic"),
            ("DEG", "Real"), ("DEG", "Real+Synthetic")
        ]
    else:
        lora_experiments = [
            ("HVG1000", "Real"), ("DEG", "Real")
        ]

    for ds_name, train_setup in lora_experiments:
        safe_setup = train_setup.replace("+", "_")
        condition = "augmented" if "Synthetic" in train_setup else "original"
       
        df_base = df_hvg_1000 if ds_name == "HVG1000" else df_deg
        gene_list = gene_names_hvg1000 if ds_name == "HVG1000" else gene_names_deg
       
        if train_setup == "Real" and PROBLEM_NAME == "ppmi":
            mask_real = ~df_base.index.str.contains("Synthetic", case=False)
            df_current = df_base[mask_real].copy()
        else:
            df_current = df_base.copy()
          
        X_current = df_current[gene_list].values.astype(np.float32)
        
        y_true_str = df_current[LABEL_COL].values
        le = LabelEncoder()
        y_true = le.fit_transform(y_true_str) 
     
        # Dynamic path (Phase 1 path no longer needed for fallback)
        if PROBLEM_NAME == "ppmi":
            phase2_model_dir = Path(PROJ_PATH) / "results" / "phase2" / condition / ds_name
        else:
            phase2_model_dir = Path(PROJ_PATH) / "results" / "tejidos" / "phase2" / PROBLEM_NAME / condition / ds_name

        model_path = phase2_model_dir / f"best_model_{OUTPUT_TAG}.pt"
        
        # ---------------------------------------------------------
        # LOAD LORA & MLP (Strictly Phase 2)
        # ---------------------------------------------------------
        if not model_path.exists():
            print(f"\n   -> [Error] Phase 2 model not found at: {model_path}. Skipping...")
            continue
            
        print(f"\nLoading best LoRA model for {ds_name} ({train_setup}) from {model_path}...")
        ckpt = torch.load(model_path, map_location=device)
        lora_params = ckpt["lora_params"]
        lora_state = ckpt["lora_state"]
        
        # 1. Load MLP
        mlp_head = CustomMLP(input_dim=200, num_classes=len(le.classes_))
        if "mlp_state" in ckpt:
            mlp_head.load_state_dict(ckpt["mlp_state"])
        else:
            print("   -> [Error] No 'mlp_state' found in checkpoint. Cannot extract TP/TN. Skipping...")
            continue
        mlp_head.to(device).eval()
        
        # 2. Load LoRA model
        lora_model = build_lora_model(
            base_checkpoint=CHECKPOINT, 
            lora_rank=lora_params["lora_rank"],       
            alpha_mult=lora_params["alpha_mult"],     
            lora_dropout=lora_params.get("lora_dropout", 0.0), 
            device=device
        )
        lora_model.load_state_dict(lora_state)
        lora_model = lora_model.eval()
       
        # ---------------------------------------------------------
        # STEP 1: Extract Embeddings to diagnose patients
        # ---------------------------------------------------------
        print(f"Step 1: Diagnosing all patients ({ds_name} - {train_setup})...")
        emb_out = au.get_generain_embeddings(
            X_current, gene_list, lora_model, device, batch_size=8, attention=False
        )
        
        # Calculate TP and TN strictly using the Phase 2 MLP
        emb_tensor = torch.tensor(emb_out).to(device)
        with torch.no_grad():
            logits = mlp_head(emb_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Assuming class 1 is Positive (PD/Cortex/Sigmoid) and 0 is Negative (Control/Blood/Transverse)
        tp_mask = (preds == 1) & (y_true == 1)
        tn_mask = (preds == 0) & (y_true == 0)
        correct_mask = tp_mask | tn_mask
        
        print(f"   -> Performance: {correct_mask.sum()} correct out of {len(y_true)} patients.")
        print(f"   -> Breakdown: {tp_mask.sum()} True Positives, {tn_mask.sum()} True Negatives.")

        # ---------------------------------------------------------
        # STEP 2: Extract PURE Attention by filtering correct predictions
        # ---------------------------------------------------------
        X_correct = X_current[correct_mask]
        
        if len(X_correct) > 0:
            print(f"Step 2: Extracting averaged attention maps ONLY for the {len(X_correct)} correctly classified patients...")
            _, df_att_out = au.get_generain_embeddings(
                X_correct, gene_list, lora_model, device, batch_size=8, attention=True
            )
            
            # Save the CSV with the _TP_TN tag
            out_file = OUTPUT_DIR / f"generain_lora_attention_{ds_name}_{safe_setup}_TP_TN.csv"
            df_att_out.to_csv(out_file, index=False)
            print(f"   -> CSV successfully saved to: {out_file}\n")
        else:
            print("   -> [Error] No patients were correctly classified. Cannot calculate attention.")

"""elif ETAPA == "phase2":
    print("----------- PHASE 2 (LoRA) -----------")
    
    device = get_device()
    sys.path.append(f"{PROJ_PATH}/src/LoRA/phase2")
    from lora_utils import build_lora_model
    import torch

    OUTPUT_TAG = "10_06"
    if PROBLEM_NAME == "ppmi":
        lora_experiments = [
            ("HVG1000", "Real"), ("HVG1000", "Real+Synthetic"),
            ("DEG", "Real"), ("DEG", "Real+Synthetic")
        ]
    else:
        lora_experiments = [
            ("HVG1000", "Real"), ("DEG", "Real")
        ]

    for ds_name, train_setup in lora_experiments:
        safe_setup = train_setup.replace("+", "_")
        condition = "augmented" if "Synthetic" in train_setup else "original"
        
        # 1. Select the gene list and the base dataframe (RAW DATA)
        df_base = df_hvg_1000 if ds_name == "HVG1000" else df_deg
        gene_list = gene_names_hvg1000 if ds_name == "HVG1000" else gene_names_deg
        
        # 2. Isolate real patients if necessary
        if train_setup == "Real" and PROBLEM_NAME == "ppmi":
            mask_real = ~df_base.index.str.contains("Synthetic", case=False)
            df_current = df_base[mask_real].copy()
        else:
            df_current = df_base.copy()
            
        # 3. Extract the raw numerical matrix (Just as the model saw it during fine-tuning)
        X_current = df_current[gene_list].values.astype(np.float32)
       
        # 4. Setup paths to load the model (Adapted to your new structure)
        if PROBLEM_NAME == "ppmi":
            phase2_model_dir = Path(PROJ_PATH) / "results"  / "phase2" / condition / ds_name
        else:
            phase2_model_dir = Path(PROJ_PATH) / "results" / "tejidos" / "phase2" / PROBLEM_NAME / condition / ds_name

        model_path = phase2_model_dir / f"best_model_{OUTPUT_TAG}.pt"

        # 5. Load model parameters and weights with fallback logic
        if not model_path.exists():
            print(f"\n[Warning] Model not found at: {model_path}")
            print("   -> Using default LoRA parameters (Rank=8, Alpha=2) as fallback...")
            lora_params = {"lora_rank": 8, "alpha_mult": 2, "lora_dropout": 0.0}
            lora_state = None # No trained weights to inject
        else:
            print(f"\nLoading best LoRA model for {ds_name} ({train_setup}) from {model_path}...")
            ckpt = torch.load(model_path, map_location=device)
            lora_params = ckpt["lora_params"]
            lora_state = ckpt["lora_state"]
            print(f"   -> Loaded hyperparameters: Rank={lora_params['lora_rank']}, Alpha={lora_params['alpha_mult']}")
            
        lora_model = build_lora_model(
            base_checkpoint=CHECKPOINT, 
            lora_rank=lora_params["lora_rank"],       
            alpha_mult=lora_params["alpha_mult"],     
            lora_dropout=lora_params.get("lora_dropout", 0.0), 
            device=device
        )
        
        if lora_state is not None:
            lora_model.load_state_dict(lora_state)
            
        lora_model = lora_model.eval()
       
        # 6. Extract attention by passing the raw matrix
        print(f"Extracting Attention Scores for {ds_name} ({train_setup}) using raw data...")
        emb_out, df_att_out = au.get_generain_embeddings(
            X_current, gene_list, lora_model, device, batch_size=8, attention=True
        )
        
        out_file = OUTPUT_DIR / f"generain_lora_attention_{ds_name}_{safe_setup}.csv"
        df_att_out.to_csv(out_file, index=False)
        print(f" CSV successfully saved to: {out_file}")
"""