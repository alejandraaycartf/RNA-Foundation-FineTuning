import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DEFAULT_INPUT_CSV = "combined_balanced_tissues.csv"
SPLITS_DIR = "splits/tejidos/colon_sigmoid_vs_colon_transverse"
GENES_DIR = "genes/tejidos/colon_sigmoid_vs_colon_transverse"
RANDOM_STATE = 42
N_TOP_GENES = 1000

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess RNA-seq data and split into train/val/test (80/10/10). "
            "Saves full-gene splits to splits/ and gene selection lists to genes/."
        )
    )
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT_CSV,
                        help=f"Path to the combined CSV file (default: {DEFAULT_INPUT_CSV}).")
    parser.add_argument("--deg-file", nargs="?", default=None,
                        help="Path to a DEG CSV with a 'padj' column. "
                             "When provided, saves genes/degs_filtered.txt in addition to top HVGs.")
    parser.add_argument("--label-col", default="tissue",
                        help="Target column name in the CSV (default: 'tissue').")
    parser.add_argument("--n-top-genes", type=int, default=N_TOP_GENES,
                        help="Number of top HVGs to select and save (default: 1000).")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE,
                        help="Random seed for reproducibility.")
    parser.add_argument("--splits-dir", default=SPLITS_DIR,
                        help="Output directory for train/val/test split CSVs.")
    parser.add_argument("--genes-dir", default=GENES_DIR,
                        help="Output directory for gene selection lists.")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.splits_dir, exist_ok=True)
    os.makedirs(args.genes_dir, exist_ok=True)

    # -------------------------
    # 1. Load Data
    # -------------------------
    print(f"Loading data from: {args.input_csv}")
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"File not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv, index_col=0)

    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in CSV.")

    label_series = df[args.label_col]
    feature_df = df.drop(columns=[args.label_col])

    print(f"Loaded {feature_df.shape[0]} samples and {feature_df.shape[1]} genes.")

    # -------------------------
    # 2. Numeric conversion and imputation
    # -------------------------
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    
    if feature_df.isna().any().any():
        print("Missing values detected. Imputing with median...")
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    # -------------------------
    # 3. Normalization and log-transform
    # -------------------------
    row_sums = feature_df.sum(axis=1)
    feature_df_normalized = feature_df.div(row_sums, axis=0) * 1e7
    print("Normalized to 10 million reads per sample")

    feature_df_log = np.log10(feature_df_normalized + 1)
    print("Applied log10 transformation.")

    # -------------------------
    # 4. Save Gene Lists (HVGs & Optional DEGs)
    # -------------------------
    # 4A. Always compute and save top HVGs
    gene_variances = feature_df_log.var(axis=0)
    n_genes_to_keep = min(args.n_top_genes, len(gene_variances))
    top_hvg_genes = gene_variances.nlargest(n_genes_to_keep).index.tolist()
    
    hvg_path = os.path.join(args.genes_dir, f"top{args.n_top_genes}_hvg.txt")
    pd.Series(top_hvg_genes).to_csv(hvg_path, index=False, header=False)
    print(f"Top {len(top_hvg_genes)} HVGs saved to: {hvg_path}")

    # 4B. Save preprocessed DEGs if deg-file is provided
    if args.deg_file:
        if os.path.exists(args.deg_file):
            deg_df = pd.read_csv(args.deg_file, index_col=0)
            if "padj" in deg_df.columns:
                sig_genes = deg_df[deg_df["padj"] < 0.05].index.tolist()
                print(f"DEG file loaded: {len(deg_df)} genes total, {len(sig_genes)} with padj < 0.05")
                available_genes = [g for g in sig_genes if g in feature_df_log.columns]
                missing = len(sig_genes) - len(available_genes)
                if missing > 0:
                    print(f"Warning: {missing} DEG gene(s) not found in the expression matrix and will be skipped.")
                
                deg_path = os.path.join(args.genes_dir, "degs_filtered.txt")
                pd.Series(available_genes).to_csv(deg_path, index=False, header=False)
                print(f"Preprocessed DEGs ({len(available_genes)} genes, padj < 0.05) saved to: {deg_path}")
            else:
                print("Warning: 'padj' column not found in DEG file. Skipping DEG filtering.")
        else:
            print(f"Warning: DEG file not found at {args.deg_file}")

    # -------------------------
    # 5. Label encoding 
    # -------------------------
    le = LabelEncoder()
    encoded_labels = le.fit_transform(label_series)
    
    mapping_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label mapping ({args.label_col}): {mapping_dict}")

    # -------------------------
    # 6. 80 / 10 / 10 Stratified Split
    # -------------------------
    strat_values = encoded_labels
    print(f"Stratifying split by '{args.label_col}' distribution: "
          f"{dict(zip(*np.unique(strat_values, return_counts=True)))}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        feature_df_log, encoded_labels, 
        test_size=0.2, random_state=args.random_state, stratify=strat_values
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, random_state=args.random_state, stratify=y_temp
    )

    print(f"\nSplit sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # -------------------------
    # 7. Save splits
    # -------------------------
    def save_split(X, y, name):
        df_split = X.reset_index()          
        df_split.insert(1, args.label_col, y) 
        
        path = os.path.join(args.splits_dir, f"{name}.csv")
        df_split.to_csv(path, index=False)
        
        c0 = (y == 0).sum()
        c1 = (y == 1).sum()
        print(f"  {name}.csv → {len(df_split)} samples (class 0: {c0}, class 1: {c1})")

    print("\nSaving splits:")
    save_split(X_train, y_train, "train")
    save_split(X_val,   y_val,   "val")
    save_split(X_test,  y_test,  "test")

    print(f"\nDone. Splits saved to '{args.splits_dir}/'.")

if __name__ == "__main__":
    main()