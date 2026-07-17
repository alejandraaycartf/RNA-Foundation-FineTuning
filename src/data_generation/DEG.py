import os
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_folder = "/home/alejandrayf/GeneRAIN/data/processed/splits/tejidos/colon_sigmoid_vs_colon_transverse"
file_name = "combined_balanced_tissues.csv"
label = "tissue"

csv_path = os.path.join(data_folder, file_name)

print(f"Reading file from: {csv_path}")
df = pd.read_csv(csv_path, index_col=0)

metadata_df = df[[label]]
counts_df = df.drop(columns=[label])

counts_df = counts_df.fillna(0).astype(int)

print(f"\nNumber of samples: {len(counts_df)}")
print(f"Number of genes: {len(counts_df.columns)}")

print("\nCreating DESeq2 dataset...")
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata_df,
    design_factors=label,
    n_cpus=8
)

print("Fitting statistical model...")
dds.deseq2()

print("Calculating contrasts (Colon_Sigmoid vs Colon_Transverse)...")
stat_res = DeseqStats(dds, contrast=[label, "Colon_Sigmoid", "Colon_Transverse"], n_cpus=8)
stat_res.summary()

results_df = stat_res.results_df

padj_threshold = 0.05

significant_genes = results_df[
    (results_df['padj'] < padj_threshold)
].sort_values('padj')

display(significant_genes)

print(f"\nFound {len(significant_genes)} significant genes.")
print("\n--- TOP 10 BIOMARKERS ---")
print(significant_genes['padj'].head(10))

output_dir = "/home/alejandrayf/GeneRAIN/data/processed/splits/tejidos/colon_sigmoid_vs_colon_transverse/DEG_output"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "biomarkers_colon_sigmoid_vs_colon_transverse.csv")
significant_genes.to_csv(output_path)
print(f"\nSaved significant genes to: {output_path}")

# 1. Clean the full dataset (drop rows with NaN in padj to avoid math errors)
plot_df = results_df.dropna(subset=['padj', 'log2FoldChange']).copy()

# 2. Apply the mathematical transformation: -log10(padj)
# (Adding a tiny number 1e-300 prevents errors if a padj is exactly 0.0)
plot_df['neg_log10_padj'] = -np.log10(plot_df['padj'] + 1e-300)

# 3. Create a new column to classify each gene (Up, Down, or Not Sig)
plot_df['Regulation'] = 'Not Sig'

# Apply the conditions based ONLY on the adjusted p-value and the middle split (0)
plot_df.loc[(plot_df['padj'] < padj_threshold) & (plot_df['log2FoldChange'] > 0), 'Regulation'] = 'Up'
plot_df.loc[(plot_df['padj'] < padj_threshold) & (plot_df['log2FoldChange'] < 0), 'Regulation'] = 'Down'

# 4. Set up the plot aesthetics
plt.figure(figsize=(10, 8))
colors = {'Not Sig': 'lightgrey', 'Up': '#B31B21', 'Down': '#1465CC'}

# 5. Draw the scatter plot
sns.scatterplot(
    data=plot_df, 
    x='log2FoldChange', 
    y='neg_log10_padj',
    hue='Regulation',
    palette=colors,
    alpha=0.8,
    edgecolor=None,
    s=30
)

# 6. Add threshold line
plt.axhline(y=-np.log10(padj_threshold), color='black', linestyle='--', alpha=0.5)

# 7. Add text labels for the top 10 most significant genes
top_genes = plot_df[plot_df['Regulation'] != 'Not Sig'].sort_values('padj').head(10)

for index, row in top_genes.iterrows():
    plt.text(
        row['log2FoldChange'], 
        row['neg_log10_padj'] + 0.2, 
        index, 
        ha='center', 
        va='bottom', 
        fontsize=9, 
        color='black'
    )

plt.title('Colon Sigmoid vs Colon Transverse')
plt.xlabel('Log2 Fold Change')
plt.ylabel('-Log10 (Adjusted P-value)')
plt.legend(title='Regulation')
plt.grid(False)

plt.show()