"""
Feature importance for the cp_measure feature space.

Answers "when we did PCA/UMAP, which features contribute the most?" and grounds
the answer in interpretable biology by correlating the cp_measure principal
components against 8 handpicked handcrafted descriptors (Fig. 4G analog), plus
image strips that confirm a feature's values actually track a visible trend in
the cells (Fig. 4H analog).

Input is already fully cleaned by phenotyping/feature_extraction/clean_rotation_invariant.py
(Type merged, degenerate columns dropped, NaN rows dropped, rotation-invariant
features only) - this script does no cleaning of its own. Run from the repo root.
"""
import os
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

CP_INPUT = 'cp_measure_features_rotinv.csv'
HC_INPUT = 'hc_features_cleaned.csv'
OUTDIR = "phenotyping/unsupervised_analysis/onlycellprofiler_features"
mask_folder_path = r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Mask_proofread'

META_COLS = ['image_file', 'combined_mask_file', 'cell_mask_file', 'nucleus_mask_file', 'Type']

font_path = r'C:\Users\anous\Downloads\Roboto (1)\Roboto-Regular.ttf'
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
sns.set_palette('deep')
sns.set_context('talk')


def pooled_std(area_a, mean_a, std_a, area_b, mean_b, std_b):
    """Exact std of two compartments pooled into one region.

    Areas stand in for pixel counts (they are a fixed scale factor apart, which
    cancels), and skimage's intensity_std is a population std (ddof=0), so
    combining via E[x^2] - E[x]^2 is exact rather than an approximation.
    """
    n = area_a + area_b
    pooled_mean = (area_a * mean_a + area_b * mean_b) / n
    pooled_sq = (area_a * (std_a ** 2 + mean_a ** 2) + area_b * (std_b ** 2 + mean_b ** 2)) / n
    return np.sqrt(np.clip(pooled_sq - pooled_mean ** 2, 0, None))


# ---------------------------------------------------------------------------
# 1. Load the clean cp_measure matrix and build the 8 handpicked descriptors
#    label_1=cytoplasm, label_2=nucleoplasm, label_3=nucleoli, label_4=lipid,
#    so nucleus = nucleoplasm + nucleoli.
# ---------------------------------------------------------------------------
prop = pd.read_csv(CP_INPUT)
propnumeric = prop.drop(columns=[c for c in META_COLS if c in prop.columns])
feature_names = propnumeric.columns.tolist()
print(f"Loaded {CP_INPUT}: {prop.shape[0]} cells x {len(feature_names)} rotation-invariant features")

# Select explicitly (not the whole frame) - hc also carries a 'Type' column,
# and merging it against cp's 'Type' would silently produce Type_x/Type_y.
hc_cols = ['image_file', 'cell_area_um2', 'cell_mean_intensity', 'cell_eccentricity',
           'label_2_area_um2', 'label_2_intensity_mean', 'label_2_intensity_std',
           'label_3_area_um2', 'label_3_intensity_mean', 'label_3_intensity_std',
           'label_4_area_um2']
hc = pd.read_csv(HC_INPUT)[hc_cols]

merged = prop[['image_file']].merge(hc, on='image_file', how='left', validate='one_to_one')
if merged['cell_area_um2'].isna().any():
    raise ValueError(f"{int(merged['cell_area_um2'].isna().sum())} cells in {CP_INPUT} have no match in {HC_INPUT}")

# hc_ prefix keeps these unambiguous against similarly-named cp_measure columns
# (hc_cell_area vs cp's cell_Area) once the two sets are clustered together.
nucleus_area = merged['label_2_area_um2'] + merged['label_3_area_um2']
handpicked = pd.DataFrame({
    'hc_cell_area': merged['cell_area_um2'],
    'hc_cell_RI_mean': merged['cell_mean_intensity'],
    'hc_cell_eccentricity': merged['cell_eccentricity'],
    'hc_nucleus_area': nucleus_area,
    'hc_nucleus_RI_std': pooled_std(
        merged['label_2_area_um2'], merged['label_2_intensity_mean'], merged['label_2_intensity_std'],
        merged['label_3_area_um2'], merged['label_3_intensity_mean'], merged['label_3_intensity_std'],
    ),
    'hc_nucleus_to_cell_area_ratio': nucleus_area / merged['cell_area_um2'],
    'hc_nucleolus_to_nucleus_area_ratio': merged['label_3_area_um2'] / nucleus_area,
    'hc_lipid_area': merged['label_4_area_um2'],
})
handpicked = handpicked.replace([np.inf, -np.inf], np.nan)
print(f"Built {handpicked.shape[1]} handpicked descriptors: {list(handpicked.columns)}")
if handpicked.isna().any().any():
    print("  NaN counts:\n" + handpicked.isna().sum()[handpicked.isna().sum() > 0].to_string())

# ---------------------------------------------------------------------------
# 2. PCA on the cp_measure features - same scale->PCA as unsupervised_cellprofiler.py
#    so these loadings describe the same embedding as the UMAP/Leiden figures.
# ---------------------------------------------------------------------------
adata = sc.AnnData(X=propnumeric.values.astype(np.float32))
adata.var_names = feature_names
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=15, svd_solver='arpack')

variance_ratio = adata.uns['pca']['variance_ratio']
cumvar = np.cumsum(variance_ratio)
n_pcs_90 = next((i + 1 for i, v in enumerate(cumvar) if v >= 0.90), len(cumvar))
print(f"90% variance explained at PC{n_pcs_90} (PC1={variance_ratio[0]*100:.1f}%, PC2={variance_ratio[1]*100:.1f}%)")

loadings = pd.DataFrame(adata.varm['PCs'], index=feature_names,
                        columns=[f'PC{i+1}' for i in range(adata.varm['PCs'].shape[1])])
loadings.to_csv(f"{OUTDIR}/pca_loadings.csv")

# ---------------------------------------------------------------------------
# 3. Which cp_measure features drive each PC
# ---------------------------------------------------------------------------
N_TOP = 15
N_PCS_PLOT = 3
fig, axes = plt.subplots(1, N_PCS_PLOT, figsize=(6.5 * N_PCS_PLOT, 7))
for i, ax in enumerate(axes):
    pc = f'PC{i+1}'
    ranked = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)[:N_TOP].iloc[::-1]
    ax.barh(ranked.index, ranked.values,
            color=['#d62728' if v < 0 else '#1f77b4' for v in ranked.values])
    ax.set_title(f'Top {N_TOP} loadings - {pc}\n({variance_ratio[i]*100:.1f}% var)')
    ax.axvline(0, color='black', lw=0.8)
    ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/pca_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Correlate cp_measure PCs against the 8 handpicked descriptors (Fig. 4G analog)
# ---------------------------------------------------------------------------
N_PCS_CORR = 5
pc_scores = pd.DataFrame(adata.obsm['X_pca'][:, :N_PCS_CORR],
                         columns=[f'PC{i+1}' for i in range(N_PCS_CORR)])
pc_vs_handpicked = pd.concat([pc_scores, handpicked], axis=1).corr().loc[
    pc_scores.columns, handpicked.columns]
pc_vs_handpicked.to_csv(f"{OUTDIR}/pc_vs_handpicked_correlation.csv")

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(pc_vs_handpicked, cmap='vlag', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 9}, ax=ax)
ax.set_title('cp_measure PCs vs handpicked descriptors')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/pc_vs_handpicked_correlation.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# 4b. Combined correlation map (Fig. 2A analog): handpicked descriptors and
#     cp_measure features clustered together at the paper's |r|>0.85 redundancy
#     threshold. Each cluster collapses to ONE representative - that is the
#     feature worth box-plotting for that axis of variation.
# ---------------------------------------------------------------------------
combined = pd.concat([handpicked, propnumeric.reset_index(drop=True)], axis=1)
combined_corr = combined.corr()

dist = (1 - combined_corr.abs()).to_numpy(copy=True)
np.fill_diagonal(dist, 0)
Z = linkage(squareform(dist, checks=False), method='average')
cluster_ids = fcluster(Z, t=0.15, criterion='distance')   # distance 0.15 <-> |r| > 0.85

clusters = {}
for feat, cid in zip(combined.columns, cluster_ids):
    clusters.setdefault(cid, []).append(feat)

handpicked_set = set(handpicked.columns)


def representative(feats):
    """Prefer a handpicked descriptor (interpretable); else the most central cp feature."""
    pool = [f for f in feats if f in handpicked_set] or feats
    if len(pool) == 1:
        return pool[0]
    return combined_corr.loc[pool, feats].abs().mean(axis=1).idxmax()


rows = []
for cid, feats in clusters.items():
    rep = representative(feats)
    for f in feats:
        rows.append({'feature': f, 'cluster': cid, 'cluster_size': len(feats),
                     'representative': rep, 'has_handpicked': bool(set(feats) & handpicked_set),
                     'source': 'handpicked' if f in handpicked_set else 'cp_measure'})
cluster_table = pd.DataFrame(rows).sort_values(['cluster_size', 'cluster'], ascending=[False, True])
cluster_table.to_csv(f"{OUTDIR}/combined_feature_clusters.csv", index=False)
print(f"\n{combined.shape[1]} combined features -> {len(clusters)} clusters (|r|>0.85)")

# Readable panel: all 8 handpicked descriptors plus the cp representatives of the
# LARGEST clusters. Ranking by cluster size (rather than a fixed maxclust cut)
# keeps the major axes of variation and pushes singleton noise features to the
# bottom, which is what a maxclust cut got wrong.
N_CP_REPS = 25
cp_reps_by_size = []
for cid, feats in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
    rep = representative(feats)
    if rep not in handpicked_set:
        cp_reps_by_size.append(rep)
panel = list(handpicked.columns) + cp_reps_by_size[:N_CP_REPS]

panel_colors = ['#d62728' if f in handpicked_set else '#4c72b0' for f in panel]
g = sns.clustermap(combined[panel].corr(), cmap='vlag', center=0, vmin=-1, vmax=1,
                   figsize=(15, 15), xticklabels=True, yticklabels=True,
                   row_colors=panel_colors, col_colors=panel_colors)
g.ax_heatmap.tick_params(labelsize=7)
g.fig.suptitle('Combined correlation: handpicked (red) + top cp_measure cluster representatives (blue)', y=1.01)
g.savefig(f"{OUTDIR}/combined_correlation.png", dpi=300, bbox_inches='tight')
plt.close(g.fig)

# Box-plot shortlist: one representative per cluster, largest clusters first.
# 'orphan' clusters (no handpicked member) are cp_measure axes the handcrafted
# feature set does not capture - the most interesting candidates for new panels.
shortlist = (cluster_table[cluster_table['feature'] == cluster_table['representative']]
             [['representative', 'cluster_size', 'has_handpicked', 'source']]
             .sort_values('cluster_size', ascending=False))
shortlist.to_csv(f"{OUTDIR}/boxplot_shortlist.csv", index=False)
print("\nBox-plot representative shortlist (top 20 by cluster size):")
print(shortlist.head(20).to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Confirm feature extraction is meaningful: sort cells low->high on a feature
#    and check the images show the expected trend (Fig. 4H analog).
# ---------------------------------------------------------------------------
def feature_trend_strip(values, mask_files, feature_label, filepath, n_bins=8):
    values = np.asarray(values, dtype=float)
    bin_edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(values, bin_edges[1:-1]), 0, n_bins - 1)

    reps = []
    for b in range(n_bins):
        idxs = np.where(bin_idx == b)[0]
        if len(idxs) == 0:
            continue
        med = np.median(values[idxs])
        reps.append(idxs[np.argmin(np.abs(values[idxs] - med))])
    reps = sorted(reps, key=lambda i: values[i])

    fig, axes = plt.subplots(1, len(reps), figsize=(2.3 * len(reps), 3.2))
    for ax, i in zip(np.atleast_1d(axes), reps):
        img = np.array(Image.open(os.path.join(mask_folder_path, mask_files[i])).convert('L')).astype(float)
        img[img == 0] = np.nan
        ax.imshow(img, cmap='viridis')
        ax.set_title(f"{values[i]:.3g}", fontsize=11)
        ax.axis('off')
    fig.suptitle(f"{feature_label}: representative cells, low -> high", y=1.03)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


mask_files = prop['combined_mask_file'].tolist()
top_pc1_feature = loadings['PC1'].abs().idxmax()
top_pc2_feature = loadings['PC2'].abs().idxmax()

trend_targets = {name: handpicked[name].values for name in
                 ['hc_cell_area', 'hc_nucleolus_to_nucleus_area_ratio', 'hc_lipid_area']}
for feat in dict.fromkeys([top_pc1_feature, top_pc2_feature, 'cell_Eccentricity']):
    if feat in propnumeric.columns:
        trend_targets[feat] = propnumeric[feat].values

for name, values in trend_targets.items():
    feature_trend_strip(values, mask_files, name,
                        f"{OUTDIR}/feature_trend_{name.replace('/', '_')}.png")
    print(f"Saved image-trend strip for '{name}'")

print("\nDone. Outputs written to", OUTDIR)
