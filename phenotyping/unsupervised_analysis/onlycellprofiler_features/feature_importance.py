#create a feature importance insight
#for eg while doing PCA which feature contributes the most to the variance?
#
# Mirrors the analysis style of Fig. 2A (correlation-based redundancy clustering),
# Fig. 4F-G (PC loadings / interpretable feature correlation) and Fig. 4H
# (representative cells spanning the range of a dominant feature, used here to
# sanity-check that a CellProfiler feature actually tracks a visible morphological
# trend in the images) from the breast cancer differentiation paper.

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

mask_folder_path = r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Mask_proofread'
OUTDIR = "phenotyping/unsupervised_analysis/onlycellprofiler_features"

font_path = r'C:\Users\anous\Downloads\Roboto (1)\Roboto-Regular.ttf'
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
sns.set_palette('deep')
sns.set_context('talk')

META_COLS = ['image_file', 'combined_mask_file', 'cell_mask_file', 'nucleus_mask_file', 'Type']

# ---------------------------------------------------------------------------
# 1. Load + clean the cp_measure feature matrix
# ---------------------------------------------------------------------------
prop = pd.read_csv('cp_measure_features_cleaned.csv')
propnumeric = prop.drop(columns=META_COLS).replace([np.inf, -np.inf], np.nan)

zero_var_cols = propnumeric.columns[propnumeric.std() < 1e-8].tolist()
print(f"Dropping {len(zero_var_cols)} near-zero-variance columns (Z-location / degenerate compartment correlations / degenerate Zernike phase-0 terms)")
propnumeric = propnumeric.drop(columns=zero_var_cols)

feature_names = propnumeric.columns.tolist()
print(f"Feature matrix: {propnumeric.shape[0]} cells x {len(feature_names)} features")

# ---------------------------------------------------------------------------
# 2. Correlation-based redundancy clustering (Fig. 2A analog)
#    Cluster features by 1-|r| distance and pick one representative per
#    cluster at the paper's stated redundancy threshold (|r| > 0.85).
# ---------------------------------------------------------------------------
corr = propnumeric.corr()
dist = (1 - corr.abs()).to_numpy(copy=True)
np.fill_diagonal(dist, 0)
condensed = squareform(dist, checks=False)
Z = linkage(condensed, method='average')
cluster_ids = fcluster(Z, t=0.15, criterion='distance')  # distance 0.15 <-> |r| > 0.85

clusters = {}
for feat, cid in zip(feature_names, cluster_ids):
    clusters.setdefault(cid, []).append(feat)

representatives = []
for cid, feats in clusters.items():
    if len(feats) == 1:
        representatives.append(feats[0])
    else:
        centrality = corr.loc[feats, feats].abs().mean(axis=1)
        representatives.append(centrality.idxmax())

print(f"{len(feature_names)} features -> {len(clusters)} correlation clusters (|r|>0.85) -> {len(representatives)} representative features")

# Full feature correlation structure (labels suppressed - too many to read individually,
# the point is to show the block/redundancy structure)
g_full = sns.clustermap(corr, row_linkage=Z, col_linkage=Z, cmap='vlag', center=0,
                         xticklabels=False, yticklabels=False, figsize=(12, 12))
g_full.fig.suptitle('cp_measure feature correlation structure (all features)', y=1.02)
g_full.savefig(f"{OUTDIR}/feature_correlation_full.png", dpi=300, bbox_inches='tight')
plt.close(g_full.fig)

pd.Series(cluster_ids, index=feature_names, name='cluster').to_csv(f"{OUTDIR}/feature_correlation_clusters.csv")

# The |r|>0.85 threshold above (matching the paper's stated method) still leaves
# 314 representative features out of 539 - too many to label on one readable
# plot, and a purely statistical cut (e.g. maxclust=25) mostly isolates niche
# Zernike-phase terms rather than the biologically interpretable descriptors a
# reader actually wants to see (unlike the paper, which started from only 16
# hand-picked features). Instead, mirror Fig. 2A with a curated, interpretable
# panel spanning cell size, nuclear organization, shape/texture and intensity -
# then show how redundant THIS set is.
curated_bases = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
                  'Eccentricity', 'Solidity', 'FormFactor', 'Compactness', 'Extent',
                  'EulerNumber', 'Intensity_MeanIntensity', 'Intensity_StdIntensity',
                  'Granularity_1', 'Granularity_5', 'Granularity_10', 'Granularity_15']
curated_features = [f'{comp}_{b}' for comp in ['cell', 'nucleus'] for b in curated_bases
                     if f'{comp}_{b}' in propnumeric.columns]

derived = pd.DataFrame(index=propnumeric.index)
if 'nucleus_Area' in propnumeric.columns and 'cell_Area' in propnumeric.columns:
    derived['nucleus_to_cell_area_ratio'] = propnumeric['nucleus_Area'] / propnumeric['cell_Area']
curated_matrix = pd.concat([propnumeric[curated_features], derived], axis=1)

repcorr = curated_matrix.corr()
g_rep = sns.clustermap(repcorr, cmap='vlag', center=0, figsize=(12, 12),
                        xticklabels=True, yticklabels=True, annot=False)
g_rep.ax_heatmap.tick_params(labelsize=8)
g_rep.fig.suptitle('Curated, interpretable cp_measure features (Fig. 2A analog)', y=1.02)
g_rep.savefig(f"{OUTDIR}/feature_correlation_representatives.png", dpi=300, bbox_inches='tight')
plt.close(g_rep.fig)

# ---------------------------------------------------------------------------
# 3. PCA + which features drive each PC (Fig. 4F-G analog)
#    Same scale->PCA pipeline as unsupervised_cellprofiler.py so the loadings
#    describe the same embedding used for the UMAP/Leiden figures.
# ---------------------------------------------------------------------------
adata = sc.AnnData(X=propnumeric.values.astype(np.float32))
adata.var_names = feature_names
adata.obs['Type'] = pd.Categorical(prop['Type'].values,
                                    categories=['Parental', 'CarboplatinR', 'PaclitaxelR', 'EpirubicinR'],
                                    ordered=True)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=15, svd_solver='arpack')

cumvar = np.cumsum(adata.uns['pca']['variance_ratio'])
n_pcs_90 = next((i + 1 for i, v in enumerate(cumvar) if v >= 0.90), len(cumvar))
print(f"90% variance explained at PC{n_pcs_90}")

loadings = pd.DataFrame(adata.varm['PCs'], index=feature_names,
                         columns=[f'PC{i+1}' for i in range(adata.varm['PCs'].shape[1])])
loadings.to_csv(f"{OUTDIR}/pca_loadings.csv")

N_TOP = 15
n_pcs_to_plot = min(3, n_pcs_90)
fig, axes = plt.subplots(1, n_pcs_to_plot, figsize=(6.5 * n_pcs_to_plot, 7))
if n_pcs_to_plot == 1:
    axes = [axes]
top_pc1_features = None
for i, ax in enumerate(axes):
    pc = f'PC{i+1}'
    ranked = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)[:N_TOP]
    ranked = ranked.iloc[::-1]
    colors = ['#d62728' if v < 0 else '#1f77b4' for v in ranked.values]
    ax.barh(ranked.index, ranked.values, color=colors)
    ax.set_title(f'Top {N_TOP} loadings - {pc}\n({adata.uns["pca"]["variance_ratio"][i]*100:.1f}% var)')
    ax.axvline(0, color='black', lw=0.8)
    ax.tick_params(axis='y', labelsize=8)
    if i == 0:
        top_pc1_features = loadings['PC1'].abs().sort_values(ascending=False).index.tolist()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/pca_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# Correlation of top PCs against a handful of interpretable representative features
interpretable_candidates = [f for f in representatives if any(
    k in f for k in ['_Area', '_Eccentricity', '_MajorAxisLength', '_Compactness',
                      '_Solidity', 'Intensity_MeanIntensity', 'Granularity_1',
                      '_Perimeter', '_FormFactor'])]
interpretable_candidates = interpretable_candidates[:20]
pc_scores = pd.DataFrame(adata.obsm['X_pca'][:, :5], columns=[f'PC{i+1}' for i in range(5)])
interp_vals = propnumeric[interpretable_candidates].reset_index(drop=True)
pc_feature_corr = pd.concat([pc_scores, interp_vals], axis=1).corr().loc[
    [f'PC{i+1}' for i in range(5)], interpretable_candidates]

fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(interpretable_candidates)), 5))
sns.heatmap(pc_feature_corr, cmap='vlag', center=0, annot=True, fmt='.2f',
            annot_kws={'size': 7}, ax=ax)
ax.set_title('Correlation of top PCs with interpretable cp_measure features')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/pca_vs_interpretable_features.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Validate that feature extraction is meaningful: sort cells by increasing
#    feature value and check the images actually show the expected trend
#    (analog of Fig. 4H, which did this for the DINO PC2 axis).
# ---------------------------------------------------------------------------
def feature_trend_strip(values, mask_files, feature_label, filepath, n_bins=8):
    values = np.asarray(values, dtype=float)
    order_vals = np.sort(values)
    bin_edges = np.quantile(order_vals, np.linspace(0, 1, n_bins + 1))
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
    for ax, i in zip(axes, reps):
        img = np.array(Image.open(os.path.join(mask_folder_path, mask_files[i])).convert('L')).astype(float)
        img[img == 0] = np.nan
        ax.imshow(img, cmap='viridis')
        ax.set_title(f"{values[i]:.2g}", fontsize=11)
        ax.axis('off')
    fig.suptitle(f"{feature_label}: representative cells, low -> high", y=1.03)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


mask_files = prop['combined_mask_file'].tolist()
trend_features = ['cell_Area']
if top_pc1_features:
    for f in top_pc1_features:
        if f not in trend_features:
            trend_features.append(f)
            break
for f in ['cell_Eccentricity']:
    if f not in trend_features:
        trend_features.append(f)

for feat in trend_features:
    safe_name = feat.replace('/', '_')
    feature_trend_strip(
        propnumeric[feat].values, mask_files, feat,
        f"{OUTDIR}/feature_trend_{safe_name}.png"
    )
    print(f"Saved image-trend validation strip for '{feat}'")

print("Done. Figures written to", OUTDIR)
