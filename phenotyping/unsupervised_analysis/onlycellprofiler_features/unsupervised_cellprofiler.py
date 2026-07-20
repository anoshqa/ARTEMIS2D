import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
from PIL import Image
import os
import phenotyping.umap_grid as grid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

mask_folder_path = r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Mask_proofread'

font_path=r'C:\Users\anous\Downloads\Roboto (1)\Roboto-Regular.ttf'
fm.fontManager.addfont(font_path)
font_prop=fm.FontProperties(fname=font_path)
plt.rcParams['font.family']=font_prop.get_name()
sns.set_palette('deep')
sns.set_context('talk')

def get_pca_component_for_variance(adata, variance_threshold=0.90):
    cumvar = np.cumsum(adata.uns['pca']['variance_ratio'])
    for i, v in enumerate(cumvar):
        if v >= variance_threshold:
            return i + 1
    return len(cumvar)

# Already fully cleaned by phenotyping/feature_extraction/clean_rotation_invariant.py:
# Type merged, degenerate/all-NaN columns dropped, NaN rows dropped, and only
prophc=pd.read_csv('hc_features_cleaned.csv')
prop = pd.read_csv('cp_selected_features.csv')
propnumeric = prop.drop(columns=['Type', 'image_file', 'combined_mask_file', 'cell_mask_file', 'nucleus_mask_file'])
masklist = prop['combined_mask_file'].tolist()
supervised_labels = np.array(prop['Type'])
print(f'Loaded {prop.shape[0]} cells x {propnumeric.shape[1]} rotation-invariant features')

adata = sc.AnnData(X=propnumeric)
groups=['Parental','CarboplatinR','PaclitaxelR','EpirubicinR']
adata.obs['true_labels'] = pd.Categorical(supervised_labels, categories=groups, ordered=True)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=100, svd_solver='arpack')

pc_component = get_pca_component_for_variance(adata)
print(f"90% variance explained at PC{pc_component}")

sc.tl.pca(adata, n_comps=pc_component, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca', metric='euclidean')
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5')
sc.tl.umap(adata, min_dist=0.1, spread=1, random_state=42)


X = adata.obsm["X_umap"]
x, y = X[:, 0], X[:, 1]

# Normalize to [0,1]
x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

labels = adata.obs["true_labels"].values

fig, ax = plt.subplots(figsize=(8,5), dpi=300)
palette = { "Parental": "red", "CarboplatinR": "blue", "PaclitaxelR": "yellow", "EpirubicinR": "green" }
for lab, col in palette.items():
    m = (labels == lab)
    ax.scatter(x_norm[m], y_norm[m], s=50, c=col, alpha=0.4, edgecolors="none", label=lab)

# axis matches grid
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.savefig("phenotyping\\unsupervised_analysis\\onlycellprofiler_features\\umap_supervisedlabels_norm01.svg", bbox_inches="tight")

# NEAREST: these are categorical label masks (1=cytoplasm, 2=nucleoplasm,
# 3=nucleoli, 4=lipid) - bilinear resizing would invent intermediate label values.
processed_images_np = np.array([np.array(Image.open(os.path.join(mask_folder_path, img_name)).convert('L').resize((150, 150), Image.NEAREST)).astype(np.float32) for img_name in masklist])
processed_images_np[processed_images_np == 0] = np.nan
print(processed_images_np.shape)
cell_area=prophc['cell_area_um2'].values

selected_idx = grid.umap_grid_bin_thumbnails(
    np.array(adata.obsm['X_umap']),
    processed_images_np,
    cell_area=cell_area,
    n_bins_x = 20,
    n_bins_y = 20,
    zoom=0.5,
    place_at="bin_center",
    pick="closest_to_center" ,
    filepath="phenotyping\\unsupervised_analysis\\onlycellprofiler_features\\umap_grid.svg"
)

# ---------------------------------------------------------------------------
# Leiden clusters: UMAP by cluster, treatment composition, and the features
# that make each cluster unique (Fig. 4C-F analog).
# ---------------------------------------------------------------------------
OUTDIR = "phenotyping\\unsupervised_analysis\\onlycellprofiler_features"
N_TOP = 5   # top cluster-defining features per cluster

clusters = adata.obs['leiden_0.5'].astype(str).values
uniq = sorted(adata.obs['leiden_0.5'].cat.categories, key=int)
cmap = plt.get_cmap('tab20b', len(uniq))
print(f"Leiden: {len(uniq)} clusters {uniq}")

# 1. UMAP colored by Leiden cluster (same [0,1] frame as the thumbnail grid)
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
for i, u in enumerate(uniq):
    m = clusters == u
    ax.scatter(x_norm[m], y_norm[m], s=50, color=cmap(i), alpha=0.8,
               edgecolors='none', label=f'cluster {u}')
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.savefig(f"{OUTDIR}\\umap_leiden.svg", bbox_inches="tight")
plt.close(fig)

# 2. Treatment x cluster composition (Fig. 4D analog): fraction of each
#    treatment's cells falling in each cluster.
comp = (adata.obs.groupby('true_labels', observed=False)['leiden_0.5']
        .value_counts(normalize=True).unstack().reindex(columns=uniq))
comp.to_csv(f"{OUTDIR}\\cluster_composition.csv")
ax = comp.plot(kind='bar', stacked=True, figsize=(10, 6), rot=0,
               color=[cmap(i) for i in range(len(uniq))])
ax.set_ylabel('Fraction of cells'); ax.set_xlabel('')
ax.legend(title='Leiden cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{OUTDIR}\\cluster_composition.svg", bbox_inches='tight')
plt.close()

# 3. Cluster-defining features via Wilcoxon rank-sum (each cluster vs rest).
sc.tl.rank_genes_groups(adata, groupby='leiden_0.5', method='wilcoxon')
top = []
for u in uniq:
    top.extend(list(adata.uns['rank_genes_groups']['names'][u][:N_TOP]))
seen = set()
top_features = [f for f in top if not (f in seen or seen.add(f))]  # unique, order kept

rank_df = sc.get.rank_genes_groups_df(adata, group=None)
rank_df.to_csv(f"{OUTDIR}\\cluster_rank_features.csv", index=False)

# 4. Heatmaps of those unique features.
#    (a) scanpy per-cell heatmap grouped by cluster (adata.X is z-scored)
sc.settings.figdir = OUTDIR
sc.pl.heatmap(adata, var_names=top_features, groupby='leiden_0.5', swap_axes=True,
              vmin=-2.5, vmax=2.5, cmap='viridis', show_gene_labels=True,
              show=False, save='_leiden_topfeatures.svg',
              figsize=(10, max(5, 0.35 * len(top_features))))

#    (b) compact, readable heatmap of z-scored CLUSTER MEANS for those features
cluster_means = (pd.DataFrame(adata.X, columns=adata.var_names)
                 .assign(cluster=adata.obs['leiden_0.5'].values)
                 .groupby('cluster').mean())
cluster_means.to_csv(f"{OUTDIR}\\cluster_mean_features.csv")

g = sns.clustermap(cluster_means[top_features].T, cmap='vlag', center=0, vmin=-2, vmax=2,
                   col_cluster=False, annot=False,
                   figsize=(1.2 * len(uniq) + 4, 0.35 * len(top_features) + 3),
                   xticklabels=True, yticklabels=True)
g.ax_heatmap.set_xlabel('Leiden cluster')
g.ax_heatmap.tick_params(labelsize=8)
g.fig.suptitle('Cluster-defining features (z-scored cluster means)', y=1.02)
g.savefig(f"{OUTDIR}\\cluster_feature_heatmap.png", dpi=300, bbox_inches='tight')
plt.close(g.fig)

# 5. Representative cells per cluster: the N cells closest to each cluster's
#    centroid in PCA space (the representation Leiden/neighbors actually used),
#    shown as mask thumbnails - one row per cluster (Fig. 4C/E analog).
N_REP = 5
Xpca = adata.obsm['X_pca']
fig, axes = plt.subplots(len(uniq), N_REP, figsize=(2.2 * N_REP, 2.4 * len(uniq)))
axes = np.atleast_2d(axes)
rep_records = []
for r, u in enumerate(uniq):
    idx = np.where(clusters == u)[0]
    centroid = Xpca[idx].mean(axis=0)
    order = idx[np.argsort(np.linalg.norm(Xpca[idx] - centroid, axis=1))]
    reps = order[:N_REP]
    for c in range(N_REP):
        ax = axes[r, c]
        if c < len(reps):
            ax.imshow(processed_images_np[reps[c]], cmap='viridis', vmin=0, vmax=4, interpolation='nearest')
            rep_records.append({'cluster': u, 'rank': c, 'image_file': prop['image_file'].iloc[reps[c]],
                                'combined_mask_file': masklist[reps[c]]})
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[r, 0].set_ylabel(f'Cluster {u}', rotation=0, ha='right', va='center',
                          fontsize=13, labelpad=28)
fig.suptitle('Representative cells per cluster (closest to PCA centroid)', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTDIR}\\cluster_representative_cells.svg", bbox_inches='tight')
plt.close(fig)
pd.DataFrame(rep_records).to_csv(f"{OUTDIR}\\cluster_representative_cells.csv", index=False)

print(f"Top cluster-defining features ({len(top_features)}): {top_features}")
print("Wrote umap_leiden.svg, cluster_composition.svg/.csv, heatmap_leiden_topfeatures.svg, "
      "cluster_feature_heatmap.png, cluster_representative_cells.svg/.csv, "
      "cluster_rank_features.csv, cluster_mean_features.csv")
