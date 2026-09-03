import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
from PIL import Image
import os
import skimage.io
import phenotyping.umap_grid as grid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

mask_folder_path = r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Mask_proofread'
# full, uncropped MIPs (1350x1350) for representative-cell display, as opposed
# to the categorical Mask_proofread thumbnails used in the bin grid above
images_folder = r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\MIP_proofread'
# display-only contrast window for the representative-cell MIP grid - tighter
# than the 13300-14100 bounds used for actual feature extraction, so this does
# NOT need to match cp_measure_extraction.py's normalization.
RAW_INTENSITY_MIN = 13380.0
RAW_INTENSITY_MAX = 13800.0


def load_mip(image_file):
    raw = skimage.io.imread(os.path.join(images_folder, image_file)).astype(np.float32)
    return np.clip((raw - RAW_INTENSITY_MIN) / (RAW_INTENSITY_MAX - RAW_INTENSITY_MIN), 0, 1)

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

# Clean UMAP style shared by both the supervised-label and Leiden-cluster
# plots: no boxed frame, minimal arrow axes instead of ticks, proportional
# arrow/label sizing (scaled for our [0,1] frame, not a raw ~10-unit range).
fontSize = 10
axisSep = 0.02
arrowProp = 0.15

fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
palette = { "Parental": "red", "CarboplatinR": "blue", "PaclitaxelR": "darkorange", "EpirubicinR": "green" }
for lab, col in palette.items():
    m = (labels == lab)
    ax.scatter(x_norm[m], y_norm[m], s=10, c=col, alpha=0.4, edgecolors="none", label=lab)

ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

ax.set_xlabel("UMAP 1", loc="left", fontsize=fontSize)
ax.set_ylabel("UMAP 2", loc="bottom", fontsize=fontSize)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.xaxis.set_label_coords(xmin - axisSep, ymin - axisSep, transform=ax.transData)
ax.yaxis.set_label_coords(xmin - axisSep, ymin - axisSep, transform=ax.transData)

arrow_len = arrowProp * (xmax - xmin)
head_size = 0.015 * (xmax - xmin)
ax.arrow(xmin, ymin, arrow_len, 0, fc="k", ec="k", lw=1,
         head_width=head_size, head_length=head_size, overhang=0.3,
         length_includes_head=True, clip_on=False)
ax.arrow(xmin, ymin, 0, arrow_len, fc="k", ec="k", lw=1,
         head_width=head_size, head_length=head_size, overhang=0.3,
         length_includes_head=True, clip_on=False)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Supervised Labels", loc="left")
fig.legend(markerscale=2, fontsize=fontSize, loc="center right",
           bbox_to_anchor=[1.1, 0.5], title='Type')

plt.savefig("phenotyping\\unsupervised_analysis\\onlycellprofiler_features\\umap_supervisedlabels_norm01.svg", dpi=500, bbox_inches="tight")
plt.close(fig)

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

# 1. UMAP colored by Leiden cluster - same clean style (fontSize/axisSep/
#    arrowProp) and [0,1] frame as the supervised-label plot above, plus the
#    cmap/uniq ordering shared with the rest of this script.
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
for i, u in enumerate(uniq):
    m = clusters == u
    ax.scatter(x_norm[m], y_norm[m], c=[cmap(i)], s=10, label=u)

ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

ax.set_xlabel("UMAP 1", loc="left", fontsize=fontSize)
ax.set_ylabel("UMAP 2", loc="bottom", fontsize=fontSize)

# Put labels at origin with some small separation from arrows
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.xaxis.set_label_coords(xmin - axisSep, ymin - axisSep, transform=ax.transData)
ax.yaxis.set_label_coords(xmin - axisSep, ymin - axisSep, transform=ax.transData)

# Add arrows
arrow_len = arrowProp * (xmax - xmin)
head_size = 0.015 * (xmax - xmin)
ax.arrow(xmin, ymin, arrow_len, 0, fc="k", ec="k", lw=1,
         head_width=head_size, head_length=head_size, overhang=0.3,
         length_includes_head=True, clip_on=False)
ax.arrow(xmin, ymin, 0, arrow_len, fc="k", ec="k", lw=1,
         head_width=head_size, head_length=head_size, overhang=0.3,
         length_includes_head=True, clip_on=False)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Leiden Clusters", loc="left")
fig.legend(markerscale=2, fontsize=fontSize, loc="center right",
           bbox_to_anchor=[1.1, 0.5], title='Cluster')

plt.savefig(f"{OUTDIR}\\umap_leiden.svg", dpi=500, bbox_inches="tight")
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
#    shown as full MIPs from MIP_proofread (not masks) - one row per cluster
#    (Fig. 4C/E analog). Loaded on demand, not eagerly for the whole dataset -
#    these are full 1350x1350 images, unlike the pre-resized mask thumbnails.
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
            image_file = prop['image_file'].iloc[reps[c]]
            ax.imshow(load_mip(image_file), cmap='gray', vmin=0, vmax=1)
            rep_records.append({'cluster': u, 'rank': c, 'image_file': image_file,
                                'combined_mask_file': masklist[reps[c]]})
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[r, 0].set_ylabel(f'Cluster {u}', rotation=0, ha='right', va='center',
                          fontsize=13, labelpad=28)
fig.suptitle('Representative cells per cluster (closest to PCA centroid) - full MIP', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTDIR}\\cluster_representative_cells.svg", bbox_inches='tight')
plt.close(fig)
pd.DataFrame(rep_records).to_csv(f"{OUTDIR}\\cluster_representative_cells.csv", index=False)

print(f"Top cluster-defining features ({len(top_features)}): {top_features}")
print("Wrote umap_leiden.svg, cluster_composition.svg/.csv, heatmap_leiden_topfeatures.svg, "
      "cluster_feature_heatmap.png, cluster_representative_cells.svg/.csv, "
      "cluster_rank_features.csv, cluster_mean_features.csv")
