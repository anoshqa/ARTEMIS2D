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
prop = pd.read_csv('cp_measure_features_rotinv.csv')
propnumeric = prop.drop(columns=['Type', 'image_file', 'combined_mask_file', 'cell_mask_file', 'nucleus_mask_file'])
masklist = prop['combined_mask_file'].tolist()
supervised_labels = np.array(prop['Type'])
print(f'Loaded {prop.shape[0]} cells x {propnumeric.shape[1]} rotation-invariant features')

adata = sc.AnnData(X=propnumeric)
groups=['Parental','CarboplatinR','PaclitaxelR','EpirubicinR']
adata.obs['true_labels'] = pd.Categorical(supervised_labels, categories=groups, ordered=True)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=15, svd_solver='arpack')

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

processed_images_np = np.array([np.array(Image.open(os.path.join(mask_folder_path, img_name)).convert('L').resize((150, 150))).astype(np.float32) for img_name in masklist])
processed_images_np[processed_images_np == 0] = np.nan
print(processed_images_np.shape)
cell_area=prop['cell_Area'].values

selected_idx = grid.umap_grid_bin_thumbnails(
    np.array(adata.obsm['X_umap']),
    processed_images_np,
    cell_area=cell_area,
    n_bins_x = 20,
    n_bins_y = 20,
    zoom=0.5,
    place_at="bin_center",
    pick="max_area" ,
    filepath="phenotyping\\unsupervised_analysis\\onlycellprofiler_features\\umap_grid.svg"
)
