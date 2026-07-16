# Combined hc_features + cp_measure feature importance / representative-feature
# boxplot analysis.
#
# Mirrors Fig. 2A-G: correlation-based redundancy clustering to find
# representative, non-redundant descriptors, then per-representative boxplots
# across Parental/CarboplatinR/PaclitaxelR/EpirubicinR with Wilcoxon effect
# sizes (Wendt rank-biserial correlation), annotated with the paper's
# significance thresholds (****>0.7, ***>0.4, **>0.2, *>0.1, ns<0.1).
#
# Two-stage clustering: hc_features (34 cols, physical units, includes
# compartment areas/intensities not in cp_measure) and cp_measure (539 cols,
# pixel-based shape/texture) are each redundancy-clustered on their own first,
# then the pooled representative sets are re-clustered together so that
# cross-source duplicates (e.g. hc's cell_area_um2 vs cp's cell_Area) collapse
# into one cluster. When a cluster contains an hc_feature, it is kept as the
# representative in preference to a cp_measure feature, since hc_features are
# the physically-interpretable, paper-matching descriptors.

import os
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu

OUTDIR = "phenotyping/unsupervised_analysis/combined_features"
BOXPLOT_DIR = f"{OUTDIR}/cluster_representative_boxplots"
os.makedirs(BOXPLOT_DIR, exist_ok=True)

font_path = r'C:\Users\anous\Downloads\Roboto (1)\Roboto-Regular.ttf'
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
sns.set_palette('deep')
sns.set_context('talk')

GROUPS = ['Parental', 'CarboplatinR', 'PaclitaxelR', 'EpirubicinR']
PALETTE = {'Parental': 'red', 'CarboplatinR': 'blue', 'PaclitaxelR': 'yellow', 'EpirubicinR': 'green'}

# ---------------------------------------------------------------------------
# 1. Load hc_features + cp_measure and merge on the shared per-cell mask file
# ---------------------------------------------------------------------------
hc = pd.read_csv('hc_features_cleaned.csv')
cp = pd.read_csv('cp_measure_features_cleaned.csv')

hc = hc.rename(columns={'mask_file': '_cell_key'})
cp = cp.rename(columns={'combined_mask_file': '_cell_key'})

hc_meta = ['Type', 'image_file', '_cell_key', 'mask_key', 'row_index']
cp_meta = ['image_file', '_cell_key', 'cell_mask_file', 'nucleus_mask_file', 'Type']

hc_numeric_cols = [c for c in hc.columns if c not in hc_meta]
cp_numeric_cols = [c for c in cp.columns if c not in cp_meta]

merged = hc[['_cell_key', 'Type'] + hc_numeric_cols].merge(
    cp[['_cell_key'] + cp_numeric_cols], on='_cell_key', how='inner', validate='one_to_one')
print(f"Merged hc ({len(hc)}) + cp ({len(cp)}) on per-cell mask file -> {len(merged)} matched cells")

# Derived, paper-matching compartment ratios (label_2=nucleoplasm, label_3=nucleoli, label_4=lipid)
merged['nucleolus_to_nucleus_area_ratio'] = merged['label_3_area_um2'] / merged['label_2_area_um2']
merged = merged.rename(columns={'label_4_area_um2': 'lipid_area_um2'})
hc_numeric_cols = [c for c in hc_numeric_cols if c != 'label_4_area_um2'] + ['lipid_area_um2', 'nucleolus_to_nucleus_area_ratio']

all_numeric_cols = hc_numeric_cols + cp_numeric_cols
X = merged[all_numeric_cols].replace([np.inf, -np.inf], np.nan)
nan_rows = X.isna().any(axis=1)
if nan_rows.any():
    print(f"Dropping {nan_rows.sum()} rows with NaN (degenerate compartment ratios etc.)")
merged = merged.loc[~nan_rows].reset_index(drop=True)
X = X.loc[~nan_rows].reset_index(drop=True)

zero_var_cols = X.columns[X.std() < 1e-8].tolist()
X = X.drop(columns=zero_var_cols)
hc_numeric_cols = [c for c in hc_numeric_cols if c in X.columns]
cp_numeric_cols = [c for c in cp_numeric_cols if c in X.columns]
print(f"Dropped {len(zero_var_cols)} near-zero-variance columns. Combined matrix: {X.shape[0]} cells x {X.shape[1]} features "
      f"({len(hc_numeric_cols)} hc + {len(cp_numeric_cols)} cp)")


def redundancy_cluster(df, threshold=0.85):
    """Cluster columns of df by 1-|r| distance; return {cluster_id: [features]}."""
    corr = df.corr()
    dist = (1 - corr.abs()).to_numpy(copy=True)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    cluster_ids = fcluster(Z, t=1 - threshold, criterion='distance')
    clusters = {}
    for feat, cid in zip(df.columns, cluster_ids):
        clusters.setdefault(cid, []).append(feat)
    return clusters, corr


def pick_representative(feats, corr, priority_set=None):
    if len(feats) == 1:
        return feats[0]
    if priority_set:
        prio_feats = [f for f in feats if f in priority_set]
        if prio_feats:
            feats_for_centrality = prio_feats
        else:
            feats_for_centrality = feats
    else:
        feats_for_centrality = feats
    centrality = corr.loc[feats_for_centrality, feats].abs().mean(axis=1)
    return centrality.idxmax()


# ---------------------------------------------------------------------------
# 2. Stage 1: redundancy-cluster each source separately
# ---------------------------------------------------------------------------
hc_clusters, hc_corr = redundancy_cluster(X[hc_numeric_cols])
hc_representatives = [pick_representative(feats, hc_corr) for feats in hc_clusters.values()]
print(f"hc_features: {len(hc_numeric_cols)} -> {len(hc_clusters)} clusters -> {len(hc_representatives)} representatives")

cp_clusters, cp_corr = redundancy_cluster(X[cp_numeric_cols])
cp_representatives = [pick_representative(feats, cp_corr) for feats in cp_clusters.values()]
print(f"cp_measure: {len(cp_numeric_cols)} -> {len(cp_clusters)} clusters -> {len(cp_representatives)} representatives")

# ---------------------------------------------------------------------------
# 3. Stage 2: pool representatives, re-cluster to catch cross-source duplicates,
#    keep hc_features as representative whenever a cluster contains one.
# ---------------------------------------------------------------------------
pooled_features = hc_representatives + cp_representatives
pooled_clusters, pooled_corr = redundancy_cluster(X[pooled_features])
hc_rep_set = set(hc_representatives)
final_representatives = [pick_representative(feats, pooled_corr, priority_set=hc_rep_set)
                          for feats in pooled_clusters.values()]
print(f"Pooled: {len(pooled_features)} -> {len(pooled_clusters)} clusters -> {len(final_representatives)} final representatives")

cluster_table = []
for cid, feats in pooled_clusters.items():
    rep = pick_representative(feats, pooled_corr, priority_set=hc_rep_set)
    for f in feats:
        cluster_table.append({'feature': f, 'cluster': cid, 'representative': rep,
                               'source': 'hc' if f in hc_rep_set else 'cp'})
pd.DataFrame(cluster_table).to_csv(f"{OUTDIR}/combined_feature_clusters.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Per-representative boxplots with Wilcoxon effect size vs Parental
# ---------------------------------------------------------------------------
def rank_biserial_effect_size(a, b):
    n1, n2 = len(a), len(b)
    U, _ = mannwhitneyu(a, b, alternative='two-sided')
    return 1 - (2 * U) / (n1 * n2)


def effect_stars(r):
    ar = abs(r)
    if ar > 0.7:
        return '****'
    if ar > 0.4:
        return '***'
    if ar > 0.2:
        return '**'
    if ar > 0.1:
        return '*'
    return 'ns'


def boxplot_feature(ax, feature, values_by_group, effect_sizes):
    data = [values_by_group[g] for g in GROUPS]
    bp = ax.boxplot(data, labels=GROUPS, showfliers=False, patch_artist=True, widths=0.6)
    for patch, g in zip(bp['boxes'], GROUPS):
        patch.set_facecolor(PALETTE[g])
        patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for i, g in enumerate(GROUPS):
        y = values_by_group[g]
        x = rng.normal(i + 1, 0.05, size=len(y))
        ax.scatter(x, y, s=8, c=PALETTE[g], alpha=0.5, edgecolors='none')
    ax.set_title(feature, fontsize=11)
    ax.tick_params(axis='x', labelsize=8, rotation=20)
    ymax = max(values_by_group[g].max() for g in GROUPS)
    ymin = min(values_by_group[g].min() for g in GROUPS)
    span = ymax - ymin if ymax > ymin else 1
    for i, g in enumerate(GROUPS[1:], start=1):
        r = effect_sizes[g]
        ax.text(i + 1, ymax + 0.03 * span, effect_stars(r), ha='center', fontsize=10)
    ax.set_ylim(ymin - 0.05 * span, ymax + 0.15 * span)


effect_size_records = []
values_by_feature = {}
for feature in final_representatives:
    values_by_group = {g: X.loc[merged['Type'] == g, feature].values for g in GROUPS}
    values_by_feature[feature] = values_by_group
    parental = values_by_group['Parental']
    effect_sizes = {}
    for g in GROUPS[1:]:
        r = rank_biserial_effect_size(values_by_group[g], parental)
        effect_sizes[g] = r
        effect_size_records.append({'feature': feature, 'group': g, 'effect_size': r,
                                     'stars': effect_stars(r)})
    fig, ax = plt.subplots(figsize=(5, 4.5))
    boxplot_feature(ax, feature, values_by_group, effect_sizes)
    plt.tight_layout()
    safe_name = feature.replace('/', '_')
    plt.savefig(f"{BOXPLOT_DIR}/{safe_name}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

effect_df = pd.DataFrame(effect_size_records)
effect_df.to_csv(f"{OUTDIR}/effect_sizes_vs_parental.csv", index=False)
print(f"Saved {len(final_representatives)} representative boxplots to {BOXPLOT_DIR}")

# ---------------------------------------------------------------------------
# 5. Headline panel (Fig. 2B-G analog): largest max |effect size| representatives,
#    always including the two features called out from last time (nucleolus/
#    nucleus ratio, lipid area) if they survived as representatives.
# ---------------------------------------------------------------------------
max_effect = effect_df.groupby('feature')['effect_size'].apply(lambda s: s.abs().max()).sort_values(ascending=False)
headline = list(max_effect.index[:6])
for must_have in ['nucleolus_to_nucleus_area_ratio', 'lipid_area_um2']:
    if must_have in final_representatives and must_have not in headline:
        headline.append(must_have)
headline = headline[:8]

n = len(headline)
ncols = 4
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
axes = np.array(axes).reshape(-1)
for i, feature in enumerate(headline):
    values_by_group = values_by_feature[feature]
    effect_sizes = {g: effect_df.loc[(effect_df.feature == feature) & (effect_df.group == g), 'effect_size'].iloc[0]
                     for g in GROUPS[1:]}
    boxplot_feature(axes[i], feature, values_by_group, effect_sizes)
for j in range(n, len(axes)):
    axes[j].axis('off')
fig.suptitle('Headline representative features vs Parental (Fig. 2B-G analog)', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/headline_representative_boxplots.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("Headline features:", headline)
print("Done. Outputs written to", OUTDIR)
