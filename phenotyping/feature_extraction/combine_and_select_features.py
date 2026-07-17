"""
Build the combined (handpicked handcrafted + cp_measure) feature matrix and
reduce it with pycytominer.feature_select.

Follows the Cimini lab protocol notebook
(2023_Garcia-Fossa_Cruz_CurrentProtocols, basic_protocol_1 Data_preparation.ipynb):
normalize to the negative control, then feature_select. Two deliberate
deviations from that notebook, both forced by this dataset:

  * samples = Parental, not DMSO. Parental MDA-MB-231 is this experiment's
    negative control that the resistant lines are compared against.
  * the 'blocklist' operation is omitted. pycytominer's blocklist matches
    standard CellProfiler feature names (Cells_AreaShape_*, Nuclei_Correlation_*);
    our columns are cp_measure-style (cell_Area, nucleus_Intensity_*), so the
    blocklist cannot match anything and would be a silent no-op.

Note on normalization: feature_select's drop_outliers compares values against
outlier_cutoff (500), so it is only meaningful on normalized data. On raw
values (cell_Area ~ 1e5) every feature would exceed the cutoff and be dropped.

Feature selection runs on the normalized matrix, but the written output carries
the RAW values of the selected features, since box plots want physical units
(lipid area in um2, etc).

Inputs : cp_measure_features_rotinv.csv (from clean_rotation_invariant.py)
         hc_features_cleaned.csv
Outputs: cp_hc_combined.csv           all combined features, raw values
         cp_hc_combined_selected.csv  selected features only, raw values
         cp_hc_feature_selection_report.csv  per-feature keep/drop + reason
Run from the repo root.
"""
import numpy as np
import pandas as pd
from pycytominer import feature_select, normalize
from pycytominer.operations import correlation_threshold, get_na_columns, variance_threshold

CP_INPUT = 'cp_measure_features_rotinv.csv'
HC_INPUT = 'hc_features_cleaned.csv'
OUT_COMBINED = 'cp_hc_combined.csv'
OUT_SELECTED = 'cp_hc_combined_selected.csv'
OUT_REPORT = 'cp_hc_feature_selection_report.csv'

META_COLS = ['image_file', 'combined_mask_file', 'cell_mask_file', 'nucleus_mask_file', 'Type']
CONTROL_QUERY = "Metadata_Type == 'Parental'"

# Paper's stated redundancy threshold (|r| > 0.85); pycytominer defaults to 0.9.
CORR_THRESHOLD = 0.85
OUTLIER_CUTOFF = 500.0
OPERATIONS = ['drop_na_columns', 'variance_threshold', 'correlation_threshold', 'drop_outliers']


def pooled_std(area_a, mean_a, std_a, area_b, mean_b, std_b):
    """Exact std of two compartments pooled into one region.

    Areas stand in for pixel counts (the scale factor cancels), and skimage's
    intensity_std is a population std (ddof=0), so combining via
    E[x^2] - E[x]^2 is exact rather than an approximation.
    """
    n = area_a + area_b
    pooled_mean = (area_a * mean_a + area_b * mean_b) / n
    pooled_sq = (area_a * (std_a ** 2 + mean_a ** 2) + area_b * (std_b ** 2 + mean_b ** 2)) / n
    return np.sqrt(np.clip(pooled_sq - pooled_mean ** 2, 0, None))


def build_handpicked(image_files):
    """The 8 interpretable descriptors spanning the paper's Fig. 2 axes.

    label_1=cytoplasm, label_2=nucleoplasm, label_3=nucleoli, label_4=lipid,
    so nucleus = nucleoplasm + nucleoli.
    """
    hc_cols = ['image_file', 'cell_area_um2', 'cell_mean_intensity', 'cell_eccentricity',
               'label_2_area_um2', 'label_2_intensity_mean', 'label_2_intensity_std',
               'label_3_area_um2', 'label_3_intensity_mean', 'label_3_intensity_std',
               'label_4_area_um2']
    # Select explicitly: hc also carries a 'Type' column, and merging it against
    # cp's 'Type' would silently produce Type_x/Type_y.
    hc = pd.read_csv(HC_INPUT)[hc_cols]
    m = pd.DataFrame({'image_file': image_files}).merge(
        hc, on='image_file', how='left', validate='one_to_one')
    if m['cell_area_um2'].isna().any():
        raise ValueError(f"{int(m['cell_area_um2'].isna().sum())} cells in {CP_INPUT} "
                         f"have no match in {HC_INPUT}")

    nucleus_area = m['label_2_area_um2'] + m['label_3_area_um2']
    return pd.DataFrame({
        'hc_cell_area': m['cell_area_um2'],
        'hc_cell_RI_mean': m['cell_mean_intensity'],
        'hc_cell_eccentricity': m['cell_eccentricity'],
        'hc_nucleus_area': nucleus_area,
        'hc_nucleus_RI_std': pooled_std(
            m['label_2_area_um2'], m['label_2_intensity_mean'], m['label_2_intensity_std'],
            m['label_3_area_um2'], m['label_3_intensity_mean'], m['label_3_intensity_std'],
        ),
        'hc_nucleus_to_cell_area_ratio': nucleus_area / m['cell_area_um2'],
        'hc_nucleolus_to_nucleus_area_ratio': m['label_3_area_um2'] / nucleus_area,
        'hc_lipid_area': m['label_4_area_um2'],
    })


def main():
    cp = pd.read_csv(CP_INPUT)
    meta = [c for c in META_COLS if c in cp.columns]
    cp_features = [c for c in cp.columns if c not in meta]
    print(f"Loaded {CP_INPUT}: {cp.shape[0]} cells x {len(cp_features)} cp_measure features")

    handpicked = build_handpicked(cp['image_file'])
    hc_features = list(handpicked.columns)
    print(f"Built {len(hc_features)} handpicked descriptors: {hc_features}")

    combined = pd.concat([cp.reset_index(drop=True), handpicked], axis=1)
    combined.to_csv(OUT_COMBINED, index=False)
    features = hc_features + cp_features
    print(f"Wrote {OUT_COMBINED}: {combined.shape[0]} cells x {len(features)} combined features")

    # pycytominer identifies metadata by the Metadata_ prefix, and infers
    # features from CellProfiler compartment names - which our cp_measure-style
    # columns do not use, so pass the feature list explicitly.
    prof = combined.rename(columns={c: f'Metadata_{c}' for c in meta})

    n_control = int((prof['Metadata_Type'] == 'Parental').sum())
    print(f"\nNormalizing (mad_robustize) against {n_control} Parental control cells")
    normalized = normalize(
        profiles=prof,
        features=features,
        meta_features=[f'Metadata_{c}' for c in meta],
        samples=CONTROL_QUERY,
        method='mad_robustize',
        mad_robustize_epsilon=0,
    )
    # epsilon=0 (as in the reference notebook) means a feature whose MAD is 0
    # within the control divides by zero -> inf. Make those NaN so the
    # drop_na_columns operation removes them, which is the intent.
    n_inf = int(np.isinf(normalized[features].to_numpy()).sum())
    if n_inf:
        print(f"  {n_inf} non-finite values from zero-MAD control features -> NaN (drop_na_columns will remove)")
        normalized[features] = normalized[features].replace([np.inf, -np.inf], np.nan)

    # Per-operation breakdown, so a dropped feature has a traceable reason.
    # Each op is evaluated independently against the same input and the
    # exclusions are unioned, so order does not matter.
    reasons = {
        'drop_na_columns': get_na_columns(normalized, features=features, samples='all', cutoff=0.05),
        'variance_threshold': variance_threshold(normalized, features=features, samples='all',
                                                 freq_cut=0.05, unique_cut=0.01),
        'correlation_threshold': correlation_threshold(normalized, features=features, samples='all',
                                                       threshold=CORR_THRESHOLD, method='pearson'),
    }

    selected_df = feature_select(
        profiles=normalized,
        features=features,
        operation=OPERATIONS,
        corr_threshold=CORR_THRESHOLD,
        outlier_cutoff=OUTLIER_CUTOFF,
        output_file=None,
    )
    selected = [c for c in selected_df.columns if not c.startswith('Metadata_')]
    print(f"\nfeature_select: {len(features)} -> {len(selected)} features "
          f"({len(features) - len(selected)} removed) at corr_threshold={CORR_THRESHOLD}")
    for op, excluded in reasons.items():
        print(f"    {op:22s} would exclude {len(excluded)}")

    first_reason = {}
    for op, excluded in reasons.items():
        for f in excluded:
            first_reason.setdefault(f, op)
    report = pd.DataFrame({
        'feature': features,
        'source': ['handpicked' if f in hc_features else 'cp_measure' for f in features],
        'selected': [f in selected for f in features],
        'excluded_by': [first_reason.get(f, 'drop_outliers/other' if f not in selected else '')
                        for f in features],
    }).sort_values(['selected', 'source', 'feature'], ascending=[False, True, True])
    report.to_csv(OUT_REPORT, index=False)

    kept_hc = [f for f in hc_features if f in selected]
    dropped_hc = [f for f in hc_features if f not in selected]
    print(f"\nHandpicked descriptors surviving selection: {len(kept_hc)}/{len(hc_features)}")
    for f in kept_hc:
        print(f"    KEPT    {f}")
    for f in dropped_hc:
        print(f"    dropped {f}  ({first_reason.get(f, 'drop_outliers/other')})")

    # Write RAW values of the selected features - box plots want physical units.
    out = combined[meta + selected]
    out.to_csv(OUT_SELECTED, index=False)
    print(f"\nWrote {OUT_SELECTED}: {out.shape[0]} cells x {len(selected)} selected features (raw values)")
    print(f"Wrote {OUT_REPORT}")


if __name__ == '__main__':
    main()
