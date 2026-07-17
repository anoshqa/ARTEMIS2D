"""
pycytominer feature selection for the combined 6-object cp_measure matrix
(output of extract_cp_features.py: cell_ / nucleus_ / cytoplasm_ / nucleoplasm_ /
nucleoli_ / lipid_ blocks).


  * WHOLE-POPULATION normalization (samples='all')
  

Pipeline: (1) drop rotation-sensitive / frame-dependent features by name
(orientation, ZernikePhase, raw moments, InertiaTensor entries, positions -
cp_measure has no module toggle for these, and they are meaningless when cells
have no canonical orientation in the crop); (2) collapse exact-duplicate columns;
(3) normalize; (4) pycytominer feature_select. No hand-protected shortlist - the
two derived interpretable ratios go through selection like any other feature.

Note on units: feature_select's drop_outliers compares against outlier_cutoff
(500), so it only makes sense on normalized data. The written output carries the
RAW values of the selected features (box plots want physical units: areas via
*_total_area_um2 / Area*0.095^2, intensities linearly back-transformable to RI).

Inputs : cp_measure_features.csv     (attached extract_cp_features.py output)
         cp_measure_features_cleaned.csv  (image_file -> Type map; also gates rows)
Outputs: cp_normalized.csv            all features, mad_robustize-NORMALIZED values
         cp_normalized_selected.csv   selected features, NORMALIZED values (for ML/UMAP)
         cp_selected_features.csv     selected features, RAW values (for box plots)
         cp_feature_selection_report.csv  per-feature keep/drop + first reason
Run from the repo root.
"""
import re
import numpy as np
import pandas as pd
from pycytominer import feature_select, normalize
from pycytominer.operations import correlation_threshold, get_na_columns, variance_threshold

# Attached latest extract_cp_features.py output (6-object matrix).
INPUT = r'cp_measure_features.csv'
# Provides per-cell Type and defines the valid row set (join on image_file).
LABEL_SOURCE = 'hc_features_cleaned.csv'

OUT_NORMALIZED = 'cp_normalized.csv'
OUT_NORMALIZED_SELECTED = 'cp_normalized_selected.csv'
OUT_SELECTED = 'cp_selected_features.csv'
OUT_REPORT = 'cp_feature_selection_report.csv'

META_COLS = ['image_file', 'combined_mask_file', 'cell_mask_file', 'nucleus_mask_file']
# Whole-population normalization: median/MAD over all cells (see module docstring).
NORMALIZE_SAMPLES = 'all'

# Paper's stated redundancy threshold (|r| > 0.85); pycytominer's default is 0.9.
CORR_THRESHOLD = 0.85
OUTLIER_CUTOFF = 500.0
OPERATIONS = ['drop_na_columns', 'variance_threshold', 'correlation_threshold', 'drop_outliers']


def rotation_sensitive_reason(col):
    """Return a category if a column changes with absolute image-frame rotation
    (cells have no canonical orientation in these crops), else None. Applies to
    ALL object prefixes (cell/nucleus/cytoplasm/nucleoplasm/nucleoli/lipid), so
    matching is on the suffix, not a fixed prefix. KEEPS rotation-invariant
    descriptors: Area, Eccentricity, Solidity, FormFactor, Compactness, feret,
    radius, ZernikeMagnitude, HuMoment, InertiaTensorEigenvalues, intensity,
    granularity, radial FracAtD/MeanFrac/RadialCV.
    """
    if re.search(r'_Orientation$', col):
        return 'orientation'                       # absolute major-axis angle
    if re.search(r'_ZernikePhase_\d+_\d+$', col):
        return 'zernike_phase'                     # phase rotates with the object
    m = re.search(r'_(Spatial|Central|Normalized)Moment_(\d+)_(\d+)$', col)
    if m and not (m.group(2) == '0' and m.group(3) == '0'):
        return 'raw_moment'                        # only Hu combinations are invariant
    if re.search(r'_InertiaTensor_\d+_\d+$', col):  # NOT InertiaTensorEigenvalues_*
        return 'inertia_tensor_entry'              # only eigenvalues are invariant
    if re.search(r'_(BoundingBoxArea|Extent)(_um2)?$', col):
        return 'bounding_box_dependent'            # axis-aligned box turns with object
    if re.search(r'_Center_[XY]$', col):
        return 'position'
    if re.search(r'_BoundingBox(Minimum|Maximum)_[XY]$', col):
        return 'position'
    if re.search(r'_Location_(CenterMassIntensity|MaxIntensity)_[XYZ]$', col):
        return 'position'
    return None


def add_derived_ratios(df):
    """Interpretable compartment-scaling ratios (Fig. 2C/2F axes), if inputs exist.

    nucleus_Area / cell_Area are cp_measure single-object pixel areas;
    nucleoli_total_area_px2 is the summed multi-blob nucleoli area (same px^2
    units), so the ratios are unit-consistent.
    """
    ratios = {}
    if {'nucleus_Area', 'cell_Area'} <= set(df.columns):
        ratios['nucleus_to_cell_area_ratio'] = df['nucleus_Area'] / df['cell_Area']
    if {'nucleoli_total_area_px2', 'nucleus_Area'} <= set(df.columns):
        ratios['nucleolus_to_nucleus_area_ratio'] = df['nucleoli_total_area_px2'] / df['nucleus_Area']
    if ratios:
        df = pd.concat([df, pd.DataFrame(ratios, index=df.index)], axis=1)
    return df, list(ratios)


def _keep_rank(name):
    """Canonical preference within an exact-duplicate group: physical units
    (_um2 for areas, _um for lengths) > plain Area > shorter/simpler name.
    So a px2/um2 area pair keeps um2, a px/um length pair keeps um, and an Area
    vs its SpatialMoment_0_0 / CentralMoment_0_0 twins keeps the (um2) Area."""
    return (0 if name.endswith(('_um2', '_um')) else 1,
            0 if name.endswith('_Area') else 1,
            len(name), name)


def find_exact_duplicates(df, features, decimals=6):
    """Group features that are identical up to a constant positive scale (|r|=1),
    e.g. area in px2 vs um2, or Area vs its 0th-moment twins. Returns
    (columns_to_drop, [(kept, [dropped...]), ...]). Constant columns are left for
    variance_threshold; merely-highly-correlated (|r|<1) pairs are left for
    pycytominer's correlation_threshold - only exact duplicates are collapsed here.
    """
    sig_to_cols = {}
    for f in features:
        v = df[f].to_numpy(dtype=float)
        finite = ~np.isnan(v)
        if finite.sum() < 2:
            continue
        sd = v[finite].std()
        if sd == 0:
            continue
        z = (v - v[finite].mean()) / sd
        # NaN positions must match too (a compartment absent in the same cells);
        # None keeps them in the signature without float-equality issues.
        sig = tuple(None if np.isnan(x) else round(float(x), decimals) for x in z)
        # Key by (object prefix, signature) so only WITHIN-object structural
        # duplicates merge (px/um, Area vs its moment twins, tensor symmetry).
        # Cross-object coincidences (e.g. cell vs cytoplasm MinIntensity happening
        # to coincide in a partial dataset) are left to correlation_threshold,
        # which decides them statistically on the full data.
        key = (f.split('_')[0], sig)
        sig_to_cols.setdefault(key, []).append(f)

    to_drop, groups = [], []
    for cols in sig_to_cols.values():
        if len(cols) > 1:
            keep = min(cols, key=_keep_rank)
            dropped = [c for c in cols if c != keep]
            to_drop.extend(dropped)
            groups.append((keep, dropped))
    return to_drop, groups


def main():
    df = pd.read_csv(INPUT)
    meta_present = [c for c in META_COLS if c in df.columns]

    # Restrict to the valid (cleaned) row set and attach Type, joined on image_file.
    labels = pd.read_csv(LABEL_SOURCE)[['image_file', 'Type']]
    df = df.merge(labels, on='image_file', how='inner')
    print(f"Loaded {INPUT}: {df.shape[0]} cells matched to Type via {LABEL_SOURCE}")
    print(df['Type'].value_counts().to_string())

    df, derived = add_derived_ratios(df)
    if derived:
        print(f"Added derived ratio columns: {derived}")

    meta = meta_present + ['Type']
    all_features = [c for c in df.columns if c not in meta]
    df[all_features] = df[all_features].replace([np.inf, -np.inf], np.nan)

    # Step 1: drop rotation-sensitive / frame-dependent features (cp_measure has
    # no module-level toggle for these - orientation/phase/position live inside
    # fixed measurement bundles - so filter them by name here). Rotation-invariant
    # descriptors are kept.
    rot_reason = {c: rotation_sensitive_reason(c) for c in all_features}
    rot_drop = [c for c, r in rot_reason.items() if r is not None]
    by_cat = {}
    for c in rot_drop:
        by_cat.setdefault(rot_reason[c], []).append(c)
    print(f"Dropping {len(rot_drop)} rotation/frame-sensitive features:")
    for cat, cols in sorted(by_cat.items()):
        print(f"    {cat:24s} {len(cols)}")

    # Step 2: collapse exact-duplicate columns (px2/um2, Area vs its moment twins,
    # etc.) deterministically, so pycytominer's correlation step doesn't
    # arbitrarily keep the wrong representative.
    kept_after_rot = [c for c in all_features if c not in rot_drop]
    dup_drop, dup_groups = find_exact_duplicates(df, kept_after_rot)
    if dup_drop:
        print(f"Dropping {len(dup_drop)} exact-duplicate columns (identical up to scale), keeping canonical:")
        for keep, dropped in dup_groups:
            print(f"    keep {keep:40s} <- drop {dropped}")
    features = [f for f in kept_after_rot if f not in dup_drop]
    print(f"\n{len(features)} candidate features going into selection")

    # pycytominer identifies metadata by the Metadata_ prefix and infers features
    # from CellProfiler compartment names - which our cp_measure-style columns do
    # not use, so rename meta and pass the feature list explicitly.
    prof = df.rename(columns={c: f'Metadata_{c}' for c in meta})
    meta_renamed = [f'Metadata_{c}' for c in meta]

    print(f"Normalizing (mad_robustize) over all {len(prof)} cells (whole-population)")
    normalized = normalize(
        profiles=prof,
        features=features,
        meta_features=meta_renamed,
        samples=NORMALIZE_SAMPLES,
        method='mad_robustize',
        mad_robustize_epsilon=0,
    )
    # epsilon=0 (as in the reference notebook) divides by zero for any feature
    # whose MAD is 0 across all cells -> inf; make those NaN so drop_na_columns
    # removes them, which is the intent.
    n_inf = int(np.isinf(normalized[features].to_numpy()).sum())
    if n_inf:
        print(f"  {n_inf} non-finite values from zero-MAD control features -> NaN")
        normalized[features] = normalized[features].replace([np.inf, -np.inf], np.nan)

    # Save the full normalized matrix (the normalize() product), meta names restored.
    restore = {f'Metadata_{c}': c for c in meta}
    normalized.rename(columns=restore).to_csv(OUT_NORMALIZED, index=False)
    print(f"Wrote {OUT_NORMALIZED}: {normalized.shape[0]} cells x {len(features)} normalized features")

    # Per-operation breakdown so each dropped feature has a traceable reason.
    # Each op is evaluated independently against the same input and unioned, so
    # order does not matter (drop_outliers is row/value based, not enumerated here).
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
    dup_of = {d: keep for keep, dropped in dup_groups for d in dropped}

    def reason_for(f):
        if rot_reason.get(f):
            return f'rotation_sensitive:{rot_reason[f]}'
        if f in dup_of:
            return f'exact_duplicate_of:{dup_of[f]}'
        if f in first_reason:
            return first_reason[f]
        return 'drop_outliers/other' if f not in selected else ''

    report = pd.DataFrame({
        'feature': all_features,
        'object': [f.split('_')[0] for f in all_features],
        'selected': [f in selected for f in all_features],
        'excluded_by': [reason_for(f) for f in all_features],
    }).sort_values(['selected', 'object', 'feature'], ascending=[False, True, True])
    report.to_csv(OUT_REPORT, index=False)

    # NORMALIZED selected matrix (pycytominer canonical product, for ML/UMAP).
    selected_df.rename(columns=restore).to_csv(OUT_NORMALIZED_SELECTED, index=False)
    print(f"Wrote {OUT_NORMALIZED_SELECTED}: {selected_df.shape[0]} cells x {len(selected)} selected features (normalized)")

    # RAW values of the selected features (physical units for box plots).
    out = df[meta + selected]
    out.to_csv(OUT_SELECTED, index=False)
    print(f"Wrote {OUT_SELECTED}: {out.shape[0]} cells x {len(selected)} selected features (raw values)")
    print(f"Wrote {OUT_REPORT}")
    if derived:
        for r in derived:
            print(f"    derived {r}: {'KEPT' if r in selected else 'dropped (' + first_reason.get(r, 'drop_outliers/other') + ')'}")


if __name__ == '__main__':
    main()
