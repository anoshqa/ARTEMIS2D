"""
Clean cp_measure_features.csv down to rotation-invariant features only and write
cp_measure_features_rotinv.csv for downstream unsupervised + feature-importance
analysis.

KEPT (rotation-invariant by construction):
  area / perimeter / axis lengths / equivalent+feret diameters, eccentricity,
  solidity, form factor, compactness, euler number, min/mean/median radius,
  convex+filled area, all intensity statistics, mass displacement, granularity,
  radial-distribution FracAtD/MeanFrac/RadialCV, Zernike MAGNITUDE (radial &
  shape), Hu moments, inertia-tensor EIGENVALUES, and the direction-averaged
  Haralick texture summary (see below).

DROPPED:
  - Orientation                         absolute major-axis angle vs the frame
  - ZernikePhase_n_m (all n,m)          phase rotates with the object by design
  - raw Spatial/Central/NormalizedMoment_i_j (except _0_0)
                                        transform under rotation; only their
                                        invariant combinations (Hu) are kept
  - InertiaTensor_i_j matrix entries    only the eigenvalues are invariant
  - BoundingBoxArea, Extent             axis-aligned box changes with rotation
  - Center_X/Y, BoundingBox*_X/Y,       absolute positions in the frame, not
    Location_*_X/Y/Z                    morphology
  - all-NaN columns                     degenerate (e.g. NormalizedMoment_0_0)
  - (near-)zero-variance columns        e.g. Z-locations on 2D MIPs, and the
                                        structurally-degenerate colocalization
                                        columns that are constant across cells

SUMMARIZED:
  - the 4 fixed-direction Haralick texture columns per stat (0/45/90/135 deg,
    *_3_00_256 .. *_3_03_256) are each orientation-dependent individually, but
    their mean across directions is rotation-invariant - collapse to one
    *_3_mean_256 column per stat instead of dropping texture entirely.
"""
import re
import numpy as np
import pandas as pd

CP_INPUT = 'cp_measure_features.csv'
LABEL_SOURCE = 'hc_features_cleaned.csv'   # provides per-cell Type, joined on image_file
OUTPUT = 'cp_measure_features_rotinv.csv'

META_COLS = ['image_file', 'combined_mask_file', 'cell_mask_file', 'nucleus_mask_file']

TEXTURE_DIRECTIONAL_RE = re.compile(r'^(cell|nucleus)_([A-Za-z0-9]+)_3_0([0-3])_256$')


def drop_reason(col):
    """Return a short category string if col is rotation/frame-sensitive, else None."""
    if re.search(r'_Orientation$', col):
        return 'orientation'
    if re.search(r'_ZernikePhase_\d+_\d+$', col):
        return 'zernike_phase'
    m = re.search(r'_(Spatial|Central|Normalized)Moment_(\d+)_(\d+)$', col)
    if m and not (m.group(2) == '0' and m.group(3) == '0'):
        return 'raw_moment'
    if re.search(r'_InertiaTensor_\d+_\d+$', col):   # NOT InertiaTensorEigenvalues_*
        return 'inertia_tensor_entry'
    if re.search(r'_(BoundingBoxArea|Extent)$', col):
        return 'bounding_box_dependent'
    if re.search(r'_Center_[XY]$', col):
        return 'position'
    if re.search(r'_BoundingBox(Minimum|Maximum)_[XY]$', col):
        return 'position'
    if re.search(r'_Location_(CenterMassIntensity|MaxIntensity)_[XYZ]$', col):
        return 'position'
    return None


def collapse_directional_texture(df):
    """Replace 4 orientation-specific Haralick columns per stat with their mean."""
    groups = {}
    for col in df.columns:
        m = TEXTURE_DIRECTIONAL_RE.match(col)
        if m:
            groups.setdefault((m.group(1), m.group(2)), []).append(col)
    collapsed = {}
    directional_cols = []
    for (comp, stat), cols in groups.items():
        collapsed[f'{comp}_{stat}_3_mean_256'] = df[cols].mean(axis=1)
        directional_cols.extend(cols)
    df = df.drop(columns=directional_cols)
    df = pd.concat([df, pd.DataFrame(collapsed, index=df.index)], axis=1)
    return df, len(directional_cols), len(collapsed)


def main():
    cp = pd.read_csv(CP_INPUT)

    # cp_measure_features.csv carries no Type label of its own; pull it from the
    # handcrafted-features table, joined on the shared image_file identifier.
    labels = pd.read_csv(LABEL_SOURCE)[['image_file', 'Type']]
    cp = cp.merge(labels, on='image_file', how='inner')
    print(f"Loaded {CP_INPUT}: {cp.shape[0]} cells x {cp.shape[1]} columns (Type merged from {LABEL_SOURCE})")

    meta = [c for c in META_COLS + ['Type'] if c in cp.columns]
    feats = cp.drop(columns=meta).replace([np.inf, -np.inf], np.nan)

    # 1. Drop all-NaN columns (e.g. NormalizedMoment_0_0/0_1/1_0, undefined by
    #    construction) BEFORE the row-wise dropna, or they would kill every row.
    all_nan = feats.columns[feats.isna().all()].tolist()
    feats = feats.drop(columns=all_nan)
    print(f"Dropped {len(all_nan)} all-NaN columns (degenerate moments)")

    # 2. Drop rotation-sensitive / frame-dependent columns, logged by category.
    reasons = {c: drop_reason(c) for c in feats.columns}
    to_drop = [c for c, r in reasons.items() if r is not None]
    by_cat = {}
    for c in to_drop:
        by_cat.setdefault(reasons[c], []).append(c)
    feats = feats.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} rotation/frame-sensitive columns:")
    for cat, cols in sorted(by_cat.items()):
        print(f"    {cat:24s} {len(cols)}")

    # 3. Collapse directional texture into a rotation-invariant per-stat mean.
    feats, n_dir, n_mean = collapse_directional_texture(feats)
    print(f"Collapsed {n_dir} directional texture columns into {n_mean} direction-averaged means")

    # 4. Drop rows still carrying NaN in any kept feature (missing compartment etc).
    keep_rows = ~feats.isna().any(axis=1)
    if (~keep_rows).any():
        print(f"Dropped {int((~keep_rows).sum())} rows with NaN in a kept feature")
    feats = feats.loc[keep_rows]
    cp = cp.loc[keep_rows]

    # 5. Drop (near-)zero-variance columns (Z-locations on 2D MIPs, structurally
    #    constant colocalization columns, etc.) - carry no discriminative signal.
    zero_var = feats.columns[feats.std() < 1e-8].tolist()
    feats = feats.drop(columns=zero_var)
    print(f"Dropped {len(zero_var)} (near-)zero-variance columns")

    out = pd.concat([cp[meta].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    out.to_csv(OUTPUT, index=False)
    n_feat = out.shape[1] - len(meta)
    print(f"\nWrote {OUTPUT}: {out.shape[0]} cells x {n_feat} rotation-invariant features (+ {len(meta)} meta columns)")


if __name__ == '__main__':
    main()
