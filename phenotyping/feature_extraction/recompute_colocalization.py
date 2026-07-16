"""
Recompute only the compartment colocalization block of cp_measure_features.csv,
using the fixed cell_mask/nucleus_mask-based channels from cp_measure_extraction.py
instead of the old combined_mask-only partition (which forced Manders/Costes/RWC
to 0.0 for every cell - see measure_pairwise_colocalization in that file).

Does NOT re-run the ~80%-of-runtime core cell/nucleus measurements or granularity;
only the six pairwise-correlation channels are recomputed, keyed by the crop
identifiers already present in cp_measure_features.csv. Run on the machine with
access to the D:\\TRAINING_DATA_FINAL source images/masks.
"""
import concurrent.futures
import os

import pandas as pd
import skimage.io
from tqdm import tqdm

import cp_measure_extraction as cpm_module
from cp_measure_extraction import (
    images_folder, combined_mask_folder, cell_mask_folder, nucleus_mask_folder,
    RAW_INTENSITY_MIN, RAW_INTENSITY_MAX, MAX_WORKERS,
    measure_pairwise_colocalization,
)
from cp_measure.bulk import get_correlation_measurements

INPUT_PATH = 'cp_measure_features.csv'
OUTPUT_PATH = 'cp_measure_features_fixed_coloc.csv'

# Old, mutually-exclusive-mask columns to drop (structurally degenerate/misleading).
OLD_COLOC_PREFIXES = (
    'cytoplasm_nucleoplasm_', 'cytoplasm_nucleoli_', 'cytoplasm_lipid_',
    'nucleoplasm_nucleoli_', 'nucleoplasm_lipid_', 'nucleoli_lipid_',
)


def _init_worker():
    # measure_pairwise_colocalization reads correlation_measurements as a
    # global inside cp_measure_extraction's own module namespace - set it
    # there directly (this script's `from ... import measure_pairwise_...`
    # is the same module object as cpm_module, not a separate copy).
    cpm_module.correlation_measurements = get_correlation_measurements()


def recompute_crop(image_file, combined_mask_file, cell_mask_file, nucleus_mask_file):
    raw = skimage.io.imread(os.path.join(images_folder, image_file)).astype('float64')
    image = (raw - RAW_INTENSITY_MIN) / (RAW_INTENSITY_MAX - RAW_INTENSITY_MIN)
    image = image.clip(0, 1)

    combined_mask = skimage.io.imread(os.path.join(combined_mask_folder, combined_mask_file))
    cell_mask = skimage.io.imread(os.path.join(cell_mask_folder, cell_mask_file)) > 0
    nucleus_mask = skimage.io.imread(os.path.join(nucleus_mask_folder, nucleus_mask_file)) > 0

    coloc_df = measure_pairwise_colocalization(image, cell_mask, nucleus_mask, combined_mask)
    coloc_df.insert(0, 'image_file', image_file)
    coloc_df.insert(1, 'combined_mask_file', combined_mask_file)
    return coloc_df


def main():
    existing = pd.read_csv(INPUT_PATH)
    old_coloc_cols = [c for c in existing.columns if c.startswith(OLD_COLOC_PREFIXES)]
    print(f"Dropping {len(old_coloc_cols)} old (structurally degenerate) colocalization columns")
    base = existing.drop(columns=old_coloc_cols)

    crops = existing[['image_file', 'combined_mask_file', 'cell_mask_file', 'nucleus_mask_file']].drop_duplicates()
    print(f"Recomputing colocalization for {len(crops)} crops with {MAX_WORKERS} workers")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker) as executor:
        futures = {executor.submit(recompute_crop, *row): row[0] for row in crops.itertuples(index=False)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            image_file = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"FAILED: {image_file}: {exc}")

    new_coloc = pd.concat(results, ignore_index=True)
    merged = base.merge(new_coloc, on=['image_file', 'combined_mask_file'], how='left')
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} ({merged.shape[0]} rows x {merged.shape[1]} cols)")


if __name__ == "__main__":
    main()
