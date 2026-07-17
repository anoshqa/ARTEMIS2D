"""
Object -> measurement matrix
----------------------------
  cell         : shape (sizeshape+zernike+feret) + intensity + texture
                 (texture/granularity/radial computed WHOLE-CELL only)
  nucleus      : shape + intensity
  cytoplasm    : shape + intensity        (combined == 1)
  nucleoplasm  : shape + intensity        (combined == 2)
  nucleoli     : area + count + intensity (combined == 3)  # multi-droplet: NO shape geometry
  lipid        : area + count + intensity (combined == 4)  # multi-droplet: NO shape geometry
"""
import concurrent.futures
import functools
import os
import warnings

import numpy as np
import pandas as pd
import skimage.io
from scipy.ndimage import label as ndi_label
from scipy.stats import ConstantInputWarning

from cp_measure.core.measureobjectsizeshape import get_sizeshape, get_zernike, get_feret
from cp_measure.core.measureobjectintensity import get_intensity
from cp_measure.core.measureobjectintensitydistribution import (
    get_radial_distribution,
    get_radial_zernikes,
)
from cp_measure.core.measuretexture import get_texture
from cp_measure.core.measuregranularity import get_granularity
# cp_measure's docs note granularity is ~80% of runtime; downsample it for speed.
GRANULARITY_SUBSAMPLE_SIZE = 0.25
GRANULARITY_IMAGE_SAMPLE_SIZE = 0.25

# One worker process per core by default. Lower this if the machine is shared.
MAX_WORKERS = os.cpu_count()

images_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\MIP_proofread'
combined_mask_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\Mask_proofread'
cell_mask_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\Cell_mask_cleaned'
nucleus_mask_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\Nucleus_mask_cleaned'

OUTPUT_PATH = 'cp_measure_features.csv'

# QPI raw-intensity range used to min-max scale into [0, 1] for cp_measure.
RAW_INTENSITY_MIN = 13300.0
RAW_INTENSITY_MAX = 14100.0

# 0.095 x 0.095 um per pixel (for recovering physical areas downstream).
PIXEL_SIZE_UM = 0.095
PIXEL_AREA_UM2 = PIXEL_SIZE_UM ** 2

CYTOPLASM_LABEL = 1
NUCLEOPLASM_LABEL = 2
NUCLEOLI_LABEL = 3
LIPID_LABEL = 4

SHAPE_FUNCS = None       # sizeshape + zernike + feret
INTENSITY_FUNCS = None   # intensity
TEXTURE_FUNCS = None     # texture + granularity + radial (whole-cell only)
TEMPLATES = None         # {func_name: [feature keys]} for NaN-filling empty objects

# Must survive get_granularity's subsample_size=0.1 downsampling with margin to
# spare for its element_size=10 erosion, or it degenerates to a zero-size array
# and crashes - hence 256x256, not a token few pixels.
_TEMPLATE_MASK = np.zeros((256, 256), dtype=np.int32)
_TEMPLATE_MASK[38:218, 38:218] = 1
_TEMPLATE_PIXELS = np.random.default_rng(0).random((256, 256))


def _init_worker():
    global SHAPE_FUNCS, INTENSITY_FUNCS, TEXTURE_FUNCS, TEMPLATES

    warnings.filterwarnings("ignore", category=FutureWarning, module="cp_measure")
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    granularity = functools.partial(
        get_granularity,
        subsample_size=GRANULARITY_SUBSAMPLE_SIZE,
        image_sample_size=GRANULARITY_IMAGE_SAMPLE_SIZE,
    )

    SHAPE_FUNCS = {"sizeshape": get_sizeshape, "zernike": get_zernike, "feret": get_feret}
    INTENSITY_FUNCS = {"intensity": get_intensity}
    TEXTURE_FUNCS = {
        "texture": get_texture,
        "granularity": granularity,
        "radial_distribution": get_radial_distribution,
        "radial_zernikes": get_radial_zernikes,
    }

    # Column names each function would emit on a real object, so empty objects
    # can be NaN-filled with the exact same columns (some funcs hard-crash on an
    # empty mask instead of returning NaN, so they must never be called on one).
    TEMPLATES = {}
    for group in (SHAPE_FUNCS, INTENSITY_FUNCS, TEXTURE_FUNCS):
        for name, func in group.items():
            TEMPLATES[name] = list(func(_TEMPLATE_MASK, _TEMPLATE_PIXELS).keys())


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def _run_group(funcs, binary_mask, pixels, prefix):
    """Run a group of cp_measure funcs on one object; NaN-fill if mask empty."""
    if not binary_mask.any():
        cols = {}
        for name in funcs:
            for key in TEMPLATES[name]:
                cols[f"{prefix}{key}"] = [np.nan]
        return pd.DataFrame(cols)

    label_mask = binary_mask.astype(np.int32)
    results = {}
    for func in funcs.values():
        for key, value in func(label_mask, pixels).items():
            # cp_measure returns length-1 arrays for single objects; unwrap.
            results[f"{prefix}{key}"] = np.asarray(value).ravel()[:1]
    return pd.DataFrame(results)


def measure_shape_intensity(binary_mask, pixels, prefix):
    return pd.concat(
        [
            _run_group(SHAPE_FUNCS, binary_mask, pixels, prefix),
            _run_group(INTENSITY_FUNCS, binary_mask, pixels, prefix),
        ],
        axis=1,
    )


def measure_intensity_only(binary_mask, pixels, prefix):
    return _run_group(INTENSITY_FUNCS, binary_mask, pixels, prefix)


def measure_texture_wholecell(binary_mask, pixels, prefix):
    return _run_group(TEXTURE_FUNCS, binary_mask, pixels, prefix)


def measure_droplets(binary_mask, pixels, prefix):
    """Multi-droplet compartments (nucleoli, lipid): area + count + intensity only."""
    area_px = int(binary_mask.sum())
    count = int(ndi_label(binary_mask)[1]) if area_px else 0
    base = pd.DataFrame({
        f"{prefix}total_area_px2": [area_px],
        f"{prefix}total_area_um2": [area_px * PIXEL_AREA_UM2],
        f"{prefix}count": [count],
    })
    return pd.concat([base, measure_intensity_only(binary_mask, pixels, prefix)], axis=1)


# ---------------------------------------------------------------------------
# Unit conversion (append micron companions to pixel-based features)
# ---------------------------------------------------------------------------
_LENGTH_SUFFIXES = (
    "MajorAxisLength", "MinorAxisLength", "Perimeter", "PerimeterCrofton",
    "EquivalentDiameter", "MaxFeretDiameter", "MinFeretDiameter",
    "MeanRadius", "MedianRadius", "MaximumRadius",
)
_AREA_SUFFIXES = ("Area", "ConvexArea", "FilledArea", "BoundingBoxArea")


def add_micron_columns(df):
    """Append _um / _um2 companions for recognised pixel-based length/area features."""
    new = {}
    for col in df.columns:
        if col.endswith(_AREA_SUFFIXES):
            new[f"{col}_um2"] = df[col] * PIXEL_AREA_UM2
        elif col.endswith(_LENGTH_SUFFIXES):
            new[f"{col}_um"] = df[col] * PIXEL_SIZE_UM
    for k, v in new.items():
        df[k] = v
    return df


# ---------------------------------------------------------------------------
# Per-crop processing
# ---------------------------------------------------------------------------
def process_crop(image_file, combined_mask_file, cell_mask_file, nucleus_mask_file):
    raw = skimage.io.imread(os.path.join(images_folder, image_file)).astype(np.float64)
    image = np.clip((raw - RAW_INTENSITY_MIN) / (RAW_INTENSITY_MAX - RAW_INTENSITY_MIN), 0, 1)

    combined = skimage.io.imread(os.path.join(combined_mask_folder, combined_mask_file))
    cell_mask = skimage.io.imread(os.path.join(cell_mask_folder, cell_mask_file)) > 0
    nucleus_mask = skimage.io.imread(os.path.join(nucleus_mask_folder, nucleus_mask_file)) > 0

    cytoplasm_mask = combined == CYTOPLASM_LABEL
    nucleoplasm_mask = combined == NUCLEOPLASM_LABEL
    nucleoli_mask = combined == NUCLEOLI_LABEL
    lipid_mask = combined == LIPID_LABEL

    parts = [
        pd.DataFrame({
            "image_file": [image_file],
            "combined_mask_file": [combined_mask_file],
            "cell_mask_file": [cell_mask_file],
            "nucleus_mask_file": [nucleus_mask_file],
        }),
        # cell: shape + intensity + whole-cell texture/granularity/radial
        measure_shape_intensity(cell_mask, image, "cell_"),
        measure_texture_wholecell(cell_mask, image, "cell_"),
        # nucleus / cytoplasm / nucleoplasm: shape + intensity
        measure_shape_intensity(nucleus_mask, image, "nucleus_"),
        measure_shape_intensity(cytoplasm_mask, image, "cytoplasm_"),
        measure_shape_intensity(nucleoplasm_mask, image, "nucleoplasm_"),
        # nucleoli / lipid: area + count + intensity (no shape geometry)
        measure_droplets(nucleoli_mask, image, "nucleoli_"),
        measure_droplets(lipid_mask, image, "lipid_"),
    ]
    row = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    return add_micron_columns(row)


def main():
    image_files = sorted(os.listdir(images_folder))
    combined_files = sorted(os.listdir(combined_mask_folder))
    cell_files = sorted(os.listdir(cell_mask_folder))
    nucleus_files = sorted(os.listdir(nucleus_mask_folder))

    n = len(image_files)
    if not (n == len(combined_files) == len(cell_files) == len(nucleus_files)):
        raise ValueError(
            f"File count mismatch: {n} images, {len(combined_files)} combined, "
            f"{len(cell_files)} cell, {len(nucleus_files)} nucleus masks"
        )

    already = set()
    if os.path.exists(OUTPUT_PATH):
        already = set(pd.read_csv(OUTPUT_PATH, usecols=["image_file"])["image_file"])

    remaining = [
        f for f in zip(image_files, combined_files, cell_files, nucleus_files)
        if f[0] not in already
    ]
    if already:
        print(f"Resuming: {len(already)} done, {len(remaining)} remaining")
    print(f"Processing {len(remaining)} crops with {MAX_WORKERS} workers")

    header_written = os.path.exists(OUTPUT_PATH)
    try:
        from tqdm import tqdm
        progress = lambda it, total: tqdm(it, total=total)
    except ImportError:
        progress = lambda it, total: it

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS, initializer=_init_worker
    ) as executor:
        futures = {executor.submit(process_crop, *f): f[0] for f in remaining}
        for future in progress(concurrent.futures.as_completed(futures), len(futures)):
            image_file = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                print(f"FAILED: {image_file}: {exc}")
                continue
            row.to_csv(OUTPUT_PATH, mode="a", header=not header_written, index=False)
            header_written = True

    print(f"Done -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
