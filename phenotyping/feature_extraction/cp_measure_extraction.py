import concurrent.futures
import functools
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import skimage.io
from scipy.stats import ConstantInputWarning
from tqdm import tqdm

from cp_measure.bulk import get_core_measurements, get_correlation_measurements
from cp_measure.core.measuregranularity import get_granularity

# cp_measure's docs note granularity is ~80% of runtime; downsample it for speed.
GRANULARITY_SUBSAMPLE_SIZE = 0.1
GRANULARITY_IMAGE_SAMPLE_SIZE = 0.1

# One worker process per core by default. Lower this if the machine is shared
# with other users/jobs.
MAX_WORKERS = os.cpu_count()

images_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\MIP_proofread'
combined_mask_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\Mask_proofread'
cell_mask_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\Cell_mask_cleaned'
nucleus_mask_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\Nucleus_mask_cleaned'

# label values produced by qpi_seg.train.split_mask_5_channels.split_into_channels,
# used only to pull out the nucleoli/lipid sub-regions below - cell/nucleus
# come from their own whole-region masks instead (see measure_pairwise_colocalization).
NUCLEOLI_LABEL = 3
LIPID_LABEL = 4

OUTPUT_PATH = 'cp_measure_features.csv'

# Populated once per worker process by _init_worker, not pickled across processes.
measurements = None
correlation_measurements = None
core_feature_names = None

# Must survive get_granularity's subsample_size=0.1 downsampling with margin
# to spare for its element_size=10 erosion, or it degenerates to a zero-size
# array and crashes - hence 256x256, not a token few pixels.
_TEMPLATE_MASK = np.zeros((256, 256), dtype=np.int32)
_TEMPLATE_MASK[38:218, 38:218] = 1
_TEMPLATE_PIXELS = np.random.default_rng(0).random((256, 256))


def _init_worker():
    global measurements, correlation_measurements, core_feature_names
    # cp_measure/skimage 0.26 API deprecation, not actionable from here.
    warnings.filterwarnings("ignore", category=FutureWarning, module="cp_measure")
    # Expected when a class (e.g. lipid, nucleoli) is absent from a crop's ROI;
    # the resulting correlation/radial features are legitimately NaN, not a bug.
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)

    measurements = get_core_measurements()
    measurements["granularity"] = functools.partial(
        get_granularity,
        subsample_size=GRANULARITY_SUBSAMPLE_SIZE,
        image_sample_size=GRANULARITY_IMAGE_SAMPLE_SIZE,
    )
    correlation_measurements = get_correlation_measurements()

    # Column names a real object would produce, used to NaN-fill rows for
    # crops where a mask (e.g. a missing nucleus) has zero foreground pixels -
    # some measurements (e.g. get_zernike, via centrosome) hard-crash on an
    # empty object instead of returning NaN, so those must never be called on one.
    core_feature_names = []
    for _, func in measurements.items():
        core_feature_names.extend(func(_TEMPLATE_MASK, _TEMPLATE_PIXELS).keys())


def measure_single_object(binary_mask, pixels, prefix):
    if not binary_mask.any():
        return pd.DataFrame({name: [np.nan] for name in core_feature_names}).add_prefix(prefix)
    label_mask = binary_mask.astype(np.int32)
    results = {}
    for _, func in measurements.items():
        results.update(func(label_mask, pixels))
    return pd.DataFrame(results).add_prefix(prefix)


def grayscale_channel(image, mask):
    return np.where(mask, image, 0.0)


def measure_pairwise_colocalization(image, cell_mask, nucleus_mask, combined_mask):
    # cell/nucleus use their own whole-region masks (nucleus_mask is a genuine
    # subset of cell_mask, so overlap is real) rather than the combined_mask's
    # mutually-exclusive cytoplasm/nucleoplasm partition, which by construction
    # never overlaps anything and forces Manders/Costes/RWC to 0 for every cell.
    # nucleoli/lipid still come from combined_mask since they have no separate
    # whole-region mask of their own.
    channels = {
        "cell": grayscale_channel(image, cell_mask),
        "nucleus": grayscale_channel(image, nucleus_mask),
        "nucleoli": grayscale_channel(image, combined_mask == NUCLEOLI_LABEL),
        "lipid": grayscale_channel(image, combined_mask == LIPID_LABEL),
    }
    # Real QPI intensity per class (not a 0/1 indicator), restricted to the
    # whole-cell ROI so background pixels don't swamp the correlation.
    cell_roi = cell_mask.astype(np.int32)
    results = {}
    for name_1, name_2 in itertools.combinations(channels, 2):
        pixels_1 = channels[name_1]
        pixels_2 = channels[name_2]
        pair_prefix = f"{name_1}_{name_2}_"
        for _, func in correlation_measurements.items():
            for feature_name, value in func(pixels_1, pixels_2, cell_roi).items():
                results[f"{pair_prefix}{feature_name}"] = value
    return pd.DataFrame(results)


RAW_INTENSITY_MIN = 13300.0
RAW_INTENSITY_MAX = 14100.0


def process_crop(image_file, combined_mask_file, cell_mask_file, nucleus_mask_file):
    # cp_measure's skimage-backed measurements require float images bounded
    # in [0, 1]. Raw QPI values only span ~13300-14100, so min-max scale to
    # that observed range (not a fixed divisor) to use the full [0,1] output
    # range and keep real contrast; clip only catches rare outlier pixels
    # past 14100, not a systematic flattening like clipping raw/1e4 did.
    raw = skimage.io.imread(os.path.join(images_folder, image_file)).astype(np.float64)
    image = (raw - RAW_INTENSITY_MIN) / (RAW_INTENSITY_MAX - RAW_INTENSITY_MIN)
    image = np.clip(image, 0, 1)

    combined_mask = skimage.io.imread(os.path.join(combined_mask_folder, combined_mask_file))
    cell_mask = skimage.io.imread(os.path.join(cell_mask_folder, cell_mask_file)) > 0
    nucleus_mask = skimage.io.imread(os.path.join(nucleus_mask_folder, nucleus_mask_file)) > 0

    cell_df = measure_single_object(cell_mask, image, "cell_")
    nucleus_df = measure_single_object(nucleus_mask, image, "nucleus_")
    coloc_df = measure_pairwise_colocalization(image, cell_mask, nucleus_mask, combined_mask)

    return pd.concat(
        [
            pd.DataFrame({
                "image_file": [image_file],
                "combined_mask_file": [combined_mask_file],
                "cell_mask_file": [cell_mask_file],
                "nucleus_mask_file": [nucleus_mask_file],
            }),
            cell_df.reset_index(drop=True),
            nucleus_df.reset_index(drop=True),
            coloc_df.reset_index(drop=True),
        ],
        axis=1,
    )


def main():
    image_files = sorted(os.listdir(images_folder))
    combined_mask_files = sorted(os.listdir(combined_mask_folder))
    cell_mask_files = sorted(os.listdir(cell_mask_folder))
    nucleus_mask_files = sorted(os.listdir(nucleus_mask_folder))

    if not (len(image_files) == len(combined_mask_files) == len(cell_mask_files) == len(nucleus_mask_files)):
        raise ValueError(
            f"File count mismatch: {len(image_files)} images, {len(combined_mask_files)} combined masks, "
            f"{len(cell_mask_files)} cell masks, {len(nucleus_mask_files)} nucleus masks"
        )

    # Resume support: skip crops already written by a previous (interrupted) run.
    already_processed = set()
    if os.path.exists(OUTPUT_PATH):
        already_processed = set(pd.read_csv(OUTPUT_PATH, usecols=["image_file"])["image_file"])

    remaining = [
        files for files in zip(image_files, combined_mask_files, cell_mask_files, nucleus_mask_files)
        if files[0] not in already_processed
    ]
    if already_processed:
        print(f"Resuming: {len(already_processed)} crops already in {OUTPUT_PATH}, {len(remaining)} left")
    print(f"Processing {len(remaining)} crops with {MAX_WORKERS} worker processes")

    header_written = os.path.exists(OUTPUT_PATH)

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker) as executor:
        futures = {executor.submit(process_crop, *files): files[0] for files in remaining}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            image_file = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                print(f"FAILED: {image_file}: {exc}")
                continue
            row.to_csv(OUTPUT_PATH, mode='a', header=not header_written, index=False)
            header_written = True


if __name__ == "__main__":
    main()
