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

# cp_measure/skimage 0.26 API deprecation, not actionable from here.
warnings.filterwarnings("ignore", category=FutureWarning, module="cp_measure")
# Expected when a class (e.g. lipid, nucleoli) is absent from a crop's ROI;
# the resulting correlation features are legitimately NaN, not a bug.
warnings.filterwarnings("ignore", category=ConstantInputWarning)

# cp_measure's docs note granularity is ~80% of runtime; downsample it for speed.
GRANULARITY_SUBSAMPLE_SIZE = 0.1
GRANULARITY_IMAGE_SAMPLE_SIZE = 0.1

images_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\MIP_proofread'
combined_mask_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\Mask_proofread'
cell_mask_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\Cell_mask_cleaned'
nucleus_mask_folder = r'D:\TRAINING_DATA_FINAL\Phenotyping_phase\phenotyping phase\Nucleus_mask_cleaned'

# label values produced by qpi_seg.train.split_mask_5_channels.split_into_channels
CHANNEL_LABELS = {
    1: "cytoplasm",
    2: "nucleoplasm",
    3: "nucleoli",
    4: "lipid",
}

OUTPUT_PATH = 'cp_measure_features.csv'

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

measurements = get_core_measurements()
measurements["granularity"] = functools.partial(
    get_granularity,
    subsample_size=GRANULARITY_SUBSAMPLE_SIZE,
    image_sample_size=GRANULARITY_IMAGE_SAMPLE_SIZE,
)
correlation_measurements = get_correlation_measurements()


def measure_single_object(binary_mask, pixels, prefix):
    label_mask = binary_mask.astype(np.int32)
    results = {}
    for _, func in measurements.items():
        results.update(func(label_mask, pixels))
    return pd.DataFrame(results).add_prefix(prefix)


def grayscale_channel(image, combined_mask, label):
    return np.where(combined_mask == label, image, 0.0)


def measure_pairwise_colocalization(image, combined_mask):
    # Real QPI intensity per class (not a 0/1 indicator), restricted to the
    # whole-cell ROI so background pixels don't swamp the correlation.
    cell_roi = (combined_mask > 0).astype(np.int32)
    results = {}
    for label_1, label_2 in itertools.combinations(CHANNEL_LABELS, 2):
        pixels_1 = grayscale_channel(image, combined_mask, label_1)
        pixels_2 = grayscale_channel(image, combined_mask, label_2)
        pair_prefix = f"{CHANNEL_LABELS[label_1]}_{CHANNEL_LABELS[label_2]}_"
        for _, func in correlation_measurements.items():
            for feature_name, value in func(pixels_1, pixels_2, cell_roi).items():
                results[f"{pair_prefix}{feature_name}"] = value
    return pd.DataFrame(results)


remaining = [
    files for files in zip(image_files, combined_mask_files, cell_mask_files, nucleus_mask_files)
    if files[0] not in already_processed
]
if already_processed:
    print(f"Resuming: {len(already_processed)} crops already in {OUTPUT_PATH}, {len(remaining)} left")

header_written = os.path.exists(OUTPUT_PATH)
progress = tqdm(remaining, total=len(remaining))
for image_file, combined_mask_file, cell_mask_file, nucleus_mask_file in progress:
    progress.set_description(image_file)
    image = skimage.io.imread(os.path.join(images_folder, image_file)) / 1e4
    image = np.clip(image, 0, 1).astype(np.float64)

    combined_mask = skimage.io.imread(os.path.join(combined_mask_folder, combined_mask_file))
    cell_mask = skimage.io.imread(os.path.join(cell_mask_folder, cell_mask_file)) > 0
    nucleus_mask = skimage.io.imread(os.path.join(nucleus_mask_folder, nucleus_mask_file)) > 0

    cell_df = measure_single_object(cell_mask, image, "cell_")
    nucleus_df = measure_single_object(nucleus_mask, image, "nucleus_")
    coloc_df = measure_pairwise_colocalization(image, combined_mask)

    row = pd.concat(
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
    row.to_csv(OUTPUT_PATH, mode='a', header=not header_written, index=False)
    header_written = True
