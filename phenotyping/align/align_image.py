"""
Centers the largest connected component of a binary mask in the middle of
the image frame, and applies the same translation to a companion
intensity/label image (e.g. the raw MIP or a cluster mask).
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops


def align_image_org(binary_image, multimask_image, intensity_image):
    gray_image = np.asarray(binary_image)
    rows, columns = gray_image.shape

    # --- Label connected components (bwlabel / bwconncomp equivalent) ---
    labeled_image, num_features = ndi.label(gray_image > 0)

    if num_features == 0:
        # No components found; nothing to align.
        return gray_image.astype(bool), intensity_image

    # Find the largest connected component by pixel count.
    sizes = ndi.sum(np.ones_like(labeled_image), labeled_image, index=range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1  # +1 because labels start at 1
    largest_component = labeled_image == largest_label

    # --- Region properties (centroid) ---
    props = regionprops(largest_component.astype(np.uint8))[0]
    y_centroid, x_centroid = props.centroid  # skimage returns (row, col)

    # --- Find translation needed to move centroid to image center ---
    middle_x = columns / 2.0
    middle_y = rows / 2.0
    delta_x = middle_x - x_centroid
    delta_y = middle_y - y_centroid

    # scipy.ndimage.shift uses (row_shift, col_shift) ordering.

    # --- Translate the intensity/companion image (nearest-neighbor) ---
    intensity_image = np.asarray(intensity_image)
    translated_intensity = ndi.shift(intensity_image,
            shift=(delta_y, delta_x),
            order=0,
            mode="constant",
            cval=13300)
    
    multimask_image = np.asarray(multimask_image)
    translated_multimask = ndi.shift(multimask_image,
            shift=(delta_y, delta_x),
            order=0,
            mode="constant",
            cval=0)
    
    translated_intensity = translated_intensity.astype(intensity_image.dtype)

    return translated_multimask,translated_intensity