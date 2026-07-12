import os
import tifffile
import skimage
from scipy.ndimage import binary_fill_holes, label as ndi_label
from skimage.measure import label, regionprops
import numpy as np
import pandas as pd
import qpi_seg.train.split_mask_5_channels as split
import torch
import torchvision.transforms.v2 as transforms_v2
images_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\MIP_proofread'


combined_mask_folder=r'C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\dlmi2\phenotyping phase\Mask_proofread'

image_files = sorted(os.listdir(images_folder))
mask_files = sorted(os.listdir(combined_mask_folder))

# pixel to um conversion
# 0.095 x 0.095 um
channel_names = ['Cell', 'Nucleus', 'Nucleolus', 'Lipid']
regionprops_rows = []
pixel_size_um = 0.095
pixel_area_um2 = pixel_size_um ** 2
def intensity_std(mask, intensity):
    return np.std(intensity[mask])
def clean_channel_mask(channel_mask):
    binary_mask = channel_mask.astype(bool)
    binary_mask = binary_fill_holes(binary_mask)

    if binary_mask.any():
        labeled_mask, num_labels = ndi_label(binary_mask)
        if num_labels > 1:
            component_sizes = np.bincount(labeled_mask.ravel())
            largest_label = np.argmax(component_sizes[1:]) + 1
            binary_mask = labeled_mask == largest_label

    return binary_mask.astype(int)

feature_rows = []
cols_to_drop = [
    "cell_area",
    "cell_convex_area",
    "cell_perimeter",
    "cell_axis_major_length",
    "cell_axis_minor_length",
    "cell_feret_diameter_max"
    ]
from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x))
for idx in range(len(image_files)):
    image = skimage.io.imread(os.path.join(images_folder, image_files[idx]))/1e4
    combined_mask=skimage.io.imread(os.path.join(combined_mask_folder,mask_files[idx]))
    
    channeled_mask = split.split_into_channels(from_np(combined_mask))
    unstacked_masks = torch.unbind(channeled_mask, dim=0)
    cell_mask= unstacked_masks[1].squeeze().numpy()
    cell_mask_cleaned=clean_channel_mask(cell_mask)
    cellprops = skimage.measure.regionprops_table(cell_mask_cleaned.astype(int), intensity_image=np.asarray(image),properties=['area', 'mean_intensity', 'convex_area', 'perimeter', 'axis_major_length', 'axis_minor_length', 'extent', 'eccentricity', 'solidity', 'feret_diameter_max', 'moments_hu'])
    cellprops_df=pd.DataFrame(cellprops)
    cellprops_df = cellprops_df.add_prefix("cell_")
    if "cell_area" in cellprops_df.columns:
        cellprops_df["cell_area_um2"] = cellprops_df["cell_area"] * pixel_area_um2

    if "cell_convex_area" in cellprops_df.columns:
        cellprops_df["cell_convex_area_um2"] = cellprops_df["cell_convex_area"] * pixel_area_um2

    if "cell_perimeter" in cellprops_df.columns:
        cellprops_df["cell_perimeter_um"] = cellprops_df["cell_perimeter"] * pixel_size_um

    if "cell_axis_major_length" in cellprops_df.columns:
        cellprops_df["cell_axis_major_length_um"] = cellprops_df["cell_axis_major_length"] * pixel_size_um

    if "cell_axis_minor_length" in cellprops_df.columns:
        cellprops_df["cell_axis_minor_length_um"] = cellprops_df["cell_axis_minor_length"] * pixel_size_um

    if "cell_feret_diameter_max" in cellprops_df.columns:
        cellprops_df["cell_feret_diameter_max_um"] = cellprops_df["cell_feret_diameter_max"] * pixel_size_um
    

    cellprops_df = cellprops_df.drop(
    columns=[c for c in cols_to_drop if c in cellprops_df.columns]
    )
    props_table = skimage.measure.regionprops_table(combined_mask, intensity_image=image, properties=['label', 'mean_intensity','area','intensity_std'])
    props_df = pd.DataFrame(props_table)

    # Make per-label table into one row
    # Example columns:
    # label_1_area, label_1_intensity_mean, label_2_area, ...
    label_features = {}

    for _, row in props_df.iterrows():
        label_id = int(row["label"])
        label_features[f"label_{label_id}_area_um2"] = row["area"] * pixel_area_um2
        label_features[f"label_{label_id}_intensity_mean"] = row["mean_intensity"]
        label_features[f"label_{label_id}_intensity_std"] = row["intensity_std"]

    labelprops_df = pd.DataFrame([label_features])
    if len(cellprops_df) == 0:
        cellprops_df = pd.DataFrame([{}])

    one_row_df = pd.concat(
        [
            pd.DataFrame({
                "image_file": [image_files[idx]],
                "mask_file": [mask_files[idx]]
            }),
            cellprops_df.reset_index(drop=True),
            labelprops_df.reset_index(drop=True)
        ],
        axis=1
    )

    feature_rows.append(one_row_df)

featuresdf = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
print(featuresdf.head())
featuresdf.to_csv('hc_features.csv', index=False)