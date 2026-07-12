import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label as ndi_label
from skimage.measure import regionprops


def extract_zernike_features(mask_path, nuclei_path, output_path, size=64, pixel_size_um=0.095):
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
    nuclei = np.array(Image.open(nuclei_path).convert("L"), dtype=np.float32)

    mask = mask > 0
    nuclei = nuclei > 0

    unique_vals = np.unique(mask)
    rows = []

    for val in unique_vals[1:]:
        cell_mask = np.zeros_like(mask, dtype=np.uint8)
        cell_mask[mask == val] = 1

        nuclei_mask = np.zeros_like(nuclei, dtype=np.uint8)
        nuclei_mask[(nuclei > 0) & (cell_mask > 0)] = 1

        labeled_mask, num_labels = ndi_label(cell_mask)
        if num_labels == 0:
            continue

        props = regionprops(labeled_mask)
        for prop in props:
            bbox = prop.bbox
            cell_crop = cell_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            nuc_crop = nuclei_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            if cell_crop.size == 0:
                continue

            if cell_crop.shape[0] % 2 == 1:
                cell_crop = np.pad(cell_crop, ((0, 1), (0, 0)), mode="constant")
                nuc_crop = np.pad(nuc_crop, ((0, 1), (0, 0)), mode="constant")
            if cell_crop.shape[1] % 2 == 1:
                cell_crop = np.pad(cell_crop, ((0, 0), (0, 1)), mode="constant")
                nuc_crop = np.pad(nuc_crop, ((0, 0), (0, 1)), mode="constant")

            if cell_crop_arr.size == 0:
                continue

            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            xx, yy = np.meshgrid(x, y)
            rr = np.sqrt(xx**2 + yy**2)
            tt = np.arctan2(yy, xx)
            mask_circle = rr <= 1

            zernike_basis = zernfun(n, m, rr[mask_circle], tt[mask_circle])
            cell_features = zernike_basis.T @ cell_crop_arr[mask_circle]
            nuc_features = zernike_basis.T @ nuc_crop_arr[mask_circle]

            cell_area_um2 = np.count_nonzero(cell_crop_arr) * pixel_size_um**2
            nuc_area_um2 = np.count_nonzero(nuc_crop_arr) * pixel_size_um**2

            rows.append(
                {
                    "cell_area_um2": cell_area_um2,
                    "nucleus_area_um2": nuc_area_um2,
                    "cell_zernike": cell_features.tolist(),
                    "nucleus_zernike": nuc_features.tolist(),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    base_dir = Path(r"")
    cell_dir = base_dir / "Cell3"
    nucleus_dir = base_dir / "CroppedN"
    output_csv = base_dir / "zernike_features.csv"

    cell_files = sorted(cell_dir.glob("*.tif"))
    nucleus_files = sorted(nucleus_dir.glob("*.tif"))

    if len(cell_files) != len(nucleus_files):
        raise ValueError(f"Mismatch: {len(cell_files)} cell files vs {len(nucleus_files)} nucleus files")

    records = []
    for cell_path, nucleus_path in zip(cell_files, nucleus_files):
        df = extract_zernike_features(cell_path, nucleus_path, output_csv)
        records.append(df)

    combined_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    combined_df.to_csv(output_csv, index=False)
    #TODO: save zernike cell and zernike nucleus features as separate columns in the CSV
    print(f"Saved {len(combined_df)} rows to {output_csv}")
