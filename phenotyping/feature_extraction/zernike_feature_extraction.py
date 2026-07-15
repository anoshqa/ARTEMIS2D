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

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    tt = np.arctan2(yy, xx)
    mask_circle = rr <= 1

    zernike_basis = zernfun(n, m, rr[mask_circle], tt[mask_circle])
    cell_features = zernike_basis.T @ mask[mask_circle]
    nuc_features = zernike_basis.T @ nuclei[mask_circle]
    rows.append({
                    "cell_zernike": cell_features.tolist(),
                    "nucleus_zernike": nuc_features.tolist(),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    base_dir = Path(r"")
    cell_dir = base_dir / "Cell_mask_cleaned"
    nucleus_dir = base_dir / "Nucleus_mask_cleaned"
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
