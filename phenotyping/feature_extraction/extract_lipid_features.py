"""Extract lipid features from saved segmentation masks (no re-segmentation).

Reads the outputs of batch_segment_allfol.py plus the raw RI MIP images:

  masks_semantic/<stem>_semantic.tiff  uint8 binary (0/255)
  masks_instance/<stem>_labels.tiff    uint16 instance labels
  <raw_dir>/**/<name>.tiff             uint16 raw MIP, values = RI x 10^4

From the semantic mask (per image):
  - total lipid area (px^2 and um^2), lipid area fraction
  - lipid count (connected components of the semantic mask)

From the instance mask (per droplet):
  - mean lipid RI (mean raw intensity inside the droplet / 10^4)
  - diameter (area-equivalent circular diameter, um)

Outputs: features_per_image.csv, features_per_droplet.csv in <out_dir>.

Usage:
  python extract_lipid_features.py <raw_dir> <masks_dir> [<out_dir>]
  <masks_dir> must contain masks_semantic/ and masks_instance/.
  <out_dir> defaults to <masks_dir>.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage import measure

PIXEL_SIZE_UM = 0.095  # Tomocube HT-2 MIP pixel size; set None for pixel units
RI_SCALE = 1e-4        # raw uint16 value * RI_SCALE = refractive index


def build_stem_map(raw_dir):
    """Map the mask stem (subfolders + cleaned filename, joined by '_')
    back to the raw image path, mirroring batch_segment_allfol.py."""
    stems = {}
    for f in sorted(raw_dir.rglob("*.tif*")):
        rel = f.relative_to(raw_dir)
        base = re.sub(r"-[0-9a-f]{8}(?=\.tiff?$)", "", f.name)
        stem = "_".join(list(rel.parts[:-1]) + [Path(base).stem])
        stems.setdefault(stem, f)
    return stems


def parse_cond_day(stem):
    cond = re.search(r"(?:^|[_ .-])c(?:ond)?[_ ]?(\d+)", stem, re.IGNORECASE)
    day = re.search(r"(?:^|[_ .-])d(?:ay)?[_ ]?(\d+)", stem, re.IGNORECASE)
    return (f"Cond{cond.group(1)}" if cond else "?",
            f"Day{day.group(1)}" if day else "?")


def main(raw_dir, masks_dir, out_dir=None):
    raw_dir, masks_dir = Path(raw_dir), Path(masks_dir)
    out_dir = Path(out_dir) if out_dir else masks_dir
    sem_dir, inst_dir = masks_dir / "masks_semantic", masks_dir / "masks_instance"

    px_area = PIXEL_SIZE_UM ** 2 if PIXEL_SIZE_UM else 1.0
    px_len = PIXEL_SIZE_UM if PIXEL_SIZE_UM else 1.0
    unit = "um" if PIXEL_SIZE_UM else "px"

    stem_to_raw = build_stem_map(raw_dir)
    per_image, per_droplet, missing = [], [], []

    for sem_path in sorted(sem_dir.glob("*_semantic.tif*")):
        stem = re.sub(r"_semantic$", "", sem_path.stem)
        inst_path = inst_dir / f"{stem}_labels.tiff"
        if not inst_path.exists():
            missing.append(f"{stem}: no instance mask")
            continue
        raw_path = stem_to_raw.get(stem)
        if raw_path is None:
            missing.append(f"{stem}: no raw image")
            continue

        semantic = tifffile.imread(sem_path) > 0
        labels = tifffile.imread(inst_path)
        img = tifffile.imread(raw_path).astype(float)
        cond, day = parse_cond_day(stem)

        n_components = int(measure.label(semantic).max())
        per_image.append({
            "image": stem, "condition": cond, "day": day,
            "lipid_area_px2": int(semantic.sum()),
            f"lipid_area_{unit}2": float(semantic.sum() * px_area),
            "lipid_area_fraction": float(semantic.mean()),
            "lipid_count": n_components,
            "n_droplets_instance": int(labels.max()),
        })

        props = measure.regionprops_table(
            labels, intensity_image=img,
            properties=["label", "area", "equivalent_diameter_area",
                        "mean_intensity"])
        d = pd.DataFrame(props)
        d["mean_RI"] = d.pop("mean_intensity") * RI_SCALE
        d[f"diameter_{unit}"] = d.pop("equivalent_diameter_area") * px_len
        d["area_px2"] = d.pop("area").astype(int)
        d[f"area_{unit}2"] = d["area_px2"] * px_area
        d.insert(0, "image", stem)
        d.insert(1, "condition", cond)
        d.insert(2, "day", day)
        per_droplet.append(d)

    pd.DataFrame(per_image).to_csv(out_dir / "features_per_image.csv", index=False)
    pd.concat(per_droplet, ignore_index=True).to_csv(
        out_dir / "features_per_droplet.csv", index=False)

    print(f"processed {len(per_image)} images, "
          f"{sum(len(d) for d in per_droplet)} droplets")
    for m in missing:
        print("SKIPPED", m)


if __name__ == "__main__":
    main(*sys.argv[1:])
