"""Batch lipid droplet segmentation for RI MIP (holotomography) images.

One scale-agnostic pipeline for all images (tiny puncta AND large droplets):
  1. Small-scale channel: white tophat (r=6) + Otsu -> puncta.
  2. Large-scale channel: white tophat (r=40) + Otsu, hole-filled, then
     filtered to round, filled components (solidity > 0.88, circularity >
     0.65) so ragged clusters of puncta are not bridged into false blobs.
  3. Union -> semantic mask (lipid area per image).
  4. Distance-transform watershed with adaptive non-max suppression of
     markers (suppression radius ~ droplet radius) -> instance labels,
     splitting touching droplets at every scale without over-splitting
     large ones.
  5. Roundness filter (eccentricity/solidity) removes streaks and debris.

Outputs per image:
  masks_semantic/<name>_semantic.tiff  uint8 binary (0/255)
  masks_instance/<name>_labels.tiff    uint16 instance labels
  overlays/<name>_overlay.png          QC overlay
Tables: per_image_summary.csv, per_droplet.csv

Set PIXEL_SIZE_UM for micron units (None -> pixels).

Usage: python batch_segment.py <input_dir> <output_dir>
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import imageio.v2 as iio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, morphology, measure, segmentation, feature, color

PIXEL_SIZE_UM = 0.095  # e.g. 0.095 for Tomocube HT-2

P = dict(
    sigma_small=1, tophat_small=6,          # puncta channel
    sigma_large=2, tophat_large=40,         # big-droplet channel
    min_solidity_large=0.88, min_circularity_large=0.65,
    min_size_large=80, min_size_final=4,
    nms_factor=0.9,                          # marker suppression radius = nms_factor * droplet radius
    artifact_min_area=20, max_eccentricity=0.95, min_solidity=0.8,
    # nucleolus / low-contrast rejection: big objects must stand out from
    # their local surroundings. Nucleoli sit in bright nucleoplasm ->
    # low ring contrast; genuine droplets sit in dimmer cytoplasm -> high.
    contrast_check_area=200,                 # apply to objects >= this area (px^2)
    min_ring_contrast=150,                   # raw intensity units (RI x 1e4: 150 = deltaRI 0.015)
    ring_width=6,
)


def segment(img):
    img = img.astype(float)

    # small-scale channel: puncta
    t1 = morphology.white_tophat(filters.gaussian(img, P["sigma_small"]),
                                 morphology.disk(P["tophat_small"]))
    m1 = t1 > filters.threshold_otsu(t1[t1 > 0])

    # large-scale channel: medium/big droplets
    t2 = morphology.white_tophat(filters.gaussian(img, P["sigma_large"]),
                                 morphology.disk(P["tophat_large"]))
    m2 = t2 > filters.threshold_otsu(t2[t2 > 0])
    m2 = ndi.binary_fill_holes(m2)
    m2 = morphology.binary_opening(m2, morphology.disk(2))
    m2 = morphology.remove_small_objects(m2, P["min_size_large"])
    lab2 = measure.label(m2)
    keep = np.zeros_like(m2)
    for r in measure.regionprops(lab2):
        circ = 4 * np.pi * r.area / max(r.perimeter, 1) ** 2
        if r.solidity > P["min_solidity_large"] and circ > P["min_circularity_large"]:
            keep[lab2 == r.label] = True

    # semantic mask
    mask = ndi.binary_fill_holes(m1 | keep)
    mask = morphology.remove_small_objects(mask, P["min_size_final"])

    # instance labels: watershed, markers via adaptive NMS on distance map
    dist = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(dist, min_distance=2, labels=mask)
    peaks = peaks[np.argsort(-dist[tuple(peaks.T)])]
    accepted, tree = [], None
    for pk in peaks:
        r = max(2.0, P["nms_factor"] * dist[tuple(pk)])
        if tree is None or not tree.query_ball_point(pk, r):
            accepted.append(pk)
            tree = cKDTree(np.array(accepted))
    markers = np.zeros(img.shape, int)
    for i, pk in enumerate(accepted, 1):
        markers[tuple(pk)] = i
    labels = segmentation.watershed(-dist, markers, mask=mask)

    # remove non-round artifacts (streaks, debris)
    for r in measure.regionprops(labels):
        if r.area > P["artifact_min_area"] and (
                r.eccentricity > P["max_eccentricity"] or r.solidity < P["min_solidity"]):
            labels[labels == r.label] = 0

    # remove nucleoli / low-contrast large objects (ring contrast test)
    allmask = labels > 0
    w = P["ring_width"]
    for r in measure.regionprops(labels):
        if r.area < P["contrast_check_area"]:
            continue
        r0, c0, r1, c1 = r.bbox
        r0, c0 = max(0, r0 - 3 * w), max(0, c0 - 3 * w)
        r1, c1 = min(img.shape[0], r1 + 3 * w), min(img.shape[1], c1 + 3 * w)
        sub = labels[r0:r1, c0:c1] == r.label
        ring = morphology.binary_dilation(sub, morphology.disk(w)) & ~allmask[r0:r1, c0:c1]
        if ring.sum() < 20:
            continue
        contrast = img[r0:r1, c0:c1][sub].mean() - img[r0:r1, c0:c1][ring].mean()
        if contrast < P["min_ring_contrast"]:
            labels[labels == r.label] = 0

    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels


def parse_name(fpath, in_dir):
    """Condition/day from the filename, else from parent folder names
    (supports layouts like <in_dir>/Cond3/Day7/img.tiff or
    <in_dir>/Day14_Cond5/img.tiff)."""
    rel = Path(fpath).relative_to(in_dir)
    sources = [rel.name] + list(rel.parts[:-1])  # filename first, then folders
    cond = day = None
    for s in sources:
        cond = cond or re.search(r"(?:^|[_ .-])c(?:ond)?[_ ]?(\d+)", s, re.IGNORECASE)
        day = day or re.search(r"(?:^|[_ .-])d(?:ay)?[_ ]?(\d+)", s, re.IGNORECASE)
    return (f"Cond{cond.group(1)}" if cond else "?",
            f"Day{day.group(1)}" if day else "?")


def main(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    for d in ("masks_semantic", "masks_instance", "overlays"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    px_area = PIXEL_SIZE_UM ** 2 if PIXEL_SIZE_UM else 1.0
    px_len = PIXEL_SIZE_UM if PIXEL_SIZE_UM else 1.0
    unit = "um" if PIXEL_SIZE_UM else "px"

    summary, droplets, seen = [], [], set()
    for f in sorted(in_dir.rglob("*.tif*")):
        rel = f.relative_to(in_dir)
        base = re.sub(r"-[0-9a-f]{8}(?=\.tiff?$)", "", f.name)  # duplicate uploads
        # unique stem: prefix subfolder names to avoid collisions across folders
        stem = "_".join(list(rel.parts[:-1]) + [Path(base).stem])
        if stem in seen:
            continue
        seen.add(stem)

        img = tifffile.imread(f)
        labels = segment(img)
        semantic = labels > 0
        cond, day = parse_name(f, in_dir)

        tifffile.imwrite(out_dir / "masks_semantic" / f"{stem}_semantic.tiff",
                         (semantic * 255).astype(np.uint8))
        tifffile.imwrite(out_dir / "masks_instance" / f"{stem}_labels.tiff",
                         labels.astype(np.uint16))

        lo, hi = np.percentile(img, [0.5, 99.8])
        n = np.clip((img.astype(float) - lo) / (hi - lo), 0, 1)
        ov = color.label2rgb(labels, image=n, bg_label=0, alpha=0.35)
        ov = segmentation.mark_boundaries(ov, labels, color=(1, 1, 0))
        iio.imwrite(out_dir / "overlays" / f"{stem}_overlay.png",
                    (ov * 255).astype(np.uint8))

        summary.append({
            "image": str(rel), "condition": cond, "day": day,
            "n_droplets": int(labels.max()),
            f"lipid_area_{unit}2": float(semantic.sum() * px_area),
            "lipid_area_fraction": float(semantic.mean())})

        props = measure.regionprops_table(
            labels, intensity_image=img,
            properties=["label", "area", "equivalent_diameter_area",
                        "eccentricity", "solidity", "mean_intensity"])
        d = pd.DataFrame(props)
        d[f"area_{unit}2"] = d.pop("area") * px_area
        d[f"equiv_diameter_{unit}"] = d.pop("equivalent_diameter_area") * px_len
        d.insert(0, "image", str(rel))
        d.insert(1, "condition", cond)
        d.insert(2, "day", day)
        droplets.append(d)
        print(f"{rel}: {cond} {day} n={labels.max()} "
              f"area_frac={semantic.mean():.4f}")

    pd.DataFrame(summary).to_csv(out_dir / "per_image_summary.csv", index=False)
    pd.concat(droplets).to_csv(out_dir / "per_droplet.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
