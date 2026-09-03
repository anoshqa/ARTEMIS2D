"""
Grad-CAM / integrated-gradient attribution maps for the condition classifier
---------------------------------------------------------------------------
Answers "which pixels drove the call?" for a checkpoint written by
resnet_condition_classifier.py, and lays the answer out as publication panels.

Two complementary attributions:
  gradcam - gradients of the class logit w.r.t. the last conv block (layer4),
            global-average-pooled into channel weights (Selvaraju et al. 2017).
            Coarse (7x7 at 224 px input, upsampled) but robust: it shows which
            regions of the field the network used.
  ig      - integrated gradients (Sundararajan et al. 2017) along a straight
            path from an empty-field baseline (the image's own 5th percentile,
            i.e. background medium) to the image. Pixel-level and signed; the
            panels show |IG| clipped at the 99th percentile.

Outputs (into the run directory):
  gradcam_panel.svg/png            one row per condition, QPI next to overlay
  integrated_gradients_panel.svg/png
  lipid_enrichment.svg/png         Grad-CAM mass over lipid-dense pixels
  attribution_stats.csv            per-image enrichment values

Example:
  python -m phenotyping.supervised_classification.attribution_maps \
      --run-dir ".../resnet_classification/resnet18_finetune"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

from phenotyping.supervised_classification.resnet_condition_classifier import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    load_image,
    save_figure,
)

# Pixels this far up the intensity distribution are lipid-droplet dense in QPI.
LIPID_PERCENTILE = 95.0
IG_STEPS = 32


# --------------------------------------------------------------------------- #
# attribution
# --------------------------------------------------------------------------- #
def to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W) in [0, 1] -> normalised (1, 3, H, W) batch."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensor = torch.from_numpy(img)[None, None].repeat(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device)


def grad_cam(model: torch.nn.Module, batch: torch.Tensor, class_idx: int,
             size: int) -> np.ndarray:
    """Grad-CAM for one image, upsampled to (size, size) and scaled to [0, 1]."""
    activations, gradients = {}, {}
    layer = model.layer4[-1]
    handles = [
        layer.register_forward_hook(
            lambda _m, _i, out: activations.__setitem__("value", out)
        ),
        layer.register_full_backward_hook(
            lambda _m, _gi, gout: gradients.__setitem__("value", gout[0])
        ),
    ]
    try:
        model.zero_grad(set_to_none=True)
        logits = model(batch)
        logits[0, class_idx].backward()
        # channel weights = GAP of the gradients; CAM = ReLU of the weighted sum
        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(size, size), mode="bilinear",
                            align_corners=False)[0, 0].detach().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def integrated_gradients(model: torch.nn.Module, batch: torch.Tensor,
                         class_idx: int, baseline_value: float,
                         steps: int = IG_STEPS) -> np.ndarray:
    """|IG| for one image, summed over the (replicated) channels."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(batch.device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(batch.device)
    # empty-field baseline, pushed through the same normalisation as the image
    baseline = ((torch.full_like(batch, baseline_value) - mean) / std)

    total = torch.zeros_like(batch)
    for alpha in np.linspace(1.0 / steps, 1.0, steps):
        point = (baseline + alpha * (batch - baseline)).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(point)[0, class_idx].backward()
        total += point.grad
    attribution = (batch - baseline) * total / steps
    return attribution.sum(dim=1)[0].abs().detach().cpu().numpy()


def lipid_enrichment(cam: np.ndarray, img: np.ndarray,
                     percentile: float = LIPID_PERCENTILE) -> float:
    """Mean attribution over lipid-dense pixels / mean attribution overall.

    1.0 means the map is indifferent to droplets; > 1 means it concentrates
    there. Uses raw RI intensity as the droplet proxy, so it needs no masks.
    """
    threshold = np.percentile(img, percentile)
    dense = img >= threshold
    if not dense.any() or cam.mean() == 0:
        return np.nan
    return float(cam[dense].mean() / cam.mean())


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_attribution_panel(records: list, class_names: list, out_dir: Path,
                           stem: str, title: str, cmap: str = "inferno",
                           alpha: float = 0.72) -> None:
    """One row per condition: QPI | overlay, repeated for each example."""
    n_examples = max(len(r["examples"]) for r in records)
    n_cols = 2 * n_examples
    fig, axs = plt.subplots(
        nrows=len(records), ncols=n_cols,
        figsize=(n_cols * 1.45, len(records) * 1.55),
        squeeze=False,
    )
    for row, record in enumerate(records):
        for col in range(n_cols):
            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])
            axs[row][col].axis("off")
        for i, example in enumerate(record["examples"]):
            ax_img, ax_cam = axs[row][2 * i], axs[row][2 * i + 1]
            for ax in (ax_img, ax_cam):
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.4)
            ax_img.imshow(example["image"], cmap="gray", vmin=0, vmax=1)
            ax_cam.imshow(example["image"], cmap="gray", vmin=0, vmax=1)
            # per-pixel alpha: weak attribution stays transparent so the
            # underlying QPI morphology is still readable in print
            ax_cam.imshow(example["map"], cmap=cmap, vmin=0, vmax=1,
                          alpha=np.clip(example["map"], 0, 1) ** 1.5 * alpha)
            # thin contour at the top quartile of attribution
            ax_cam.contour(example["map"], levels=[0.75], colors="w",
                           linewidths=0.6, alpha=0.8)
            # misclassified top-ups are flagged so a row is never misread
            label = f"p={example['confidence']:.2f}"
            colour = "0.2" if example.get("correct", True) else "C3"
            if not example.get("correct", True):
                label += " (wrong)"
            ax_cam.set_title(label, fontsize=7, pad=2, color=colour)
            if row == 0:
                ax_img.set_title("QPI\n ", fontsize=8, pad=2)
                ax_cam.set_title(f"attribution\n{label}", fontsize=8, pad=2,
                                 color=colour)
            if i == 0:
                ax_img.set_ylabel(record["class_name"], fontsize=9,
                                  fontweight="bold", rotation=0,
                                  ha="right", va="center", labelpad=6)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    cax = fig.add_axes((0.90, 0.12, 0.012, 0.72))
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap),
        cax=cax, ticks=[0, 0.5, 1.0],
    )
    bar.set_label("attribution (scaled)", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    save_figure(fig, out_dir / stem)


def plot_lipid_enrichment(stats: pd.DataFrame, column: str, method: str,
                          class_names: list, out_dir: Path, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=stats, x="true_class", y=column,
                order=class_names, color="#9ecae1", fliersize=0, ax=ax)
    sns.stripplot(data=stats, x="true_class", y=column,
                  order=class_names, color="0.25", size=3, alpha=0.7, ax=ax)
    ax.axhline(1.0, color="C3", ls="--", lw=1,
               label="no enrichment over the whole field")
    ax.set_xlabel("")
    ax.set_ylabel(f"{method} in top {100 - LIPID_PERCENTILE:.0f}% RI pixels\n"
                  "(fold over field mean)", fontsize=11)
    ax.set_title(f"{method} attribution over lipid-dense pixels", fontsize=13)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(bottom=0.9)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save_figure(fig, out_dir / stem)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="a run folder written by resnet_condition_classifier")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n-per-class", type=int, default=3,
                        help="examples shown per condition in the panels")
    parser.add_argument("--ig-steps", type=int, default=IG_STEPS)
    parser.add_argument("--no-ig", action="store_true",
                        help="skip the (slower) integrated-gradients panel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_context("talk")
    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(run_dir / "resnet_condition_classifier.pt",
                            map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    size = checkpoint["target_size"]
    model = build_model(checkpoint["arch"], len(class_names), "finetune")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    for params in model.parameters():  # gradients flow, weights stay fixed
        params.requires_grad_(True)

    splits = pd.read_csv(run_dir / "image_splits.csv")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    frame = predictions[predictions["split"] == args.split].merge(
        splits[["filename", "path"]], on="filename", how="left"
    )
    prob_cols = [f"prob_{name}" for name in class_names]
    frame["confidence"] = frame[prob_cols].max(axis=1)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"{len(frame)} {args.split} images, model {checkpoint['arch']} @ {size} px")

    # Attributions for every image: the panels use a few, the enrichment
    # statistic uses all of them.
    images, cams, igs = {}, {}, {}
    for row in tqdm(list(frame.itertuples()), desc="attributions"):
        img = load_image(row.path, size)
        batch = to_tensor(img, device)
        target = class_to_idx[row.true_class]
        images[row.filename] = img
        cams[row.filename] = grad_cam(model, batch, target, size)
        if not args.no_ig:
            attribution = integrated_gradients(
                model, batch, target, float(np.percentile(img, 5)),
                steps=args.ig_steps,
            )
            ceiling = np.percentile(attribution, 99)
            igs[row.filename] = (np.clip(attribution / ceiling, 0, 1)
                                 if ceiling > 0 else attribution)

    frame["gradcam_lipid_enrichment"] = [
        lipid_enrichment(cams[f], images[f]) for f in frame["filename"]
    ]
    stat_cols = ["filename", "true_class", "predicted_class", "correct",
                 "confidence", "gradcam_lipid_enrichment"]
    print("\nmedian lipid enrichment per condition (1.0 = indifferent):")
    if igs:
        frame["ig_lipid_enrichment"] = [
            lipid_enrichment(igs[f], images[f]) for f in frame["filename"]
        ]
        stat_cols.append("ig_lipid_enrichment")
        plot_lipid_enrichment(frame, "ig_lipid_enrichment",
                              "Integrated gradients", class_names, run_dir,
                              "lipid_enrichment_ig")
    print(frame.groupby("true_class")[stat_cols[5:]].median().round(2).to_string())

    stats_path = run_dir / "attribution_stats.csv"
    stats = frame[stat_cols]
    if args.no_ig and stats_path.exists():  # keep IG values from an earlier pass
        previous = pd.read_csv(stats_path)
        if "ig_lipid_enrichment" in previous:
            stats = stats.merge(previous[["filename", "ig_lipid_enrichment"]],
                                on="filename", how="left")
    stats.to_csv(stats_path, index=False)

    # Cache the maps so figures can be re-rendered without recomputing them
    # (float16 is plenty for maps that are already scaled to [0, 1]).
    np.savez_compressed(
        run_dir / "gradcam_maps.npz",
        **{name: cam.astype(np.float16) for name, cam in cams.items()},
    )
    if igs:
        np.savez_compressed(
            run_dir / "ig_maps.npz",
            **{name: ig.astype(np.float16) for name, ig in igs.items()},
        )
    plot_lipid_enrichment(frame, "gradcam_lipid_enrichment", "Grad-CAM",
                          class_names, run_dir, "lipid_enrichment_gradcam")

    # Panels lead with the most confident correct call per condition; if a
    # condition has too few correct calls, the row is topped up with its
    # remaining images so every row stays the same width.
    picks = {}
    for class_name in class_names:
        of_class = frame[frame["true_class"] == class_name]
        correct = of_class[of_class["correct"]].nlargest(
            args.n_per_class, "confidence")
        if len(correct) < args.n_per_class:
            top_up = of_class[~of_class["correct"]].nlargest(
                args.n_per_class - len(correct), "confidence")
            correct = pd.concat([correct, top_up])
        picks[class_name] = correct

    cam_records = [
        {
            "class_name": class_name,
            "examples": [
                {
                    "image": images[row.filename],
                    "map": cams[row.filename],
                    "confidence": row.confidence,
                    "correct": bool(row.correct),
                }
                for row in picks[class_name].itertuples()
            ],
        }
        for class_name in class_names
    ]
    plot_attribution_panel(
        cam_records, class_names, run_dir, "gradcam_panel",
        "Grad-CAM: pixels driving the condition call", cmap="inferno",
        alpha=0.55,  # smooth map: stay translucent so the cell reads through
    )

    if igs:
        ig_records = [
            {
                "class_name": class_name,
                "examples": [
                    {
                        "image": images[row.filename],
                        "map": igs[row.filename],
                        "confidence": row.confidence,
                        "correct": bool(row.correct),
                    }
                    for row in picks[class_name].itertuples()
                ],
            }
            for class_name in class_names
        ]
        plot_attribution_panel(
            ig_records, class_names, run_dir, "integrated_gradients_panel",
            "Integrated gradients: pixel-level evidence", cmap="magma",
            alpha=0.85,  # sparse map: opaque where it fires, clear elsewhere
        )

    print(f"\nwrote attribution figures to {run_dir}")


if __name__ == "__main__":
    main()
