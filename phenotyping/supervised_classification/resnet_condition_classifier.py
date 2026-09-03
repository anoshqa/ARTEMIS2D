"""
ResNet transfer-learning classifier for WAT QPI MIPs
---------------------------------------------------
Classifies single-field RI MIP images into the 8 culture conditions held in
`Remaining_MIP` (one folder per condition, e.g. WAT_D7_C3_MIP -> class D7_C3).

Adapted from the earlier Colab fibroblast notebook (ResNet-18 transfer
learning) with the changes this dataset needs:
  * 16-bit RI TIFFs instead of 8-bit PNGs -> min-max scaled with the same raw
    QPI window used elsewhere in this repo, then replicated to 3 channels.
  * labels come from the parent folder, not a filename prefix.
  * stratified splits instead of a plain random split, because the classes are
    imbalanced (39-70 images).
  * images are decoded once into RAM (471 x 3 x 112 x 112 float32 ~ 71 MB),
    so epochs are fast even on CPU.

Two training modes:
  linear   - freeze the backbone, train the new fc layer only (notebook default)
  finetune - additionally unfreeze layer4 with a lower learning rate

Example:
  python -m phenotyping.supervised_classification.resnet_condition_classifier \
      --mode linear --epochs 30
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
import torch
import torch.nn as nn
import torchvision
from skimage.transform import resize as sk_resize
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_ROOT = Path(
    r"C:\Users\anous\OneDrive - Johns Hopkins\2026_datanalysis\MISC_NONBC_NONFIBRO"
    r"\WAT_AYAN\Total HWAT data\All_WAT_MIP\Remaining_MIP"
)
# Results are written next to the data, not into this repo.
OUTPUT_DIR = DATA_ROOT.parent / "resnet_classification"

# QPI raw-intensity window used to min-max scale into [0, 1]
# (same constants as phenotyping/feature_extraction/extract_cp_features.py).
RAW_INTENSITY_MIN = 13300.0
RAW_INTENSITY_MAX = 14100.0

TARGET_SIZE = 112
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 42

# e.g. "...HWAT_Day7_Cond3_R1-001_RI MIP.tiff" -> replicate "R1"
REPLICATE_RE = re.compile(r"_R(\d+)-", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def folder_to_class(folder_name: str) -> str:
    """WAT_D14_C1_MIP -> D14_C1 (case-normalised, so WAT_d7_c1_MIP -> D7_C1)."""
    name = folder_name.upper()
    if name.startswith("WAT_"):
        name = name[len("WAT_"):]
    if name.endswith("_MIP"):
        name = name[: -len("_MIP")]
    return name


def discover_images(root: Path) -> pd.DataFrame:
    """One row per image: path, class name, replicate id."""
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(folder.glob("*.tif*")):
            match = REPLICATE_RE.search(path.name)
            rows.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "folder": folder.name,
                    "class_name": folder_to_class(folder.name),
                    "replicate": f"R{match.group(1)}" if match else "R?",
                }
            )
    if not rows:
        raise FileNotFoundError(f"no .tif/.tiff images found under {root}")
    return pd.DataFrame(rows)


def load_image(path: str, size: int) -> np.ndarray:
    """16-bit RI MIP -> float32 (size, size) scaled to [0, 1]."""
    img = tifffile.imread(path)
    if img.ndim == 3:  # defensive: collapse any stray z/channel axis
        img = img.max(axis=0) if img.shape[0] < img.shape[-1] else img.max(axis=-1)
    img = img.astype(np.float32)
    img = (img - RAW_INTENSITY_MIN) / (RAW_INTENSITY_MAX - RAW_INTENSITY_MIN)
    img = np.clip(img, 0.0, 1.0)
    img = sk_resize(img, (size, size), order=1, anti_aliasing=True, preserve_range=True)
    return img.astype(np.float32)


class MIPDataset(Dataset):
    """Pre-decoded MIPs kept in RAM; augmentation is applied on the tensor."""

    def __init__(self, frame: pd.DataFrame, class_to_idx: dict, size: int,
                 augment: bool = False, normalize: bool = True):
        self.labels = torch.tensor(
            [class_to_idx[c] for c in frame["class_name"]], dtype=torch.long
        )
        self.filenames = frame["filename"].tolist()
        self.augment = augment
        self.normalize = normalize
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

        images = [
            load_image(p, size)
            for p in tqdm(frame["path"].tolist(), desc="loading", leave=False)
        ]
        # (N, 3, H, W) in [0, 1] -- grayscale replicated across RGB
        self.images = torch.from_numpy(np.stack(images))[:, None].repeat(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.augment:
            # dihedral group only: microscopy fields have no canonical orientation
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[1])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img = torch.rot90(img, k, dims=[1, 2])
            img = img.contiguous()
        if self.normalize:
            img = (img - self.mean) / self.std
        return img, self.labels[idx]


def make_splits(frame: pd.DataFrame, val_frac: float, test_frac: float,
                seed: int) -> pd.DataFrame:
    """Add a 'split' column (train/val/test), stratified by class."""
    frame = frame.copy()
    frame["split"] = "train"

    idx = np.arange(len(frame))
    train_idx, hold_idx = train_test_split(
        idx, test_size=val_frac + test_frac, random_state=seed,
        stratify=frame["class_name"].to_numpy(),
    )
    val_idx, test_idx = train_test_split(
        hold_idx, test_size=test_frac / (val_frac + test_frac),
        random_state=seed,
        stratify=frame["class_name"].to_numpy()[hold_idx],
    )

    frame.iloc[val_idx, frame.columns.get_loc("split")] = "val"
    frame.iloc[test_idx, frame.columns.get_loc("split")] = "test"
    return frame


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(arch: str, n_classes: int, mode: str) -> nn.Module:
    weights = torchvision.models.get_model_weights(arch).DEFAULT
    model = torchvision.models.get_model(arch, weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    for params in model.parameters():
        params.requires_grad = False
    for params in model.fc.parameters():
        params.requires_grad = True
    if mode == "finetune":
        for params in model.layer4.parameters():
            params.requires_grad = True
    return model


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels_all, probs_all = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
    labels_all = np.concatenate(labels_all)
    probs_all = np.concatenate(probs_all)
    return labels_all, probs_all.argmax(axis=1), probs_all


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_sample_grid(dataset: MIPDataset, class_names: list, out_dir: Path,
                     n_per_class: int = 6) -> None:
    by_class = {name: [] for name in class_names}
    for i in range(len(dataset)):
        name = class_names[int(dataset.labels[i])]
        if len(by_class[name]) < n_per_class:
            by_class[name].append(dataset.images[i])
        if all(len(v) >= n_per_class for v in by_class.values()):
            break

    fig, axs = plt.subplots(
        nrows=len(class_names), ncols=n_per_class,
        figsize=(n_per_class * 1.6, len(class_names) * 1.7),
    )
    for row, name in enumerate(class_names):
        for col in range(n_per_class):
            ax = axs[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < len(by_class[name]):
                ax.imshow(by_class[name][col][0].numpy(), cmap="gray", vmin=0, vmax=1)
            if col == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold")
    fig.suptitle("Representative QPI MIPs per condition")
    fig.tight_layout()
    save_figure(fig, out_dir / "representative_images")


def plot_example_predictions(frame: pd.DataFrame, predictions: pd.DataFrame,
                             class_names: list, out_dir: Path,
                             size: int = TARGET_SIZE, n_examples: int = 6,
                             seed: int = SEED, stem: str = "example_predictions") -> None:
    """Correct (top row) and misclassified (bottom row) test images.

    Works purely from the two result tables, so it can also be regenerated
    after a run from image_splits.csv + predictions.csv.
    """
    merged = predictions.merge(frame[["filename", "path"]], on="filename", how="left")
    prob_cols = [f"prob_{name}" for name in class_names]
    merged["confidence"] = merged[prob_cols].max(axis=1)
    correct = merged["correct"].astype(bool)

    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(nrows=2, ncols=n_examples,
                            figsize=(n_examples * 2.0, 5.6))
    for row, (subset, title) in enumerate(
        [(merged[correct], "Correct"), (merged[~correct], "Misclassified")]
    ):
        # spread the picks over classes rather than over-sampling one condition
        subset = subset.sample(frac=1.0, random_state=seed).sort_values(
            "true_class", kind="stable"
        )
        take = (
            rng.choice(len(subset), size=min(n_examples, len(subset)), replace=False)
            if len(subset) > n_examples else np.arange(len(subset))
        )
        for col in range(n_examples):
            ax = axs[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col >= len(take):
                ax.axis("off")
                continue
            item = subset.iloc[int(sorted(take)[col])]
            ax.imshow(load_image(item["path"], size), cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"{item['true_class']} -> {item['predicted_class']}\n"
                f"p={item['confidence']:.2f}",
                fontsize=9, color="C2" if row == 0 else "C3",
            )
            for spine in ax.spines.values():
                spine.set_edgecolor("C2" if row == 0 else "C3")
                spine.set_linewidth(2)
            if col == 0:
                ax.set_ylabel(title, fontsize=11, fontweight="bold")
    fig.suptitle("Test-set predictions (true -> predicted)")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.30)  # keep row-2 titles off row-1 images
    save_figure(fig, out_dir / stem)


def plot_losses(losses_train: list, losses_val: list, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(losses_train, "C0", label="train")
    ax.plot(losses_val, "C1", label="val")
    ax.axhline(min(losses_val), c="C2", ls=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, out_dir / "training_loss")


def plot_confusion(cm: np.ndarray, class_names: list, title: str,
                   out_dir: Path, stem: str) -> None:
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        cbar_kws={"label": "fraction of true class"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, out_dir / stem)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                        help="results go to <out-dir>/<arch>_<mode>")
    parser.add_argument("--arch", default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--mode", default="linear", choices=["linear", "finetune"],
                        help="linear: fc only; finetune: fc + layer4")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None,
                        help="default 1e-3 (linear) / 1e-4 (finetune)")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sns.set_context("talk")

    # one sub-folder per configuration, so runs do not overwrite each other
    out_dir = Path(args.out_dir) / f"{args.arch}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    frame = discover_images(Path(args.data_root))
    class_names = sorted(frame["class_name"].unique())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"{len(frame)} images, {len(class_names)} classes -> {class_to_idx}")
    print(frame.groupby(["class_name", "replicate"]).size().to_string())

    frame = make_splits(frame, args.val_frac, args.test_frac, args.seed)
    print("\nsplit sizes:")
    print(pd.crosstab(frame["class_name"], frame["split"]).to_string())

    datasets = {
        split: MIPDataset(
            frame[frame["split"] == split], class_to_idx, args.target_size,
            augment=(split == "train" and not args.no_augment),
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"))
        for split, ds in datasets.items()
    }
    # deterministic order so per-image predictions line up with the filenames
    eval_loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        for split, ds in datasets.items()
    }
    plot_sample_grid(datasets["train"], class_names, out_dir)

    model = build_model(args.arch, len(class_names), args.mode).to(device)
    lr = args.lr if args.lr is not None else (1e-3 if args.mode == "linear" else 1e-4)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    if args.no_class_weights:
        weight = None
    else:  # inverse-frequency weights over the training split
        counts = np.bincount(datasets["train"].labels.numpy(),
                             minlength=len(class_names)).astype(np.float32)
        weight = torch.tensor(
            counts.sum() / (len(class_names) * np.clip(counts, 1, None))
        ).to(device)
    cost = nn.CrossEntropyLoss(weight=weight)

    losses_train, losses_val, bal_acc_val = [], [], []
    best_state, best_score, best_epoch = None, -np.inf, -1

    for epoch in tqdm(range(args.epochs), desc="epochs"):
        model.train()
        loss_train = 0.0
        for images, labels in loaders["train"]:
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = cost(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        losses_train.append(loss_train / len(loaders["train"]))

        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for images, labels in loaders["val"]:
                outputs = model(images.to(device))
                loss_val += cost(outputs, labels.to(device)).item()
        losses_val.append(loss_val / len(loaders["val"]))

        labels_val, preds_val, _ = predict(model, eval_loaders["val"], device)
        score = balanced_accuracy_score(labels_val, preds_val)
        bal_acc_val.append(score)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    print(f"\nbest val balanced accuracy {best_score:.3f} at epoch {best_epoch}")
    model.load_state_dict(best_state)
    plot_losses(losses_train, losses_val, out_dir)

    metrics = {
        "arch": args.arch,
        "mode": args.mode,
        "epochs": args.epochs,
        "lr": lr,
        "target_size": args.target_size,
        "class_names": class_names,
        "best_epoch": best_epoch,
        "losses_train": losses_train,
        "losses_val": losses_val,
        "balanced_accuracy_val_per_epoch": bal_acc_val,
    }

    datasets["train"].augment = False  # score the training split un-augmented
    prediction_rows = []
    for split in ("train", "val", "test"):
        labels_true, preds, probs = predict(model, eval_loaders[split], device)
        bal_acc = balanced_accuracy_score(labels_true, preds)
        metrics[f"balanced_accuracy_{split}"] = float(bal_acc)
        metrics[f"accuracy_{split}"] = float((labels_true == preds).mean())
        if split in ("val", "test"):
            report = classification_report(
                labels_true, preds, labels=range(len(class_names)),
                target_names=class_names, zero_division=0,
            )
            print(f"\n=== {split} (balanced accuracy {bal_acc:.3f}) ===\n{report}")
            metrics[f"classification_report_{split}"] = classification_report(
                labels_true, preds, labels=range(len(class_names)),
                target_names=class_names, zero_division=0, output_dict=True,
            )
            cm = confusion_matrix(labels_true, preds, labels=range(len(class_names)))
            plot_confusion(
                cm, class_names,
                f"{split.capitalize()} confusion matrix ({args.arch})",
                out_dir, f"confusion_matrix_{split}",
            )

        split_frame = pd.DataFrame(
            {
                "filename": datasets[split].filenames,
                "split": split,
                "true_class": [class_names[i] for i in labels_true],
                "predicted_class": [class_names[i] for i in preds],
                "correct": labels_true == preds,
            }
        )
        for i, name in enumerate(class_names):
            split_frame[f"prob_{name}"] = probs[:, i]
        if split == "test":
            plot_example_predictions(frame, split_frame, class_names, out_dir,
                                     args.target_size)
        prediction_rows.append(split_frame)

    pd.concat(prediction_rows).to_csv(out_dir / "predictions.csv", index=False)
    torch.save(
        {"state_dict": best_state, "class_names": class_names,
         "arch": args.arch, "target_size": args.target_size,
         "raw_intensity_range": [RAW_INTENSITY_MIN, RAW_INTENSITY_MAX]},
        out_dir / "resnet_condition_classifier.pt",
    )
    with open(out_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    frame.to_csv(out_dir / "image_splits.csv", index=False)
    print(f"\nwrote results to {out_dir}")


if __name__ == "__main__":
    main()
