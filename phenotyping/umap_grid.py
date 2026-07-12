import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
def umap_grid_bin_thumbnails(
    embedding, images,
    cell_area=None,
    n_bins_x=35,
    n_bins_y=15,
    zoom=0.10,
    figsize=(20, 20),
    place_at="bin_center",          # "bin_center" or "point"
    pick="closest_to_center",       # "closest_to_center", "max_area", "median_area"
    min_area=None,               # optional: ignore tiny cells
    filepath=''
):
    """
    embedding: (N,2)
    images: list/array length N, each (H,W) or (H,W,3)
    cell_area: array length N (optional)
    """

    x = embedding[:, 0].astype(float)
    y = embedding[:, 1].astype(float)

    # Normalize to [0,1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # Bin indices
    bx = np.clip((x_norm * n_bins_x).astype(int), 0, n_bins_x - 1)
    by = np.clip((y_norm * n_bins_y).astype(int), 0, n_bins_y - 1)

    x_centers = (np.arange(n_bins_x) + 0.5) / n_bins_x
    y_centers = (np.arange(n_bins_y) + 0.5) / n_bins_y

    # Group points by bin
    bin_to_indices = {}
    for i in range(len(images)):
        if min_area is not None and cell_area is not None and cell_area[i] < min_area:
            continue
        key = (bx[i], by[i])
        bin_to_indices.setdefault(key, []).append(i)

    # Pick representative per bin
    bin_to_rep = {}
    for key, idxs in bin_to_indices.items():
        cx = x_centers[key[0]]
        cy = y_centers[key[1]]

        if pick == "closest_to_center":
            d2 = [(i, (x_norm[i]-cx)**2 + (y_norm[i]-cy)**2) for i in idxs]
            rep = min(d2, key=lambda t: t[1])[0]

        elif pick == "max_area":
            if cell_area is None:
                raise ValueError("cell_area is required for pick='max_area'")
            rep = max(idxs, key=lambda i: cell_area[i])

        elif pick == "median_area":
            if cell_area is None:
                raise ValueError("cell_area is required for pick='median_area'")
            # pick index whose area is closest to the median in that bin
            areas = np.array([cell_area[i] for i in idxs])
            med = np.median(areas)
            rep = idxs[int(np.argmin(np.abs(areas - med)))]

        else:
            raise ValueError("pick must be: closest_to_center | max_area | median_area")

        bin_to_rep[key] = rep

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")

    for key, rep_i in bin_to_rep.items():
        img = images[rep_i]
        if img is None:
            continue

        imagebox = OffsetImage(img, zoom=zoom)

        if place_at == "bin_center":
            px = x_centers[key[0]]
            py = y_centers[key[1]]
        elif place_at == "point":
            px = x_norm[rep_i]
            py = y_norm[rep_i]
        else:
            raise ValueError("place_at must be 'bin_center' or 'point'")

        ab = AnnotationBbox(imagebox, (px, py), frameon=False, pad=0)
        ax.add_artist(ab)

    plt.tight_layout()
    plt.savefig(filepath,bbox_inches='tight',dpi=300)
    #plt.show()

    selected_idx = list(bin_to_rep.values())
    return selected_idx
