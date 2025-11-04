# 6) Plot: distinct color/marker per campaign; centroids as stars with black edge
def _nongray_colors(n: int):
    # Curated, high-contrast, non-gray colors (repeat if n > len(base))
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#4c72b0",
        "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860",
        "#da8bc3", "#8c8c00", "#00a2d3", "#a55194", "#636efa",
    ]
    if n <= len(base):
        return base[:n]
    # If more needed, cycle (colors stay non-gray)
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out

def plot_campaigns(X_2d, labels, cluster_ids, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 8), dpi=160)

    # --- Noise fixed gray ---
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx):
        plt.scatter(
            X_2d[noise_idx, 0], X_2d[noise_idx, 1],
            s=50, c="#9e9e9e", marker=".", alpha=0.7, label="Noise", linewidths=0
        )

    # --- Colors that are explicitly NOT gray ---
    colors = _nongray_colors(len(cluster_ids))

    # --- Clusters ---
    for i, c in enumerate(cluster_ids):
        idx = np.where(labels == c)[0]
        col = colors[i]
        # points
        plt.scatter(
            X_2d[idx, 0], X_2d[idx, 1],
            s=80, c=col, marker="o", alpha=0.85,
            linewidths=0.4, edgecolors="white", label=f"Campaign {c}"
        )
        # centroid (mean in 2D) with big star + black edge
        cx, cy = X_2d[idx].mean(axis=0)
        plt.scatter([cx], [cy],
            s=700, c=col, marker="*", edgecolors="black", linewidths=1.8,
            label=f"Centroid {c}", zorder=5
        )
        # label
        plt.text(cx, cy, f"C{c}", fontsize=11, weight="bold", color="black",
                 ha="center", va="center", zorder=6,
                 bbox=dict(facecolor="white", edgecolor=col, boxstyle="round,pad=0.25", alpha=0.9))

    plt.title("UMAP (2D) + HDBSCAN campaigns\nColors = campaigns (non-gray), Stars = centroids (C#)")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.grid(True, alpha=0.3)

    # de-dup legend
    handles, labels_ = plt.gca().get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels_):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    plt.legend(H, L, loc="best", fontsize=8, ncol=2, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.show()
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb

def _shade_by_cosine(base_hex: str, cos_vals: np.ndarray, vmin=0.80, vmax=1.0):
    """
    Return an array of RGB colors with the same hue as base_hex but brightness
    scaled by cosine similarity in [vmin, vmax]. Values below vmin are clipped.
    """
    base_rgb = np.array(to_rgb(base_hex))
    h, s, v = rgb_to_hsv(base_rgb.reshape(1, 1, 3))[0, 0]
    # normalize cos to 0..1 then map into brightness range
    cos = np.clip((cos_vals - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)
    V = (0.35 + 0.65 * cos)  # keep some floor brightness to remain visible
    HSV = np.stack([np.full_like(V, h), np.full_like(V, s), V], axis=1)
    RGB = hsv_to_rgb(HSV.reshape(-1, 1, 3)).reshape(-1, 3)
    return RGB


