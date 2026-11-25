"""Surface micro-relief textures added to the height grid before lighting."""
import numpy as np
from scipy.ndimage import gaussian_filter


def forest_canopy(forest, res, cfg, seed=21):
    """Canopy height (m) over forest: a raised block (edges cast shadows) plus individual crowns.

    Crowns larger than ~1.5 px are drawn as Gaussian bumps; smaller crowns become
    a roughness texture so the canopy still reads as 'trees' at coarser scales.
    """
    f = cfg["forest"]
    rng = np.random.default_rng(seed)
    mask = gaussian_filter(forest.astype(np.float32), 0.8)
    crown_px = f["crown_m"] / res
    if crown_px >= 1.5:
        sigma = crown_px / 2.5
        p = f["density"] * 1.6 / (crown_px * crown_px)  # tree centres per pixel
        pts = (rng.random(forest.shape) < p).astype(np.float32)
        crowns = gaussian_filter(pts, sigma) * (2 * np.pi * sigma * sigma)  # one tree peaks at 1
        crowns = np.clip(crowns, 0.0, 1.3) / 1.3
    else:
        crowns = gaussian_filter(rng.random(forest.shape).astype(np.float32), 0.6)
        crowns = (crowns - crowns.min()) / max(np.ptp(crowns), 1e-6)
    canopy = 0.55 + 0.45 * crowns
    return (f["tree_height_m"] * mask * canopy).astype(np.float32), crowns
