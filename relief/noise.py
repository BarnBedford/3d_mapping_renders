"""Procedural noise helpers."""
import math

import numpy as np
from scipy import sparse
from scipy.ndimage import map_coordinates, rotate
from scipy.special import ndtr


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bspline_matrix(n_out, n_in, scale):
    """Sparse (n_out x n_in) cubic B-spline upsampling weights."""
    x = np.arange(n_out) / scale
    base = np.floor(x).astype(int)
    t = x - base
    w = [(1 - t) ** 3 / 6, (3 * t**3 - 6 * t**2 + 4) / 6, (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6, t**3 / 6]
    rows = np.repeat(np.arange(n_out), 4)
    cols = np.clip(np.stack([base - 1, base, base + 1, base + 2], 1).ravel(), 0, n_in - 1)
    vals = np.stack(w, 1).ravel().astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_out, n_in))


def upsample(g, scale, shape):
    """Smoothly upsample a coarse grid g by `scale` (scalar or (rows, cols)) and crop to `shape`."""
    sr, sc = scale if isinstance(scale, tuple) else (scale, scale)
    wr = _bspline_matrix(shape[0], g.shape[0], sr)
    wc = _bspline_matrix(shape[1], g.shape[1], sc)
    return np.asarray((wc @ np.asarray(wr @ g).T).T, dtype=np.float32)


def fbm(shape, scale_px, seed, octaves=5, persistence=0.5, stretch=1.0):
    """Fractal noise, ~N(0, 1). scale_px = size of the largest features in pixels.
    stretch > 1 elongates features along columns (east-west)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32)
    amp = 1.0
    for _ in range(octaves):
        if scale_px < 1.0:
            break
        sr, sc = scale_px, scale_px * stretch
        g = rng.standard_normal((int(np.ceil(shape[0] / sr)) + 4,
                                 int(np.ceil(shape[1] / sc)) + 4)).astype(np.float32)
        out += amp * (upsample(g, (sr, sc), shape) if sr > 1.0 else g[: shape[0], : shape[1]])
        amp *= persistence
        scale_px /= 2.0
    std = out.std()
    return (out - out.mean()) / (std if std > 0 else 1.0)


def fbm01(shape, scale_px, seed, **kw):
    """Fractal noise mapped to a ~uniform [0, 1], so thresholds equal coverage fractions."""
    return ndtr(fbm(shape, scale_px, seed, **kw)).astype(np.float32)


def billow(shape, scale_px, seed, octaves=3):
    """Puffy 0..1 noise (|fbm|, inverted): rounded lumps with creases between them."""
    n = fbm(shape, scale_px, seed, octaves=octaves)
    soft_abs = np.sqrt(n * n + 0.25)  # rounded creases (plain |n| looks like crumpled foil)
    b = 1.0 - np.clip((soft_abs - 0.5) / 2.0, 0, 1)
    return b * b


def warp(field, amount_px, scale_px, seed):
    """Domain-warp a grid: swirls edges so noise shapes look less blobby."""
    if amount_px < 0.5:
        return field
    dy = fbm(field.shape, scale_px, seed, octaves=3) * amount_px
    dx = fbm(field.shape, scale_px, seed + 1, octaves=3) * amount_px
    r, c = np.indices(field.shape, dtype=np.float32)
    return map_coordinates(field, [r + dy, c + dx], order=1, mode="reflect").astype(np.float32)


def streaks01(shape, scale_px, stretch, toward_deg, seed):
    """~uniform 0..1 noise elongated along a compass direction (e.g. wind)."""
    n = int(math.ceil(math.hypot(*shape))) + 2
    f = fbm((n, n), scale_px, seed, stretch=stretch)
    # stretch runs along columns (east); rotate so it runs along toward_deg.
    f = rotate(f, 90.0 - toward_deg, reshape=False, order=1, mode="reflect")
    r0, c0 = (n - shape[0]) // 2, (n - shape[1]) // 2
    return ndtr(f[r0:r0 + shape[0], c0:c0 + shape[1]]).astype(np.float32)
