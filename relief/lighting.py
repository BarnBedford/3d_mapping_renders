"""Direct light, cast shadows and sky occlusion on a height grid.

Heights are in (exaggerated) metres, rows run north->south, columns west->east,
square pixels of `res` metres.

Cast shadows use a horizon sweep: walking away from the sun one row at a time,
track the height H of the lowest sun ray that clears everything sun-ward.
A point lower than H is in shadow. Each row is a handful of vectorised numpy
ops, so a full sweep of a 4k x 4k grid takes well under a second.
"""
import math

import numpy as np
from scipy.ndimage import zoom

from .noise import smoothstep

NEG = np.float32(-1e9)


def sun_vector(az_deg, alt_deg):
    az, alt = math.radians(az_deg), math.radians(alt_deg)
    return (math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt))


def gradients(z, res):
    """dz/dx (east) and dz/dy (north)."""
    dzdrow, dzdcol = np.gradient(z, res)
    return dzdcol, -dzdrow


def lambert(z, res, az_deg, alt_deg):
    """cos(angle between surface normal and sun), normalised so flat ground = 1."""
    gx, gy = gradients(z, res)
    sx, sy, sz = sun_vector(az_deg, alt_deg)
    dot = (-gx * sx - gy * sy + sz) / np.sqrt(gx * gx + gy * gy + 1.0)
    return np.clip(dot / sz, 0.0, None).astype(np.float32)


def _sweep(a, k, step):
    """Horizon heights with the sun toward row 0; sun-ward neighbour of (r, c) is (r-1, c+k)."""
    n, m = a.shape
    H = np.empty_like(a)
    H[0] = NEG
    i0 = int(math.floor(k))
    f = np.float32(k - i0)
    pad = np.full(m + 3, NEG, np.float32)
    step = np.float32(step)
    for r in range(1, n):
        np.maximum(a[r - 1], H[r - 1], out=pad[1:m + 1])
        lo = pad[1 + i0:1 + i0 + m]
        hi = pad[2 + i0:2 + i0 + m]
        H[r] = lo + f * (hi - lo) - step
    return H


def horizon(occ, res, az_deg, alt_deg):
    """Height of the shadow boundary at each cell for a sun at (az, alt)."""
    az = math.radians(az_deg)
    dr, dc = -math.cos(az), math.sin(az)  # toward the sun, in (row, col)
    a = occ
    transposed = abs(dc) > abs(dr)
    if transposed:
        a, dr, dc = a.T, dc, dr
    flipped = dr > 0
    if flipped:
        a = a[::-1]
    k = dc / abs(dr)
    step = res * math.sqrt(1.0 + k * k) * math.tan(math.radians(alt_deg))
    H = _sweep(np.ascontiguousarray(a, dtype=np.float32), k, step)
    if flipped:
        H = H[::-1]
    if transposed:
        H = H.T
    return H


def _disk(n, radius):
    """n points on a disk (golden-angle spiral), as (d_az, d_alt) offsets in degrees."""
    if n <= 1:
        return [(0.0, 0.0)]
    pts = []
    for i in range(n):
        r = radius * math.sqrt((i + 0.5) / n)
        th = i * 2.399963
        pts.append((r * math.cos(th), r * math.sin(th)))
    return pts


def sun_visibility(occ, targets, res, az, alt, samples, radius_deg, softness):
    """Fraction of the (jittered) sun visible from each target surface.

    occ: heights that cast shadows. targets: list of height grids to test against
    (e.g. the terrain and a cloud top). Returns one visibility grid per target.
    """
    vis = [np.zeros(occ.shape, np.float32) for _ in targets]
    pts = _disk(samples, radius_deg)
    for daz, dalt in pts:
        H = horizon(occ, res, az + daz / max(math.cos(math.radians(alt)), 0.1), alt + dalt)
        for v, t in zip(vis, targets):
            v += 1.0 - smoothstep(0.0, softness, H - t)
    return [v / len(pts) for v in vis]


def sky_visibility(z, res, directions, altitudes, downsample=1):
    """Ambient occlusion: fraction of sampled sky directions not blocked by terrain."""
    d = max(int(downsample), 1)
    zs = z[::d, ::d] if d > 1 else z
    acc = np.zeros(zs.shape, np.float32)
    weights = 0.0
    for alt in altitudes:
        w = math.cos(math.radians(alt))  # low sky contributes more horizon blocking
        for i in range(directions):
            az = 360.0 * i / directions + 7.0
            H = horizon(zs, res * d, az, alt)
            acc += w * (1.0 - smoothstep(0.0, 5.0, H - zs))
            weights += w
    acc /= weights
    if d > 1:
        acc = zoom(acc, (z.shape[0] / acc.shape[0], z.shape[1] / acc.shape[1]), order=1)
        acc = acc[: z.shape[0], : z.shape[1]]
    return acc
