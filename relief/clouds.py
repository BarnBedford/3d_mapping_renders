"""Cloud layers.

Each layer type builds three grids:
  alpha    opacity 0..1
  surface  cloud-top altitude (m), with billows so it catches light and self-shadows
  floating True if the cloud is detached from the ground (its shadow lands offset)

Types: valley_fog, blanket, cumulus, orographic, cirrus. Ready-made combinations of
settings live in cloud_presets.yaml ("preset: faint_mist" etc.).
"""
import math
import re
import zlib

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter, shift, uniform_filter

from .noise import billow, fbm, fbm01, smoothstep, streaks01, warp


ALTITUDE_KEYS = ("top_m", "min_ground_m", "base_m", "altitude_m")


def resolve_altitudes(L, z, ocean):
    """Allow altitudes as land-elevation percentiles: "p25" or "p90+100" (metres added)."""
    land = z[~ocean] if (~ocean).any() else z.ravel()
    out = dict(L)
    for k in ALTITUDE_KEYS:
        v = out.get(k)
        if isinstance(v, str):
            m = re.fullmatch(r"\s*p(\d+(?:\.\d+)?)\s*(?:([+-])\s*(\d+(?:\.\d+)?))?\s*", v)
            if not m:
                raise ValueError(f"{k}: {v!r} is not a number or 'pNN[+/-metres]'")
            val = float(np.percentile(land, float(m.group(1))))
            if m.group(2):
                val += float(m.group(3)) * (1 if m.group(2) == "+" else -1)
            out[k] = val
    return out


def _texture(a, grid, L, seed):
    """Internal wisps (thin, see-through patches) + swirled edges, shared by all types."""
    res, shape = grid.res, a.shape
    if L["warp"] > 0:
        scale = L["wisp_scale_m"] * 3 / res
        a = warp(a, L["warp"] * scale * 0.5, scale, seed + 50)
    if L["wisp"] > 0:
        u = fbm01(shape, L["wisp_scale_m"] / res, seed + 51)
        a = a * (1.0 - L["wisp"] + L["wisp"] * smoothstep(0.15, 0.85, u))
    return a


def _shift_m(a, metres, toward_deg, res, mode="constant"):
    """Move a grid `metres` toward a compass direction."""
    d = metres / res
    th = math.radians(toward_deg)
    return shift(a, (-d * math.cos(th), d * math.sin(th)), order=1, mode=mode, cval=0.0)


def _valley_fog(z, ocean, grid, L, seed):
    res, shape = grid.res, z.shape
    top = L["top_m"] + L["top_variation_m"] * fbm(shape, 25000 / res, seed + 1, octaves=3)
    a = 1.0 - np.exp(-np.clip(top - z, 0, None) / max(L["fade_m"], 1.0))
    region = fbm01(shape, L["region_scale_m"] / res, seed + 2, octaves=3)
    a *= smoothstep(1.0 - L["coverage"] - 0.08, 1.0 - L["coverage"] + 0.08, region)
    if L["valleys_only"]:
        win = max(int(L["valley_scale_m"] / res) | 1, 3)
        land = (~ocean).astype(np.float32)
        mean = uniform_filter(z * land, win) / np.maximum(uniform_filter(land, win), 1e-3)
        depth = mean - z  # >0 in valleys, ~0 on plains
        a *= gaussian_filter(smoothstep(0.0, L["valley_depth_m"], depth), win / 6)
    if L["min_ground_m"] is not None:
        zs = gaussian_filter(z, 1500 / res)  # smoothed, so gorges don't punch holes
        a *= smoothstep(L["min_ground_m"] - 80, L["min_ground_m"] + 80, zs)
    if L["breakup"] > 0:
        u = fbm01(shape, L["breakup_scale_m"] / res, seed + 3)
        a *= smoothstep(L["breakup"] - 0.25, L["breakup"] + 0.25, u)
    if L["coast_clearance_m"] > 0 and ocean.any():
        # keep inland: otherwise low coastal strips read as a white ring round the land
        inland = distance_transform_edt(~ocean) * res
        a *= smoothstep(0.3 * L["coast_clearance_m"], L["coast_clearance_m"], inland)
    surface = top + L["thickness_m"] * billow(shape, L["wisp_scale_m"] / res, seed + 4)
    return a, surface, None


def _blanket(z, ocean, grid, L, seed):
    """A sheet of stratus at top_m: covers everything lower, hills poke through."""
    res, shape = grid.res, z.shape
    top = L["top_m"] + L["top_variation_m"] * fbm(shape, 20000 / res, seed + 1, octaves=3)
    a = 1.0 - np.exp(-np.clip(top - z, 0, None) / max(L["fade_m"], 1.0))
    holes = fbm01(shape, L["holes_scale_m"] / res, seed + 2)
    edge = 1.0 - L["coverage"]
    a *= smoothstep(edge - L["softness"], edge + L["softness"], holes)
    surface = top + L["thickness_m"] * billow(shape, L["billow_scale_m"] / res, seed + 3)
    return a, surface, None


def _cumulus(z, ocean, grid, L, seed):
    """Floating puffy clouds: domed tops with billows, flat bases at altitude_m."""
    res, shape = grid.res, z.shape
    u = fbm01(shape, L["noise_scale_m"] / res, seed + 5, octaves=6)
    edge = 1.0 - L["coverage"]
    a = smoothstep(edge - L["softness"], edge + L["softness"], u)
    dome = np.sqrt(smoothstep(edge - L["softness"], 1.0, u))
    puffs = billow(shape, L["billow_scale_m"] / res, seed + 6)
    surface = L["altitude_m"] + L["thickness_m"] * dome * (0.75 + 0.25 * puffs)
    return a, surface, "free"


def _orographic(z, ocean, grid, L, seed):
    """Cap and banner clouds clinging to high ground, dragged downwind."""
    res, shape = grid.res, z.shape
    # Envelope of the peaks (max filter, then smooth): caps spread around summits
    # instead of being flattened away like a plain blur would.
    win = max(int(L["smooth_m"] / res), 3)
    zs = gaussian_filter(maximum_filter(z, size=win), win / 3)
    high = smoothstep(L["base_m"], L["base_m"] + L["range_m"], zs)
    downwind = (L["wind_from_deg"] + 180.0) % 360.0
    # Plume: the summit zone smeared downwind, fading with distance.
    banner = high.copy()
    steps = 8 if L["drift_m"] > 0 else 0
    for k in range(1, steps + 1):
        f = k / steps
        banner = np.maximum(banner, (1.0 - 0.7 * f) * _shift_m(high, L["drift_m"] * f, downwind, res))
    banner = gaussian_filter(banner, max(win / 6, 1.0))
    # banner = where cloud can form; noise (streaked along the wind) picks which parts have cloud
    u = (streaks01(shape, L["noise_scale_m"] / res, L["stretch"], downwind, seed + 7)
         if L["stretch"] > 1 else fbm01(shape, L["noise_scale_m"] / res, seed + 7, octaves=4))
    edge = 1.0 - L["coverage"]
    patches = smoothstep(edge - L["softness"], edge + L["softness"], u)
    a = smoothstep(0.05, 0.6, banner) * patches
    puffs = billow(shape, L["billow_scale_m"] / res, seed + 8)
    surface = zs + L["clearance_m"] + L["thickness_m"] * np.sqrt(a) * (0.75 + 0.25 * puffs)
    surface = gaussian_filter(surface, max(win / 8, 1.5))
    return a, surface, "attached"


def _cirrus(z, ocean, grid, L, seed):
    """High, thin, wind-streaked cloud."""
    res, shape = grid.res, z.shape
    downwind = (L["wind_from_deg"] + 180.0) % 360.0
    u = streaks01(shape, L["noise_scale_m"] / res, L["stretch"], downwind, seed + 9)
    edge = 1.0 - L["coverage"]
    a = smoothstep(edge - L["softness"], edge + L["softness"], u)
    surface = np.full(shape, L["altitude_m"], np.float32) + L["thickness_m"] * a
    return a, surface, "free"


BUILDERS = {"valley_fog": _valley_fog, "blanket": _blanket, "cumulus": _cumulus,
            "orographic": _orographic, "cirrus": _cirrus}


def _footprint_shadow(a, height_above_ground_m, grid, exag, az_deg, alt_deg, wrap):
    """Cloud opacity moved to where its shadow lands (away from the sun).
    wrap: free clouds wrap round the map edge, standing in for the off-map clouds
    whose shadows would fall inside it (the noise is random, so they look the same)."""
    d = height_above_ground_m * exag / math.tan(math.radians(alt_deg))
    return _shift_m(a, d, (az_deg + 180.0) % 360.0, grid.res, mode="grid-wrap" if wrap else "constant")


def build_clouds(z, ocean, grid, cfg):
    """List of layers, lowest first: dicts with alpha, surface (m), cast (opacity of this
    layer's shadow on everything below it), altitude (mean m), colour (hex or None)."""
    c = cfg["clouds"]
    if not c["enabled"]:
        return []
    exag = cfg["terrain"]["z_exaggeration"]
    layers = []
    seen = {}
    for L in c["layers"]:
        # Seed from the layer's name (+ repeat count), so adding/removing other layers
        # doesn't reshuffle this one.
        name = L.get("preset") or L["type"]
        seen[name] = seen.get(name, 0) + 1
        seed = L["seed"] if L.get("seed") is not None else (
            c["seed"] * 7919 + zlib.crc32(f"{name}{seen[name]}".encode())) % 2**31
        L = resolve_altitudes(L, z, ocean)
        a, surf, floating = BUILDERS[L["type"]](z, ocean, grid, L, seed)
        a = gaussian_filter(_texture(a, grid, L, seed), max(0.8, 12 / grid.res))
        if not L.get("over_sea", True):
            a = a * gaussian_filter((~ocean).astype(np.float32), 1.0)
        if L.get("bbox"):
            a = a * grid.bbox_mask(L["bbox"])
        a = np.clip(a * L["density"], 0, 1).astype(np.float32)
        surf = np.maximum(surf, z).astype(np.float32)
        w = a + 1e-6
        altitude = float(np.average(surf, weights=w))
        cast = np.zeros_like(a)
        if floating and c["cast_shadows"]:
            above = float(np.average(surf - z, weights=w))
            s = _footprint_shadow(a, above, grid, exag, cfg["sun"]["azimuth_deg"], cfg["sun"]["altitude_deg"],
                                  wrap=floating == "free")
            cast = gaussian_filter(s, max(3.0, 40 / grid.res)) * c["shadow_strength"]
        layers.append(dict(alpha=a, surface=surf, cast=cast, altitude=altitude,
                           colour=L.get("colour"), name=L.get("preset") or L["type"]))
    return sorted(layers, key=lambda l: l["altitude"])
