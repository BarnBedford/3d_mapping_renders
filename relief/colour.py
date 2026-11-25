"""Base colours: elevation ramp, rock, landcover, water. All maths in linear RGB."""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .noise import fbm, smoothstep

# [elevation_m, colour] stops. Add your own here or pass a list in the config.
PRESETS = {
    # Close to the attached Visual Geomatics look: tan lowlands, forest-green uplands.
    "aegean": [
        [0, "#b4b98a"], [120, "#c8c196"], [400, "#a9ad78"], [800, "#6f9150"],
        [1300, "#4a7a3e"], [1900, "#5f7752"], [2400, "#a39d8c"], [2800, "#f2f2ee"],
    ],
    "natural": [
        [0, "#8fae78"], [300, "#b9c48e"], [700, "#d8cf9c"], [1200, "#c2a477"],
        [1800, "#a08466"], [2500, "#b7b0a8"], [3200, "#f4f4f2"],
    ],
    "alpine": [
        [0, "#7da36a"], [600, "#9fbb7c"], [1200, "#6d8f55"], [1800, "#8c8a6c"],
        [2300, "#a39e94"], [2800, "#e6e6e4"], [3300, "#ffffff"],
    ],
    "desert": [
        [0, "#e3cfa2"], [400, "#d8b98a"], [900, "#c49a6c"], [1500, "#a7795a"],
        [2200, "#8d6a55"], [3000, "#d8d0c8"],
    ],
    # Scottish hills: green pasture, brown heather moorland, grey granite tops.
    "highland": [
        [0, "#7f9a5f"], [60, "#8fa166"], [180, "#8e9461"], [320, "#857c58"],
        [550, "#7d7260"], [750, "#958d80"], [900, "#b4aea4"],
    ],
    "grey": [[0, "#d0d0d0"], [3000, "#d0d0d0"]],
}


def to_linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def to_srgb(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def hex_lin(h):
    h = h.lstrip("#")
    return to_linear(np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32))


def lerp(a, b, t):
    t = t[..., None] if np.ndim(t) == 2 else t
    return a + (b - a) * t


def ramp(z, stops):
    if isinstance(stops, str):
        stops = PRESETS[stops]
    stops = sorted(stops, key=lambda s: s[0])
    xs = [s[0] for s in stops]
    cols = np.array([hex_lin(s[1]) for s in stops])
    return np.stack([np.interp(z, xs, cols[:, c]) for c in range(3)], -1).astype(np.float32)


def terrain_colour(z, res, cfg, seed=11):
    c = cfg["colour"]
    rgb = ramp(z, c["ramp"])
    gx, gy = np.gradient(z, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    s0, s1 = c["rock_slope_deg"]
    rgb = lerp(rgb, hex_lin(c["rock_colour"]), smoothstep(s0, s1, slope) * c["rock_strength"])
    if c["variation"]:
        n = fbm(z.shape, c["variation_scale_m"] / res, seed)
        rgb *= (1.0 + c["variation"] * np.clip(n, -2.5, 2.5))[..., None]
    return rgb


def apply_landcover(rgb, lc, cfg, seed=12):
    lcc = cfg["landcover"]
    if lc is None or not lcc["tint"]:
        return rgb
    n = fbm(lc.shape, 6.0, seed)  # fine-grain variation inside each class
    for code, hexc in lcc["colours"].items():
        m = lc == int(code)
        if not m.any():
            continue
        w = gaussian_filter(m.astype(np.float32), 0.7) * lcc["tint"]
        col = hex_lin(hexc)[None, None, :] * (1.0 + 0.06 * n)[..., None]
        rgb = lerp(rgb, col, w)
    return rgb


def water_colour(rgb, ocean, lakes, res, cfg, seed=13):
    w = cfg["water"]
    shape = ocean.shape
    if ocean.any():
        dist = distance_transform_edt(ocean) * res  # metres to nearest land
        t = np.exp(-dist / max(w["shallow_width_m"], 1.0))
        sea = lerp(hex_lin(w["ocean_colour"])[None, None, :] * np.ones(shape + (1,), np.float32),
                   hex_lin(w["shallow_colour"]), t)
    else:
        sea = np.zeros(shape + (3,), np.float32)
    lake = hex_lin(w["lake_colour"])[None, None, :] * np.ones(shape + (1,), np.float32)
    water = np.where(lakes[..., None], lake, sea)
    if w["texture"]:
        n = fbm(shape, w["texture_scale_m"] / res, seed) + 0.4 * fbm(shape, w["texture_scale_m"] / res / 8, seed + 1)
        water *= (1.0 + w["texture"] * np.clip(n, -2.5, 2.5))[..., None]
    m = gaussian_filter((ocean | lakes).astype(np.float32), 0.6)
    return lerp(rgb, water, m)
