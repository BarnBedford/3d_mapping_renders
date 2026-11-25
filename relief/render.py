"""The full pipeline: config -> image."""
import math
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label, maximum_filter, minimum_filter

from . import colour as C
from .clouds import build_clouds
from .data import load_dem, load_landcover, make_grid
from .lighting import NEG, lambert, sky_visibility, sun_visibility
from .textures import forest_canopy


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def __call__(self, msg):
        now = time.perf_counter()
        print(f"[{now - self.t:6.1f}s] {msg}")
        self.t = now


def detect_lakes(z, ocean, res, min_area_km2):
    """Copernicus DEM flattens lake surfaces, so large perfectly flat areas are lakes."""
    flat = (maximum_filter(z, 3) - minimum_filter(z, 3) < 0.05) & ~ocean
    lab, n = label(flat)
    if n == 0:
        return np.zeros_like(ocean)
    sizes = np.bincount(lab.ravel()) * res * res / 1e6
    keep = sizes >= min_area_km2
    keep[0] = False
    return binary_dilation(keep[lab], iterations=1) & ~ocean


def finish(lin, cfg):
    L = cfg["lighting"]
    srgb = C.to_srgb(lin * L["exposure"])
    if L["gamma"] != 1.0:
        srgb = srgb ** (1.0 / L["gamma"])
    srgb = (srgb - 0.5) * L["contrast"] + 0.5
    if L["saturation"] != 1.0:
        luma = srgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        srgb = luma[..., None] + (srgb - luma[..., None]) * L["saturation"]
    return np.clip(srgb, 0.0, 1.0)


def save_gray(path, a, lo=None, hi=None):
    lo = np.nanmin(a) if lo is None else lo
    hi = np.nanmax(a) if hi is None else hi
    Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)).save(path)


def render(cfg):
    T = Timer()
    out_path = Path(cfg["output"]["path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grid = make_grid(cfg["aoi"]["bbox"], cfg["aoi"]["crs"], cfg["output"]["width_px"])
    res = grid.res
    print(f"Grid {grid.width} x {grid.height} px, {res:.1f} m/px, {grid.crs.to_string()}")

    # --- data ---------------------------------------------------------------
    z = load_dem(grid, cfg["dem"]["source"])
    ocean = ~np.isfinite(z) | (np.nan_to_num(z, nan=-1.0) <= cfg["dem"]["sea_level_m"])
    z = np.where(ocean, 0.0, z).astype(np.float32)
    T("DEM loaded")

    lakes = (detect_lakes(z, ocean, res, cfg["water"]["min_lake_area_km2"])
             if cfg["water"]["detect_lakes"] else np.zeros_like(ocean))

    lc = None
    want_lc = cfg["landcover"]["enabled"] or cfg["forest"]["enabled"]
    if want_lc and res <= cfg["landcover"]["max_pixel_m"]:
        lc = load_landcover(grid)
        if cfg["landcover"]["use_water"]:
            lakes |= (lc == 80) & ~ocean
        T("landcover loaded")
    elif want_lc:
        print(f"  landcover/forest skipped: {res:.0f} m/px > landcover.max_pixel_m")

    # --- surface heights ------------------------------------------------------
    exag = cfg["terrain"]["z_exaggeration"]
    surf = z * exag
    forest = crowns = None
    if cfg["forest"]["enabled"] and lc is not None:
        forest = (lc == 10)
        canopy, crowns = forest_canopy(forest, res, cfg)
        surf = surf + canopy
        T("forest canopy")

    clouds = build_clouds(z, ocean, grid, cfg)
    if clouds:
        T(f"clouds built ({len(clouds)} layers)")

    # --- lighting -------------------------------------------------------------
    L, S = cfg["lighting"], cfg["sun"]
    az, alt = S["azimuth_deg"], S["altitude_deg"]
    direct = lambert(surf, res, az, alt)

    shadow_args = (res, az, alt, L["shadow_samples"], L["sun_radius_deg"], L["shadow_softness_m"])
    cc = cfg["clouds"]
    smooth_px = max(1.0, 15 / res)  # in metres, so previews match full-size renders
    tops = [gaussian_filter(cl["surface"] * exag, smooth_px) for cl in clouds]
    if cc["self_shadow"]:
        # Each layer is shadowed by the terrain and by its own billows (thin edges excluded).
        vis = sun_visibility(surf, [surf], *shadow_args)
        for cl, top in zip(clouds, tops):
            occ = np.maximum(surf, np.where(cl["alpha"] > 0.25, top, NEG))
            vis += sun_visibility(occ, [top], *shadow_args)
    else:
        vis = sun_visibility(surf, [surf] + tops, *shadow_args)
    # Floating clouds shade the terrain and every cloud layer below them.
    for i, cl in enumerate(clouds):
        for j in range(i + 1):
            vis[j] = vis[j] * (1.0 - cl["cast"])
    T(f"cast shadows ({L['shadow_samples']} samples)")

    sky = sky_visibility(surf, res, L["ambient_directions"], L["ambient_altitudes_deg"],
                         L["ambient_downsample"])
    T("ambient occlusion")

    shadow = 1.0 - L["shadow_strength"] * (1.0 - vis[0])
    light = L["ambient"] * sky + L["direct"] * direct * shadow

    # --- colour ---------------------------------------------------------------
    base = C.terrain_colour(z, res, cfg)
    base = C.apply_landcover(base, lc, cfg)
    if forest is not None and cfg["forest"]["colour_strength"]:
        w = gaussian_filter(forest.astype(np.float32), 0.7) * cfg["forest"]["colour_strength"]
        tree = C.hex_lin(cfg["forest"]["colour"])[None, None, :] * (0.8 + 0.35 * crowns)[..., None]
        base = C.lerp(base, tree, w)
    base = C.water_colour(base, ocean, lakes, res, cfg)
    T("colour")

    # Soft roll-off (shadows ~linear, flat ground ~0.85, sun-facing slopes keep separating
    # instead of clipping); whatever goes past 1 pushes toward white as a highlight.
    tone = 1.133 * light / (1.0 + 0.333 * light)
    lit = base * np.minimum(tone, 1.0)[..., None]
    lit += (1.0 - base) * (np.clip(tone - 1.0, 0, 1) * L["highlight"])[..., None]

    sun_c, shade_c = C.hex_lin(cc["sun_colour"]), C.hex_lin(cc["shade_colour"])
    for cl, top, v in zip(clouds, tops, vis[1:]):  # lowest layer first
        # Clouds scatter light: wrap the diffuse term and keep self-shadows soft and partial.
        lam = (np.minimum(lambert(gaussian_filter(top, 1.5 * smooth_px), res, az, alt), 1.4) + 0.6) / 1.6
        v = 1.0 - cc["self_shadow_strength"] * (1.0 - gaussian_filter(v, 2.0 * smooth_px))
        sunlit = cc["direct"] * lam * v
        cloud = shade_c * cc["ambient"] + sun_c * sunlit[..., None]
        if cl["colour"]:
            cloud = cloud * C.hex_lin(cl["colour"]) / sun_c
        lit = C.lerp(lit, cloud, cl["alpha"])

    rgb = finish(lit, cfg)
    img = (rgb * 255 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(out_path)
    print(f"Saved {out_path}")

    if cfg["output"]["geotiff"]:
        tif = out_path.with_suffix(".tif")
        with rasterio.open(tif, "w", driver="GTiff", width=grid.width, height=grid.height, count=3,
                           dtype="uint8", crs=grid.crs, transform=grid.transform,
                           compress="deflate", photometric="RGB") as dst:
            dst.write(np.moveaxis(img, -1, 0))
        print(f"Saved {tif}")

    if cfg["output"]["save_layers"]:
        d = out_path.with_name(out_path.stem + "_layers")
        d.mkdir(exist_ok=True)
        save_gray(d / "direct.png", direct, 0, 1.6)
        save_gray(d / "shadow.png", vis[0], 0, 1)
        save_gray(d / "ambient.png", sky, 0, 1)
        save_gray(d / "light.png", light, 0, 1.4)
        save_gray(d / "elevation.png", z)
        for i, cl in enumerate(clouds):
            save_gray(d / f"cloud{i}_{cl['name']}_alpha.png", cl["alpha"], 0, 1)
        print(f"Saved layers to {d}")
    T("done")
    return out_path
