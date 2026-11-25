# 3D shaded relief renderer

Blender-style shaded relief maps in pure Python. It handles cast shadows, ambient occlusion, forest canopy texture and clouds, and it is driven by a YAML config.

## Quick start

```bash
.venv/Scripts/python render.py configs/olympus.yaml --preview
```

- `--preview`: 1200 px wide render, takes a few seconds. Use it while tuning.
- No flag: full `output.width_px`.
- `--set key=value`: override any setting for one run. It is repeatable and takes dotted paths; list items are indexed.

```bash
.venv/Scripts/python render.py configs/olympus.yaml --preview --set terrain.z_exaggeration=3 --set clouds.layers.0.density=0.5 --set sun.altitude_deg=20
```

Outputs go to `renders/`: a PNG plus a georeferenced GeoTIFF. Set `output.save_layers: true` to also dump the shadow, ambient and cloud layers.

## Config

Every option, with comments, is in [relief/defaults.yaml](relief/defaults.yaml). Your config only needs the keys you change.

| Want to change | Key |
|---|---|
| Area | `aoi.bbox` (lon/lat), `aoi.crs` |
| Z amplification | `terrain.z_exaggeration` |
| Light direction / shadow length | `sun.azimuth_deg`, `sun.altitude_deg` |
| Shadow darkness / softness | `lighting.shadow_strength`, `lighting.sun_radius_deg` |
| Colour ramp | `colour.ramp`: a preset (`aegean`, `natural`, `alpine`, `desert`, `grey`) or a list of `[elevation_m, "#hex"]` |
| Sea / lake colour | `water.ocean_colour`, `water.shallow_colour`, `water.shallow_width_m`, `water.lake_colour` |
| Landcover tints | `landcover.enabled`, `landcover.tint`, `landcover.colours` |
| Forest texture | `forest.enabled`, `tree_height_m`, `crown_m`, `density`, `colour` |
| Clouds | `clouds.layers` (see below) |

## Clouds

Clouds are a list of layers. Pick a preset from the library ([relief/cloud_presets.yaml](relief/cloud_presets.yaml)), then override any setting.

```yaml
clouds:
  enabled: true
  seed: 4                       # change to re-roll the random arrangement
  layers:
    - preset: valley_fog
      coverage: 0.45
    - preset: mountain_cap
      wind_from_deg: 240
    - preset: fair_weather_cumulus
      coverage: 0.07
      bbox: [-5.3, 55.5, -5.0, 55.7]   # optional: only in this lon/lat box
```

To see every preset on your area as one contact sheet (`renders/<name>_cloud_gallery.png`):

```bash
.venv/Scripts/python cloud_gallery.py configs/arran.yaml
```

| Preset | Type | Look |
|---|---|---|
| `faint_mist` | valley_fog | See-through haze in valley bottoms |
| `valley_fog` | valley_fog | Thicker fog filling glens |
| `morning_inversion` | blanket | Sea of cloud, hills poking through |
| `thin_blanket` | blanket | Translucent stratus with soft holes |
| `fair_weather_cumulus` | cumulus | Small scattered puffs with offset shadows |
| `big_cumulus` | cumulus | Large towering clouds |
| `mountain_cap` | orographic | Cloud sitting on summits |
| `banner_cloud` | orographic | Cloud streaming downwind of peaks |
| `cirrus` | cirrus | High, thin streaks along the wind |

- **Altitudes adapt to the area.** `top_m`, `base_m`, `min_ground_m` and `altitude_m` take metres or a land-elevation percentile: `"p35"`, or `"p90+100"` for 100 m above it.
- **Shadows.** Floating clouds (cumulus, cirrus, caps) cast shadows offset away from the sun. Fog and blankets receive terrain shadows but cast none.
- **Main knobs.** `density` = opacity. `coverage` = how much of the eligible area. `wisp` = see-through texture. `warp` = swirled edges. `softness` = edge softness.
- **Stable seeds.** Each layer's randomness comes from `clouds.seed` plus its preset name, so adding a layer doesn't reshuffle the others. Set `seed:` on one layer to re-roll just that layer.

## Data

| Data | Source | When |
|---|---|---|
| DEM | Copernicus GLO-30 / GLO-90 from AWS, no key needed | GLO-30 below 60 m/px, GLO-90 above |
| Landcover | ESA WorldCover 2021 (10 m) | Only when landcover/forest is on and pixels are ≤ `landcover.max_pixel_m` |
| Your own DEM | Any GeoTIFF | Set `dem.source: path/to/dem.tif` |

Downloads and reprojected grids are cached in `cache/`, so re-renders of the same area/size skip the fetch.

## How it works

1. Reproject the DEM to the output grid. Missing tiles become sea, and flat areas become lakes.
2. Build the surface heights: DEM × exaggeration, plus forest canopy bumps (so forest edges cast shadows).
3. Compute cast shadows with a horizon sweep, averaged over jittered sun positions for soft edges ([relief/lighting.py](relief/lighting.py)).
4. Compute ambient occlusion with the same sweep from many sky directions.
5. Colour (ramp → rock on steep slopes → landcover → forest → water), all in linear RGB.
6. Light = ambient × occlusion + direct × Lambert × shadow, with sunlit slopes highlighted.
7. Composite cloud layers lowest first, then apply exposure, contrast and saturation.
