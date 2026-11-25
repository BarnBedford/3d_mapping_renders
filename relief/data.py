"""Output grid definition, DEM and landcover fetching (cached)."""
import hashlib
import json
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

CACHE = Path(__file__).resolve().parent.parent / "cache"


@dataclass
class Grid:
    crs: CRS
    transform: object
    width: int
    height: int
    res: float

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def bounds(self):
        l, t = self.transform.c, self.transform.f
        return (l, t - self.height * self.res, l + self.width * self.res, t)

    def lonlat_bounds(self):
        return transform_bounds(self.crs, "EPSG:4326", *self.bounds, densify_pts=21)

    def key(self):
        return dict(crs=self.crs.to_string(), bounds=[round(b, 3) for b in self.bounds],
                    width=self.width, height=self.height)

    def bbox_mask(self, bbox, feather_px=8):
        """Soft 0..1 mask of a lon/lat bbox on this grid."""
        from scipy.ndimage import gaussian_filter
        l, b, r, t = transform_bounds("EPSG:4326", self.crs, *bbox, densify_pts=21)
        inv = ~self.transform
        c0, r0 = inv * (l, t)
        c1, r1 = inv * (r, b)
        m = np.zeros(self.shape, np.float32)
        m[max(int(r0), 0):max(int(r1), 0), max(int(c0), 0):max(int(c1), 0)] = 1.0
        return gaussian_filter(m, feather_px) if feather_px else m


def make_grid(bbox, crs, width_px):
    lon0, lat0, lon1, lat1 = bbox
    if crs == "auto":
        zone = int((0.5 * (lon0 + lon1) + 180) // 6) + 1
        crs = f"EPSG:{(32600 if lat0 + lat1 >= 0 else 32700) + zone}"
    crs = CRS.from_user_input(crs)
    l, b, r, t = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    res = (r - l) / width_px
    height = int(round((t - b) / res))
    return Grid(crs, from_origin(l, t, res, res), width_px, height, res)


def _cache_path(name, key):
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return CACHE / f"{name}_{h}.npy"


def _cached(name, key, fn):
    p = _cache_path(name, key)
    if p.exists():
        return np.load(p)
    arr = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)
    return arr


def _download(url, dest):
    """Download url to dest; return None if the tile does not exist (e.g. open ocean)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def _copernicus_tiles(lonlat_bounds, arcsec):
    """URLs of Copernicus DEM tiles covering a lon/lat box. arcsec: 10 (GLO-30) or 30 (GLO-90)."""
    l, b, r, t = lonlat_bounds
    bucket = "copernicus-dem-30m" if arcsec == 10 else "copernicus-dem-90m"
    for lat in range(math.floor(b), math.ceil(t)):
        for lon in range(math.floor(l), math.ceil(r)):
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            name = f"Copernicus_DSM_COG_{arcsec}_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
            yield f"https://{bucket}.s3.amazonaws.com/{name}/{name}.tif", name


def _warp_into(dst, src_path, grid, resampling, src_nodata=None):
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            dst_transform=grid.transform, dst_crs=grid.crs,
            src_nodata=src_nodata if src_nodata is not None else src.nodata,
            dst_nodata=np.nan if dst.dtype.kind == "f" else 0,
            resampling=resampling, init_dest_nodata=False,
        )


def load_dem(grid, source="auto"):
    """Elevation in metres on the output grid. NaN where there is no data (open sea)."""
    if source not in ("auto", "glo30", "glo90"):
        def from_file():
            dem = np.full(grid.shape, np.nan, np.float32)
            with rasterio.open(source) as src:
                src_res = abs(src.res[0])
                if src.crs and src.crs.is_geographic:
                    lat = 0.5 * (src.bounds.bottom + src.bounds.top)
                    src_res *= 111320 * math.cos(math.radians(lat))
            _warp_into(dem, source, grid,
                       Resampling.average if grid.res > 1.5 * src_res else Resampling.bilinear)
            return dem
        return _cached("dem_file", dict(grid.key(), src=str(Path(source).resolve())), from_file)

    if source == "auto":
        source = "glo30" if grid.res < 60 else "glo90"
    arcsec, src_res = (10, 30) if source == "glo30" else (30, 90)

    def fetch():
        tiles = list(_copernicus_tiles(grid.lonlat_bounds(), arcsec))
        print(f"  DEM: {len(tiles)} {source} tiles (missing = sea)")
        with ThreadPoolExecutor(8) as ex:
            paths = list(ex.map(lambda u: _download(u[0], CACHE / "dem_tiles" / f"{u[1]}.tif"), tiles))
        dem = np.full(grid.shape, np.nan, np.float32)
        rs = Resampling.average if grid.res > 1.5 * src_res else Resampling.bilinear
        for p in paths:
            if p is not None:
                _warp_into(dem, p, grid, rs)
        return dem

    return _cached(f"dem_{source}", grid.key(), fetch)


def load_landcover(grid):
    """ESA WorldCover 2021 class codes on the output grid (0 = no data)."""
    def fetch():
        l, b, r, t = grid.lonlat_bounds()
        lc = np.zeros(grid.shape, np.uint8)
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
        with rasterio.Env(**env):
            for lat in range(math.floor(b / 3) * 3, math.ceil(t / 3) * 3, 3):
                for lon in range(math.floor(l / 3) * 3, math.ceil(r / 3) * 3, 3):
                    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
                    tile = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
                    url = ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
                           f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
                    try:
                        print(f"  landcover tile {tile}")
                        _warp_into(lc, url, grid, Resampling.mode, src_nodata=0)
                    except rasterio.errors.RasterioIOError:
                        pass  # no tile (open ocean)
        return lc

    return _cached("worldcover", grid.key(), fetch)
