"""Data access for the Istanbul nighttime heat island project.

ECOSTRESS comes from NASA Earthdata (login required); MODIS and Sentinel-2
come from Microsoft Planetary Computer (no login).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# Central Istanbul (European and Asian core), lon/lat.
AOI_BBOX = (28.6, 40.95, 29.3, 41.2)
# Istanbul has no daylight saving time since 2016: local time = UTC + 3 h.
UTC_OFFSET_H = 3
# Local hours treated as "night" (21:00-05:00).
NIGHT_START_H, NIGHT_END_H = 21, 5
# UTM zone 35N covers the whole AOI; ECOSTRESS tiles (e.g. 35TPF) use it too.
UTM_CRS = "EPSG:32635"
# The coarse grid is 14 x 14 fine pixels, so the two grids nest exactly.
FINE_RES = 70.0
COARSE_RES = 980.0
FACTOR = int(COARSE_RES // FINE_RES)

ECO_SHORT_NAME = "ECO_L2T_LSTE"
ECO_VERSION = "002"
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MODIS_COLLECTION = "modis-11A1-061"
MODIS_LST_SCALE = 0.02    # MOD11A1 LST is stored as uint16 * 0.02 K
MODIS_TIME_SCALE = 0.1    # view time is stored as uint8 * 0.1 h (local solar time)
MODIS_PLATFORMS = ("terra", "aqua")


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------

def grids(bbox=AOI_BBOX, crs=UTM_CRS):
    """Coarse (980 m) and fine (70 m) GeoBoxes over the AOI that nest exactly."""
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import BoundingBox

    utm = BoundingBox(*bbox, crs="EPSG:4326").to_crs(crs)
    coarse = GeoBox.from_bbox(utm, resolution=COARSE_RES)
    fine = GeoBox.from_bbox(coarse.boundingbox, resolution=FINE_RES)
    return coarse, fine


def _stac_client():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)


# --------------------------------------------------------------------------
# ECOSTRESS
# --------------------------------------------------------------------------

def _is_night(local_time: pd.Timestamp) -> bool:
    return local_time.hour >= NIGHT_START_H or local_time.hour < NIGHT_END_H


def search_ecostress(start: str, end: str, bbox=AOI_BBOX) -> pd.DataFrame:
    """List ECOSTRESS L2T LSTE granules over the AOI, one row per granule.

    `night_date` is the calendar date of the evening the night belongs to,
    so an acquisition at 01:30 local time on 5 July belongs to the night of 4 July.
    """
    import earthaccess

    results = earthaccess.search_data(
        short_name=ECO_SHORT_NAME,
        version=ECO_VERSION,
        bounding_box=bbox,
        temporal=(start, end),
    )
    rows = []
    for granule in results:
        umm = granule["umm"]
        start_utc = pd.Timestamp(umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
        local = start_utc.tz_convert(None) + pd.Timedelta(hours=UTC_OFFSET_H)
        night_date = (local - pd.Timedelta(hours=12)).date()
        # Each layer is a separate COG named ..._<layer>.tif
        links = {link.rsplit("_", 1)[-1].removesuffix(".tif"): link
                 for link in granule.data_links()
                 if link.endswith(".tif")}
        rows.append({
            "granule": umm["GranuleUR"],
            "tile": umm["GranuleUR"].split("_")[5],
            "utc": start_utc.tz_convert(None),
            "local": local,
            "is_night": _is_night(local),
            "night_date": night_date,
            "url_LST": links.get("LST"),
            "url_cloud": links.get("cloud"),
            "url_water": links.get("water"),
        })
    df = pd.DataFrame(rows)
    # CMR can return the same granule twice; keep one row per granule.
    df = df.drop_duplicates("granule")
    return df.sort_values("utc").reset_index(drop=True)


def download(urls: list[str], folder: Path) -> list[Path]:
    """Download files with earthaccess, skipping ones already in `folder`."""
    import earthaccess

    folder.mkdir(parents=True, exist_ok=True)
    missing = [u for u in urls if not (folder / u.rsplit("/", 1)[-1]).exists()]
    if missing:
        earthaccess.download(missing, local_path=str(folder))
    return [folder / u.rsplit("/", 1)[-1] for u in urls]


def _open(path: Path):
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr

    return xr.open_dataarray(path, engine="rasterio").squeeze("band", drop=True)


def _open_clipped(path: Path, bbox=AOI_BBOX):
    return _open(path).rio.clip_box(*bbox, crs="EPSG:4326")


def clear_land_pixels(lst_path: Path, cloud_path: Path, water_path: Path, bbox=AOI_BBOX) -> int:
    """Number of AOI land pixels with a valid, cloud-free LST value.

    The cloud and water layers are 0 outside the swath as well, so a pixel is
    only counted when the LST layer also holds a valid value there.
    """
    lst = _open_clipped(lst_path, bbox).values
    cloud = _open_clipped(cloud_path, bbox).values
    water = _open_clipped(water_path, bbox).values
    valid = np.isfinite(lst) & (lst > 0)
    return int((valid & (cloud == 0) & (water == 0)).sum())


def add_clear_fraction(granules: pd.DataFrame, folder: Path, bbox=AOI_BBOX) -> pd.DataFrame:
    """Add `clear_pixels` and `clear_frac` columns.

    `clear_frac` is relative to the best-covered granule, which stands in for
    "all land pixels of the AOI observed under clear sky".
    """
    lst_paths = download(granules["url_LST"].tolist(), folder)
    cloud_paths = download(granules["url_cloud"].tolist(), folder)
    water_paths = download(granules["url_water"].tolist(), folder)
    out = granules.copy()
    out["clear_pixels"] = [clear_land_pixels(l, c, w, bbox)
                           for l, c, w in zip(lst_paths, cloud_paths, water_paths)]
    out["clear_frac"] = out["clear_pixels"] / out["clear_pixels"].max()
    return out


def best_granule_per_night(granules: pd.DataFrame) -> pd.DataFrame:
    """One row per night: the granule with the most clear land pixels."""
    idx = granules.groupby("night_date")["clear_pixels"].idxmax()
    return granules.loc[idx].sort_values("night_date").reset_index(drop=True)


def load_ecostress_lst(lst_path: Path, bbox=AOI_BBOX):
    """ECOSTRESS LST in degrees Celsius, clipped to the AOI (native 70 m grid)."""
    da = _open_clipped(lst_path, bbox)
    da = da.where(np.isfinite(da) & (da > 0))
    return (da - 273.15).rename("lst_c")


def ecostress_on_grid(lst_path: Path, cloud_path: Path, water_path: Path, geobox):
    """Cloud- and water-masked ECOSTRESS LST (degrees C) resampled onto `geobox`."""
    from rasterio.enums import Resampling

    lst = _open(lst_path)
    bad = (~np.isfinite(lst)) | (lst <= 0) | (_open(cloud_path) != 0) | (_open(water_path) != 0)
    lst = (lst.where(~bad) - 273.15).astype("float32").rio.write_nodata(np.nan)
    out = lst.rio.reproject(geobox.crs.to_wkt(), shape=geobox.shape.yx,
                            transform=geobox.transform, resampling=Resampling.average)
    return out.rename("lst_c")


# --------------------------------------------------------------------------
# MODIS
# --------------------------------------------------------------------------

def _modis_platform(item) -> str:
    """'terra' or 'aqua'. Some catalogue items leave `platform` empty, so fall back
    to the product name (MOD11A1 = Terra, MYD11A1 = Aqua)."""
    platform = (item.properties.get("platform") or "").lower()
    if platform:
        return platform
    return {"MOD": "terra", "MYD": "aqua"}.get(item.id[:3].upper(), "")


def load_modis_night_passes(night_date: dt.date, geobox, bbox=AOI_BBOX) -> dict:
    """Terra and Aqua night LST for the night that starts on `night_date`.

    The MODIS daily file is organised by UTC day. Both night passes of the night
    that starts on `night_date` fall in that UTC day's file.
    Returns {platform: {"lst": DataArray in degrees C, "clock_h": local clock
    hour of the pass, "valid_frac": share of grid pixels with a value}}.
    """
    import odc.stac

    day = night_date.isoformat()
    items = _stac_client().search(collections=[MODIS_COLLECTION], bbox=bbox,
                                  datetime=f"{day}T00:00:00Z/{day}T23:59:59Z").item_collection()
    lon_center = (bbox[0] + bbox[2]) / 2
    out = {}
    for platform in MODIS_PLATFORMS:
        its = [it for it in items if _modis_platform(it) == platform]
        if not its:
            continue
        ds = odc.stac.load(its, bands=["LST_Night_1km", "Night_view_time"],
                           geobox=geobox, groupby="solar_day", resampling="nearest")
        raw = ds["LST_Night_1km"].isel(time=0).astype("float32")
        lst = (raw.where(raw > 0) * MODIS_LST_SCALE - 273.15).rename("lst_c")
        vt = ds["Night_view_time"].isel(time=0).astype("float32")
        solar_h = vt.where(vt < 255) * MODIS_TIME_SCALE
        if not np.isfinite(solar_h.values).any():   # no retrieval over the AOI
            continue
        # local solar time -> local clock time (UTC+3)
        clock = (float(np.nanmedian(solar_h.values)) + UTC_OFFSET_H - lon_center / 15) % 24
        out[platform] = {"lst": lst, "clock_h": clock,
                         "valid_frac": float(np.isfinite(lst.values).mean())}
    return out


def hours_apart(h1: float, h2: float) -> float:
    """Absolute difference between two clock hours, across midnight."""
    d = (h1 - h2) % 24
    return min(d, 24 - d)


def closest_modis_pass(passes: dict, eco_local: pd.Timestamp, min_valid: float = 0.2):
    """Pick the MODIS pass closest in time to the ECOSTRESS acquisition.

    Passes with fewer than `min_valid` valid pixels are skipped when possible.
    Returns (platform, pass dict, gap in hours) or None.
    """
    if not passes:
        return None
    eco_h = eco_local.hour + eco_local.minute / 60
    usable = {k: v for k, v in passes.items() if v["valid_frac"] >= min_valid} or passes
    platform = min(usable, key=lambda k: hours_apart(usable[k]["clock_h"], eco_h))
    return platform, usable[platform], hours_apart(usable[platform]["clock_h"], eco_h)


# --------------------------------------------------------------------------
# Sentinel-2
# --------------------------------------------------------------------------

S2_CLEAR_SCL = (4, 5, 6, 7, 11)  # vegetation, bare, water, unclassified, snow


def sentinel2_ndvi(year: int, geobox, bbox=AOI_BBOX, months=("06-01", "08-31"),
                   max_cloud: float = 20.0):
    """Median summer NDVI from Sentinel-2 L2A on `geobox` (cloud-masked with SCL)."""
    import odc.stac

    items = _stac_client().search(
        collections=["sentinel-2-l2a"], bbox=bbox,
        datetime=f"{year}-{months[0]}/{year}-{months[1]}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()
    ds = odc.stac.load(items, bands=["B04", "B08", "SCL"], geobox=geobox,
                       groupby="solar_day", resampling={"*": "average", "SCL": "mode"})
    # Processing baseline 04.00 (from 25 Jan 2022) adds an offset of 1000 to L2A digital numbers.
    offset = {pd.Timestamp(it.datetime).tz_convert(None).normalize():
              1000.0 if float(it.properties.get("s2:processing_baseline", "0")) >= 4 else 0.0
              for it in items}
    off = xr_offsets(ds.time.values, offset)
    red = (ds["B04"].astype("float32") - off) / 10000
    nir = (ds["B08"].astype("float32") - off) / 10000
    ok = ds["SCL"].isin(list(S2_CLEAR_SCL)) & (red > 0) & (nir > 0)
    ndvi = ((nir - red) / (nir + red)).where(ok)
    return ndvi.median("time").rename("ndvi")


def xr_offsets(times, offset_by_day: dict):
    """DataArray of per-time digital-number offsets, aligned with `times`."""
    import xarray as xr

    vals = [offset_by_day.get(pd.Timestamp(t).normalize(), 0.0) for t in times]
    return xr.DataArray(np.array(vals, dtype="float32"), dims=["time"], coords={"time": times})
