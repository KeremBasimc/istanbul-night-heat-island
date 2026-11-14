"""Data access for the Istanbul nighttime heat island project.

ECOSTRESS comes from NASA Earthdata (login required), MODIS from Microsoft
Planetary Computer (no login).
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

ECO_SHORT_NAME = "ECO_L2T_LSTE"
ECO_VERSION = "002"
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MODIS_COLLECTION = "modis-11A1-061"
MODIS_LST_SCALE = 0.02  # MOD11A1 LST is stored as uint16 * 0.02 K


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


def _open_clipped(path: Path, bbox=AOI_BBOX):
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr

    da = xr.open_dataarray(path, engine="rasterio").squeeze("band", drop=True)
    return da.rio.clip_box(*bbox, crs="EPSG:4326")


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


def load_ecostress_lst(lst_path: Path, bbox=AOI_BBOX):
    """ECOSTRESS LST in degrees Celsius, clipped to the AOI (70 m grid)."""
    da = _open_clipped(lst_path, bbox)
    da = da.where(np.isfinite(da) & (da > 0))
    return (da - 273.15).rename("lst_c")


def load_modis_night(night_date: dt.date, platform: str = "aqua", bbox=AOI_BBOX,
                     crs: str = UTM_CRS, resolution: float = 1000.0):
    """MODIS night LST (degrees Celsius) for the night that starts on `night_date`.

    The MODIS daily file is organised by UTC day. The Terra night pass (about
    22:30 local) and the Aqua night pass (about 01:30 local = 22:30 UTC) of the
    night starting on `night_date` both fall in that UTC day's file.
    Pixels without a valid retrieval (clouds) are NaN.
    """
    import odc.stac
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    day = night_date.isoformat()
    items = catalog.search(collections=[MODIS_COLLECTION], bbox=bbox,
                           datetime=f"{day}T00:00:00Z/{day}T23:59:59Z").item_collection()
    items = [it for it in items if it.properties.get("platform", "").lower() == platform]
    if not items:
        return None
    ds = odc.stac.load(items, bands=["LST_Night_1km"], bbox=bbox, crs=crs,
                       resolution=resolution, groupby="solar_day")
    raw = ds["LST_Night_1km"].isel(time=0).astype("float32")
    lst = raw.where(raw > 0) * MODIS_LST_SCALE - 273.15
    return lst.rename("lst_c")
