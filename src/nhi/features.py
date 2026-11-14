"""Predictor layers on the 70 m grid for the night-time downscaling model.

Every function returns numpy arrays with the fine grid's shape (see
`nhi.data.grids`). Static layers come from 10-30 m products aggregated to 70 m.
"""

from __future__ import annotations

import numpy as np

from .data import AOI_BBOX, FINE_RES, _stac_client
from .downscale import block_mean

WORLDCOVER_CLASSES = {
    "tree": 10, "shrub": 20, "grass": 30, "crop": 40,
    "built": 50, "bare": 60, "water": 80, "wetland": 90,
}
LANDSAT_ST_SCALE, LANDSAT_ST_OFFSET = 0.00341802, 149.0   # Collection 2 ST_B10 -> K
# QA_PIXEL bits: 1 dilated cloud, 3 cloud, 4 cloud shadow
LANDSAT_QA_BAD = (1 << 1) | (1 << 3) | (1 << 4)


def _sub_geobox(fine, resolution: float):
    """A finer GeoBox that nests inside `fine` (resolution must divide 70 m)."""
    from odc.geo.geobox import GeoBox

    return GeoBox.from_bbox(fine.boundingbox, resolution=resolution)


def worldcover_fractions(fine, bbox=AOI_BBOX) -> dict:
    """Share of each ESA WorldCover 2021 class inside every 70 m pixel."""
    import odc.stac

    gb10 = _sub_geobox(fine, 10.0)
    items = [it for it in _stac_client().search(collections=["esa-worldcover"], bbox=bbox).items()
             if "2021" in it.id]
    ds = odc.stac.load(items, bands=["map"], geobox=gb10, resampling="nearest")
    classes = ds["map"].max("time").values
    factor = int(FINE_RES // 10)
    return {name: block_mean((classes == code).astype("float32"), factor)
            for name, code in WORLDCOVER_CLASSES.items()}


def distance_to_sea(water_frac: np.ndarray, resolution: float = FINE_RES) -> np.ndarray:
    """Distance (m) from each pixel to the sea.

    The sea is the largest connected water body (Marmara Sea, Bosphorus and
    Golden Horn); lakes and reservoirs are ignored.
    """
    from scipy import ndimage

    water = water_frac > 0.5
    labels, n = ndimage.label(water)
    if n == 0:
        return np.full(water.shape, np.nan, dtype="float32")
    sizes = ndimage.sum(water, labels, index=range(1, n + 1))
    sea = labels == (int(np.argmax(sizes)) + 1)
    return (ndimage.distance_transform_edt(~sea) * resolution).astype("float32")


def elevation(fine, bbox=AOI_BBOX) -> np.ndarray:
    """Copernicus DEM GLO-30 elevation (m), averaged onto the 70 m grid."""
    import odc.stac

    items = list(_stac_client().search(collections=["cop-dem-glo-30"], bbox=bbox).items())
    ds = odc.stac.load(items, bands=["data"], geobox=fine, resampling="average")
    return ds["data"].max("time").values.astype("float32")


def sentinel2_indices(year: int, fine, bbox=AOI_BBOX, months=("06-01", "08-31"),
                      max_cloud: float = 20.0) -> dict:
    """Median summer NDVI and NDBI from Sentinel-2 L2A (cloud-masked with SCL)."""
    import odc.stac
    import pandas as pd

    from .data import S2_CLEAR_SCL, xr_offsets

    items = _stac_client().search(
        collections=["sentinel-2-l2a"], bbox=bbox,
        datetime=f"{year}-{months[0]}/{year}-{months[1]}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()
    ds = odc.stac.load(items, bands=["B04", "B08", "B11", "SCL"], geobox=fine,
                       groupby="solar_day", resampling={"*": "average", "SCL": "mode"})
    offset = {pd.Timestamp(it.datetime).tz_convert(None).normalize():
              1000.0 if float(it.properties.get("s2:processing_baseline", "0")) >= 4 else 0.0
              for it in items}
    off = xr_offsets(ds.time.values, offset)
    red, nir, swir = [(ds[b].astype("float32") - off) / 10000 for b in ("B04", "B08", "B11")]
    ok = ds["SCL"].isin(list(S2_CLEAR_SCL)) & (red > 0) & (nir > 0) & (swir > 0)
    ndvi = ((nir - red) / (nir + red)).where(ok).median("time")
    ndbi = ((swir - nir) / (swir + nir)).where(ok).median("time")
    return {"ndvi": ndvi.values.astype("float32"), "ndbi": ndbi.values.astype("float32")}


def landsat_day_lst(year: int, fine, bbox=AOI_BBOX, months=("06-01", "08-31"),
                    max_cloud: float = 30.0) -> np.ndarray:
    """Median summer daytime LST (degrees C) from Landsat 8/9, a heat-storage proxy."""
    import odc.stac

    items = [it for it in _stac_client().search(
        collections=["landsat-c2-l2"], bbox=bbox,
        datetime=f"{year}-{months[0]}/{year}-{months[1]}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).items() if it.properties.get("platform") in ("landsat-8", "landsat-9")]
    ds = odc.stac.load(items, bands=["lwir11", "qa_pixel"], geobox=fine, groupby="solar_day",
                       resampling={"lwir11": "average", "qa_pixel": "nearest"})
    dn = ds["lwir11"].astype("float32")
    bad = (ds["qa_pixel"].astype("uint16") & LANDSAT_QA_BAD) != 0
    lst = (dn * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET - 273.15).where((dn > 0) & ~bad)
    return lst.median("time").values.astype("float32")
