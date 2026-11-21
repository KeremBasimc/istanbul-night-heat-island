# Data manifest

Every dataset used in the project, with its source, access route and licence. Fill in the access date the first time a dataset is downloaded, and re-check the licence on the product page at that time.

| Dataset | Provider / product | Access | Licence | Access date | Used for |
|---|---|---|---|---|---|
| MODIS night LST | NASA, MOD11A1 / MYD11A1 v061 (`LST_Night_1km`) | Microsoft Planetary Computer, collection `modis-11A1-061` | NASA open data, no restrictions on reuse | 2026-11-14 | Coarse input |
| ECOSTRESS LST | NASA JPL, ECO_L2T_LSTE v002 | NASA Earthdata (LP DAAC), `earthaccess` | NASA open data, no restrictions on reuse | 2026-11-14 | Fine target and independent reference |
| Sentinel-2 L2A | ESA Copernicus | Planetary Computer, `sentinel-2-l2a` | Copernicus free and open data licence | 2026-11-14 | Surface predictors (NDVI, built-up indices) |
| ESA WorldCover | ESA, WorldCover 10 m 2021 v200 | Planetary Computer, `esa-worldcover` | CC BY 4.0 | 2026-11-14 | Land-cover fractions (trees, grass, built) |
| GHSL building height | European Commission JRC, GHS-BUILT-H R2023A, average net building height 2018, 100 m (tile R5_C21) | JRC GHSL FTP, downloaded by `nhi.features.ghsl_building_height` | CC BY 4.0 | 2026-11-21 | Building height (night heat storage, street canyons) |
| OpenStreetMap buildings | OpenStreetMap contributors | Geofabrik extract or Overpass API | ODbL | | Building density |
| Copernicus DEM GLO-30 | ESA Copernicus | Planetary Computer, `cop-dem-glo-30` | Copernicus DEM licence | 2026-11-14 | Elevation |
| Landsat 8/9 surface temperature | USGS, Collection 2 Level-2 | Planetary Computer, `landsat-c2-l2` | Public domain (USGS) | 2026-11-14 | Daytime heat-storage predictor |
| Meteosat LST (optional) | EUMETSAT LSA SAF | LSA SAF data portal (free registration) | Check product page | | Correcting the MODIS–ECOSTRESS time gap |
| Sea mask | Derived from ESA WorldCover (largest connected water body) | computed in `nhi.features.distance_to_sea` | CC BY 4.0 | 2026-11-14 | Distance to the sea and the Bosphorus |
