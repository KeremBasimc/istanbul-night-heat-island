# Data manifest

Every dataset used in the project, with its source, access route and licence. Fill in the access date the first time a dataset is downloaded, and re-check the licence on the product page at that time.

| Dataset | Provider / product | Access | Licence | Access date | Used for |
|---|---|---|---|---|---|
| MODIS night LST | NASA, MOD11A1 / MYD11A1 v061 (`LST_Night_1km`) | Microsoft Planetary Computer, collection `modis-11A1-061` | NASA open data, no restrictions on reuse | | Coarse input |
| ECOSTRESS LST | NASA JPL, ECO_L2T_LSTE v002 | NASA Earthdata (LP DAAC), `earthaccess` | NASA open data, no restrictions on reuse | | Fine target and independent reference |
| Sentinel-2 L2A | ESA Copernicus | Planetary Computer, `sentinel-2-l2a` | Copernicus free and open data licence | | Surface predictors (NDVI, built-up indices) |
| ESA WorldCover | ESA, WorldCover 10 m 2021 v200 | ESA / Planetary Computer | CC BY 4.0 | | Land-cover fractions (trees, grass, built) |
| GHSL | European Commission JRC, built-up surface and height | JRC GHSL data portal | Check product page | | Built density and building height |
| OpenStreetMap buildings | OpenStreetMap contributors | Geofabrik extract or Overpass API | ODbL | | Building density |
| Copernicus DEM GLO-30 | ESA Copernicus | Planetary Computer, `cop-dem-glo-30` | Copernicus DEM licence | | Elevation |
| Landsat 8/9 surface temperature | USGS, Collection 2 Level-2 | Planetary Computer, `landsat-c2-l2` | Public domain (USGS) | | Daytime heat-storage predictor |
| Meteosat LST (optional) | EUMETSAT LSA SAF | LSA SAF data portal (free registration) | Check product page | | Correcting the MODIS–ECOSTRESS time gap |
| Coastline | OpenStreetMap contributors | Geofabrik extract | ODbL | | Distance to the sea and the Bosphorus |
