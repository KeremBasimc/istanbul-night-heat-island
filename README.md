# Istanbul Nighttime Heat Island at Block Scale

Daily 1 km MODIS nighttime land surface temperature (LST) is downscaled to 70 m for Istanbul. The model is trained on sparse ECOSTRESS night observations, which also serve as the independent reference.

Term project for **EHB 466E Remote Sensing**, Istanbul Technical University, Fall 2026.

## Why

Heat waves in Istanbul were linked to 4,281 excess deaths in 2004–2017 (Çulpan et al., 2022), and warm nights shorten sleep (Minor et al., 2022). The health risk depends on where the city stays hot at night. Existing Istanbul heat maps are either daytime (Landsat) or coarse at night (MODIS, 1 km). No satellite maps nighttime temperature at block scale every night.

## Research questions

1. Can daily 1 km MODIS night LST be downscaled to 70 m with a model trained on sparse ECOSTRESS nights, and do night-specific predictors beat the NDVI-based standard method (TsHARP)?
2. Which surface properties explain night LST at block scale (built density, imperviousness, tree vs. grass cover, distance to the sea)?
3. Which neighbourhoods stayed hottest at night during the 2018–2025 heat waves, and how much do night and day hotspots overlap?

## Team and task distribution

| Member | Responsibility |
|---|---|
| Kerem Basım | All parts of the project |
| Murad Ahmadli | All parts of the project |

Both members work on all parts: data pipeline, predictor layers, benchmark, model, validation, figures and the paper. Each member commits under their own GitHub account.

## Data

All inputs are open and accessed from code. Licences and access dates are recorded in [DATA_MANIFEST.md](DATA_MANIFEST.md).

| Role | Dataset | Resolution |
|---|---|---|
| Coarse input | MODIS MOD11A1 / MYD11A1 night LST | 1 km, daily |
| Fine target and reference | ECOSTRESS ECO_L2T_LSTE v002 | 70 m, irregular |
| Predictors | Sentinel-2, ESA WorldCover, GHSL, OpenStreetMap buildings, Copernicus DEM, Landsat daytime LST | 10–100 m |
| Time-gap correction (optional) | Meteosat LST (EUMETSAT LSA SAF) | 3–5 km, 15 min |

## Method

- **Pairing:** each cloud-free ECOSTRESS night is matched with the closest MODIS night overpass (Terra about 23:00, Aqua about 02:00–04:00 local time). Days missing from Planetary Computer are read from NASA's HDF files.
- **Cloud screen:** nights where the two sensors disagree grossly at 980 m (median difference above 5 K or negative correlation) are dropped for every method.
- **Proposed two-step fusion:**
  1. *Block correction (980 m):* a random forest learns where the MODIS night pattern differs from the ECOSTRESS block means (MODIS neighbourhood pattern, coast, elevation, time gap) and corrects the MODIS block values.
  2. *70 m anomaly:* a second random forest predicts each 70 m pixel's deviation from its block mean from vegetation, built-up, building height, elevation, distance to the sea and daytime heat storage; the re-centred anomaly is added to the corrected block value.
- **Benchmarks:** plain upsampling of MODIS, fixed spatial filters of MODIS (bilinear, 3×3 and 5×5 mean, Gaussian), TsHARP (Agam et al., 2007) and a Sentinel-2-only random forest.
- **Validation:** 6-fold leave-nights-out against held-out ECOSTRESS nights, with the same folds for every method. Bias-removed RMSE, RMSE, MAE, bias (°C) and r, a paired bootstrap over nights, an ablation of predictor groups and the within-block skill at 70, 140 and 490 m.

### First results (51 clear nights, 2019–2025)

| Method               | Bias-removed RMSE (°C), mean | median | r     |
|----------------------|------------------------------|--------|-------|
| MODIS upsampled      | 1.255                        | 1.044  | 0.578 |
| Best MODIS filter (5×5 mean) | 1.245                | 1.059  | –     |
| TsHARP               | 1.314                        | 1.108  | 0.557 |
| Sentinel-2-only RF   | 1.254                        | 1.046  | 0.579 |
| **Proposed fusion**  | **1.105**                    | **0.938** | **0.627** |

The fusion lowers the bias-removed RMSE by 12 % against MODIS and 16 % against TsHARP (95 % bootstrap intervals exclude zero; better on 46 and 48 of 50 nights). Most of the night error sits between 980 m blocks; the 70 m detail inside a block is small (median SD 0.54 °C) and only partly predictable, while TsHARP's NDVI rule carries almost no skill at night.

## Repository structure

```
notebooks/   Colab notebooks, numbered in run order
src/nhi/     reusable Python code
data/        local cache, not tracked by git
figures/     figures for slides and the paper
docs/        project documents
```

## Getting started (Google Colab)

1. Open a notebook from `notebooks/` in Colab.
2. Install the requirements: `pip install -r requirements.txt`
3. ECOSTRESS needs a free NASA Earthdata account. Log in with `earthaccess.login()`. **Never commit credentials** (`.netrc` is ignored by git).
4. MODIS, Sentinel-2 and the other layers are read from Microsoft Planetary Computer without an account.

## Timeline

- [x] Topic approved by the instructor
- [ ] Interim presentation and first demo: 30 November 2026
- [ ] Final presentation: 28 December 2026
- [ ] IEEE conference paper: 4 January 2027

## Key references

- N. Agam et al., "A vegetation index based technique for spatial sharpening of thermal imagery," *Remote Sens. Environ.*, 107(4), 545–558, 2007. doi:10.1016/j.rse.2006.10.006
- G. C. Hulley et al., "Validation and quality assessment of the ECOSTRESS level-2 land surface temperature and emissivity product," *IEEE TGRS*, 60, 2022. doi:10.1109/TGRS.2021.3079879
- Z. Wan, "New refinements and validation of the collection-6 MODIS land-surface temperature/emissivity product," *Remote Sens. Environ.*, 140, 36–45, 2014. doi:10.1016/j.rse.2013.08.027
- H. Çulpan, Ü. Şahin and G. Can, "A step to develop heat-health action plan: Assessing heat waves' impacts on mortality," *Atmosphere*, 13, 2126, 2022. doi:10.3390/atmos13122126
- T. Knight et al., "How effective is 'greening' of urban areas in reducing human exposure to ground-level ozone concentrations, UV exposure and the 'urban heat island effect'? An updated systematic review," *Environ. Evid.*, 10, 12, 2021. doi:10.1186/s13750-021-00226-y

## License

Code: MIT (see [LICENSE](LICENSE)). Data remain under their own licences, listed in [DATA_MANIFEST.md](DATA_MANIFEST.md).
