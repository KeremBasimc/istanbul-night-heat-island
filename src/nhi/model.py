"""Proposed method: random-forest downscaling with night-cooling predictors.

The model learns the *within-block anomaly* of ECOSTRESS: how much each 70 m
pixel deviates from the mean of its 980 m block. At prediction time the
anomaly is added to the MODIS value of the block and re-centred so that the
block mean equals MODIS again (the same residual correction as TsHARP).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .downscale import block_mean, upsample

# Static predictors (70 m), grouped for the ablation.
STATIC_GROUPS = {
    "vegetation": ["ndvi", "tree", "grass", "shrub", "crop"],
    "built": ["built", "ndbi", "bare"],
    "location": ["sea_dist_km", "elev"],
    "heat_storage": ["landsat_day"],
    "building_height": ["bldg_height"],
}
# Night-level context, the same value for every pixel of a night
# (except modis_coarse, which varies by 980 m block).
NIGHT_FEATURES = ["modis_coarse", "modis_scene_mean", "hour_after_21",
                  "month_sin", "month_cos", "gap_h"]
# Benchmark 3: a generic Sentinel-2-only random forest.
S2_ONLY = ["ndvi", "ndbi", "modis_coarse"]


def features_for(groups=tuple(STATIC_GROUPS), night_context: bool = True) -> list[str]:
    names = [f for g in groups for f in STATIC_GROUPS[g]]
    return names + (NIGHT_FEATURES if night_context else [])


def _with_deviation(name: str, arr: np.ndarray) -> dict:
    """The layer itself and its deviation from its 980 m block mean."""
    return {name: arr, name + "_dev": arr - upsample(block_mean(arr))}


def static_with_deviations(static: dict) -> dict:
    """Static layers plus their block deviations; computed once per year."""
    layers = {}
    for name, arr in static.items():
        if name != "land":
            layers.update(_with_deviation(name, np.asarray(arr, dtype="float32")))
    return layers


def night_layers(static_dev: dict, modis_coarse: np.ndarray, eco_local: pd.Timestamp,
                 gap_h: float) -> dict:
    """Per-pixel layers for one night: the shared static layers plus night context.

    Night-level constants are broadcast views, so they cost no memory.
    """
    layers = dict(static_dev)
    shape = next(iter(static_dev.values())).shape
    hour = (eco_local.hour + eco_local.minute / 60 - 21) % 24
    month = eco_local.month
    layers["modis_coarse"] = upsample(np.asarray(modis_coarse, dtype="float32"))
    for name, value in [("modis_scene_mean", float(np.nanmean(modis_coarse))),
                        ("hour_after_21", hour),
                        ("month_sin", np.sin(2 * np.pi * month / 12)),
                        ("month_cos", np.cos(2 * np.pi * month / 12)),
                        ("gap_h", gap_h)]:
        layers[name] = np.broadcast_to(np.float32(value), shape)
    return layers


def design_matrix(layers: dict, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Pixels x features; each static feature also brings its `_dev` column."""
    cols = []
    for n in names:
        cols.append(n)
        if n + "_dev" in layers:
            cols.append(n + "_dev")
    X = np.stack([layers[c].ravel() for c in cols], axis=1).astype("float32")
    return X, cols


def anomaly_target(eco: np.ndarray, min_valid: float = 0.5) -> np.ndarray:
    """ECOSTRESS minus its 980 m block mean; NaN where the block is mostly unobserved."""
    from .data import FACTOR

    valid = np.isfinite(eco).astype("float32")
    enough = upsample(block_mean(valid, FACTOR)) >= min_valid
    y = eco - upsample(block_mean(eco, FACTOR))
    return np.where(enough, y, np.nan)


def make_model(seed: int = 0):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_features=0.5,
                                 n_jobs=-1, random_state=seed)


def predict_night(model, layers: dict, names: list[str], modis_coarse: np.ndarray,
                  land: np.ndarray) -> np.ndarray:
    """Fine LST (degrees C): MODIS block value plus the re-centred predicted anomaly."""
    X, _ = design_matrix(layers, names)
    ok = np.all(np.isfinite(X), axis=1) & land.ravel()
    anom = np.full(X.shape[0], np.nan, dtype="float32")
    if ok.any():
        anom[ok] = model.predict(X[ok])
    anom = anom.reshape(land.shape)
    anom = anom - upsample(block_mean(anom))          # block mean of the anomaly = 0
    return upsample(modis_coarse) + anom


def training_sample(layers: dict, names: list[str], y: np.ndarray, land: np.ndarray,
                    n: int, rng: np.random.Generator):
    X, _ = design_matrix(layers, names)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y.ravel()) & land.ravel()
    idx = np.flatnonzero(ok)
    if idx.size > n:
        idx = rng.choice(idx, n, replace=False)
    return X[idx], y.ravel()[idx]


def cross_validate(nights: list[dict], names: list[str], n_folds: int | None = None,
                   n_per_night: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Leave-nights-out cross-validation; returns one metrics row per test night.

    Each element of `nights` holds: key, layers, y (anomaly target), eco,
    modis_coarse, land. `n_folds=None` means leave-one-night-out.
    """
    from .downscale import metrics

    rng = np.random.default_rng(seed)
    samples = [training_sample(n["layers"], names, n["y"], n["land"], n_per_night, rng)
               for n in nights]
    k = len(nights) if n_folds is None else n_folds
    fold_of = np.arange(len(nights)) % k
    rows = []
    for f in range(k):
        train = [s for s, fo in zip(samples, fold_of) if fo != f]
        model = make_model(seed)
        model.fit(np.concatenate([s[0] for s in train]), np.concatenate([s[1] for s in train]))
        for night, fo in zip(nights, fold_of):
            if fo != f:
                continue
            pred = predict_night(model, night["layers"], names, night["modis_coarse"], night["land"])
            m = metrics(pred, night["eco"])
            m["night_date"] = night["key"]
            rows.append(m)
    return pd.DataFrame(rows)
