"""Proposed method: two-step MODIS-ECOSTRESS fusion for night LST.

Step 1, block correction (980 m): a random forest learns where the MODIS night
pattern differs from the ECOSTRESS block means (sensor and time-gap effects,
coast, elevation) and corrects the MODIS block values.

Step 2, within-block anomaly (70 m): a second random forest learns how much each
70 m pixel deviates from its 980 m block mean. The anomaly is re-centred and added
to the corrected block value, so the block mean equals the corrected MODIS value.

The diagnostics in notebook 03 show that most of the night error sits at the
980 m scale, which is why step 1 carries most of the gain.
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


# --------------------------------------------------------------------------
# Quality screen
# --------------------------------------------------------------------------

def screen_nights(nights: list[dict], max_median_diff: float = 5.0, min_r: float = 0.0):
    """Drop nights where the two sensors disagree grossly at 980 m.

    Such nights carry cloud that one of the two cloud masks missed. The rule uses
    only the paired data and is applied to every method in the same way.
    Returns (kept nights, table of dropped nights).
    """
    kept, dropped = [], []
    for n in nights:
        eco_blk, mc = block_mean(n["eco"]), n["modis_coarse"]
        ok = np.isfinite(eco_blk) & np.isfinite(mc)
        d = float(np.median(eco_blk[ok] - mc[ok]))
        r = float(np.corrcoef(eco_blk[ok], mc[ok])[0, 1])
        if abs(d) > max_median_diff or r < min_r:
            dropped.append({"night_date": n["key"], "median_eco_minus_modis": round(d, 2),
                            "r_980m": round(r, 2)})
        else:
            kept.append(n)
    return kept, pd.DataFrame(dropped)


# --------------------------------------------------------------------------
# Step 1: block correction at 980 m
# --------------------------------------------------------------------------

BLOCK_PATTERN = ["modis_dev", "modis_minus_3x3", "modis_minus_5x5"]
BLOCK_CONTEXT = ["gap_h", "hour_after_21", "month"]


def _nan_smooth(a: np.ndarray, size: int) -> np.ndarray:
    from scipy import ndimage

    ok = np.isfinite(a)
    s = ndimage.uniform_filter(np.where(ok, a, 0.0), size)
    w = ndimage.uniform_filter(ok.astype(float), size)
    return np.where(w > 0, s / np.maximum(w, 1e-9), np.nan)


def block_columns(groups=tuple(STATIC_GROUPS), modis_pattern: bool = True,
                  night_context: bool = True) -> list[str]:
    cols = [f for g in groups for f in STATIC_GROUPS[g]]
    return (BLOCK_PATTERN if modis_pattern else []) + cols + (BLOCK_CONTEXT if night_context else [])


def block_table(night: dict, min_valid: float = 0.5) -> pd.DataFrame:
    """One row per 980 m block: predictors and the target
    (ECOSTRESS block mean minus MODIS, with the night-wide offset removed)."""
    mc = np.asarray(night["modis_coarse"], dtype="float64")
    land = night["land"]
    cols = {"modis_dev": mc - np.nanmean(mc),
            "modis_minus_3x3": mc - _nan_smooth(mc, 3),
            "modis_minus_5x5": mc - _nan_smooth(mc, 5)}
    for g in STATIC_GROUPS.values():
        for f in g:
            cols[f] = block_mean(np.where(land, night["layers"][f], np.nan))
    local = pd.Timestamp(night["local"])
    cols["gap_h"] = np.full(mc.shape, night["gap_h"])
    cols["hour_after_21"] = np.full(mc.shape, (local.hour + local.minute / 60 - 21) % 24)
    cols["month"] = np.full(mc.shape, local.month)
    df = pd.DataFrame({c: np.asarray(v, dtype="float32").ravel() for c, v in cols.items()})
    if night.get("eco") is None:                    # prediction-only night (no ECOSTRESS)
        df["target"], df["ok"] = np.nan, False
        return df
    eco_blk = block_mean(night["eco"])
    valid = block_mean(np.isfinite(night["eco"]).astype("float32"))
    target = (eco_blk - mc).ravel()
    df["target"] = target - np.nanmean(target)      # the night-wide offset is not learnable
    df["ok"] = (valid.ravel() >= min_valid) & np.isfinite(df["target"]) & np.isfinite(mc.ravel())
    return df


def make_block_model(seed: int = 0, n_estimators: int = 150):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=5, max_features=0.5,
                                 n_jobs=-1, random_state=seed)


def within_block_skill(pred: np.ndarray, eco: np.ndarray, scales=(1, 2, 7),
                       min_valid: float = 0.5) -> dict:
    """Correlation between predicted and observed within-block anomalies, after
    averaging both to 70 m x scale (70, 140 and 490 m by default)."""
    from .data import FACTOR

    out = {}
    valid = upsample(block_mean(np.isfinite(eco).astype("float32"))) >= min_valid
    eco_v = np.where(valid, eco, np.nan)
    for s in scales:
        e = block_mean(eco_v, s) if s > 1 else eco_v
        p = block_mean(pred, s) if s > 1 else pred
        b = FACTOR // s
        te = e - upsample(block_mean(e, b), b)
        tp = p - upsample(block_mean(p, b), b)
        v = np.isfinite(te) & np.isfinite(tp)
        enough = v.sum() > 50
        out[f"r_{70 * s}m"] = float(np.corrcoef(te[v], tp[v])[0, 1]) if enough else np.nan
        out[f"sd_eco_{70 * s}m"] = float(te[v].std()) if enough else np.nan
    return out


def cross_validate_fusion(nights: list[dict], fine_names: list[str], block_cols: list[str],
                          n_folds: int = 6, n_per_night: int = 2000, seed: int = 0,
                          fine: bool = True, block_trees: int = 150):
    """Leave-nights-out evaluation of the two-step fusion.

    Methods scored per test night: "Block correction" (step 1 only), "70 m anomaly"
    (step 2 on the raw MODIS) and "Proposed fusion" (both steps).
    Returns (metrics per night and method, within-block skill per night,
    mean feature importances {"block": Series, "fine": Series}).
    """
    from .downscale import metrics, tsharp

    rng = np.random.default_rng(seed)
    tables = [block_table(n) for n in nights]
    samples = ([training_sample(n["layers"], fine_names, n["y"], n["land"], n_per_night, rng)
                for n in nights] if fine else None)
    fold_of = np.arange(len(nights)) % n_folds
    rows, skill = [], []
    imp_b, imp_f = 0.0, 0.0
    for f in range(n_folds):
        btr = pd.concat([t[t.ok] for t, fo in zip(tables, fold_of) if fo != f])
        bm = make_block_model(seed, block_trees).fit(btr[block_cols].fillna(-999).values,
                                                     btr["target"].values)
        imp_b = imp_b + bm.feature_importances_ / n_folds
        if fine:
            tr = [s for s, fo in zip(samples, fold_of) if fo != f]
            fm = make_model(seed).fit(np.concatenate([s[0] for s in tr]),
                                      np.concatenate([s[1] for s in tr]))
            imp_f = imp_f + fm.feature_importances_ / n_folds
        for n, t, fo in zip(nights, tables, fold_of):
            if fo != f:
                continue
            mc, land, eco = n["modis_coarse"], n["land"], n["eco"]
            mc_corr = mc + bm.predict(t[block_cols].fillna(-999).values).reshape(mc.shape)
            preds = {"Block correction": np.where(land, upsample(mc_corr), np.nan)}
            if fine:
                anom_only = np.where(land, predict_night(fm, n["layers"], fine_names, mc, land), np.nan)
                preds["70 m anomaly"] = anom_only
                preds["Proposed fusion"] = np.where(
                    land, predict_night(fm, n["layers"], fine_names, mc_corr, land), np.nan)
                tsh = tsharp(mc, np.where(land, n["layers"]["ndvi"], np.nan))[0]
                for name, p in (("Proposed", anom_only), ("TsHARP", tsh)):
                    skill.append({"night_date": n["key"], "method": name, **within_block_skill(p, eco)})
            for name, p in preds.items():
                m = metrics(p, eco)
                m.update(night_date=n["key"], method=name)
                rows.append(m)
    imps = {"block": pd.Series(imp_b, index=block_cols)}
    if fine:
        imps["fine"] = pd.Series(imp_f, index=design_matrix(nights[0]["layers"], fine_names)[1])
    return pd.DataFrame(rows), pd.DataFrame(skill), imps


def fit_fusion(nights: list[dict], fine_names: list[str], block_cols: list[str],
               n_per_night: int = 2000, seed: int = 0):
    """Train both steps on all given nights (for maps and examples)."""
    rng = np.random.default_rng(seed)
    btr = pd.concat([t[t.ok] for t in (block_table(n) for n in nights)])
    bm = make_block_model(seed).fit(btr[block_cols].fillna(-999).values, btr["target"].values)
    Xs, ys = zip(*[training_sample(n["layers"], fine_names, n["y"], n["land"], n_per_night, rng)
                   for n in nights])
    fm = make_model(seed).fit(np.concatenate(Xs), np.concatenate(ys))
    return bm, fm


def predict_fusion(bm, fm, night: dict, fine_names: list[str], block_cols: list[str]) -> np.ndarray:
    """70 m night LST for one night: corrected MODIS blocks plus the 70 m anomaly."""
    mc = night["modis_coarse"]
    t = block_table(night)
    mc_corr = mc + bm.predict(t[block_cols].fillna(-999).values).reshape(mc.shape)
    return np.where(night["land"], predict_night(fm, night["layers"], fine_names, mc_corr, night["land"]), np.nan)
