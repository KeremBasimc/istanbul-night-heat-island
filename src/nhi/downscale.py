"""Downscaling methods and validation metrics.

All functions work on plain numpy arrays: a coarse array of shape (H, W) and
fine arrays of shape (H * FACTOR, W * FACTOR) on grids that nest exactly
(see `nhi.data.grids`).
"""

from __future__ import annotations

import numpy as np

from .data import FACTOR


def upsample(coarse: np.ndarray, factor: int = FACTOR) -> np.ndarray:
    """Repeat each coarse pixel onto its factor x factor fine pixels."""
    return np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)


def block_mean(fine: np.ndarray, factor: int = FACTOR) -> np.ndarray:
    """Mean of the finite fine pixels inside each coarse pixel (NaN if none)."""
    h, w = fine.shape[0] // factor, fine.shape[1] // factor
    blocks = fine[: h * factor, : w * factor].reshape(h, factor, w, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=(1, 3))


def fractional_cover(ndvi: np.ndarray, ndvi_min: float, ndvi_max: float) -> np.ndarray:
    """Vegetation fraction as used in TsHARP (Agam et al. 2007)."""
    scaled = np.clip((ndvi_max - ndvi) / (ndvi_max - ndvi_min), 0, 1)
    return 1 - scaled ** 0.625


def tsharp(lst_coarse: np.ndarray, ndvi_fine: np.ndarray, factor: int = FACTOR,
           ndvi_range: tuple[float, float] | None = None):
    """TsHARP thermal sharpening (Agam et al. 2007) with residual correction.

    1. Fit LST = a + b * fc at the coarse scale.
    2. Apply the fit to the fine fc.
    3. Add each coarse pixel's fit residual to its fine pixels, so that the
       fine prediction averages back to the coarse observation.
    Returns (fine prediction, (a, b)).
    """
    if ndvi_range is None:
        ndvi_range = tuple(np.nanpercentile(ndvi_fine, [2, 98]))
    fc_fine = fractional_cover(ndvi_fine, *ndvi_range)
    fc_coarse = block_mean(fc_fine, factor)

    ok = np.isfinite(lst_coarse) & np.isfinite(fc_coarse)
    if ok.sum() < 3:
        return np.full(ndvi_fine.shape, np.nan), (np.nan, np.nan)
    b, a = np.polyfit(fc_coarse[ok], lst_coarse[ok], 1)

    residual = lst_coarse - (a + b * fc_coarse)
    pred = a + b * fc_fine + upsample(residual, factor)
    return pred, (a, b)


def metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Agreement between a fine prediction and the fine reference (both degrees C).

    `crmse` removes the mean bias first, so it measures how well the spatial
    pattern is reproduced; `rmse` includes the bias.
    """
    ok = np.isfinite(pred) & np.isfinite(ref)
    n = int(ok.sum())
    if n < 10:
        return {"n": n, "rmse": np.nan, "crmse": np.nan, "mae": np.nan,
                "bias": np.nan, "r2": np.nan, "r": np.nan}
    p, r = pred[ok], ref[ok]
    err = p - r
    bias = float(err.mean())
    ss_res = float((err ** 2).sum())
    ss_tot = float(((r - r.mean()) ** 2).sum())
    return {
        "n": n,
        "rmse": float(np.sqrt((err ** 2).mean())),
        "crmse": float(np.sqrt(((err - bias) ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "bias": bias,
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "r": float(np.corrcoef(p, r)[0, 1]),
    }
