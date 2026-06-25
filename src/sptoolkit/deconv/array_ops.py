from __future__ import annotations

import numpy as np

from .constants import MAX_PLOT_POINTS


def window_mask(x: np.ndarray, xlim: tuple[float, float]) -> np.ndarray:
    lo, hi = sorted(xlim)
    return (x >= lo) & (x <= hi)


def baseline_edge_mask(x: np.ndarray, xlim: tuple[float, float]) -> np.ndarray:
    lo, hi = sorted(xlim)
    span = hi - lo
    edge = min(10.0, max(span * 0.1, np.finfo(float).eps))
    return ((x >= lo) & (x <= lo + edge)) | ((x >= hi - edge) & (x <= hi))


def decimate_for_plot(
    x: np.ndarray, y: np.ndarray, max_points: int = MAX_PLOT_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    step = int(np.ceil(x.size / max_points))
    return x[::step], y[::step]


def finite_bound(value: float | None) -> float:
    if value is None:
        return np.nan
    return float(value)

