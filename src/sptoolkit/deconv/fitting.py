from __future__ import annotations

from typing import Any

import lmfit.models as md
import numpy as np
from lmfit.model import ModelResult

from .constants import PEAK_MODELS
from .models import PeakSpec


def build_model_and_params(
    peaks: list[PeakSpec],
    baseline_coeffs: tuple[float, float],
    fit_baseline: bool,
) -> tuple[Any, Any]:
    model = md.LinearModel(prefix="bl_")
    for index, peak in enumerate(peaks, start=1):
        model += PEAK_MODELS[peak.kind](prefix=f"p{index}_")
    params = model.make_params()

    slope, intercept = baseline_coeffs
    params["bl_intercept"].set(value=intercept, vary=fit_baseline)
    params["bl_slope"].set(value=slope, vary=fit_baseline)

    for index, peak in enumerate(peaks, start=1):
        prefix = f"p{index}_"
        for name, spec in peak.all_params().items():
            full_name = prefix + name
            if full_name not in params:
                continue
            if name == "gamma":
                params[full_name].expr = None
            params[full_name].set(
                value=spec.value,
                min=-np.inf if spec.minimum is None else spec.minimum,
                max=np.inf if spec.maximum is None else spec.maximum,
                vary=spec.vary,
            )
    return model, params


def update_peak_specs_from_result(peaks: list[PeakSpec], result: ModelResult) -> None:
    for index, peak in enumerate(peaks, start=1):
        prefix = f"p{index}_"
        for name, spec in peak.all_params().items():
            full_name = prefix + name
            if full_name in result.params:
                param = result.params[full_name]
                spec.value = float(param.value)
                spec.minimum = None if np.isneginf(param.min) else float(param.min)
                spec.maximum = None if np.isposinf(param.max) else float(param.max)
                spec.vary = bool(param.vary)

