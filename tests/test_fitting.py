from __future__ import annotations

from sptoolkit.deconv.fitting import build_model_and_params
from sptoolkit.deconv.models import ParamSpec, PeakSpec


def test_build_model_and_params_uses_peak_prefixes_and_baseline_settings():
    peaks = [
        PeakSpec(
            kind="Gaussian",
            center=ParamSpec(1600.0, 1598.0, 1602.0),
            amplitude=ParamSpec(100.0, 0.0, 1000.0),
            sigma=ParamSpec(1.0, 0.1, 10.0),
        )
    ]

    _model, params = build_model_and_params(peaks, baseline_coeffs=(0.5, 10.0), fit_baseline=False)

    assert params["bl_slope"].value == 0.5
    assert params["bl_intercept"].value == 10.0
    assert params["bl_slope"].vary is False
    assert params["p1_center"].value == 1600.0
    assert params["p1_center"].min == 1598.0
    assert params["p1_center"].max == 1602.0

