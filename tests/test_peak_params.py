from __future__ import annotations

import pytest

from sptoolkit.deconv.models import ParamSpec, PeakSpec
from sptoolkit.deconv.peak_params import (
    read_peak_params,
    read_peak_params_document,
    read_peak_params_csv,
    read_peak_params_json,
    read_peak_params_json_document,
    write_peak_params,
    write_peak_params_csv,
    write_peak_params_json,
)


def test_peak_params_roundtrip(tmp_path):
    peaks = [
        PeakSpec(
            kind="Voigt",
            center=ParamSpec(1600.0, 1598.0, 1602.0),
            amplitude=ParamSpec(2500.0, 0.0, 10000.0),
            sigma=ParamSpec(1.2, 0.1, 5.0),
            extras={"gamma": ParamSpec(1.5, 0.0, 10.0, vary=False)},
        )
    ]
    path = tmp_path / "peaks.csv"

    write_peak_params_csv(path, peaks)
    loaded = read_peak_params_csv(path)

    assert loaded == peaks


def test_peak_params_json_roundtrip(tmp_path):
    peaks = [
        PeakSpec(
            kind="PseudoVoigt",
            center=ParamSpec(1600.0, 1598.0, 1602.0),
            amplitude=ParamSpec(2500.0, 0.0, 10000.0),
            sigma=ParamSpec(1.2, 0.1, 5.0),
            extras={"fraction": ParamSpec(0.4, 0.0, 1.0, vary=False)},
        )
    ]
    path = tmp_path / "peaks.json"

    write_peak_params_json(path, peaks)
    loaded = read_peak_params_json(path)

    assert loaded == peaks


def test_peak_params_json_roundtrip_with_deconvolution_range(tmp_path):
    peaks = [
        PeakSpec(
            kind="Gaussian",
            center=ParamSpec(1600.0, 1598.0, 1602.0),
            amplitude=ParamSpec(2500.0, 0.0, 10000.0),
            sigma=ParamSpec(1.2, 0.1, 5.0),
        )
    ]
    path = tmp_path / "peaks.json"

    write_peak_params_json(path, peaks, deconvolution_range=(1500.0, 1700.0))
    loaded = read_peak_params_json_document(path)

    assert loaded.peaks == peaks
    assert loaded.deconvolution_range == (1500.0, 1700.0)


def test_peak_params_json_without_deconvolution_range_is_supported(tmp_path):
    peaks = [
        PeakSpec(
            kind="Gaussian",
            center=ParamSpec(1600.0, 1598.0, 1602.0),
            amplitude=ParamSpec(2500.0, 0.0, 10000.0),
            sigma=ParamSpec(1.2, 0.1, 5.0),
        )
    ]
    path = tmp_path / "peaks.json"

    write_peak_params_json(path, peaks)
    loaded = read_peak_params_json_document(path)

    assert loaded.peaks == peaks
    assert loaded.deconvolution_range is None


def test_peak_params_dispatches_by_extension(tmp_path):
    peaks = [
        PeakSpec(
            kind="Gaussian",
            center=ParamSpec(1600.0, 1598.0, 1602.0),
            amplitude=ParamSpec(2500.0, 0.0, 10000.0),
            sigma=ParamSpec(1.2, 0.1, 5.0),
        )
    ]
    path = tmp_path / "peaks.json"

    write_peak_params(path, peaks, deconvolution_range=(1500.0, 1700.0))
    loaded = read_peak_params(path)
    document = read_peak_params_document(path)

    assert loaded == peaks
    assert document.peaks == peaks
    assert document.deconvolution_range == (1500.0, 1700.0)


def test_peak_params_rejects_unknown_model(tmp_path):
    path = tmp_path / "peaks.csv"
    path.write_text(
        "peak,kind,param,value,min,max,vary\n"
        "1,Unknown,center,1600,1598,1602,true\n"
        "1,Unknown,amplitude,2500,0,10000,true\n"
        "1,Unknown,sigma,1,0,5,true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported peak type"):
        read_peak_params_csv(path)
