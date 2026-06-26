from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import PEAK_MODELS
from .models import ParamSpec, PeakSpec

REQUIRED_PARAM_NAMES = {"center", "amplitude", "sigma"}


@dataclass
class PeakParamsDocument:
    peaks: list[PeakSpec]
    deconvolution_range: tuple[float, float] | None = None


def optional_float_from_text(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return float(text)


def bool_from_text(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def peak_param_to_data(spec: ParamSpec) -> dict[str, Any]:
    return {
        "value": spec.value,
        "min": spec.minimum,
        "max": spec.maximum,
        "vary": spec.vary,
    }


def peak_to_data(peak: PeakSpec) -> dict[str, Any]:
    return {
        "kind": peak.kind,
        "params": {
            name: peak_param_to_data(spec)
            for name, spec in peak.all_params().items()
        },
    }


def deconvolution_range_to_data(xlim: tuple[float, float] | None) -> dict[str, float] | None:
    if xlim is None:
        return None
    return {
        "min": float(xlim[0]),
        "max": float(xlim[1]),
    }


def deconvolution_range_from_data(data: Any) -> tuple[float, float] | None:
    if data is None:
        return None
    if isinstance(data, dict):
        if "min" not in data or "max" not in data:
            raise ValueError("deconvolution_range must contain min and max")
        return (float(data["min"]), float(data["max"]))
    if isinstance(data, list | tuple) and len(data) == 2:
        return (float(data[0]), float(data[1]))
    raise ValueError("deconvolution_range must be an object with min/max or a two-value list")


def param_from_data(data: Any, context: str) -> ParamSpec:
    if not isinstance(data, dict):
        raise ValueError(f"{context}: parameter must be an object")
    if "value" not in data:
        raise ValueError(f"{context}: missing value")
    return ParamSpec(
        value=float(data["value"]),
        minimum=optional_float_from_text(data.get("min")),
        maximum=optional_float_from_text(data.get("max")),
        vary=bool_from_text(data.get("vary", True)),
    )


def peak_from_params(kind: str, params: dict[str, ParamSpec], context: str) -> PeakSpec:
    if kind not in PEAK_MODELS:
        raise ValueError(f"{context}: unsupported peak type {kind!r}")
    missing = REQUIRED_PARAM_NAMES - set(params)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{context} is missing required parameter(s): {names}")
    extras = {
        name: spec
        for name, spec in params.items()
        if name not in REQUIRED_PARAM_NAMES
    }
    return PeakSpec(
        kind=kind,
        center=params["center"],
        amplitude=params["amplitude"],
        sigma=params["sigma"],
        extras=extras,
    )


def write_peak_params_csv(path: Path, peaks: list[PeakSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["peak", "kind", "param", "value", "min", "max", "vary"])
        for index, peak in enumerate(peaks, start=1):
            for name, spec in peak.all_params().items():
                writer.writerow(
                    [
                        index,
                        peak.kind,
                        name,
                        spec.value,
                        "" if spec.minimum is None else spec.minimum,
                        "" if spec.maximum is None else spec.maximum,
                        spec.vary,
                    ]
                )


def read_peak_params_csv(path: Path) -> list[PeakSpec]:
    grouped: dict[int, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"peak", "kind", "param", "value", "min", "max", "vary"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Peak CSV must contain columns: peak, kind, param, value, min, max, vary")
        for row_number, row in enumerate(reader, start=2):
            peak_index = int(row["peak"])
            kind = str(row["kind"]).strip()
            name = str(row["param"]).strip()
            if peak_index < 1:
                raise ValueError(f"Row {row_number}: peak index must be >= 1")
            if kind not in PEAK_MODELS:
                raise ValueError(f"Row {row_number}: unsupported peak type {kind!r}")
            if not name:
                raise ValueError(f"Row {row_number}: empty parameter name")
            grouped.setdefault(peak_index, {"kind": kind, "params": {}})
            if grouped[peak_index]["kind"] != kind:
                raise ValueError(f"Row {row_number}: mixed peak types for p{peak_index}")
            grouped[peak_index]["params"][name] = ParamSpec(
                value=float(row["value"]),
                minimum=optional_float_from_text(row["min"]),
                maximum=optional_float_from_text(row["max"]),
                vary=bool_from_text(row["vary"]),
            )

    peaks: list[PeakSpec] = []
    for peak_index in sorted(grouped):
        data = grouped[peak_index]
        peaks.append(peak_from_params(data["kind"], data["params"], f"Peak p{peak_index}"))
    if not peaks:
        raise ValueError("Peak CSV does not contain any peaks")
    return peaks


def write_peak_params_json(
    path: Path,
    peaks: list[PeakSpec],
    deconvolution_range: tuple[float, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "peaks": [peak_to_data(peak) for peak in peaks],
    }
    range_data = deconvolution_range_to_data(deconvolution_range)
    if range_data is not None:
        data["deconvolution_range"] = range_data
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_peak_params_json_document(path: Path) -> PeakParamsDocument:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    deconvolution_range = None
    if isinstance(data, list):
        peak_items = data
    elif isinstance(data, dict) and isinstance(data.get("peaks"), list):
        peak_items = data["peaks"]
        deconvolution_range = deconvolution_range_from_data(
            data.get("deconvolution_range", data.get("xlim"))
        )
    else:
        raise ValueError("Peak JSON must contain a peaks list")

    peaks: list[PeakSpec] = []
    for index, item in enumerate(peak_items, start=1):
        context = f"Peak p{index}"
        if not isinstance(item, dict):
            raise ValueError(f"{context}: peak must be an object")
        kind = str(item.get("kind", "")).strip()
        raw_params = item.get("params")
        if not isinstance(raw_params, dict):
            raise ValueError(f"{context}: params must be an object")
        params = {
            str(name).strip(): param_from_data(raw_spec, f"{context}.{name}")
            for name, raw_spec in raw_params.items()
            if str(name).strip()
        }
        peaks.append(peak_from_params(kind, params, context))

    if not peaks:
        raise ValueError("Peak JSON does not contain any peaks")
    return PeakParamsDocument(peaks=peaks, deconvolution_range=deconvolution_range)


def read_peak_params_json(path: Path) -> list[PeakSpec]:
    return read_peak_params_json_document(path).peaks


def write_peak_params(
    path: Path,
    peaks: list[PeakSpec],
    deconvolution_range: tuple[float, float] | None = None,
) -> None:
    if path.suffix.lower() == ".json":
        write_peak_params_json(path, peaks, deconvolution_range)
    else:
        write_peak_params_csv(path, peaks)


def read_peak_params_document(path: Path) -> PeakParamsDocument:
    if path.suffix.lower() == ".json":
        return read_peak_params_json_document(path)
    return PeakParamsDocument(peaks=read_peak_params_csv(path))


def read_peak_params(path: Path) -> list[PeakSpec]:
    return read_peak_params_document(path).peaks
