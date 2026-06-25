from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .constants import PEAK_MODELS
from .models import ParamSpec, PeakSpec


def optional_float_from_text(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return float(text)


def bool_from_text(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


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
        params = data["params"]
        missing = {"center", "amplitude", "sigma"} - set(params)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Peak p{peak_index} is missing required parameter(s): {names}")
        extras = {
            name: spec
            for name, spec in params.items()
            if name not in {"center", "amplitude", "sigma"}
        }
        peaks.append(
            PeakSpec(
                kind=data["kind"],
                center=params["center"],
                amplitude=params["amplitude"],
                sigma=params["sigma"],
                extras=extras,
            )
        )
    if not peaks:
        raise ValueError("Peak CSV does not contain any peaks")
    return peaks

