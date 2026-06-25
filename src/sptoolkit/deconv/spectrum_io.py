from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def find_csv_files(input_dir: Path) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in input_dir.rglob("*.csv") if path.is_file())


def resolve_initial_file(input_dir: Path, file_arg: Path | None, files: list[Path]) -> Path:
    if file_arg is None:
        if not files:
            raise FileNotFoundError(f"No CSV files found under {input_dir}")
        return files[0]
    if file_arg.is_absolute():
        return file_arg
    candidate = input_dir / file_arg
    return candidate if candidate.exists() else file_arg.resolve()


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns")
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    return x[order], y[order]

