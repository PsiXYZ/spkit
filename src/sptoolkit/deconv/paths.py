from __future__ import annotations

import itertools
import re
from pathlib import Path


def safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "spectrum"


def make_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in itertools.count(1):
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate output directory")

