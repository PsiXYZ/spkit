from __future__ import annotations

import lmfit.models as md

PEAK_MODELS = {
    "Gaussian": md.GaussianModel,
    "Lorentzian": md.LorentzianModel,
    "Voigt": md.VoigtModel,
    "PseudoVoigt": md.PseudoVoigtModel,
}

DEFAULT_XLIM = (1550.0, 1700.0)
DEFAULT_YLIM = (0.0, 4300.0)
DEFAULT_SIGMA = 1.0
DEFAULT_CENTER_SPAN = 4.0
DEFAULT_AMPLITUDE_MAX = 100000.0
DEFAULT_SIGMA_MAX = 20.0
DRAG_PICK_RADIUS_PX = 12.0
MAX_PLOT_POINTS = 5000

COLOR_SPECTRUM = "#c2410c"
COLOR_FIT = "#2563eb"
COLOR_BASELINE = "#6b7280"
COLOR_SELECTED = "#f59e0b"
COLOR_MARKER = "#111827"

