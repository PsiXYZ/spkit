from __future__ import annotations

import argparse
import copy
import csv
import itertools
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lmfit.models as md
import matplotlib
import numpy as np
import pandas as pd
from lmfit.model import ModelResult
from matplotlib.backend_bases import MouseEvent

os.environ.setdefault("QT_API", "PySide6")
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

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


@dataclass
class ParamSpec:
    value: float
    minimum: float | None = None
    maximum: float | None = None
    vary: bool = True


@dataclass
class PeakSpec:
    kind: str
    center: ParamSpec
    amplitude: ParamSpec
    sigma: ParamSpec
    extras: dict[str, ParamSpec] = field(default_factory=dict)

    def all_params(self) -> dict[str, ParamSpec]:
        params = {
            "center": self.center,
            "amplitude": self.amplitude,
            "sigma": self.sigma,
        }
        params.update(self.extras)
        return params


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="PySide6 Raman spectrum deconvolution.")
    parser.add_argument("--input-dir", type=Path, default=base_dir / "in")
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--xlim", type=float, nargs=2, default=DEFAULT_XLIM)
    parser.add_argument("--ylim", type=float, nargs=2, default=DEFAULT_YLIM)
    parser.add_argument("--method", default="least_squares")
    return parser.parse_args()


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


def finite_bound(value: float | None) -> float:
    if value is None:
        return np.nan
    return float(value)


def optional_float_from_text(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return float(text)


def bool_from_text(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


class OptionalFloatEdit(QLineEdit):
    def value(self) -> float | None:
        text = self.text().strip()
        if not text:
            return None
        return float(text)

    def set_optional_value(self, value: float | None) -> None:
        self.setText("" if value is None else f"{value:.8g}")


class DeconvolutionWindow(QMainWindow):
    def __init__(
        self,
        files: list[Path],
        initial_file: Path,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        fit_method: str,
    ) -> None:
        super().__init__()
        self.files = files
        self.file_index = self.files.index(initial_file) if initial_file in self.files else 0
        self.current_file = initial_file
        self.xlim = (float(xlim[0]), float(xlim[1]))
        self.ylim = (float(ylim[0]), float(ylim[1]))
        self.fit_method = fit_method

        self.x = np.array([], dtype=float)
        self.y = np.array([], dtype=float)
        self.peaks: list[PeakSpec] = []
        self.selected_index: int | None = None
        self.drag_index: int | None = None
        self.drag_from_center_axis = False
        self.result: ModelResult | None = None
        self.baseline_coeffs: tuple[float, float] | None = None
        self.spectrum_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        self.syncing_controls = False

        self.setWindowTitle("VIP DECONVOLUTION TOOL 5000$ PER LICENSE")
        self.resize(1420, 820)
        self._build_ui()
        self._connect_canvas()
        self.load_file(initial_file)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(10, 7), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.figure.add_subplot(211)
        self.center_ax = self.figure.add_subplot(212, sharex=self.ax)
        self.center_ax.set_box_aspect(0.12)
        left.addWidget(self.toolbar)
        left.addWidget(self.canvas, 1)
        self.splitter.addWidget(left_widget)

        panel = QVBoxLayout()
        panel.setSpacing(10)
        panel.addWidget(self._file_group())
        panel.addWidget(self._model_group())
        panel.addWidget(self._parameter_group())
        panel.addWidget(self._fit_group())
        panel.addWidget(self._peak_table())
        panel.addStretch(1)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        panel.addWidget(self.status)
        side = QWidget()
        side.setLayout(panel)
        side.setMinimumWidth(280)
        self.splitter.addWidget(side)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([1040, 360])
        layout.addWidget(self.splitter, 1)

        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QGroupBox { font-weight: 600; }
            QPushButton { padding: 6px 10px; }
            QLineEdit, QComboBox, QDoubleSpinBox { padding: 3px; }
            """
        )

    def _file_group(self) -> QGroupBox:
        group = QGroupBox("Spectrum")
        layout = QGridLayout(group)
        self.file_combo = QComboBox()
        self.file_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.file_combo.addItems([str(path) for path in self.files])
        self.file_combo.currentIndexChanged.connect(self.on_file_combo_changed)
        self.browse_button = QPushButton("Browse")
        self.prev_button = QPushButton("Prev")
        self.next_button = QPushButton("Next")
        self.browse_button.clicked.connect(self.browse_file)
        self.prev_button.clicked.connect(lambda: self.switch_file(-1))
        self.next_button.clicked.connect(lambda: self.switch_file(1))
        layout.addWidget(self.file_combo, 0, 0, 1, 3)
        layout.addWidget(self.browse_button, 1, 0)
        layout.addWidget(self.prev_button, 1, 1)
        layout.addWidget(self.next_button, 1, 2)
        return group

    def _model_group(self) -> QGroupBox:
        group = QGroupBox("Model")
        layout = QFormLayout(group)
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(PEAK_MODELS.keys())
        self.kind_combo.currentTextChanged.connect(self.on_kind_change)
        self.method_edit = QLineEdit(self.fit_method)
        self.method_edit.editingFinished.connect(self.on_method_changed)
        self.fit_baseline_box = QCheckBox("Fit linear baseline")
        layout.addRow("Peak type", self.kind_combo)
        layout.addRow("Method", self.method_edit)
        layout.addRow("", self.fit_baseline_box)
        return group

    def _parameter_group(self) -> QGroupBox:
        group = QGroupBox("Selected Peak")
        layout = QGridLayout(group)
        layout.addWidget(QLabel("param"), 0, 0)
        layout.addWidget(QLabel("value"), 0, 1)
        layout.addWidget(QLabel("min"), 0, 2)
        layout.addWidget(QLabel("max"), 0, 3)
        self.param_edits: dict[str, tuple[QDoubleSpinBox, OptionalFloatEdit, OptionalFloatEdit]] = {}
        for row, name in enumerate(["center", "amplitude", "sigma", "gamma", "fraction"], start=1):
            value = QDoubleSpinBox()
            value.setDecimals(6)
            value.setRange(-1.0e12, 1.0e12)
            value.setKeyboardTracking(False)
            value.valueChanged.connect(lambda _value, key=name: self.on_param_changed(key))
            min_edit = OptionalFloatEdit()
            max_edit = OptionalFloatEdit()
            min_edit.editingFinished.connect(lambda key=name: self.on_param_changed(key))
            max_edit.editingFinished.connect(lambda key=name: self.on_param_changed(key))
            self.param_edits[name] = (value, min_edit, max_edit)
            layout.addWidget(QLabel(name), row, 0)
            layout.addWidget(value, row, 1)
            layout.addWidget(min_edit, row, 2)
            layout.addWidget(max_edit, row, 3)
        self.delete_button = QPushButton("Delete Peak")
        self.delete_button.clicked.connect(self.delete_selected_peak)
        layout.addWidget(self.delete_button, 6, 0, 1, 4)
        return group

    def _fit_group(self) -> QGroupBox:
        group = QGroupBox("Fit / Export")
        layout = QGridLayout(group)
        self.fit_button = QPushButton("Fit")
        self.export_button = QPushButton("Export")
        self.export_peaks_button = QPushButton("Export Peaks")
        self.import_peaks_button = QPushButton("Import Peaks")
        self.batch_button = QPushButton("Process Folder")
        self.xmin_spin = self._axis_spin(self.xlim[0])
        self.xmax_spin = self._axis_spin(self.xlim[1])
        self.ymin_spin = self._axis_spin(self.ylim[0])
        self.ymax_spin = self._axis_spin(self.ylim[1])
        self.fit_button.clicked.connect(self.fit)
        self.export_button.clicked.connect(self.export)
        self.export_peaks_button.clicked.connect(self.export_peak_params)
        self.import_peaks_button.clicked.connect(self.import_peak_params)
        self.batch_button.clicked.connect(self.process_folder)
        for spin in [self.xmin_spin, self.xmax_spin, self.ymin_spin, self.ymax_spin]:
            spin.valueChanged.connect(self.on_axes_changed)
        layout.addWidget(QLabel("x min"), 0, 0)
        layout.addWidget(self.xmin_spin, 0, 1)
        layout.addWidget(QLabel("x max"), 0, 2)
        layout.addWidget(self.xmax_spin, 0, 3)
        layout.addWidget(QLabel("y min"), 1, 0)
        layout.addWidget(self.ymin_spin, 1, 1)
        layout.addWidget(QLabel("y max"), 1, 2)
        layout.addWidget(self.ymax_spin, 1, 3)
        layout.addWidget(self.fit_button, 2, 0, 1, 2)
        layout.addWidget(self.export_button, 2, 2, 1, 2)
        layout.addWidget(self.export_peaks_button, 3, 0, 1, 2)
        layout.addWidget(self.import_peaks_button, 3, 2, 1, 2)
        layout.addWidget(self.batch_button, 4, 0, 1, 4)
        return group

    def _peak_table(self) -> QTableWidget:
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "type", "center", "amplitude"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.on_table_selection)
        return self.table

    @staticmethod
    def _axis_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(-1.0e9, 1.0e9)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    def _connect_canvas(self) -> None:
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def set_status(self, message: str) -> None:
        self.status.setText(message)

    def load_file(self, path: Path) -> None:
        try:
            self.current_file = path
            self.x, self.y = self.load_spectrum_cached(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.baseline_coeffs = None
        self.result = None
        if path not in self.files:
            self.files.append(path)
            self.files.sort()
        self.file_index = self.files.index(path)
        self.sync_file_combo()
        self.sync_controls()
        self.set_status(f"Loaded {path.name}. Shift+click on plot adds a peak.")
        self.redraw()

    def load_spectrum_cached(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        cached = self.spectrum_cache.get(path)
        if cached is not None:
            return cached
        data = read_spectrum(path)
        self.spectrum_cache[path] = data
        return data

    def sync_file_combo(self) -> None:
        self.syncing_controls = True
        try:
            self.file_combo.blockSignals(True)
            self.file_combo.clear()
            self.file_combo.addItems([str(path) for path in self.files])
            self.file_combo.setCurrentIndex(self.file_index)
        finally:
            self.file_combo.blockSignals(False)
            self.syncing_controls = False

    def on_file_combo_changed(self, index: int) -> None:
        if self.syncing_controls or index < 0 or index >= len(self.files):
            return
        self.load_file(self.files[index])

    def browse_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open spectrum",
            str(Path(__file__).resolve().parent / "in"),
            "CSV files (*.csv *.CSV);;All files (*.*)",
        )
        if filename:
            self.load_file(Path(filename))

    def switch_file(self, step: int) -> None:
        if not self.files:
            return
        self.file_index = (self.file_index + step) % len(self.files)
        self.load_file(self.files[self.file_index])

    def on_method_changed(self) -> None:
        method = self.method_edit.text().strip()
        if method:
            self.fit_method = method
            self.set_status(f"Fit method set to {method}")
        else:
            self.method_edit.setText(self.fit_method)

    def on_axes_changed(self) -> None:
        self.xlim = (self.xmin_spin.value(), self.xmax_spin.value())
        self.ylim = (self.ymin_spin.value(), self.ymax_spin.value())
        self.baseline_coeffs = None
        self.result = None
        self.redraw(draw_preview=True)

    def create_peak(self, center: float, y_value: float) -> PeakSpec:
        mask = window_mask(self.x, self.xlim)
        y_window = self.y[mask] if np.any(mask) else self.y
        y_max = float(np.nanmax(y_window)) if y_window.size else max(float(y_value), 1.0)
        amplitude = max(float(y_value), y_max, 1.0)
        return PeakSpec(
            kind=self.kind_combo.currentText() or "Gaussian",
            center=ParamSpec(center, center - DEFAULT_CENTER_SPAN, center + DEFAULT_CENTER_SPAN),
            amplitude=ParamSpec(amplitude, 0.01, max(DEFAULT_AMPLITUDE_MAX, amplitude * 10.0)),
            sigma=ParamSpec(DEFAULT_SIGMA, 0.001, DEFAULT_SIGMA_MAX),
        )

    def on_press(self, event: MouseEvent) -> None:
        if event.inaxes not in (self.ax, self.center_ax) or event.xdata is None:
            return
        y_value = self.estimate_y_at(float(event.xdata)) if event.ydata is None else float(event.ydata)
        if self.has_shift_modifier(event):
            self.peaks.append(self.create_peak(float(event.xdata), y_value))
            self.selected_index = len(self.peaks) - 1
            self.result = None
            self.sync_controls()
            self.redraw()
            self.set_status(f"Added peak p{self.selected_index + 1}")
            return
        picked = self.pick_peak(event)
        if picked is not None:
            self.selected_index = picked
            self.drag_index = picked
            self.drag_from_center_axis = event.inaxes == self.center_ax
            self.sync_controls()
            self.redraw(draw_preview=False)

    @staticmethod
    def has_shift_modifier(event: MouseEvent) -> bool:
        key = str(event.key or "").lower()
        if "shift" in key:
            return True
        modifiers = QApplication.keyboardModifiers()
        return bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

    def on_motion(self, event: MouseEvent) -> None:
        if self.drag_index is None or event.inaxes not in (self.ax, self.center_ax) or event.xdata is None:
            return
        peak = self.peaks[self.drag_index]
        center = float(event.xdata)
        peak.center.value = center
        peak.center.minimum = center - DEFAULT_CENTER_SPAN
        peak.center.maximum = center + DEFAULT_CENTER_SPAN
        if not self.drag_from_center_axis and event.ydata is not None:
            peak.amplitude.value = max(float(event.ydata), 0.01)
        self.result = None
        self.sync_controls(update_table_only=True)
        self.redraw(draw_preview=False)

    def on_release(self, _event: MouseEvent) -> None:
        self.drag_index = None
        self.drag_from_center_axis = False
        self.sync_controls()

    def pick_peak(self, event: MouseEvent) -> int | None:
        if event.x is None or event.y is None:
            return None
        best_index = None
        best_distance = DRAG_PICK_RADIUS_PX
        for index, peak in enumerate(self.peaks):
            if event.inaxes == self.center_ax:
                px, py = self.center_ax.transData.transform((peak.center.value, 0.0))
            else:
                px, py = self.ax.transData.transform((peak.center.value, peak.amplitude.value))
            distance = float(np.hypot(px - event.x, py - event.y))
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def estimate_y_at(self, center: float) -> float:
        if not self.x.size:
            return 1.0
        return float(np.interp(center, self.x, self.y))

    def selected_peak(self) -> PeakSpec | None:
        if self.selected_index is None or not (0 <= self.selected_index < len(self.peaks)):
            return None
        return self.peaks[self.selected_index]

    def on_kind_change(self, label: str) -> None:
        if self.syncing_controls:
            return
        peak = self.selected_peak()
        if peak is None:
            return
        peak.kind = label
        if label == "Voigt":
            peak.extras.setdefault("gamma", ParamSpec(peak.sigma.value, 0.001, DEFAULT_SIGMA_MAX))
        elif label == "PseudoVoigt":
            peak.extras.setdefault("fraction", ParamSpec(0.5, 0.0, 1.0))
        for extra in list(peak.extras):
            if extra == "gamma" and label != "Voigt":
                del peak.extras[extra]
            if extra == "fraction" and label != "PseudoVoigt":
                del peak.extras[extra]
        self.result = None
        self.sync_controls()
        self.redraw()

    def on_param_changed(self, name: str) -> None:
        if self.syncing_controls:
            return
        peak = self.selected_peak()
        if peak is None:
            return
        params = peak.all_params()
        if name not in params:
            return
        value_edit, min_edit, max_edit = self.param_edits[name]
        spec = params[name]
        try:
            spec.value = value_edit.value()
            spec.minimum = min_edit.value()
            spec.maximum = max_edit.value()
        except ValueError:
            self.sync_controls()
            return
        self.result = None
        self.sync_controls(update_table_only=True)
        self.redraw()

    def on_table_selection(self) -> None:
        if self.syncing_controls:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        self.selected_index = rows[0].row()
        self.sync_controls()
        self.redraw(draw_preview=False)

    def delete_selected_peak(self) -> None:
        if self.selected_index is None:
            return
        if 0 <= self.selected_index < len(self.peaks):
            del self.peaks[self.selected_index]
        self.selected_index = min(self.selected_index, len(self.peaks) - 1) if self.peaks else None
        self.result = None
        self.sync_controls()
        self.redraw()

    def build_model_and_params(self) -> tuple[Any, Any]:
        model = md.LinearModel(prefix="bl_")
        for index, peak in enumerate(self.peaks, start=1):
            model += PEAK_MODELS[peak.kind](prefix=f"p{index}_")
        params = model.make_params()
        self.apply_baseline_params(params)
        for index, peak in enumerate(self.peaks, start=1):
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

    def apply_baseline_params(self, params: Any) -> None:
        slope, intercept = self.get_baseline_coeffs()
        vary = self.fit_baseline_box.isChecked()
        params["bl_intercept"].set(value=intercept, vary=vary)
        params["bl_slope"].set(value=slope, vary=vary)

    def get_baseline_coeffs(self) -> tuple[float, float]:
        if self.baseline_coeffs is not None:
            return self.baseline_coeffs
        x_fit, y_fit = self.baseline_data()
        try:
            slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
            self.baseline_coeffs = (float(slope), float(intercept))
        except Exception:
            intercept = float(np.nanmedian(y_fit)) if y_fit.size else 0.0
            self.baseline_coeffs = (0.0, intercept)
        return self.baseline_coeffs

    def baseline_data(self) -> tuple[np.ndarray, np.ndarray]:
        mask = baseline_edge_mask(self.x, self.xlim)
        if np.count_nonzero(mask) < 2:
            mask = window_mask(self.x, self.xlim)
        return self.x[mask], self.y[mask]

    def fit_data(self) -> tuple[np.ndarray, np.ndarray]:
        mask = window_mask(self.x, self.xlim)
        return self.x[mask], self.y[mask]

    def fit(self) -> None:
        if not self.peaks:
            self.set_status("Add at least one peak before fitting.")
            return
        try:
            self.fit_current_spectrum()
        except Exception as exc:
            self.set_status(f"Fit failed: {exc}")
            return
        self.update_peak_specs_from_result(sync=True)
        self.redraw(draw_preview=False)
        self.set_status(
            f"Fit complete: method={self.fit_method}, redchi={self.result.redchi:.4g}, nfev={self.result.nfev}"
        )

    def fit_current_spectrum(self) -> ModelResult:
        x_fit, y_fit = self.fit_data()
        if x_fit.size < 3:
            raise ValueError("Fit window has too few data points.")
        if not self.peaks:
            raise ValueError("Add at least one peak before fitting.")
        model, params = self.build_model_and_params()
        self.result = model.fit(y_fit, params, x=x_fit, method=self.fit_method)
        return self.result

    def update_peak_specs_from_result(self, sync: bool = False) -> None:
        if self.result is None:
            return
        for index, peak in enumerate(self.peaks, start=1):
            prefix = f"p{index}_"
            for name, spec in peak.all_params().items():
                full_name = prefix + name
                if full_name in self.result.params:
                    param = self.result.params[full_name]
                    spec.value = float(param.value)
                    spec.minimum = None if np.isneginf(param.min) else float(param.min)
                    spec.maximum = None if np.isposinf(param.max) else float(param.max)
                    spec.vary = bool(param.vary)
        if sync:
            self.sync_controls()

    def redraw(self, draw_preview: bool = True) -> None:
        self.ax.clear()
        self.center_ax.clear()
        x_plot, y_plot = decimate_for_plot(self.x, self.y)
        self.ax.plot(x_plot, y_plot, "-", color=COLOR_SPECTRUM, label="Spectrum", linewidth=1.15)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlabel("Raman shift (cm-1)")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True, color="#e5e7eb", linewidth=0.8)

        if self.result is not None:
            self.draw_fit_result()
        elif draw_preview and self.peaks:
            self.draw_initial_model()
        self.draw_peak_scale_bar()
        self.ax.legend(loc="upper right", fontsize=8)
        self.figure.suptitle(self.current_file.name)
        self.canvas.draw_idle()

    def draw_fit_result(self) -> None:
        if self.result is None:
            return
        x_fit, _ = self.fit_data()
        self.ax.plot(x_fit, self.result.best_fit, "-", color=COLOR_FIT, label="best fit", linewidth=1.5)
        components = self.result.eval_components(x=x_fit)
        for name, values in components.items():
            if name == "bl_":
                self.ax.plot(x_fit, values, "-", color=COLOR_BASELINE, label="baseline", linewidth=1.0)
            else:
                self.ax.plot(x_fit, values, "-", linewidth=0.9, label=name.rstrip("_"))

    def draw_initial_model(self) -> None:
        x_fit, _ = self.fit_data()
        if x_fit.size < 2:
            return
        model, params = self.build_model_and_params()
        try:
            values = model.eval(params, x=x_fit)
            components = model.eval_components(params, x=x_fit)
        except Exception:
            return
        self.ax.plot(x_fit, values, "-", color=COLOR_FIT, alpha=0.55, label="initial model")
        for name, component in components.items():
            if name == "bl_":
                self.ax.plot(x_fit, component, "-", color=COLOR_BASELINE, alpha=0.7, label="baseline")
            elif np.any(np.isfinite(component)):
                self.ax.plot(x_fit, component, "-", alpha=0.65, linewidth=0.8, label=name.rstrip("_"))

    def draw_peak_scale_bar(self) -> None:
        self.center_ax.set_xlim(self.xlim)
        self.center_ax.set_ylim(-1.0, 1.0)
        self.center_ax.set_yticks([])
        self.center_ax.set_xlabel("Peak centers")
        self.center_ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
        self.center_ax.spines["left"].set_visible(False)
        self.center_ax.spines["right"].set_visible(False)
        self.center_ax.spines["top"].set_visible(False)
        self.center_ax.axhline(0.0, color=COLOR_BASELINE, linewidth=0.8, alpha=0.75)
        for index, peak in enumerate(self.peaks):
            color = COLOR_SELECTED if index == self.selected_index else COLOR_MARKER
            linewidth = 2.0 if index == self.selected_index else 1.2
            self.center_ax.vlines(peak.center.value, -0.45, 0.45, color=color, linewidth=linewidth)
            self.center_ax.scatter(
                [peak.center.value], [0.0], marker="v", c=[color], edgecolors="white", s=70, zorder=5
            )
            self.center_ax.text(
                peak.center.value, 0.62, f"p{index + 1}", ha="center", va="bottom", fontsize=8, color=color
            )

    def sync_controls(self, update_table_only: bool = False) -> None:
        self.syncing_controls = True
        try:
            self.sync_table()
            if update_table_only:
                return
            peak = self.selected_peak()
            enabled_names = {"center", "amplitude", "sigma"}
            if peak is not None:
                enabled_names.update(peak.extras.keys())
                self.kind_combo.blockSignals(True)
                self.kind_combo.setCurrentText(peak.kind)
                self.kind_combo.blockSignals(False)
            for name, edits in self.param_edits.items():
                value_edit, min_edit, max_edit = edits
                active = peak is not None and name in enabled_names
                value_edit.setEnabled(active)
                min_edit.setEnabled(active)
                max_edit.setEnabled(active)
                if not active or peak is None:
                    value_edit.setValue(0.0)
                    min_edit.clear()
                    max_edit.clear()
                    continue
                spec = peak.all_params()[name]
                value_edit.blockSignals(True)
                value_edit.setValue(spec.value)
                value_edit.blockSignals(False)
                min_edit.set_optional_value(spec.minimum)
                max_edit.set_optional_value(spec.maximum)
        finally:
            self.syncing_controls = False

    def sync_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.peaks))
        for row, peak in enumerate(self.peaks):
            values = [
                f"p{row + 1}",
                peak.kind,
                f"{peak.center.value:.6g}",
                f"{peak.amplitude.value:.6g}",
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
        if self.selected_index is not None and 0 <= self.selected_index < len(self.peaks):
            self.table.selectRow(self.selected_index)
        else:
            self.table.clearSelection()
        self.table.blockSignals(False)

    def export(self) -> None:
        if self.result is None:
            self.set_status("Run fit before export.")
            return
        out_base = Path(__file__).resolve().parent / "out" / safe_name(self.current_file)
        out_dir = make_unique_dir(out_base)
        self.write_fit_outputs(out_dir)
        self.set_status(f"Exported to {out_dir}")

    def write_fit_outputs(self, out_dir: Path) -> None:
        if self.result is None:
            raise ValueError("Run fit before export.")
        out_dir.mkdir(parents=True, exist_ok=False)
        report_path = out_dir / "fit_report.txt"
        params_path = out_dir / "fit_params.csv"
        peak_params_path = out_dir / "peak_params.csv"
        peak_areas_path = out_dir / "peak_areas.csv"
        figure_path = out_dir / "fit_plot.png"
        report = f"method: {self.fit_method}\n\n{self.result.fit_report(min_correl=0.5)}"
        report_path.write_text(report, encoding="utf-8")
        self.write_params_csv(params_path)
        self.write_peak_params_csv(peak_params_path)
        self.write_peak_areas_csv(peak_areas_path)
        self.figure.savefig(figure_path, dpi=300)

    def process_folder(self) -> None:
        if not self.peaks:
            self.set_status("Add or import peaks before processing a folder.")
            return
        dirname = QFileDialog.getExistingDirectory(
            self,
            "Select folder with spectra",
            str(Path(__file__).resolve().parent / "in"),
        )
        if not dirname:
            return
        input_dir = Path(dirname)
        files = find_csv_files(input_dir)
        if not files:
            self.set_status(f"No CSV files found under {input_dir}")
            return

        original_file = self.current_file
        original_x = self.x
        original_y = self.y
        original_peaks = copy.deepcopy(self.peaks)
        original_selected_index = self.selected_index
        original_result = self.result
        original_baseline_coeffs = self.baseline_coeffs

        out_root_base = Path(__file__).resolve().parent / "out" / f"batch_{safe_name(input_dir)}"
        out_root = make_unique_dir(out_root_base)
        out_root.mkdir(parents=True, exist_ok=False)
        summary_path = out_root / "batch_summary.csv"
        rows: list[list[Any]] = [["file", "status", "message", "redchi", "nfev", "out_dir"]]
        ok_count = 0

        try:
            for index, path in enumerate(files, start=1):
                self.set_status(f"Processing {index}/{len(files)}: {path.name}")
                QApplication.processEvents()
                out_dir = make_unique_dir(out_root / safe_name(path))
                try:
                    self.current_file = path
                    self.x, self.y = self.load_spectrum_cached(path)
                    self.peaks = copy.deepcopy(original_peaks)
                    self.selected_index = None
                    self.result = None
                    self.baseline_coeffs = None
                    self.fit_current_spectrum()
                    self.update_peak_specs_from_result(sync=False)
                    self.redraw(draw_preview=False)
                    self.write_fit_outputs(out_dir)
                    rows.append(
                        [
                            str(path),
                            "ok",
                            "",
                            self.result.redchi if self.result is not None else "",
                            self.result.nfev if self.result is not None else "",
                            str(out_dir),
                        ]
                    )
                    ok_count += 1
                except Exception as exc:
                    rows.append([str(path), "failed", str(exc), "", "", ""])
        finally:
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerows(rows)
            self.current_file = original_file
            self.x = original_x
            self.y = original_y
            self.peaks = original_peaks
            self.selected_index = original_selected_index
            self.result = original_result
            self.baseline_coeffs = original_baseline_coeffs
            self.sync_controls()
            self.redraw(draw_preview=False)

        failed_count = len(files) - ok_count
        self.set_status(
            f"Batch complete: {ok_count} ok, {failed_count} failed. Results: {out_root}"
        )

    def write_params_csv(self, path: Path) -> None:
        if self.result is None:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "value", "stderr", "min", "max", "vary", "expr"])
            for name, param in self.result.params.items():
                writer.writerow(
                    [
                        name,
                        param.value,
                        param.stderr,
                        finite_bound(param.min),
                        finite_bound(param.max),
                        param.vary,
                        param.expr or "",
                    ]
                )

    def write_peak_areas_csv(self, path: Path) -> None:
        if self.result is None:
            return
        x_fit, _ = self.fit_data()
        components = self.result.eval_components(x=x_fit)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["peak", "kind", "center", "area", "x_min", "x_max"])
            for index, peak in enumerate(self.peaks, start=1):
                prefix = f"p{index}_"
                values = components.get(prefix)
                if values is None:
                    area = np.nan
                else:
                    area = float(np.trapezoid(values, x_fit))
                center_param = self.result.params.get(prefix + "center")
                center = center_param.value if center_param is not None else peak.center.value
                writer.writerow(
                    [
                        index,
                        peak.kind,
                        center,
                        area,
                        x_fit[0] if x_fit.size else "",
                        x_fit[-1] if x_fit.size else "",
                    ]
                )

    def export_peak_params(self) -> None:
        if not self.peaks:
            self.set_status("No peaks to export.")
            return
        default_dir = Path(__file__).resolve().parent / "out"
        default_dir.mkdir(parents=True, exist_ok=True)
        default_path = default_dir / f"{safe_name(self.current_file)}_peak_params.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export peak parameters",
            str(default_path),
            "CSV files (*.csv *.CSV);;All files (*.*)",
        )
        if not filename:
            return
        path = Path(filename)
        if path.suffix == "":
            path = path.with_suffix(".csv")
        try:
            self.write_peak_params_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Peak export failed", str(exc))
            return
        self.set_status(f"Exported peak parameters to {path}")

    def import_peak_params(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import peak parameters",
            str(Path(__file__).resolve().parent / "out"),
            "CSV files (*.csv *.CSV);;All files (*.*)",
        )
        if not filename:
            return
        path = Path(filename)
        try:
            peaks = self.read_peak_params_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Peak import failed", str(exc))
            return
        self.peaks = peaks
        self.selected_index = 0 if self.peaks else None
        self.result = None
        self.baseline_coeffs = None
        self.sync_controls()
        self.redraw()
        self.set_status(f"Imported {len(self.peaks)} peak(s) from {path}")

    def write_peak_params_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["peak", "kind", "param", "value", "min", "max", "vary"])
            for index, peak in enumerate(self.peaks, start=1):
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

    def read_peak_params_csv(self, path: Path) -> list[PeakSpec]:
        grouped: dict[int, dict[str, Any]] = {}
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"peak", "kind", "param", "value", "min", "max", "vary"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    "Peak CSV must contain columns: peak, kind, param, value, min, max, vary"
                )
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


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    files = find_csv_files(input_dir)

    app = QApplication.instance() or QApplication(sys.argv)
    try:
        initial_file = resolve_initial_file(input_dir, args.file, files).resolve()
    except FileNotFoundError as exc:
        QMessageBox.warning(
            None,
            "No input files",
            f"{exc}\n\nPut CSV files into:\n{input_dir}",
        )
        return 0

    if initial_file not in files:
        files.append(initial_file)
        files.sort()

    window = DeconvolutionWindow(files, initial_file, tuple(args.xlim), tuple(args.ylim), args.method)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
