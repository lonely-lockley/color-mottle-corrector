#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "mplconfig")
from matplotlib import cm
from scipy import ndimage as ndi
from skimage import color, filters, morphology
from tifffile import TiffFile, imread, imwrite

from PyQt6.QtCore import QObject, QThread, Qt, QtMsgType, pyqtSignal, pyqtSlot, qInstallMessageHandler
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBox,
    QVBoxLayout,
    QWidget,
)


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


APP_LOGGER_NAME = "chroma_blotch"
LOG = logging.getLogger(APP_LOGGER_NAME)
QT_LOG = logging.getLogger(f"{APP_LOGGER_NAME}.qt")


def setup_logging(log_dir: Path, console_level: str = "INFO"):
    log_dir.mkdir(parents=True, exist_ok=True)
    info_path = log_dir / "chroma_blotch_info.log"
    debug_path = log_dir / "chroma_blotch_debug.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    info_handler = logging.FileHandler(info_path, encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(fmt)

    debug_handler = logging.FileHandler(debug_path, encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_handler.setFormatter(fmt)

    LOG.handlers.clear()
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False
    LOG.addHandler(info_handler)
    LOG.addHandler(debug_handler)
    LOG.addHandler(console_handler)

    QT_LOG.handlers.clear()
    QT_LOG.setLevel(logging.DEBUG)
    QT_LOG.propagate = False
    QT_LOG.addHandler(debug_handler)

    LOG.info("Logging initialized (console=%s)", console_level.upper())
    LOG.info("Info log: %s", info_path)
    LOG.debug("Debug log: %s", debug_path)


def qt_message_handler(msg_type, context, message):
    file_part = context.file if context is not None and context.file else "?"
    line_part = context.line if context is not None else 0
    func_part = context.function if context is not None and context.function else "?"
    text = f"{message} | {file_part}:{line_part} | {func_part}"
    if msg_type == QtMsgType.QtDebugMsg:
        QT_LOG.debug(text)
    elif msg_type == QtMsgType.QtInfoMsg:
        QT_LOG.debug("INFO: %s", text)
    elif msg_type == QtMsgType.QtWarningMsg:
        QT_LOG.debug("WARNING: %s", text)
    elif msg_type == QtMsgType.QtCriticalMsg:
        QT_LOG.debug("CRITICAL: %s", text)
    else:
        QT_LOG.debug("FATAL: %s", text)


def find_default_input(search_dir: Path) -> Optional[Path]:
    candidates = sorted(
        p
        for p in search_dir.glob("*.tif*")
        if "_equalized" not in p.stem.lower() and "_corrected" not in p.stem.lower()
    )
    if not candidates:
        return None

    preferred = [p for p in candidates if "starless" in p.name.lower()] + candidates

    seen = set()
    ordered = []
    for p in preferred:
        if p in seen:
            continue
        seen.add(p)
        ordered.append(p)

    for path in ordered:
        try:
            with TiffFile(path) as tf:
                if tf.pages[0].dtype == np.uint16:
                    return path
        except Exception:
            continue

    return None


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_equalized.tiff")


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.ndim == 3 and rgb.shape[-1] > 3:
        rgb = rgb[..., :3]

    if np.issubdtype(rgb.dtype, np.integer):
        rgb_float = rgb.astype(np.float32) / np.iinfo(rgb.dtype).max
    else:
        rgb_float = rgb.astype(np.float32)
        maxv = float(np.nanmax(rgb_float)) if rgb_float.size else 1.0
        if maxv > 1.0:
            rgb_float /= maxv

    rgb_float = np.nan_to_num(rgb_float, nan=0.0, posinf=1.0, neginf=0.0)
    return clamp01(rgb_float)


def robust_center(x: np.ndarray, mask: np.ndarray) -> float:
    vals = x[mask]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))


def masked_normalized_blur(x: np.ndarray, mask: np.ndarray, sigma: float, eps: float = 1e-6):
    mask_f = mask.astype(np.float32)
    x_masked = np.where(mask, x, 0.0)
    num = filters.gaussian(x_masked, sigma=sigma, preserve_range=True)
    den = filters.gaussian(mask_f, sigma=sigma, preserve_range=True)
    y = num / (den + eps)
    return y.astype(np.float32), den.astype(np.float32)


def build_base_background_mask(L: np.ndarray):
    L_low, L_high = np.percentile(L, [1, 99])
    L_norm = clamp01((L - L_low) / (L_high - L_low + 1e-8))

    l_med = np.median(L_norm)
    l_mad = np.median(np.abs(L_norm - l_med)) + 1e-6
    bright_thresh = min(1.0, l_med + 3.5 * l_mad)
    dark_thresh = max(0.0, l_med - 4.0 * l_mad)

    local_base = filters.gaussian(L_norm, sigma=3, preserve_range=True)
    local_contrast = np.abs(L_norm - local_base)
    contrast_thresh = np.percentile(local_contrast, 88)

    edge_map = filters.sobel(L_norm)
    edge_thresh = np.percentile(edge_map, 90)

    foreground_seed = (
        (L_norm > bright_thresh)
        | (L_norm < dark_thresh)
        | (local_contrast > contrast_thresh)
        | (edge_map > edge_thresh)
    )
    # Preserve old strict behavior (< threshold) when switching to max_size (<= threshold).
    foreground_seed = morphology.remove_small_objects(foreground_seed, max_size=127)

    foreground_expanded = morphology.dilation(foreground_seed, morphology.disk(10))
    background_mask = ~foreground_expanded
    background_mask = morphology.remove_small_objects(background_mask, max_size=511)
    background_mask = morphology.remove_small_holes(background_mask, max_size=2047)

    stats = {
        "median_L_norm": float(l_med),
        "mad_L_norm": float(l_mad),
        "bright_thresh": float(bright_thresh),
        "dark_thresh": float(dark_thresh),
        "contrast_thresh": float(contrast_thresh),
        "edge_thresh": float(edge_thresh),
        "background_ratio_pct": float(np.mean(background_mask) * 100.0),
    }

    return background_mask.astype(bool), L_norm.astype(np.float32), stats


def rgb_to_qpixmap(rgb: np.ndarray, width: int, height: int) -> QPixmap:
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    h, w, _ = rgb8.shape
    qimg = QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy()).scaled(
        max(32, width),
        max(32, height),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def heatmap_rgb(data: np.ndarray, mask: np.ndarray, cmap_name: str) -> np.ndarray:
    if np.any(mask):
        lim = np.percentile(np.abs(data[mask]), 98)
        lim = max(float(lim), 1e-6)
    else:
        lim = 1.0

    norm = clamp01((data / lim) * 0.5 + 0.5)
    rgba = cm.get_cmap(cmap_name)(norm)
    rgb = rgba[..., :3]

    bg = np.ones_like(rgb) * 0.08
    return np.where(mask[..., None], rgb, bg).astype(np.float32)


@dataclass
class PipelineData:
    input_path: Path
    output_path: Path
    rgb_raw: np.ndarray
    rgb_float: np.ndarray
    lab: np.ndarray
    L: np.ndarray
    a: np.ndarray
    b: np.ndarray
    base_background_mask: np.ndarray
    L_norm: np.ndarray
    range_soft: np.ndarray
    working_mask: np.ndarray
    RG_field: np.ndarray
    BY_field: np.ndarray
    apply_alpha: np.ndarray


class CurveWidget(QWidget):
    curveChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        self._ys = self._xs.copy()
        self._active_idx = -1

    def reset_curve(self):
        self._ys = self._xs.copy()
        self.curveChanged.emit()
        self.update()

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        return np.interp(values, self._xs, self._ys).astype(np.float32)

    def control_points(self):
        return self._xs.copy(), self._ys.copy()

    def _plot_rect(self):
        margin = 16
        return margin, margin, self.width() - 2 * margin, self.height() - 2 * margin

    def _to_widget(self, x: float, y: float):
        px, py, pw, ph = self._plot_rect()
        wx = int(px + x * pw)
        wy = int(py + (1.0 - y) * ph)
        return wx, wy

    def _from_widget(self, wx: int, wy: int):
        px, py, pw, ph = self._plot_rect()
        x = (wx - px) / max(pw, 1)
        y = 1.0 - (wy - py) / max(ph, 1)
        return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())

        px, py, pw, ph = self._plot_rect()
        painter.setPen(QPen(Qt.GlobalColor.gray, 1))
        painter.drawRect(px, py, pw, ph)

        painter.setPen(QPen(Qt.GlobalColor.darkGray, 1))
        for t in [0.25, 0.5, 0.75]:
            x, _ = self._to_widget(t, 0.0)
            painter.drawLine(x, py, x, py + ph)
            _, y = self._to_widget(0.0, t)
            painter.drawLine(px, y, px + pw, y)

        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        pts = [self._to_widget(float(x), float(y)) for x, y in zip(self._xs, self._ys)]
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])

        painter.setPen(QPen(Qt.GlobalColor.cyan, 2))
        for wx, wy in pts:
            painter.drawEllipse(wx - 4, wy - 4, 8, 8)

    def mousePressEvent(self, event):
        ex = event.position().x() if hasattr(event, "position") else event.x()
        ey = event.position().y() if hasattr(event, "position") else event.y()

        pts = [self._to_widget(float(x), float(y)) for x, y in zip(self._xs, self._ys)]
        d2 = [((ex - wx) ** 2 + (ey - wy) ** 2, i) for i, (wx, wy) in enumerate(pts)]
        d2.sort(key=lambda t: t[0])
        if d2 and d2[0][0] <= 12 * 12:
            self._active_idx = d2[0][1]

    def mouseMoveEvent(self, event):
        if self._active_idx < 0:
            return
        ex = event.position().x() if hasattr(event, "position") else event.x()
        ey = event.position().y() if hasattr(event, "position") else event.y()
        _, y = self._from_widget(int(ex), int(ey))
        self._ys[self._active_idx] = y
        self.curveChanged.emit()
        self.update()

    def mouseReleaseEvent(self, event):
        self._active_idx = -1


class FitImageLabel(QLabel):
    def __init__(self, placeholder: str):
        super().__init__(placeholder)
        self.placeholder = placeholder
        self._image: Optional[np.ndarray] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background: #1f1f1f; border: 1px solid #3a3a3a; color: #cfcfcf; }")
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, rgb: np.ndarray):
        self._image = rgb.astype(np.float32)
        self._refresh()

    def set_placeholder(self, text: Optional[str] = None):
        self._image = None
        self.setPixmap(QPixmap())
        self.setText(text if text is not None else self.placeholder)

    def _refresh(self):
        if self._image is None:
            return
        pix = rgb_to_qpixmap(self._image, self.width(), self.height())
        self.setPixmap(pix)
        self.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


class MaskWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        revision: int,
        L_norm: np.ndarray,
        curve_xs: np.ndarray,
        curve_ys: np.ndarray,
        invert_output: bool,
        blur_radius: float,
    ):
        super().__init__()
        self.revision = revision
        self.L_norm = L_norm
        self.curve_xs = curve_xs
        self.curve_ys = curve_ys
        self.invert_output = invert_output
        self.blur_radius = blur_radius

    @pyqtSlot()
    def run(self):
        t0 = time.perf_counter()
        try:
            h, w = self.L_norm.shape
            max_dim = max(h, w)
            scale = min(1.0, 2200.0 / float(max_dim))

            if scale < 0.999:
                L_work = ndi.zoom(self.L_norm, scale, order=1).astype(np.float32)
            else:
                L_work = self.L_norm

            range_soft = np.interp(L_work, self.curve_xs, self.curve_ys).astype(np.float32)
            if self.invert_output:
                range_soft = 1.0 - range_soft

            if self.blur_radius > 0:
                sigma = self.blur_radius * scale
                range_soft = filters.gaussian(range_soft, sigma=sigma, preserve_range=True)

            range_soft = clamp01(range_soft).astype(np.float32)
            working_mask = np.ones_like(range_soft, dtype=bool)

            if scale < 0.999:
                zoom_back = (h / float(range_soft.shape[0]), w / float(range_soft.shape[1]))
                range_soft = ndi.zoom(range_soft, zoom_back, order=1).astype(np.float32)
                range_soft = range_soft[:h, :w]

                working_mask = ndi.zoom(working_mask.astype(np.float32), zoom_back, order=0) > 0.5
                working_mask = working_mask[:h, :w]

            self.finished.emit(
                {
                    "revision": self.revision,
                    "range_soft": range_soft,
                    "working_mask": working_mask.astype(bool),
                    "range_min": float(np.min(range_soft)),
                    "range_max": float(np.max(range_soft)),
                    "range_mean": float(np.mean(range_soft)),
                    "working_mask_pct": float(np.mean(working_mask) * 100.0),
                    "elapsed": time.perf_counter() - t0,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class FieldsWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(
        self,
        revision: int,
        a: np.ndarray,
        b: np.ndarray,
        working_mask: np.ndarray,
        range_soft: np.ndarray,
        sigma_lo: int,
    ):
        super().__init__()
        self.revision = revision
        self.a = a
        self.b = b
        self.working_mask = working_mask
        self.range_soft = range_soft
        self.sigma_lo = sigma_lo

    @pyqtSlot()
    def run(self):
        t0 = time.perf_counter()
        def emit_progress(percent: int, text: str):
            self.progress.emit(int(np.clip(percent, 0, 100)), text)

        try:
            emit_progress(2, "Validating mask")
            if not np.any(self.working_mask):
                raise RuntimeError("Background mask is empty. Tune mask controls first.")

            sigma_hi = self.sigma_lo * 2
            emit_progress(5, f"Preparing scales sigma={self.sigma_lo}/{sigma_hi}")

            emit_progress(14, f"Blur A @ sigma={self.sigma_lo}")
            a_lo, den_lo = masked_normalized_blur(self.a, self.working_mask, self.sigma_lo)
            emit_progress(28, f"Blur B @ sigma={self.sigma_lo}")
            b_lo, _ = masked_normalized_blur(self.b, self.working_mask, self.sigma_lo)
            emit_progress(42, f"Blur A @ sigma={sigma_hi}")
            a_hi, den_hi = masked_normalized_blur(self.a, self.working_mask, sigma_hi)
            emit_progress(56, f"Blur B @ sigma={sigma_hi}")
            b_hi, _ = masked_normalized_blur(self.b, self.working_mask, sigma_hi)

            emit_progress(68, "Centering RG/BY components")
            rg_sign = 1.0
            RG_lo = rg_sign * (a_lo - robust_center(a_lo, self.working_mask))
            RG_hi = rg_sign * (a_hi - robust_center(a_hi, self.working_mask))
            BY_lo = b_lo - robust_center(b_lo, self.working_mask)
            BY_hi = b_hi - robust_center(b_hi, self.working_mask)

            def masked_corr(x, y, mask):
                xv = x[mask].astype(np.float64)
                yv = y[mask].astype(np.float64)
                xv -= xv.mean()
                yv -= yv.mean()
                denom = np.linalg.norm(xv) * np.linalg.norm(yv) + 1e-12
                return float(np.dot(xv, yv) / denom)

            emit_progress(76, "Checking scale stability")
            rg_corr = masked_corr(RG_lo, RG_hi, self.working_mask)
            by_corr = masked_corr(BY_lo, BY_hi, self.working_mask)
            thr = np.percentile(np.abs(RG_lo[self.working_mask]), 25)
            valid_rg = self.working_mask & (np.abs(RG_lo) >= thr)
            rg_sign_agree = (
                float(np.mean(np.sign(RG_lo[valid_rg]) == np.sign(RG_hi[valid_rg])) * 100.0)
                if np.any(valid_rg)
                else float("nan")
            )

            emit_progress(84, "Blending fields 32/64")
            w_lo = 0.85
            w_hi = 0.15
            RG_field = (w_lo * RG_lo + w_hi * RG_hi).astype(np.float32)
            BY_field = (w_lo * BY_lo + w_hi * BY_hi).astype(np.float32)

            emit_progress(92, "Building correction alpha")
            edge_soft_px = 28.0
            bg_dist = ndi.distance_transform_edt(self.working_mask)
            edge_falloff = clamp01(bg_dist / edge_soft_px).astype(np.float32)

            support_mix = w_lo * den_lo + w_hi * den_hi
            support_conf = clamp01((support_mix - 0.15) / 0.85).astype(np.float32)
            mask_apply = (1.0 - self.range_soft).astype(np.float32)

            apply_alpha = np.where(
                self.working_mask,
                edge_falloff * support_conf * mask_apply,
                0.0,
            ).astype(np.float32)

            elapsed = time.perf_counter() - t0
            emit_progress(98, "Finalizing output")
            self.finished.emit(
                {
                    "revision": self.revision,
                    "sigma_lo": self.sigma_lo,
                    "sigma_hi": sigma_hi,
                    "RG_field": RG_field,
                    "BY_field": BY_field,
                    "apply_alpha": apply_alpha,
                    "coverage": float(np.mean(apply_alpha > 0.5) * 100.0),
                    "rg_sign": rg_sign,
                    "rg_corr": rg_corr,
                    "by_corr": by_corr,
                    "rg_sign_agree_pct": rg_sign_agree,
                    "rg_std_lo": float(np.std(RG_lo[self.working_mask])),
                    "by_std_lo": float(np.std(BY_lo[self.working_mask])),
                    "rg_std_hi": float(np.std(RG_hi[self.working_mask])),
                    "by_std_hi": float(np.std(BY_hi[self.working_mask])),
                    "elapsed": elapsed,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class CorrectionWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        lab: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        RG_field: np.ndarray,
        BY_field: np.ndarray,
        apply_alpha: np.ndarray,
        rg_k: float,
        by_k: float,
    ):
        super().__init__()
        self.lab = lab
        self.a = a
        self.b = b
        self.RG_field = RG_field
        self.BY_field = BY_field
        self.apply_alpha = apply_alpha
        self.rg_k = rg_k
        self.by_k = by_k

    @pyqtSlot()
    def run(self):
        t0 = time.perf_counter()
        try:
            a_corr = self.a - self.rg_k * self.RG_field * self.apply_alpha
            b_corr = self.b - self.by_k * self.BY_field * self.apply_alpha

            lab_corr = self.lab.copy()
            lab_corr[..., 1] = a_corr
            lab_corr[..., 2] = b_corr

            rgb_corr = clamp01(color.lab2rgb(lab_corr).astype(np.float32))

            def robust_sigma(x, eps=1e-6):
                med = np.median(x)
                mad = np.median(np.abs(x - med)) + eps
                return float(1.4826 * mad)

            delta_a = robust_sigma((self.RG_field * self.apply_alpha).ravel())
            delta_b = robust_sigma((self.BY_field * self.apply_alpha).ravel())

            self.finished.emit(
                {
                    "rgb_corr": rgb_corr,
                    "elapsed": time.perf_counter() - t0,
                    "delta_rg_sigma": delta_a,
                    "delta_by_sigma": delta_b,
                    "rg_k": float(self.rg_k),
                    "by_k": float(self.by_k),
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class BlotchEqualizerWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Blotch Equalizer")
        self.resize(1500, 920)
        self.step_titles = [
            "1. Source File",
            "2. Mask Controls",
            "3. Field Calculation",
            "4. Correction",
            "5. Save",
        ]
        self.step_dirty = [False] * len(self.step_titles)

        self.pipeline: Optional[PipelineData] = None
        self.current_corrected_rgb: Optional[np.ndarray] = None
        self.base_mask_stats = {}

        self.mask_ready = False
        self.mask_running = False
        self.mask_revision = 0

        self.fields_ready = False
        self.fields_preview_available = False
        self.fields_running = False
        self.fields_revision = 0

        self.correction_ready = False
        self.correction_preview_available = False
        self.correction_running = False

        self.mask_thread: Optional[QThread] = None
        self.mask_worker: Optional[MaskWorker] = None
        self.fields_thread: Optional[QThread] = None
        self.fields_worker: Optional[FieldsWorker] = None
        self.corr_thread: Optional[QThread] = None
        self.corr_worker: Optional[CorrectionWorker] = None
        self._active_threads: set[QThread] = set()

        self._build_ui()
        self._init_paths(args.input, args.output)
        LOG.info("UI initialized")

    def _style_primary_action_button(self, button: QPushButton):
        button.setProperty("primaryAction", True)
        button.setMinimumHeight(44)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _refresh_step_titles(self):
        for i, base in enumerate(self.step_titles):
            suffix = " *" if self.step_dirty[i] else ""
            self.toolbox.setItemText(i, f"{base}{suffix}")

    def _set_step_dirty(self, idx: int, dirty: bool):
        if idx < 0 or idx >= len(self.step_dirty):
            return
        if self.step_dirty[idx] == dirty:
            return
        self.step_dirty[idx] = dirty
        self._refresh_step_titles()

    def _mark_steps_dirty_from(self, start_idx: int):
        changed = False
        for i in range(max(0, start_idx), len(self.step_dirty)):
            if not self.step_dirty[i]:
                self.step_dirty[i] = True
                changed = True
        if changed:
            self._refresh_step_titles()

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # macOS-like accent styling for primary action buttons on each step.
        self.setStyleSheet(
            """
            QPushButton[primaryAction="true"] {
                background-color: #1E88E5;
                color: #ffffff;
                border: 1px solid #1E88E5;
                border-radius: 8px;
                font-weight: 600;
                padding: 8px 14px;
            }
            QPushButton[primaryAction="true"]:hover {
                background-color: #2791EF;
                border-color: #2791EF;
            }
            QPushButton[primaryAction="true"]:pressed {
                background-color: #1976D2;
                border-color: #1976D2;
            }
            QPushButton[primaryAction="true"]:disabled {
                background-color: #84BFF1;
                border-color: #84BFF1;
                color: #EAF4FF;
            }
            """
        )

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.toolbox = QToolBox()
        left_layout.addWidget(self.toolbox)

        self._build_step1()
        self._build_step2()
        self._build_step3()
        self._build_step4()
        self._build_step5()

        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)

        self.original_view = self._make_image_label("Source image")
        self.mask_view = self._make_image_label("Mask preview")
        self.corrected_view = self._make_image_label("Corrected preview")

        self.fields_container = QWidget()
        fbox = QVBoxLayout(self.fields_container)
        fbox.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        self.rg_view = self._make_image_label("RG field")
        self.by_view = self._make_image_label("BY field")
        row.addWidget(self.rg_view)
        row.addWidget(self.by_view)
        fbox.addLayout(row)

        self.right_layout.addWidget(self.original_view)
        self.right_layout.addWidget(self.mask_view)
        self.right_layout.addWidget(self.fields_container)
        self.right_layout.addWidget(self.corrected_view)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        splitter.setSizes([450, 1050])

        root.addWidget(splitter)

        self.setCentralWidget(central)

        status = QStatusBar()
        status.setSizeGripEnabled(False)
        self.setStatusBar(status)
        self.status("Ready")

        self.toolbox.currentChanged.connect(self.on_step_changed)

    def _make_image_label(self, text: str) -> FitImageLabel:
        return FitImageLabel(text)

    def _build_step1(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.input_edit = QLineEdit()
        self.input_browse_btn = QPushButton("Browse")
        self.load_btn = QPushButton("Load")

        row = QWidget()
        r = QHBoxLayout(row)
        r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(self.input_edit)
        r.addWidget(self.input_browse_btn)
        self.input_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.input_browse_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._style_primary_action_button(self.load_btn)

        lay.addWidget(QLabel("Input 16-bit TIFF:"))
        lay.addWidget(row)
        lay.addWidget(self.load_btn)
        lay.addStretch(1)

        self.toolbox.addItem(page, self.step_titles[0])

        self.input_browse_btn.clicked.connect(self.on_browse_input)
        self.input_edit.editingFinished.connect(self.on_input_changed)
        self.load_btn.clicked.connect(self.on_load_clicked)

    def _build_step2(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        box = QGroupBox("Range Mask Controls")
        g = QVBoxLayout(box)

        self.curve = CurveWidget()
        self.reset_curve_btn = QPushButton("Reset Curve")
        self.apply_mask_btn = QPushButton("Apply mask")
        self.invert_check = QCheckBox("Invert output")

        blur_row = QWidget()
        br = QHBoxLayout(blur_row)
        br.setContentsMargins(0, 0, 0, 0)
        self.range_blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.range_blur_slider.setRange(0, 64)
        self.range_blur_slider.setValue(0)
        self.range_blur_value = QLabel("0")
        br.addWidget(QLabel("Mask blur radius:"))
        br.addWidget(self.range_blur_slider)
        br.addWidget(self.range_blur_value)

        g.addWidget(self.curve)
        g.addWidget(self.invert_check)
        g.addWidget(blur_row)
        g.addWidget(self.reset_curve_btn)
        g.addWidget(self.apply_mask_btn)
        self._style_primary_action_button(self.apply_mask_btn)

        lay.addWidget(box)
        lay.setStretch(0, 1)
        lay.addStretch(1)

        self.toolbox.addItem(page, self.step_titles[1])

        self.reset_curve_btn.clicked.connect(self.on_reset_curve)
        self.apply_mask_btn.clicked.connect(self.on_apply_mask_clicked)
        self.curve.curveChanged.connect(self.on_mask_controls_changed)
        self.invert_check.stateChanged.connect(self.on_mask_controls_changed)
        self.range_blur_slider.valueChanged.connect(self.on_range_blur_changed)

    def _build_step3(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        info = QLabel(
            "Build RG/BY fields from current mask.\n"
            "Press Preview to run. This can take a long time."
        )
        info.setWordWrap(True)

        sigma_row = QWidget()
        sr = QHBoxLayout(sigma_row)
        sr.setContentsMargins(0, 0, 0, 0)
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(8, 64)
        self.sigma_slider.setValue(32)
        self.sigma_value = QLabel("32")
        sr.addWidget(QLabel("Gaussian blur tile size:"))
        sr.addWidget(self.sigma_slider)
        sr.addWidget(self.sigma_value)

        self.fields_preview_btn = QPushButton("Preview")
        self._style_primary_action_button(self.fields_preview_btn)

        lay.addWidget(info)
        lay.addWidget(sigma_row)
        lay.addWidget(self.fields_preview_btn)
        lay.addStretch(1)

        self.toolbox.addItem(page, self.step_titles[2])

        self.sigma_slider.valueChanged.connect(self.on_sigma_changed)
        self.fields_preview_btn.clicked.connect(self.on_fields_preview)

    def _build_step4(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        rg_row = QWidget()
        r1 = QHBoxLayout(rg_row)
        r1.setContentsMargins(0, 0, 0, 0)
        self.rg_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.rg_strength_slider.setRange(0, 100)
        self.rg_strength_slider.setValue(70)
        self.rg_strength_value = QLabel("70")
        r1.addWidget(QLabel("RG strength (%):"))
        r1.addWidget(self.rg_strength_slider)
        r1.addWidget(self.rg_strength_value)

        by_row = QWidget()
        r2 = QHBoxLayout(by_row)
        r2.setContentsMargins(0, 0, 0, 0)
        self.by_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.by_strength_slider.setRange(0, 100)
        self.by_strength_slider.setValue(30)
        self.by_strength_value = QLabel("30")
        r2.addWidget(QLabel("BY strength (%):"))
        r2.addWidget(self.by_strength_slider)
        r2.addWidget(self.by_strength_value)

        lay.addWidget(rg_row)
        lay.addWidget(by_row)
        self.correction_preview_btn = QPushButton("Preview")
        self._style_primary_action_button(self.correction_preview_btn)
        lay.addWidget(self.correction_preview_btn)
        lay.addStretch(1)

        self.toolbox.addItem(page, self.step_titles[3])

        self.rg_strength_slider.valueChanged.connect(self.on_strength_changed)
        self.by_strength_slider.valueChanged.connect(self.on_strength_changed)
        self.correction_preview_btn.clicked.connect(self.on_correction_preview)

    def _build_step5(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.output_edit = QLineEdit()
        self.output_browse_btn = QPushButton("Browse")

        row = QWidget()
        r = QHBoxLayout(row)
        r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(self.output_edit)
        r.addWidget(self.output_browse_btn)
        self.output_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.output_browse_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        lay.addWidget(QLabel("Output TIFF:"))
        lay.addWidget(row)

        self.save_btn = QPushButton("Save")
        self._style_primary_action_button(self.save_btn)
        lay.addWidget(self.save_btn)
        lay.addStretch(1)

        self.toolbox.addItem(page, self.step_titles[4])

        self.output_browse_btn.clicked.connect(self.on_browse_output)
        self.save_btn.clicked.connect(self.on_save)

    def status(self, text: str):
        self.statusBar().showMessage(text)
        LOG.info("STATUS | %s", text)

    def _register_thread(self, thread: QThread):
        self._active_threads.add(thread)
        thread.finished.connect(lambda thr=thread: self._active_threads.discard(thr))

    def _stop_thread(self, thread: Optional[QThread]):
        if thread is None:
            return
        self._active_threads.discard(thread)
        if thread.isRunning():
            LOG.info("Stopping thread: %s", thread.objectName() or "worker")
            thread.requestInterruption()
            thread.quit()
            thread.wait()
        thread.deleteLater()

    def show_error(self, message: str):
        LOG.error(message)
        QMessageBox.critical(self, "Error", message)
        self.status(message)

    def _init_paths(self, input_arg: str, output_arg: str):
        input_path = Path(input_arg).expanduser().resolve() if input_arg else find_default_input(Path.cwd())
        if input_path is None:
            LOG.info("No default 16-bit TIFF found in %s", Path.cwd())
            self.status("No 16-bit TIFF found. Select source file.")
            self.toolbox.setCurrentIndex(0)
            self.on_step_changed(0)
            self._refresh_step_titles()
            return

        self.input_edit.setText(str(input_path))
        out = Path(output_arg).expanduser().resolve() if output_arg else default_output_path(input_path)
        self.output_edit.setText(str(out))
        LOG.info("Default input candidate: %s", input_path)
        LOG.info("Default output candidate: %s", out)

        self.original_view.set_placeholder("Select input and press Load.")
        self.mask_view.set_placeholder("Load source first.")
        self.rg_view.set_placeholder("Load source first.")
        self.by_view.set_placeholder("Load source first.")
        self.corrected_view.set_placeholder("Load source first.")
        self.status("Source preselected. Press Load to start.")
        self.toolbox.setCurrentIndex(0)
        self.on_step_changed(0)
        self._refresh_step_titles()

    def invalidate_fields(self):
        self.fields_revision += 1
        self.fields_ready = False
        self.correction_ready = False

    def load_input(self, path: Path):
        LOG.info("Loading input file: %s", path)
        try:
            rgb_raw = imread(path)
        except Exception as exc:
            self.show_error(f"Failed to read TIFF: {exc}")
            return

        if rgb_raw.dtype != np.uint16:
            self.show_error(f"Only 16-bit TIFF is supported right now. Got dtype={rgb_raw.dtype}")
            return

        try:
            rgb_float = normalize_rgb(rgb_raw)
            lab = color.rgb2lab(rgb_float).astype(np.float32)
        except Exception as exc:
            self.show_error(f"Failed to convert to Lab: {exc}")
            return

        L = lab[..., 0]
        a = lab[..., 1]
        b = lab[..., 2]

        base_background_mask, L_norm, base_stats = build_base_background_mask(L)
        self.base_mask_stats = base_stats

        self.pipeline = PipelineData(
            input_path=path,
            output_path=Path(self.output_edit.text().strip()),
            rgb_raw=rgb_raw,
            rgb_float=rgb_float,
            lab=lab,
            L=L,
            a=a,
            b=b,
            base_background_mask=base_background_mask,
            L_norm=L_norm,
            range_soft=np.ones_like(L_norm, dtype=np.float32),
            working_mask=base_background_mask.copy(),
            RG_field=np.zeros_like(L_norm, dtype=np.float32),
            BY_field=np.zeros_like(L_norm, dtype=np.float32),
            apply_alpha=np.zeros_like(L_norm, dtype=np.float32),
        )

        self.mask_ready = False
        self.fields_preview_available = False
        self.correction_preview_available = False
        self.correction_ready = False
        self.current_corrected_rgb = None
        self.invalidate_fields()
        self._set_step_dirty(0, False)
        self._mark_steps_dirty_from(1)
        self.update_original_view()
        self.update_fields_view()
        self.update_corrected_view()
        self.request_mask_recompute()

        LOG.info("Loaded: %s", path)
        LOG.info("Raw shape=%s, dtype=%s", rgb_raw.shape, rgb_raw.dtype)
        LOG.info("L range: %.3f .. %.3f", float(np.min(L)), float(np.max(L)))
        LOG.info("a range: %.3f .. %.3f", float(np.min(a)), float(np.max(a)))
        LOG.info("b range: %.3f .. %.3f", float(np.min(b)), float(np.max(b)))
        LOG.info(
            "median(L)=%.4f, MAD=%.4f",
            base_stats["median_L_norm"],
            base_stats["mad_L_norm"],
        )
        LOG.info(
            "bright_thresh=%.4f, dark_thresh=%.4f",
            base_stats["bright_thresh"],
            base_stats["dark_thresh"],
        )
        LOG.info(
            "contrast_thresh=%.4f, edge_thresh=%.4f",
            base_stats["contrast_thresh"],
            base_stats["edge_thresh"],
        )
        LOG.info("background pixels: %.2f%%", base_stats["background_ratio_pct"])

        self.status(f"Loaded: {path.name} | shape={rgb_raw.shape} | dtype={rgb_raw.dtype}")

    def on_browse_input(self):
        current = self.input_edit.text().strip() or str(Path.cwd())
        LOG.debug("Browse input from: %s", current)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 16-bit TIFF",
            str(Path(current).parent if Path(current).exists() else Path.cwd()),
            "TIFF files (*.tif *.tiff)",
        )
        if not path:
            return
        self.input_edit.setText(path)
        LOG.info("Input selected via dialog: %s", path)
        self.on_input_changed()

    def on_input_changed(self):
        raw = self.input_edit.text().strip()
        if not raw:
            return
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            self.show_error(f"Input file not found: {path}")
            return

        self.input_edit.setText(str(path))
        self.output_edit.setText(str(default_output_path(path)))
        self._set_step_dirty(0, True)
        LOG.info("Input path set: %s", path)
        LOG.info("Output path auto-set: %s", self.output_edit.text().strip())
        self.status("Input path updated. Press Load.")

    def on_load_clicked(self):
        raw = self.input_edit.text().strip()
        if not raw:
            self.show_error("Select input file first.")
            return
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            self.show_error(f"Input file not found: {path}")
            return

        self.load_input(path)

    def on_browse_output(self):
        current = self.output_edit.text().strip() or str(Path.cwd() / "output_equalized.tiff")
        LOG.debug("Browse output from: %s", current)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output TIFF",
            current,
            "TIFF files (*.tif *.tiff)",
        )
        if path:
            self.output_edit.setText(path)
            LOG.info("Output selected via dialog: %s", path)

    def on_reset_curve(self):
        self.curve.reset_curve()

    def on_apply_mask_clicked(self):
        LOG.info("Apply mask clicked (UI-only action).")
        self.status("Mask settings are already applied automatically.")

    def on_range_blur_changed(self, value: int):
        self.range_blur_value.setText(str(value))
        self.on_mask_controls_changed()

    def on_mask_controls_changed(self, *_):
        if self.pipeline is None:
            return
        self._mark_steps_dirty_from(1)
        LOG.debug(
            "Mask controls changed: invert=%s, blur=%s",
            self.invert_check.isChecked(),
            self.range_blur_slider.value(),
        )
        self.request_mask_recompute()

    def on_sigma_changed(self, value: int):
        self.sigma_value.setText(str(value))
        self.invalidate_fields()
        self._mark_steps_dirty_from(2)
        LOG.info("Field sigma slider changed: sigma_lo=%d sigma_hi=%d", value, value * 2)
        self.status("Gaussian tile size changed. RG/BY fields need new Preview.")

    def on_strength_changed(self):
        self.rg_strength_value.setText(str(self.rg_strength_slider.value()))
        self.by_strength_value.setText(str(self.by_strength_slider.value()))
        self.correction_ready = False
        self._mark_steps_dirty_from(3)
        LOG.info(
            "Correction strengths changed: RG=%d%% BY=%d%%",
            self.rg_strength_slider.value(),
            self.by_strength_slider.value(),
        )

    def request_mask_recompute(self):
        if self.pipeline is None:
            return

        self.mask_ready = False
        self.invalidate_fields()
        self.mask_revision += 1
        revision = self.mask_revision

        if self.mask_running:
            self.status("Mask recalculation queued...")
            return

        p = self.pipeline
        xs, ys = self.curve.control_points()
        invert_output = self.invert_check.isChecked()
        blur_radius = float(self.range_blur_slider.value())
        LOG.info(
            "Mask recompute request #%d | invert=%s | blur=%.2f | curve_ys=%s",
            revision,
            invert_output,
            blur_radius,
            np.array2string(ys, precision=3),
        )

        self.mask_thread = QThread(self)
        self.mask_thread.setObjectName("mask_worker")
        self._register_thread(self.mask_thread)
        self.mask_worker = MaskWorker(
            revision=revision,
            L_norm=p.L_norm.copy(),
            curve_xs=xs,
            curve_ys=ys,
            invert_output=invert_output,
            blur_radius=blur_radius,
        )
        self.mask_worker.moveToThread(self.mask_thread)

        self.mask_thread.started.connect(self.mask_worker.run)
        self.mask_worker.finished.connect(self.on_mask_finished)
        self.mask_worker.failed.connect(self.on_mask_failed)

        self.mask_worker.finished.connect(self.cleanup_mask_worker)
        self.mask_worker.failed.connect(self.cleanup_mask_worker)

        self.mask_running = True
        self.status("Recomputing mask...")
        self.mask_thread.start()

    @pyqtSlot(object)
    def on_mask_finished(self, result):
        self.mask_running = False
        if self.pipeline is None:
            return

        if result["revision"] != self.mask_revision:
            if self.mask_revision > result["revision"]:
                self.request_mask_recompute()
            return

        self.pipeline.range_soft = result["range_soft"]
        self.pipeline.working_mask = result["working_mask"]
        self.mask_ready = True
        self._set_step_dirty(1, False)
        self.invalidate_fields()

        if self.toolbox.currentIndex() == 1:
            self.update_mask_view()

        LOG.info(
            "Mask result #%d | elapsed=%.2fs | range[min/max/mean]=%.4f/%.4f/%.4f | mask=%.2f%%",
            result["revision"],
            result["elapsed"],
            result["range_min"],
            result["range_max"],
            result["range_mean"],
            result["working_mask_pct"],
        )
        self.status(f"Mask updated in {result['elapsed']:.1f}s. RG/BY fields need new Preview.")

        if self.mask_revision > result["revision"]:
            self.request_mask_recompute()

    @pyqtSlot(str)
    def on_mask_failed(self, message: str):
        self.mask_running = False
        self.mask_ready = False
        self.show_error(f"Mask recomputation failed: {message}")

    def cleanup_mask_worker(self, *_):
        worker = self.sender()
        thread = worker.thread() if isinstance(worker, QObject) else self.mask_thread
        if not isinstance(thread, QThread):
            thread = self.mask_thread

        self._stop_thread(thread)

        if worker is self.mask_worker:
            self.mask_worker = None
        if thread is self.mask_thread:
            self.mask_thread = None
        if isinstance(worker, QObject):
            worker.deleteLater()

    def on_fields_preview(self):
        if self.pipeline is None:
            self.show_error("Load source file first.")
            return
        if self.mask_running:
            self.show_error("Mask is being recalculated. Wait and run Preview again.")
            return
        if not self.mask_ready:
            self.show_error("Mask is not ready yet. Wait for mask recalculation to finish.")
            return
        if self.fields_running:
            self.status("Field calculation already running...")
            return

        p = self.pipeline
        if not np.any(p.working_mask):
            self.show_error("Working mask is empty. Adjust mask controls first.")
            return

        rev = self.fields_revision
        sigma_lo = int(self.sigma_slider.value())
        LOG.info(
            "Fields preview start | revision=%d | sigma_lo=%d sigma_hi=%d | mask=%.2f%%",
            rev,
            sigma_lo,
            sigma_lo * 2,
            float(np.mean(p.working_mask) * 100.0),
        )

        self.fields_thread = QThread(self)
        self.fields_thread.setObjectName("fields_worker")
        self._register_thread(self.fields_thread)
        self.fields_worker = FieldsWorker(
            revision=rev,
            a=p.a,
            b=p.b,
            working_mask=p.working_mask.copy(),
            range_soft=p.range_soft.copy(),
            sigma_lo=sigma_lo,
        )
        self.fields_worker.moveToThread(self.fields_thread)

        self.fields_thread.started.connect(self.fields_worker.run)
        self.fields_worker.finished.connect(self.on_fields_finished)
        self.fields_worker.failed.connect(self.on_fields_failed)
        self.fields_worker.progress.connect(self.on_fields_progress)

        self.fields_worker.finished.connect(self.cleanup_fields_worker)
        self.fields_worker.failed.connect(self.cleanup_fields_worker)

        self.fields_running = True
        self.fields_preview_btn.setEnabled(False)
        self.status("Calculating RG/BY fields...")
        self.fields_thread.start()

    @pyqtSlot(int, str)
    def on_fields_progress(self, percent: int, message: str):
        self.status(f"Calculating RG/BY fields... {percent}% | {message}")

    @pyqtSlot(object)
    def on_fields_finished(self, result):
        self.fields_running = False
        self.fields_preview_btn.setEnabled(True)

        if self.pipeline is None:
            return
        if result["revision"] != self.fields_revision:
            self.status("Outdated field result ignored.")
            return

        self.pipeline.RG_field = result["RG_field"]
        self.pipeline.BY_field = result["BY_field"]
        self.pipeline.apply_alpha = result["apply_alpha"]
        self.fields_ready = True
        self.fields_preview_available = True
        self.correction_ready = False
        self._set_step_dirty(2, False)

        LOG.info("RG axis sign convention: rg_sign=%+.0f", result["rg_sign"])
        LOG.info(
            "RG corr(%d,%d)=%.4f, RG sign agreement=%.2f%%",
            result["sigma_lo"],
            result["sigma_hi"],
            result["rg_corr"],
            result["rg_sign_agree_pct"],
        )
        LOG.info(
            "BY corr(%d,%d)=%.4f",
            result["sigma_lo"],
            result["sigma_hi"],
            result["by_corr"],
        )
        LOG.info(
            "Scale-space debug: sigma=%d std(RG)=%.4f std(BY)=%.4f | sigma=%d std(RG)=%.4f std(BY)=%.4f",
            result["sigma_lo"],
            result["rg_std_lo"],
            result["by_std_lo"],
            result["sigma_hi"],
            result["rg_std_hi"],
            result["by_std_hi"],
        )
        LOG.info("field blend: w32=0.85, w64=0.15")
        LOG.info("apply_alpha coverage (>0.5): %.2f%%", result["coverage"])

        self.status(
            f"Fields ready in {result['elapsed']:.1f}s | "
            f"sigma={result['sigma_lo']}/{result['sigma_hi']} | "
            f"coverage(alpha>0.5)={result['coverage']:.2f}%"
        )
        self.update_fields_view()

    @pyqtSlot(str)
    def on_fields_failed(self, message: str):
        self.fields_running = False
        self.fields_preview_btn.setEnabled(True)
        LOG.error("Field calculation failed: %s", message)
        self.show_error(f"Field calculation failed: {message}")

    def cleanup_fields_worker(self, *_):
        worker = self.sender()
        thread = worker.thread() if isinstance(worker, QObject) else self.fields_thread
        if not isinstance(thread, QThread):
            thread = self.fields_thread

        self._stop_thread(thread)

        if worker is self.fields_worker:
            self.fields_worker = None
        if thread is self.fields_thread:
            self.fields_thread = None
        if isinstance(worker, QObject):
            worker.deleteLater()

    def on_correction_preview(self):
        if self.pipeline is None:
            self.show_error("Load source file first.")
            return
        if self.mask_running:
            self.show_error("Mask is recalculating. Wait for completion.")
            return
        if self.fields_running:
            self.show_error("RG/BY field calculation is running. Wait for completion.")
            return
        if not self.fields_ready:
            self.show_error("Run Preview in step 3 first.")
            return
        if self.correction_running:
            self.status("Correction preview already running...")
            return

        rg_k = float(self.rg_strength_slider.value()) / 100.0
        by_k = float(self.by_strength_slider.value()) / 100.0
        p = self.pipeline
        LOG.info(
            "Correction preview start | RG=%.2f BY=%.2f | fields_ready=%s",
            rg_k,
            by_k,
            self.fields_ready,
        )

        self.corr_thread = QThread(self)
        self.corr_thread.setObjectName("correction_worker")
        self._register_thread(self.corr_thread)
        self.corr_worker = CorrectionWorker(
            lab=p.lab.copy(),
            a=p.a.copy(),
            b=p.b.copy(),
            RG_field=p.RG_field.copy(),
            BY_field=p.BY_field.copy(),
            apply_alpha=p.apply_alpha.copy(),
            rg_k=rg_k,
            by_k=by_k,
        )
        self.corr_worker.moveToThread(self.corr_thread)

        self.corr_thread.started.connect(self.corr_worker.run)
        self.corr_worker.finished.connect(self.on_correction_finished)
        self.corr_worker.failed.connect(self.on_correction_failed)

        self.corr_worker.finished.connect(self.cleanup_correction_worker)
        self.corr_worker.failed.connect(self.cleanup_correction_worker)

        self.correction_running = True
        self.correction_preview_btn.setEnabled(False)
        self.status("Building corrected preview...")
        self.corr_thread.start()

    @pyqtSlot(object)
    def on_correction_finished(self, result):
        self.correction_running = False
        self.correction_preview_btn.setEnabled(True)
        self.current_corrected_rgb = result["rgb_corr"]
        self.correction_preview_available = True
        self.correction_ready = True
        self._set_step_dirty(3, False)
        self._set_step_dirty(4, False)
        if self.toolbox.currentIndex() in (3, 4):
            self.update_corrected_view()
        LOG.info(
            "Correction preview done | elapsed=%.2fs | delta_rg_sigma=%.4f | delta_by_sigma=%.4f | RG=%.2f | BY=%.2f",
            result["elapsed"],
            result["delta_rg_sigma"],
            result["delta_by_sigma"],
            result["rg_k"],
            result["by_k"],
        )
        self.status(f"Correction preview ready in {result['elapsed']:.2f}s")

    @pyqtSlot(str)
    def on_correction_failed(self, message: str):
        self.correction_running = False
        self.correction_preview_btn.setEnabled(True)
        LOG.error("Correction preview failed: %s", message)
        self.show_error(f"Correction preview failed: {message}")

    def cleanup_correction_worker(self, *_):
        worker = self.sender()
        thread = worker.thread() if isinstance(worker, QObject) else self.corr_thread
        if not isinstance(thread, QThread):
            thread = self.corr_thread

        self._stop_thread(thread)

        if worker is self.corr_worker:
            self.corr_worker = None
        if thread is self.corr_thread:
            self.corr_thread = None
        if isinstance(worker, QObject):
            worker.deleteLater()

    def save_tiff(self, rgb_corr: np.ndarray, output_path: Path) -> Path:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        out = np.round(rgb_corr * np.iinfo(np.uint16).max).astype(np.uint16)
        imwrite(output_path, out)
        return output_path

    def on_save(self):
        if self.pipeline is None:
            self.show_error("Load source file first.")
            return
        if not self.fields_ready:
            self.show_error("Run Preview in step 3 before saving.")
            return
        if not self.correction_ready or self.current_corrected_rgb is None:
            self.show_error("Run Preview in step 4 before saving.")
            return

        try:
            rgb_corr = self.current_corrected_rgb
            out_path = Path(self.output_edit.text().strip())
            if not out_path.suffix:
                out_path = out_path.with_suffix(".tiff")
                self.output_edit.setText(str(out_path))
            if out_path.suffix.lower() not in {".tif", ".tiff"}:
                out_path = out_path.with_suffix(".tiff")
                self.output_edit.setText(str(out_path))

            saved = self.save_tiff(rgb_corr, out_path)
            LOG.info("Saved output TIFF: %s", saved)
            LOG.info("Saved shape=%s dtype=%s", rgb_corr.shape, np.uint16)
            self.status(f"Saved: {saved}")
            QMessageBox.information(self, "Saved", f"Saved: {saved}")
        except Exception as exc:
            LOG.exception("Save failed")
            self.show_error(f"Save failed: {exc}")

    def on_step_changed(self, idx: int):
        LOG.debug("Step changed: %d", idx + 1)
        self.original_view.setVisible(False)
        self.mask_view.setVisible(False)
        self.fields_container.setVisible(False)
        self.corrected_view.setVisible(False)

        if idx == 0:
            self.update_original_view()
            self.original_view.setVisible(True)
        elif idx == 1:
            self.update_mask_view()
            self.mask_view.setVisible(True)
        elif idx == 2:
            self.update_fields_view()
            self.fields_container.setVisible(True)
        elif idx in (3, 4):
            self.update_corrected_view()
            self.corrected_view.setVisible(True)

    def update_original_view(self):
        if self.pipeline is None:
            self.original_view.set_placeholder("Source image")
            return
        self.original_view.set_image(self.pipeline.rgb_float)

    def update_mask_view(self):
        if self.pipeline is None:
            self.mask_view.set_placeholder("Mask preview")
            return

        soft = self.pipeline.range_soft

        # Show soft grayscale mask driven by curve settings over the whole image.
        gray = soft.astype(np.float32)
        mask_rgb = np.repeat(gray[..., None], 3, axis=2)
        self.mask_view.set_image(mask_rgb)

    def update_fields_view(self):
        if self.pipeline is None:
            self.rg_view.set_placeholder("RG field")
            self.by_view.set_placeholder("BY field")
            return

        if not self.fields_preview_available:
            self.rg_view.set_placeholder("RG preview not available")
            self.by_view.set_placeholder("BY preview not available")
            return

        rg_rgb = heatmap_rgb(self.pipeline.RG_field, self.pipeline.working_mask, "RdYlGn_r")
        by_rgb = heatmap_rgb(self.pipeline.BY_field, self.pipeline.working_mask, "coolwarm")

        self.rg_view.set_image(rg_rgb)
        self.by_view.set_image(by_rgb)

    def update_corrected_view(self):
        if self.pipeline is None:
            self.corrected_view.set_placeholder("Corrected preview")
            return

        if not self.correction_preview_available or self.current_corrected_rgb is None:
            self.corrected_view.set_placeholder("Press Preview in step 4.")
            return
        self.corrected_view.set_image(self.current_corrected_rgb)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        idx = self.toolbox.currentIndex()
        self.on_step_changed(idx)

    def closeEvent(self, event):
        all_threads = set(self._active_threads)
        for thr in (self.mask_thread, self.fields_thread, self.corr_thread):
            if thr is not None:
                all_threads.add(thr)

        for thr in all_threads:
            if thr is not None and thr.isRunning():
                self.status("Waiting for background calculations to finish...")
                LOG.info("Waiting for active thread to finish: %s", thr.objectName() or "worker")
            self._stop_thread(thr)

        self.mask_thread = None
        self.fields_thread = None
        self.corr_thread = None
        self.mask_worker = None
        self.fields_worker = None
        self.corr_worker = None
        LOG.info("Application close")
        super().closeEvent(event)


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description="Standalone blotch equalizer (RG/BY background field correction)")
    parser.add_argument("--input", default="", help="Input 16-bit TIFF")
    parser.add_argument("--output", default="", help="Output TIFF path")
    parser.add_argument("--log-dir", default="logs", help="Directory for info/debug logs")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (file logs stay info/debug).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    setup_logging(Path(args.log_dir).expanduser().resolve(), console_level=args.log_level)
    qInstallMessageHandler(qt_message_handler)
    LOG.info("Application start")
    LOG.info("CWD: %s", Path.cwd())
    LOG.info("Args: input=%s output=%s log_dir=%s", args.input, args.output, args.log_dir)
    app = QApplication(sys.argv)
    win = BlotchEqualizerWindow(args)
    win.show()
    exit_code = app.exec()
    qInstallMessageHandler(None)
    LOG.info("Application exit with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
