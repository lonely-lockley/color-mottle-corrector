#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "mplconfig")
from matplotlib import cm
from scipy import ndimage as ndi
from skimage import color, filters, morphology
from tifffile import TiffFile, imwrite
try:
    from astropy.io import fits as astrofits
    ASTROPY_AVAILABLE = True
except Exception:
    astrofits = None
    ASTROPY_AVAILABLE = False

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, QtMsgType, pyqtSignal, pyqtSlot, qInstallMessageHandler
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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


def choose_blur_workers(max_workers: int = 4) -> int:
    cores = os.cpu_count()
    if cores is None:
        return 1
    return int(max(1, min(max_workers, cores)))


SUPPORTED_INPUT_SUFFIXES = {".tif", ".tiff", ".fit", ".fits", ".fts"}
TIFF_SUFFIXES = {".tif", ".tiff"}
FITS_SUFFIXES = {".fit", ".fits", ".fts"}
BIT_DEPTH_CHOICES = [
    ("8-bit int", "uint8"),
    ("16-bit int", "uint16"),
    ("32-bit float", "float32"),
]
BIT_DEPTH_CODE_TO_LABEL = {code: label for label, code in BIT_DEPTH_CHOICES}


def infer_image_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TIFF_SUFFIXES:
        return "tiff"
    if suffix in FITS_SUFFIXES:
        return "fits"
    raise RuntimeError(f"Unsupported file extension: {path.suffix}")


def bit_depth_code_from_dtype(dtype: np.dtype) -> Optional[str]:
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        if dt.itemsize == 4:
            return "float32"
        return None
    if np.issubdtype(dt, np.integer):
        if dt.itemsize <= 1:
            return "uint8"
        if dt.itemsize <= 2:
            return "uint16"
        return None
    return None


def bit_depth_label_from_dtype(dtype: np.dtype) -> Optional[str]:
    code = bit_depth_code_from_dtype(dtype)
    if code is None:
        return None
    return BIT_DEPTH_CODE_TO_LABEL[code]


def default_output_extension_for_format(fmt: str) -> str:
    return ".fits" if fmt == "fits" else ".tiff"


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
        for p in search_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        and "_equalized" not in p.stem.lower()
        and "_corrected" not in p.stem.lower()
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
            fmt = infer_image_format(path)
            if fmt == "tiff":
                with TiffFile(path) as tf:
                    code = bit_depth_code_from_dtype(tf.pages[0].dtype)
                    if code in {"uint8", "uint16", "float32"}:
                        return path
            elif fmt == "fits":
                if not ASTROPY_AVAILABLE:
                    continue
                # Avoid memmap here: scaled FITS data (BZERO/BSCALE) may fail with memmap.
                with astrofits.open(path, memmap=False) as hdul:  # type: ignore[union-attr]
                    image_hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
                    if image_hdu is None:
                        continue
                    bitpix = int(image_hdu.header.get("BITPIX", 0))
                    if bitpix in (8, 16, -32):
                        return path
        except Exception:
            continue

    return None


def default_output_path(input_path: Path) -> Path:
    suffix = ".fits" if input_path.suffix.lower() in FITS_SUFFIXES else ".tiff"
    return input_path.with_name(f"{input_path.stem}_equalized{suffix}")


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.ndim == 3 and rgb.shape[-1] not in (1, 3, 4) and rgb.shape[0] in (1, 3, 4):
        rgb = np.moveaxis(rgb, 0, -1)

    if rgb.ndim != 3:
        raise RuntimeError(f"Expected 2D/3D image, got shape={rgb.shape}")

    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    elif rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    elif rgb.shape[-1] != 3:
        raise RuntimeError(f"Expected RGB-like channels, got shape={rgb.shape}")

    if np.issubdtype(rgb.dtype, np.integer):
        rgb_float = rgb.astype(np.float32) / np.iinfo(rgb.dtype).max
    else:
        rgb_float = rgb.astype(np.float32)
        maxv = float(np.nanmax(rgb_float)) if rgb_float.size else 1.0
        if maxv > 1.0:
            rgb_float /= maxv

    rgb_float = np.nan_to_num(rgb_float, nan=0.0, posinf=1.0, neginf=0.0)
    return clamp01(rgb_float)


def _fits_require_astropy():
    if not ASTROPY_AVAILABLE:
        raise RuntimeError(
            "FITS support requires astropy. Install it with: pip install astropy"
        )


def _extract_tiff_metadata(path: Path) -> dict[str, Any]:
    with TiffFile(path) as tf:
        page = tf.pages[0]
        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        ru_tag = page.tags.get("ResolutionUnit")
        resolution = None
        if xres_tag is not None and yres_tag is not None:
            xv = xres_tag.value
            yv = yres_tag.value
            try:
                resolution = (float(xv[0]) / float(xv[1]), float(yv[0]) / float(yv[1]))
            except Exception:
                resolution = None

        compression = None
        try:
            compression = str(page.compression.name).lower()
        except Exception:
            compression = None

        description = page.description if isinstance(page.description, str) and page.description else None
        metadata = tf.imagej_metadata if getattr(tf, "is_imagej", False) else None
        photometric = "rgb"
        planarconfig = None
        try:
            planarconfig = str(page.planarconfig.name).lower()
        except Exception:
            planarconfig = None
        software = page.tags.get("Software").value if page.tags.get("Software") is not None else None

    return {
        "tiff_description": description,
        "tiff_metadata": metadata,
        "tiff_is_imagej": bool(metadata),
        "tiff_resolution": resolution,
        "tiff_resolutionunit": int(ru_tag.value) if ru_tag is not None else None,
        "tiff_compression": compression,
        "tiff_photometric": photometric,
        "tiff_planarconfig": planarconfig,
        "tiff_software": software,
    }


def read_image_for_pipeline(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    fmt = infer_image_format(path)

    if fmt == "tiff":
        with TiffFile(path) as tf:
            page = tf.pages[0]
            raw = page.asarray()
            input_dtype = np.dtype(page.dtype)
        bit_code = bit_depth_code_from_dtype(input_dtype)
        if bit_code is None:
            raise RuntimeError(f"Unsupported TIFF dtype: {input_dtype}. Expected 8/16-bit int or 32-bit float.")

        rgb_raw = raw
        rgb_float = normalize_rgb(rgb_raw)
        io_meta = {
            "source_format": "tiff",
            "source_bit_depth": bit_code,
            "source_dtype": str(input_dtype),
            **_extract_tiff_metadata(path),
        }
        return {
            "path": str(path),
            "rgb_raw": rgb_raw,
            "rgb_float": rgb_float,
            "input_format": "tiff",
            "input_bit_depth": bit_code,
            "input_dtype_str": str(input_dtype),
            "io_meta": io_meta,
        }

    _fits_require_astropy()
    # memmap=False is required for robust reads when FITS uses scaling keywords
    # (common for unsigned integer images written via BZERO/BSCALE).
    with astrofits.open(path, memmap=False) as hdul:  # type: ignore[union-attr]
        hdu_idx = None
        image_hdu = None
        for i, hdu in enumerate(hdul):
            if getattr(hdu, "data", None) is not None:
                hdu_idx = i
                image_hdu = hdu
                break
        if image_hdu is None or hdu_idx is None:
            raise RuntimeError("No image HDU with data found in FITS file.")

        bitpix = int(image_hdu.header.get("BITPIX", 0))
        if bitpix not in (8, 16, -32):
            raise RuntimeError(f"Unsupported FITS BITPIX={bitpix}. Expected 8, 16, or -32.")
        raw = np.asarray(image_hdu.data)

    input_dtype = np.dtype(raw.dtype)
    bit_code = "float32" if bitpix == -32 else ("uint8" if bitpix == 8 else "uint16")
    rgb_raw = raw
    rgb_float = normalize_rgb(rgb_raw)
    io_meta = {
        "source_format": "fits",
        "source_bit_depth": bit_code,
        "source_dtype": str(input_dtype),
        "fits_template_path": str(path),
        "fits_hdu_index": int(hdu_idx),
        "fits_bitpix": bitpix,
        "fits_original_shape": tuple(int(v) for v in raw.shape),
        "fits_channel_axis_first": bool(raw.ndim == 3 and raw.shape[0] in (3, 4) and raw.shape[-1] not in (3, 4)),
    }
    return {
        "path": str(path),
        "rgb_raw": rgb_raw,
        "rgb_float": rgb_float,
        "input_format": "fits",
        "input_bit_depth": bit_code,
        "input_dtype_str": str(input_dtype),
        "io_meta": io_meta,
    }


def encode_rgb_output(rgb_corr: np.ndarray, bit_depth_code: str) -> np.ndarray:
    x = clamp01(rgb_corr).astype(np.float32, copy=False)
    if bit_depth_code == "float32":
        return x.astype(np.float32, copy=False)
    if bit_depth_code == "uint16":
        return np.round(x * np.iinfo(np.uint16).max).astype(np.uint16)
    if bit_depth_code == "uint8":
        return np.round(x * np.iinfo(np.uint8).max).astype(np.uint8)
    raise RuntimeError(f"Unsupported output bit depth: {bit_depth_code}")


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


def quick_l_norm(L: np.ndarray) -> np.ndarray:
    L_low, L_high = np.percentile(L, [1, 99])
    return clamp01((L - L_low) / (L_high - L_low + 1e-8)).astype(np.float32)


def prepare_mask_work_luminance(L_norm: np.ndarray, max_dim: int = 2200):
    h, w = L_norm.shape
    max_side = max(h, w)
    scale = min(1.0, float(max_dim) / float(max_side))
    if scale < 0.999:
        L_work = ndi.zoom(L_norm, scale, order=1).astype(np.float32)
    else:
        L_work = L_norm.astype(np.float32, copy=False)
        scale = 1.0
    return L_work, float(scale)


def fit_array_to_shape(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ah, aw = arr.shape
    if (ah, aw) == (h, w):
        return arr.astype(np.float32, copy=False)

    out = np.empty((h, w), dtype=np.float32)
    hh = min(h, ah)
    ww = min(w, aw)
    out[:hh, :ww] = arr[:hh, :ww]

    if hh < h:
        out[hh:, :ww] = out[hh - 1 : hh, :ww]
    if ww < w:
        out[:hh, ww:] = out[:hh, ww - 1 : ww]
    if hh < h and ww < w:
        out[hh:, ww:] = out[hh - 1, ww - 1]

    return out


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
    input_format: str
    input_bit_depth: str
    input_dtype_str: str
    io_meta: dict[str, Any]
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
        self._drag_changed = False

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
        self._drag_changed = False

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
        if float(self._ys[self._active_idx]) != float(y):
            self._ys[self._active_idx] = y
            self._drag_changed = True
        self.update()

    def mouseReleaseEvent(self, event):
        if self._active_idx >= 0 and self._drag_changed:
            self.curveChanged.emit()
        self._active_idx = -1
        self._drag_changed = False


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


class LoadWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    @pyqtSlot()
    def run(self):
        t0 = time.perf_counter()

        def emit_progress(percent: int, text: str):
            self.progress.emit(int(np.clip(percent, 0, 100)), text)

        try:
            p = Path(self.path).expanduser().resolve()
            emit_progress(5, "Reading image from disk")
            loaded = read_image_for_pipeline(p)
            rgb_raw = loaded["rgb_raw"]
            rgb_float = loaded["rgb_float"]
            input_format = loaded["input_format"]
            input_bit_depth = loaded["input_bit_depth"]
            input_dtype_str = loaded["input_dtype_str"]
            io_meta = loaded["io_meta"]

            emit_progress(38, f"Input validated: {input_format.upper()} / {BIT_DEPTH_CODE_TO_LABEL[input_bit_depth]}")

            emit_progress(58, "Converting RGB -> Lab")
            lab = color.rgb2lab(rgb_float).astype(np.float32)
            L = lab[..., 0]
            a = lab[..., 1]
            b = lab[..., 2]

            emit_progress(82, "Preparing luminance normalization")
            L_norm = quick_l_norm(L)
            l_med = float(np.median(L_norm))
            l_mad = float(np.median(np.abs(L_norm - l_med)) + 1e-6)
            base_stats = {
                "median_L_norm": l_med,
                "mad_L_norm": l_mad,
                "bright_thresh": float(min(1.0, l_med + 3.5 * l_mad)),
                "dark_thresh": float(max(0.0, l_med - 4.0 * l_mad)),
                "contrast_thresh": 0.0,
                "edge_thresh": 0.0,
                "background_ratio_pct": 100.0,
            }

            emit_progress(98, "Finalizing load")
            self.finished.emit(
                {
                    "path": str(p),
                    "rgb_raw": rgb_raw,
                    "rgb_float": rgb_float,
                    "input_format": input_format,
                    "input_bit_depth": input_bit_depth,
                    "input_dtype_str": input_dtype_str,
                    "io_meta": io_meta,
                    "lab": lab,
                    "L": L,
                    "a": a,
                    "b": b,
                    "L_norm": L_norm,
                    "base_stats": base_stats,
                    "elapsed": time.perf_counter() - t0,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


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
        prepared_L_work: Optional[np.ndarray] = None,
        prepared_scale: float = 1.0,
    ):
        super().__init__()
        self.revision = revision
        self.L_norm = L_norm
        self.curve_xs = curve_xs
        self.curve_ys = curve_ys
        self.invert_output = invert_output
        self.blur_radius = blur_radius
        self.prepared_L_work = prepared_L_work
        self.prepared_scale = prepared_scale

    @pyqtSlot()
    def run(self):
        t0 = time.perf_counter()
        try:
            h, w = self.L_norm.shape

            t_prep = time.perf_counter()
            cache_used = self.prepared_L_work is not None
            cache_generated = False
            if cache_used:
                L_work = self.prepared_L_work
                scale = float(self.prepared_scale)
            else:
                L_work, scale = prepare_mask_work_luminance(self.L_norm)
                cache_generated = scale < 0.999
            prep_elapsed = time.perf_counter() - t_prep

            t_curve = time.perf_counter()
            lut_size = 4096
            lut_x = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
            lut_y = np.interp(lut_x, self.curve_xs, self.curve_ys).astype(np.float32)
            lut_idx = np.clip((L_work * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)
            range_soft = lut_y[lut_idx]
            if self.invert_output:
                range_soft = 1.0 - range_soft
            curve_elapsed = time.perf_counter() - t_curve

            t_blur = time.perf_counter()
            if self.blur_radius > 0:
                sigma = self.blur_radius * scale
                if sigma > 1e-3:
                    range_soft = ndi.gaussian_filter(range_soft, sigma=sigma, mode="nearest")
            blur_elapsed = time.perf_counter() - t_blur

            range_soft = clamp01(range_soft).astype(np.float32)

            t_up = time.perf_counter()
            if scale < 0.999:
                zoom_back = (h / float(range_soft.shape[0]), w / float(range_soft.shape[1]))
                range_soft = ndi.zoom(range_soft, zoom_back, order=1).astype(np.float32)
                range_soft = fit_array_to_shape(range_soft, (h, w))
            up_elapsed = time.perf_counter() - t_up

            working_mask = np.ones((h, w), dtype=bool)

            self.finished.emit(
                {
                    "revision": self.revision,
                    "range_soft": range_soft,
                    "working_mask": working_mask.astype(bool),
                    "range_min": float(np.min(range_soft)),
                    "range_max": float(np.max(range_soft)),
                    "range_mean": float(np.mean(range_soft)),
                    "working_mask_pct": float(np.mean(working_mask) * 100.0),
                    "mask_scale": float(scale),
                    "work_shape": tuple(int(v) for v in L_work.shape),
                    "cache_used": bool(cache_used),
                    "cache_generated": bool(cache_generated),
                    "prepared_L_work": L_work if cache_generated else None,
                    "prepared_scale": float(scale if cache_generated else 1.0),
                    "prep_elapsed": float(prep_elapsed),
                    "curve_elapsed": float(curve_elapsed),
                    "blur_elapsed": float(blur_elapsed),
                    "up_elapsed": float(up_elapsed),
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

            workers = choose_blur_workers(4)
            emit_progress(8, f"Starting parallel field calculation jobs ({workers} workers)")
            blur_jobs = [
                ("a_lo", self.a, self.sigma_lo, f"Blur A @ sigma={self.sigma_lo}"),
                ("b_lo", self.b, self.sigma_lo, f"Blur B @ sigma={self.sigma_lo}"),
                ("a_hi", self.a, sigma_hi, f"Blur A @ sigma={sigma_hi}"),
                ("b_hi", self.b, sigma_hi, f"Blur B @ sigma={sigma_hi}"),
            ]
            progress_marks = [20, 35, 50, 62]
            blur_results = {}

            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="field_blur") as pool:
                futures = {
                    pool.submit(masked_normalized_blur, channel, self.working_mask, sigma): (key, label)
                    for key, channel, sigma, label in blur_jobs
                }
                done_count = 0
                for future in as_completed(futures):
                    key, label = futures[future]
                    blur_results[key] = future.result()
                    done_count += 1
                    emit_progress(progress_marks[min(done_count - 1, len(progress_marks) - 1)], f"{label} done ({done_count}/4)")

            a_lo, den_lo = blur_results["a_lo"]
            b_lo, _ = blur_results["b_lo"]
            a_hi, den_hi = blur_results["a_hi"]
            b_hi, _ = blur_results["b_hi"]

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
        revision: int,
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
        self.revision = revision
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
                    "revision": self.revision,
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
        self.setWindowTitle("Chroma Blotch Corrector")
        self.resize(1500, 920)
        self.step_titles = [
            "1. Source File",
            "2. Mask Controls",
            "3. Field Calculation",
            "4. Correction",
            "5. Save",
        ]
        self.sigma_options = [4, 8, 16, 32, 64]
        self.step_dirty = [False] * len(self.step_titles)

        self.pipeline: Optional[PipelineData] = None
        self.current_corrected_rgb: Optional[np.ndarray] = None
        self.base_mask_stats = {}

        self.mask_ready = False
        self.mask_running = False
        self.mask_revision = 0
        self.mask_apply_pending = False
        self.mask_apply_auto = False
        self.mask_queue_notified = False
        self.auto_open_step2_after_mask = False
        self.mask_prepared_L_work: Optional[np.ndarray] = None
        self.mask_prepared_scale: float = 1.0
        self.mask_prepared_src_shape: Optional[tuple[int, int]] = None
        self.auto_target_step = 0

        self.fields_ready = False
        self.fields_preview_available = False
        self.fields_running = False
        self.fields_revision = 0

        self.correction_revision = 0
        self.correction_ready = False
        self.correction_preview_available = False
        self.correction_running = False

        self.load_running = False
        self.load_thread: Optional[QThread] = None
        self.load_worker: Optional[LoadWorker] = None

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

    def _style_large_action_button(self, button: QPushButton):
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
            if i == 2 and hasattr(self, "fields_next_btn"):
                self.fields_next_btn.setVisible(False)
            if i == 3 and hasattr(self, "correction_next_btn"):
                self.correction_next_btn.setVisible(False)
        if changed:
            self._refresh_step_titles()

    def _open_next_step(self, current_step_idx: int):
        next_idx = current_step_idx + 1
        if next_idx >= self.toolbox.count():
            return
        self.toolbox.setCurrentIndex(next_idx)
        LOG.info("Auto-opened next step: %d -> %d", current_step_idx + 1, next_idx + 1)

    def _clear_auto_target(self):
        if self.auto_target_step != 0:
            LOG.info("Auto-run target cleared (was step %d)", self.auto_target_step)
        self.auto_target_step = 0

    def _complete_mask_apply(self, auto: bool):
        self.mask_apply_pending = False
        self.mask_apply_auto = False
        self._set_step_dirty(1, False)
        if auto:
            LOG.info("Mask apply completed automatically.")
            self.status("Mask applied (auto).")
            return
        LOG.info("Apply mask completed.")
        self.status("Mask applied.")
        self._open_next_step(1)

    def _request_auto_run(self, target_step: int, reason: str):
        target_step = int(np.clip(target_step, 1, 5))
        prev = self.auto_target_step
        self.auto_target_step = max(self.auto_target_step, target_step)
        LOG.info(
            "Auto-run requested: reason=%s | target_step=%d | previous_target=%d",
            reason,
            self.auto_target_step,
            prev,
        )
        self._drive_auto_pipeline()

    def _drive_auto_pipeline(self):
        if self.auto_target_step == 0:
            return

        if self.load_running or self.mask_running or self.fields_running or self.correction_running:
            return

        target = self.auto_target_step

        if self.pipeline is None or self.step_dirty[0]:
            raw = self.input_edit.text().strip()
            if not raw:
                self._clear_auto_target()
                self.show_error("Select input file first.")
                return
            path = Path(raw).expanduser().resolve()
            if not path.exists():
                self._clear_auto_target()
                self.show_error(f"Input file not found: {path}")
                return
            self.status("Auto-run: loading source image...")
            self.load_input(path)
            return

        if target >= 2:
            if self.step_dirty[1]:
                if self.mask_ready:
                    self._complete_mask_apply(auto=True)
                    self._drive_auto_pipeline()
                    return
                self.mask_apply_pending = True
                self.mask_apply_auto = True
                self.status("Auto-run: applying mask...")
                self.request_mask_recompute()
                return
            if not self.mask_ready:
                self.mask_apply_pending = True
                self.mask_apply_auto = True
                self.status("Auto-run: recalculating mask...")
                self.request_mask_recompute()
                return

        if target >= 3 and (self.step_dirty[2] or not self.fields_ready):
            self.status("Auto-run: calculating RG/BY fields...")
            try:
                self._start_fields_preview()
            except Exception as exc:
                self._clear_auto_target()
                self.show_error(f"Field calculation failed: {exc}")
            return

        if target >= 4 and (self.step_dirty[3] or not self.correction_ready or self.current_corrected_rgb is None):
            self.status("Auto-run: building correction preview...")
            try:
                self._start_correction_preview()
            except Exception as exc:
                self._clear_auto_target()
                self.show_error(f"Correction preview failed: {exc}")
            return

        if target >= 5:
            self.status("Auto-run: saving result...")
            try:
                self._perform_save()
            except Exception as exc:
                LOG.exception("Auto-run save failed")
                self.show_error(f"Save failed: {exc}")
            self._clear_auto_target()
            return

        self._clear_auto_target()

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

        lay.addWidget(QLabel("Input image (TIFF/FITS, 8/16-bit int or 32-bit float):"))
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

        controls = QWidget()
        g = QVBoxLayout(controls)
        g.setContentsMargins(0, 0, 0, 0)

        self.curve = CurveWidget()
        self.reset_curve_btn = QPushButton("Reset Curve")
        self.apply_mask_btn = QPushButton("Apply mask")
        self.invert_check = QCheckBox("Invert output")

        blur_row = QWidget()
        br = QHBoxLayout(blur_row)
        br.setContentsMargins(0, 0, 0, 0)
        self.range_blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.range_blur_slider.setRange(0, 15)
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

        lay.addWidget(QLabel("Range Mask Controls"))
        lay.addWidget(controls)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        lay.addStretch(1)

        self.toolbox.addItem(page, self.step_titles[1])

        self.reset_curve_btn.clicked.connect(self.on_reset_curve)
        self.apply_mask_btn.clicked.connect(self.on_apply_mask_clicked)
        self.curve.curveChanged.connect(self.on_mask_controls_changed)
        self.invert_check.stateChanged.connect(self.on_mask_controls_changed)
        self.range_blur_slider.valueChanged.connect(self.on_range_blur_changed)
        self.range_blur_slider.sliderReleased.connect(self.on_mask_controls_changed)

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
        self.sigma_slider.setRange(0, len(self.sigma_options) - 1)
        self.sigma_slider.setSingleStep(1)
        self.sigma_slider.setPageStep(1)
        self.sigma_slider.setValue(self.sigma_options.index(32))
        self.sigma_value = QLabel("32")
        sr.addWidget(QLabel("Gaussian blur tile size:"))
        sr.addWidget(self.sigma_slider)
        sr.addWidget(self.sigma_value)

        self.fields_preview_btn = QPushButton("Calculate")
        self._style_large_action_button(self.fields_preview_btn)
        self.fields_next_btn = QPushButton("Next")
        self._style_primary_action_button(self.fields_next_btn)
        self.fields_next_btn.setVisible(False)

        lay.addWidget(info)
        lay.addWidget(sigma_row)
        lay.addWidget(self.fields_preview_btn)
        lay.addWidget(self.fields_next_btn)
        lay.addStretch(1)

        self.toolbox.addItem(page, self.step_titles[2])

        self.sigma_slider.valueChanged.connect(self.on_sigma_changed)
        self.fields_preview_btn.clicked.connect(self.on_fields_preview)
        self.fields_next_btn.clicked.connect(lambda: self._open_next_step(2))

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
        self._style_large_action_button(self.correction_preview_btn)
        self.correction_next_btn = QPushButton("Next")
        self._style_primary_action_button(self.correction_next_btn)
        self.correction_next_btn.setVisible(False)
        lay.addWidget(self.correction_preview_btn)
        lay.addWidget(self.correction_next_btn)
        lay.addStretch(1)

        self.toolbox.addItem(page, self.step_titles[3])

        self.rg_strength_slider.valueChanged.connect(self.on_strength_changed)
        self.by_strength_slider.valueChanged.connect(self.on_strength_changed)
        self.correction_preview_btn.clicked.connect(self.on_correction_preview)
        self.correction_next_btn.clicked.connect(lambda: self._open_next_step(3))

    def _build_step5(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.output_edit = QLineEdit()
        self.output_browse_btn = QPushButton("Browse")
        self.save_format_combo = QComboBox()
        self.save_bitdepth_combo = QComboBox()

        row = QWidget()
        r = QHBoxLayout(row)
        r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(self.output_edit)
        r.addWidget(self.output_browse_btn)
        self.output_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.output_browse_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        format_row = QWidget()
        fr = QHBoxLayout(format_row)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.addWidget(QLabel("Format:"))
        fr.addWidget(self.save_format_combo, 1)
        fr.addWidget(QLabel("Bit depth:"))
        fr.addWidget(self.save_bitdepth_combo, 1)

        self.save_format_combo.addItem("TIFF", "tiff")
        self.save_format_combo.addItem("FITS", "fits")
        self._populate_bitdepth_combo("tiff")

        lay.addWidget(QLabel("Output file:"))
        lay.addWidget(row)
        lay.addWidget(format_row)

        self.save_btn = QPushButton("Save")
        self._style_primary_action_button(self.save_btn)
        lay.addWidget(self.save_btn)
        lay.addStretch(1)

        self.toolbox.addItem(page, self.step_titles[4])

        self.output_browse_btn.clicked.connect(self.on_browse_output)
        self.output_edit.editingFinished.connect(self.on_output_changed)
        self.save_format_combo.currentIndexChanged.connect(self.on_save_format_changed)
        self.save_bitdepth_combo.currentIndexChanged.connect(self.on_save_bitdepth_changed)
        self.save_btn.clicked.connect(self.on_save)

    def _selected_save_format(self) -> str:
        return str(self.save_format_combo.currentData() or "tiff")

    def _selected_save_bit_depth(self) -> str:
        return str(self.save_bitdepth_combo.currentData() or "uint16")

    def _populate_bitdepth_combo(self, fmt: str, preferred_code: Optional[str] = None):
        prev_code = self._selected_save_bit_depth() if self.save_bitdepth_combo.count() > 0 else None
        keep_code = preferred_code or prev_code or "uint16"
        self.save_bitdepth_combo.blockSignals(True)
        self.save_bitdepth_combo.clear()
        for label, code in BIT_DEPTH_CHOICES:
            self.save_bitdepth_combo.addItem(label, code)
        idx = self.save_bitdepth_combo.findData(keep_code)
        if idx < 0:
            idx = self.save_bitdepth_combo.findData("uint16")
        if idx < 0:
            idx = 0
        self.save_bitdepth_combo.setCurrentIndex(idx)
        self.save_bitdepth_combo.blockSignals(False)

    def _sync_output_extension_to_format(self, mark_dirty: bool = True):
        text = self.output_edit.text().strip()
        if not text:
            return
        fmt = self._selected_save_format()
        allowed_suffixes = FITS_SUFFIXES if fmt == "fits" else TIFF_SUFFIXES
        suffix = default_output_extension_for_format(fmt)
        p = Path(text).expanduser().resolve()
        if p.suffix.lower() not in allowed_suffixes:
            p = p.with_suffix(suffix)
            self.output_edit.setText(str(p))
            if mark_dirty:
                self._set_step_dirty(4, True)
                self.status("Output extension updated for selected format.")

    def _set_save_defaults_from_loaded(self, input_format: str, input_bit_depth: str):
        self.save_format_combo.blockSignals(True)
        fmt_idx = self.save_format_combo.findData(input_format)
        self.save_format_combo.setCurrentIndex(fmt_idx if fmt_idx >= 0 else 0)
        self.save_format_combo.blockSignals(False)
        self._populate_bitdepth_combo(input_format, preferred_code=input_bit_depth)
        self._sync_output_extension_to_format(mark_dirty=False)

    def on_save_format_changed(self):
        fmt = self._selected_save_format()
        self._populate_bitdepth_combo(fmt)
        self._sync_output_extension_to_format(mark_dirty=True)
        self._set_step_dirty(4, True)
        LOG.info("Save format changed: %s", fmt)
        self.status("Save format changed.")

    def on_save_bitdepth_changed(self):
        code = self._selected_save_bit_depth()
        self._set_step_dirty(4, True)
        LOG.info("Save bit depth changed: %s", code)
        self.status("Save bit depth changed.")

    def status(self, text: str):
        self.statusBar().showMessage(text)
        LOG.info("STATUS | %s", text)

    def _set_load_controls_enabled(self, enabled: bool):
        self.load_btn.setEnabled(enabled)
        self.input_browse_btn.setEnabled(enabled)
        self.input_edit.setEnabled(enabled)

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
            LOG.info("No default supported TIFF/FITS found in %s", Path.cwd())
            self.status("No supported TIFF/FITS found. Select source file.")
            self.toolbox.setCurrentIndex(0)
            self.on_step_changed(0)
            self._refresh_step_titles()
            return

        self.input_edit.setText(str(input_path))
        out = Path(output_arg).expanduser().resolve() if output_arg else default_output_path(input_path)
        self.output_edit.setText(str(out))
        input_fmt = infer_image_format(input_path)
        self._set_save_defaults_from_loaded(input_fmt, "uint16")
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
        self.correction_revision += 1
        self.correction_ready = False

    def load_input(self, path: Path):
        if self.load_running:
            self.status("Loading is already running...")
            return

        path = path.expanduser().resolve()
        if not path.exists():
            self.show_error(f"Input file not found: {path}")
            return
        if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            self.show_error(f"Unsupported input extension: {path.suffix}. Use TIFF/FITS.")
            return

        self.load_thread = QThread(self)
        self.load_thread.setObjectName("load_worker")
        self._register_thread(self.load_thread)
        self.load_worker = LoadWorker(str(path))
        self.load_worker.moveToThread(self.load_thread)

        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.progress.connect(self.on_load_progress)
        self.load_worker.finished.connect(self.on_load_finished)
        self.load_worker.failed.connect(self.on_load_failed)
        self.load_worker.finished.connect(self.cleanup_load_worker)
        self.load_worker.failed.connect(self.cleanup_load_worker)

        self.load_running = True
        self._set_load_controls_enabled(False)
        self.status(f"Loading image: {path}")
        LOG.info("Loading input file: %s", path)
        self.load_thread.start()

    @pyqtSlot(int, str)
    def on_load_progress(self, percent: int, message: str):
        self.status(f"Loading image... {percent}% | {message}")

    @pyqtSlot(object)
    def on_load_finished(self, result):
        self.load_running = False
        self._set_load_controls_enabled(True)

        path = Path(result["path"])
        rgb_raw = result["rgb_raw"]
        rgb_float = result["rgb_float"]
        input_format = result["input_format"]
        input_bit_depth = result["input_bit_depth"]
        input_dtype_str = result["input_dtype_str"]
        io_meta = result["io_meta"]
        lab = result["lab"]
        L = result["L"]
        a = result["a"]
        b = result["b"]
        L_norm = result["L_norm"]
        base_stats = result["base_stats"]
        self.base_mask_stats = base_stats
        base_background_mask = np.ones_like(L_norm, dtype=bool)

        self.pipeline = PipelineData(
            input_path=path,
            output_path=Path(self.output_edit.text().strip()),
            input_format=input_format,
            input_bit_depth=input_bit_depth,
            input_dtype_str=input_dtype_str,
            io_meta=io_meta,
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
        self.mask_prepared_L_work = None
        self.mask_prepared_scale = 1.0
        self.mask_prepared_src_shape = None
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
        self._set_save_defaults_from_loaded(input_format, input_bit_depth)
        self.auto_open_step2_after_mask = self.auto_target_step == 0
        self.status("Image displayed. Recomputing mask in background...")
        QTimer.singleShot(0, self.request_mask_recompute)

        LOG.info("Loaded: %s", path)
        LOG.info("Input format=%s bit_depth=%s dtype=%s", input_format, input_bit_depth, input_dtype_str)
        LOG.info("Load elapsed: %.2fs", result["elapsed"])
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
        LOG.info("Base mask precompute during load is skipped; step-2 mask is built asynchronously.")

        self.status(
            f"Loaded: {path.name} | format={input_format.upper()} | "
            f"bit={BIT_DEPTH_CODE_TO_LABEL.get(input_bit_depth, input_bit_depth)} | dtype={rgb_raw.dtype}"
        )

    @pyqtSlot(str)
    def on_load_failed(self, message: str):
        self.load_running = False
        self.auto_open_step2_after_mask = False
        self._set_load_controls_enabled(True)
        self._clear_auto_target()
        LOG.error("Load failed: %s", message)
        self.show_error(f"Load failed: {message}")

    def cleanup_load_worker(self, *_):
        worker = self.sender()
        thread = worker.thread() if isinstance(worker, QObject) else self.load_thread
        if not isinstance(thread, QThread):
            thread = self.load_thread

        self._stop_thread(thread)

        if worker is self.load_worker:
            self.load_worker = None
        if thread is self.load_thread:
            self.load_thread = None
        if isinstance(worker, QObject):
            worker.deleteLater()

    def on_browse_input(self):
        current = self.input_edit.text().strip() or str(Path.cwd())
        LOG.debug("Browse input from: %s", current)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select TIFF/FITS image",
            str(Path(current).parent if Path(current).exists() else Path.cwd()),
            "Images (*.tif *.tiff *.fit *.fits *.fts)",
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
        try:
            self._set_save_defaults_from_loaded(infer_image_format(path), "uint16")
        except Exception:
            pass
        self._set_step_dirty(0, True)
        self._mark_steps_dirty_from(1)
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
        current = self.output_edit.text().strip() or str(Path.cwd() / f"output_equalized{default_output_extension_for_format(self._selected_save_format())}")
        LOG.debug("Browse output from: %s", current)
        fmt = self._selected_save_format()
        filt = "FITS files (*.fit *.fits *.fts)" if fmt == "fits" else "TIFF files (*.tif *.tiff)"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output image",
            current,
            filt,
        )
        if path:
            self.output_edit.setText(path)
            self._sync_output_extension_to_format(mark_dirty=False)
            self._set_step_dirty(4, True)
            LOG.info("Output selected via dialog: %s", path)
            self.status("Output path updated. Press Save to write result.")

    def on_output_changed(self):
        if self.pipeline is None:
            return
        self._set_step_dirty(4, True)
        LOG.info("Output path edited: %s", self.output_edit.text().strip())
        self.status("Output path updated. Press Save to write result.")

    def on_reset_curve(self):
        self.curve.reset_curve()

    def on_apply_mask_clicked(self):
        if self.pipeline is None:
            self._request_auto_run(2, "apply_mask_clicked_without_loaded_pipeline")
            return

        if self.mask_running:
            self.mask_apply_pending = True
            self.mask_apply_auto = False
            LOG.info("Apply mask clicked while mask is running; waiting for recalculation to finish.")
            self.status("Apply mask queued. Waiting for recalculation to finish...")
            return

        if not self.mask_ready:
            self.mask_apply_pending = True
            self.mask_apply_auto = False
            LOG.info("Apply mask clicked while mask is not ready; starting recalculation.")
            self.status("Applying mask: recalculating...")
            self.request_mask_recompute()
            return

        self._complete_mask_apply(auto=False)

    def on_range_blur_changed(self, value: int):
        self.range_blur_value.setText(str(value))

    def on_mask_controls_changed(self, *_):
        if self.pipeline is None:
            return
        self.mask_apply_pending = False
        self._mark_steps_dirty_from(1)
        LOG.debug(
            "Mask controls changed: invert=%s, blur=%s",
            self.invert_check.isChecked(),
            self.range_blur_slider.value(),
        )
        self.request_mask_recompute()

    def on_sigma_changed(self, value: int):
        sigma_lo = self.sigma_options[int(value)]
        self.sigma_value.setText(str(sigma_lo))
        self.invalidate_fields()
        self._mark_steps_dirty_from(2)
        LOG.info("Field sigma slider changed: sigma_lo=%d sigma_hi=%d", sigma_lo, sigma_lo * 2)
        self.status("Gaussian tile size changed. RG/BY fields need new Preview.")

    def on_strength_changed(self):
        self.rg_strength_value.setText(str(self.rg_strength_slider.value()))
        self.by_strength_value.setText(str(self.by_strength_slider.value()))
        self.correction_revision += 1
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
            if not self.mask_queue_notified:
                self.status("Mask recalculation queued...")
                self.mask_queue_notified = True
            return

        p = self.pipeline
        xs, ys = self.curve.control_points()
        invert_output = self.invert_check.isChecked()
        blur_radius = float(self.range_blur_slider.value())
        use_mask_cache = (
            self.mask_prepared_L_work is not None
            and self.mask_prepared_src_shape == p.L_norm.shape
        )
        LOG.info(
            "Mask recompute request #%d | invert=%s | blur=%.2f | cache=%s | curve_ys=%s",
            revision,
            invert_output,
            blur_radius,
            "hit" if use_mask_cache else "miss",
            np.array2string(ys, precision=3),
        )

        self.mask_thread = QThread(self)
        self.mask_thread.setObjectName("mask_worker")
        self._register_thread(self.mask_thread)
        self.mask_worker = MaskWorker(
            revision=revision,
            L_norm=p.L_norm,
            curve_xs=xs,
            curve_ys=ys,
            invert_output=invert_output,
            blur_radius=blur_radius,
            prepared_L_work=self.mask_prepared_L_work if use_mask_cache else None,
            prepared_scale=self.mask_prepared_scale if use_mask_cache else 1.0,
        )
        self.mask_worker.moveToThread(self.mask_thread)

        self.mask_thread.started.connect(self.mask_worker.run)
        self.mask_worker.finished.connect(self.on_mask_finished)
        self.mask_worker.failed.connect(self.on_mask_failed)

        self.mask_worker.finished.connect(self.cleanup_mask_worker)
        self.mask_worker.failed.connect(self.cleanup_mask_worker)

        self.mask_running = True
        self.mask_queue_notified = False
        self.status("Recomputing mask...")
        self.mask_thread.start()

    @pyqtSlot(object)
    def on_mask_finished(self, result):
        self.mask_running = False
        self.mask_queue_notified = False
        if self.pipeline is None:
            return

        if result["revision"] != self.mask_revision:
            if self.mask_revision > result["revision"]:
                self.request_mask_recompute()
            return

        self.pipeline.range_soft = result["range_soft"]
        self.pipeline.working_mask = result["working_mask"]
        self.mask_ready = True
        self.invalidate_fields()

        if result.get("cache_generated") and isinstance(result.get("prepared_L_work"), np.ndarray):
            self.mask_prepared_L_work = result["prepared_L_work"]
            self.mask_prepared_scale = float(result.get("prepared_scale", 1.0))
            self.mask_prepared_src_shape = self.pipeline.L_norm.shape
            LOG.info(
                "Mask cache prepared: work_shape=%s scale=%.4f",
                result.get("work_shape"),
                self.mask_prepared_scale,
            )

        if self.toolbox.currentIndex() == 1:
            self.update_mask_view()

        LOG.info(
            "Mask result #%d | elapsed=%.2fs | prep=%.3fs curve=%.3fs blur=%.3fs up=%.3fs | "
            "scale=%.4f work_shape=%s cache_used=%s | range[min/max/mean]=%.4f/%.4f/%.4f | mask=%.2f%%",
            result["revision"],
            result["elapsed"],
            result.get("prep_elapsed", 0.0),
            result.get("curve_elapsed", 0.0),
            result.get("blur_elapsed", 0.0),
            result.get("up_elapsed", 0.0),
            result.get("mask_scale", 1.0),
            result.get("work_shape"),
            result.get("cache_used", False),
            result["range_min"],
            result["range_max"],
            result["range_mean"],
            result["working_mask_pct"],
        )
        self.status(f"Mask updated in {result['elapsed']:.1f}s. RG/BY fields need new Preview.")

        if self.auto_open_step2_after_mask:
            self.auto_open_step2_after_mask = False
            self._open_next_step(0)

        if self.mask_apply_pending:
            self._complete_mask_apply(auto=self.mask_apply_auto)

        if self.mask_revision > result["revision"]:
            self.request_mask_recompute()
            return

        self._drive_auto_pipeline()

    @pyqtSlot(str)
    def on_mask_failed(self, message: str):
        self.mask_running = False
        self.mask_ready = False
        self.mask_apply_pending = False
        self.mask_apply_auto = False
        self.auto_open_step2_after_mask = False
        self.mask_queue_notified = False
        self._clear_auto_target()
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

    def _start_fields_preview(self):
        p = self.pipeline
        if p is None:
            raise RuntimeError("Pipeline is not loaded.")
        if not np.any(p.working_mask):
            raise RuntimeError("Working mask is empty. Adjust mask controls first.")

        rev = self.fields_revision
        sigma_lo = self.sigma_options[int(self.sigma_slider.value())]
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
        self.fields_next_btn.setVisible(False)
        self.status("Calculating RG/BY fields...")
        self.fields_thread.start()

    def on_fields_preview(self):
        self._request_auto_run(3, "step3_calculate_clicked")

    @pyqtSlot(int, str)
    def on_fields_progress(self, percent: int, message: str):
        self.status(f"Calculating RG/BY fields... {percent}% | {message}")

    @pyqtSlot(object)
    def on_fields_finished(self, result):
        self.fields_running = False
        self.fields_preview_btn.setEnabled(True)

        if self.pipeline is None:
            self._clear_auto_target()
            return
        if result["revision"] != self.fields_revision:
            self.status("Outdated field result ignored.")
            self._drive_auto_pipeline()
            return

        self.pipeline.RG_field = result["RG_field"]
        self.pipeline.BY_field = result["BY_field"]
        self.pipeline.apply_alpha = result["apply_alpha"]
        self.fields_ready = True
        self.fields_preview_available = True
        self.correction_revision += 1
        self.correction_ready = False
        self._set_step_dirty(2, False)
        self.fields_next_btn.setVisible(True)

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
        self._drive_auto_pipeline()

    @pyqtSlot(str)
    def on_fields_failed(self, message: str):
        self.fields_running = False
        self.fields_preview_btn.setEnabled(True)
        self.fields_next_btn.setVisible(False)
        self._clear_auto_target()
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

    def _start_correction_preview(self):
        p = self.pipeline
        if p is None:
            raise RuntimeError("Pipeline is not loaded.")
        rg_k = float(self.rg_strength_slider.value()) / 100.0
        by_k = float(self.by_strength_slider.value()) / 100.0
        rev = self.correction_revision
        LOG.info(
            "Correction preview start | revision=%d | RG=%.2f BY=%.2f | fields_ready=%s",
            rev,
            rg_k,
            by_k,
            self.fields_ready,
        )

        self.corr_thread = QThread(self)
        self.corr_thread.setObjectName("correction_worker")
        self._register_thread(self.corr_thread)
        self.corr_worker = CorrectionWorker(
            revision=rev,
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
        self.correction_next_btn.setVisible(False)
        self.status("Building corrected preview...")
        self.corr_thread.start()

    def on_correction_preview(self):
        self._request_auto_run(4, "step4_preview_clicked")

    @pyqtSlot(object)
    def on_correction_finished(self, result):
        self.correction_running = False
        self.correction_preview_btn.setEnabled(True)

        if result["revision"] != self.correction_revision:
            LOG.info(
                "Outdated correction result ignored: result_rev=%d current_rev=%d",
                result["revision"],
                self.correction_revision,
            )
            self.status("Outdated correction result ignored.")
            self._drive_auto_pipeline()
            return

        self.current_corrected_rgb = result["rgb_corr"]
        self.correction_preview_available = True
        self.correction_ready = True
        self._set_step_dirty(3, False)
        self.correction_next_btn.setVisible(True)
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
        self._drive_auto_pipeline()

    @pyqtSlot(str)
    def on_correction_failed(self, message: str):
        self.correction_running = False
        self.correction_preview_btn.setEnabled(True)
        self.correction_next_btn.setVisible(False)
        self._clear_auto_target()
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

    def _perform_save(self):
        if self.pipeline is None:
            raise RuntimeError("Load source file first.")
        if self.current_corrected_rgb is None:
            raise RuntimeError("Corrected preview is empty.")
        rgb_corr = self.current_corrected_rgb
        out_fmt = self._selected_save_format()
        out_bit = self._selected_save_bit_depth()
        out_path = Path(self.output_edit.text().strip())
        if not out_path.suffix:
            out_path = out_path.with_suffix(default_output_extension_for_format(out_fmt))
            self.output_edit.setText(str(out_path))
        expected_suffixes = FITS_SUFFIXES if out_fmt == "fits" else TIFF_SUFFIXES
        if out_path.suffix.lower() not in expected_suffixes:
            out_path = out_path.with_suffix(default_output_extension_for_format(out_fmt))
            self.output_edit.setText(str(out_path))

        saved = self.save_image(
            rgb_corr=rgb_corr,
            output_path=out_path,
            output_format=out_fmt,
            bit_depth_code=out_bit,
            io_meta=self.pipeline.io_meta,
        )
        LOG.info("Saved output image: %s", saved)
        LOG.info("Saved shape=%s format=%s bit_depth=%s", rgb_corr.shape, out_fmt, out_bit)
        self._set_step_dirty(4, False)
        self.status(f"Saved: {saved}")
        QMessageBox.information(self, "Saved", f"Saved: {saved}")

    def on_save(self):
        self._request_auto_run(5, "step5_save_clicked")

    def _save_tiff(self, rgb_corr: np.ndarray, output_path: Path, bit_depth_code: str, io_meta: dict[str, Any]) -> Path:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = encode_rgb_output(rgb_corr, bit_depth_code)

        kwargs: dict[str, Any] = {
            "photometric": "rgb",
            "planarconfig": "CONTIG",
        }

        if io_meta.get("source_format") == "tiff":
            resolution = io_meta.get("tiff_resolution")
            resolutionunit = io_meta.get("tiff_resolutionunit")
            compression = io_meta.get("tiff_compression")
            software = io_meta.get("tiff_software")
            if resolution is not None:
                kwargs["resolution"] = resolution
            if resolutionunit is not None:
                kwargs["resolutionunit"] = resolutionunit
            if compression and compression not in {"none", "uncompressed", "1"}:
                kwargs["compression"] = compression
            if software:
                kwargs["software"] = software
            if io_meta.get("tiff_is_imagej") and io_meta.get("tiff_metadata"):
                kwargs["imagej"] = True
                kwargs["metadata"] = io_meta["tiff_metadata"]
            elif io_meta.get("tiff_description"):
                kwargs["description"] = io_meta["tiff_description"]

        imwrite(output_path, out, **kwargs)
        return output_path

    def _save_fits(self, rgb_corr: np.ndarray, output_path: Path, bit_depth_code: str, io_meta: dict[str, Any]) -> Path:
        _fits_require_astropy()
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        out = encode_rgb_output(rgb_corr, bit_depth_code)
        if out.ndim == 3:
            # FITS expects image cube axes as (C, Y, X) for sane viewer behavior.
            # Keep source FITS axis order when we have a FITS template;
            # otherwise (e.g. TIFF->FITS) force channel-first.
            if io_meta.get("source_format") == "fits":
                if io_meta.get("fits_channel_axis_first"):
                    out = np.moveaxis(out, -1, 0)
            else:
                out = np.moveaxis(out, -1, 0)

        template_path = io_meta.get("fits_template_path")
        hdu_index = int(io_meta.get("fits_hdu_index", 0))
        if template_path and Path(template_path).exists():
            with astrofits.open(template_path, memmap=False) as hdul:  # type: ignore[union-attr]
                if hdu_index >= len(hdul):
                    raise RuntimeError(f"Template FITS HDU index out of range: {hdu_index}")
                hdul[hdu_index].data = out
                hdul.writeto(output_path, overwrite=True)
        else:
            hdu = astrofits.PrimaryHDU(out)  # type: ignore[union-attr]
            astrofits.HDUList([hdu]).writeto(output_path, overwrite=True)  # type: ignore[union-attr]
        return output_path

    def save_image(
        self,
        rgb_corr: np.ndarray,
        output_path: Path,
        output_format: str,
        bit_depth_code: str,
        io_meta: dict[str, Any],
    ) -> Path:
        if output_format == "fits":
            return self._save_fits(rgb_corr, output_path, bit_depth_code, io_meta)
        return self._save_tiff(rgb_corr, output_path, bit_depth_code, io_meta)

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
            self.corrected_view.set_placeholder("Preview not available")
            return
        self.corrected_view.set_image(self.current_corrected_rgb)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        idx = self.toolbox.currentIndex()
        self.on_step_changed(idx)

    def closeEvent(self, event):
        all_threads = set(self._active_threads)
        for thr in (self.load_thread, self.mask_thread, self.fields_thread, self.corr_thread):
            if thr is not None:
                all_threads.add(thr)

        for thr in all_threads:
            if thr is not None and thr.isRunning():
                self.status("Waiting for background calculations to finish...")
                LOG.info("Waiting for active thread to finish: %s", thr.objectName() or "worker")
            self._stop_thread(thr)

        self.load_thread = None
        self.mask_thread = None
        self.fields_thread = None
        self.corr_thread = None
        self.load_worker = None
        self.mask_worker = None
        self.fields_worker = None
        self.corr_worker = None
        LOG.info("Application close")
        super().closeEvent(event)


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description="Standalone blotch equalizer (RG/BY background field correction)")
    parser.add_argument("--input", default="", help="Input TIFF/FITS image")
    parser.add_argument("--output", default="", help="Output image path (TIFF/FITS)")
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
