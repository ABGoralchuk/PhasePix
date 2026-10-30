"""
PhasePix - SEM Phase Analyzer Streamlit Application.
Provides automated Cellpose deep learning segmentation, morphological quantification,
phase classification (air bubbles vs. fat particles), and comparative KDE size distribution.
"""

from typing import Iterable, Optional, Tuple
from pathlib import Path
import base64
import inspect
import io
import json
import math
import re
import time

# ---------------- CPU THREADING GUARD (v1.7.3) ----------------
# The strict MKL/OpenMP/OpenCV cap only exists for the Windows CPU-only fallback,
# where the oversubscribed pools freeze the backend for 10+ minutes on tiny
# images. On macOS (darwin), Linux and CUDA-enabled Windows machines both helpers
# below are complete no-ops, so OpenCV keeps its multi-threaded bilateralFilter /
# medianBlur / morphology kernels and PyTorch keeps a full-size intra-op pool.
# They still sit above every third-party import because the guarded path (and the
# PHASEPIX_FORCE_THREAD_CAP=1 override) must export OMP_NUM_THREADS /
# MKL_NUM_THREADS before numpy and torch read them at initialisation.
from cpu_thread_guard import apply_cpu_thread_env, apply_opencv_thread_limits  # noqa: E402

# No-op on macOS / CUDA machines (returns the native logical CPU count there);
# logs which threading mode is active.
CPU_THREADS = apply_cpu_thread_env()
# cv2.setNumThreads(0) only on the Windows CPU fallback; same log line reports it.
OPENCV_THREADS = apply_opencv_thread_limits()

import cv2  # noqa: E402  (must follow the CPU thread cap)
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import find_objects

# Internal application modules
from cellpose_model import get_cellpose_model, run_inference, get_device, detect_hardware, cached_segment_air_bubbles_auto, cached_segment_fat_in_bubbles_deepscan, cached_segment_crystallized_multiscale, smooth_internal_crystals
from analyzer import (
    extract_metrics,
    calculate_granulometry_metrics,
    parse_canvas_manual_objects,
    calculate_manual_object_metrics,
    snap_contour_to_edges,
    build_combined_training_mask,
    create_combined_training_mask,
    prepare_cellpose_training_package,
    compute_interfacial_adsorption_metrics,
    compute_matrix_coalescence_metrics,
)
from visualizer import (
    classify_by_size,
    plot_single_phase_distribution,
    compute_phase_histogram_table,
    PALETTE_COLORS,
    apply_footer_dimming,
    create_mask_overlay,
    render_selected_bubbles_inspection_overlay,
    render_coalescence_network_overlay,
)
from scale_detector import detect_scale_bar_pixels, detect_scale_bar

# Drawable canvas used by the "Manual Correction" tab. On the pinned Streamlit
# 1.39 this widget is fully native: `streamlit.elements.image.image_to_url` still
# exists with the positional signature the component expects, so the background
# micrograph is served through Streamlit's own MediaFileManager - no patching.
from streamlit_drawable_canvas import st_canvas


def safe_st_canvas(**kwargs):
    """Call ``st_canvas`` while tolerating keyword changes across releases.

    Only unsupported keyword *names* are removed (``display_toolbar`` is the one
    that differs between streams of the component). Every value - in particular
    ``background_image``, which stays the untouched PIL image handed in by the
    caller - is forwarded exactly as supplied, so Streamlit keeps encoding the
    micrograph natively through ``image_to_url``.
    """
    cleaned_kwargs = dict(kwargs)
    try:
        parameters = inspect.signature(st_canvas).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if parameters and not accepts_var_kwargs:
        cleaned_kwargs = {
            name: value
            for name, value in cleaned_kwargs.items()
            if name in parameters
        }
    for _ in range(4):
        try:
            return st_canvas(**cleaned_kwargs)
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if not match or match.group(1) not in cleaned_kwargs:
                raise
            cleaned_kwargs.pop(match.group(1))
    return st_canvas(**cleaned_kwargs)


# Streamlit 1.39 sizes images with ``use_column_width``; 1.40 renamed it to
# ``use_container_width`` for ``st.image``. Detect the supported keyword once so
# every ``st.image`` call below renders full-width on both APIs.
try:
    _FULL_WIDTH_IMAGE_KWARGS = (
        {"use_container_width": True}
        if "use_container_width" in inspect.signature(st.image).parameters
        else {"use_column_width": True}
    )
except (TypeError, ValueError):
    _FULL_WIDTH_IMAGE_KWARGS = {"use_column_width": True}


# ---------------- LIGHT MICROSCOPY CALIBRATION PRESETS ----------------
# Laboratory-validated objective coefficients for transmitted-light micrographs.
# Stored values: um_per_px (physical size of one pixel) and the reference
# stage-micrometer / scale-bar geometry used to derive it.
DEFAULT_LIGHT_OBJECTIVES = {
    "10x Objective": {"um_per_px": 0.72098, "scale_bar_um": 10.0, "scale_bar_px": 13.87},
    "10x + 1.6x Objective": {"um_per_px": 0.45998, "scale_bar_um": 10.0, "scale_bar_px": 21.74},
    "20x Objective": {"um_per_px": 0.35348, "scale_bar_um": 10.0, "scale_bar_px": 28.29},
    "20x + 1.6x Objective": {"um_per_px": 0.22989, "scale_bar_um": 10.0, "scale_bar_px": 43.50},
    "40x Objective": {"um_per_px": 0.17915, "scale_bar_um": 10.0, "scale_bar_px": 55.82},
    "40x + 1.6x Objective": {"um_per_px": 0.11662, "scale_bar_um": 10.0, "scale_bar_px": 85.75},
    "63x Objective": {"um_per_px": 0.11424, "scale_bar_um": 10.0, "scale_bar_px": 87.54},
    "63x + 1.6x Objective": {"um_per_px": 0.07420, "scale_bar_um": 10.0, "scale_bar_px": 134.77},
    "100x Objective": {"um_per_px": 0.07149, "scale_bar_um": 10.0, "scale_bar_px": 139.87},
    "100x + 1.6x Objective": {"um_per_px": 0.04667, "scale_bar_um": 10.0, "scale_bar_px": 214.28},
    "Custom Calibration": {"um_per_px": 0.200, "scale_bar_um": 10.0, "scale_bar_px": 50.0},
}

LIGHT_PRESETS_FILE = "light_calibration_presets.json"
# Resolve relative to the application root so presets survive any working directory.
LIGHT_PRESETS_PATH = Path(__file__).resolve().parent / LIGHT_PRESETS_FILE


def load_light_presets() -> dict:
    """
    Load light-microscopy objective presets from ``light_calibration_presets.json``.

    Defaults are always available; any user-saved / edited entries found in the
    JSON file override their built-in counterparts. Malformed entries or a
    corrupted file fall back silently to the built-in defaults so the UI never
    breaks during a session.

    Returns:
        dict: Mapping ``objective name -> {"um_per_px", "scale_bar_um", "scale_bar_px"}``.
    """
    presets = {name: dict(values) for name, values in DEFAULT_LIGHT_OBJECTIVES.items()}

    try:
        if LIGHT_PRESETS_PATH.exists():
            with open(LIGHT_PRESETS_PATH, "r", encoding="utf-8") as fh:
                stored = json.load(fh)
            if isinstance(stored, dict):
                for name, values in stored.items():
                    if not isinstance(values, dict):
                        continue
                    try:
                        um_per_px = float(values.get("um_per_px"))
                        scale_bar_um = float(values.get("scale_bar_um", 10.0))
                        scale_bar_px = float(values.get("scale_bar_px", 10.0 / um_per_px))
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
                    if um_per_px <= 0 or scale_bar_um <= 0 or scale_bar_px <= 0:
                        continue
                    presets[str(name)] = {
                        "um_per_px": um_per_px,
                        "scale_bar_um": scale_bar_um,
                        "scale_bar_px": scale_bar_px,
                    }
    except (OSError, json.JSONDecodeError):
        # Corrupted / unreadable file: keep the in-memory defaults.
        pass

    if "Custom Calibration" not in presets:
        presets["Custom Calibration"] = dict(DEFAULT_LIGHT_OBJECTIVES["Custom Calibration"])

    return presets


def save_light_presets(presets: dict) -> None:
    """
    Persist light-microscopy objective presets to ``light_calibration_presets.json``.

    Args:
        presets: Mapping ``objective name -> {"um_per_px", "scale_bar_um", "scale_bar_px"}``.

    Raises:
        OSError: If the JSON file cannot be written (caller surfaces the message in the UI).
    """
    with open(LIGHT_PRESETS_PATH, "w", encoding="utf-8") as fh:
        json.dump(presets, fh, indent=2, ensure_ascii=False)


# ---------------- AI PARAMETER ADVISOR + CUSTOM PARAMETER PRESETS ----------------
PARAMETER_PRESETS_FILE = "parameter_presets.json"
# Resolve relative to the application root so student recipes survive any cwd.
PARAMETER_PRESETS_PATH = Path(__file__).resolve().parent / PARAMETER_PRESETS_FILE

SEM_FOCUS_AIR = "🫧 Air Bubbles Only (Fast & Precise)"
SEM_FOCUS_TWO_PASS = "🫧+🟠 Both Air Bubbles & Fat Particles (Two-Pass)"
LIGHT_PROFILE_LIQUID = "💧 Liquid Fat (Smooth Globules)"
LIGHT_PROFILE_CRYSTALLIZED = "❄️ Crystallized Fat (Textured / Internal Crystals)"

# Stable widget-key defaults for every Section 4 / Section 5 control.
# They are used (a) as fallbacks when a control is hidden by the active
# profile/sample-type branch and (b) to snapshot "current settings" when the
# student saves a custom preset.
PARAMETER_WIDGET_DEFAULTS: dict = {
    "SEM": {
        "sem_sample_type_radio": "Whipped Emulsion",
        "sem_detection_focus_radio": SEM_FOCUS_AIR,
        "sem_min_air_um": 10.0,
        "sem_max_air_um": 100.0,
        "sem_min_fat_um": 0.20,
        "sem_max_fat_um": 5.0,
        "shared_flow_slider": 0.70,
        "shared_cellprob_slider": -1.0,
        "shared_fat_circ_slider": 0.60,
        "sem_air_circ_slider": 0.50,
        "sem_ads_circ_slider": 0.30,
        "sem_deepscan_checkbox": False,
        "shared_clahe_checkbox": True,
        "shared_model_type_select": "cyto2",
    },
    "Light": {
        "light_min_fat_size_input": 0.50,
        "light_max_fat_size_input": 50.0,
        "light_emulsion_profile_radio": LIGHT_PROFILE_CRYSTALLIZED,
        "light_expected_droplet_um": 9.0,
        "light_crystal_smooth_slider": 3,
        "light_multiscale_checkbox": True,
        "light_invert_bf_checkbox": False,
        "light_solidity_slider": 0.84,
        "shared_fat_circ_slider": 0.60,
        "shared_flow_slider": 0.70,
        "shared_cellprob_slider": -1.0,
        "shared_clahe_checkbox": False,
        "shared_model_type_select": "cyto2",
    },
}


def default_parameter_presets() -> dict:
    """
    Built-in laboratory recipes offered by the Parameter Preset Manager.

    Returns:
        dict: ``{"SEM": {preset_name: {widget_key: value}}, "Light": {...}}``.
    """
    return {
        "Light": {
            "💧 Liquid Emulsion (Standard 40x)": {
                "light_emulsion_profile_radio": LIGHT_PROFILE_LIQUID,
                "light_expected_droplet_um": 5.0,
                "light_min_fat_size_input": 0.50,
                "light_max_fat_size_input": 40.0,
                "light_crystal_smooth_slider": 0,
                "light_multiscale_checkbox": False,
                "light_invert_bf_checkbox": True,
                "light_solidity_slider": 0.75,
                "shared_fat_circ_slider": 0.68,
                "shared_flow_slider": 0.70,
                "shared_cellprob_slider": -1.0,
                "shared_clahe_checkbox": False,
            },
            "❄️ Crystallized Fat Emulsion (Multi-Scale 40x/63x)": {
                "light_emulsion_profile_radio": LIGHT_PROFILE_CRYSTALLIZED,
                "light_expected_droplet_um": 9.0,
                "light_min_fat_size_input": 0.50,
                "light_max_fat_size_input": 60.0,
                "light_crystal_smooth_slider": 3,
                "light_multiscale_checkbox": True,
                "light_invert_bf_checkbox": False,
                "light_solidity_slider": 0.84,
                "shared_fat_circ_slider": 0.60,
                "shared_flow_slider": 0.70,
                "shared_cellprob_slider": -1.0,
                "shared_clahe_checkbox": False,
            },
        },
        "SEM": {
            "🫧+🟠 Whipped Emulsion (Two-Pass + Deep-Scan)": {
                "sem_sample_type_radio": "Whipped Emulsion",
                "sem_detection_focus_radio": SEM_FOCUS_TWO_PASS,
                "sem_min_air_um": 5.0,
                "sem_max_air_um": 150.0,
                "sem_min_fat_um": 0.20,
                "sem_max_fat_um": 5.0,
                "shared_flow_slider": 0.75,
                "shared_cellprob_slider": -1.5,
                "shared_fat_circ_slider": 0.60,
                "sem_air_circ_slider": 0.50,
                "sem_ads_circ_slider": 0.30,
                "sem_deepscan_checkbox": True,
                "shared_clahe_checkbox": True,
            },
            "🫧 Whipped Emulsion (Fast Air Bubbles Preview)": {
                "sem_sample_type_radio": "Whipped Emulsion",
                "sem_detection_focus_radio": SEM_FOCUS_AIR,
                "sem_min_air_um": 10.0,
                "sem_max_air_um": 100.0,
                "sem_min_fat_um": 0.20,
                "sem_max_fat_um": 5.0,
                "shared_flow_slider": 0.70,
                "shared_cellprob_slider": -1.0,
                "shared_fat_circ_slider": 0.60,
                "sem_air_circ_slider": 0.50,
                "sem_ads_circ_slider": 0.30,
                "sem_deepscan_checkbox": False,
                "shared_clahe_checkbox": True,
            },
            "🟠 SEM Single-Phase Emulsion": {
                "sem_sample_type_radio": "Emulsion",
                "sem_detection_focus_radio": SEM_FOCUS_AIR,
                "sem_min_air_um": 10.0,
                "sem_max_air_um": 100.0,
                "sem_min_fat_um": 0.20,
                "sem_max_fat_um": 5.0,
                "shared_flow_slider": 0.70,
                "shared_cellprob_slider": -1.0,
                "shared_fat_circ_slider": 0.60,
                "sem_air_circ_slider": 0.50,
                "sem_ads_circ_slider": 0.30,
                "sem_deepscan_checkbox": False,
                "shared_clahe_checkbox": True,
            },
        },
    }


def load_parameter_presets() -> dict:
    """
    Load Section 4 / Section 5 widget recipes from ``parameter_presets.json``.

    A modality that is stored in ``parameter_presets.json`` is authoritative: the
    file holds the exact dictionary of active recipes, so presets deleted from the
    sidebar stay deleted across restarts. Modalities that were never written (or a
    missing / corrupted file) fall back to the built-in laboratory recipes.
    :func:`restore_builtin_parameter_presets` re-seeds the built-ins on demand.

    Returns:
        dict: ``{"SEM": {name: {widget_key: value}}, "Light": {...}}``.
    """
    presets = {
        modality: {name: dict(values) for name, values in recipes.items()}
        for modality, recipes in default_parameter_presets().items()
    }
    try:
        if PARAMETER_PRESETS_PATH.exists():
            with open(PARAMETER_PRESETS_PATH, "r", encoding="utf-8") as fh:
                stored = json.load(fh)
            if isinstance(stored, dict):
                for modality in ("SEM", "Light"):
                    if modality not in stored:
                        # Never written yet: keep the built-in laboratory recipes.
                        continue
                    recipes = stored.get(modality)
                    if not isinstance(recipes, dict):
                        continue
                    # The stored dictionary replaces the built-ins verbatim so that
                    # user deletions survive the next app start.
                    presets[modality] = {
                        str(name): dict(values)
                        for name, values in recipes.items()
                        if isinstance(values, dict)
                    }
    except (OSError, json.JSONDecodeError):
        pass
    return presets


def restore_builtin_parameter_presets(modality: str) -> dict:
    """
    Re-seed the built-in laboratory recipes for one modality.

    Custom presets saved by the operator are preserved; only missing built-in
    recipes are re-added (e.g. after they were deleted from the sidebar).

    Args:
        modality: ``"SEM"`` or ``"Light"``.

    Returns:
        dict: The updated ``modality -> {preset_name: {widget_key: value}}`` map,
        ready to be handed to :func:`save_parameter_presets`.
    """
    presets = load_parameter_presets()
    builtin_recipes = default_parameter_presets().get(modality, {})
    # Built-ins first (canonical order), then any user-only presets.
    merged = {name: dict(values) for name, values in builtin_recipes.items()}
    for name, values in presets.get(modality, {}).items():
        if name not in merged:
            merged[name] = dict(values)
    presets[modality] = merged
    return presets


def save_parameter_presets(presets: dict) -> None:
    """
    Persist Section 4 / Section 5 widget recipes to ``parameter_presets.json``.

    Args:
        presets: Mapping ``modality -> {preset_name: {widget_key: value}}``.

    Raises:
        OSError: If the JSON file cannot be written (surfaced by the caller).
    """
    with open(PARAMETER_PRESETS_PATH, "w", encoding="utf-8") as fh:
        json.dump(presets, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _json_safe_parameter(value):
    """Coerce numpy scalars into plain JSON-serialisable Python values."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def collect_parameter_snapshot(modality: str, live_values: Optional[dict] = None) -> dict:
    """
    Build a complete ``widget_key -> value`` recipe for the active modality.

    Hidden controls (e.g. crystal smoothing while the liquid profile is active)
    fall back to their canonical defaults so a saved preset can always be
    re-applied deterministically.

    Args:
        modality: ``"SEM"`` or ``"Light"``.
        live_values: Values read from the currently rendered sidebar controls.

    Returns:
        dict: Filtered, JSON-safe widget state for the modality.
    """
    snapshot = dict(PARAMETER_WIDGET_DEFAULTS.get(modality, {}))
    for key in list(st.session_state.keys()):
        if key in snapshot and st.session_state[key] is not None:
            snapshot[key] = st.session_state[key]
    for key, value in (live_values or {}).items():
        if key in snapshot and value is not None:
            snapshot[key] = value
    return {key: _json_safe_parameter(value) for key, value in snapshot.items()}


def build_default_parameter_state(
    is_light_mode: bool,
    light_profile: Optional[str] = None,
) -> dict:
    """
    Return the clean Section 4/5 defaults for the active modality and optical profile.

    Light mode follows the selected droplet profile (crystallized vs liquid), while
    SEM mode uses the canonical whipped-emulsion recipe. Used by the sidebar
    "🔄 Reset to Defaults" action so a student can always return to the validated
    starting point (e.g. 9.0 µm target, cyto2, 0.50-50.0 µm, smoothing 3,
    solidity 0.84 for crystallized fat).

    Args:
        is_light_mode: Whether Light Microscopy is active.
        light_profile: Current light-mode profile string (ignored in SEM mode).

    Returns:
        dict: JSON-safe ``widget_key -> default_value`` mapping.
    """
    modality = "Light" if is_light_mode else "SEM"
    defaults = dict(PARAMETER_WIDGET_DEFAULTS.get(modality, {}))

    if is_light_mode:
        crystallized = str(light_profile or LIGHT_PROFILE_CRYSTALLIZED).startswith("❄️")
        defaults.update(
            {
                "light_emulsion_profile_radio": (
                    LIGHT_PROFILE_CRYSTALLIZED if crystallized else LIGHT_PROFILE_LIQUID
                ),
                "light_expected_droplet_um": 9.0 if crystallized else 5.0,
                "light_crystal_smooth_slider": 3 if crystallized else 0,
                "light_multiscale_checkbox": bool(crystallized),
                "light_invert_bf_checkbox": not crystallized,
                "light_solidity_slider": 0.84 if crystallized else 0.72,
                "shared_fat_circ_slider": 0.60,
            }
        )

    return {key: _json_safe_parameter(value) for key, value in defaults.items()}


# ---------------- DUAL-MODALITY DIAGNOSTIC & RECOMMENDATION ENGINE ----------------
def _ellipse(size: int) -> np.ndarray:
    """Return an elliptical structuring element of the requested odd size."""
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _prepare_analysis_gray(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normalise an analysis image to 8-bit grayscale and limit its working size.

    Args:
        image: Raw analysis image (footer already excluded).

    Returns:
        Tuple[np.ndarray, float]: working grayscale image and the resize factor
        (``working_px / original_px``) needed to convert calibrated µm/px.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("Empty analysis image - load a micrograph first.")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    max_dim = int(max(arr.shape[:2]))
    if max_dim > 900:
        factor = 900.0 / float(max_dim)
        new_size = (
            max(1, int(round(arr.shape[1] * factor))),
            max(1, int(round(arr.shape[0] * factor))),
        )
        return cv2.resize(arr, new_size, interpolation=cv2.INTER_AREA), factor
    return arr, 1.0


def _component_areas(mask_u8: np.ndarray, min_area: int) -> list:
    """Return component areas that are plausibly objects (not the background)."""
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8.astype(np.uint8), 8)
    limit = 0.45 * float(mask_u8.size)
    areas = []
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if min_area <= area <= limit:
            areas.append(area)
    return areas


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Fill enclosed holes so dark refractive rims become solid droplet footprints."""
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def _remove_border_background(mask_u8: np.ndarray, min_fraction: float = 0.05) -> np.ndarray:
    """Drop large connected regions touching the frame edge (background phase)."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8.astype(np.uint8), 8)
    height, width = mask_u8.shape[:2]
    out = mask_u8.copy()
    for idx in range(1, n_labels):
        x, y, w, h, area = stats[idx]
        touches_border = x <= 1 or y <= 1 or (x + w) >= width - 1 or (y + h) >= height - 1
        if touches_border and area > min_fraction * float(mask_u8.size):
            out[labels == idx] = 0
    return out


def _distance_transform_diameters(mask_u8: np.ndarray, um_per_px: float) -> list:
    """Estimate object diameters (µm) from distance-transform local maxima."""
    dist = cv2.distanceTransform(mask_u8.astype(np.uint8), cv2.DIST_L2, 5)
    if float(dist.max()) <= 0:
        return []
    peaks = ((dist >= cv2.dilate(dist, _ellipse(5)) - 1e-3) & (dist > 1.0)).astype(np.uint8)
    n_peaks, labels = cv2.connectedComponents(peaks, 8)
    diameters = []
    for idx in range(1, n_peaks):
        radii = dist[labels == idx]
        if radii.size == 0:
            continue
        diameter_um = 2.0 * float(radii.max()) * float(um_per_px)
        if 0.15 <= diameter_um <= 2000.0:
            diameters.append(diameter_um)
    return diameters


def _watershed_droplet_diameters(
    mask_u8: np.ndarray,
    um_per_px: float,
    min_peak_separation_px: int = 6,
) -> Tuple[list, list]:
    """
    Split touching droplets and return (region diameters, peak diameters) in µm.

    Distance-transform maxima are used as watershed seeds (one seed per droplet
    lobe), which prevents a merged doublet from being reported as one giant
    droplet. Peak radii are converted with ``2 * r_px * um_per_px`` and kept as a
    fallback; the primary distribution uses each watershed region's area-equivalent
    diameter so D[2,1] is not inflated by merged lobes.

    Args:
        mask_u8: Binary (0/1 or 0/255) droplet mask.
        um_per_px: Physical size of one pixel (µm).
        min_peak_separation_px: Minimum spacing between accepted maxima.

    Returns:
        Tuple[list, list]: (watershed region diameters, local-maximum diameters),
        both filtered to 0.8-200 µm.
    """
    mask = (mask_u8 > 0).astype(np.uint8)
    if not np.any(mask):
        return [], []
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if float(dist.max()) <= 0:
        return [], []

    separation = max(3, int(min_peak_separation_px))
    local_max = cv2.dilate(dist, _ellipse(separation))
    peaks = ((dist >= local_max - 1e-3) & (dist > 1.0)).astype(np.uint8)
    n_peaks, peak_labels = cv2.connectedComponents(peaks, 8)
    if n_peaks <= 1:
        return [], []

    # Literal peak-radius diameter (2 * r_px * µm/px), filtered below 0.8 µm.
    peak_diameters = []
    for label in range(1, n_peaks):
        radii = dist[peak_labels == label]
        if radii.size == 0:
            continue
        diameter_um = 2.0 * float(radii.max()) * float(um_per_px)
        if 0.8 <= diameter_um <= 200.0:
            peak_diameters.append(diameter_um)

    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[mask == 0] = 1  # background seed
    markers[peaks > 0] = peak_labels[peaks > 0] + 1  # one seed per droplet lobe
    vis = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(vis, markers)

    diameters = []
    for region_id in range(2, n_peaks + 1):
        region = (markers == region_id) & (mask > 0)
        area = int(np.count_nonzero(region))
        if area < 4:
            continue
        diameter_um = 2.0 * math.sqrt(area / math.pi) * float(um_per_px)
        if 0.8 <= diameter_um <= 200.0:
            diameters.append(diameter_um)
    return diameters, peak_diameters


def _contour_diameters(mask_u8: np.ndarray, um_per_px: float) -> list:
    """Fallback equivalent-circle diameters (µm) from external contours."""
    contours, _ = cv2.findContours(mask_u8.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diameters = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 4.0:
            continue
        diameter_um = 2.0 * math.sqrt(area / math.pi) * float(um_per_px)
        if 0.15 <= diameter_um <= 2000.0:
            diameters.append(diameter_um)
    return diameters


def _floor_to_step(value: float, step: float) -> float:
    """Round ``value`` DOWN to the nearest multiple of ``step`` (artifact-free)."""
    if step <= 0:
        return float(value)
    return float(round(math.floor(round(float(value) / step, 6)) * step, 6))


def _ceil_to_step(value: float, step: float) -> float:
    """Round ``value`` UP to the nearest multiple of ``step`` (artifact-free)."""
    if step <= 0:
        return float(value)
    return float(round(math.ceil(round(float(value) / step, 6)) * step, 6))


def _clean_micron_min(value: float) -> float:
    """
    Round a *lower* size bound DOWNWARD to clean, human-readable laboratory values.

    Directional flooring is essential for minimums: symmetric rounding can push a
    minimum above the smallest object that was actually measured and silently clip
    it (0.62 µm -> 0.6 µm is safe, 0.64 µm -> 0.7 µm is not). A lower bound must
    always err towards smaller values.

    Args:
        value: Raw lower bound in µm.

    Returns:
        float: Floored bound on the same 10 / 5 / 0.5 / 0.1 / 0.01 µm ladder that
        the Section 4 number inputs use.
    """
    val = max(0.0, float(value))
    if val >= 50.0:
        return _floor_to_step(val, 10.0)
    if val >= 10.0:
        return _floor_to_step(val, 5.0)
    if val >= 2.0:
        return _floor_to_step(val, 0.5)
    if val >= 0.5:
        return _floor_to_step(val, 0.1)
    return _floor_to_step(val, 0.01)


def _clean_micron_max(value: float) -> float:
    """
    Round an *upper* size bound UPWARD to clean, human-readable laboratory values.

    An upper bound must never be rounded down: a 54 µm recommendation rounded
    symmetrically to 50 µm would clip every detected bubble larger than 50 µm.
    Ceiling rounding preserves the full measured range plus its safety headroom.

    Args:
        value: Raw upper bound in µm.

    Returns:
        float: Ceiled bound on the same 10 / 5 / 0.5 / 0.1 / 0.01 µm ladder that
        the Section 4 number inputs use.
    """
    val = max(0.0, float(value))
    if val >= 50.0:
        return _ceil_to_step(val, 10.0)
    if val >= 10.0:
        return _ceil_to_step(val, 5.0)
    if val >= 2.0:
        return _ceil_to_step(val, 0.5)
    if val >= 0.5:
        return _ceil_to_step(val, 0.1)
    return _ceil_to_step(val, 0.01)


def _fmt_um(value) -> str:
    """Format a physical size for the comparison table."""
    if value is None:
        return "-"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if val >= 100.0:
        return f"{val:.0f} µm"
    if val >= 10.0:
        return f"{val:.1f} µm"
    return f"{val:.2f} µm"


def _current_value(current_params: dict, key: str):
    """Read a current widget value, falling back to its canonical default."""
    if isinstance(current_params, dict) and current_params.get(key) is not None:
        return current_params[key]
    for defaults in PARAMETER_WIDGET_DEFAULTS.values():
        if key in defaults:
            return defaults[key]
    return None


def _format_param_value(key: str, value) -> str:
    """Render a widget value for the educational comparison table."""
    if value is None:
        return "-"
    if key in (
        "sem_min_air_um",
        "sem_max_air_um",
        "sem_min_fat_um",
        "sem_max_fat_um",
        "light_min_fat_size_input",
        "light_max_fat_size_input",
        "light_expected_droplet_um",
    ):
        return _fmt_um(value)
    if isinstance(value, (bool, np.bool_)):
        return "On" if bool(value) else "Off"
    if key in (
        "shared_flow_slider",
        "shared_cellprob_slider",
        "shared_fat_circ_slider",
        "sem_air_circ_slider",
        "sem_ads_circ_slider",
        "light_solidity_slider",
    ):
        return f"{float(value):.2f}"
    if key == "light_crystal_smooth_slider":
        return f"{int(value)} / 4" if int(value) > 0 else "Off"
    return str(value)


def _comparison_row(key: str, label: str, recommended, current_params: dict, why: str) -> dict:
    """Build one fully formatted row of the educational comparison table."""
    return {
        "Parameter": label,
        "Current": _format_param_value(key, _current_value(current_params, key)),
        "Recommended": _format_param_value(key, recommended),
        "Why (Educational Rationale)": why,
    }


def _diagnose_light_microscopy(gray: np.ndarray, um_per_px: float, current_params: dict) -> dict:
    """
    Brightfield diagnostics: internal crystal striations, packing, polarity and scale.

    Internal crystal needles are dark lines *inside* bright fat globules, so plain
    Otsu thresholding labels them as "background" and shreds each globule into tiny
    bright patches. The fix is to separate two views of the same image:

    * a crystal-smoothed copy used to build macro-droplet masks and size estimates;
    * the raw (unsmoothed) image used to measure crystal striation physics strictly
      inside the eroded macro-droplet interiors (rims excluded).
    """
    work, resize_factor = _prepare_analysis_gray(gray)
    try:
        ratio_value = float(um_per_px)
    except (TypeError, ValueError):
        ratio_value = 0.0
    px_um = ratio_value if math.isfinite(ratio_value) and ratio_value > 0 else 0.18
    px_um = px_um / resize_factor if resize_factor > 0 else px_um

    # ── 1. Crystal-smoothed copy: erase internal needles, keep outer rims ─────
    try:
        smoothed = smooth_internal_crystals(work, strength=3)
    except Exception:
        smoothed = work.copy()
    blur_s = cv2.GaussianBlur(smoothed, (5, 5), 0)

    p5, _, p95 = (float(np.percentile(blur_s, q)) for q in (5.0, 50.0, 95.0))
    contrast = p95 - p5

    # Polarity is decided on the smoothed image so striations cannot masquerade
    # as the "dark object" class (the root cause of the old 0.77x misclassification).
    otsu_thr, _ = cv2.threshold(blur_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_mask = (blur_s >= otsu_thr).astype(np.uint8)
    dark_mask = (1 - bright_mask).astype(np.uint8)
    min_area = max(12, int(round(0.00004 * float(work.size))))

    # ── 2. Macro-droplet interior mask (bridges thin internal crystal lines) ──
    # Dark inter-droplet ridges are detected first: the closing bridges internal
    # crystal needles, but restoring those outer ridges prevents neighbouring
    # globules from merging into one giant sheet.
    ridge_k = max(5, int(round(1.2 / max(px_um, 1e-4))) | 1)
    outer_ridges = (
        cv2.morphologyEx(blur_s, cv2.MORPH_BLACKHAT, _ellipse(ridge_k)) > 14.0
    ).astype(np.uint8)

    close_k = max(9, int(round(2.5 / max(px_um, 1e-4))) | 1)
    macro_seed = cv2.morphologyEx(blur_s, cv2.MORPH_CLOSE, _ellipse(close_k))
    macro_seed[outer_ridges > 0] = 0
    macro_blur = cv2.GaussianBlur(macro_seed, (5, 5), 0)
    macro_thr, _ = cv2.threshold(macro_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    macro_bright = (macro_blur >= macro_thr).astype(np.uint8)
    macro_dark = (1 - macro_bright).astype(np.uint8)

    def _macro_candidate(binary_mask: np.ndarray):
        """Return (mask, coverage, n_components, plausible) for one polarity."""
        candidate = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, _ellipse(3))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, _ellipse(5))
        candidate = _fill_holes(candidate)
        foreground = int(np.count_nonzero(candidate))
        coverage_value = foreground / float(candidate.size)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        valid_areas = [
            int(stats[i, cv2.CC_STAT_AREA])
            for i in range(1, n_labels)
            if min_area <= stats[i, cv2.CC_STAT_AREA] <= 0.45 * candidate.size
        ]
        all_areas = [
            int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)
        ]
        largest_fraction = (
            max(all_areas) / float(foreground) if all_areas and foreground > 0 else 0.0
        )
        # A bright background sheet enclosing dilute droplets looks like "objects"
        # by component count, but its near-total coverage and dominant single region
        # reveal that it is the background phase.
        is_plausible = (
            0.02 <= coverage_value <= 0.97
            and len(valid_areas) >= 3
            and (coverage_value <= 0.90 or largest_fraction <= 0.85)
        )
        return candidate, coverage_value, len(valid_areas), is_plausible

    bright_candidate = _macro_candidate(macro_bright)
    dark_candidate = _macro_candidate(macro_dark)
    if bright_candidate[3] and dark_candidate[3]:
        bright_objects = bright_candidate[2] >= dark_candidate[2]
    elif bright_candidate[3]:
        bright_objects = True
    elif dark_candidate[3]:
        bright_objects = False
    else:
        # Fallback for ambiguous fields: the polarity with more compact components wins.
        bright_objects = bright_candidate[2] >= dark_candidate[2]

    macro_mask, packing, _, _ = bright_candidate if bright_objects else dark_candidate
    macro_bool = macro_mask > 0

    # ── 3. Internal crystal striations measured on the UNSMOOTHED image ───────
    blackhat = cv2.morphologyEx(work, cv2.MORPH_BLACKHAT, _ellipse(7))
    lap = np.abs(cv2.Laplacian(work.astype(np.float32), cv2.CV_32F, ksize=3))
    # Erode each macro-droplet proportionally to its own radius (>= 3 px) so the
    # dark outer rim / refraction halo never leaks into the texture sample.
    macro_dist = cv2.distanceTransform(macro_mask, cv2.DIST_L2, 5)
    n_macro, macro_labels, macro_stats, _ = cv2.connectedComponentsWithStats(
        macro_mask, 8
    )
    interior = np.zeros_like(macro_mask)
    for idx in range(1, n_macro):
        area = int(macro_stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w_box, h_box, _ = (int(v) for v in macro_stats[idx])
        if w_box < 2 or h_box < 2:
            continue
        crop_labels = macro_labels[y : y + h_box, x : x + w_box]
        crop_dist = macro_dist[y : y + h_box, x : x + w_box]
        component = crop_labels == idx
        if not np.any(component):
            continue
        max_radius = float(crop_dist[component].max())
        max_radius = min(max_radius, 0.5 * float(min(macro_mask.shape)))
        if max_radius < 5.0:
            continue
        erode_px = max(3, int(round(0.45 * max_radius)))
        kernel_size = 2 * erode_px + 1
        if kernel_size >= min(w_box, h_box) or kernel_size >= 2 * max_radius:
            continue
        core = cv2.erode(component.astype(np.uint8), _ellipse(kernel_size))
        interior[y : y + h_box, x : x + w_box] |= core
    if np.count_nonzero(interior) < 80:
        interior = cv2.erode(macro_mask, _ellipse(7))
    interior_bool = interior.astype(bool)
    if not np.any(interior_bool):
        interior_bool = macro_bool.copy()

    striation_mask = (blackhat > 12) & interior_bool
    striation_density = float(np.count_nonzero(striation_mask)) / float(
        np.count_nonzero(interior_bool)
    )
    striation_intensity = float(blackhat[interior_bool].mean())
    interior_values = work[interior_bool]
    if interior_values.size > 20:
        low_pct, high_pct = np.percentile(interior_values, [10.0, 90.0])
        trimmed_values = interior_values[
            (interior_values >= low_pct) & (interior_values <= high_pct)
        ]
        internal_std = (
            float(trimmed_values.std()) if trimmed_values.size else float(interior_values.std())
        )
    else:
        internal_std = float(interior_values.std())
    lap_energy = float(lap[interior_bool].mean())
    outer_bool = ~macro_bool
    lap_outside = float(lap[outer_bool].mean()) if np.any(outer_bool) else lap_energy
    lap_ratio = lap_energy / max(lap_outside, 1e-6)

    crystallized = bool(
        striation_density >= 0.035
        or internal_std > 18.0
        or (packing > 0.55 and (lap_energy >= 6.0 or lap_ratio >= 1.15))
    )
    crystal_score = float(
        np.clip(
            0.55 * (striation_density / 0.12)
            + 0.30 * (internal_std / 30.0)
            + 0.15 * (packing / 0.75),
            0.0,
            1.0,
        )
    )

    # ── 4. Droplet size on the crystal-smoothed mask ──────────────────────────
    # Dark inter-droplet borders are removed via a black-hat ridge map, and the
    # distance-transform peaks seed a marker-controlled watershed so merged
    # doublets are split into their true lobes.
    if bright_objects:
        # Bright bodies: subtract dark contact ridges, then fill to get footprints.
        size_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, _ellipse(3))
        size_mask = (size_mask.astype(bool) & ~outer_ridges.astype(bool)).astype(np.uint8)
        size_mask = cv2.morphologyEx(size_mask, cv2.MORPH_OPEN, _ellipse(3))
        size_mask = _fill_holes(size_mask)
        size_mask = cv2.erode(size_mask, _ellipse(3))
    else:
        # Dark-rim droplets: the rim IS the droplet boundary. Subtract nothing and
        # fill the enclosed bright interior so each ring becomes a solid footprint.
        size_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, _ellipse(3))
        size_mask = _fill_holes(size_mask)

    min_peak_sep_px = max(6, int(round(1.5 / max(px_um, 1e-4))))
    diameters, peak_diameters = _watershed_droplet_diameters(
        size_mask, px_um, min_peak_separation_px=min_peak_sep_px
    )
    if len(diameters) < 3:
        diameters = peak_diameters
    if len(diameters) < 3:
        diameters = _distance_transform_diameters(size_mask, px_um)
    if len(diameters) < 3:
        diameters = _contour_diameters(size_mask, px_um)
    if not diameters:
        diameters = [3.0, 6.0, 9.0]
    diam_arr = np.asarray(diameters, dtype=float)
    d5, d50, d70, d75, d90, d95, d98 = (
        float(np.percentile(diam_arr, q))
        for q in (5.0, 50.0, 70.0, 75.0, 90.0, 95.0, 98.0)
    )
    polydispersity = d90 / max(d50, 1e-6)

    # Liquid emulsions: area-weighted D[2,1] + P75 avoids being dragged down by
    # tiny interstitial droplets. Crystallized/polydisperse packed emulsions use a
    # robust Sauter D[3,2] + P82 instead, because the 1.0x fine pass must not cut
    # inside 15-24 µm globules.
    d_area_weighted = float(
        np.sum(diam_arr ** 2) / max(np.sum(diam_arr), 1e-6)
    )
    # Sauter D[3,2] is extremely tail-sensitive: a handful of coalesced/merged
    # lobes can double it. Winsorizing at P85 keeps the volume-weighted scale
    # anchored to the bulk globule population instead of a few outliers.
    sauter_cap = float(np.percentile(diam_arr, 85.0))
    sauter_arr = np.minimum(diam_arr, sauter_cap)
    d_sauter = float(
        np.sum(sauter_arr ** 3) / max(np.sum(sauter_arr ** 2), 1e-6)
    )
    p82 = float(np.percentile(diam_arr, 82.0))
    use_crystal_scale = bool(crystallized or polydispersity > 1.6)
    if use_crystal_scale:
        crystal_target = 0.5 * d_sauter + 0.5 * p82
        # Large central macro-globules: never let the base pass fall below 8 µm.
        if d95 >= 11.0:
            crystal_target = max(crystal_target, 8.0)
    else:
        crystal_target = 0.5 * d_area_weighted + 0.5 * d75
    expected_droplet = float(
        np.clip(round(crystal_target * 2.0) / 2.0, 3.0, 22.0)
    )
    # Never truncate the large central globules of a crystallized emulsion:
    # 18-25 µm macro-globules must stay classified as fat, not "Unclassified".
    min_fat = float(round(max(0.40, min(1.0, d5 * 0.65)), 2))
    max_fat = float(round(max(25.0, d98 * 1.6, expected_droplet * 3.2), 1))
    multiscale = bool(crystallized or polydispersity > 1.6)

    # Brightfield polarity: packed bright bodies (dark contact borders) vs. dilute
    # dark-rimmed droplets on a bright background.
    if bright_objects and (crystallized or packing >= 0.28):
        invert_bf = False
        invert_why = (
            "Tightly packed bright-body globules separated by thin dark contact "
            f"borders (packing {packing:.0%}): the native polarity already gives "
            "Cellpose filled bodies, whereas inversion would turn every globule into "
            "a hole."
        )
    elif (not bright_objects) and packing <= 0.70:
        invert_bf = True
        invert_why = (
            "Dilute droplets present dark refractive rims/halos on a bright background; "
            "inverting converts those rims into solid bright bodies so Cellpose can close "
            "the true outer droplet boundary."
        )
    else:
        invert_bf = not crystallized
        invert_why = (
            "Mixed/ambiguous polarity: inversion follows the optical profile "
            "(crystallized-packed keeps OFF, liquid/dispersed keeps ON)."
        )

    cellprob = -1.5 if contrast < 65.0 else -1.0
    profile = LIGHT_PROFILE_CRYSTALLIZED if crystallized else LIGHT_PROFILE_LIQUID
    solidity = 0.84 if crystallized else 0.75
    fat_circularity = 0.60 if crystallized else 0.68
    smoothing_strength = 3 if crystallized else 0

    crystal_pipeline_active = bool(
        crystallized and smoothing_strength >= 3 and multiscale
    )
    # Foundation-model choice is derived ONLY from the measured optics: contrast,
    # crystal striation density and packing. current_model_choice is reported for
    # the "Current" column but never feeds the recommendation, so a cyto3 selected
    # for a previous image can no longer be echoed back as a "measured" finding.
    current_model_choice = str(
        _current_value(current_params, "shared_model_type_select") or "cyto2"
    )
    if crystallized:
        # Smoothing 3 + multi-scale already suppress internal crystal lines; cyto2
        # is faster and equally clean unless the optics are genuinely faint.
        model_recommendation = "cyto3" if contrast < 75.0 else "cyto2"
    else:
        model_recommendation = "cyto3" if contrast < 65.0 else "cyto2"

    recommended_state = {
        "light_emulsion_profile_radio": profile,
        "light_crystal_smooth_slider": smoothing_strength,
        "light_solidity_slider": solidity,
        "shared_fat_circ_slider": fat_circularity,
        "light_expected_droplet_um": expected_droplet,
        "light_min_fat_size_input": min_fat,
        "light_max_fat_size_input": max_fat,
        "light_multiscale_checkbox": multiscale,
        "light_invert_bf_checkbox": bool(invert_bf),
        "shared_clahe_checkbox": False,
        "shared_flow_slider": 0.70,
        "shared_cellprob_slider": float(cellprob),
        "shared_model_type_select": model_recommendation,
    }
    return {
        "profile": profile,
        "crystallized": crystallized,
        "contrast": contrast,
        "d5": d5,
        "d50": d50,
        "d70": d70,
        "d75": d75,
        "d90": d90,
        "d95": d95,
        "d98": d98,
        "d_area_weighted": d_area_weighted,
        "d_sauter": d_sauter,
        "sauter_cap": sauter_cap,
        "p82": p82,
        "use_crystal_scale": use_crystal_scale,
        "crystal_target": crystal_target,
        "expected_droplet": expected_droplet,
        "smoothing_strength": smoothing_strength,
        "crystal_pipeline_active": crystal_pipeline_active,
        "model_recommendation": model_recommendation,
        "current_model_choice": current_model_choice,
        "min_peak_sep_px": min_peak_sep_px,
        "striation_density": striation_density,
        "striation_intensity": striation_intensity,
        "internal_std": internal_std,
        "lap_energy": lap_energy,
        "lap_ratio": lap_ratio,
        "packing": packing,
        "coverage": packing,
        "crystal_score": crystal_score,
        "texture_ratio": lap_ratio,
        "polydispersity": polydispersity,
        "recommended": recommended_state,
        "invert_why": invert_why,
        "cellprob": cellprob,
    }


def _diagnose_sem_micrograph(gray: np.ndarray, um_per_px: float, current_params: dict) -> dict:
    """
    SEM diagnostics: dark craters/pores, fine matrix globules, shadow depth.

    The recommended widget state is a *pure function* of the pixels and the
    calibration. ``current_params`` is accepted only so the caller can render the
    "Current" column of the comparison table next to the proposal - it must never
    influence a recommended value, otherwise a range tuned on a previous
    micrograph would be echoed back as if it had been measured on this one.

    Args:
        gray: Cropped SEM micrograph (footer already excluded).
        um_per_px: Active calibration in µm per pixel.
        current_params: Sidebar values, used by the comparison table only.

    Returns:
        dict: Measured statistics plus the ``recommended`` widget-state mapping.
    """

    work, resize_factor = _prepare_analysis_gray(gray)
    try:
        ratio_value = float(um_per_px)
    except (TypeError, ValueError):
        ratio_value = 0.0
    px_um = ratio_value if math.isfinite(ratio_value) and ratio_value > 0 else 0.2
    px_um = px_um / resize_factor if resize_factor > 0 else px_um

    blur = cv2.GaussianBlur(work, (5, 5), 0)
    p5, p95 = (float(np.percentile(blur, q)) for q in (5.0, 95.0))
    contrast = p95 - p5
    bg_level = float(np.percentile(blur, 85.0))

    # 1a. Reference dark mask (22nd percentile, Otsu fallback). It doubles as the
    #     base mask for the adsorbed-fat speckle statistic computed further below.
    p22 = float(np.percentile(blur, 22.0))
    bg_median = float(np.median(blur))
    dark_threshold = p22
    if p22 > bg_median - 10.0:
        otsu_threshold, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_threshold = float(min(otsu_threshold, p22))
    dark = (blur <= dark_threshold).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, _ellipse(3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, _ellipse(5))

    # Matrix anchor for the "is this really part of the dark phase?" test below.
    # Percentile anchors collapse when dark craters cover most of the frame (dense
    # foam), so the test is skipped there and the local ring test alone decides.
    dark_fraction = float(np.count_nonzero(dark)) / float(dark.size)
    dark_reference = (
        bg_median - max(8.0, 0.25 * (bg_median - p5))
        if dark_fraction <= 0.45
        else None
    )

    # 1b. Pore candidates: multi-threshold + morphological contour pass.
    #     A single 22nd-percentile cut silently misses valid bubbles whenever a
    #     field holds only 1-2 large craters (the percentile then lands on the
    #     matrix, not on the pores) or when a crater interior is coated with bright
    #     adsorbed fat. We therefore probe several independent dark cuts - raw
    #     percentiles p18/p25/p35 plus Otsu on a CLAHE-enhanced copy and on its
    #     inverse (the two views segment_air_bubbles_auto also uses in Step B) -
    #     flood-fill internal holes so bright adsorbed fat cannot break the pore
    #     mask, and keep only the largest valid contour per spatial region.
    #     The cuts are probed on a mildly denoised view (~0.5 µm of physical
    #     smoothing): pixel noise would otherwise make every percentile cut split
    #     into hundreds of pseudo-craters, while 20-100 µm bubbles are unaffected.
    def _clahe_view(img: np.ndarray) -> np.ndarray:
        """Return the CLAHE-enhanced analysis view (identity if CLAHE fails)."""
        try:
            return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        except cv2.error:
            return img.copy()

    pore_sigma = float(np.clip(0.5 / max(px_um, 1e-6), 1.5, 9.0))
    pore_view = cv2.GaussianBlur(
        work, (0, 0), sigmaX=pore_sigma, sigmaY=pore_sigma
    )
    clahe_img = _clahe_view(pore_view)
    inv_clahe = np.clip(255 - clahe_img, 0, 255).astype(np.uint8)

    candidate_masks = [
        (pore_view <= float(np.percentile(pore_view, pct))).astype(np.uint8)
        for pct in (18.0, 22.0, 25.0, 35.0)
    ]
    # Otsu on the CLAHE view isolates locally boosted (shadowed) pore walls.
    clahe_otsu, _ = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidate_masks.append((clahe_img <= clahe_otsu).astype(np.uint8))
    # Otsu on the inverted view: craters become bright islands there, so the bright
    # class of the inverted histogram is the dark-object class of the original.
    inv_otsu, _ = cv2.threshold(inv_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidate_masks.append((inv_clahe >= inv_otsu).astype(np.uint8))

    min_pore_px = max(
        9,
        min(
            int(round(math.pi * (0.35 / max(px_um, 1e-6)) ** 2)),
            max(9, int(0.01 * float(blur.size))),
        ),
    )
    max_pore_px = 0.35 * float(blur.size)
    # (filled area, circularity, bbox, local mask, pore record) kept together so
    # overlapping candidates can be deduplicated by size before the masks are freed.
    pore_candidates = []
    for candidate_mask in candidate_masks:
        cleaned = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, _ellipse(3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, _ellipse(5))
        # Bright adsorbed fat / internal shadow speckle inside a crater must not
        # split or puncture the pore footprint.
        cleaned = _fill_holes(cleaned)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, _ellipse(3))
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if contour_area < float(min_pore_px) or contour_area > max_pore_px:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if contour_area <= 0.0 or perimeter <= 0.0:
                continue
            circularity = float(4.0 * math.pi * contour_area / (perimeter ** 2))
            if circularity < 0.30:
                continue
            diameter_um = 2.0 * math.sqrt(contour_area / math.pi) * px_um
            if not 1.5 <= diameter_um <= 5000.0:
                continue
            box_x, box_y, box_w, box_h = cv2.boundingRect(contour)
            if box_w < 2 or box_h < 2:
                continue
            local = np.zeros((box_h, box_w), np.uint8)
            shifted = contour.copy()
            shifted[:, :, 0] -= box_x
            shifted[:, :, 1] -= box_y
            cv2.drawContours(local, [shifted], -1, 1, thickness=cv2.FILLED)
            local_bool = local.astype(bool)
            filled_px = float(np.count_nonzero(local_bool))
            if filled_px <= 0.0:
                continue
            values = blur[box_y : box_y + box_h, box_x : box_x + box_w][local_bool]
            shadow = float(bg_level - float(values.mean()))
            # Every cut can still grab flat matrix patches; genuine craters must be
            # measurably darker than the surrounding material.
            if shadow < 4.0:
                continue
            # Absolute darkness: a crater interior belongs to the dark phase, i.e. it
            # sits clearly below the matrix mode. A percentile cut applied to a
            # featureless field produces thin bands whose interior median equals the
            # matrix mode, and those must never be reported as air bubbles.
            if dark_reference is not None and float(np.median(values)) > dark_reference:
                continue
            # Local contrast-to-noise: a real crater stands out of the material that
            # directly surrounds it. Flat matrix texture - which any percentile cut
            # inevitably splits - has inside ≈ ring and is discarded here, while a
            # medium-gray bubble interior sitting in a bright matrix still passes.
            # Medians / IQR keep neighbouring craters inside the ring from faking a
            # shadow or inflating the local noise estimate.
            ring_pad = max(2, int(round(math.sqrt(filled_px / math.pi) * 0.35)))
            crop_y0 = max(0, box_y - ring_pad)
            crop_y1 = min(blur.shape[0], box_y + box_h + ring_pad)
            crop_x0 = max(0, box_x - ring_pad)
            crop_x1 = min(blur.shape[1], box_x + box_w + ring_pad)
            crop = blur[crop_y0:crop_y1, crop_x0:crop_x1]
            padded_local = np.zeros(crop.shape, np.uint8)
            padded_local[
                box_y - crop_y0 : box_y - crop_y0 + box_h,
                box_x - crop_x0 : box_x - crop_x0 + box_w,
            ] = local
            ring = (cv2.dilate(padded_local, _ellipse(2 * ring_pad + 1)) > 0) & (
                padded_local == 0
            )
            if np.any(ring):
                ring_values = crop[ring].astype(np.float32)
                ring_level = float(np.median(ring_values))
                ring_q25, ring_q75 = np.percentile(ring_values, [25.0, 75.0])
                ring_sigma = float((ring_q75 - ring_q25) / 1.349)
                inside_level = float(np.median(crop[padded_local > 0]))
                prominence_gate = max(
                    6.0, min(1.5 * ring_sigma, 0.5 * max(float(contrast), 1.0))
                )
                if (ring_level - inside_level) < prominence_gate:
                    continue
            pore_candidates.append(
                (
                    filled_px,
                    circularity,
                    (box_x, box_y, box_w, box_h),
                    local_bool,
                    {
                        "d_um": diameter_um,
                        "circ": circularity,
                        "shadow": shadow,
                        "inner_std": float(values.std()),
                    },
                )
            )

    # Deduplicate overlapping candidates: the largest valid contour wins its
    # spatial region, so a wide percentile cut cannot re-add a bubble that the
    # darkest (or a tighter) cut already explained.
    pore_candidates.sort(key=lambda item: (-item[0], -item[1]))
    pores = []
    occupied = np.zeros(blur.shape[:2], dtype=bool)
    for filled_px, _circularity, bbox, local_bool, pore in pore_candidates:
        box_x, box_y, box_w, box_h = bbox
        seen = occupied[box_y : box_y + box_h, box_x : box_x + box_w]
        overlap_fraction = float(np.count_nonzero(seen & local_bool)) / filled_px
        if overlap_fraction >= 0.50:
            continue
        seen |= local_bool
        pores.append(pore)

    # 2. Fine bright matrix globules via white top-hat (isolates objects up to ~3 µm).
    max_kernel = max(3, min(work.shape[:2]) - 1)
    tophat_len = int(np.clip(round(3.0 / max(px_um, 1e-6)), 5, min(41, max_kernel)))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, _ellipse(tophat_len))
    fat_threshold = max(6.0, float(np.percentile(tophat, 99.0)))
    fat_mask = (tophat >= fat_threshold).astype(np.uint8)
    fat_mask = cv2.morphologyEx(fat_mask, cv2.MORPH_OPEN, _ellipse(3))
    n_fat, _, fat_stats, _ = cv2.connectedComponentsWithStats(fat_mask, 8)
    fat_diams = []
    for idx in range(1, n_fat):
        area = int(fat_stats[idx, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        diameter_um = 2.0 * math.sqrt(area / math.pi) * px_um
        if 0.1 <= diameter_um <= 40.0:
            fat_diams.append(diameter_um)

    # A pore population must be genuinely darker than the matrix noise floor.
    # (Kept as a final safety net: a handful of residual blobs must never be
    # promoted to an air phase.)
    if pores:
        noise_floor = max(6.0, 0.35 * float(contrast))
        if float(np.max([p["shadow"] for p in pores])) < noise_floor:
            pores = []

    air_vals = np.asarray([p["d_um"] for p in pores], dtype=float)
    fat_vals = np.asarray(fat_diams, dtype=float)
    # Even a single prominent air bubble is real evidence: requiring >= 3 pores used
    # to discard micrographs that contain only 1-2 large craters.
    has_pores = bool(air_vals.size >= 1 and float(np.max(air_vals)) >= 2.0)
    has_fine = bool(fat_vals.size >= 15)
    if air_vals.size:
        air_p10, air_p50, air_p95 = (float(np.percentile(air_vals, q)) for q in (10.0, 50.0, 95.0))
        air_d_min = float(np.min(air_vals))
        air_d_max = float(np.max(air_vals))
    else:
        air_p10 = air_p50 = air_p95 = 0.0
        air_d_min = air_d_max = 0.0
    if fat_vals.size:
        fat_p5, fat_p10, fat_p50, fat_p95, fat_p98 = (
            float(np.percentile(fat_vals, q)) for q in (5.0, 10.0, 50.0, 95.0, 98.0)
        )
    else:
        fat_p5 = fat_p10 = fat_p50 = fat_p95 = fat_p98 = 0.0

    # 3. Shadow depth + internal crater variance (adsorbed-fat evidence).
    shadow_depth = float(np.median([p["shadow"] for p in pores])) if pores else 0.0
    speck_density = 0.0
    if np.any(dark):
        speck_density = float((tophat[dark.astype(bool)] >= max(8.0, fat_threshold * 0.5)).mean())
    inner_stds = [p["inner_std"] for p in pores if p["d_um"] >= 2.0]
    internal_speck = bool(
        (inner_stds and float(np.median(inner_stds)) >= 14.0) or speck_density >= 0.02
    )
    # 4. Sample type / detection target - derived from the measured phases ONLY.
    #    current_params is deliberately not read here: the recommendation must be
    #    identical for identical pixels, regardless of what the sidebar currently
    #    holds (the "Current" column of the comparison table shows those values).
    both_present = bool(has_pores and has_fine and fat_p95 < max(air_p50 * 0.75, 1.0))
    if both_present:
        rec_type, rec_focus = "Whipped Emulsion", SEM_FOCUS_TWO_PASS
    else:
        # Canonical modality defaults keep the air pass enabled, so a weakly
        # detected (or genuinely absent) crater population can never silently
        # disable bubble search for this image.
        rec_type = str(PARAMETER_WIDGET_DEFAULTS["SEM"]["sem_sample_type_radio"])
        rec_focus = str(PARAMETER_WIDGET_DEFAULTS["SEM"]["sem_detection_focus_radio"])
    if rec_focus not in (SEM_FOCUS_AIR, SEM_FOCUS_TWO_PASS):
        rec_focus = SEM_FOCUS_AIR

    # 5. Directional size bounds: minima are floored, maxima are ceiled. Every
    #    bound is derived from the measured pixels alone, so the recommendation is
    #    a pure function of (image, calibration).
    default_min_air = float(PARAMETER_WIDGET_DEFAULTS["SEM"]["sem_min_air_um"])
    default_max_air = float(PARAMETER_WIDGET_DEFAULTS["SEM"]["sem_max_air_um"])
    default_min_fat = float(PARAMETER_WIDGET_DEFAULTS["SEM"]["sem_min_fat_um"])
    default_max_fat = float(PARAMETER_WIDGET_DEFAULTS["SEM"]["sem_max_fat_um"])

    if has_pores:
        # min(10.0, 0.65 * smallest bubble): never above the validated 10 µm
        # default, so a field of 25-80 µm bubbles still admits 10-20 µm ones.
        rec_min_air = _clean_micron_min(
            max(0.5, min(default_min_air, air_d_min * 0.65))
        )
        # max(100.0, 1.35 * largest bubble): never below the validated 100 µm
        # default, unless the whole frame is narrower than 80 µm (high magnification)
        # where a 100 µm object cannot physically exist.
        frame_width_um = float(work.shape[1]) * px_um
        air_max_floor = (
            default_max_air
            if frame_width_um >= 80.0
            else max(default_max_air * 0.1, frame_width_um)
        )
        rec_max_air = _clean_micron_max(max(air_max_floor, air_d_max * 1.35))
    else:
        # No pore population measured: fall back to the canonical SEM defaults,
        # never to whatever range the previous image left behind.
        rec_min_air, rec_max_air = default_min_air, default_max_air

    if has_fine:
        rec_min_fat = _clean_micron_min(max(0.1, fat_p5 * 0.60))
        rec_max_fat = _clean_micron_max(max(default_max_fat, fat_p98 * 1.40))
    else:
        rec_min_fat, rec_max_fat = default_min_fat, default_max_fat

    low_contrast = bool(contrast < 65.0 or shadow_depth > 55.0)
    if contrast < 45.0 or shadow_depth > 75.0:
        flow = 0.80
    elif low_contrast:
        flow = 0.75
    else:
        flow = 0.70
    cellprob = -1.5 if low_contrast else -1.0
    # Deep-scan is an image property (adsorbed-fat speckle inside craters) and is
    # never inherited from the currently checked checkbox.
    deepscan = bool(internal_speck)
    dust_score = float(
        np.clip(0.5 * (shadow_depth / 80.0) + 0.5 * (1.0 if internal_speck else 0.0), 0.0, 1.0)
    )

    recommended_state = {
        "sem_sample_type_radio": rec_type,
        "sem_detection_focus_radio": rec_focus,
        "sem_min_air_um": float(rec_min_air),
        "sem_max_air_um": float(rec_max_air),
        "sem_min_fat_um": float(rec_min_fat),
        "sem_max_fat_um": float(rec_max_fat),
        "shared_flow_slider": float(flow),
        "shared_cellprob_slider": float(cellprob),
        "shared_fat_circ_slider": 0.60,
        "sem_air_circ_slider": 0.50,
        "sem_ads_circ_slider": 0.30,
        "sem_deepscan_checkbox": bool(deepscan),
        "shared_clahe_checkbox": True,
        # Low-contrast craters or internal shadow speckle benefit from cyto3's
        # much broader training distribution.
        "shared_model_type_select": (
            "cyto3" if (low_contrast or internal_speck) else "cyto2"
        ),
    }
    return {
        "n_pores": int(air_vals.size),
        "n_globules": int(fat_vals.size),
        "air_p10": air_p10,
        "air_p50": air_p50,
        "air_p95": air_p95,
        "air_d_min": air_d_min,
        "air_d_max": air_d_max,
        "fat_p5": fat_p5,
        "fat_p10": fat_p10,
        "fat_p50": fat_p50,
        "fat_p95": fat_p95,
        "fat_p98": fat_p98,
        "has_pores": has_pores,
        "has_fine": has_fine,
        "both_present": both_present,
        "contrast": contrast,
        "shadow_depth": shadow_depth,
        "speck_density": speck_density,
        "internal_speck": internal_speck,
        "dust_score": dust_score,
        "low_contrast": low_contrast,
        "flow": flow,
        "cellprob": cellprob,
        "recommended": recommended_state,
    }



def _light_summary_metrics(diag: dict) -> dict:
    """Measured facts shown as diagnostic badges."""
    return {
        "Droplet Ø (P5-P95)": f"{diag['d5']:.1f}-{diag['d95']:.1f} µm",
        "AI Target Ø": f"{diag['expected_droplet']:.1f} µm",
        "Crystal Striation / Packing": (
            f"{diag['striation_density'] * 100:.0f}% · {diag['packing']:.2f}"
        ),
        "Optical Contrast (P95-P5)": f"{diag['contrast']:.0f} GL",
        "Recommended Profile": (
            "❄️ Crystallized Fat" if diag["crystallized"] else "💧 Liquid Fat"
        ),
    }


def _light_comparison_rows(diag: dict, current_params: dict) -> list:
    """Educational Current vs. Recommended table for brightfield light microscopy."""
    rec = diag["recommended"]
    crystallized = bool(diag["crystallized"])
    if crystallized:
        profile_why = (
            f"Black-hat striation density inside eroded globule interiors is "
            f"{diag['striation_density']:.1%} (internal std {diag['internal_std']:.1f} GL, "
            f"packing {diag['packing']:.0%}, Laplacian energy {diag['lap_energy']:.1f}). "
            "These dark refraction lines are the optical signature of internal fat "
            "crystals; edge-preserving smoothing plus strict convexity keeps each "
            "globule intact instead of splitting it along crystal needles."
        )
    else:
        profile_why = (
            f"Globule interiors are optically smooth (striation density "
            f"{diag['striation_density']:.1%}, internal std {diag['internal_std']:.1f} GL, "
            f"packing {diag['packing']:.0%}); smoothing and the extra convexity "
            "constraint would only erode genuine diffuse droplet boundaries."
        )

    if rec["shared_model_type_select"] == "cyto3":
        if diag["crystal_pipeline_active"] and diag["contrast"] < 75.0:
            model_why = (
                f"Contrast is {diag['contrast']:.0f} GL (< 75), so even with smoothing "
                f"{diag['smoothing_strength']} + multi-scale the faint crystal boundaries "
                "benefit from cyto3's broader multi-modality training. Measured striation "
                f"density {diag['striation_density']:.1%}, internal std "
                f"{diag['internal_std']:.1f} GL."
            )
        else:
            model_why = (
                f"Low optical contrast ({diag['contrast']:.0f} GL < 65) makes faint "
                "droplet boundaries ambiguous; cyto3's super-generalist training recovers "
                f"them better than cyto2. Striation density "
                f"{diag['striation_density']:.1%}, internal std {diag['internal_std']:.1f} GL."
            )
    elif diag["crystal_pipeline_active"]:
        model_why = (
            f"Smoothing {diag['smoothing_strength']} + multi-scale already erase internal "
            f"crystal refraction lines cleanly, and contrast is {diag['contrast']:.0f} GL "
            "(≥ 75), so cyto2 reaches the same boundaries much faster than cyto3."
        )
    else:
        model_why = (
            f"Globule interiors are smooth (striation density "
            f"{diag['striation_density']:.1%}, internal std {diag['internal_std']:.1f} GL) "
            f"and contrast is adequate ({diag['contrast']:.0f} GL); cyto2 reaches the "
            "same boundaries much faster."
        )

    rows = [
        _comparison_row(
            "light_emulsion_profile_radio",
            "Emulsion Droplet Profile",
            rec["light_emulsion_profile_radio"],
            current_params,
            profile_why,
        ),
        _comparison_row(
            "light_expected_droplet_um",
            "Typical Droplet Diameter for AI (µm)",
            rec["light_expected_droplet_um"],
            current_params,
            (
                (
                    "A marker-controlled watershed seeded by distance-transform maxima "
                    f"(min peak separation {diag['min_peak_sep_px']} px) separates touching "
                    "globules. For crystallized multi-scale segmentation the robust Sauter "
                    f"D[3,2] = {diag['d_sauter']:.2f} µm (winsorized at P85 = "
                    f"{diag['sauter_cap']:.2f} µm) is blended 50/50 with P82 = "
                    f"{diag['p82']:.2f} µm, giving {diag['expected_droplet']:.1f} µm. "
                    "That keeps the 1.0× pass from cutting inside 15-24 µm globules, "
                    "where a number-median scale of ~6.5 µm split them into two lobes."
                )
                if diag["use_crystal_scale"]
                else (
                    "A marker-controlled watershed seeded by distance-transform maxima "
                    f"(min peak separation {diag['min_peak_sep_px']} px) separates touching "
                    f"globules. Area-weighted D[2,1] = {diag['d_area_weighted']:.2f} µm and "
                    f"P75 = {diag['d75']:.2f} µm blend to {diag['expected_droplet']:.1f} µm, "
                    "so tiny interstitial droplets cannot drag the AI scale down."
                )
            ),
        ),
        _comparison_row(
            "light_min_fat_size_input",
            "Min Fat Particle Size (µm)",
            rec["light_min_fat_size_input"],
            current_params,
            (
                f"The P5 droplet diameter is {diag['d5']:.2f} µm; 65% of it, capped at "
                f"1.0 µm, gives {rec['light_min_fat_size_input']:.2f} µm - enough to "
                "suppress sub-resolution noise while keeping the genuine small-droplet "
                "tail."
            ),
        ),
        _comparison_row(
            "light_max_fat_size_input",
            "Max Fat Particle Size (µm)",
            rec["light_max_fat_size_input"],
            current_params,
            (
                f"P98 = {diag['d98']:.1f} µm; 1.6× P98 and 3.2× the AI target give "
                f"{rec['light_max_fat_size_input']:.1f} µm (always ≥ 25 µm), so "
                "18-25 µm central macro-globules are classified as fat instead of "
                "falling into Unclassified."
            ),
        ),
    ]
    rows.extend(
        [
            _comparison_row(
                "light_multiscale_checkbox",
                "Multi-Scale Globule Detection",
                rec["light_multiscale_checkbox"],
                current_params,
                (
                    f"Size spread P90/P50 = {diag['polydispersity']:.2f} "
                    f"({'broad' if diag['polydispersity'] > 1.6 else 'narrow'}) with "
                    f"{'crystallized' if crystallized else 'liquid'} droplets. Multi-scale "
                    "runs a coarse pass for large globules (avoiding internal splitting) "
                    "plus a fine pass for the small droplet population."
                ),
            ),
            _comparison_row(
                "light_crystal_smooth_slider",
                "Internal Crystal Smoothing Strength",
                rec["light_crystal_smooth_slider"],
                current_params,
                (
                    f"Measured internal black-hat striation density is "
                    f"{diag['striation_density']:.1%}; strength 3 edge-preserving "
                    "smoothing erases those crystal needles before inference, so "
                    "Cellpose sees one continuous droplet instead of several shards."
                    if crystallized
                    else f"Striation density is only {diag['striation_density']:.1%}, so "
                    "smoothing is switched off to preserve the faint refractive rim of "
                    "liquid droplets."
                ),
            ),
            _comparison_row(
                "light_invert_bf_checkbox",
                "Invert Brightfield (Dark-Rim Droplets)",
                rec["light_invert_bf_checkbox"],
                current_params,
                diag["invert_why"],
            ),
            _comparison_row(
                "light_solidity_slider",
                "Matrix Fat Min Solidity",
                rec["light_solidity_slider"],
                current_params,
                (
                    "Crystallized globules are solid convex bodies: Solidity ≥ 0.84 "
                    "rejects crescent shards and inter-droplet background fragments that "
                    "survive circularity filtering."
                    if crystallized
                    else "Liquid droplets are deformable; 0.75 tolerates slight elongation "
                    "while still rejecting concave fragments and shards."
                ),
            ),
        ]
    )
    rows.extend(
        [
            _comparison_row(
                "shared_fat_circ_slider",
                "Matrix Fat Min Circularity",
                rec["shared_fat_circ_slider"],
                current_params,
                (
                    "0.60 accommodates faceted crystal outlines and touching boundaries "
                    "between packed globules."
                    if crystallized
                    else "0.68 rejects rim fragments and elongated debris while keeping "
                    "near-spherical liquid droplets."
                ),
            ),
            _comparison_row(
                "shared_flow_slider",
                "flow_threshold",
                rec["shared_flow_slider"],
                current_params,
                (
                    "Brightfield droplet boundaries are optically crisp, so 0.70 keeps the "
                    "learned flow field stable while tolerating slightly irregular "
                    "crystallized outlines."
                ),
            ),
            _comparison_row(
                "shared_cellprob_slider",
                "cellprob_threshold",
                rec["shared_cellprob_slider"],
                current_params,
                (
                    f"Measured P95-P5 optical contrast is {diag['contrast']:.0f} GL "
                    f"({'< 65, so -1.5 recovers faint droplets' if diag['contrast'] < 65 else 'adequate, so -1.0 avoids background false positives'})."
                ),
            ),
            _comparison_row(
                "shared_clahe_checkbox",
                "Enable CLAHE Preprocessing",
                False,
                current_params,
                "CLAHE is disabled in brightfield: it amplifies the dark refractive halos "
                "and uneven illumination gradients that already dominate the gradient "
                "field, producing spurious boundaries.",
            ),
            _comparison_row(
                "shared_model_type_select",
                "Cellpose Foundation Model",
                rec["shared_model_type_select"],
                current_params,
                model_why,
            ),
        ]
    )
    return rows


def _sem_summary_metrics(diag: dict) -> dict:
    """Three-to-four measured facts shown as diagnostic badges."""

    def _range_text(p10: float, p95: float) -> str:
        if p10 <= 0.0 or p95 <= 0.0:
            return "not detected"
        return f"{p10:.1f}-{p95:.1f} µm"

    return {
        "Pore Ø (P10-P95)": _range_text(diag["air_p10"], diag["air_p95"]),
        "Matrix Globule Ø (P10-P95)": _range_text(diag["fat_p10"], diag["fat_p95"]),
        "Optical Contrast (P95-P5)": f"{diag['contrast']:.0f} GL",
        "Crater Shadow / Speck Score": (
            f"−{diag['shadow_depth']:.0f} GL · {diag['dust_score']:.2f}"
        ),
    }


def _sem_comparison_rows(diag: dict, current_params: dict) -> list:
    """Educational Current vs. Recommended table for SEM micrographs."""
    rec = diag["recommended"]
    pore_text = (
        f"{diag['n_pores']} near-circular dark craters spanning "
        f"{diag['air_d_min']:.1f}-{diag['air_d_max']:.1f} µm (min-max)"
        if diag["has_pores"]
        else "no large dark craters were segmented"
    )
    globule_text = (
        f"{diag['n_globules']} fine bright matrix globules with P95 = "
        f"{diag['fat_p95']:.2f} µm"
        if diag["has_fine"]
        else "no dense fine-globule population was isolated by the top-hat filter"
    )
    sample_why = (
        f"{pore_text} coexist with {globule_text}. Two physically distinct phases "
        "require a two-pass workflow so large air pores and fine fat globules are "
        "measured with separate detection scales."
        if diag["both_present"]
        else f"{pore_text}; {globule_text}. Only one phase is statistically dominant, "
        "so the validated Whipped Emulsion recipe (air pass enabled) is used instead "
        "of switching the classification logic on weak evidence."
    )
    focus_why = (
        "Two-pass is recommended because the pore and globule size distributions barely "
        "overlap: the air pass captures dark craters at the air-bubble scale, then the "
        "fat pass captures the fine matrix emulsion."
        if diag["both_present"]
        else "Air-bubbles-only keeps the dedicated dark-crater detector on the single "
        "dominant phase measured in this field of view; a second pass would only add "
        "runtime and false positives."
    )
    rows = [
        _comparison_row(
            "sem_sample_type_radio",
            "Sample Type",
            rec["sem_sample_type_radio"],
            current_params,
            sample_why,
        ),
        _comparison_row(
            "sem_detection_focus_radio",
            "AI Detection Target",
            rec["sem_detection_focus_radio"],
            current_params,
            focus_why,
        ),
        _comparison_row(
            "sem_min_air_um",
            "Min Air Bubble Size (µm)",
            rec["sem_min_air_um"],
            current_params,
            (
                f"min(10.0 µm, 0.65 × the smallest measured bubble "
                f"{diag['air_d_min']:.1f} µm) = {rec['sem_min_air_um']:.1f} µm after "
                "rounding DOWN to a clean laboratory step. The cap keeps the bound at "
                "or below the validated default, so 10-20 µm bubbles are never excluded "
                "even when every detected crater is large."
                if diag["has_pores"]
                else "No pore population was measured, so the canonical SEM default of "
                "10.0 µm is used (never the range left over from the previous image)."
            ),
        ),
        _comparison_row(
            "sem_max_air_um",
            "Max Air Bubble Size (µm)",
            rec["sem_max_air_um"],
            current_params,
            (
                f"max(100.0 µm, 1.35 × the largest measured bubble "
                f"{diag['air_d_max']:.1f} µm) = {rec['sem_max_air_um']:.1f} µm after "
                "rounding UP to a clean laboratory step. Ceiling rounding plus the 35% "
                "headroom guarantees that no detected bubble can ever be clipped, and "
                "the bound only drops below 100 µm when the whole frame is narrower "
                "than 80 µm."
                if diag["has_pores"]
                else "No pore population was measured, so the canonical SEM default of "
                "100.0 µm is used (never the range left over from the previous image)."
            ),
        ),
    ]
    rows.extend(
        [
            _comparison_row(
                "sem_min_fat_um",
                "Min Fat Particle Size (µm)",
                rec["sem_min_fat_um"],
                current_params,
                (
                    f"0.60 × the top-hat globule P5 ({diag['fat_p5']:.2f} µm) = "
                    f"{rec['sem_min_fat_um']:.2f} µm after rounding DOWN to a clean step: "
                    "sub-resolution texture is removed while the genuine small-droplet "
                    "tail is preserved."
                    if diag["has_fine"]
                    else "The fine-globule population was too sparse to re-bound, so the "
                    "canonical SEM default of 0.20 µm is used."
                ),
            ),
            _comparison_row(
                "sem_max_fat_um",
                "Max Fat Particle Size (µm)",
                rec["sem_max_fat_um"],
                current_params,
                (
                    f"max(5.0 µm, 1.40 × the globule P98 ({diag['fat_p98']:.2f} µm)) = "
                    f"{rec['sem_max_fat_um']:.1f} µm after rounding UP to a clean step: "
                    "the coarse tail of the matrix emulsion is captured without "
                    "swallowing air-crater rims."
                    if diag["has_fine"]
                    else "The fine-globule population was too sparse to re-bound, so the "
                    "canonical SEM default of 5.0 µm is used."
                ),
            ),
            _comparison_row(
                "shared_flow_slider",
                "flow_threshold",
                rec["shared_flow_slider"],
                current_params,
                (
                    f"Global contrast is {diag['contrast']:.0f} GL with a median pore "
                    f"shadow of {diag['shadow_depth']:.0f} GL; a stricter flow bound "
                    "keeps the learned flow field from leaking across blurred crater walls."
                ),
            ),
            _comparison_row(
                "shared_cellprob_slider",
                "cellprob_threshold",
                rec["shared_cellprob_slider"],
                current_params,
                (
                    f"Low contrast ({diag['contrast']:.0f} GL < 65) or deep crater shadow "
                    f"({diag['shadow_depth']:.0f} GL) means faint pores fall below the "
                    "default probability: -1.5 recovers them."
                    if diag["low_contrast"]
                    else f"Contrast ({diag['contrast']:.0f} GL) and crater shadow "
                    f"({diag['shadow_depth']:.0f} GL) are adequate, so -1.0 suppresses "
                    "background false positives."
                ),
            ),
            _comparison_row(
                "sem_air_circ_slider",
                "Air Bubbles Min Circularity",
                0.50,
                current_params,
                "Air bubbles are near-spherical; 0.50 removes irregular inter-particle "
                "gaps while tolerating partially occluded bubble rims.",
            ),
            _comparison_row(
                "shared_fat_circ_slider",
                "Matrix Fat Min Circularity",
                0.60,
                current_params,
                "Matrix fat globules are spherical but often touching; 0.60 keeps merged "
                "round pairs and rejects angular mineral debris.",
            ),
            _comparison_row(
                "sem_ads_circ_slider",
                "Adsorbed Fat Min Circularity",
                0.30,
                current_params,
                "Adsorbed fat sits as caps/patches on curved bubble walls, so a low bound "
                "(0.30) is required to retain these non-spherical interfacial objects.",
            ),
            _comparison_row(
                "sem_deepscan_checkbox",
                "Deep-Scan Adsorbed Fat inside Bubbles",
                rec["sem_deepscan_checkbox"],
                current_params,
                (
                    f"Crater interiors show internal variance (speck density "
                    f"{diag['speck_density']:.3f}), consistent with adsorbed fat globules "
                    "hiding in the dark shadow - a localized deep-scan recovers them."
                    if diag["internal_speck"]
                    else "Crater interiors are homogeneous, so a localized deep-scan would "
                    "only add runtime; deep-scan stays off for this image."
                ),
            ),
            _comparison_row(
                "shared_clahe_checkbox",
                "Enable CLAHE Preprocessing",
                True,
                current_params,
                "SEM craters and shadowed pore walls are locally low-contrast; CLAHE "
                "(clipLimit=2.0) normalises those dark regions before the network sees "
                "the image.",
            ),
            _comparison_row(
                "shared_model_type_select",
                "Cellpose Foundation Model",
                rec["shared_model_type_select"],
                current_params,
                (
                    f"Low contrast ({diag['contrast']:.0f} GL), deep crater shadow "
                    f"({diag['shadow_depth']:.0f} GL) or internal speckle (density "
                    f"{diag['speck_density']:.3f}) requires cyto3's broader multi-modality "
                    "training to separate faint pore walls from matrix texture."
                    if rec["shared_model_type_select"] == "cyto3"
                    else f"Contrast ({diag['contrast']:.0f} GL) and crater definition "
                    f"({diag['shadow_depth']:.0f} GL shadow) are sufficient for the faster "
                    "cyto2 baseline."
                ),
            ),
        ]
    )
    return rows


def diagnose_and_recommend_parameters(
    analysis_img: np.ndarray,
    pixel_to_um_ratio: float,
    is_light_mode: bool,
    current_params: dict,
) -> dict:
    """
    Measure the current micrograph and translate the optics into Cellpose settings.

    The advisor never mutates widget state: it returns a structured proposal that
    the UI shows in an interactive review card. Physical sizes are derived from the
    active calibration (``pixel_to_um_ratio`` µm/px) so both SEM and light
    microscopy modes produce values in µm.

    Args:
        analysis_img: Cropped grayscale micrograph (SEM footer already excluded).
        pixel_to_um_ratio: Active calibration in µm per pixel.
        is_light_mode: ``True`` for brightfield light microscopy, ``False`` for SEM.
        current_params: Mapping of Section 4 / 5 widget keys to their current values.

    Returns:
        dict: ``modality``, ``summary_metrics``, ``recommended_widget_state`` and
        ``comparison_rows`` (Parameter / Current / Recommended / Rationale).

    Raises:
        ValueError: If the analysis image is empty.
    """
    gray = np.asarray(analysis_img)
    if gray.size == 0:
        raise ValueError("Empty analysis image - load a micrograph before running the advisor.")

    if is_light_mode:
        diag = _diagnose_light_microscopy(gray, pixel_to_um_ratio, current_params)
        return {
            "modality": "💡 Light Microscopy",
            "summary_metrics": _light_summary_metrics(diag),
            "recommended_widget_state": dict(diag["recommended"]),
            "comparison_rows": _light_comparison_rows(diag, current_params),
        }

    diag = _diagnose_sem_micrograph(gray, pixel_to_um_ratio, current_params)
    return {
        "modality": "🔬 SEM Micrograph",
        "summary_metrics": _sem_summary_metrics(diag),
        "recommended_widget_state": dict(diag["recommended"]),
        "comparison_rows": _sem_comparison_rows(diag, current_params),
    }


# ---------------------------------------------------------------------------
# Visual Ruler: self-contained HTML5 caliper (no websocket, zero latency)
# ---------------------------------------------------------------------------
# The template is rendered with ``st.components.v1.html`` and therefore lives
# entirely in the browser: dragging a line never triggers a Streamlit rerun, so
# the measurement appears instantly and cannot be lost by a websocket hiccup.
#
# The markup is deliberately styled like a native Streamlit card (white card,
# 1px #e6e9ef border, 8px radius, "Source Sans Pro" typography, dark slate
# #31333F text) so the caliper blends into the light dashboard instead of
# looking like an embedded third-party widget.
#
# __NATIVE_W__ / __NATIVE_H__ are the *native* micrograph dimensions,
# __BITMAP_W__ / __BITMAP_H__ the embedded JPEG dimensions and __UM_PER_PX__ the
# active calibration. Because every pointer event is mapped from CSS pixels to
# bitmap pixels and then to native pixels, the µm value is exact regardless of
# browser zoom, canvas downscaling or CSS responsiveness.
_VISUAL_RULER_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: #ffffff; }
  body {
    font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px; color: #31333F;
    -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
  }
  .pp-ruler {
    width: 100%; background: #ffffff; border: 1px solid #e6e9ef;
    border-radius: 8px; overflow: hidden;
  }
  /* ---- Compact top bar: measurements left, compact controls right ---- */
  .pp-topbar {
    display: flex; flex-wrap: wrap; align-items: center;
    justify-content: space-between; gap: 6px 16px;
    padding: 7px 10px; border-bottom: 1px solid #e6e9ef; background: #ffffff;
  }
  .pp-metrics { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 16px; }
  .pp-metric { display: inline-flex; align-items: baseline; gap: 5px; white-space: nowrap; }
  .pp-label { font-size: 11.5px; font-weight: 400; color: #555867; }
  .pp-value { font-size: 14px; font-weight: 600; color: #31333F; font-variant-numeric: tabular-nums; }
  .pp-value.accent { color: #ff4b4b; }
  .pp-unit { font-size: 11.5px; font-weight: 400; color: #808495; font-variant-numeric: tabular-nums; }
  .pp-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; }
  .pp-control { display: inline-flex; align-items: center; gap: 5px; font-size: 11.5px; color: #555867; }
  input[type="color"] {
    width: 26px; height: 22px; padding: 0; border: 1px solid #e6e9ef;
    border-radius: 6px; background: #ffffff; cursor: pointer;
  }
  input[type="range"] { width: 76px; accent-color: #ff4b4b; cursor: pointer; }
  .pp-width-val { min-width: 28px; color: #808495; font-variant-numeric: tabular-nums; }
  button {
    font-family: inherit; font-size: 12px; font-weight: 400; line-height: 1.4;
    color: #31333F; background: #ffffff; border: 1px solid #e6e9ef;
    border-radius: 8px; padding: 3px 10px; cursor: pointer;
    transition: border-color .12s ease, color .12s ease, background .12s ease;
  }
  button:hover { border-color: #ff4b4b; color: #ff4b4b; }
  button:active { background: #f6f7f9; }
  /* ---- Micrograph stage: always full column width, zero letterboxing ---- */
  .pp-stage { position: relative; line-height: 0; background: #ffffff; }
  canvas { display: block; width: 100%; height: auto; cursor: crosshair; touch-action: none; }
  .pp-badge {
    position: absolute; transform: translate(-50%, -135%); pointer-events: none;
    background: #ffffff; border: 1px solid #e6e9ef; border-radius: 8px;
    padding: 3px 8px; line-height: 1.35; white-space: nowrap; opacity: 0;
    box-shadow: 0 2px 10px rgba(49, 51, 63, 0.14);
    transition: opacity .07s linear;
  }
  .pp-badge-um {
    display: block; font-size: 12.5px; font-weight: 600; color: #31333F;
    font-variant-numeric: tabular-nums;
  }
  .pp-badge-px {
    display: block; font-size: 10.5px; font-weight: 400; color: #808495;
    font-variant-numeric: tabular-nums;
  }
  /* ---- Auto-hiding measurement history (only after the first line) ---- */
  .pp-table-wrap { display: none; max-height: 168px; overflow-y: auto; border-top: 1px solid #e6e9ef; }
  .pp-table-wrap.visible { display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
  th, td { padding: 5px 10px; text-align: right; border-bottom: 1px solid #f0f2f6; }
  th {
    position: sticky; top: 0; background: #f6f7f9; color: #555867;
    font-size: 10.5px; font-weight: 600; letter-spacing: .03em; text-transform: uppercase;
  }
  th:first-child, td:first-child { text-align: left; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: #fafbfc; }
</style>
</head>
<body>
<div class="pp-ruler" id="pp-ruler">
  <div class="pp-topbar">
    <div class="pp-metrics">
      <span class="pp-metric">
        <span class="pp-label">Measured Size</span>
        <span class="pp-value accent" id="v-um">0.00 µm</span>
        <span class="pp-unit" id="v-px">(0.0 px)</span>
      </span>
      <span class="pp-metric">
        <span class="pp-label">Suggested Min (80%)</span>
        <span class="pp-value" id="v-min">0.00 µm</span>
      </span>
      <span class="pp-metric">
        <span class="pp-label">Suggested Max (120%)</span>
        <span class="pp-value" id="v-max">0.00 µm</span>
      </span>
    </div>
    <div class="pp-controls">
      <label class="pp-control" title="Measurement line colour">Line Color
        <input type="color" id="line-color" value="#ff4b4b" />
      </label>
      <label class="pp-control" title="Measurement line thickness">Line Width
        <input type="range" id="line-width" min="1" max="10" step="1" value="3" />
        <span class="pp-width-val" id="line-width-val">3 px</span>
      </label>
      <button type="button" id="btn-undo" title="Remove the last measurement line">↩️ Undo</button>
      <button type="button" id="btn-clear" title="Remove every measurement line">🗑️ Clear</button>
    </div>
  </div>
  <div class="pp-stage">
    <canvas id="ruler-canvas"></canvas>
    <div class="pp-badge" id="badge">
      <span class="pp-badge-um" id="badge-um">0.00 µm</span>
      <span class="pp-badge-px" id="badge-px">0.0 px</span>
    </div>
  </div>
  <div class="pp-table-wrap" id="table-wrap">
    <table>
      <thead>
        <tr><th>Line #</th><th>Length (px)</th><th>Length (µm)</th><th>Suggested Min (µm)</th><th>Suggested Max (µm)</th></tr>
      </thead>
      <tbody id="history"></tbody>
    </table>
  </div>
</div>
<script>
(function () {
  "use strict";

  // ---- Streamlit iframe bootstrap -----------------------------------------
  // The component lives in a same-origin iframe created by
  // ``st.components.v1.html``. Announcing componentReady and then
  // setFrameHeight keeps the iframe exactly as tall as its content, so the
  // auto-hiding history table never leaves a gap and is never clipped.
  function postToStreamlit(message) {
    try { window.parent.postMessage(message, "*"); } catch (err) { /* standalone */ }
  }
  postToStreamlit({ isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 });

  // ---- Native micrograph geometry & calibration (injected by app.py) --------
  var NATIVE_W = __NATIVE_W__;
  var NATIVE_H = __NATIVE_H__;
  var BITMAP_W = __BITMAP_W__;
  var BITMAP_H = __BITMAP_H__;
  var UM_PER_PX = __UM_PER_PX__;
  var IMAGE_SRC = "__IMAGE_SRC__";

  var root = document.getElementById("pp-ruler");
  var canvas = document.getElementById("ruler-canvas");
  var stage = canvas.parentNode;
  var ctx = canvas.getContext("2d");
  var badge = document.getElementById("badge");
  var badgeUm = document.getElementById("badge-um");
  var badgePx = document.getElementById("badge-px");
  var historyBody = document.getElementById("history");
  var tableWrap = document.getElementById("table-wrap");
  var colorInput = document.getElementById("line-color");
  var widthInput = document.getElementById("line-width");
  var widthLabel = document.getElementById("line-width-val");

  var lines = [];
  var drag = null;
  var preview = null;
  var lineSeq = 0;
  var bgReady = false;
  var bg = new Image();

  canvas.width = BITMAP_W;
  canvas.height = BITMAP_H;

  // Keep the surrounding iframe in sync with the rendered content height.
  var frameTimer = null;
  function syncFrameHeight() {
    if (!root) { return; }
    var height = Math.ceil(root.getBoundingClientRect().height) + 2;
    postToStreamlit({
      isStreamlitMessage: true,
      type: "streamlit:setFrameHeight",
      height: height
    });
  }
  function scheduleFrameHeight() {
    if (frameTimer) { window.clearTimeout(frameTimer); }
    frameTimer = window.setTimeout(syncFrameHeight, 30);
  }

  // ---- Geometry helpers ----------------------------------------------------
  // CSS pixel -> canvas bitmap pixel.
  function toBitmap(evt) {
    var rect = canvas.getBoundingClientRect();
    var scaleX = rect.width ? canvas.width / rect.width : 1;
    var scaleY = rect.height ? canvas.height / rect.height : 1;
    return {
      x: (evt.clientX - rect.left) * scaleX,
      y: (evt.clientY - rect.top) * scaleY
    };
  }

  // Bitmap pixel delta -> native micrograph pixel length.
  // X and Y are scaled independently so a non-square canvas / downscaled bitmap
  // can never skew the result.
  function nativePx(dxBitmap, dyBitmap) {
    var dx = dxBitmap * (NATIVE_W / BITMAP_W);
    var dy = dyBitmap * (NATIVE_H / BITMAP_H);
    return Math.sqrt(dx * dx + dy * dy);
  }

  function fmt(value, digits) {
    var number = Number(value);
    if (!isFinite(number)) { return (0).toFixed(digits); }
    return number.toFixed(digits);
  }

  // Bitmap pixels per CSS pixel, so strokes and labels stay readable while the
  // bitmap is scaled down to the column width.
  function bitmapScale() {
    var cssWidth = canvas.getBoundingClientRect().width;
    if (!(cssWidth > 0)) { return 1; }
    return BITMAP_W / cssWidth;
  }

  // ---- Live dashboard ------------------------------------------------------
  function updateDashboard(pixelLength) {
    var um = pixelLength * UM_PER_PX;
    document.getElementById("v-um").textContent = fmt(um, 2) + " µm";
    document.getElementById("v-px").textContent = "(" + fmt(pixelLength, 1) + " px)";
    document.getElementById("v-min").textContent = fmt(um * 0.8, 2) + " µm";
    document.getElementById("v-max").textContent = fmt(um * 1.2, 2) + " µm";
    return um;
  }

  function updateBadge(evt, pixelLength, um) {
    if (!(um > 0)) { hideBadge(); return; }
    var rect = stage.getBoundingClientRect();
    badge.style.left = (evt.clientX - rect.left) + "px";
    badge.style.top = (evt.clientY - rect.top) + "px";
    badge.style.opacity = "1";
    badgeUm.textContent = fmt(um, 2) + " µm";
    badgePx.textContent = fmt(pixelLength, 1) + " px";
  }

  function hideBadge() { badge.style.opacity = "0"; }

  // Keep stroke thickness visually constant even when the bitmap is downscaled.
  function strokeWidth() {
    var requested = parseFloat(widthInput.value) || 3;
    return Math.max(1, requested * bitmapScale());
  }

  // ---- Rendering -----------------------------------------------------------
  function drawScaleBar() {
    if (!(UM_PER_PX > 0)) { return; }
    var candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];
    var target = BITMAP_W * 0.16;
    var chosen = candidates[0];
    for (var i = 0; i < candidates.length; i++) {
      var candidatePx = (candidates[i] / UM_PER_PX) * (BITMAP_W / NATIVE_W);
      if (candidatePx <= target) { chosen = candidates[i]; }
    }
    var barPx = (chosen / UM_PER_PX) * (BITMAP_W / NATIVE_W);
    var pad = Math.max(9, BITMAP_W * 0.012);
    var y = BITMAP_H - pad;
    var fontSize = Math.max(11, Math.round(BITMAP_W * 0.015));

    ctx.save();
    ctx.lineCap = "butt";
    ctx.strokeStyle = "rgba(0, 0, 0, 0.85)";
    ctx.lineWidth = Math.max(4, BITMAP_W * 0.006);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(pad + barPx, y); ctx.stroke();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = Math.max(2, BITMAP_W * 0.0025);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(pad + barPx, y); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad, y - fontSize * 0.45); ctx.lineTo(pad, y + fontSize * 0.45);
    ctx.moveTo(pad + barPx, y - fontSize * 0.45); ctx.lineTo(pad + barPx, y + fontSize * 0.45);
    ctx.stroke();
    ctx.font = "bold " + fontSize + "px sans-serif";
    ctx.textBaseline = "bottom";
    var label = chosen + " µm";
    ctx.lineWidth = Math.max(2, BITMAP_W * 0.0035);
    ctx.strokeStyle = "rgba(0, 0, 0, 0.85)";
    ctx.strokeText(label, pad, y - fontSize * 0.7);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, pad, y - fontSize * 0.7);
    ctx.restore();
  }

  function drawSegment(segment, isPreview) {
    var lw = strokeWidth();
    ctx.save();
    ctx.strokeStyle = segment.color;
    ctx.fillStyle = segment.color;
    ctx.lineWidth = lw;
    ctx.lineCap = "round";
    if (isPreview) { ctx.setLineDash([lw * 2.4, lw * 2.0]); }
    ctx.beginPath();
    ctx.moveTo(segment.x1, segment.y1);
    ctx.lineTo(segment.x2, segment.y2);
    ctx.stroke();
    ctx.setLineDash([]);

    var radius = Math.max(2.5, lw * 1.25);
    ctx.beginPath(); ctx.arc(segment.x1, segment.y1, radius, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(segment.x2, segment.y2, radius, 0, Math.PI * 2); ctx.fill();

    if (!isPreview) {
      var fontSize = Math.max(12, Math.round(BITMAP_W * 0.017));
      var text = fmt(segment.um, 2) + " µm";
      var midX = (segment.x1 + segment.x2) / 2;
      var midY = (segment.y1 + segment.y2) / 2 - fontSize * 0.85;
      ctx.font = "bold " + fontSize + "px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      var textWidth = ctx.measureText(text).width;
      var padX = fontSize * 0.4;
      var boxHeight = fontSize * 1.5;
      // Light chip keeps the label readable on any micrograph without breaking
      // the light UI look of the component.
      ctx.fillStyle = "rgba(255, 255, 255, 0.93)";
      ctx.fillRect(
        midX - textWidth / 2 - padX, midY - boxHeight / 2,
        textWidth + padX * 2, boxHeight
      );
      ctx.strokeStyle = segment.color;
      ctx.lineWidth = Math.max(1, lw * 0.35);
      ctx.strokeRect(
        midX - textWidth / 2 - padX, midY - boxHeight / 2,
        textWidth + padX * 2, boxHeight
      );
      ctx.fillStyle = "#31333F";
      ctx.fillText(text, midX, midY);
    }
    ctx.restore();
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (bgReady) { ctx.drawImage(bg, 0, 0, BITMAP_W, BITMAP_H); }
    drawScaleBar();
    for (var i = 0; i < lines.length; i++) { drawSegment(lines[i], false); }
    if (preview) { drawSegment(preview, true); }
  }

  // ---- Measurement history (hidden until the first line is drawn) ----------
  function renderHistory() {
    historyBody.innerHTML = "";
    if (lines.length > 0) {
      tableWrap.classList.add("visible");
    } else {
      tableWrap.classList.remove("visible");
    }
    for (var i = lines.length - 1; i >= 0; i--) {
      var line = lines[i];
      var cells = [
        "#" + line.id,
        fmt(line.px, 1),
        fmt(line.um, 2),
        fmt(line.um * 0.8, 2),
        fmt(line.um * 1.2, 2)
      ];
      var row = document.createElement("tr");
      for (var c = 0; c < cells.length; c++) {
        var cell = document.createElement("td");
        cell.textContent = cells[c];
        row.appendChild(cell);
      }
      historyBody.appendChild(row);
    }
    scheduleFrameHeight();
  }

  // ---- Interaction ---------------------------------------------------------
  function finishDrag(evt) {
    if (!drag) { return; }
    var pixelLength = nativePx(drag.x2 - drag.x1, drag.y2 - drag.y1);
    if (pixelLength >= 1.5) {
      lineSeq += 1;
      lines.push({
        id: lineSeq,
        x1: drag.x1, y1: drag.y1, x2: drag.x2, y2: drag.y2,
        px: pixelLength,
        um: pixelLength * UM_PER_PX,
        color: colorInput.value
      });
      updateDashboard(pixelLength);
      renderHistory();
    }
    drag = null;
    preview = null;
    hideBadge();
    draw();
    if (evt && evt.preventDefault) { evt.preventDefault(); }
  }

  canvas.addEventListener("pointerdown", function (evt) {
    if (evt.button !== undefined && evt.button > 0) { return; }
    var point = toBitmap(evt);
    drag = { x1: point.x, y1: point.y, x2: point.x, y2: point.y };
    preview = { x1: point.x, y1: point.y, x2: point.x, y2: point.y, color: colorInput.value };
    if (canvas.setPointerCapture) {
      try { canvas.setPointerCapture(evt.pointerId); } catch (err) { /* ignore */ }
    }
    updateDashboard(0);
    hideBadge();
    draw();
    evt.preventDefault();
  });

  canvas.addEventListener("pointermove", function (evt) {
    if (!drag) { return; }
    var point = toBitmap(evt);
    drag.x2 = point.x;
    drag.y2 = point.y;
    preview.x2 = point.x;
    preview.y2 = point.y;
    var pixelLength = nativePx(drag.x2 - drag.x1, drag.y2 - drag.y1);
    var um = updateDashboard(pixelLength);
    updateBadge(evt, pixelLength, um);
    draw();
    evt.preventDefault();
  });

  canvas.addEventListener("pointerup", finishDrag);
  canvas.addEventListener("pointercancel", function () {
    drag = null; preview = null; hideBadge(); draw();
  });
  canvas.addEventListener("pointerleave", function () {
    if (!drag) { hideBadge(); }
  });

  document.getElementById("btn-undo").addEventListener("click", function () {
    if (lines.length) { lines.pop(); }
    renderHistory();
    if (lines.length) {
      updateDashboard(lines[lines.length - 1].px);
    } else {
      updateDashboard(0);
    }
    hideBadge();
    draw();
  });

  document.getElementById("btn-clear").addEventListener("click", function () {
    lines = [];
    lineSeq = 0;
    renderHistory();
    updateDashboard(0);
    hideBadge();
    draw();
  });

  colorInput.addEventListener("input", function () { draw(); });
  widthInput.addEventListener("input", function () {
    widthLabel.textContent = widthInput.value + " px";
    draw();
  });

  // ---- Boot ----------------------------------------------------------------
  window.addEventListener("resize", scheduleFrameHeight);

  bg.onload = function () {
    bgReady = true;
    draw();
    scheduleFrameHeight();
  };
  bg.onerror = function () {
    draw();
    scheduleFrameHeight();
  };
  bg.src = IMAGE_SRC;

  renderHistory();
  updateDashboard(0);
  draw();
  syncFrameHeight();
  window.setTimeout(syncFrameHeight, 250);
})();
</script>
</body>
</html>
"""


def build_visual_ruler_html(
    image_src: str,
    native_w: int,
    native_h: int,
    bitmap_w: int,
    bitmap_h: int,
    um_per_px: float,
) -> str:
    """Render the self-contained HTML5 caliper used by the Interactive Dashboard.

    Args:
        image_src: ``data:image/jpeg;base64,...`` URL of the measurement image
            (raw micrograph or segmentation overlay).
        native_w: Native image width in calibration pixels.
        native_h: Native image height in calibration pixels.
        bitmap_w: Width of the embedded JPEG (after optional downscaling).
        bitmap_h: Height of the embedded JPEG.
        um_per_px: Active calibration factor (micrometres per native pixel).

    Returns:
        A complete HTML document string for ``st.components.v1.html``.
    """
    replacements = {
        "__NATIVE_W__": str(int(native_w)),
        "__NATIVE_H__": str(int(native_h)),
        "__BITMAP_W__": str(int(bitmap_w)),
        "__BITMAP_H__": str(int(bitmap_h)),
        "__UM_PER_PX__": repr(float(um_per_px)),
        "__IMAGE_SRC__": image_src,
    }
    html = _VISUAL_RULER_HTML
    for token, value in replacements.items():
        html = html.replace(token, value)
    return html


def encode_image_data_url(
    image: np.ndarray,
    max_width: int = 1600,
    quality: int = 92,
) -> Tuple[str, int, int]:
    """Encode an image as a self-contained ``data:image/jpeg;base64,...`` URL.

    A high-quality JPEG keeps the HTML component payload small (a 1600 px
    micrograph lands well below 1 MB), so the browser-side caliper appears
    instantly, whereas a multi-megabyte base64 PNG used to stall the component.
    The returned bitmap dimensions are used by the browser-side caliper to
    convert dragged (CSS) distances back into *native* micrograph pixels, so
    downscaling never changes the physical measurement.

    Args:
        image: Grayscale (H, W) or colour (H, W, 3/4) uint8 array.
        max_width: Downscale ceiling for the embedded bitmap; ``0`` disables it.
        quality: JPEG quality (1-95); 92 is visually lossless for this purpose.

    Returns:
        ``(data_url, bitmap_width, bitmap_height)``.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        pil_image = Image.fromarray(array).convert("RGB")
    else:
        pil_image = Image.fromarray(
            np.ascontiguousarray(array[:, :, :3])
        ).convert("RGB")

    if max_width and pil_image.width > max_width:
        ratio = max_width / float(pil_image.width)
        new_size = (
            int(max_width),
            max(1, int(round(pil_image.height * ratio))),
        )
        resample = getattr(Image, "Resampling", Image).LANCZOS
        pil_image = pil_image.resize(new_size, resample)

    buffer = io.BytesIO()
    pil_image.save(
        buffer,
        format="JPEG",
        quality=int(quality),
        optimize=True,
        subsampling=0,
    )
    data_url = (
        "data:image/jpeg;base64,"
        + base64.b64encode(buffer.getvalue()).decode("ascii")
    )
    return data_url, int(pil_image.width), int(pil_image.height)


# Configure page layout and aesthetics
st.set_page_config(
    page_title="PhasePix - Quantitative Microstructure & Colloid Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Running Cellpose AI segmentation...")
def cached_segmentation(
    image: np.ndarray,
    model_name: str = "cyto2",
    custom_model_path: Optional[str] = None,
    diameter: Optional[float] = 30.0,
    flow_threshold: float = 0.7,
    cellprob_threshold: float = -1.0,
    enable_clahe: bool = True,
) -> np.ndarray:
    """
    Run and cache Cellpose segmentation on cropped analysis image array.
    Streamlit caching guarantees that changing downstream parameters
    (e.g., intensity threshold slider or spatial scale) will NEVER
    re-trigger the expensive deep learning inference.

    Args:
        image: 2D numpy array of the micrograph to analyze.
        model_name: Cellpose model identifier or 'Custom Model File...'.
        custom_model_path: Optional filesystem path to user-supplied custom .pth weights.
        diameter: Estimated average particle diameter in pixels.
        flow_threshold: Flow error threshold for detection.
        cellprob_threshold: Cell probability threshold for detection.
        enable_clahe: Whether to apply CLAHE preprocessing for low-contrast pores.

    Returns:
        np.ndarray: 2D integer array of segmented object masks.
    """
    model = get_cellpose_model(model_type=model_name, custom_model_path=custom_model_path)
    masks = run_inference(
        model=model,
        image=image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        enable_clahe=enable_clahe,
    )
    return masks


@st.cache_data(show_spinner=False, max_entries=16)
def decode_image_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes into a 2D grayscale uint8 numpy array.
    Supports JPG, PNG, and multi-mode TIFF files.

    Cached via ``@st.cache_data`` (16 entries) because the same bytes are decoded
    several times per rerun (sidebar calibration probe, preview rendering and the
    analysis pass).  Re-decoding a 20 MP TIFF on every widget interaction is pure
    overhead - the decoded array is a deterministic function of the raw bytes.
    """
    try:
        # First attempt decoding via tifffile (handles scientific 16-bit/8-bit TIFFs)
        try:
            arr = tifffile.imread(io.BytesIO(file_bytes))
            if isinstance(arr, np.ndarray) and arr.size > 0:
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                if arr.dtype != np.uint8:
                    # Normalize to 0-255 uint8 range for consistent intensity metrics
                    arr = (
                        (arr - arr.min()) / (max(arr.max() - arr.min(), 1e-6)) * 255.0
                    ).astype(np.uint8)
                return arr
        except Exception:
            pass

        # Fallback to standard PIL decoding
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
        return np.array(pil_img, dtype=np.uint8)

    except Exception as exc:
        st.error(f"Failed to read image file: {exc}")
        raise exc


@st.cache_data(show_spinner=False, max_entries=16)
def cached_detect_scale_bar(
    file_bytes: bytes,
    footer_height: int,
) -> Tuple[int, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Cached SEM scale-bar probe used by the sidebar calibration panel.

    ``detect_scale_bar`` runs a full morphological search (thresholding, contour
    analysis, tick pairing) on the uploaded micrograph.  Re-running it on every
    widget interaction of the Streamlit rerun loop is wasteful, so both the image
    decoding and the detection are memoised on ``(file_bytes, footer_height)``.

    Args:
        file_bytes: Raw bytes of the uploaded micrograph.
        footer_height: Height in pixels of the databar/footer band to search in.

    Returns:
        Tuple[int, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        Tick-to-tick bar length in pixels, an annotated RGB preview crop, and the
        bar bounding box ``(x, y, w, h)`` in full-image coordinates.
    """
    probe_img = decode_image_bytes(file_bytes)
    return detect_scale_bar(probe_img, footer_height=footer_height)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_extract_metrics(
    original_image: np.ndarray,
    mask: np.ndarray,
    pixel_to_um_ratio: float,
) -> pd.DataFrame:
    """Cached ``regionprops`` extraction for a segmentation mask.

    ``extract_metrics`` (from ``analyzer.py``) walks every labeled object to compute
    area, perimeter-derived circularity, convex-hull solidity and the fitted ellipse
    axes.  On micrographs with thousands of particles this dominates runtime, yet its
    output depends ONLY on ``(image, mask, calibration)`` - it must never re-run when
    the user merely moves an opacity slider, edits a size range or switches tabs.

    Args:
        original_image: 2D grayscale image the mask was computed on.
        mask: Labeled integer mask (0 = background).
        pixel_to_um_ratio: Spatial calibration in micrometres per pixel.

    Returns:
        pd.DataFrame: The morphological/photometric metrics table from
        ``analyzer.extract_metrics`` (a fresh copy on every cache hit).
    """
    return extract_metrics(
        original_image=original_image,
        mask=mask,
        pixel_to_um_ratio=pixel_to_um_ratio,
    )


def extract_scale_um_from_filename(image_name: str) -> float:
    """
    Extract nominal scale-bar length in micrometers from a filename.

    Looks for patterns like '_5um_', '_200um_', '5.00 um' via
    re.search(r'(\\d+(?:\\.\\d+)?)\\s*um', ..., re.IGNORECASE).

    Returns:
        float: Parsed µm value, or 0.0 if not found.
    """
    if not image_name:
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)\s*um", str(image_name), re.IGNORECASE)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return 0.0


def create_sample_sem_image() -> bytes:
    """Generate a realistic synthetic SEM micrograph byte stream for instant demo testing."""
    height, width = 400, 400
    rng = np.random.default_rng(42)
    # Background matrix
    base = np.full((height, width), 130, dtype=np.int16)
    grid_y, grid_x = np.ogrid[:height, :width]

    # Air bubbles (dark pores in SEM, low mean intensity < 100)
    bubble_centers = [(80, 90, 26), (100, 280, 32), (250, 100, 22), (320, 260, 28)]
    for cy, cx, r in bubble_centers:
        dist_sq = (grid_y - cy) ** 2 + (grid_x - cx) ** 2
        mask = dist_sq <= r ** 2
        base[mask] = 50 + rng.integers(-8, 8, np.sum(mask))

    # Fat particles (brighter features in SEM, high mean intensity > 150)
    fat_centers = [(180, 200, 35), (290, 320, 24), (120, 190, 20), (330, 120, 30), (220, 300, 25)]
    for cy, cx, r in fat_centers:
        dist_sq = (grid_y - cy) ** 2 + (grid_x - cx) ** 2
        mask = dist_sq <= r ** 2
        base[mask] = 205 + rng.integers(-10, 10, np.sum(mask))

    # Add realistic SEM Gaussian noise
    base += rng.normal(0, 7, (height, width)).astype(np.int16)

    # Add SEM databar band with a 150px white calibration scale bar line
    base[355:, :] = 20
    base[375:378, 200:350] = 255

    img_uint8 = np.clip(base, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()




def get_local_images(directory: str = "images") -> list:
    """Return list of supported image paths from the local directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    supported_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in supported_exts])


IMAGE_ROOT_DIR = Path("images")
SEM_IMAGE_DIR = IMAGE_ROOT_DIR / "sem"
LIGHT_IMAGE_DIR = IMAGE_ROOT_DIR / "light"


def ensure_modality_image_folders() -> None:
    """Create images/, images/sem/ and images/light/ automatically on startup."""
    for _dir in (IMAGE_ROOT_DIR, SEM_IMAGE_DIR, LIGHT_IMAGE_DIR):
        _dir.mkdir(parents=True, exist_ok=True)


def get_modality_images(is_light_mode: bool) -> Tuple[list, str, bool]:
    """
    Return the image list for the active imaging modality.

    Light Microscopy reads ``images/light/`` and SEM reads ``images/sem/``. If the
    modality folder is empty, files directly under ``images/`` are listed instead
    so existing projects keep working.

    Args:
        is_light_mode: Whether the sidebar modality switch is set to Light Microscopy.

    Returns:
        Tuple[list, str, bool]: (image_paths, primary_folder, used_legacy_fallback)
    """
    ensure_modality_image_folders()
    primary_dir = LIGHT_IMAGE_DIR if is_light_mode else SEM_IMAGE_DIR
    modality_files = get_local_images(str(primary_dir))
    if modality_files:
        return modality_files, str(primary_dir), False
    return get_local_images(str(IMAGE_ROOT_DIR)), str(primary_dir), True


def _rasterize_manual_air_bubbles(
    manual_items: Optional[list],
    shape: Tuple[int, int],
    phase_overrides: Optional[dict] = None,
) -> np.ndarray:
    """Rasterize manual objects classified as air bubbles into a boolean canvas."""
    canvas = np.zeros(shape[:2], dtype=np.uint8)
    if not manual_items:
        return canvas.astype(bool)
    overrides = phase_overrides or {}
    for item in manual_items:
        if not isinstance(item, dict):
            continue
        obj_id = item.get("id")
        phase = str(
            overrides.get(obj_id, item.get("assigned_class", "air_bubbles"))
        ).lower()
        if phase != "air_bubbles":
            continue
        polygon = item.get("polygon")
        points = polygon.get("points") if isinstance(polygon, dict) else None
        if points is None:
            continue
        try:
            pts = np.asarray(points, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
            continue
        cv2.fillPoly(canvas, [np.round(pts[:, :2]).astype(np.int32)], 1)
    return canvas.astype(bool)


def _lut_label_filter(
    mask: np.ndarray,
    labels: Iterable[int],
    keep: bool = True,
) -> np.ndarray:
    """Filter a labeled mask through an O(1) NumPy look-up table (LUT).

    ``np.where(np.isin(mask, list(ids)), mask, 0)`` sorts every label of the *whole*
    image and binary-searches it, which dominates runtime on multi-megapixel masks
    holding thousands of objects.  Building a dense ``mask.max() + 1`` boolean LUT and
    gathering it with the mask is a single vectorised pass:

        keep_lut = np.zeros(int(mask.max()) + 1, dtype=bool)
        keep_lut[valid_ids] = True
        mask = mask * keep_lut[mask]

    Args:
        mask: Labeled 2D (or ``(H, W, 1)``) integer mask, 0 = background.
        labels: Object ids to keep (``keep=True``) or to zero out (``keep=False``).
        keep: When True only ``labels`` survive, otherwise ``labels`` are removed.

    Returns:
        np.ndarray: Filtered 2D mask with the original dtype and shape.
    """
    mask_2d = mask.squeeze() if mask.ndim == 3 else mask
    if not np.issubdtype(mask_2d.dtype, np.integer):
        mask_2d = mask_2d.astype(np.int32)
    if mask_2d.size == 0:
        return mask_2d

    max_label = int(mask_2d.max())
    if max_label <= 0:
        # Only background present - nothing to keep or drop.
        return mask_2d

    accepted = np.zeros(max_label + 1, dtype=bool) if keep else np.ones(max_label + 1, dtype=bool)
    # Background must always stay background, in both modes.
    accepted[0] = True
    for value in labels:
        try:
            label = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < label <= max_label:
            accepted[label] = keep

    return np.where(accepted[mask_2d], mask_2d, 0).astype(mask_2d.dtype, copy=False)


def _remove_nested_matryoshka_masks(
    df_in: pd.DataFrame,
    mask_arr: Optional[np.ndarray],
    prior_occupied: Optional[np.ndarray] = None,
    overlap_thresh: float = 0.50,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], np.ndarray]:
    """Remove nested duplicate detections ("matryoshka" masks).

    Auto objects (id < 10000) are processed from the largest to the smallest. For each
    object the hole-filled external contour is compared with the canvas already occupied
    by accepted bigger objects; if at least ``overlap_thresh`` of the filled footprint
    lies inside that canvas, the object is discarded. Manual objects (id >= 10000) are
    always preserved.

    Args:
        df_in: Metrics DataFrame with 'id' and 'area_um2'/'diameter_um'.
        mask_arr: Optional labeled mask aligned with ``prior_occupied``.
        prior_occupied: Optional boolean canvas of already accepted regions (e.g. manual
            air bubbles) that should suppress auto detections lying inside them.
        overlap_thresh: Rejection threshold on the filled-footprint overlap fraction.

    Returns:
        Tuple[pd.DataFrame, Optional[np.ndarray], np.ndarray]: cleaned DataFrame, cleaned
        mask (rejected ids zeroed) and the updated occupied canvas.
    """
    if mask_arr is None:
        occupied = (
            prior_occupied.astype(bool, copy=True)
            if prior_occupied is not None
            else np.zeros((0, 0), dtype=bool)
        )
        return df_in, mask_arr, occupied

    mask_2d = mask_arr.squeeze() if mask_arr.ndim == 3 else mask_arr
    if not np.issubdtype(mask_2d.dtype, np.integer):
        mask_2d = mask_2d.astype(np.int32)

    if prior_occupied is not None and prior_occupied.shape[:2] == mask_2d.shape[:2]:
        occupied = prior_occupied.astype(bool, copy=True)
    else:
        occupied = np.zeros(mask_2d.shape[:2], dtype=bool)

    if df_in is None or df_in.empty:
        return df_in, mask_arr, occupied

    sort_col = "area_um2" if "area_um2" in df_in.columns else "diameter_um"
    auto_df = df_in[df_in["id"] < 10000].sort_values(by=sort_col, ascending=False)
    if auto_df.empty:
        return df_in, mask_arr, occupied

    try:
        label_slices = find_objects(mask_2d)
    except Exception:
        label_slices = None

    valid_auto_ids: set = set()
    for _, row in auto_df.iterrows():
        obj_id = int(row["id"])
        sl = None
        if label_slices is not None and 0 < obj_id <= len(label_slices):
            sl = label_slices[obj_id - 1]
        if sl is None:
            full_mask = mask_2d == obj_id
            if not np.any(full_mask):
                continue
            ys, xs = np.nonzero(full_mask)
            sl = (
                slice(int(ys.min()), int(ys.max()) + 1),
                slice(int(xs.min()), int(xs.max()) + 1),
            )

        local = (mask_2d[sl] == obj_id).astype(np.uint8)
        if not np.any(local):
            continue
        filled = np.zeros(local.shape, dtype=np.uint8)
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        filled_bool = filled.astype(bool)
        filled_area = float(np.count_nonzero(filled_bool))
        if filled_area <= 0:
            continue

        overlap_fraction = (
            float(np.count_nonzero(filled_bool & occupied[sl])) / filled_area
        )
        if overlap_fraction >= float(overlap_thresh):
            continue

        valid_auto_ids.add(obj_id)
        occupied[sl] |= filled_bool

    manual_ids = set(df_in.loc[df_in["id"] >= 10000, "id"].tolist())
    valid_ids = valid_auto_ids | manual_ids
    df_out = df_in[df_in["id"].isin(valid_ids)].copy().reset_index(drop=True)
    mask_out = _lut_label_filter(mask_2d, valid_ids, keep=True)
    return df_out, mask_out, occupied


@st.cache_data(show_spinner=False, max_entries=8)
def cached_remove_nested_matryoshka_masks(
    df_in: pd.DataFrame,
    mask_arr: Optional[np.ndarray],
    prior_occupied: Optional[np.ndarray] = None,
    overlap_thresh: float = 0.50,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], np.ndarray]:
    """Streamlit-cached wrapper around :func:`_remove_nested_matryoshka_masks`.

    The de-duplication pass walks every auto object (largest first), fills its external
    contour and measures the overlap with the already-accepted canvas - i.e. one
    ``cv2.findContours`` call per particle.  The result is a pure function of the
    metrics table, the mask and the prior canvas, so it is memoised to keep slider,
    tab and opacity re-runs instantaneous.

    Args:
        df_in: Metrics DataFrame with 'id' and 'area_um2'/'diameter_um'.
        mask_arr: Labeled mask aligned with ``df_in`` (0 = background).
        prior_occupied: Optional boolean canvas of already accepted regions.
        overlap_thresh: Rejection threshold on the filled-footprint overlap fraction.

    Returns:
        Tuple[pd.DataFrame, Optional[np.ndarray], np.ndarray]: cleaned DataFrame,
        cleaned mask and the updated occupied canvas.
    """
    return _remove_nested_matryoshka_masks(
        df_in=df_in,
        mask_arr=mask_arr,
        prior_occupied=prior_occupied,
        overlap_thresh=overlap_thresh,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_prune_nested_auto_objects(
    df_in: pd.DataFrame,
    mask_arr: np.ndarray,
    overlap_thresh: float = 0.60,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Cached "matryoshka" cleanup of the combined segmentation mask (Step 3b).

    Auto objects (id < 10000) are visited from the largest to the smallest.  Each
    external contour is hole-filled and compared with the canvas already occupied by
    accepted, larger objects; candidates lying at least ``overlap_thresh`` of their
    own area inside that canvas are detection artifacts and are dropped.  Manual
    objects (id >= 10000) are always preserved.

    The bookkeeping (``cv2.findContours`` + ``cv2.drawContours`` per particle, plus
    ``find_objects`` slicing) is the single most expensive Python-level loop in the
    analysis, so the whole pass is memoised on ``(df, mask, threshold)``: changing an
    opacity slider, a size range or a tab must never re-run it.

    Args:
        df_in: Metrics DataFrame with 'id' and 'area_um2'/'diameter_um'.
        mask_arr: Labeled mask aligned with ``df_in`` (0 = background).
        overlap_thresh: Rejection threshold on the object-area overlap fraction.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: cleaned DataFrame and cleaned mask (rejected
        labels zeroed out with an O(1) look-up table).
    """
    auto_df = df_in[df_in["id"] < 10000].sort_values(by="area_um2", ascending=False)
    valid_auto_ids = []
    occupied_canvas = np.zeros(mask_arr.shape, dtype=bool)
    masks_int = (
        mask_arr
        if np.issubdtype(mask_arr.dtype, np.integer)
        else mask_arr.astype(np.int32)
    )
    label_slices = None
    try:
        label_slices = find_objects(masks_int)
    except Exception:
        label_slices = None

    for _, row in auto_df.iterrows():
        obj_id = int(row["id"])
        sl = None
        if label_slices is not None and 0 < obj_id <= len(label_slices):
            sl = label_slices[obj_id - 1]
        if sl is None:
            obj_mask_full = masks_int == obj_id
            if not np.any(obj_mask_full):
                continue
            ys, xs = np.nonzero(obj_mask_full)
            sl = (
                slice(int(ys.min()), int(ys.max()) + 1),
                slice(int(xs.min()), int(xs.max()) + 1),
            )
        obj_mask = masks_int[sl] == obj_id
        obj_area = float(np.count_nonzero(obj_mask))
        if obj_area <= 0:
            continue
        overlap_area = float(np.count_nonzero(obj_mask & occupied_canvas[sl]))
        # If the object lies more than 60% inside an already accepted bigger object,
        # it is a "matryoshka" detection artifact and is skipped.
        if overlap_area / obj_area < float(overlap_thresh):
            valid_auto_ids.append(obj_id)
            filled = np.zeros(obj_mask.shape, dtype=np.uint8)
            contours, _ = cv2.findContours(
                obj_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
            occupied_canvas[sl] |= filled.astype(bool)

    # Keep valid auto objects and all manual objects (id >= 10000)
    valid_all_ids = set(valid_auto_ids) | set(
        df_in[df_in["id"] >= 10000]["id"].tolist()
    )
    if not valid_all_ids:
        return df_in, mask_arr

    df_out = df_in[df_in["id"].isin(valid_all_ids)].copy().reset_index(drop=True)
    # Clean the mask itself from rejected artifacts through a fast LUT gather
    mask_out = _lut_label_filter(mask_arr, valid_all_ids, keep=True)
    return df_out, mask_out


# ---------------- SIDEBAR CONFIGURATION ----------------
with st.sidebar:
    # ── Pre-widget session-state synchronisation ─────────────────────────────
    # The AI Parameter Advisor "Apply" action and the preset loader stage their
    # widget values here; writing them BEFORE any widget is instantiated avoids
    # StreamlitAPIException ("cannot be modified after widget is instantiated").
    _pending_parameter_updates = st.session_state.pop("_pending_parameter_updates", None)
    if _pending_parameter_updates:
        for _param_key, _param_value in _pending_parameter_updates.items():
            st.session_state[_param_key] = _param_value
        # Drop sticky particle-editor table state so the new recipe reclassifies
        # immediately on this very run.
        for _stale_editor_key in [
            k for k in list(st.session_state.keys())
            if str(k).startswith("particle_data_editor")
        ]:
            del st.session_state[_stale_editor_key]
        st.session_state["manual_phase_overrides"] = {}
        st.session_state["force_class_refresh"] = True
        # Keep the profile tracker aligned with the applied profile so the
        # profile-driven defaults below cannot overwrite the applied recipe.
        if "light_emulsion_profile_radio" in _pending_parameter_updates:
            st.session_state["light_profile_tracker"] = _pending_parameter_updates[
                "light_emulsion_profile_radio"
            ]

    _preset_saved_flash = st.session_state.pop("_preset_saved_flash", None)
    if _preset_saved_flash:
        st.toast(f"💾 Saved preset: {_preset_saved_flash}", icon="💾")
    _preset_loaded_flash = st.session_state.pop("_preset_loaded_flash", None)
    if _preset_loaded_flash:
        st.toast(f"📂 Loaded preset: {_preset_loaded_flash}", icon="📂")
    _preset_deleted_flash = st.session_state.pop("_preset_deleted_flash", None)
    if _preset_deleted_flash:
        st.toast(f"🗑️ Deleted preset: {_preset_deleted_flash}", icon="🗑️")
    _preset_restored_flash = st.session_state.pop("_preset_restored_flash", None)
    if _preset_restored_flash:
        st.toast(f"🔄 Restored built-in presets: {_preset_restored_flash}", icon="🔄")

    st.title("🔬 PhasePix Controls")
    st.markdown("Quantitative SEM & Light Microscopy Colloid Analyzer.")
    st.caption("📦 **Build: 2026.10.30 (v1.7.3 - Native Speed Restoration)**")

    # Hardware Acceleration Status — the fastest backend is picked automatically at
    # startup by get_device(): NVIDIA CUDA -> Apple Metal (MPS) -> CPU.
    device, is_gpu, accelerator_name = detect_hardware()
    if device.type == "cuda":
        st.success(f"🚀 Hardware Acceleration: CUDA ({accelerator_name})")
    elif device.type == "mps":
        st.success("🚀 Hardware Acceleration: Apple Metal (MPS)")
    else:
        st.info("🖥️ Hardware Backend: CPU Mode")
        with st.expander("ℹ️ GPU Acceleration Guide", expanded=False):
            st.markdown(
                "**PhasePix is currently running in CPU mode.**\n\n"
                "Hardware acceleration is detected and enabled **automatically** at startup:\n"
                "* **NVIDIA GPU (Windows/Linux):** Automatically activates CUDA acceleration "
                "if a compatible NVIDIA GPU and up-to-date NVIDIA Driver (531+) are installed "
                "on this PC.\n"
                "* **Apple Silicon (macOS):** Automatically activates Apple Metal (MPS) "
                "acceleration.\n"
                "* **Standard PC (Intel / AMD Graphics):** Runs safely in CPU mode. All features "
                "work identically, though neural network segmentation takes slightly longer per "
                "image.\n"
            )

    st.divider()

    st.subheader("0. Imaging Modality")
    imaging_mode = st.radio(
        "Microscopy Mode",
        options=["🔬 SEM Micrograph", "💡 Light Microscopy"],
        index=0,
        horizontal=True,
        key="imaging_modality_radio",
        help=(
            "**SEM Micrograph** — electron micrographs with an embedded databar / scale bar.  \n"
            "**Light Microscopy** — transmitted-light images (e.g. emulsion fat droplets) "
            "calibrated from a stored objective preset instead of the SEM scale bar."
        ),
    )
    is_light_mode = (imaging_mode == "💡 Light Microscopy")

    # Shared preprocessing toggles keep different physical defaults per modality
    # (CLAHE helps recover shadowed SEM craters but amplifies brightfield halos).
    # Reset them only when the modality actually changes and no pending recipe
    # from the advisor / preset loader explicitly supplied a value.
    if st.session_state.get("_last_imaging_modality") != imaging_mode:
        st.session_state["_last_imaging_modality"] = imaging_mode
        # A snapshot from the other modality cannot be applied here.
        st.session_state.pop("_previous_parameter_snapshot", None)
        if not (
            _pending_parameter_updates
            and "shared_clahe_checkbox" in _pending_parameter_updates
        ):
            st.session_state["shared_clahe_checkbox"] = not is_light_mode

    st.divider()

    st.subheader("1. Image Input")
    available_images, images_primary_dir, images_used_fallback = get_modality_images(is_light_mode)
    _folder_hint = "light micrographs" if is_light_mode else "SEM micrographs"
    st.caption(f"📁 Folder: {images_primary_dir}/ (drop {_folder_hint} here)")
    if images_used_fallback and available_images:
        st.caption(
            f"↳ `{images_primary_dir}/` is empty — listing images from `images/` (legacy folder)."
        )

    input_source = st.radio(
        "Source Method",
        ["From 'images/' folder", "Upload via Browser"],
        horizontal=True,
    )

    image_bytes: Optional[bytes] = None
    image_name = "SEM Micrograph"

    if input_source == "From 'images/' folder":
        if available_images:
            selected_path = st.selectbox(
                f"Choose File ({len(available_images)} found)",
                options=available_images,
                format_func=lambda p: p.name,
                help=f"Automatically lists all micrographs located in the '{images_primary_dir}/' directory.",
            )
            if selected_path:
                image_bytes = selected_path.read_bytes()
                image_name = selected_path.name
        else:
            st.warning(
                f"No images found in `{images_primary_dir}/` or `images/`. "
                "Drop your micrographs there to begin."
            )
            if st.button(f"Generate Demo Sample in '{images_primary_dir}/'", use_container_width=True):
                demo_bytes = create_sample_sem_image()
                Path(images_primary_dir).mkdir(parents=True, exist_ok=True)
                (Path(images_primary_dir) / "demo_sem_sample.png").write_bytes(demo_bytes)
                st.rerun()
    else:
        uploaded_file = st.file_uploader(
            "Upload SEM Image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="Select a SEM micrograph in JPG, PNG, or TIFF format.",
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image_name = uploaded_file.name
        elif st.button("Load Demo SEM Image", use_container_width=True):
            image_bytes = create_sample_sem_image()
            image_name = "Synthetic_SEM_Micrograph.png"

    # ── 🔒 Parameter isolation between micrographs ────────────────────────────
    # Section 4 & 5 controls are keyed in st.session_state, so a recipe tuned for
    # Image #1 would otherwise leak into Image #2. That leak is not cosmetic: in
    # SEM Air-Bubble / Two-Pass mode min_air_size & max_air_size drive both the
    # Cellpose target diameter (sqrt(min * max)) and the geometric fallback gates
    # (0.85 * min .. 1.15 * max) inside cached_segment_air_bubbles_auto, so a stale
    # range silently biases the very first segmentation of the newly loaded image.
    keep_params_on_switch = st.checkbox(
        "🔒 Keep current parameters when switching images",
        value=False,
        key="keep_params_on_image_switch",
        help=(
            "When unchecked (default), loading a new micrograph automatically "
            "resets Section 4 & 5 segmentation parameters to clean defaults so "
            "previous image adjustments never bias the new image. Check this box "
            "only when batch-analyzing a uniform series with identical settings."
        ),
    )

    _previous_image_for_params = st.session_state.get("_last_loaded_image_for_params")
    if _previous_image_for_params != image_name:
        st.session_state["_last_loaded_image_for_params"] = image_name
        if _previous_image_for_params is not None:
            # Per-image analysis artefacts must never survive an image switch:
            # a proposal / undo snapshot / manual correction measured on Image #1
            # is meaningless for Image #2 and would be silently applied to it.
            st.session_state.pop("autotune_proposal", None)
            st.session_state.pop("_previous_parameter_snapshot", None)
            st.session_state["manual_phase_overrides"] = {}
            st.session_state["interactive_excluded_ids"] = set()
            for _stale_editor_key in [
                _k
                for _k in list(st.session_state.keys())
                if str(_k).startswith("particle_data_editor")
            ]:
                del st.session_state[_stale_editor_key]

            if not keep_params_on_switch:
                # Written BEFORE any Section 4 / 5 widget is instantiated in this
                # run, so the controls below render with the clean defaults on this
                # very run - an image switch therefore behaves exactly like opening
                # that micrograph as the first image of a fresh session.
                _active_light_profile = st.session_state.get(
                    "light_emulsion_profile_radio"
                )
                for _default_key, _default_value in build_default_parameter_state(
                    is_light_mode, _active_light_profile
                ).items():
                    st.session_state[_default_key] = _default_value
                if is_light_mode:
                    # Keep the profile tracker aligned with the freshly rendered
                    # profile radio so the polarity default is not re-applied twice.
                    st.session_state["light_profile_tracker"] = st.session_state.get(
                        "light_emulsion_profile_radio", LIGHT_PROFILE_CRYSTALLIZED
                    )
                st.session_state["force_class_refresh"] = True
                st.toast(
                    "♻️ New micrograph loaded — parameters reset to clean defaults.",
                    icon="♻️",
                )

    st.divider()

    st.subheader("2. Footer Panel Exclusion")
    if is_light_mode:
        # Light micrographs have no SEM databar — keep the panel collapsed and
        # default the crop to 0 px so the full frame is analysed.
        with st.expander("Optional Banner Crop (Light Mode)", expanded=False):
            footer_height = int(
                st.number_input(
                    "Footer Height (pixels)",
                    min_value=0,
                    value=0,
                    step=5,
                    help=(
                        "Optional: crop a bottom banner from the light micrograph. "
                        "Leave at 0 for standard microscope camera images."
                    ),
                    key="light_footer_height_input",
                )
            )
    else:
        footer_height = int(
            st.number_input(
                "Footer Height (pixels)",
                min_value=0,
                value=70,
                step=5,
                help=(
                    "Height in pixels of the bottom SEM information panel. "
                    "Also used to locate the scale bar in the upper half of the banner."
                ),
                key="footer_height_input",
            )
        )

    st.divider()

    st.subheader("3. Spatial Calibration")

    # Objective preset chosen in Light Microscopy mode (None while in SEM mode)
    selected_objective: Optional[str] = None

    if is_light_mode:
        # ── Light Microscopy: objective-preset calibration ───────────────────
        # Light micrographs carry no SEM databar/scale-bar line, therefore
        # detect_scale_bar() is never invoked in this mode.
        detected_bar_pixels = 0
        scale_preview = None
        use_auto_scale = False

        light_presets = load_light_presets()
        preset_names = list(light_presets.keys())
        _default_objective = "40x Objective"
        _default_objective_idx = (
            preset_names.index(_default_objective) if _default_objective in preset_names else 0
        )

        selected_objective = st.selectbox(
            "Microscope Objective Preset",
            options=preset_names,
            index=_default_objective_idx,
            key="light_objective_preset",
            help=(
                "Laboratory-validated objective calibration (µm per pixel). "
                "Pick 'Custom Calibration' for an arbitrary coefficient."
            ),
        )

        _active_preset = light_presets.get(
            selected_objective, DEFAULT_LIGHT_OBJECTIVES["Custom Calibration"]
        )
        _preset_um_per_px = float(_active_preset.get("um_per_px", 0.2))
        _preset_bar_um = float(_active_preset.get("scale_bar_um", 10.0))
        _preset_bar_px = float(_active_preset.get("scale_bar_px", 50.0))

        # Widget state for keys that disappear from the layout (e.g. after
        # switching to SEM mode and back) is discarded by Streamlit, therefore
        # both the objective change AND missing keys are handled here.
        calib_input_style = st.radio(
            "Calibration Input Method",
            ["Direct Ratio (µm/px)", "Known Scale (µm / pixels)"],
            horizontal=True,
            key="light_calib_input_style",
        )

        _objective_changed = st.session_state.get("light_preset_tracker") != selected_objective
        if _objective_changed:
            st.session_state["light_preset_tracker"] = selected_objective
            st.session_state["light_um_per_px_input"] = _preset_um_per_px
            st.session_state["light_scale_bar_um_input"] = _preset_bar_um
            st.session_state["light_scale_bar_px_input"] = _preset_bar_px

        if calib_input_style == "Direct Ratio (µm/px)":
            if "light_um_per_px_input" not in st.session_state:
                st.session_state["light_um_per_px_input"] = _preset_um_per_px
            pixel_to_um_ratio = float(
                st.number_input(
                    "Calibration Factor (µm / pixel)",
                    min_value=0.001,
                    max_value=100.0,
                    step=0.001,
                    format="%.5f",
                    help="Physical size represented by a single pixel at the selected objective.",
                    key="light_um_per_px_input",
                )
            )
            pixels_per_um = 1.0 / pixel_to_um_ratio if pixel_to_um_ratio > 0 else 5.0
            # Manifest compatibility: emulate a nominal 10 µm reference scale bar
            scale_bar_um = 10.0
            scale_bar_length_px = (
                scale_bar_um / pixel_to_um_ratio if pixel_to_um_ratio > 0 else 50.0
            )
        else:
            if "light_scale_bar_um_input" not in st.session_state:
                st.session_state["light_scale_bar_um_input"] = _preset_bar_um
            if "light_scale_bar_px_input" not in st.session_state:
                st.session_state["light_scale_bar_px_input"] = _preset_bar_px
            scale_bar_um = float(
                st.number_input(
                    "Real Stage Micrometer / Scale Size (µm)",
                    min_value=0.1,
                    max_value=5000.0,
                    step=0.5,
                    help="Known physical length of the stage micrometer division or embedded scale bar.",
                    key="light_scale_bar_um_input",
                )
            )
            scale_bar_length_px = float(
                st.number_input(
                    "Measured Length (pixels)",
                    min_value=1.0,
                    max_value=20000.0,
                    step=1.0,
                    help="Length of that same reference distance measured in pixels.",
                    key="light_scale_bar_px_input",
                )
            )
            if scale_bar_length_px > 0 and scale_bar_um > 0:
                pixel_to_um_ratio = float(scale_bar_um) / float(scale_bar_length_px)
                pixels_per_um = float(scale_bar_length_px) / float(scale_bar_um)
            else:
                pixel_to_um_ratio = 1.0 / 5.0
                pixels_per_um = 5.0
                st.warning("Enter valid reference size (µm) and measured length (pixels).")

        st.info(
            f"Active Scale: **{pixels_per_um:.3f} px/µm** ({pixel_to_um_ratio:.4f} µm/px)"
        )

        if st.button(
            "💾 Save Current Value to Objective Preset",
            use_container_width=True,
            key="light_save_preset_btn",
        ):
            light_presets[selected_objective] = {
                "um_per_px": float(pixel_to_um_ratio),
                "scale_bar_um": float(scale_bar_um),
                "scale_bar_px": float(scale_bar_length_px),
            }
            try:
                save_light_presets(light_presets)
                st.success(
                    f"Saved **{selected_objective}** → "
                    f"{float(pixel_to_um_ratio):.5f} µm/px "
                    f"({pixels_per_um:.3f} px/µm → `{LIGHT_PRESETS_FILE}`)",
                    icon="💾",
                )
                st.toast(f"Preset saved: {selected_objective}", icon="💾")
            except OSError as exc:
                st.error(f"Could not write `{LIGHT_PRESETS_FILE}`: {exc}")

    else:
        # ── SEM Mode: unchanged automatic scale-bar detection ────────────────
        # Detect scale bar in UPPER-RIGHT banner only (avoids table separator line)
        detected_bar_pixels = 0
        scale_preview = None
        if image_bytes is not None:
            try:
                # Cached decode + morphological probe: the sidebar reruns on every
                # widget interaction but the scale bar never changes.
                detected_bar_pixels, scale_preview, _ = cached_detect_scale_bar(
                    image_bytes, max(10, int(footer_height) or 70)
                )
            except Exception:
                detected_bar_pixels = 0
                scale_preview = None

        # Auto-fill Real Scale Bar Size (µm) from filename patterns like _5um_ / _200um_
        filename_scale_um = extract_scale_um_from_filename(image_name)
        default_scale_um = float(filename_scale_um) if filename_scale_um > 0 else 200.0

        # Reset widget defaults when the active image changes
        if st.session_state.get("calib_image_name") != image_name:
            st.session_state["calib_image_name"] = image_name
            st.session_state["scale_bar_um_input"] = default_scale_um
            st.session_state["scale_bar_length_px_input"] = (
                float(detected_bar_pixels) if detected_bar_pixels > 0 else 200.0
            )

        if "scale_bar_um_input" not in st.session_state:
            st.session_state["scale_bar_um_input"] = default_scale_um
        if "scale_bar_length_px_input" not in st.session_state:
            st.session_state["scale_bar_length_px_input"] = (
                float(detected_bar_pixels) if detected_bar_pixels > 0 else 200.0
            )

        if detected_bar_pixels > 0:
            st.success(f"Detected scale bar: **{detected_bar_pixels} pixels**", icon="📏")
        elif image_bytes is not None:
            st.caption("No prominent scale bar line detected in bottom-right region.")

        if filename_scale_um > 0:
            st.caption(f"Filename scale hint: **{filename_scale_um:g} µm** parsed from `{image_name}`")

        if scale_preview is not None and detected_bar_pixels > 0:
            st.image(
                scale_preview,
                caption=f"Detected scale-bar line preview ({detected_bar_pixels} px)",
                **_FULL_WIDTH_IMAGE_KWARGS,
            )

        scale_bar_um = st.number_input(
            "Real Scale Bar Size (µm)",
            min_value=0.1,
            max_value=5000.0,
            step=0.5,
            help="Physical length corresponding to the scale bar marked on the SEM micrograph. Auto-filled from filename when possible (e.g. _5um_).",
            key="scale_bar_um_input",
        )

        scale_bar_length_px = st.number_input(
            "Scale Bar Length (pixels)",
            min_value=1.0,
            max_value=20000.0,
            step=1.0,
            help="Detected (or manually corrected) length of the scale bar line in pixels.",
            key="scale_bar_length_px_input",
        )

        # Calibration: pixel_to_um_ratio = scale_bar_um / scale_bar_length_px
        if float(scale_bar_length_px) > 0 and float(scale_bar_um) > 0:
            pixel_to_um_ratio = float(scale_bar_um) / float(scale_bar_length_px)
            pixels_per_um = float(scale_bar_length_px) / float(scale_bar_um)
            use_auto_scale = detected_bar_pixels > 0 and abs(
                float(scale_bar_length_px) - float(detected_bar_pixels)
            ) < 0.5
            st.info(
                f"Active Scale: **{pixels_per_um:.3f} px/µm** ({pixel_to_um_ratio:.4f} µm/px)  \n"
                f"`pixel_to_um_ratio = {float(scale_bar_um):g} / {float(scale_bar_length_px):g}`"
            )
        else:
            use_auto_scale = False
            pixels_per_um = 5.0
            pixel_to_um_ratio = 1.0 / pixels_per_um
            st.warning("Enter valid Real Scale Bar Size (µm) and Scale Bar Length (pixels).")

    st.divider()

    st.subheader("4. Phase Classification")

    if is_light_mode:
        # Phase classification is locked in light mode; the radio is hidden.
        sample_type = "Emulsion"
    else:
        sample_type = st.radio(
            "Sample Type",
            options=["Emulsion", "Whipped Emulsion", "Foam"],
            index=1,
            help="Select 'Emulsion' for single-phase fat droplets, 'Whipped Emulsion' for multi-phase air and fat sorting, or 'Foam' for single-phase air bubbles.",
            key="sem_sample_type_radio",
        )

    if is_light_mode:
        # ── Light Microscopy: phase locked to single-phase Emulsion ──────────
        detection_focus = None
        st.info(
            "🟠 **Light Microscopy Mode: Emulsion (Fat Droplets Analysis)** — "
            "phase classification is locked to fat droplet sizing.",
            icon="🔒",
        )
        st.markdown("**Fat Particle Size Range (µm):**")
        col_fat1, col_fat2 = st.columns(2)
        with col_fat1:
            min_fat_size = float(st.number_input(
                "Min Fat Particle Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=0.50,
                step=0.05,
                help="Minimum equivalent diameter in micrometers for fat droplets.",
                key="light_min_fat_size_input",
            ))
        with col_fat2:
            max_fat_size = float(st.number_input(
                "Max Fat Particle Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=50.0,
                step=0.5,
                help="Maximum equivalent diameter in micrometers for fat droplets.",
                key="light_max_fat_size_input",
            ))
        # Internal defaults for hidden air bubble settings
        min_air_size = 10.0
        max_air_size = 100.0
        overlap_priority = "Air Bubbles"
        out_of_range_handling = "Unclassified"
        st.caption(f"🟧 Fat: [{min_fat_size:.2f}, {max_fat_size:.2f}] µm")

    elif sample_type == "Emulsion":
        detection_focus = None  # Not applicable for single-phase Emulsion
        st.info(
            "🟠 **Sample Type: Emulsion** — Objects within Min/Max Fat size become Fat Particles; "
            "out-of-range objects become Unclassified (Air phase search disabled)."
        )
        st.markdown("**Fat Particle Size Range (µm):**")
        col_fat1, col_fat2 = st.columns(2)
        with col_fat1:
            min_fat_size = float(st.number_input(
                "Min Fat Particle Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=0.20,
                step=0.1,
                help="Minimum equivalent diameter in micrometers for fat particles.",
                key="sem_min_fat_um",
            ))
        with col_fat2:
            max_fat_size = float(st.number_input(
                "Max Fat Particle Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=5.0,
                step=0.5,
                help="Maximum equivalent diameter in micrometers for fat particles.",
                key="sem_max_fat_um",
            ))
        # Internal defaults for hidden air bubble settings
        min_air_size = 10.0
        max_air_size = 100.0
        overlap_priority = "Air Bubbles"
        out_of_range_handling = "Unclassified"
        st.caption(f"🟧 Fat: [{min_fat_size:.1f}, {max_fat_size:.1f}] µm")

    elif sample_type == "Foam":
        detection_focus = "🫧 Air Bubbles Only (Fast & Precise)"  # Foam always uses air-bubble detector
        st.info(
            "🫧 **Sample Type: Foam** — Objects within Min/Max Air size become Air Bubbles; "
            "out-of-range objects become Unclassified (Fat phase search disabled)."
        )
        st.markdown("**Air Bubble Size Range (µm):**")
        col_air1, col_air2 = st.columns(2)
        with col_air1:
            min_air_size = float(st.number_input(
                "Min Air Bubble Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=10.0,
                step=1.0,
                help="Minimum equivalent diameter in micrometers for air bubbles.",
                key="sem_min_air_um",
            ))
        with col_air2:
            max_air_size = float(st.number_input(
                "Max Air Bubble Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=100.0,
                step=5.0,
                help="Maximum equivalent diameter in micrometers for air bubbles.",
                key="sem_max_air_um",
            ))
        # Internal defaults for hidden fat particle settings
        min_fat_size = 0.20
        max_fat_size = 5.0
        overlap_priority = "Air Bubbles"
        out_of_range_handling = "Unclassified"
        st.caption(f"🟦 Air: [{min_air_size:.1f}, {max_air_size:.1f}] µm")

    else:
        detection_focus = st.radio(
            "AI Detection Target",
            options=[
                "🫧 Air Bubbles Only (Fast & Precise)",
                "🫧+🟠 Both Air Bubbles & Fat Particles (Two-Pass)",
            ],
            index=0,
            help=(
                "**Air Bubbles Only** — focuses AI strictly on your Min/Max Air Bubble "
                "size range using the specialized dark-crater detector (fastest, most accurate "
                "for large pores).  "
                "**Two-Pass** — dedicated pass for large air bubbles + second pass for fine "
                "fat globules; both layers shown simultaneously on the overlay."
            ),
            key="sem_detection_focus_radio",
        )

        st.markdown("**Air Bubble Size Range (µm):**")
        col_air1, col_air2 = st.columns(2)
        with col_air1:
            min_air_size = float(st.number_input(
                "Min Air Bubble Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=10.0,
                step=1.0,
                help="Minimum equivalent diameter in micrometers for air bubbles.",
                key="sem_min_air_um",
            ))
        with col_air2:
            max_air_size = float(st.number_input(
                "Max Air Bubble Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=100.0,
                step=5.0,
                help="Maximum equivalent diameter in micrometers for air bubbles.",
                key="sem_max_air_um",
            ))

        st.markdown("**Fat Particle Size Range (µm):**")
        col_fat1, col_fat2 = st.columns(2)
        with col_fat1:
            min_fat_size = float(st.number_input(
                "Min Fat Particle Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=0.20,
                step=0.1,
                help="Minimum equivalent diameter in micrometers for fat particles.",
                key="sem_min_fat_um",
            ))
        with col_fat2:
            max_fat_size = float(st.number_input(
                "Max Fat Particle Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=5.0,
                step=0.5,
                help="Maximum equivalent diameter in micrometers for fat particles.",
                key="sem_max_fat_um",
            ))

        overlap_priority = "Air Bubbles"
        out_of_range_handling = "Unclassified"

        st.caption(
            f"🟦 Air: [{min_air_size:.1f}, {max_air_size:.1f}] µm | "
            f"🟧 Fat: [{min_fat_size:.1f}, {max_fat_size:.1f}] µm"
        )


    if st.button("🔄 Apply Size Ranges & Classification", use_container_width=True):
        # Force clear sticky table/editor state so classification refreshes immediately
        editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
        for k in editor_keys:
            del st.session_state[k]
        st.session_state["manual_phase_overrides"] = {}
        st.session_state["force_class_refresh"] = True
        st.rerun()

    st.divider()

    st.subheader("5. Cellpose AI Segmentation")

    # Emulsion optical profile (Light Microscopy only)
    emulsion_profile = "N/A (SEM Mode)"
    is_crystallized_mode = False
    if is_light_mode:
        emulsion_profile = st.radio(
            "Emulsion Droplet Profile",
            options=[
                "💧 Liquid Fat (Smooth Globules)",
                "❄️ Crystallized Fat (Textured / Internal Crystals)",
            ],
            index=1,
            help=(
                "**Liquid Fat:** Standard inference for smooth, homogeneous droplets.\n"
                "**Crystallized Fat:** Applies edge-preserving smoothing to erase internal crystal lines "
                "so Cellpose does not split globules into fragments, and enforces stricter convexity (Solidity)."
            ),
            key="light_emulsion_profile_radio",
        )
        is_crystallized_mode = emulsion_profile.startswith("❄️")

    model_type = st.selectbox(
        "Pretrained Model",
        options=["cyto3", "cyto2", "nuclei", "Custom Model File..."],
        index=1,
        key="shared_model_type_select",
        help=(
            "**cyto3:** Super-generalist model trained on a massive multi-modality "
            "dataset (best for complex/textured micrographs).  \n"
            "**cyto2:** Classic high-speed baseline for standard SEM & emulsions.  \n"
            "**nuclei:** Optimized for round, high-contrast spherical droplets."
        ),
    )

    custom_model_path: Optional[str] = None
    if model_type == "Custom Model File...":
        uploaded_custom_file = st.file_uploader(
            "Upload Custom .pth File",
            type=None,
            help="Select a custom Cellpose weights file (.pth or checkpoint).",
        )
        if uploaded_custom_file is not None:
            models_cache_dir = Path("models_cache")
            models_cache_dir.mkdir(parents=True, exist_ok=True)
            saved_custom_path = models_cache_dir / uploaded_custom_file.name
            saved_custom_path.write_bytes(uploaded_custom_file.getvalue())
            custom_model_path = str(saved_custom_path.resolve())
            st.success(f"Custom weights loaded: `{uploaded_custom_file.name}`", icon="📦")
        else:
            st.info("Upload a .pth or custom weights file above to run inference.")

    # Smart automatic calculation of Cellpose effective diameter from physical size boundaries
    # For Air Bubbles Only / Foam: diameter computed inside segment_air_bubbles_auto
    # For Emulsion / Two-Pass fat pass: fat-optimised small diameter
    _whipped_air_only = (
        sample_type == "Whipped Emulsion"
        and detection_focus == "\U0001fae7 Air Bubbles Only (Fast & Precise)"
    )
    _whipped_two_pass = (
        sample_type == "Whipped Emulsion"
        and detection_focus == "\U0001fae7+\U0001f7e0 Both Air Bubbles & Fat Particles (Two-Pass)"
    )

    expected_droplet_um: Optional[float] = None

    if sample_type == "Emulsion":
        if is_light_mode:
            # Physical (µm) target scale — automatically converted to pixels using the
            # active objective calibration, and no longer capped at 45 px so large
            # crystallized globules are captured whole instead of split into shards.
            default_cp_um = 9.0 if is_crystallized_mode else 5.0
            expected_droplet_um = float(
                st.slider(
                    "🎯 Typical Droplet Diameter for AI (µm)",
                    min_value=1.0,
                    max_value=40.0,
                    value=default_cp_um,
                    step=0.5,
                    help=(
                        "Physical diameter (µm) of typical droplets you want AI to focus on. "
                        "Automatically converted to pixels using your active objective calibration. "
                        "Set higher (e.g., 8–14 µm) for large crystallized globules so AI captures them whole instead of splitting internal crystal lines."
                    ),
                    key="light_expected_droplet_um",
                )
            )
            eff_diameter = float(np.clip(expected_droplet_um / max(pixel_to_um_ratio, 1e-6), 12.0, 220.0))
            st.caption(
                f"🎯 AI Target Scale: **{expected_droplet_um:.1f} µm** → **{eff_diameter:.0f} px** (at {pixels_per_um:.2f} px/µm)"
            )
        else:
            ref_um = float(min_fat_size) * 1.25 if float(min_fat_size) > 0 else 1.0
            calc_px = ref_um / pixel_to_um_ratio if pixel_to_um_ratio > 0 else 30.0
            eff_diameter = float(np.clip(calc_px, 18.0, 45.0))
            st.caption(f"\U0001f3af Auto Cellpose diameter: **{eff_diameter:.1f} px** (fat-optimised)")
    elif sample_type == "Foam" or _whipped_air_only:
        _air_target = float(np.clip(
            math.sqrt(max(float(min_air_size), 0.5) * max(float(max_air_size), 1.0))
            / (pixel_to_um_ratio if pixel_to_um_ratio > 0 else 0.2),
            25.0, 220.0,
        ))
        eff_diameter = _air_target
        st.caption(
            f"\U0001fae7 Air bubble target: **~{_air_target:.1f} px** "
            f"(geometric mean [{min_air_size:.1f}\u2013{max_air_size:.1f} \u00b5m])"
        )
    elif _whipped_two_pass:
        _air_target = float(np.clip(
            math.sqrt(max(float(min_air_size), 0.5) * max(float(max_air_size), 1.0))
            / (pixel_to_um_ratio if pixel_to_um_ratio > 0 else 0.2),
            25.0, 220.0,
        ))
        fat_ref_um = float(min_fat_size) * 1.25 if float(min_fat_size) > 0 else 1.0
        fat_calc_px = float(np.clip(
            fat_ref_um / pixel_to_um_ratio if pixel_to_um_ratio > 0 else 25.0,
            15.0, 45.0,
        ))
        eff_diameter = fat_calc_px
        st.caption(
            f"\U0001fae7 Air pass: **~{_air_target:.1f} px** | \U0001f7e0 Fat pass: **{fat_calc_px:.1f} px**"
        )
    else:
        ref_um = float(min_fat_size) * 1.25 if float(min_fat_size) > 0 else 1.0
        calc_px = ref_um / pixel_to_um_ratio if pixel_to_um_ratio > 0 else 30.0
        eff_diameter = float(np.clip(calc_px, 18.0, 45.0))
        st.caption(f"\U0001f3af Auto Cellpose diameter: **{eff_diameter:.1f} px** (derived from size boundaries)")

    flow_threshold = st.slider(
        "flow_threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Maximum allowed flow error. Higher values detect more irregular or faint boundaries (default: 0.7).",
        key="shared_flow_slider",
    )

    cellprob_threshold = st.slider(
        "cellprob_threshold",
        min_value=-6.0,
        max_value=6.0,
        value=-1.0,
        step=0.5,
        help="Cell probability detection threshold. Negative values (e.g. -1.0 to -2.0) drastically increase detection of low-contrast pores and bubbles (default: -1.0).",
        key="shared_cellprob_slider",
    )

    fat_min_solidity = 0.0
    crystal_smooth_strength = 0
    enable_multiscale_crystal = False

    if is_light_mode:
        # Light microscopy (single-phase emulsion): air-bubble / adsorbed-fat
        # controls are irrelevant, so they stay hidden with safe internal values.
        air_min_circularity = 0.50
        ads_min_circularity = 0.30
        enable_deepscan = False
        fat_min_circularity = st.slider(
            "🟧 Matrix Fat Min Circularity", 0.0, 1.0, 0.60, 0.01,
            key="shared_fat_circ_slider",
        )
        default_solidity = 0.84 if is_crystallized_mode else 0.72
        fat_min_solidity = float(
            st.slider(
                "🛡️ Matrix Fat Min Solidity (Convexity — rejects shards)",
                min_value=0.50,
                max_value=0.98,
                value=default_solidity,
                step=0.01,
                help="Ratio of droplet area to its convex hull area. True spherical/oval droplets have Solidity > 0.84; crescent shards and inter-droplet background fragments have low solidity and are rejected.",
                key="light_solidity_slider",
            )
        )
        if is_crystallized_mode:
            crystal_smooth_strength = int(
                st.slider(
                    "❄️ Internal Crystal Smoothing Strength",
                    min_value=0,
                    max_value=4,
                    value=3,
                    help="0 = off. 2–3 smooths out internal crystal needles before AI segmentation so whole globules are detected intact.",
                    key="light_crystal_smooth_slider",
                )
            )
            enable_multiscale_crystal = st.checkbox(
                "🔍 Multi-Scale Globule Detection (Large + Small Globules)",
                value=True,
                help="Runs a coarse AI pass for large crystallized globules (preventing internal splitting) and a fine pass for small globules, merging intact droplets.",
                key="light_multiscale_checkbox",
            )
        st.caption("ℹ️ Light Microscopy: only fat-droplet circularity/convexity filtering is applied.")
    else:
        air_min_circularity = st.slider(
            "🫧 Air Bubbles Min Circularity", 0.0, 1.0, 0.50, 0.05,
            key="sem_air_circ_slider",
        )
        fat_min_circularity = st.slider(
            "🟧 Matrix Fat Min Circularity", 0.0, 1.0, 0.60, 0.01,
            key="shared_fat_circ_slider",
        )
        ads_min_circularity = st.slider(
            "🟨 Adsorbed Fat Min Circularity (in Bubbles)", 0.0, 1.0, 0.30, 0.05,
            key="sem_ads_circ_slider",
        )
        enable_deepscan = st.checkbox(
            "🔬 Deep-Scan Adsorbed Fat inside Bubbles",
            value=False,
            help="Runs localized shadow-recovery Cellpose inside confirmed air bubbles. Keep unchecked for fast preview.",
            key="sem_deepscan_checkbox",
        )
        st.caption("ℹ️ Colloid Chemistry: Fat within bubble regions is classified as adsorbed_fat.")

    min_circularity = fat_min_circularity  # Fallback alias for backward compatibility

    # Optical preprocessing for transmitted-light (brightfield) micrographs
    invert_brightfield = False
    if is_light_mode:
        # Keep the widget's polarity default in sync with the selected optical
        # profile, unless the user (or the AI advisor) explicitly overrode it.
        if st.session_state.get("light_profile_tracker") != emulsion_profile:
            st.session_state["light_profile_tracker"] = emulsion_profile
            st.session_state["light_invert_bf_checkbox"] = not is_crystallized_mode
        invert_brightfield = st.checkbox(
            "💡 Invert Brightfield (Dark-Rim Droplets)",
            value=not is_crystallized_mode,
            help=(
                "Inverts the image before segmentation. Essential for brightfield light "
                "microscopy where fat droplets have a dark refraction halo against a bright "
                "background. This helps Cellpose capture the true outer boundary. "
                "Crystallized profiles default to OFF because packed globules have bright "
                "bodies separated by dark borders."
            ),
            key="light_invert_bf_checkbox",
        )

    enable_clahe = st.checkbox(
        "Enable CLAHE Preprocessing",
        value=not is_light_mode,
        help="Applies Contrast Limited Adaptive Histogram Equalization with OpenCV (clipLimit=2.0, tileGridSize=(8,8)) to boost contrast for faint pores before segmentation.",
        key="shared_clahe_checkbox",
    )

    # ── 🪄 AI Parameter Advisor (controlled apply) ────────────────────────────
    st.divider()
    st.markdown("**🪄 AI Parameter Advisor**")
    st.caption(
        "Measures the current micrograph in physical µm and *proposes* Section 4/5 "
        "settings. Nothing changes until you review the diagnostics and apply."
    )

    # Single source of truth for "current" values (also used by preset saving).
    _live_parameter_values = {
        "sem_sample_type_radio": sample_type,
        "sem_detection_focus_radio": detection_focus,
        "sem_min_air_um": float(min_air_size),
        "sem_max_air_um": float(max_air_size),
        "sem_min_fat_um": float(min_fat_size),
        "sem_max_fat_um": float(max_fat_size),
        "light_min_fat_size_input": float(min_fat_size),
        "light_max_fat_size_input": float(max_fat_size),
        "light_emulsion_profile_radio": emulsion_profile if is_light_mode else None,
        "light_expected_droplet_um": expected_droplet_um,
        "light_crystal_smooth_slider": int(crystal_smooth_strength),
        "light_multiscale_checkbox": bool(enable_multiscale_crystal),
        "light_invert_bf_checkbox": bool(invert_brightfield) if is_light_mode else None,
        "light_solidity_slider": float(fat_min_solidity) if is_light_mode else None,
        "shared_fat_circ_slider": float(fat_min_circularity),
        "shared_flow_slider": float(flow_threshold),
        "shared_cellprob_slider": float(cellprob_threshold),
        "shared_clahe_checkbox": bool(enable_clahe),
        "shared_model_type_select": model_type,
        "sem_air_circ_slider": float(air_min_circularity) if not is_light_mode else None,
        "sem_ads_circ_slider": float(ads_min_circularity) if not is_light_mode else None,
        "sem_deepscan_checkbox": bool(enable_deepscan) if not is_light_mode else None,
    }

    if st.button(
        "🪄 Analyze Image & Recommend Parameters",
        type="primary",
        use_container_width=True,
        key="autotune_analyze_btn",
        help=(
            "Runs optical and morphological diagnostics on the current image and opens a "
            "review card with measured µm ranges, a parameter comparison table and the "
            "scientific rationale for every suggestion."
        ),
    ):
        if image_bytes is None:
            st.warning("Load a micrograph before running the AI Parameter Advisor.")
        else:
            try:
                _advisor_raw = decode_image_bytes(image_bytes)
                _advisor_footer = max(0, min(int(footer_height), _advisor_raw.shape[0] - 1))
                _advisor_img = (
                    _advisor_raw[:-_advisor_footer, :]
                    if _advisor_footer > 0
                    else _advisor_raw
                )
                _advisor_proposal = diagnose_and_recommend_parameters(
                    _advisor_img,
                    float(pixel_to_um_ratio),
                    bool(is_light_mode),
                    dict(_live_parameter_values),
                )
                _advisor_proposal["image_name"] = image_name
                _advisor_proposal["is_light_mode"] = bool(is_light_mode)
                _advisor_proposal["calibration_um_per_px"] = float(pixel_to_um_ratio)
                st.session_state["autotune_proposal"] = _advisor_proposal
            except Exception as exc:
                st.error(f"AI Parameter Advisor could not analyse this image: {exc}")

    # ── ↩️ Undo Last Apply / 🔄 Reset to Defaults (one-click recovery) ─────────
    _undo_col, _reset_col = st.columns(2)
    with _undo_col:
        if st.button(
            "↩️ Undo Last Apply",
            use_container_width=True,
            key="autotune_undo_btn",
            disabled=not bool(st.session_state.get("_previous_parameter_snapshot")),
            help=(
                "Restore the parameter values that were active before the last "
                "Apply, preset load, or reset."
            ),
        ):
            _undo_snapshot = st.session_state.pop("_previous_parameter_snapshot", None)
            if _undo_snapshot:
                st.session_state["_pending_parameter_updates"] = dict(_undo_snapshot)
                st.session_state.pop("autotune_proposal", None)
                st.toast("↩️ Reverted to previous parameters!")
                st.rerun()

    with _reset_col:
        if st.button(
            "🔄 Reset to Defaults",
            use_container_width=True,
            key="autotune_reset_btn",
            help=(
                "Restore the clean defaults for the active modality and optical profile "
                "(9.0 µm target, cyto2, 0.50-50.0 µm, smoothing 3, solidity 0.84 for "
                "crystallized fat)."
            ),
        ):
            st.session_state["_previous_parameter_snapshot"] = collect_parameter_snapshot(
                "Light" if is_light_mode else "SEM", _live_parameter_values
            )
            st.session_state["_pending_parameter_updates"] = build_default_parameter_state(
                is_light_mode,
                emulsion_profile if is_light_mode else None,
            )
            # Clear manual classification/exclusion state so the reset recipe is
            # evaluated against a clean object set.
            st.session_state["manual_phase_overrides"] = {}
            st.session_state["interactive_excluded_ids"] = set()
            st.session_state.pop("autotune_proposal", None)
            st.toast("🔄 Parameters reset to defaults!")
            st.rerun()

    # ── 📂 Custom Parameter Preset Manager ────────────────────────────────────
    with st.expander("📂 Parameter Presets (Save / Load Recipes)", expanded=False):
        _preset_modality = "Light" if is_light_mode else "SEM"
        _parameter_presets = load_parameter_presets()
        _modality_presets = _parameter_presets.get(_preset_modality, {})
        st.caption(
            f"Recipes for **{_preset_modality}** mode are stored in "
            f"`{PARAMETER_PRESETS_FILE}` in the project root."
        )

        if _modality_presets:
            _preset_names = list(_modality_presets.keys())
            _selected_preset_name = st.selectbox(
                "Saved Recipe",
                options=_preset_names,
                key=f"parameter_preset_select_{_preset_modality.lower()}",
                help="Built-in laboratory recipes plus every preset you saved yourself.",
            )
            _preset_select_key = f"parameter_preset_select_{_preset_modality.lower()}"
            # Load / delete side-by-side so stale recipes can be pruned from the UI
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                _load_preset_clicked = st.button(
                    "📂 Load Preset",
                    use_container_width=True,
                    key=f"load_parameter_preset_{_preset_modality.lower()}",
                )
            with p_col2:
                _delete_preset_clicked = st.button(
                    "🗑️ Delete Preset",
                    use_container_width=True,
                    key=f"delete_parameter_preset_{_preset_modality.lower()}",
                    help=(
                        f"Permanently removes '{_selected_preset_name}' from "
                        f"`{PARAMETER_PRESETS_FILE}`."
                    ),
                )

            if _load_preset_clicked:
                st.session_state["_previous_parameter_snapshot"] = collect_parameter_snapshot(
                    _preset_modality, _live_parameter_values
                )
                st.session_state["_pending_parameter_updates"] = dict(
                    _modality_presets[_selected_preset_name]
                )
                st.session_state.pop("autotune_proposal", None)
                st.session_state["_preset_loaded_flash"] = _selected_preset_name
                st.rerun()
            elif _delete_preset_clicked:
                _parameter_presets[_preset_modality] = {
                    _name: _values
                    for _name, _values in _modality_presets.items()
                    if _name != _selected_preset_name
                }
                try:
                    save_parameter_presets(_parameter_presets)
                    # Discard the stale widget value so the selectbox re-seeds from
                    # the remaining recipes instead of a deleted entry.
                    if st.session_state.get(_preset_select_key) == _selected_preset_name:
                        del st.session_state[_preset_select_key]
                    st.session_state["_preset_deleted_flash"] = _selected_preset_name
                    st.rerun()
                except OSError as exc:
                    st.error(f"Could not write `{PARAMETER_PRESETS_FILE}`: {exc}")
        else:
            st.info(
                "No presets stored for this modality yet — the file now holds the exact "
                "recipe list, so deleted presets stay deleted. Use "
                "**🔄 Restore Built-in Presets** below to bring the built-ins back."
            )

        _new_preset_name = st.text_input(
            "New preset name",
            placeholder="Student Lab - Coconut Bar Emulsion 40x",
            key=f"parameter_preset_name_{_preset_modality.lower()}",
            help="Saves every Section 4 & Section 5 control for the active modality.",
        )
        if st.button(
            "💾 Save Current Settings as Preset",
            use_container_width=True,
            key=f"save_parameter_preset_{_preset_modality.lower()}",
        ):
            _clean_preset_name = str(_new_preset_name or "").strip()
            if not _clean_preset_name:
                st.warning("Type a preset name before saving.")
            else:
                _preset_snapshot = collect_parameter_snapshot(
                    _preset_modality, _live_parameter_values
                )
                _parameter_presets.setdefault(_preset_modality, {})[
                    _clean_preset_name
                ] = _preset_snapshot
                try:
                    save_parameter_presets(_parameter_presets)
                    st.session_state["_preset_saved_flash"] = _clean_preset_name
                    st.rerun()
                except OSError as exc:
                    st.error(f"Could not write `{PARAMETER_PRESETS_FILE}`: {exc}")

        if st.button(
            "🔄 Restore Built-in Presets",
            use_container_width=True,
            key=f"restore_parameter_presets_{_preset_modality.lower()}",
            help=(
                "Re-adds the built-in laboratory recipes for this modality. Custom "
                "presets you saved are kept."
            ),
        ):
            try:
                save_parameter_presets(restore_builtin_parameter_presets(_preset_modality))
                st.session_state["_preset_restored_flash"] = _preset_modality
                st.rerun()
            except OSError as exc:
                st.error(f"Could not write `{PARAMETER_PRESETS_FILE}`: {exc}")

        st.caption(
            "ℹ️ Presets capture the full Section 4 & 5 recipe so a lab protocol can be "
            "reproduced exactly on another image or microscope."
        )

    st.divider()

    st.subheader("6. Visualization Settings")

    if sample_type == "Foam":
        show_air = st.checkbox("Show Air Bubbles", value=True)
        show_fat = False
        show_adsorbed = False
        adsorbed_opacity = 0.0
        air_opacity = st.slider(
            "Air Bubbles Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Transparency level for air bubbles overlay layer (0.0=fully transparent, 1.0=fully opaque).",
        )
        fat_opacity = 0.0
        z_order = "Air over Fat"

    elif sample_type == "Emulsion":
        show_air = False
        show_fat = st.checkbox("Show Fat Particles", value=True)
        show_adsorbed = False
        adsorbed_opacity = 0.0
        air_opacity = 0.0
        fat_opacity = st.slider(
            "Fat Particles Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Transparency level for fat particles overlay layer (0.0=fully transparent, 1.0=fully opaque).",
        )
        z_order = "Fat over Air"

    else:
        col_vis1, col_vis2, col_vis3 = st.columns(3)
        with col_vis1:
            show_air = st.checkbox("Show Air Bubbles", value=True)
        with col_vis2:
            show_fat = st.checkbox("Show Fat Particles", value=True)
        with col_vis3:
            show_adsorbed = st.checkbox("Show Adsorbed Fat (in Bubbles)", value=True)

        air_opacity = st.slider(
            "Air Bubbles Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Transparency level for air bubbles overlay layer (0.0=fully transparent, 1.0=fully opaque).",
        )

        fat_opacity = st.slider(
            "Fat Particles Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Transparency level for fat particles overlay layer (0.0=fully transparent, 1.0=fully opaque).",
        )

        adsorbed_opacity = st.slider(
            "Adsorbed Fat Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Transparency level for adsorbed fat particles on bubble walls (0.0=fully transparent, 1.0=fully opaque).",
        )

        z_order = st.radio(
            "Z-Order (Layering)",
            options=["Fat over Air", "Air over Fat"],
            index=0,
            help="Layering priority: 'Fat over Air' renders fat particles on top of air bubbles; 'Air over Fat' renders air bubbles on top.",
        )

    # Universal for all sample types (Foam / Emulsion / Whipped Emulsion)
    show_labels = st.checkbox(
        "🔢 Show Particle ID Numbers",
        value=True,
        key="show_particle_labels_toggle",
        help="Uncheck to hide ID numbers and get a clean color-coded phase distribution map for publications/reports."
    )

    st.markdown("**📏 Burn-in Scale Bar (Image Export)**")
    draw_scale_bar = st.checkbox("Draw Scale Bar on Images", value=False)

    if draw_scale_bar:
        col_sb1, col_sb2, col_sb3 = st.columns([1.2, 1.2, 1.0])
        with col_sb1:
            sb_length = st.number_input("Length (µm)", min_value=0.1, max_value=5000.0, value=10.0, step=1.0)
        with col_sb2:
            sb_color = st.selectbox("Color", ["White", "Black", "Yellow", "Red"], index=0)
        with col_sb3:
            sb_thick = st.slider("Thickness", min_value=1, max_value=10, value=3)


# ---------------- MAIN APPLICATION DISPLAY ----------------
if is_light_mode:
    st.title("PhasePix: Quantitative Light Microscopy Emulsion Analyzer")
    st.markdown(
        "Automated brightfield & crystallized fat droplet segmentation, granulometry ($D[3,2], D[4,3]$), and coalescence network analytics."
    )
else:
    st.title("PhasePix: Quantitative SEM Microstructure Analyzer")
    st.markdown(
        "High-throughput segmentation of SEM phases using Cellpose deep learning with instant phase sorting and size distribution."
    )

if image_bytes is None:
    st.info("👆 Please select an image from the 'images/' folder or upload one via the sidebar to begin analysis.")
else:
    clean_stem = Path(image_name).stem

    # Single-phase Emulsion (incl. Light Microscopy) hides all air-bubble /
    # adsorbed-fat UI controls, since those phases cannot exist in this mode.
    is_emulsion_ui = (sample_type == "Emulsion") or is_light_mode

    # 1. Decode original full micrograph
    original_img = decode_image_bytes(image_bytes)
    img_h, img_w = original_img.shape[:2]

    # Guard against invalid footer height exceeding image bounds
    effective_footer = footer_height
    if effective_footer >= img_h:
        st.error(
            f"Footer Height ({effective_footer}px) cannot exceed total image height ({img_h}px). Footer exclusion disabled."
        )
        effective_footer = 0

    # Cropping: Prepare analysis image excluding bottom information banner AFTER scale detection
    analysis_img = (
        original_img[:-effective_footer, :]
        if effective_footer > 0
        else original_img
    )

    # 2. Run / Retrieve cached Cellpose segmentation on cropped analysis_img
    #    Mode selection controls which detector(s) fire:
    #      • Emulsion / None          → standard Cellpose (fat-optimised diameter)
    #      • Foam / Air Bubbles Only  → segment_air_bubbles_auto only
    #      • Two-Pass                 → air pass first, then fat pass; results merged

    _is_air_only = (
        sample_type == "Foam"
        or detection_focus == "🫧 Air Bubbles Only (Fast & Precise)"
    )
    _is_two_pass = (detection_focus == "🫧+🟠 Both Air Bubbles & Fat Particles (Two-Pass)")

    air_masks: Optional[np.ndarray] = None
    filtered_air_masks: Optional[np.ndarray] = None
    _air_df_pass: pd.DataFrame = pd.DataFrame()

    # Additive persistent storage for manually drawn objects (read globally for bubble mask building)
    manual_items_key = f"saved_manual_items_{image_name}"
    if manual_items_key not in st.session_state:
        st.session_state[manual_items_key] = []
    saved_manual_items = st.session_state.get(manual_items_key, [])
    active_excl = set(st.session_state.get("interactive_excluded_ids", set()))
    confirmed_air_ids = set()
    manual_air_ids = set()
    combined_bubble_mask = np.zeros(analysis_img.shape[:2], dtype=bool)
    manual_air_canvas = _rasterize_manual_air_bubbles(
        saved_manual_items,
        analysis_img.shape[:2],
        st.session_state.get("manual_phase_overrides", {}),
    )

    def compute_cumulative_bubble_mask(auto_air_layer: Optional[np.ndarray]) -> np.ndarray:
        """
        Build cumulative binary mask of all confirmed air bubbles.
        Combines:
        1. Binary mask of confirmed automatically detected bubbles (LUT gather, no
           ``np.isin`` sorting pass): if auto_air_layer is not None and confirmed_air_ids:
               c_mask = c_mask | (_lut_label_filter(auto_air_layer, confirmed_air_ids) > 0)
        2. Filled polygon masks of active manual objects from saved_manual_items marked
           as 'air_bubbles' via cv2.fillPoly.
        If no confirmed auto air bubbles and no manual air bubbles, mask is completely False.
        """
        c_mask = np.zeros(analysis_img.shape[:2], dtype=bool)
        if auto_air_layer is not None and confirmed_air_ids:
            c_mask = c_mask | (_lut_label_filter(auto_air_layer, confirmed_air_ids, keep=True) > 0)

        cur_saved = st.session_state.get(manual_items_key, [])
        if cur_saved:
            manual_layer = np.zeros(analysis_img.shape[:2], dtype=np.uint8)
            overrides = st.session_state.get("manual_phase_overrides", {})
            has_manual = False
            for itm in cur_saved:
                m_id = itm.get("id")
                if m_id in active_excl:
                    continue
                m_phase = str(overrides.get(m_id, itm.get("assigned_class", "air_bubbles"))).lower()
                if m_phase == "air_bubbles":
                    pts = np.asarray(itm.get("polygon", {}).get("points", []), dtype=np.float32)
                    if len(pts) >= 3:
                        pts_int = np.round(pts).astype(np.int32)
                        cv2.fillPoly(manual_layer, [pts_int], 1)
                        has_manual = True
            if has_manual:
                c_mask = c_mask | (manual_layer > 0)
        return c_mask

    if model_type == "Custom Model File..." and not custom_model_path:
        st.warning("⚠️ Please upload a custom .pth model file in the sidebar to run segmentation.")
        masks = np.zeros(analysis_img.shape[:2], dtype=np.int32)
        _air_df_pass = pd.DataFrame()
    elif _is_air_only:
        # ── Air Bubbles Only / Foam ───────────────────────────────────────────
        min_air_px = float(min_air_size) / float(pixel_to_um_ratio) if pixel_to_um_ratio > 0 else 20.0
        max_air_px = float(max_air_size) / float(pixel_to_um_ratio) if pixel_to_um_ratio > 0 else 200.0
        masks = cached_segment_air_bubbles_auto(
            image=analysis_img,
            min_diam_px=min_air_px,
            max_diam_px=max_air_px,
            model_type=model_type,
            custom_model_path=custom_model_path,
            flow_threshold=float(flow_threshold),
            cellprob_threshold=float(cellprob_threshold),
        )
        air_masks = masks
        if masks is not None and np.any(masks > 0):
            _air_df_raw = cached_extract_metrics(
                original_image=analysis_img,
                mask=masks,
                pixel_to_um_ratio=pixel_to_um_ratio,
            )
            if not _air_df_raw.empty:
                _in_air = (
                    (_air_df_raw["diameter_um"] >= float(min_air_size))
                    & (_air_df_raw["diameter_um"] <= float(max_air_size))
                    & (_air_df_raw["circularity"] >= float(air_min_circularity))
                )
                _air_df_pass = _air_df_raw[_in_air].copy()
                _air_df_pass["phase_class"] = "air_bubbles"
                confirmed_air_ids = set(_air_df_pass[~_air_df_pass["id"].isin(active_excl)]["id"].tolist())
            else:
                _air_df_pass = pd.DataFrame()
        else:
            _air_df_pass = pd.DataFrame()

        # Pass 1 cleanup: no air bubble may live inside a bigger air bubble/manual bubble
        (
            _air_df_pass,
            air_masks,
            _air_occupied,
        ) = cached_remove_nested_matryoshka_masks(
            _air_df_pass,
            air_masks,
            prior_occupied=manual_air_canvas,
            overlap_thresh=0.50,
        )
        masks = air_masks
        confirmed_air_ids = (
            set(
                _air_df_pass[~_air_df_pass["id"].isin(active_excl)]["id"].tolist()
            )
            if not _air_df_pass.empty
            else set()
        )
        combined_bubble_mask = compute_cumulative_bubble_mask(air_masks)
    elif _is_two_pass:
        # ── Two-Pass: air first, fat second ──────────────────────────────────
        min_air_px = float(min_air_size) / float(pixel_to_um_ratio) if pixel_to_um_ratio > 0 else 20.0
        max_air_px = float(max_air_size) / float(pixel_to_um_ratio) if pixel_to_um_ratio > 0 else 200.0
        with st.spinner("🫧 Pass 1/2 — Detecting air bubbles (multi-scale + dark crater)..."):
            air_masks = cached_segment_air_bubbles_auto(
                image=analysis_img,
                min_diam_px=min_air_px,
                max_diam_px=max_air_px,
                model_type=model_type,
                custom_model_path=custom_model_path,
                flow_threshold=float(flow_threshold),
                cellprob_threshold=float(cellprob_threshold),
            )
        # Extract air metrics immediately so we can label them 'air_bubbles'
        _air_df_raw = cached_extract_metrics(
            original_image=analysis_img,
            mask=air_masks,
            pixel_to_um_ratio=pixel_to_um_ratio,
        )
        if not _air_df_raw.empty:
            _in_air = (
                (_air_df_raw["diameter_um"] >= float(min_air_size))
                & (_air_df_raw["diameter_um"] <= float(max_air_size))
                & (_air_df_raw["circularity"] >= float(air_min_circularity))
            )
            _air_df_pass = _air_df_raw[_in_air].copy()
            _air_df_pass["phase_class"] = "air_bubbles"
            confirmed_air_ids = set(_air_df_pass[~_air_df_pass["id"].isin(active_excl)]["id"].tolist())
        else:
            _air_df_pass = pd.DataFrame()

        # Pass 1 cleanup: air_masks and _air_df_pass must contain no nested bubbles
        (
            _air_df_pass,
            air_masks,
            _air_occupied,
        ) = cached_remove_nested_matryoshka_masks(
            _air_df_pass,
            air_masks,
            prior_occupied=manual_air_canvas,
            overlap_thresh=0.50,
        )
        confirmed_air_ids = (
            set(
                _air_df_pass[~_air_df_pass["id"].isin(active_excl)]["id"].tolist()
            )
            if not _air_df_pass.empty
            else set()
        )

        # Build cumulative bubble mask (Rule 1): combines automated air_masks > 0 and ALL manual air_bubbles
        combined_bubble_mask = compute_cumulative_bubble_mask(air_masks)

        # Fat pass: optimise diameter for small globules
        fat_eff_diameter = float(np.clip(
            (float(min_fat_size) * 1.25) / float(pixel_to_um_ratio) if pixel_to_um_ratio > 0 else 25.0,
            15.0, 45.0,
        ))
        with st.spinner("🟠 Pass 2/2 — Detecting fat particles..."):
            masks = cached_segmentation(
                image=analysis_img,
                model_name=model_type,
                custom_model_path=custom_model_path,
                diameter=fat_eff_diameter,
                flow_threshold=float(flow_threshold),
                cellprob_threshold=float(cellprob_threshold),
                enable_clahe=enable_clahe,
            )

        # Shift fat mask labels so fat and air IDs never collide and stay strictly positive (> 0)
        fat_id_offset = int(air_masks.max()) + 100 if air_masks is not None else 100
        masks = np.where(masks > 0, masks + fat_id_offset, 0).astype(np.int32)

        # Pass 3: Deep-scan for hidden fat globules inside bubble craters (including manual bubbles)
        if enable_deepscan and np.any(combined_bubble_mask):
            min_fat_px = float(min_fat_size) / float(pixel_to_um_ratio) if pixel_to_um_ratio > 0 else 4.0
            max_fat_px = float(max_fat_size) / float(pixel_to_um_ratio) if pixel_to_um_ratio > 0 else 50.0
            with st.spinner("🔬 Pass 3/3 — Deep-scanning bubble craters for hidden fat globules..."):
                masks = cached_segment_fat_in_bubbles_deepscan(
                    image=analysis_img,
                    bubble_region_mask=combined_bubble_mask.astype(np.int32),
                    base_fat_masks=masks,
                    min_fat_px=min_fat_px,
                    max_fat_px=max_fat_px,
                    model_type=model_type,
                    custom_model_path=custom_model_path,
                    flow_threshold=float(flow_threshold),
                    cellprob_threshold=-2.0,
                )
    else:
        # ── Standard single-pass (Emulsion) ──────────────────────────────────
        with st.spinner("Executing AI particle segmentation..."):
            if is_light_mode and is_crystallized_mode and enable_multiscale_crystal:
                # Two-scale hierarchical pass: intact large crystallized globules +
                # small droplets, merged with shard rejection.
                masks = cached_segment_crystallized_multiscale(
                    image=analysis_img,
                    base_diameter_px=eff_diameter,
                    model_name=model_type,
                    custom_model_path=custom_model_path,
                    flow_threshold=float(flow_threshold),
                    cellprob_threshold=float(cellprob_threshold),
                    enable_clahe=enable_clahe,
                    smooth_strength=int(crystal_smooth_strength),
                    invert_brightfield=bool(invert_brightfield),
                )
            else:
                seg_input = analysis_img.copy()
                if is_light_mode and is_crystallized_mode and crystal_smooth_strength > 0:
                    from cellpose_model import smooth_internal_crystals
                    seg_input = smooth_internal_crystals(
                        seg_input, strength=int(crystal_smooth_strength)
                    )
                if is_light_mode and invert_brightfield:
                    seg_input = cv2.bitwise_not(seg_input)

                masks = cached_segmentation(
                    image=seg_input,
                    model_name=model_type,
                    custom_model_path=custom_model_path,
                    diameter=eff_diameter,
                    flow_threshold=float(flow_threshold),
                    cellprob_threshold=float(cellprob_threshold),
                    enable_clahe=enable_clahe,
                )
        _air_df_pass = pd.DataFrame()
        air_masks = None
        combined_bubble_mask = compute_cumulative_bubble_mask(None)

    # Ensure manual air bubbles are included for all modes
    combined_bubble_mask = compute_cumulative_bubble_mask(air_masks)

    # 3. Extract morphological and photometric metrics on analysis_img
    raw_df = cached_extract_metrics(
        original_image=analysis_img,
        mask=masks,
        pixel_to_um_ratio=pixel_to_um_ratio,
    )

    # 3b. Reject nested "matryoshka" double detections (small objects inside bigger ones).
    #     The occupied canvas stores hole-filled footprints, so a smaller detection sitting
    #     inside the hole of a donut-shaped parent is correctly flagged as an artifact.
    #     Delegated to the memoised helper: one cv2.findContours + fill pass per particle
    #     must never be repeated just because a slider or tab changed.
    if not raw_df.empty and masks is not None and np.any(masks > 0):
        raw_df, masks = cached_prune_nested_auto_objects(
            raw_df,
            masks,
            overlap_thresh=0.60,
        )

    # 3c. Two-Pass safety net: Pass 2 (fat detector) must never recreate air bubbles.
    #     Any auto Pass-2 object larger than max_fat_size that overlaps the cumulative
    #     bubble mask by >=35% of its own area is a duplicate of the bubble floor.
    if (
        _is_two_pass
        and not raw_df.empty
        and masks is not None
        and np.any(combined_bubble_mask)
    ):
        bubble_binary = combined_bubble_mask.astype(bool)
        pass2_rejected_ids = set()
        masks_int = (
            masks if np.issubdtype(masks.dtype, np.integer) else masks.astype(np.int32)
        )
        try:
            pass2_slices = find_objects(masks_int)
        except Exception:
            pass2_slices = None

        # Only oversized Pass-2 leftovers are candidates: filtering the frame up front
        # keeps the Python loop over a handful of rows instead of thousands.
        pass2_candidates = raw_df[
            (raw_df["id"] < 10000) & (raw_df["diameter_um"] > float(max_fat_size))
        ]
        for _, row in pass2_candidates.iterrows():
            obj_id = int(row["id"])
            sl = None
            if pass2_slices is not None and 0 < obj_id <= len(pass2_slices):
                sl = pass2_slices[obj_id - 1]
            if sl is None:
                full_mask = masks_int == obj_id
                if not np.any(full_mask):
                    continue
                ys, xs = np.nonzero(full_mask)
                sl = (
                    slice(int(ys.min()), int(ys.max()) + 1),
                    slice(int(xs.min()), int(xs.max()) + 1),
                )
            local = masks_int[sl] == obj_id
            local_area = float(np.count_nonzero(local))
            if local_area <= 0:
                continue
            overlap_area = float(np.count_nonzero(local & bubble_binary[sl]))
            if overlap_area / local_area >= 0.35:
                pass2_rejected_ids.add(obj_id)

        if pass2_rejected_ids:
            raw_df = (
                raw_df[~raw_df["id"].isin(pass2_rejected_ids)]
                .copy()
                .reset_index(drop=True)
            )
            masks = _lut_label_filter(masks, pass2_rejected_ids, keep=False)

    # Re-synchronize manual objects with current spatial calibration
    saved_manual_items = st.session_state.get(manual_items_key, [])
    manual_metrics_rows = []
    manual_polys = []
    for item in saved_manual_items:
        m = dict(item["metrics"])
        area_px = m.get("area_px")
        if area_px is not None and area_px > 0:
            m["area_um2"] = float(area_px) * (float(pixel_to_um_ratio) ** 2)
            m["diameter_um"] = (
                math.sqrt(4.0 * m["area_um2"] / math.pi) if m["area_um2"] > 0 else 0.0
            )
        manual_metrics_rows.append(m)
        manual_polys.append(item["polygon"])

    st.session_state["manual_objects_list"] = manual_metrics_rows
    st.session_state["manual_polygons_list"] = manual_polys

    # Add manual objects directly into raw_df as new rows (no deletion of existing masks)
    if manual_metrics_rows:
        manual_df = pd.DataFrame(manual_metrics_rows)
        raw_df = pd.concat([raw_df, manual_df], ignore_index=True)

    # Refresh cumulative bubble mask to ensure all latest manual items are integrated (Rule 1)
    combined_bubble_mask = compute_cumulative_bubble_mask(air_masks)

    # 4. Spatial Phase Filtering & Classification (Colloid Chemistry Rules)
    partial_ids = set()
    if not raw_df.empty:
        overrides_dict = st.session_state.get("manual_phase_overrides", {})
        manual_air_ids = set()
        for item in saved_manual_items:
            m_id = item.get("id")
            if m_id in active_excl:
                continue
            m_phase = str(overrides_dict.get(m_id, item.get("assigned_class", "air_bubbles"))).lower()
            if m_phase == "air_bubbles":
                manual_air_ids.add(m_id)

        all_air_ids = confirmed_air_ids | manual_air_ids

        # Rule 2: Coordinate centroid (centroid_x, centroid_y) of each automated object (id < 10000)
        # is tested against cumulative bubble mask (combined_bubble_mask).
        # Fully vectorised: a row-wise DataFrame.apply() costs ~100x more than a single
        # fancy-index gather into the boolean bubble mask.
        if combined_bubble_mask.any() and not raw_df.empty:
            cx_arr = np.clip(
                np.rint(raw_df["centroid_x"].to_numpy(dtype=float)).astype(np.intp),
                0,
                combined_bubble_mask.shape[1] - 1,
            )
            cy_arr = np.clip(
                np.rint(raw_df["centroid_y"].to_numpy(dtype=float)).astype(np.intp),
                0,
                combined_bubble_mask.shape[0] - 1,
            )
            in_bub = combined_bubble_mask[cy_arr, cx_arr] & (
                ~raw_df["id"].isin(all_air_ids).to_numpy()
            )
            in_bubble_vec = pd.Series(in_bub, index=raw_df.index)
        else:
            in_bubble_vec = pd.Series(False, index=raw_df.index)

        raw_df["in_bubble"] = in_bubble_vec

        is_manual = raw_df["id"] >= 10000
        is_auto = raw_df["id"] < 10000
        is_air_bubble_obj = raw_df["id"].isin(all_air_ids)

        in_fat_size_range = (
            (raw_df["diameter_um"] >= float(min_fat_size))
            & (raw_df["diameter_um"] <= float(max_fat_size))
        )
        in_air_size_range = (
            (raw_df["diameter_um"] >= float(min_air_size))
            & (raw_df["diameter_um"] <= float(max_air_size))
        )

        # Rule 3 & Rule 4:
        # If the centroid of an automated object lies inside the cumulative bubble mask,
        # its size falls into fat range [min_fat_size, max_fat_size], and its circularity >= ads_min_circularity,
        # it is identified as adsorbed fat.
        is_partial = (
            is_auto
            & (~is_air_bubble_obj)
            & in_bubble_vec
            & in_fat_size_range
            & (raw_df["circularity"] >= float(ads_min_circularity))
        )
        partial_ids = set(raw_df.loc[is_partial, "id"].tolist())

        # Outside bubbles: standard shape filtering by circularity threshold.
        # Light Microscopy additionally enforces a convexity (solidity) gate for
        # fat-range objects, which rejects crescent shards and inter-droplet wedges.
        if is_light_mode and float(fat_min_solidity) > 0 and "solidity" in raw_df.columns:
            solidity_ok = np.where(
                in_fat_size_range,
                raw_df["solidity"] >= float(fat_min_solidity),
                True,
            )
        else:
            solidity_ok = True

        circ_ok_outside = np.where(
            in_fat_size_range,
            (raw_df["circularity"] >= float(fat_min_circularity)) & solidity_ok,
            np.where(
                in_air_size_range,
                raw_df["circularity"] >= float(air_min_circularity),
                raw_df["circularity"] >= float(fat_min_circularity),
            ),
        )

        circ_ok = np.where(
            is_manual,
            True,
            np.where(
                is_partial,
                True,  # Adsorbed fat inside bubbles that meets ads_min_circularity passes
                np.where(
                    is_air_bubble_obj,
                    raw_df["circularity"] >= float(air_min_circularity),
                    circ_ok_outside,
                ),
            ),
        )

        filtered_df = raw_df[circ_ok].copy().reset_index(drop=True)
        valid_ids = set(filtered_df["id"].tolist())
        # Zero out rejected object IDs from segmentation mask (non-destructive copy)
        filtered_masks = _lut_label_filter(masks, valid_ids, keep=True)
    else:
        filtered_df = raw_df.copy()
        filtered_masks = masks.copy()
        partial_ids = set()
        in_bubble_vec = pd.Series(dtype=bool)

    # 5. Exclude particle IDs via interactive table deletions only
    if "interactive_excluded_ids" not in st.session_state:
        st.session_state["interactive_excluded_ids"] = set()
    if st.session_state.get("active_image_name") != image_name:
        st.session_state["active_image_name"] = image_name
        st.session_state["interactive_excluded_ids"] = set()

    active_excluded_ids = set(st.session_state.get("interactive_excluded_ids", set()))

    if active_excluded_ids:
        if not filtered_df.empty:
            filtered_df = filtered_df[~filtered_df["id"].isin(active_excluded_ids)].copy().reset_index(drop=True)
        # Zero out excluded IDs in mask array
        filtered_masks = _lut_label_filter(filtered_masks, active_excluded_ids, keep=False)

    # Filter air pass masks and dataframe in Two-Pass mode
    if _is_two_pass and air_masks is not None and not _air_df_pass.empty:
        valid_air_ids = set(_air_df_pass["id"].tolist()) - active_excluded_ids
        _air_df_pass = _air_df_pass[_air_df_pass["id"].isin(valid_air_ids)].copy()
        filtered_air_masks = _lut_label_filter(air_masks, valid_air_ids, keep=True)
    else:
        filtered_air_masks = None

    # 6. Track classification-parameter changes to clear sticky data_editor / overrides
    current_class_params = (
        sample_type,
        float(min_air_size),
        float(max_air_size),
        float(min_fat_size),
        float(max_fat_size),
        str(overlap_priority),
        str(out_of_range_handling),
    )
    if st.session_state.get("last_class_params") != current_class_params or st.session_state.pop("force_class_refresh", False):
        st.session_state["last_class_params"] = current_class_params
        editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
        for k in editor_keys:
            del st.session_state[k]
        # Clear overrides so Out-of-Range / Min-Max reclassification is not blocked
        st.session_state["manual_phase_overrides"] = {}
        st.session_state["particle_editor_rev"] = int(st.session_state.get("particle_editor_rev", 0)) + 1

    # 7. Classify objects into phases based on Sample Type & Size Ranges
    if sample_type == "Emulsion":
        classified_df = filtered_df.copy()
        if not classified_df.empty:
            in_fat = (
                (classified_df["diameter_um"] >= float(min_fat_size))
                & (classified_df["diameter_um"] <= float(max_fat_size))
            )
            oor = str(out_of_range_handling).lower()
            oor_label = "fat_particles" if "fat" in oor else "Unclassified"
            classified_df["phase_class"] = np.where(in_fat, "fat_particles", oor_label)
        else:
            classified_df["phase_class"] = pd.Series(dtype=str)

    elif sample_type == "Foam" or _is_air_only:
        # Air Bubbles Only: strict size-range gate on air-bubble detector output
        classified_df = filtered_df.copy()
        if not classified_df.empty:
            in_air = (
                (classified_df["diameter_um"] >= float(min_air_size))
                & (classified_df["diameter_um"] <= float(max_air_size))
            )
            oor = str(out_of_range_handling).lower()
            oor_label = "air_bubbles" if "air" in oor else "Unclassified"
            classified_df["phase_class"] = np.where(in_air, "air_bubbles", oor_label)
        else:
            classified_df["phase_class"] = pd.Series(dtype=str)

    elif _is_two_pass:
        # Two-Pass: Pass 2 detects fat only — it may NEVER create 'air_bubbles'.
        # All air bubbles come from Pass 1 (_air_df_pass); Pass-2 objects outside the
        # fat size range are labelled 'Unclassified' (or fat if the user chose that).
        if not filtered_df.empty:
            in_fat_size_pass2 = (
                (filtered_df["diameter_um"] >= float(min_fat_size))
                & (filtered_df["diameter_um"] <= float(max_fat_size))
            )
            oor = str(out_of_range_handling).lower()
            oor_label = "fat_particles" if "fat" in oor else "Unclassified"
            fat_classified = filtered_df.copy()
            fat_classified["phase_class"] = np.where(
                in_fat_size_pass2, "fat_particles", oor_label
            )
        else:
            fat_classified = filtered_df.copy()
            fat_classified["phase_class"] = pd.Series(dtype=str)

        # Merge air and fat dataframes with positive IDs (fat IDs already offset by fat_id_offset)
        if not _air_df_pass.empty:
            classified_df = pd.concat(
                [_air_df_pass, fat_classified],
                ignore_index=True,
            )
        else:
            classified_df = fat_classified

    else:
        classified_df = classify_by_size(
            filtered_df,
            min_air_size=min_air_size,
            max_air_size=max_air_size,
            min_fat_size=min_fat_size,
            max_fat_size=max_fat_size,
            overlap_priority=overlap_priority,
            out_of_range_class=out_of_range_handling,
        )

    # Normalize phase_class naming to standard options ('air_bubbles', 'fat_particles', 'adsorbed_fat', 'Unclassified')
    if not classified_df.empty:
        classified_df["phase_class"] = classified_df["phase_class"].replace({
            "unclassified": "Unclassified",
            "partial_fat": "adsorbed_fat",
        })

    # Step 7 (After classify_by_size):
    # Rule 3 & Rule 4: Enforce 'adsorbed_fat' for all partial_ids inside bubbles, forbid outside
    if not classified_df.empty:
        if "in_bubble" not in classified_df.columns:
            # Recompute in_bubble if not already attached (vectorised centroid gather)
            cx_arr = np.clip(
                np.rint(classified_df["centroid_x"].to_numpy(dtype=float)).astype(np.intp),
                0,
                combined_bubble_mask.shape[1] - 1,
            )
            cy_arr = np.clip(
                np.rint(classified_df["centroid_y"].to_numpy(dtype=float)).astype(np.intp),
                0,
                combined_bubble_mask.shape[0] - 1,
            )
            not_air_row = (
                classified_df["phase_class"].astype(str).str.lower().to_numpy() != "air_bubbles"
            )
            classified_df["in_bubble"] = not_air_row & combined_bubble_mask[cy_arr, cx_arr]

        # Rule 3: Forcibly assign 'adsorbed_fat' to all partial_ids
        if partial_ids:
            classified_df.loc[classified_df["id"].isin(partial_ids), "phase_class"] = "adsorbed_fat"

        # Also catch any automated fat particles inside combined_bubble_mask within fat size range and circularity
        is_auto_obj = classified_df["id"] < 10000
        in_bub_obj = classified_df["in_bubble"] == True
        is_in_fat_size = (
            (classified_df["diameter_um"] >= float(min_fat_size))
            & (classified_df["diameter_um"] <= float(max_fat_size))
        )
        circ_ads_ok = classified_df["circularity"] >= float(ads_min_circularity)
        not_air_phase = classified_df["phase_class"] != "air_bubbles"
        auto_bubble_fat = is_auto_obj & in_bub_obj & is_in_fat_size & circ_ads_ok & not_air_phase
        if auto_bubble_fat.any():
            classified_df.loc[auto_bubble_fat, "phase_class"] = "adsorbed_fat"

        # Rule 4: Outside bubble field, status 'adsorbed_fat' is strictly FORBIDDEN!
        # Objects outside bubble field cannot have adsorbed_fat status
        outside_partial = (~classified_df["in_bubble"].fillna(False).astype(bool)) & classified_df["phase_class"].astype(str).str.lower().isin(["adsorbed_fat"])
        if outside_partial.any():
            classified_df.loc[outside_partial, "phase_class"] = np.where(
                is_in_fat_size[outside_partial], "fat_particles", "Unclassified"
            )

        classified_df["is_partial_in_bubble"] = classified_df["phase_class"].astype(str).str.lower().eq("adsorbed_fat")

    # State synchronization: Apply manual phase reclassifications from st.session_state
    if "manual_phase_overrides" not in st.session_state:
        st.session_state["manual_phase_overrides"] = {}
    if st.session_state.get("active_image_name_overrides") != image_name:
        st.session_state["active_image_name_overrides"] = image_name
        st.session_state["manual_phase_overrides"] = {}

    if st.session_state.get("manual_phase_overrides") and not classified_df.empty:
        for obj_id, override_phase in list(st.session_state["manual_phase_overrides"].items()):
            norm_ov = "adsorbed_fat" if str(override_phase).lower() in ("adsorbed_fat", "partial_fat") else override_phase
            match_row = classified_df[classified_df["id"] == obj_id]
            if not match_row.empty:
                row_in_b = bool(match_row["in_bubble"].values[0]) if "in_bubble" in match_row.columns else False
                if norm_ov == "adsorbed_fat" and not row_in_b:
                    # Dynamically clean override when outside bubble: object can NEVER have adsorbed_fat status outside bubble!
                    del st.session_state["manual_phase_overrides"][obj_id]
                    diam = float(match_row["diameter_um"].values[0]) if "diameter_um" in match_row.columns else 0.0
                    in_fat_sz = (float(min_fat_size) <= diam <= float(max_fat_size))
                    norm_ov = "fat_particles" if in_fat_sz else "Unclassified"
                classified_df.loc[classified_df["id"] == obj_id, "phase_class"] = norm_ov
            else:
                if norm_ov == "adsorbed_fat":
                    del st.session_state["manual_phase_overrides"][obj_id]

        # Enforce outside bubble rule again after manual overrides
        outside_ads = (~classified_df["in_bubble"].fillna(False).astype(bool)) & classified_df["phase_class"].astype(str).str.lower().isin(["adsorbed_fat"])
        if outside_ads.any():
            is_in_fat = (
                (classified_df["diameter_um"] >= float(min_fat_size))
                & (classified_df["diameter_um"] <= float(max_fat_size))
            )
            classified_df.loc[outside_ads, "phase_class"] = np.where(
                is_in_fat[outside_ads], "fat_particles", "Unclassified"
            )

        classified_df["is_partial_in_bubble"] = classified_df["phase_class"].astype(str).str.lower().eq("adsorbed_fat")

    # Final air-bubble deduplication: drop smaller air bubbles that sit inside bigger ones
    # (auto Pass 1, Pass 2 leftovers and manual bubbles; overlap >= 45% of filled area).
    if not classified_df.empty and "phase_class" in classified_df.columns:
        air_bubble_rows = classified_df[
            classified_df["phase_class"].astype(str).str.lower() == "air_bubbles"
        ].sort_values(by="area_um2", ascending=False)
        if len(air_bubble_rows) > 1:
            analysis_shape = analysis_img.shape[:2]
            manual_poly_lookup = {}
            for polygon in st.session_state.get("manual_polygons_list", []) or []:
                if not isinstance(polygon, dict):
                    continue
                try:
                    poly_id = int(polygon.get("id"))
                except (TypeError, ValueError):
                    continue
                pts = polygon.get("points")
                if pts is None:
                    continue
                try:
                    pts_arr = np.asarray(pts, dtype=np.float32)
                except (TypeError, ValueError):
                    continue
                if pts_arr.ndim == 2 and pts_arr.shape[0] >= 3:
                    manual_poly_lookup[poly_id] = pts_arr[:, :2]

            def _bubble_binary_mask(bubble_id: int) -> Optional[np.ndarray]:
                if bubble_id >= 10000:
                    poly_pts = manual_poly_lookup.get(int(bubble_id))
                    if poly_pts is None:
                        return None
                    canvas = np.zeros(analysis_shape, dtype=np.uint8)
                    cv2.fillPoly(canvas, [np.round(poly_pts).astype(np.int32)], 1)
                    return canvas
                if filtered_air_masks is not None:
                    air_mask = filtered_air_masks == bubble_id
                    if np.any(air_mask):
                        return air_mask.astype(np.uint8)
                if filtered_masks is not None:
                    fat_mask = filtered_masks == bubble_id
                    if np.any(fat_mask):
                        return fat_mask.astype(np.uint8)
                return None

            occupied_air = np.zeros(analysis_shape, dtype=bool)
            rejected_air_ids = set()
            for _, air_row in air_bubble_rows.iterrows():
                air_id = int(air_row["id"])
                air_mask = _bubble_binary_mask(air_id)
                if air_mask is None:
                    continue
                filled_air = np.zeros(analysis_shape, dtype=np.uint8)
                air_contours, _ = cv2.findContours(
                    air_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if air_contours:
                    cv2.drawContours(
                        filled_air, air_contours, -1, 1, thickness=cv2.FILLED
                    )
                filled_bool = filled_air.astype(bool)
                filled_area = float(np.count_nonzero(filled_bool))
                if filled_area <= 0:
                    continue
                overlap_fraction = (
                    float(np.count_nonzero(filled_bool & occupied_air)) / filled_area
                )
                if overlap_fraction >= 0.45:
                    rejected_air_ids.add(air_id)
                else:
                    occupied_air |= filled_bool

            if rejected_air_ids:
                classified_df = (
                    classified_df[~classified_df["id"].isin(rejected_air_ids)]
                    .copy()
                    .reset_index(drop=True)
                )
                if filtered_masks is not None:
                    filtered_masks = _lut_label_filter(
                        filtered_masks, rejected_air_ids, keep=False
                    )
                if filtered_air_masks is not None:
                    filtered_air_masks = _lut_label_filter(
                        filtered_air_masks, rejected_air_ids, keep=False
                    )
                confirmed_air_ids = set(confirmed_air_ids) - rejected_air_ids
                manual_air_ids = set(manual_air_ids) - rejected_air_ids
                if not _air_df_pass.empty:
                    _air_df_pass = _air_df_pass[
                        ~_air_df_pass["id"].isin(rejected_air_ids)
                    ].copy()

    # High-level statistics
    total_objects = len(classified_df)
    air_count = int((classified_df["phase_class"] == "air_bubbles").sum()) if not classified_df.empty else 0
    fat_count = int((classified_df["phase_class"] == "fat_particles").sum()) if not classified_df.empty else 0
    adsorbed_fat_count = int((classified_df["phase_class"] == "adsorbed_fat").sum()) if not classified_df.empty else 0
    unclass_count = int(classified_df["phase_class"].astype(str).str.lower().eq("unclassified").sum()) if not classified_df.empty else 0

    # Visualization preparations: Dimming & Padding
    # Apply translucent dimming to footer panel on full-size display image
    base_display_img = (
        apply_footer_dimming(original_img, footer_height=effective_footer, dim_factor=0.4)
        if effective_footer > 0
        else np.stack([original_img, original_img, original_img], axis=-1)
    )

    # Zero-pad filtered masks along the bottom to match full micrograph dimensions
    padded_masks = (
        np.pad(filtered_masks, ((0, effective_footer), (0, 0)), mode="constant")
        if effective_footer > 0
        else filtered_masks
    )

    padded_air_masks = (
        (
            np.pad(filtered_air_masks, ((0, effective_footer), (0, 0)), mode="constant")
            if effective_footer > 0
            else filtered_air_masks
        )
        if filtered_air_masks is not None
        else None
    )

    # Create full-resolution translucent color overlay with particle ID numbers
    active_manual_ids = set(classified_df["id"].tolist()) if not classified_df.empty else set()
    active_manual_polygons = [
        p for p in st.session_state.get("manual_polygons_list", [])
        if p.get("id") in active_manual_ids
    ]

    # Apparent contact angles for every confirmed/manual bubble (dashboard theta labels)
    _classified_air_ids = set()
    if not classified_df.empty and "phase_class" in classified_df.columns:
        _classified_air_ids = {
            int(value)
            for value in classified_df.loc[
                classified_df["phase_class"].astype(str).str.lower() == "air_bubbles",
                "id",
            ].tolist()
        }
    selected_bubble_ids_for_ads = sorted({
        int(value)
        for value in (set(confirmed_air_ids) | set(manual_air_ids) | _classified_air_ids)
    })

    ads_particles_df_global = pd.DataFrame()
    if (
        selected_bubble_ids_for_ads
        and show_adsorbed
        and show_labels
        and adsorbed_fat_count > 0
    ):
        try:
            _, ads_particles_df_global, _ = compute_interfacial_adsorption_metrics(
                df=classified_df,
                selected_bubble_ids=selected_bubble_ids_for_ads,
                air_masks=padded_air_masks,
                fat_masks=padded_masks,
                manual_polygons=active_manual_polygons,
                image_shape=base_display_img.shape[:2],
                pixel_to_um_ratio=float(pixel_to_um_ratio),
            )
        except Exception:
            # Keep the dashboard renderable even if contact-angle estimation fails
            ads_particles_df_global = pd.DataFrame()

    try:
        overlay = create_mask_overlay(
            image=base_display_img,
            masks=padded_masks,
            air_masks=padded_air_masks,
            df=classified_df,
            air_opacity=float(air_opacity),
            fat_opacity=float(fat_opacity),
            show_air=bool(show_air),
            show_fat=bool(show_fat),
            show_adsorbed=bool(show_adsorbed),
            adsorbed_opacity=float(adsorbed_opacity),
            z_order=str(z_order),
            manual_polygons=active_manual_polygons,
            show_labels=bool(show_labels),
            ads_particles_df=ads_particles_df_global,
        )
    except TypeError:
        import importlib
        import visualizer as vis_mod
        importlib.reload(vis_mod)
        overlay = vis_mod.create_mask_overlay(
            image=base_display_img,
            masks=padded_masks,
            air_masks=padded_air_masks,
            df=classified_df,
            air_opacity=float(air_opacity),
            fat_opacity=float(fat_opacity),
            show_air=bool(show_air),
            show_fat=bool(show_fat),
            show_adsorbed=bool(show_adsorbed),
            adsorbed_opacity=float(adsorbed_opacity),
            z_order=str(z_order),
            manual_polygons=active_manual_polygons,
            show_labels=bool(show_labels),
            ads_particles_df=ads_particles_df_global,
        )

    # Apply burn-in scale bar if requested
    if draw_scale_bar:
        try:
            try:
                from visualizer import burn_in_scale_bar
            except ImportError:
                # A Streamlit server already running before visualizer.py gained this
                # helper keeps the old module cached in sys.modules; reload it once.
                import importlib
                import visualizer as _vis_sb
                importlib.reload(_vis_sb)
                burn_in_scale_bar = _vis_sb.burn_in_scale_bar

            base_display_img = burn_in_scale_bar(
                image=base_display_img,
                length_um=float(sb_length),
                pixels_per_um=float(pixels_per_um),
                color_name=sb_color,
                thickness=int(sb_thick),
                footer_height=effective_footer
            )
            overlay = burn_in_scale_bar(
                image=overlay,
                length_um=float(sb_length),
                pixels_per_um=float(pixels_per_um),
                color_name=sb_color,
                thickness=int(sb_thick),
                footer_height=effective_footer
            )
        except Exception as e:
            st.warning(f"Could not draw scale bar: {e}")

    # Summary KPI Metric cards
    def _phase_mean_diameter(phase_key: str) -> float:
        if classified_df.empty:
            return 0.0
        subset = classified_df.loc[classified_df["phase_class"] == phase_key, "diameter_um"]
        return float(subset.mean()) if len(subset) else 0.0

    show_mean_diam = sample_type in ("Emulsion", "Foam")
    if sample_type == "Emulsion":
        avg_diam = _phase_mean_diameter("fat_particles")
    elif sample_type == "Foam":
        avg_diam = _phase_mean_diameter("air_bubbles")
    else:
        avg_diam = 0.0

    show_adsorbed_card = (adsorbed_fat_count > 0)

    # Dynamic column counting (Drop Air Bubbles card for Emulsion entirely)
    base_cols = 2 if sample_type in ("Emulsion", "Foam") else 3
    metric_cols = base_cols + (1 if unclass_count > 0 else 0) + (1 if show_mean_diam else 0) + (1 if show_adsorbed_card else 0)
    cols = st.columns(metric_cols)

    c_idx = 0
    cols[c_idx].metric("Total Objects", f"{total_objects:,}")
    c_idx += 1

    if sample_type == "Emulsion":
        fat_label = "Fat Droplets" if ('is_light_mode' in locals() and is_light_mode) else "Fat Particles"
        cols[c_idx].metric(
            fat_label,
            f"{fat_count:,}",
            f"{(fat_count / max(total_objects, 1) * 100):.1f}%",
        )
        c_idx += 1
    elif sample_type == "Foam":
        cols[c_idx].metric(
            "Air Bubbles",
            f"{air_count:,}",
            f"{(air_count / max(total_objects, 1) * 100):.1f}%",
        )
        c_idx += 1
    else:
        cols[c_idx].metric(
            "Air Bubbles",
            f"{air_count:,}",
            f"{(air_count / max(total_objects, 1) * 100):.1f}%",
        )
        c_idx += 1
        cols[c_idx].metric(
            "Fat Particles",
            f"{fat_count:,}",
            f"{(fat_count / max(total_objects, 1) * 100):.1f}%",
        )
        c_idx += 1

    if unclass_count > 0:
        cols[c_idx].metric(
            "Unclassified",
            f"{unclass_count:,}",
            f"{(unclass_count / max(total_objects, 1) * 100):.1f}%",
        )
        c_idx += 1

    if show_adsorbed_card:
        cols[c_idx].metric(
            "Adsorbed Fat",
            f"{adsorbed_fat_count:,}",
            "Excluded from PSD",
        )
        c_idx += 1

    if show_mean_diam:
        cols[c_idx].metric("Mean Diameter", f"{avg_diam:.2f} µm")
        c_idx += 1

    # ---------------- 🪄 AI PARAMETER ADVISOR REVIEW CARD ----------------
    _autotune_proposal = st.session_state.get("autotune_proposal")
    if _autotune_proposal is not None:
        _proposal_matches_image = (
            _autotune_proposal.get("image_name") == image_name
            and bool(_autotune_proposal.get("is_light_mode")) == bool(is_light_mode)
        )
        if not _proposal_matches_image:
            # Stale proposal from another image/modality: silently discard it.
            st.session_state.pop("autotune_proposal", None)
        else:
            with st.container(border=True):
                st.markdown(
                    "### 🪄 AI Microstructure Diagnostics & Recommended Parameters "
                    f"({_autotune_proposal.get('modality', 'Micrograph')})"
                )
                st.caption(
                    f"Measured on `{image_name}` at **"
                    f"{float(_autotune_proposal.get('calibration_um_per_px', pixel_to_um_ratio)):.4f} µm/px**. "
                    "Review the physical evidence below before applying — no sidebar "
                    "parameter has been changed yet."
                )

                _proposal_metrics = _autotune_proposal.get("summary_metrics", {})
                if _proposal_metrics:
                    _badge_cols = st.columns(len(_proposal_metrics))
                    for _badge_col, (_metric_label, _metric_value) in zip(
                        _badge_cols, _proposal_metrics.items()
                    ):
                        _badge_col.metric(_metric_label, _metric_value)

                st.markdown("**📊 Parameter Comparison & Scientific Rationale**")
                _proposal_rows = _autotune_proposal.get("comparison_rows", [])
                if _proposal_rows:
                    st.dataframe(
                        pd.DataFrame(_proposal_rows),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Parameter": st.column_config.TextColumn(
                                "Parameter", width="medium"
                            ),
                            "Current": st.column_config.TextColumn("Current", width="small"),
                            "Recommended": st.column_config.TextColumn(
                                "Recommended", width="small"
                            ),
                            "Why (Educational Rationale)": st.column_config.TextColumn(
                                "Why (Educational Rationale)", width="large"
                            ),
                        },
                    )
                else:
                    st.info("No parameter changes are required for this micrograph.")

                _apply_col, _dismiss_col = st.columns(2)
                if _apply_col.button(
                    "✅ Apply Recommended Parameters",
                    type="primary",
                    use_container_width=True,
                    key="autotune_apply_btn",
                ):
                    st.session_state["_previous_parameter_snapshot"] = collect_parameter_snapshot(
                        "Light" if is_light_mode else "SEM",
                        _live_parameter_values,
                    )
                    st.session_state["_pending_parameter_updates"] = dict(
                        _autotune_proposal.get("recommended_widget_state", {})
                    )
                    st.session_state.pop("autotune_proposal", None)
                    st.toast("✅ Optimal parameters applied!")
                    st.rerun()
                if _dismiss_col.button(
                    "✖️ Dismiss (Keep Current Settings)",
                    use_container_width=True,
                    key="autotune_dismiss_btn",
                ):
                    st.session_state.pop("autotune_proposal", None)
                    st.rerun()

    # Navigation View Bar
    tab_options = [
        "📊 Interactive Dashboard",
        "🖌️ Manual Correction",
        "🔬 Metrics",
        "🛡️ Interfacial Analytics",
        "🍇 Matrix Coalescence",
        "💾 Reproducibility & Export",
    ]
    if is_light_mode:
        # Single-phase emulsion light micrographs contain no air bubbles.
        tab_options = [t for t in tab_options if t != "🛡️ Interfacial Analytics"]
    # Streamlit 1.39 has no `st.segmented_control` (added in 1.40), so the tab bar
    # is a horizontal radio group - same look, same "always one tab active"
    # semantics, supported by every Streamlit release from 1.10 onwards.
    _default_tab = "📊 Interactive Dashboard"
    active_tab = st.radio(
        "Navigation",
        options=tab_options,
        index=tab_options.index(_default_tab) if _default_tab in tab_options else 0,
        horizontal=True,
        label_visibility="collapsed",
    )
    if not active_tab:
        active_tab = _default_tab

    # Active Smart Box SAM backbone (persisted across tabs for the session manifest).
    _sam_backbone_state = str(st.session_state.get("sam_backbone_select", "vit_b"))
    active_sam_type = _sam_backbone_state.split(" ")[0].strip() or "vit_b"
    if active_sam_type not in ("vit_b", "vit_l", "vit_h"):
        active_sam_type = "vit_b"

    # ---------------- TAB 1: DASHBOARD (COLUMNS) ----------------
    if active_tab == "📊 Interactive Dashboard":
        col_img, col_df = st.columns([1.15, 1.0])

        # Column 1: Original Image & Segmentation Overlay
        with col_img:
            st.subheader("1. Micrograph & Overlay")
            view_mode = st.radio(
                "Display Mode",
                ["Segmentation Overlay", "Raw Micrograph", "📏 Interactive Ruler"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if "Ruler" in view_mode:
                # Interactive HTML5 caliper rendered inside the same column as
                # the two standard views, so any bubble, pore or fat globule can
                # be measured directly on the dashboard and the suggested
                # Min/Max size pasted into Section 4 of the sidebar.
                ruler_base_mode = st.radio(
                    "Ruler Base Image",
                    ["Raw Micrograph", "Segmentation Overlay"],
                    horizontal=True,
                    key="dashboard_ruler_base_mode",
                )
                ruler_base_img = (
                    base_display_img
                    if ruler_base_mode == "Raw Micrograph"
                    else overlay
                )
                try:
                    (
                        ruler_data_url,
                        ruler_bitmap_w,
                        ruler_bitmap_h,
                    ) = encode_image_data_url(ruler_base_img, max_width=1600)
                    ruler_native_h, ruler_native_w = ruler_base_img.shape[:2]
                    # The component resizes its own iframe from the inside
                    # (streamlit:setFrameHeight); this is only the initial
                    # fallback height for a typical wide-layout column.
                    _ruler_est_w = max(320, min(int(ruler_bitmap_w), 800))
                    _ruler_est_canvas_h = int(
                        round(
                            _ruler_est_w
                            * ruler_bitmap_h
                            / float(max(1, ruler_bitmap_w))
                        )
                    )
                    components.html(
                        build_visual_ruler_html(
                            image_src=ruler_data_url,
                            native_w=int(ruler_native_w),
                            native_h=int(ruler_native_h),
                            bitmap_w=int(ruler_bitmap_w),
                            bitmap_h=int(ruler_bitmap_h),
                            um_per_px=float(pixel_to_um_ratio),
                        ),
                        height=int(_ruler_est_canvas_h + 250),
                        scrolling=False,
                    )
                    st.caption(
                        f"Drag a line across any bubble, pore or fat globule. Calibration: "
                        f"**{pixels_per_um:.2f} px/µm** ({pixel_to_um_ratio:.4f} µm/px) — copy the "
                        "suggested Min/Max size into **Section 4** of the sidebar. Measurements "
                        "reset whenever the app re-runs (any sidebar change)."
                    )
                except Exception as ruler_ex:
                    st.warning(f"Interactive ruler unavailable: {ruler_ex}")
            elif "Segmentation" in view_mode:
                st.image(
                    overlay,
                    caption=f"{image_name} — Labeled Objects: {total_objects}"
                    + (f" ({effective_footer}px footer excluded)" if effective_footer > 0 else ""),
                    **_FULL_WIDTH_IMAGE_KWARGS,
                )
                # Export segmented overlay button(s) directly beneath
                buf_overlay = io.BytesIO()
                Image.fromarray(overlay).save(buf_overlay, format="PNG", compress_level=1)
                if effective_footer > 0:
                    # Publication workflow: full micrograph (dimmed databar) next to a
                    # footer-free crop, which keeps a burned-in scale bar intact.
                    dl_c1, dl_c2 = st.columns(2)
                    with dl_c1:
                        st.download_button(
                            label="📥 Download Full Overlay (PNG)",
                            data=buf_overlay.getvalue(),
                            file_name=f"segmented_overlay_{clean_stem}.png",
                            mime="image/png",
                            use_container_width=True,
                            help=(
                                "Complete micrograph including the dimmed SEM databar "
                                f"({effective_footer}px) for traceability."
                            ),
                        )
                    with dl_c2:
                        buf_overlay_crop = io.BytesIO()
                        Image.fromarray(overlay[:-effective_footer, :]).save(
                            buf_overlay_crop, format="PNG", compress_level=1
                        )
                        st.download_button(
                            label="✂️ Download Cropped Overlay — No Footer (PNG)",
                            data=buf_overlay_crop.getvalue(),
                            file_name=f"segmented_overlay_cropped_{clean_stem}.png",
                            mime="image/png",
                            use_container_width=True,
                            help=(
                                "Publication-ready micrograph with the bottom SEM databar "
                                "cropped out (preserves burned-in scale bar if enabled)."
                            ),
                        )
                else:
                    st.download_button(
                        label="📥 Download Segmented Overlay (PNG)",
                        data=buf_overlay.getvalue(),
                        file_name=f"segmented_overlay_{clean_stem}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
            else:
                st.image(
                    base_display_img,
                    caption=f"{image_name} (Full SEM Micrograph"
                    + (f", {effective_footer}px Footer Dimmed)" if effective_footer > 0 else ")"),
                    **_FULL_WIDTH_IMAGE_KWARGS,
                )
                # Export raw micrograph (with the burned-in scale bar when enabled)
                buf_raw = io.BytesIO()
                Image.fromarray(base_display_img).save(buf_raw, format="PNG", compress_level=1)
                if effective_footer > 0:
                    dl_r1, dl_r2 = st.columns(2)
                    with dl_r1:
                        st.download_button(
                            label="📥 Download Full Raw Micrograph (PNG)",
                            data=buf_raw.getvalue(),
                            file_name=f"raw_micrograph_{clean_stem}.png",
                            mime="image/png",
                            use_container_width=True,
                            help=(
                                "Complete micrograph including the dimmed SEM databar "
                                f"({effective_footer}px) for traceability."
                            ),
                        )
                    with dl_r2:
                        buf_raw_crop = io.BytesIO()
                        Image.fromarray(base_display_img[:-effective_footer, :]).save(
                            buf_raw_crop, format="PNG", compress_level=1
                        )
                        st.download_button(
                            label="✂️ Download Cropped Raw — No Footer (PNG)",
                            data=buf_raw_crop.getvalue(),
                            file_name=f"raw_micrograph_cropped_{clean_stem}.png",
                            mime="image/png",
                            use_container_width=True,
                            help=(
                                "Publication-ready raw micrograph with the bottom SEM "
                                "databar cropped out (preserves burned-in scale bar if enabled)."
                            ),
                        )
                else:
                    st.download_button(
                        label="📥 Download Raw Micrograph (PNG)",
                        data=buf_raw.getvalue(),
                        file_name=f"raw_micrograph_{clean_stem}.png",
                        mime="image/png",
                        use_container_width=True,
                    )

        # Column 2: Filtered Metrics Table & Interactive Exclusion
        with col_df:
            st.subheader("2. Phase Classifications")
            editor_rev = int(st.session_state.get("particle_editor_rev", 0))
            editor_key = f"particle_data_editor_{editor_rev}"
            if classified_df.empty:
                st.warning("No particles passed the circularity and segmentation filter.")
                if st.session_state.get("interactive_excluded_ids"):
                    if st.button("Reset Interactive Deletions", use_container_width=True):
                        st.session_state["interactive_excluded_ids"] = set()
                        editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
                        for k in editor_keys:
                            del st.session_state[k]
                        st.rerun()
            else:
                editor_df = classified_df[[
                    "id", "phase_class", "diameter_um", "area_um2", "circularity", "mean_intensity"
                ]].copy()
                editor_df.insert(0, "Excluded", False)

                # Single-phase modes only expose their physically possible phases
                if is_emulsion_ui:
                    phase_class_options = ["fat_particles", "Unclassified"]
                elif sample_type == "Foam":
                    phase_class_options = ["air_bubbles", "Unclassified"]
                else:
                    phase_class_options = ["air_bubbles", "fat_particles", "adsorbed_fat", "Unclassified"]

                edited_table = st.data_editor(
                    editor_df,
                    column_config={
                        "Excluded": st.column_config.CheckboxColumn(
                            "Excluded",
                            help="Check to mark this particle for deletion as an artifact.",
                            default=False,
                        ),
                        "id": st.column_config.NumberColumn("ID", format="%d", disabled=True),
                        "phase_class": st.column_config.SelectboxColumn(
                            "Phase Class",
                            help="Manually reclassify this object (options depend on the selected Sample Type).",
                            options=phase_class_options,
                            required=True,
                        ),
                        "diameter_um": st.column_config.NumberColumn("Diameter (µm)", format="%.2f", disabled=True),
                        "area_um2": st.column_config.NumberColumn("Area (µm²)", format="%.2f", disabled=True),
                        "circularity": st.column_config.NumberColumn("Circularity", format="%.2f", disabled=True),
                        "mean_intensity": st.column_config.NumberColumn("Intensity", format="%.1f", disabled=True),
                    },
                    disabled=["id", "diameter_um", "area_um2", "circularity", "mean_intensity"],
                    hide_index=True,
                    height=320,
                    use_container_width=True,
                    key=editor_key,
                )

                if unclass_count > 0:
                    unclass_mask = classified_df["phase_class"].astype(str).str.lower().eq("unclassified")
                    unclass_df = classified_df[unclass_mask]

                    if "in_bubble" in unclass_df.columns:
                        unclass_in_bub = unclass_df[unclass_df["in_bubble"] == True]
                        unclass_out_bub = unclass_df[unclass_df["in_bubble"] == False]
                    else:
                        in_bub_mask_unclass = unclass_df.apply(_centroid_in_bubble, axis=1)
                        unclass_in_bub = unclass_df[in_bub_mask_unclass]
                        unclass_out_bub = unclass_df[~in_bub_mask_unclass]

                    n_unclass_in_bub = len(unclass_in_bub)
                    n_unclass_out_bub = len(unclass_out_bub)

                    st.markdown(f"**⚡ Smart Group Bulk Actions for {unclass_count} Unclassified Particles:**")
                    if is_emulsion_ui:
                        # Single-phase Emulsion: no air/adsorbed phases exist
                        e_col1, e_col2 = st.columns(2)
                        with e_col1:
                            if st.button(
                                f"🟧 Assign All Unclassified ({unclass_count}) → Fat Droplets",
                                use_container_width=True,
                                help="Reclassifies every Unclassified object as a fat droplet.",
                            ):
                                if "manual_phase_overrides" not in st.session_state:
                                    st.session_state["manual_phase_overrides"] = {}
                                for u_id in unclass_df["id"]:
                                    st.session_state["manual_phase_overrides"][int(u_id)] = "fat_particles"
                                editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
                                for k in editor_keys:
                                    del st.session_state[k]
                                st.rerun()
                        with e_col2:
                            if st.button(
                                f"🗑️ Delete All Unclassified ({unclass_count})",
                                use_container_width=True,
                                help="Removes all Unclassified particles as artifacts.",
                            ):
                                unclass_ids = set(unclass_df["id"].tolist())
                                if "interactive_excluded_ids" not in st.session_state:
                                    st.session_state["interactive_excluded_ids"] = set()
                                st.session_state["interactive_excluded_ids"].update(unclass_ids)
                                editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
                                for k in editor_keys:
                                    del st.session_state[k]
                                st.rerun()
                    else:
                        # Row 1: Smart Spatial 1-Click Actions (4 buttons)
                        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                        with b_col1:
                            if st.button(
                                f"✨ Smart Auto-Split All ({unclass_count})",
                                use_container_width=True,
                                help="Reclassifies all Unclassified inside bubbles to adsorbed_fat and outside bubbles to fat_particles.",
                            ):
                                if "manual_phase_overrides" not in st.session_state:
                                    st.session_state["manual_phase_overrides"] = {}
                                for u_id in unclass_in_bub["id"]:
                                    st.session_state["manual_phase_overrides"][int(u_id)] = "adsorbed_fat"
                                for u_id in unclass_out_bub["id"]:
                                    st.session_state["manual_phase_overrides"][int(u_id)] = "fat_particles"
                                editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
                                for k in editor_keys:
                                    del st.session_state[k]
                                st.rerun()

                        with b_col2:
                            if st.button(
                                f"🟨 In Bubbles ({n_unclass_in_bub}) → Adsorbed Fat",
                                use_container_width=True,
                                help="Reclassifies only Unclassified particles whose centroid is inside an air bubble to adsorbed_fat.",
                            ):
                                if "manual_phase_overrides" not in st.session_state:
                                    st.session_state["manual_phase_overrides"] = {}
                                for u_id in unclass_in_bub["id"]:
                                    st.session_state["manual_phase_overrides"][int(u_id)] = "adsorbed_fat"
                                editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
                                for k in editor_keys:
                                    del st.session_state[k]
                                st.rerun()

                        with b_col3:
                            if st.button(
                                f"🟧 Outside ({n_unclass_out_bub}) → Matrix Fat",
                                use_container_width=True,
                                help="Reclassifies only Unclassified particles outside air bubble regions to fat_particles.",
                            ):
                                if "manual_phase_overrides" not in st.session_state:
                                    st.session_state["manual_phase_overrides"] = {}
                                for u_id in unclass_out_bub["id"]:
                                    st.session_state["manual_phase_overrides"][int(u_id)] = "fat_particles"
                                editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
                                for k in editor_keys:
                                    del st.session_state[k]
                                st.rerun()

                        with b_col4:
                            if st.button(
                                "🗑️ Delete All Unclassified",
                                use_container_width=True,
                                help="Removes all Unclassified particles as artifacts.",
                            ):
                                unclass_ids = set(unclass_df["id"].tolist())
                                if "interactive_excluded_ids" not in st.session_state:
                                    st.session_state["interactive_excluded_ids"] = set()
                                st.session_state["interactive_excluded_ids"].update(unclass_ids)
                                editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
                                for k in editor_keys:
                                    del st.session_state[k]
                                st.rerun()

                    # Row 2: Grouping Unclassified by Size Range in a compact expander
                    with st.expander("📏 Group & Assign Unclassified by Size Range (µm)"):
                        col_g1, col_g2, col_g3, col_g4 = st.columns([1, 1, 1.2, 1.2])
                        with col_g1:
                            def_min_um = float(unclass_df["diameter_um"].min()) if not unclass_df.empty else 0.0
                            grp_min_um = float(st.number_input(
                                "Min Diam (µm)",
                                min_value=0.0,
                                max_value=5000.0,
                                value=round(def_min_um, 2),
                                step=0.1,
                                key="grp_unclass_min",
                            ))
                        with col_g2:
                            def_max_um = float(unclass_df["diameter_um"].max()) if not unclass_df.empty else 10.0
                            grp_max_um = float(st.number_input(
                                "Max Diam (µm)",
                                min_value=0.0,
                                max_value=5000.0,
                                value=round(def_max_um, 2),
                                step=0.1,
                                key="grp_unclass_max",
                            ))
                        with col_g3:
                            grp_target = st.selectbox(
                                "Assign to",
                                options=(
                                    ["fat_particles", "Delete"]
                                    if is_emulsion_ui
                                    else ["fat_particles", "adsorbed_fat", "air_bubbles", "Delete"]
                                ),
                                index=0,
                                key="grp_unclass_target",
                            )
                        with col_g4:
                            st.write("")
                            st.write("")
                            if st.button("Apply to Unclassified in Range", use_container_width=True, key="btn_apply_unclass_range"):
                                in_range_ids = unclass_df.loc[
                                    (unclass_df["diameter_um"] >= grp_min_um) & (unclass_df["diameter_um"] <= grp_max_um),
                                    "id",
                                ].tolist()
                                if in_range_ids:
                                    if grp_target == "Delete":
                                        if "interactive_excluded_ids" not in st.session_state:
                                            st.session_state["interactive_excluded_ids"] = set()
                                        st.session_state["interactive_excluded_ids"].update(set(in_range_ids))
                                    else:
                                        if "manual_phase_overrides" not in st.session_state:
                                            st.session_state["manual_phase_overrides"] = {}
                                        for t_id in in_range_ids:
                                            st.session_state["manual_phase_overrides"][int(t_id)] = grp_target
                                    editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
                                    for k in editor_keys:
                                        del st.session_state[k]
                                    st.rerun()

                col_btn1, col_btn2 = st.columns([1.2, 1.0])
                with col_btn1:
                    if st.button("Apply Changes", type="primary", use_container_width=True, help="Apply particle deletions and phase reclassifications"):
                        marked_ids = set(edited_table.loc[edited_table["Excluded"] == True, "id"].tolist())
                        if marked_ids:
                            st.session_state["interactive_excluded_ids"].update(marked_ids)

                        for _, row in edited_table.iterrows():
                            p_id = int(row["id"])
                            if p_id in marked_ids:
                                continue
                            new_phase = str(row["phase_class"])
                            orig_match = classified_df.loc[classified_df["id"] == p_id, "phase_class"].values
                            if len(orig_match) > 0 and str(orig_match[0]) != new_phase:
                                st.session_state["manual_phase_overrides"][p_id] = new_phase

                        editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
                        for k in editor_keys:
                            del st.session_state[k]
                        st.rerun()

                with col_btn2:
                    has_modifications = bool(st.session_state.get("interactive_excluded_ids")) or bool(st.session_state.get("manual_phase_overrides"))
                    if has_modifications:
                        if st.button("Reset Modifications", use_container_width=True, help="Reset all manual deletions and reclassifications"):
                            st.session_state["interactive_excluded_ids"] = set()
                            st.session_state["manual_phase_overrides"] = {}
                            editor_keys = [k for k in list(st.session_state.keys()) if str(k).startswith("particle_data_editor")]
                            for k in editor_keys:
                                del st.session_state[k]
                            st.rerun()

                if st.session_state.get("interactive_excluded_ids"):
                    st.caption(
                        f"🗑️ Excluded {len(st.session_state['interactive_excluded_ids'])} particles: "
                        f"`{sorted(list(st.session_state['interactive_excluded_ids']))}`"
                    )
                if st.session_state.get("manual_phase_overrides"):
                    st.caption(
                        f"✏️ Reclassified {len(st.session_state['manual_phase_overrides'])} particles: "
                        f"`{dict(list(st.session_state['manual_phase_overrides'].items())[:4])}`"
                    )

        # Full-width Size Distribution section beneath image + table
        st.subheader("3. Size Distribution")

        # Fixed physical bar width (∆µm): every bar covers the same diameter interval, so
        # bar heights (% of that phase) stay directly comparable between micrographs with
        # very different particle counts (e.g. n = 3176 fat particles).
        if sample_type == "Emulsion":
            # Single-phase: only the fat histogram exists — no unused air column
            fat_bin_width = st.select_slider(
                "🟧 Fat Histogram Step (∆µm)",
                options=[
                    0.2, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
                ],
                value=0.5,
                key=f"fat_bin_width_{clean_stem}",
                help=(
                    "Width of each fat-particle histogram bar. 0.5 µm resolves fine emulsion "
                    "globules while keeping every bar comparable (all bars sum to 100%)."
                ),
            )
            air_bin_width = 10.0
        else:
            col_bin_fat, col_bin_air = st.columns(2)
            with col_bin_fat:
                fat_bin_width = st.select_slider(
                    "🟧 Fat Histogram Step (∆µm)",
                    options=[
                        0.2, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                        5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
                    ],
                    value=0.5,
                    key=f"fat_bin_width_{clean_stem}",
                    help=(
                        "Width of each fat-particle histogram bar. 0.5 µm resolves fine emulsion "
                        "globules while keeping every bar comparable (all bars sum to 100%)."
                    ),
                )
            with col_bin_air:
                air_bin_width = st.select_slider(
                    "🟦 Air Histogram Step (∆µm)",
                    options=[2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
                    value=10.0,
                    key=f"air_bin_width_{clean_stem}",
                    help=(
                        "Width of each air-bubble histogram bar. Bubble diameters span a much "
                        "wider range, so a coarser step (10 µm by default) avoids hundreds of bins."
                    ),
                )

        def _render_phase_distribution(phase_key: str, title: str, png_name: str, csv_name: str, dl_prefix: str):
            phase_csv = classified_df[classified_df["phase_class"] == phase_key] if not classified_df.empty else classified_df
            bin_width = float(fat_bin_width if phase_key == "fat_particles" else air_bin_width)
            fig_phase, _ = plot_single_phase_distribution(
                classified_df,
                phase_class=phase_key,
                title=title,
                color=PALETTE_COLORS.get(phase_key),
                bin_width=bin_width,
            )
            st.pyplot(fig_phase, use_container_width=True)
            buf_png = io.BytesIO()
            fig_phase.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                label=f"📥 Download {dl_prefix} Plot (PNG)",
                data=buf_png.getvalue(),
                file_name=png_name,
                mime="image/png",
                use_container_width=True,
                key=f"dl_png_{phase_key}_{clean_stem}",
            )
            plt.close(fig_phase)
            if phase_csv is not None and not phase_csv.empty:
                st.download_button(
                    label=f"📥 Download {dl_prefix} (CSV)",
                    data=phase_csv.to_csv(index=False).encode("utf-8"),
                    file_name=csv_name,
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_csv_{phase_key}_{clean_stem}",
                )

            # Exact binned percentage table (Σ = 100%) mirroring the rendered bars
            binned_table = compute_phase_histogram_table(
                classified_df,
                phase_class=phase_key,
                bin_width=bin_width,
            )
            with st.expander(
                f"📊 View Exact Binned Percentage Table — {dl_prefix} (Σ = 100%)",
                expanded=False,
            ):
                if binned_table.empty:
                    st.info(
                        f"No {dl_prefix.lower()} objects are available for a binned percentage table."
                    )
                else:
                    st.caption(
                        f"∆ = {bin_width:g} µm · "
                        f"n = {int(binned_table['Count (n)'].sum()):,} · "
                        f"Σ = 100% of the {dl_prefix.lower()} population · "
                        f"{len(binned_table)} intervals"
                    )
                    st.dataframe(
                        binned_table,
                        use_container_width=True,
                        hide_index=True,
                        height=260,
                    )
                    st.download_button(
                        label=f"📥 Download {dl_prefix} Binned Table (CSV)",
                        data=binned_table.to_csv(index=False).encode("utf-8"),
                        file_name=f"phasepix_{phase_key}_binned_table_{clean_stem}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_binned_{phase_key}_{clean_stem}",
                    )

        if sample_type == "Whipped Emulsion":
            col_plot_fat, col_plot_air = st.columns(2)
            with col_plot_fat:
                _render_phase_distribution(
                    "fat_particles",
                    "Fat Particles Size Distribution",
                    f"phasepix_fat_distribution_{clean_stem}.png",
                    f"phasepix_fat_particles_{clean_stem}.csv",
                    "Fat",
                )
            with col_plot_air:
                _render_phase_distribution(
                    "air_bubbles",
                    "Air Bubbles Size Distribution",
                    f"phasepix_air_distribution_{clean_stem}.png",
                    f"phasepix_air_bubbles_{clean_stem}.csv",
                    "Air",
                )
        elif sample_type == "Emulsion":
            _render_phase_distribution(
                "fat_particles",
                "Fat Particles Size Distribution",
                f"phasepix_fat_distribution_{clean_stem}.png",
                f"phasepix_fat_particles_{clean_stem}.csv",
                "Fat",
            )
        else:
            _render_phase_distribution(
                "air_bubbles",
                "Air Bubbles Size Distribution",
                f"phasepix_air_distribution_{clean_stem}.png",
                f"phasepix_air_bubbles_{clean_stem}.csv",
                "Air",
            )

    # ---------------- TAB 2: MANUAL CORRECTION ----------------
    elif active_tab == "🖌️ Manual Correction":
        st.subheader("🖌️ Manual Object Addition & Overlapping Pore Correction")
        st.markdown(
            "Draw missed features (e.g. large air bubbles or pores) directly on the micrograph. "
            "**Additive Workflow**: Drawn objects are permanently saved to your session dataset and rendered "
            "directly on the canvas background, so switching tabs will never erase your manual corrections."
        )

        manual_items_key = f"saved_manual_items_{image_name}"
        if manual_items_key not in st.session_state:
            st.session_state[manual_items_key] = []
        saved_items = st.session_state[manual_items_key]

        # Inject modern sticky CSS for toolbar
        st.markdown(
            """
<style>
div[data-testid="stVerticalBlock"] > div:has(div.sticky-toolbar-marker) {
    position: sticky;
    top: 2.8rem;
    z-index: 999;
    background-color: rgba(255, 255, 255, 0.96);
    padding: 10px 14px;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    backdrop-filter: blur(6px);
}
@media (prefers-color-scheme: dark) {
    div[data-testid="stVerticalBlock"] > div:has(div.sticky-toolbar-marker) {
        background-color: rgba(26, 28, 36, 0.96);
        border: 1px solid #3a3f50;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
}
</style>
""",
            unsafe_allow_html=True,
        )

        # Micrograph PIL conversion
        if len(analysis_img.shape) == 2:
            manual_pil = Image.fromarray(analysis_img).convert("RGB")
        else:
            manual_pil = Image.fromarray(analysis_img)
        m_img_w, m_img_h = manual_pil.size

        # Sticky Floating Toolbar
        with st.container():
            st.markdown('<div class="sticky-toolbar-marker"></div>', unsafe_allow_html=True)

            # Row 1: Selection Method
            assist_mode = st.radio(
                "Selection Method",
                [
                    "📦 AI Smart Box (Draw Rectangle around bubble/particle — SAM AI fit)",
                    "📐 Exact Smooth Curve by Points (Click points — 100% exact pass-through)",
                    "🪄 1-Click Dark Pore Center (Click center — auto-detect dark pore)",
                ],
                index=0,
                horizontal=True,
                help=(
                    "📦 AI Smart Box: Draw a rectangle around a bubble/particle — SAM AI (with a Scale-Normalized Cellpose fallback) fits the true SEM boundary and converts it into a smooth contour.\n"
                    "📐 Exact Smooth Curve: Click points around the object — a periodic spline passes exactly through your clicks (no AI shift, no Fourier distortion).\n"
                    "🪄 1-Click Dark Pore Center: Click in the dark core of a pore — PhasePix auto-detects the crater boundary."
                ),
            )

            # Row 1b: SAM backbone selector (Smart Box only). Heavier backbones trade
            # loading time / unified memory for maximum boundary precision.
            active_sam_type = "vit_b"
            if assist_mode.startswith("📦 AI Smart Box"):
                sam_model_choice = st.selectbox(
                    "🧠 SAM Backbone (Smart Box AI)",
                    options=[
                        "vit_b — Base (375 MB, Instant)",
                        "vit_l — Large (1.25 GB, Balanced)",
                        "vit_h — Huge (2.56 GB, Max Precision for Complex Edges)",
                    ],
                    index=0,
                    key="sam_backbone_select",
                    help=(
                        "Select 'vit_h' (2.56 GB) to leverage your 48 GB RAM for maximum "
                        "boundary accuracy on faint, blurry, or crystallized droplets/pores. "
                        "vit_l is the balanced option; vit_b loads instantly."
                    ),
                )
                active_sam_type = sam_model_choice.split(" ")[0].strip()

            # Row 2: Target Phase Assignment (Col 1) and Action Buttons (Col 2)
            t_row2_col1, t_row2_col2 = st.columns([1.6, 1.2])
            with t_row2_col1:
                if is_emulsion_ui:
                    default_manual_phase = st.radio(
                        "Assign New Manual Objects As",
                        options=["fat_particles", "Auto (by Min/Max Size)"],
                        index=0,
                        horizontal=True,
                        help=(
                            "Single-phase emulsion: manual objects are fat droplets, or classified "
                            "automatically by the Min/Max Fat size range."
                        ),
                    )
                else:
                    default_manual_phase = st.radio(
                        "Assign New Manual Objects As",
                        options=["air_bubbles", "fat_particles", "adsorbed_fat", "Auto (by Min/Max Size)"],
                        index=0,
                        horizontal=True,
                        help=(
                            "By default, manually added pores/bubbles are locked as air_bubbles regardless of their size, "
                            "so small or huge pores are never misclassified as fat or Unclassified."
                        ),
                    )
            with t_row2_col2:
                if assist_mode.startswith("📦 AI Smart Box"):
                    btn_add_label = "➕ Detect & Add from Box(es)"
                elif assist_mode.startswith("🪄 1-Click"):
                    btn_add_label = "➕ Add Droplet" if is_emulsion_ui else "➕ Add Bubble"
                else:
                    btn_add_label = "➕ Add Manual Droplet" if is_emulsion_ui else "➕ Add Manual Bubble"

                col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1.0, 1.0])
                with col_btn1:
                    add_drawn_btn = st.button(btn_add_label, type="primary", use_container_width=True)
                with col_btn2:
                    undo_last_btn = st.button(
                        "↩️ Undo Last Added",
                        type="secondary",
                        use_container_width=True,
                        disabled=not saved_items,
                        help="Removes the most recently added manual object and clears its phase override.",
                    )
                with col_btn3:
                    clear_canvas_btn = st.button("🔄 Clear Unsaved Drawing", type="secondary", use_container_width=True)

        # Fixed 100% Native 1:1 Canvas Size to prevent coordinate shifting on clicks
        canvas_scale_ratio = 1.0
        target_w = m_img_w
        target_h = m_img_h
        base_canvas_img = manual_pil.copy()

        # Pre-render existing saved manual objects and automated air bubbles onto canvas background
        canvas_bg_np = np.array(base_canvas_img)

        # 1. Thin teal outline around already detected automated air bubbles
        auto_air_binary = np.zeros(analysis_img.shape[:2], dtype=np.uint8)
        if filtered_air_masks is not None and isinstance(filtered_air_masks, np.ndarray) and filtered_air_masks.size > 0:
            auto_air_binary = (filtered_air_masks > 0).astype(np.uint8)
        elif not classified_df.empty and filtered_masks is not None and isinstance(filtered_masks, np.ndarray):
            air_ids = classified_df[
                (classified_df["phase_class"] == "air_bubbles") & (classified_df["id"] < 10000)
            ]["id"].tolist()
            if air_ids:
                # Fast LUT gather instead of an np.isin() sort over the whole mask
                auto_air_binary = (
                    _lut_label_filter(filtered_masks, air_ids, keep=True) > 0
                ).astype(np.uint8)

        if np.any(auto_air_binary > 0):
            auto_cnts, _ = cv2.findContours(auto_air_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for acnt in auto_cnts:
                if len(acnt) >= 8:
                    try:
                        from analyzer import _smooth_contour_fourier
                        smoothed_acnt = _smooth_contour_fourier(acnt.reshape(-1, 2).astype(np.float32), n_harmonics=8, num_points=160)
                        pts_scaled = (smoothed_acnt * canvas_scale_ratio).round().astype(np.int32)
                        cv2.polylines(canvas_bg_np, [pts_scaled], isClosed=True, color=(0, 230, 180), thickness=1, lineType=cv2.LINE_AA)
                    except Exception:
                        pts_scaled = (acnt.reshape(-1, 2).astype(np.float32) * canvas_scale_ratio).round().astype(np.int32)
                        cv2.polylines(canvas_bg_np, [pts_scaled], isClosed=True, color=(0, 230, 180), thickness=1, lineType=cv2.LINE_AA)
                elif len(acnt) >= 3:
                    pts_scaled = (acnt.reshape(-1, 2).astype(np.float32) * canvas_scale_ratio).round().astype(np.int32)
                    cv2.polylines(canvas_bg_np, [pts_scaled], isClosed=True, color=(0, 230, 180), thickness=1, lineType=cv2.LINE_AA)

        # 2. Pre-render saved manual objects (color-coded by phase class with border and ID)
        manual_phase_lookup = {}
        if not classified_df.empty and "phase_class" in classified_df.columns:
            for _, r in classified_df[classified_df["id"] >= 10000].iterrows():
                manual_phase_lookup[int(r["id"])] = str(r["phase_class"])

        if saved_items:
            for item in saved_items:
                poly = item.get("polygon", {})
                pts = poly.get("points")
                if pts is not None and len(pts) >= 3:
                    item_id = int(item.get("id", 10000))
                    item_phase = manual_phase_lookup.get(
                        item_id,
                        st.session_state.get("manual_phase_overrides", {}).get(item_id, "air_bubbles"),
                    )

                    # Color coding: Sky blue for air_bubbles, Orange for fat_particles,
                    # Gold for adsorbed_fat, Green for Unclassified
                    if str(item_phase).lower() == "fat_particles":
                        obj_color = (255, 165, 0)
                    elif str(item_phase).lower() == "adsorbed_fat":
                        obj_color = (255, 215, 0)
                    elif str(item_phase).lower() == "unclassified":
                        obj_color = (50, 205, 50)
                    else:
                        obj_color = (0, 191, 255)

                    pts_scaled = (np.asarray(pts, dtype=np.float32) * canvas_scale_ratio).round().astype(np.int32)
                    tint_layer = canvas_bg_np.copy()
                    cv2.fillPoly(tint_layer, [pts_scaled], obj_color)
                    cv2.addWeighted(tint_layer, 0.35, canvas_bg_np, 0.65, 0, dst=canvas_bg_np)
                    cv2.polylines(canvas_bg_np, [pts_scaled], isClosed=True, color=obj_color, thickness=2, lineType=cv2.LINE_AA)
                    
                    m = item.get("metrics", {})
                    cx_raw = m.get("centroid_x", float(pts_scaled[:, 0].mean()))
                    cy_raw = m.get("centroid_y", float(pts_scaled[:, 1].mean()))
                    cx_canv = int(round(cx_raw * canvas_scale_ratio))
                    cy_canv = int(round(cy_raw * canvas_scale_ratio))
                    lbl = str(item_id)
                    cv2.putText(canvas_bg_np, lbl, (cx_canv - 12, cy_canv + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, lineType=cv2.LINE_AA)
                    cv2.putText(canvas_bg_np, lbl, (cx_canv - 12, cy_canv + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        canvas_bg = Image.fromarray(canvas_bg_np)

        # Informative captions and full-size tips
        if is_emulsion_ui:
            st.caption(
                "🟢 **Thin Teal Outline** = Already detected by AI &nbsp;|&nbsp; "
                "🟠/🟢 **Solid Fill** = Manual Additions (Orange: Fat Droplets, Green: Unclassified) &nbsp;|&nbsp; "
                "💡 **Pro-Tip:** Collapse the left sidebar (**<<** at top-left) to view the micrograph in full native 100% resolution. The floating toolbar stays pinned to the top as you scroll!"
            )
        else:
            st.caption(
                "🟢 **Thin Teal Outline** = Already detected by AI &nbsp;|&nbsp; "
                "🔵/🟠/🟡/🟢 **Solid Fill** = Manual Additions (Blue: Air, Orange: Fat, Gold: Adsorbed Fat, Green: Unclassified) &nbsp;|&nbsp; "
                "💡 **Pro-Tip:** Collapse the left sidebar (**<<** at top-left) to view the micrograph in full native 100% resolution. The floating toolbar stays pinned to the top as you scroll!"
            )

        if assist_mode.startswith("📦 AI Smart Box"):
            st.caption("💡 **Tip:** Drag one or more rectangles around bubbles/particles, then click **Detect & Add from Box(es)**. SAM AI (or the Scale-Normalized Cellpose fallback) fits the real SEM boundary.")
        elif assist_mode.startswith("🪄 1-Click"):
            st.caption(
                f"💡 **Tip:** Click once in the dark center of each missed {'droplet' if is_emulsion_ui else 'bubble/pore'}, "
                f"then click **{'Add Droplet' if is_emulsion_ui else 'Add Bubble'}**."
            )
        else:
            st.caption("💡 **Tip:** Click points around the particle. PhasePix will smoothly interpolate exactly through your points (0.0 px shift).")

        # Full-width Canvas
        if default_manual_phase == "fat_particles":
            canvas_fill_color = "rgba(255, 165, 0, 0.4)"
            canvas_stroke_color = "#FFA500"
        elif default_manual_phase == "adsorbed_fat":
            canvas_fill_color = "rgba(255, 215, 0, 0.4)"
            canvas_stroke_color = "#FFD700"
        elif default_manual_phase == "Unclassified":
            canvas_fill_color = "rgba(50, 205, 50, 0.4)"
            canvas_stroke_color = "#32CD32"
        else:
            canvas_fill_color = "rgba(0, 191, 255, 0.4)"
            canvas_stroke_color = "#00BFFF"

        # Drawing mode: rectangles for Smart Box (SAM), clicks for point-based modes
        canvas_drawing_mode = "rect" if assist_mode.startswith("📦 AI Smart Box") else "point"

        curr_ver = st.session_state.get("manual_canvas_ver", 0)
        canvas_result = safe_st_canvas(
            fill_color=canvas_fill_color,
            stroke_width=2,
            stroke_color=canvas_stroke_color,
            background_image=canvas_bg,
            update_streamlit=True,
            height=target_h,
            width=target_w,
            drawing_mode=canvas_drawing_mode,
            point_display_radius=4,
            display_toolbar=False,
            key=f"manual_correction_canvas_{image_name}_{curr_ver}",
        )

        if clear_canvas_btn:
            st.session_state["manual_canvas_ver"] = curr_ver + 1
            st.rerun()

        if undo_last_btn:
            live_items = st.session_state.get(manual_items_key, [])
            if live_items:
                removed_item = live_items.pop()
                try:
                    removed_id = int(removed_item.get("id"))
                except (TypeError, ValueError):
                    removed_id = None
                if removed_id is not None:
                    st.session_state.get("manual_phase_overrides", {}).pop(removed_id, None)
                st.session_state[manual_items_key] = live_items
                st.session_state["manual_canvas_ver"] = curr_ver + 1
                st.rerun()

        if add_drawn_btn:
            has_raster = (
                canvas_result is not None
                and canvas_result.image_data is not None
                and (
                    np.any(canvas_result.image_data[:, :, 3] > 0)
                    if canvas_result.image_data.ndim == 3 and canvas_result.image_data.shape[2] >= 4
                    else np.any(canvas_result.image_data > 0)
                )
            )
            has_json = (
                canvas_result is not None
                and canvas_result.json_data is not None
                and bool(canvas_result.json_data.get("objects"))
            )

            if has_raster or has_json:
                existing_ids = [item["id"] for item in saved_items]
                start_new_id = max(existing_ids, default=10000) + 1

                cp_model = None
                if assist_mode.startswith("📦 AI Smart Box") or assist_mode.startswith("🪄 1-Click"):
                    try:
                        cp_model = get_cellpose_model(
                            model_type=model_type,
                            custom_model_path=custom_model_path,
                        )
                    except Exception as e:
                        st.warning(f"Could not load Cellpose model for AI assistance: {e}")
                        cp_model = None

                parsed_metrics, parsed_polys = parse_canvas_manual_objects(
                    canvas_json=canvas_result.json_data if has_json else None,
                    scale_ratio=canvas_scale_ratio,
                    original_image=analysis_img,
                    pixel_to_um_ratio=pixel_to_um_ratio,
                    start_id=start_new_id,
                    image_data=canvas_result.image_data if has_raster else None,
                    # Exact Smooth Curve mode -> clean periodic spline through the clicks
                    assist_mode=(
                        "exact_manual"
                        if assist_mode.startswith("📐 Exact Smooth")
                        else assist_mode
                    ),
                    cellpose_model=cp_model,
                    max_air_size_um=max_air_size,
                    target_phase=default_manual_phase,
                    invert_brightfield=(
                        invert_brightfield if 'invert_brightfield' in locals() else False
                    ),
                    sam_model_type=active_sam_type,
                )

                if parsed_metrics:
                    for m, p in zip(parsed_metrics, parsed_polys):
                        if "area_px" not in m:
                            m["area_px"] = (
                                m["area_um2"] / (pixel_to_um_ratio ** 2)
                                if pixel_to_um_ratio > 0
                                else 0.0
                            )
                        st.session_state[manual_items_key].append({
                            "id": m["id"],
                            "metrics": m,
                            "polygon": p,
                            "assigned_class": (
                                default_manual_phase
                                if default_manual_phase != "Auto (by Min/Max Size)"
                                else "air_bubbles"
                            ),
                        })

                        if default_manual_phase != "Auto (by Min/Max Size)":
                            if "manual_phase_overrides" not in st.session_state:
                                st.session_state["manual_phase_overrides"] = {}
                            st.session_state["manual_phase_overrides"][m["id"]] = default_manual_phase

                    st.session_state["manual_canvas_ver"] = curr_ver + 1
                    st.success(f"Added {len(parsed_metrics)} object(s) to dataset!", icon="✅")
                    st.rerun()
                else:
                    if assist_mode.startswith("📦 AI Smart Box"):
                        st.warning("Could not extract an object from the drawn box(es). Draw a larger rectangle around a single bubble/particle and try again.", icon="⚠️")
                    elif assist_mode.startswith("🪄 1-Click"):
                        st.warning("Could not detect bubble boundaries from the clicked center(s). Try clicking closer to the center of the dark pore.", icon="⚠️")
                    else:
                        st.warning("Please click at least 2 points to form a closed manual shape.", icon="⚠️")
            else:
                if assist_mode.startswith("📦 AI Smart Box"):
                    st.info("Canvas is empty. Draw a rectangle around a bubble/particle, then click '➕ Detect & Add from Box(es)'.", icon="💡")
                elif assist_mode.startswith("🪄 1-Click"):
                    st.info(
                        f"Canvas is empty. Click in the center of one or more {'droplets' if is_emulsion_ui else 'dark bubbles'}, "
                        f"then click '➕ {'Add Droplet' if is_emulsion_ui else 'Add Bubble'}'.",
                        icon="💡",
                    )
                else:
                    st.info(
                        f"Canvas is empty. Click points on the image, then click "
                        f"'➕ {'Add Manual Droplet' if is_emulsion_ui else 'Add Manual Bubble'}'.",
                        icon="💡",
                    )

        # Two-column layout beneath the full-width canvas
        st.markdown("---")
        col_sum, col_prev = st.columns([1.0, 1.2])

        with col_sum:
            st.markdown("### Manual Objects Summary")
            current_saved_items = st.session_state.get(manual_items_key, [])
            st.metric("Total Manual Objects", f"{len(current_saved_items):,}")

            if current_saved_items:
                # Lookup current metrics and phase classifications from classified_df
                cf_lookup = {}
                if not classified_df.empty:
                    cf_manual = classified_df[classified_df["id"] >= 10000]
                    for _, r in cf_manual.iterrows():
                        cf_lookup[int(r["id"])] = {
                            "diameter_um": float(r["diameter_um"]),
                            "area_um2": float(r["area_um2"]),
                            "phase_class": str(r["phase_class"]),
                        }

                rows = []
                for it in current_saved_items:
                    obj_id = int(it["id"])
                    m = it.get("metrics", {})
                    diam = cf_lookup.get(obj_id, {}).get("diameter_um", float(m.get("diameter_um", 0.0)))
                    area = cf_lookup.get(obj_id, {}).get("area_um2", float(m.get("area_um2", 0.0)))
                    curr_phase = cf_lookup.get(obj_id, {}).get(
                        "phase_class",
                        st.session_state.get("manual_phase_overrides", {}).get(
                            obj_id, "fat_particles" if is_emulsion_ui else "air_bubbles"
                        ),
                    )
                    if str(curr_phase).lower() == "unclassified":
                        curr_phase = "Unclassified"
                    elif is_emulsion_ui:
                        # Single-phase Emulsion only allows fat droplets / Unclassified
                        if curr_phase not in ["fat_particles", "Unclassified"]:
                            curr_phase = "fat_particles"
                    elif curr_phase not in ["air_bubbles", "fat_particles", "adsorbed_fat", "Unclassified"]:
                        curr_phase = "air_bubbles"

                    rows.append({
                        "Delete": False,
                        "id": obj_id,
                        "diameter_um": diam,
                        "area_um2": area,
                        "phase_class": curr_phase,
                    })

                edit_df = pd.DataFrame(rows)

                edited_data = st.data_editor(
                    edit_df,
                    column_config={
                        "Delete": st.column_config.CheckboxColumn(
                            "Delete",
                            help="Check to delete this manual object",
                            default=False,
                        ),
                        "id": st.column_config.NumberColumn(
                            "ID",
                            disabled=True,
                            format="%d",
                        ),
                        "diameter_um": st.column_config.NumberColumn(
                            "Diam (µm)",
                            disabled=True,
                            format="%.2f",
                        ),
                        "area_um2": st.column_config.NumberColumn(
                            "Area (µm²)",
                            disabled=True,
                            format="%.2f",
                        ),
                        "phase_class": st.column_config.SelectboxColumn(
                            "Phase Class",
                            options=(
                                ["fat_particles", "Unclassified"]
                                if is_emulsion_ui
                                else ["air_bubbles", "fat_particles", "adsorbed_fat", "Unclassified"]
                            ),
                            required=True,
                            help="Select phase class for this manual object (options depend on the selected Sample Type).",
                        ),
                    },
                    disabled=["id", "diameter_um", "area_um2"],
                    hide_index=True,
                    use_container_width=True,
                    key=f"manual_data_editor_{image_name}_{curr_ver}",
                )

                col_apply, col_clear = st.columns([1.5, 1.0])
                with col_apply:
                    if st.button("✅ Apply Table Changes", type="primary", use_container_width=True):
                        if edited_data is not None and isinstance(edited_data, pd.DataFrame):
                            if "manual_phase_overrides" not in st.session_state:
                                st.session_state["manual_phase_overrides"] = {}

                            deleted_ids = set()
                            for _, row in edited_data.iterrows():
                                obj_id = int(row["id"])
                                if bool(row.get("Delete", False)):
                                    deleted_ids.add(obj_id)
                                else:
                                    new_phase = str(row["phase_class"])
                                    st.session_state["manual_phase_overrides"][obj_id] = new_phase

                            if deleted_ids:
                                st.session_state[manual_items_key] = [
                                    it for it in current_saved_items if int(it["id"]) not in deleted_ids
                                ]
                                for d_id in deleted_ids:
                                    st.session_state["manual_phase_overrides"].pop(d_id, None)

                            st.session_state["manual_canvas_ver"] = curr_ver + 1
                            st.rerun()

                with col_clear:
                    if st.button("🗑️ Clear All Manual", type="secondary", use_container_width=True):
                        st.session_state[manual_items_key] = []
                        if "manual_phase_overrides" in st.session_state:
                            st.session_state["manual_phase_overrides"] = {
                                k: v for k, v in st.session_state["manual_phase_overrides"].items() if k < 10000
                            }
                        st.session_state["manual_canvas_ver"] = curr_ver + 1
                        st.rerun()
            else:
                st.info("No manual objects added yet. Draw on the canvas above to add.")

            st.markdown("#### 💾 Manual Mask Persistence")
            save_path = Path("images") / f"{clean_stem}_manual_masks.json"

            load_notice = st.session_state.pop("manual_load_notice", None)
            if load_notice:
                st.success(load_notice)

            def _manual_json_default(value):
                """Serialize NumPy values stored inside manual object records."""
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, Path):
                    return str(value)
                if isinstance(value, (set, tuple)):
                    return list(value)
                return str(value)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if st.button("💾 Save Manual Objects to Disk", use_container_width=True):
                    Path("images").mkdir(parents=True, exist_ok=True)
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(
                            current_saved_items,
                            f,
                            default=_manual_json_default,
                            indent=2,
                        )
                    st.success(
                        f"Saved {len(current_saved_items)} objects to {save_path.name}"
                    )
            with col_s2:
                if save_path.exists():
                    if st.button("📂 Load Saved Manual Objects", use_container_width=True):
                        with open(save_path, "r", encoding="utf-8") as f:
                            loaded_items = json.load(f)
                        if isinstance(loaded_items, list):
                            restored_items = [
                                item
                                for item in loaded_items
                                if isinstance(item, dict)
                                and "id" in item
                                and "polygon" in item
                            ]
                            st.session_state[manual_items_key] = restored_items
                            st.session_state["manual_canvas_ver"] = (
                                st.session_state.get("manual_canvas_ver", 0) + 1
                            )
                            st.session_state["manual_load_notice"] = (
                                f"Loaded {len(restored_items)} manual objects from {save_path.name}"
                            )
                            st.rerun()
                        else:
                            st.error("Saved manual-masks file has an unexpected format.")
                else:
                    st.caption(
                        "No saved manual-mask file for this image yet — use Save above."
                    )

        with col_prev:
            st.markdown("### Live Composite Preview")
            st.image(
                overlay,
                caption=f"Composite Preview with {len(current_saved_items)} manual objects",
                **_FULL_WIDTH_IMAGE_KWARGS,
            )

    # ---------------- TAB 3: DETAILED METRICS & TABLES ----------------
    elif active_tab == "🔬 Metrics":
        st.subheader("Granulometry & Phase Metrics")
        if classified_df.empty:
            st.warning("No data available.")
        else:
            st.markdown("#### 📐 Advanced Granulometry Metrics ($D[3,2], D[4,3], D_{10}, D_{50}, D_{90}$)")
            st.markdown(
                "Key colloidal and microstructural dispersion metrics calculated per phase and overall: "
                "Sauter surface mean diameter $D[3,2]$, De Brouckere volume mean diameter $D[4,3]$, "
                "cumulative percentiles ($D_{10}, D_{50}, D_{90}$), and Polydispersity Span."
            )
            granulo_df = calculate_granulometry_metrics(classified_df)
            st.dataframe(
                granulo_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Phase": st.column_config.TextColumn("Phase"),
                    "Count": st.column_config.NumberColumn("Object Count", format="%d"),
                    "D[3,2] (µm)": st.column_config.NumberColumn("D[3,2] (µm)", format="%.2f"),
                    "D[4,3] (µm)": st.column_config.NumberColumn("D[4,3] (µm)", format="%.2f"),
                    "D10 (µm)": st.column_config.NumberColumn("D10 (µm)", format="%.2f"),
                    "D50 (µm)": st.column_config.NumberColumn("D50 / Median (µm)", format="%.2f"),
                    "D90 (µm)": st.column_config.NumberColumn("D90 (µm)", format="%.2f"),
                    "Span": st.column_config.NumberColumn("Span", format="%.2f"),
                },
            )

            # Export button directly below the metrics table
            csv_granulo = granulo_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results Table (CSV)",
                data=csv_granulo,
                file_name=f"phasepix_granulometry_metrics_{clean_stem}.csv",
                mime="text/csv",
                help="Export the granulometry results table (D[3,2], D[4,3], D10, D50, D90, Span) as a CSV file.",
            )

            st.divider()

            c_left, c_right = st.columns([1.6, 1.0])
            with c_left:
                st.markdown("#### Individual Particle Measurements")
                st.dataframe(classified_df, use_container_width=True, height=340)
            with c_right:
                st.markdown("#### Morphological Summary")
                stats_summary = classified_df.groupby("phase_class").agg(
                    Count=("id", "count"),
                    Mean_Diameter_um=("diameter_um", "mean"),
                    Std_Diameter_um=("diameter_um", "std"),
                    Mean_Circularity=("circularity", "mean"),
                    Total_Area_um2=("area_um2", "sum"),
                    Mean_Intensity=("mean_intensity", "mean"),
                ).round(2)
                st.dataframe(stats_summary, use_container_width=True)

    # ---------------- TAB: INTERFACIAL ADSORPTION ANALYTICS ----------------
    elif active_tab == "🛡️ Interfacial Analytics":
        st.subheader("🛡️ Interfacial Adsorption Analytics (Pickering Stabilization)")
        st.markdown(
            "Quantify adsorbed fat globules on verified air bubbles: surface coverage, rim coverage, "
            "packing density and apparent contact angle $\\theta = \\arcsin(L_c / D_{ref})$."
        )

        def _safe_ui_float(value, default: float = 0.0) -> float:
            """Convert a widget/table value to a finite float for captions and metrics."""
            try:
                converted = float(value)
            except (TypeError, ValueError):
                return float(default)
            return converted if np.isfinite(converted) else float(default)

        if classified_df.empty or "phase_class" not in classified_df.columns:
            st.warning("No classified objects are available yet. Run the segmentation first.")
        else:
            bubble_df = classified_df[
                classified_df["phase_class"].astype(str).str.lower() == "air_bubbles"
            ].copy()
            bubble_ids_numeric = (
                pd.to_numeric(bubble_df["id"], errors="coerce").dropna()
                if "id" in bubble_df.columns
                else pd.Series(dtype=float)
            )
            all_bubble_ids = sorted({int(value) for value in bubble_ids_numeric.tolist()})

            if not all_bubble_ids:
                st.info(
                    "💡 No air bubbles are available for interfacial analysis. "
                    "Add a bubble in the **🖌️ Manual Correction** tab, or enable air-bubble "
                    "detection in the sidebar (Foam / Two-Pass mode)."
                )
            else:
                qc_col, ref_col = st.columns([1.35, 1.0])
                with qc_col:
                    selected_bubble_ids = st.multiselect(
                        "🎯 Select Verified Air Bubbles for Interfacial Analysis (ID):",
                        options=all_bubble_ids,
                        default=all_bubble_ids,
                        help="Choose only the air bubbles where you have verified that all adsorbed fat particles are properly segmented.",
                    )
                with ref_col:
                    d_ref_mode = st.radio(
                        "Reference Fat Globule Diameter (D_ref) for Contact Angle θ:",
                        [
                            "Mean Matrix Fat Diameter",
                            "Matrix Fat Sauter D[3,2]",
                            "Custom Value (µm)",
                        ],
                        horizontal=True,
                    )
                    d_ref_custom = 1.0
                    if d_ref_mode == "Custom Value (µm)":
                        d_ref_custom = st.number_input(
                            "Custom D_ref (µm)",
                            min_value=0.01,
                            max_value=10000.0,
                            value=1.0,
                            step=0.1,
                            format="%.2f",
                        )

                # Resolve the reference globule diameter for theta = arcsin(L_c / D_ref)
                matrix_fat_diameters = pd.to_numeric(
                    classified_df.loc[
                        classified_df["phase_class"].astype(str).str.lower() == "fat_particles",
                        "diameter_um",
                    ],
                    errors="coerce",
                ).to_numpy(dtype=float)
                matrix_fat_diameters = matrix_fat_diameters[
                    np.isfinite(matrix_fat_diameters) & (matrix_fat_diameters > 0)
                ]
                mean_fat_diameter = (
                    float(matrix_fat_diameters.mean()) if matrix_fat_diameters.size else 0.0
                )
                sauter_fat_diameter = (
                    float(
                        np.sum(matrix_fat_diameters ** 3)
                        / max(float(np.sum(matrix_fat_diameters ** 2)), 1e-12)
                    )
                    if matrix_fat_diameters.size
                    else 0.0
                )
                if d_ref_mode == "Matrix Fat Sauter D[3,2]":
                    resolved_d_ref = sauter_fat_diameter
                elif d_ref_mode == "Custom Value (µm)":
                    resolved_d_ref = float(d_ref_custom)
                else:
                    resolved_d_ref = mean_fat_diameter
                d_ref_argument = resolved_d_ref if resolved_d_ref > 0 else None

                (
                    per_bubble_df,
                    ads_particles_df,
                    ads_summary,
                ) = compute_interfacial_adsorption_metrics(
                    df=classified_df,
                    selected_bubble_ids=selected_bubble_ids,
                    air_masks=padded_air_masks,
                    fat_masks=padded_masks,
                    manual_polygons=active_manual_polygons,
                    image_shape=base_display_img.shape[:2],
                    pixel_to_um_ratio=float(pixel_to_um_ratio),
                    d_ref_um=d_ref_argument,
                )
                resolved_d_ref = float(ads_summary.get("d_ref_um", resolved_d_ref or 0.0))

                kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)
                kpi_1.metric(
                    "Selected Bubbles",
                    f"{int(ads_summary.get('Selected Bubbles', 0)):,}",
                )
                kpi_2.metric(
                    "Adsorbed Globules",
                    f"{int(ads_summary.get('Total Adsorbed Globules', 0)):,}",
                )
                kpi_3.metric(
                    "Surface Coverage (%)",
                    f"{float(ads_summary.get('Overall Surface Coverage (%)', 0.0)):.2f}%",
                )
                kpi_4.metric(
                    "Packing Density (globules / µm²)",
                    f"{float(ads_summary.get('Mean Packing Density (glob/µm²)', 0.0)):.4f}",
                )
                kpi_5.metric(
                    "Mean Contact Angle θ (°)",
                    f"{float(ads_summary.get('Mean Contact Angle θ (°)', 0.0)):.1f}°",
                    delta=f"D_ref = {resolved_d_ref:.2f} µm",
                    delta_color="off",
                )

                summary_df = pd.DataFrame([ads_summary])
                st.download_button(
                    "📥 Download Overall Adsorption Summary (CSV)",
                    data=summary_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"phasepix_adsorption_summary_{clean_stem}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.divider()

                if per_bubble_df.empty or not selected_bubble_ids:
                    st.warning(
                        "No valid bubble masks were found for the current selection. "
                        "Verify the selected bubble IDs and their segmentation masks."
                    )
                else:
                    col_img, col_data = st.columns([1.15, 1.0])

                    with col_img:
                        st.markdown("#### 🖼️ Selected Bubble Inspection Overlay")
                        show_theta_labels = st.checkbox(
                            "Show Contact Angle Labels (°) on Image",
                            value=True,
                        )
                        inspection_img = render_selected_bubbles_inspection_overlay(
                            image=base_display_img,
                            df=classified_df,
                            selected_bubble_ids=selected_bubble_ids,
                            air_masks=padded_air_masks,
                            fat_masks=padded_masks,
                            manual_polygons=active_manual_polygons,
                            ads_particles_df=ads_particles_df,
                            show_contact_angles=bool(show_theta_labels),
                        )
                        st.image(
                            inspection_img,
                            caption=(
                                "Cyan: verified bubbles · Gold: adsorbed fat · "
                                "Labels: apparent contact angle θ"
                            ),
                            **_FULL_WIDTH_IMAGE_KWARGS,
                        )
                        buf_insp = io.BytesIO()
                        Image.fromarray(inspection_img).save(buf_insp, format="PNG", compress_level=1)
                        st.download_button(
                            "📥 Download Inspection Overlay (PNG)",
                            data=buf_insp.getvalue(),
                            file_name=f"Selected_Bubble_Inspection_{clean_stem}.png",
                            mime="image/png",
                            use_container_width=True,
                        )

                        st.markdown("#### 🔍 Single Bubble Crop Inspector")
                        inspect_options = per_bubble_df["Bubble ID"].astype(int).tolist()
                        inspect_id = st.selectbox(
                            "Bubble to inspect closely (ID):",
                            options=inspect_options,
                            format_func=lambda value: f"Bubble #{int(value)}",
                        )
                        inspect_row = per_bubble_df.loc[
                            per_bubble_df["Bubble ID"] == inspect_id
                        ].iloc[0]
                        geometry_rows = bubble_df.loc[
                            pd.to_numeric(bubble_df["id"], errors="coerce") == inspect_id
                        ]
                        if geometry_rows.empty:
                            st.info("Bubble geometry is unavailable for a close-up crop.")
                        else:
                            bubble_cx = _safe_ui_float(
                                geometry_rows.iloc[0].get("centroid_x"), np.nan
                            )
                            bubble_cy = _safe_ui_float(
                                geometry_rows.iloc[0].get("centroid_y"), np.nan
                            )
                            bubble_diam_um = _safe_ui_float(
                                inspect_row.get("Bubble Diam (µm)"), 0.0
                            )
                            if bubble_diam_um <= 0:
                                bubble_area_um2 = _safe_ui_float(
                                    inspect_row.get("Bubble Area (µm²)"), 0.0
                                )
                                bubble_diam_um = (
                                    math.sqrt(4.0 * bubble_area_um2 / math.pi)
                                    if bubble_area_um2 > 0
                                    else 0.0
                                )
                            diameter_px = bubble_diam_um / max(
                                float(pixel_to_um_ratio), 1e-6
                            )
                            half_side = int(max(24, round(diameter_px * 0.9)))
                            full_h, full_w = inspection_img.shape[:2]
                            if np.isfinite(bubble_cx) and np.isfinite(bubble_cy):
                                center_x = int(round(bubble_cx))
                                center_y = int(round(bubble_cy))
                                y0 = max(0, center_y - half_side)
                                y1 = min(full_h, center_y + half_side)
                                x0 = max(0, center_x - half_side)
                                x1 = min(full_w, center_x + half_side)
                                if y1 > y0 and x1 > x0:
                                    crop_img = inspection_img[y0:y1, x0:x1]
                                    st.image(
                                        crop_img,
                                        caption=f"Bubble #{inspect_id} — close-up",
                                        **_FULL_WIDTH_IMAGE_KWARGS,
                                    )
                                    st.caption(
                                        f"Coverage {_safe_ui_float(inspect_row.get('Surface Coverage (%)')):.1f}% · "
                                        f"Rim {_safe_ui_float(inspect_row.get('Rim Coverage (%)')):.1f}% · "
                                        f"θ = {_safe_ui_float(inspect_row.get('Mean Contact Angle θ (°)')):.1f}° · "
                                        f"Adsorbed globules: {int(_safe_ui_float(inspect_row.get('Adsorbed Count')))}"
                                    )

                                    # Фільтруємо адсорбовані частинки тільки для поточної бульбашки
                                    inspect_particles = ads_particles_df[
                                        pd.to_numeric(
                                            ads_particles_df.get(
                                                "parent_bubble_id", pd.Series(dtype=float)
                                            ),
                                            errors="coerce",
                                        )
                                        == inspect_id
                                    ].copy()

                                    st.markdown(
                                        f"##### 🟡 Adsorbed Globules on Bubble #{inspect_id}"
                                    )
                                    st.dataframe(
                                        inspect_particles,
                                        use_container_width=True,
                                        hide_index=True,
                                        height=200,
                                    )

                                    # Формуємо комплексний CSV-протокол (Summary + Detail)
                                    protocol_csv = "BUBBLE SUMMARY\n"
                                    protocol_csv += pd.DataFrame([inspect_row]).to_csv(
                                        index=False
                                    )
                                    protocol_csv += "\nADSORBED GLOBULES DETAIL\n"
                                    protocol_csv += inspect_particles.to_csv(index=False)

                                    buf_crop = io.BytesIO()
                                    Image.fromarray(crop_img).save(buf_crop, format="PNG", compress_level=1)
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.download_button(
                                            f"📥 Download Crop #{inspect_id} (PNG)",
                                            data=buf_crop.getvalue(),
                                            file_name=f"Bubble_Crop_{inspect_id}_{clean_stem}.png",
                                            mime="image/png",
                                            use_container_width=True,
                                        )
                                    with c2:
                                        st.download_button(
                                            f"📥 Download Protocol #{inspect_id} (CSV)",
                                            data=protocol_csv.encode("utf-8"),
                                            file_name=f"Bubble_Protocol_{inspect_id}_{clean_stem}.csv",
                                            mime="text/csv",
                                            use_container_width=True,
                                            help="Download a complete protocol containing bubble summary and specific globule data.",
                                        )

                    with col_data:
                        st.markdown("#### 📋 Per-Bubble Interfacial Metrics")
                        st.dataframe(
                            per_bubble_df,
                            use_container_width=True,
                            hide_index=True,
                            height=240,
                        )
                        st.download_button(
                            label="📥 Download Per-Bubble Metrics (CSV)",
                            data=per_bubble_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"phasepix_interfacial_per_bubble_{clean_stem}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                        st.markdown("#### 📈 Contact Angle Distribution")
                        theta_values = pd.to_numeric(
                            ads_particles_df.get(
                                "contact_angle_deg", pd.Series(dtype=float)
                            ),
                            errors="coerce",
                        ).dropna()
                        if theta_values.empty:
                            st.info(
                                "No adsorbed globules were detected inside the selected bubbles."
                            )
                        else:
                            fig_theta, ax_theta = plt.subplots(
                                figsize=(6.4, 3.0), dpi=130
                            )
                            sns.histplot(
                                theta_values,
                                bins=int(min(20, max(5, theta_values.size // 2))),
                                kde=bool(theta_values.size >= 3),
                                color="#ffd700",
                                ax=ax_theta,
                            )
                            ax_theta.set_xlabel("Apparent contact angle θ (°)")
                            ax_theta.set_ylabel("Globule count")
                            ax_theta.set_title("Adsorbed globule contact-angle distribution")
                            ax_theta.grid(alpha=0.25)
                            fig_theta.tight_layout()
                            st.pyplot(fig_theta, use_container_width=True)
                            buf_theta = io.BytesIO()
                            fig_theta.savefig(
                                buf_theta, format="png", dpi=300, bbox_inches="tight"
                            )
                            st.download_button(
                                "📥 Download Contact Angle Plot (PNG)",
                                data=buf_theta.getvalue(),
                                file_name=f"Contact_Angle_Distribution_{clean_stem}.png",
                                mime="image/png",
                                use_container_width=True,
                            )
                            plt.close(fig_theta)

                        st.markdown("#### 🟡 Adsorbed Globule Detail")
                        st.dataframe(
                            ads_particles_df,
                            use_container_width=True,
                            hide_index=True,
                            height=260,
                        )
                        st.download_button(
                            label="📥 Download Adsorbed Globules (CSV)",
                            data=ads_particles_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"phasepix_adsorbed_globules_{clean_stem}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

    # ---------------- TAB: MATRIX COALESCENCE NETWORK ----------------
    elif active_tab == "🍇 Matrix Coalescence":
        st.subheader("🍇 Matrix Partial Coalescence Network Analysis")
        st.markdown(
            "Build the contact network (RAG) of free matrix fat globules and quantify partial "
            "coalescence: coordination number $Z$, agglomerate size $N_{agg}$, branching nodes "
            "and the mean solidity (neck index)."
        )

        if classified_df.empty or "phase_class" not in classified_df.columns:
            st.warning("No classified objects are available yet. Run the segmentation first.")
        else:
            roi_col, tolerance_col = st.columns([1.4, 1.0])
            with roi_col:
                use_roi_crop = st.checkbox(
                    "🎯 Analyze Specific Verified Sub-Region (ROI Box) of Matrix",
                    value=False,
                )
            with tolerance_col:
                contact_tol_px = st.slider(
                    "Inter-Globule Contact Bridge Tolerance (px)",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Maximum pixel gap between fat masks to consider two globules partially coalesced.",
                )

            roi_box_px = None
            display_h, display_w = base_display_img.shape[:2]
            if use_roi_crop:
                roi_x_pct = st.slider(
                    "Horizontal ROI Range (% of width)",
                    min_value=0,
                    max_value=100,
                    value=(15, 85),
                )
                roi_y_pct = st.slider(
                    "Vertical ROI Range (% of height)",
                    min_value=0,
                    max_value=100,
                    value=(15, 85),
                )
                roi_box_px = (
                    int(round(display_w * float(roi_x_pct[0]) / 100.0)),
                    int(round(display_h * float(roi_y_pct[0]) / 100.0)),
                    int(round(display_w * float(roi_x_pct[1]) / 100.0)),
                    int(round(display_h * float(roi_y_pct[1]) / 100.0)),
                )
                st.caption(
                    f"ROI box (pixels): x = {roi_box_px[0]}–{roi_box_px[2]}, "
                    f"y = {roi_box_px[1]}–{roi_box_px[3]}"
                )

            (
                particles_coal_df,
                clusters_df,
                coalescence_edges,
                coal_summary,
            ) = compute_matrix_coalescence_metrics(
                df=classified_df,
                fat_masks=padded_masks,
                roi_box_px=roi_box_px,
                pixel_to_um_ratio=float(pixel_to_um_ratio),
                contact_dilation_px=int(contact_tol_px),
                bubble_mask=combined_bubble_mask,
            )

            kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)
            kpi_1.metric(
                "Coalesced Fat Fraction (%)",
                f"{float(coal_summary.get('Coalesced Fraction (%)', 0.0)):.1f}%",
                help="Share of matrix globules with at least one contact (Z >= 1).",
            )
            kpi_2.metric(
                "Mean Coordination (Z)",
                f"{float(coal_summary.get('Mean Coordination Number (Z)', 0.0)):.2f}",
                help="Average number of contacting neighbours per globule.",
            )
            kpi_3.metric(
                "Branching Nodes (Z≥3)",
                f"{float(coal_summary.get('Branching Nodes Z>=3 (%)', 0.0)):.1f}%",
                help="Globules acting as branching nodes of the coalescence network.",
            )
            kpi_4.metric(
                "Mean Cluster Size (N_agg)",
                f"{float(coal_summary.get('Mean Aggregation Number (N_agg)', 0.0)):.2f}",
                help="Average number of globules per multi-globule agglomerate.",
            )
            kpi_5.metric(
                "Mean Solidity / Neck Index",
                f"{float(coal_summary.get('Mean Solidity', 0.0)):.3f}",
                help="Mean convexity of matrix globules; lower values indicate stronger sintering necks.",
            )

            coal_summary_df = pd.DataFrame([coal_summary])
            st.download_button(
                "📥 Download Overall Coalescence Summary (CSV)",
                data=coal_summary_df.to_csv(index=False).encode("utf-8"),
                file_name=f"phasepix_coalescence_summary_{clean_stem}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if particles_coal_df.empty:
                st.info(
                    "💡 No matrix fat globules were found in the analysed region. "
                    "Matrix coalescence requires free fat particles outside the bubbles "
                    "(Emulsion / Whipped Emulsion mode)."
                )

            net_col, stats_col = st.columns([1.15, 1.0])
            with net_col:
                st.markdown("#### 🕸️ Coalescence Network Map")
                st.caption(
                    "🟦 Z = 0 isolated monomer · 🟩 Z = 1 dimer / chain end · "
                    "🟨 Z = 2 chain link · 🟥 Z ≥ 3 branch node"
                )
                network_img = render_coalescence_network_overlay(
                    image=base_display_img,
                    particles_coal_df=particles_coal_df,
                    fat_masks=padded_masks,
                    edges=coalescence_edges,
                    roi_box_px=roi_box_px,
                    show_bridges=True,
                )
                st.image(
                    network_img,
                    caption=(
                        "Contact network of matrix fat globules"
                        + (
                            f" · {len(coalescence_edges)} contact bridges "
                            f"(tolerance {int(contact_tol_px)} px)"
                            if coalescence_edges
                            else ""
                        )
                    ),
                    **_FULL_WIDTH_IMAGE_KWARGS,
                )
                buf_network = io.BytesIO()
                Image.fromarray(network_img).save(buf_network, format="PNG", compress_level=1)
                st.download_button(
                    label="📥 Download Network Map (PNG)",
                    data=buf_network.getvalue(),
                    file_name=f"phasepix_coalescence_network_{clean_stem}.png",
                    mime="image/png",
                    use_container_width=True,
                )

            with stats_col:
                st.markdown("#### 📊 Coordination Number Distribution")
                z_series = pd.to_numeric(
                    particles_coal_df.get(
                        "coordination_number", pd.Series(dtype=float)
                    ),
                    errors="coerce",
                ).dropna()
                if z_series.empty:
                    st.info("No globules are available for a coordination-number distribution.")
                else:
                    total_z = max(int(z_series.size), 1)
                    z_share_df = pd.DataFrame({
                        "Coordination (Z)": ["Z = 0", "Z = 1", "Z = 2", "Z ≥ 3"],
                        "Share of Globules (%)": [
                            float((z_series == 0).sum()) / total_z * 100.0,
                            float((z_series == 1).sum()) / total_z * 100.0,
                            float((z_series == 2).sum()) / total_z * 100.0,
                            float((z_series >= 3).sum()) / total_z * 100.0,
                        ],
                    })
                    fig_z, ax_z = plt.subplots(figsize=(6, 3), dpi=130)
                    ax_z.bar(
                        z_share_df["Coordination (Z)"],
                        z_share_df["Share of Globules (%)"],
                        color="#4c78a8",
                    )
                    ax_z.set_ylabel("Share of Globules (%)")
                    ax_z.set_title("Coordination Number Distribution")
                    fig_z.tight_layout()
                    st.pyplot(fig_z, use_container_width=True)
                    buf_z = io.BytesIO()
                    fig_z.savefig(buf_z, format="png", dpi=300, bbox_inches="tight")
                    st.download_button(
                        "📥 Download Z-Distribution Plot (PNG)",
                        data=buf_z.getvalue(),
                        file_name=f"Coordination_Number_Distribution_{clean_stem}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                    plt.close(fig_z)
                    st.caption(
                        f"{int(z_series.size)} matrix globules analysed · "
                        f"{len(coalescence_edges)} contact bridges"
                    )

                st.markdown("#### 🍇 Cluster Agglomerates (N_agg ≥ 2)")
                if clusters_df.empty:
                    st.info("No multi-globule agglomerates were detected in the analysed region.")
                else:
                    st.dataframe(
                        clusters_df,
                        use_container_width=True,
                        hide_index=True,
                        height=210,
                    )
                    st.download_button(
                        label="📥 Export Clusters (CSV)",
                        data=clusters_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"phasepix_coalescence_clusters_{clean_stem}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                st.markdown("#### 🟠 All Matrix Globules in ROI")
                st.dataframe(
                    particles_coal_df,
                    use_container_width=True,
                    hide_index=True,
                    height=260,
                )
                st.download_button(
                    label="📥 Export Particles (CSV)",
                    data=particles_coal_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"phasepix_coalescence_particles_{clean_stem}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ---------------- TAB 4: REPRODUCIBILITY & EXPORT ----------------
    elif active_tab == "💾 Reproducibility & Export":
        st.subheader("Reproducibility Parameters & Session Manifest")
        st.markdown(
            "Every analysis session parameters can be inspected and exported to ensure rigorous scientific reproducibility."
        )

        session_meta = {
            "Analysis Timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "Source Image": image_name,
            "Imaging Modality": imaging_mode,
            "Objective Preset": selected_objective if is_light_mode else "N/A (SEM Scale Bar)",
            "Full Micrograph Dimensions (px)": f"{img_w}x{img_h}",
            "Footer Height Excluded (px)": effective_footer,
            "Analysis Region Dimensions (px)": f"{analysis_img.shape[1]}x{analysis_img.shape[0]}",
            "Spatial Scale (px/µm)": pixels_per_um,
            "Pixel to µm Ratio": pixel_to_um_ratio,
            "Scale Bar Detected (px)": detected_bar_pixels if detected_bar_pixels > 0 else "None",
            "Scale Bar Nominal (µm)": scale_bar_um,
            "Auto Scale Used": use_auto_scale,
            "Sample Type": sample_type,
            "Air Bubble Size Range (µm)": (
                f"{min_air_size:.1f} - {max_air_size:.1f}" if sample_type in ["Whipped Emulsion", "Foam"] else "Disabled (Emulsion)"
            ),
            "Fat Particle Size Range (µm)": (
                f"{min_fat_size:.1f} - {max_fat_size:.1f}" if sample_type in ["Whipped Emulsion", "Emulsion"] else "Disabled (Foam)"
            ),
            "Overlap Priority": overlap_priority if sample_type == "Whipped Emulsion" else "N/A",
            "Out-of-Range Handling": out_of_range_handling,
            "Air Bubbles Min Circularity": float(air_min_circularity),
            "Matrix Fat Min Circularity": float(fat_min_circularity),
            "Matrix Fat Min Solidity": float(fat_min_solidity),
            "Emulsion Droplet Profile": emulsion_profile,
            "Typical Droplet Diameter for AI (µm)": (
                float(expected_droplet_um) if expected_droplet_um is not None else "N/A (SEM Mode)"
            ),
            "Multi-Scale Globule Detection": (
                bool(enable_multiscale_crystal) if is_light_mode else "N/A (SEM Mode)"
            ),
            "Internal Crystal Smoothing": (
                int(crystal_smooth_strength) if is_light_mode else "N/A (SEM Mode)"
            ),
            "Adsorbed Fat Min Circularity": float(ads_min_circularity),
            "Manual Phase Overrides": st.session_state.get("manual_phase_overrides", {}),
            "Show Air Bubbles": bool(show_air),
            "Show Fat Particles": bool(show_fat),
            "Air Bubbles Opacity": float(air_opacity),
            "Fat Particles Opacity": float(fat_opacity),
            "Z-Order (Layering)": str(z_order),
            "Cellpose Model": model_type,
            "SAM Backbone (Manual Smart Box)": active_sam_type,
            "Particle Diameter Setting (px)": eff_diameter if eff_diameter else "Auto",
            "Flow Threshold": float(flow_threshold),
            "Cell Probability Threshold": float(cellprob_threshold),
            "CLAHE Preprocessing": enable_clahe,
            "Brightfield Inversion": (
                bool(invert_brightfield) if is_light_mode else "N/A (SEM Mode)"
            ),
            "Hardware Accelerator": f"{device} ({accelerator_name}, GPU: {is_gpu})",
            "Total Segmented Objects (Raw)": len(raw_df),
            "Retained Objects (After Filters)": total_objects,
            "Air Bubbles Count": air_count,
            "Fat Particles Count": fat_count,
            "Adsorbed Fat Count": adsorbed_fat_count,
            "Unclassified Count": unclass_count,
        }

        # Генерація гарно відформатованого тексту для експорту
        manifest_text = "PhasePix Analysis Session Manifest\n"
        manifest_text += "=" * 40 + "\n\n"
        for key, value in session_meta.items():
            # Якщо значення є списком або словником, робимо відступи для читабельності
            if isinstance(value, (list, dict)):
                manifest_text += f"{key}:\n"
                if isinstance(value, list):
                    for item in value:
                        manifest_text += f"  - {item}\n"
                else:  # dict
                    for k, v in value.items():
                        manifest_text += f"  - {k}: {v}\n"
            else:
                manifest_text += f"{key}: {value}\n"

        col_meta, col_dl = st.columns([2, 1])
        with col_meta:
            st.json(session_meta)

        with col_dl:
            st.markdown("#### Export Session Data")
            st.download_button(
                label="📥 Download Manifest (.txt)",
                data=manifest_text.encode("utf-8"),
                file_name=f"phasepix_session_manifest_{clean_stem}.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download session parameters as a formatted text file.",
            )
            if not classified_df.empty:
                csv_bytes = classified_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Metrics (CSV)",
                    data=csv_bytes,
                    file_name=f"phasepix_morphology_metrics_{clean_stem}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Export segmented overlay
            buf_overlay_tab = io.BytesIO()
            Image.fromarray(overlay).save(buf_overlay_tab, format="PNG", compress_level=1)
            st.download_button(
                label="📥 Export Overlay (PNG)",
                data=buf_overlay_tab.getvalue(),
                file_name=f"segmented_overlay_{clean_stem}.png",
                mime="image/png",
                use_container_width=True,
            )

            # Publication-ready export with the SEM databar cropped away
            if effective_footer > 0:
                buf_overlay_tab_crop = io.BytesIO()
                Image.fromarray(overlay[:-effective_footer, :]).save(
                    buf_overlay_tab_crop, format="PNG", compress_level=1
                )
                st.download_button(
                    label="✂️ Export Cropped Overlay (No Footer, PNG)",
                    data=buf_overlay_tab_crop.getvalue(),
                    file_name=f"segmented_overlay_cropped_{clean_stem}.png",
                    mime="image/png",
                    use_container_width=True,
                    help=(
                        "Publication-ready micrograph with the bottom SEM databar "
                        "cropped out (preserves burned-in scale bar if enabled)."
                    ),
                )

            # Export for Cellpose retraining
            training_zip_export = prepare_cellpose_training_package(
                image=analysis_img,
                base_masks=filtered_masks,
                manual_polygons=[
                    it["polygon"]
                    for it in st.session_state.get(f"saved_manual_items_{image_name}", [])
                ],
                base_name=clean_stem,
                df=classified_df,
            )
            st.download_button(
                label="🎓 Export for Cellpose Training (.tif)",
                data=training_zip_export,
                file_name=f"{clean_stem}_cellpose_training.zip",
                mime="application/zip",
                use_container_width=True,
                help="Exports original micrograph and 3 specialized 16-bit TIFF mask variants (combined fat-on-top, air-only, fat-only) for Cellpose retraining.",
            )
