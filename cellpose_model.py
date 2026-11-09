"""
Module for SEM image segmentation using Cellpose with Apple Silicon (MPS) acceleration.
Provides model caching and inference utilities for PhasePix.
"""

from typing import Optional, Tuple
import logging
import math
import os
import shutil
import ssl
import urllib.request
from pathlib import Path

# ---------------- CPU THREADING GUARD (v1.7.3) ----------------
# PyTorch's CPU backend can deadlock for 10+ minutes on tiny images (e.g. 700x700)
# on Windows, because MKL / OpenMP / OpenCV oversubscribe the logical cores. That
# guard now applies ONLY to the Windows CPU-only fallback: on macOS (darwin),
# Linux and CUDA-enabled Windows machines these helpers are complete no-ops, so
# native multi-core (and multi-threaded OpenCV) performance is preserved.
# On the guarded path the safe cap has to be exported *before* numpy (MKL) and
# torch are imported -- those runtimes read OMP_NUM_THREADS / MKL_NUM_THREADS
# exactly once, at initialisation.
from cpu_thread_guard import (  # noqa: E402
    apply_cpu_thread_env,
    apply_opencv_thread_limits,
    apply_torch_thread_cap,
)

# No-op on macOS / CUDA machines (returns the native logical CPU count there).
CPU_THREADS = apply_cpu_thread_env()
# OpenCV is serialised only on the Windows CPU fallback; everywhere else it keeps
# its full multi-threaded pool while Cellpose resizes images and runs the flow
# dynamics (v1.7.3 restored native speed).
OPENCV_THREADS = apply_opencv_thread_limits()

import cv2  # noqa: E402  (must follow the threading guard)
import numpy as np  # noqa: E402  (must follow the threading guard)
import scipy.ndimage as ndi  # noqa: E402
import torch  # noqa: E402  (must follow the threading guard)
import streamlit as st  # noqa: E402
from skimage.segmentation import watershed as skimage_watershed  # noqa: E402

# PyTorch sizes its intra-op pool through its own API, not through OMP_*.
# Called after ``import torch`` so the CUDA decision uses the authoritative
# torch.cuda.is_available(): no-op on macOS/Linux, os.cpu_count() on Windows+CUDA
# and a 4-thread cap only for the Windows CPU-only fallback.
apply_torch_thread_cap(CPU_THREADS)

# Setup module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ---------------- PORTABLE MODEL CACHE LOCATION ----------------
# The Windows portable distribution ships ``PhasePix_Start.bat``, which exports
# ``CELLPOSE_LOCAL_MODELS_PATH`` so that every checkpoint lands inside the
# portable folder instead of ``C:\Users\<user>\.cellpose``. Cellpose resolves that
# variable only once, at import time, so it must be exported *before* the cellpose
# import further down.
def _resolve_models_cache_dir() -> Path:
    """
    Return the folder that stores every downloaded AI checkpoint.

    Priority:
        1. ``CELLPOSE_LOCAL_MODELS_PATH`` - exported by the portable Windows
           launcher and honoured identically by the Cellpose library itself.
        2. ``PHASEPIX_MODELS_DIR`` - optional project-level override.
        3. ``models_cache/`` in the working directory - the classic layout used by
           the Docker image and the native launchers.

    Returns:
        Path: Cache directory, created when missing.
    """
    cache_dir = Path("models_cache")
    for env_name in ("CELLPOSE_LOCAL_MODELS_PATH", "PHASEPIX_MODELS_DIR"):
        override = os.environ.get(env_name, "").strip()
        if override:
            cache_dir = Path(override).expanduser()
            break
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create the models cache folder %s: %s", cache_dir, exc)
    return cache_dir


MODELS_CACHE_DIR = _resolve_models_cache_dir()
# Export the resolved location so Cellpose itself writes cyto2 / cyto3 / cpsam
# weights into the same folder (it reads this variable once, at import time).
os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(MODELS_CACHE_DIR)

from cellpose import models  # noqa: E402  (must follow the cache-path export)


# ---------------- SEGMENT ANYTHING (SAM) BACKBONE REGISTRY ----------------
# All three official SAM architectures stay available so users with large
# unified-memory Macs (e.g. 48 GB) can trade loading time for boundary accuracy.
SAM_MODEL_CONFIGS = {
    "vit_b": {
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "min_bytes": 300_000_000,
    },
    "vit_l": {
        "filename": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "min_bytes": 1_100_000_000,
    },
    "vit_h": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "min_bytes": 2_400_000_000,
    },
}

# ---------------- LOCALIZED (ROI) SMART-BOX INFERENCE ----------------------
# SAM resizes its input to 1024 px on the long side, so feeding it the full
# high-resolution micrograph wastes preprocessing time and memory. Every drawn
# box is therefore served from a padded crop (box + 40 % on each side, never
# smaller than 256x256 px) and the predicted mask is shifted back into full-image
# coordinates. Neighbouring boxes reuse the already-computed embedding.
SAM_ROI_PAD_FRACTION = 0.40
SAM_ROI_MIN_SIDE = 256
# A set of boxes may share one union embedding when the union ROI is at most this
# many times larger than the largest single-box ROI (i.e. the boxes are close).
SAM_ROI_UNION_BUDGET = 2.5


def _normalize_sam_model_type(sam_model_type: Optional[str]) -> str:
    """Return a valid SAM backbone key, defaulting safely to ``vit_b``."""
    if isinstance(sam_model_type, str) and sam_model_type in SAM_MODEL_CONFIGS:
        return sam_model_type
    if sam_model_type:
        logger.warning(
            "Unknown SAM backbone '%s'; defaulting to 'vit_b'.", sam_model_type
        )
    return "vit_b"


def get_models_cache_dir() -> Path:
    """
    Return the shared checkpoint folder used by every AI backend.

    Resolved once at import time from ``CELLPOSE_LOCAL_MODELS_PATH`` (portable
    Windows build), then ``PHASEPIX_MODELS_DIR``, then ``models_cache/``.

    Returns:
        Path: Directory holding the SAM and Cellpose weights.
    """
    return MODELS_CACHE_DIR


def ensure_sam_checkpoint(sam_model_type: str = "vit_b") -> Optional[Path]:
    """
    Download a SAM checkpoint into the shared cache when it is missing.

    The file is streamed in 8 MB chunks into a ``.part`` file and atomically
    moved into place, so an interrupted download can never poison the cache.
    Used both by the interactive predictor and by the portable Windows builder
    to pre-seed ``models_cache`` before the folder is handed to colleagues.

    Args:
        sam_model_type: One of ``"vit_b"``, ``"vit_l"``, ``"vit_h"``.

    Returns:
        Optional[Path]: Path to the checkpoint, or None when it cannot be fetched.
    """
    sam_type = _normalize_sam_model_type(sam_model_type)
    config = SAM_MODEL_CONFIGS[sam_type]
    ckpt_path = get_models_cache_dir() / config["filename"]
    min_bytes = int(config["min_bytes"])

    # A truncated/partial checkpoint would break torch.load, so anything
    # suspiciously small is treated as missing and re-downloaded.
    if ckpt_path.exists() and ckpt_path.stat().st_size >= min_bytes:
        logger.info("SAM %s checkpoint already cached at %s.", sam_type, ckpt_path)
        return ckpt_path

    url = config["url"]
    logger.info("SAM %s checkpoint missing or truncated. Downloading from %s ...", sam_type, url)
    tmp_path = ckpt_path.with_name(ckpt_path.name + ".part")
    try:
        # macOS Python often lacks a proper CA bundle, which makes
        # urllib.request.urlretrieve fail with SSLCertVerificationError.
        # Use an explicitly unverified SSL context for this one download.
        ssl_ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ssl_ctx) as response, open(
            tmp_path, "wb"
        ) as out_file:
            # Stream in 8 MB chunks: vit_h is ~2.56 GB, so never buffer it whole.
            shutil.copyfileobj(response, out_file, length=8 * 1024 * 1024)

        downloaded_bytes = tmp_path.stat().st_size if tmp_path.exists() else 0
        if downloaded_bytes < min_bytes:
            logger.warning(
                "SAM %s download incomplete (%d of at least %d bytes); aborting.",
                sam_type,
                downloaded_bytes,
                min_bytes,
            )
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return None

        os.replace(tmp_path, ckpt_path)
        logger.info(
            "SAM %s checkpoint downloaded to %s (%.1f MB).",
            sam_type,
            ckpt_path,
            downloaded_bytes / 1e6,
        )
        return ckpt_path
    except Exception as net_exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        logger.warning("Failed to download SAM checkpoint (%s): %s", url, net_exc)
        return None


def ensure_cellpose_weights(model_type: str = "cyto2") -> Optional[Path]:
    """
    Download Cellpose pretrained weights into the shared cache folder.

    Supports both Cellpose 3.x (``models.model_path``) and Cellpose 4.x
    (``models.cache_model_path`` + ``MODEL_NAMES``), so the portable Windows
    builder can pre-seed the cache for whatever version pip resolved. Missing
    weight files are skipped gracefully - PhasePix would simply download them on
    first use.

    Args:
        model_type: Preferred Cellpose model name, e.g. ``"cyto2"`` or ``"cyto3"``.

    Returns:
        Optional[Path]: Path to the first weight file that exists, else None.
    """
    try:
        # Local import: keeps this module importable when Cellpose is absent.
        from cellpose import models as cellpose_models
    except Exception as exc:
        logger.warning("Cellpose is not installed; cannot pre-download weights: %s", exc)
        return None

    cache_dir = get_models_cache_dir()
    cellpose_dir = getattr(cellpose_models, "MODEL_DIR", None)
    if cellpose_dir is not None:
        try:
            if Path(cellpose_dir).resolve() != cache_dir.resolve():
                logger.warning(
                    "Cellpose MODEL_DIR is %s but PhasePix caches weights in %s. "
                    "Export CELLPOSE_LOCAL_MODELS_PATH before importing cellpose.",
                    cellpose_dir,
                    cache_dir,
                )
        except OSError:
            pass

    # Cellpose 3.x exposes model_path(); Cellpose 4.x exposes cache_model_path().
    model_path_fn = getattr(cellpose_models, "model_path", None)
    cache_path_fn = getattr(cellpose_models, "cache_model_path", None)
    bundled_names = tuple(getattr(cellpose_models, "MODEL_NAMES", ()) or ())

    candidates = []
    for name in (model_type, "cyto2", "cyto3", "cpsam_v2", "cpsam", "cpdino-vitb"):
        if name and name not in candidates:
            candidates.append(name)

    for name in candidates:
        if callable(cache_path_fn) and name in bundled_names:
            try:
                resolved = cache_path_fn(name)
            except Exception as exc:
                logger.info("Cellpose cache_model_path('%s') failed: %s", name, exc)
                resolved = None
            if resolved and Path(resolved).exists():
                logger.info("Cellpose weights ready (%s): %s", name, resolved)
                return Path(resolved)
        if callable(model_path_fn):
            try:
                resolved = model_path_fn(name)
            except Exception as exc:
                logger.info("Cellpose model_path('%s') failed: %s", name, exc)
                resolved = None
            if resolved and Path(resolved).exists():
                logger.info("Cellpose weights ready (%s): %s", name, resolved)
                return Path(resolved)

    logger.warning(
        "Could not pre-download Cellpose weights for %s; they will be fetched on first use.",
        model_type,
    )
    return None


def to_grayscale_2d(image: np.ndarray) -> np.ndarray:
    """
    Collapse an image array to the single intensity plane PhasePix analyses on.

    Confocal laser scanning microscopy (CLSM) acquisitions are usually stored as
    multi-channel stacks (``(H, W, 3)`` RGB or ``(H, W, 4)`` RGBA), while Cellpose
    inference, CLAHE preprocessing and every morphological metric operate on one
    2D plane. Those colour stacks are therefore projected onto the OpenCV
    luminance plane, which weighs the channels by perceived brightness and keeps
    the visible contrast of a mixed multi-channel overlay.

    Args:
        image: ``(H, W)`` grayscale, ``(H, W, 1)`` single-channel, ``(H, W, 3)``
            RGB or ``(H, W, 4)`` RGBA array. Stacks with more than 4 channels are
            reduced by a channel mean projection.

    Returns:
        np.ndarray: 2D array. Grayscale / single-channel input is returned
        unchanged (no copy), colour input is converted with
        ``cv2.COLOR_RGB2GRAY`` / ``cv2.COLOR_RGBA2GRAY`` while preserving dtype.

    Raises:
        TypeError: If ``image`` is not a numpy array.
        ValueError: If the array is not 2D or 3D.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(
            f"Expected a 2D image or an (H, W, C) stack, but got shape {image.shape}"
        )

    n_channels = image.shape[2]
    if n_channels == 1:
        return image[:, :, 0]
    if n_channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if n_channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    # Hyperstacks (>4 channels, e.g. unmixed spectral CLSM): mean projection.
    mean_projection = image.astype(np.float32).mean(axis=2)
    if np.issubdtype(image.dtype, np.integer):
        return np.rint(mean_projection).astype(image.dtype)
    return mean_projection


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to a grayscale image.
    Enhances local contrast to reveal subtle, low-contrast boundaries such as large air pores
    in SEM micrographs without over-amplifying background noise.

    Args:
        image: 2D numpy array (grayscale) or an ``(H, W, C)`` stack, which is
            first projected to its luminance plane by ``to_grayscale_2d`` so
            multi-channel CLSM frames can be preprocessed as well.
        clip_limit: Threshold for contrast limiting. Default is 2.0.
        tile_grid_size: Size of grid for histogram equalization (e.g., (8, 8)).

    Returns:
        np.ndarray: Contrast-enhanced 2D uint8 numpy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array.")

    img_2d = to_grayscale_2d(image)
    if img_2d.dtype != np.uint8:
        # Normalize to uint8 [0, 255]
        min_val, max_val = float(img_2d.min()), float(img_2d.max())
        img_uint8 = np.clip(
            ((img_2d - min_val) / max(max_val - min_val, 1e-6)) * 255.0, 0, 255
        ).astype(np.uint8)
    else:
        img_uint8 = img_2d

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_uint8)


# ---------------- DOWNSCALE-FOR-INFERENCE FAST PATHS (v1.8.6) ----------------
# Cellpose's native receptive field is only ~30 px: ``CellposeModel.eval()`` rescales
# every frame by ``diameter / diam_mean`` (``diam_mean`` = 30 px for cyto2/cyto3), so
# asking for a 210 px Light Microscopy globule upsamples a 3000x3000 frame by 7x
# *before* the network even starts (7x7 = 49x the pixel budget).  Beyond ~45 px the
# extra resolution carries no additional boundary information, only cost, so every
# request above ``CELLPOSE_FAST_DIAMETER_TRIGGER_PX`` is answered at
# ``CELLPOSE_FAST_DIAMETER_PX`` on a proportionally downscaled frame.  The integer
# label mask is mapped back with ``cv2.INTER_NEAREST`` (labels copied one-to-one --
# no blending, no interpolated ids), exactly like the v1.5 air-bubble fast path.
CELLPOSE_FAST_DIAMETER_TRIGGER_PX = 45.0
CELLPOSE_FAST_DIAMETER_PX = 35.0

# ``CellposeModel.eval`` also rescales *upward* whenever the requested diameter is
# BELOW its 30 px receptive field (``diam_mean``).  A 5-9 um target on a 10x objective
# (0.72 um/px) is only 7-14 px, so ``eval`` resampled the whole micrograph 2.1x-4.3x
# per axis -- 4.5x-18x the pixel budget, with dozens of overlapping tiles -- and one
# pass took ~60 s (174 s for the v1.8.7 three-pass Light Microscopy run).  Requests are
# therefore never handed to the network below ``CELLPOSE_MIN_INFERENCE_DIAMETER_PX``
# (28 px < 30 px, i.e. at most 1.07x per axis) and the frame itself is capped at
# ``CELLPOSE_MAX_INFERENCE_SIDE_PX``, which pins one pass at ~1-1.3 Mpx of network
# budget -- single-digit seconds on the reference Mac (the whole two-pass Light
# Microscopy run measured 174 s -> 9.5 s on the 1808x1808 10x fixture at the worst-case
# 5-9 um setting) -- no matter what the "Typical Droplet Diameter for AI" slider says.
#
# Both clamps are OPT-IN (``run_inference(..., allow_downscale=True)``): the SEM paths,
# and ``segment_air_bubbles_auto`` above all, already own their multi-scale resizing
# (``scale_factor``), so touching them again in ``run_inference`` double-downscales the
# micrograph and destroys large air-bubble craters.  SEM therefore stays byte-identical
# to v1.8.5.
CELLPOSE_MIN_INFERENCE_DIAMETER_PX = 28.0
CELLPOSE_MAX_INFERENCE_SIDE_PX = 1024

# ``cv2.medianBlur`` and ``cv2.bilateralFilter`` both cost O(k^2) *per pixel*, so the
# crystal smoothing of a 2048-4096 px Light Microscopy frame runs on a half-size
# copy (4x less work) and the sharp droplet walls are restored afterwards.
CRYSTAL_SMOOTH_HALF_RES_MAX_SIDE = 1024
# Canny band used to re-inject the full-resolution outer rims after the upscale.  It
# runs on the already smoothed half-size image (striations are gone by then, droplet
# walls survive), so the pair only has to separate a wall step from residual noise.
CRYSTAL_SMOOTH_EDGE_CANNY = (80, 200)


def smooth_internal_crystals(image: np.ndarray, strength: int = 2) -> np.ndarray:
    """
    Suppress internal crystal needle/fibril lines inside crystallized fat globules
    while preserving the dark outer droplet boundaries (edge-preserving filtering).

    v1.8.6 - half-resolution fast path:
        A Light Microscopy frame (long side > ``CRYSTAL_SMOOTH_HALF_RES_MAX_SIDE``)
        is smoothed on a half-size ``cv2.INTER_AREA`` copy with proportionally
        scaled kernels and upscaled again with ``cv2.INTER_LINEAR``.  Median and
        bilateral filtering cost O(k^2) per pixel, so the same visual result costs
        ~1/4 of the filter work (measured 28 ms -> 3 ms on a 3000x3000 frame).
        The outer rims that the upscale softens are then copied back from the
        original along a Canny mask detected on the *smoothed* half-size image
        (where the crystal striations are already gone), so the droplet walls
        Cellpose segments on stay razor sharp (measured rim contrast 20 -> 94, i.e.
        sharper than the full-resolution filter) while every striation inside a
        globule stays erased (interior residual 10.1 -> 9.1).

    Args:
        image: 2D grayscale image (any numeric dtype; float is min-max normalised).
        strength: Smoothing strength; the median kernel is ``2*strength + 1``.

    Returns:
        np.ndarray: uint8 smoothed 2D image with the input spatial shape.
    """
    img_2d = image.squeeze() if image.ndim == 3 else image
    if img_2d.dtype != np.uint8:
        min_v, max_v = float(img_2d.min()), float(img_2d.max())
        img_u8 = np.clip(((img_2d - min_v) / max(max_v - min_v, 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img_2d.copy()

    s = max(1, int(strength))
    # 1. Median filter wipes out thin dark crystal striations inside droplets
    k_med = 2 * s + 1  # e.g., 5x5 for strength=2, 7x7 for strength=3

    full_h, full_w = img_u8.shape[:2]
    if max(full_h, full_w) > CRYSTAL_SMOOTH_HALF_RES_MAX_SIDE:
        half_w = max(2, full_w // 2)
        half_h = max(2, full_h // 2)
        small = cv2.resize(img_u8, (half_w, half_h), interpolation=cv2.INTER_AREA)
        # Kernel footprints are pixel sizes, so they halve with the grid; sigmaColor
        # does not (grey levels are resolution independent).
        k_med_half = max(3, (k_med // 2) | 1)
        smooth_small = cv2.bilateralFilter(
            cv2.medianBlur(small, k_med_half),
            d=5,
            sigmaColor=35 * s,
            sigmaSpace=max(1.0, 3.5 * s),
        )
        smooth_up = cv2.resize(
            smooth_small, (full_w, full_h), interpolation=cv2.INTER_LINEAR
        )
        # 2. Blending the high-frequency outer edges back.  The walls are detected on
        #    the *smoothed* half-size image, where the crystal striations no longer
        #    exist, so the mask can only follow real droplet rims; it is upscaled with
        #    INTER_NEAREST and re-injected with one masked copy (no float temporaries).
        edge_mask = cv2.Canny(smooth_small, *CRYSTAL_SMOOTH_EDGE_CANNY)
        if np.any(edge_mask):
            edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
            edge_mask = cv2.resize(
                edge_mask, (full_w, full_h), interpolation=cv2.INTER_NEAREST
            )
            smooth_up = cv2.copyTo(img_u8, edge_mask, smooth_up)
        return smooth_up

    med = cv2.medianBlur(img_u8, k_med)
    # 2. Bilateral filter homogenizes droplet interiors while keeping outer rims sharp
    smooth = cv2.bilateralFilter(med, d=9, sigmaColor=35 * s, sigmaSpace=7 * s)
    return smooth


def get_cuda_device_name() -> Optional[str]:
    """
    Return the marketing name of CUDA device 0 (e.g. ``NVIDIA GeForce RTX 4060``).

    Returns:
        Optional[str]: The GPU model name, or None when CUDA is not available.
    """
    try:
        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name(0))
    except Exception as exc:
        logger.warning("Could not read the CUDA device name: %s", exc)
        return None


def get_device() -> Tuple[torch.device, bool]:
    """
    Detect and return the fastest compute device available at startup.

    Priority order:
        1. NVIDIA CUDA (``torch.cuda.is_available()``) - fastest backend.
        2. Apple Silicon MPS (Metal Performance Shaders).
        3. CPU fallback - every feature keeps working, just slower.

    The result is consumed by :func:`get_cellpose_model` (``gpu=`` / ``device=``)
    and :func:`get_sam_predictor`, so the backend is selected without any user
    configuration.

    Returns:
        Tuple[torch.device, bool]: (device, is_gpu_active)
    """
    try:
        if torch.cuda.is_available():
            logger.info(
                "NVIDIA CUDA detected and enabled (%s).", get_cuda_device_name() or "unknown GPU"
            )
            return torch.device("cuda"), True
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logger.info("Apple Silicon MPS (Metal Performance Shaders) detected and enabled.")
            return torch.device("mps"), True
    except Exception as exc:
        logger.warning("Hardware acceleration probe encountered an issue: %s. Defaulting to CPU.", exc)

    logger.info("Using CPU backend.")
    return torch.device("cpu"), False


def detect_hardware() -> Tuple[torch.device, bool, str]:
    """
    Detect the active backend and return a short human-readable accelerator label.

    Used by the sidebar badge so the exact hardware in use is always visible:
        * CUDA   -> the GPU model reported by ``torch.cuda.get_device_name(0)``
        * MPS    -> ``Apple Metal (MPS)``
        * CPU    -> ``CPU``

    Returns:
        Tuple[torch.device, bool, str]: (device, is_gpu_active, accelerator_name)
    """
    device, is_gpu = get_device()
    if device.type == "cuda":
        return device, True, get_cuda_device_name() or "NVIDIA GPU"
    if device.type == "mps":
        return device, True, "Apple Metal (MPS)"
    return device, False, "CPU"


def get_sam_device() -> torch.device:
    """
    Return the device used by Segment Anything (SAM) predictors.

    SAM follows the Cellpose policy: NVIDIA CUDA when present, **Apple Silicon
    MPS** otherwise (the ViT image encoder runs ~2x faster on Metal than on CPU),
    and CPU as the final fallback. SAM's prompt/decoder tensors are all created as
    ``float32``, so the historical ``float64`` Metal crash does not occur with the
    current ``segment_anything`` release; if an unsupported op ever surfaces, the
    caller moves the model back to CPU transparently (see
    :func:`get_cached_sam_predictor_for_roi`).

    Returns:
        torch.device: ``cuda`` / ``mps`` when available, otherwise ``cpu``.
    """
    device, is_gpu = get_device()
    if is_gpu and device.type == "cuda":
        return device
    if device.type == "mps":
        logger.info("SAM image encoder runs on Apple Silicon MPS (metal).")
        return device
    return torch.device("cpu")


def _sam_move_to_cpu(predictor) -> bool:
    """
    Move a SAM predictor back to CPU after an MPS-op failure (best effort).

    The Metal backend supports every op used by the ViT-B/L/H encoders and the
    mask decoder in the pinned ``segment_anything`` release, but a future/odd
    build could still raise on an individual kernel. Falling back to CPU keeps
    interactive Smart Box segmentation working instead of returning an empty mask.

    Args:
        predictor: A ``SamPredictor`` instance.

    Returns:
        bool: True when the predictor now lives on CPU and its cache was cleared.
    """
    model = getattr(predictor, "model", None)
    if model is None:
        return False
    try:
        model.to(device=torch.device("cpu"))
        predictor.reset_image()
        predictor._pp_force_cpu = True
        logger.warning("SAM fell back to CPU (unsupported Metal op on this build).")
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("SAM CPU fallback failed: %s", exc)
        return False


def _instantiate_cellpose_model(
    model_type: str,
    custom_model_path: Optional[str],
    device: torch.device,
    use_gpu: bool,
) -> models.CellposeModel:
    """
    Build a CellposeModel, degrading ``cyto3`` -> ``cyto2`` when unavailable.

    ``cyto3`` is the super-generalist Cellpose model, but not every Cellpose build
    bundles it. A missing/broken ``cyto3`` must never crash the app, so the request
    falls back to the proven ``cyto2`` baseline.
    """
    if custom_model_path:
        return models.CellposeModel(
            pretrained_model=custom_model_path,
            gpu=use_gpu,
            device=device,
        )

    requested = model_type
    bundled_models = getattr(models, "MODEL_NAMES", None)
    if (
        requested == "cyto3"
        and isinstance(bundled_models, (list, tuple, set, frozenset))
        and "cyto3" not in bundled_models
    ):
        logger.warning(
            "This Cellpose build does not bundle 'cyto3' (available: %s); using 'cyto2'.",
            ", ".join(str(name) for name in bundled_models) or "unknown",
        )
        requested = "cyto2"

    try:
        return models.CellposeModel(
            pretrained_model=requested,
            model_type=requested,
            gpu=use_gpu,
            device=device,
        )
    except Exception as exc:
        if requested != "cyto3":
            raise
        logger.warning(
            "Failed to initialize Cellpose 'cyto3' (%s); falling back to 'cyto2'.", exc
        )
        return models.CellposeModel(
            pretrained_model="cyto2",
            model_type="cyto2",
            gpu=use_gpu,
            device=device,
        )


@st.cache_resource(show_spinner="Loading Cellpose segmentation model...")
def get_cellpose_model(
    model_type: str = "cyto2",
    custom_model_path: Optional[str] = None,
) -> models.CellposeModel:
    """
    Load and cache the Cellpose model instance.
    Uses @st.cache_resource to prevent re-initialization during Streamlit app re-runs.

    Args:
        model_type: The Cellpose model identifier ('cyto3', 'cyto2', 'nuclei').
        custom_model_path: Optional filesystem path to user-supplied custom .pth weights.

    Returns:
        models.CellposeModel: Initialized CellposeModel instance.
    """
    device, is_gpu = get_device()
    try:
        model = _instantiate_cellpose_model(model_type, custom_model_path, device, is_gpu)
        logger.info(
            "CellposeModel request '%s' ready on %s (gpu=%s).",
            model_type,
            device,
            is_gpu,
        )
        return model
    except Exception as exc:
        logger.error(
            "Failed to initialize Cellpose model '%s' on %s: %s. Falling back to CPU.",
            model_type,
            device,
            exc,
        )
        return _instantiate_cellpose_model(
            model_type, custom_model_path, torch.device("cpu"), False
        )


@st.cache_data(show_spinner="Running Cellpose segmentation inference...")
def segment_image_cached(
    image: np.ndarray,
    model_type: str = "cyto2",
    custom_model_path: Optional[str] = None,
    diameter: Optional[float] = None,
    flow_threshold: float = 0.7,
    cellprob_threshold: float = -1.0,
    enable_clahe: bool = True,
) -> np.ndarray:
    """
    Run Cellpose segmentation on a grayscale image with Streamlit caching.

    Cached via @st.cache_data so changing unrelated UI components (such as brightness,
    contrast or visual styling sliders) does not re-trigger expensive deep learning inference.

    Args:
        image: 2D grayscale image as a numpy array, or a multi-channel
            ``(H, W, 3)`` CLSM stack (projected to luminance before inference).
        model_type: Model name ('cyto2', 'nuclei', or 'Custom Model File...').
        custom_model_path: Optional filesystem path to custom trained weights.
        diameter: Estimated average particle/cell diameter in pixels. None for auto-estimation.
        flow_threshold: Maximum allowed flow error threshold (default: 0.7).
        cellprob_threshold: Cell probability threshold for detection (default: -1.0).
        enable_clahe: Whether to apply CLAHE contrast enhancement before inference.

    Returns:
        np.ndarray: 2D array of labeled masks (0 for background, 1, 2, ... for objects).
    """
    model = get_cellpose_model(model_type=model_type, custom_model_path=custom_model_path)
    return run_inference(
        model=model,
        image=image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        enable_clahe=enable_clahe,
    )


def run_inference(
    model: models.CellposeModel,
    image: np.ndarray,
    diameter: Optional[float] = None,
    flow_threshold: float = 0.7,
    cellprob_threshold: float = -1.0,
    enable_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    allow_downscale: bool = False,
) -> np.ndarray:
    """
    Perform direct inference on a 2D grayscale image using a loaded Cellpose model.

    Args:
        model: An initialized CellposeModel.
        image: 2D numpy array (grayscale image), a single-channel 3D array
            (H, W, 1) or a multi-channel CLSM stack (H, W, 3) / (H, W, 4), which
            is projected to its luminance plane by ``to_grayscale_2d`` before the
            network sees it.
        diameter: Average diameter of objects in pixels.  Values above
            ``CELLPOSE_FAST_DIAMETER_TRIGGER_PX`` (45 px) are answered at
            ``CELLPOSE_FAST_DIAMETER_PX`` (35 px) on a proportionally downscaled
            frame, because Cellpose's receptive field is only ~30 px wide.  Values
            below ``CELLPOSE_MIN_INFERENCE_DIAMETER_PX`` (28 px) are raised to that
            floor, because ``CellposeModel.eval`` would otherwise *upscale* the whole
            frame (v1.8.6/v1.8.8).
        flow_threshold: Flow error threshold (default 0.7).
        cellprob_threshold: Cell probability threshold (default -1.0).
        enable_clahe: Apply CLAHE contrast enhancement prior to model feeding.
        clahe_clip_limit: Contrast clip limit for CLAHE.
        clahe_tile_grid_size: Grid size for CLAHE tiles.
        allow_downscale: Opt in to the diameter/frame-side clamps above (Light
            Microscopy AI passes only).  Default ``False`` keeps SEM inference
            byte-identical to v1.8.5: ``segment_air_bubbles_auto`` manages its own
            multi-scale resizing, so a second downscale here would destroy large
            air-bubble craters (v1.8.8).

    Returns:
        np.ndarray: 2D numpy array of integer labels (0=background, 1, 2, ...=objects)
                    matching the input spatial dimensions (H, W).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    # v2.0 CLSM: multi-channel (H, W, 3) / (H, W, 4) confocal stacks are reduced
    # to their luminance plane; grayscale and (H, W, 1) input pass through
    # unchanged, so SEM / Light inference behaviour is untouched.
    img_2d = to_grayscale_2d(image)

    # Validate non-empty dimensions
    if img_2d.size == 0 or img_2d.shape[0] == 0 or img_2d.shape[1] == 0:
        raise ValueError("Image array cannot be empty.")

    # Apply CLAHE preprocessing to enhance contrast for pores & bubbles if enabled
    if enable_clahe:
        input_for_model = apply_clahe(
            img_2d,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )
    else:
        input_for_model = img_2d

    # ── Opt-in downscale-for-inference (v1.8.6, gated in v1.8.8) ──────────────
    # ``CellposeModel.eval`` rescales the frame by ``diameter / diam_mean`` (30 px for
    # cyto2/cyto3), i.e. the requested diameter *is* a resampling factor:
    #   * above 45 px  -> the network processes a frame several times larger per axis
    #     than the one handed in (a 210 px Light Microscopy request = 7x7 more pixels);
    #   * below 28 px  -> the network *upscales* the frame (a 12 px request = 2.5x per
    #     axis, 6.25x the pixel budget), which is what froze Light Microscopy for 174 s
    #     once the "Typical Droplet Diameter for AI (µm)" slider reached 5-9 µm.
    # Both directions are clamped here and the frame itself is capped at
    # ``CELLPOSE_MAX_INFERENCE_SIDE_PX``, so one pass costs ~1-1.3 Mpx of network budget
    # (single-digit seconds) for ANY slider value.  The integer label mask is mapped back
    # with INTER_NEAREST, so every id and every boundary pixel is preserved exactly.
    #
    # This is strictly opt-in: ``segment_air_bubbles_auto`` already downscales its own
    # passes with ``scale_factor`` (42 px target), so a second clamp/downscale here made
    # it run SEM frames at a fraction of the intended resolution and destroyed the large
    # air-bubble craters (v1.8.6 regression).  SEM callers keep ``allow_downscale=False``
    # and therefore get the untouched v1.8.5 behaviour.
    pass_diameter = diameter
    downscale = 1.0
    orig_h, orig_w = img_2d.shape[:2]
    if allow_downscale:
        eff_cp_diam = (
            max(float(diameter), CELLPOSE_MIN_INFERENCE_DIAMETER_PX)
            if diameter is not None
            else None
        )
        if eff_cp_diam is not None and eff_cp_diam > CELLPOSE_FAST_DIAMETER_TRIGGER_PX:
            downscale = CELLPOSE_FAST_DIAMETER_PX / eff_cp_diam
        max_side = float(max(orig_h, orig_w))
        if max_side > 0 and max_side * downscale > float(CELLPOSE_MAX_INFERENCE_SIDE_PX):
            downscale = float(CELLPOSE_MAX_INFERENCE_SIDE_PX) / max_side
        if downscale < 1.0:
            small_size = (
                max(1, int(round(orig_w * downscale))),
                max(1, int(round(orig_h * downscale))),
            )
            input_for_model = cv2.resize(
                np.asarray(input_for_model), small_size, interpolation=cv2.INTER_AREA
            )
        if eff_cp_diam is not None:
            # Never hand Cellpose a diameter below its receptive field: that request
            # alone would make ``eval`` resample the frame upward (v1.8.8).
            pass_diameter = max(
                eff_cp_diam * downscale, CELLPOSE_MIN_INFERENCE_DIAMETER_PX
            )
        if downscale < 1.0 or pass_diameter != diameter:
            logger.info(
                "Downscale-for-inference: scale %.3f (%dx%d -> %dx%d), diameter %s -> %s px",
                downscale, orig_w, orig_h, input_for_model.shape[1],
                input_for_model.shape[0],
                f"{float(diameter):.1f}" if diameter is not None else "auto",
                f"{pass_diameter:.1f}" if pass_diameter is not None else "auto",
            )

    try:
        # For grayscale images in Cellpose, channels=[0, 0] denotes single-channel / grayscale
        eval_result = model.eval(
            input_for_model,
            channels=[0, 0],
            diameter=pass_diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

        # Unpack masks from eval result tuple (masks, flows, styles)
        if isinstance(eval_result, tuple):
            masks = eval_result[0]
        else:
            masks = eval_result

        # Ensure return type is 2D numpy array matching input dimensions
        masks_2d = np.asarray(masks, dtype=np.int32)
        if downscale < 1.0:
            # Nearest-neighbour round-trip: label ids are copied one-to-one back to
            # native resolution (no interpolated ids, no boundary blending).
            masks_2d = cv2.resize(
                masks_2d, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )
        if masks_2d.shape != img_2d.shape:
            logger.warning(
                "Mask shape %s does not match input image shape %s. Reshaping or handling.",
                masks_2d.shape,
                img_2d.shape
            )
        return masks_2d

    except Exception as exc:
        logger.error("Error occurred during Cellpose inference: %s", exc, exc_info=True)
        # Graceful fallback: return empty zero-mask of identical spatial shape
        return np.zeros(img_2d.shape, dtype=np.int32)


def segment_air_bubbles_auto(
    image: np.ndarray,
    min_diam_px: float,
    max_diam_px: float,
    model: models.CellposeModel,
    flow_threshold: float = 0.7,
    cellprob_threshold: float = -1.5,
) -> np.ndarray:
    """
    Specialized two-step detector optimized for dark crater-shaped air bubbles on SEM micrographs.

    Step A — Cellpose dual-pass (direct CLAHE + inverted CLAHE):
        Runs Cellpose on the standard CLAHE image and on the inverted CLAHE image.
        Inverted images reveal deep dark pores as bright 'cells'.  Masks found only on the
        inverted pass are accepted when:
          * IoU overlap with any direct-pass mask is < 40 %
          * Median pixel intensity of the candidate region < image median (darker than background)
        For large bubbles (``air_target_px > 55``) both passes are executed on a
        downscaled copy (``scale_factor = 42 / air_target_px``, clamped to >= 0.35)
        because Cellpose's native receptive field is only ~30 px; the masks are mapped
        back with exact nearest-neighbour interpolation, which keeps every boundary
        pixel and label intact while cutting the Pass-1 pixel count 3x-4x.

    Step B — Dark Pore Fallback (geometric, always at native resolution):
        Finds connected dark regions below the 22nd percentile, removes those already
        covered by Cellpose masks, and keeps components whose equivalent diameter lies in
        [min_diam_px * 0.85, max_diam_px * 1.15] and circularity >= 0.35.

    Args:
        image:               2D uint8 grayscale SEM image (analysis region, footer removed).
        min_diam_px:         Minimum bubble diameter in pixels (from µm / px ratio).
        max_diam_px:         Maximum bubble diameter in pixels (from µm / px ratio).
        model:               Loaded CellposeModel instance (from get_cellpose_model).
        flow_threshold:      Cellpose flow error threshold (default 0.7).
        cellprob_threshold:  Cellpose cell probability threshold (default -1.5, more sensitive).

    Returns:
        np.ndarray: 2D int32 label array (0 = background, 1, 2, … = individual air bubbles).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("segment_air_bubbles_auto expects a 2D grayscale numpy array.")

    min_diam_px = max(float(min_diam_px), 4.0)
    max_diam_px = max(float(max_diam_px), min_diam_px + 4.0)

    # Target diameter: geometric mean, clamped to [25, 220] px
    air_target_px = float(np.clip(
        math.sqrt(max(min_diam_px, 8.0) * max(max_diam_px, 15.0)),
        25.0, 220.0,
    ))
    logger.info(
        "Air bubble target diameter: %.1f px  (min=%.1f, max=%.1f)",
        air_target_px, min_diam_px, max_diam_px,
    )

    # ── Step A: Cellpose dual-pass ──────────────────────────────────────────
    clahe_img = apply_clahe(image)
    inv_img   = np.clip(255 - clahe_img, 0, 255).astype(np.uint8)
    img_median = float(np.median(image))

    # Cellpose's native receptive field is ~30 px, so asking for a >55 px object diameter
    # on a full-resolution micrograph only adds pyramid levels that carry no extra boundary
    # detail.  Downscaling so the target lands at ~42 px reduces the Pass-1 pixel count by
    # 3x-4x; the clamp at 0.35 keeps the target comfortably above the receptive field.
    scale_factor = 1.0
    full_h, full_w = image.shape[:2]
    if air_target_px > 55.0:
        scale_factor = max(0.35, 42.0 / air_target_px)

    if scale_factor < 1.0:
        small_size = (
            max(1, int(round(full_w * scale_factor))),
            max(1, int(round(full_h * scale_factor))),
        )
        clahe_pass = cv2.resize(clahe_img, small_size, interpolation=cv2.INTER_AREA)
        inv_pass = cv2.resize(inv_img, small_size, interpolation=cv2.INTER_AREA)
        pass_diameter = air_target_px * scale_factor
        logger.info(
            "Large-bubble fast path: scale %.3f (%dx%d -> %dx%d), diameter %.1f -> %.1f px",
            scale_factor, full_w, full_h, small_size[0], small_size[1],
            air_target_px, pass_diameter,
        )
    else:
        clahe_pass, inv_pass = clahe_img, inv_img
        pass_diameter = air_target_px

    masks_direct = run_inference(
        model=model,
        image=clahe_pass,
        diameter=pass_diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        enable_clahe=False,  # already applied
    )

    masks_inv = run_inference(
        model=model,
        image=inv_pass,
        diameter=pass_diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        enable_clahe=False,
    )

    # Map both passes back to native resolution with exact nearest-neighbour sampling:
    # label ids are copied one-to-one (no interpolation, no blending of boundaries).
    if scale_factor < 1.0:
        masks_direct = cv2.resize(
            np.asarray(masks_direct, dtype=np.int32),
            (full_w, full_h),
            interpolation=cv2.INTER_NEAREST,
        )
        masks_inv = cv2.resize(
            np.asarray(masks_inv, dtype=np.int32),
            (full_w, full_h),
            interpolation=cv2.INTER_NEAREST,
        )

    # Merge: start with direct-pass masks; add non-overlapping inverted-pass masks
    # v1.7.8: every inverted candidate is evaluated strictly inside its tight bounding
    # box (``ndi.find_objects``). A full-frame ``masks_inv == label`` boolean is 1 byte
    # per pixel (≈2 MB on a 1440×1440 micrograph) *per candidate* -- L candidates meant
    # L full-image allocations plus L full-image scans. The bounding-box slice keeps a
    # single candidate's work at O(bbox) memory and O(bbox) time, so the merge cost now
    # scales with the summed object area instead of ``L × image area``.
    combined = masks_direct.copy()
    n_direct = int(masks_direct.max())
    next_label = n_direct + 1
    n_inv_added = 0

    for inv_idx, inv_slice in enumerate(ndi.find_objects(masks_inv), start=1):
        if inv_slice is None:
            continue

        inv_sub = masks_inv[inv_slice]
        inv_region = inv_sub == inv_idx            # bbox-sized, never full-frame
        area_inv = float(np.count_nonzero(inv_region))
        if area_inv <= 0.0:
            continue

        # Intensity gate: candidate must be darker than image median
        if float(np.median(image[inv_slice][inv_region])) >= img_median:
            continue

        # IoU gate: skip if > 40 % overlap with any already-accepted mask
        combined_sub = combined[inv_slice]
        overlap = float(np.count_nonzero(combined_sub[inv_region]))
        iou = overlap / max(area_inv, 1.0)
        if iou > 0.40:
            continue

        combined_sub[inv_region] = next_label       # view -> writes into ``combined``
        next_label += 1
        n_inv_added += 1

    # ── Step B: Dark Pore Fallback ─────────────────────────────────────────
    blur = cv2.GaussianBlur(image, (9, 9), 0)
    dark_thresh = float(np.percentile(blur, 22))
    dark_binary = (blur.astype(np.float32) <= dark_thresh).astype(np.uint8)

    # Morphological open→close to clean noise and fill thin gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Label connected components. v1.7.8: ``connectedComponentsWithStats`` returns the
    # area and bounding box of every component in the same C pass, so the diameter gate
    # below rejects candidates without ever touching their pixels (previously every
    # component -- including the thousands outside the size window -- triggered a
    # full-frame ``cc_labels == cc_id`` boolean mask).
    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        dark_binary, connectivity=8
    )

    lo_px = min_diam_px * 0.85
    hi_px = max_diam_px * 1.15

    # Components of ``dark_binary`` are pixel-disjoint, so one static coverage mask stays
    # exact while accepted fallback labels are appended to ``combined`` (a fresh dark
    # component can never contain a pixel that another dark component just claimed).
    covered = combined > 0
    n_fallback_added = 0

    for cc_id in range(1, num_cc):
        # O(1) area lookup -> cheap early rejection before any pixel work
        area_px = float(cc_stats[cc_id, cv2.CC_STAT_AREA])
        eq_diam = math.sqrt(4.0 * area_px / math.pi)
        if not (lo_px <= eq_diam <= hi_px):
            continue

        bx = int(cc_stats[cc_id, cv2.CC_STAT_LEFT])
        by = int(cc_stats[cc_id, cv2.CC_STAT_TOP])
        bw = int(cc_stats[cc_id, cv2.CC_STAT_WIDTH])
        bh = int(cc_stats[cc_id, cv2.CC_STAT_HEIGHT])
        cc_slice = (slice(by, by + bh), slice(bx, bx + bw))

        # bbox-sized candidate mask (boundary pixels are kept 1:1, so contours and the
        # circularity/perimeter measurement are identical to the full-frame version)
        cc_region = cc_labels[cc_slice] == cc_id

        # Circularity filter
        contours, _ = cv2.findContours(
            cc_region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area_px / (perimeter ** 2)
        if circularity < 0.35:
            continue

        # Skip if already covered by Cellpose (IoU > 40 %)
        covered_sub = covered[cc_slice]
        overlap = float(np.count_nonzero(covered_sub[cc_region]))
        iou = overlap / max(area_px, 1.0)
        if iou > 0.40:
            continue

        combined_sub = combined[cc_slice]
        combined_sub[cc_region] = next_label          # view -> writes into ``combined``
        next_label += 1
        n_fallback_added += 1

    # v1.7.8 logging fix: ``next_label`` keeps incrementing through Step B, so deriving
    # the counts from it at the end reported ``fallback = next_label - combined.max() - 1
    # == 0`` and inflated ``inv_added`` by every fallback bubble. The explicit counters
    # tracked in Step A / Step B now report the exact per-pass numbers.
    logger.info(
        "segment_air_bubbles_auto: %d bubbles found (direct=%d, inv_added=%d, fallback=%d)",
        n_direct + n_inv_added + n_fallback_added,
        n_direct,
        n_inv_added,
        n_fallback_added,
    )
    return combined.astype(np.int32)


@st.cache_data(show_spinner="🫧 Running Air Bubble detection (multi-scale + dark crater)...")
def cached_segment_air_bubbles_auto(
    image: np.ndarray,
    min_diam_px: float,
    max_diam_px: float,
    model_type: str = "cyto2",
    custom_model_path: Optional[str] = None,
    flow_threshold: float = 0.7,
    cellprob_threshold: float = -1.5,
) -> np.ndarray:
    """
    Cached wrapper for segment_air_bubbles_auto.
    Uses @st.cache_data so the expensive dual-pass inference is NOT re-run
    when unrelated UI sliders (opacity, contrast, etc.) change.

    Args:
        image:               2D uint8 grayscale analysis image.
        min_diam_px:         Minimum expected bubble diameter in pixels.
        max_diam_px:         Maximum expected bubble diameter in pixels.
        model_type:          Cellpose pretrained model name.
        custom_model_path:   Optional path to custom .pth weights.
        flow_threshold:      Cellpose flow error threshold.
        cellprob_threshold:  Cellpose cell probability threshold.

    Returns:
        np.ndarray: 2D int32 label mask of air bubbles.
    """
    model = get_cellpose_model(model_type=model_type, custom_model_path=custom_model_path)
    return segment_air_bubbles_auto(
        image=image,
        min_diam_px=min_diam_px,
        max_diam_px=max_diam_px,
        model=model,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )


def segment_fat_in_bubbles_deepscan(
    image: np.ndarray,
    bubble_region_mask: np.ndarray,
    base_fat_masks: np.ndarray,
    min_fat_px: float,
    max_fat_px: float,
    model: models.CellposeModel,
    flow_threshold: float = 0.75,
    cellprob_threshold: float = -2.0,
) -> np.ndarray:
    """
    High-speed deep-scan segmentation of fat globules strictly INSIDE air bubble craters.

    Optimization:
      Instead of running heavy inference across the full micrograph, this function:
      1. Extracts connected components of bubbles from bubble_binary.
      2. Crops small bounding-box ROIs around each bubble with pad = 8 px.
      3. Enhances local contrast inside the bubble (p2-p98 stretch, CLAHE, gamma=0.85),
         filling background pixels outside the bubble with the local median.
      4. Runs Cellpose inference on the lightweight cropped ROI.
      5. Strictly clips each found candidate mask to the bubble interior (cand_region = (m == lbl) & (roi_bub > 0)),
         guaranteeing no objects outside the bubble can ever be formed.
      6. Merges non-overlapping candidates meeting the minimum equivalent diameter into result_masks.

    Args:
        image:              2D uint8 grayscale SEM micrograph (analysis region).
        bubble_region_mask: 2D labeled or binary mask of air bubbles.
        base_fat_masks:     2D int32 labeled fat mask from the prior pass.
        min_fat_px:         Minimum expected fat globule diameter in pixels.
        max_fat_px:         Maximum expected fat globule diameter in pixels.
        model:              Loaded CellposeModel instance.
        flow_threshold:     Cellpose flow error threshold (default 0.75).
        cellprob_threshold: Cellpose cell probability threshold (default -2.0).

    Returns:
        np.ndarray: Updated 2D int32 labeled fat mask with newly found bubble-interior
                    fat globules added under new unique IDs.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        return base_fat_masks
    if not isinstance(bubble_region_mask, np.ndarray) or bubble_region_mask.ndim != 2:
        return base_fat_masks

    bubble_binary = (bubble_region_mask > 0).astype(np.uint8)
    if not np.any(bubble_binary):
        return base_fat_masks

    h, w = image.shape[:2]
    result_masks = base_fat_masks.astype(np.int32).copy()
    next_id = int(result_masks.max()) + 1

    min_fat_px = max(float(min_fat_px), 4.0)
    d_small = float(np.clip(min_fat_px * 1.15, 12.0, 28.0))
    min_accept_diam = min_fat_px * 0.7

    # Gamma correction lookup table (gamma=0.85 lifts crater shadow details)
    gamma = 0.85
    lut = np.array([
        int(np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255))
        for i in range(256)
    ], dtype=np.uint8)

    # Connected components of individual bubbles
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(bubble_binary, connectivity=8)
    pad = 8

    for lbl_idx in range(1, num_labels):
        area = int(stats[lbl_idx, cv2.CC_STAT_AREA])
        if area < 50:
            continue

        bx = int(stats[lbl_idx, cv2.CC_STAT_LEFT])
        by = int(stats[lbl_idx, cv2.CC_STAT_TOP])
        bw = int(stats[lbl_idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl_idx, cv2.CC_STAT_HEIGHT])

        # Bounding box with pad = 8 px clamped to image dimensions
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)

        roi_img = image[y1:y2, x1:x2]
        roi_bub = bubble_binary[y1:y2, x1:x2]

        bubble_pixels = roi_img[roi_bub > 0]
        if len(bubble_pixels) < 10:
            continue

        # Percentile stretch [2nd, 98th] to [0, 255]
        p2 = float(np.percentile(bubble_pixels, 2))
        p98 = float(np.percentile(bubble_pixels, 98))
        span = max(p98 - p2, 1.0)
        stretched = np.clip((roi_img.astype(np.float32) - p2) / span * 255.0, 0.0, 255.0).astype(np.uint8)

        # CLAHE with tileGridSize adapted to small crop dimensions (max (8, 8))
        tile_h = max(1, min(8, roi_img.shape[0]))
        tile_w = max(1, min(8, roi_img.shape[1]))
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(tile_w, tile_h))
        enhanced = clahe.apply(stretched)
        enhanced = cv2.LUT(enhanced, lut)

        # Fill background outside bubble with local median of bubble interior pixels
        median_bg = int(np.median(bubble_pixels))
        enhanced[roi_bub == 0] = median_bg

        # Run fast inference on the cropped ROI
        try:
            m_roi = run_inference(
                model=model,
                image=enhanced,
                diameter=d_small,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                enable_clahe=False,
            )
        except Exception as exc:
            logger.warning("Deepscan ROI inference failed for bubble at (%d, %d): %s", bx, by, exc)
            continue

        if m_roi is None or m_roi.max() == 0:
            continue

        m_roi_2d = m_roi.squeeze() if m_roi.ndim == 3 else m_roi
        if m_roi_2d.ndim != 2 or m_roi_2d.shape != roi_bub.shape:
            continue

        roi_existing = result_masks[y1:y2, x1:x2]

        # v1.7.8: one vectorised pass per bubble ROI instead of a full-ROI boolean mask
        # per candidate label. ``np.bincount`` over the labels that actually fall inside
        # the bubble yields the clipped area *and* the overlap with already-accepted fat
        # globules for every label simultaneously, and ``find_objects`` gives the tight
        # bounding box used to write each accepted label back without a full-ROI mask.
        roi_bubble_bool = roi_bub > 0
        if not np.any(roi_bubble_bool):
            continue

        labels_in_bubble = m_roi_2d[roi_bubble_bool]
        if labels_in_bubble.size == 0:
            continue
        if int(labels_in_bubble.min()) < 0:
            labels_in_bubble = np.clip(labels_in_bubble, 0, None)
        n_labels = int(labels_in_bubble.max()) + 1
        if n_labels <= 1:
            continue

        area_per_label = np.bincount(labels_in_bubble, minlength=n_labels).astype(np.float64)
        area_per_label[0] = 0.0        # background is never a candidate
        occupied = roi_existing[roi_bubble_bool] > 0
        overlap_per_label = np.bincount(labels_in_bubble[occupied], minlength=n_labels)

        # Vectorised gates: area >= 4 px, eq. diameter >= min_fat_px * 0.7,
        # overlap with existing fat particles < 30 % (identical thresholds to v1.7.7).
        eq_diam_per_label = np.sqrt(4.0 * area_per_label / math.pi)
        overlap_ratio = overlap_per_label.astype(np.float64) / np.maximum(area_per_label, 1.0)
        candidates = np.flatnonzero(
            (area_per_label >= 4.0)
            & (eq_diam_per_label >= min_accept_diam)
            & (overlap_ratio < 0.30)
        )

        if candidates.size:
            label_slices = ndi.find_objects(m_roi_2d)
            for lbl in candidates:
                lbl_index = int(lbl) - 1
                lbl_slice = label_slices[lbl_index] if lbl_index < len(label_slices) else None
                if lbl_slice is None:
                    continue

                # Strictly clip candidate mask to the bubble interior (bbox-sized)
                cand_region = (m_roi_2d[lbl_slice] == lbl) & (roi_bub[lbl_slice] > 0)
                if not cand_region.any():
                    continue

                # Accept candidate with new unique next_id
                existing_sub = roi_existing[lbl_slice]     # view -> writes into the ROI
                existing_sub[cand_region] = next_id
                next_id += 1

        result_masks[y1:y2, x1:x2] = roi_existing

    logger.info(
        "segment_fat_in_bubbles_deepscan: %d new fat globules found inside bubbles.",
        int(result_masks.max()) - int(base_fat_masks.max()),
    )
    return result_masks


@st.cache_data(show_spinner="🔬 Deep-scanning fat inside air bubbles...")
def cached_segment_fat_in_bubbles_deepscan(
    image: np.ndarray,
    bubble_region_mask: np.ndarray,
    base_fat_masks: np.ndarray,
    min_fat_px: float,
    max_fat_px: float,
    model_type: str = "cyto2",
    custom_model_path: Optional[str] = None,
    flow_threshold: float = 0.75,
    cellprob_threshold: float = -2.0,
) -> np.ndarray:
    """
    Cached wrapper for segment_fat_in_bubbles_deepscan.
    Uses @st.cache_data so re-runs are NOT triggered by unrelated UI changes.
    """
    model = get_cellpose_model(model_type=model_type, custom_model_path=custom_model_path)
    return segment_fat_in_bubbles_deepscan(
        image=image,
        bubble_region_mask=bubble_region_mask,
        base_fat_masks=base_fat_masks,
        min_fat_px=min_fat_px,
        max_fat_px=max_fat_px,
        model=model,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )


class CellposeSegmenter:
    """
    Object-oriented wrapper for Cellpose segmentation in SEM / material phase analysis.
    """

    def __init__(
        self,
        model_type: str = "cyto2",
        custom_model_path: Optional[str] = None,
        diameter: Optional[float] = None,
    ) -> None:
        """
        Initialize the segmenter with specified model architecture and device.

        Args:
            model_type: Name of the pre-trained model ('cyto2', 'nuclei', or 'Custom Model File...').
            custom_model_path: Optional filesystem path to user-supplied custom .pth weights.
            diameter: Default diameter parameter for particle segmentation.
        """
        self.model_type = model_type
        self.custom_model_path = custom_model_path
        self.diameter = diameter
        self.model: models.CellposeModel = get_cellpose_model(
            model_type=self.model_type,
            custom_model_path=self.custom_model_path,
        )

    def segment(
        self,
        image: np.ndarray,
        diameter: Optional[float] = None,
        flow_threshold: float = 0.7,
        cellprob_threshold: float = -1.0,
        enable_clahe: bool = True,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Segment a grayscale SEM image.

        Args:
            image: 2D numpy array representing grayscale image.
            diameter: Particle diameter override. If None, uses default diameter.
            flow_threshold: Flow error threshold (default 0.7).
            cellprob_threshold: Cell probability threshold (default -1.0).
            enable_clahe: Whether to apply CLAHE contrast enhancement.
            use_cache: Whether to use Streamlit data cache for inference.

        Returns:
            np.ndarray: 2D labeled mask array where 0 is background and positive ints are objects.
        """
        effective_diameter = diameter if diameter is not None else self.diameter
        if use_cache:
            return segment_image_cached(
                image=image,
                model_type=self.model_type,
                diameter=effective_diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                enable_clahe=enable_clahe,
            )
        return run_inference(
            model=self.model,
            image=image,
            diameter=effective_diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            enable_clahe=enable_clahe,
        )


@st.cache_resource(show_spinner="Loading Interactive SAM Predictor...")
def get_sam_predictor(sam_model_type: str = "vit_b"):
    """
    Load and cache a Segment Anything (SAM) ``SamPredictor`` for one backbone.

    - ``vit_b`` (375 MB) is fast and ideal for interactive use.
    - ``vit_l`` (1.25 GB) and ``vit_h`` (2.56 GB) trade loading time/RAM for
      noticeably better boundary precision on faint, blurry or crystallized edges.

    The checkpoint is stored under ``get_models_cache_dir()/{filename}``; missing
    or truncated files are downloaded in 8 MB chunks into a ``.part`` file and
    atomically moved into place (see :func:`ensure_sam_checkpoint`). The device
    follows :func:`get_sam_device`: CUDA when an NVIDIA GPU is present, Apple
    Silicon MPS when available, CPU otherwise. A failed MPS load degrades to CPU
    instead of disabling Smart Box.

    Args:
        sam_model_type: One of ``"vit_b"``, ``"vit_l"``, ``"vit_h"``.

    Returns:
        SamPredictor | None: A ready-to-use predictor, or None if SAM is unavailable.
    """
    sam_type = _normalize_sam_model_type(sam_model_type)

    try:
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    except Exception as exc:
        logger.warning("segment_anything is not installed; SAM interactive segmentation disabled: %s", exc)
        return None

    try:
        ckpt_path = ensure_sam_checkpoint(sam_type)
        if ckpt_path is None:
            return None

        sam_device = get_sam_device()
        sam = sam_model_registry[sam_type](checkpoint=str(ckpt_path))
        try:
            sam.to(device=sam_device)
        except Exception as exc:
            if sam_device.type == "cpu":
                raise
            logger.warning(
                "SAM could not move to %s (%s); falling back to CPU.",
                sam_device.type,
                exc,
            )
            sam_device = torch.device("cpu")
            sam.to(device=sam_device)
        predictor = SamPredictor(sam)
        # Book-keeping used by the ROI embedding cache / CPU fallback.
        predictor._pp_force_cpu = sam_device.type == "cpu"
        predictor._pp_cached_hash = None
        predictor._pp_cached_region = None
        logger.info("SAM SamPredictor (%s) initialized on %s.", sam_type, sam_device.type)
        return predictor
    except Exception as exc:
        logger.warning("Failed to initialize SAM predictor (%s): %s", sam_type, exc)
        return None


def get_cached_sam_predictor_for_roi(
    image_2d: np.ndarray,
    roi_xyxy: Optional[Tuple[float, float, float, float]] = None,
    sam_model_type: str = "vit_b",
):
    """
    Return a SAM predictor whose image embeddings cover ``roi_xyxy`` only.

    SAM's encoder always resizes its input to 1024 px, so encoding the full
    micrograph costs the same as encoding a crop while forcing every drawn box to
    pay that price again. This helper:

      * encodes only the padded ROI requested by the caller,
      * reuses the cached embedding whenever the requested ROI lies inside the
        region that is already encoded (neighbouring boxes / a pre-warmed union),
      * falls back to CPU transparently when an MPS kernel raises, then retries once.

    Args:
        image_2d: 2D grayscale image array.
        roi_xyxy: Box that must be covered by the embedding; None encodes the whole image.
        sam_model_type: SAM backbone key (``"vit_b"``, ``"vit_l"``, ``"vit_h"``).

    Returns:
        Tuple[SamPredictor, Tuple[int, int, int, int]] | None: the predictor and the
        encoded region ``(x0, y0, x1, y1)`` in full-image pixels, or None.
    """
    predictor = get_sam_predictor(sam_model_type=sam_model_type)
    if predictor is None:
        return None

    if image_2d.ndim == 3:
        image_2d = image_2d[:, :, 0]
    full_h, full_w = image_2d.shape[:2]
    if full_h <= 0 or full_w <= 0:
        return None

    region = _sam_roi_bounds(roi_xyxy, (full_h, full_w))
    try:
        img_hash = hash(image_2d.tobytes())
    except Exception:  # pragma: no cover - defensive
        img_hash = None

    cached_region = getattr(predictor, "_pp_cached_region", None)
    if (
        img_hash is not None
        and getattr(predictor, "_pp_cached_hash", None) == img_hash
        and cached_region is not None
    ):
        cx0, cy0, cx1, cy1 = cached_region
        if region[0] >= cx0 and region[1] >= cy0 and region[2] <= cx1 and region[3] <= cy1:
            return predictor, cached_region

    x0, y0, x1, y1 = region
    crop = image_2d[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    try:
        clahe = apply_clahe(crop)
        rgb = cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)
        try:
            predictor.set_image(rgb)
        except Exception as exc:
            # Metal op unsupported on this build -> move the whole model to CPU once.
            if predictor.device.type == "cpu":
                raise
            logger.warning(
                "SAM ROI encoding failed on %s (%s); retrying on CPU.",
                predictor.device,
                exc,
            )
            if not _sam_move_to_cpu(predictor):
                raise
            predictor.set_image(rgb)

        predictor._pp_cached_hash = img_hash
        predictor._pp_cached_region = region
        return predictor, region
    except Exception as exc:
        logger.warning("Failed to compute SAM image embeddings: %s", exc)
        return None


def _sam_roi_bounds(
    roi_xyxy: Optional[Tuple[float, float, float, float]],
    shape: Tuple[int, int],
    pad_fraction: float = SAM_ROI_PAD_FRACTION,
    min_side: int = SAM_ROI_MIN_SIDE,
) -> Tuple[int, int, int, int]:
    """
    Return the padded, image-clamped ROI that SAM should encode for one box.

    The box is expanded by ``pad_fraction`` of its own size on every side (40 % by
    default, so the surrounding matrix context stays visible), grown to at least
    ``min_side`` x ``min_side`` px around its centre, and finally clamped to the
    image bounds.

    Args:
        roi_xyxy: Box as ``(x1, y1, x2, y2)`` in image pixels, or None for the whole image.
        shape: Image shape ``(height, width)``.
        pad_fraction: Fraction of the box size added on each side.
        min_side: Minimum ROI side length in pixels.

    Returns:
        Tuple[int, int, int, int]: ``(x0, y0, x1, y1)`` integer ROI bounds.
    """
    height, width = int(shape[0]), int(shape[1])
    if roi_xyxy is None:
        return 0, 0, width, height
    try:
        x1, y1, x2, y2 = (float(value) for value in roi_xyxy)
    except (TypeError, ValueError):
        return 0, 0, width, height
    if not all(np.isfinite([x1, y1, x2, y2])) or x2 <= x1 or y2 <= y1:
        return 0, 0, width, height

    pad_x = (x2 - x1) * float(pad_fraction)
    pad_y = (y2 - y1) * float(pad_fraction)
    fx0, fx1 = x1 - pad_x, x2 + pad_x
    fy0, fy1 = y1 - pad_y, y2 + pad_y

    if (fx1 - fx0) < float(min_side):
        grow = (float(min_side) - (fx1 - fx0)) / 2.0
        fx0 -= grow
        fx1 += grow
    if (fy1 - fy0) < float(min_side):
        grow = (float(min_side) - (fy1 - fy0)) / 2.0
        fy0 -= grow
        fy1 += grow

    x0 = int(max(0, math.floor(fx0)))
    y0 = int(max(0, math.floor(fy0)))
    x1i = int(min(width, math.ceil(fx1)))
    y1i = int(min(height, math.ceil(fy1)))
    if x1i <= x0:
        x1i = min(width, x0 + 1)
    if y1i <= y0:
        y1i = min(height, y0 + 1)
    return x0, y0, x1i, y1i


def get_cached_sam_predictor_for_image(
    image_2d: np.ndarray,
    sam_model_type: str = "vit_b",
):
    """
    Return a SAM predictor with embeddings for the whole ``image_2d``.

    Thin wrapper around :func:`get_cached_sam_predictor_for_roi` kept for callers
    that genuinely need a full-field embedding; the interactive Smart Box path
    always requests a local ROI instead.

    Args:
        image_2d: 2D grayscale image array.
        sam_model_type: SAM backbone key (``"vit_b"``, ``"vit_l"``, ``"vit_h"``).

    Returns:
        SamPredictor | None: Predictor with embeddings set for the image, or None.
    """
    state = get_cached_sam_predictor_for_roi(image_2d, None, sam_model_type)
    return state[0] if state is not None else None


def prefetch_sam_roi(
    image_2d: np.ndarray,
    roi_xyxy: Optional[Tuple[float, float, float, float]] = None,
    sam_model_type: str = "vit_b",
) -> bool:
    """
    Warm the SAM embedding cache for one ROI (best effort, never raises).

    Args:
        image_2d: 2D grayscale image array.
        roi_xyxy: Region that must be covered, or None for the whole image.
        sam_model_type: SAM backbone key.

    Returns:
        bool: True when an embedding covering ``roi_xyxy`` is available.
    """
    try:
        return get_cached_sam_predictor_for_roi(image_2d, roi_xyxy, sam_model_type) is not None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("SAM ROI prefetch failed: %s", exc)
        return False


def prefetch_sam_roi_union(
    image_2d: np.ndarray,
    boxes: list,
    sam_model_type: str = "vit_b",
    budget_factor: float = SAM_ROI_UNION_BUDGET,
) -> bool:
    """
    Encode ONE union embedding for several *close* boxes (Smart Box batch).

    When several rectangles are drawn around neighbouring objects their padded
    ROIs overlap heavily. Encoding the tight union once and serving every box from
    that embedding turns N encoder passes into one — only the mask decoder then
    runs per box (~0.02 s instead of ~0.4 s).

    Scattered boxes (whose union is far larger than a single ROI) are left alone so
    no box is ever served from a needlessly downscaled embedding.

    Args:
        image_2d: 2D grayscale image array.
        boxes: Iterable of ``(x1, y1, x2, y2)`` boxes in image pixels.
        sam_model_type: SAM backbone key.
        budget_factor: Maximum allowed ``union_area / largest_box_roi_area``.

    Returns:
        bool: True when a single union embedding was computed (or already cached).
    """
    if image_2d is None or getattr(image_2d, "size", 0) == 0:
        return False
    img_2d = image_2d[:, :, 0] if image_2d.ndim == 3 else image_2d
    shape = img_2d.shape[:2]

    rois = [_sam_roi_bounds(box, shape) for box in (boxes or [])]
    rois = [roi for roi in rois if roi[2] > roi[0] and roi[3] > roi[1]]
    if len(rois) < 2:
        return False

    ux0 = min(roi[0] for roi in rois)
    uy0 = min(roi[1] for roi in rois)
    ux1 = max(roi[2] for roi in rois)
    uy1 = max(roi[3] for roi in rois)
    union_area = float(max(1, ux1 - ux0) * max(1, uy1 - uy0))
    largest_area = max(float((roi[2] - roi[0]) * (roi[3] - roi[1])) for roi in rois)
    if largest_area <= 0 or union_area > float(budget_factor) * largest_area:
        return False

    return prefetch_sam_roi(img_2d, (ux0, uy0, ux1, uy1), sam_model_type)


def _box_prompt_corners(
    x1: float, y1: float, x2: float, y2: float, dx: float, dy: float
) -> np.ndarray:
    """4 inset corners of the user box — they are guaranteed to be background."""
    return np.array(
        [
            [x1 + dx, y1 + dy],
            [x2 - dx, y1 + dy],
            [x1 + dx, y2 - dy],
            [x2 - dx, y2 - dy],
        ],
        dtype=np.float32,
    )


def _mask_corner_hits(mask: np.ndarray, corners: np.ndarray, pad: int = 2) -> int:
    """Count how many box corners lie inside the mask (small tolerance window)."""
    h, w = mask.shape[:2]
    hits = 0
    for px, py in corners:
        cx = int(np.clip(round(float(px)), 0, w - 1))
        cy = int(np.clip(round(float(py)), 0, h - 1))
        y0, y1_ = max(0, cy - pad), min(h, cy + pad + 1)
        x0, x1_ = max(0, cx - pad), min(w, cx + pad + 1)
        if bool(mask[y0:y1_, x0:x1_].any()):
            hits += 1
    return hits


def _is_box_like_mask(mask: np.ndarray, corners: np.ndarray) -> bool:
    """A mask touching all 4 box corners is the drawn rectangle, not a particle."""
    return _mask_corner_hits(mask, corners, pad=2) >= 4


def _build_edge_hint(mask_layer: np.ndarray) -> np.ndarray:
    """Binary mask (full image) -> dilated boundary map used as a rim-snapping bonus."""
    edge_u8 = (mask_layer > 0).astype(np.uint8) * 255
    if not np.any(edge_u8):
        return np.zeros(mask_layer.shape[:2], dtype=np.uint8)
    grad = cv2.morphologyEx(edge_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return cv2.dilate(grad, np.ones((5, 5), np.uint8))


def _radial_rim_mask(
    image_2d: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    edge_hint: Optional[np.ndarray] = None,
    num_rays: int = 72,
    edge_bonus: float = 40.0,
) -> np.ndarray:
    """
    Elliptical Rim Snapper: 72 rays from the box center, each searching a brightness
    gradient peak strictly inside the annular corridor [0.62 * r_ell, 1.05 * r_ell].

    The ellipse inscribed in the user box already hugs the true object wall, therefore
    the returned star-shaped polygon always follows the crater/globule rim and can
    never collapse into the drawn rectangle. A circular median filter (size=5, wrap)
    removes single-ray spikes while keeping the loop closed.

    Args:
        image_2d:  2D grayscale image.
        box_xyxy:  Bounding box (x1, y1, x2, y2).
        edge_hint: Optional binary boundary hint (SAM/Cellpose edges) for a score bonus.
        num_rays:  Number of radial rays (default: 72).
        edge_bonus: Score bonus applied on edge-hint pixels.

    Returns:
        np.ndarray: 2D uint8 binary rim mask (0/1) with the image shape.
    """
    h, w = image_2d.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return np.zeros((h, w), dtype=np.uint8)

    cx_box = (x1 + x2) / 2.0
    cy_box = (y1 + y2) / 2.0
    rx = max(bw / 2.0, 0.5)
    ry = max(bh / 2.0, 0.5)

    # Smoothed brightness gradient (CLAHE + Gaussian -> Sobel), normalized to 0..255
    enhanced = apply_clahe(image_2d)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.GaussianBlur(cv2.magnitude(gx, gy), (3, 3), 0)
    g_max = float(gmag.max())
    if g_max > 1e-6:
        gmag = gmag / g_max * 255.0

    if edge_hint is None or edge_hint.shape[:2] != (h, w):
        edge_hint = np.zeros((h, w), dtype=np.uint8)

    thetas = np.linspace(0.0, 2.0 * np.pi, int(max(num_rays, 8)), endpoint=False)
    radii = np.zeros(len(thetas), dtype=np.float32)

    for i, theta in enumerate(thetas):
        cos_t, sin_t = float(math.cos(theta)), float(math.sin(theta))
        denom = math.sqrt((ry * cos_t) ** 2 + (rx * sin_t) ** 2)
        if denom <= 1e-6:
            radii[i] = min(rx, ry)
            continue

        r_ell = (rx * ry) / denom
        r_lo = max(1.0, 0.62 * r_ell)
        r_hi = max(r_lo + 1.0, 1.05 * r_ell)
        r_samples = np.arange(r_lo, r_hi + 0.25, 0.5, dtype=np.float32)
        if r_samples.size == 0:
            radii[i] = r_ell
            continue

        xs = cx_box + r_samples * cos_t
        ys = cy_box + r_samples * sin_t
        valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
        if not np.any(valid):
            radii[i] = r_ell
            continue

        xi = np.clip(np.round(xs[valid]).astype(np.int32), 0, w - 1)
        yi = np.clip(np.round(ys[valid]).astype(np.int32), 0, h - 1)
        r_valid = r_samples[valid]

        score = (
            gmag[yi, xi]
            + np.where(edge_hint[yi, xi] > 0, edge_bonus, 0.0)
            - 0.45 * np.abs(r_valid - 0.92 * r_ell)
        )
        radii[i] = float(r_valid[int(np.argmax(score))])

    radii = ndi.median_filter(radii, size=5, mode="wrap")

    rim_pts = np.column_stack(
        [cx_box + radii * np.cos(thetas), cy_box + radii * np.sin(thetas)]
    )
    rim_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(rim_mask, [np.round(rim_pts).astype(np.int32)], 1)
    return rim_mask


def _merge_pass_labels(
    combined: np.ndarray,
    occupied: np.ndarray,
    masks: np.ndarray,
    next_id: int,
    min_area_px: int = 12,
    max_overlap: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Merge one Cellpose label array into the running multi-scale mask.

    Because ``masks`` is a single label array, two of its labels can never share a
    pixel, so the per-label overlap with ``occupied`` is exactly
    ``bincount(masks[occupied])`` - one gather instead of N boolean scans.  For the
    same reason accepted masks must not be folded into ``occupied`` (they cannot
    overlap each other), which keeps the histogram exact.

    Args:
        combined: Running output label mask (native resolution).
        occupied: Running footprint canvas of the accepted objects.
        masks: Label array returned by one Cellpose pass.
        next_id: Label id the next accepted object receives.
        min_area_px: Smallest accepted object area, in pixels.
        max_overlap: Rejection threshold on the overlap fraction with ``occupied``.

    Returns:
        Tuple[np.ndarray, np.ndarray, int, int]: (combined, occupied, next_id, added).
    """
    if masks is None or masks.size == 0:
        return combined, occupied, next_id, 0
    n_labels = int(masks.max())
    if n_labels <= 0:
        return combined, occupied, next_id, 0

    label_area = np.bincount(masks.ravel(), minlength=n_labels + 1)
    occupied_any = bool(occupied.any())
    if occupied_any:
        overlap_area = np.bincount(masks[occupied], minlength=n_labels + 1)
    else:
        overlap_area = np.zeros(n_labels + 1, dtype=np.int64)
    area_f = label_area.astype(np.float64)
    keep = (
        (label_area >= int(min_area_px))
        & ((label_area - overlap_area) >= int(min_area_px))
        & (overlap_area / np.maximum(area_f, 1.0) < float(max_overlap))
    )
    keep[0] = False
    added = 0
    if not keep.any():
        return combined, occupied, next_id, 0

    for lbl, sl in enumerate(ndi.find_objects(masks), start=1):
        if sl is None or not keep[lbl]:
            continue
        clean = masks[sl] == lbl
        if occupied_any:
            # Reject internal crystal shards touching an accepted large globule
            clean &= ~occupied[sl]
        combined[sl][clean] = next_id
        next_id += 1
        added += 1
    return combined, occupied, next_id, added


# ---------------- SMALL-DROPLET DARK-RIM RECOVERY (v1.8.7) ----------------
# Wide-field Light Microscopy (e.g. 10x with a 100 um scale bar) renders small
# droplets as a *dark refractive ring with a bright core*.  ``smooth_internal_crystals()``
# blurs an 8-22 px droplet away before Cellpose sees it, and a ring-shaped object is not
# a blob at all, so the network cannot grow a mask over it.  v1.8.7 answered that with a
# third ("fine") Cellpose pass on a lightly denoised *inverted* copy; v1.8.8 removed that
# pass again -- asking for 14-22 px, i.e. below Cellpose's 30 px receptive field, made
# ``eval`` upsample every frame and cost 174 s per slider change.  The purely
# morphological dark-ring detector below is what recovers these droplets now: ~25 ms, at
# native resolution, with no neural network involved.
DARK_RING_MIN_DIAM_PX = 6.0
DARK_RING_MAX_DIAM_FRACTION = 0.85
DARK_RING_MIN_CIRCULARITY = 0.62
DARK_RING_MIN_SOLIDITY = 0.80
DARK_RING_MAX_OVERLAP = 0.20
# A black-hat response below this many grey levels is sensor noise, not a ring.
DARK_RING_MIN_RESPONSE = 6
# A real brightfield droplet is a dark rim around a *bright* core: the enclosed core
# must be at least this many grey levels brighter than the rim it sits in.
DARK_RING_MIN_CONTRAST = 6
# ... and it must fill at least this fraction of the droplet disc (a solid blob, i.e. a
# crystal shard or a noise peak, has no enclosed core at all).
DARK_RING_MIN_CORE_FRACTION = 0.08
DARK_RING_MAX_KERNEL_PX = 61
DARK_RING_MIN_AREA_PX = 12


def _is_round_droplet(
    region: np.ndarray,
    min_circularity: float = DARK_RING_MIN_CIRCULARITY,
    min_solidity: float = DARK_RING_MIN_SOLIDITY,
) -> bool:
    """Return True when a binary patch is a compact round droplet (circ + solidity)."""
    region_u8 = region.astype(np.uint8, copy=False)
    if not np.any(region_u8):
        return False
    contours, _ = cv2.findContours(
        region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return False
    perimeter = float(cv2.arcLength(contour, True))
    circularity = (
        (4.0 * math.pi * area) / (perimeter ** 2) if perimeter > 0.0 else 0.0
    )
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0.0 else 0.0
    return circularity >= min_circularity and solidity >= min_solidity


def _split_touching_droplets(region: np.ndarray, min_diam_px: float) -> list:
    """
    Split one merged pair of touching droplets with a distance-transform partition.

    A single round droplet has exactly one distance-transform maximum, so the patch is
    returned untouched; only patches with two or more well separated maxima are cut
    into a nearest-peak (Voronoi) partition.  Everything runs on the candidate's own
    bounding box, so the cost is bounded by the droplet size, not the frame size.

    Args:
        region: Binary patch holding one candidate component (bbox sized).
        min_diam_px: Smallest droplet diameter of interest (peak separation scale).

    Returns:
        list: One binary patch per droplet (``[region]`` when nothing was split).
    """
    if int(np.count_nonzero(region)) < 4 * max(1.0, float(min_diam_px)):
        return [region]
    dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
    size = max(3, int(round(float(min_diam_px))))
    peaks = (dist > 0.0) & (dist >= ndi.maximum_filter(dist, size=size))
    peak_labels, n_peaks = ndi.label(peaks)
    if n_peaks < 2:
        return [region]
    _, (iy, ix) = ndi.distance_transform_edt(~peaks, return_indices=True)
    nearest = peak_labels[iy, ix]
    parts = []
    for peak_id in np.unique(nearest[region]):
        part = region & (nearest == peak_id)
        if part.any():
            parts.append(part)
    return parts if len(parts) > 1 else [region]

def _recover_dark_rim_droplets(
    image: np.ndarray,
    base_diameter_px: float,
    mask: np.ndarray,
    min_diam_px: float = DARK_RING_MIN_DIAM_PX,
    max_diam_fraction: float = DARK_RING_MAX_DIAM_FRACTION,
    max_overlap: float = DARK_RING_MAX_OVERLAP,
) -> Tuple[np.ndarray, int]:
    """
    Fast morphological catcher for the small dark-rimmed droplets Cellpose misses.

    Runs entirely on the background the AI passes left unclaimed and finishes in a few
    tens of milliseconds on a multi-megapixel frame:

    1. A ``MORPH_BLACKHAT`` (closing - source) with a square kernel sized for the
       largest droplet of interest isolates *dark structures smaller than the kernel*,
       i.e. the refractive rings, while the bright matrix and the already accepted
       globules contribute nothing.  A square kernel is used on purpose: OpenCV
       evaluates it in O(1) per pixel (separable), where an ellipse costs 3x-6x more.
    2. A 3x3 median kills salt noise and a fixed grey-level floor rejects the rest.
    3. ``connectedComponentsWithStats`` gives area and bounding box for every blob in
       one C pass; only blobs in the ``6 px .. 0.85 * base_diameter_px`` equivalent
       diameter window are kept.
    4. Each candidate is hole-filled on its own bounding box (a bright core inside a
       dark ring becomes a full disc), touching pairs are cut with a distance-transform
       Voronoi partition, and a droplet is accepted only when it is compact
       (``circularity >= 0.62``, ``solidity >= 0.80``) and overlaps the existing mask by
       at most ``max_overlap``.

    Args:
        image: 2D grayscale Light Microscopy frame (uint8 or any numeric dtype).
        base_diameter_px: Typical droplet diameter in pixels (from the sidebar scale).
        mask: Label mask produced by the AI passes (``0`` = background). Written in
            place when it is a writable array.
        min_diam_px: Smallest equivalent diameter accepted, in pixels.
        max_diam_fraction: Largest accepted diameter as a fraction of
            ``base_diameter_px``.
        max_overlap: Rejection threshold on the overlap fraction with ``mask > 0``.

    Returns:
        Tuple[np.ndarray, int]: the mask with the recovered droplets appended using
        unique label ids, and the number of droplets that were added.
    """
    if mask is None or image is None:
        return mask, 0
    img_2d = image.squeeze() if image.ndim == 3 else image
    if img_2d.ndim != 2 or img_2d.size == 0 or mask.shape[:2] != img_2d.shape[:2]:
        return mask, 0

    if img_2d.dtype != np.uint8:
        min_v, max_v = float(img_2d.min()), float(img_2d.max())
        img_u8 = np.clip(
            ((img_2d - min_v) / max(max_v - min_v, 1e-6)) * 255.0, 0, 255
        ).astype(np.uint8)
    else:
        img_u8 = img_2d

    max_diam = float(
        np.clip(
            max(float(max_diam_fraction) * float(base_diameter_px), float(min_diam_px) + 2.0),
            float(min_diam_px) + 2.0,
            400.0,
        )
    )
    # A dark rim is only 1-2 px thick, so the size gates are applied to the *bounding
    # box* first (which equals the droplet diameter) and to the hole-filled area later.
    min_area = max(DARK_RING_MIN_AREA_PX, math.pi * ((float(min_diam_px) - 1.0) ** 2) / 4.0)
    max_area = math.pi * (max_diam ** 2) / 4.0

    # 1.-3. Black-hat rings -> adaptive threshold -> one C-pass statistics table.
    ksize = int(np.clip(round(min(max_diam, DARK_RING_MAX_KERNEL_PX - 2)) + 2, 15,
                        DARK_RING_MAX_KERNEL_PX)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    response = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, kernel)
    # A 3x3 *Gaussian* (not median!) suppresses single-pixel noise while keeping the
    # 1-2 px rim itself: a 3x3 median would erase a thin ring completely.
    response = cv2.GaussianBlur(response, (3, 3), 0)
    # Adaptive threshold, i.e. the rim must stand DARK_RING_MIN_RESPONSE grey levels
    # above its *own* neighbourhood.  A global floor cannot work here: the closing of a
    # large kernel sits a few grey levels above the local mean, and on a noisy matrix
    # that offset alone lifts every background pixel over a fixed threshold.
    block = int(np.clip((ksize // 2) | 1, 15, DARK_RING_MAX_KERNEL_PX)) | 1
    ring_u8 = cv2.adaptiveThreshold(
        response,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block,
        -float(DARK_RING_MIN_RESPONSE),
    )
    ring_binary = (ring_u8 > 0) & (response >= DARK_RING_MIN_RESPONSE)
    del response, ring_u8, kernel

    occupied = mask > 0
    ring_binary &= ~occupied

    n_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(
        ring_binary.view(np.uint8), connectivity=8
    )
    if n_blobs <= 1:
        return mask, 0
    areas = stats[:, cv2.CC_STAT_AREA]
    spans = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = (
        (areas >= 8)
        & (spans >= max(4, int(round(float(min_diam_px))) - 2))
        & (spans <= int(round(max_diam)) + 2)
    )
    keep[0] = False
    candidates = np.flatnonzero(keep)
    if candidates.size == 0:
        return mask, 0

    out = mask if getattr(mask.flags, "writeable", True) else mask.copy()
    next_id = int(out.max()) + 1 if out.size else 1
    if next_id < 1:
        next_id = 1
    added = 0

    for blob_id in candidates:
        left = int(stats[blob_id, cv2.CC_STAT_LEFT])
        top = int(stats[blob_id, cv2.CC_STAT_TOP])
        width = int(stats[blob_id, cv2.CC_STAT_WIDTH])
        height = int(stats[blob_id, cv2.CC_STAT_HEIGHT])
        sl = (slice(top, top + height), slice(left, left + width))

        # 4. Ring signature: a brightfield droplet is a dark rim around a bright core,
        #    so the enclosed core must exist and be brighter than the rim.  Without this
        #    test a crystal striation fragment or a noise peak (a *solid* blob) would be
        #    accepted as soon as it happens to be roughly round.
        component = labels[sl] == blob_id
        if not component.any():
            continue
        ring_area = int(np.count_nonzero(component))
        filled = ndi.binary_fill_holes(component)
        filled_area = int(np.count_nonzero(filled))
        core = filled & ~component
        core_area = filled_area - ring_area
        if core_area < max(2, int(DARK_RING_MIN_CORE_FRACTION * filled_area)):
            continue
        image_sub = img_u8[sl]
        core_contrast = float(image_sub[core].mean()) - float(image_sub[component].mean())
        if core_contrast < DARK_RING_MIN_CONTRAST:
            continue

        component = filled
        occupied_sub = occupied[sl]
        combine_sub = out[sl]

        for part in _split_touching_droplets(component, min_diam_px):
            part_area = int(np.count_nonzero(part))
            if part_area < min_area or part_area > max_area:
                continue
            overlap = int(np.count_nonzero(part & occupied_sub)) / float(part_area)
            if overlap > float(max_overlap):
                continue
            if not _is_round_droplet(part):
                continue
            combine_sub[part] = next_id
            next_id += 1
            added += 1

    if added:
        logger.info(
            "Dark-rim recovery appended %d small droplet(s) (diam %.1f-%.1f px)",
            added, float(min_diam_px), max_diam,
        )
    return out, added


@st.cache_data(show_spinner="❄️ Running Multi-Scale Crystallized Globule AI segmentation...")
def cached_segment_crystallized_multiscale(

    image: np.ndarray,
    base_diameter_px: float,
    model_name: str = "cyto2",
    custom_model_path: Optional[str] = None,
    flow_threshold: float = 0.7,
    cellprob_threshold: float = -1.0,
    enable_clahe: bool = False,
    smooth_strength: int = 3,
    invert_brightfield: bool = False,
) -> np.ndarray:
    """
    Two-scale hierarchical segmentation for polydisperse crystallized emulsions:
    1. Coarse Pass (1.75x diameter, stronger crystal smoothing): captures large crystallized
       globules intact without splitting along internal crystal needles.
    2. Standard Pass (1.0x diameter): captures medium/small globules in the remaining matrix.
    3. Merges both passes: large globules take priority; small detections overlapping >25%
       with an accepted large globule are discarded as internal crystal shards, and the
       morphological dark-ring catcher appends the tiny dark-rimmed droplets the network
       cannot resolve.

    v1.8.6 - Light Microscopy scale handling:
        ``base_diameter_px`` reaches 80-220 px at 40x/63x/100x, so the coarse pass asks
        for up to 240 px.  ``run_inference`` answers those requests at a 35 px Cellpose
        diameter on a proportionally downscaled frame, which keeps the network budget
        at ~1 Mpx instead of ~600 Mpx on a 3000x3000 frame.  The two label arrays are
        still returned at native resolution, and the merge below is vectorised with
        ``scipy.ndimage.find_objects`` + ``np.bincount`` so the number of full-frame
        scans no longer grows with the droplet count.

    v1.8.8 - two passes only, and never outside [28, 45] px:
        The v1.8.7 third "fine" pass is gone.  It was asked for 14-22 px, i.e. *below*
        Cellpose's 30 px receptive field, so ``eval`` upscaled every axis 1.4x-2.1x on
        top of the two macro passes -- three heavily upsampled network runs that took
        174 s on a 10x frame.  Both remaining passes are now clamped to
        ``CELLPOSE_MIN_INFERENCE_DIAMETER_PX`` (28 px) and mapped by
        ``run_inference(..., allow_downscale=True)`` onto a frame of at most
        ``CELLPOSE_MAX_INFERENCE_SIDE_PX`` (1024 px) at a 28-45 px network scale, so each
        pass costs ~1-1.3 Mpx of network work (single-digit seconds; measured 174 s ->
        9.5 s end-to-end at the worst-case 5-9 um slider value) for ANY "Typical Droplet
        Diameter for AI" value.  The
        small dark-rimmed droplets the fine pass used to find are recovered instead by
        the purely morphological :func:`_recover_dark_rim_droplets` (~25 ms, native
        resolution, no neural network).
    """
    model = get_cellpose_model(model_type=model_name, custom_model_path=custom_model_path)
    img_2d = image.squeeze() if image.ndim == 3 else image

    # Pass A: Large globules (stronger smoothing + 1.75x diameter)
    s_large = max(int(smooth_strength), 2) + 1
    img_large = smooth_internal_crystals(img_2d, strength=s_large)
    if invert_brightfield:
        img_large = cv2.bitwise_not(img_large)
    d_large = float(
        np.clip(base_diameter_px * 1.75, CELLPOSE_MIN_INFERENCE_DIAMETER_PX, 240.0)
    )
    masks_large = run_inference(
        model=model,
        image=img_large,
        diameter=d_large,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        enable_clahe=enable_clahe,
        allow_downscale=True,
    )

    # Pass B: Standard/medium globules (1.0x diameter, crystal-smoothed)
    img_small = smooth_internal_crystals(img_2d, strength=max(int(smooth_strength), 1)) if smooth_strength > 0 else img_2d.copy()
    if invert_brightfield:
        img_small = cv2.bitwise_not(img_small)
    # v1.8.8: the floor is ``CELLPOSE_MIN_INFERENCE_DIAMETER_PX`` (28 px), never 14 px.
    # A 5-9 um target on a 10x frame (0.72 um/px) is only 12-14 px, and handing that to
    # ``CellposeModel.eval`` upscales the whole frame 2.5x per axis (6.25x the pixels).
    d_small = float(
        np.clip(base_diameter_px, CELLPOSE_MIN_INFERENCE_DIAMETER_PX, 180.0)
    )
    masks_small = run_inference(
        model=model,
        image=img_small,
        diameter=d_small,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        enable_clahe=enable_clahe,
        allow_downscale=True,
    )
    del img_small, img_large

    # NOTE (v1.8.8): the v1.8.7 third Cellpose pass ("fine_pass" at 14-22 px on a lightly
    # denoised inverted copy) is intentionally REMOVED.  Below Cellpose's 30 px receptive
    # field that request made ``eval`` upsample the frame 1.4x-2.1x per axis, and three
    # such passes per slider change cost 174 s.  Everything it recovered is now handled by
    # the morphological dark-ring catcher below (~25 ms, native resolution).

    # Merge: keep all hole-filled large globules with solidity >= 0.78 first, then add non-overlapping small globules
    # v1.8.6 - fully vectorised merge.  Per-object work is confined to the tight
    # bounding box returned by ``ndi.find_objects`` (instead of an full-frame
    # ``masks_large == lbl`` boolean allocation per candidate) and every count that
    # used to be a per-object scan comes from one ``np.bincount`` histogram.  On a
    # Light Microscopy frame with 1000+ droplets the small pass therefore needs two
    # full-frame passes in total instead of ~2 * N.
    combined = np.zeros(img_2d.shape[:2], dtype=np.int32)
    occupied = np.zeros(img_2d.shape[:2], dtype=bool)
    next_id = 1

    # ---- Coarse pass: hole-filled large globules with solidity >= 0.76 ----------
    n_large = int(masks_large.max()) if masks_large.size else 0
    if n_large > 0:
        large_area = np.bincount(masks_large.ravel(), minlength=n_large + 1)
        large_slices = ndi.find_objects(masks_large)
        for lbl, sl in enumerate(large_slices, start=1):
            if sl is None or int(large_area[lbl]) < 36:
                continue
            region = (masks_large[sl] == lbl).astype(np.uint8)
            area = float(large_area[lbl])
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            main_cnt = max(cnts, key=cv2.contourArea)
            hull_area = float(cv2.contourArea(cv2.convexHull(main_cnt)))
            solidity = area / max(hull_area, 1.0)
            if solidity < 0.76:
                continue
            filled = np.zeros_like(region)
            cv2.drawContours(filled, [main_cnt], -1, 1, thickness=cv2.FILLED)
            filled_bool = filled.astype(bool)
            combined[sl][filled_bool] = next_id
            occupied[sl] |= filled_bool
            next_id += 1

    # ---- Medium pass: add the globules the coarse pass did not claim ----
    combined, occupied, next_id, n_medium = _merge_pass_labels(
        combined, occupied, masks_small, next_id
    )

    # ---- Residual morphological dark-ring catcher (v1.8.7, now the ONLY small-object
    #      detector since the v1.8.8 removal of the third neural-network pass) --------
    # Anything the two AI passes left in the background: small dark-rimmed brightfield
    # droplets that are too small for the network's receptive field.
    ring_source = cv2.bitwise_not(img_2d) if invert_brightfield else img_2d
    combined, n_recovered = _recover_dark_rim_droplets(
        ring_source, base_diameter_px, combined
    )
    logger.info(
        "Multi-scale merge: %d coarse + %d medium + %d dark-ring = %d objects",
        (next_id - 1) - n_medium, n_medium, n_recovered,
        (next_id - 1) + n_recovered,
    )

    return combined

# ---------------- CLSM SILICA MICROSPHERE SEGMENTATION (v2.0) ----------------
# Confocal laser scanning microscopy (CLSM) of colloidal silica arrives in three
# file flavours, and all three must present the spheres to Cellpose as *solid,
# round, bright discs* on a dark background:
#
#   1. Composite RGB  (``Snap-22.tif``) - green continuous matrix + magenta
#      fluorescent shell.  The silica core is the *negative* (dark) stain inside
#      the green matrix, so ``core_inv = 255 - G`` turns every sphere into a
#      solid bright disc, and the magenta/red shell (``max(R, B)``) sharpens the
#      inter-particle boundaries inside dense clusters.
#   2. Channel c1 - negative-contrast matrix (dark spheres on a bright
#      background), i.e. the stored channel of the very same acquisition.  A
#      plain inversion converts it into bright discs.
#   3. Channel c0 - fluorescent shell / rim only (bright rings on black).  A ring
#      is not a blob, so the enclosed dark core is hole-filled and a
#      distance-transform dome is painted inside it: every hollow ring becomes a
#      solid disc that Cellpose (and the watershed catcher) can measure.
#
# ``CLSM_CHANNEL_*`` are the canonical tokens; the sidebar passes its own labels
# and :func:`resolve_clsm_channel_mode` maps them by keyword, so UI wording can be
# reworded without invalidating the ``@st.cache_data`` entry point.
CLSM_CHANNEL_AUTO = "auto"
CLSM_CHANNEL_COMPOSITE = "composite"
CLSM_CHANNEL_C1 = "c1"
CLSM_CHANNEL_C0 = "c0"

# ``median > 90`` grey levels = bright background (c1 negative contrast matrix);
# ``median <= 90`` = dark background (c0 fluorescent shells on black).
CLSM_AUTO_BRIGHT_BACKGROUND_MEDIAN = 90.0
# Residual catcher band: 5 px up to 0.8 x the Cellpose target diameter.
CLSM_MIN_MICROSPHERE_DIAM_PX = 5.0
CLSM_RESIDUAL_DIAM_FRACTION = 0.80
CLSM_RESIDUAL_MIN_CIRCULARITY = 0.55
CLSM_RESIDUAL_MIN_SOLIDITY = 0.70
CLSM_RESIDUAL_MAX_OVERLAP = 0.25
# Anything smaller than this is sensor noise / a diffraction speck, not a sphere.
CLSM_MAIN_MIN_AREA_PX = 9

# ---------------- v2.0.0 CLUSTER SPLITTING (magenta shell = separator) --------
# In the RGB composite the magenta shell is BRIGHT exactly where two silica cores
# touch, so adding it to the core image glued neighbours into peanut blobs.  The
# shell is therefore subtracted instead: every contact neck becomes a dark groove
# and each dark core stays an independent bright disc.
CLSM_SHELL_SEPARATION_WEIGHT = 0.90
# Dense magenta-rich beds (e.g. the bottom-left agglomerate of Snap-22.tif) hide
# dozens of small spheres whose cores Cellpose merges into one irregular blob.
CLSM_AGGLOMERATE_MAGENTA_MIN = 110      # R at/above this level = fluorescent magenta
CLSM_AGGLOMERATE_MIN_DIAM_PX = 3.0
CLSM_AGGLOMERATE_MAX_DIAM_PX = 24.0
CLSM_AGGLOMERATE_MIN_CIRCULARITY = 0.60
CLSM_AGGLOMERATE_MIN_SOLIDITY = 0.70
CLSM_AGGLOMERATE_PEAK_SEPARATION_PX = 3.0
CLSM_AGGLOMERATE_MAX_OVERLAP = 0.25
# A dark core only becomes a sphere when a fluorescent shell ring encloses it.
CLSM_AGGLOMERATE_CORE_MIN_LEVEL = 150.0   # minimum 255 - G level of a real core
CLSM_AGGLOMERATE_RING_PAD_PX = 3          # shell-ring width measured around a core
CLSM_AGGLOMERATE_RING_MIN_RIDGE = 15      # R - G (grey levels) that marks a shell
CLSM_AGGLOMERATE_RING_MIN_FRACTION = 0.45 # share of the ring that must be shell
CLSM_AGGLOMERATE_MIN_CONTRAST = 18.0      # core must be this much darker than its ring
# Post-segmentation split: any Cellpose mask this irregular is 2+ cemented spheres.
CLSM_MERGED_MAX_CIRCULARITY = 0.72
CLSM_MERGED_MIN_SOLIDITY = 0.88
CLSM_MERGED_PEAK_FRACTION = 0.35


def _mask_circularity_solidity(region: np.ndarray) -> Tuple[float, float]:
    """
    Measure the circularity and convexity of one binary mask patch.

    Args:
        region: Binary patch (bounding box of a single object).

    Returns:
        Tuple[float, float]: (circularity, solidity); ``(0.0, 0.0)`` for an empty
        patch or a degenerate contour.
    """
    region_u8 = np.ascontiguousarray(region, dtype=np.uint8)
    if not np.any(region_u8):
        return 0.0, 0.0
    contours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return 0.0, 0.0
    perimeter = float(cv2.arcLength(contour, True))
    circularity = (4.0 * math.pi * area) / (perimeter ** 2) if perimeter > 0.0 else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0.0 else 0.0
    return float(circularity), float(solidity)


def _clsm_rgb_channels(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a CLSM composite frame into its green matrix and magenta shell planes.

    Args:
        image: ``(H, W, 3)`` RGB frame, or a grayscale ``(H, W)`` plane.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(green, magenta)`` uint8 planes, where
        ``green = rgb[:, :, 1]`` (continuous matrix) and
        ``magenta = max(rgb[:, :, 0], rgb[:, :, 2])`` (fluorescent shell).
        For grayscale input the single plane is returned as both channels.
    """
    if image.ndim == 3 and image.shape[2] >= 3:
        green = np.ascontiguousarray(image[:, :, 1], dtype=np.uint8)
        magenta = np.ascontiguousarray(
            np.maximum(image[:, :, 0], image[:, :, 2]), dtype=np.uint8
        )
        return green, magenta
    gray = np.ascontiguousarray(to_grayscale_2d(image), dtype=np.uint8)
    return gray, np.zeros_like(gray)


def resolve_clsm_channel_mode(channel_mode: str, image: np.ndarray) -> str:
    """
    Resolve the CLSM channel / optical mode for one frame.

    Accepts the sidebar label (``"🎨 Composite RGB (Green Matrix + Magenta
    Shell)"``), a short token (``"composite"``/``"c1"``/``"c0"``) or anything
    containing ``"auto"``.  Auto-detection follows the acquisition physics:

    * a 3-channel stack is the RGB composite (green matrix + magenta shell);
    * a single plane whose median exceeds ``CLSM_AUTO_BRIGHT_BACKGROUND_MEDIAN``
      (90 grey levels) is the bright-matrix c1 negative contrast channel;
    * anything darker is the c0 fluorescent-shell channel on black.

    Args:
        channel_mode: Sidebar label, token, or ``"auto"``.
        image: Frame as decoded (``(H, W)`` or ``(H, W, 3)`` uint8).

    Returns:
        str: One of ``CLSM_CHANNEL_COMPOSITE``, ``CLSM_CHANNEL_C1``,
        ``CLSM_CHANNEL_C0``.

    Raises:
        TypeError: If ``image`` is not a numpy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    label = str(channel_mode or "").strip().lower()
    if label and "auto" not in label:
        # "Composite" must be tested before "shell": its own label contains
        # "Magenta Shell" as well.
        if "composite" in label or "rgb" in label:
            return CLSM_CHANNEL_COMPOSITE
        if "c1" in label or "negative" in label:
            return CLSM_CHANNEL_C1
        if "c0" in label or "shell" in label or "ring" in label or "rim" in label:
            return CLSM_CHANNEL_C0

    if image.ndim == 3 and image.shape[2] >= 3:
        return CLSM_CHANNEL_COMPOSITE
    gray = to_grayscale_2d(image)
    median_value = float(np.median(gray)) if gray.size else 0.0
    if median_value > CLSM_AUTO_BRIGHT_BACKGROUND_MEDIAN:
        return CLSM_CHANNEL_C1
    return CLSM_CHANNEL_C0



def _otsu_foreground(image_u8: np.ndarray) -> np.ndarray:
    """
    Otsu-threshold one plane into a boolean foreground mask.

    A one-grey-level floor protects the degenerate "all black" / "all white"
    cases where OpenCV reports a threshold of 0 and the comparison would either
    select nothing or the entire frame.

    Args:
        image_u8: 2D uint8 intensity plane.

    Returns:
        np.ndarray: Boolean mask, True where the plane is above the Otsu value.
    """
    if image_u8.size == 0:
        return np.zeros(image_u8.shape, dtype=bool)
    otsu_value, _ = cv2.threshold(
        image_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return image_u8 > max(float(otsu_value), 1.0)


def _clsm_segmentation_planes(
    image: np.ndarray,
    resolved_mode: str,
    base_diameter_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the Cellpose input plane and the residual-catcher source for one mode.

    Args:
        image: Decoded frame, ``(H, W)`` grayscale or ``(H, W, 3)`` RGB.
        resolved_mode: One of the ``CLSM_CHANNEL_*`` tokens.
        base_diameter_px: Sidebar "Typical Microsphere Diameter for AI" in pixels
            (scales the c0 distance-transform dome so small rings stay visible).

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(cellpose_plane, residual_source)``,
        both 2D uint8, where dark silica cores / bright fluorescent shells are
        represented as bright solid discs.
    """
    if resolved_mode == CLSM_CHANNEL_COMPOSITE:
        green_u8, magenta_u8 = _clsm_rgb_channels(image)
        core_channel = cv2.bitwise_not(green_u8)
        # The magenta shell is a *separator*, never a bridge: the thin fluorescent
        # neck between two touching silica cores is bright magenta, so adding it to
        # the core image (the v2.0.0 recipe, ``0.80*core + 0.35*R``) glued
        # neighbours into peanut blobs and turned the dense bottom-left agglomerate
        # into one irregular mass.  Subtracting the shell ridge instead turns every
        # contact neck into a dark groove Cellpose cannot cross, so each dark core
        # stays an independent bright disc.
        shell_ridge = np.clip(
            magenta_u8.astype(np.int16) - green_u8.astype(np.int16), 0, 255
        ).astype(np.float32)
        separated = np.clip(
            core_channel.astype(np.float32)
            - CLSM_SHELL_SEPARATION_WEIGHT * shell_ridge,
            0.0,
            255.0,
        ).astype(np.uint8)
        # Residual catcher source: same shell-separated cores, so the small spheres
        # hidden in magenta beds stay individual bright discs there too.
        return separated, separated

    gray = to_grayscale_2d(image)

    if resolved_mode == CLSM_CHANNEL_C1:
        # Negative contrast matrix: 255 - gray turns the dark spheres into bright,
        # solid discs (no hole filling needed - the core is already solid).
        inverted = cv2.bitwise_not(gray)
        return inverted, inverted

    # c0: bright fluorescent shells on a black background.  A ring is not a blob,
    # so the enclosed dark core is hole-filled and a distance-transform dome is
    # painted inside it - hollow rings become solid discs.
    rings_u8 = cv2.morphologyEx(
        _otsu_foreground(gray).astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    solid = ndi.binary_fill_holes(rings_u8.astype(bool))
    if not solid.any():
        return gray, gray
    dist = cv2.distanceTransform(solid.astype(np.uint8), cv2.DIST_L2, 5)
    # A sphere of the sidebar target diameter reaches full brightness, so small
    # rings keep a usable dome instead of being normalised away by the largest one.
    dome_scale = 255.0 / max(2.0, 0.5 * float(base_diameter_px))
    dome = np.clip(dist * dome_scale, 0.0, 255.0).astype(np.uint8)
    return cv2.max(gray, dome), dome


def _watershed_split_microspheres(mask: np.ndarray, min_diam_px: float) -> np.ndarray:
    """
    Split touching microspheres with a distance-transform marker watershed.

    Colloidal silica beds are densely packed, so thresholded clusters of small
    spheres arrive as one merged blob.  A marker watershed on the **inverted
    distance transform** cuts that blob along the ridges between its distance
    maxima, i.e. exactly between the sphere centres, and returns one label per
    particle instead of one label per cluster.  The 1-px watershed lines are
    handed back to their nearest label afterwards, so a 5-10 px sphere does not
    lose the rim pixels that its circularity is measured on.

    Args:
        mask: Boolean candidate mask (any number of disjoint clusters).
        min_diam_px: Smallest sphere diameter of interest; also the local-maxima
            window size, so a single sphere never produces two peaks.

    Returns:
        np.ndarray: int32 label image (0 = background, 1..N = particles).
    """
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return labels

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if float(dist.max()) <= 0.0:
        return labels

    peak_size = max(3, int(round(float(min_diam_px))))
    local_max = ndi.maximum_filter(dist, size=peak_size)
    peaks = (dist >= local_max - 1e-6) & (dist > 0.0)
    peak_labels, n_peaks = ndi.label(peaks)
    if n_peaks <= 1:
        # A single blob with a single distance maximum: nothing to split.
        return ndi.label(mask)[0].astype(np.int32)

    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[peaks] = peak_labels[peaks]
    # Inverted distance transform = topographic surface: the markers are basins
    # and the watershed floods the ridges (saddles) between touching spheres.
    # ``skimage.segmentation.watershed`` is used instead of ``cv2.watershed``
    # because OpenCV's variant floods on the *image gradient*, which is flat
    # inside a distance-transform plateau and produced wildly unbalanced splits
    # (one sphere absorbing most of its neighbour).
    labels = skimage_watershed(
        -dist.astype(np.float32),
        markers=markers,
        mask=mask,
    ).astype(np.int32)

    # Watershed lines (-1/0) inside the mask are handed back to their nearest
    # label, so a 5-10 px sphere does not lose the rim pixels its circularity is
    # measured on.
    boundary = mask & (labels <= 0)
    if boundary.any():
        _, (iy, ix) = ndi.distance_transform_edt(labels <= 0, return_indices=True)
        nearest = labels[iy, ix]
        labels = np.where(boundary & (nearest > 0), nearest, labels).astype(np.int32)
    labels = np.maximum(labels, 0)
    if int(labels.max()) <= 0:
        return ndi.label(mask)[0].astype(np.int32)

    # Seam fragments: where three or more discs overlap, the watershed can carve a
    # few-pixel pocket that gets its own peak.  Such a pocket is far thinner than
    # a real sphere, so a label survives only when its inradius (``dist`` maximum)
    # reaches 40 % of the smallest sphere diameter of interest.
    n_labels_found = int(labels.max())
    if n_labels_found > 0:
        label_ids = np.arange(1, n_labels_found + 1)
        max_inradius = np.asarray(
            ndi.maximum(dist, labels, index=label_ids), dtype=float
        )
        label_areas = np.bincount(labels.ravel(), minlength=n_labels_found + 1)
        keep = np.zeros(n_labels_found + 1, dtype=bool)
        keep[1:] = (max_inradius >= 0.40 * float(min_diam_px)) & (
            label_areas[1:] >= 4
        )
        labels = np.where(keep[labels], labels, 0).astype(np.int32)
    if int(labels.max()) <= 0:
        return ndi.label(mask)[0].astype(np.int32)
    return labels


def _recover_clsm_residual_microspheres(
    source_u8: np.ndarray,
    base_diameter_px: float,
    combined: np.ndarray,
    occupied: np.ndarray,
    next_id: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Append the small microspheres Cellpose's receptive field cannot resolve.

    The residual "catcher" runs purely on morphology (Otsu + distance-transform
    watershed, no neural network), so it costs milliseconds at native resolution.
    It only looks at the band ``5 px .. 0.8 x base_diameter_px``; anything the
    main Cellpose pass already claimed is rejected by the overlap gate.

    Args:
        source_u8: Residual source plane from ``_clsm_segmentation_planes``
            (dark cores / bright shells already turned into bright discs).
        base_diameter_px: Cellpose target diameter in pixels.
        combined: Running output label mask (modified in place).
        occupied: Running footprint canvas of the accepted objects.
        next_id: Label id the next accepted sphere receives.

    Returns:
        Tuple[np.ndarray, int, int]: (combined, next_id, added).
    """
    min_diam = float(CLSM_MIN_MICROSPHERE_DIAM_PX)
    max_diam = max(min_diam + 1.0, CLSM_RESIDUAL_DIAM_FRACTION * float(base_diameter_px))
    min_area = max(
        CLSM_MAIN_MIN_AREA_PX, int(round(math.pi * (min_diam / 2.0) ** 2 * 0.5))
    )

    candidates = cv2.morphologyEx(
        _otsu_foreground(source_u8).astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    if not candidates.any():
        return combined, next_id, 0

    labels = _watershed_split_microspheres(candidates.astype(bool), min_diam)
    added = 0
    area_lookup = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
    for lbl, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None or int(area_lookup[lbl]) < min_area:
            continue
        region = labels[sl] == lbl
        area = int(np.count_nonzero(region))
        if area <= 0:
            continue
        equiv_diam = 2.0 * math.sqrt(area / math.pi)
        if equiv_diam < min_diam or equiv_diam > max_diam:
            continue
        if not _is_round_droplet(
            region, CLSM_RESIDUAL_MIN_CIRCULARITY, CLSM_RESIDUAL_MIN_SOLIDITY
        ):
            continue
        overlap = int(np.count_nonzero(region & occupied[sl])) / float(area)
        if overlap > CLSM_RESIDUAL_MAX_OVERLAP:
            continue
        # Never overwrite an accepted sphere: only the free pixels of the candidate
        # are claimed, otherwise a 25 %-overlapping write would shred the neighbour
        # that already owns them into 1-px fragments.
        free = region & ~occupied[sl]
        if int(np.count_nonzero(free)) < min_area:
            continue
        combined[sl][free] = next_id
        next_id += 1
        added += 1
    if added:
        logger.info(
            "CLSM residual catcher appended %d small microsphere(s) (diam %.1f-%.1f px)",
            added, min_diam, max_diam,
        )
    return combined, next_id, added


def _split_merged_clsm_objects(
    masks: np.ndarray,
    min_peak_sep_px: float,
) -> Tuple[np.ndarray, int]:
    """
    Split every merged (non-round) Cellpose mask into its individual spheres.

    A single silica microsphere is compact by definition, so a mask whose
    circularity or solidity falls below ``CLSM_MERGED_MAX_CIRCULARITY`` /
    ``CLSM_MERGED_MIN_SOLIDITY`` is 2+ cemented spheres that Cellpose grew as one
    blob.  Such masks are cut with a distance-transform marker watershed whose
    local-maxima window is ``min_peak_sep_px`` (``0.35 x base_diameter_px``), and
    each round fragment receives its own label.

    Args:
        masks: int32 label mask (0 = background).
        min_peak_sep_px: Minimum distance between two sphere centres, in pixels.

    Returns:
        Tuple[np.ndarray, int]: (relabelled mask with consecutive ids, number of
        masks that were actually split).
    """
    shape = masks.shape[:2]
    if masks is None or masks.size == 0 or int(masks.max()) <= 0:
        return np.zeros(shape, dtype=np.int32), 0

    out = np.zeros(shape, dtype=np.int32)
    next_id = 1
    splits = 0
    for lbl, sl in enumerate(ndi.find_objects(masks), start=1):
        if sl is None:
            continue
        patch = masks[sl] == lbl
        if not patch.any():
            continue
        circularity, solidity = _mask_circularity_solidity(patch)
        if circularity >= CLSM_MERGED_MAX_CIRCULARITY and solidity >= CLSM_MERGED_MIN_SOLIDITY:
            out[sl][patch] = next_id
            next_id += 1
            continue

        parts = _watershed_split_microspheres(patch, float(min_peak_sep_px))
        n_parts = int(parts.max())
        if n_parts <= 1:
            out[sl][patch] = next_id
            next_id += 1
            continue

        # ``parts`` is patch-sized, so every part slice must be shifted back into
        # full-frame coordinates before it can index the output canvas.
        y_offset = int(sl[0].start)
        x_offset = int(sl[1].start)
        min_part_area = max(CLSM_MAIN_MIN_AREA_PX, int(0.12 * int(np.count_nonzero(patch))))
        kept = 0
        for part_lbl, part_sl in enumerate(ndi.find_objects(parts), start=1):
            if part_sl is None:
                continue
            part = parts[part_sl] == part_lbl
            if int(np.count_nonzero(part)) < min_part_area:
                continue
            part_circ, part_sol = _mask_circularity_solidity(part)
            # A real sphere fragment must still look round; slivers between three
            # cores are rejected here instead of reaching the size distribution.
            if part_circ < CLSM_RESIDUAL_MIN_CIRCULARITY or part_sol < CLSM_RESIDUAL_MIN_SOLIDITY:
                continue
            target = (
                slice(y_offset + part_sl[0].start, y_offset + part_sl[0].stop),
                slice(x_offset + part_sl[1].start, x_offset + part_sl[1].stop),
            )
            out[target][part] = next_id
            next_id += 1
            kept += 1
        if kept == 0:
            out[sl][patch] = next_id
            next_id += 1
        else:
            splits += 1

    if splits:
        logger.info("CLSM cluster split: %d merged mask(s) separated into single spheres", splits)
    return out, splits


def _recover_clsm_agglomerate_spheres(
    image: np.ndarray,
    base_diameter_px: float,
    combined: np.ndarray,
    occupied: np.ndarray,
    next_id: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Recover the small spheres hidden inside dense magenta agglomerates.

    A tight bed of small silica spheres (the bottom-left agglomerate of
    ``Snap-22.tif``, x ≈ 45-105, y ≈ 295-355) is blanketed in fluorescent magenta,
    so its cores are dim and touching and Cellpose grows one irregular blob for the
    whole bed - which then fails the sidebar circularity gate.  Global colour
    regions cannot isolate such a bed (the bright green matrix and the bed have
    practically identical R/G statistics), so the discriminator used here is
    strictly LOCAL: a dark core (high ``255 - G``) becomes a sphere only when

    * a fluorescent shell ring surrounds it - at least
      ``CLSM_AGGLOMERATE_RING_MIN_FRACTION`` of the annulus reaches
      ``R - G >= CLSM_AGGLOMERATE_RING_MIN_RIDGE`` (or ``R >= 110``), and
    * the core itself is ``CLSM_AGGLOMERATE_MIN_CONTRAST`` grey levels darker than
      that ring (a flat shadow pocket has no core at all).

    The surviving core mask is split with a distance-transform marker watershed
    (peak separation ``CLSM_AGGLOMERATE_PEAK_SEPARATION_PX``), so every individual
    small sphere inside the bed becomes its own compact mask
    (``3 px .. 24 px`` diameter, ``circularity >= 0.60``) instead of one merged
    irregular blob.

    Args:
        image: Original CLSM frame (``(H, W, 3)`` RGB composite).
        base_diameter_px: Cellpose target diameter in pixels (log only).
        combined: Running output label mask (modified in place).
        occupied: Running footprint canvas of the accepted objects.
        next_id: Label id the next accepted sphere receives.

    Returns:
        Tuple[np.ndarray, int, int]: (combined, next_id, added).
    """
    if image.ndim != 3 or image.shape[2] < 3 or combined.size == 0:
        return combined, next_id, 0

    green, magenta = _clsm_rgb_channels(image)
    if not np.any(magenta >= CLSM_AGGLOMERATE_MAGENTA_MIN):
        return combined, next_id, 0

    # The magenta shell ridge (R - G) is the accent that surrounds every sphere in
    # a dense bed.  Global colour regions cannot isolate an agglomerate (on
    # Snap-22.tif the bright matrix and the bottom-left bed have practically
    # identical R/G statistics), so the discriminator is LOCAL: a dark core only
    # becomes a sphere when a magenta ridge ring encloses it.
    core_inv = cv2.bitwise_not(green)
    ridge = np.clip(
        magenta.astype(np.int16) - green.astype(np.int16), 0, 255
    ).astype(np.uint8)

    # Dark cores = the dark tail of the green matrix (high 255 - G).
    otsu_value, _ = cv2.threshold(
        core_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    core_threshold = max(float(otsu_value), CLSM_AGGLOMERATE_CORE_MIN_LEVEL)
    cores = (core_inv >= core_threshold) & ~occupied
    if not np.any(cores):
        return combined, next_id, 0

    cores_u8 = cv2.morphologyEx(
        cores.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    if not cores_u8.any():
        return combined, next_id, 0

    labels = _watershed_split_microspheres(
        cores_u8.astype(bool), CLSM_AGGLOMERATE_PEAK_SEPARATION_PX
    )
    if int(labels.max()) <= 0:
        return combined, next_id, 0

    label_area = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
    min_area = max(
        4,
        int(
            round(
                math.pi
                * (CLSM_AGGLOMERATE_MIN_DIAM_PX / 2.0) ** 2
                * 0.65
            )
        ),
    )
    height, width = labels.shape[:2]
    pad = int(CLSM_AGGLOMERATE_RING_PAD_PX) + 1
    added = 0
    for lbl, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None or int(label_area[lbl]) < min_area:
            continue
        # Work in a padded window so the surrounding shell ring is measurable.
        y0 = max(0, sl[0].start - pad)
        y1 = min(height, sl[0].stop + pad)
        x0 = max(0, sl[1].start - pad)
        x1 = min(width, sl[1].stop + pad)
        local = labels[y0:y1, x0:x1]
        region = local == lbl
        area = int(np.count_nonzero(region))
        if area <= 0:
            continue
        equiv_diam = 2.0 * math.sqrt(area / math.pi)
        if (
            equiv_diam < CLSM_AGGLOMERATE_MIN_DIAM_PX
            or equiv_diam > CLSM_AGGLOMERATE_MAX_DIAM_PX
        ):
            continue
        circularity, solidity = _mask_circularity_solidity(region)
        if (
            circularity < CLSM_AGGLOMERATE_MIN_CIRCULARITY
            or solidity < CLSM_AGGLOMERATE_MIN_SOLIDITY
        ):
            continue

        # 1. Magenta ring signature: the candidate must be enclosed by the
        #    fluorescent shell of the agglomerate, not by plain green matrix.
        ring = cv2.dilate(
            region.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * int(CLSM_AGGLOMERATE_RING_PAD_PX) + 1,) * 2,
            ),
        ).astype(bool) & ~region
        if not np.any(ring):
            continue
        ring_ridge = ridge[y0:y1, x0:x1][ring]
        ring_magenta = magenta[y0:y1, x0:x1][ring]
        shell_ring = (ring_ridge >= CLSM_AGGLOMERATE_RING_MIN_RIDGE) | (
            ring_magenta >= int(CLSM_AGGLOMERATE_MAGENTA_MIN)
        )
        if float(np.mean(shell_ring)) < CLSM_AGGLOMERATE_RING_MIN_FRACTION:
            continue

        # 2. Core contrast: the candidate core must be measurably darker than the
        #    shell ring around it (a flat shadow pocket has no core at all).
        local_core = core_inv[y0:y1, x0:x1]
        contrast = float(local_core[region].mean()) - float(local_core[ring].mean())
        if contrast < CLSM_AGGLOMERATE_MIN_CONTRAST:
            continue

        overlap = int(np.count_nonzero(region & occupied[y0:y1, x0:x1])) / float(area)
        if overlap > CLSM_AGGLOMERATE_MAX_OVERLAP:
            continue
        # Claim only unoccupied pixels so an accepted sphere is never shredded.
        free = region & ~occupied[y0:y1, x0:x1]
        if int(np.count_nonzero(free)) < min_area:
            continue
        combined[y0:y1, x0:x1][free] = next_id
        next_id += 1
        added += 1

    if added:
        logger.info(
            "CLSM agglomerate recovery appended %d sphere(s) enclosed by magenta "
            "shells (diam %.1f-%.1f px, base %.1f px)",
            added,
            CLSM_AGGLOMERATE_MIN_DIAM_PX,
            CLSM_AGGLOMERATE_MAX_DIAM_PX,
            float(base_diameter_px),
        )
    return combined, next_id, added





@st.cache_data(show_spinner="Running CLSM Confocal Microsphere Segmentation...")
def cached_segment_clsm_microspheres(
    image: np.ndarray,
    base_diameter_px: float,
    channel_mode: str = CLSM_CHANNEL_AUTO,
    model_name: str = "cyto2",
    custom_model_path: Optional[str] = None,
    flow_threshold: float = 0.7,
    cellprob_threshold: float = -1.0,
    enable_clahe: bool = False,
) -> np.ndarray:
    """
    Segment silica microspheres in confocal (CLSM) micrographs - all 3 file types.

    Pipeline:
        1. ``resolve_clsm_channel_mode`` maps the sidebar label / token to the
           concrete optical mode (``auto`` inspects the channel count and the
           median grey level: RGB -> composite, ``median > 90`` -> c1 bright
           matrix, ``median <= 90`` -> c0 dark-background shells).
        2. ``_clsm_segmentation_planes`` builds the Cellpose input: the negative
           stain core ``255 - G`` with the magenta shell ridge **subtracted** (the
           shell is a boundary separator, never a bridge), a plain inversion for
           c1, and hole-filling + a distance-transform dome for hollow c0 rings.
        3. One Cellpose pass at ``base_diameter_px`` (clamped into Cellpose's
           28-45 px comfort zone by ``run_inference(allow_downscale=True)``, so a
           1024x1024 confocal frame costs ~1 Mpx of network work).  Every mask is
           hole-filled and compacted to consecutive label ids.
        4. ``_recover_clsm_residual_microspheres`` appends the 5 px -
           ``0.8 x base_diameter_px`` spheres the network cannot resolve, split
           apart by a distance-transform watershed instead of being merged into
           one blob.

    Args:
        image: Analysis frame - ``(H, W, 3)`` RGB (``Snap-22.tif``) or
            ``(H, W)`` single channel (c1 / c0 exports).
        base_diameter_px: Typical microsphere diameter in pixels (sidebar
            "Typical Microsphere Diameter for AI" converted with the active
            objective calibration).
        channel_mode: Sidebar label, ``"composite"``/``"c1"``/``"c0"``, or
            ``"auto"``.
        model_name: Cellpose model name ('cyto2', 'cyto3', 'nuclei', ...).
        custom_model_path: Optional path to custom Cellpose weights.
        flow_threshold: Cellpose flow error threshold.
        cellprob_threshold: Cellpose cell-probability threshold.
        enable_clahe: Apply CLAHE contrast enhancement before inference.

    Returns:
        np.ndarray: int32 label mask (0 = background, 1..N = individual
        microspheres) at the input spatial resolution.
    """
    model = get_cellpose_model(model_type=model_name, custom_model_path=custom_model_path)

    resolved_mode = resolve_clsm_channel_mode(channel_mode, image)
    base_diameter = max(float(base_diameter_px), CLSM_MIN_MICROSPHERE_DIAM_PX)
    seg_img, residual_source = _clsm_segmentation_planes(
        image, resolved_mode, base_diameter
    )

    masks_main = run_inference(
        model=model,
        image=seg_img,
        diameter=float(
            np.clip(base_diameter, CELLPOSE_MIN_INFERENCE_DIAMETER_PX, 240.0)
        ),
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        enable_clahe=enable_clahe,
        allow_downscale=True,
    )

    combined = np.zeros(seg_img.shape[:2], dtype=np.int32)
    occupied = np.zeros(seg_img.shape[:2], dtype=bool)
    next_id = 1

    # ---- Cellpose pass: hole-filled spheres, compacted label ids ---------------
    n_labels = int(masks_main.max()) if masks_main.size else 0
    if n_labels > 0:
        label_area = np.bincount(masks_main.ravel(), minlength=n_labels + 1)
        for lbl, sl in enumerate(ndi.find_objects(masks_main), start=1):
            if sl is None or int(label_area[lbl]) < CLSM_MAIN_MIN_AREA_PX:
                continue
            # Pure hole filling (binary_fill_holes) instead of redrawing the filled
            # external contour: a redrawn disc can spill over a neighbouring label
            # that was written earlier and shred it into 1-px fragments.  Filling
            # enclosed holes turns a hollow c0 ring into a solid disc without ever
            # growing past the object's own boundary.
            region = masks_main[sl] == lbl
            filled_bool = ndi.binary_fill_holes(region)
            if not filled_bool.any():
                continue
            combined[sl][filled_bool] = next_id
            occupied[sl] |= filled_bool
            next_id += 1
    n_main = next_id - 1

    # ---- Post-segmentation cluster split: every merged mask becomes N spheres ---
    # A mask that is not compact is 2+ cemented microspheres, so it is cut with a
    # distance-transform marker watershed before any filtering can reject it as
    # "irregular" (circularity < 0.55 in the sidebar).
    min_peak_sep_px = max(3, int(round(CLSM_MERGED_PEAK_FRACTION * base_diameter)))
    combined, n_split = _split_merged_clsm_objects(combined, min_peak_sep_px)
    next_id = int(combined.max()) + 1 if combined.size else 1
    occupied = combined > 0
    n_after_split = next_id - 1

    # ---- Residual catcher: small spheres + dense-cluster watershed split ------
    combined, next_id, n_residual = _recover_clsm_residual_microspheres(
        residual_source, base_diameter, combined, occupied, next_id
    )
    occupied = combined > 0

    # ---- Dense magenta agglomerates: individual small spheres inside the bed ---
    combined, next_id, n_agglomerate = _recover_clsm_agglomerate_spheres(
        image, base_diameter, combined, occupied, next_id
    )

    logger.info(
        "CLSM microspheres [%s]: %d Cellpose (%d after splitting %d merged mask(s)) "
        "+ %d residual + %d agglomerate = %d objects",
        resolved_mode,
        n_main,
        n_after_split,
        n_split,
        n_residual,
        n_agglomerate,
        next_id - 1,
    )
    return combined





def predict_interactive_box_mask(
    image_2d: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    target_phase: str = "air_bubbles",
    cellpose_model=None,
    invert_brightfield: bool = False,
    sam_model_type: str = "vit_b",
) -> np.ndarray:
    """
    Instant interactive segmentation of an object drawn with a bounding box.

    Step A (SAM on a local ROI, combined Box + 5-Point prompt):
        The user's rectangle tightly frames one object, so its center is foreground
        while its 4 inset corners are guaranteed background. SAM receives all 5 points
        plus the box at once — but only for the **padded ROI crop** around that box
        (box + 40 % per side, min 256x256 px), never the full-resolution micrograph.
        The predicted mask is shifted back into full-image coordinates, so nothing
        downstream changes. Neighbouring boxes reuse the already-computed embedding.
        Of the 3 candidate masks only the one that covers the center, fills 30-88 % of
        the box and does NOT touch all 4 corners (i.e. is not the rectangle itself) is
        accepted. Accepted masks are hole-filled and opened with a 5x5 ellipse to cut
        thin necks to neighbouring particles.

    Step B (fallback: Cellpose in the box + Elliptical Rim Snapper):
        - A padded crop of the box is segmented with Cellpose (direct + inverted CLAHE;
          in brightfield mode only the inverted polarity is tested) using
          diameter = 0.85 * min(bw, bh), cellprob_threshold = -1.8 and
          flow_threshold = 0.8. A centre object filling 30-88% of the box is kept as a
          base mask and contributes its boundary to the rim-snapping edge hint.
        - 72 rays from the box center search the brightness-gradient peak inside the
          annular corridor [0.62 * r_ell, 1.05 * r_ell] of the inscribed ellipse, with a
          soft penalty for deviating from 0.92 * r_ell. Radii are median-filtered
          (size=5, wrap) and filled into the returned mask, so the result always follows
          the real bubble/particle wall and can never become a square.

    Args:
        image_2d:       2D grayscale image.
        box_xyxy:       Bounding box as (x1, y1, x2, y2).
        target_phase:   Phase hint kept for UI/API compatibility ("air_bubbles", ...).
                        The current pipeline is phase-agnostic: absolute-gradient rim
                        snapping works for dark pores and bright fat globules alike.
        cellpose_model: Optional preloaded CellposeModel used by the fallback path.
        invert_brightfield: When True (brightfield light microscopy), the fallback only
                        tests the inverted ROI polarity — the correct one for dark-rim
                        droplets — instead of running Cellpose twice.
        sam_model_type: SAM backbone used by Step A (``"vit_b"`` fast, ``"vit_l"``
                        balanced, ``"vit_h"`` maximum boundary precision).

    Returns:
        np.ndarray: 2D uint8 binary mask (0/1) with shape ``image_2d.shape[:2]``.
    """
    if not isinstance(image_2d, np.ndarray) or image_2d.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if image_2d.ndim == 3:
        image_2d = image_2d[:, :, 0]
    h, w = image_2d.shape[:2]
    out_mask = np.zeros((h, w), dtype=np.uint8)

    if not isinstance(box_xyxy, (tuple, list)) or len(box_xyxy) != 4:
        return out_mask

    try:
        x1 = int(np.clip(round(float(box_xyxy[0])), 0, w - 1))
        y1 = int(np.clip(round(float(box_xyxy[1])), 0, h - 1))
        x2 = int(np.clip(round(float(box_xyxy[2])), 0, w))
        y2 = int(np.clip(round(float(box_xyxy[3])), 0, h))
    except (TypeError, ValueError):
        return out_mask
    if x2 <= x1 or y2 <= y1:
        return out_mask

    bw = float(x2 - x1)
    bh = float(y2 - y1)
    box_area = max(bw * bh, 1.0)
    cx_box = (x1 + x2) / 2.0
    cy_box = (y1 + y2) / 2.0
    cx_i = int(np.clip(round(cx_box), 0, w - 1))
    cy_i = int(np.clip(round(cy_box), 0, h - 1))

    # Inset corner prompts (4% of the side, at least 2 px) — guaranteed background
    dx = max(2.0, bw * 0.04)
    dy = max(2.0, bh * 0.04)
    corners = _box_prompt_corners(x1, y1, x2, y2, dx, dy)

    hint_layer = np.zeros((h, w), dtype=np.uint8)

    # ── Step A: SAM on the LOCAL ROI — box + 1 positive centre point + 4 negative corners
    #    Only the padded crop around the drawn box is encoded (40 % per side, min 256 px),
    #    so the full-resolution micrograph is never pushed through the ViT encoder.
    sam_state = get_cached_sam_predictor_for_roi(
        image_2d, (x1, y1, x2, y2), sam_model_type=sam_model_type
    )
    if sam_state is not None:
        predictor, (reg_x0, reg_y0, reg_x1, reg_y1) = sam_state
        reg_h = int(reg_y1 - reg_y0)
        reg_w = int(reg_x1 - reg_x0)
        try:
            # Prompt geometry in ROI-local coordinates.
            local_box = np.array(
                [
                    x1 - reg_x0,
                    y1 - reg_y0,
                    x2 - reg_x0,
                    y2 - reg_y0,
                ],
                dtype=np.float32,
            )
            local_corners = _box_prompt_corners(
                float(local_box[0]),
                float(local_box[1]),
                float(local_box[2]),
                float(local_box[3]),
                dx,
                dy,
            )
            pt_coords = np.vstack(
                [
                    np.array(
                        [[cx_box - reg_x0, cy_box - reg_y0]], dtype=np.float32
                    ),
                    local_corners,
                ]
            )
            pt_labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=pt_coords,
                point_labels=pt_labels,
                box=local_box,
                multimask_output=True,
            )

            best_mask = None
            best_score = -1.0
            for i in range(len(masks)):
                local = np.asarray(masks[i]).astype(bool)
                if local.shape[:2] != (reg_h, reg_w):
                    continue
                # Shift the ROI-local prediction back into full-image coordinates.
                cand = np.zeros((h, w), dtype=bool)
                cand[reg_y0:reg_y1, reg_x0:reg_x1] = local
                ratio = float(np.count_nonzero(cand)) / box_area
                if ratio < 0.30 or ratio > 0.88:
                    continue
                if not bool(cand[cy_i, cx_i]):
                    continue
                if _is_box_like_mask(cand, corners):
                    continue
                hint_layer[cand] = 1
                score = float(scores[i]) if scores is not None else 0.0
                score += 0.15 * ratio
                if score > best_score:
                    best_score = score
                    best_mask = cand

            if best_mask is not None:
                # Fill internal holes (bright fat droplets) and cut thin necks
                filled = ndi.binary_fill_holes(best_mask)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                opened = (
                    cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
                )
                clipped = np.zeros((h, w), dtype=bool)
                clipped[y1:y2, x1:x2] = opened[y1:y2, x1:x2]
                area = float(np.count_nonzero(clipped))
                ratio = area / box_area
                if (
                    int(area) > 15
                    and bool(clipped[cy_i, cx_i])
                    and 0.25 <= ratio <= 0.92
                    and not _is_box_like_mask(clipped, corners)
                ):
                    return clipped.astype(np.uint8)
        except Exception as exc:
            logger.warning(
                "SAM box+point prediction failed, using rim snapping: %s", exc
            )

    # ── Step B: Cellpose in the box + Elliptical Rim Snapper ────────────────────
    pad = int(max(14, min(bw, bh) * 0.25))
    rx1 = max(0, int(math.floor(x1)) - pad)
    ry1 = max(0, int(math.floor(y1)) - pad)
    rx2 = min(w, int(math.ceil(x2)) + pad)
    ry2 = min(h, int(math.ceil(y2)) + pad)
    roi = image_2d[ry1:ry2, rx1:rx2]

    if roi.size > 0:
        base_mask = None
        roi_clahe = apply_clahe(roi)
        roi_inv = np.clip(255 - roi_clahe, 0, 255).astype(np.uint8)
        cy_roi = int(np.clip(round(cy_box) - ry1, 0, roi.shape[0] - 1))
        cx_roi = int(np.clip(round(cx_box) - rx1, 0, roi.shape[1] - 1))
        diameter = float(max(min(bw, bh) * 0.85, 8.0))

        try:
            if cellpose_model is None:
                cellpose_model = get_cellpose_model()
            # Brightfield droplets: only the inverted polarity is meaningful, so skip
            # the wrong polarity entirely (halves inference time in Light mode).
            test_images = [roi_inv] if invert_brightfield else [roi_clahe, roi_inv]
            for test_img in test_images:
                m_res = run_inference(
                    model=cellpose_model,
                    image=test_img,
                    diameter=diameter,
                    flow_threshold=0.8,
                    cellprob_threshold=-1.8,
                    enable_clahe=False,
                )
                if m_res is None or int(np.max(m_res)) <= 0:
                    continue

                # Label under the box center (or the dominant neighbour within 6 px)
                lbl = int(m_res[cy_roi, cx_roi])
                if lbl == 0:
                    sub = m_res[
                        max(0, cy_roi - 6): cy_roi + 7,
                        max(0, cx_roi - 6): cx_roi + 7,
                    ]
                    nz = sub[sub > 0]
                    if nz.size:
                        lbl = int(np.bincount(nz).argmax())
                if lbl <= 0:
                    continue

                cand = (m_res == lbl).astype(np.uint8)
                hint_layer[ry1:ry2, rx1:rx2] |= cand
                ratio = float(np.count_nonzero(cand)) / box_area
                if base_mask is None and 0.30 <= ratio <= 0.88:
                    base_mask = np.zeros((h, w), dtype=bool)
                    base_mask[ry1:ry2, rx1:rx2] = cand.astype(bool)
        except Exception as exc:
            logger.warning(
                "Cellpose box segmentation failed, using rim snapping only: %s", exc
            )

        edge_hint = _build_edge_hint(hint_layer)
        rim_mask = _radial_rim_mask(image_2d, (x1, y1, x2, y2), edge_hint=edge_hint)
        rim_area = float(np.count_nonzero(rim_mask))
        if int(rim_area) > 15:
            if rim_area > 0.95 * box_area and base_mask is not None:
                return base_mask.astype(np.uint8)
            return rim_mask
        if base_mask is not None and int(np.count_nonzero(base_mask)) > 15:
            return base_mask.astype(np.uint8)

    return out_mask


if __name__ == "__main__":
    # Self-test when executed directly
    print("Running Cellpose model self-test on Apple Silicon / CPU...")
    active_device, is_gpu_enabled = get_device()
    print(f"Detected device: {active_device} (GPU enabled: {is_gpu_enabled})")

    # Generate synthetic grayscale SEM-like image with 2 particles
    canvas_size = 256
    test_image = np.full((canvas_size, canvas_size), 35, dtype=np.uint8)
    grid_y, grid_x = np.ogrid[:canvas_size, :canvas_size]

    # Particle 1 & 2
    test_image[(grid_y - 80) ** 2 + (grid_x - 80) ** 2 <= 30 ** 2] = 210
    test_image[(grid_y - 170) ** 2 + (grid_x - 175) ** 2 <= 25 ** 2] = 230

    segmenter = CellposeSegmenter(model_type="cyto2")
    output_masks = segmenter.segment(test_image, diameter=55.0, use_cache=False)

    print(f"Input shape: {test_image.shape}, Output mask shape: {output_masks.shape}")
    print(f"Unique segmented object labels: {np.unique(output_masks)}")
    assert output_masks.shape == test_image.shape, "Shape mismatch between input and output!"
    assert 0 in np.unique(output_masks), "Background label 0 not present!"
    print("Verification successfully completed!")
