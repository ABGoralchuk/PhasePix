"""
Module for SEM image segmentation using Cellpose with Apple Silicon (MPS) acceleration.
Provides model caching and inference utilities for PhasePix.
"""

from typing import Optional, Tuple
import cv2
import logging
import math
import os
import shutil
import ssl
import urllib.request
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import torch
from cellpose import models
import streamlit as st

# Setup module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


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


def _normalize_sam_model_type(sam_model_type: Optional[str]) -> str:
    """Return a valid SAM backbone key, defaulting safely to ``vit_b``."""
    if isinstance(sam_model_type, str) and sam_model_type in SAM_MODEL_CONFIGS:
        return sam_model_type
    if sam_model_type:
        logger.warning(
            "Unknown SAM backbone '%s'; defaulting to 'vit_b'.", sam_model_type
        )
    return "vit_b"


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
        image: 2D numpy array (grayscale).
        clip_limit: Threshold for contrast limiting. Default is 2.0.
        tile_grid_size: Size of grid for histogram equalization (e.g., (8, 8)).

    Returns:
        np.ndarray: Contrast-enhanced 2D uint8 numpy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array.")

    img_2d = image.squeeze() if image.ndim == 3 else image
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


def smooth_internal_crystals(image: np.ndarray, strength: int = 2) -> np.ndarray:
    """
    Suppress internal crystal needle/fibril lines inside crystallized fat globules
    while preserving the dark outer droplet boundaries (edge-preserving filtering).
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
    med = cv2.medianBlur(img_u8, k_med)
    # 2. Bilateral filter homogenizes droplet interiors while keeping outer rims sharp
    smooth = cv2.bilateralFilter(med, d=9, sigmaColor=35 * s, sigmaSpace=7 * s)
    return smooth


def get_device() -> Tuple[torch.device, bool]:
    """
    Detect and return the optimal compute device.
    Prioritizes Apple Silicon MPS (Metal Performance Shaders), then CUDA, and falls back to CPU.

    Returns:
        Tuple[torch.device, bool]: (device, is_gpu_active)
    """
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logger.info("Apple Silicon MPS (Metal Performance Shaders) detected and enabled.")
            return torch.device("mps"), True
        if torch.cuda.is_available():
            logger.info("NVIDIA CUDA detected and enabled.")
            return torch.device("cuda"), True
    except Exception as exc:
        logger.warning("Hardware acceleration probe encountered an issue: %s. Defaulting to CPU.", exc)

    logger.info("Using CPU backend.")
    return torch.device("cpu"), False


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
        image: 2D grayscale image as a numpy array.
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
) -> np.ndarray:
    """
    Perform direct inference on a 2D grayscale image using a loaded Cellpose model.

    Args:
        model: An initialized CellposeModel.
        image: 2D numpy array (grayscale image) or single-channel 3D array (H, W, 1).
        diameter: Average diameter of objects in pixels.
        flow_threshold: Flow error threshold (default 0.7).
        cellprob_threshold: Cell probability threshold (default -1.0).
        enable_clahe: Apply CLAHE contrast enhancement prior to model feeding.
        clahe_clip_limit: Contrast clip limit for CLAHE.
        clahe_tile_grid_size: Grid size for CLAHE tiles.

    Returns:
        np.ndarray: 2D numpy array of integer labels (0=background, 1, 2, ...=objects)
                    matching the input spatial dimensions (H, W).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if image.ndim == 3 and image.shape[2] == 1:
        img_2d = image[:, :, 0]
    elif image.ndim == 2:
        img_2d = image
    else:
        raise ValueError(
            f"Expected a 2D grayscale image or (H, W, 1), but got shape {image.shape}"
        )

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

    try:
        # For grayscale images in Cellpose, channels=[0, 0] denotes single-channel / grayscale
        eval_result = model.eval(
            input_for_model,
            channels=[0, 0],
            diameter=diameter,
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

    Step B — Dark Pore Fallback (geometric):
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

    masks_direct = run_inference(
        model=model,
        image=clahe_img,
        diameter=air_target_px,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        enable_clahe=False,  # already applied
    )

    masks_inv = run_inference(
        model=model,
        image=inv_img,
        diameter=air_target_px,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        enable_clahe=False,
    )

    # Merge: start with direct-pass masks; add non-overlapping inverted-pass masks
    combined = masks_direct.copy()
    next_label = int(combined.max()) + 1

    for inv_label in np.unique(masks_inv):
        if inv_label == 0:
            continue
        inv_region = masks_inv == inv_label
        # Intensity gate: candidate must be darker than image median
        if float(np.median(image[inv_region])) >= img_median:
            continue
        # IoU gate: skip if > 40 % overlap with any already-accepted mask
        overlap = np.sum(inv_region & (combined > 0))
        iou = overlap / max(float(np.sum(inv_region)), 1.0)
        if iou > 0.40:
            continue
        combined[inv_region] = next_label
        next_label += 1

    # ── Step B: Dark Pore Fallback ─────────────────────────────────────────
    blur = cv2.GaussianBlur(image, (9, 9), 0)
    dark_thresh = float(np.percentile(blur, 22))
    dark_binary = (blur.astype(np.float32) <= dark_thresh).astype(np.uint8)

    # Morphological open→close to clean noise and fill thin gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Label connected components
    num_cc, cc_labels = cv2.connectedComponents(dark_binary, connectivity=8)

    lo_px = min_diam_px * 0.85
    hi_px = max_diam_px * 1.15

    for cc_id in range(1, num_cc):
        cc_mask = cc_labels == cc_id
        area_px = float(np.sum(cc_mask))
        eq_diam = math.sqrt(4.0 * area_px / math.pi)
        if not (lo_px <= eq_diam <= hi_px):
            continue

        # Circularity filter
        contours, _ = cv2.findContours(
            cc_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
        overlap = float(np.sum(cc_mask & (combined > 0)))
        iou = overlap / max(area_px, 1.0)
        if iou > 0.40:
            continue

        combined[cc_mask] = next_label
        next_label += 1

    logger.info(
        "segment_air_bubbles_auto: %d bubbles found (direct=%d, inv_added=%d, fallback=%d)",
        int(combined.max()),
        int(masks_direct.max()),
        next_label - int(masks_direct.max()) - 1,
        next_label - int(combined.max()) - 1,
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
        for lbl in np.unique(m_roi_2d):
            if lbl == 0:
                continue

            # Strictly clip candidate mask to the bubble interior
            cand_region = (m_roi_2d == lbl) & (roi_bub > 0)
            cand_area = float(np.sum(cand_region))
            if cand_area < 4.0:
                continue

            # Equivalent diameter check >= min_fat_px * 0.7
            eq_diam = math.sqrt(4.0 * cand_area / math.pi)
            if eq_diam < min_accept_diam:
                continue

            # Overlap gate: skip if >= 30% overlaps existing fat particles in ROI
            overlap = float(np.sum(cand_region & (roi_existing > 0)))
            if overlap / max(cand_area, 1.0) >= 0.30:
                continue

            # Accept candidate with new unique next_id
            roi_existing[cand_region] = next_id
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

    The checkpoint is stored under ``models_cache/{filename}``; missing or truncated
    files are downloaded in 8 MB chunks into a ``.part`` file and atomically moved
    into place. SAM always runs on CPU (safe default; avoids float64/MPS decoder
    issues on Apple Silicon).

    Args:
        sam_model_type: One of ``"vit_b"``, ``"vit_l"``, ``"vit_h"``.

    Returns:
        SamPredictor | None: A ready-to-use predictor, or None if SAM is unavailable.
    """
    sam_type = _normalize_sam_model_type(sam_model_type)
    config = SAM_MODEL_CONFIGS[sam_type]

    try:
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    except Exception as exc:
        logger.warning("segment_anything is not installed; SAM interactive segmentation disabled: %s", exc)
        return None

    try:
        models_dir = Path("models_cache")
        models_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = models_dir / config["filename"]
        min_bytes = int(config["min_bytes"])

        # A truncated/partial checkpoint would break torch.load, so anything
        # suspiciously small is treated as missing and re-downloaded.
        if not ckpt_path.exists() or ckpt_path.stat().st_size < min_bytes:
            url = config["url"]
            logger.info(
                "SAM %s checkpoint missing or truncated. Downloading from %s ...",
                sam_type,
                url,
            )
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
            except Exception as net_exc:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                logger.warning("Failed to download SAM checkpoint (%s): %s", url, net_exc)
                return None

        # On macOS Apple Silicon, the MPS decoder can hit float64 errors.
        # Running SAM on CPU is the safe, reliable choice.
        sam = sam_model_registry[sam_type](checkpoint=str(ckpt_path))
        sam.to(device="cpu")
        predictor = SamPredictor(sam)
        logger.info("SAM SamPredictor (%s) initialized on CPU.", sam_type)
        return predictor
    except Exception as exc:
        logger.warning("Failed to initialize SAM predictor (%s): %s", sam_type, exc)
        return None


def get_cached_sam_predictor_for_image(
    image_2d: np.ndarray,
    sam_model_type: str = "vit_b",
):
    """
    Return a SAM predictor whose image embeddings are computed for ``image_2d``.

    Image embeddings are expensive to compute, so they are cached on the predictor
    instance via the ``_cached_img_hash`` attribute. Each backbone has its own cached
    predictor instance, therefore switching ``vit_b`` <-> ``vit_h`` never reuses
    embeddings computed by a different architecture.

    Args:
        image_2d: 2D grayscale image array.
        sam_model_type: SAM backbone key (``"vit_b"``, ``"vit_l"``, ``"vit_h"``).

    Returns:
        SamPredictor | None: Predictor with embeddings set for the image, or None.
    """
    predictor = get_sam_predictor(sam_model_type=sam_model_type)
    if predictor is None:
        return None

    try:
        if image_2d.ndim == 3:
            image_2d = image_2d[:, :, 0]
        img_hash = hash(image_2d.tobytes())

        if getattr(predictor, "_cached_img_hash", None) == img_hash:
            return predictor

        clahe = apply_clahe(image_2d)
        rgb = cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)
        predictor.set_image(rgb)
        predictor._cached_img_hash = img_hash
        return predictor
    except Exception as exc:
        logger.warning("Failed to compute SAM image embeddings: %s", exc)
        return None


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
       with an accepted large globule are discarded as internal crystal shards.
    """
    model = get_cellpose_model(model_type=model_name, custom_model_path=custom_model_path)
    img_2d = image.squeeze() if image.ndim == 3 else image

    # Pass A: Large globules (stronger smoothing + 1.75x diameter)
    s_large = max(int(smooth_strength), 2) + 1
    img_large = smooth_internal_crystals(img_2d, strength=s_large)
    if invert_brightfield:
        img_large = cv2.bitwise_not(img_large)
    d_large = float(np.clip(base_diameter_px * 1.75, 30.0, 240.0))
    masks_large = run_inference(
        model=model,
        image=img_large,
        diameter=d_large,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        enable_clahe=enable_clahe,
    )

    # Pass B: Standard/small globules (1.0x diameter)
    img_small = smooth_internal_crystals(img_2d, strength=max(int(smooth_strength), 1)) if smooth_strength > 0 else img_2d.copy()
    if invert_brightfield:
        img_small = cv2.bitwise_not(img_small)
    d_small = float(np.clip(base_diameter_px, 14.0, 180.0))
    masks_small = run_inference(
        model=model,
        image=img_small,
        diameter=d_small,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        enable_clahe=enable_clahe,
    )

    # Merge: keep all hole-filled large globules with solidity >= 0.78 first, then add non-overlapping small globules
    combined = np.zeros(img_2d.shape[:2], dtype=np.int32)
    occupied = np.zeros(img_2d.shape[:2], dtype=bool)
    next_id = 1

    for lbl in np.unique(masks_large):
        if lbl == 0:
            continue
        region = (masks_large == lbl).astype(np.uint8)
        area = float(np.count_nonzero(region))
        if area < 36:
            continue
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
        combined[filled_bool] = next_id
        occupied |= filled_bool
        next_id += 1

    for lbl in np.unique(masks_small):
        if lbl == 0:
            continue
        region = (masks_small == lbl)
        area = float(np.count_nonzero(region))
        if area < 12:
            continue
        overlap = float(np.count_nonzero(region & occupied)) / area
        if overlap >= 0.25:
            # Reject internal shard inside an already-captured large globule!
            continue
        clean_region = region & (~occupied)
        if np.count_nonzero(clean_region) < 12:
            continue
        combined[clean_region] = next_id
        occupied |= clean_region
        next_id += 1

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

    Step A (SAM, combined Box + 5-Point prompt):
        The user's rectangle tightly frames one object, so its center is foreground
        while its 4 inset corners are guaranteed background. SAM receives all 5 points
        plus the box at once. Of the 3 candidate masks only the one that covers the
        center, fills 30-88% of the box and does NOT touch all 4 corners (i.e. is not
        the rectangle itself) is accepted. Accepted masks are hole-filled and opened
        with a 5x5 ellipse to cut thin necks to neighbouring particles.

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

    # ── Step A: SAM — box + 1 positive centre point + 4 negative corner points ──
    predictor = get_cached_sam_predictor_for_image(
        image_2d, sam_model_type=sam_model_type
    )
    if predictor is not None:
        try:
            pt_coords = np.vstack(
                [np.array([[cx_box, cy_box]], dtype=np.float32), corners]
            )
            pt_labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=pt_coords,
                point_labels=pt_labels,
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                multimask_output=True,
            )

            best_mask = None
            best_score = -1.0
            for i in range(len(masks)):
                cand = np.asarray(masks[i]).astype(bool)
                if cand.shape[:2] != (h, w):
                    continue
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
