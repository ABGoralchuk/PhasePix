"""
PhasePix - SEM Phase Analyzer Streamlit Application.
Provides automated Cellpose deep learning segmentation, morphological quantification,
phase classification (air bubbles vs. fat particles), and comparative KDE size distribution.
"""

from typing import Optional, Tuple
from pathlib import Path
import io
import json
import math
import re
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import find_objects

# Internal application modules
from cellpose_model import get_cellpose_model, run_inference, get_device, cached_segment_air_bubbles_auto, cached_segment_fat_in_bubbles_deepscan
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
    PALETTE_COLORS,
    apply_footer_dimming,
    create_mask_overlay,
    render_selected_bubbles_inspection_overlay,
    render_coalescence_network_overlay,
)
from scale_detector import detect_scale_bar_pixels, detect_scale_bar

# Compatibility shim: adapt image_to_url for streamlit_drawable_canvas in modern Streamlit
from types import SimpleNamespace
import streamlit.elements.image as _sei
import streamlit.elements.lib.image_utils as _image_utils

if not getattr(_image_utils, "_phasepix_patched", False):
    _original_image_to_url = _image_utils.image_to_url

    def _compat_image_to_url(image, layout_config, *args, **kwargs):
        if isinstance(layout_config, int) or layout_config is None:
            layout_config = SimpleNamespace(width=layout_config if layout_config is not None else -1)
        return _original_image_to_url(image, layout_config, *args, **kwargs)

    _image_utils.image_to_url = _compat_image_to_url
    _sei.image_to_url = _compat_image_to_url
    _image_utils._phasepix_patched = True

from streamlit_drawable_canvas import st_canvas


def calculate_canvas_line_length(line_obj: dict) -> float:
    """Calculate Euclidean length of a line object from Fabric.js canvas JSON."""
    scale_x = float(line_obj.get("scaleX", 1.0))
    scale_y = float(line_obj.get("scaleY", 1.0))
    if "x1" in line_obj and "x2" in line_obj and "y1" in line_obj and "y2" in line_obj:
        dx = (float(line_obj["x2"]) - float(line_obj["x1"])) * scale_x
        dy = (float(line_obj["y2"]) - float(line_obj["y1"])) * scale_y
        return float(np.sqrt(dx ** 2 + dy ** 2))
    w = float(line_obj.get("width", 0.0)) * scale_x
    h = float(line_obj.get("height", 0.0)) * scale_y
    return float(np.sqrt(w ** 2 + h ** 2))


# Configure page layout and aesthetics
st.set_page_config(
    page_title="PhasePix - SEM Phase Analyzer",
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


def decode_image_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes into a 2D grayscale uint8 numpy array.
    Supports JPG, PNG, and multi-mode TIFF files.
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
    Image.fromarray(img_uint8).save(buf, format="PNG")
    return buf.getvalue()




def get_local_images(directory: str = "images") -> list:
    """Return list of supported image paths from the local directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    supported_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in supported_exts])


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
    mask_out = np.where(np.isin(mask_2d, list(valid_ids)), mask_2d, 0)
    return df_out, mask_out, occupied


# ---------------- SIDEBAR CONFIGURATION ----------------
with st.sidebar:
    st.title("🔬 PhasePix Controls")
    st.markdown("Automated SEM microstructural segmentation and phase quantification.")

    # Hardware Acceleration Status
    device, is_gpu = get_device()
    if is_gpu and device.type == "mps":
        st.success("⚡ Hardware Acceleration: **Apple Silicon (MPS)** active", icon="🍏")
    elif is_gpu:
        st.success(f"⚡ Hardware Acceleration: **CUDA ({device})** active", icon="🚀")
    else:
        st.info("💻 Hardware Backend: **CPU**", icon="⚙️")

    st.divider()

    st.subheader("1. Image Input")
    available_images = get_local_images("images")

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
                help="Automatically lists all micrographs located in the local 'images/' directory.",
            )
            if selected_path:
                image_bytes = selected_path.read_bytes()
                image_name = selected_path.name
        else:
            st.warning("The 'images/' folder is currently empty. Drop your SEM images into 'images/'.")
            if st.button("Generate Demo Sample in 'images/'", use_container_width=True):
                demo_bytes = create_sample_sem_image()
                Path("images/demo_sem_sample.png").write_bytes(demo_bytes)
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

    st.divider()

    st.subheader("2. Footer Panel Exclusion")
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

    # Detect scale bar in UPPER-RIGHT banner only (avoids table separator line)
    detected_bar_pixels = 0
    scale_preview = None
    if image_bytes is not None:
        try:
            probe_img = decode_image_bytes(image_bytes)
            detected_bar_pixels, scale_preview, _ = detect_scale_bar(
                probe_img, footer_height=max(10, int(footer_height) or 70)
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
            use_container_width=True,
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
    sample_type = st.radio(
        "Sample Type",
        options=["Emulsion", "Whipped Emulsion", "Foam"],
        index=1,
        help="Select 'Emulsion' for single-phase fat droplets, 'Whipped Emulsion' for multi-phase air and fat sorting, or 'Foam' for single-phase air bubbles.",
    )

    if sample_type == "Emulsion":
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
            ))
        with col_fat2:
            max_fat_size = float(st.number_input(
                "Max Fat Particle Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=5.0,
                step=0.5,
                help="Maximum equivalent diameter in micrometers for fat particles.",
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
            ))
        with col_air2:
            max_air_size = float(st.number_input(
                "Max Air Bubble Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=100.0,
                step=5.0,
                help="Maximum equivalent diameter in micrometers for air bubbles.",
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
            key="detection_focus_radio",
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
            ))
        with col_air2:
            max_air_size = float(st.number_input(
                "Max Air Bubble Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=100.0,
                step=5.0,
                help="Maximum equivalent diameter in micrometers for air bubbles.",
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
            ))
        with col_fat2:
            max_fat_size = float(st.number_input(
                "Max Fat Particle Size (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=5.0,
                step=0.5,
                help="Maximum equivalent diameter in micrometers for fat particles.",
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
    model_type = st.selectbox(
        "Pretrained Model",
        options=["cyto2", "nuclei", "Custom Model File..."],
        index=0,
        help="Architecture model family for particle detection or user-trained custom weights.",
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

    if sample_type == "Emulsion":
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
    )

    cellprob_threshold = st.slider(
        "cellprob_threshold",
        min_value=-6.0,
        max_value=6.0,
        value=-1.0,
        step=0.5,
        help="Cell probability detection threshold. Negative values (e.g. -1.0 to -2.0) drastically increase detection of low-contrast pores and bubbles (default: -1.0).",
    )

    air_min_circularity = st.slider(
        "🫧 Air Bubbles Min Circularity", 0.0, 1.0, 0.50, 0.05
    )
    fat_min_circularity = st.slider(
        "🟧 Matrix Fat Min Circularity", 0.0, 1.0, 0.60, 0.05
    )
    ads_min_circularity = st.slider("🟨 Adsorbed Fat Min Circularity (in Bubbles)", 0.0, 1.0, 0.30, 0.05)
    min_circularity = fat_min_circularity  # Fallback alias for backward compatibility

    enable_deepscan = st.checkbox(
        "🔬 Deep-Scan Adsorbed Fat inside Bubbles",
        value=False,
        help="Runs localized shadow-recovery Cellpose inside confirmed air bubbles. Keep unchecked for fast preview.",
    )
    st.caption("ℹ️ Colloid Chemistry: Fat within bubble regions is classified as adsorbed_fat.")

    enable_clahe = st.checkbox(
        "Enable CLAHE Preprocessing",
        value=True,
        help="Applies Contrast Limited Adaptive Histogram Equalization with OpenCV (clipLimit=2.0, tileGridSize=(8,8)) to boost contrast for faint pores before segmentation.",
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
        show_labels = True

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
        show_labels = True

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

        show_labels = st.checkbox(
            "🔢 Show Particle ID Numbers",
            value=True,
            key="show_particle_labels_toggle",
            help="Uncheck to hide ID numbers and get a clean color-coded phase distribution map for publications/reports.",
        )


# ---------------- MAIN APPLICATION DISPLAY ----------------
st.title("PhasePix: Quantitative SEM Microstructure Analyzer")
st.markdown(
    "High-throughput segmentation of SEM phases using Cellpose deep learning with instant phase sorting and size distribution."
)

if image_bytes is None:
    st.info("👆 Please select an image from the 'images/' folder or upload one via the sidebar to begin analysis.")
else:
    clean_stem = Path(image_name).stem

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
        1. Binary mask of confirmed automatically detected bubbles:
           if auto_air_layer is not None and confirmed_air_ids:
               c_mask = c_mask | np.isin(auto_air_layer, list(confirmed_air_ids))
        2. Filled polygon masks of active manual objects from saved_manual_items marked
           as 'air_bubbles' via cv2.fillPoly.
        If no confirmed auto air bubbles and no manual air bubbles, mask is completely False.
        """
        c_mask = np.zeros(analysis_img.shape[:2], dtype=bool)
        if auto_air_layer is not None and confirmed_air_ids:
            c_mask = c_mask | np.isin(auto_air_layer, list(confirmed_air_ids))

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
            _air_df_raw = extract_metrics(
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
        ) = _remove_nested_matryoshka_masks(
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
        _air_df_raw = extract_metrics(
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
        ) = _remove_nested_matryoshka_masks(
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
            masks = cached_segmentation(
                image=analysis_img,
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
    raw_df = extract_metrics(
        original_image=analysis_img,
        mask=masks,
        pixel_to_um_ratio=pixel_to_um_ratio,
    )

    # 3b. Reject nested "matryoshka" double detections (small objects inside bigger ones).
    #     The occupied canvas stores hole-filled footprints, so a smaller detection sitting
    #     inside the hole of a donut-shaped parent is correctly flagged as an artifact.
    if not raw_df.empty and masks is not None and np.any(masks > 0):
        auto_df = raw_df[raw_df["id"] < 10000].sort_values(
            by="area_um2", ascending=False
        )
        valid_auto_ids = []
        occupied_canvas = np.zeros(masks.shape, dtype=bool)
        masks_int = (
            masks
            if np.issubdtype(masks.dtype, np.integer)
            else masks.astype(np.int32)
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
            if overlap_area / obj_area < 0.60:
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
            raw_df[raw_df["id"] >= 10000]["id"].tolist()
        )
        if valid_all_ids:
            raw_df = (
                raw_df[raw_df["id"].isin(valid_all_ids)]
                .copy()
                .reset_index(drop=True)
            )
            # Clean the mask itself from rejected artifacts
            masks = np.where(np.isin(masks, list(valid_all_ids)), masks, 0)

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

        for _, row in raw_df[raw_df["id"] < 10000].iterrows():
            if float(row.get("diameter_um", 0.0)) <= float(max_fat_size):
                continue
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
            masks = np.where(np.isin(masks, list(pass2_rejected_ids)), 0, masks)

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
        # is tested against cumulative bubble mask (combined_bubble_mask)
        def _centroid_in_bubble(row: pd.Series) -> bool:
            if int(row["id"]) in all_air_ids:
                return False
            cx = int(round(float(row["centroid_x"])))
            cy = int(round(float(row["centroid_y"])))
            cy = np.clip(cy, 0, combined_bubble_mask.shape[0] - 1)
            cx = np.clip(cx, 0, combined_bubble_mask.shape[1] - 1)
            return bool(combined_bubble_mask[cy, cx])

        if combined_bubble_mask.any() and not raw_df.empty:
            in_bubble_vec = raw_df.apply(_centroid_in_bubble, axis=1)
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

        # Outside bubbles: standard shape filtering by circularity threshold
        circ_ok_outside = np.where(
            in_fat_size_range,
            raw_df["circularity"] >= float(fat_min_circularity),
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
        filtered_masks = np.where(np.isin(masks, list(valid_ids)), masks, 0)
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
        filtered_masks = np.where(np.isin(filtered_masks, list(active_excluded_ids)), 0, filtered_masks)

    # Filter air pass masks and dataframe in Two-Pass mode
    if _is_two_pass and air_masks is not None and not _air_df_pass.empty:
        valid_air_ids = set(_air_df_pass["id"].tolist()) - active_excluded_ids
        _air_df_pass = _air_df_pass[_air_df_pass["id"].isin(valid_air_ids)].copy()
        filtered_air_masks = np.where(np.isin(air_masks, list(valid_air_ids)), air_masks, 0)
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
            # Recompute in_bubble if not already attached
            def _check_in_b(row: pd.Series) -> bool:
                if str(row.get("phase_class", "")).lower() == "air_bubbles":
                    return False
                cx = int(round(float(row["centroid_x"])))
                cy = int(round(float(row["centroid_y"])))
                cy = np.clip(cy, 0, combined_bubble_mask.shape[0] - 1)
                cx = np.clip(cx, 0, combined_bubble_mask.shape[1] - 1)
                return bool(combined_bubble_mask[cy, cx])
            classified_df["in_bubble"] = classified_df.apply(_check_in_b, axis=1)

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
                    filtered_masks = np.where(
                        np.isin(filtered_masks, list(rejected_air_ids)),
                        0,
                        filtered_masks,
                    )
                if filtered_air_masks is not None:
                    filtered_air_masks = np.where(
                        np.isin(filtered_air_masks, list(rejected_air_ids)),
                        0,
                        filtered_air_masks,
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
    if selected_bubble_ids_for_ads and show_adsorbed and show_labels:
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

    # Summary KPI Metric cards
    # Mean Diameter only for Emulsion (fat phase) / Foam (air phase) — never for Whipped Emulsion
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
    metric_cols = 3 + (1 if unclass_count > 0 else 0) + (1 if show_mean_diam else 0) + (1 if show_adsorbed_card else 0)
    cols = st.columns(metric_cols)
    cols[0].metric("Total Objects", f"{total_objects:,}")
    if sample_type == "Emulsion":
        cols[1].metric("Air Bubbles", "0", "Disabled (Emulsion)")
        cols[2].metric(
            "Fat Particles",
            f"{fat_count:,}",
            f"{(fat_count / max(total_objects, 1) * 100):.1f}%",
        )
    elif sample_type == "Foam":
        cols[1].metric(
            "Air Bubbles",
            f"{air_count:,}",
            f"{(air_count / max(total_objects, 1) * 100):.1f}%",
        )
        cols[2].metric("Fat Particles", "0", "Disabled (Foam)")
    else:
        cols[1].metric(
            "Air Bubbles",
            f"{air_count:,}",
            f"{(air_count / max(total_objects, 1) * 100):.1f}%",
        )
        cols[2].metric(
            "Fat Particles",
            f"{fat_count:,}",
            f"{(fat_count / max(total_objects, 1) * 100):.1f}%",
        )
    next_idx = 3
    if unclass_count > 0:
        cols[next_idx].metric(
            "Unclassified",
            f"{unclass_count:,}",
            f"{(unclass_count / max(total_objects, 1) * 100):.1f}%",
        )
        next_idx += 1
    if show_adsorbed_card:
        cols[next_idx].metric(
            "Adsorbed Fat",
            f"{adsorbed_fat_count:,}",
            "Excluded from PSD",
        )
        next_idx += 1
    if show_mean_diam:
        cols[next_idx].metric("Mean Diameter", f"{avg_diam:.2f} µm")

    # Navigation View Bar
    tab_options = [
        "📊 Interactive Dashboard",
        "🖌️ Manual Correction",
        "📏 Visual Ruler",
        "🔬 Metrics",
        "🛡️ Interfacial Analytics",
        "🍇 Matrix Coalescence",
        "💾 Reproducibility & Export",
    ]
    active_tab = st.segmented_control(
        "Navigation",
        options=tab_options,
        default="📊 Interactive Dashboard",
        label_visibility="collapsed",
    )
    if active_tab is None:
        active_tab = "📊 Interactive Dashboard"

    # ---------------- TAB 1: DASHBOARD (COLUMNS) ----------------
    if active_tab == "📊 Interactive Dashboard":
        col_img, col_df = st.columns([1.15, 1.0])

        # Column 1: Original Image & Segmentation Overlay
        with col_img:
            st.subheader("1. Micrograph & Overlay")
            view_mode = st.radio(
                "Display Mode",
                ["Segmentation Overlay", "Raw Micrograph"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if "Segmentation" in view_mode:
                st.image(
                    overlay,
                    caption=f"{image_name} — Labeled Objects: {total_objects}"
                    + (f" ({effective_footer}px footer excluded)" if effective_footer > 0 else ""),
                    use_container_width=True,
                )
                # Export segmented overlay button directly beneath
                buf_overlay = io.BytesIO()
                Image.fromarray(overlay).save(buf_overlay, format="PNG")
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
                            help="Manually reclassify object between air_bubbles, fat_particles, adsorbed_fat (adsorbed on bubble wall, excluded from PSD), or Unclassified.",
                            options=["air_bubbles", "fat_particles", "adsorbed_fat", "Unclassified"],
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
                                options=["fat_particles", "adsorbed_fat", "air_bubbles", "Delete"],
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

        def _render_phase_distribution(phase_key: str, title: str, png_name: str, csv_name: str, dl_prefix: str):
            phase_csv = classified_df[classified_df["phase_class"] == phase_key] if not classified_df.empty else classified_df
            fig_phase, _ = plot_single_phase_distribution(
                classified_df,
                phase_class=phase_key,
                title=title,
                color=PALETTE_COLORS.get(phase_key),
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

            # Row 2: Target Phase Assignment (Col 1) and Action Buttons (Col 2)
            t_row2_col1, t_row2_col2 = st.columns([1.6, 1.2])
            with t_row2_col1:
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
                    btn_add_label = "➕ Add Bubble"
                else:
                    btn_add_label = "➕ Add Manual Bubble"

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
                auto_air_binary = np.isin(filtered_masks, air_ids).astype(np.uint8)

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
        st.caption(
            "🟢 **Thin Teal Outline** = Already detected by AI &nbsp;|&nbsp; "
            "🔵/🟠/🟡/🟢 **Solid Fill** = Manual Additions (Blue: Air, Orange: Fat, Gold: Adsorbed Fat, Green: Unclassified) &nbsp;|&nbsp; "
            "💡 **Pro-Tip:** Collapse the left sidebar (**<<** at top-left) to view the micrograph in full native 100% resolution. The floating toolbar stays pinned to the top as you scroll!"
        )

        if assist_mode.startswith("📦 AI Smart Box"):
            st.caption("💡 **Tip:** Drag one or more rectangles around bubbles/particles, then click **Detect & Add from Box(es)**. SAM AI (or the Scale-Normalized Cellpose fallback) fits the real SEM boundary.")
        elif assist_mode.startswith("🪄 1-Click"):
            st.caption("💡 **Tip:** Click once in the dark center of each missed bubble/pore, then click **Add Bubble**.")
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
        canvas_result = st_canvas(
            fill_color=canvas_fill_color,
            stroke_width=2,
            stroke_color=canvas_stroke_color,
            background_image=canvas_bg,
            update_streamlit=True,
            height=target_h,
            width=target_w,
            drawing_mode=canvas_drawing_mode,
            point_display_radius=4,
            display_toolbar=True,
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
                    st.info("Canvas is empty. Click in the center of one or more dark bubbles, then click '➕ Add Bubble'.", icon="💡")
                else:
                    st.info("Canvas is empty. Click points on the image, then click '➕ Add Manual Bubble'.", icon="💡")

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
                        st.session_state.get("manual_phase_overrides", {}).get(obj_id, "air_bubbles"),
                    )
                    if str(curr_phase).lower() == "unclassified":
                        curr_phase = "Unclassified"
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
                            options=["air_bubbles", "fat_particles", "adsorbed_fat", "Unclassified"],
                            required=True,
                            help="Select phase class for this manual object. 'adsorbed_fat' = adsorbed globule on bubble wall (excluded from PSD stats).",
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
                use_container_width=True,
            )

    # ---------------- TAB 3: VISUAL RULER ----------------
    elif active_tab == "📏 Visual Ruler":
        st.subheader("📏 Visual Ruler: Interactive Microstructural Caliper")
        st.markdown(
            "Draw a line across any bubble, pore, or fat droplet to measure its physical dimension. "
            "Use the measured length to accurately set the Min/Max size boundaries in the sidebar."
        )

        # Convert analysis_img (RGB or grayscale) to PIL Image
        if len(analysis_img.shape) == 2:
            ruler_pil = Image.fromarray(analysis_img).convert("RGB")
        else:
            ruler_pil = Image.fromarray(analysis_img)

        ruler_w, ruler_h = ruler_pil.size
        # Responsive canvas sizing fitting browser view
        max_canvas_width = 750
        scale_ratio = min(1.0, max_canvas_width / float(ruler_w))
        target_w = int(ruler_w * scale_ratio)
        target_h = int(ruler_h * scale_ratio)
        canvas_bg = ruler_pil.resize((target_w, target_h)) if scale_ratio < 1.0 else ruler_pil

        col_canvas, col_measure = st.columns([2.0, 1.0])

        with col_canvas:
            col_tools1, col_tools2 = st.columns(2)
            with col_tools1:
                line_color = st.color_picker("Ruler Color", "#00FF66", help="Color of the caliper measurement line.")
            with col_tools2:
                line_width = st.slider("Line Width", min_value=1, max_value=8, value=3)

            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=line_width,
                stroke_color=line_color,
                background_image=canvas_bg,
                update_streamlit=True,
                height=target_h,
                width=target_w,
                drawing_mode="line",
                key=f"visual_ruler_{image_name}",
            )

        with col_measure:
            st.markdown("### Caliper Measurement")
            st.info(f"📏 Scale: **{pixels_per_um:.2f} px/µm** ({pixel_to_um_ratio:.4f} µm/px)")

            measured_lines = []
            if canvas_result is not None and canvas_result.json_data is not None:
                canvas_objects = canvas_result.json_data.get("objects", [])
                for obj in canvas_objects:
                    if obj.get("type") == "line":
                        adj_factor = 1.0 / scale_ratio
                        raw_px = calculate_canvas_line_length(obj) * adj_factor
                        real_um = raw_px * pixel_to_um_ratio
                        measured_lines.append((raw_px, real_um))

            if measured_lines:
                latest_px, latest_um = measured_lines[-1]
                st.metric(
                    label="Measured Length (µm)",
                    value=f"{latest_um:.2f} µm",
                    delta=f"{latest_px:.1f} px",
                )
                st.success(
                    f"💡 Size guidance: Enter **{latest_um:.1f} µm** in the sidebar's "
                    "Min/Max size input for optimal classification."
                )

                if len(measured_lines) > 1:
                    st.markdown("#### Measurement History")
                    history_df = pd.DataFrame([
                        {"Line #": i + 1, "Length (px)": round(p, 1), "Length (µm)": round(u, 2)}
                        for i, (p, u) in enumerate(measured_lines)
                    ])
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.caption("👈 Click and drag a line on the micrograph to measure a pore or particle.")

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
                            use_container_width=True,
                        )
                        buf_insp = io.BytesIO()
                        Image.fromarray(inspection_img).save(buf_insp, format="PNG")
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
                                        use_container_width=True,
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
                                    Image.fromarray(crop_img).save(buf_crop, format="PNG")
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
                    use_container_width=True,
                )
                buf_network = io.BytesIO()
                Image.fromarray(network_img).save(buf_network, format="PNG")
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
            "Adsorbed Fat Min Circularity": float(ads_min_circularity),
            "Manual Phase Overrides": st.session_state.get("manual_phase_overrides", {}),
            "Show Air Bubbles": bool(show_air),
            "Show Fat Particles": bool(show_fat),
            "Air Bubbles Opacity": float(air_opacity),
            "Fat Particles Opacity": float(fat_opacity),
            "Z-Order (Layering)": str(z_order),
            "Cellpose Model": model_type,
            "Particle Diameter Setting (px)": eff_diameter if eff_diameter else "Auto",
            "Flow Threshold": float(flow_threshold),
            "Cell Probability Threshold": float(cellprob_threshold),
            "CLAHE Preprocessing": enable_clahe,
            "Hardware Accelerator": f"{device} (GPU: {is_gpu})",
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
            Image.fromarray(overlay).save(buf_overlay_tab, format="PNG")
            st.download_button(
                label="📥 Export Overlay (PNG)",
                data=buf_overlay_tab.getvalue(),
                file_name=f"segmented_overlay_{clean_stem}.png",
                mime="image/png",
                use_container_width=True,
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
