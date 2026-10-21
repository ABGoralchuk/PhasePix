"""
Module for classification and statistical visualization of segmented SEM phases.
Separates objects into air bubbles and fat particles based on mean intensity thresholds
and plots comparative kernel density estimations (KDE).
"""

from typing import Optional, Tuple
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import find_objects

logger = logging.getLogger(__name__)

# Standard color palette specified for PhasePix
PALETTE_COLORS = {
    "air_bubbles": "#1f77b4",   # Blue for air bubbles (dark pores in SEM)
    "fat_particles": "#ff7f0e",  # Orange for fat particles (bright features in SEM)
    "adsorbed_fat": "#ffd700",   # Gold for adsorbed fat particles on bubble walls
    "Unclassified": "#32cd32",   # Lime Green for unclassified particles (RGB 50, 205, 50)
    "unclassified": "#32cd32",   # Lime Green alias
}


def classify_by_size(
    df: pd.DataFrame,
    min_air_size: float,
    max_air_size: float,
    min_fat_size: float,
    max_fat_size: float,
    overlap_priority: str = "Air Bubbles",
    out_of_range_class: str = "Unclassified",
) -> pd.DataFrame:
    """
    Classify segmented objects into 'air_bubbles', 'fat_particles', or 'unclassified' based on equivalent diameter (diameter_um).

    Logic:
    - Object belongs to 'air_bubbles' if min_air_size <= diameter_um <= max_air_size.
    - Object belongs to 'fat_particles' if min_fat_size <= diameter_um <= max_fat_size.
    - Overlap conflict (matches both air and fat ranges): resolved via overlap_priority.
    - Out-of-range objects (matches neither range): resolved via out_of_range_class (defaults to 'unclassified').

    Args:
        df: DataFrame containing particle measurements with 'diameter_um' column.
        min_air_size: Lower bound diameter for air bubbles in micrometers.
        max_air_size: Upper bound diameter for air bubbles in micrometers.
        min_fat_size: Lower bound diameter for fat particles in micrometers.
        max_fat_size: Upper bound diameter for fat particles in micrometers.
        overlap_priority: Priority when an object falls into both size ranges
            ('Air Bubbles', 'Fat Particles', 'Unclassified'). Defaults to 'Air Bubbles'.
        out_of_range_class: Label for objects falling outside both ranges
            ('Unclassified', 'Air Bubbles', 'Fat Particles'). Defaults to 'Unclassified'.

    Returns:
        pd.DataFrame: Copy of DataFrame with assigned 'phase_class'.
    """
    if df.empty:
        df_copy = df.copy()
        df_copy["phase_class"] = pd.Series(dtype=str)
        return df_copy

    if "diameter_um" not in df.columns:
        raise ValueError("DataFrame must contain a 'diameter_um' column for size classification.")

    df_copy = df.copy()
    diameters = df_copy["diameter_um"].values

    in_air = (diameters >= float(min_air_size)) & (diameters <= float(max_air_size))
    in_fat = (diameters >= float(min_fat_size)) & (diameters <= float(max_fat_size))

    # Resolve priority options
    norm_overlap = overlap_priority.lower().strip()
    if "air" in norm_overlap:
        overlap_target = "air_bubbles"
    elif "fat" in norm_overlap:
        overlap_target = "fat_particles"
    else:
        overlap_target = "unclassified"

    norm_oor = out_of_range_class.lower().strip()
    if "air" in norm_oor:
        oor_target = "air_bubbles"
    elif "fat" in norm_oor:
        oor_target = "fat_particles"
    else:
        oor_target = "unclassified"

    classes = np.full(len(df_copy), oor_target, dtype=object)

    # In air only
    classes[in_air & (~in_fat)] = "air_bubbles"
    # In fat only
    classes[in_fat & (~in_air)] = "fat_particles"
    # In both
    classes[in_air & in_fat] = overlap_target

    df_copy["phase_class"] = classes
    return df_copy


def classify_by_intensity(
    df: pd.DataFrame,
    intensity_threshold: float = 120.0,
) -> pd.DataFrame:
    """
    Legacy classifier: separates into 'air_bubbles' and 'fat_particles' based on mean intensity.
    Kept for backwards compatibility.
    """
    if not (0.0 <= intensity_threshold <= 255.0):
        raise ValueError(f"intensity_threshold must be between 0 and 255, got {intensity_threshold}")

    if df.empty:
        df_copy = df.copy()
        df_copy["phase_class"] = pd.Series(dtype=str)
        return df_copy

    if "mean_intensity" not in df.columns:
        raise ValueError("DataFrame must contain a 'mean_intensity' column for classification.")

    df_copy = df.copy()
    df_copy["phase_class"] = np.where(
        df_copy["mean_intensity"] < intensity_threshold,
        "air_bubbles",
        "fat_particles",
    )
    return df_copy


def split_by_intensity(
    df: pd.DataFrame,
    intensity_threshold: float = 120.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into two separate DataFrames: (air_bubbles, fat_particles).

    Args:
        df: Input DataFrame containing 'mean_intensity'.
        intensity_threshold: Intensity cutoff threshold (0 to 255).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (air_bubbles_df, fat_particles_df)
    """
    classified_df = classify_by_intensity(df, intensity_threshold=intensity_threshold)
    air_df = classified_df[classified_df["phase_class"] == "air_bubbles"].copy()
    fat_df = classified_df[classified_df["phase_class"] == "fat_particles"].copy()
    return air_df, fat_df


def plot_single_phase_distribution(
    df: pd.DataFrame,
    phase_class: str,
    title: Optional[str] = None,
    color: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4.5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot size distribution for a single phase with Y-axis as percentage of particles.

    Uses sns.histplot(..., stat="percent", kde=True, bins="auto") so the Y axis
    shows Percentage of Particles (%) within that phase (not probability density).

    Args:
        df: DataFrame with 'diameter_um' and 'phase_class' columns.
        phase_class: Target phase key ('fat_particles' or 'air_bubbles').
        title: Optional plot title.
        color: Optional matplotlib/seaborn color; defaults from PALETTE_COLORS.
        figsize: Figure dimensions (width, height) in inches.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize)

    phase_key = str(phase_class).strip()
    display_name = phase_key.replace("_", " ").title()
    plot_color = color or PALETTE_COLORS.get(phase_key, "#4c78a8")
    plot_title = title or f"{display_name} Size Distribution"

    if df is None or df.empty or "diameter_um" not in df.columns:
        ax.text(
            0.5, 0.5, "No segmented objects detected to plot distribution.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#555555", style="italic",
        )
        ax.set_title(plot_title, fontsize=13, weight="bold")
        ax.set_xlabel("Equivalent Diameter (µm)", fontsize=11)
        ax.set_ylabel("Percentage of Particles (%)", fontsize=11)
        fig.tight_layout()
        return fig, ax

    data = df.copy()
    if "phase_class" in data.columns:
        if phase_key.lower() == "fat_particles":
            # PSD strictly excludes adsorbed_fat globules to prevent size underestimation
            valid_mask = (data["phase_class"].astype(str).str.lower() == "fat_particles") & (
                ~data["phase_class"].astype(str).str.lower().isin(["adsorbed_fat"])
            )
            if "is_partial_in_bubble" in data.columns:
                valid_mask = valid_mask & (~data["is_partial_in_bubble"].fillna(False).astype(bool))
            phase_data = data[valid_mask]["diameter_um"].dropna()
        else:
            phase_data = data[
                data["phase_class"].astype(str).str.lower() == phase_key.lower()
            ]["diameter_um"].dropna()
    else:
        phase_data = data["diameter_um"].dropna()

    if len(phase_data) == 0:
        ax.text(
            0.5, 0.5, f"No {display_name} objects to plot.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#555555", style="italic",
        )
        ax.set_title(plot_title, fontsize=13, weight="bold")
        ax.set_xlabel("Equivalent Diameter (µm)", fontsize=11)
        ax.set_ylabel("Percentage of Particles (%)", fontsize=11)
        fig.tight_layout()
        return fig, ax

    if len(phase_data) == 1 or phase_data.nunique() == 1:
        single_val = float(phase_data.iloc[0])
        ax.bar([single_val], [100.0], width=max(single_val * 0.05, 0.2),
               color=plot_color, alpha=0.55, edgecolor=plot_color,
               label=f"{display_name} (n={len(phase_data)})")
        ax.axvline(single_val, color=plot_color, linestyle="--", linewidth=1.5)
    else:
        sns.histplot(
            data=phase_data,
            stat="percent",
            kde=True,
            bins="auto",
            color=plot_color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.55,
            ax=ax,
            label=f"{display_name} (n={len(phase_data)})",
        )

    ax.legend(frameon=True, loc="upper right", fontsize=10)
    ax.set_title(plot_title, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Equivalent Diameter (µm)", fontsize=11, labelpad=8)
    ax.set_ylabel("Percentage of Particles (%)", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    current_xlim = ax.get_xlim()
    ax.set_xlim(left=max(0.0, current_xlim[0]))
    fig.tight_layout()
    return fig, ax


def plot_size_distribution(
    df: pd.DataFrame,
    intensity_threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (9, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Legacy combined plot kept for compatibility. Prefer plot_single_phase_distribution.

    Plots percentage histograms for both air and fat phases on one axes when present.
    """
    fig, ax = plt.subplots(figsize=figsize)

    data = df.copy() if df is not None else pd.DataFrame()
    if intensity_threshold is not None or (not data.empty and "phase_class" not in data.columns):
        thresh = intensity_threshold if intensity_threshold is not None else 120.0
        data = classify_by_intensity(data, intensity_threshold=thresh)

    if data.empty or "diameter_um" not in data.columns:
        ax.text(
            0.5, 0.5, "No segmented objects detected to plot distribution.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#555555", style="italic",
        )
        ax.set_title("Size Distribution by Phase", fontsize=13, weight="bold")
        ax.set_xlabel(r"Equivalent Diameter ($\mu$m)", fontsize=11)
        ax.set_ylabel("Percentage of Particles (%)", fontsize=11)
        fig.tight_layout()
        return fig, ax

    plotted = False
    for phase_key, label in (
        ("air_bubbles", "Air Bubbles"),
        ("fat_particles", "Fat Particles"),
    ):
        phase_data = data[data["phase_class"] == phase_key]["diameter_um"].dropna()
        if len(phase_data) == 0:
            continue
        plotted = True
        if len(phase_data) == 1 or phase_data.nunique() == 1:
            single_val = float(phase_data.iloc[0])
            ax.axvline(
                single_val,
                color=PALETTE_COLORS[phase_key],
                linestyle="--",
                linewidth=2.0,
                label=f"{label} (n={len(phase_data)}, d={single_val:.2f} µm)",
            )
        else:
            sns.histplot(
                data=phase_data,
                stat="percent",
                kde=True,
                bins="auto",
                color=PALETTE_COLORS[phase_key],
                element="step",
                fill=True,
                alpha=0.25,
                linewidth=1.5,
                ax=ax,
                label=f"{label} (n={len(phase_data)})",
            )

    if plotted:
        ax.legend(frameon=True, loc="upper right", fontsize=10)
    else:
        ax.text(
            0.5, 0.5, "No air/fat objects to plot.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#555555", style="italic",
        )

    ax.set_title("Equivalent Diameter Distribution by Phase", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Equivalent Diameter (µm)", fontsize=11, labelpad=8)
    ax.set_ylabel("Percentage of Particles (%)", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    current_xlim = ax.get_xlim()
    ax.set_xlim(left=max(0.0, current_xlim[0]))
    fig.tight_layout()
    return fig, ax


def apply_footer_dimming(
    image: np.ndarray,
    footer_height: int,
    dim_factor: float = 0.4,
) -> np.ndarray:
    """
    Overlay a translucent dimmed rectangle on the footer portion of an image.
    Preserves visibility of SEM status bar text while indicating that the region
    is excluded from segmentation analysis.

    Args:
        image: 2D grayscale or 3D RGB SEM micrograph.
        footer_height: Height of the footer region in pixels from the bottom edge.
        dim_factor: Multiplier for footer intensity (0.0=black, 1.0=unchanged). Defaults to 0.4.

    Returns:
        np.ndarray: 3D uint8 RGB image with dimmed footer band.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray")

    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1).astype(np.uint8)
    elif image.ndim == 3:
        rgb = image.copy().astype(np.uint8)
    else:
        raise ValueError(f"Expected 2D or 3D image, got ndim={image.ndim}")

    h = rgb.shape[0]
    if 0 < footer_height < h:
        footer_slice = rgb[h - footer_height : h, :].astype(np.float32)
        rgb[h - footer_height : h, :] = np.clip(footer_slice * dim_factor, 0, 255).astype(np.uint8)

    return rgb


def create_mask_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    df: Optional[pd.DataFrame] = None,
    alpha: float = 0.4,
    air_opacity: float = 0.3,
    fat_opacity: float = 0.3,
    show_air: bool = True,
    show_fat: bool = True,
    show_adsorbed: bool = True,
    adsorbed_opacity: float = 0.45,
    z_order: str = "Fat over Air",
    mask: Optional[np.ndarray] = None,
    manual_polygons: Optional[list] = None,
    air_masks: Optional[np.ndarray] = None,
    show_labels: bool = True,
    ads_particles_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> np.ndarray:
    """
    Create an RGB image with flexible phase layering, individual opacities, and particle ID numbering.

    Color coding (RGB):
    - 'fat_particles': Orange (RGB: 255, 165, 0)
    - 'air_bubbles': Deep Sky Blue (RGB: 0, 191, 255)
    - Other / unlabeled: Yellow (RGB: 255, 235, 59)

    Features:
    - Layering order controlled by z_order: 'Fat over Air' (fat on top) or 'Air over Fat' (air on top).
    - Class-specific opacities via air_opacity and fat_opacity using cv2.addWeighted.
    - Class visibility toggles: show_air and show_fat (disabled classes are neither drawn nor numbered).
    - Particle ID numbering drawn at centroids with a crisp black outline and white interior text (toggleable via show_labels).
    - Supports manual_polygons rendered underneath automatic masks for seamless overlapping objects.
    - Supports separate air_masks layer for two-pass segmentation overlays.

    Args:
        image: 2D grayscale or 3D RGB base image.
        masks: 2D integer labeled mask array where 0=background, 1..N=objects.
        df: Optional DataFrame containing columns 'id', 'phase_class', 'centroid_x', 'centroid_y'.
        alpha: Fallback opacity if class-specific opacities are not applicable (defaults to 0.4).
        air_opacity: Opacity for air bubbles overlay layer (0.0 to 1.0, defaults to 0.3).
        fat_opacity: Opacity for fat particles overlay layer (0.0 to 1.0, defaults to 0.3).
        show_air: Whether to render and number air bubbles (defaults to True).
        show_fat: Whether to render and number fat particles (defaults to True).
        z_order: Layer rendering sequence ('Fat over Air' or 'Air over Fat'). Defaults to 'Fat over Air'.
        mask: Optional alias for masks (for backward compatibility).
        manual_polygons: Optional list of manual contour dicts ({'id': int, 'points': np.ndarray}).
        air_masks: Optional 2D integer labeled mask array specifically for air bubbles (two-pass mode).
        show_labels: Whether to render particle ID text labels at centroids (defaults to True).
        ads_particles_df: Optional table with 'id' and 'contact_angle_deg' (output of
            compute_interfacial_adsorption_metrics). When supplied, adsorbed fat globules are
            labelled with their apparent contact angle f"{theta:.0f}°" instead of their ID.

    Returns:
        np.ndarray: 3D uint8 RGB composite image with tinted particles and ID labels.
    """
    if "show_labels" in kwargs:
        show_labels = bool(kwargs.pop("show_labels"))
    if masks is None and mask is not None:
        masks = mask
    if masks is None:
        raise ValueError("masks array must be provided.")

    if not isinstance(image, np.ndarray) or not isinstance(masks, np.ndarray):
        raise TypeError("Both image and masks must be numpy ndarrays")

    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1).astype(np.uint8)
    elif image.ndim == 3:
        rgb = image.copy().astype(np.uint8)
    else:
        raise ValueError(f"Expected 2D or 3D image, got ndim={image.ndim}")

    mask_2d = masks.squeeze() if masks.ndim == 3 else masks
    if mask_2d.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {masks.shape}")

    if rgb.shape[:2] != mask_2d.shape:
        raise ValueError(
            f"Image shape {rgb.shape[:2]} does not match mask shape {mask_2d.shape}"
        )

    # If masks are empty or max is 0 and no manual polygons are present, return original rgb
    has_air = air_masks is not None and bool(air_masks.max() > 0)
    if mask_2d.max() == 0 and not has_air and not manual_polygons:
        return rgb

    blended = rgb.copy()

    # Phase color definitions (RGB format)
    FAT_COLOR = (255, 165, 0)        # Orange for fat particles
    AIR_COLOR = (0, 191, 255)        # Deep sky blue for air bubbles
    UNCLASS_COLOR = (50, 205, 50)    # Lime Green for unclassified particles (RGB 50, 205, 50)
    DEFAULT_COLOR = (255, 235, 59)   # Fallback yellow

    # Fallback for unclassified / raw mask when df is not provided
    if df is None or df.empty:
        if mask_2d.max() > 0 and alpha > 0.0:
            fallback_layer = np.zeros_like(blended)
            fallback_layer[mask_2d > 0] = DEFAULT_COLOR
            weighted = cv2.addWeighted(blended, 1.0 - alpha, fallback_layer, alpha, 0)
            blended[mask_2d > 0] = weighted[mask_2d > 0]
        return blended

    def _draw_manual_for_phase(target_phase: str, phase_color, phase_opacity: float):
        """Draw manual polygons that belong to target_phase (respects Z-order)."""
        if not manual_polygons:
            return
        id_to_phase = dict(zip(df["id"], df["phase_class"]))
        for mp in manual_polygons:
            poly_id = mp.get("id")
            phase = id_to_phase.get(poly_id, "Unclassified")
            phase_norm = str(phase).lower()
            target_norm = str(target_phase).lower()
            if phase_norm != target_norm:
                if (
                    target_norm == "unclassified"
                    and phase_norm in ("unclassified", "unclassified")
                ):
                    pass
                else:
                    continue
            pts = np.asarray(mp.get("points"), dtype=np.float32)
            if len(pts) < 3:
                continue
            pts_int = np.round(pts).astype(np.int32)
            temp_poly_layer = np.zeros(blended.shape[:2], dtype=np.uint8)
            cv2.fillPoly(temp_poly_layer, [pts_int], 1)
            poly_fill_mask = temp_poly_layer > 0
            if np.any(poly_fill_mask):
                draw_opacity = float(np.clip(phase_opacity, 0.0, 1.0))
                blended[poly_fill_mask] = (
                    blended[poly_fill_mask].astype(np.float32) * (1.0 - draw_opacity)
                    + np.array(phase_color, dtype=np.float32) * draw_opacity
                ).clip(0, 255).astype(np.uint8)

            outline_color = (
                (255, 200, 50)
                if (target_phase == "fat_particles" and z_order == "Fat over Air")
                else phase_color
            )
            cv2.polylines(blended, [pts_int], isClosed=True, color=outline_color, thickness=1, lineType=cv2.LINE_AA)

    def _draw_id_label(cx: float, cy: float, obj_id: int, text: Optional[str] = None):
        text = str(obj_id) if text is None else str(text)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        tx = max(1, min(rgb.shape[1] - tw - 2, int(round(cx - tw / 2.0))))
        ty = max(th + 1, min(rgb.shape[0] - 2, int(round(cy + th / 2.0))))
        cv2.putText(
            blended, text, (tx, ty), font, font_scale, (0, 0, 0),
            thickness=outline_thickness, lineType=cv2.LINE_AA,
        )
        cv2.putText(
            blended, text, (tx, ty), font, font_scale, (255, 255, 255),
            thickness=font_thickness, lineType=cv2.LINE_AA,
        )

    # Determine Z-order layering sequence:
    # Bottom layer rendered first, top layer rendered second on top.
    if z_order == "Air over Fat":
        phase_sequence = ["fat_particles", "air_bubbles"]
    else:
        # Default: "Fat over Air" (fat rendered over air)
        phase_sequence = ["air_bubbles", "fat_particles"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    font_thickness = 1
    outline_thickness = 2

    # 1. Render unclassified (auto + manual) first as base underlay
    unclass_mask_series = df["phase_class"].astype(str).str.lower() == "unclassified"
    unclass_ids = df.loc[unclass_mask_series, "id"].tolist()
    if unclass_ids:
        unclass_mask = np.isin(mask_2d, unclass_ids)
        if air_masks is not None:
            air_m2d = air_masks.squeeze() if air_masks.ndim == 3 else air_masks
            if air_m2d.shape == mask_2d.shape:
                unclass_mask = unclass_mask | np.isin(air_m2d, unclass_ids)
        if np.any(unclass_mask):
            unclass_opacity = 0.35
            blended[unclass_mask] = (
                blended[unclass_mask].astype(np.float32) * (1.0 - unclass_opacity)
                + np.array(UNCLASS_COLOR, dtype=np.float32) * unclass_opacity
            ).clip(0, 255).astype(np.uint8)
        _draw_manual_for_phase("Unclassified", UNCLASS_COLOR, 0.35)

    # 2. Render fills (auto masks + matching manual polygons) in Z-order
    for phase in phase_sequence:
        if phase == "air_bubbles":
            if not show_air or air_opacity <= 0.0:
                continue
            phase_opacity = float(np.clip(air_opacity, 0.0, 1.0))
            phase_color = AIR_COLOR
        elif phase == "fat_particles":
            if not show_fat or fat_opacity <= 0.0:
                continue
            phase_opacity = float(np.clip(fat_opacity, 0.0, 1.0))
            phase_color = FAT_COLOR
        else:
            continue

        phase_ids = df.loc[df["phase_class"] == phase, "id"].tolist()
        if phase_ids:
            phase_mask = np.isin(mask_2d, phase_ids)
            if air_masks is not None:
                air_m2d = air_masks.squeeze() if air_masks.ndim == 3 else air_masks
                if air_m2d.shape == mask_2d.shape:
                    phase_mask = phase_mask | np.isin(air_m2d, phase_ids)
        else:
            phase_mask = np.zeros_like(mask_2d, dtype=bool)

        if np.any(phase_mask):
            blended[phase_mask] = (
                blended[phase_mask].astype(np.float32) * (1.0 - phase_opacity)
                + np.array(phase_color, dtype=np.float32) * phase_opacity
            ).clip(0, 255).astype(np.uint8)

            # Highlight fat particles with a crisp 1 px outline in Fat over Air mode
            if phase == "fat_particles" and z_order == "Fat over Air":
                fat_cnts, _ = cv2.findContours(phase_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                if fat_cnts:
                    cv2.drawContours(blended, fat_cnts, -1, (255, 200, 50), 1, lineType=cv2.LINE_AA)

            # Smooth crater wall contour for air bubbles (unified quality with manual AI mode)
            if phase == "air_bubbles":
                air_cnts, _ = cv2.findContours(phase_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                for acnt in air_cnts:
                    if len(acnt) >= 8:
                        try:
                            from analyzer import _smooth_contour_fourier
                            smoothed_acnt = _smooth_contour_fourier(acnt.reshape(-1, 2).astype(np.float32), n_harmonics=8, num_points=160)
                            cv2.polylines(blended, [smoothed_acnt.round().astype(np.int32)], isClosed=True, color=(100, 220, 255), thickness=1, lineType=cv2.LINE_AA)
                        except Exception:
                            cv2.drawContours(blended, [acnt], -1, (100, 220, 255), 1, lineType=cv2.LINE_AA)

        # Manual polygons of this phase are drawn in the SAME Z-order pass
        _draw_manual_for_phase(phase, phase_color, phase_opacity)

    # 2b. Render adsorbed_fat (adsorbed globules inside bubble craters)
    # Rendered with distinct vibrant Gold color (255, 215, 0)
    ADSORBED_COLOR = (255, 215, 0)
    if show_adsorbed and adsorbed_opacity > 0.0:
        pf_opacity = float(np.clip(adsorbed_opacity, 0.0, 1.0))
        partial_ids = df.loc[df["phase_class"].astype(str).str.lower().isin(["adsorbed_fat"]), "id"].tolist()
        if partial_ids:
            partial_mask = np.isin(mask_2d, partial_ids)
            if np.any(partial_mask):
                blended[partial_mask] = (
                    blended[partial_mask].astype(np.float32) * (1.0 - pf_opacity)
                    + np.array(ADSORBED_COLOR, dtype=np.float32) * pf_opacity
                ).clip(0, 255).astype(np.uint8)
            pf_cnts, _ = cv2.findContours(partial_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for cnt in pf_cnts:
                cv2.drawContours(blended, [cnt], -1, (255, 240, 100), 1, lineType=cv2.LINE_AA)
        _draw_manual_for_phase("adsorbed_fat", ADSORBED_COLOR, pf_opacity)

    # 3. Draw particle ID numbers in the same Z-order (top phase labels last = on top)
    if show_labels:
        if unclass_ids:
            unclass_df = df[unclass_mask_series]
            for _, row in unclass_df.iterrows():
                centroid_x = row.get("centroid_x")
                centroid_y = row.get("centroid_y")
                if pd.isna(centroid_x) or pd.isna(centroid_y):
                    continue
                cx, cy = float(centroid_x), float(centroid_y)
                if not (0 <= cx < rgb.shape[1] and 0 <= cy < rgb.shape[0]):
                    continue
                _draw_id_label(cx, cy, int(row["id"]))

        for phase in phase_sequence:
            if phase == "air_bubbles" and not show_air:
                continue
            if phase == "fat_particles" and not show_fat:
                continue

            phase_df = df[df["phase_class"] == phase]
            for _, row in phase_df.iterrows():
                centroid_x = row.get("centroid_x")
                centroid_y = row.get("centroid_y")
                if pd.isna(centroid_x) or pd.isna(centroid_y):
                    continue
                cx, cy = float(centroid_x), float(centroid_y)
                if not (0 <= cx < rgb.shape[1] and 0 <= cy < rgb.shape[0]):
                    continue
                _draw_id_label(cx, cy, int(row["id"]))

        # Draw labels for adsorbed_fat objects when show_adsorbed is enabled
        if show_adsorbed:
            theta_lookup: dict = {}
            if (
                isinstance(ads_particles_df, pd.DataFrame)
                and not ads_particles_df.empty
                and "id" in ads_particles_df.columns
                and "contact_angle_deg" in ads_particles_df.columns
            ):
                for _, theta_row in ads_particles_df.iterrows():
                    try:
                        theta_id = int(theta_row["id"])
                    except (TypeError, ValueError):
                        continue
                    theta_value = pd.to_numeric(
                        theta_row.get("contact_angle_deg"), errors="coerce"
                    )
                    if pd.notna(theta_value):
                        theta_lookup[theta_id] = float(theta_value)

            pf_df = df[df["phase_class"].astype(str).str.lower().isin(["adsorbed_fat"])]
            for _, row in pf_df.iterrows():
                centroid_x = row.get("centroid_x")
                centroid_y = row.get("centroid_y")
                if pd.isna(centroid_x) or pd.isna(centroid_y):
                    continue
                cx, cy = float(centroid_x), float(centroid_y)
                if not (0 <= cx < rgb.shape[1] and 0 <= cy < rgb.shape[0]):
                    continue
                obj_id = int(row["id"])
                theta_value = theta_lookup.get(obj_id)
                label_text = (
                    f"{theta_value:.0f}°" if theta_value is not None else str(obj_id)
                )
                _draw_id_label(cx, cy, obj_id, text=label_text)

    return blended


# ---------------------------------------------------------------------------
# Quality-control overlays for Pickering adsorption & matrix coalescence
# ---------------------------------------------------------------------------


def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a 2D grayscale or 3D color image into an RGB uint8 array."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray")
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1).astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 1:
            gray = image[:, :, 0]
            return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
        if image.shape[2] == 4:
            return np.ascontiguousarray(image[:, :, :3]).astype(np.uint8)
        if image.shape[2] == 3:
            return image.copy().astype(np.uint8)
        raise ValueError(f"Expected 1, 3 or 4 image channels, got {image.shape[2]}")
    raise ValueError(f"Expected 2D or 3D image, got ndim={image.ndim}")


def _rasterize_manual_object(
    manual_polygons: Optional[list],
    obj_id: int,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Rasterize manual polygons belonging to obj_id into a uint8 mask."""
    canvas = np.zeros(shape, dtype=np.uint8)
    if not manual_polygons:
        return canvas
    for polygon in manual_polygons:
        if not isinstance(polygon, dict):
            continue
        try:
            if int(polygon.get("id")) != int(obj_id):
                continue
        except (TypeError, ValueError):
            continue
        pts = polygon.get("points")
        if pts is None:
            continue
        pts_arr = np.asarray(pts, dtype=np.float32)
        if pts_arr.ndim != 2 or pts_arr.shape[0] < 3 or pts_arr.shape[1] < 2:
            continue
        cv2.fillPoly(canvas, [np.round(pts_arr[:, :2]).astype(np.int32)], 1)
    return canvas


def _object_binary_mask(
    obj_id: int,
    shape: Tuple[int, int],
    manual_polygons: Optional[list],
    *mask_sources: Optional[np.ndarray],
) -> np.ndarray:
    """Build a binary mask of one object from manual polygons or labeled mask stacks.

    The first labeled stack that actually contains ``obj_id`` wins, so identical numeric
    labels used by different stacks (e.g. bubble id 1 and globule id 1) are never merged.
    """
    obj_id = int(obj_id)
    if obj_id >= 10000:
        return _rasterize_manual_object(manual_polygons, obj_id, shape)

    for source in mask_sources:
        if isinstance(source, np.ndarray) and source.ndim >= 2:
            src = source.squeeze() if source.ndim == 3 else source
            if src.shape[:2] == tuple(shape):
                candidate = src == obj_id
                if np.any(candidate):
                    return candidate.astype(np.uint8)
    return np.zeros(shape, dtype=np.uint8)


def _apply_color_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    opacity: float,
) -> None:
    """Alpha-blend a solid RGB color into rgb pixels selected by a boolean mask (in place)."""
    if mask is None or not np.any(mask):
        return
    alpha = float(np.clip(opacity, 0.0, 1.0))
    if alpha <= 0.0:
        return
    color_arr = np.array(color, dtype=np.float32)
    rgb[mask] = (
        rgb[mask].astype(np.float32) * (1.0 - alpha) + color_arr * alpha
    ).clip(0, 255).astype(np.uint8)


def render_selected_bubbles_inspection_overlay(
    image: np.ndarray,
    df: pd.DataFrame,
    selected_bubble_ids: list[int],
    air_masks: Optional[np.ndarray],
    fat_masks: Optional[np.ndarray],
    manual_polygons: Optional[list[dict]],
    ads_particles_df: Optional[pd.DataFrame] = None,
    show_contact_angles: bool = True,
) -> np.ndarray:
    """
    Build a QA inspection overlay for the user-selected bubbles and their adsorbed fat.

    Everything outside the selected bubble masks is dimmed (x0.45) so the inspected
    bubbles stay the only bright regions. Their contours are drawn in bright cyan
    (RGB 0, 210, 255), adsorbed fat globules inside them are tinted gold
    (RGB 255, 215, 0, alpha 0.50) with a crisp outline (RGB 255, 240, 120), and each
    adsorbed globule can be annotated with its apparent contact angle theta.

    Args:
        image: 2D grayscale or 3D RGB base micrograph.
        df: Classified metrics DataFrame ('id', 'phase_class', 'centroid_x', 'centroid_y').
        selected_bubble_ids: Bubble IDs selected for quality-control inspection.
        air_masks: Optional labeled air-bubble mask (0 = background).
        fat_masks: Optional labeled fat/particle mask (0 = background).
        manual_polygons: Optional manual contour dicts ({'id': int, 'points': (N, 2)}).
        ads_particles_df: Optional output of compute_interfacial_adsorption_metrics
            containing 'id', 'parent_bubble_id' and 'contact_angle_deg'.
        show_contact_angles: Annotate adsorbed globules with theta when
            ads_particles_df is supplied (defaults to True).

    Returns:
        np.ndarray: RGB uint8 inspection overlay.
    """
    rgb = _ensure_rgb_uint8(image)
    shape = rgb.shape[:2]

    selected_ids: list[int] = []
    id_iterable = selected_bubble_ids if selected_bubble_ids is not None else []
    for raw_id in id_iterable:
        try:
            bubble_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if bubble_id not in selected_ids:
            selected_ids.append(bubble_id)

    bubble_masks: dict[int, np.ndarray] = {}
    selected_union = np.zeros(shape, dtype=np.uint8)
    for bubble_id in selected_ids:
        b_mask = _object_binary_mask(
            bubble_id, shape, manual_polygons, air_masks, fat_masks
        )
        if np.any(b_mask):
            bubble_masks[bubble_id] = b_mask
            selected_union |= b_mask

    # Nothing to inspect -> return the untouched base image
    if not bubble_masks:
        return rgb

    # Dim everything outside the selected bubbles (x0.45)
    selected_bool = selected_union > 0
    rgb[~selected_bool] = (
        rgb[~selected_bool].astype(np.float32) * 0.45
    ).clip(0, 255).astype(np.uint8)


    # Collect adsorbed-fat ids from the classified table and/or the adsorption results
    ads_ids: set = set()
    if isinstance(df, pd.DataFrame) and not df.empty and "id" in df.columns:
        if "phase_class" in df.columns:
            phase_norm = df["phase_class"].astype(str).str.lower()
            for value in df.loc[phase_norm == "adsorbed_fat", "id"].tolist():
                try:
                    ads_ids.add(int(value))
                except (TypeError, ValueError):
                    continue
    if (
        isinstance(ads_particles_df, pd.DataFrame)
        and not ads_particles_df.empty
        and "id" in ads_particles_df.columns
    ):
        for value in ads_particles_df["id"].tolist():
            try:
                ads_ids.add(int(value))
            except (TypeError, ValueError):
                continue

    # Build the adsorbed-fat mask (auto labels + manual polygons) inside selected bubbles.
    # fat_masks is authoritative; air_masks is only consulted for ids absent from it, so
    # colliding numeric labels in the two stacks cannot flood a whole bubble.
    ads_union = np.zeros(shape, dtype=np.uint8)
    auto_ads_ids = [pid for pid in ads_ids if pid < 10000]
    if auto_ads_ids:
        auto_ads_set = set(auto_ads_ids)
        fat_present: set = set()
        if isinstance(fat_masks, np.ndarray) and fat_masks.ndim >= 2:
            fat_2d = fat_masks.squeeze() if fat_masks.ndim == 3 else fat_masks
            if fat_2d.shape[:2] == shape:
                ads_union |= np.isin(fat_2d, auto_ads_ids).astype(np.uint8)
                if isinstance(air_masks, np.ndarray):
                    fat_present = {
                        int(value)
                        for value in np.unique(fat_2d)
                        if int(value) in auto_ads_set
                    }
        fallback_ids = [pid for pid in auto_ads_ids if pid not in fat_present]
        if fallback_ids and isinstance(air_masks, np.ndarray) and air_masks.ndim >= 2:
            air_2d = air_masks.squeeze() if air_masks.ndim == 3 else air_masks
            if air_2d.shape[:2] == shape:
                ads_union |= np.isin(air_2d, fallback_ids).astype(np.uint8)
    for pid in [pid for pid in ads_ids if pid >= 10000]:
        ads_union |= _object_binary_mask(pid, shape, manual_polygons, fat_masks, air_masks)

    ads_union = (ads_union > 0) & selected_bool
    _apply_color_overlay(rgb, ads_union, (255, 215, 0), 0.50)
    if np.any(ads_union):
        ads_cnts, _ = cv2.findContours(
            ads_union.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(rgb, ads_cnts, -1, (255, 240, 120), 1, lineType=cv2.LINE_AA)

    # Bright cyan outlines of the selected bubbles
    for b_mask in bubble_masks.values():
        bubble_cnts, _ = cv2.findContours(
            b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(rgb, bubble_cnts, -1, (0, 210, 255), 2, lineType=cv2.LINE_AA)

    # Compact contact-angle annotations next to each adsorbed globule centroid
    if (
        show_contact_angles
        and isinstance(ads_particles_df, pd.DataFrame)
        and not ads_particles_df.empty
        and "id" in ads_particles_df.columns
    ):
        centroid_lookup: dict = {}
        if (
            isinstance(df, pd.DataFrame)
            and not df.empty
            and {"id", "centroid_x", "centroid_y"}.issubset(df.columns)
        ):
            for _, entry in df.iterrows():
                try:
                    obj_id = int(entry["id"])
                except (TypeError, ValueError):
                    continue
                cx_val = entry.get("centroid_x")
                cy_val = entry.get("centroid_y")
                if pd.notna(cx_val) and pd.notna(cy_val):
                    centroid_lookup[obj_id] = (float(cx_val), float(cy_val))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.34
        font_thickness = 1
        outline_thickness = 2
        for _, row in ads_particles_df.iterrows():
            try:
                particle_id = int(row["id"])
            except (TypeError, ValueError):
                continue

            parent_id = row.get("parent_bubble_id")
            if pd.notna(parent_id):
                try:
                    if int(parent_id) not in bubble_masks:
                        continue
                except (TypeError, ValueError):
                    pass

            centroid = centroid_lookup.get(particle_id)
            if centroid is None:
                p_mask = _object_binary_mask(
                    particle_id, shape, manual_polygons, fat_masks, air_masks
                )
                ys, xs = np.nonzero(p_mask)
                if len(xs) == 0:
                    continue
                centroid = (float(xs.mean()), float(ys.mean()))

            cx, cy = centroid
            if not (0 <= cx < shape[1] and 0 <= cy < shape[0]):
                continue
            if not selected_bool[int(round(cy)), int(round(cx))]:
                continue

            theta_val = row.get("contact_angle_deg")
            try:
                theta = float(theta_val)
            except (TypeError, ValueError):
                theta = float("nan")
            text = f"{theta:.0f}°" if np.isfinite(theta) else str(particle_id)

            (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            tx = int(np.clip(round(cx - tw / 2.0), 1, max(1, shape[1] - tw - 2)))
            ty = int(np.clip(round(cy - 5.0), th + 1, shape[0] - 2))
            cv2.putText(
                rgb, text, (tx, ty), font, font_scale, (0, 0, 0),
                thickness=outline_thickness, lineType=cv2.LINE_AA,
            )
            cv2.putText(
                rgb, text, (tx, ty), font, font_scale, (255, 255, 255),
                thickness=font_thickness, lineType=cv2.LINE_AA,
            )

    return rgb


def render_coalescence_network_overlay(
    image: np.ndarray,
    particles_coal_df: pd.DataFrame,
    fat_masks: np.ndarray,
    edges: list[Tuple[int, int]],
    roi_box_px: Optional[Tuple[int, int, int, int]] = None,
    show_bridges: bool = True,
) -> np.ndarray:
    """
    Render the matrix coalescence contact network (RAG) on top of the micrograph.

    Globule masks from ``fat_masks`` are tinted according to their coordination number Z:
    Z = 0 -> light blue (60, 180, 255), Z = 1 -> lime green (80, 235, 100),
    Z = 2 -> golden yellow (255, 210, 40), Z >= 3 -> bright red/purple (255, 65, 85).
    When ``show_bridges`` is True every contact edge is drawn as a white-blue bridge
    between the two globule centroids with small node dots.

    Args:
        image: 2D grayscale or 3D RGB base micrograph.
        particles_coal_df: Output of compute_matrix_coalescence_metrics with columns
            'id', 'coordination_number', 'centroid_x', 'centroid_y'.
        fat_masks: Labeled fat/particle mask (0 = background) used to rasterize globules.
        edges: Contact edges as (id_1, id_2) pairs.
        roi_box_px: Optional region of interest (x1, y1, x2, y2) in pixels.
        show_bridges: Draw coalescence bridges between contacting centroids (default True).

    Returns:
        np.ndarray: RGB uint8 coalescence-network overlay.
    """
    rgb = _ensure_rgb_uint8(image)
    shape = rgb.shape[:2]

    # ROI: dim everything outside the box (x0.40) and highlight the border in green
    if roi_box_px is not None:
        try:
            rx1, ry1, rx2, ry2 = (int(round(float(v))) for v in roi_box_px)
            x_lo, x_hi = sorted((rx1, rx2))
            y_lo, y_hi = sorted((ry1, ry2))
            x_lo = int(np.clip(x_lo, 0, max(0, shape[1] - 1)))
            x_hi = int(np.clip(x_hi, 0, max(0, shape[1] - 1)))
            y_lo = int(np.clip(y_lo, 0, max(0, shape[0] - 1)))
            y_hi = int(np.clip(y_hi, 0, max(0, shape[0] - 1)))

            roi_mask = np.zeros(shape, dtype=bool)
            roi_mask[y_lo:y_hi + 1, x_lo:x_hi + 1] = True
            rgb[~roi_mask] = (
                rgb[~roi_mask].astype(np.float32) * 0.40
            ).clip(0, 255).astype(np.uint8)
            cv2.rectangle(
                rgb, (x_lo, y_lo), (x_hi, y_hi), (0, 255, 128), 2, lineType=cv2.LINE_AA
            )
        except (TypeError, ValueError):
            pass

    if (
        particles_coal_df is None
        or not isinstance(particles_coal_df, pd.DataFrame)
        or particles_coal_df.empty
        or "id" not in particles_coal_df.columns
    ):
        return rgb

    mask_2d = None
    if isinstance(fat_masks, np.ndarray) and fat_masks.ndim >= 2:
        mask_2d = fat_masks.squeeze() if fat_masks.ndim == 3 else fat_masks
        if not np.issubdtype(mask_2d.dtype, np.integer):
            mask_2d = mask_2d.astype(np.int32)
        if mask_2d.shape[:2] != shape:
            mask_2d = None

    label_slices = None
    if mask_2d is not None and mask_2d.size > 0 and int(mask_2d.max()) > 0:
        try:
            label_slices = find_objects(mask_2d)
        except (RuntimeError, TypeError, ValueError):
            label_slices = None


    # Coordination-number colour palette (RGB)
    role_colors = {
        0: (60, 180, 255),   # Isolated Monomer (Z=0)
        1: (80, 235, 100),   # Dimer / Chain End (Z=1)
        2: (255, 210, 40),   # Chain Link (Z=2)
    }
    branch_color = (255, 65, 85)  # Network Branch Node (Z>=3)
    fill_alpha = 0.55
    centroids: dict = {}

    for _, row in particles_coal_df.iterrows():
        try:
            particle_id = int(row["id"])
        except (TypeError, ValueError):
            continue
        if particle_id <= 0:
            continue

        z_raw = row.get("coordination_number", 0)
        try:
            z_value = int(z_raw) if pd.notna(z_raw) else 0
        except (TypeError, ValueError):
            z_value = 0
        color = (
            branch_color
            if z_value >= 3
            else role_colors.get(max(z_value, 0), role_colors[0])
        )

        # Centroid from the table (used for bridge endpoints and node dots)
        cx_val = row.get("centroid_x")
        cy_val = row.get("centroid_y")
        centroid = None
        if pd.notna(cx_val) and pd.notna(cy_val):
            cx_f, cy_f = float(cx_val), float(cy_val)
            if 0 <= cx_f < shape[1] and 0 <= cy_f < shape[0]:
                centroid = (cx_f, cy_f)
                centroids[particle_id] = centroid

        if mask_2d is None:
            continue

        sl = None
        if label_slices is not None and 0 < particle_id <= len(label_slices):
            sl = label_slices[particle_id - 1]
        if sl is None:
            full_mask = mask_2d == particle_id
            if not np.any(full_mask):
                continue
            ys_all, xs_all = np.nonzero(full_mask)
            sl = (
                slice(int(ys_all.min()), int(ys_all.max()) + 1),
                slice(int(xs_all.min()), int(xs_all.max()) + 1),
            )

        local = mask_2d[sl] == particle_id
        if not np.any(local):
            continue

        y_off = int(sl[0].start or 0)
        x_off = int(sl[1].start or 0)
        if centroid is None:
            ys, xs = np.nonzero(local)
            cx_f = float(xs.mean() + x_off)
            cy_f = float(ys.mean() + y_off)
            if 0 <= cx_f < shape[1] and 0 <= cy_f < shape[0]:
                centroids[particle_id] = (cx_f, cy_f)

        # Tint the globule mask by its topological role
        region = rgb[sl]
        region[local] = (
            region[local].astype(np.float32) * (1.0 - fill_alpha)
            + np.array(color, dtype=np.float32) * fill_alpha
        ).clip(0, 255).astype(np.uint8)

        cnts, _ = cv2.findContours(
            local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if cnts:
            offset = np.array([[[x_off, y_off]]], dtype=np.int32)
            shifted = [cnt + offset for cnt in cnts]
            cv2.drawContours(rgb, shifted, -1, color, 1, lineType=cv2.LINE_AA)

    # White-blue coalescence bridges between contacting globule centroids
    if show_bridges and edges is not None and len(edges) > 0:
        for edge in edges:
            try:
                id_1, id_2 = int(edge[0]), int(edge[1])
            except (TypeError, ValueError, IndexError):
                continue
            point_1 = centroids.get(id_1)
            point_2 = centroids.get(id_2)
            if point_1 is None or point_2 is None:
                continue

            pt1 = (int(round(point_1[0])), int(round(point_1[1])))
            pt2 = (int(round(point_2[0])), int(round(point_2[1])))
            cv2.line(rgb, pt1, pt2, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(rgb, pt1, pt2, (140, 220, 255), 1, lineType=cv2.LINE_AA)
            for point in (pt1, pt2):
                cv2.circle(rgb, point, 3, (20, 35, 50), -1, lineType=cv2.LINE_AA)
                cv2.circle(rgb, point, 2, (235, 245, 255), -1, lineType=cv2.LINE_AA)

    return rgb


if __name__ == "__main__":
    print("Running visualizer.py self-tests...")

    # Case 1: Standard case with both classes populated
    df_both = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "mean_intensity": [50.0, 80.0, 110.0, 140.0, 180.0, 220.0],
        "diameter_um": [1.5, 2.0, 2.2, 3.1, 4.0, 4.5],
        "centroid_x": [15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
        "centroid_y": [15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
    })

    classified_df = classify_by_intensity(df_both, intensity_threshold=120.0)
    print("Classified DataFrame:")
    print(classified_df[["id", "mean_intensity", "diameter_um", "phase_class"]])

    # Assert correct filtering
    assert (classified_df.loc[classified_df["mean_intensity"] < 120.0, "phase_class"] == "air_bubbles").all()
    assert (classified_df.loc[classified_df["mean_intensity"] >= 120.0, "phase_class"] == "fat_particles").all()
    print("Classification threshold assertion passed!")

    fig1, _ = plot_size_distribution(classified_df)
    plt.close(fig1)
    fig1b, _ = plot_single_phase_distribution(classified_df, "fat_particles", title="Fat % Dist")
    plt.close(fig1b)
    fig1c, _ = plot_single_phase_distribution(classified_df, "air_bubbles", title="Air % Dist")
    plt.close(fig1c)
    print("Test 1 (both classes + single-phase percent plots) passed.")

    # Case 2: Only one class has objects (0 air bubbles)
    df_only_fat = df_both[df_both["mean_intensity"] >= 120.0].copy()
    fig2, _ = plot_size_distribution(df_only_fat, intensity_threshold=120.0)
    plt.close(fig2)
    print("Test 2 (0 air bubbles) passed.")

    # Case 3: Empty DataFrame
    df_empty = pd.DataFrame(columns=["id", "mean_intensity", "diameter_um"])
    fig3, _ = plot_size_distribution(df_empty, intensity_threshold=120.0)
    plt.close(fig3)
    print("Test 3 (empty DataFrame) passed.")

    # Case 4: Footer dimming and mask overlay tests
    test_img = np.full((100, 100), 200, dtype=np.uint8)
    dimmed_img = apply_footer_dimming(test_img, footer_height=20, dim_factor=0.4)
    assert dimmed_img.shape == (100, 100, 3)
    assert dimmed_img[:80, :, 0].mean() == 200
    assert dimmed_img[80:, :, 0].mean() == 80  # 200 * 0.4 = 80
    print("Test 4 (footer dimming) passed.")

    # Case 5: Translucent mask overlay with Z-order, individual opacities, and visibility toggles
    test_mask = np.zeros((100, 100), dtype=np.int32)
    test_mask[10:30, 10:30] = 1  # Air bubble
    test_mask[40:60, 40:60] = 4  # Fat particle

    overlay_img = create_mask_overlay(
        dimmed_img,
        test_mask,
        df=classified_df,
        air_opacity=0.3,
        fat_opacity=0.5,
        show_air=True,
        show_fat=True,
        z_order="Fat over Air",
    )
    assert overlay_img.shape == (100, 100, 3)

    # Particle 1 (air bubble, blue: [0, 191, 255]) with air_opacity=0.3
    # Red: 200 * 0.7 + 0 * 0.3 = 140
    # Green: 200 * 0.7 + 191 * 0.3 = 140 + 57.3 = 197
    # Blue: 200 * 0.7 + 255 * 0.3 = 140 + 76.5 = 216 / 217
    assert overlay_img[12, 12, 0] == 140
    assert overlay_img[12, 12, 2] in (216, 217)

    # Case 6: Visibility toggle test (hide air bubbles)
    hidden_air_overlay = create_mask_overlay(
        dimmed_img,
        test_mask,
        df=classified_df,
        show_air=False,
        show_fat=True,
    )
    # Air bubble area at (12, 12) should be completely unchanged from original 200
    assert hidden_air_overlay[12, 12, 0] == 200
    assert hidden_air_overlay[12, 12, 1] == 200
    assert hidden_air_overlay[12, 12, 2] == 200
    print("Test 6 (show_air=False hides fill and numbers) passed.")

    # Case 7: Test classify_by_size
    df_sizes = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "diameter_um": [2.0, 15.0, 50.0, 150.0],
    })
    size_classified = classify_by_size(
        df_sizes,
        min_air_size=10.0,
        max_air_size=100.0,
        min_fat_size=1.0,
        max_fat_size=5.0,
        overlap_priority="Air Bubbles",
        out_of_range_class="Unclassified",
    )
    assert size_classified.loc[size_classified["id"] == 1, "phase_class"].iloc[0] == "fat_particles"
    assert size_classified.loc[size_classified["id"] == 2, "phase_class"].iloc[0] == "air_bubbles"
    assert size_classified.loc[size_classified["id"] == 3, "phase_class"].iloc[0] == "air_bubbles"
    assert size_classified.loc[size_classified["id"] == 4, "phase_class"].iloc[0] == "unclassified"
    print("Test 7 (classify_by_size) passed.")

    print("All visualizer.py checks passed successfully!")
