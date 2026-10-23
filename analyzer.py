"""
Module for quantitative morphological and photometric feature extraction from SEM images.
Calculates area, equivalent diameter, and mean intensity from labeled segmentation masks.
"""

from typing import Any, Optional, Tuple, Union
import io
import math
import zipfile
import cv2
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from scipy.ndimage import find_objects, median_filter
from skimage.measure import regionprops
from skimage.segmentation import watershed
import tifffile


def extract_metrics(
    original_image: np.ndarray,
    mask: np.ndarray,
    pixel_to_um_ratio: float = 1.0,
) -> pd.DataFrame:
    """
    Extract morphological and photometric metrics for each segmented object in the mask.

    Uses skimage.measure.regionprops with intensity_image=original_image to extract:
    1) Object Area converted to square micrometers (area_um2).
    2) Mean pixel intensity on the original grayscale image (mean_intensity).
    3) Equivalent spherical/circular diameter in micrometers (diameter_um) calculated
       using the formula: sqrt(4 * area / pi).
    4) Circularity shape factor calculated as: (4 * pi * area_px) / (perimeter ** 2).
    5) Spatial centroid coordinates: centroid_x (column), centroid_y (row).

    Args:
        original_image: 2D numpy array containing the raw grayscale SEM image.
        mask: 2D integer numpy array containing labeled object masks (0=background, 1..N=objects).
        pixel_to_um_ratio: Spatial calibration factor in micrometers per pixel (um/pixel).
            Must be greater than 0. Defaults to 1.0.

    Returns:
        pd.DataFrame: DataFrame containing metrics with columns:
            - 'id': Unique integer label of each segmented object.
            - 'area_um2': Object area in square micrometers (um^2).
            - 'diameter_um': Equivalent circular diameter in micrometers (um).
            - 'mean_intensity': Mean grayscale pixel intensity of the object.
            - 'circularity': Shape factor (1.0 for perfect circle, <1.0 for irregular shapes).
            - 'centroid_x': Horizontal centroid coordinate (column index in pixels).
            - 'centroid_y': Vertical centroid coordinate (row index in pixels).
            - 'minor_axis_um': Minor axis of the fitted ellipse in micrometers (um).
            - 'solidity': Convexity ratio area / convex_hull_area (1.0 for a perfect convex object).
            - 'aspect_ratio': major_axis_um / minor_axis_um (elongation; 1.0 for round objects,
              used for coalescence and apparent contact angle analysis).

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If spatial dimensions mismatch or calibration ratio is non-positive.
    """
    if not isinstance(original_image, np.ndarray) or not isinstance(mask, np.ndarray):
        raise TypeError("Both original_image and mask must be numpy arrays.")

    if pixel_to_um_ratio <= 0:
        raise ValueError(f"pixel_to_um_ratio must be positive, got {pixel_to_um_ratio}")

    # Standardize original_image to 2D
    img_2d = original_image.squeeze() if original_image.ndim == 3 else original_image
    mask_2d = mask.squeeze() if mask.ndim == 3 else mask

    if img_2d.ndim != 2:
        raise ValueError(
            f"original_image must be a 2D grayscale image, got shape {original_image.shape}"
        )
    if mask_2d.ndim != 2:
        raise ValueError(f"mask must be a 2D labeled mask, got shape {mask.shape}")

    if img_2d.shape != mask_2d.shape:
        raise ValueError(
            f"Shape mismatch: original_image shape {img_2d.shape} != mask shape {mask_2d.shape}"
        )

    # Convert mask to integer type for regionprops
    mask_labeled = mask_2d.astype(np.int32, copy=False)

    # Use intensity_image=original_image inside regionprops
    props = regionprops(mask_labeled, intensity_image=img_2d)

    columns = [
        "id",
        "area_um2",
        "diameter_um",
        "major_axis_um",
        "minor_axis_um",
        "solidity",
        "aspect_ratio",
        "mean_intensity",
        "circularity",
        "centroid_x",
        "centroid_y",
    ]
    if not props:
        return pd.DataFrame(columns=columns)

    pixel_area_scale = pixel_to_um_ratio ** 2
    records = []

    for prop in props:
        obj_id = int(prop.label)
        # 1) Calculate area in micrometers squared
        area_px = float(prop.area)
        area_um2 = area_px * pixel_area_scale

        # 2) Calculate mean intensity of object on original image
        # Support both intensity_mean (skimage >= 0.26) and mean_intensity (earlier versions)
        if hasattr(prop, "intensity_mean") and prop.intensity_mean is not None:
            mean_intensity = float(prop.intensity_mean)
        else:
            mean_intensity = float(prop.mean_intensity)

        # 3) Calculate equivalent diameter using sqrt(4 * area / pi)
        diameter_um = math.sqrt(4.0 * area_um2 / math.pi)

        # 4) Calculate perimeter and centroid coordinates (row, col)
        perimeter = float(prop.perimeter)
        row, col = prop.centroid
        centroid_y = float(row)
        centroid_x = float(col)

        # 5) Calculate circularity: (4 * pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0
        circularity = (
            (4.0 * math.pi * area_px) / (perimeter ** 2)
            if perimeter > 0.0
            else 0.0
        )

        # 6) Major axis of the fitted ellipse in micrometers (chord length for partial globules)
        major_axis_um = float(prop.major_axis_length) * pixel_to_um_ratio

        # 7) Minor axis + convexity/elongation descriptors used for coalescence
        #    and apparent contact angle analysis.
        minor_axis_um = float(getattr(prop, "minor_axis_length", 0.0)) * pixel_to_um_ratio
        solidity = float(getattr(prop, "solidity", 1.0))
        aspect_ratio = (
            float(major_axis_um / max(minor_axis_um, 1e-6))
            if minor_axis_um > 0
            else 1.0
        )

        records.append({
            "id": obj_id,
            "area_um2": area_um2,
            "diameter_um": diameter_um,
            "major_axis_um": major_axis_um,
            "minor_axis_um": minor_axis_um,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "mean_intensity": mean_intensity,
            "circularity": circularity,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
        })

    df = pd.DataFrame(records, columns=columns)
    return df


def _compute_granulometry_dict(diameters: np.ndarray) -> dict:
    """Helper to compute D[3,2], D[4,3], D10, D50, D90, Span on 1D diameter array."""
    d = diameters[np.isfinite(diameters) & (diameters > 0)].astype(np.float64)
    n = len(d)
    if n == 0:
        return {
            "Count": 0,
            "D[3,2] (µm)": 0.0,
            "D[4,3] (µm)": 0.0,
            "D10 (µm)": 0.0,
            "D50 (µm)": 0.0,
            "D90 (µm)": 0.0,
            "Span": 0.0,
        }
    sum_d2 = np.sum(d ** 2)
    sum_d3 = np.sum(d ** 3)
    sum_d4 = np.sum(d ** 4)

    d32 = float(sum_d3 / sum_d2) if sum_d2 > 0 else 0.0
    d43 = float(sum_d4 / sum_d3) if sum_d3 > 0 else 0.0
    d10 = float(np.percentile(d, 10))
    d50 = float(np.percentile(d, 50))
    d90 = float(np.percentile(d, 90))
    span = float((d90 - d10) / d50) if d50 > 0 else 0.0

    return {
        "Count": int(n),
        "D[3,2] (µm)": round(d32, 2),
        "D[4,3] (µm)": round(d43, 2),
        "D10 (µm)": round(d10, 2),
        "D50 (µm)": round(d50, 2),
        "D90 (µm)": round(d90, 2),
        "Span": round(span, 2),
    }


def calculate_granulometry_metrics(
    df: pd.DataFrame,
    diameter_col: str = "diameter_um",
    phase_col: str = "phase_class",
) -> pd.DataFrame:
    """
    Calculate advanced Food Science granulometry metrics for segmented particle phases:
    1) Sauter mean surface diameter D[3,2] = sum(d^3) / sum(d^2)
    2) De Brouckere volume mean diameter D[4,3] = sum(d^4) / sum(d^3)
    3) Percentile diameters: D10 (10%), D50 (median, 50%), D90 (90%)
    4) Relative span = (D90 - D10) / D50

    Args:
        df: DataFrame containing particle measurements with diameter and phase class.
        diameter_col: Column name for equivalent diameter (um). Defaults to 'diameter_um'.
        phase_col: Column name for phase classification. Defaults to 'phase_class'.

    Returns:
        pd.DataFrame: Table with granulometry statistics per phase.
    """
    cols = ["Phase", "Count", "D[3,2] (µm)", "D[4,3] (µm)", "D10 (µm)", "D50 (µm)", "D90 (µm)", "Span"]
    if df.empty or diameter_col not in df.columns:
        return pd.DataFrame(columns=cols)

    records = []
    # adsorbed_fat is explicitly excluded from PSD statistics (Sauter/De Brouckere/histogram)
    # because partially-visible globules on bubble walls underestimate the true size.
    # They appear as a separate informational row only.
    canonical_phases = [
        ("fat_particles", "Fat Particles"),
        ("air_bubbles", "Air Bubbles"),
        ("unclassified", "Unclassified"),
    ]

    for p_key, p_name in canonical_phases:
        if phase_col in df.columns:
            mask_phase = df[phase_col].astype(str).str.lower() == p_key.lower()
            # Explicit scientific safety: fat_particles PSD strictly excludes any adsorbed globules
            if p_key.lower() == "fat_particles":
                mask_phase = mask_phase & (~df[phase_col].astype(str).str.lower().isin(["adsorbed_fat"]))
                if "is_partial_in_bubble" in df.columns:
                    mask_phase = mask_phase & (~df["is_partial_in_bubble"].fillna(False).astype(bool))
            subset = df[mask_phase][diameter_col].values
            if len(subset) > 0:
                stats = _compute_granulometry_dict(subset)
                stats["Phase"] = p_name
                records.append(stats)

    # Informational row for adsorbed fat globules (excluded from PSD)
    if phase_col in df.columns:
        partial_mask = df[phase_col].astype(str).str.lower().isin(["adsorbed_fat"])
        partial_count = int(partial_mask.sum())
        if partial_count > 0:
            partial_row = {
                "Phase": "Adsorbed Fat (in Bubbles, Excluded from PSD)",
                "Count": partial_count,
                "D[3,2] (µm)": float("nan"),
                "D[4,3] (µm)": float("nan"),
                "D10 (µm)": float("nan"),
                "D50 (µm)": float("nan"),
                "D90 (µm)": float("nan"),
                "Span": float("nan"),
            }
            records.append(partial_row)

    if phase_col in df.columns:
        known_keys = {k.lower() for k, _ in canonical_phases} | {"adsorbed_fat"}
        other_phases = [p for p in df[phase_col].dropna().unique() if str(p).lower() not in known_keys]
        for p in other_phases:
            subset = df[df[phase_col] == p][diameter_col].values
            if len(subset) > 0:
                stats = _compute_granulometry_dict(subset)
                stats["Phase"] = str(p).replace("_", " ").title()
                records.append(stats)

    if not records:
        return pd.DataFrame(columns=cols)

    res_df = pd.DataFrame(records)[cols]
    return res_df


def _clean_contour_vertices(pts: np.ndarray) -> np.ndarray:
    """
    Remove duplicated or degenerate consecutive vertices and trailing closed loops.
    """
    pts_arr = np.asarray(pts, dtype=np.float32)
    if len(pts_arr) < 3:
        return pts_arr
    cleaned = [pts_arr[0]]
    for p in pts_arr[1:]:
        if np.linalg.norm(p - cleaned[-1]) > 1e-2:
            cleaned.append(p)
    res = np.array(cleaned, dtype=np.float32)
    # If the last point duplicates the first point, drop it for periodic spline
    if len(res) > 2 and np.linalg.norm(res[0] - res[-1]) < 1e-2:
        res = res[:-1]
    return res


def _smooth_contour_fourier(
    pts: np.ndarray,
    n_harmonics: int = 7,
    num_points: int = 140,
) -> np.ndarray:
    """
    Fourier contour regularizer for closed biological and physical interfaces (air bubbles / pores).
    Resamples closed contour along cumulative arc length and truncates high-frequency spatial harmonics
    above n_harmonics. Guarantees 100% smooth, natural, low-order closed curves without pixel steps or
    cubic spline ringing artifacts.

    Args:
        pts: (N, 2) array of contour coordinates.
        n_harmonics: Number of low-frequency Fourier harmonics to retain (default: 7).
        num_points: Number of uniformly-spaced output vertices along arc length (default: 140).

    Returns:
        np.ndarray: (num_points, 2) float32 coordinates of the regularized contour.
    """
    raw = np.asarray(pts, dtype=np.float32)
    if len(raw) < 3:
        return raw

    # Ensure closed loop for cumulative arc length calculation
    if np.linalg.norm(raw[0] - raw[-1]) > 1e-4:
        closed = np.vstack([raw, raw[0]])
    else:
        closed = raw.copy()

    diffs = np.diff(closed, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dist = np.concatenate([[0.0], np.cumsum(dists)])
    total_len = float(cum_dist[-1])
    if total_len < 1e-4:
        return raw

    # Uniform arc-length resampling
    u = np.linspace(0.0, total_len, num_points, endpoint=False)
    x_res = np.interp(u, cum_dist, closed[:, 0])
    y_res = np.interp(u, cum_dist, closed[:, 1])

    # Complex representation & Discrete Fourier Transform
    z = x_res + 1j * y_res
    coeffs = np.fft.fft(z)

    # Low-pass filter in Fourier domain: zero out frequencies above n_harmonics
    N = len(z)
    k = max(1, min(int(n_harmonics), N // 2 - 1))
    filtered = np.zeros_like(coeffs)
    filtered[:k + 1] = coeffs[:k + 1]
    if k > 0:
        filtered[-k:] = coeffs[-k:]

    # Inverse Fourier Transform to recover smooth spatial coordinates
    z_smooth = np.fft.ifft(filtered)
    return np.column_stack([z_smooth.real, z_smooth.imag]).astype(np.float32)


def _generate_periodic_bspline(pts: np.ndarray, num_points: int = 160) -> np.ndarray:
    """
    Generate a closed smooth periodic spline through user control points.
    Uses scipy.interpolate.CubicSpline with bc_type='periodic' and chord-length
    parameterization for reliable, guaranteed-smooth closed curves.
    """
    cleaned = _clean_contour_vertices(pts)
    if len(cleaned) < 3:
        return np.asarray(pts, dtype=np.float32)

    # Close the contour: last point EXACTLY equals the first for bc_type='periodic'
    closed_pts = np.vstack([cleaned, cleaned[0]])

    # Remove consecutive near-duplicate points (chord < 1e-3 px) that break CubicSpline
    diffs = np.diff(closed_pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    keep = np.concatenate([[True], seg_lens > 1e-3])
    closed_pts = closed_pts[keep]
    if len(closed_pts) < 4:
        return cleaned.astype(np.float32)

    # Ensure the last point is exactly the first for strict periodicity
    closed_pts[-1] = closed_pts[0]

    # Chord-length parameterization: t in [0, 1]
    t = np.zeros(len(closed_pts), dtype=np.float64)
    t[1:] = np.cumsum(np.hypot(np.diff(closed_pts[:, 0]), np.diff(closed_pts[:, 1])))
    if t[-1] <= 0:
        return cleaned.astype(np.float32)
    t /= t[-1]

    try:
        cs_x = interp.CubicSpline(t, closed_pts[:, 0], bc_type="periodic")
        cs_y = interp.CubicSpline(t, closed_pts[:, 1], bc_type="periodic")
        u_dense = np.linspace(0.0, 1.0, num_points, endpoint=False)
        return np.column_stack([cs_x(u_dense), cs_y(u_dense)]).astype(np.float32)
    except Exception:
        return cleaned.astype(np.float32)


def _fit_smooth_ellipse_contour(pts: np.ndarray, num_points: int = 120) -> np.ndarray:
    """
    Fit a smooth parametric ellipse using cv2.fitEllipse and generate num_points coordinates.
    """
    cleaned = _clean_contour_vertices(pts)
    if len(cleaned) < 3:
        return pts.astype(np.float32)

    # When exactly 3 points are provided, analytically construct the exact circumcircle
    if len(cleaned) == 3:
        p1, p2, p3 = cleaned[0], cleaned[1], cleaned[2]
        d = 2.0 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        if abs(d) > 1e-4:
            ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / d
            uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / d
            r = float(np.hypot(p1[0] - ux, p1[1] - uy))
            if 2.0 <= r <= 2000.0:
                t = np.linspace(0.0, 2.0 * math.pi, num_points, endpoint=False)
                return np.column_stack([ux + r * np.cos(t), uy + r * np.sin(t)]).astype(np.float32)

    # cv2.fitEllipse requires at least 5 points
    if len(cleaned) < 5:
        dense = _generate_periodic_bspline(cleaned, num_points=60)
    else:
        dense = cleaned.astype(np.float32)

    try:
        ellipse = cv2.fitEllipse(dense)
        (cx, cy), (d1, d2), angle_deg = ellipse
        a = float(d1) / 2.0
        b = float(d2) / 2.0
        if a <= 0 or b <= 0 or not math.isfinite(a) or not math.isfinite(b):
            return _generate_periodic_bspline(cleaned, num_points=num_points)

        theta = np.radians(angle_deg)
        t = np.linspace(0, 2.0 * math.pi, num_points, endpoint=False)
        x_rot = a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
        y_rot = a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
        return np.column_stack([cx + x_rot, cy + y_rot]).astype(np.float32)
    except Exception:
        return _generate_periodic_bspline(cleaned, num_points=num_points)


def auto_grow_dark_bubble_contour(
    seed_xy: Tuple[float, float],
    image_2d: np.ndarray,
    r_max_px: float = 120.0,
    r_min_px: float = 8.0,
    num_rays: int = 36,
    num_spline_points: int = 140,
    cellpose_model: Optional[Any] = None,
    intensity_threshold_offset: float = 18.0,
) -> Optional[np.ndarray]:
    """
    Grow a bubble / pore / fat-globule contour from a single clicked centre point.

    Step B (priority) — Multi-scale Cellpose AI on a WIDE window:
        The click is no longer assumed to be dark. A wide ROI around it is segmented on
        both direct and inverted CLAHE with diameters [28, 55, 100, 160] px
        (cellprob_threshold=-1.8, flow_threshold=0.8). As soon as Cellpose finds the
        object under the click, its Fourier-regularised contour is returned immediately.
    Step A (fallback) — Radial |dI/dr| search:
        Used only when Cellpose found nothing. Each ray takes the maximum ABSOLUTE
        brightness derivative, so dark crater walls and bright fat globules are treated
        equally (the old brightness-rise test stopped rays after ~8 px on bright fat).
        Radii are outlier-filtered, median-filtered (size=5, wrap), ellipse-fitted and
        Fourier-smoothed.

    Args:
        seed_xy: (sx, sy) clicked centre coordinates in image pixels.
        image_2d: 2D grayscale SEM micrograph array.
        r_max_px: Maximum radial search radius in pixels (also scales the AI window).
        r_min_px: Minimum radial search radius in pixels (defaults to 8.0).
        num_rays: Number of angular rays for the radial fallback (defaults to 36).
        num_spline_points: Number of output contour vertices (defaults to 140).
        cellpose_model: Optional initialized CellposeModel (loaded lazily when None).
        intensity_threshold_offset: Deprecated compatibility parameter (unused).

    Returns:
        np.ndarray: (num_spline_points, 2) float32 contour coordinates,
                    or None if detection fails.
    """
    sx, sy = float(seed_xy[0]), float(seed_xy[1])
    img_2d = image_2d.squeeze() if image_2d.ndim == 3 else image_2d
    if img_2d.ndim != 2:
        return None
    h, w = img_2d.shape[:2]
    if not (0 <= sx < w and 0 <= sy < h):
        return None

    r_max_eff = float(max(r_min_px + 8.0, min(r_max_px, min(h, w) * 0.48)))
    ray_angles = np.linspace(0.0, 2.0 * math.pi, num_rays, endpoint=False)

    # ── Step B (priority): Multi-scale Cellpose AI on a wide window ──────────
    model = cellpose_model
    if model is None:
        try:
            from cellpose_model import get_cellpose_model

            model = get_cellpose_model()
        except Exception:
            model = None

    if model is not None:
        try:
            from cellpose_model import apply_clahe, run_inference

            clahe_img = apply_clahe(img_2d)
            win = int(max(64.0, min(r_max_eff, 0.5 * min(h, w))))
            x1 = max(0, int(round(sx)) - win)
            y1 = max(0, int(round(sy)) - win)
            x2 = min(w, int(round(sx)) + win)
            y2 = min(h, int(round(sy)) + win)
            roi_clahe = clahe_img[y1:y2, x1:x2]
            roi_inv = np.clip(255 - roi_clahe, 0, 255).astype(np.uint8)
            cx_roi = int(np.clip(round(sx) - x1, 0, roi_clahe.shape[1] - 1))
            cy_roi = int(np.clip(round(sy) - y1, 0, roi_clahe.shape[0] - 1))
            roi_area = float(roi_clahe.size)

            if roi_clahe.shape[0] >= 16 and roi_clahe.shape[1] >= 16:
                for d_run in (28.0, 55.0, 100.0, 160.0):
                    contour = None
                    for test_img in (roi_clahe, roi_inv):
                        try:
                            m_res = run_inference(
                                model=model,
                                image=test_img,
                                diameter=float(d_run),
                                flow_threshold=0.8,
                                cellprob_threshold=-1.8,
                                enable_clahe=False,
                            )
                        except Exception:
                            m_res = None
                        if m_res is None or int(np.max(m_res)) <= 0:
                            continue

                        # Object under the click (or dominant neighbour within 6 px)
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

                        cand_mask = (m_res == lbl).astype(np.uint8)
                        cnts, _ = cv2.findContours(
                            cand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                        )
                        if not cnts:
                            continue
                        best_cnt = max(cnts, key=cv2.contourArea)
                        area = float(cv2.contourArea(best_cnt))
                        if area < 25.0 or area > 0.9 * roi_area:
                            continue

                        pts = best_cnt.reshape(-1, 2).astype(np.float32)
                        pts[:, 0] += float(x1)
                        pts[:, 1] += float(y1)

                        perimeter = float(cv2.arcLength(best_cnt, True))
                        circ = (
                            4.0 * math.pi * area / (perimeter ** 2)
                            if perimeter > 0
                            else 1.0
                        )
                        n_harm = 3 if circ >= 0.72 else 5
                        contour = _smooth_contour_fourier(
                            pts, n_harmonics=n_harm, num_points=num_spline_points
                        )
                        break

                    if contour is not None and len(contour) >= 3:
                        return contour
        except Exception:
            pass

    # ── Step A (fallback): radial search on the maximum |dI/dr| ───────────────
    blurred = cv2.GaussianBlur(img_2d.astype(np.float32), (15, 15), 0)
    radii = []
    for angle in ray_angles:
        cos_a = float(math.cos(angle))
        sin_a = float(math.sin(angle))
        r_steps = np.arange(float(r_min_px), r_max_eff + 1.0, 1.0)
        ray_x = sx + r_steps * cos_a
        ray_y = sy + r_steps * sin_a

        valid = (ray_x >= 0) & (ray_x < w) & (ray_y >= 0) & (ray_y < h)
        if not np.any(valid):
            radii.append(float(r_min_px))
            continue

        r_valid = r_steps[valid]
        xs = np.clip(np.round(ray_x[valid]).astype(int), 0, w - 1)
        ys = np.clip(np.round(ray_y[valid]).astype(int), 0, h - 1)
        dI_dr = np.gradient(blurred[ys, xs].astype(np.float32))

        # |dI/dr| peak = wall of a dark crater OR the edge of a bright fat globule
        radii.append(float(r_valid[int(np.argmax(np.abs(dI_dr)))]))

    if not radii:
        return None

    med_r = float(np.median(radii))
    if med_r <= 1.0:
        return None

    # Outlier rays -> median radius, then circular median filter around the loop
    clean_radii = [r if abs(r - med_r) <= 0.35 * med_r else med_r for r in radii]
    clean_arr = median_filter(
        np.array(clean_radii, dtype=np.float32), size=5, mode="wrap"
    )

    rim_pts = np.column_stack(
        [
            sx + clean_arr * np.cos(ray_angles),
            sy + clean_arr * np.sin(ray_angles),
        ]
    ).astype(np.float32)

    # Ellipse fit + low-order Fourier smoothing guarantees a natural oval outline
    ell_pts = _fit_smooth_ellipse_contour(rim_pts, num_points=num_spline_points)
    return _smooth_contour_fourier(ell_pts, n_harmonics=3, num_points=num_spline_points)


def snap_contour_to_edges(
    points: np.ndarray,
    original_image: np.ndarray,
    expand_px: int = 18,
    method: str = "🧲 Smart Spline & Edge Snap (Irregular Shapes)",
    search_dist: int = 14,
) -> np.ndarray:
    """
    Refine sketched contour points to true SEM bubble/pore boundaries.

    Supports three modes:
    1) '🧲 Smart Spline & Edge Snap (Irregular Shapes)' (Default):
       - Smooth closed B-spline (N=160 points).
       - Macro-gradient map via cv2.GaussianBlur(img_2d, (15, 15), 0) + Sobel magnitude.
       - Snapping along outward normals within search_dist (14 px) with penalty score = grad_val - 0.6 * |offset|.
    2) '⭕ Fit Smooth Ellipse / Circle':
       - Fits an exact smooth parametric ellipse via cv2.fitEllipse.
    3) '📐 Smooth Spline Only (Follow My Points)':
       - Periodic cubic B-spline through user control points without image gradient snapping.

    Args:
        points: (N, 2) float array of contour coordinates.
        original_image: 2D grayscale SEM micrograph.
        expand_px: Fallback boundary search margin in pixels (defaults to 18).
        method: Refinement mode name.
        search_dist: Normal search radius in pixels (defaults to 14).

    Returns:
        np.ndarray: Refined (M, 2) float32 contour coordinates.
    """
    pts_raw = np.asarray(points, dtype=np.float32)
    if len(pts_raw) < 3:
        return pts_raw

    img_2d = original_image.squeeze() if original_image.ndim == 3 else original_image
    if img_2d.ndim != 2:
        return pts_raw
    h, w = img_2d.shape[:2]

    mode_str = str(method or "").strip()

    # 1. Fit Smooth Ellipse / Circle
    if "ellipse" in mode_str.lower() or "circle" in mode_str.lower() or "fit smooth" in mode_str.lower():
        fitted_ell = _fit_smooth_ellipse_contour(pts_raw, num_points=120)
        fitted_ell[:, 0] = np.clip(fitted_ell[:, 0], 0, w - 1)
        fitted_ell[:, 1] = np.clip(fitted_ell[:, 1], 0, h - 1)
        return fitted_ell

    # 2. Smooth Spline Only (Follow My Points)
    if (
        "smooth spline only" in mode_str.lower()
        or "follow my points" in mode_str.lower()
        or mode_str.lower() == "spline_only"
    ):
        spline_only = _generate_periodic_bspline(pts_raw, num_points=120)
        spline_only[:, 0] = np.clip(spline_only[:, 0], 0, w - 1)
        spline_only[:, 1] = np.clip(spline_only[:, 1], 0, h - 1)
        return spline_only

    # If method explicitly requests no snapping / None
    if mode_str.lower() in ("none", "false", ""):
        return pts_raw

    # 3. Smart Spline & Edge Snap (Irregular Shapes) [Default]
    # Step A: Compute macro-gradient map on blurred image
    img_f = img_2d.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (15, 15), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobel_x, sobel_y)
    max_grad = float(np.max(grad_mag))
    norm_grad = (grad_mag / (max_grad + 1e-6)) * 100.0

    # Step B: Select control points for snapping.
    # If the input already has few points (user-clicked polygon 3..24 pts),
    # use them directly as control nodes. For dense raster contours (> 24 pts),
    # thin down to K = 14 evenly-spaced control nodes along the perimeter.
    num_raw = len(pts_raw)
    K_TARGET = 14
    if num_raw <= 24:
        control_pts = pts_raw.copy()
    else:
        indices = np.round(np.linspace(0, num_raw - 1, K_TARGET)).astype(int)
        control_pts = pts_raw[indices]

    # Step C: Normal search radius — 14 px as instructed
    eff_search_dist = int(search_dist)

    # Step D: Snap each control point along its outward normal to the
    # nearest gradient peak, with distance penalty score = grad - 0.6 * |offset|
    K = len(control_pts)
    snapped_control_pts = np.zeros_like(control_pts)
    for i in range(K):
        prev_p = control_pts[(i - 1) % K]
        next_p = control_pts[(i + 1) % K]
        tangent = next_p - prev_p
        norm_t = float(np.linalg.norm(tangent))
        if norm_t < 1e-6:
            normal = np.array([0.0, 1.0], dtype=np.float32)
        else:
            normal = np.array([-tangent[1] / norm_t, tangent[0] / norm_t], dtype=np.float32)

        cur_p = control_pts[i]
        best_score = -1e9
        best_offset = 0

        for offset in range(-eff_search_dist, eff_search_dist + 1):
            px = cur_p[0] + offset * normal[0]
            py = cur_p[1] + offset * normal[1]
            ix = int(round(float(np.clip(px, 0, w - 1))))
            iy = int(round(float(np.clip(py, 0, h - 1))))
            grad_val = norm_grad[iy, ix]
            score = grad_val - 0.6 * abs(float(offset))
            if score > best_score:
                best_score = score
                best_offset = offset

        snapped_control_pts[i] = cur_p + best_offset * normal

    # Step E: AFTER snapping control points, generate the smooth closed spline.
    # This guarantees 100% smooth, rounded output — no jagged edges possible.
    result = _generate_periodic_bspline(snapped_control_pts, num_points=160)

    # Clip to micrograph bounds
    result[:, 0] = np.clip(result[:, 0], 0, w - 1)
    result[:, 1] = np.clip(result[:, 1], 0, h - 1)

    return result



def build_combined_training_mask(
    auto_masks: np.ndarray,
    manual_polygons: list,
    start_manual_id: Optional[int] = None,
) -> np.ndarray:
    """
    Merge Cellpose auto masks with manual polygon fills into one labeled mask
    suitable for Cellpose human-in-the-loop training (0=bg, 1..N=objects).
    """
    base = np.asarray(auto_masks).astype(np.int32, copy=True)
    if base.ndim != 2:
        base = base.squeeze()
    combined = base.copy()
    next_id = int(combined.max()) + 1 if combined.size else 1
    if start_manual_id is not None:
        next_id = max(next_id, int(start_manual_id))

    for mp in manual_polygons or []:
        pts = np.asarray(mp.get("points"), dtype=np.float32)
        if len(pts) < 3:
            continue
        pts_int = np.round(pts).astype(np.int32)
        cv2.fillPoly(combined, [pts_int], next_id)
        next_id += 1

    return combined


def calculate_manual_object_metrics(
    points: np.ndarray,
    original_image: np.ndarray,
    pixel_to_um_ratio: float = 1.0,
    obj_id: int = 10001,
) -> dict:
    """
    Calculate morphological and photometric metrics for a manually drawn contour.

    Args:
        points: (N, 2) array of coordinates [x, y] in original image pixel units.
        original_image: 2D numpy array containing the raw grayscale SEM image.
        pixel_to_um_ratio: Spatial resolution in micrometers per pixel (um/pixel).
        obj_id: Unique integer identifier for the manual object (>= 10000).

    Returns:
        dict: Dictionary containing metrics with keys:
            - 'id': Integer identifier.
            - 'area_um2': Area in square micrometers (um^2).
            - 'diameter_um': Equivalent circular diameter in micrometers (um).
            - 'mean_intensity': Mean pixel intensity inside the drawn shape.
            - 'circularity': Shape factor (4 * pi * area) / (perimeter^2).
            - 'centroid_x': Horizontal centroid coordinate (pixels).
            - 'centroid_y': Vertical centroid coordinate (pixels).
            - 'minor_axis_um': Minor axis of the fitted ellipse / minAreaRect in micrometers.
            - 'solidity': area / convex_hull_area (1.0 for a perfect convex shape).
            - 'aspect_ratio': major_axis_um / minor_axis_um (1.0 for round objects).
    """
    pts_arr = np.asarray(points, dtype=np.float32)
    if len(pts_arr) < 3:
        raise ValueError("Manual contour must have at least 3 points.")

    area_px = float(cv2.contourArea(pts_arr))
    pixel_area_scale = float(pixel_to_um_ratio) ** 2
    area_um2 = area_px * pixel_area_scale
    diameter_um = math.sqrt(4.0 * area_um2 / math.pi) if area_um2 > 0 else 0.0

    # Calculate centroid coordinates
    M = cv2.moments(pts_arr)
    if M["m00"] != 0:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        cx = float(np.mean(pts_arr[:, 0]))
        cy = float(np.mean(pts_arr[:, 1]))

    # Calculate circularity
    perimeter = float(cv2.arcLength(pts_arr, closed=True))
    circularity = (
        (4.0 * math.pi * area_px) / (perimeter ** 2)
        if perimeter > 0.0
        else 1.0
    )

    # Compute mean intensity on original_image
    img_2d = original_image.squeeze() if original_image.ndim == 3 else original_image
    h, w = img_2d.shape[:2]
    mask_poly = np.zeros((h, w), dtype=np.uint8)
    pts_int = np.round(pts_arr).astype(np.int32)
    cv2.fillPoly(mask_poly, [pts_int], 1)

    if np.any(mask_poly):
        mean_intensity = float(img_2d[mask_poly == 1].mean())
    else:
        mean_intensity = float(img_2d.mean())

    # Calculate major/minor axes of fitted ellipse (chord length for contact angle estimation)
    if len(pts_arr) >= 5:
        try:
            ellipse = cv2.fitEllipse(pts_arr)
            major_axis_px = max(float(ellipse[1][0]), float(ellipse[1][1]))
            minor_axis_px = min(float(ellipse[1][0]), float(ellipse[1][1]))
        except Exception:
            rect = cv2.minAreaRect(pts_arr)
            major_axis_px = max(float(rect[1][0]), float(rect[1][1]))
            minor_axis_px = min(float(rect[1][0]), float(rect[1][1]))
    else:
        rect = cv2.minAreaRect(pts_arr)
        major_axis_px = max(float(rect[1][0]), float(rect[1][1]))
        minor_axis_px = min(float(rect[1][0]), float(rect[1][1]))
    major_axis_um = float(major_axis_px) * float(pixel_to_um_ratio)
    minor_axis_um = float(minor_axis_px) * float(pixel_to_um_ratio)

    # Convexity (solidity): contour area / convex hull area (mirrors regionprops.solidity)
    try:
        hull_area_px = float(cv2.contourArea(cv2.convexHull(pts_arr)))
    except Exception:
        hull_area_px = 0.0
    solidity = float(area_px / max(hull_area_px, 1.0))

    aspect_ratio = (
        float(major_axis_um / max(minor_axis_um, 1e-6))
        if minor_axis_um > 0
        else 1.0
    )

    return {
        "id": obj_id,
        "area_um2": area_um2,
        "diameter_um": diameter_um,
        "major_axis_um": major_axis_um,
        "minor_axis_um": minor_axis_um,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "mean_intensity": mean_intensity,
        "circularity": circularity,
        "centroid_x": cx,
        "centroid_y": cy,
    }


def parse_canvas_manual_objects_from_raster(
    image_data: np.ndarray,
    original_image: np.ndarray,
    pixel_to_um_ratio: float,
    start_id: int = 10001,
    min_area_px: float = 25.0,
    auto_snap_mode: Optional[Union[str, bool]] = "🧲 Smart Spline & Edge Snap (Irregular Shapes)",
    invert_brightfield: bool = False,
    **kwargs,
) -> Tuple[list[dict], list[dict]]:
    """
    Extract manually drawn objects from a canvas RGBA raster (fallback for freedraw).

    Uses the alpha channel or drawn strokes, finds external contours, and computes
    morphological metrics scaled to the original analysis micrograph.

    Args:
        image_data: Canvas RGBA (or RGB) array from canvas_result.image_data.
        original_image: 2D analysis micrograph (target coordinate space).
        pixel_to_um_ratio: Micrometers per pixel calibration factor.
        start_id: Starting integer ID for manual objects (defaults to 10001).
        min_area_px: Ignore tiny contour noise below this area (canvas pixels).
        auto_snap_mode: Selected refinement mode string or bool.
        invert_brightfield: Brightfield-light-mode flag (accepted for API parity with
            ``parse_canvas_manual_objects``; the raster fallback performs no AI inference).

    Returns:
        Tuple[list[dict], list[dict]]: (metrics_list, polygons_list)
    """
    if image_data is None or not isinstance(image_data, np.ndarray) or image_data.size == 0:
        return [], []

    # Backward compatibility with auto_snap boolean argument
    if "auto_snap" in kwargs and kwargs["auto_snap"] is False:
        auto_snap_mode = "none"

    if auto_snap_mode is True:
        snap_mode = "🧲 Smart Spline & Edge Snap (Irregular Shapes)"
    elif auto_snap_mode is False or auto_snap_mode is None or str(auto_snap_mode).lower() == "none":
        snap_mode = "none"
    else:
        snap_mode = str(auto_snap_mode)

    img_2d = original_image.squeeze() if original_image.ndim == 3 else original_image
    if img_2d.ndim != 2:
        return [], []
    target_h, target_w = img_2d.shape[:2]

    canvas = image_data
    if canvas.ndim == 3 and canvas.shape[2] >= 4:
        draw_mask = (canvas[:, :, 3] > 0).astype(np.uint8) * 255
    elif canvas.ndim == 3:
        draw_mask = (np.any(canvas[:, :, :3] > 8, axis=2)).astype(np.uint8) * 255
    else:
        draw_mask = (canvas > 0).astype(np.uint8) * 255

    if not np.any(draw_mask):
        return [], []

    # Close thin freedraw strokes into solid filled regions before contour extraction
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    draw_mask = cv2.morphologyEx(draw_mask, cv2.MORPH_CLOSE, close_k)

    contours, _ = cv2.findContours(draw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], []

    canvas_h, canvas_w = draw_mask.shape[:2]
    sx = float(target_w) / float(max(canvas_w, 1))
    sy = float(target_h) / float(max(canvas_h, 1))

    metrics_list: list[dict] = []
    polygons_list: list[dict] = []
    curr_id = start_id

    for contour in contours:
        area_canvas = float(cv2.contourArea(contour))
        if area_canvas < float(min_area_px):
            continue

        # Detect thin unclosed strokes: if area / convexHull_area < 0.25, the contour
        # is an open line rather than a closed polygon.  Re-order points angularly around
        # the centroid (atan2) so they form a proper convex hull and represent the bubble.
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        is_thin_stroke = hull_area > 0 and (area_canvas / hull_area) < 0.25
        if is_thin_stroke:
            # Reorder raw contour points by angle around centroid → closed polygon
            pts_all = contour.reshape(-1, 2).astype(np.float32)
            cx_c = float(pts_all[:, 0].mean())
            cy_c = float(pts_all[:, 1].mean())
            angles = np.arctan2(pts_all[:, 1] - cy_c, pts_all[:, 0] - cx_c)
            order = np.argsort(angles)
            pts_ordered = pts_all[order].astype(np.int32)
            # Re-render the reordered polygon as a filled region and find its contour
            thin_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            cv2.fillPoly(thin_canvas, [pts_ordered], 255)
            new_contours, _ = cv2.findContours(
                thin_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if new_contours:
                contour = max(new_contours, key=cv2.contourArea)
                area_canvas = float(cv2.contourArea(contour))
                if area_canvas < float(min_area_px):
                    continue

        filled_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillPoly(filled_canvas, [contour], 1)
        filled_full = cv2.resize(
            filled_canvas, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
        area_px = float(np.count_nonzero(filled_full))
        if area_px < 3:
            continue

        pts_canvas = contour.reshape(-1, 2).astype(np.float32)
        pts_arr = np.column_stack([
            pts_canvas[:, 0] * sx,
            pts_canvas[:, 1] * sy,
        ]).astype(np.float32)

        if len(pts_arr) < 3:
            continue

        # Boundary refinement using selected mode
        if snap_mode.lower() != "none":
            try:
                # Downsample dense pixel perimeter to control vertices for robust spline fitting
                epsilon = max(1.2, 0.007 * cv2.arcLength(contour, True))
                approx_c = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)
                pts_to_snap = np.column_stack([approx_c[:, 0] * sx, approx_c[:, 1] * sy]).astype(np.float32)
                if len(pts_to_snap) >= 3:
                    pts_arr = snap_contour_to_edges(
                        points=pts_to_snap,
                        original_image=img_2d,
                        method=snap_mode,
                    )
                else:
                    pts_arr = snap_contour_to_edges(
                        points=pts_arr,
                        original_image=img_2d,
                        method=snap_mode,
                    )
            except Exception:
                pass

        try:
            metrics = calculate_manual_object_metrics(
                points=pts_arr,
                original_image=img_2d,
                pixel_to_um_ratio=pixel_to_um_ratio,
                obj_id=curr_id,
            )
            # Re-rasterize refined contour to get exact pixel-based metrics
            refined_mask = np.zeros((target_h, target_w), dtype=np.uint8)
            pts_int = np.round(pts_arr).astype(np.int32)
            cv2.fillPoly(refined_mask, [pts_int], 1)
            ref_area_px = float(np.count_nonzero(refined_mask))
            if ref_area_px > 0:
                pixel_area_scale = float(pixel_to_um_ratio) ** 2
                metrics["area_um2"] = ref_area_px * pixel_area_scale
                metrics["diameter_um"] = (
                    math.sqrt(4.0 * metrics["area_um2"] / math.pi)
                    if metrics["area_um2"] > 0
                    else 0.0
                )
                ys, xs = np.where(refined_mask > 0)
                if len(xs) > 0:
                    metrics["centroid_x"] = float(xs.mean())
                    metrics["centroid_y"] = float(ys.mean())
                    metrics["mean_intensity"] = float(img_2d[refined_mask > 0].mean())

            metrics_list.append(metrics)
            polygons_list.append({
                "id": curr_id,
                "points": pts_arr,
                "shape_type": "raster",
            })
            curr_id += 1
        except Exception:
            continue

    return metrics_list, polygons_list


def ai_segment_bubble_from_points(
    points_xy: np.ndarray,
    image_2d: np.ndarray,
    cellpose_model: Optional[Any] = None,
    num_spline_points: int = 140,
    invert_brightfield: bool = False,
) -> Optional[np.ndarray]:
    """
    Locally segment an air bubble/pore from user points using Local Normal Snapping,
    adaptive geometry (perfect ellipse for 2-5 points, faithful spline for >= 6 points),
    and low-order Fourier regularization.

    Args:
        points_xy: (N, 2) array of clicked points in micrograph pixel coordinates.
        image_2d: 2D grayscale SEM micrograph array.
        cellpose_model: Optional initialized CellposeModel instance.
        num_spline_points: Number of points on output smooth regularized contour (default: 140).
        invert_brightfield: When True (brightfield light microscopy), only the inverted
            ROI polarity is tested — the physically correct one for dark-rim droplets —
            which avoids a redundant second Cellpose pass.

    Returns:
        np.ndarray: (num_spline_points, 2) float32 coordinates of the refined contour, or None.
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(pts) < 1:
        return None

    img_2d = image_2d.squeeze() if image_2d.ndim == 3 else image_2d
    if img_2d.ndim != 2:
        return None
    h, w = img_2d.shape[:2]

    # Mode 1: Single point seed -> 1-Click Dark Pore detection
    if len(pts) == 1:
        cx, cy = float(pts[0, 0]), float(pts[0, 1])
        return auto_grow_dark_bubble_contour(
            seed_xy=(cx, cy),
            image_2d=img_2d,
            cellpose_model=cellpose_model,
            num_spline_points=num_spline_points,
        )

    # Mode 2: Multi-point (>= 2 points)
    center = pts.mean(axis=0)

    # Macro-gradient map for local normal snapping
    blurred = cv2.GaussianBlur(img_2d.astype(np.float32), (15, 15), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)

    # Local Normal Snap: shift each user click along its radial direction within ±10 px to max dI/dr
    snapped_pts = []
    for p in pts:
        dx = float(p[0] - center[0])
        dy = float(p[1] - center[1])
        dist = math.hypot(dx, dy)
        ux = dx / max(1e-6, dist)
        uy = dy / max(1e-6, dist)

        best_score = -1e9
        best_k = 0
        for k in range(-10, 11):
            xs = int(np.clip(round(p[0] + k * ux), 0, w - 1))
            ys = int(np.clip(round(p[1] + k * uy), 0, h - 1))
            sc = gmag[ys, xs] - 0.7 * abs(float(k))
            if sc > best_score:
                best_score = sc
                best_k = k

        snapped_pts.append([p[0] + best_k * ux, p[1] + best_k * uy])

    snapped_arr = np.array(snapped_pts, dtype=np.float32)
    angles = np.arctan2(snapped_arr[:, 1] - center[1], snapped_arr[:, 0] - center[0])
    order = np.argsort(angles)
    ordered_pts = snapped_arr[order]
    N_pts = len(ordered_pts)

    # Unified high-quality contour for ALL point counts (>= 2):
    # Step 1: Build initial dense spline through snapped user points (120 vertices).
    # Step 2: Run Cellpose on a wide ROI around the centroid to get the AI mask contour.
    # Step 3: Refine each of 120 spline points along its outward normal in an adaptive
    #         corridor: ±4 px near user-clicked points and ±14 px between clicks.
    #         Refinement combines Cellpose mask edge and Sobel gradient on global CLAHE.
    # Step 4: Apply _smooth_contour_fourier(n_harmonics=8, num_points=160) — same quality
    #         as automatic segment_air_bubbles_auto output.

    # Handle degenerate 2-point case: synthesise circle as initial spline
    if N_pts == 2:
        p1, p2 = ordered_pts[0], ordered_pts[1]
        c2 = (p1 + p2) / 2.0
        eff_r = max(8.0, float(np.linalg.norm(p1 - p2)) / 2.0)
        angles_circ = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
        seed_pts = np.column_stack([
            c2[0] + eff_r * np.cos(angles_circ),
            c2[1] + eff_r * np.sin(angles_circ),
        ]).astype(np.float32)
    else:
        seed_pts = ordered_pts

    # Dense periodic spline from seed points (120 vertices)
    spline_pts = _generate_periodic_bspline(seed_pts, num_points=120)

    # Adaptive Cellpose + CLAHE gradient refinement
    from cellpose_model import apply_clahe  # local import to avoid circular at module level
    clahe_img = apply_clahe(img_2d)  # global CLAHE for Sobel gradient
    gx_c = cv2.Sobel(clahe_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy_c = cv2.Sobel(clahe_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gmag_clahe = np.hypot(gx_c, gy_c)

    # Build Cellpose AI mask for the whole estimated bubble region
    ai_mask_region = np.zeros(img_2d.shape[:2], dtype=np.uint8)
    if cellpose_model is not None:
        try:
            from cellpose_model import run_inference
            # Estimate bounding box from snapped points + generous margin
            all_x = spline_pts[:, 0]
            all_y = spline_pts[:, 1]
            margin = int(max(30, np.ptp(all_x) * 0.3, np.ptp(all_y) * 0.3))
            x1_roi = max(0, int(all_x.min()) - margin)
            y1_roi = max(0, int(all_y.min()) - margin)
            x2_roi = min(w, int(all_x.max()) + margin)
            y2_roi = min(h, int(all_y.max()) + margin)
            if (x2_roi - x1_roi) >= 16 and (y2_roi - y1_roi) >= 16:
                roi_clahe = clahe_img[y1_roi:y2_roi, x1_roi:x2_roi]
                roi_inv   = (255 - roi_clahe).astype(np.uint8)
                # Estimate bubble diameter from spline bounding box
                bub_diam = max(float(np.ptp(all_x)), float(np.ptp(all_y)))
                cx_roi = int(round(float(center[0]) - x1_roi))
                cy_roi = int(round(float(center[1]) - y1_roi))
                # Brightfield droplets: the inverted polarity is the physically correct
                # one, so skip the wrong polarity entirely in Light mode.
                test_images = [roi_inv] if invert_brightfield else [roi_clahe, roi_inv]
                for test_img in test_images:
                    m_res = run_inference(
                        model=cellpose_model,
                        image=test_img,
                        diameter=float(np.clip(bub_diam, 20.0, 400.0)),
                        flow_threshold=0.8,
                        cellprob_threshold=-1.8,
                        enable_clahe=False,
                    )
                    if m_res is not None and m_res.max() > 0:
                        cy_clip = min(cy_roi, m_res.shape[0] - 1)
                        cx_clip = min(cx_roi, m_res.shape[1] - 1)
                        target_lbl = m_res[cy_clip, cx_clip]
                        if target_lbl == 0:
                            sub = m_res[
                                max(0, cy_clip - 6):cy_clip + 7,
                                max(0, cx_clip - 6):cx_clip + 7,
                            ]
                            nz = sub[sub > 0]
                            if len(nz) > 0:
                                target_lbl = int(np.bincount(nz).argmax())
                        if target_lbl > 0:
                            ai_mask_roi = (m_res == target_lbl).astype(np.uint8)
                            # Embed ROI mask back into full-image array
                            ai_mask_region[y1_roi:y2_roi, x1_roi:x2_roi] = np.maximum(
                                ai_mask_region[y1_roi:y2_roi, x1_roi:x2_roi],
                                ai_mask_roi,
                            )
                            break  # found a good AI mask — no need to try inverted
        except Exception:
            pass

    # Build lookup: which spline vertices are "near" a user-clicked point?
    # Near = within the snap radius used when initially snapping clicked points.
    NEAR_CLICK_RADIUS = 12.0  # px — tighter corridor near actual click positions
    FAR_CORRIDOR      = 14    # px — generous corridor between clicks
    NEAR_CORRIDOR     = 4     # px — tight corridor right at click positions

    near_click = np.zeros(len(spline_pts), dtype=bool)
    for user_pt in ordered_pts:
        dists_to_user = np.hypot(
            spline_pts[:, 0] - float(user_pt[0]),
            spline_pts[:, 1] - float(user_pt[1]),
        )
        near_click |= (dists_to_user <= NEAR_CLICK_RADIUS)

    # Refine each spline point along its outward normal
    K = len(spline_pts)
    refined = spline_pts.copy()
    for i in range(K):
        prev_p = spline_pts[(i - 1) % K]
        next_p = spline_pts[(i + 1) % K]
        tangent = next_p - prev_p
        norm_t = float(np.linalg.norm(tangent))
        if norm_t < 1e-6:
            continue
        normal = np.array([-tangent[1] / norm_t, tangent[0] / norm_t], dtype=np.float32)

        # Adaptive corridor width
        corridor = NEAR_CORRIDOR if near_click[i] else FAR_CORRIDOR
        cur_p = spline_pts[i]

        best_score = -1e9
        best_offset = 0
        for offset in range(-corridor, corridor + 1):
            px = float(cur_p[0]) + offset * float(normal[0])
            py = float(cur_p[1]) + offset * float(normal[1])
            ix = int(round(np.clip(px, 0, w - 1)))
            iy = int(round(np.clip(py, 0, h - 1)))

            grad_val = float(gmag_clahe[iy, ix])
            # Bonus if point is on the AI mask boundary (edge pixel)
            ai_bonus = 30.0 if ai_mask_region[iy, ix] > 0 else 0.0
            score = grad_val + ai_bonus - 0.6 * abs(float(offset))
            if score > best_score:
                best_score = score
                best_offset = offset

        refined[i] = cur_p + best_offset * normal

    # Final Fourier smoothing — 8 harmonics for natural, non-rippled oval shape
    # same quality target as segment_air_bubbles_auto auto contours
    return _smooth_contour_fourier(refined, n_harmonics=8, num_points=160)


def segment_box_to_smooth_contour(
    box_xyxy: Tuple[float, float, float, float],
    image_2d: np.ndarray,
    target_phase: str = "air_bubbles",
    cellpose_model: Optional[Any] = None,
    num_points: int = 160,
    invert_brightfield: bool = False,
) -> Optional[np.ndarray]:
    """
    Convert a user-drawn bounding box ("Smart Box") into a smooth closed contour.

    The heavy lifting is delegated to ``cellpose_model.predict_interactive_box_mask``
    (SAM image embeddings with a Scale-Normalized Cellpose + GrabCut fallback). The
    largest external contour of the returned binary mask is then regularized with
    ``_smooth_contour_fourier`` so the final outline is a natural, ripple-free SEM
    pore/bubble boundary. If no contour can be extracted (or segmentation failed),
    a smooth ellipse inscribed in the box is returned instead, so a drawn rectangle
    always produces a usable object.

    Args:
        box_xyxy: Bounding box as (x1, y1, x2, y2) in analysis-image pixel coordinates.
        image_2d: 2D grayscale analysis micrograph.
        target_phase: Phase hint forwarded to the segmenter (default: "air_bubbles").
        cellpose_model: Optional preloaded CellposeModel used by the fallback path.
        num_points: Number of output contour vertices (default: 160).
        invert_brightfield: Forwarded to ``predict_interactive_box_mask``; when True the
            fallback tests only the inverted ROI polarity (brightfield dark-rim droplets).

    Returns:
        Optional[np.ndarray]: (num_points, 2) float32 closed contour, or None when the
        box/image inputs are invalid.
    """
    n_out = max(int(num_points), 16)

    if image_2d is None or not isinstance(image_2d, np.ndarray) or image_2d.size == 0:
        return None

    img_2d = image_2d.squeeze() if image_2d.ndim == 3 else image_2d
    if img_2d.ndim != 2:
        return None

    if box_xyxy is None or not isinstance(box_xyxy, (tuple, list)) or len(box_xyxy) != 4:
        return None

    try:
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None

    def _inscribed_ellipse() -> np.ndarray:
        """Smooth ellipse inscribed in the bounding box (fallback outline)."""
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        rx = max((x2 - x1) / 2.0, 0.5)
        ry = max((y2 - y1) / 2.0, 0.5)
        theta = np.linspace(0.0, 2.0 * math.pi, n_out, endpoint=False)
        return np.column_stack(
            [cx + rx * np.cos(theta), cy + ry * np.sin(theta)]
        ).astype(np.float32)

    # Local import to avoid a circular dependency at module import time.
    try:
        from cellpose_model import predict_interactive_box_mask
    except Exception:
        return _inscribed_ellipse()

    mask = None
    try:
        mask = predict_interactive_box_mask(
            image_2d=img_2d,
            box_xyxy=(x1, y1, x2, y2),
            target_phase=str(target_phase or "air_bubbles"),
            cellpose_model=cellpose_model,
            invert_brightfield=bool(invert_brightfield),
        )
    except Exception:
        mask = None

    if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
        return _inscribed_ellipse()

    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return _inscribed_ellipse()

    cnt = max(contours, key=cv2.contourArea)
    if cnt is None or len(cnt) < 6:
        return _inscribed_ellipse()

    cnt_pts = cnt.reshape(-1, 2).astype(np.float32)
    area_px = float(abs(cv2.contourArea(cnt)))
    perimeter_px = float(cv2.arcLength(cnt, True))
    circularity = (
        4.0 * math.pi * area_px / (perimeter_px ** 2) if perimeter_px > 1e-6 else 1.0
    )

    _, _, cnt_w, cnt_h = cv2.boundingRect(cnt)
    bbox_area = float(max(int(cnt_w), 1) * max(int(cnt_h), 1))
    box_fill = area_px / max((x2 - x1) * (y2 - y1), 1.0)
    rectangularity = area_px / bbox_area if bbox_area > 0 else 1.0

    # Safety net: a contour that fills almost the entire drawn box AND almost its own
    # bounding box is the user's rectangle leaking through segmentation — never let it
    # masquerade as a particle (the inscribed ellipse is always a valid fallback).
    if box_fill >= 0.93 and rectangularity >= 0.93:
        return _inscribed_ellipse()

    # Small or highly circular objects need fewer harmonics (6) to stay round;
    # larger irregular pores keep 8 harmonics for the true SEM wall shape.
    is_small = area_px < 400.0 or max(int(cnt_w), int(cnt_h)) < 24
    n_harmonics = 6 if (is_small or circularity >= 0.78) else 8

    smooth = _smooth_contour_fourier(cnt_pts, n_harmonics=n_harmonics, num_points=n_out)
    if smooth is None or len(smooth) < 3:
        return _inscribed_ellipse()
    return np.asarray(smooth, dtype=np.float32)


def parse_canvas_manual_objects(
    canvas_json: Optional[dict],
    scale_ratio: float,
    original_image: np.ndarray,
    pixel_to_um_ratio: float,
    start_id: int = 10001,
    image_data: Optional[np.ndarray] = None,
    assist_mode: Optional[str] = "🎯 AI-Assisted Points (Click around bubble — AI snaps to real edge)",
    cellpose_model: Optional[Any] = None,
    target_phase: str = "air_bubbles",
    drawing_mode_label: Optional[str] = None,
    max_air_size_um: Optional[float] = None,
    auto_snap_mode: Optional[Union[str, bool]] = None,
    invert_brightfield: bool = False,
    **kwargs,
) -> Tuple[list[dict], list[dict]]:
    """
    Parse manually drawn canvas objects into metrics + overlay polygons.

    Supports four responsive interactive modes:
    1) '🎯 AI-Assisted Points (Click around bubble — AI snaps to real edge)':
       - Employs local Cellpose inference + edge-constrained watershed to snap
         to the real physical SEM pore boundary, correcting user click inaccuracies.
    2) '🪄 1-Click Dark Pore Center (Auto-detect from 1 click)':
       - Treats each clicked point as a bubble seed, expanding 36 radial rays to detect
         the crater rim and generating a smooth closed spline.
    3) '📐 Exact Manual Points (No AI — connect my points smoothly)':
       - Strictly preserves user click points, constructing a smooth periodic B-spline without shifting.
    4) Smart Box rectangles (fabric.js `rect` objects):
       - Each drawn rectangle is sent to `segment_box_to_smooth_contour` (SAM image
         embeddings with Scale-Normalized Cellpose fallback) and the resulting mask
         contour is Fourier-smoothed (n_harmonics=8) before being committed.

    Args:
        canvas_json: Canvas JSON dictionary from canvas_result.json_data.
        scale_ratio: Responsive scale factor between display canvas and analysis micrograph.
        original_image: 2D analysis micrograph.
        pixel_to_um_ratio: Micrometers per pixel calibration factor.
        start_id: Starting integer ID for manual objects (defaults to 10001).
        image_data: Optional canvas RGBA image array for raster fallback.
        assist_mode: Selected Selection Method mode string.
        cellpose_model: Optional initialized CellposeModel instance for local AI snapping.
        target_phase: Phase hint forwarded to Smart Box segmentation (default: "air_bubbles").
        drawing_mode_label: Optional backward-compatible drawing mode string.
        max_air_size_um: Optional maximum air bubble size in um for bounding ray searches.
        auto_snap_mode: Optional backward-compatible snap mode.
        invert_brightfield: True for brightfield light microscopy; forwarded down the AI
            chain so only the inverted ROI polarity is evaluated (faster + more accurate
            for dark-rim droplets).

    Returns:
        Tuple[list[dict], list[dict]]: (metrics_list, polygons_list)
    """
    eff_scale = max(float(scale_ratio), 1e-6)
    img_2d = original_image.squeeze() if original_image.ndim == 3 else original_image
    target_h, target_w = img_2d.shape[:2]

    metrics_list: list[dict] = []
    polygons_list: list[dict] = []
    curr_id = start_id

    objects = canvas_json.get("objects", []) if isinstance(canvas_json, dict) else []
    has_valid_json_objects = False

    def _commit_contour(pts_arr: np.ndarray, shape_type: str) -> bool:
        nonlocal curr_id, has_valid_json_objects
        if pts_arr is None or len(pts_arr) < 3:
            return False
        try:
            metrics = calculate_manual_object_metrics(
                points=pts_arr,
                original_image=img_2d,
                pixel_to_um_ratio=pixel_to_um_ratio,
                obj_id=curr_id,
            )
            # Re-rasterize refined contour to get exact pixel-based metrics
            refined_mask = np.zeros((target_h, target_w), dtype=np.uint8)
            pts_int = np.round(pts_arr).astype(np.int32)
            cv2.fillPoly(refined_mask, [pts_int], 1)
            ref_area_px = float(np.count_nonzero(refined_mask))
            if ref_area_px > 0:
                pixel_area_scale = float(pixel_to_um_ratio) ** 2
                metrics["area_um2"] = ref_area_px * pixel_area_scale
                metrics["diameter_um"] = (
                    math.sqrt(4.0 * metrics["area_um2"] / math.pi)
                    if metrics["area_um2"] > 0
                    else 0.0
                )
                ys, xs = np.where(refined_mask > 0)
                if len(xs) > 0:
                    metrics["centroid_x"] = float(xs.mean())
                    metrics["centroid_y"] = float(ys.mean())
                    metrics["mean_intensity"] = float(img_2d[refined_mask > 0].mean())

            metrics_list.append(metrics)
            polygons_list.append({
                "id": curr_id,
                "points": pts_arr,
                "shape_type": shape_type,
            })
            curr_id += 1
            has_valid_json_objects = True
            return True
        except Exception:
            return False

    mode_tag = str(assist_mode or drawing_mode_label or "").lower()

    if objects:
        point_circles = []
        regular_objects = []

        for obj in objects:
            obj_type = str(obj.get("type", "")).lower()
            if obj_type == "circle":
                radius = float(obj.get("radius", 0.0))
                scale_x = float(obj.get("scaleX", 1.0))
                scale_y = float(obj.get("scaleY", 1.0))
                origin_x = obj.get("originX", "left")
                origin_y = obj.get("originY", "top")

                if origin_x == "center":
                    cx_canvas = float(obj.get("left", 0.0))
                else:
                    cx_canvas = float(obj.get("left", 0.0)) + radius * scale_x

                if origin_y == "center":
                    cy_canvas = float(obj.get("top", 0.0))
                else:
                    cy_canvas = float(obj.get("top", 0.0)) + radius * scale_y

                cx = cx_canvas / eff_scale
                cy = cy_canvas / eff_scale
                eff_r = (radius * (scale_x + scale_y) / 2.0) / eff_scale
                point_circles.append((cx, cy, eff_r, radius))
            else:
                regular_objects.append(obj)

        is_ai_mode = ("ai" in mode_tag) or ("assisted" in mode_tag) or ("rim" in mode_tag) or ("multi-point" in mode_tag)
        is_center_mode = ("1-click" in mode_tag) or ("pore center" in mode_tag) or ("auto-grow" in mode_tag) or ("dark pore" in mode_tag)
        is_exact_mode = ("exact" in mode_tag) or ("no ai" in mode_tag)
        is_geometric_mode = ("circle" in mode_tag) or ("ellipse" in mode_tag) or ("geometr" in mode_tag)

        # 1. AI-Assisted Mode (Default)
        if is_ai_mode and not is_center_mode and not is_exact_mode and point_circles:
            pts_xy = np.array([[c[0], c[1]] for c in point_circles], dtype=np.float32)
            refined_contour = ai_segment_bubble_from_points(
                points_xy=pts_xy,
                image_2d=img_2d,
                cellpose_model=cellpose_model,
                num_spline_points=140,
                invert_brightfield=bool(invert_brightfield),
            )
            if refined_contour is not None and len(refined_contour) >= 3:
                _commit_contour(refined_contour, "ai_bubble")

        # 2. 1-Click Dark Pore Center Mode
        elif is_center_mode and point_circles:
            if max_air_size_um is not None and float(pixel_to_um_ratio) > 0:
                r_max_calc = (float(max_air_size_um) / float(pixel_to_um_ratio)) / 2.0 * 1.25
                r_max = float(np.clip(r_max_calc, 15.0, min(target_h, target_w) * 0.48))
            else:
                r_max = 120.0

            for cx, cy, _, _ in point_circles:
                bubble_contour = auto_grow_dark_bubble_contour(
                    seed_xy=(cx, cy),
                    image_2d=img_2d,
                    r_max_px=r_max,
                    r_min_px=8.0,
                    num_rays=36,
                    num_spline_points=140,
                    cellpose_model=cellpose_model,
                )
                if bubble_contour is not None and len(bubble_contour) >= 3:
                    _commit_contour(bubble_contour, "center_seed")

        # 3. Exact Manual Points Mode (No AI — connect my points smoothly)
        elif is_exact_mode and point_circles:
            pts_xy = np.array([[c[0], c[1]] for c in point_circles], dtype=np.float32)
            if len(pts_xy) >= 3:
                mean_x = float(pts_xy[:, 0].mean())
                mean_y = float(pts_xy[:, 1].mean())
                angles = np.arctan2(pts_xy[:, 1] - mean_y, pts_xy[:, 0] - mean_x)
                ordered_pts = pts_xy[np.argsort(angles)]
                # Interpolating periodic cubic spline: the curve passes EXACTLY through
                # every user click (0.0 px offset) — no Fourier harmonics, no gradients.
                if is_geometric_mode:
                    refined_contour = _fit_smooth_ellipse_contour(ordered_pts, num_points=160)
                else:
                    refined_contour = _generate_periodic_bspline(ordered_pts, num_points=160)
                _commit_contour(refined_contour, "exact_manual")
            elif len(pts_xy) == 2:
                p1, p2 = pts_xy[0], pts_xy[1]
                center = (p1 + p2) / 2.0
                eff_r = float(np.linalg.norm(p1 - p2)) / 2.0
                if eff_r > 0:
                    angles = np.linspace(0.0, 2.0 * math.pi, 60, endpoint=False)
                    circle_pts = np.column_stack([
                        center[0] + eff_r * np.cos(angles),
                        center[1] + eff_r * np.sin(angles),
                    ]).astype(np.float32)
                    # Geometric circle/ellipse through the 2 clicked points (no shifting).
                    refined_contour = _fit_smooth_ellipse_contour(circle_pts, num_points=160)
                    _commit_contour(refined_contour, "exact_circle")

        # Fallback for point circles
        elif point_circles:
            pts_xy = np.array([[c[0], c[1]] for c in point_circles], dtype=np.float32)
            refined_contour = ai_segment_bubble_from_points(
                points_xy=pts_xy,
                image_2d=img_2d,
                cellpose_model=cellpose_model,
                num_spline_points=140,
                invert_brightfield=bool(invert_brightfield),
            )
            if refined_contour is not None and len(refined_contour) >= 3:
                _commit_contour(refined_contour, "ai_bubble")

        # Process freehand paths and polygons
        for obj in regular_objects:
            obj_type = str(obj.get("type", "")).lower()

            if obj_type in ("path", "freedraw"):
                path = obj.get("path") or obj.get("points") or []
                pts_raw = []
                for item in path:
                    if isinstance(item, dict) and "x" in item and "y" in item:
                        pts_raw.append([float(item["x"]), float(item["y"])])
                    elif isinstance(item, (list, tuple)) and len(item) >= 3:
                        try:
                            pts_raw.append([float(item[-2]), float(item[-1])])
                        except (TypeError, ValueError):
                            continue
                if len(pts_raw) < 3:
                    continue

                scale_x = float(obj.get("scaleX", 1.0))
                scale_y = float(obj.get("scaleY", 1.0))
                left = float(obj.get("left", 0.0))
                top = float(obj.get("top", 0.0))
                path_offset = obj.get("pathOffset", {})
                po_x = float(path_offset.get("x", 0.0))
                po_y = float(path_offset.get("y", 0.0))

                pts = []
                for px, py in pts_raw:
                    c_x = left + (px - po_x) * scale_x
                    c_y = top + (py - po_y) * scale_y
                    pts.append([c_x / eff_scale, c_y / eff_scale])

                pts_arr = np.array(pts, dtype=np.float32)
                if len(pts_arr) < 3:
                    continue

                pts_arr = snap_contour_to_edges(
                    points=pts_arr,
                    original_image=img_2d,
                    search_dist=14,
                )
                _commit_contour(pts_arr, "freedraw")

            elif obj_type == "polygon":
                raw_pts = obj.get("points", [])
                if len(raw_pts) < 3:
                    continue

                scale_x = float(obj.get("scaleX", 1.0))
                scale_y = float(obj.get("scaleY", 1.0))
                path_offset = obj.get("pathOffset", {})
                po_x = float(path_offset.get("x", 0.0))
                po_y = float(path_offset.get("y", 0.0))

                origin_x = obj.get("originX", "left")
                origin_y = obj.get("originY", "top")

                if origin_x == "center":
                    center_x = float(obj.get("left", 0.0))
                else:
                    center_x = float(obj.get("left", 0.0)) + (float(obj.get("width", 0.0)) / 2.0) * scale_x

                if origin_y == "center":
                    center_y = float(obj.get("top", 0.0))
                else:
                    center_y = float(obj.get("top", 0.0)) + (float(obj.get("height", 0.0)) / 2.0) * scale_y

                pts = []
                for p in raw_pts:
                    c_x = center_x + (float(p["x"]) - po_x) * scale_x
                    c_y = center_y + (float(p["y"]) - po_y) * scale_y
                    pts.append([c_x / eff_scale, c_y / eff_scale])

                pts_arr = np.array(pts, dtype=np.float32)
                if len(pts_arr) < 3:
                    continue

                pts_arr = snap_contour_to_edges(
                    points=pts_arr,
                    original_image=img_2d,
                    search_dist=14,
                )
                _commit_contour(pts_arr, "polygon")

            elif obj_type == "rect":
                # 📦 Smart Box: convert the drawn rectangle into a SAM/Cellpose mask
                left = float(obj.get("left", 0.0)) / eff_scale
                top = float(obj.get("top", 0.0)) / eff_scale
                w_box = (float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))) / eff_scale
                h_box = (float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))) / eff_scale
                if w_box >= 5.0 and h_box >= 5.0:
                    x1, y1 = max(0.0, left), max(0.0, top)
                    x2, y2 = min(float(target_w - 1), left + w_box), min(float(target_h - 1), top + h_box)
                    box_contour = segment_box_to_smooth_contour(
                        box_xyxy=(x1, y1, x2, y2),
                        image_2d=img_2d,
                        target_phase=str(target_phase or "air_bubbles"),
                        cellpose_model=cellpose_model,
                        num_points=160,
                        invert_brightfield=bool(invert_brightfield),
                    )
                    if box_contour is not None and len(box_contour) >= 3:
                        _commit_contour(box_contour, "smart_box")

    if not has_valid_json_objects and image_data is not None:
        raster_metrics, raster_polys = parse_canvas_manual_objects_from_raster(
            image_data=image_data,
            original_image=img_2d,
            pixel_to_um_ratio=pixel_to_um_ratio,
            start_id=start_id,
            invert_brightfield=bool(invert_brightfield),
        )
        if raster_metrics:
            return raster_metrics, raster_polys

    return metrics_list, polygons_list
    return metrics_list, polygons_list


def create_combined_training_mask(
    base_masks: Optional[np.ndarray],
    manual_polygons: Optional[list[dict]],
    target_shape: Optional[Tuple[int, int]] = None,
    fat_on_top: bool = True,
) -> np.ndarray:
    """
    Create a unified 2D integer label mask combining automated Cellpose masks and
    manual polygons for Cellpose retraining (Human-in-the-Loop).

    When fat_on_top is True, manual polygons are written only where base_masks == 0,
    ensuring that small fat particles residing inside large air pores are NEVER overwritten.

    Args:
        base_masks: Optional 2D integer mask from automated segmentation (0=bg, 1..N=objects).
        manual_polygons: Optional list of manual contour dicts ({'id': int, 'points': np.ndarray}).
        target_shape: (H, W) spatial dimensions if base_masks is None or empty.
        fat_on_top: Whether fine fat particles take priority over manual pores (defaults to True).

    Returns:
        np.ndarray: 2D uint16 array with 0=background, unique positive integers per particle.
    """
    if base_masks is not None and isinstance(base_masks, np.ndarray) and base_masks.size > 0:
        h, w = base_masks.shape[:2]
        combined = base_masks.astype(np.int32).copy()
    elif target_shape is not None:
        h, w = target_shape
        combined = np.zeros((h, w), dtype=np.int32)
    else:
        raise ValueError("Either base_masks or target_shape must be provided.")

    if manual_polygons:
        max_existing_id = int(combined.max()) if combined.size > 0 else 0
        next_label = max(max_existing_id + 1, 1)

        for mp in manual_polygons:
            pts = mp.get("points")
            if pts is None or len(pts) < 3:
                continue
            pts_int = np.round(np.asarray(pts, dtype=np.float32)).astype(np.int32)
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [pts_int], 1)
            if np.any(temp_mask):
                if fat_on_top:
                    # Write manual pore ONLY on background to preserve fine overlying fat particles
                    target_zone = (temp_mask > 0) & (combined == 0)
                else:
                    target_zone = (temp_mask > 0)

                if np.any(target_zone):
                    combined[target_zone] = next_label
                    next_label += 1

    # Compact labels to ensure dense range [0, 1, 2, ..., N]
    unique_labels = np.unique(combined)
    unique_labels = unique_labels[unique_labels > 0]
    dense_mask = np.zeros_like(combined, dtype=np.uint16)
    for new_idx, old_id in enumerate(unique_labels, start=1):
        dense_mask[combined == old_id] = np.uint16(new_idx)

    return dense_mask


def prepare_cellpose_training_package(
    image: np.ndarray,
    base_masks: Optional[np.ndarray],
    manual_polygons: Optional[list[dict]],
    base_name: str = "micrograph",
    df: Optional[pd.DataFrame] = None,
) -> bytes:
    """
    Generate an in-memory ZIP package containing:
      1. {base_name}_img.tif: Original SEM micrograph (uint8 grayscale).
      2. {base_name}_masks.tif: Standard Cellpose unified mask (fat on top).
      3. {base_name}_masks_combined_fat_on_top.tif: Combined mask preserving internal fat inclusions.
      4. {base_name}_masks_air_only.tif: Air bubbles / pores only.
      5. {base_name}_masks_fat_only.tif: Fat particles only.

    Args:
        image: 2D or 3D grayscale/RGB array of the micrograph.
        base_masks: 2D array of automated masks.
        manual_polygons: List of manual polygon dicts.
        base_name: Base filename prefix for exported files.
        df: Optional classified DataFrame to separate air bubbles from fat particles.

    Returns:
        bytes: Compressed ZIP file buffer.
    """
    img_2d = image.squeeze() if image.ndim == 3 and image.shape[2] == 1 else image
    if img_2d.ndim == 3:
        img_2d = cv2.cvtColor(img_2d, cv2.COLOR_RGB2GRAY)
    if img_2d.dtype != np.uint8:
        min_v, max_v = float(img_2d.min()), float(img_2d.max())
        img_u8 = np.clip(((img_2d - min_v) / max(max_v - min_v, 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img_2d.copy()

    h, w = img_u8.shape[:2]

    # 1. Combined mask with fat on top (manual pores do not overwrite fat droplets)
    mask_fat_on_top = create_combined_training_mask(
        base_masks=base_masks,
        manual_polygons=manual_polygons,
        target_shape=(h, w),
        fat_on_top=True,
    )

    # 2. Air bubbles only mask
    air_mask = np.zeros((h, w), dtype=np.int32)
    next_air_label = 1
    air_ids = set()
    if df is not None and not df.empty and "phase_class" in df.columns:
        air_ids = set(df[df["phase_class"] == "air_bubbles"]["id"].tolist())

    if base_masks is not None and np.any(base_masks) and air_ids:
        for aid in air_ids:
            if np.any(base_masks == aid):
                air_mask[base_masks == aid] = next_air_label
                next_air_label += 1

    if manual_polygons:
        for mp in manual_polygons:
            poly_id = mp.get("id")
            # Include if classified as air or if no explicit classification exists
            is_air = (poly_id in air_ids) if air_ids else True
            if is_air:
                pts = mp.get("points")
                if pts is not None and len(pts) >= 3:
                    pts_int = np.round(np.asarray(pts, dtype=np.float32)).astype(np.int32)
                    temp_p = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(temp_p, [pts_int], 1)
                    if np.any(temp_p):
                        air_mask[temp_p > 0] = next_air_label
                        next_air_label += 1

    # Compact air mask
    air_unique = np.unique(air_mask)
    air_unique = air_unique[air_unique > 0]
    dense_air_mask = np.zeros_like(air_mask, dtype=np.uint16)
    for idx, old_id in enumerate(air_unique, start=1):
        dense_air_mask[air_mask == old_id] = np.uint16(idx)

    # 3. Fat particles only mask
    fat_mask = np.zeros((h, w), dtype=np.int32)
    next_fat_label = 1
    fat_ids = set()
    if df is not None and not df.empty and "phase_class" in df.columns:
        fat_ids = set(df[df["phase_class"] == "fat_particles"]["id"].tolist())

    if base_masks is not None and np.any(base_masks):
        if fat_ids:
            for fid in fat_ids:
                if np.any(base_masks == fid):
                    fat_mask[base_masks == fid] = next_fat_label
                    next_fat_label += 1
        else:
            # When df is not specified, all base_masks are assumed to be fat droplets
            unique_base = np.unique(base_masks)
            unique_base = unique_base[unique_base > 0]
            for ub in unique_base:
                fat_mask[base_masks == ub] = next_fat_label
                next_fat_label += 1

    if manual_polygons and fat_ids:
        for mp in manual_polygons:
            poly_id = mp.get("id")
            if poly_id in fat_ids:
                pts = mp.get("points")
                if pts is not None and len(pts) >= 3:
                    pts_int = np.round(np.asarray(pts, dtype=np.float32)).astype(np.int32)
                    temp_p = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(temp_p, [pts_int], 1)
                    if np.any(temp_p):
                        fat_mask[temp_p > 0] = next_fat_label
                        next_fat_label += 1

    # Compact fat mask
    fat_unique = np.unique(fat_mask)
    fat_unique = fat_unique[fat_unique > 0]
    dense_fat_mask = np.zeros_like(fat_mask, dtype=np.uint16)
    for idx, old_id in enumerate(fat_unique, start=1):
        dense_fat_mask[fat_mask == old_id] = np.uint16(idx)

    clean_base = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(base_name)).strip("_")
    if not clean_base:
        clean_base = "micrograph"

    # Encode files to in-memory TIFF buffers
    def _to_tif_bytes(arr: np.ndarray) -> bytes:
        b = io.BytesIO()
        tifffile.imwrite(b, arr, photometric="minisblack")
        return b.getvalue()

    img_bytes = _to_tif_bytes(img_u8)
    fat_on_top_bytes = _to_tif_bytes(mask_fat_on_top)
    air_only_bytes = _to_tif_bytes(dense_air_mask)
    fat_only_bytes = _to_tif_bytes(dense_fat_mask)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Standard Cellpose input pair
        zf.writestr(f"{clean_base}_img.tif", img_bytes)
        zf.writestr(f"{clean_base}_masks.tif", fat_on_top_bytes)
        # Dedicated specialized variants
        zf.writestr(f"{clean_base}_masks_combined_fat_on_top.tif", fat_on_top_bytes)
        zf.writestr(f"{clean_base}_masks_air_only.tif", air_only_bytes)
        zf.writestr(f"{clean_base}_masks_fat_only.tif", fat_only_bytes)

    return zip_buf.getvalue()



# Alias for backward compatibility
build_combined_training_mask = create_combined_training_mask


# ---------------------------------------------------------------------------
# Interfacial adsorption (Pickering) & matrix coalescence analysis
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert an arbitrary value to a finite float, falling back to ``default``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _finite_or_none(value: Any) -> Optional[float]:
    """Return a finite float or None (used to detect missing DataFrame fields)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _polygon_points_array(polygon: Any) -> Optional[np.ndarray]:
    """Normalize a manual polygon record to an (N, 2) float32 array of (x, y) points."""
    if polygon is None:
        return None
    points = polygon.get("points") if isinstance(polygon, dict) else polygon
    if points is None:
        return None
    try:
        pts = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return None
    pts = pts[:, :2]
    if not np.all(np.isfinite(pts)):
        return None
    return pts


def _rasterize_manual_object(
    manual_polygons: Optional[list],
    obj_id: int,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Rasterize every manual polygon whose 'id' equals ``obj_id`` into a uint8 mask."""
    canvas = np.zeros(shape, dtype=np.uint8)
    if not manual_polygons:
        return canvas
    target = float(int(obj_id))
    for polygon in manual_polygons:
        if not isinstance(polygon, dict):
            continue
        if _finite_or_none(polygon.get("id")) != target:
            continue
        pts = _polygon_points_array(polygon)
        if pts is None:
            continue
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return canvas


def _object_binary_mask(
    obj_id: int,
    shape: Tuple[int, int],
    manual_polygons: Optional[list],
    *mask_sources: Optional[np.ndarray],
) -> np.ndarray:
    """Build a single-object binary mask from labeled masks and/or manual polygons.

    Objects with id >= 10000 are always treated as manually drawn contours and are
    rasterized from ``manual_polygons``; automatic objects are looked up in every
    supplied labeled mask stack.
    """
    obj_id = int(obj_id)
    if obj_id >= 10000:
        return _rasterize_manual_object(manual_polygons, obj_id, shape)

    mask = np.zeros(shape, dtype=np.uint8)
    for source in mask_sources:
        if isinstance(source, np.ndarray) and source.ndim >= 2:
            src = source.squeeze() if source.ndim == 3 else source
            if src.shape[:2] == tuple(shape):
                mask |= (src == obj_id).astype(np.uint8)
    return mask


def _mask_contour_metrics(mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (largest external contour perimeter in pixels, largest contour)."""
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return 0.0, np.empty((0, 2), dtype=np.float32)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, np.empty((0, 2), dtype=np.float32)
    main = max(contours, key=cv2.contourArea)
    return float(cv2.arcLength(main, True)), main


def _mask_shape_descriptors(mask: np.ndarray) -> Tuple[float, float]:
    """Return (solidity, aspect_ratio) computed directly from a binary mask."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 1.0, 1.0
    main = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(main))
    hull_area_px = float(cv2.contourArea(cv2.convexHull(main)))
    solidity = area_px / max(hull_area_px, 1.0)
    try:
        if len(main) >= 5:
            ellipse = cv2.fitEllipse(main)
            w, h = float(ellipse[1][0]), float(ellipse[1][1])
        else:
            rect = cv2.minAreaRect(main)
            w, h = float(rect[1][0]), float(rect[1][1])
        major_axis, minor_axis = max(w, h), min(w, h)
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
    except Exception:
        aspect_ratio = 1.0
    return float(np.clip(solidity, 0.0, 1.0)), float(aspect_ratio)


def _mask_crop(mask: np.ndarray) -> Optional[dict]:
    """Crop a binary mask to its bounding box for fast pairwise gap tests."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "mask": (mask[y0:y1, x0:x1] > 0),
    }


def _masks_within_dilation(crop_a: dict, crop_b: dict, dilation_px: int) -> bool:
    """True if two cropped masks are separated by <= ``dilation_px`` pixels."""
    x0 = min(crop_a["x0"], crop_b["x0"])
    y0 = min(crop_a["y0"], crop_b["y0"])
    x1 = max(crop_a["x1"], crop_b["x1"])
    y1 = max(crop_a["y1"], crop_b["y1"])
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return False

    canvas_a = np.zeros((h, w), dtype=np.uint8)
    canvas_b = np.zeros((h, w), dtype=np.uint8)
    canvas_a[
        crop_a["y0"] - y0:crop_a["y1"] - y0,
        crop_a["x0"] - x0:crop_a["x1"] - x0,
    ] = crop_a["mask"].astype(np.uint8)
    canvas_b[
        crop_b["y0"] - y0:crop_b["y1"] - y0,
        crop_b["x0"] - x0:crop_b["x1"] - x0,
    ] = crop_b["mask"].astype(np.uint8)

    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
        )
        canvas_a = cv2.dilate(canvas_a, kernel)

    return bool(np.any((canvas_a > 0) & (canvas_b > 0)))


def compute_interfacial_adsorption_metrics(
    df: pd.DataFrame,
    selected_bubble_ids: list[int],
    air_masks: Optional[np.ndarray],
    fat_masks: Optional[np.ndarray],
    manual_polygons: Optional[list[dict]],
    image_shape: Tuple[int, int],
    pixel_to_um_ratio: float,
    d_ref_um: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Quantify Pickering-type interfacial adsorption of fat globules on selected air bubbles.

    For every bubble the function rasterizes its individual mask (automatic ids from
    ``air_masks`` / ``fat_masks`` or manual polygons with id >= 10000), then collects all
    fat globules (``phase_class`` in {'adsorbed_fat', 'fat_particles'}) whose centroid lies
    inside that mask. From these it derives surface coverage, packing density, rim coverage
    and an apparent contact angle theta for every adsorbed globule:

        theta = arcsin(L_c / d_ref)  [degrees]

    where L_c is the visible chord (major_axis_um, falling back to diameter_um) and d_ref is
    the mean diameter of free matrix fat globules (auto-derived when not supplied).
    Adsorbed area is measured as the union of the globule masks clipped by the bubble mask
    (falling back to the summed row areas when no masks are available), which prevents
    double counting of overlapping globules.

    Args:
        df: Classified metrics DataFrame (extract_metrics + 'phase_class').
        selected_bubble_ids: Bubble IDs selected by the user for interfacial analysis.
        air_masks: Optional labeled air-bubble mask (0 = background).
        fat_masks: Optional labeled fat/particle mask (0 = background).
        manual_polygons: Optional list of manual contour dicts ({'id': int, 'points': (N, 2)}).
        image_shape: Spatial shape of the analysis image as (height, width).
        pixel_to_um_ratio: Spatial calibration in micrometers per pixel (um/px).
        d_ref_um: Optional reference globule diameter in um. When None/<=0 it is derived
            from the mean diameter of free matrix fat globules.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, dict]:
            1) per-bubble table (one row per bubble that produced a non-empty mask),
            2) table of every adsorbed globule with its parent bubble and contact angle,
            3) summary dictionary of overall Pickering metrics and the used d_ref_um.
    """
    per_bubble_columns = [
        "Bubble ID",
        "Bubble Diam (µm)",
        "Bubble Area (µm²)",
        "Adsorbed Count",
        "Adsorbed Area (µm²)",
        "Surface Coverage (%)",
        "Rim Coverage (%)",
        "Packing Density (glob/µm²)",
        "Mean Contact Angle θ (°)",
        "Mean Ads Globule Diam (µm)",
    ]
    ads_columns = [
        "id",
        "parent_bubble_id",
        "diameter_um",
        "major_axis_um",
        "area_um2",
        "circularity",
        "contact_angle_deg",
        "protrusion_ratio",
    ]

    shape = (
        (int(image_shape[0]), int(image_shape[1]))
        if image_shape is not None
        else (0, 0)
    )
    ratio = _safe_float(pixel_to_um_ratio, 1.0)
    if ratio <= 0:
        ratio = 1.0
    pixel_area_um2 = ratio ** 2
    d_ref = _finite_or_none(d_ref_um)
    if d_ref is not None and d_ref <= 0:
        d_ref = None

    summary = {
        "Selected Bubbles": 0,
        "Total Bubble Area (µm²)": 0.0,
        "Total Adsorbed Area (µm²)": 0.0,
        "Overall Surface Coverage (%)": 0.0,
        "Mean Packing Density (glob/µm²)": 0.0,
        "Mean Contact Angle θ (°)": 0.0,
        "Total Adsorbed Globules": 0,
        "Mean Bubble Perimeter (µm)": 0.0,
        "d_ref_um": float(d_ref) if d_ref is not None else 0.0,
    }

    if shape[0] <= 0 or shape[1] <= 0:
        return (
            pd.DataFrame(columns=per_bubble_columns),
            pd.DataFrame(columns=ads_columns),
            summary,
        )
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "id" not in df.columns:
        return (
            pd.DataFrame(columns=per_bubble_columns),
            pd.DataFrame(columns=ads_columns),
            summary,
        )

    work = df.copy()
    if "phase_class" not in work.columns:
        work["phase_class"] = ""
    work["phase_class"] = work["phase_class"].astype(str).str.lower()
    for col in (
        "centroid_x",
        "centroid_y",
        "diameter_um",
        "area_um2",
        "major_axis_um",
        "circularity",
    ):
        if col not in work.columns:
            work[col] = np.nan
    work["id"] = pd.to_numeric(work["id"], errors="coerce")
    work = work[work["id"].notna()].copy()
    if work.empty:
        return (
            pd.DataFrame(columns=per_bubble_columns),
            pd.DataFrame(columns=ads_columns),
            summary,
        )
    work["id"] = work["id"].astype(int)

    if d_ref is None:
        fat_vals = pd.to_numeric(
            work.loc[work["phase_class"] == "fat_particles", "diameter_um"],
            errors="coerce",
        ).to_numpy(dtype=float)
        fat_vals = fat_vals[np.isfinite(fat_vals) & (fat_vals > 0)]
        if fat_vals.size == 0:
            adsorbed_vals = pd.to_numeric(
                work.loc[work["phase_class"] == "adsorbed_fat", "diameter_um"],
                errors="coerce",
            ).to_numpy(dtype=float)
            fat_vals = adsorbed_vals[np.isfinite(adsorbed_vals) & (adsorbed_vals > 0)]
        d_ref = float(fat_vals.mean()) if fat_vals.size > 0 else 1.0
    d_ref = max(float(d_ref), 1e-6)
    summary["d_ref_um"] = float(d_ref)

    # Pre-extract candidate adsorbed/matrix globules once (avoids repeated iterrows)
    particle_df = work[work["phase_class"].isin(["adsorbed_fat", "fat_particles"])]
    particles = []
    for _, prow in particle_df.iterrows():
        px = _finite_or_none(prow.get("centroid_x"))
        py = _finite_or_none(prow.get("centroid_y"))
        if px is None or py is None:
            continue
        particles.append({
            "id": int(prow["id"]),
            "cx": float(px),
            "cy": float(py),
            "diameter_um": _safe_float(prow.get("diameter_um"), 0.0),
            "area_um2": _safe_float(prow.get("area_um2"), 0.0),
            "major_axis_um": _finite_or_none(prow.get("major_axis_um")),
            "circularity": _safe_float(prow.get("circularity"), 0.0),
        })

    per_bubble_rows: list[dict] = []
    ads_rows: list[dict] = []
    perimeters_um: list[float] = []
    processed_ids: set = set()

    for raw_id in selected_bubble_ids or []:
        try:
            b_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if b_id in processed_ids:
            continue
        processed_ids.add(b_id)

        # 1) Rasterize the single bubble mask (manual polygons or labeled masks)
        b_mask = _object_binary_mask(b_id, shape, manual_polygons, air_masks, fat_masks)
        bubble_area_px = float(np.count_nonzero(b_mask))
        if bubble_area_px <= 0:
            continue
        bubble_area_um2 = bubble_area_px * pixel_area_um2

        # 2) Perimeter and equivalent diameter in physical units
        bubble_perimeter_px, _ = _mask_contour_metrics(b_mask)
        bubble_perimeter_um = bubble_perimeter_px * ratio
        perimeters_um.append(bubble_perimeter_um)
        bubble_diam_um = math.sqrt(4.0 * bubble_area_um2 / math.pi)
        b_rows = work.loc[work["id"] == b_id]
        if not b_rows.empty:
            row_diam = _finite_or_none(b_rows.iloc[0].get("diameter_um"))
            if row_diam is not None and row_diam > 0:
                bubble_diam_um = float(row_diam)

        # 3) Globules whose centroid falls inside this bubble
        ys_all, xs_all = np.nonzero(b_mask)
        bx0, bx1 = int(xs_all.min()), int(xs_all.max())
        by0, by1 = int(ys_all.min()), int(ys_all.max())
        inside: list[dict] = []
        for particle in particles:
            if particle["id"] == b_id:
                continue
            ix = int(round(particle["cx"]))
            iy = int(round(particle["cy"]))
            if ix < bx0 or ix > bx1 or iy < by0 or iy > by1:
                continue
            if 0 <= iy < shape[0] and 0 <= ix < shape[1] and b_mask[iy, ix] > 0:
                inside.append(particle)

        # 4) Union of adsorbed particle masks clipped by the bubble mask
        ads_union_mask = np.zeros(shape, dtype=np.uint8)
        row_area_sum_um2 = 0.0
        angles: list[float] = []
        auto_ids: list[int] = []
        manual_ids: list[int] = []
        for particle in inside:
            p_id = int(particle["id"])
            if p_id >= 10000:
                manual_ids.append(p_id)
            else:
                auto_ids.append(p_id)
            row_area_sum_um2 += float(particle["area_um2"])

            l_c = particle["major_axis_um"]
            if l_c is None or l_c <= 0:
                l_c = float(particle["diameter_um"])
            protrusion_ratio = float(l_c) / d_ref
            theta_deg = float(
                np.degrees(np.arcsin(np.clip(protrusion_ratio, 0.0, 1.0)))
            )
            angles.append(theta_deg)
            ads_rows.append({
                "id": p_id,
                "parent_bubble_id": b_id,
                "diameter_um": float(particle["diameter_um"]),
                "major_axis_um": float(l_c),
                "area_um2": float(particle["area_um2"]),
                "circularity": float(particle["circularity"]),
                "contact_angle_deg": theta_deg,
                "protrusion_ratio": protrusion_ratio,
            })

        # Automatic globules: vectorized label union restricted to the bubble bbox
        crop_y0, crop_y1 = by0, by1 + 1
        crop_x0, crop_x1 = bx0, bx1 + 1
        bubble_crop = b_mask[crop_y0:crop_y1, crop_x0:crop_x1] > 0
        if auto_ids:
            for source in (fat_masks, air_masks):
                if not isinstance(source, np.ndarray) or source.ndim < 2:
                    continue
                src = source.squeeze() if source.ndim == 3 else source
                if src.shape[:2] != shape:
                    continue
                region = src[crop_y0:crop_y1, crop_x0:crop_x1]
                union_crop = np.isin(region, auto_ids) & bubble_crop
                ads_union_mask[crop_y0:crop_y1, crop_x0:crop_x1] |= union_crop.astype(
                    np.uint8
                )

        # Manual globules: rasterize their contour directly
        for p_id in manual_ids:
            p_mask = _object_binary_mask(
                p_id, shape, manual_polygons, fat_masks, air_masks
            )
            if np.any(p_mask):
                ads_union_mask |= (p_mask & b_mask)

        # 5) Pickering stabilization metrics for this bubble
        union_px = float(np.count_nonzero(ads_union_mask))
        adsorbed_area_um2 = (
            union_px * pixel_area_um2 if union_px > 0 else row_area_sum_um2
        )
        surface_coverage_pct = min(
            100.0,
            (adsorbed_area_um2 / max(bubble_area_um2, 1e-6)) * 100.0,
        )
        packing_density = len(inside) / max(bubble_area_um2, 1e-6)

        # Perimeter ring of thickness max(3, d_ref) px measured inwards from the edge
        rim_thickness_px = max(3, int(round(d_ref / max(ratio, 1e-6))))
        rim_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * rim_thickness_px + 1, 2 * rim_thickness_px + 1),
        )
        eroded = cv2.erode(b_mask, rim_kernel)
        rim_mask = (b_mask > 0) & (eroded == 0)
        if not np.any(rim_mask):
            rim_mask = b_mask > 0
        rim_px = float(np.count_nonzero(rim_mask))
        rim_coverage_pct = min(
            100.0,
            float(np.count_nonzero(rim_mask & (ads_union_mask > 0)))
            / max(rim_px, 1e-6)
            * 100.0,
        )

        mean_theta = float(np.mean(angles)) if angles else 0.0
        mean_globule_diam = (
            float(np.mean([p["diameter_um"] for p in inside])) if inside else 0.0
        )

        per_bubble_rows.append({
            "Bubble ID": b_id,
            "Bubble Diam (µm)": bubble_diam_um,
            "Bubble Area (µm²)": bubble_area_um2,
            "Adsorbed Count": len(inside),
            "Adsorbed Area (µm²)": adsorbed_area_um2,
            "Surface Coverage (%)": surface_coverage_pct,
            "Rim Coverage (%)": rim_coverage_pct,
            "Packing Density (glob/µm²)": packing_density,
            "Mean Contact Angle θ (°)": mean_theta,
            "Mean Ads Globule Diam (µm)": mean_globule_diam,
        })

    per_bubble_df = pd.DataFrame(per_bubble_rows, columns=per_bubble_columns)
    ads_particles_df = pd.DataFrame(ads_rows, columns=ads_columns)

    total_bubble_area = float(per_bubble_df["Bubble Area (µm²)"].sum())
    total_ads_area = float(per_bubble_df["Adsorbed Area (µm²)"].sum())
    total_ads_count = int(per_bubble_df["Adsorbed Count"].sum())
    summary.update({
        "Selected Bubbles": int(len(per_bubble_df)),
        "Total Bubble Area (µm²)": round(total_bubble_area, 3),
        "Total Adsorbed Area (µm²)": round(total_ads_area, 3),
        "Overall Surface Coverage (%)": round(
            min(100.0, total_ads_area / max(total_bubble_area, 1e-6) * 100.0), 3
        ),
        "Mean Packing Density (glob/µm²)": round(
            float(per_bubble_df["Packing Density (glob/µm²)"].mean())
            if not per_bubble_df.empty
            else 0.0,
            4,
        ),
        "Mean Contact Angle θ (°)": round(
            float(ads_particles_df["contact_angle_deg"].mean())
            if not ads_particles_df.empty
            else 0.0,
            3,
        ),
        "Total Adsorbed Globules": total_ads_count,
        "Mean Bubble Perimeter (µm)": round(
            float(np.mean(perimeters_um)) if perimeters_um else 0.0, 3
        ),
        "d_ref_um": round(float(d_ref), 4),
    })

    return per_bubble_df, ads_particles_df, summary


def compute_matrix_coalescence_metrics(
    df: pd.DataFrame,
    fat_masks: np.ndarray,
    roi_box_px: Optional[Tuple[int, int, int, int]] = None,
    pixel_to_um_ratio: float = 1.0,
    contact_dilation_px: int = 3,
    bubble_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[Tuple[int, int]], dict]:
    """
    Quantify partial coalescence of matrix fat globules through a contact network (RAG).

    Only globules classified as ``phase_class == 'fat_particles'`` (free matrix fat outside
    the bubbles) are considered. Two globules are linked when their masks are separated by
    at most ``contact_dilation_px`` pixels (physical neck / contact) or when their centroid
    distance satisfies ``d <= 0.55 * (d_i + d_j) + contact_dilation_px * pixel_to_um_ratio``.
    Connected components of this graph are the coalescence clusters; the per-globule degree
    is the coordination number Z. When the DataFrame has no 'phase_class' column, every row
    is treated as a matrix globule.

    Args:
        df: Classified metrics DataFrame (extract_metrics + 'phase_class').
        fat_masks: Labeled fat/particle mask (0 = background) used for contact detection.
        roi_box_px: Optional region of interest (x1, y1, x2, y2) in pixels; only globules
            whose centroid falls inside the box are analysed.
        pixel_to_um_ratio: Spatial calibration in micrometers per pixel (um/px).
        contact_dilation_px: Maximum mask gap (pixels) still counted as physical contact.
        bubble_mask: Optional binary/labeled air-bubble mask. Any globule whose centroid
            falls inside a bubble is adsorbed fat and is excluded from the matrix network,
            regardless of its (possibly stale) phase_class.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, list[Tuple[int, int]], dict]:
            1) per-globule table with Z, cluster id, cluster size and topological role,
            2) cluster table for agglomerates with N_agg >= 2,
            3) list of contact edges (id_1, id_2) for drawing coalescence bridges,
            4) summary dictionary of matrix coalescence metrics.
    """
    particle_columns = [
        "id",
        "diameter_um",
        "area_um2",
        "circularity",
        "solidity",
        "aspect_ratio",
        "coordination_number",
        "cluster_id",
        "cluster_size",
        "network_role",
        "centroid_x",
        "centroid_y",
    ]
    cluster_columns = [
        "Cluster ID",
        "Globules in Cluster (N_agg)",
        "Total Cluster Area (µm²)",
        "Equivalent Cluster Diam (µm)",
        "Mean Globule Diam (µm)",
        "Mean Coordination Z",
        "Mean Solidity",
    ]

    ratio = _safe_float(pixel_to_um_ratio, 1.0)
    if ratio <= 0:
        ratio = 1.0
    dilation_px = max(int(_safe_float(contact_dilation_px, 3)), 0)
    dilation_um = dilation_px * ratio

    summary = {
        "Total Matrix Globules": 0,
        "Coalesced Fraction (%)": 0.0,
        "Isolated Monomers (%)": 0.0,
        "Branching Nodes Z>=3 (%)": 0.0,
        "Mean Coordination Number (Z)": 0.0,
        "Mean Aggregation Number (N_agg)": 0.0,
        "Mean Solidity": 0.0,
        "Mean Aspect Ratio": 0.0,
        "Contact Edges": 0,
        "Clusters (N>=2)": 0,
    }

    empty_particles = pd.DataFrame(columns=particle_columns)
    empty_clusters = pd.DataFrame(columns=cluster_columns)

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
        or "id" not in df.columns
    ):
        return empty_particles, empty_clusters, [], summary

    masks = None
    if isinstance(fat_masks, np.ndarray) and fat_masks.ndim >= 2:
        masks = fat_masks.squeeze() if fat_masks.ndim == 3 else fat_masks
        if not np.issubdtype(masks.dtype, np.integer):
            masks = masks.astype(np.int32)
    max_label = int(masks.max()) if masks is not None and masks.size > 0 else 0
    label_slices = None
    if masks is not None and max_label > 0:
        try:
            label_slices = find_objects(masks)
        except (RuntimeError, TypeError, ValueError):
            label_slices = None

    work = df.copy()
    if "phase_class" in work.columns:
        work = work[
            work["phase_class"].astype(str).str.lower() == "fat_particles"
        ].copy()
    work["id"] = pd.to_numeric(work["id"], errors="coerce")
    work = work[work["id"].notna()].copy()
    if work.empty:
        return empty_particles, empty_clusters, [], summary
    work["id"] = work["id"].astype(int)
    for col in (
        "centroid_x",
        "centroid_y",
        "diameter_um",
        "area_um2",
        "circularity",
        "solidity",
        "aspect_ratio",
    ):
        if col not in work.columns:
            work[col] = np.nan

    if roi_box_px is not None:
        try:
            rx1, ry1, rx2, ry2 = (float(v) for v in roi_box_px)
            lo_x, hi_x = sorted((rx1, rx2))
            lo_y, hi_y = sorted((ry1, ry2))
            roi_x = pd.to_numeric(work["centroid_x"], errors="coerce")
            roi_y = pd.to_numeric(work["centroid_y"], errors="coerce")
            work = work[
                (roi_x >= lo_x) & (roi_x <= hi_x) & (roi_y >= lo_y) & (roi_y <= hi_y)
            ].copy()
        except (TypeError, ValueError):
            pass
    if work.empty:
        return empty_particles, empty_clusters, [], summary


    # Build graph nodes (one per matrix globule inside the ROI)
    nodes: list[dict] = []
    for _, row in work.iterrows():
        p_id = int(row["id"])
        cx = _finite_or_none(row.get("centroid_x"))
        cy = _finite_or_none(row.get("centroid_y"))
        if cx is None or cy is None:
            continue

        # Hard geometric guard: a globule whose centroid lies INSIDE an air bubble is
        # adsorbed fat and must never enter the matrix coalescence network, even if its
        # phase_class became stale through manual edits or delayed Streamlit state.
        if isinstance(bubble_mask, np.ndarray) and bubble_mask.size > 0:
            bubble_2d = bubble_mask.squeeze() if bubble_mask.ndim == 3 else bubble_mask
            if bubble_2d.ndim > 2:
                bubble_2d = bubble_2d[..., 0]
            iy_check = int(round(cy))
            ix_check = int(round(cx))
            if (
                0 <= iy_check < bubble_2d.shape[0]
                and 0 <= ix_check < bubble_2d.shape[1]
                and bubble_2d[iy_check, ix_check] > 0
            ):
                continue

        area_um2 = _safe_float(row.get("area_um2"), 0.0)
        diameter_um = _finite_or_none(row.get("diameter_um"))
        if diameter_um is None or diameter_um <= 0:
            diameter_um = math.sqrt(4.0 * area_um2 / math.pi) if area_um2 > 0 else 0.0
        if diameter_um <= 0:
            continue

        crop = None
        if masks is not None and 0 < p_id <= max_label:
            if label_slices is not None and p_id - 1 < len(label_slices):
                sl = label_slices[p_id - 1]
                if sl is not None:
                    local = (masks[sl] == p_id).astype(np.uint8)
                    if np.any(local):
                        rows_sl, cols_sl = sl
                        crop = {
                            "x0": int(cols_sl.start),
                            "y0": int(rows_sl.start),
                            "x1": int(cols_sl.stop),
                            "y1": int(rows_sl.stop),
                            "mask": local.astype(bool),
                        }
            else:
                candidate = masks == p_id
                if np.any(candidate):
                    crop = _mask_crop(candidate.astype(np.uint8))
        local_mask = crop["mask"] if crop is not None else None

        solidity = _finite_or_none(row.get("solidity"))
        aspect_ratio = _finite_or_none(row.get("aspect_ratio"))
        if local_mask is not None and (solidity is None or aspect_ratio is None):
            mask_solidity, mask_aspect = _mask_shape_descriptors(local_mask)
            if solidity is None:
                solidity = mask_solidity
            if aspect_ratio is None:
                aspect_ratio = mask_aspect
        solidity = float(np.clip(solidity if solidity is not None else 1.0, 0.0, 1.0))
        aspect_ratio = float(
            max(aspect_ratio if aspect_ratio is not None else 1.0, 1.0)
        )

        if crop is not None:
            bbox_w = float(crop["x1"] - crop["x0"])
            bbox_h = float(crop["y1"] - crop["y0"])
        else:
            bbox_w = float(diameter_um / ratio)
            bbox_h = bbox_w

        nodes.append({
            "id": p_id,
            "cx": float(cx),
            "cy": float(cy),
            "diameter_um": float(diameter_um),
            "area_um2": float(area_um2),
            "circularity": _safe_float(row.get("circularity"), 0.0),
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "crop": crop,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
        })

    if not nodes:
        return empty_particles, empty_clusters, [], summary

    # Contact network (RAG): spatial hashing + mask-gap / centroid contact criteria
    n_nodes = len(nodes)
    adjacency: list[set] = [set() for _ in range(n_nodes)]
    edges: list[Tuple[int, int]] = []

    if n_nodes > 1:
        max_span = max(max(nd["bbox_w"], nd["bbox_h"]) for nd in nodes)
        cell_size = max(8.0, float(max_span) + float(dilation_px) + 1.0)
        grid: dict = {}
        for idx, nd in enumerate(nodes):
            key = (int(nd["cx"] // cell_size), int(nd["cy"] // cell_size))
            grid.setdefault(key, []).append(idx)

        checked: set = set()
        for (gx, gy), bucket in grid.items():
            for ox in range(-2, 3):
                for oy in range(-2, 3):
                    neighbours = grid.get((gx + ox, gy + oy))
                    if not neighbours:
                        continue
                    for i in bucket:
                        for j in neighbours:
                            if i >= j:
                                continue
                            pair = (i, j)
                            if pair in checked:
                                continue
                            checked.add(pair)

                            nd_i, nd_j = nodes[i], nodes[j]
                            dx = abs(nd_i["cx"] - nd_j["cx"])
                            dy = abs(nd_i["cy"] - nd_j["cy"])

                            contact = False
                            crop_i, crop_j = nd_i["crop"], nd_j["crop"]
                            if crop_i is not None and crop_j is not None:
                                x_gap = max(crop_i["x0"], crop_j["x0"]) - min(
                                    crop_i["x1"], crop_j["x1"]
                                )
                                y_gap = max(crop_i["y0"], crop_j["y0"]) - min(
                                    crop_i["y1"], crop_j["y1"]
                                )
                                if x_gap <= dilation_px and y_gap <= dilation_px:
                                    contact = _masks_within_dilation(
                                        crop_i, crop_j, dilation_px
                                    )

                            if not contact:
                                centroid_dist_um = math.hypot(dx, dy) * ratio
                                contact = centroid_dist_um <= (
                                    0.55
                                    * (nd_i["diameter_um"] + nd_j["diameter_um"])
                                    + dilation_um
                                )

                            if contact:
                                adjacency[i].add(j)
                                adjacency[j].add(i)
                                edges.append((int(nd_i["id"]), int(nd_j["id"])))

    coordination = np.array([len(adjacency[i]) for i in range(n_nodes)], dtype=int)


    # Connected clusters via BFS/DFS over the contact graph
    cluster_ids = np.zeros(n_nodes, dtype=int)
    cluster_members: dict = {}
    current_cluster = 0
    for start in range(n_nodes):
        if cluster_ids[start] != 0:
            continue
        current_cluster += 1
        cluster_ids[start] = current_cluster
        stack = [start]
        members: list[int] = []
        while stack:
            node_idx = stack.pop()
            members.append(node_idx)
            for neighbour in sorted(adjacency[node_idx]):
                if cluster_ids[neighbour] == 0:
                    cluster_ids[neighbour] = current_cluster
                    stack.append(neighbour)
        cluster_members[current_cluster] = members

    cluster_sizes = np.zeros(n_nodes, dtype=int)
    for members in cluster_members.values():
        for idx in members:
            cluster_sizes[idx] = len(members)

    records = []
    for idx, nd in enumerate(nodes):
        z = int(coordination[idx])
        if z <= 0:
            role = "Isolated Monomer (Z=0)"
        elif z == 1:
            role = "Dimer / Chain End (Z=1)"
        elif z == 2:
            role = "Chain Link (Z=2)"
        else:
            role = "Network Branch Node (Z>=3)"
        records.append({
            "id": int(nd["id"]),
            "diameter_um": float(nd["diameter_um"]),
            "area_um2": float(nd["area_um2"]),
            "circularity": float(nd["circularity"]),
            "solidity": float(nd["solidity"]),
            "aspect_ratio": float(nd["aspect_ratio"]),
            "coordination_number": z,
            "cluster_id": int(cluster_ids[idx]),
            "cluster_size": int(cluster_sizes[idx]),
            "network_role": role,
            "centroid_x": float(nd["cx"]),
            "centroid_y": float(nd["cy"]),
        })
    particles_coal_df = pd.DataFrame(records, columns=particle_columns)

    cluster_records = []
    for cluster_id, members in cluster_members.items():
        if len(members) < 2:
            continue
        total_area = float(sum(nodes[i]["area_um2"] for i in members))
        mean_diam = float(np.mean([nodes[i]["diameter_um"] for i in members]))
        cluster_records.append({
            "Cluster ID": int(cluster_id),
            "Globules in Cluster (N_agg)": int(len(members)),
            "Total Cluster Area (µm²)": total_area,
            "Equivalent Cluster Diam (µm)": (
                math.sqrt(4.0 * total_area / math.pi) if total_area > 0 else 0.0
            ),
            "Mean Globule Diam (µm)": mean_diam,
            "Mean Coordination Z": float(
                np.mean([coordination[i] for i in members])
            ),
            "Mean Solidity": float(np.mean([nodes[i]["solidity"] for i in members])),
        })
    clusters_df = pd.DataFrame(cluster_records, columns=cluster_columns)


    total_globules = int(n_nodes)
    multi_sizes = [len(members) for members in cluster_members.values() if len(members) >= 2]
    summary.update({
        "Total Matrix Globules": total_globules,
        "Coalesced Fraction (%)": round(
            float(np.mean(coordination >= 1)) * 100.0, 3
        ),
        "Isolated Monomers (%)": round(
            float(np.mean(coordination == 0)) * 100.0, 3
        ),
        "Branching Nodes Z>=3 (%)": round(
            float(np.mean(coordination >= 3)) * 100.0, 3
        ),
        "Mean Coordination Number (Z)": round(float(coordination.mean()), 4),
        "Mean Aggregation Number (N_agg)": round(
            float(np.mean(multi_sizes)) if multi_sizes else 0.0, 3
        ),
        "Mean Solidity": round(
            float(np.mean([nd["solidity"] for nd in nodes])), 4
        ),
        "Mean Aspect Ratio": round(
            float(np.mean([nd["aspect_ratio"] for nd in nodes])), 4
        ),
        "Contact Edges": int(len(edges)),
        "Clusters (N>=2)": int(len(multi_sizes)),
    })

    return particles_coal_df, clusters_df, edges, summary


if __name__ == "__main__":
    # Self-test block to verify metrics calculation and formulas
    print("Running analyzer.py verification test...")

    # Create synthetic test data
    img_test = np.array([
        [100, 100, 10],
        [100, 100, 10],
        [10,  10,  200],
    ], dtype=np.float32)

    mask_test = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 2],
    ], dtype=np.int32)

    ratio = 0.5  # 0.5 um per pixel
    df_result = extract_metrics(
        original_image=img_test,
        mask=mask_test,
        pixel_to_um_ratio=ratio,
    )

    print("Result DataFrame:")
    print(df_result)

    # Verification: Object 1 has 4 pixels of intensity 100
    # Area px = 4 -> Area um2 = 4 * (0.5^2) = 1.0
    # Diameter um = sqrt(4 * 1.0 / pi)
    expected_area_1 = 4.0 * (0.5 ** 2)
    expected_diam_1 = math.sqrt(4.0 * expected_area_1 / math.pi)
    expected_mean_1 = 100.0

    row_1 = df_result.loc[df_result["id"] == 1].iloc[0]
    assert math.isclose(row_1["area_um2"], expected_area_1, rel_tol=1e-5), "Area calculation mismatch"
    assert math.isclose(row_1["diameter_um"], expected_diam_1, rel_tol=1e-5), "Diameter formula mismatch"
    assert math.isclose(row_1["mean_intensity"], expected_mean_1, rel_tol=1e-5), "Mean intensity mismatch"
    assert "circularity" in df_result.columns, "Missing circularity column"
    assert "centroid_x" in df_result.columns, "Missing centroid_x column"
    assert "centroid_y" in df_result.columns, "Missing centroid_y column"
    assert row_1["circularity"] > 0, "Circularity should be positive"
    assert math.isclose(row_1["centroid_x"], 0.5, rel_tol=1e-5), "Centroid X mismatch"
    assert math.isclose(row_1["centroid_y"], 0.5, rel_tol=1e-5), "Centroid Y mismatch"

    # Granulometry metrics verification
    # Two particles: d = [2.0, 4.0]
    # sum(d^2) = 4 + 16 = 20
    # sum(d^3) = 8 + 64 = 72 => D[3,2] = 72 / 20 = 3.6
    # sum(d^4) = 16 + 256 = 272 => D[4,3] = 272 / 72 = 3.78
    test_granulo_df = pd.DataFrame({
        "id": [1, 2],
        "diameter_um": [2.0, 4.0],
        "phase_class": ["fat_particles", "fat_particles"],
    })
    g_res = calculate_granulometry_metrics(test_granulo_df)
    print("Granulometry test result:")
    print(g_res)
    fat_row = g_res.loc[g_res["Phase"] == "Fat Particles"].iloc[0]
    assert math.isclose(fat_row["D[3,2] (µm)"], 3.6, rel_tol=1e-2), "D[3,2] formula mismatch"
    assert math.isclose(fat_row["D[4,3] (µm)"], 3.78, rel_tol=1e-2), "D[4,3] formula mismatch"
    assert math.isclose(fat_row["D50 (µm)"], 3.0, rel_tol=1e-2), "D50 median mismatch"

    print("All mathematical checks and assertions passed successfully!")
