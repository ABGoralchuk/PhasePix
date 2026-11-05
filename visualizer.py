"""
Module for classification and statistical visualization of segmented SEM phases.
Separates objects into air bubbles and fat particles based on mean intensity thresholds
and plots comparative kernel density estimations (KDE).
"""

from typing import Optional, Tuple
import logging
import math
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
    "adsorbed_fat": "#E6AF00",   # Warm honey-gold for interior crater adsorbed fat globules
    # Pure bright lemon-yellow (golden-amber edge) for equatorial Pickering corona
    # globules: unmistakably yellow, so it can never clash with the lime-green
    # Unclassified underlay (#32CD32), and always lighter than the interior
    # adsorbed_fat honey-gold (#E6AF00) above.
    "corona_fat": "#FFF220",
    "Unclassified": "#32cd32",   # Lime Green for unclassified particles (RGB 50, 205, 50)
    "unclassified": "#32cd32",   # Lime Green alias
}

# Number of histogram bars used when the caller supplies no binning preference.
# A fixed default (instead of bins="auto") keeps bar heights comparable across
# micrographs with different particle counts.
DEFAULT_HIST_BINS = 15

# ---- Interfacial inspection role filter (v1.8.0) --------------------------------
# A cryo-SEM freeze-fractured bubble carries two physically different globule
# populations on two different surfaces, so the inspection overlay can be restricted
# to one of them:
#   interior crater globules (is_corona_globule == False) -> Surface Coverage (%)
#   equatorial corona globules (is_corona_globule == True) -> contact angle theta + Rim Coverage
# v1.8.4: only compact, convex, short globules can carry is_corona_globule == True —
# elongated / crescent rim-wall shards are rejected by the analyzer's compactness gates,
# so they are never rendered in 👑 Contact Angle θ Mode.
GLOBULE_ROLE_ALL = "all"
GLOBULE_ROLE_INTERIOR = "interior"
GLOBULE_ROLE_CORONA = "corona"
GLOBULE_ROLE_FILTERS = (
    GLOBULE_ROLE_ALL,
    GLOBULE_ROLE_INTERIOR,
    GLOBULE_ROLE_CORONA,
)


def normalize_globule_role(globule_role: Optional[str]) -> str:
    """Return a valid ``GLOBULE_ROLE_*`` filter value (unknown -> ``all``)."""
    role = str(globule_role or GLOBULE_ROLE_ALL).strip().lower()
    return role if role in GLOBULE_ROLE_FILTERS else GLOBULE_ROLE_ALL


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


def _extract_phase_diameters(df: pd.DataFrame, phase_class: str) -> pd.Series:
    """
    Return the cleaned numeric equivalent diameters that belong to a single phase.

    Adsorbed fat globules are strictly excluded from the ``fat_particles`` PSD, and so
    are particles clipped by a bubble wall (``is_partial_in_bubble``), because their
    visible cap would bias the distribution towards smaller apparent diameters.

    Args:
        df: DataFrame with 'diameter_um' and (optionally) 'phase_class' columns.
        phase_class: Target phase key ('fat_particles' or 'air_bubbles').

    Returns:
        pd.Series: Float Series of diameters in µm (empty when nothing matches).
    """
    if df is None or df.empty or "diameter_um" not in df.columns:
        return pd.Series(dtype=float)

    phase_key = str(phase_class).strip().lower()
    data = df

    if "phase_class" in data.columns:
        classes = data["phase_class"].astype(str).str.lower()
        if phase_key == "fat_particles":
            # PSD strictly excludes adsorbed_fat globules to prevent size underestimation
            mask = (classes == "fat_particles") & (~classes.isin(["adsorbed_fat"]))
            if "is_partial_in_bubble" in data.columns:
                mask = mask & (~data["is_partial_in_bubble"].fillna(False).astype(bool))
        else:
            mask = classes == phase_key
        series = data.loc[mask, "diameter_um"]
    else:
        series = data["diameter_um"]

    return pd.to_numeric(series, errors="coerce").dropna()


def _resolve_hist_binning(
    values: np.ndarray,
    bin_width: Optional[float] = None,
    num_bins: Optional[int] = None,
) -> Tuple[dict, str]:
    """
    Translate binning preferences into seaborn histplot kwargs plus a readable note.

    Priority: ``bin_width`` (physical bar width) > ``num_bins`` > DEFAULT_HIST_BINS.
    For ``bin_width`` the aligned ``binrange`` (multiples of the width) is passed as
    well, which keeps the real bar width equal to the requested ∆µm instead of the
    ``int(round(range / binwidth))`` approximation seaborn 0.13 would otherwise use.

    Args:
        values: 1D array of numeric diameters in µm.
        bin_width: Requested bar width in µm (must be > 0 to be used).
        num_bins: Requested number of bars (must be > 0 to be used).

    Returns:
        Tuple[dict, str]: (kwargs for sns.histplot, human readable binning note).
    """
    if bin_width is not None:
        try:
            candidate_width = float(bin_width)
        except (TypeError, ValueError):
            candidate_width = 0.0
        if math.isfinite(candidate_width) and candidate_width > 0:
            edges = _resolve_bin_edges(values, bin_width=candidate_width)
            if edges.size >= 2:
                return (
                    {
                        "binwidth": candidate_width,
                        "binrange": (float(edges[0]), float(edges[-1])),
                    },
                    f"Δ = {candidate_width:g} µm per bar",
                )

    if num_bins is not None:
        try:
            candidate_bins = int(num_bins)
        except (TypeError, ValueError):
            candidate_bins = 0
        if candidate_bins > 0:
            return {"bins": candidate_bins}, f"{candidate_bins} bins"

    return {"bins": DEFAULT_HIST_BINS}, f"{DEFAULT_HIST_BINS} bins (default)"


def _resolve_bin_edges(
    values: np.ndarray,
    bin_width: Optional[float] = None,
    num_bins: Optional[int] = None,
) -> np.ndarray:
    """
    Build the exact histogram edges used by sns.histplot for the same settings.

    When ``bin_width`` is given the range is aligned to multiples of that width
    (0.0 – 0.5 – 1.0 µm ...) so every bar covers a *true* physical interval. The
    aligned range is also handed to seaborn as ``binrange`` by
    :func:`_resolve_hist_binning`, which makes the percentage table and the
    rendered bars describe byte-identical intervals regardless of seaborn version.

    Args:
        values: 1D array of numeric diameters in µm.
        bin_width: Requested bar width in µm (must be > 0 to be used).
        num_bins: Requested number of bars (must be > 0 to be used).

    Returns:
        np.ndarray: Monotonically increasing bin edges (empty for empty input).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([], dtype=float)

    data_min = float(values.min())
    data_max = float(values.max())

    if bin_width is not None:
        try:
            width = float(bin_width)
        except (TypeError, ValueError):
            width = 0.0
        if math.isfinite(width) and width > 0:
            start = math.floor(data_min / width) * width
            stop = math.ceil(data_max / width) * width
            if stop <= start:
                stop = start + width
            steps = max(1, int(round((stop - start) / width)))
            return np.round(np.linspace(start, start + steps * width, steps + 1), 10)

    if num_bins is not None:
        try:
            bins = int(num_bins)
        except (TypeError, ValueError):
            bins = 0
        if bins > 0:
            return np.histogram_bin_edges(values, bins=bins)

    return np.histogram_bin_edges(values, bins=DEFAULT_HIST_BINS)


def plot_single_phase_distribution(
    df: pd.DataFrame,
    phase_class: str,
    title: Optional[str] = None,
    color: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4.5),
    bin_width: Optional[float] = None,
    num_bins: Optional[int] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot size distribution for a single phase with Y-axis as percentage of particles.

    Uses sns.histplot(..., stat="percent") so the Y axis shows the Percentage of
    Particles (%) within that phase and every bar sum to exactly 100%. Bars use a
    fixed physical width (``bin_width``) whenever supplied, which replaces the
    count-dependent ``bins="auto"`` behaviour that produced ~0.12 µm wide bars and
    artificially low percentages for large populations.

    Args:
        df: DataFrame with 'diameter_um' and 'phase_class' columns.
        phase_class: Target phase key ('fat_particles' or 'air_bubbles').
        title: Optional plot title.
        color: Optional matplotlib/seaborn color; defaults from PALETTE_COLORS.
        figsize: Figure dimensions (width, height) in inches.
        bin_width: Optional physical bar width in µm (e.g. 0.5 or 1.0 for fat
            particles, 5.0 or 10.0 for air bubbles). Takes priority over ``num_bins``.
        num_bins: Optional fixed number of bars, used when ``bin_width`` is not given.

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

    # The shared helper keeps adsorbed_fat globules (and bubble-clipped caps) out of
    # the fat_particles PSD, so plot and percentage table always share one population.
    phase_data = _extract_phase_diameters(df, phase_class)

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

    legend_label = f"{display_name} (n={len(phase_data):,}, Σ=100%)"
    hist_kwargs, binning_note = _resolve_hist_binning(
        phase_data.to_numpy(dtype=float), bin_width, num_bins
    )

    if len(phase_data) == 1 or phase_data.nunique() == 1:
        single_val = float(phase_data.iloc[0])
        ax.bar([single_val], [100.0], width=max(single_val * 0.05, 0.2),
               color=plot_color, alpha=0.55, edgecolor=plot_color,
               label=legend_label)
        ax.axvline(single_val, color=plot_color, linestyle="--", linewidth=1.5)
    else:
        sns.histplot(
            data=phase_data,
            stat="percent",
            kde=True,
            **hist_kwargs,
            color=plot_color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.55,
            ax=ax,
            label=legend_label,
        )

    # Subtitle documents the active binning and confirms the mass balance (Σ = 100%)
    ax.text(
        0.5, 1.01,
        f"{binning_note} · {len(ax.patches)} bars · n={len(phase_data):,} · "
        f"Σ = 100% of {display_name}",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=9, color="#555555",
    )

    ax.legend(frameon=True, loc="upper right", fontsize=10)
    ax.set_title(plot_title, fontsize=13, weight="bold", pad=24)
    ax.set_xlabel("Equivalent Diameter (µm)", fontsize=11, labelpad=8)
    ax.set_ylabel("Percentage of Particles (%)", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    current_xlim = ax.get_xlim()
    ax.set_xlim(left=max(0.0, current_xlim[0]))
    fig.tight_layout()
    return fig, ax


def compute_histogram_plot_table(
    values,
    *,
    bin_width: Optional[float] = None,
    num_bins: Optional[int] = None,
    bin_edges: Optional[np.ndarray] = None,
    interval_label: str = "Bin Interval (µm)",
    center_label: str = "Bin Center (µm)",
    lower_label: str = "Bin Lower (µm)",
    upper_label: str = "Bin Upper (µm)",
    decimals: int = 2,
    max_rows: int = 200,
) -> pd.DataFrame:
    """Build a publication-ready binned table for any numeric histogram.

    Every row describes exactly one rendered bar, so the table can recreate the
    chart in Excel / Origin / GraphPad Prism without the raw particle list: the
    editable interval label, the numeric X-axis centre, the numeric interval bounds,
    the raw count, the bar height in percent (Σ = 100 %) and the cumulative percent
    curve (closing at exactly 100 %).

    Args:
        values: Iterable of numeric values to bin (NaN/inf entries are dropped).
        bin_width: Optional physical bar width; aligned to multiples of that width.
        num_bins: Optional fixed number of bars (used when ``bin_width`` is absent).
        bin_edges: Optional explicit edges (highest priority) so the table matches a
            histogram plotted with those exact edges.
        interval_label: Column name of the textual interval (e.g. "1.00 – 1.50").
        center_label: Column name of the numeric bar centre (X axis).
        lower_label: Column name of the numeric interval start.
        upper_label: Column name of the numeric interval end.
        decimals: Decimal places used for the interval text and numeric bounds.
        max_rows: When more intervals than this are produced (wide outliers), only
            non-empty intervals are kept so the table stays readable.

    Returns:
        pd.DataFrame: Columns ``interval_label``, ``center_label``, ``lower_label``,
        ``upper_label``, 'Count (n)', 'Percentage (%)' and 'Cumulative (%)'
        (empty frame with those columns when there is nothing to bin).
    """
    columns = [
        interval_label,
        center_label,
        lower_label,
        upper_label,
        "Count (n)",
        "Percentage (%)",
        "Cumulative (%)",
    ]
    array = np.asarray(values if isinstance(values, np.ndarray) else list(values))
    try:
        array = array.astype(float)
    except (TypeError, ValueError):
        return pd.DataFrame(columns=columns)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return pd.DataFrame(columns=columns)

    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.size < 2:
            return pd.DataFrame(columns=columns)
    else:
        edges = _resolve_bin_edges(array, bin_width=bin_width, num_bins=num_bins)
    if edges.size < 2:
        return pd.DataFrame(columns=columns)

    counts, edges = np.histogram(array, bins=edges)
    total = int(counts.sum())
    if total <= 0:
        return pd.DataFrame(columns=columns)

    percentages = counts.astype(float) / float(total) * 100.0
    cumulative = np.cumsum(percentages)

    rows = []
    for idx in range(len(edges) - 1):
        lower = float(edges[idx])
        upper = float(edges[idx + 1])
        rows.append(
            {
                interval_label: f"{lower:.{decimals}f} – {upper:.{decimals}f}",
                center_label: round((lower + upper) / 2.0, decimals),
                lower_label: round(lower, decimals),
                upper_label: round(upper, decimals),
                "Count (n)": int(counts[idx]),
                "Percentage (%)": round(float(percentages[idx]), 2),
                "Cumulative (%)": round(float(cumulative[idx]), 2),
            }
        )

    table = pd.DataFrame(rows, columns=columns)
    if not table.empty:
        # Percentages are rounded per row, so pin the closing balance to exactly 100 %
        table.loc[table.index[-1], "Cumulative (%)"] = 100.0
    if len(table) > max_rows:
        table = table[table["Count (n)"] > 0].reset_index(drop=True)
    return table
def compute_phase_histogram_table(
    df: pd.DataFrame,
    phase_class: str,
    bin_width: Optional[float] = None,
    num_bins: Optional[int] = None,
    max_rows: int = 200,
) -> pd.DataFrame:
    """Build the exact, plot-ready binned table (Σ = 100%) for a single phase.

    The intervals are produced with the very same edges seaborn uses for the matching
    ``plot_single_phase_distribution`` call, so the table always explains the rendered
    bars. Adsorbed fat globules are excluded from ``fat_particles`` exactly as in the
    plot.

    Args:
        df: DataFrame with 'diameter_um' and (optionally) 'phase_class' columns.
        phase_class: Target phase key ('fat_particles' or 'air_bubbles').
        bin_width: Optional physical bar width in µm; takes priority over ``num_bins``.
        num_bins: Optional fixed number of bars.
        max_rows: When more bins than this are produced (wide outliers), only
            non-empty intervals are kept so the table stays readable.

    Returns:
        pd.DataFrame: Plot-ready columns 'Bin Interval (µm)', 'Bin Center (µm)',
        'Bin Lower (µm)', 'Bin Upper (µm)', 'Count (n)', 'Percentage (%)' and
        'Cumulative (%)' (the final row closes at exactly 100.0%).
    """
    values = _extract_phase_diameters(df, phase_class).to_numpy(dtype=float)
    return compute_histogram_plot_table(
        values,
        bin_width=bin_width,
        num_bins=num_bins,
        max_rows=max_rows,
    )


def build_coordination_plot_table(z_values) -> pd.DataFrame:
    """Build the publication-ready coordination-number (Z) bar-chart table.

    Args:
        z_values: Iterable of per-globule coordination numbers.

    Returns:
        pd.DataFrame: Columns 'Coordination Motif', 'Coordination (Z)',
        'Globule Count (n)' and 'Share of Globules (%)' (shares sum to 100 %).
    """
    columns = [
        "Coordination Motif",
        "Coordination (Z)",
        "Globule Count (n)",
        "Share of Globules (%)",
    ]
    series = pd.to_numeric(pd.Series(list(z_values or [])), errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame(columns=columns)

    total = max(int(series.size), 1)
    counts = [
        int((series == 0).sum()),
        int((series == 1).sum()),
        int((series == 2).sum()),
        int((series >= 3).sum()),
    ]
    motifs = [
        "Isolated globule",
        "Single contact (chain end)",
        "Two contacts (chain interior)",
        "Branching node (Z >= 3)",
    ]
    labels = ["Z = 0", "Z = 1", "Z = 2", "Z ≥ 3"]

    table = pd.DataFrame(
        {
            "Coordination Motif": motifs,
            "Coordination (Z)": labels,
            "Globule Count (n)": counts,
            "Share of Globules (%)": [
                round(count / float(total) * 100.0, 2) for count in counts
            ],
        },
        columns=columns,
    )
    # Rounded shares are pinned so the exported column sums to exactly 100 %
    if not table.empty:
        difference = round(100.0 - float(table["Share of Globules (%)"].sum()), 2)
        if abs(difference) >= 0.01:
            last_index = table.index[-1]
            table.loc[last_index, "Share of Globules (%)"] = round(
                float(table.loc[last_index, "Share of Globules (%)"]) + difference, 2
            )
    return table


#: Publication-ready column order of the adsorbed-globule table. Internal bookkeeping
#: flags (``in_bubble``, ``is_partial_in_bubble``, ``area_px``, …) are deliberately
#: never exported — full raw particle data lives in 🔬 Metrics / 💾 Reproducibility.
ADSORBED_GLOBULE_EXPORT_COLUMNS = [
    "Particle ID",
    "Bubble ID",
    "Globule Role",
    "Diameter (µm)",
    "Major Axis (µm)",
    "True Globule Ø (µm)",
    "Contact Chord L_c (µm)",
    "Area (µm²)",
    "Circularity (0-1)",
    "Contact Angle θ (°)",
    "Excluded from θ",
    "Protrusion Ratio d/R",
    "Protrusion Distance (µm)",
    "θ Protrusion (°)",
]


def clean_adsorbed_globule_table(ads_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return the adsorbed-globule table with clean, journal-ready column names.

    Only scientific quantities are kept — every internal bookkeeping flag
    (``in_bubble``, ``is_partial_in_bubble``, ``area_px``, …) is dropped by the
    explicit whitelist, so the on-screen detail table and its CSV can be pasted
    straight into a manuscript or a plotting package.

    Args:
        ads_df: Adsorbed-globule table returned by
            ``analyzer.compute_interfacial_adsorption_metrics``.

    Returns:
        pd.DataFrame: Columns ``ADSORBED_GLOBULE_EXPORT_COLUMNS`` (empty with those
        columns when the input is empty).
    """
    if not isinstance(ads_df, pd.DataFrame) or ads_df.empty:
        return pd.DataFrame(columns=ADSORBED_GLOBULE_EXPORT_COLUMNS)

    def _numeric(column: str) -> pd.Series:
        return pd.to_numeric(
            ads_df.get(column, pd.Series(np.nan, index=ads_df.index)),
            errors="coerce",
        )

    corona_flags = (
        ads_df.get("is_corona_globule", pd.Series(False, index=ads_df.index))
        .fillna(False)
        .astype(bool)
    )
    excluded_flags = (
        ads_df.get("excluded_from_theta", pd.Series(False, index=ads_df.index))
        .fillna(False)
        .astype(bool)
    )

    return pd.DataFrame(
        {
            "Particle ID": _numeric("id").astype("Int64"),
            "Bubble ID": _numeric("parent_bubble_id").astype("Int64"),
            "Globule Role": np.where(corona_flags, "Corona Rim", "Interior Crater"),
            "Diameter (µm)": _numeric("diameter_um").round(3),
            "Major Axis (µm)": _numeric("major_axis_um").round(3),
            "True Globule Ø (µm)": _numeric("globule_diam_um").round(3),
            "Contact Chord L_c (µm)": _numeric("corona_chord_um").round(3),
            "Area (µm²)": _numeric("area_um2").round(3),
            "Circularity (0-1)": _numeric("circularity").round(3),
            "Contact Angle θ (°)": _numeric("contact_angle_deg").round(2),
            "Excluded from θ": excluded_flags,
            "Protrusion Ratio d/R": _numeric("protrusion_ratio").round(3),
            "Protrusion Distance (µm)": _numeric("protrusion_distance_um").round(3),
            "θ Protrusion (°)": _numeric("contact_angle_protrusion_deg").round(2),
        },
        columns=ADSORBED_GLOBULE_EXPORT_COLUMNS,
    )


    return table



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
    show_corona: bool = True,
    show_unclassified: bool = True,
    adsorbed_opacity: float = 0.45,
    corona_opacity: float = 0.50,
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
    - 'adsorbed_fat': Warm honey-gold (RGB: 230, 175, 0) — interior crater globules only
    - 'corona_fat' / ``is_corona_globule == True``: Pure lemon-yellow
      (RGB: 255, 242, 32) with a warm golden-amber outer contour (RGB 255, 190, 0) —
      equatorial Pickering rim globules
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
        show_adsorbed: Whether to render and number adsorbed fat globules (defaults to True).
        show_corona: Whether to render and number equatorial Pickering corona globules —
            rows with ``phase_class == 'corona_fat'`` or ``is_corona_globule == True`` in
            ``ads_particles_df`` — in pure lemon-yellow (RGB 255, 242, 32) with a warm
            golden-amber outer contour (RGB 255, 190, 0); defaults to True. This layer is
            100% independent of ``show_adsorbed``: corona globules stay visible even when
            the interior adsorbed-fat layer is switched off. Interior ``adsorbed_fat``
            globules keep the warm honey-gold tint (RGB 230, 175, 0) and stay visible when
            ``show_corona`` is False, so the two globule populations never merge and never
            clash with the lime-green Unclassified underlay (RGB 50, 205, 50).
        corona_opacity: Opacity for the lemon-yellow corona_fat overlay layer (0.0 to 1.0,
            defaults to 0.50).
        show_unclassified: Whether to render and number green Unclassified particles
            (defaults to True). Turning it off keeps the overlay focused on the
            explicitly classified phases.
        z_order: Layer rendering sequence ('Fat over Air' or 'Air over Fat'). Defaults to 'Fat over Air'.
        mask: Optional alias for masks (for backward compatibility).
        manual_polygons: Optional list of manual contour dicts ({'id': int, 'points': np.ndarray}).
        air_masks: Optional 2D integer labeled mask array specifically for air bubbles (two-pass mode).
        show_labels: Whether to render particle ID text labels at centroids (defaults to True).
        ads_particles_df: Optional table with 'id' and 'contact_angle_deg' (output of
            compute_interfacial_adsorption_metrics). When supplied, adsorbed fat globules are
            labelled with their apparent contact angle f"{theta:.0f}°" instead of their ID.
            Rows flagged ``excluded_from_theta`` keep the plain ID label.

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
    ADSORBED_COLOR = (230, 175, 0)   # Warm honey-gold for interior crater adsorbed globules
    CORONA_COLOR = (255, 242, 32)    # Pure lemon-yellow for equatorial corona globules
    CORONA_EDGE_COLOR = (255, 190, 0)  # Warm golden-amber outline around corona globules
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
    unclass_ids = df.loc[unclass_mask_series, "id"].tolist() if show_unclassified else []
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
    # Rendered with the warm honey-gold ADSORBED_COLOR (230, 175, 0) defined above:
    # interior crater globules only — corona ids are peeled off for the lemon-yellow 2c pass.
    # Corona ids are collected first and UNCONDITIONALLY (never inside ``if show_corona``):
    # the two globule layers are 100% independent, so hiding the corona layer must not
    # repaint equatorial rim globules as interior honey-gold ones, and hiding the adsorbed
    # layer must never hide the corona globules.
    corona_ids: set = set()
    if "phase_class" in df.columns:
        corona_phase = (
            df["phase_class"].astype(str).str.lower() == "corona_fat"
        )
        for value in df.loc[corona_phase, "id"].tolist():
            try:
                corona_ids.add(int(value))
            except (TypeError, ValueError):
                continue
    if (
        isinstance(ads_particles_df, pd.DataFrame)
        and not ads_particles_df.empty
        and {"id", "is_corona_globule"}.issubset(ads_particles_df.columns)
    ):
        corona_flags = (
            ads_particles_df["is_corona_globule"].fillna(False).astype(bool)
        )
        for value in ads_particles_df.loc[corona_flags, "id"].tolist():
            try:
                corona_ids.add(int(value))
            except (TypeError, ValueError):
                continue
    if show_adsorbed and adsorbed_opacity > 0.0:
        pf_opacity = float(np.clip(adsorbed_opacity, 0.0, 1.0))
        partial_ids = df.loc[df["phase_class"].astype(str).str.lower().isin(["adsorbed_fat"]), "id"].tolist()
        interior_ids = [
            pid for pid in partial_ids if int(pid) not in corona_ids
        ]
        if interior_ids:
            partial_mask = np.isin(mask_2d, interior_ids)
            if np.any(partial_mask):
                blended[partial_mask] = (
                    blended[partial_mask].astype(np.float32) * (1.0 - pf_opacity)
                    + np.array(ADSORBED_COLOR, dtype=np.float32) * pf_opacity
                ).clip(0, 255).astype(np.uint8)
            pf_cnts, _ = cv2.findContours(partial_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for cnt in pf_cnts:
                cv2.drawContours(blended, [cnt], -1, (205, 150, 0), 1, lineType=cv2.LINE_AA)
        _draw_manual_for_phase("adsorbed_fat", ADSORBED_COLOR, pf_opacity)

    # 2c. Render corona_fat (equatorial Pickering rim globules) in pure lemon-yellow.
    # A corona globule sits on the fractured rim cross-section and is the only globule
    # population that carries a three-phase contact angle, so it must never be confused
    # with the honey-gold interior crater globules above. This pass is gated on
    # ``show_corona`` ALONE — never on ``show_adsorbed`` — so the two globule layers stay
    # 100% independent in both directions.
    if show_corona and corona_opacity > 0.0 and corona_ids:
        corona_fill_ids = sorted(corona_ids)
        corona_fill_opacity = float(np.clip(corona_opacity, 0.0, 1.0))
        # v1.8.5 — paint the WHOLE globule: a rim globule that the segmenter captured
        # only as its protruding cap is completed into its full disc before it is
        # blended, so the lemon-yellow highlight is never a half-moon.
        corona_mask = _completed_globule_union(
            mask_2d, corona_fill_ids, manual_polygons
        )
        if air_masks is not None:
            air_m2d_c = air_masks.squeeze() if air_masks.ndim == 3 else air_masks
            if air_m2d_c.shape == mask_2d.shape:
                corona_mask = corona_mask | np.isin(air_m2d_c, corona_fill_ids)
        if np.any(corona_mask):
            blended[corona_mask] = (
                blended[corona_mask].astype(np.float32) * (1.0 - corona_fill_opacity)
                + np.array(CORONA_COLOR, dtype=np.float32) * corona_fill_opacity
            ).clip(0, 255).astype(np.uint8)
            corona_cnts, _ = cv2.findContours(
                corona_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for ccnt in corona_cnts:
                cv2.drawContours(
                    blended, [ccnt], -1, CORONA_EDGE_COLOR, 1, lineType=cv2.LINE_AA
                )
        _draw_manual_for_phase("corona_fat", CORONA_COLOR, corona_fill_opacity)

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

        # Draw labels for adsorbed_fat / corona_fat objects when their layer is enabled
        if show_adsorbed or show_corona:
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
                    excluded_value = theta_row.get("excluded_from_theta", False)
                    if bool(excluded_value) if pd.notna(excluded_value) else False:
                        # Excluded from theta by the researcher (or an interior crater
                        # globule): keep the plain ID label instead of a fake angle.
                        continue
                    theta_value = pd.to_numeric(
                        theta_row.get("contact_angle_deg"), errors="coerce"
                    )
                    if pd.notna(theta_value):
                        theta_lookup[theta_id] = float(theta_value)

            label_phases = ["adsorbed_fat"] if show_adsorbed else []
            if show_corona:
                label_phases.append("corona_fat")
            pf_df = df[df["phase_class"].astype(str).str.lower().isin(label_phases)]
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


def _completed_globule_union(
    mask_2d: np.ndarray,
    globule_ids,
    manual_polygons: Optional[list] = None,
    allowed_region: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean union of **full** (circle-completed) globule masks for ``globule_ids``.

    A Pickering rim globule is frequently segmented as its bright protruding cap only,
    while the micrograph shows a complete circle. Painting the raw label therefore turns
    the particle into a half-moon. Every globule is passed through
    ``analyzer.complete_circular_globule_mask`` before it is punched into the union, so
    the overlay shows the same whole particle the interfacial metrics measure.

    The completion is a strict superset of the visible mask and only fires for an
    unambiguous circular cap, so intact globules and rim-wall shards are untouched.

    Args:
        mask_2d: Labeled (fat) mask, already padded to the display image shape.
        globule_ids: Iterable of globule ids. Ids >= 10000 are manual contours and are
            rasterized from ``manual_polygons``.
        manual_polygons: Manual contour dicts (``{'id': int, 'points': (N, 2)}``).
        allowed_region: Optional boolean region. A globule is drawn only when its own
            *visible* mask touches this region (keeps far-away globules out of a
            bubble-centred overlay); the globule itself is then drawn in full, so a
            rim-straddling disc is never clipped into a crescent.

    Returns:
        np.ndarray: Boolean union mask (``True`` where a globule is drawn).
    """
    shape = tuple(mask_2d.shape[:2])
    union = np.zeros(shape, dtype=bool)

    ids: set = set()
    for value in globule_ids or ():
        try:
            ids.add(int(value))
        except (TypeError, ValueError):
            continue
    if not ids:
        return union

    try:
        from analyzer import complete_circular_globule_mask
    except Exception:  # pragma: no cover - analyzer is always importable in the app
        complete_circular_globule_mask = None

    def _punch(local: np.ndarray, x0: int, y0: int) -> None:
        """Draw one globule (completed when possible) into ``union``."""
        visible = local > 0
        if not np.any(visible):
            return
        height, width = visible.shape
        y0 = max(0, int(y0))
        x0 = max(0, int(x0))
        y1 = min(shape[0], y0 + height)
        x1 = min(shape[1], x0 + width)
        if y1 <= y0 or x1 <= x0:
            return
        visible = visible[: y1 - y0, : x1 - x0]
        if allowed_region is not None:
            region_crop = allowed_region[y0:y1, x0:x1]
            if region_crop.shape != visible.shape or not np.any(visible & region_crop):
                return
        completed = None
        if complete_circular_globule_mask is not None:
            try:
                completed = complete_circular_globule_mask(local, x0, y0, shape)
            except Exception:
                completed = None
        if completed is None:
            union[y0:y1, x0:x1] |= visible
            return
        completed_local, comp_x0, comp_y0, _radius = completed
        comp_y1 = min(shape[0], comp_y0 + completed_local.shape[0])
        comp_x1 = min(shape[1], comp_x0 + completed_local.shape[1])
        if comp_y1 > comp_y0 and comp_x1 > comp_x0:
            union[comp_y0:comp_y1, comp_x0:comp_x1] |= (
                completed_local[: comp_y1 - comp_y0, : comp_x1 - comp_x0] > 0
            )

    auto_ids = sorted(pid for pid in ids if pid < 10000)
    manual_ids = sorted(pid for pid in ids if pid >= 10000)

    if auto_ids:
        label_slices = None
        try:
            from scipy.ndimage import find_objects

            label_slices = find_objects(mask_2d, max(auto_ids))
        except Exception:
            label_slices = None
        for pid in auto_ids:
            local = None
            x0 = y0 = 0
            if label_slices is not None:
                # scipy >= 1.18 indexes by (label - 1), older releases by the label.
                for index in (pid - 1, pid):
                    if 0 <= index < len(label_slices):
                        window = label_slices[index]
                        if window is None:
                            continue
                        candidate = mask_2d[window] == pid
                        if np.any(candidate):
                            local = candidate.astype(np.uint8)
                            x0, y0 = int(window[1].start), int(window[0].start)
                            break
            if local is None:
                # Fast path when the label is absent (it may live in the air stack).
                local = (mask_2d == pid).astype(np.uint8)
                ys, xs = np.nonzero(local)
                if xs.size == 0:
                    continue
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                local = local[y0:y1, x0:x1]
            _punch(local, x0, y0)

    for pid in manual_ids:
        full = _rasterize_manual_object(manual_polygons, pid, shape)
        ys, xs = np.nonzero(full)
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        _punch(full[y0:y1, x0:x1], x0, y0)

    return union


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


def _draw_dashed_polyline(
    rgb: np.ndarray,
    pts: np.ndarray,
    color: Tuple[int, int, int],
    dash_px: float = 6.0,
    gap_px: float = 4.0,
    thickness: int = 1,
) -> None:
    """Draw a closed dashed polyline (in place) from an ordered (N, 2) point list.

    Used to render the Fourier-regularised bubble baseline as a clean reference line:
    a dashed stroke reads as an annotation, so it can never be mistaken for a second
    segmentation edge burned into the micrograph.
    """
    if pts is None:
        return
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] < 3:
        return
    dash = max(float(dash_px), 0.5)
    period = max(dash + max(float(gap_px), 0.0), 1.0)
    closed = np.vstack([arr, arr[0]])
    phase = 0.0  # dash-phase carried across vertices so the pattern stays continuous
    for start, end in zip(closed[:-1], closed[1:]):
        seg = end - start
        seg_len = float(np.hypot(seg[0], seg[1]))
        if seg_len < 1e-6:
            continue
        direction = seg / seg_len
        travelled = 0.0
        while travelled < seg_len:
            remaining = (dash - phase) if phase < dash else (period - phase)
            run = min(remaining, seg_len - travelled)
            if run <= 0.0:
                phase = (phase + 1e-3) % period
                continue
            if phase < dash:
                p0 = start + direction * travelled
                p1 = p0 + direction * run
                cv2.line(
                    rgb,
                    (int(round(p0[0])), int(round(p0[1]))),
                    (int(round(p1[0])), int(round(p1[1]))),
                    color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
            travelled += run
            phase = (phase + run) % period


def render_selected_bubbles_inspection_overlay(
    image: np.ndarray,
    df: pd.DataFrame,
    selected_bubble_ids: list[int],
    air_masks: Optional[np.ndarray],
    fat_masks: Optional[np.ndarray],
    manual_polygons: Optional[list[dict]],
    ads_particles_df: Optional[pd.DataFrame] = None,
    show_contact_angles: bool = True,
    show_labels: bool = True,
    corona_margin_px: int = 0,
    show_particle_ids: bool = False,
    globule_role: str = GLOBULE_ROLE_ALL,
    show_globule_ids: bool = False,
    show_bubble_ids: bool = True,
    show_fat_ids: bool = False,
) -> np.ndarray:
    """
    Build a QA inspection overlay for the user-selected bubbles and their adsorbed fat.

    Everything outside the selected bubble masks is dimmed (x0.45) so the inspected
    bubbles stay the only bright regions. Their contours are drawn in bright cyan
    (RGB 0, 210, 255), adsorbed fat globules inside them are tinted gold
    (RGB 255, 215, 0, alpha 0.50) with a crisp outline (RGB 255, 240, 120). In
    ``GLOBULE_ROLE_CORONA`` (👑 Contact Angle θ Mode) the Fourier-regularised baseline
    bubble interface is additionally drawn as a clean dashed cyan line (5 harmonics,
    180 vertices), so the exact interface across which L_c,i / θ is measured is visible
    instead of the staircase-jagged segmentation edge. Only
    numeric identifiers are burned into the image: the ID of every inspected bubble and,
    optionally, the particle ID of every rendered globule. Contact angle values (°) are
    never drawn any more, because a degree label on the picture cannot be trusted to
    match the θ table and simply covers the thin interfacial sliver.

    Args:
        image: 2D grayscale or 3D RGB base micrograph.
        df: Classified metrics DataFrame ('id', 'phase_class', 'centroid_x', 'centroid_y').
        selected_bubble_ids: Bubble IDs selected for quality-control inspection.
        air_masks: Optional labeled air-bubble mask (0 = background).
        fat_masks: Optional labeled fat/particle mask (0 = background).
        manual_polygons: Optional manual contour dicts ({'id': int, 'points': (N, 2)}).
        ads_particles_df: Optional output of compute_interfacial_adsorption_metrics
            containing 'id', 'parent_bubble_id', 'is_corona_globule' and
            'contact_angle_deg'.
        show_labels: Master switch for every text annotation (bubble IDs and particle
            IDs). Set to False for a label-free close-up that shows only the cyan bubble
            boundary and the gold adsorbed-fat masks, so the adsorption corona can be
            inspected without text clutter.
        corona_margin_px: Width in pixels of the corona rim band. When > 0 the gold
            adsorbed-fat layer is allowed to extend that far beyond the bubble
            contour, so Pickering globules straddling the interface are drawn in
            full instead of being clipped at the rim (0 keeps the strict clipping,
            which is what the areal Surface Coverage definition uses).
        globule_role: Which globule population is rendered and annotated —
            ``GLOBULE_ROLE_ALL`` (default, historic behaviour: interior + corona),
            ``GLOBULE_ROLE_INTERIOR`` (Surface Coverage mode: only
            ``is_corona_globule == False`` globules on the crater floor) or
            ``GLOBULE_ROLE_CORONA`` (Contact Angle mode: only
            ``is_corona_globule == True`` globules on the equatorial rim). Unknown
            values fall back to ``GLOBULE_ROLE_ALL``. The role is read from
            ``ads_particles_df['is_corona_globule']``, so a rim-wall shard that
            ``analyzer.compute_interfacial_adsorption_metrics`` rejected with its v1.8.4
            compactness gates (non-convex curved strip or elongated segment) can never be
            painted in 👑 Contact Angle θ Mode — and if it also lies on the outer rim band
            it never reaches this overlay at all.
        show_bubble_ids: Print the numeric ID of every inspected bubble at its mask
            centroid (defaults to True).
        show_fat_ids: Print only the bare numeric particle ID (e.g. ``142``) of every
            rendered gold adsorbed/corona globule (defaults to False). Globules flagged
            ``excluded_from_theta`` in ``ads_particles_df`` keep their ID but it is
            drawn in dim grey so it cannot be mistaken for a valid θ globule.
        show_globule_ids: Deprecated alias of ``show_fat_ids`` — still accepted so
            legacy Surface Coverage callers keep their particle IDs on the overlay.
        show_contact_angles: Deprecated and ignored — contact angle values (°) are
            never drawn on the image any more. Kept in the signature so legacy callers
            keep working.
        show_particle_ids: Deprecated and ignored (it used to prefix a θ label with
            ``#<id>: ``). Superseded by ``show_fat_ids``.

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


    # Collect adsorbed-fat ids from the classified table and/or the adsorption results.
    # ``corona_fat`` rows are collected as well: an explicitly classified (e.g. manually
    # drawn) corona globule must be painted even when the interfacial analysis could not
    # attach it to a bubble, otherwise a manual corona particle silently disappears from
    # the inspection overlay and the crop inspector.
    ads_ids: set = set()
    corona_phase_ids: set = set()
    if isinstance(df, pd.DataFrame) and not df.empty and "id" in df.columns:
        if "phase_class" in df.columns:
            phase_norm = df["phase_class"].astype(str).str.lower()
            for phase_key, target in (
                ("adsorbed_fat", ads_ids),
                ("corona_fat", corona_phase_ids),
            ):
                for value in df.loc[phase_norm == phase_key, "id"].tolist():
                    try:
                        target.add(int(value))
                    except (TypeError, ValueError):
                        continue
            ads_ids |= corona_phase_ids
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

    # Role filter (v1.8.0): the overlay can be restricted to ONE globule population.
    # Interior crater globules define the areal Surface Coverage and corona rim globules
    # define theta + Rim Coverage, so mixing them on one overlay hides the structure the
    # researcher actually wants to verify. ``role_filter_ids`` is None when no filter is
    # requested (or when the role column is unavailable, e.g. legacy callers).
    role_filter = normalize_globule_role(globule_role)
    role_filter_ids: Optional[set] = None
    if role_filter != GLOBULE_ROLE_ALL:
        if (
            isinstance(ads_particles_df, pd.DataFrame)
            and not ads_particles_df.empty
            and {"id", "is_corona_globule"}.issubset(ads_particles_df.columns)
        ):
            want_corona = role_filter == GLOBULE_ROLE_CORONA
            corona_flags = (
                ads_particles_df["is_corona_globule"].fillna(False).astype(bool).to_numpy()
            )
            role_particle_ids = (
                pd.to_numeric(ads_particles_df["id"], errors="coerce")
                .fillna(-1)
                .astype(int)
                .to_numpy()
            )
            role_filter_ids = {
                int(particle_id)
                for particle_id, is_corona in zip(role_particle_ids, corona_flags)
                if bool(is_corona) == want_corona
            }
        if role_filter == GLOBULE_ROLE_CORONA:
            # An explicit ``corona_fat`` phase is authoritative: keep the particle in the
            # θ population even when the interfacial table does not list it (manual rim
            # globules, or a globule whose chord could not be measured).
            if role_filter_ids is None:
                role_filter_ids = set()
            role_filter_ids |= corona_phase_ids
        elif role_filter == GLOBULE_ROLE_INTERIOR:
            # Surface Coverage mode must never paint an explicit ``corona_fat`` globule,
            # even when the interfacial table is empty (manual rim globules).
            ads_ids -= corona_phase_ids
        if role_filter_ids is not None:
            ads_ids &= role_filter_ids

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

    # Corona band: corona (Pickering) globules straddle the rim, so their gold mask is
    # allowed to extend a few pixels beyond the bubble contour instead of being clipped.
    ads_region_bool = selected_bool
    corona_margin = max(0, int(corona_margin_px or 0))
    if corona_margin > 0:
        corona_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * corona_margin + 1, 2 * corona_margin + 1)
        )
        ads_region_bool = (
            cv2.dilate(selected_union.astype(np.uint8), corona_kernel) > 0
        )

    ads_union = (ads_union > 0) & ads_region_bool

    # v1.8.5 — corona globules are painted as WHOLE particles: the circular completion
    # rebuilds a cap-only segmentation into its full disc, and the rim band clip is not
    # applied to them (a globule straddling the interface is not a crater globule, so
    # clipping it to Ω_b used to shave its outer half into a crescent).
    corona_draw_ids = (
        [pid for pid in ads_ids if pid in corona_phase_ids]
        if role_filter != GLOBULE_ROLE_INTERIOR
        else []
    )
    if (
        role_filter != GLOBULE_ROLE_INTERIOR
        and isinstance(ads_particles_df, pd.DataFrame)
        and not ads_particles_df.empty
        and {"id", "is_corona_globule"}.issubset(ads_particles_df.columns)
    ):
        corona_flag_series = (
            ads_particles_df["is_corona_globule"].fillna(False).astype(bool)
        )
        for value in ads_particles_df.loc[corona_flag_series, "id"].tolist():
            try:
                corona_draw_ids.append(int(value))
            except (TypeError, ValueError):
                continue
    if corona_draw_ids:
        fat_display = np.zeros(shape, dtype=np.int32)
        auto_corona_ids = {pid for pid in corona_draw_ids if pid < 10000}
        fat_present_ids: set = set()
        if isinstance(fat_masks, np.ndarray) and fat_masks.ndim >= 2:
            fat_candidate = fat_masks.squeeze() if fat_masks.ndim == 3 else fat_masks
            if fat_candidate.shape[:2] == shape:
                fat_display = fat_candidate
                if auto_corona_ids:
                    # Only labels that really exist in the fat stack count as "present":
                    # a numeric collision with a bubble label must never make the globule
                    # lookup fall back to the air mask (which would paint a whole bubble).
                    fat_present_ids = {
                        int(value)
                        for value in np.unique(fat_display)
                        if int(value) in auto_corona_ids
                    }
        corona_full_union = _completed_globule_union(
            fat_display,
            set(corona_draw_ids),
            manual_polygons,
            allowed_region=ads_region_bool,
        )
        air_fallback_ids = auto_corona_ids - fat_present_ids
        if air_fallback_ids and isinstance(air_masks, np.ndarray):
            air_2d_c = air_masks.squeeze() if air_masks.ndim == 3 else air_masks
            if air_2d_c.shape[:2] == shape:
                corona_full_union |= _completed_globule_union(
                    air_2d_c,
                    air_fallback_ids,
                    manual_polygons,
                    allowed_region=ads_region_bool,
                )
        # Only globules that touch the rim band are selected (inside the helper); once
        # selected, the WHOLE completed disc is painted, never a clipped crescent.
        ads_union = ads_union | corona_full_union

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

    # 👑 Contact Angle θ mode only: draw the SMOOTHED baseline bubble interface — the
    # exact line across which the contact chord L_c,i (and therefore θ) is measured. The
    # raw segmentation edge is staircase-jagged around every rim globule, so it cannot
    # serve as the visual reference: the Fourier-regularised contour is drawn as a clean
    # dashed cyan line so the researcher can see the true interface at a glance.
    if role_filter == GLOBULE_ROLE_CORONA:
        for b_mask in bubble_masks.values():
            baseline_cnts, _ = cv2.findContours(
                b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not baseline_cnts:
                continue
            main_baseline = max(baseline_cnts, key=cv2.contourArea)
            baseline_pts = (
                np.asarray(main_baseline, dtype=np.float32).reshape(-1, 2)
            )
            try:
                from analyzer import _smooth_contour_fourier

                smoothed_baseline = np.asarray(
                    _smooth_contour_fourier(
                        baseline_pts, n_harmonics=5, num_points=180
                    ),
                    dtype=np.float32,
                ).reshape(-1, 2)
                if smoothed_baseline.shape[0] >= 3:
                    baseline_pts = smoothed_baseline
            except Exception:
                # A raw contour is still better than no reference line at all.
                pass
            _draw_dashed_polyline(
                rgb,
                baseline_pts,
                (0, 255, 255),
                dash_px=6.0,
                gap_px=4.0,
                thickness=1,
            )

    # Role-aware globule ID annotations. Only the globules the role filter actually
    # draws are annotated, so a Surface Coverage overlay never labels a hidden corona
    # rim globule and a Contact Angle overlay never labels an interior crater globule.
    # Contact angle values (°) are never burned in any more: the image carries numeric
    # identifiers only, and the legacy ``show_globule_ids`` flag is honoured as an
    # alias of ``show_fat_ids``.
    if (
        show_labels
        and (show_fat_ids or show_globule_ids)
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

            # Never annotate a globule that the active role filter hides.
            if role_filter_ids is not None and particle_id not in role_filter_ids:
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
            # Corona globules sit on the rim, so the label anchor may fall just
            # outside the bubble contour: accept the widened corona band as well.
            if not ads_region_bool[int(round(cy)), int(round(cx))]:
                continue

            excluded_val = row.get("excluded_from_theta", False)
            excluded = bool(excluded_val) if pd.notna(excluded_val) else False

            # Bare numeric particle ID only (e.g. ``142``) — the contact angle value is
            # never burned into the image. Globules excluded from θ keep their ID for
            # traceability but are drawn in dim grey so they cannot be mistaken for a
            # globule that contributes to the θ statistics.
            text = str(particle_id)
            text_color = (150, 150, 150) if excluded else (255, 255, 255)

            (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            tx = int(np.clip(round(cx - tw / 2.0), 1, max(1, shape[1] - tw - 2)))
            ty = int(np.clip(round(cy - 5.0), th + 1, shape[0] - 2))
            cv2.putText(
                rgb, text, (tx, ty), font, font_scale, (0, 0, 0),
                thickness=outline_thickness, lineType=cv2.LINE_AA,
            )
            cv2.putText(
                rgb, text, (tx, ty), font, font_scale, text_color,
                thickness=font_thickness, lineType=cv2.LINE_AA,
            )

    # Large numeric bubble ID labels exactly at each bubble-mask centroid (drawn on
    # top). Skipped when ``show_bubble_ids`` or the ``show_labels`` master switch is
    # False, so the close-up crop can stay label-free.
    if show_labels and show_bubble_ids:
        for b_id, b_mask in bubble_masks.items():
            ys_b, xs_b = np.nonzero(b_mask)
            if len(xs_b) > 0:
                cx, cy = int(xs_b.mean()), int(ys_b.mean())
                text = str(b_id)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tx = int(np.clip(cx - tw // 2, 1, max(1, shape[1] - tw - 2)))
                ty = int(np.clip(cy + th // 2, th + 1, shape[0] - 2))
                cv2.putText(
                    rgb, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    rgb, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA,
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


def burn_in_scale_bar(
    image: np.ndarray,
    length_um: float,
    pixels_per_um: float,
    color_name: str,
    thickness: int,
    footer_height: int = 0
) -> np.ndarray:
    """Draws a scale bar with text in the bottom right corner of the image."""
    if length_um <= 0 or pixels_per_um <= 0:
        return image

    color_map = {
        "White": (255, 255, 255),
        "Black": (0, 0, 0),
        "Yellow": (255, 255, 0),
        "Red": (255, 0, 0)
    }
    color = color_map.get(color_name, (255, 255, 255))

    # Ensure image is contiguous and writable
    img = np.ascontiguousarray(image.copy())
    h, w = img.shape[:2]

    length_px = int(round(length_um * pixels_per_um))
    if length_px > w - 40:
        length_px = w - 40  # prevent drawing outside bounds

    margin_x = max(20, int(w * 0.03))

    # Text sizing
    text = f"{length_um:g} um"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, 0.45 + (w / 3000.0))
    text_thick = max(1, thickness - 1)

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, text_thick)

    # Ensure the bar doesn't get clipped into the footer or the bottom edge
    margin_y = max(20, int(h * 0.03)) + th + 15
    base_y = int(h - footer_height - margin_y)

    # Safety clamp for y
    base_y = max(th + 20, min(base_y, h - 20))

    x2 = w - margin_x
    x1 = x2 - length_px

    # Safety clamp for x
    x1 = max(10, x1)

    # Draw line with flat ends
    cv2.line(img, (x1, base_y), (x2, base_y), color, thickness, cv2.LINE_AA)

    tx = x1 + (length_px - tw) // 2
    # Safety clamp for the label so it can never leave the image on short bars
    tx = max(5, min(tx, max(5, w - tw - 5)))
    ty = base_y - 12  # hover slightly above the line

    # Draw text shadow/outline for readability against any background
    shadow_color = (0, 0, 0) if color != (0, 0, 0) else (255, 255, 255)
    cv2.putText(img, text, (tx, ty), font, font_scale, shadow_color, text_thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, font_scale, color, text_thick, cv2.LINE_AA)

    return img


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
