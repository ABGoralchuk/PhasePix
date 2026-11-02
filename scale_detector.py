"""
Module for automated detection and measurement of SEM scale bars.
Extracts calibration metrics from the upper-right databar region only,
avoiding the horizontal table separator that spans the info banner.

Since v1.7.4 the detector understands FEI / Thermo Fisher *split* databars
(``|---- 10 µm ----|``): the horizontal ray is interrupted by the central
magnification label, so both collinear halves are merged back into a single
full-span bar measured from the outermost left tick to the outermost right tick.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


# --- Split scale-bar (``|---- 10 µm ----|``) tuning -------------------------
_SPLIT_GAP_MIN_PX = 8      # narrower notches are just anti-aliased line breaks
_SPLIT_GAP_MAX_PX = 140    # the central label gap is never wider than this
_LONG_SEGMENT_PX = 20      # segment length that may bridge a text-free gap
_BAR_COVERAGE_MIN = 0.25   # ray pixels / span: rejects sparse text-only rows
_TICK_EXPAND_PX = 4        # how far the span may grow onto the outer end ticks
_HALF_BAND_PX = 3          # baseline tolerance for collinear rays (± px)


def _prepare_u8(image: np.ndarray) -> Optional[np.ndarray]:
    """Convert input image to 2D uint8 grayscale, or None if invalid."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None

    img_2d = image.squeeze() if image.ndim == 3 else image
    if img_2d.ndim != 2:
        return None

    if img_2d.dtype != np.uint8:
        min_v, max_v = float(img_2d.min()), float(img_2d.max())
        return np.clip(
            ((img_2d - min_v) / max(max_v - min_v, 1e-6)) * 255.0, 0, 255
        ).astype(np.uint8)
    return img_2d


def _band_runs(
    mask: np.ndarray, y_lo: int, y_hi: int, min_len: int = 6
) -> List[Tuple[int, int]]:
    """
    Return inclusive column runs of ``mask`` pixels inside the row band [y_lo, y_hi).

    Args:
        mask: Binary/uint8 mask (non-zero = foreground).
        y_lo: First row of the band (inclusive).
        y_hi: Last row of the band (exclusive).
        min_len: Shortest run (in px) that is reported.

    Returns:
        List of ``(x_start, x_end)`` inclusive column runs, left to right.
    """
    if mask is None or mask.size == 0 or y_hi <= y_lo:
        return []

    col_has = np.any(mask[y_lo:y_hi, :] > 0, axis=0)
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for x, has in enumerate(col_has):
        if has and start is None:
            start = int(x)
        elif not has and start is not None:
            runs.append((start, int(x) - 1))
            start = None
    if start is not None:
        runs.append((start, int(col_has.shape[0]) - 1))
    return [run for run in runs if (run[1] - run[0] + 1) >= min_len]


def _gap_has_label_text(
    binary: np.ndarray, y_lo: int, y_hi: int, gap_x0: int, gap_x1: int
) -> bool:
    """
    Return True when a gap between two collinear bar halves holds the bar label.

    On FEI / Thermo Fisher split databars the magnification string (``10 µm``) is
    drawn inside the bar, so its bright glyph pixels sit in the same baseline band
    only a few pixels above the ray. A blank notch between two anti-aliased dashes
    contains almost nothing.
    """
    if gap_x1 < gap_x0:
        return False

    height = int(binary.shape[0])
    pad = 6
    y0 = max(0, y_lo - pad)
    y1 = min(height, y_hi + pad)
    window = binary[y0:y1, gap_x0 : gap_x1 + 1]
    if window.size == 0:
        return False

    gap_width = gap_x1 - gap_x0 + 1
    bright = int(np.count_nonzero(window))
    return bright >= max(6, gap_width // 8)


def _merge_split_bar_runs(
    runs: List[Tuple[int, int]], binary: np.ndarray, y_lo: int, y_hi: int
) -> List[Tuple[int, int]]:
    """
    Merge collinear horizontal segments that flank the central databar label.

    ``|---- 10 µm ----|`` is drawn as two horizontal rays separated by the label,
    both on the same Y-row (±2 px). Their gap (typically 20–120 px) carries the
    bright ``10 µm`` glyphs, so the halves are fused into one full-span bar
    instead of being measured separately (which halved the reported length).

    Args:
        runs: Inclusive column runs found on the baseline.
        binary: Raw thresholded ROI used to confirm label text inside the gap.
        y_lo: First row of the baseline band (inclusive).
        y_hi: Last row of the baseline band (exclusive).

    Returns:
        List of merged ``(x_start, x_end)`` runs.
    """
    if not runs:
        return []

    ordered = sorted(runs)
    merged: List[List[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1
        if not (_SPLIT_GAP_MIN_PX <= gap <= _SPLIT_GAP_MAX_PX):
            merged.append([start, end])
            continue

        left_len = prev_end - prev_start + 1
        right_len = end - start + 1
        has_text = _gap_has_label_text(binary, y_lo, y_hi, prev_end + 1, start - 1)
        if has_text or (left_len >= _LONG_SEGMENT_PX and right_len >= _LONG_SEGMENT_PX):
            merged[-1][1] = end
            continue
        merged.append([start, end])

    return [(int(start), int(end)) for start, end in merged]


def _expand_over_ticks(
    binary: np.ndarray, y_lo: int, y_hi: int, x_left: int, x_right: int
) -> Tuple[int, int]:
    """Grow the span outward onto the vertical end ticks (max ``_TICK_EXPAND_PX``)."""
    width = int(binary.shape[1])
    steps = 0
    while (
        x_left > 0
        and steps < _TICK_EXPAND_PX
        and np.any(binary[y_lo:y_hi, x_left - 1] > 0)
    ):
        x_left -= 1
        steps += 1
    steps = 0
    while (
        x_right < width - 1
        and steps < _TICK_EXPAND_PX
        and np.any(binary[y_lo:y_hi, x_right + 1] > 0)
    ):
        x_right += 1
        steps += 1
    return x_left, x_right


def _span_at_row(
    valid_mask: np.ndarray,
    binary: np.ndarray,
    y_bar: int,
    half_band: int = _HALF_BAND_PX,
) -> Optional[Tuple[int, int, int]]:
    """
    Resolve a full-scale-bar candidate on one baseline.

    All horizontal runs within ``y_bar ± half_band`` are merged across a central
    label gap and the widest well-covered span is returned, already extended onto
    the outer vertical ticks.

    Returns:
        ``(span_px, x_left, x_right)`` or ``None`` when this baseline holds no bar.
    """
    height = int(valid_mask.shape[0])
    y_lo = max(0, y_bar - half_band)
    y_hi = min(height, y_bar + half_band + 1)

    runs = _band_runs(valid_mask, y_lo, y_hi)
    if not runs:
        return None

    best: Optional[Tuple[int, int]] = None
    for start, end in _merge_split_bar_runs(runs, binary, y_lo, y_hi):
        span = end - start + 1
        if span < 20:
            continue
        covered = sum(
            min(run_end, end) - max(run_start, start) + 1
            for run_start, run_end in runs
            if run_end >= start and run_start <= end
        )
        if covered / span < _BAR_COVERAGE_MIN:
            continue
        if best is None or span > best[1] - best[0] + 1:
            best = (start, end)

    if best is None:
        return None

    x_left, x_right = _expand_over_ticks(binary, y_lo, y_hi, best[0], best[1])
    return int(x_right - x_left + 1), int(x_left), int(x_right)


def _find_tick_span(
    binary: np.ndarray,
) -> Tuple[int, Optional[Tuple[int, int, int, int]], Optional[int]]:
    """
    Locate horizontal rays of a SEM scale bar and return tick-to-tick span.

    Strategy:
      1. Detect horizontal ray segments using morphology OPEN with a (15, 1) kernel.
      2. Discard only full-width components (table separators / column rules).
         Rays merely clipped by the ROI edge are kept, so a split databar whose
         left half starts at the border is not thrown away — that rejection used
         to report only the right half (e.g. 237 px instead of the full ~557 px).
      3. Rank candidate baselines ``y_bar`` by the number of horizontal ray pixels.
      4. On every candidate baseline (Y-row ±3 px) collect the horizontal runs and
         merge the two collinear halves flanking the central ``10 µm`` label.
      5. Keep the widest well-covered span and grow it onto the outer end ticks.
      6. Scale bar length = x_right - x_left + 1.

    Returns:
        (length_px, bbox_xywh_in_roi, y_bar)
    """
    if binary is None or binary.size == 0 or not np.any(binary):
        return 0, None, None

    h, w = binary.shape[:2]

    # 1. Horizontal ray segments
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

    # 2. Reject only full-width rules (table dividers / column separators); a bar
    #    half clipped by the ROI edge is a legitimate split-bar segment.
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    valid_mask = np.zeros_like(horiz)
    full_width_limit = int(0.9 * w)
    for i in range(1, n_lab):
        x, y, bw, bh, area = stats[i]
        if bw >= full_width_limit:
            continue
        valid_mask[lab == i] = 255

    # 3. Rank candidate baselines by horizontal-ray pixel count
    row_sums = np.sum(valid_mask > 0, axis=1)
    candidate_rows: List[int] = []
    for raw_row in np.argsort(row_sums)[::-1]:
        if row_sums[raw_row] <= 0:
            break
        row_candidate = int(raw_row)
        if all(abs(row_candidate - kept) > _HALF_BAND_PX for kept in candidate_rows):
            candidate_rows.append(row_candidate)
        if len(candidate_rows) >= 4:
            break
    if not candidate_rows:
        return 0, None, None

    # 4./5. Merge split halves on every candidate baseline, keep the widest span
    best: Optional[Tuple[int, int, int, int]] = None  # (span, x_left, x_right, y)
    for row_candidate in candidate_rows:
        resolved = _span_at_row(valid_mask, binary, row_candidate)
        if resolved is None:
            continue
        span, x_left, x_right = resolved
        if best is None or span > best[0]:
            best = (span, x_left, x_right, row_candidate)

    if best is None:
        return 0, None, None

    span, x_left, x_right, y_bar = best
    if span < 20:
        return 0, None, None

    y_min = max(0, y_bar - _HALF_BAND_PX)
    y_max = min(h, y_bar + _HALF_BAND_PX + 1)
    bbox = (int(x_left), int(y_min), int(span), int(y_max - y_min))
    return span, bbox, y_bar


def _recover_clipped_span(
    bbox_roi: Tuple[int, int, int, int],
    binary_full: np.ndarray,
    roi_x0: int,
    roi_x1: int,
    roi_y0: int,
    max_extra: int = 90,
) -> Tuple[int, int, int, int]:
    """
    Widen a bar bbox that the right-hand ROI window clipped on one of its edges.

    A split databar (``|---- 10 µm ----|``) whose outer left tick sits just left of
    the ROI window loses part of its left half at the border. Walking outward along
    the same baseline in the full image recovers that clipped part, so the measured
    span stays tick-to-tick instead of dropping to the right half only
    (the reported 237 px instead of ~557 px bug).

    Args:
        bbox_roi: ``(x, y, w, h)`` bar box in ROI coordinates.
        binary_full: Thresholded full-image mask matching the ROI threshold.
        roi_x0: ROI left edge in full-image pixels.
        roi_x1: ROI right edge in full-image pixels.
        roi_y0: ROI top row in full-image pixels.
        max_extra: Maximum number of pixels the span may grow per side.

    Returns:
        Tuple[int, int, int, int]: bbox in ROI coordinates (may hold negative ``x``).
    """
    x, y, bw, bh = bbox_roi
    height, width = binary_full.shape[:2]
    band_y0 = int(np.clip(roi_y0 + y, 0, height))
    band_y1 = int(np.clip(roi_y0 + y + bh, 0, height))
    if band_y1 <= band_y0:
        return bbox_roi

    left_extra = 0
    if x <= 2:
        probe = roi_x0
        steps = 0
        while (
            probe > 0
            and steps < max_extra
            and np.any(binary_full[band_y0:band_y1, probe - 1] > 0)
        ):
            probe -= 1
            steps += 1
        left_extra = roi_x0 - probe

    right_extra = 0
    if x + bw >= (roi_x1 - roi_x0) - 2:
        probe = roi_x1
        steps = 0
        while (
            probe < width - 1
            and steps < max_extra
            and np.any(binary_full[band_y0:band_y1, probe] > 0)
        ):
            probe += 1
            steps += 1
        right_extra = max(0, probe - roi_x1)

    if left_extra == 0 and right_extra == 0:
        return bbox_roi
    return (x - left_extra, y, bw + left_extra + right_extra, bh)


def detect_scale_bar(
    image: np.ndarray,
    footer_height: int = 70,
) -> Tuple[int, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Detect SEM scale-bar length from horizontal rays in the upper-right banner.

    ROI is restricted to:
      - X: right ~37% of the image (from width*0.63 to width-5) to safely cut off
        the metadata columns. A bar half clipped by that border is recovered
        afterwards by walking outward along the same baseline in the full image
        (``_recover_clipped_span``) — this is what keeps FEI / Thermo Fisher split
        databars (``|---- 10 µm ----|``) at their full tick-to-tick length instead
        of collapsing to the right half only.
      - Y: upper half of the info banner only
        [height - footer_height + 3, height - footer_height*0.48)

    This excludes the long horizontal table separator and the sample-name text
    in the lower banner floor.

    Returns:
        (length_px, annotated_preview_rgb_or_None, bbox_xywh_in_full_image_or_None)
    """
    img_u8 = _prepare_u8(image)
    if img_u8 is None:
        return 0, None, None

    height, width = img_u8.shape
    fh = int(max(10, min(footer_height, height - 5)))

    banner_top = height - fh
    roi_y0 = banner_top + 3
    roi_y1 = height - int(fh * 0.48)
    roi_x0 = int(width * 0.63)
    roi_x1 = max(roi_x0 + 10, width - 5)

    if roi_y1 <= roi_y0 + 4 or roi_x1 <= roi_x0 + 10:
        return 0, None, None

    roi = img_u8[roi_y0:roi_y1, roi_x0:roi_x1]
    if roi.size == 0:
        return 0, None, None

    # Grey scale-bar lines are typically ~110–180; threshold > 90
    threshold = 90
    binary = (roi > threshold).astype(np.uint8) * 255
    length, bbox_roi, _ = _find_tick_span(binary)

    # Milder threshold fallback for dark bars
    if length < 20:
        alt_binary = (roi > 70).astype(np.uint8) * 255
        alt_length, alt_bbox, _ = _find_tick_span(alt_binary)
        if alt_length > length:
            length, bbox_roi, threshold = alt_length, alt_bbox, 70

    if bbox_roi is None or length < 20:
        return 0, None, None

    # Recover a split-bar half clipped by the right-hand ROI window, then convert
    # the ROI-local box to full-image coordinates.
    binary_full = img_u8 > threshold
    x_roi, y_roi, bw, bh = _recover_clipped_span(
        bbox_roi, binary_full, roi_x0, roi_x1, roi_y0
    )
    length = int(bw)
    full_bbox = (int(roi_x0 + x_roi), int(roi_y0 + y_roi), int(bw), int(bh))

    # Tight QA preview around the measured bar: the framed crop guarantees that a
    # span recovered beyond the ROI border is still fully inside the annotation
    # and that both outer end ticks (tick-to-tick width, across the label gap)
    # carry a red mark.
    fx, fy, fw, fhb = full_bbox
    pad_x = 16
    pad_y = max(8, int(fhb))
    px0 = int(max(0, fx - pad_x))
    px1 = int(min(width, fx + fw + pad_x))
    py0 = int(max(0, fy - pad_y))
    py1 = int(min(height, fy + fhb + pad_y))
    preview = cv2.cvtColor(img_u8[py0:py1, px0:px1].copy(), cv2.COLOR_GRAY2RGB)

    preview_h, preview_w = preview.shape[:2]
    y_line = int(np.clip(fy + fhb // 2 - py0, 0, preview_h - 1))
    x0 = int(np.clip(fx - px0, 0, preview_w - 1))
    x1 = int(np.clip(fx + fw - 1 - px0, 0, preview_w - 1))

    # Semi-transparent thin green centerline spanning the full bar
    overlay = preview.copy()
    cv2.line(overlay, (x0, y_line), (x1, y_line), (0, 220, 80), 1, lineType=cv2.LINE_AA)
    preview = cv2.addWeighted(overlay, 0.65, preview, 0.35, 0)

    # Clear vertical end marks (red) at the outer tick positions
    tick_half = max(6, min(12, preview_h // 3))
    cv2.line(
        preview,
        (x0, max(0, y_line - tick_half)),
        (x0, min(preview_h - 1, y_line + tick_half)),
        (255, 50, 50),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        preview,
        (x1, max(0, y_line - tick_half)),
        (x1, min(preview_h - 1, y_line + tick_half)),
        (255, 50, 50),
        2,
        lineType=cv2.LINE_AA,
    )

    return int(length), preview, full_bbox


def detect_scale_bar_pixels(image: np.ndarray, footer_height: int = 70) -> int:
    """
    Detect the length of the SEM scale bar in pixels (tick-to-tick).

    Args:
        image: 2D or 3D grayscale SEM image.
        footer_height: Info-banner height in pixels from the bottom edge.

    Returns:
        int: Scale-bar length in pixels, or 0 if not found.
    """
    length, _, _ = detect_scale_bar(image, footer_height=footer_height)
    return length


if __name__ == "__main__":
    print("Testing scale_detector.py on synthetic SEM banner...")
    # Image 500x600, footer 80 px; upper banner floor holds the scale bar
    canvas = np.full((500, 600), 100, dtype=np.uint8)
    canvas[420:, :] = 8  # black databar
    # Table separator at mid-banner (must NOT be detected) — full width, 655-like
    canvas[458:460, :] = 160
    # Sample name in lower floor
    canvas[470:490, 340:580] = 40
    # Real scale bar in UPPER half of banner (right section in 0.58*600 = 348..595):
    # Left tick at 400, left ray 402..460, text gap 460..500, right ray 500..560, right tick at 561
    canvas[432:448, 400:402] = 170  # left tick
    canvas[438:441, 402:460] = 150  # left ray
    canvas[438:441, 500:560] = 150  # right ray
    canvas[432:448, 559:561] = 170  # right tick

    length, preview, bbox = detect_scale_bar(canvas, footer_height=80)
    print(f"Detected line length: {length} px (Expected: ~160 px)")
    assert 150 <= length <= 170, f"Expected ~160 px tick span, got {length}"
    assert preview is not None
    print("Synthetic tick-span test passed!")

    # --- Split databar ``|---- 10 µm ----|`` (v1.7.4 regression) --------------
    # The label sits INSIDE the bar, so the ray is drawn as two collinear halves.
    split = np.full((500, 600), 100, dtype=np.uint8)
    split[420:, :] = 8
    split[458:460, :] = 160  # full-width separator (must stay rejected)
    split[432:448, 400:402] = 170  # outer left tick
    split[438:441, 402:470] = 150  # left ray half
    split[438:441, 530:590] = 150  # right ray half
    split[432:448, 590:592] = 170  # outer right tick
    cv2.putText(
        split, "10 um", (480, 446), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 210, 1, cv2.LINE_AA
    )

    split_len, split_preview, split_bbox = detect_scale_bar(split, footer_height=80)
    print(f"Split-bar span: {split_len} px (Expected: ~193 px full tick-to-tick)")
    assert 185 <= split_len <= 200, f"Split bar measured as a half: {split_len} px"
    assert split_bbox is not None and split_bbox[2] >= 185, (
        f"Preview bbox does not span the label gap: {split_bbox}"
    )
    # The red end marks must sit on the outermost ticks, i.e. across the text gap
    mid_row = split_preview[split_preview.shape[0] // 2]
    red_cols = np.where((mid_row[:, 0] > 140) & (mid_row[:, 1] < 140))[0]
    assert red_cols.size >= 2, "Green centerline / red end marks missing"
    assert int(red_cols[-1] - red_cols[0]) >= 185, (
        f"Preview annotation does not span the full bar: {int(red_cols[-1] - red_cols[0])} px"
    )
    print("Split databar merge test passed!")

    # --- Split databar clipped by the ROI edge ------------------------------
    # Left half starts before the ROI window; it must not be discarded.
    clipped = np.full((500, 600), 100, dtype=np.uint8)
    clipped[420:, :] = 8
    clipped[458:460, :] = 160
    clipped[432:448, 336:338] = 170  # left tick outside the ROI (x0 = 348)
    clipped[438:441, 338:470] = 150  # clipped left ray half
    clipped[438:441, 530:590] = 150  # right ray half
    clipped[432:448, 590:592] = 170  # right tick
    cv2.putText(
        clipped, "10 um", (480, 446), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 210, 1, cv2.LINE_AA
    )

    clipped_len = detect_scale_bar_pixels(clipped, footer_height=80)
    print(f"Clipped split-bar span: {clipped_len} px (Expected: >235 px from ROI edge)")
    assert clipped_len >= 235, f"Clipped half discarded: {clipped_len} px"
    print("Clipped split-bar test passed!")

    # Ensure full-width separator alone is not measured as the scale bar
    canvas_sep = np.full((500, 600), 100, dtype=np.uint8)
    canvas_sep[420:, :] = 8
    canvas_sep[458:460, 330:595] = 180  # only separator in mid/lower zone
    length_sep = detect_scale_bar_pixels(canvas_sep, footer_height=80)
    print(f"Separator-only detection: {length_sep} px (Expected: 0)")
    assert length_sep < 40, f"Separator should not dominate, got {length_sep}"
    print("Separator rejection test passed!")
