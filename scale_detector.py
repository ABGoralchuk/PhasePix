"""
Module for automated detection and measurement of SEM scale bars.
Extracts calibration metrics from the upper-right databar region only,
avoiding the horizontal table separator that spans the info banner.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


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


def _find_tick_span(
    binary: np.ndarray,
) -> Tuple[int, Optional[Tuple[int, int, int, int]], Optional[int]]:
    """
    Locate horizontal rays of a SEM scale bar and return tick-to-tick span.

    Strategy:
      1. Detect horizontal ray segments using morphology OPEN with a (15, 1) kernel.
      2. Discard any component touching ROI borders (x <= 5 or x + bw >= w - 5).
      3. Find row y_bar where the sum of horizontal ray pixels is maximized.
      4. Take all horizontal rays at height y_bar +- 3 px (left and right rays around text).
      5. Left boundary x_left = leftmost pixel of left ray (adjusted 1-2 px for adjacent vertical tick).
         Right boundary x_right = rightmost pixel of right ray.
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

    # 2. Reject components touching ROI left/right borders
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    valid_mask = np.zeros_like(horiz)
    for i in range(1, n_lab):
        x, y, bw, bh, area = stats[i]
        if x <= 5 or (x + bw) >= w - 5:
            continue
        valid_mask[lab == i] = 255

    # 3. Find row y_bar where sum of horizontal ray pixels is maximized
    row_sums = np.sum(valid_mask > 0, axis=1)
    if not np.any(row_sums > 0):
        return 0, None, None

    y_bar = int(np.argmax(row_sums))

    # 4. Take all horizontal rays at height y_bar +- 3 px
    y_min = max(0, y_bar - 3)
    y_max = min(h, y_bar + 4)
    rays = valid_mask[y_min:y_max, :]
    col_has = np.any(rays > 0, axis=0)
    cols = np.where(col_has)[0]

    if len(cols) < 10:
        return 0, None, None

    x_left = int(cols[0])
    x_right = int(cols[-1])

    # 5. Expand 1-2 px outward if vertical tick pixels sit immediately adjacent in raw binary
    while x_left > 0 and np.any(binary[y_min:y_max, x_left - 1] > 0) and (x_left >= cols[0] - 2):
        x_left -= 1
    while x_right < w - 1 and np.any(binary[y_min:y_max, x_right + 1] > 0) and (x_right <= cols[-1] + 2):
        x_right += 1

    span = int(x_right - x_left + 1)
    if span < 20:
        return 0, None, None

    bbox = (x_left, y_min, span, y_max - y_min)
    return span, bbox, y_bar


def detect_scale_bar(
    image: np.ndarray,
    footer_height: int = 70,
) -> Tuple[int, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Detect SEM scale-bar length from horizontal rays in the upper-right banner.

    ROI is restricted to:
      - X: right ~37% of the image (from width*0.63 to width-5) to safely cut off the det column.
      - Y: upper half of the info banner only
        [height - footer_height + 3, height - footer_height*0.48)

    This excludes the long horizontal table separator and the sample-name text
    in the lower banner floor.

    Returns:
        (length_px, annotated_roi_preview_rgb_or_None, bbox_xywh_in_full_image_or_None)
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
    binary = (roi > 90).astype(np.uint8) * 255
    length, bbox_roi, y_mid = _find_tick_span(binary)

    # Milder threshold fallback for dark bars
    if length < 20:
        binary2 = (roi > 70).astype(np.uint8) * 255
        length2, bbox2, y_mid2 = _find_tick_span(binary2)
        if length2 > length:
            length, bbox_roi, y_mid = length2, bbox2, y_mid2

    preview = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    full_bbox = None

    if bbox_roi is not None and length > 0:
        x, y, bw, bh = bbox_roi
        y_line = int(y_mid) if y_mid is not None else y + bh // 2
        x0, x1 = int(x), int(x + bw - 1)

        # Semi-transparent thin green centerline
        overlay = preview.copy()
        cv2.line(overlay, (x0, y_line), (x1, y_line), (0, 220, 80), 1, lineType=cv2.LINE_AA)
        preview = cv2.addWeighted(overlay, 0.65, preview, 0.35, 0)

        # Clear vertical end marks (red) at outer tick positions
        tick_half = max(6, min(12, preview.shape[0] // 3))
        cv2.line(
            preview,
            (x0, max(0, y_line - tick_half)),
            (x0, min(preview.shape[0] - 1, y_line + tick_half)),
            (255, 50, 50),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            preview,
            (x1, max(0, y_line - tick_half)),
            (x1, min(preview.shape[0] - 1, y_line + tick_half)),
            (255, 50, 50),
            2,
            lineType=cv2.LINE_AA,
        )

        full_bbox = (roi_x0 + x0, roi_y0 + y, bw, bh)

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
    # Real scale bar in UPPER half of banner (right section in 0.63*600 = 378..595):
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

    # Ensure full-width separator alone is not measured as the scale bar
    canvas_sep = np.full((500, 600), 100, dtype=np.uint8)
    canvas_sep[420:, :] = 8
    canvas_sep[458:460, 330:595] = 180  # only separator in mid/lower zone
    length_sep = detect_scale_bar_pixels(canvas_sep, footer_height=80)
    print(f"Separator-only detection: {length_sep} px (Expected: 0)")
    assert length_sep < 40, f"Separator should not dominate, got {length_sep}"
    print("Separator rejection test passed!")
