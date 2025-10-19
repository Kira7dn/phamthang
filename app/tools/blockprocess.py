"""
Frame Detection Module - Simplified approach to detect aluminum frames.

Strategy:
1. Use Canny edge detection
2. Apply Hough Lines to find frame edges
3. Detect closed rectangles using contours on morphologically processed edges
4. Combine results for accurate frame dimensions
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import logging
import math
from itertools import combinations
import cv2
import numpy as np

from app.models import BoundingRect


# Constants for frame detection
MIN_FRAME_PERCENTAGE = 0.015  # 1.5% of image area (lowered to catch smaller frames)
MIN_FRAME_PIXELS = 15000  # Absolute minimum size (lowered from 20000)
MIN_DIMENSION = 40  # Minimum width or height
MAX_ASPECT_RATIO = 15.0  # Maximum elongation
MIN_FILL_RATIO = 0.7  # Minimum fill ratio for closed contours
MIN_CHILDREN_FOR_BOUNDARY = 2  # Parent with 2+ children is boundary

# Constants for line detection and morphology
SEGMENT_GAP_TOLERANCE = 30
SEGMENT_POSITION_TOLERANCE = 5
MORPH_KERNEL_SIZE = (5, 5)
TEXT_MAX_SIZE = 120
TEXT_MAX_AREA = 10000
MAX_CORNER_ENDPOINT_DISTANCE = 30

logger = logging.getLogger(__name__)


def draw_rectangles(
    image: np.ndarray,
    rects: Sequence[BoundingRect],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    labels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Draw rectangles on image with labels."""
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    for i, rect in enumerate(rects):
        # Support both dict and object
        if isinstance(rect, dict):
            x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
        else:
            x, y, w, h = rect.x, rect.y, rect.w, rect.h
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        # Draw label
        label: Optional[str] = None
        if labels is not None and i < len(labels):
            label = labels[i]
        if not label:
            label = f"Frame {i+1}: {w}x{h}"
        cv2.putText(
            annotated,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def apply_otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Apply OTSU threshold."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def remove_text_regions(binary: np.ndarray) -> np.ndarray:
    """Remove text/number regions (dimension annotations) before line detection.

    Strategy:
    - Remove margin regions (dimension annotations are typically outside main area)
    - Detect and remove small dense components (text)
    - Use morphology to clean up
    """
    logger = logging.getLogger("remove_text_regions")
    h, w = binary.shape
    cleaned = binary.copy()

    # Step 1: Remove outer margin regions (where dimension text/arrows typically are)
    # Find the main content area by detecting largest solid regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find contours to locate main content
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find bounding box of all main contours
        all_points = np.vstack(contours)
        x_min, y_min = all_points.min(axis=0)[0]
        x_max, y_max = all_points.max(axis=0)[0]

        # Add margin to remove dimension text that's close to edges
        margin = 0  # Reduced to avoid removing important frame parts
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        # Clear regions outside main area (including dimension text)
        # CRITICAL: ONLY remove LEFT/RIGHT margins to preserve horizontal frame edges
        # Do NOT remove top/bottom margins as they may contain frame edges
        # cleaned[:y_min, :] = 0  # Top margin - DISABLED to preserve top edges
        # cleaned[y_max:, :] = 0  # Bottom margin - DISABLED to preserve bottom edges
        cleaned[:, :x_min] = 0  # Left margin
        cleaned[:, x_max:] = 0  # Right margin

    # Step 2: Remove small isolated text components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    for i in range(1, num_labels):
        x, y, w_comp, h_comp, area = stats[i]

        # Remove small compact regions (likely text)
        # BUT preserve horizontal/vertical lines (high aspect ratio)
        is_small = w_comp < TEXT_MAX_SIZE and h_comp < TEXT_MAX_SIZE
        is_compact = area < TEXT_MAX_AREA

        # Calculate aspect ratio
        aspect_ratio = max(w_comp, h_comp) / max(1, min(w_comp, h_comp))

        # CRITICAL: Only remove if aspect ratio is LOW (text-like)
        # High aspect ratio means it's a line (horizontal or vertical edge of frame)
        is_text_like = aspect_ratio < 5  # Text has low aspect ratio

        if is_small and is_compact and is_text_like:
            cleaned[labels == i] = 0

    return cleaned


def connect_lines(binary: np.ndarray) -> np.ndarray:
    """Connect broken lines by detecting and extending line segments.

    Strategy:
    - Use Hough Lines to detect all line segments
    - Group nearby collinear segments
    - Extend and connect segments in the same group
    - Redraw connected lines
    """
    logger = logging.getLogger("connect_lines")
    h, w = binary.shape

    # Detect edges
    edges = cv2.Canny(binary, 30, 100, apertureSize=3)

    # Detect line segments with relaxed parameters to catch broken pieces
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,  # Lower to detect weak segments
        minLineLength=50,  # Shorter to catch small pieces
        maxLineGap=20,  # Allow gaps
    )

    if lines is None:
        return binary

    # Separate horizontal and vertical lines
    h_segments = []
    v_segments = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        if angle < 15 or angle > 165:  # Horizontal
            h_segments.append((min(x1, x2), y1, max(x1, x2), y2))
        elif 75 < angle < 105:  # Vertical
            v_segments.append((x1, min(y1, y2), x2, max(y1, y2)))

    # Connect nearby collinear segments
    def connect_horizontal_segments(
        segments,
        gap_tolerance=SEGMENT_GAP_TOLERANCE,
        y_tolerance=SEGMENT_POSITION_TOLERANCE,
    ):
        """Connect horizontal segments that are on the same line."""
        if not segments:
            return []

        # Sort by y, then x
        segments = sorted(segments, key=lambda s: (s[1], s[0]))
        connected = []

        for seg in segments:
            x1, y1, x2, y2 = seg
            merged = False

            for i, conn in enumerate(connected):
                cx1, cy1, cx2, cy2 = conn

                # Check if on same horizontal line (y-coordinate similar)
                if abs(y1 - cy1) <= y_tolerance:
                    # Check if segments are close enough to connect
                    if cx1 - gap_tolerance <= x1 <= cx2 + gap_tolerance:
                        # Extend the connected segment
                        connected[i] = (min(cx1, x1), cy1, max(cx2, x2), cy2)
                        merged = True
                        break

            if not merged:
                connected.append(seg)

        return connected

    def connect_vertical_segments(
        segments,
        gap_tolerance=SEGMENT_GAP_TOLERANCE,
        x_tolerance=SEGMENT_POSITION_TOLERANCE,
    ):
        """Connect vertical segments that are on the same line."""
        if not segments:
            return []

        # Sort by x, then y
        segments = sorted(segments, key=lambda s: (s[0], s[1]))
        connected = []

        for seg in segments:
            x1, y1, x2, y2 = seg
            merged = False

            for i, conn in enumerate(connected):
                cx1, cy1, cx2, cy2 = conn

                # Check if on same vertical line (x-coordinate similar)
                if abs(x1 - cx1) <= x_tolerance:
                    # Check if segments are close enough to connect
                    if cy1 - gap_tolerance <= y1 <= cy2 + gap_tolerance:
                        # Extend the connected segment
                        connected[i] = (cx1, min(cy1, y1), cx2, max(cy2, y2))
                        merged = True
                        break

            if not merged:
                connected.append(seg)

        return connected

    # Connect segments - 2 iterations sufficient for most cases
    for _ in range(2):
        h_segments = connect_horizontal_segments(h_segments)
        v_segments = connect_vertical_segments(v_segments)

    # Create mask with connected lines
    line_mask = np.zeros_like(binary)

    # Draw connected horizontal lines
    for x1, y1, x2, y2 in h_segments:
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    # Draw connected vertical lines
    for x1, y1, x2, y2 in v_segments:
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    # Slight thickening to ensure continuity
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    line_mask = cv2.dilate(line_mask, kernel, iterations=1)

    # CRITICAL: Combine with original to preserve corners and details
    result = cv2.bitwise_or(binary, line_mask)

    return result


def restore_missing_corners(binary: np.ndarray) -> np.ndarray:
    """Khôi phục các góc bị thiếu bằng cách kéo dài các đoạn thẳng gần nhau."""
    logger = logging.getLogger("restore_missing_corners")
    h, w = binary.shape

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=20,
    )

    if lines is None:
        return binary

    def compute_intersection(
        line1: np.ndarray, line2: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        x1, y1, x2, y2 = map(float, line1)
        x3, y3, x4, y4 = map(float, line2)

        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1

        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3

        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None

        px = (b2 * c1 - b1 * c2) / det
        py = (a1 * c2 - a2 * c1) / det
        return int(round(px)), int(round(py))

    def endpoint_distance(line: np.ndarray, point: Tuple[int, int]) -> float:
        x1, y1, x2, y2 = line
        dist_start = math.hypot(point[0] - x1, point[1] - y1)
        dist_end = math.hypot(point[0] - x2, point[1] - y2)
        return min(dist_start, dist_end)

    line_mask = np.zeros_like(binary)
    restored = 0
    seen_corners: Set[Tuple[int, int]] = set()

    for line1, line2 in combinations(lines, 2):
        intersection = compute_intersection(line1[0], line2[0])
        if intersection is None:
            continue

        ix, iy = intersection
        if ix < -5 or iy < -5 or ix > w + 5 or iy > h + 5:
            continue

        v1 = np.array(
            [line1[0][2] - line1[0][0], line1[0][3] - line1[0][1]], dtype=float
        )
        v2 = np.array(
            [line2[0][2] - line2[0][0], line2[0][3] - line2[0][1]], dtype=float
        )

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        if not (70 < angle < 110):
            continue

        if endpoint_distance(line1[0], (ix, iy)) > MAX_CORNER_ENDPOINT_DISTANCE:
            continue
        if endpoint_distance(line2[0], (ix, iy)) > MAX_CORNER_ENDPOINT_DISTANCE:
            continue

        corner = (ix, iy)
        if corner in seen_corners:
            continue

        cv2.line(line_mask, (line1[0][0], line1[0][1]), corner, 255, 3)
        cv2.line(line_mask, (line2[0][0], line2[0][1]), corner, 255, 3)
        seen_corners.add(corner)
        restored += 1

    if restored == 0:
        return binary

    logger.debug(f"Khôi phục {restored} góc bị thiếu")
    return cv2.bitwise_or(binary, line_mask)


def is_axis_aligned(angle: float) -> bool:
    """Check if line angle is horizontal or vertical."""
    return angle < 15 or angle > 165 or (75 < angle < 105)


def remove_diagonal_lines(binary: np.ndarray) -> np.ndarray:
    """Remove diagonal lines (dimension annotations) by keeping only horizontal/vertical structures.

    Strategy:
    - Use Hough Lines to detect and classify lines by angle
    - Remove diagonal lines by painting them black
    """
    logger = logging.getLogger("remove_diagonal_lines")
    # Detect all lines using Hough
    edges = cv2.Canny(binary, 30, 100, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=20,
        maxLineGap=10,
    )

    cleaned = binary.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Remove only diagonal lines by painting them black
            if not is_axis_aligned(angle):
                cv2.line(cleaned, (x1, y1), (x2, y2), 0, 4)
    return cleaned


def remove_fold_lines(binary: np.ndarray, border_margin: int = 10) -> np.ndarray:
    """Remove vertical/horizontal lines near image borders.

    Strategy:
    - Detect all lines using Hough
    - Find vertical lines within border_margin from left/right edges
    - Find horizontal lines within border_margin from top/bottom edges
    - Paint the ENTIRE line black (not just border region)
    """
    logger = logging.getLogger("remove_fold_lines")
    h, w = binary.shape
    cleaned = binary.copy()

    # Detect all lines
    edges = cv2.Canny(binary, 30, 100, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=50,  # Catch shorter lines too
        maxLineGap=10,
    )

    if lines is None:
        return cleaned

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Check if ANY endpoint touches border
        touches_border = (
            # Point 1 (x1, y1) touches any border
            x1 <= border_margin
            or x1 >= w - border_margin
            or y1 <= border_margin
            or y1 >= h - border_margin
            or
            # Point 2 (x2, y2) touches any border
            x2 <= border_margin
            or x2 >= w - border_margin
            or y2 <= border_margin
            or y2 >= h - border_margin
        )

        if touches_border:
            # Paint the ENTIRE line black
            cv2.line(cleaned, (x1, y1), (x2, y2), 0, 10)

    return cleaned


def reconnect_broken_frames(binary: np.ndarray) -> np.ndarray:
    """Aggressively reconnect broken frame edges after diagonal removal.

    This is stronger than connect_lines and specifically targets
    gaps created by diagonal line removal and text removal.
    """
    logger = logging.getLogger("app.tools.reconnect_broken_frames")
    h, w = binary.shape

    # Step 1: Strong horizontal closing to connect horizontal breaks
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    horizontal_closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, h_kernel, iterations=2
    )

    # Step 2: Strong vertical closing to connect vertical breaks
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vertical_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, v_kernel, iterations=2)

    # Step 3: Combine both
    combined = cv2.bitwise_or(horizontal_closed, vertical_closed)

    # Step 4: CRITICAL - Close corner gaps with larger square kernel
    # This specifically targets broken corners from text removal
    kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    reconnected = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, kernel_square, iterations=6
    )

    # Step 5: Dilate to make lines thicker and more continuous
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    reconnected = cv2.dilate(reconnected, kernel_dilate, iterations=2)

    # Step 6: Final corner filling - one more close to ensure corners are solid
    kernel_corner = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    reconnected = cv2.morphologyEx(
        reconnected, cv2.MORPH_CLOSE, kernel_corner, iterations=2
    )

    return reconnected


def detect_frames_by_contours(
    cleaned_binary: np.ndarray,
    max_aspect: float = MAX_ASPECT_RATIO,
    min_frame_percentage: float = MIN_FRAME_PERCENTAGE,
    min_frame_pixels: int = MIN_FRAME_PIXELS,
) -> List[Dict[str, Any]]:
    """Detect frame rectangles using contours.

    Strategy: Use bounding rectangles instead of contour area to avoid
    issues with internal annotations (dimension lines, diagonal lines, etc.)
    Uses hierarchy to filter out parent contours that only serve as boundaries.
    """
    logger = logging.getLogger("detect_frames_by_contours")

    h, w = cleaned_binary.shape
    image_area = w * h
    min_frame_area = max(image_area * min_frame_percentage, min_frame_pixels)

    # Find all contours with hierarchy
    # Hierarchy format: [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(
        cleaned_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    logger.debug(
        f"Found {len(contours)} total contours (image: {w}x{h}, min_area: {min_frame_area:.0f})"
    )

    # Get all contour areas for debugging
    all_areas = [cv2.contourArea(cnt) for cnt in contours]
    all_areas.sort(reverse=True)
    if all_areas:
        logger.debug(f"  Top 5 contour areas: {all_areas[:5]}")
        logger.debug(f"  Area range: {all_areas[0]:.0f} -> {all_areas[-1]:.0f}")

    # logger.debug ALL contours with bounding box info for debugging
    for idx, cnt in enumerate(contours):
        x, y, rect_w, rect_h = cv2.boundingRect(cnt)
        bbox_area = rect_w * rect_h
        contour_area = cv2.contourArea(cnt)
        fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0

    frames = []
    filtered_stats = {
        "too_small": 0,
        "bad_aspect": 0,
        "bad_fill": 0,
        "parent_boundary": 0,
    }
    debug_info = []

    # Flatten hierarchy for easier access
    if hierarchy is not None:
        hierarchy = hierarchy[0]

    # ============================================================
    # PRIORITY 1-5: Filter obvious invalid contours FIRST
    # ============================================================
    # These are definite filters - no ambiguity
    for idx, cnt in enumerate(contours):
        x, y, rect_w, rect_h = cv2.boundingRect(cnt)
        bbox_area = rect_w * rect_h

        # 1. Filter out very small frames (< 2% image or < 20k pixels)
        if bbox_area < min_frame_area:
            filtered_stats["too_small"] += 1
            if bbox_area > min_frame_area / 10:
                debug_info.append(
                    f"⚠ Filtered (too small): bbox={rect_w}x{rect_h}, area={bbox_area:.0f} < {min_frame_area:.0f}"
                )
            continue

        # 2. Filter out frames that are too large (> 90% of image in BOTH dimensions)
        if rect_w > w * 0.9 and rect_h > h * 0.9:
            filtered_stats["bad_aspect"] += 1
            debug_info.append(f"⚠ Filtered (too large): bbox={rect_w}x{rect_h}")
            continue

        # 3. Filter out very thin frames (likely noise or dimension lines)
        if rect_w < MIN_DIMENSION or rect_h < MIN_DIMENSION:
            filtered_stats["bad_aspect"] += 1
            debug_info.append(f"⚠ Filtered (too thin): bbox={rect_w}x{rect_h}")
            continue

        # 4. Filter by aspect ratio (not too elongated)
        aspect = max(rect_w, rect_h) / max(1, min(rect_w, rect_h))
        if aspect > max_aspect:
            filtered_stats["bad_aspect"] += 1
            debug_info.append(
                f"⚠ Filtered (aspect={aspect:.1f}): bbox={rect_w}x{rect_h}"
            )
            continue

        # 5. Validate that contour is actually closed (not just 3 sides)
        contour_area = cv2.contourArea(cnt)
        fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0
        if fill_ratio < MIN_FILL_RATIO:
            filtered_stats["bad_fill"] += 1
            debug_info.append(
                f"⚠ Filtered (open contour): bbox={rect_w}x{rect_h}, fill_ratio={fill_ratio:.2f}"
            )
            continue

        # Passed all basic filters - add to candidates
        frames.append(
            {
                "idx": idx,  # Store original index for hierarchy lookup
                "x": x,
                "y": y,
                "w": rect_w,
                "h": rect_h,
                "area": float(bbox_area),
                "aspect": float(aspect),
                "fill_ratio": float(fill_ratio),
            }
        )

    # ============================================================
    # PRIORITY 6: Parent-Child relationship filtering
    # ============================================================
    # Now check hierarchy relationships among valid candidates
    # Rule: If sum(children) >= 30% parent -> remove parent
    #       If sum(children) < 30% parent -> remove children

    # Build index map for quick lookup
    idx_to_frame = {frame["idx"]: i for i, frame in enumerate(frames)}
    frames_to_remove = set()

    for i, frame in enumerate(frames):
        if i in frames_to_remove:
            continue

        idx = frame["idx"]
        # Check if this frame has children in hierarchy
        if hierarchy is not None and hierarchy[idx][2] != -1:
            # Calculate total area of children that passed filters
            child_idx = hierarchy[idx][2]
            children_indices = []
            total_children_area = 0

            while child_idx != -1:
                if child_idx in idx_to_frame:
                    child_frame_idx = idx_to_frame[child_idx]
                    children_indices.append(child_frame_idx)
                    total_children_area += frames[child_frame_idx]["area"]
                child_idx = hierarchy[child_idx][0]  # Next sibling

            if children_indices:
                parent_area = frame["area"]
                children_ratio = (
                    total_children_area / parent_area if parent_area > 0 else 0
                )

                logger.debug(
                    f"Parent-child check: parent {frame['w']}x{frame['h']}, "
                    f"{len(children_indices)} children, ratio={children_ratio:.1%}"
                )

                if children_ratio >= 0.30:
                    # Children occupy >= 30% -> remove parent (it's a boundary)
                    frames_to_remove.add(i)
                    filtered_stats["parent_boundary"] += 1
                    debug_info.append(
                        f"⚠ Filtered (parent boundary with {len(children_indices)} children, {children_ratio:.1%} filled): "
                        f"bbox={frame['w']}x{frame['h']}"
                    )
                    logger.debug("  → Removing parent (boundary)")
                else:
                    # Children occupy < 30% -> remove children (they're noise)
                    for child_idx in children_indices:
                        frames_to_remove.add(child_idx)
                        child = frames[child_idx]
                        logger.debug(
                            f"  → Removing child noise: {child['w']}x{child['h']} "
                            f"({child['area']/parent_area:.1%} of parent)"
                        )

    # Remove marked frames
    frames = [f for i, f in enumerate(frames) if i not in frames_to_remove]

    # ============================================================
    # PRIORITY 7: Spatial containment check (fallback)
    # ============================================================
    # Some noise frames may not be in hierarchy but are spatially inside
    # Apply same parent-child logic: >= 30% remove parent, < 30% remove children

    # Build spatial parent-child relationships
    spatial_parents = {}  # parent_idx -> [child_indices]
    for i, frame in enumerate(frames):
        x1, y1, w1, h1 = frame["x"], frame["y"], frame["w"], frame["h"]

        for j, other in enumerate(frames):
            if i == j:
                continue
            x2, y2, w2, h2 = other["x"], other["y"], other["w"], other["h"]

            # Check if frame is completely inside other (with small margin)
            margin = 5
            if (
                x1 >= x2 - margin
                and y1 >= y2 - margin
                and x1 + w1 <= x2 + w2 + margin
                and y1 + h1 <= y2 + h2 + margin
            ):
                # Frame i is spatially inside frame j (j is parent of i)
                if j not in spatial_parents:
                    spatial_parents[j] = []
                spatial_parents[j].append(i)
                break  # Only one parent per child

    # Apply parent-child logic
    frames_to_remove_spatial = set()
    for parent_idx, children_indices in spatial_parents.items():
        parent = frames[parent_idx]
        parent_area = parent["area"]
        total_children_area = sum(frames[c]["area"] for c in children_indices)
        children_ratio = total_children_area / parent_area if parent_area > 0 else 0

        logger.debug(
            f"Spatial parent-child: parent {parent['w']}x{parent['h']}, "
            f"{len(children_indices)} children, ratio={children_ratio:.1%}"
        )

        if children_ratio >= 0.30:
            # Remove parent (it's a boundary)
            frames_to_remove_spatial.add(parent_idx)
            logger.debug("  → Removing spatial parent (boundary)")
        else:
            # Remove children (they're noise)
            for child_idx in children_indices:
                frames_to_remove_spatial.add(child_idx)
                child = frames[child_idx]
                logger.debug(
                    f"  → Removing spatial child noise: {child['w']}x{child['h']} "
                    f"({child['area']/parent_area:.1%} of parent)"
                )

    # Remove spatially contained noise frames
    frames = [f for i, f in enumerate(frames) if i not in frames_to_remove_spatial]

    # ============================================================
    # PRIORITY 8: Merge or Remove overlapping frames
    # ============================================================
    # Strategy:
    # - If frames overlap significantly and are similar in size -> MERGE them
    # - If one frame is much smaller and inside/overlapping -> REMOVE smaller one
    # This handles both broken frame detection and corner artifacts

    def calculate_iou(frame1, frame2):
        """Calculate Intersection over Union (IoU) between two frames."""
        x1, y1, w1, h1 = frame1["x"], frame1["y"], frame1["w"], frame1["h"]
        x2, y2, w2, h2 = frame2["x"], frame2["y"], frame2["w"], frame2["h"]

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0, 0.0, 0.0  # No overlap - return 3 values

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0.0

        # Also calculate overlap ratio relative to smaller frame
        smaller_area = min(area1, area2)
        overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0.0

        return iou, overlap_ratio, intersection_area

    def merge_frames(frame1, frame2):
        """Merge two overlapping frames into a single bounding box."""
        x1, y1, w1, h1 = frame1["x"], frame1["y"], frame1["w"], frame1["h"]
        x2, y2, w2, h2 = frame2["x"], frame2["y"], frame2["w"], frame2["h"]

        # Calculate bounding box that contains both frames
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2)
        y_max = max(y1 + h1, y2 + h2)

        merged_w = x_max - x_min
        merged_h = y_max - y_min
        merged_area = merged_w * merged_h

        return {
            "x": x_min,
            "y": y_min,
            "w": merged_w,
            "h": merged_h,
            "area": float(merged_area),
            "aspect": float(max(merged_w, merged_h) / max(1, min(merged_w, merged_h))),
            "fill_ratio": (frame1.get("fill_ratio", 0) + frame2.get("fill_ratio", 0))
            / 2,
            "merged": True,  # Mark as merged
        }

    # Sort by area (largest first)
    frames.sort(key=lambda r: r["area"], reverse=True)

    # Apply smart overlap handling
    frames_to_keep = []
    frames_merged = 0
    frames_removed_by_overlap = 0
    processed = set()  # Track which frames have been processed

    for i, frame in enumerate(frames):
        if i in processed:
            continue

        # Check against remaining unprocessed frames
        merged_with_any = False

        for j in range(i + 1, len(frames)):
            if j in processed:
                continue

            other = frames[j]
            iou, overlap_ratio, intersection = calculate_iou(frame, other)

            # Decision logic:
            # MERGE if overlap > 80% of the smaller block
            # Otherwise REMOVE the smaller block

            # overlap_ratio is already calculated as intersection / smaller_area
            if overlap_ratio > 0.80:  # Overlap > 80% of smaller block
                # MERGE: The two frames are likely parts of the same frame
                merged_frame = merge_frames(frame, other)
                logger.debug(
                    f"Merging frames: {frame['w']}x{frame['h']} + {other['w']}x{other['h']} "
                    f"-> {merged_frame['w']}x{merged_frame['h']} (overlap={overlap_ratio:.1%} of smaller)"
                )
                logger.debug(
                    f"  🔗 Merged overlapping frames: {frame['w']}x{frame['h']} + {other['w']}x{other['h']} "
                    f"-> {merged_frame['w']}x{merged_frame['h']} (overlap={overlap_ratio:.1%} of smaller block)"
                )
                frame = merged_frame  # Update current frame to merged one
                processed.add(j)
                frames_merged += 1
                merged_with_any = True
                # Continue checking for more frames to merge

            # elif overlap_ratio > 0.0:  # Has overlap but < 80%
            #     # REMOVE smaller frame (it's likely noise or artifact)
            #     logger.debug(
            #         f"  ⚠ Removed overlapping frame: {other['w']}x{other['h']}px "
            #         f"(overlap={overlap_ratio:.1%} of smaller block, threshold=80%)"
            #     )
            #     processed.add(j)
            #     frames_removed_by_overlap += 1

        frames_to_keep.append(frame)
        processed.add(i)

    frames = frames_to_keep

    # Clean up idx field
    for frame in frames:
        frame.pop("idx", None)  # Remove idx, not needed by caller

    logger.debug(
        f"⚠ Filtered: {filtered_stats['too_small']} too small, "
        f"{filtered_stats['bad_aspect']} bad aspect, "
        f"{filtered_stats['bad_fill']} bad fill ratio, "
        f"{filtered_stats['parent_boundary']} parent boundaries"
    )
    if frames_merged > 0:
        logger.debug(f"  Merged: {frames_merged} frames combined")
    if frames_removed_by_overlap > 0:
        logger.debug(f"  Removed: {frames_removed_by_overlap} overlapping frames")

    if debug_info:
        logger.debug(f"  Debug - ALL {len(debug_info)} significant filtered contours:")
        for info in debug_info:
            logger.debug(info)
    logger.debug(f"Found {len(frames)} potential frames")
    for i, frame in enumerate(frames):
        logger.debug(
            f"  Frame {i+1}: {frame['w']}x{frame['h']}px, Area: {frame['area']:.0f}, Aspect: {frame['aspect']:.1f}, Fill: {frame['fill_ratio']:.1%}"
        )
    return frames


def extract_edges_from_frames(
    frames: List[Dict[str, Any]], image_shape: Tuple[int, int]
) -> Tuple[Tuple[List, List], np.ndarray]:
    """Extract 4 edges from each validated frame.

    This function only processes frames that have already been validated
    in detect_frames_by_contours, avoiding re-detection and filtering.
    """
    h, w = image_shape
    line_image = np.zeros((h, w), dtype=np.uint8)
    horizontal_lines = []
    vertical_lines = []

    # Extract edges from each validated frame
    for frame in frames:
        x, y, rect_w, rect_h = frame["x"], frame["y"], frame["w"], frame["h"]

        # Extract 4 edges of the rectangle as lines
        # Top edge (horizontal)
        horizontal_lines.append((x, y, x + rect_w, y))
        # Bottom edge (horizontal)
        horizontal_lines.append((x, y + rect_h, x + rect_w, y + rect_h))
        # Left edge (vertical)
        vertical_lines.append((x, y, x, y + rect_h))
        # Right edge (vertical)
        vertical_lines.append((x + rect_w, y, x + rect_w, y + rect_h))

        # Draw edges on line image
        cv2.line(line_image, (x, y), (x + rect_w, y), 255, 2)  # Top
        cv2.line(
            line_image, (x, y + rect_h), (x + rect_w, y + rect_h), 255, 2
        )  # Bottom
        cv2.line(line_image, (x, y), (x, y + rect_h), 255, 2)  # Left
        cv2.line(line_image, (x + rect_w, y), (x + rect_w, y + rect_h), 255, 2)  # Right

    logger.debug(
        f"Detected {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines"
    )

    return (horizontal_lines, vertical_lines), line_image


def extract_frame_dimensions(
    frames: List[Dict[str, Any]], lines: Tuple[List, List]
) -> List[Dict[str, Any]]:
    """Extract precise dimensions from frames using detected lines and filter invalid frames."""
    logger = logging.getLogger("extract_frame_dimensions")

    h_lines, v_lines = lines

    enhanced_frames = []
    for frame in frames:
        x, y, w, h = frame["x"], frame["y"], frame["w"], frame["h"]

        # Find lines within frame bounds (with margin)
        margin = 20
        frame_h_lines = []
        frame_v_lines = []

        for x1, y1, x2, y2 in h_lines:
            if (
                y - margin <= y1 <= y + h + margin
                and y - margin <= y2 <= y + h + margin
            ):
                if x - margin <= min(x1, x2) and max(x1, x2) <= x + w + margin:
                    frame_h_lines.append((x1, y1, x2, y2))

        for x1, y1, x2, y2 in v_lines:
            if (
                x - margin <= x1 <= x + w + margin
                and x - margin <= x2 <= x + w + margin
            ):
                if y - margin <= min(y1, y2) and max(y1, y2) <= y + h + margin:
                    frame_v_lines.append((x1, y1, x2, y2))

        enhanced_frame = frame.copy()
        enhanced_frame["h_lines_count"] = len(frame_h_lines)
        enhanced_frame["v_lines_count"] = len(frame_v_lines)
        enhanced_frame["total_lines"] = len(frame_h_lines) + len(frame_v_lines)
        enhanced_frames.append(enhanced_frame)

    # Filter out noisy frames with insufficient line support
    # Valid frames should have strong line support AND reasonable size
    filtered_frames = []
    for frame in enhanced_frames:
        total_lines = frame["total_lines"]
        v_lines_count = frame["v_lines_count"]
        h_lines_count = frame["h_lines_count"]
        w, h = frame["w"], frame["h"]
        area = w * h

        # Criteria for MAIN FRAMES (black thick borders):
        # Problem: Thick black borders → horizontal edges hard to detect by Hough
        # Solution: Accept frames with STRONG VERTICAL support even if horizontal is weak

        # STRICT FILTER: Only keep PRIMARY FRAMES (black thick borders)
        # Strategy: These frames are LARGE and have STRONG vertical lines

        # Primary frames: Very large with some support
        is_primary = (
            area >= 150000  # Very large (>= 300x500)
            and v_lines_count >= 2  # At least some vertical support
        )

        # Secondary: Moderately large with STRONG vertical
        is_secondary = (
            area >= 100000  # Large (>= 250x400)
            and v_lines_count >= 4  # Very strong vertical
        )

        # Only for thick black borders: Strong vertical, no horizontal needed
        is_thick_border = (
            v_lines_count >= 4  # Very strong vertical (4+ lines)
            and area >= 80000  # Reasonably large
            and total_lines <= 8  # Not too many lines (avoid noisy regions)
        )

        # Small frames: For resized/small images (e.g., from pipeline blocks)
        # Accept smaller frames if they have reasonable line support
        is_small_frame = (
            area >= 30000  # Small but valid (>= 150x200)
            and v_lines_count >= 2  # At least 2 vertical lines
            and h_lines_count >= 2  # At least 2 horizontal lines
            and total_lines >= 4  # Total 4+ lines
            and total_lines <= 10  # Not too noisy
        )

        is_valid = is_primary or is_secondary or is_thick_border or is_small_frame

        if is_valid:
            filtered_frames.append(frame)
        else:
            logger.debug(
                f"  ⚠ Filtered noisy frame: {frame['w']}x{frame['h']}px "
                f"(only {frame['h_lines_count']}H + {frame['v_lines_count']}V lines, area={area:.0f})"
            )

    return filtered_frames
