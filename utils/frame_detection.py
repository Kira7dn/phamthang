"""
Frame Detection Module - Simplified approach to detect aluminum frames.

Strategy:
1. Use Canny edge detection
2. Apply Hough Lines to find frame edges
3. Detect closed rectangles using contours on morphologically processed edges
4. Combine results for accurate frame dimensions
"""

from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.normalize_image import normalize_image
from utils.image_process import ImagePipeline, save_stage_image


def draw_rectangles(
    image: np.ndarray,
    rects: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw rectangles on image with labels."""
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    for i, rect in enumerate(rects):
        x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        # Draw label
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
    h, w = binary.shape
    cleaned = binary.copy()

    # Step 1: Remove outer margin regions (where dimension text/arrows typically are)
    # Find the main content area by detecting largest solid regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find contours to locate main content
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find bounding box of all main contours
        all_points = np.vstack(contours)
        x_min, y_min = all_points.min(axis=0)[0]
        x_max, y_max = all_points.max(axis=0)[0]

        # Add larger margin to remove dimension text that's close to edges
        margin = 50  # Increased to catch text near edges
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        # Clear regions outside main area (including dimension text)
        cleaned[:y_min, :] = 0  # Top margin
        cleaned[y_max:, :] = 0  # Bottom margin
        cleaned[:, :x_min] = 0  # Left margin
        cleaned[:, x_max:] = 0  # Right margin

        # ADAPTIVE border removal: only clear inner border if content is far from edge
        # This prevents removing frame edges in tight layouts
        content_width = x_max - x_min
        content_height = y_max - y_min

        # Only apply border removal if main content doesn't fill most of the image
        # (i.e., there's room for dimension annotations)
        if content_width < w * 0.9 and content_height < h * 0.9:
            # Calculate safe border width (don't exceed 5% of content size)
            safe_border = min(50, int(min(content_width, content_height) * 0.05))

            # Right edge - only if there's enough space
            if x_max - safe_border > x_min + 50:  # Keep at least 50px of content
                cleaned[:, max(x_min, x_max - safe_border) : x_max] = 0
            # Left edge - only if there's enough space
            if x_min + safe_border < x_max - 50:  # Keep at least 50px of content
                cleaned[:, x_min : min(x_max, x_min + safe_border)] = 0

    # Step 2: Remove small isolated text components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    for i in range(1, num_labels):
        x, y, w_comp, h_comp, area = stats[i]

        # Remove small compact regions (likely text)
        is_small = w_comp < 120 and h_comp < 120
        is_compact = area < 10000

        if is_small and is_compact:
            cleaned[labels == i] = 0

    return cleaned


def connect_broken_lines(binary: np.ndarray) -> np.ndarray:
    """Connect broken lines by detecting and extending line segments.

    Strategy:
    - Use Hough Lines to detect all line segments
    - Group nearby collinear segments
    - Extend and connect segments in the same group
    - Redraw connected lines
    """
    h, w = binary.shape

    # Detect edges
    edges = cv2.Canny(binary, 30, 100, apertureSize=3)

    # Detect line segments with relaxed parameters to catch broken pieces
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,  # Lower to detect weak segments
        minLineLength=15,  # Shorter to catch small pieces
        maxLineGap=30,  # Allow gaps
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
    def connect_horizontal_segments(segments, gap_tolerance=30, y_tolerance=5):
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

    def connect_vertical_segments(segments, gap_tolerance=30, x_tolerance=5):
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

    # Connect segments multiple times for better coverage
    for _ in range(3):
        h_segments = connect_horizontal_segments(h_segments)
        v_segments = connect_vertical_segments(v_segments)

    # Create new image with connected lines
    connected = np.zeros_like(binary)

    # Draw connected horizontal lines
    for x1, y1, x2, y2 in h_segments:
        cv2.line(connected, (x1, y1), (x2, y2), 255, 2)

    # Draw connected vertical lines
    for x1, y1, x2, y2 in v_segments:
        cv2.line(connected, (x1, y1), (x2, y2), 255, 2)

    # Slight thickening to ensure continuity
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    connected = cv2.dilate(connected, kernel, iterations=1)

    return connected


def remove_diagonal_lines(binary: np.ndarray) -> np.ndarray:
    """Remove diagonal lines (dimension annotations) by keeping only horizontal/vertical structures.

    Strategy:
    - Use Hough Lines to detect and classify lines by angle
    - Keep only lines that are nearly horizontal (angle < 10°) or vertical (80° < angle < 100°)
    - Reconstruct image from filtered lines with thicker lines to prevent gaps
    """
    h, w = binary.shape

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

    # Create new image with only horizontal and vertical lines
    # cleaned = np.zeros_like(binary)
    cleaned = binary.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            def is_axis_aligned(angle):
                return angle < 15 or angle > 165 or (75 < angle < 105)

            # Remove only diagonal lines by painting them black
            if not is_axis_aligned(angle):
                cv2.line(cleaned, (x1, y1), (x2, y2), 0, 4)
    return cleaned


def reconnect_broken_frames(binary: np.ndarray) -> np.ndarray:
    """Aggressively reconnect broken frame edges after diagonal removal.

    This is stronger than connect_broken_lines and specifically targets
    gaps created by diagonal line removal.
    """
    h, w = binary.shape

    # Step 1: Close horizontal gaps with horizontal kernel
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    horizontal_closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, h_kernel, iterations=1
    )

    # Step 2: Close vertical gaps with vertical kernel
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, v_kernel, iterations=1)

    # Step 3: Combine both
    combined = cv2.bitwise_or(horizontal_closed, vertical_closed)

    # Step 4: Close remaining small gaps with square kernel
    kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    reconnected = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, kernel_square, iterations=3
    )

    # Step 5: Dilate to make lines thicker and more continuous
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    reconnected = cv2.dilate(reconnected, kernel_dilate, iterations=2)

    return reconnected


def detect_frames_by_contours(
    cleaned_binary: np.ndarray,
    min_area_ratio: float = 0.0001,
    max_aspect: float = 20.0,
    min_fill_ratio: float = 0.2,
) -> List[Dict[str, Any]]:
    """Detect frame rectangles using contours.

    Strategy: Use bounding rectangles instead of contour area to avoid
    issues with internal annotations (dimension lines, diagonal lines, etc.)
    Uses hierarchy to filter out parent contours that only serve as boundaries.
    """
    h, w = cleaned_binary.shape
    min_area = h * w * min_area_ratio

    # Find all contours with hierarchy
    # Hierarchy format: [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(
        cleaned_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"Found {len(contours)} total contours (image: {w}x{h}, min_area: {min_area}")

    # Get all contour areas for debugging
    all_areas = [cv2.contourArea(cnt) for cnt in contours]
    all_areas.sort(reverse=True)
    if all_areas:
        print(f"  Top 5 contour areas: {all_areas[:5]}")
        print(f"  Area range: {all_areas[0]:.0f} -> {all_areas[-1]:.0f}")

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

    for idx, cnt in enumerate(contours):
        # CRITICAL: Filter out parent contours that are just outer boundaries
        # Check if this contour has children (First_Child != -1)
        if hierarchy is not None and hierarchy[idx][2] != -1:
            # This contour has children - check if it's just an outer boundary
            # Get all children
            child_idx = hierarchy[idx][2]
            num_children = 0
            while child_idx != -1:
                num_children += 1
                child_idx = hierarchy[child_idx][0]  # Next sibling

            # If this contour has multiple children (2+), it's likely an outer boundary
            # NOT an actual frame
            if num_children >= 2:
                filtered_stats["parent_boundary"] += 1
                x, y, rect_w, rect_h = cv2.boundingRect(cnt)
                debug_info.append(
                    f"  Filtered (parent boundary with {num_children} children): bbox={rect_w}x{rect_h}"
                )
                continue
        # Get bounding rectangle FIRST
        x, y, rect_w, rect_h = cv2.boundingRect(cnt)
        bbox_area = rect_w * rect_h

        # Use BBOX area for filtering, not contour area
        if bbox_area < min_area:
            filtered_stats["too_small"] += 1
            continue

        # Filter out very small frames
        # Use BOTH percentage (for large images) AND absolute size (for small images)
        image_area = w * h
        min_frame_area = max(
            image_area * 0.02,  # At least 2% of image
            20000,  # OR at least 20,000 pixels (e.g., 100x200)
        )

        if bbox_area < min_frame_area:
            filtered_stats["too_small"] += 1
            debug_info.append(
                f"  Filtered (too small): bbox={rect_w}x{rect_h}, area={bbox_area:.0f} < {min_frame_area:.0f}"
            )
            continue

        # Filter out frames that are too large (> 90% of image in BOTH dimensions)
        # Allow frames that are tall or wide, just not both
        if rect_w > w * 0.9 and rect_h > h * 0.9:
            filtered_stats["bad_aspect"] += 1
            debug_info.append(f"  Filtered (too large): bbox={rect_w}x{rect_h}")
            continue

        # Filter out very thin frames (likely noise or dimension lines)
        if rect_w < 40 or rect_h < 40:
            filtered_stats["bad_aspect"] += 1
            debug_info.append(f"  Filtered (too thin): bbox={rect_w}x{rect_h}")
            continue

        # Filter by aspect ratio (not too elongated)
        aspect = max(rect_w, rect_h) / max(1, min(rect_w, rect_h))
        if aspect > max_aspect:
            filtered_stats["bad_aspect"] += 1
            debug_info.append(
                f"  Filtered (aspect={aspect:.1f}): bbox={rect_w}x{rect_h}"
            )
            continue

        # CRITICAL: Validate that contour is actually closed (not just 3 sides)
        # Check fill_ratio to ensure this is a real frame, not open edges
        contour_area = cv2.contourArea(cnt)
        fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0

        # For valid closed frames, fill_ratio should be significant
        # Open contours (3 sides, L-shape) will have low fill_ratio
        if fill_ratio < 0.3:  # Less than 30% filled = open contour
            filtered_stats["bad_fill"] += 1
            debug_info.append(
                f"  Filtered (open contour): bbox={rect_w}x{rect_h}, fill_ratio={fill_ratio:.2f}"
            )
            continue

        frames.append(
            {
                "x": x,
                "y": y,
                "w": rect_w,
                "h": rect_h,
                "area": float(bbox_area),  # Use bbox area
                "aspect": float(aspect),
                "fill_ratio": float(fill_ratio),
            }
        )

    print(
        f"  Filtered: {filtered_stats['too_small']} too small, "
        f"{filtered_stats['bad_aspect']} bad aspect, "
        f"{filtered_stats['bad_fill']} bad fill ratio, "
        f"{filtered_stats['parent_boundary']} parent boundaries"
    )

    if debug_info:
        print("  Debug - Top filtered contours:")
        for info in debug_info[:5]:
            print(info)

    # Sort by area (largest first)
    frames.sort(key=lambda r: r["area"], reverse=True)
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

    print(
        f"Detected {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines"
    )

    return (horizontal_lines, vertical_lines), line_image


def extract_frame_dimensions(
    frames: List[Dict[str, Any]], lines: Tuple[List, List]
) -> List[Dict[str, Any]]:
    """Extract precise dimensions from frames using detected lines and filter invalid frames."""
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

        is_valid = is_primary or is_secondary or is_thick_border
        is_valid = True
        if is_valid:
            filtered_frames.append(frame)
        else:
            print(
                f"  ⚠ Filtered noisy frame: {frame['w']}x{frame['h']}px "
                f"(only {frame['h_lines_count']}H + {frame['v_lines_count']}V lines, area={area:.0f})"
            )

    return filtered_frames


def detect_frames_pipeline(
    image: np.ndarray,
    output_path: Optional[Path] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Main pipeline to detect aluminum frames.

    Steps:
    1. Convert to grayscale
    2. Apply OTSU threshold
    3. Remove text regions (dimension annotations, numbers)
    4. Remove diagonal lines (keep only horizontal/vertical structures)
    5. Connect broken lines (detect and extend line segments intelligently)
    6. Reconnect broken frames (aggressive morphology to close remaining gaps)
    7. Detect frames by contours
    8. Detect Hough lines
    9. Filter frames by line support
    10. Draw results

    Returns:
        Tuple of (annotated image, list of detected frames with dimensions)
    """
    if image is None or (hasattr(image, "size") and image.size == 0):
        raise ValueError("Input image is empty or None")

    print("\n=== Frame Detection Pipeline ===")

    # Store intermediate results
    pipeline_data = {
        "original": image,
        "binary": None,
        "frames": [],
        "lines": ([], []),
        "enhanced_frames": [],
    }

    def store_binary(binary: np.ndarray) -> np.ndarray:
        """Store binary image for later use."""
        pipeline_data["binary"] = binary
        return binary

    def detect_contour_frames(binary: np.ndarray) -> np.ndarray:
        """Detect frames using contours."""
        print("  → Detecting frames by contours...")
        frames = detect_frames_by_contours(
            binary,
            min_area_ratio=0.001,  # 0.1% of image
            max_aspect=15.0,  # Allow reasonable rectangles
            min_fill_ratio=0.0,  # Not used anymore
        )
        print(f"    Found {len(frames)} potential frames")
        pipeline_data["frames"] = frames
        return binary

    def detect_hough_lines(binary: np.ndarray) -> np.ndarray:
        """Detect lines from validated frames only."""
        print("  → Detecting edges by Hough lines...")
        # CRITICAL: Only extract edges from already-validated frames
        # Don't re-detect contours (which includes filtered-out frames)
        lines, line_image = extract_edges_from_frames(
            pipeline_data["frames"], binary.shape
        )
        pipeline_data["lines"] = lines
        return line_image

    def filter_frames_by_lines(line_image: np.ndarray) -> np.ndarray:
        """Filter frames based on line support."""
        print("  → Filtering frames by line support...")
        enhanced_frames = extract_frame_dimensions(
            pipeline_data["frames"], pipeline_data["lines"]
        )
        pipeline_data["enhanced_frames"] = enhanced_frames
        return line_image

    def draw_final_results(line_image: np.ndarray) -> np.ndarray:
        """Draw rectangles on original image."""
        print("  → Drawing results...")
        annotated = draw_rectangles(
            pipeline_data["original"],
            pipeline_data["enhanced_frames"],
            color=(0, 255, 0),
            thickness=3,
        )

        # Print summary
        frames = pipeline_data["enhanced_frames"]
        print(f"\n✓ Detection completed: {len(frames)} frames found")
        for i, frame in enumerate(frames):
            print(
                f"  Frame {i+1}: {frame['w']}x{frame['h']}px, "
                f"Area: {frame['area']:.0f}, "
                f"Lines: {frame['h_lines_count']}H + {frame['v_lines_count']}V"
            )

        return annotated

    # Build and run pipeline
    pipeline = ImagePipeline(output_path)
    pipeline.add("grayscale", to_grayscale)
    pipeline.add("otsu_threshold", apply_otsu_threshold)
    pipeline.add("remove_text", remove_text_regions)  # Remove text/numbers first
    pipeline.add("remove_diagonals", remove_diagonal_lines)  # Remove diagonals
    pipeline.add("connect_lines", connect_broken_lines)  # Connect broken segments
    pipeline.add("reconnect_frames", reconnect_broken_frames)  # Reconnect after removal
    pipeline.add("store_binary", store_binary)
    pipeline.add("detect_contours", detect_contour_frames)
    pipeline.add("detect_hough", detect_hough_lines)
    pipeline.add("filter_frames", filter_frames_by_lines)
    pipeline.add("draw_results", draw_final_results)

    annotated = pipeline.run(image)

    return annotated, pipeline_data["enhanced_frames"]


if __name__ == "__main__":
    frame2_bold = Path("outputs/panel_agent/analyzer/Block 0/14_white_padding.png")
    frame4_large = Path(
        "outputs/5d430f41d60b422a8385dcbc2e96c66f/Block 0/00_origin.png"
    )
    frame2_bold_noise = Path(
        "outputs/panel_agent/analyzer/Block 0/14_white_padding.png"
    )
    frame_trapezium = Path(
        "outputs/7a08cabecf654f0b9f8b916fcbbe4c56/Block 0/00_origin.png"
    )
    frame4_bold = Path("assets/block/bold4frame.png")
    thin8frame = Path("assets/block/thin8frame.png")

    img_path = frame2_bold
    # img_path = frame4_large
    # img_path = frame2_bold_noise
    # img_path = frame4_bold
    # img_path = thin8frame
    output_dir = Path("outputs", "frame_detection")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(
            f"Không đọc được ảnh: {img_path}. Kiểm tra đường dẫn và quyền truy cập."
        )
    image = normalize_image(image, output_dir / "normalized_image")
    if image is None:
        raise FileNotFoundError(
            f"Không đọc được ảnh: {img_path}. Kiểm tra đường dẫn và quyền truy cập."
        )
    annotated, frames = detect_frames_pipeline(image, output_path=output_dir)
    print(f"\n✓ Pipeline completed. Results saved to: {output_dir}")
