from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.normalize_image import normalize_image
from utils.image_process import save_stage_image


def preprocess_for_frame_detection(image: np.ndarray) -> np.ndarray:
    """Preprocess image for frame detection using edge detection."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Detect edges using Canny
    edges = cv2.Canny(filtered, 30, 100, apertureSize=3)
    
    return edges


def find_rectangles_from_lines(
    h_lines: List[Tuple[int, int, int, int]],
    v_lines: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    tolerance: int = 20
) -> List[Dict[str, Any]]:
    """Find rectangles by intersecting horizontal and vertical lines."""
    
    # Group lines by position
    def group_horizontal_lines(lines, tol=tolerance):
        """Group horizontal lines by y-coordinate."""
        if not lines:
            return []
        sorted_lines = sorted(lines, key=lambda l: (l[1] + l[3]) // 2)
        groups = []
        current_group = [sorted_lines[0]]
        
        for line in sorted_lines[1:]:
            y_curr = (line[1] + line[3]) // 2
            y_prev = (current_group[-1][1] + current_group[-1][3]) // 2
            if abs(y_curr - y_prev) < tol:
                current_group.append(line)
            else:
                groups.append(current_group)
                current_group = [line]
        groups.append(current_group)
        
        # Merge lines in each group
        merged = []
        for group in groups:
            x_min = min(min(l[0], l[2]) for l in group)
            x_max = max(max(l[0], l[2]) for l in group)
            y_avg = int(np.mean([l[1] for l in group] + [l[3] for l in group]))
            merged.append((x_min, y_avg, x_max, y_avg))
        return merged
    
    def group_vertical_lines(lines, tol=tolerance):
        """Group vertical lines by x-coordinate."""
        if not lines:
            return []
        sorted_lines = sorted(lines, key=lambda l: (l[0] + l[2]) // 2)
        groups = []
        current_group = [sorted_lines[0]]
        
        for line in sorted_lines[1:]:
            x_curr = (line[0] + line[2]) // 2
            x_prev = (current_group[-1][0] + current_group[-1][2]) // 2
            if abs(x_curr - x_prev) < tol:
                current_group.append(line)
            else:
                groups.append(current_group)
                current_group = [line]
        groups.append(current_group)
        
        # Merge lines in each group
        merged = []
        for group in groups:
            y_min = min(min(l[1], l[3]) for l in group)
            y_max = max(max(l[1], l[3]) for l in group)
            x_avg = int(np.mean([l[0] for l in group] + [l[2] for l in group]))
            merged.append((x_avg, y_min, x_avg, y_max))
        return merged
    
    h_grouped = group_horizontal_lines(h_lines)
    v_grouped = group_vertical_lines(v_lines)
    
    print(f"  Grouped: {len(h_grouped)} horizontal, {len(v_grouped)} vertical lines")
    
    # Find rectangles from intersecting lines
    rectangles = []
    for i, h1 in enumerate(h_grouped):
        for h2 in h_grouped[i+1:]:
            y1, y2 = min(h1[1], h2[1]), max(h1[1], h2[1])
            if y2 - y1 < 50:  # Too small
                continue
            
            for j, v1 in enumerate(v_grouped):
                for v2 in v_grouped[j+1:]:
                    x1, x2 = min(v1[0], v2[0]), max(v1[0], v2[0])
                    if x2 - x1 < 50:  # Too small
                        continue
                    
                    # Check if lines actually form a rectangle
                    # (horizontal lines should span the vertical lines' x range)
                    h1_x_range = (h1[0], h1[2])
                    h2_x_range = (h2[0], h2[2])
                    v1_y_range = (v1[1], v1[3])
                    v2_y_range = (v2[1], v2[3])
                    
                    # Check overlap
                    if (min(h1_x_range[1], h2_x_range[1]) - max(h1_x_range[0], h2_x_range[0]) > (x2 - x1) * 0.5 and
                        min(v1_y_range[1], v2_y_range[1]) - max(v1_y_range[0], v2_y_range[0]) > (y2 - y1) * 0.5):
                        
                        w = x2 - x1
                        h = y2 - y1
                        aspect = max(w, h) / max(1, min(w, h))
                        
                        if aspect < 20:  # Reasonable aspect ratio
                            rectangles.append({
                                "x": x1,
                                "y": y1,
                                "w": w,
                                "h": h,
                                "area": float(w * h),
                                "aspect": float(aspect),
                            })
    
    # Remove duplicates
    unique_rects = []
    for rect in rectangles:
        is_dup = False
        for existing in unique_rects:
            if (abs(rect["x"] - existing["x"]) < tolerance and
                abs(rect["y"] - existing["y"]) < tolerance and
                abs(rect["w"] - existing["w"]) < tolerance and
                abs(rect["h"] - existing["h"]) < tolerance):
                is_dup = True
                break
        if not is_dup:
            unique_rects.append(rect)
    
    # Sort by area
    unique_rects.sort(key=lambda r: r["area"], reverse=True)
    return unique_rects


def detect_frames_pipeline(
    image: np.ndarray,
    output_path: Optional[Path] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Simplified pipeline to detect aluminum frames using Hough lines."""
    
    if image is None or (hasattr(image, "size") and image.size == 0):
        raise ValueError("Input image is empty or None")
    
    print("\n=== Frame Detection Pipeline (V2 - Hough-based) ===")
    
    # Step 1: Preprocess for edge detection
    print("Step 1: Edge detection...")
    edges = preprocess_for_frame_detection(image)
    if output_path:
        save_stage_image("edges", edges, output_path, 1)
    
    # Step 2: Detect lines using Hough Transform
    print("Step 2: Detecting lines with Hough Transform...")
    h, w = edges.shape
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=max(min(h, w) // 15, 40),
        maxLineGap=30,
    )
    
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:  # Horizontal
                horizontal_lines.append((x1, y1, x2, y2))
            elif 80 < angle < 100:  # Vertical
                vertical_lines.append((x1, y1, x2, y2))
    
    print(f"  Found {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines")
    
    # Draw lines
    line_image = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    for x1, y1, x2, y2 in vertical_lines:
        cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    
    if output_path:
        save_stage_image("hough_lines", line_image, output_path, 2)
    
    # Step 3: Find rectangles from line intersections
    print("Step 3: Finding rectangles from line intersections...")
    frames = find_rectangles_from_lines(horizontal_lines, vertical_lines, w, h)
    print(f"  → Found {len(frames)} potential frames")
    
    # Step 4: Draw results
    print("Step 4: Drawing results...")
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    
    for i, rect in enumerate(frames):
        x, y, rect_w, rect_h = rect["x"], rect["y"], rect["w"], rect["h"]
        cv2.rectangle(annotated, (x, y), (x + rect_w, y + rect_h), (0, 255, 0), 3)
        
        label = f"Frame {i+1}: {rect_w}x{rect_h}"
        cv2.putText(
            annotated,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    
    if output_path:
        save_stage_image("final_result", annotated, output_path, 99)
    
    print(f"\n✓ Detection completed: {len(frames)} frames found")
    for i, frame in enumerate(frames):
        print(f"  Frame {i+1}: {frame['w']}x{frame['h']}px, Area: {frame['area']:.0f}")
    
    return annotated, frames


if __name__ == "__main__":
    img_path = Path("outputs/24a94f1546374c16b54e1e411cc96010/Block 1/00_origin.png")
    output_dir = Path("outputs", "frame_detection_v2")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    
    print("Normalizing image...")
    image = normalize_image(image, output_dir / "normalized")
    if image is None:
        raise ValueError("Image normalization failed")
    
    print("\nStarting frame detection...")
    annotated, frames = detect_frames_pipeline(image, output_path=output_dir)
    
    print(f"\n✓ Pipeline completed. Results saved to: {output_dir}")
