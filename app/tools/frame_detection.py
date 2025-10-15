"""
Frame Detection Module - Simplified approach to detect aluminum frames.

Strategy:
1. Use Canny edge detection
2. Apply Hough Lines to find frame edges
3. Detect closed rectangles using contours on morphologically processed edges
4. Combine results for accurate frame dimensions
"""

import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.tools.blockprocess import (
    apply_otsu_threshold,
    connect_lines,
    detect_frames_by_contours,
    draw_rectangles,
    extract_edges_from_frames,
    extract_frame_dimensions,
    reconnect_broken_frames,
    remove_diagonal_lines,
    remove_text_regions,
    to_grayscale,
)
from app.tools.normalize_image import normalize_image
from app.tools.image_process import ImagePipeline

# Constants for frame detection
MIN_FRAME_PERCENTAGE = 0.02  # 2% of image area
MIN_FRAME_PIXELS = 20000  # Absolute minimum size
MIN_DIMENSION = 40  # Minimum width or height
MAX_ASPECT_RATIO = 15.0  # Maximum elongation
MIN_FILL_RATIO = 0.3  # Minimum fill ratio for closed contours
MIN_CHILDREN_FOR_BOUNDARY = 2  # Parent with 2+ children is boundary

# Constants for line detection and morphology
SEGMENT_GAP_TOLERANCE = 30
SEGMENT_POSITION_TOLERANCE = 5
MORPH_KERNEL_SIZE = (5, 5)
TEXT_MAX_SIZE = 120
TEXT_MAX_AREA = 10000


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
        frames = detect_frames_by_contours(binary)
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

    def draw_final_results(image: np.ndarray) -> np.ndarray:
        """Draw rectangles on original image."""
        print("  → Drawing results...", image.shape)
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
    pipeline.add(
        "remove_text_regions", remove_text_regions
    )  # Remove text/numbers first
    pipeline.add("remove_diagonals", remove_diagonal_lines)  # Remove diagonals
    pipeline.add("connect_lines", connect_lines)  # Connect broken segments
    pipeline.add("reconnect_frames", reconnect_broken_frames)  # Reconnect after removal
    pipeline.add("store_binary", store_binary)
    pipeline.add("detect_contours", detect_contour_frames)
    pipeline.add("detect_hough", detect_hough_lines)
    pipeline.add("filter_frames", filter_frames_by_lines)
    pipeline.add("draw_results", draw_final_results)

    annotated = pipeline.run(image)
    # save frame to json if output_path is not None
    if output_path is not None:
        with open(output_path / "frames.json", "w", encoding="utf-8") as f:
            json.dump(pipeline_data["enhanced_frames"], f, ensure_ascii=False)
    return annotated, pipeline_data["enhanced_frames"]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    frame2_bold = Path("assets/block/frame2_bold.png")
    frame4_large = Path(
        "outputs/5d430f41d60b422a8385dcbc2e96c66f/Block 0/00_origin.png"
    )
    frame_trapezium = Path(
        "outputs/7a08cabecf654f0b9f8b916fcbbe4c56/Block 0/00_origin.png"
    )
    frame4_bold = Path("assets/block/bold4frame.png")
    thin8frame = Path("assets/block/thin8frame.png")
    frame3_thin = Path(
        "outputs/67f6ef3dadba4dc6b84c23c66e078b73/Block 0/normalized/00_origin.png"
    )
    thin5 = Path(
        "outputs/67f6ef3dadba4dc6b84c23c66e078b73/Block 1/normalized/00_origin.png"
    )
    thin_3 = Path("outputs/pipeline/15234d05/Block 0/frame_detection/00_origin.png")
    test = Path("outputs/pipeline/ae436b30/Block 0/normalized/00_origin.png")
    # img_path = frame3_thin
    img_path = thin5
    # img_path = test
    # img_path = test
    # img_path = frame4_bold
    # img_path = thin8frame
    # img_path = thin_3
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
    if annotated is None:
        raise FileNotFoundError(
            f"Không vẽ được ảnh: {img_path}. Kiểm tra đường dẫn và quyền truy cập."
        )
    print(f"\n✓ Pipeline completed. Results saved to: {output_dir}")
