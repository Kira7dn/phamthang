import logging
from pathlib import Path
import shutil
from typing import Optional, Tuple

import numpy as np
import cv2
from app.tools.blockprocess import remove_diagonal_lines
from app.tools.image_process import (
    ImagePipeline,
    adaptive_threshold,
    add_white_padding,
    bridge_horizontal,
    clahe,
    invert_background,
    morph_close,
    morph_open,
    normalize_bg,
    remove_small_components,
    resize_with_limit,
    to_gray,
)


def remove_line_noise_binary(binary_img: np.ndarray) -> np.ndarray:
    """
    Remove line noise from binary image (after adaptive threshold).
    Removes horizontal, vertical, and diagonal lines while preserving text.
    Works on binary images (black text on white background).
    """
    h, w = binary_img.shape[:2]

    # Invert if needed (we want black lines on white background)
    if np.mean(binary_img) < 127:
        working = cv2.bitwise_not(binary_img)
        inverted = True
    else:
        working = binary_img.copy()
        inverted = False

    # Detect horizontal lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1))
    horizontal_lines = cv2.morphologyEx(
        working, cv2.MORPH_OPEN, horiz_kernel, iterations=1
    )

    # Detect vertical lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 20))
    vertical_lines = cv2.morphologyEx(
        working, cv2.MORPH_OPEN, vert_kernel, iterations=1
    )

    # Detect diagonal lines using HoughLinesP
    edges = cv2.Canny(working, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min(h, w) // 8,
        maxLineGap=15,
    )

    diagonal_mask = np.zeros_like(working)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # Only remove long lines (dimension lines, not text strokes)
            if length > min(h, w) * 0.1:
                cv2.line(diagonal_mask, (x1, y1), (x2, y2), 255, 3)

    # Combine all line masks
    line_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    line_mask = cv2.bitwise_or(line_mask, diagonal_mask)

    # Remove lines from image (set to white)
    result = working.copy()
    result[line_mask > 0] = 255

    # Invert back if needed
    if inverted:
        result = cv2.bitwise_not(result)

    return result


def _resize_short_edge(image: np.ndarray, target_short: int) -> np.ndarray:
    h, w = image.shape[:2]
    if target_short <= 0 or min(h, w) == target_short:
        return image
    if h <= w:
        new_h = target_short
        new_w = int(w * (target_short / h))
    else:
        new_w = target_short
        new_h = int(h * (target_short / w))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def smart_resize(
    img: np.ndarray,
    min_size: int = 960,
    max_size: int = 1920,
) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    logger = logging.getLogger("smart_resize")

    # Scale up nếu ảnh nhỏ (OCR thường lỗi khi chiều ngắn < 800)
    if min(h, w) < min_size:
        scale = min_size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        logger.debug(f"Upscale image: {w}x{h} → {new_w}x{new_h} (scale={scale:.2f})")
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return resized, scale

    # Scale down nếu ảnh quá lớn (tăng tốc contour detect)
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        logger.debug(f"Downscale image: {w}x{h} → {new_w}x{new_h} (scale={scale:.2f})")
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    logger.debug(f"Keep original size: {w}x{h}")
    return img, 1.0


# def smart_resize(
#     img: np.ndarray,
#     min_short_side: int = 960,
#     max_long_side: int = 1920,
#     max_pixels: int = 6_000_000,
# ) -> Tuple[np.ndarray, float]:
#     h, w = img.shape[:2]
#     logger = logging.getLogger("smart_resize")

#     short_side = min(h, w)
#     long_side = max(h, w)

#     # Upscale if too small, but cap the upscale factor to avoid artifacts
#     if short_side < min_short_side:
#         desired_scale = min_short_side / short_side
#         # cap scale to 2.0; if >2 recommend super-resolution instead of naive resize
#         scale = min(desired_scale, 2.0)
#         if desired_scale > 2.0:
#             logger.warning(
#                 f"Requested upscale {desired_scale:.2f}x > 2.0x; "
#                 "consider using a super-resolution model for better OCR results."
#             )
#         new_w, new_h = int(round(w * scale)), int(round(h * scale))
#         logger.info(f"Upscale: {w}x{h} → {new_w}x{new_h} (scale={scale:.2f})")
#         resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
#         return resized, scale

#     # Downscale if too large by long side or by pixel count
#     scale_by_long = max_long_side / long_side if long_side > max_long_side else 1.0
#     scale_by_pixels = (max_pixels / (w * h)) ** 0.5 if (w * h) > max_pixels else 1.0
#     scale = min(scale_by_long, scale_by_pixels)

#     if scale < 1.0:
#         new_w, new_h = int(round(w * scale)), int(round(h * scale))
#         logger.info(f"Downscale: {w}x{h} → {new_w}x{new_h} (scale={scale:.2f})")
#         resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
#         return resized, scale

#     logger.debug(f"Keep original size: {w}x{h}")
#     return img, 1.0


def light_sharpen(img):
    kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5.0, -0.5], [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(img, -1, kernel)
    # Blend with original to avoid over-sharpening
    return cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)


def normalize_frame(
    image: np.ndarray,
    output_path: Optional[Path] = None,
    padding_pct: float = 0.05,
    final_short_edge: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Optimize image for OCR: enhance contrast, denoise, binarize.

    Pipeline:
    1. Upscale if too small (better OCR on small text)
    2. Convert to grayscale
    3. Normalize background
    4. CLAHE for local contrast enhancement
    5. Denoise with bilateral filter
    6. Adaptive threshold (better for uneven lighting)
    7. Clean up with morphology
    8. Add padding
    """
    if image is None or (hasattr(image, "size") and image.size == 0):
        raise ValueError("normalize_image: input image is empty or None")

    pipeline = ImagePipeline(output_path)

    # pipeline.add("smart_resize", smart_resize)

    # 2. Convert to grayscale
    # pipeline.add("to_gray", to_gray)

    # 3. Normalize background (remove shadows, uneven lighting)
    # pipeline.add("normalize_bg", normalize_bg)

    # 4. CLAHE for local contrast enhancement (makes text clearer)
    # pipeline.add("clahe", clahe)

    # 6. Light denoise - very gentle to avoid blurring text
    # pipeline.add("denoise", lambda img: cv2.GaussianBlur(img, (3, 3), 0))

    # 7. Very light sharpen to enhance text edges near lines

    # pipeline.add("light_sharpen", light_sharpen)

    # pipeline.add("blur", lambda image: cv2.medianBlur(image, 3))

    # 8. Adaptive threshold (better for varying lighting)
    # Use very small block_size (7) to preserve tiny text near lines like "120"
    # pipeline.add(
    #     "adaptive_threshold", lambda img: adaptive_threshold(img, block_size=15, c=10)
    # )
    # pipeline.add("remove_line_noise", remove_line_noise_binary)

    # 9. Remove line noise from binary image (after threshold)
    # This removes dashed lines and solid lines without blurring text
    # pipeline.add("blur", lambda image: cv2.medianBlur(image, 3))
    # 10. Very light morphology to clean up (remove tiny noise only)
    # pipeline.add("morph_open", lambda img: morph_open(img, (1, 1), 2))
    # pipeline.add("morph_close", lambda image: morph_close(image, (1, 1), 1))
    # pipeline.add("smart_resize", smart_resize)
    # Capture scale factor from resize
    resize_result = {"scale": 1.0}

    def resize_and_capture(img):
        # resized, scale = resize_with_limit(img, 1920, 1920)
        resized, scale = smart_resize(img)
        resize_result["scale"] = scale
        return resized

    pipeline.add("resize", resize_and_capture)
    pipeline.add("to_gray", to_gray)
    pipeline.add("normalize_bg", normalize_bg)
    pipeline.add("blur", lambda image: cv2.medianBlur(image, 3))
    pipeline.add("blur", lambda image: cv2.GaussianBlur(image, (3, 3), 0))
    # pipeline.add("clahe", clahe)
    pipeline.add("adaptive_threshold", adaptive_threshold)
    # pipeline.add("invert_background", invert_background)
    pipeline.add("blur", lambda image: cv2.medianBlur(image, 1))
    pipeline.add("blur", lambda image: cv2.GaussianBlur(image, (1, 1), 0))
    pipeline.add(
        "enhance_edges",
        lambda img: cv2.addWeighted(img, 1, cv2.GaussianBlur(img, (0, 0), 3), -0.5, 0),
    )
    # pipeline.add("clahe", clahe)
    pipeline.add(
        "otsu",
        lambda image: cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1],
    )
    pipeline.add("morph_open", lambda image: morph_open(image, (2, 2), 1))
    pipeline.add("morph_close", lambda image: morph_close(image, (3, 3), 2))
    pipeline.add("remove_small_components", remove_small_components)
    # pipeline.add("bridge_horizontal", lambda image: bridge_horizontal(image, (1, 1), 2))
    # pipeline.add("invert_background", invert_background)
    output_image = pipeline.run(image)
    return output_image, resize_result["scale"]


def normalize_text(
    image: np.ndarray, output_path: Optional[Path] = None
) -> Tuple[np.ndarray, float]:
    if image is None or (hasattr(image, "size") and image.size == 0):
        raise ValueError("normalize_image: input image is empty or None")
    pipeline = ImagePipeline(output_path)

    resize_result = {"scale": 1.0}

    def resize_and_capture(img: np.ndarray) -> np.ndarray:
        resized, scale = smart_resize(img)
        resize_result["scale"] = scale
        return resized

    pipeline.add("resize", resize_and_capture)
    pipeline.add("to_gray", to_gray)
    pipeline.add("normalize_bg", normalize_bg)
    # pipeline.add("remove_diagonals", remove_diagonal_lines)

    # pipeline.add("blur", lambda image: cv2.medianBlur(image, 3))
    # pipeline.add("blur", lambda image: cv2.GaussianBlur(image, (3, 3), 0))
    # pipeline.add("clahe", clahe)
    # pipeline.add("adaptive_threshold", adaptive_threshold)
    # pipeline.add("invert_background", invert_background)
    # pipeline.add("blur", lambda image: cv2.medianBlur(image, 1))
    # pipeline.add("blur", lambda image: cv2.GaussianBlur(image, (1, 1), 0))
    # pipeline.add(
    #     "enhance_edges",
    #     lambda img: cv2.addWeighted(img, 1, cv2.GaussianBlur(img, (0, 0), 3), -0.5, 0),
    # )
    # pipeline.add("clahe", clahe)
    # pipeline.add(
    #     "otsu",
    #     lambda image: cv2.threshold(
    #         image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    #     )[1],
    # )
    # pipeline.add("morph_open", lambda image: morph_open(image, (2, 2), 1))
    # pipeline.add("morph_close", lambda image: morph_close(image, (3, 3), 2))
    # pipeline.add("remove_small_components", remove_small_components)
    # pipeline.add("bridge_horizontal", lambda image: bridge_horizontal(image, (1, 1), 2))
    # pipeline.add("invert_background", invert_background)
    output_image = pipeline.run(image)
    return output_image, resize_result["scale"]


if __name__ == "__main__":
    # img_path = Path("outputs/panel_analyze_pipeline/Block 2/00_origin.png")
    # img_path = Path("outputs/5d430f41d60b422a8385dcbc2e96c66f/Block 0/00_origin.png")
    # img_path = Path("outputs/24a94f1546374c16b54e1e411cc96010/Block 1/00_origin.png")
    # img_path = Path("outputs/c7895b9a60794796bcdb6568edda235b/Block 0/00_origin.png")
    # img_path = Path("outputs/22609b998d2c4ce28e6bceb77b92ffb2/Block 0/00_origin.png")
    img_path = Path("outputs2/pipeline/6333e83e/Block 0/normalized_ocr/00_origin.png")

    output_dir = Path("outputs", "normalize_text")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(
            f"Không đọc được ảnh: {img_path}. Kiểm tra đường dẫn và quyền truy cập."
        )
    normalized_image = normalize_text(image, output_dir)
    print("Normalized image saved to:", output_dir)
