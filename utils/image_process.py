import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


def make_text_mask(
    shape: Tuple[int, int], boxes: List[Tuple[int, int, int, int, float, int, str, str]]
) -> np.ndarray:
    img_h, img_w = shape
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for x1, y1, x2, y2, *_ in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def make_whitened_image(
    image: np.ndarray, boxes: List[Tuple[int, int, int, int, float, int, str, str]]
) -> np.ndarray:
    whitened = image.copy()
    if whitened.ndim == 2:
        for x1, y1, x2, y2, *_ in boxes:
            cv2.rectangle(whitened, (x1, y1), (x2, y2), 255, -1)
    else:
        for x1, y1, x2, y2, *_ in boxes:
            cv2.rectangle(whitened, (x1, y1), (x2, y2), (255, 255, 255), -1)
    return whitened


def _ensure_binary(img_input: np.ndarray) -> Tuple[np.ndarray, bool]:
    if img_input.ndim == 3:
        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_input.copy()

    unique_vals = np.unique(gray)
    if unique_vals.size <= 2 and set(unique_vals.tolist()).issubset({0, 255}):
        bin_img = (gray > 0).astype(np.uint8) * 255
    else:
        if np.std(gray) < 20:
            gray = cv2.equalizeHist(gray)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    inverted = False
    if np.count_nonzero(bin_img) > bin_img.size // 2:
        bin_img = cv2.bitwise_not(bin_img)
        inverted = True

    return bin_img, inverted


def clean_line(img_input: np.ndarray) -> np.ndarray:
    """
    Loại bỏ các đường thẳng mảnh/dài trên ảnh màu BGR mà không phá chữ.
    Kết hợp adaptive threshold động theo nền sáng/tối, morphology và HoughLinesP.
    Bảo vệ vùng chữ bằng mask riêng và lưu mask đường thẳng thành `mask.png`.
    """

    is_color = img_input.ndim == 3 and img_input.shape[2] == 3

    if is_color:
        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_input.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    mean_intensity = float(np.mean(gray))
    if mean_intensity > 127:
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 4
        )
    else:
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 4
        )

    h, w = gray.shape
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 35), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 35)))

    morph_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, 1)
    morph_mask = cv2.bitwise_or(
        morph_mask, cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, 1)
    )

    # Hough cho đường chéo / line còn sót nhưng với ngưỡng cao hơn để bớt nhạy
    edges = cv2.Canny(blurred, 90, 200, apertureSize=3)
    min_line_len = max(int(min(h, w) * 0.35), 30)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=110,
        minLineLength=min_line_len,
        maxLineGap=8,
    )

    hough_mask = np.zeros_like(gray, dtype=np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if np.hypot(x2 - x1, y2 - y1) < min_line_len:
                continue
            cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 2)
    # mask = cv2.bitwise_or(morph_mask, hough_mask)
    mask = morph_mask.copy()
    mask_path = Path.cwd() / "mask0.png"
    if not mask_path.parent.exists():
        mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)), 1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    filtered_mask = np.zeros_like(mask)
    length_threshold = max(int(min_line_len * 0.9), 30)
    for idx in range(1, num_labels):
        _x, _y, comp_w, comp_h, _area = stats[idx]
        if comp_w >= length_threshold or comp_h >= length_threshold:
            filtered_mask[labels == idx] = 255

    mask = filtered_mask

    mask_path = Path.cwd() / "mask.png"
    if not mask_path.parent.exists():
        mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    repaired = cv2.inpaint(img_input.copy(), mask, 1, cv2.INPAINT_TELEA)

    if is_color and repaired.ndim == 2:
        repaired = cv2.cvtColor(repaired, cv2.COLOR_GRAY2BGR)
    if not is_color and repaired.ndim == 3:
        repaired = cv2.cvtColor(repaired, cv2.COLOR_BGR2GRAY)

    return repaired
    # return cv2.addWeighted(repaired, 0.9, img_input, 0.1, 0)


def bold_text(
    img_input: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    strength: int = 100,
) -> np.ndarray:
    """
    Làm đậm nét chữ bằng cách dãn mask chữ và giảm sáng vùng chữ.

    Giữ nguyên định dạng kênh như đầu vào.
    """
    is_color = img_input.ndim == 3 and img_input.shape[2] == 3

    if is_color:
        gray = img_input.mean(axis=2).astype(np.uint8)
    else:
        gray = img_input

    _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    thick_mask = cv2.dilate(text_mask, kernel, iterations=1)

    mask_scale = (thick_mask.astype(np.int16) * strength) // 255

    if is_color:
        base = img_input.astype(np.int16)
        enhanced = np.clip(base - mask_scale[..., None], 0, 255).astype(np.uint8)
        return enhanced

    enhanced = np.clip(gray.astype(np.int16) - mask_scale, 0, 255).astype(np.uint8)
    return enhanced


def enhance_lines(img_input: np.ndarray) -> np.ndarray:
    """
    Phát hiện và nối liền các đường thẳng trong `img_input` bằng HoughLinesP và morphology.
    """
    bin_img, inverted = _ensure_binary(img_input)

    h, w = bin_img.shape
    mask_lines = np.zeros_like(bin_img)

    # Thicken lines trước khi detect để nối đứt
    thickened = cv2.morphologyEx(
        bin_img,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    edges = cv2.Canny(
        thickened, 30, 200, apertureSize=3
    )  # Giảm ngưỡng Canny để detect edges mảnh hơn
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,  # Giảm threshold để detect lines yếu hơn
        minLineLength=25,  # Giảm min length để detect lines ngắn
        maxLineGap=50,  # Tăng maxGap để nối lines đứt xa hơn
    )

    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(mask_lines, (x1, y1), (x2, y2), 255, 2)

    # Dilate nhiều hơn để nối liền các line đứt và làm dày
    enhanced_lines = cv2.dilate(
        mask_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=2,  # Tăng iterations để nối tốt hơn
    )

    # Kết hợp với ảnh nhị phân gốc để giữ lại các chi tiết khác
    result = cv2.bitwise_or(bin_img, enhanced_lines)
    if inverted:
        result = cv2.bitwise_not(result)

    return result


class ImagePipeline:
    def __init__(self, output_path: Optional[Path] = None) -> None:
        self.output_path = output_path
        self._steps: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = []

    def add(
        self, name: str, func: Callable[[np.ndarray], np.ndarray]
    ) -> "ImagePipeline":
        self._steps.append((name, func))
        return self

    def run(self, image: np.ndarray) -> np.ndarray:
        current = image
        should_save = self.output_path is not None
        order = 1
        if should_save:
            save_stage_image("origin", current, self.output_path, 0)
        for name, func in self._steps:
            current = func(current)
            if should_save:
                save_stage_image(name, current, self.output_path, order)
            order += 1

        return current


def save_stage_image(
    stage_name: str, image: np.ndarray, output_path: Path, order: int
) -> Path:
    if output_path.suffix:
        stage_path = output_path.with_name(
            f"{output_path.stem}_{order:02d}_{stage_name}.png"
        )
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        stage_path = output_path / f"{order:02d}_{stage_name}.png"
    cv2.imwrite(str(stage_path), image)
    return stage_path


def visualize_boxes(
    image: np.ndarray, boxes: List[Tuple[int, int, int, int, float, int, str, str]]
) -> np.ndarray:
    vis = image.copy()
    for x1, y1, x2, y2, _conf, _psm, text, orient in boxes:
        # draw rectangle
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        txt = (text or "").strip().replace("\n", " ")
        if not txt:
            continue
        font = cv2.FONT_HERSHEY_SIMPLEX
        if orient == "v" and len(txt) > 1:
            # draw vertical by stacking characters inside top area
            font_scale = 0.7
            thickness_fg = 2
            # compute single char size (use widest char as approx)
            (cw, ch), _ = cv2.getTextSize("0", font, font_scale, thickness_fg)
            total_h = len(txt) * (ch + 4) - 4
            cx = x1 + max(0, (x2 - x1 - cw) // 2)
            cy = y1 + 6 + ch
            # background band
            cv2.rectangle(
                vis, (cx - 4, y1 + 2), (cx + cw + 4, y1 + 8 + total_h), (0, 0, 0), -1
            )
            for i, chrs in enumerate(txt):
                cv2.putText(
                    vis,
                    chrs,
                    (cx, cy + i * (ch + 4)),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness_fg,
                    cv2.LINE_AA,
                )
        else:
            # horizontal label centered on top border
            font_scale = 0.6
            thickness_fg = 2
            (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness_fg)
            tx = x1 + max(0, (x2 - x1 - tw) // 2)
            ty_above = y1 - 6
            draw_above = ty_above - th >= 0
            if draw_above:
                ty = ty_above
                cv2.rectangle(
                    vis, (tx - 4, ty - th - 2), (tx + tw + 4, ty + 4), (0, 0, 0), -1
                )
            else:
                ty = y1 + th + 6
                cv2.rectangle(
                    vis, (tx - 4, y1 + 2), (tx + tw + 4, y1 + th + 10), (0, 0, 0), -1
                )
            cv2.putText(
                vis,
                txt,
                (tx, ty),
                font,
                font_scale,
                (255, 255, 255),
                thickness_fg,
                cv2.LINE_AA,
            )
    return vis


def resize_with_limit(img, max_width=1920, max_height=1920):
    h, w = img.shape[:2]

    # Tính tỷ lệ thu phóng
    scale = min(max_width / w, max_height / h)
    # Nếu scale >= 1 → upscale, ngược lại → downscale
    if scale >= 1:
        interpolation = cv2.INTER_CUBIC  # phóng to, giữ chi tiết
    else:
        interpolation = cv2.INTER_AREA  # thu nhỏ, giảm nhiễu

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    return resized


def morph_close(
    img: np.ndarray, ksize: Tuple[int, int] = (1, 1), iterations: int = 3
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
    morphology = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations)
    return morphology


def morph_open(
    img: np.ndarray, ksize: Tuple[int, int] = (1, 1), iterations: int = 1
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
    morphology = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations)
    return morphology


def binary_threshold(img: np.ndarray) -> np.ndarray:
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def normalize_bg(img: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(img) if float(np.mean(img)) < 127.0 else img


def adaptive_threshold(
    img: np.ndarray, block_size: int = 15, c: int = 10
) -> np.ndarray:
    return cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def clahe(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        # Ảnh grayscale: áp dụng CLAHE trực tiếp
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe_obj.apply(img)
    elif img.ndim == 3:
        # Ảnh màu: chuyển sang LAB, áp dụng CLAHE trên kênh L, chuyển lại BGR
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        raise ValueError("Ảnh phải là grayscale (2D) hoặc màu (3D)")


def apply_clahe(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def unsharp_mask(
    img: np.ndarray, blur_ksize=(3, 3), amount: float = 0.8, threshold: int = 10
) -> np.ndarray:
    # Làm mờ ảnh gốc để tách chi tiết
    blurred = cv2.GaussianBlur(img, blur_ksize, 0)

    # Kết hợp ảnh gốc và ảnh mờ theo trọng số
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    # Giữ nguyên vùng đồng nhất (tránh nổi noise)
    low_contrast_mask = (
        np.abs(img.astype(np.int16) - blurred.astype(np.int16)) < threshold
    )
    np.copyto(sharpened, img, where=low_contrast_mask)

    return sharpened


def invert_background(img_input: np.ndarray) -> np.ndarray:
    if img_input.ndim == 2:
        return cv2.bitwise_not(img_input)
    return cv2.bitwise_not(img_input)


def add_white_padding(img_input: np.ndarray, padding_pct: float = 0.02) -> np.ndarray:
    """Add white padding around the image based on percentage of image size.

    padding_pct is applied to width and height separately.
    Example: 0.05 = 5% padding on each side.
    """
    if padding_pct is None or padding_pct <= 0:
        return img_input

    h, w = img_input.shape[:2]
    pad_w = max(0, int(round(w * padding_pct)))
    pad_h = max(0, int(round(h * padding_pct)))

    if img_input.ndim == 2:
        canvas = np.full((h + 2 * pad_h, w + 2 * pad_w), 255, dtype=img_input.dtype)
        canvas[pad_h : pad_h + h, pad_w : pad_w + w] = img_input
        return canvas

    channels = img_input.shape[2]
    canvas = np.full(
        (h + 2 * pad_h, w + 2 * pad_w, channels), 255, dtype=img_input.dtype
    )
    canvas[pad_h : pad_h + h, pad_w : pad_w + w, :] = img_input
    return canvas


def bridge_horizontal(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bridged = cv2.dilate(binary, kernel, iterations=1)
    return bridged


def remove_small_components(image_in: np.ndarray) -> np.ndarray:

    if image_in.ndim == 3 and image_in.shape[2] == 3:
        gray_mask = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    else:
        gray_mask = image_in

    _, bin_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)
    h, w = gray_mask.shape[:2]
    derived_min_area = 10
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )

    keep_mask = np.zeros_like(bin_mask)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= derived_min_area:
            keep_mask[labels == idx] = 255

    cleaned = cv2.bitwise_and(gray_mask, keep_mask)
    if image_in.ndim == 3 and image_in.shape[2] == 3:
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    return cleaned
