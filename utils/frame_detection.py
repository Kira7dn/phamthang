from pathlib import Path
import math
import shutil
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.normalize_image import normalize_image
from utils.image_process import (
    ImagePipeline,
    clahe,
    enhance_lines,
    invert_background,
    morph_close,
    morph_open,
    normalize_bg,
    resize_with_limit,
    to_gray,
)


def _resize_for_detection(
    image: np.ndarray, max_dim: int = 1600
) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        image = cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image, scale


def _adaptive_close(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    k = max(3, int(round(min(h, w) / 80)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)


def _draw_rectangles(
    image: np.ndarray,
    rects: Sequence[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 0),
    center_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]
    for rect in rects:
        corners = rect.get("corners")
        if corners is None:
            x, y = rect.get("x"), rect.get("y")
            w, h = rect.get("w"), rect.get("h")
            if None in (x, y, w, h):
                continue
            corners = [
                (int(x), int(y)),
                (int(x + w), int(y)),
                (int(x + w), int(y + h)),
                (int(x), int(y + h)),
            ]
        pts = np.array(corners, dtype=np.float32)
        pts_int = np.array(pts, dtype=np.int32)
        if pts_int.size == 0:
            continue
        cv2.polylines(annotated, [pts_int], True, color, thickness)
        center = rect.get("center")
        if center is None:
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            center = (float(np.mean(x_vals)), float(np.mean(y_vals)))
        if center:
            cv2.circle(
                annotated,
                (int(round(center[0])), int(round(center[1]))),
                3,
                center_color,
                -1,
            )
        label = rect.get("label")
        if label is not None:
            x_min = int(np.min(pts_int[:, 0]))
            y_min = int(np.min(pts_int[:, 1]))
            text_x = min(max(x_min, 0), img_w - 1)
            text_y = min(max(y_min + 20, 20), img_h - 5)
            cv2.putText(
                annotated,
                str(label),
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return annotated


def detect_by_hough(
    image: np.ndarray, min_area: float = 1000, merge_thresh: int = 10
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Phát hiện hình chữ nhật thẳng hàng (vuông góc, xếp chồng hoặc cạnh nhau)
    bằng cách dò các line ngang/dọc và ghép thành hộp.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis = image.copy()
    else:
        gray = image
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120, minLineLength=50, maxLineGap=5
    )
    if lines is None:
        return [], cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    lines = lines.reshape(-1, 4)
    h_lines, v_lines = [], []

    for x1, y1, x2, y2 in lines:
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 10 or angle > 170:
            h_lines.append((min(y1, y2), min(x1, x2), max(x1, x2)))  # y, x_start, x_end
        elif 80 < angle < 100:
            v_lines.append((min(x1, x2), min(y1, y2), max(y1, y2)))  # x, y_start, y_end

    # Gom nhóm các đường gần nhau
    def merge_lines(lines, axis_idx=0):
        lines.sort(key=lambda x: x[axis_idx])
        merged = []
        for l in lines:
            if not merged or abs(l[axis_idx] - merged[-1][axis_idx]) > merge_thresh:
                merged.append(l)
        return merged

    h_lines = merge_lines(h_lines, 0)
    v_lines = merge_lines(v_lines, 0)

    rects = []
    for i, (x1, _, _) in enumerate(v_lines):
        for x2, _, _ in v_lines[i + 1 :]:
            if abs(x2 - x1) < 10:  # quá gần, bỏ qua
                continue

            for k, (y1, _, _) in enumerate(h_lines):
                for y2, _, _ in h_lines[k + 1 :]:
                    if abs(y2 - y1) < 10:
                        continue

                    w, h = abs(x2 - x1), abs(y2 - y1)
                    area = w * h
                    if area < min_area:
                        continue

                    rects.append(
                        {
                            "x": int(min(x1, x2)),
                            "y": int(min(y1, y2)),
                            "width": int(w),
                            "height": int(h),
                            "area": float(area),
                        }
                    )

    # Vẽ để debug
    for r in rects:
        cv2.rectangle(
            vis,
            (r["x"], r["y"]),
            (r["x"] + r["width"], r["y"] + r["height"]),
            (0, 255, 0),
            2,
        )

    return rects, vis


def is_closed_rect(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0 or area == 0:
        return False
    circularity = 4 * np.pi * area / (peri * peri)
    return circularity > 0.4  # đủ kín (có thể chỉnh 0.4–0.6 tùy ảnh)


def _extract_rect_from_contour(
    cnt: np.ndarray,
    min_area: float,
    min_fill_ratio: float = 0.8,
    max_aspect: float = 10.0,
    min_circularity: float = 0.0004,
) -> Optional[Dict[str, Any]]:
    area = cv2.contourArea(cnt)
    if area < min_area:
        return None

    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) != 4:
        return None

    x, y, w, h = cv2.boundingRect(approx)
    rect_area = w * h
    if rect_area == 0 or area / rect_area < min_fill_ratio:
        return None

    aspect = max(w, h) / max(1, min(w, h))
    if aspect > max_aspect:
        return None

    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return None
    circularity = 4 * np.pi * area / (peri * peri)
    if circularity < min_circularity:
        return None

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": float(area),
        "aspect": float(aspect),
    }


def _filter_and_dedup_rects(
    rects: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    margin: int = 25,
    max_ratio: float = 0.92,
    area_ratio_threshold: float = 1.02,
    duplicate_pos_tol: int = 5,
    duplicate_size_tol: float = 0.05,
) -> List[Dict[str, Any]]:
    if len(rects) <= 1:
        return rects

    def _contains(outer: Dict[str, Any], inner: Dict[str, Any]) -> bool:
        return (
            outer["x"] <= inner["x"] + margin
            and outer["y"] <= inner["y"] + margin
            and outer["x"] + outer["w"] >= inner["x"] + inner["w"] - margin
            and outer["y"] + outer["h"] >= inner["y"] + inner["h"] - margin
        )

    working = rects[:]
    max_idx = max(range(len(working)), key=lambda i: working[i]["area"])
    candidate = working[max_idx]
    if all(
        _contains(candidate, working[idx])
        for idx in range(len(working))
        if idx != max_idx
    ):
        working.pop(max_idx)

    filtered: List[Dict[str, Any]] = []
    for idx, rect in enumerate(working):
        h_ratio = rect["h"] / float(max(1, img_h))
        w_ratio = rect["w"] / float(max(1, img_w))
        if h_ratio > max_ratio or w_ratio > max_ratio:
            continue
        contains_inner = any(
            _contains(rect, working[j])
            and rect["area"] >= working[j]["area"] * area_ratio_threshold
            for j in range(len(working))
            if j != idx
        )
        if contains_inner:
            continue
        filtered.append(rect)

    if filtered:
        working = filtered

    if len(working) <= 1:
        return working

    working.sort(key=lambda r: r["area"], reverse=True)
    deduped: List[Dict[str, Any]] = []
    for rect in working:
        is_duplicate = False
        for kept in deduped:
            if (
                abs(rect["x"] - kept["x"]) < duplicate_pos_tol
                and abs(rect["y"] - kept["y"]) < duplicate_pos_tol
                and abs(rect["w"] - kept["w"]) / max(1.0, kept["w"])
                < duplicate_size_tol
                and abs(rect["h"] - kept["h"]) / max(1.0, kept["h"])
                < duplicate_size_tol
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(rect)

    return deduped


def detect_by_contours(
    binary: np.ndarray, canvas: Optional[np.ndarray] = None, area_ratio: float = 0.0025
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects: List[Dict[str, Any]] = []
    adaptive_min_area = max(
        int(float(binary.shape[0] * binary.shape[1]) * max(area_ratio, 0.0)), 0
    )
    debug_img = canvas.copy() if canvas is not None else binary.copy()
    if debug_img.ndim == 2:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
    img_h, img_w = debug_img.shape[:2]
    for cnt in contours:
        rect = _extract_rect_from_contour(cnt, min_area=adaptive_min_area)
        if rect is None:
            continue
        rects.append(rect)
    rects = _filter_and_dedup_rects(rects, img_w=img_w, img_h=img_h)

    for idx, rect in enumerate(rects, start=1):
        rect.setdefault("label", idx)

    if rects:
        debug_img = _draw_rectangles(debug_img, rects)
    print(rects)
    print(len(rects), "found")
    print("----------------")

    return debug_img, rects


def remove_non_rect(
    binary: np.ndarray,
    area_ratio: float = 0.0025,
):
    # Tìm contour ngoài cùng
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    rects_dict: List[Dict[str, Any]] = []

    adaptive_min_area = max(
        int(float(binary.shape[0] * binary.shape[1]) * max(area_ratio, 0.0)), 0
    )

    for cnt in contours:
        rect = _extract_rect_from_contour(cnt, min_area=adaptive_min_area)
        if rect is None:
            continue
        rects_dict.append(rect)
    filtered_rects = _filter_and_dedup_rects(
        rects_dict, img_w=binary.shape[1], img_h=binary.shape[0]
    )
    rects: List[Tuple[int, int, int, int]] = []
    for rect in filtered_rects:
        x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        rects.append((x, y, w, h))

    return mask, rects


def detect_frames_pipeline(
    image: np.ndarray,
    output_path: Optional[Path] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if image is None or (hasattr(image, "size") and image.size == 0):
        raise ValueError("detect_frames_pipeline: input image is empty or None")
    pipeline = ImagePipeline(output_path)
    # pipeline.add("reverse", lambda image: cv2.bitwise_not(image))
    pipeline.add(
        "otsu",
        lambda image: cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1],
    )
    # Làm dày và tách vùng khép kín
    # pipeline.add(
    #     "dilate",
    #     lambda image: cv2.dilate(image, np.ones((5, 5), np.uint8), iterations=1),
    # )
    # pipeline.add(
    #     "erode", lambda image: cv2.erode(image, np.ones((3, 3), np.uint8), iterations=2)
    # )
    pipeline.add("remove_non_rect", lambda image: remove_non_rect(image)[0])
    pipeline.add(
        "detect_contours",
        lambda img: detect_by_contours(binary=img, canvas=image)[0],
    )
    # pipeline.add("detect_hough_rects", lambda img: detect_by_hough(img)[1])
    annotated = pipeline.run(image)
    print("save folder", output_path)
    return annotated


if __name__ == "__main__":
    # img_path = Path("outputs/5d430f41d60b422a8385dcbc2e96c66f/Block 0/00_origin.png")
    # img_path = Path("outputs/panel_agent/analyzer/Block 0/14_white_padding.png")
    # img_path = Path("outputs/panel_analyze_pipeline/Block 2/00_origin.png")
    # img_path = Path("outputs/7a08cabecf654f0b9f8b916fcbbe4c56/Block 0/00_origin.png")
    # img_path = Path("outputs/24a94f1546374c16b54e1e411cc96010/Block 1/00_origin.png")
    img_path = Path("assets/block/image.png")

    output_dir = Path("outputs", "frame_detection")
    # if output_dir.exists():
    #     shutil.rmtree(output_dir)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(
            f"Không đọc được ảnh: {img_path}. Kiểm tra đường dẫn và quyền truy cập."
        )
    normalized_image = normalize_image(image, output_dir / "normalized_image")
    if normalized_image is None:
        raise FileNotFoundError(
            f"Không đọc được ảnh: {img_path}. Kiểm tra đường dẫn và quyền truy cập."
        )
    print("Image loaded successfully, Starting frame detection...")
    annotated = detect_frames_pipeline(normalized_image, output_path=output_dir)
    print("Frame detection completed successfully.")
