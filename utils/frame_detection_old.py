from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.normalize_image import normalize_image
from utils.image_process import ImagePipeline, save_stage_image


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


def _draw_rectangles_basic(
    image: np.ndarray,
    rects: Sequence[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    annotated = image.copy()
    for rect in rects:
        x = rect.get("x") or rect.get("left")
        y = rect.get("y") or rect.get("top")
        w = rect.get("w") or rect.get("width")
        h = rect.get("h") or rect.get("height")
        if None in (x, y, w, h):
            continue
        cv2.rectangle(
            annotated,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            color,
            thickness,
        )
    return annotated


def _extract_rect_from_contour(
    cnt: np.ndarray,
    min_area: float,
    min_fill_ratio: float = 0.8,
    max_aspect: float = 10.0,
    min_circularity: float = 0.0004,
    filter: bool = True,
) -> Optional[Dict[str, Any]]:

    area = cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(approx)
    aspect = max(w, h) / max(1, min(w, h))
    peri = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (peri * peri)

    if filter:
        if not filter_contour(
            cnt,
            min_area=min_area,
            min_fill_ratio=min_fill_ratio,
            max_aspect=max_aspect,
            min_circularity=min_circularity,
        ):
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


def dedup_rect_contours(
    contours: Sequence[np.ndarray],
    img_w: int,
    img_h: int,
    area_ratio: float = 0.0025,
    min_fill_ratio: float = 0.8,
    max_aspect: float = 10.0,
    min_circularity: float = 0.0004,
    margin: int = 25,
    max_ratio: float = 0.92,
    area_ratio_threshold: float = 1.02,
    duplicate_pos_tol: int = 5,
    duplicate_size_tol: float = 0.05,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    if not contours:
        return [], []

    min_area = max(int(float(img_w * img_h) * max(area_ratio, 0.0)), 0)
    contour_pairs: List[Tuple[Dict[str, Any], np.ndarray]] = []

    for cnt in contours:
        rect = _extract_rect_from_contour(
            cnt,
            min_area=min_area,
            min_fill_ratio=min_fill_ratio,
            max_aspect=max_aspect,
            min_circularity=min_circularity,
            filter=True,
        )
        if rect is None:
            continue
        contour_pairs.append((rect, cnt))

    if not contour_pairs:
        return [], []

    rects = [pair[0] for pair in contour_pairs]
    rect_map = {id(pair[0]): pair[1] for pair in contour_pairs}

    dedup_rects = _filter_and_dedup_rects(
        rects,
        img_w=img_w,
        img_h=img_h,
        margin=margin,
        max_ratio=max_ratio,
        area_ratio_threshold=area_ratio_threshold,
        duplicate_pos_tol=duplicate_pos_tol,
        duplicate_size_tol=duplicate_size_tol,
    )

    dedup_contours = [
        rect_map[id(rect)] for rect in dedup_rects if id(rect) in rect_map
    ]

    return dedup_contours, dedup_rects


def remove_non_rect(
    binary: np.ndarray,
    area_ratio: float = 0.0025,
):
    # Làm sạch nhiễu nhỏ trước khi tìm contour
    cleaned = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=1
    )

    def _rect_bounds(rect: Dict[str, Any]) -> Tuple[int, int, int, int]:
        x1 = int(rect["x"])
        y1 = int(rect["y"])
        x2 = int(rect["x"] + rect["w"])
        y2 = int(rect["y"] + rect["h"])
        return x1, y1, x2, y2

    def _rects_overlap_or_close(
        r1: Dict[str, Any], r2: Dict[str, Any], padding: int = 10
    ) -> bool:
        ax1, ay1, ax2, ay2 = _rect_bounds(r1)
        bx1, by1, bx2, by2 = _rect_bounds(r2)
        ax1 -= padding
        ay1 -= padding
        ax2 += padding
        ay2 += padding
        bx1 -= padding
        by1 -= padding
        bx2 += padding
        by2 += padding
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        return inter_x1 <= inter_x2 and inter_y1 <= inter_y2

    def _union_rects(r1: Dict[str, Any], r2: Dict[str, Any]) -> Dict[str, Any]:
        ax1, ay1, ax2, ay2 = _rect_bounds(r1)
        bx1, by1, bx2, by2 = _rect_bounds(r2)
        ux1 = min(ax1, bx1)
        uy1 = min(ay1, by1)
        ux2 = max(ax2, bx2)
        uy2 = max(ay2, by2)
        w = max(0, ux2 - ux1)
        h = max(0, uy2 - uy1)
        area = float(w * h)
        aspect = float(max(w, h) / max(1, min(w, h))) if w and h else 0.0
        merged = dict(r1)
        merged.update(
            {
                "x": ux1,
                "y": uy1,
                "w": w,
                "h": h,
                "area": area,
                "aspect": aspect,
            }
        )
        merged.pop("label", None)
        return merged

    def _merge_rectangles(
        rects: Sequence[Dict[str, Any]], padding: int = 10
    ) -> List[Dict[str, Any]]:
        if not rects:
            return []
        rect_list = [dict(r) for r in rects]
        merged: List[Dict[str, Any]] = []
        consumed = [False] * len(rect_list)
        for i in range(len(rect_list)):
            if consumed[i]:
                continue
            current = dict(rect_list[i])
            consumed[i] = True
            expanded = True
            while expanded:
                expanded = False
                for j in range(len(rect_list)):
                    if consumed[j]:
                        continue
                    if _rects_overlap_or_close(current, rect_list[j], padding):
                        current = _union_rects(current, rect_list[j])
                        consumed[j] = True
                        expanded = True
            merged.append(current)
        return merged

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours), "contours found")
    filtered_contours, rects = dedup_rect_contours(
        contours,
        img_w=binary.shape[1],
        img_h=binary.shape[0],
        area_ratio=area_ratio,
        min_fill_ratio=0.6,
        max_aspect=8.0,
    )
    print(len(filtered_contours), "filtered contours found")

    merged_rects = _merge_rectangles(rects, padding=15)

    rect_mask = np.zeros_like(binary)
    for rect in merged_rects:
        x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
        cv2.rectangle(rect_mask, (x, y), (x + w, y + h), 255, -1)

    if not merged_rects and filtered_contours:
        cv2.drawContours(rect_mask, filtered_contours, -1, 255, -1)

    masked = cv2.bitwise_and(binary, rect_mask)
    return masked


def detect_frames_pipeline(
    image: np.ndarray,
    output_path: Optional[Path] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if image is None or (hasattr(image, "size") and image.size == 0):
        raise ValueError("detect_frames_pipeline: input image is empty or None")
    pipeline = ImagePipeline(output_path)
    pipeline.add(
        "otsu",
        lambda image: cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1],
    )
    pipeline.add("remove_non_rect", lambda image: remove_non_rect(image))
    pipeline.add("detect_by_hough", lambda image: detect_by_hough(image)[1])
    annotated = pipeline.run(image)
    print("save folder", output_path)
    return annotated


if __name__ == "__main__":
    # img_path = Path("outputs/5d430f41d60b422a8385dcbc2e96c66f/Block 0/00_origin.png")
    # img_path = Path("outputs/panel_agent/analyzer/Block 0/14_white_padding.png")
    # img_path = Path("outputs/panel_analyze_pipeline/Block 2/00_origin.png")
    # img_path = Path("outputs/7a08cabecf654f0b9f8b916fcbbe4c56/Block 0/00_origin.png")
    img_path = Path("outputs/24a94f1546374c16b54e1e411cc96010/Block 1/00_origin.png")
    # img_path = Path("assets/block/image.png")

    output_dir = Path("outputs", "frame_detection")
    # if output_dir.exists():
    #     shutil.rmtree(output_dir)
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
    print("Image loaded successfully, Starting frame detection...")
    annotated = detect_frames_pipeline(image, output_path=output_dir)
    print("Frame detection completed successfully.")
