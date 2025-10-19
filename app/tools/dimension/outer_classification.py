"""
Các hàm hỗ trợ chọn kích thước outer panel từ bộ ứng viên OCR.
- Chuẩn hoá dữ liệu đầu vào (frame, danh sách số đo).
- Chọn cặp width/height phù hợp dựa trên aspect ratio và thang đo.
"""

import logging
from typing import List, Optional, Tuple
from pydantic import BaseModel
from app.models import Frame, OCRTextBlock

logger = logging.getLogger(__name__)


def _extract_numeric_entries(
    entries: List[OCRTextBlock],
) -> List[Tuple[OCRTextBlock, float]]:
    numeric_entries: List[Tuple[OCRTextBlock, float]] = []
    for entry in entries:
        text = entry.text
        if text is None:
            continue

        try:
            value = float(str(text).strip())
        except (TypeError, ValueError):
            continue

        if value <= 0:
            continue

        numeric_entries.append((entry, value))

    return numeric_entries


class OuterResult(BaseModel):
    width: Optional[OCRTextBlock] = None
    height: Optional[OCRTextBlock] = None
    scale_x: Optional[float] = None
    scale_y: Optional[float] = None
    aspect_diff_ratio: Optional[float] = None


def outer_classify(
    frame: Frame, width_entries: List[OCRTextBlock], height_entries: List[OCRTextBlock]
) -> OuterResult:
    w_px, h_px = frame.w, frame.h
    result_default = OuterResult()

    if w_px <= 0 or h_px <= 0:
        logger.debug("outer_classify → invalid frame dimensions: %s", frame)
        return result_default

    width_candidates = _extract_numeric_entries(width_entries)
    height_candidates = _extract_numeric_entries(height_entries)

    if not width_candidates or not height_candidates:
        logger.debug("outer_classify → thiếu ứng viên width/height hợp lệ")
        return result_default

    aspect_px = w_px / h_px
    pairs = []

    for w_entry, w_mm in width_candidates:
        for h_entry, h_mm in height_candidates:
            if w_mm <= 0 or h_mm <= 0:
                continue
            scale_x = w_mm / w_px
            scale_y = h_mm / h_px
            aspect_mm = w_mm / h_mm
            aspect_diff_ratio = abs(aspect_mm - aspect_px) / max(aspect_px, 1e-6)

            pairs.append(
                {
                    "w_entry": w_entry,
                    "h_entry": h_entry,
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "aspect_diff_ratio": aspect_diff_ratio,
                    "sum_mm": w_mm + h_mm,
                }
            )

    if not pairs:
        logger.debug("outer_classify → không tạo được cặp ứng viên")
        return result_default

    aspect_tol = 0.1
    near_best = [p for p in pairs if p["aspect_diff_ratio"] <= aspect_tol]
    if near_best:
        best_pair = max(near_best, key=lambda p: p["sum_mm"])
    else:
        best_pair = min(pairs, key=lambda p: (p["aspect_diff_ratio"], -p["sum_mm"]))

    result = OuterResult(
        width=best_pair["w_entry"],
        height=best_pair["h_entry"],
        scale_x=round(best_pair["scale_x"], 6),
        scale_y=round(best_pair["scale_y"], 6),
        aspect_diff_ratio=round(best_pair["aspect_diff_ratio"], 6),
    )

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s"
    )
    logger.setLevel(logging.DEBUG)
    sample = {
        "frame": Frame(
            w=90,
            h=400,
            x=0,
            y=0,
            area=0,
            aspect=0,
            fill_ratio=0,
            h_lines_count=0,
            v_lines_count=0,
            total_lines=0,
        ),
        "width_entries": [
            OCRTextBlock(text="548", bounding_box=[], confidence=0),
            OCRTextBlock(text="517", bounding_box=[], confidence=0),
            OCRTextBlock(text="497", bounding_box=[], confidence=0),
        ],
        "height_entries": [
            OCRTextBlock(text="100", bounding_box=[], confidence=0),
            OCRTextBlock(text="100", bounding_box=[], confidence=0),
            OCRTextBlock(text="517", bounding_box=[], confidence=0),
            OCRTextBlock(text="517", bounding_box=[], confidence=0),
            OCRTextBlock(text="517", bounding_box=[], confidence=0),
            OCRTextBlock(text="517", bounding_box=[], confidence=0),
            OCRTextBlock(text="100", bounding_box=[], confidence=0),
            OCRTextBlock(text="2269", bounding_box=[], confidence=0),
        ],
    }

    outer_classify(**sample)
