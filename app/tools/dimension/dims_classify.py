import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.models import OCRTextBlock, OCRVertex, Frame, SimplifiedPanel, SimplifiedFrame
from app.tools.dimension.score_dims import calculate_quality_scores
from app.tools.dimension.normalize_frames import normalize_frames_fast
from app.tools.dimension.scale_estimation import estimate_scale
from app.tools.dimension.outer_candidates import generate_outer_candidates


# ============================================================================
# HELPER FUNCTIONS - Text Processing
# ============================================================================

logger = logging.getLogger(__name__)


def _calculate_image_bounds(
    text_blocks: List[OCRTextBlock],
) -> tuple[float, float, float, float]:
    x_vals = [v.x or 0 for block in text_blocks for v in block.bounding_box]
    y_vals = [v.y or 0 for block in text_blocks for v in block.bounding_box]
    return min(x_vals), max(x_vals), min(y_vals), max(y_vals)


# ============================================================================
# HELPER FUNCTIONS - Text Processing
# ============================================================================


def _is_valid_numeric_text(text: str) -> bool:
    """Kiểm tra text có phải là số hợp lệ (không chứa chữ cái)."""
    return text and not any(c.isalpha() for c in text)


def _parse_float(text: str) -> float:
    """Parse text thành float, trả về 0 nếu không hợp lệ."""
    try:
        return float(str(text).strip()) if str(text).replace(".", "").isdigit() else 0.0
    except Exception:
        return 0.0


def _create_text_entry(block: OCRTextBlock) -> Optional[Dict]:
    """Tạo entry từ text block với thông tin tọa độ và kích thước."""
    if len(block.bounding_box) < 4:
        return None

    xs = [v.x or 0 for v in block.bounding_box]
    ys = [v.y or 0 for v in block.bounding_box]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return {
        "text": block.text,
        "cx": (min(xs) + max(xs)) / 2.0,
        "cy": (min(ys) + max(ys)) / 2.0,
        "w": width,
        "h": height,
        "aspect": width / (height + 1e-6),
    }


def _is_near_border(
    cx: float,
    cy: float,
    bounds: tuple[float, float, float, float],
    margins: tuple[float, float, float, float],
) -> bool:
    """Kiểm tra tâm text có nằm gần biên ảnh hay không."""
    x_min, x_max, y_min, y_max = bounds
    margin_left, margin_right, margin_top, margin_bottom = margins

    return (
        cx - x_min < margin_left
        or x_max - cx < margin_right
        or cy - y_min < margin_top
        or y_max - cy < margin_bottom
    )


# ============================================================================
# HELPER FUNCTIONS - Outer/Inner Classification
# ============================================================================


def _collect_outer_inner(text_blocks: List[OCRTextBlock]) -> Dict[str, List[Dict]]:
    """
    Phân loại text thành outer và inner dựa trên vị trí:
    - Outer: gần mép ảnh (margin)
    - Inner: nằm bên trong
    """
    if not text_blocks:
        return {
            "left_outer": [],
            "right_outer": [],
            "top_outer": [],
            "bottom_outer": [],
            "inner": [],
        }

    # Tính toán biên ảnh
    x_min, x_max, y_min, y_max = _calculate_image_bounds(text_blocks)
    if x_min == x_max or y_min == y_max:
        return {
            "left_outer": [],
            "right_outer": [],
            "top_outer": [],
            "bottom_outer": [],
            "inner": [],
        }

    # Tạo danh sách tất cả entries
    all_entries = []
    for block in text_blocks:
        if not _is_valid_numeric_text(block.text):
            continue
        entry = _create_text_entry(block)
        if entry:
            all_entries.append(entry)

    if not all_entries:
        return {
            "left_outer": [],
            "right_outer": [],
            "top_outer": [],
            "bottom_outer": [],
            "inner": [],
        }

    def _dedup(entries: List[Dict]) -> List[Dict]:
        seen_ids = set()
        deduped = []
        for e in entries:
            eid = id(e)
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            deduped.append(e)
        return deduped

    left_outer, right_outer, top_outer, bottom_outer, inner = [], [], [], [], []

    for entry in all_entries:
        cx, cy = entry["cx"], entry["cy"]
        is_outer = False

        if cx - x_min < entry["w"]:
            left_outer.append(entry)
            is_outer = True
        if x_max - cx < entry["w"]:
            right_outer.append(entry)
            is_outer = True
        if cy - y_min < entry["h"]:
            top_outer.append(entry)
            is_outer = True
        if y_max - cy < entry["h"]:
            bottom_outer.append(entry)
            is_outer = True
        if not is_outer:
            inner.append(entry)

    left_outer = _dedup(left_outer)
    right_outer = _dedup(right_outer)
    top_outer = _dedup(top_outer)
    bottom_outer = _dedup(bottom_outer)
    inner.sort(key=lambda x: (x["cy"], x["cx"]))

    return {
        "left_outer": left_outer,
        "right_outer": right_outer,
        "top_outer": top_outer,
        "bottom_outer": bottom_outer,
        "inner": inner,
    }


# ============================================================================
# HELPER FUNCTIONS - Frame Alignment
# ============================================================================


def _is_aligned(entry: Dict, frame: Frame, threshold: int = 10) -> bool:
    """Kiểm tra entry có nằm cùng hàng hoặc cột với frame (trong ngưỡng sai số)."""
    cx, cy = entry.get("cx", 0.0), entry.get("cy", 0.0)

    # Ngưỡng động dựa trên kích thước frame
    threshold_x = max(threshold, int(0.15 * frame.w))
    threshold_y = max(threshold, int(0.15 * frame.h))

    # Kiểm tra cùng cột hoặc cùng hàng
    in_column = frame.x - threshold_x <= cx <= frame.x + frame.w + threshold_x
    in_row = frame.y - threshold_y <= cy <= frame.y + frame.h + threshold_y

    return in_column or in_row


# ============================================================================
# HELPER FUNCTIONS - Number Extraction
# ============================================================================


def _extract_numbers_from_blocks(text_blocks: List[OCRTextBlock]) -> List[float]:
    """Trích xuất tất cả các số từ text blocks."""
    return [
        float(b.text)
        for b in text_blocks
        if b.text and b.text.replace(".", "").isdigit()
    ]


def _extract_numbers_from_entries(entries: List[Dict]) -> List[float]:
    """Trích xuất các số từ danh sách entries."""
    return [
        float(e["text"])
        for e in entries
        if e.get("text", "").replace(".", "").isdigit()
    ]


# ============================================================================
# HELPER FUNCTIONS - Frame Processing
# ============================================================================


def _convert_frames_to_dict(frames: List[Frame]) -> List[Dict]:
    """Chuyển đổi frames sang định dạng dict cho normalize_frames_fast."""
    return [
        {
            "x": f.x,
            "y": f.y,
            "w": f.w,
            "h": f.h,
            "area": f.area,
            "aspect": f.aspect,
        }
        for f in frames
    ]


def _get_pixel_candidates(
    normalized_frames: List[Dict],
) -> tuple[List[float], List[float]]:
    """Lấy danh sách pixel candidates riêng cho width và height từ normalized frames."""
    width_candidates: List[float] = []
    height_candidates: List[float] = []
    for nf in normalized_frames:
        w_val = nf.get("normalized_w")
        if isinstance(w_val, (int, float)) and w_val > 0:
            width_candidates.append(float(w_val))

        h_val = nf.get("normalized_h")
        if isinstance(h_val, (int, float)) and h_val > 0:
            height_candidates.append(float(h_val))

    return width_candidates, height_candidates


# ============================================================================
# HELPER FUNCTIONS - Candidate Selection
# ============================================================================


def _select_best_candidate(
    candidates: List[Dict], estimated_mm: float, scale: Optional[float]
) -> str:
    """Chọn candidate tốt nhất dựa trên kích thước ước tính và score."""
    if not candidates:
        return str(int(estimated_mm))

    if isinstance(scale, (int, float)) and scale:
        # Ưu tiên candidate gần nhất với kích thước ước tính
        best = min(
            candidates,
            key=lambda c: (
                abs(float(c.get("value", 0)) - estimated_mm),
                -float(c.get("score", 0.0)),
            ),
        )
    else:
        best = candidates[0]

    return str(int(best.get("value", estimated_mm)))


# ============================================================================
# HELPER FUNCTIONS - Inner Candidates Processing
# ============================================================================


def _filter_inner_candidates(
    entries: List[Dict], frame: Frame, outer_width: str, outer_height: str
) -> List[Dict]:
    """Lọc và chuẩn bị inner candidates cho frame."""
    # Lấy entries cùng hàng/cột với frame
    frame_entries = [e for e in entries if _is_aligned(e, frame)]

    # Chỉ lấy số nằm trong frame height range
    margin_y = max(30, int(0.12 * frame.h))
    inner_candidates = [
        e
        for e in frame_entries
        if e.get("text", "").isdigit()
        and (frame.y - margin_y) <= e.get("cy", 0.0) <= (frame.y + frame.h + margin_y)
    ]

    # Sắp xếp theo Y (từ trên xuống)
    inner_candidates.sort(key=lambda e: e.get("cy", 0.0))

    # Loại bỏ outer dimensions
    to_remove = {outer_height, outer_width}
    return [e for e in inner_candidates if e.get("text") not in to_remove]


def _select_best_column_group(
    inner_entries: List[Dict], frame_width: float, outer_height: str
) -> List[Dict]:
    """Chọn nhóm cột tốt nhất dựa trên tổng gần outer_height."""
    if len(inner_entries) < 3:
        return inner_entries

    # Phân nhóm theo cột X
    cxs = [e.get("cx", 0.0) for e in inner_entries]
    x_tolerance = max(20.0, 0.20 * frame_width)
    left_seed, right_seed = min(cxs), max(cxs)

    left_group = [
        e for e in inner_entries if abs(e.get("cx", 0.0) - left_seed) <= x_tolerance
    ]
    right_group = [
        e for e in inner_entries if abs(e.get("cx", 0.0) - right_seed) <= x_tolerance
    ]

    # Tính tổng của mỗi nhóm
    def group_sum(group: List[Dict]) -> float:
        return sum(_parse_float(e.get("text", "")) for e in group)

    target_height = _parse_float(outer_height)
    left_sum, right_sum = group_sum(left_group), group_sum(right_group)
    left_error = abs(left_sum - target_height)
    right_error = abs(right_sum - target_height)

    # Chọn nhóm tốt nhất
    if abs(left_error - right_error) <= 1.0:
        # Nếu bằng nhau, chọn nhóm nhiều phần tử hơn
        return left_group if len(left_group) >= len(right_group) else right_group
    else:
        return left_group if left_error < right_error else right_group


def _deduplicate_entries(entries: List[Dict]) -> List[Dict]:
    """Loại bỏ các entries trùng lặp liên tiếp và gần nhau."""
    deduped = []
    for entry in entries:
        if not deduped:
            deduped.append(entry)
            continue

        prev = deduped[-1]
        # Nếu trùng giá trị và gần nhau theo Y (<30px), bỏ qua
        if (
            entry["text"] == prev["text"]
            and abs(entry.get("cy", 0.0) - prev.get("cy", 0.0)) < 30
        ):
            continue

        deduped.append(entry)
    return deduped


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def classify_dimensions(
    text_blocks: List[OCRTextBlock],
    frames: List[Frame],
    output_dir: Optional[Path] = None,
) -> List[SimplifiedFrame]:
    """
    Phân loại kích thước cho các frame dựa trên OCR text blocks.

    Args:
        text_blocks: Danh sách các text block từ OCR
        frames: Danh sách các frame cần phân loại kích thước
        output_dir: Thư mục lưu kết quả (nếu có)

    Returns:
        Danh sách SimplifiedFrame với thông tin kích thước
    """
    # Bước 1: Phân loại outer/inner entries
    dim_info = _collect_outer_inner(text_blocks)
    outer_entries = []
    for key in ("left_outer", "right_outer", "top_outer", "bottom_outer"):
        outer_entries.extend(dim_info.get(key, []))

    # Loại bỏ trùng lặp giữa các cạnh
    seen_outer = set()
    deduped_outer = []
    for entry in outer_entries:
        e_id = id(entry)
        if e_id in seen_outer:
            continue
        seen_outer.add(e_id)
        deduped_outer.append(entry)
    outer_entries = deduped_outer

    inner_entries_all = dim_info["inner"]
    entries = outer_entries + inner_entries_all

    # Bước 2: Trích xuất các số
    numbers_all = _extract_numbers_from_blocks(text_blocks)
    outer_numbers = _extract_numbers_from_entries(outer_entries)

    # Bước 3: Chuẩn hóa frames và ước tính tỷ lệ
    frames_dict = _convert_frames_to_dict(frames)
    normalized_frames = normalize_frames_fast(frames_dict, tolerance_px=5)

    width_pixels, height_pixels = _get_pixel_candidates(normalized_frames)
    width_numbers = [n for n in outer_numbers if n == n] if outer_numbers else []
    height_numbers = [n for n in outer_numbers if n == n] if outer_numbers else []

    if not width_numbers or not height_numbers:
        # Nếu không có outer numbers, fallback bằng cách dùng tất cả numbers cho cả width/height
        width_numbers = numbers_all
        height_numbers = numbers_all

    combined_numbers = list(width_numbers) + list(height_numbers)
    if not combined_numbers:
        combined_numbers = numbers_all

    scale_numbers = sorted(
        {float(n) for n in combined_numbers if isinstance(n, (int, float))}
    )
    if not scale_numbers:
        scale_numbers = numbers_all

    scale_result = estimate_scale(
        width_pixels,
        height_pixels,
        width_numbers,
        height_numbers,
        inlier_rel_tol=0.12,
    )
    scale = scale_result.get("scale")

    # Bước 4: Tạo outer candidates
    if not scale:
        raise ValueError("Scale estimation failed")

    outer_cands_raw = generate_outer_candidates(
        normalized_frames,
        scale_numbers,
        float(scale),
        k=3,
        rel_tol=0.20,
        window_ratio=0.5,
    )

    # Bước 5: Xử lý từng frame
    result: Dict[int, Dict] = {}

    def _build_candidates_from_numbers(
        nums: List[float], estimated: float, rel_tol: float = 0.20, limit: int = 3
    ) -> List[Dict]:
        candidates_map: Dict[float, Dict] = {}
        for num in nums:
            if not isinstance(num, (int, float)):
                continue
            value = float(num)
            if value <= 0:
                continue
            rel_err = abs(value - estimated) / max(value, estimated)
            if rel_tol > 0 and rel_err > rel_tol:
                continue
            score = 1.0 - (rel_err / rel_tol) if rel_tol > 0 else 0.0
            candidates_map[value] = {
                "value": value,
                "score": float(max(0.0, score)),
                "rel_err": float(rel_err),
                "shared": False,
            }
        candidates = list(candidates_map.values())
        candidates.sort(key=lambda x: (-x.get("score", 0.0), x.get("rel_err", 0.0)))
        return candidates[:limit]

    def _merge_candidate_lists(
        primary: List[Dict], fallback: List[Dict], limit: int = 3
    ) -> List[Dict]:
        merged: Dict[float, Dict] = {}
        for cand in primary:
            value = float(cand.get("value", 0.0))
            if value <= 0:
                continue
            merged[value] = cand
        for cand in fallback:
            value = float(cand.get("value", 0.0))
            if value <= 0 or value in merged:
                continue
            merged[value] = cand
        merged_list = list(merged.values())
        merged_list.sort(
            key=lambda x: (-float(x.get("score", 0.0)), float(x.get("rel_err", 0.0)))
        )
        return merged_list[:limit]

    for i, frame in enumerate(frames):
        base_cands = (
            outer_cands_raw[i]
            if i < len(outer_cands_raw)
            else {"w_cands": [], "h_cands": [], "pairs": []}
        )

        left_entries = [
            entry
            for entry in dim_info.get("left_outer", [])
            if _is_aligned(entry, frame)
        ]
        right_entries = [
            entry
            for entry in dim_info.get("right_outer", [])
            if _is_aligned(entry, frame)
        ]
        top_entries = [
            entry
            for entry in dim_info.get("top_outer", [])
            if _is_aligned(entry, frame)
        ]
        bottom_entries = [
            entry
            for entry in dim_info.get("bottom_outer", [])
            if _is_aligned(entry, frame)
        ]

        frame_width_entries = left_entries + right_entries
        frame_height_entries = top_entries + bottom_entries

        frame_width_numbers = _extract_numbers_from_entries(frame_width_entries)
        frame_height_numbers = _extract_numbers_from_entries(frame_height_entries)

        width_scale_numbers = frame_width_numbers or width_numbers
        height_scale_numbers = frame_height_numbers or height_numbers

        frame_scale_result = estimate_scale(
            [float(frame.w)],
            [float(frame.h)],
            width_scale_numbers,
            height_scale_numbers,
            inlier_rel_tol=0.12,
        )
        frame_scale = frame_scale_result.get("scale") or scale

        estimated_w = float(frame.w) * float(frame_scale)
        estimated_h = float(frame.h) * float(frame_scale)

        frame_w_candidates = _build_candidates_from_numbers(
            frame_width_numbers, estimated_w
        )
        frame_h_candidates = _build_candidates_from_numbers(
            frame_height_numbers, estimated_h
        )

        base_w_cands = base_cands.get("w_cands", [])
        base_h_cands = base_cands.get("h_cands", [])

        w_candidates = _merge_candidate_lists(frame_w_candidates, base_w_cands)
        h_candidates = _merge_candidate_lists(frame_h_candidates, base_h_cands)

        if not w_candidates:
            w_candidates = base_w_cands
        if not h_candidates:
            h_candidates = base_h_cands

        pairs = [
            {"width_cand": w_cand, "height_cand": h_cand}
            for w_cand in w_candidates
            for h_cand in h_candidates
        ]
        if not pairs:
            pairs = base_cands.get("pairs", [])

        chosen_w = _select_best_candidate(w_candidates, estimated_w, frame_scale)
        chosen_h = _select_best_candidate(h_candidates, estimated_h, frame_scale)

        frame_entries = [e for e in entries if _is_aligned(e, frame)]
        inner_entries = _filter_inner_candidates(
            frame_entries, frame, chosen_w, chosen_h
        )
        inner_entries = _select_best_column_group(inner_entries, frame.w, chosen_h)
        inner_entries = _deduplicate_entries(inner_entries)

        inner_heights = [e["text"] for e in inner_entries]
        selected_ids = {id(e) for e in inner_entries}
        unselected_entries = [e for e in frame_entries if id(e) not in selected_ids]

        result[i] = {
            "width": chosen_w,
            "height": chosen_h,
            "inner_heights": inner_heights,
            "frame_px_w": frame.w,
            "frame_px_h": frame.h,
            "outer_pairs": pairs,
            "unselected": unselected_entries,
        }

    # Bước 6: Tính toán chỉ số chất lượng
    quality_scores = calculate_quality_scores(result, frames, numbers_all)
    # Bước 7: Tạo SimplifiedFrame list
    simplified_frames = []
    for i in range(len(frames)):
        dims = result.get(i, {})

        inner_heights = [float(x) for x in dims.get("inner_heights", [])]
        outer_w = float(dims.get("width", 0))
        outer_h = float(dims.get("height", 0))

        panels = [
            SimplifiedPanel(
                outer_width=outer_w,
                outer_height=outer_h,
                inner_heights=inner_heights,
            )
        ]

        simplified_frames.append(
            SimplifiedFrame(
                id=str(i),
                panels=panels,
                quality_scores=quality_scores,
            )
        )

    # Bước 8: Lưu kết quả nếu có output_dir
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dimensions_classified.json"

        results_data = {
            "text_blocks": [block.model_dump() for block in text_blocks],
            "frames": [frame.model_dump() for frame in frames],
            "panels": [
                panel.model_dump() for sf in simplified_frames for panel in sf.panels
            ],
            "quality_scores": quality_scores,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

    return simplified_frames


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(message)s")

    # Iterate all JSON samples in sample directory
    sample_dir = Path("app/tools/dimension/sample")
    json_files = sorted(sample_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON samples found in {sample_dir}")

    overall_ok = True
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as sf:
            data = json.load(sf)

        case = data
        base_output_dir = Path("outputs", "dims_classify", json_file.stem)

        print(f"\n=== Running sample: {json_file} ===")
        print(f"Output dir: {base_output_dir}")

        cid = Path(json_file).stem
        cname = case.get("name", "")
        frames = case.get("frames", [])
        expected = case.get("panels", case.get("expected", []))
        text_blocks = case.get("text_blocks", [])

        print(f"\n-- Sample: {cid} - {cname}")

        # Convert dicts to models
        text_block_models = [
            OCRTextBlock(
                text=tb["text"],
                bounding_box=[
                    OCRVertex(x=v["x"], y=v["y"]) for v in tb.get("bounding_box", [])
                ],
                confidence=tb.get("confidence", 1.0),
            )
            for tb in text_blocks
        ]

        frame_models = [
            Frame(
                x=f["x"],
                y=f["y"],
                w=f["w"],
                h=f["h"],
                area=f.get("area", f["w"] * f["h"]),
                aspect=f.get("aspect", f["h"] / f["w"] if f["w"] > 0 else 0),
                fill_ratio=f.get("fill_ratio", 0.0),
                h_lines_count=f.get("h_lines_count", 0),
                v_lines_count=f.get("v_lines_count", 0),
                total_lines=f.get("total_lines", 0),
            )
            for f in frames
        ]

        results = classify_dimensions(
            text_block_models, frame_models, output_dir=base_output_dir
        )
        all_ok = True
        for i, frame in enumerate(frames):
            if i >= len(results):
                continue
            simplified_frame = results[i]
            panel = simplified_frame.panels[0] if simplified_frame.panels else None

            exp = expected[i] if i < len(expected) else {}

            if panel:
                got_outer = [int(panel.outer_width), int(panel.outer_height)]
                got_inner = [int(x) for x in panel.inner_heights]
            else:
                got_outer = [0, 0]
                got_inner = []

            # Parse expected
            if isinstance(exp, dict) and (
                "outer_width" in exp or "outer_height" in exp
            ):
                exp_outer = [
                    int(exp.get("outer_width", 0)),
                    int(exp.get("outer_height", 0)),
                ]
                exp_inner = [int(x) for x in exp.get("inner_heights", [])]
            else:
                exp_outer = [
                    int(exp.get("outer", [0, 0])[0]) if isinstance(exp, dict) else 0,
                    int(exp.get("outer", [0, 0])[1]) if isinstance(exp, dict) else 0,
                ]
                exp_inner = [
                    int(x)
                    for x in (exp.get("inner", []) if isinstance(exp, dict) else [])
                ]

            outer_ok = sorted(got_outer) == sorted(exp_outer)
            inner_ok = got_inner == exp_inner
            print(f"- Frame {i}")
            print(f"  px: ({frame['w']},{frame['h']})")
            print(
                f"  outer: got={got_outer} exp={exp_outer} -> {'OK' if outer_ok else 'NG'}"
            )
            print(
                f"  inner: got={got_inner} exp={exp_inner} -> {'OK' if inner_ok else 'NG'}"
            )
            all_ok = all_ok and outer_ok and inner_ok
        print("SAMPLE TEST:", "PASS" if all_ok else "FAIL")
        overall_ok = overall_ok and all_ok

    print("\nSUMMARY (ALL FILES):", "PASS" if overall_ok else "FAIL")
