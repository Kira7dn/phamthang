import json
from pathlib import Path
from typing import Dict, List, Optional

from app.models import OCRTextBlock, OCRVertex, Frame, SimplifiedPanel, SimplifiedFrame
from app.tools.dimension.score_dims import (
    score_consistency,
    score_frequency_alignment,
    score_inner_sum_quality,
    score_completeness,
)
from app.tools.dimension.normalize_frames import normalize_frames_fast
from app.tools.dimension.scale_estimation import estimate_scale
from app.tools.dimension.outer_candidates import generate_outer_candidates


def _calculate_image_bounds(text_blocks: List[OCRTextBlock]) -> tuple[float, float, float, float]:
    """Tính toán biên ảnh từ tất cả các text blocks."""
    x_vals = [v.x or 0 for block in text_blocks for v in block.bounding_box]
    y_vals = [v.y or 0 for block in text_blocks for v in block.bounding_box]
    return min(x_vals), max(x_vals), min(y_vals), max(y_vals)


def _calculate_margins(span_x: float, span_y: float, margin: int) -> tuple[float, float]:
    """Tính toán margin động dựa trên kích thước ảnh."""
    margin_x = min(margin, max(10.0, 0.08 * span_x))
    margin_y = min(margin, max(10.0, 0.08 * span_y))
    return margin_x, margin_y


def _is_valid_numeric_text(text: str) -> bool:
    """Kiểm tra text có phải là số hợp lệ (không chứa chữ cái)."""
    return text and not any(c.isalpha() for c in text)


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


def _is_near_border(cx: float, cy: float, bounds: tuple[float, float, float, float], 
                    margins: tuple[float, float]) -> bool:
    """Kiểm tra tâm text có nằm gần biên ảnh hay không."""
    x_min, x_max, y_min, y_max = bounds
    margin_x, margin_y = margins
    
    return (
        cx - x_min < margin_x or
        x_max - cx < margin_x or
        cy - y_min < margin_y or
        y_max - cy < margin_y
    )


def _collect_outer_inner(
    text_blocks: List[OCRTextBlock], margin: int = 0
) -> Dict[str, List[Dict]]:
    """
    Phân loại text thành outer và inner dựa trên vị trí:
    - Outer: gần mép ảnh (margin)
    - Inner: nằm bên trong
    """
    if not text_blocks:
        return {"outer": [], "inner": []}

    # Tính toán biên ảnh
    x_min, x_max, y_min, y_max = _calculate_image_bounds(text_blocks)
    if x_min == x_max or y_min == y_max:
        return {"outer": [], "inner": []}

    # Tính margin động
    span_x = max(1.0, x_max - x_min)
    span_y = max(1.0, y_max - y_min)
    margin_x, margin_y = _calculate_margins(span_x, span_y, margin)
    bounds = (x_min, x_max, y_min, y_max)
    margins = (margin_x, margin_y)

    outer, inner = [], []

    for block in text_blocks:
        # Chỉ xử lý text số hợp lệ
        if not _is_valid_numeric_text(block.text):
            continue

        entry = _create_text_entry(block)
        if not entry:
            continue

        # Phân loại outer/inner dựa trên vị trí
        if _is_near_border(entry["cx"], entry["cy"], bounds, margins):
            outer.append(entry)
        else:
            inner.append(entry)

    # Debug output
    print("---Outer---")
    for item in outer:
        print(item.get("text"))
    print("---Inner---")
    for item in inner:
        print(item.get("text"))

    # Sắp xếp inner theo tọa độ (từ trên xuống, trái sang phải)
    inner.sort(key=lambda x: (x["cy"], x["cx"]))

    return {"outer": outer, "inner": inner}


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


def _parse_float(text: str) -> float:
    """Parse text thành float, trả về 0 nếu không hợp lệ."""
    try:
        return float(str(text).strip()) if str(text).replace(".", "").isdigit() else 0.0
    except Exception:
        return 0.0


def classify_dimensions(
    text_blocks: List[OCRTextBlock],
    frames: List[Frame],
    margin: int = 20,
    output_dir: Optional[Path] = None,
) -> List[SimplifiedFrame]:
    # Thu thập entries toàn cục theo logic OUTER/INNER của TOÀN ẢNH (biên ảnh)
    dim_info = _collect_outer_inner(text_blocks, margin=margin)
    outer_entries = dim_info["outer"]
    inner_entries_all = dim_info["inner"]
    entries = outer_entries + inner_entries_all

    # Extract numbers from text_blocks
    numbers_all = [
        float(b.text)
        for b in text_blocks
        if b.text and b.text.replace(".", "").isdigit()
    ]

    outer_numbers = [
        float(e["text"])
        for e in outer_entries
        if e.get("text", "").replace(".", "").isdigit()
    ]

    # Convert frames to dict format for normalize_frames_fast
    frames_dict = [
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

    # Normalize frames to get consistent dimensions
    normalized_frames = normalize_frames_fast(frames_dict, tolerance_px=5)

    # Estimate scale from normalized dimensions
    pixel_candidates = []
    for nf in normalized_frames:
        for key in ("normalized_w", "normalized_h"):
            val = nf.get(key)
            if isinstance(val, (int, float)) and val > 0:
                pixel_candidates.append(float(val))

    scale_numbers = outer_numbers if outer_numbers else numbers_all
    scale_result = estimate_scale(pixel_candidates, scale_numbers, inlier_rel_tol=0.12)
    scale = scale_result.get("scale")

    # Generate outer candidates using scale-based approach
    if scale:
        outer_cands = generate_outer_candidates(
            normalized_frames,
            scale_numbers,
            float(scale),
            k=3,
            rel_tol=0.20,
            window_ratio=0.5,
        )
    else:
        # Fallback: no candidates if scale estimation failed
        # outer_cands = [
        #     {"w_cands": [], "h_cands": [], "est_mm": {"w": None, "h": None}}
        #     for _ in frames
        # ]
        raise ValueError("Scale estimation failed")

    result: Dict[int, Dict] = {}
    for i, frame in enumerate(frames):
        w, h = frame.w, frame.h
        fx, fy = frame.x, frame.y
        nf = normalized_frames[i] if i < len(normalized_frames) else {}
        row_group = nf.get("row_group")
        col_group = nf.get("col_group")

        # B2: chọn width/height từ outer candidates (scale-based)
        cands = (
            outer_cands[i] if i < len(outer_cands) else {"w_cands": [], "h_cands": []}
        )
        w_cands = cands.get("w_cands", [])
        h_cands = cands.get("h_cands", [])

        # Pick top candidate for width and height (use per-frame size criteria)
        if w_cands:
            if isinstance(scale, (int, float)) and scale:
                est_w_frame = float(w) * float(scale)
                # Prefer minimal absolute diff to estimated mm, tie-break by higher score
                best_w_cand = min(
                    w_cands,
                    key=lambda c: (
                        abs(float(c.get("value", 0)) - est_w_frame),
                        -float(c.get("score", 0.0)),
                    ),
                )
            else:
                # Fallback: original behavior
                best_w_cand = w_cands[0]
            chosen_w = str(int(best_w_cand.get("value", w)))
        else:
            chosen_w = str(int(w))

        if h_cands:
            # Prefer candidate closest to estimated mm based on this frame size if scale exists
            if isinstance(scale, (int, float)) and scale:
                est_h_frame = float(h) * float(scale)
                best_h_cand = min(
                    h_cands,
                    key=lambda c: (
                        abs(float(c.get("value", 0)) - est_h_frame),
                        -float(c.get("score", 0.0)),
                    ),
                )
            else:
                # Fallback: use per-candidate estimated mm from generator if present, else first
                est_h = None
                try:
                    est_h = cands.get("est_mm", {}).get("h")
                except Exception:
                    est_h = None
                if isinstance(est_h, (int, float)) and est_h > 0:
                    best_h_cand = min(
                        h_cands,
                        key=lambda c: abs(float(c.get("value", 0)) - float(est_h)),
                    )
                else:
                    best_h_cand = h_cands[0]
            chosen_h = str(int(best_h_cand.get("value", h)))
        else:
            chosen_h = str(int(h))

        outer_wh = {"width": chosen_w, "height": chosen_h}

        # B3: inner_heights = Dùng _is_aligned để lấy tất cả entries thuộc frame
        # Sau đó lọc chỉ lấy số nằm trong frame height range
        frame_entries = [e for e in entries if _is_aligned(e, frame)]

        # Chỉ lấy số có cy nằm trong frame height range [fy, fy+h]
        margin_y = max(30, int(0.12 * h))
        inner_candidates = [
            e
            for e in frame_entries
            if e.get("text", "").isdigit()
            and (fy - margin_y) <= e.get("cy", 0.0) <= (fy + h + margin_y)
        ]

        # Sắp xếp theo Y (từ trên xuống)
        inner_candidates.sort(key=lambda e: e.get("cy", 0.0))

        # Loại bỏ outer dimensions
        to_remove = {outer_wh["height"], outer_wh["width"]}
        inner_entries: List[Dict] = [
            e for e in inner_candidates if e.get("text") not in to_remove
        ]

        # Lọc cột X: nếu có nhiều cột, chọn cột có tổng gần outer_height nhất
        if len(inner_entries) >= 3:
            cxs = [e.get("cx", 0.0) for e in inner_entries]
            x_tol = max(20.0, 0.20 * w)
            left_seed = min(cxs)
            right_seed = max(cxs)
            left_group = [
                e for e in inner_entries if abs(e.get("cx", 0.0) - left_seed) <= x_tol
            ]
            right_group = [
                e for e in inner_entries if abs(e.get("cx", 0.0) - right_seed) <= x_tol
            ]

            def _parse_float(s: str) -> float:
                try:
                    return (
                        float(str(s).strip())
                        if str(s).replace(".", "").isdigit()
                        else 0.0
                    )
                except Exception:
                    return 0.0

            def group_sum(group: List[Dict]) -> float:
                return sum(_parse_float(e.get("text", "")) for e in group)

            chosen_h_val = _parse_float(outer_wh.get("height", "0"))
            lg_sum = group_sum(left_group)
            rg_sum = group_sum(right_group)

            # Chọn nhóm có tổng gần outer_height nhất, nếu bằng nhau thì ưu tiên nhóm nhiều phần tử hơn
            lg_err = abs(lg_sum - chosen_h_val)
            rg_err = abs(rg_sum - chosen_h_val)

            if abs(lg_err - rg_err) <= 1.0:
                # Tie-break: chọn nhóm nhiều phần tử hơn
                cand_group = (
                    left_group if len(left_group) >= len(right_group) else right_group
                )
            else:
                cand_group = left_group if lg_err < rg_err else right_group

            if cand_group:
                inner_entries = cand_group

        # Dedupe: bỏ số liên tiếp trùng nhau và gần nhau theo Y
        deduped: List[Dict] = []
        for e in inner_entries:
            if not deduped:
                deduped.append(e)
                continue
            prev = deduped[-1]
            # Nếu trùng giá trị và gần nhau theo Y (<30px), bỏ
            if (
                e["text"] == prev["text"]
                and abs(e.get("cy", 0.0) - prev.get("cy", 0.0)) < 30
            ):
                continue
            deduped.append(e)

        # Chuyển thành inner_heights
        inner_heights = [e["text"] for e in deduped]

        result[i] = {
            "width": outer_wh["width"],
            "height": outer_wh["height"],
            "inner_heights": inner_heights,
            "frame_px_w": w,
            "frame_px_h": h,
        }

    # Tính scores để đánh giá chất lượng (numbers already extracted above)

    # Calculate quality scores
    consistency_score = score_consistency(result, frames)
    frequency_score = score_frequency_alignment(result, numbers_all)
    inner_sum_score = score_inner_sum_quality(result)
    completeness_score = score_completeness(result, frames)

    # Compute overall score
    weights = {
        "consistency": 0.35,
        "frequency": 0.25,
        "inner_sum": 0.25,
        "completeness": 0.15,
    }

    components = [
        consistency_score["overall_consistency"],
        frequency_score["overall_freq_score"],
        inner_sum_score["inner_sum_score"],
        completeness_score["completeness_score"],
    ]
    component_weights = list(weights.values())
    overall_score = sum(s * w for s, w in zip(components, component_weights)) / sum(
        component_weights
    )

    # Build SimplifiedFrame list with quality scores
    simplified_frames = []

    # Calculate quality scores
    quality_scores = {
        "overall": round(overall_score, 4),
        "consistency": round(consistency_score["overall_consistency"], 4),
        "frequency": round(frequency_score["overall_freq_score"], 4),
        "inner_sum": round(inner_sum_score["inner_sum_score"], 4),
        "completeness": round(completeness_score["completeness_score"], 4),
    }

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

    # Save results if output_dir is provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dimensions_classified.json"

        # Convert to format matching dimension_samples.json
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


if __name__ == "__main__":
    # Iterate all JSON samples in sample directory
    sample_dir = Path("app/tools/dimension/sample")
    json_files = sorted(sample_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON samples found in {sample_dir}")

    overall_ok = True
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as sf:
            data = json.load(sf)

        # Each sample file contains a SINGLE object
        case = data

        # Use a separate output directory per sample file
        base_output_dir = Path("outputs", "dims_classify", json_file.stem)

        print(f"\n=== Running sample: {json_file} ===")
        print(f"Output dir: {base_output_dir}")

        cid = Path(json_file).stem
        cname = case.get("name", "")
        frames = case.get("frames", [])
        expected = case.get("panels", case.get("expected", []))  # Support both names
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

            # Parse expected in either new (panels) or legacy format
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
