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


def _collect_outer_inner(
    text_blocks: List[OCRTextBlock], margin: int = 50
) -> Dict[str, List[Dict]]:
    """
    Phân loại text thành outer và inner dựa trên vị trí:
    - Outer: gần mép (margin)
    - Inner: còn lại
    """
    if not text_blocks:
        return {"outer": [], "inner": []}

    x_vals = [v.x or 0 for block in text_blocks for v in block.bounding_box]
    y_vals = [v.y or 0 for block in text_blocks for v in block.bounding_box]
    if not x_vals or not y_vals:
        return {"outer": [], "inner": []}

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    outer, inner = [], []

    for block in text_blocks:
        text = block.text
        if not text or any(c.isalpha() for c in text):
            continue

        bounding_box = block.bounding_box
        if len(bounding_box) < 4:
            continue

        xs = [v.x or 0 for v in bounding_box]
        ys = [v.y or 0 for v in bounding_box]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        entry = {
            "text": text,
            "cx": cx,
            "cy": cy,
            "w": max(xs) - min(xs),
            "h": max(ys) - min(ys),
            "aspect": (max(xs) - min(xs)) / (max(ys) - min(ys) + 1e-6),
        }

        # Chỉ thêm vào outer nếu nằm gần biên
        if (
            cx - x_min < margin
            or x_max - cx < margin
            or cy - y_min < margin
            or y_max - cy < margin
        ):
            outer.append(entry)
        else:
            inner.append(entry)

    # Không sort outer theo giá trị số - giữ thứ tự OCR gốc
    # outer.sort(key=lambda x: float(x["text"]), reverse=True)
    # Sắp xếp inner theo tọa độ y (từ trên xuống)
    inner.sort(key=lambda x: (x["cy"], x["cx"]))

    return {"outer": outer, "inner": inner}


def _is_aligned(entry: Dict, frame: Frame, threshold: int = 10) -> bool:
    """Kiểm tra tâm của entry có cùng hàng/cột với frame trong sai số threshold."""
    cx, cy = entry.get("cx", 0.0), entry.get("cy", 0.0)
    fx, fy, fw, fh = frame.x, frame.y, frame.w, frame.h
    in_column = fx - threshold <= cx <= fx + fw + threshold
    in_row = fy - threshold <= cy <= fy + fh + threshold
    return in_column or in_row


def _select_outer_by_aspect(
    frame_w: float, frame_h: float, outer_candidates: List[Dict], tolerance: float = 0.2
) -> Dict[str, str]:
    """
    Chọn outer width và height phù hợp với frame dựa trên tỷ lệ:
    - Tính tỷ lệ khung (frame_aspect)
    - Tìm cặp width/height trong outer_candidates có tỷ lệ gần nhất với frame_aspect
    - Trả về width và height phù hợp nhất
    """
    if not outer_candidates:
        return {"width": str(int(frame_w)), "height": str(int(frame_h))}

    frame_aspect = frame_w / frame_h if frame_h != 0 else 1.0
    best_pair = None
    min_diff = float("inf")

    # Lọc các candidate hợp lệ
    valid_candidates = [
        c for c in outer_candidates if c["text"].isdigit() and float(c["text"]) > 0
    ]

    # Nếu không đủ 2 giá trị trở lên, trả về giá trị mặc định
    if len(valid_candidates) < 2:
        if valid_candidates:
            return {"width": valid_candidates[0]["text"], "height": str(int(frame_h))}
        return {"width": str(int(frame_w)), "height": str(int(frame_h))}

    # Sắp xếp theo tọa độ y (từ trên xuống)
    valid_candidates.sort(key=lambda x: x["cy"])

    # Tìm cặp có tỷ lệ gần nhất với frame_aspect
    for i in range(len(valid_candidates) - 1):
        for j in range(i + 1, len(valid_candidates)):
            width = float(valid_candidates[i]["text"])
            height = float(valid_candidates[j]["text"])
            if height == 0:
                continue

            candidate_aspect = width / height
            aspect_diff = abs(candidate_aspect - frame_aspect)

            if aspect_diff < min_diff:
                min_diff = aspect_diff
                best_pair = (valid_candidates[i], valid_candidates[j])

    # Nếu tìm được cặp phù hợp
    if best_pair and min_diff <= tolerance:
        # Sắp xếp lại để width luôn lớn hơn height (vì frame thường nằm ngang)
        w, h = best_pair
        if float(w["text"]) < float(h["text"]):
            w, h = h, w
        return {"width": w["text"], "height": h["text"]}

    # Fallback: chọn width là giá trị lớn nhất, height là giá trị phù hợp nhất
    if valid_candidates:
        max_width = max(valid_candidates, key=lambda x: float(x["text"]))
        other_heights = [c for c in valid_candidates if c != max_width]
        if other_heights:
            height = min(
                other_heights, key=lambda x: abs(float(x["text"]) / frame_h - 1)
            )
            return {"width": max_width["text"], "height": height["text"]}
        return {"width": max_width["text"], "height": str(int(frame_h))}

    return {"width": str(int(frame_w)), "height": str(int(frame_h))}


def classify_dimensions(
    text_blocks: List[OCRTextBlock],
    frames: List[Frame],
    margin: int = 50,
    output_dir: Optional[Path] = None,
) -> List[SimplifiedFrame]:
    # Thu thập entries toàn cục theo logic OUTER/INNER của TOÀN ẢNH (biên ảnh)
    dim_info = _collect_outer_inner(text_blocks, margin=margin)
    entries = dim_info["outer"] + dim_info["inner"]

    # Tính biên toàn cục từ entries để xác định ứng viên theo mép ảnh
    if entries:
        gx_min = min(e["cx"] for e in entries)
        gx_max = max(e["cx"] for e in entries)
        gy_min = min(e["cy"] for e in entries)
        gy_max = max(e["cy"] for e in entries)
    else:
        gx_min = gx_max = gy_min = gy_max = 0.0

    # Ứng viên width lấy từ mép TRÊN/DƯỚI của ảnh; height lấy từ mép TRÁI/PHẢI của ảnh
    horiz_outer = [
        o
        for o in dim_info["outer"]
        if o["text"].isdigit()
        and (abs(o["cy"] - gy_min) < margin or abs(o["cy"] - gy_max) < margin)
    ]
    # Vertical outer band: widen margin to capture all left/right border labels
    v_margin_x = max(margin, 100)
    vert_outer = [
        o
        for o in dim_info["outer"]
        if o["text"].isdigit()
        and (o["cx"] <= gx_min + v_margin_x or o["cx"] >= gx_max - v_margin_x)
    ]

    result: Dict[int, Dict] = {}
    for i, frame in enumerate(frames):
        w, h = frame.w, frame.h

        # B1: gom dims thuộc frame (cùng hàng/cột)
        frame_entries = [e for e in entries if _is_aligned(e, frame)]

        # B2: chọn width/height từ OUTER mép ảnh sao cho width/height gần frame_aspect
        frame_aspect = (w / h) if h != 0 else 1.0
        chosen_w = None
        chosen_h = None
        best_diff = None

        if horiz_outer and vert_outer:
            for w_cand in horiz_outer:
                wv = float(w_cand["text"])
                for h_cand in vert_outer:
                    hv = float(h_cand["text"])
                    if hv == 0:
                        continue
                    diff = abs((wv / hv) - frame_aspect)
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        chosen_w = w_cand["text"]
                        chosen_h = h_cand["text"]
        else:
            # fallback: dùng toàn bộ outer kết hợp cặp gần aspect
            outer_wh_tmp = _select_outer_by_aspect(w, h, dim_info["outer"])
            chosen_w = outer_wh_tmp["width"]
            chosen_h = outer_wh_tmp["height"]

        outer_wh = {"width": chosen_w or str(int(w)), "height": chosen_h or str(int(h))}

        # B4: inner_heights = TẤT CẢ số căn theo frame (cùng hàng/cột), sắp xếp STRICTLY theo (cy, cx) từ trên xuống
        # Lấy entries số từ frame_entries
        vertical_aligned = [e for e in frame_entries if e.get("text", "").isdigit()]
        vertical_aligned.sort(key=lambda e: (e.get("cy", 0.0), e.get("cx", 0.0)))
        # Loại bỏ các entry có tọa độ khác số đông theo trục x (cx) trong cùng frame
        if len(vertical_aligned) >= 3:
            # CRITICAL: Chỉ tính median từ entries CÓ cx hợp lệ (> 0)
            valid_cxs = [
                e.get("cx")
                for e in vertical_aligned
                if e.get("cx") is not None and e.get("cx") > 0
            ]

            if len(valid_cxs) >= 3:
                cxs = sorted(valid_cxs)
                mid = len(cxs) // 2
                if len(cxs) % 2 == 1:
                    median_cx = cxs[mid]
                else:
                    median_cx = 0.5 * (cxs[mid - 1] + cxs[mid])
                tol_x = max(20.0, 0.15 * w)

                # Filter: chỉ giữ entries có cx GẦN median_cx
                vertical_aligned = [
                    e
                    for e in vertical_aligned
                    if e.get("cx") is not None and abs(e.get("cx") - median_cx) <= tol_x
                ]

        # Loại bỏ outer dimensions (width và height) khỏi danh sách
        # Giữ nguyên thứ tự sắp xếp theo tọa độ
        to_remove = {outer_wh["height"], outer_wh["width"]}
        inner_heights: List[str] = []
        removed_count = {outer_wh["height"]: 0, outer_wh["width"]: 0}

        for e in vertical_aligned:
            num = e["text"]
            # Chỉ loại bỏ mỗi giá trị outer một lần
            if num in to_remove and removed_count[num] == 0:
                removed_count[num] = 1
                continue
            inner_heights.append(num)

        result[i] = {
            "width": outer_wh["width"],
            "height": outer_wh["height"],
            "inner_heights": inner_heights,
            "frame_px_w": w,
            "frame_px_h": h,
        }

    # Tính scores để đánh giá chất lượng
    numbers = [float(b.text) for b in text_blocks if b.text.replace(".", "").isdigit()]

    # Calculate quality scores
    consistency_score = score_consistency(result, frames)
    frequency_score = score_frequency_alignment(result, numbers)
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

    sample_path = "app/tools/dimension/dimension_samples.json"
    with open(sample_path, "r", encoding="utf-8") as sf:
        data = json.load(sf)
    output_dir = Path("outputs", "dims_classify")
    cases = data if isinstance(data, list) else [data]
    grand_ok = True
    for case_idx, case in enumerate(cases):
        cid = case.get("id", case_idx)
        cname = case.get("name", "")
        frames = case.get("frames", [])
        expected = case.get("panels", case.get("expected", []))  # Support both names
        text_blocks = case.get("text_blocks", [])

        print(f"Using sample: {cid} - {cname}")

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
            )
            for f in frames
        ]

        results = classify_dimensions(
            text_block_models, frame_models, output_dir=output_dir
        )
        all_ok = True
        for i, frame in enumerate(frames):
            # results is now List[SimplifiedFrame]
            if i >= len(results):
                continue
            simplified_frame = results[i]
            panel = simplified_frame.panels[0] if simplified_frame.panels else None

            exp = expected[i] if i < len(expected) else {"outer": [], "inner": []}

            if panel:
                got_outer = [int(panel.outer_width), int(panel.outer_height)]
                got_inner = [int(x) for x in panel.inner_heights]
            else:
                got_outer = [0, 0]
                got_inner = []

            exp_outer = [
                int(exp.get("outer", [0, 0])[0]),
                int(exp.get("outer", [0, 0])[1]),
            ]
            outer_ok = sorted(got_outer) == sorted(exp_outer)
            exp_inner = [int(x) for x in exp.get("inner", [])]
            inner_ok = got_inner == exp_inner
            print(
                f"Frame {i}: outer={got_outer} -> {'OK' if outer_ok else 'NG'}; "
                f"inner={got_inner} -> {'OK' if inner_ok else 'NG'}; "
                f"frame_px=({frame['w']},{frame['h']})"
            )
            all_ok = all_ok and outer_ok and inner_ok
        print("SAMPLE TEST:", "PASS" if all_ok else "FAIL")
        grand_ok = grand_ok and all_ok
    if len(cases) > 1:
        print("ALL CASES:", "PASS" if grand_ok else "FAIL")
