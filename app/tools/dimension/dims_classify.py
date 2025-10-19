import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.models import OCRTextBlock, OCRVertex, Frame, SimplifiedPanel, SimplifiedFrame
from app.tools.dimension.inner_classification import (
    deduplicate_entries,
    inner_height_select,
)
from app.tools.dimension.score_dims import calculate_quality_scores
from app.tools.dimension.outer_classification import OuterResult, outer_classify


logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS - Dimensions Processing
# ============================================================================


def _parse_float(text: str) -> float:
    """Parse text thành float, trả về 0 nếu không hợp lệ."""
    try:
        return float(str(text).strip()) if str(text).replace(".", "").isdigit() else 0.0
    except Exception:
        return 0.0


def _entry_center(block: OCRTextBlock) -> tuple[float, float]:
    if not block.bounding_box:
        return 0.0, 0.0
    xs = [v.x or 0.0 for v in block.bounding_box]
    ys = [v.y or 0.0 for v in block.bounding_box]
    return (
        (min(xs) + max(xs)) / 2.0,
        (min(ys) + max(ys)) / 2.0,
    )


# ============================================================================
# HELPER FUNCTIONS - Outer/Inner Classification
# ============================================================================


def _collect_outer_inner(
    text_blocks: List[OCRTextBlock],
) -> tuple[
    List[OCRTextBlock], List[OCRTextBlock], List[OCRTextBlock], List[OCRTextBlock]
]:
    """
    Phân loại text thành outer và inner dựa trên vị trí:
    - Outer: gần mép ảnh (margin)
    - Inner: nằm bên trong
    """
    if not text_blocks:
        return [], [], [], []

    x_vals = [v.x or 0 for block in text_blocks for v in block.bounding_box]
    y_vals = [v.y or 0 for block in text_blocks for v in block.bounding_box]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    if x_min == x_max or y_min == y_max:
        return [], [], [], []

    # Tìm 4 block xa nhất theo mỗi hướng
    left_block = min(text_blocks, key=lambda b: min(v.x for v in b.bounding_box))
    right_block = max(text_blocks, key=lambda b: max(v.x for v in b.bounding_box))
    top_block = min(text_blocks, key=lambda b: min(v.y for v in b.bounding_box))
    bottom_block = max(text_blocks, key=lambda b: max(v.y for v in b.bounding_box))

    # Lấy vùng biên dựa trên chính kích thước của 4 block đó
    margin_left = max(v.x for v in left_block.bounding_box)
    margin_right = min(v.x for v in right_block.bounding_box)
    margin_top = max(v.y for v in top_block.bounding_box)
    margin_bottom = min(v.y for v in bottom_block.bounding_box)

    left_outer, right_outer, top_outer, bottom_outer, inner = [], [], [], [], []

    for text_block in text_blocks:
        cx, cy = _entry_center(text_block)
        is_outer = False

        if cx < margin_left:
            left_outer.append(text_block)
            is_outer = True
        if cx > margin_right:
            right_outer.append(text_block)
            is_outer = True
        if cy < margin_top:
            top_outer.append(text_block)
            is_outer = True
        if cy > margin_bottom:
            bottom_outer.append(text_block)
            is_outer = True
        if not is_outer:
            inner.append(text_block)

    return left_outer, right_outer, top_outer, bottom_outer


# ============================================================================
# HELPER FUNCTIONS - Frame Alignment
# ============================================================================


def _is_aligned(entry: OCRTextBlock, frame: Frame, threshold: int = 5) -> bool:
    """Kiểm tra entry có nằm cùng hàng hoặc cột với frame (trong ngưỡng sai số)."""
    cx, cy = _entry_center(entry)

    # Ngưỡng động dựa trên kích thước frame
    threshold_x = max(threshold, int(0.02 * frame.w))
    threshold_y = max(threshold, int(0.02 * frame.h))

    # Kiểm tra cùng cột hoặc cùng hàng
    in_column = frame.x - threshold_x <= cx <= frame.x + frame.w + threshold_x
    in_row = frame.y - threshold_y <= cy <= frame.y + frame.h + threshold_y

    return in_column or in_row


# ============================================================================
# HELPER FUNCTIONS - Inner Candidates Processing
# ============================================================================


def _filter_inner_candidates(
    entries: List[OCRTextBlock],
    outer_width_entry: OCRTextBlock,
    outer_height_entry: OCRTextBlock,
) -> List[OCRTextBlock]:
    """Loại bỏ chính xác các entry outer đã chọn, dựa trên object gốc hoặc text fallback."""

    targets = []
    if outer_width_entry is not None:
        targets.append(outer_width_entry)
    if outer_height_entry is not None:
        targets.append(outer_height_entry)

    consumed = [False] * len(targets)

    def _match_entry(candidate: OCRTextBlock, target: OCRTextBlock) -> bool:
        if candidate is target:
            return True
        if not isinstance(target, dict):
            return False
        if not isinstance(candidate, dict):
            return False
        return candidate.get("text") == target.get("text")

    filtered: List[OCRTextBlock] = []
    for entry in entries:
        matched = False
        for idx, target in enumerate(targets):
            if consumed[idx]:
                continue
            if _match_entry(entry, target):
                consumed[idx] = True
                matched = True
                break
        if matched:
            continue
        filtered.append(entry)

    return filtered


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
    left_outer, right_outer, top_outer, bottom_outer = _collect_outer_inner(text_blocks)

    # Bước 5: Xử lý từng frame
    result: Dict[int, Dict] = {}

    for i, frame in enumerate(frames):
        frame_entries = [e for e in text_blocks if _is_aligned(e, frame)]
        left_entries = [entry for entry in left_outer if _is_aligned(entry, frame)]
        right_entries = [entry for entry in right_outer if _is_aligned(entry, frame)]
        top_entries = [entry for entry in top_outer if _is_aligned(entry, frame)]
        bottom_entries = [entry for entry in bottom_outer if _is_aligned(entry, frame)]
        logger.debug(f"Frame {i}: frame={[entry.text for entry in frame_entries]}")
        logger.debug(f"Frame {i}: left={[entry.text for entry in left_entries]}")
        logger.debug(f"Frame {i}: right={[entry.text for entry in right_entries]}")
        logger.debug(f"Frame {i}: top={[entry.text for entry in top_entries]}")
        logger.debug(f"Frame {i}: bottom={[entry.text for entry in bottom_entries]}")
        outer_classify_result: OuterResult = outer_classify(
            frame,
            top_entries + bottom_entries,
            left_entries + right_entries,
        )
        chosen_w = outer_classify_result.width
        chosen_h = outer_classify_result.height
        scale_x = outer_classify_result.scale_x
        scale_y = outer_classify_result.scale_y
        aspect_diff_ratio = outer_classify_result.aspect_diff_ratio

        if not chosen_w or not chosen_h:
            logger.debug(
                "Frame %s: outer_classify không tìm được width/height hợp lệ", i
            )
            continue
        logger.debug(f"Frame {i}: Entries={[entry.text for entry in frame_entries]}")
        inner_entries = _filter_inner_candidates(
            frame_entries,
            chosen_w,
            chosen_h,
        )
        logger.debug(
            f"Frame {i}: Entries={[entry.text for entry in inner_entries]} Remove outer"
        )
        # sort by Y
        inner_entries = sorted(inner_entries, key=lambda e: e.bounding_box[0].y)
        logger.debug(
            f"Frame {i}: Entries={[entry.text for entry in inner_entries]} Sort"
        )
        # remove duplicates
        inner_entries = deduplicate_entries(inner_entries, float(chosen_h.text))

        logger.debug(
            f"Frame {i}: Entries={[entry.text for entry in inner_entries]}Dedup"
        )
        inner_entries = inner_height_select(inner_entries, chosen_h)
        logger.debug(
            f"Frame {i}: Entries={[entry.text for entry in inner_entries]}Select"
        )

        inner_heights = [e.text for e in inner_entries]
        selected_ids = {id(e) for e in inner_entries}
        unselected_entries = [e for e in frame_entries if id(e) not in selected_ids]

        result[i] = {
            "frame_idx": i,
            "width": chosen_w.text,
            "height": chosen_h.text,
            "inner_heights": inner_heights,
            "unselected": [e.text for e in unselected_entries],
            "frame_px_w": frame.w,
            "frame_px_h": frame.h,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "aspect_diff_ratio": aspect_diff_ratio,
        }

    # Bước 6: Tính toán chỉ số chất lượng
    numbers_all = [entry.text for entry in text_blocks]
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
    # json_files = [f for f in json_files if f.stem == "07"]
    if not json_files:
        print(f"No JSON samples found in {sample_dir}")

    overall_ok = True
    overall_stats = {
        "samples": 0,
        "frames": 0,
        "outer_pass": 0,
        "inner_pass": 0,
        "both_pass": 0,
    }
    failed_samples: List[str] = []

    for json_file in json_files:
        overall_stats["samples"] += 1
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
        sample_total_frames = 0
        sample_outer_pass = 0
        sample_inner_pass = 0
        sample_both_pass = 0
        sample_outer_pass_idx: List[int] = []
        sample_outer_fail_idx: List[int] = []
        sample_inner_pass_idx: List[int] = []
        sample_inner_fail_idx: List[int] = []

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
            sample_total_frames += 1
            overall_stats["frames"] += 1
            if outer_ok:
                sample_outer_pass += 1
                sample_outer_pass_idx.append(i)
                overall_stats["outer_pass"] += 1
            else:
                sample_outer_fail_idx.append(i)
            if inner_ok:
                sample_inner_pass += 1
                sample_inner_pass_idx.append(i)
                overall_stats["inner_pass"] += 1
            else:
                sample_inner_fail_idx.append(i)
            if outer_ok and inner_ok:
                sample_both_pass += 1
                overall_stats["both_pass"] += 1
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
        if sample_total_frames:
            print(
                "Sample summary: frames=%d outer_pass=%d/%d inner_pass=%d/%d both_pass=%d/%d"
                % (
                    sample_total_frames,
                    sample_outer_pass,
                    sample_total_frames,
                    sample_inner_pass,
                    sample_total_frames,
                    sample_both_pass,
                    sample_total_frames,
                )
            )
            print(
                "  outer_pass_idx=%s outer_fail_idx=%s"
                % (sample_outer_pass_idx, sample_outer_fail_idx)
            )
            print(
                "  inner_pass_idx=%s inner_fail_idx=%s"
                % (sample_inner_pass_idx, sample_inner_fail_idx)
            )
        overall_ok = overall_ok and all_ok
        if not all_ok:
            failed_samples.append(cid or json_file.name)

    print("\nSUMMARY (ALL FILES):", "PASS" if overall_ok else "FAIL")
    total_frames = overall_stats["frames"]
    if total_frames:
        print(
            "Totals: samples=%d frames=%d outer_pass=%d/%d inner_pass=%d/%d both_pass=%d/%d"
            % (
                overall_stats["samples"],
                total_frames,
                overall_stats["outer_pass"],
                total_frames,
                overall_stats["inner_pass"],
                total_frames,
                overall_stats["both_pass"],
                total_frames,
            )
        )
    else:
        print("Totals: samples=%d frames=0" % overall_stats["samples"])

    if failed_samples:
        print("Failed samples:", ", ".join(failed_samples))
    else:
        print("Failed samples: None")
