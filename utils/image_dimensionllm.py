"""
Image Dimension Extraction Module

Maps detected frames from technical drawings with OCR-extracted numbers
to determine outer dimensions and inner hinge positions of aluminum door panels.

Business Logic:
- Frames: Detected rectangular aluminum frames with pixel coordinates (x, y, w, h)
- Numbers: OCR-extracted dimensions in millimeters from the technical drawing
- Outer dimensions: width and height of the frame rectangle
- Inner dimensions: vertical hinge positions that sum to the frame height
- Numbers may appear multiple times if frames share dimensions
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import logging
import json
import os
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent

logger = logging.getLogger(__name__)


def normalize_frames(
    frames: List[Dict[str, Any]], tolerance_px: int = 10
) -> List[Dict[str, Any]]:
    """
    Normalize frame dimensions by grouping similar sizes.

    Frames with similar dimensions (within tolerance) are considered to have the same size.
    This handles small detection errors.

    Args:
        frames: List of detected frames
        tolerance_px: Pixel tolerance for considering dimensions equal (default 10px)

    Returns:
        List of frames with normalized dimensions
    """
    if not frames:
        return frames

    normalized = []

    # Group frames by approximate width
    width_groups = {}
    for frame in frames:
        w = frame["w"]
        # Find existing group within tolerance
        found_group = False
        for group_w, group_frames in width_groups.items():
            if abs(w - group_w) <= tolerance_px:
                group_frames.append(frame)
                found_group = True
                break
        if not found_group:
            width_groups[w] = [frame]

    # Normalize widths within each group to the median
    for group_w, group_frames in width_groups.items():
        widths = [f["w"] for f in group_frames]
        normalized_w = int(sum(widths) / len(widths))

        for frame in group_frames:
            frame["normalized_w"] = normalized_w

    # Group frames by approximate height
    height_groups = {}
    for frame in frames:
        h = frame["h"]
        found_group = False
        for group_h, group_frames in height_groups.items():
            if abs(h - group_h) <= tolerance_px:
                group_frames.append(frame)
                found_group = True
                break
        if not found_group:
            height_groups[h] = [frame]

    # Normalize heights within each group
    for group_h, group_frames in height_groups.items():
        heights = [f["h"] for f in group_frames]
        normalized_h = int(sum(heights) / len(heights))

        for frame in group_frames:
            frame["normalized_h"] = normalized_h

    # Check for frames in same row (similar y) or same column (similar x)
    for frame in frames:
        frame["row_group"] = None
        frame["col_group"] = None

    # Group by rows (similar y coordinate)
    y_groups = {}
    for frame in frames:
        y = frame["y"]
        found_group = False
        for group_y, group_frames in y_groups.items():
            if abs(y - group_y) <= tolerance_px * 2:
                group_frames.append(frame)
                found_group = True
                break
        if not found_group:
            y_groups[y] = [frame]

    for row_id, (group_y, group_frames) in enumerate(y_groups.items()):
        group_size = len(group_frames)
        for frame in group_frames:
            frame["row_group"] = row_id
            frame["row_group_size"] = group_size

    # Group by columns (similar x coordinate)
    x_groups = {}
    for frame in frames:
        x = frame["x"]
        found_group = False
        for group_x, group_frames in x_groups.items():
            if abs(x - group_x) <= tolerance_px * 2:
                group_frames.append(frame)
                found_group = True
                break
        if not found_group:
            x_groups[x] = [frame]

    for col_id, (group_x, group_frames) in enumerate(x_groups.items()):
        group_size = len(group_frames)
        for frame in group_frames:
            frame["col_group"] = col_id
            frame["col_group_size"] = group_size

    logger.info(
        f"Normalized {len(frames)} frames: "
        f"{len(width_groups)} unique widths, {len(height_groups)} unique heights, "
        f"{len(y_groups)} rows, {len(x_groups)} columns"
    )

    return frames


class InnerHeightsLLMResult(BaseModel):
    inner_heights: List[float]
    reasoning: str


def find_inner_heights_with_llm(
    target_height: float,
    available_numbers: List[float],
    outer_width: float,
    frame_info: Dict[str, Any],
) -> Optional[List[float]]:
    """
    Use LLM to infer inner heights when algorithmic approach fails.

    Args:
        target_height: Target sum (outer height in mm)
        available_numbers: All available numbers from OCR
        outer_width: Outer width of the frame
        frame_info: Frame metadata for context

    Returns:
        List of inner heights or None if LLM fails
    """
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        logger.warning("No API key available for LLM fallback")
        return None

    model_id = (
        "google-gla:gemini-2.0-flash"
        if os.getenv("GOOGLE_API_KEY")
        else "openai:gpt-4o"
    )

    system_prompt = """
You are an expert in analyzing aluminum door technical drawings.
Given a panel's outer dimensions and available numbers from OCR, infer the most likely inner heights (hinge positions).

Rules:
1. Inner heights should sum approximately to the outer height (within 10% tolerance)
2. Common patterns: [100, 497, 100], [508, 508, 100, 100], [100, 100, ...]
3. Prefer using numbers from the available list, but can infer missing values
4. Typical hinge spacing: 100mm at top/bottom, larger segments in middle
5. Output valid JSON only with reasoning
"""

    prompt = f"""
Panel dimensions:
- Outer width: {outer_width} mm
- Outer height: {target_height} mm
- Frame aspect ratio: {frame_info.get('aspect', 'unknown')}

Available numbers from OCR: {available_numbers}

Task: Infer the inner heights (vertical hinge positions) that sum to approximately {target_height} mm.
Provide reasoning for your choice.
"""

    try:
        agent = Agent(
            model_id,
            output_type=InnerHeightsLLMResult,
            system_prompt=system_prompt,
        )
        result = agent.run_sync(prompt)

        if hasattr(result, "output") and isinstance(
            result.output, InnerHeightsLLMResult
        ):
            llm_result = result.output
        else:
            llm_result = InnerHeightsLLMResult.model_validate(result.output)

        logger.info(
            f"LLM inferred inner heights: {llm_result.inner_heights}, "
            f"reasoning: {llm_result.reasoning}"
        )
        return llm_result.inner_heights
    except Exception as e:
        logger.error(f"LLM fallback failed: {e}")
        return None


# Demo data - Multiple test cases
_samples_path = Path(__file__).resolve().parent / "image_dimension_samples.json"
with _samples_path.open("r", encoding="utf-8") as _samples_file:
    image_set = json.load(_samples_file)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    # Check if --test-llm flag is provided
    test_llm = "--test-llm" in sys.argv
    use_llm = "--no-llm" not in sys.argv

    # Select test cases
    if "--case" in sys.argv:
        case_idx = int(sys.argv[sys.argv.index("--case") + 1])
        test_cases = [image_set[case_idx]]
    else:
        test_cases = image_set

    total_tests = 0
    passed_tests = 0

    separator = "=" * 70

    for img_set in test_cases:
        print("\n" + separator)
        print(f"=== Test Case {img_set['id']}: {img_set['name']} ===")
        print(separator)
        print(f"Available numbers: {img_set['numbers']}")
        print(f"Number of frames: {len(img_set['frames'])}")
        print(f"LLM fallback: {'enabled' if use_llm else 'disabled'}")

        panels = find_inner_heights_with_llm(img_set, use_llm_fallback=True)

        print(f"\n✓ Extracted {len(panels)} panels:")

        # Compare with expected results if available
        expected = img_set.get("expected", [])
        test_passed = True

        for idx, panel in enumerate(panels):
            inner_sum = sum(panel["inner_heights"]) if panel["inner_heights"] else 0
            match_status = (
                "✓"
                if inner_sum > 0
                and abs(inner_sum - panel["outer_height"]) / panel["outer_height"]
                <= 0.05
                else "✗"
            )

            print(f"\nPanel {panel['panel_index']}:")
            print(f"  Outer: {panel['outer_width']} x {panel['outer_height']} mm")
            print(f"  Inner heights: {panel['inner_heights']}")
            if panel["inner_heights"]:
                print(f"  Inner sum: {inner_sum} mm {match_status}")
            print(
                f"  Frame (pixels): {panel['frame_pixel']['w']}x{panel['frame_pixel']['h']} at ({panel['frame_pixel']['x']}, {panel['frame_pixel']['y']})"
            )

            # Check against expected
            if idx < len(expected):
                exp = expected[idx]
                exp_outer = exp["outer"]
                exp_inner = exp["inner"]

                outer_match = (
                    panel["outer_width"] == exp_outer[0]
                    and panel["outer_height"] == exp_outer[1]
                )

                # Check inner heights - accept if sorted lists match (order doesn't matter)
                inner_match = sorted(panel["inner_heights"]) == sorted(exp_inner)

                if outer_match and inner_match:
                    print("  ✅ PASS - Matches expected result")
                else:
                    if not outer_match:
                        print(
                            f"  ❌ FAIL - Expected outer: {exp_outer}, got: ({panel['outer_width']}, {panel['outer_height']})"
                        )
                    if not inner_match:
                        print(
                            f"  ❌ FAIL - Expected inner (sorted): {sorted(exp_inner)}, got: {sorted(panel['inner_heights'])}"
                        )
                    test_passed = False

                total_tests += 1
                if outer_match and inner_match:
                    passed_tests += 1

        print("\n" + "-" * 70)
        print(
            f"Test Case {img_set['id']} Result: {'✅ PASS' if test_passed else '❌ FAIL'}"
        )
        successful = sum(1 for p in panels if p["inner_heights"])
        print(
            f"Panels with inner heights: {successful}/{len(panels)} ({successful/len(panels)*100:.1f}%)"
        )

    # Final summary
    if total_tests > 0:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        print("=" * 70)
