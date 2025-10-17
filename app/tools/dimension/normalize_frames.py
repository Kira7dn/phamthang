from typing import Any, Dict, List
import logging
from statistics import median
import json
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def normalize_frames_fast(
    frames: List[Dict[str, Any]], tolerance_px: int = 5
) -> List[Dict[str, Any]]:
    """
    Optimized & more accurate version of normalize_frames.
    - O(n log n) grouping (sort + linear merge)
    - Uses median for representative width/height
    - Tolerant grouping by x/y (rows, cols)
    - Enforces consistency per row/col
    """
    if not frames:
        return frames

    frames_copy = [dict(f) for f in frames]

    def group_by_value(values: List[float], tolerance: float) -> List[List[float]]:
        """Sort-based clustering: group nearby values within tolerance."""
        values.sort()
        groups, group = [], [values[0]]
        for v in values[1:]:
            if abs(v - group[-1]) <= tolerance:
                group.append(v)
            else:
                groups.append(group)
                group = [v]
        groups.append(group)
        return groups

    def assign_normalized_attr(frames, attr: str, norm_attr: str, tolerance: float):
        """Cluster and assign normalized values."""
        # Build mapping: value → frames
        val_to_frames = {}
        for f in frames:
            val = float(f.get(attr, 0))
            val_to_frames.setdefault(val, []).append(f)

        values = list(val_to_frames.keys())
        grouped_values = group_by_value(values, tolerance)

        for g in grouped_values:
            group_frames = sum((val_to_frames[v] for v in g), [])
            norm_val = int(round(median(g)))
            for f in group_frames:
                f[norm_attr] = norm_val
            logger.debug(f"{attr} group {g} → {norm_attr}={norm_val}")

    # Normalize widths & heights
    assign_normalized_attr(frames_copy, "w", "normalized_w", tolerance_px)
    assign_normalized_attr(frames_copy, "h", "normalized_h", tolerance_px)

    # Group by row (y) and col (x)
    def assign_group_index(frames, attr: str, group_key: str, tolerance: float):
        sorted_frames = sorted(frames, key=lambda f: f.get(attr, 0))
        groups, current = [], [sorted_frames[0]]
        for f in sorted_frames[1:]:
            if abs(f[attr] - current[-1][attr]) <= tolerance:
                current.append(f)
            else:
                groups.append(current)
                current = [f]
        groups.append(current)
        for i, g in enumerate(groups):
            for f in g:
                f[group_key] = i
        return groups

    row_groups = assign_group_index(frames_copy, "y", "row_group", tolerance_px * 2)
    col_groups = assign_group_index(frames_copy, "x", "col_group", tolerance_px * 2)

    # Enforce consistency per row & col
    for g in col_groups:
        if len(g) > 1:
            norm_w = int(round(median([f["normalized_w"] for f in g])))
            for f in g:
                f["normalized_w"] = norm_w
    for g in row_groups:
        if len(g) > 1:
            norm_h = int(round(median([f["normalized_h"] for f in g])))
            for f in g:
                f["normalized_h"] = norm_h

    logger.info(
        f"normalize_frames_fast: {len(frames_copy)} frames → "
        f"{len(row_groups)} rows, {len(col_groups)} cols"
    )
    return frames_copy


def main():
    samples_path = Path(__file__).resolve().parent.parent / "dimension_samples.json"
    with samples_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    for sample in samples:
        frames = sample.get("frames", [])
        normalized = normalize_frames_fast(frames)
        logger.info(
            "Sample %s (%s): %d frames normalized",
            sample.get("id"),
            sample.get("name"),
            len(normalized),
        )
        numbers = sample.get("numbers")
        if numbers:
            logger.info("  Numbers: %s", numbers)
        expected = sample.get("expected")
        if expected:
            logger.info("  Expected: %s", expected)
        for idx, (original, normed) in enumerate(zip(frames, normalized)):
            logger.info(
                "  Frame %d: pos=(%s,%s) size=(%s,%s) → normalized_w=%s normalized_h=%s row_group=%s col_group=%s",
                idx,
                original.get("x"),
                original.get("y"),
                original.get("w"),
                original.get("h"),
                normed.get("normalized_w"),
                normed.get("normalized_h"),
                normed.get("row_group"),
                normed.get("col_group"),
            )


if __name__ == "__main__":
    main()
