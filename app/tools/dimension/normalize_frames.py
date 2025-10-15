"""
Full pipeline:
- normalize_frames (light)
- estimate_scale_ransac
- generate_outer_candidates
- global_assign_outer (greedy local search + column/row consensus)
- recompute_scale
- find_inner_heights_improved (heuristic + backtracking)
- extract_dimension(...) - orchestrator
"""

from collections import Counter, defaultdict
from itertools import product
from typing import Any, Dict, List, Optional, Tuple
import copy
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------
# Frame normalization (simple deterministic grouping)
# ---------------------------
def normalize_frames(
    frames: List[Dict[str, Any]], tolerance_px: int = 5
) -> List[Dict[str, Any]]:
    """
    ENHANCED: Đồng bộ pixel dimensions khi chênh lệch không đáng kể.
    Ví dụ: [205, 204, 206, 205] → normalize về 205 (median/average)

    Simple clustering/grouping for normalized_w, normalized_h, row_group, col_group.
    Does not change original list but returns new list of frames (copied dicts).
    """
    if not frames:
        return frames
    frames_copy = [dict(f) for f in frames]

    # width groups - CHẶT CHẼ HƠN với tolerance nhỏ hơn
    width_groups = []
    for f in frames_copy:
        w = float(f.get("w", 0))
        placed = False
        for g in width_groups:
            if abs(w - g["rep"]) <= tolerance_px:
                g["frames"].append(f)
                # Update representative as median (more robust than mean)
                g["widths"] = g.get("widths", []) + [w]
                g["rep"] = sorted(g["widths"])[len(g["widths"]) // 2]  # median
                placed = True
                break
        if not placed:
            width_groups.append({"rep": w, "frames": [f], "widths": [w]})

    # Assign normalized_w using median of group
    for g in width_groups:
        # Use median để robust hơn với outliers
        widths = sorted(g["widths"])
        norm_w = int(round(widths[len(widths) // 2]))
        for fr in g["frames"]:
            fr["normalized_w"] = norm_w
        logger.debug(f"Width group: {widths} → normalized to {norm_w}")

    # height groups - CHẶT CHẼ HƠN
    height_groups = []
    for f in frames_copy:
        h = float(f.get("h", 0))
        placed = False
        for g in height_groups:
            if abs(h - g["rep"]) <= tolerance_px:
                g["frames"].append(f)
                g["heights"] = g.get("heights", []) + [h]
                g["rep"] = sorted(g["heights"])[len(g["heights"]) // 2]  # median
                placed = True
                break
        if not placed:
            height_groups.append({"rep": h, "frames": [f], "heights": [h]})

    # Assign normalized_h using median of group
    for g in height_groups:
        heights = sorted(g["heights"])
        norm_h = int(round(heights[len(heights) // 2]))
        for fr in g["frames"]:
            fr["normalized_h"] = norm_h
        logger.debug(f"Height group: {heights} → normalized to {norm_h}")

    # row groups by y
    row_groups = []
    for f in frames_copy:
        y = float(f.get("y", 0))
        placed = False
        for g in row_groups:
            if abs(y - g["rep"]) <= tolerance_px * 2:
                g["frames"].append(f)
                g["rep"] = sum(float(x["y"]) for x in g["frames"]) / len(g["frames"])
                placed = True
                break
        if not placed:
            row_groups.append({"rep": y, "frames": [f]})
    for idx, g in enumerate(row_groups):
        for fr in g["frames"]:
            fr["row_group"] = idx

    # col groups by x
    col_groups = []
    for f in frames_copy:
        x = float(f.get("x", 0))
        placed = False
        for g in col_groups:
            if abs(x - g["rep"]) <= tolerance_px * 2:
                g["frames"].append(f)
                g["rep"] = sum(float(x["x"]) for x in g["frames"]) / len(g["frames"])
                placed = True
                break
        if not placed:
            col_groups.append({"rep": x, "frames": [f]})
    for idx, g in enumerate(col_groups):
        for fr in g["frames"]:
            fr["col_group"] = idx

    # ENFORCE CONSISTENCY: Cùng col → same normalized_w
    for col_idx, col_g in enumerate(col_groups):
        col_frames = col_g["frames"]
        if len(col_frames) > 1:
            # Take average or median width
            widths = [f.get("normalized_w", f.get("w", 0)) for f in col_frames]
            avg_w = int(round(sum(widths) / len(widths)))
            for fr in col_frames:
                fr["normalized_w"] = avg_w

    # ENFORCE CONSISTENCY: Cùng row → same normalized_h
    for row_idx, row_g in enumerate(row_groups):
        row_frames = row_g["frames"]
        if len(row_frames) > 1:
            # Take average or median height
            heights = [f.get("normalized_h", f.get("h", 0)) for f in row_frames]
            avg_h = int(round(sum(heights) / len(heights)))
            for fr in row_frames:
                fr["normalized_h"] = avg_h

    logger.debug(
        f"normalize_frames: widths={len(width_groups)}, heights={len(height_groups)}, rows={len(row_groups)}, cols={len(col_groups)}"
    )
    return frames_copy
