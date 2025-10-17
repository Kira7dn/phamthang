from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import logging
import json
from pathlib import Path

from app.tools.dimension.normalize_frames import normalize_frames_fast
from app.tools.dimension.scale_estimation import estimate_scale

logger = logging.getLogger(__name__)


def _group_frames(
    frames: List[Dict[str, Any]],
    x_tol: float = 20.0,
    y_tol: float = 20.0,
    wh_rel_tol: float = 0.12,
) -> List[Dict[str, Any]]:
    """
    Group frames that are spatially and dimensionally similar.
    Returns groups: {
      'indices': [frame_indices],
      'agg': {'x','y','w','h'} running averages (normalized sizes if present),
      'rows': set(row_group), 'cols': set(col_group)
    }
    """
    groups: List[Dict[str, Any]] = []
    for i, f in enumerate(frames):
        fx = float(f.get("x", 0))
        fy = float(f.get("y", 0))
        fw = float(f.get("normalized_w", f.get("w", 0) or 0))
        fh = float(f.get("normalized_h", f.get("h", 0) or 0))
        row = f.get("row_group")
        col = f.get("col_group")

        placed = False
        for g in groups:
            gx, gy, gw, gh = g["agg"]["x"], g["agg"]["y"], g["agg"]["w"], g["agg"]["h"]
            # position proximity
            if abs(fx - gx) <= x_tol and abs(fy - gy) <= y_tol:
                # size similarity (relative)
                if (abs(fw - gw) / max(1.0, gw) <= wh_rel_tol) and (
                    abs(fh - gh) / max(1.0, gh) <= wh_rel_tol
                ):
                    g["indices"].append(i)
                    if row is not None:
                        g["rows"].add(row)
                    if col is not None:
                        g["cols"].add(col)
                    # update running average
                    n = len(g["indices"])
                    g["agg"]["x"] = (gx * (n - 1) + fx) / n
                    g["agg"]["y"] = (gy * (n - 1) + fy) / n
                    g["agg"]["w"] = (gw * (n - 1) + fw) / n
                    g["agg"]["h"] = (gh * (n - 1) + fh) / n
                    placed = True
                    break
        if not placed:
            groups.append(
                {
                    "indices": [i],
                    "agg": {"x": fx, "y": fy, "w": fw, "h": fh},
                    "rows": set([row]) if row is not None else set(),
                    "cols": set([col]) if col is not None else set(),
                }
            )
    return groups


def _detect_layout_from_groups(groups: List[Dict[str, Any]]) -> str:
    """
    Heuristic: look at distinct row_group and col_group across groups.
    """
    row_vals = {r for g in groups for r in g["rows"] if r is not None}
    col_vals = {c for g in groups for c in g["cols"] if c is not None}
    row_count = len(row_vals) or 1
    col_count = len(col_vals) or 1
    if col_count > row_count and col_count >= 3:
        return "horizontal"
    if row_count > col_count and row_count >= 3:
        return "vertical"
    return "mixed"


def _score_by_rel_tol(
    est: float, n: float, rel_tol: float, prefer_shared: bool = False
) -> float:
    """
    Score mapping:
      orig_rel = abs(n - est) / max(n, est)
      if orig_rel > rel_tol -> score = 0
      else score = 1 - (orig_rel / rel_tol)
      optional small boost if prefer_shared
    Returns float in [0,1].
    """
    if n <= 0 or est <= 0:
        return 0.0
    orig_rel = abs(n - est) / max(n, est)
    if orig_rel > rel_tol:
        return 0.0
    score = 1.0 - (orig_rel / rel_tol)
    if prefer_shared and score > 0:
        score = min(1.0, score * 1.15 + 0.01)
    return float(score)


def generate_outer_candidates(
    frames: List[Dict[str, Any]],
    numbers: List[float],
    scale: float,
    k: int = 3,
    rel_tol: float = 0.20,
    window_ratio: float = 0.5,
    # grouping thresholds
    x_tol: float = 20.0,
    y_tol: float = 20.0,
    wh_rel_tol: float = 0.12,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    For each frame, return candidate width/height values (from `numbers`) ranked by score.

    Returns list length == len(frames), each element:
      {
        "w_cands": [ {"value": n, "score": s, "rel_err": r, "shared": bool}, ... ],
        "h_cands": [ {...}, ... ],
        "est_mm": {"w": est_w, "h": est_h}
      }

    Rules:
      - Primary candidate pool for shared dims = numbers with freq == 1 (unique).
      - Only consider number n if it lies within window_ratio around est (est*(1±window_ratio)).
      - score computed using rel_tol (clamped), prefer shared if group's position implies sharing.
      - If no unique candidate in window, fallback to repeated numbers (lower weight).
    """
    if not frames:
        return []

    # frequency table
    freq = Counter(numbers)
    unique_numbers = sorted(float(n) for n, c in freq.items() if c == 1)
    repeated_numbers = sorted(float(n) for n, c in freq.items() if c > 1)

    # group similar frames to compute candidates once per group
    groups = _group_frames(frames, x_tol=x_tol, y_tol=y_tol, wh_rel_tol=wh_rel_tol)
    layout = _detect_layout_from_groups(groups)
    logger.info("Layout from groups: %s (groups=%d)", layout, len(groups))

    # prepare results slot for each frame
    results: List[Dict[str, Any]] = [
        {"w_cands": [], "h_cands": [], "est_mm": {"w": None, "h": None}} for _ in frames
    ]

    for g_idx, g in enumerate(groups):
        indices = g["indices"]
        agg = g["agg"]
        est_w = float(agg["w"]) * float(scale)
        est_h = float(agg["h"]) * float(scale)

        # preference heuristics: if group has >1 frame and same row_group -> prefer shared height
        prefer_shared_width = False
        prefer_shared_height = False
        if len(indices) > 1:
            row_vals = {
                frames[i].get("row_group")
                for i in indices
                if frames[i].get("row_group") is not None
            }
            col_vals = {
                frames[i].get("col_group")
                for i in indices
                if frames[i].get("col_group") is not None
            }
            if row_vals and len(row_vals) == 1:
                prefer_shared_height = True
            if col_vals and len(col_vals) == 1:
                prefer_shared_width = True

        # window filtering
        w_min, w_max = max(0.0, est_w * (1 - window_ratio)), est_w * (1 + window_ratio)
        h_min, h_max = max(0.0, est_h * (1 - window_ratio)), est_h * (1 + window_ratio)

        if debug:
            logger.info(
                "Group %d indices=%s est_w=%.3f est_h=%.3f w_window=[%.1f,%.1f] h_window=[%.1f,%.1f] prefer_w=%s prefer_h=%s",
                g_idx,
                indices,
                est_w,
                est_h,
                w_min,
                w_max,
                h_min,
                h_max,
                prefer_shared_width,
                prefer_shared_height,
            )

        # collect width candidates from unique numbers first
        w_candidates: List[Dict[str, Any]] = []
        for n in unique_numbers:
            if not (w_min <= n <= w_max):
                continue
            score = _score_by_rel_tol(
                est_w, n, rel_tol, prefer_shared=prefer_shared_width
            )
            if score <= 0.0:
                continue
            rel_err = abs(n - est_w) / max(n, est_w)
            w_candidates.append(
                {"value": n, "score": score, "rel_err": rel_err, "shared": True}
            )

        # fallback to repeated numbers if none found
        if not w_candidates:
            for n in repeated_numbers:
                if not (w_min <= n <= w_max):
                    continue
                score = (
                    _score_by_rel_tol(est_w, n, rel_tol, prefer_shared=False) * 0.6
                )  # downweight
                if score <= 0.0:
                    continue
                rel_err = abs(n - est_w) / max(n, est_w)
                w_candidates.append(
                    {"value": n, "score": score, "rel_err": rel_err, "shared": False}
                )

        # sort and trim top-k
        w_candidates.sort(
            key=lambda x: (-x["score"], x["rel_err"], -float(x["shared"]))
        )
        w_candidates = w_candidates[:k]

        # collect height candidates (same logic)
        h_candidates: List[Dict[str, Any]] = []
        for n in unique_numbers:
            if not (h_min <= n <= h_max):
                continue
            score = _score_by_rel_tol(
                est_h, n, rel_tol, prefer_shared=prefer_shared_height
            )
            if score <= 0.0:
                continue
            rel_err = abs(n - est_h) / max(n, est_h)
            h_candidates.append(
                {"value": n, "score": score, "rel_err": rel_err, "shared": True}
            )

        if not h_candidates:
            for n in repeated_numbers:
                if not (h_min <= n <= h_max):
                    continue
                score = _score_by_rel_tol(est_h, n, rel_tol, prefer_shared=False) * 0.6
                if score <= 0.0:
                    continue
                rel_err = abs(n - est_h) / max(n, est_h)
                h_candidates.append(
                    {"value": n, "score": score, "rel_err": rel_err, "shared": False}
                )

        h_candidates.sort(
            key=lambda x: (-x["score"], x["rel_err"], -float(x["shared"]))
        )
        h_candidates = h_candidates[:k]

        if debug:
            logger.info("Group %d w_cands=%s", g_idx, w_candidates)
            logger.info("Group %d h_cands=%s", g_idx, h_candidates)

        # assign to all frames in this group
        for i in indices:
            results[i]["w_cands"] = w_candidates.copy()
            results[i]["h_cands"] = h_candidates.copy()
            results[i]["est_mm"] = {"w": est_w, "h": est_h}

    return results


def main():
    samples_path = Path(__file__).resolve().parent.parent / "dimension_samples.json"
    with samples_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    for sample in samples:
        sample_id = sample.get("id")
        sample_name = sample.get("name")
        numbers = [
            float(n) for n in sample.get("numbers", []) if isinstance(n, (int, float))
        ]
        frames = sample.get("frames", [])
        normalized_frames = normalize_frames_fast(frames)

        # pixel candidates used to estimate scale
        pixel_candidates: List[float] = []
        for frame in normalized_frames:
            for key in ("normalized_w", "normalized_h"):
                val = frame.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    pixel_candidates.append(float(val))

        scale_result = estimate_scale(pixel_candidates, numbers)
        scale = scale_result.get("scale")
        if not scale:
            logger.info("Sample %s (%s): skipped (no scale)", sample_id, sample_name)
            continue

        logger.info(
            "Sample %s (%s): scale=%.6f inliers=%s rel_err=%s",
            sample_id,
            sample_name,
            scale,
            scale_result.get("inliers_count"),
            scale_result.get("avg_rel_error"),
        )

        candidates = generate_outer_candidates(
            normalized_frames,
            numbers,
            float(scale),
            k=3,
            rel_tol=0.20,
            window_ratio=0.5,
            x_tol=20.0,
            y_tol=20.0,
            wh_rel_tol=0.12,
            debug=True,
        )

        for idx, frame_cands in enumerate(candidates):
            logger.info(
                " Frame %d: est_mm=%s w_cands=%s h_cands=%s",
                idx,
                frame_cands.get("est_mm"),
                frame_cands.get("w_cands"),
                frame_cands.get("h_cands"),
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    main()
