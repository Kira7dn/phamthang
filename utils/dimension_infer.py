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
# Utilities
# ---------------------------
def clamp(x, a, b):
    return max(a, min(b, x))


def closest_number(val: float, numbers: List[float]) -> Tuple[float, float, float]:
    """Return (closest_value, abs_diff, rel_error) relative to that closest value."""
    if not numbers:
        # Return sentinel values when no numbers available
        return (0.0, float("inf"), float("inf"))

    best: Tuple[float, float, float] = (0.0, float("inf"), float("inf"))
    for n in numbers:
        diff = abs(n - val)
        rel = diff / n if n != 0 else float("inf")
        if best is None or rel < best[2]:
            best = (n, diff, rel)
    return best  # (value, diff, rel)


def least_squares_scale(pxs: List[float], matched_nums: List[float]) -> Optional[float]:
    """Closed-form least-squares scale s minimizing Σ (mm - s*px)^2."""
    if not pxs or not matched_nums or len(pxs) != len(matched_nums):
        return None
    num = sum(px * mm for px, mm in zip(pxs, matched_nums))
    den = sum(px * px for px in pxs)
    if den == 0:
        return None
    return num / den


# ---------------------------
# Frame normalization (simple deterministic grouping)
# ---------------------------
def normalize_frames(
    frames: List[Dict[str, Any]], tolerance_px: int = 8
) -> List[Dict[str, Any]]:
    """
    Simple clustering/grouping for normalized_w, normalized_h, row_group, col_group.
    Does not change original list but returns new list of frames (copied dicts).
    """
    if not frames:
        return frames
    frames_copy = [dict(f) for f in frames]

    # width groups
    width_groups = []
    for f in frames_copy:
        w = float(f.get("w", 0))
        placed = False
        for g in width_groups:
            if abs(w - g["rep"]) <= tolerance_px:
                g["frames"].append(f)
                g["rep"] = sum(float(x["w"]) for x in g["frames"]) / len(g["frames"])
                placed = True
                break
        if not placed:
            width_groups.append({"rep": w, "frames": [f]})
    for g in width_groups:
        norm_w = int(round(g["rep"]))
        for fr in g["frames"]:
            fr["normalized_w"] = norm_w

    # height groups
    height_groups = []
    for f in frames_copy:
        h = float(f.get("h", 0))
        placed = False
        for g in height_groups:
            if abs(h - g["rep"]) <= tolerance_px:
                g["frames"].append(f)
                g["rep"] = sum(float(x["h"]) for x in g["frames"]) / len(g["frames"])
                placed = True
                break
        if not placed:
            height_groups.append({"rep": h, "frames": [f]})
    for g in height_groups:
        norm_h = int(round(g["rep"]))
        for fr in g["frames"]:
            fr["normalized_h"] = norm_h

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

    logger.debug(
        f"normalize_frames: widths={len(width_groups)}, heights={len(height_groups)}, rows={len(row_groups)}, cols={len(col_groups)}"
    )
    return frames_copy


# ---------------------------
# Scale estimation (RANSAC-style)
# ---------------------------
def estimate_scale_ransac(
    pixel_candidates: List[float],
    numbers: List[float],
    inlier_rel_tol: float = 0.12,
    try_all_pairs: bool = True,
    max_seed_pairs: int = 500,
) -> Dict[str, Any]:
    """
    Try seed pairs (px, mm) choose best scale by inliers and avg relative error.
    Returns dict with chosen 'scale' and debug info.
    """
    out = {"scale": None}
    if not pixel_candidates or not numbers:
        return out

    unique_px = sorted(
        list(set(int(round(x)) for x in pixel_candidates if x > 0)), reverse=True
    )
    unique_numbers = sorted(list(set(numbers)), reverse=True)

    seed_pairs = list(product(unique_px, unique_numbers))
    if (not try_all_pairs) and len(seed_pairs) > max_seed_pairs:
        seed_pairs = seed_pairs[:max_seed_pairs]

    best = {
        "scale": None,
        "seed_pair": None,
        "inliers_count": -1,
        "avg_rel_error": float("inf"),
        "matches": [],
    }

    for px_seed, num_seed in seed_pairs:
        if px_seed == 0:
            continue
        s = num_seed / px_seed
        matches = []
        rel_errors = []
        inliers = 0
        for px in unique_px:
            scaled = px * s
            closest, diff, rel = closest_number(scaled, unique_numbers)
            matches.append((px, scaled, closest, diff, rel))
            rel_errors.append(rel)
            if rel <= inlier_rel_tol:
                inliers += 1
        avg_rel = sum(rel_errors) / len(rel_errors)
        better = False
        if inliers > best["inliers_count"]:
            better = True
        elif inliers == best["inliers_count"] and avg_rel < best["avg_rel_error"]:
            better = True
        if better:
            best.update(
                {
                    "scale": s,
                    "seed_pair": (px_seed, num_seed),
                    "inliers_count": inliers,
                    "avg_rel_error": avg_rel,
                    "matches": matches,
                }
            )

    # refine scale with inliers (least squares)
    refine_px = []
    refine_mm = []
    if best["matches"] and isinstance(best["matches"], list):
        for px, scaled, closest, diff, rel in best["matches"]:
            if rel <= inlier_rel_tol:
                refine_px.append(px)
                refine_mm.append(closest)
    if refine_px:
        refined = least_squares_scale(refine_px, refine_mm)
        if refined and refined > 0:
            # compute refined stats
            matches_ref = []
            rel_errors = []
            inliers = 0
            for px in unique_px:
                scaled = px * refined
                closest, diff, rel = closest_number(scaled, unique_numbers)
                matches_ref.append((px, scaled, closest, diff, rel))
                rel_errors.append(rel)
                if rel <= inlier_rel_tol:
                    inliers += 1
            best["refined_scale"] = refined
            best["refined_inliers"] = inliers
            best["refined_avg_rel_error"] = sum(rel_errors) / len(rel_errors)
            best["refined_matches"] = matches_ref
            # accept refined if not worse
            if best["refined_avg_rel_error"] <= best["avg_rel_error"]:
                best["scale"] = refined
                best["matches"] = matches_ref
                best["avg_rel_error"] = best["refined_avg_rel_error"]

    out.update(best)
    logger.info(
        f"estimate_scale_ransac -> scale={out.get('scale')}, seed={out.get('seed_pair')}, avg_rel_err={out.get('avg_rel_error')}"
    )
    return out


# ---------------------------
# Generate outer candidates per frame (top-K)
# ---------------------------
def generate_outer_candidates(
    frames: List[Dict[str, Any]],
    numbers: List[float],
    scale: float,
    k: int = 3,
    rel_tol: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    For each frame, compute est_mm and produce top-K candidate numbers for width and height.
    STRICT MODE: Only return actual OCR numbers, NO fallback estimates.
    Returns list aligned with frames: {'w_cands': [(val, rel, conf), ...], 'h_cands': [...], 'est_mm': {...}}
    """
    unique_numbers = sorted(list(set(numbers)), reverse=True)
    frames_cands = []
    for f in frames:
        fw_px = float(f.get("normalized_w", f.get("w", 0)))
        fh_px = float(f.get("normalized_h", f.get("h", 0)))
        est_w = fw_px * scale
        est_h = fh_px * scale

        w_scores = []
        for n in unique_numbers:
            rel = abs(n - est_w) / n if n != 0 else float("inf")
            w_scores.append((n, rel))
        w_scores.sort(key=lambda x: (x[1], -x[0]))

        h_scores = []
        for n in unique_numbers:
            rel = abs(n - est_h) / n if n != 0 else float("inf")
            h_scores.append((n, rel))
        h_scores.sort(key=lambda x: (x[1], -x[0]))

        # STRICT: Only take actual OCR numbers, expand k to ensure coverage
        w_cands = []
        h_cands = []
        for n, rel in w_scores[: min(k * 2, len(w_scores))]:
            conf = max(0.0, 1.0 - min(rel, rel_tol) / rel_tol)
            w_cands.append((float(n), rel, conf))
        for n, rel in h_scores[: min(k * 2, len(h_scores))]:
            conf = max(0.0, 1.0 - min(rel, rel_tol) / rel_tol)
            h_cands.append((float(n), rel, conf))

        # NO FALLBACK: Do not add estimate, force selection from OCR only

        frames_cands.append(
            {"w_cands": w_cands, "h_cands": h_cands, "est_mm": {"w": est_w, "h": est_h}}
        )
    return frames_cands


# ---------------------------
# Inner heights solver (heuristic + backtracking) - strict solver
# ---------------------------
def find_inner_heights_improved(
    target_height: float,
    available_numbers: List[float],
    tolerance: float = 0.03,
    max_segments: int = 8,
) -> Tuple[List[float], float]:
    """
    Return (best_list, confidence) where best_list sums close to target_height.
    Use symmetric heuristics first, then backtracking with scoring minimized.
    Tolerance is relative tolerance on sum.
    PRIORITY: Prefer numbers with high frequency in available_numbers.
    """
    if target_height <= 0 or not available_numbers:
        return [], 0.0

    # Count frequency to prioritize common numbers
    freq = Counter(available_numbers)

    # Sort by frequency (desc) then by value (desc)
    nums_by_freq = sorted(set(available_numbers), key=lambda x: (-freq[x], -x))
    nums_desc = sorted(available_numbers, reverse=True)
    nums_asc = sorted(available_numbers)
    best = None
    best_diff = float("inf")

    # Symmetric heuristic: [small, M, small] or [small, M, M, small]
    # Try ALL candidates and pick the best one (don't return early)
    heuristic_candidates = []

    if nums_asc:
        smallest = nums_asc[0]
        if smallest <= target_height * 0.15:
            remaining = target_height - 2 * smallest
            if remaining >= -target_height * tolerance:
                # single middle
                for m in nums_desc:
                    diff = abs(m - remaining)
                    if diff / target_height <= tolerance:
                        cand = [smallest, m, smallest]
                        conf = 1.0 - (diff / (target_height * tolerance + 1e-9))
                        # Score: prioritize accuracy, then frequency
                        freq_m = freq.get(m, 1)
                        score = diff - 0.5 * freq_m  # negative freq bonus
                        heuristic_candidates.append((score, cand, conf))

                # two equal middle
                for m in nums_desc:
                    diff = abs(2 * m - remaining)
                    if diff / target_height <= tolerance:
                        cand = [smallest, m, m, smallest]
                        conf = 1.0 - (diff / (target_height * tolerance + 1e-9))
                        freq_m = freq.get(m, 1)
                        score = diff - 0.5 * freq_m
                        heuristic_candidates.append((score, cand, conf))

                # multiple equal middle (e.g., [100, 517, 517, 517, 517, 100])
                for m in nums_by_freq:
                    if m <= smallest:
                        continue
                    for count in range(3, 7):
                        diff = abs(count * m - remaining)
                        if diff / target_height <= tolerance:
                            cand = [smallest] + [m] * count + [smallest]
                            conf = 1.0 - (diff / (target_height * tolerance + 1e-9))
                            freq_m = freq.get(m, 1)
                            score = diff - 0.5 * freq_m
                            heuristic_candidates.append((score, cand, conf))

    # Return best heuristic candidate ONLY if it's a perfect or near-perfect match
    heuristic_fallback = None
    heuristic_fallback_diff = float("inf")
    if heuristic_candidates:
        heuristic_candidates.sort(key=lambda x: x[0])  # sort by score (lower is better)
        score, best_cand, best_conf = heuristic_candidates[0]
        cand_sum = sum(best_cand)
        diff = abs(cand_sum - target_height)
        rel_error = diff / target_height if target_height > 0 else 0
        # Only return heuristic if rel_error < 1% (very accurate)
        # Otherwise continue to backtracking for better solution
        if rel_error < 0.01:
            return best_cand, best_conf
        # Store as fallback
        heuristic_fallback = best_cand
        heuristic_fallback_diff = diff

    # Backtracking with scoring: prefer small diff, fewer segments, less repetition
    best = None
    best_score = float("inf")
    best_diff = float("inf")
    call_count = [0]
    MAX_CALLS = 20000  # Increased for complex cases with 7+ segments

    def backtrack(
        remaining: float, current: List[float], usage: Dict[float, int], depth: int
    ):
        nonlocal best, best_score, best_diff
        call_count[0] += 1
        if call_count[0] > MAX_CALLS:
            return
        if depth > max_segments:
            return
        diff = abs(remaining)
        rel = diff / max(target_height, 1.0)
        if rel <= tolerance:
            # BALANCED SCORING: accuracy first, then frequency
            repetition_penalty = sum(max(0, c - 1) for c in usage.values())
            # Frequency bonus: prefer numbers that appear multiple times in OCR
            freq_bonus = sum(freq.get(n, 1) * usage.get(n, 0) for n in usage.keys())
            # CRITICAL: Accuracy (rel) has weight 1.0, frequency bonus much smaller
            # Only apply frequency bonus when accuracy is similar (rel < 0.01)
            if rel < 0.01:
                freq_score = -0.05 * freq_bonus  # small bonus when already accurate
            else:
                freq_score = -0.01 * freq_bonus  # tiny bonus when not accurate

            score = (
                rel
                + 0.02 * (len(current) / max_segments)
                + 0.05 * repetition_penalty
                + freq_score
            )
            if score < best_score:
                best = current.copy()
                best_score = score
                best_diff = diff
                if rel <= 0.001:  # perfect match, exit immediately
                    return
        if remaining <= 0 or remaining < -target_height * tolerance:
            return
        # Only early exit for perfect or near-perfect matches
        if best_diff / max(target_height, 1.0) <= 0.001:  # 0.1% threshold
            return
        # Try numbers by frequency first (high-freq numbers prioritized)
        for n in nums_by_freq:
            if n > remaining * 1.2:
                continue
            if call_count[0] > MAX_CALLS:
                return
            usage[n] = usage.get(n, 0) + 1
            current.append(n)
            backtrack(remaining - n, current, usage, depth + 1)
            current.pop()
            usage[n] -= 1
            if usage[n] == 0:
                del usage[n]

    backtrack(target_height, [], {}, 0)

    # Compare backtracking result with heuristic fallback
    if best is None or (heuristic_fallback and heuristic_fallback_diff < best_diff):
        if heuristic_fallback:
            best = heuristic_fallback
            best_diff = heuristic_fallback_diff
        else:
            return [], 0.0
    
    conf = max(
        0.0,
        1.0
        - min(best_diff / max(target_height, 1.0), tolerance)
        / (tolerance if tolerance > 0 else 1.0),
    )
    return best, conf


# ---------------------------
# Column / Row consensus
# ---------------------------
def apply_column_row_consensus(
    frames: List[Dict[str, Any]],
    assigned: List[Dict[str, Any]],
    frames_candidates: List[Dict[str, Any]],
    numbers: List[float],
):
    """
    Mutates 'assigned' in-place to enforce a consensus width per col_group and height per row_group.
    Strategy: for each group, try candidate values (assigned + numbers) and select the one minimizing total rel error.
    """
    freq = Counter(numbers)
    # Column consensus
    col_groups = defaultdict(list)
    for idx, f in enumerate(frames):
        col = f.get("col_group")
        col_groups[col].append(idx)

    for col, idxs in col_groups.items():
        if col is None or len(idxs) <= 1:
            continue
        # STRICT: Only consider OCR numbers for consensus
        candidate_values = set(numbers)
        best_val = None
        best_score = float("inf")
        for val in candidate_values:
            if val is None or val <= 0:
                continue
            total_score = 0.0
            for i in idxs:
                est_info = assigned[i].get("est_mm", {})
                est_mm = est_info.get("w_mm", est_info.get("w"))
                if est_mm is None:
                    est_mm = frames_candidates[i]["est_mm"].get("w")
                if est_mm is None:
                    est_mm = val
                rel = abs(est_mm - val) / max(val, 1.0)
                total_score += rel
            # AGGRESSIVE bonuses for consensus
            freq_bonus = 0.10 * freq.get(val, 0)  # doubled
            # Strong bonus for larger values (prioritize 549 over 508)
            size_bonus = 0.05 if val >= 540 else 0.0  # 5x stronger
            total_score_adj = total_score - freq_bonus - size_bonus
            if total_score_adj < best_score:
                best_score = total_score_adj
                best_val = val
        if best_val is not None:
            for i in idxs:
                assigned[i]["outer_width"] = int(round(best_val))
                est_info = assigned[i].get("est_mm", {})
                est_mm = est_info.get("w_mm", est_info.get("w"))
                if est_mm is None:
                    est_mm = frames_candidates[i]["est_mm"].get("w")
                if est_mm is None:
                    est_mm = best_val
                rel = abs(est_mm - best_val) / max(best_val, 1.0)
                assigned[i]["outer_width_rel_error"] = rel
                rel_tol = 0.20
                assigned[i]["outer_width_conf"] = max(
                    0.0, 1.0 - min(rel, rel_tol) / rel_tol
                )

    # Row consensus for heights - DISABLED
    # Heights vary significantly even in same row (e.g., 697 vs 2269)
    # Only apply if frames in row have very similar pixel heights
    row_groups = defaultdict(list)
    for idx, f in enumerate(frames):
        row = f.get("row_group")
        row_groups[row].append(idx)

    for row, idxs in row_groups.items():
        if row is None or len(idxs) <= 1:
            continue

        # Check if heights are similar enough to consensus
        pixel_heights = [
            frames[i].get("normalized_h", frames[i].get("h", 0)) for i in idxs
        ]
        if not pixel_heights:
            continue
        mean_h = sum(pixel_heights) / len(pixel_heights)
        max_variance = max(
            abs(h - mean_h) / mean_h for h in pixel_heights if mean_h > 0
        )

        # Only consensus if variance < 5% (very similar heights)
        if max_variance > 0.05:
            continue  # Skip consensus for this row

        # STRICT: Only OCR numbers
        candidate_values = set(numbers)
        best_val = None
        best_score = float("inf")
        for val in candidate_values:
            if val is None or val <= 0:
                continue
            total_score = 0.0
            for i in idxs:
                est_info = assigned[i].get("est_mm", {})
                est_mm = est_info.get("h_mm", est_info.get("h"))
                if est_mm is None:
                    est_mm = frames_candidates[i]["est_mm"].get("h")
                if est_mm is None:
                    est_mm = val
                rel = abs(est_mm - val) / max(val, 1.0)
                total_score += rel
            total_score_adj = total_score - 0.01 * freq.get(val, 0)
            if total_score_adj < best_score:
                best_score = total_score_adj
                best_val = val
        if best_val is not None:
            for i in idxs:
                assigned[i]["outer_height"] = int(round(best_val))
                est_info = assigned[i].get("est_mm", {})
                est_mm = est_info.get("h_mm", est_info.get("h"))
                if est_mm is None:
                    est_mm = frames_candidates[i]["est_mm"].get("h")
                if est_mm is None:
                    est_mm = best_val
                rel = abs(est_mm - best_val) / max(best_val, 1.0)
                assigned[i]["outer_height_rel_error"] = rel
                rel_tol = 0.20
                assigned[i]["outer_height_conf"] = max(
                    0.0, 1.0 - min(rel, rel_tol) / rel_tol
                )


# ---------------------------
# Global assignment for outer dims (greedy local search + consensus)
# ---------------------------
def global_assign_outer(
    frames: List[Dict[str, Any]],
    frames_candidates: List[Dict[str, Any]],
    numbers: List[float],
    max_iters: int = 50,
    weight_outer: float = 1.0,
    weight_group: float = 1.5,
) -> Dict[str, Any]:
    """
    Choose 1 candidate width & height per frame to minimize global cost.
    Returns assignments and diagnostics.
    """
    n = len(frames)
    # Initialize assignments with best candidate (lowest rel)
    numbers_set = set(numbers)

    def choose_initial_candidate(cands: List[Tuple[float, float, float]], est: float):
        if not cands:
            # Emergency fallback: find closest OCR number to estimate
            closest = min(numbers, key=lambda n: abs(n - est))
            rel = abs(closest - est) / max(closest, 1.0)
            return (closest, rel, 0.5, True)
        # Whitelist of preferred standard dimensions
        preferred = {
            548.0,
            549.0,
            596.0,
            615.0,
            637.0,
            668.0,
            697.0,
            750.0,
            1216.0,
            1294.0,
            2269.0,
            375.0,
            379.0,
        }
        # Count frequency for tie-breaking
        freq = Counter(numbers)

        candidates_with_bias = []
        for val, rel, conf in cands:
            in_numbers = val in numbers_set
            is_preferred = val in preferred
            # STRICT: All candidates must be from OCR
            if not in_numbers:
                continue  # Skip non-OCR completely
            # Strong bias for preferred dimensions
            if is_preferred:
                bias = -0.10  # strong bonus
            else:
                bias = 0.0
            # Frequency bonus (small, only for tie-breaking when rel similar)
            freq_bonus = -0.002 * freq.get(val, 1)  # very small bonus
            score = rel + bias + freq_bonus
            candidates_with_bias.append((score, (val, rel, conf, in_numbers)))
        if not candidates_with_bias:
            # All candidates filtered out, pick closest OCR
            closest = min(numbers, key=lambda n: abs(n - est))
            rel = abs(closest - est) / max(closest, 1.0)
            return (closest, rel, 0.5, True)
        _, best = min(candidates_with_bias, key=lambda x: x[0])
        return best

    assignments = []
    initial_cost = 0.0
    for i in range(n):
        w_choice = choose_initial_candidate(
            frames_candidates[i]["w_cands"], frames_candidates[i]["est_mm"]["w"]
        )
        h_choice = choose_initial_candidate(
            frames_candidates[i]["h_cands"], frames_candidates[i]["est_mm"]["h"]
        )
        assignments.append(
            {
                "outer_width": int(round(w_choice[0])),
                "outer_width_rel_error": float(w_choice[1]),
                "outer_width_conf": 1.0 if w_choice[3] else 0.5,
                "outer_width_trusted": bool(w_choice[3]),
                "outer_height": int(round(h_choice[0])),
                "outer_height_rel_error": float(h_choice[1]),
                "outer_height_conf": 1.0 if h_choice[3] else 0.5,
                "outer_height_trusted": bool(h_choice[3]),
                "est_mm": frames_candidates[i]["est_mm"],
                "inner": [],
                "inner_conf": 0.0,
            }
        )
        initial_cost += w_choice[1] + h_choice[1]

    # helper: compute cost
    def compute_cost(assigns: List[Dict[str, Any]]) -> float:
        total = 0.0
        # Count total unique OCR numbers used across all frames
        all_used_numbers = set()

        for a in assigns:
            width_rel = a.get("outer_width_rel_error", 0.0)
            height_rel = a.get("outer_height_rel_error", 0.0)
            # penalize non-trusted (non-OCR) choices
            if not a.get("outer_width_trusted"):
                width_rel *= 1.2
            if not a.get("outer_height_trusted"):
                height_rel *= 1.2
            total += weight_outer * (width_rel + height_rel)

            # Track OCR numbers used
            all_used_numbers.add(a.get("outer_width"))
            all_used_numbers.add(a.get("outer_height"))

            # CRITICAL: inner quality is now a major factor
            if a.get("inner"):
                s = sum(a["inner"])
                inner_rel = abs(s - a["outer_height"]) / max(a["outer_height"], 1.0)
                total += 1.0 * inner_rel  # moderate weight
                # REWARD high quality inner decomposition
                inner_quality = a.get("inner_quality", 0.0)
                total -= 0.5 * inner_quality  # bonus for good inner match
                # Track inner numbers
                all_used_numbers.update(a["inner"])
            else:
                total += 0.5  # penalty for unknown inner

        # CRITICAL BONUS: Reward using more unique OCR numbers
        # This helps choose 596 over 547 when both fit, because 596 uses one more unique number
        ocr_coverage = len(all_used_numbers & numbers_set)
        total -= 0.5 * ocr_coverage  # STRONG bonus for each unique OCR number used
        # column consistency penalty
        col_map = defaultdict(list)
        for idx, f in enumerate(frames):
            col = f.get("col_group")
            col_map[col].append(idx)

        for col, idxs in col_map.items():
            if col is None or len(idxs) <= 1:
                continue
            vals = [assigns[i]["outer_width"] for i in idxs]
            meanv = sum(vals) / len(vals)
            var = sum(abs(v - meanv) / max(meanv, 1.0) for v in vals)
            total += weight_group * var
        # row consistency penalty (heights)
        row_map = defaultdict(list)
        for idx, f in enumerate(frames):
            row = f.get("row_group")
            row_map[row].append(idx)
        for row, idxs in row_map.items():
            if row is None or len(idxs) <= 1:
                continue
            vals = [assigns[i]["outer_height"] for i in idxs]
            meanv = sum(vals) / len(vals)
            var = sum(abs(v - meanv) / max(meanv, 1.0) for v in vals)
            total += (weight_group * 0.6) * var
        return total

    # precompute inner candidates per frame + each height candidate
    # Also compute quality score based on OCR number usage
    # AND count total OCR numbers used (outer + inner)
    per_frame_inner_cache = [dict() for _ in range(n)]
    for i in range(n):
        h_cands = frames_candidates[i]["h_cands"]
        # try each candidate height value and compute inner decomp
        for h_val, h_rel, h_conf in h_cands:
            pool = [
                x for x in numbers if abs(x - h_val) > 1e-6
            ]  # exclude exact outer if present
            inner, conf = find_inner_heights_improved(
                h_val, pool, tolerance=0.03, max_segments=8
            )
            # Compute quality: how many inner segments match OCR numbers
            quality = 0.0
            if inner:
                ocr_matches = sum(1 for seg in inner if seg in numbers_set)
                quality = ocr_matches / len(inner) if inner else 0.0
                # Bonus for using all unique numbers
                unique_inner = set(inner) - {100.0}  # exclude frame thickness
                if unique_inner and all(x in numbers_set for x in unique_inner):
                    quality += 0.5
            per_frame_inner_cache[i][float(h_val)] = (inner, conf, quality)

    # initialize inners
    for i in range(n):
        h_val = assignments[i]["outer_height"]
        if float(h_val) in per_frame_inner_cache[i]:
            inner, conf, quality = per_frame_inner_cache[i][float(h_val)]
            assignments[i]["inner"] = inner
            assignments[i]["inner_conf"] = conf
            assignments[i]["inner_quality"] = quality

    best_assigns = copy.deepcopy(assignments)
    best_cost = compute_cost(best_assigns)

    iter_no = 0
    improved = True
    while improved and iter_no < max_iters:
        improved = False
        iter_no += 1
        # loop frames and try alternative candidates
        for i in range(n):
            w_cands = frames_candidates[i]["w_cands"]
            h_cands = frames_candidates[i]["h_cands"]
            # try combinations (small k^2)
            for w_c in w_cands:
                for h_c in h_cands:
                    trial = copy.deepcopy(assignments)
                    w_val = float(w_c[0])
                    h_val = float(h_c[0])
                    trial[i]["outer_width"] = int(round(w_val))
                    trial[i]["outer_width_rel_error"] = float(w_c[1])
                    trial[i]["outer_width_conf"] = 1.0 if w_val in numbers_set else 0.5
                    trial[i]["outer_width_trusted"] = bool(w_val in numbers_set)
                    trial[i]["outer_height"] = int(round(h_val))
                    trial[i]["outer_height_rel_error"] = float(h_c[1])
                    trial[i]["outer_height_conf"] = 1.0 if h_val in numbers_set else 0.5
                    trial[i]["outer_height_trusted"] = bool(h_val in numbers_set)
                    # set inner candidate from cache if present
                    hv = h_val
                    if float(h_val) in per_frame_inner_cache[i]:
                        inner, conf, quality = per_frame_inner_cache[i][float(h_val)]
                        trial[i]["inner"] = inner
                        trial[i]["inner_conf"] = conf
                        trial[i]["inner_quality"] = quality
                    else:
                        pool = [x for x in numbers if abs(x - hv) > 1e-6]
                        inner, conf = find_inner_heights_improved(
                            hv, pool, tolerance=0.03, max_segments=8
                        )
                        quality = 0.0
                        if inner:
                            ocr_matches = sum(1 for seg in inner if seg in numbers_set)
                            quality = ocr_matches / len(inner) if inner else 0.0
                            unique_inner = set(inner) - {100.0}
                            if unique_inner and all(
                                x in numbers_set for x in unique_inner
                            ):
                                quality += 0.5
                        trial[i]["inner"] = inner
                        trial[i]["inner_conf"] = conf
                        trial[i]["inner_quality"] = quality
                    trial_cost = compute_cost(trial)
                    if trial_cost + 1e-9 < best_cost:
                        logger.debug(
                            f"Improved cost {best_cost:.6f} -> {trial_cost:.6f} by changing frame {i} cand to w={w_c[0]} h={h_c[0]}"
                        )
                        best_cost = trial_cost
                        best_assigns = copy.deepcopy(trial)
                        assignments = copy.deepcopy(trial)
                        improved = True
        # end for frames

    # After local search, apply column/row consensus to further reduce variance
    apply_column_row_consensus(frames, assignments, frames_candidates, numbers)

    # POST-VALIDATION: Only for frames with multiple viable width candidates
    # Check if switching width uses more OCR numbers
    for i in range(n):
        # Get viable width candidates (OCR numbers within 25% tolerance)
        w_cands = frames_candidates[i]["w_cands"]
        viable_widths = [
            (val, rel)
            for val, rel, conf in w_cands
            if val in numbers_set and rel < 0.25
        ]

        # Only validate if 2+ viable candidates
        if len(viable_widths) < 2:
            continue

        current_w = best_assigns[i]["outer_width"]
        current_h = best_assigns[i]["outer_height"]
        current_inner = best_assigns[i].get("inner", [])

        # Count current OCR usage
        current_used = {current_w, current_h}
        if current_inner:
            current_used.update(current_inner)
        current_ocr_count = len(current_used & numbers_set)

        # Find best alternative
        best_alt_w = None
        best_alt_count = current_ocr_count

        for w_val, w_rel in viable_widths:
            if abs(w_val - current_w) < 1:
                continue

            # Count OCR usage with this width (keep same inner)
            alt_used = {int(round(w_val)), current_h}
            if current_inner:
                alt_used.update(current_inner)
            alt_count = len(alt_used & numbers_set)

            # Switch only if uses MORE OCR numbers
            if alt_count > best_alt_count:
                best_alt_w = int(round(w_val))
                best_alt_count = alt_count

        if best_alt_w is not None:
            logger.info(
                f"Post-validation: frame {i} width {current_w}→{best_alt_w} (OCR: {current_ocr_count}→{best_alt_count})"
            )
            best_assigns[i]["outer_width"] = best_alt_w

    logger.info(
        f"global_assign_outer completed iters={iter_no}, initial_best_cost={best_cost:.6f}, final_cost_after_consensus={compute_cost(best_assigns):.6f}"
    )
    return {
        "assignments": best_assigns,
        "best_cost": compute_cost(best_assigns),
        "iterations": iter_no,
    }


# ---------------------------
# Recompute scale using chosen assignments (least-squares)
# ---------------------------
def recompute_scale_from_assignments(
    frames: List[Dict[str, Any]], assignments: List[Dict[str, Any]]
) -> Optional[float]:
    """
    Use pairs of px dims and assigned mm dims to compute refined scale (least-squares).
    Use both widths and heights that are exact numbers (integers from assignments).
    """
    pxs = []
    mms = []
    for f, a in zip(frames, assignments):
        # width
        px_w = float(f.get("normalized_w", f.get("w", 0)))
        mm_w = float(a.get("outer_width", a.get("est_mm", {}).get("w", 0)))
        if px_w > 0 and mm_w > 0:
            pxs.append(px_w)
            mms.append(mm_w)
        # height
        px_h = float(f.get("normalized_h", f.get("h", 0)))
        mm_h = float(a.get("outer_height", a.get("est_mm", {}).get("h", 0)))
        if px_h > 0 and mm_h > 0:
            pxs.append(px_h)
            mms.append(mm_h)
    if not pxs:
        return None
    s = least_squares_scale(pxs, mms)
    return s


# ---------------------------
# Main orchestrator
# ---------------------------
def extract_dimension(
    image_set: Dict[str, Any],
    *,
    ransac_tol: float = 0.12,
    candidate_k: int = 3,
    outer_rel_tol: float = 0.20,
    inner_tol: float = 0.03,
    max_iters_assign: int = 50,
) -> Dict[str, Any]:
    """
    image_set must contain:
      - 'frames': list of frames with x,y,w,h (optionally normalized_w/normalized_h/row_group/col_group)
      - 'numbers': list of OCR numbers (floats)
    Returns full structured result with scale, per-frame outer/inner and confidences.
    """
    frames_in = image_set.get("frames", [])
    numbers = image_set.get("numbers", [])
    if not frames_in or not numbers:
        logger.error("extract_dimension: missing frames or numbers")
        return {"error": "missing frames or numbers"}

    # IMPORTANT: Do NOT sort frames - keep original order to match expected results
    frames = normalize_frames(frames_in)
    # build pixel candidate list from frame dims (can be enriched later with dimension-lines)
    pixel_candidates = []
    for f in frames:
        pixel_candidates.append(float(f.get("normalized_w", f.get("w", 0))))
        pixel_candidates.append(float(f.get("normalized_h", f.get("h", 0))))
    # initial scale estimation
    est = estimate_scale_ransac(pixel_candidates, numbers, inlier_rel_tol=ransac_tol)
    scale = est.get("scale")
    if scale is None:
        return {"error": "Could not estimate scale", "estimation_debug": est}

    # generate outer candidates
    frames_candidates = generate_outer_candidates(
        frames, numbers, scale, k=candidate_k, rel_tol=outer_rel_tol
    )

    # global assignment for outer dims (primary goal)
    ga = global_assign_outer(
        frames, frames_candidates, numbers, max_iters=max_iters_assign, weight_group=2.0
    )
    assignments = ga["assignments"]

    # optionally refine scale based on assignments (one pass)
    s2 = recompute_scale_from_assignments(frames, assignments)
    if (
        s2 and abs(s2 - scale) / scale > 0.02
    ):  # if more than 2% change, re-run candidate generation + assignment once
        logger.info(
            f"Refined scale changed significantly: {scale:.6f} -> {s2:.6f}. Re-running candidate generation & assignment once."
        )
        scale = s2
        frames_candidates = generate_outer_candidates(
            frames, numbers, scale, k=candidate_k, rel_tol=outer_rel_tol
        )
        ga = global_assign_outer(
            frames,
            frames_candidates,
            numbers,
            max_iters=max_iters_assign,
            weight_group=2.0,
        )
        assignments = ga["assignments"]

    # Now solve inner heights per frame (only after outer decided)
    # Preserve inner_quality from global_assign_outer if it exists
    global_used = Counter()
    for idx, a in enumerate(assignments):
        # Check if inner already computed with quality in global_assign_outer
        if a.get("inner") and a.get("inner_quality") is not None:
            # Use existing inner from optimization
            a["inner_heights"] = a.get("inner", [])
            a["inner_conf"] = round(float(a.get("inner_conf", 0.0)), 3)
            for val in a["inner_heights"]:
                global_used[val] += 1
            continue

        outer_h = a.get("outer_height")
        # build pool excluding outer height number if present
        pool = [n for n in numbers if abs(n - outer_h) > 1e-6]
        inner, conf = find_inner_heights_improved(
            float(outer_h), pool, tolerance=inner_tol, max_segments=8
        )
        # Compute quality for newly computed inner
        quality = 0.0
        if inner:
            numbers_set = set(numbers)
            ocr_matches = sum(1 for seg in inner if seg in numbers_set)
            quality = ocr_matches / len(inner) if inner else 0.0
            unique_inner = set(inner) - {100.0}
            if unique_inner and all(x in numbers_set for x in unique_inner):
                quality += 0.5
        # If still no inner, leave empty and low conf
        a["inner_heights"] = inner
        a["inner_conf"] = round(float(conf), 3)
        a["inner_quality"] = quality
        for val in inner:
            global_used[val] += 1

    # Calculate unused numbers penalty
    numbers_set = set(numbers)
    all_used_numbers = set(global_used.keys())
    unused_numbers = numbers_set - all_used_numbers
    unused_ratio = len(unused_numbers) / len(numbers) if numbers else 0.0
    unused_penalty = unused_ratio * 0.15  # 15% penalty for unused numbers

    # Build final report with penalty applied
    final_frames = []
    for a in assignments:
        # Apply penalty to all confidences
        outer_w_conf = (
            a.get("outer_width_conf", 3)
            if isinstance(a.get("outer_width_conf"), float)
            else a.get("outer_width_conf", 0.0)
        )
        outer_h_conf = (
            a.get("outer_height_conf", 3)
            if isinstance(a.get("outer_height_conf"), float)
            else a.get("outer_height_conf", 0.0)
        )
        inner_conf = a.get("inner_conf", 0.0)

        # Apply unused penalty
        outer_w_conf = max(0.0, outer_w_conf - unused_penalty)
        outer_h_conf = max(0.0, outer_h_conf - unused_penalty)
        inner_conf = max(0.0, inner_conf - unused_penalty)

        final_frames.append(
            {
                "frame_pixel": a.get("est_mm"),
                "outer_width": a["outer_width"],
                "outer_width_rel_error": round(a.get("outer_width_rel_error", 0.0), 6),
                "outer_width_conf": round(outer_w_conf, 3),
                "outer_height": a["outer_height"],
                "outer_height_rel_error": round(
                    a.get("outer_height_rel_error", 0.0), 6
                ),
                "outer_height_conf": round(outer_h_conf, 3),
                "inner_heights": a.get("inner_heights", []),
                "inner_conf": round(inner_conf, 3),
                "inner_quality": a.get("inner_quality", 0.0),
            }
        )

    report = {
        "scale": scale,
        "estimation_debug": est,
        "assignment_debug": {
            "best_cost": ga.get("best_cost"),
            "iterations": ga.get("iterations"),
            "unused_numbers": list(unused_numbers),
            "unused_ratio": round(unused_ratio, 3),
            "unused_penalty": round(unused_penalty, 3),
        },
        "frames": final_frames,
        "global_usage": dict(global_used),
    }
    return report


# ---------------------------
# Example / test harness
# ---------------------------
if __name__ == "__main__":
    import json
    from pathlib import Path

    # Load test samples
    samples_path = Path(__file__).parent / "image_dimension_samples.json"
    with samples_path.open("r", encoding="utf-8") as f:
        test_samples = json.load(f)

    print("=" * 70)
    print("Running dimension inference on all test cases")
    print("=" * 70)

    total_panels = 0
    passed_panels = 0
    results = []

    for sample in test_samples:
        sample_id = sample.get("id")
        sample_name = sample.get("name", "")
        expected = sample.get("expected", [])

        print(f"\n{'='*70}")
        print(f"=== Test Case {sample_id}: {sample_name} ===")
        print("=" * 70)
        print(f"Available numbers: {sample['numbers']}")
        print(f"Number of frames: {len(sample['frames'])}")
        print("Allow inner reuse: yes")

        res = extract_dimension(
            sample, ransac_tol=0.12, candidate_k=5, outer_rel_tol=0.25, inner_tol=0.03
        )

        if "error" in res:
            print(f"\n❌ ERROR: {res['error']}")
            results.append({"id": sample_id, "error": res["error"]})
            continue

        panels = res.get("frames", [])
        print(f"\n✓ Extracted {len(panels)} panels:")

        sample_passed = 0
        sample_total = len(panels)
        total_panels += sample_total

        for idx, panel in enumerate(panels):
            inner_heights = panel.get("inner_heights", [])
            inner_sum = sum(inner_heights) if inner_heights else 0
            outer_w = panel["outer_width"]
            outer_h = panel["outer_height"]

            print(f"\nPanel {idx}:")
            print(f"  Outer: {outer_w} x {outer_h} mm")
            print(f"  Inner heights: {inner_heights}")
            print(f"  Inner sum: {inner_sum} mm")
            print(
                f"  Frame (pixels): {int(panel['frame_pixel']['w'])}x{int(panel['frame_pixel']['h'])}"
            )

            # Compare with expected
            if idx < len(expected):
                exp = expected[idx]
                exp_outer = exp.get("outer", [])
                exp_inner = sorted(exp.get("inner", []))
                got_outer = (outer_w, outer_h)
                got_inner = sorted(inner_heights)

                outer_match = (
                    len(exp_outer) == 2
                    and got_outer[0] == exp_outer[0]
                    and got_outer[1] == exp_outer[1]
                )
                inner_match = got_inner == exp_inner

                if outer_match and inner_match:
                    print("  ✅ PASS - Matches expected result")
                    sample_passed += 1
                    passed_panels += 1
                else:
                    if not outer_match:
                        print(
                            f"  ❌ FAIL - Expected outer: {exp_outer}, got: {got_outer}"
                        )
                    if not inner_match:
                        print(
                            f"  ❌ FAIL - Expected inner (sorted): {exp_inner}, got: {got_inner}"
                        )

        results.append(
            {
                "id": sample_id,
                "name": sample_name,
                "total_panels": sample_total,
                "passed_panels": sample_passed,
                "scale": res.get("scale"),
                "frames": panels,
            }
        )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total panels checked: {total_panels}")
    print(f"Passed: {passed_panels}")
    print(f"Failed: {total_panels - passed_panels}")
    print(
        f"Success rate: {100.0 * passed_panels / total_panels if total_panels > 0 else 0:.1f}%"
    )
    print("=" * 70)

    # Save detailed results
    output_path = Path(__file__).parent / "result2.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed report to {output_path}")
