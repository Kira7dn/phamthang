"""
Faster scale estimation using vectorized operations.
- Uses subset of seeds (quantile sampling)
- Vectorized inlier check
- Least-squares refinement
"""

from typing import Any, Dict, List, Tuple
import logging
import json
from pathlib import Path
import numpy as np

from app.tools.dimension.normalize_frames import normalize_frames_fast

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def closest_number_np(
    vals: np.ndarray, numbers: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized nearest neighbor: return closest, abs diff, rel error."""
    if len(numbers) == 0 or len(vals) == 0:
        return (
            np.zeros_like(vals),
            np.full_like(vals, np.inf),
            np.full_like(vals, np.inf),
        )
    idx = np.abs(numbers[None, :] - vals[:, None]).argmin(axis=1)
    closest = numbers[idx]
    diff = np.abs(closest - vals)
    rel = np.divide(
        diff, np.abs(closest), out=np.full_like(diff, np.inf), where=closest != 0
    )
    return closest, diff, rel


def estimate_scale(
    pixel_candidates: List[float],
    numbers: List[float],
    inlier_rel_tol: float = 0.12,
    max_seed_pairs: int = 400,
) -> Dict[str, Any]:
    """
    Faster scale estimation using vectorized operations.
    - Uses subset of seeds (quantile sampling)
    - Vectorized inlier check
    - Least-squares refinement
    """
    out = {"scale": None}
    if not pixel_candidates or not numbers:
        return out

    px = np.unique(np.round(pixel_candidates).astype(float))
    nums = np.unique(np.array(numbers, dtype=float))
    px = px[px > 0]

    if len(px) == 0 or len(nums) == 0:
        return out

    # 🪄 Select a limited but representative subset of seeds (quantile sampling)
    px_sample = np.quantile(px, np.linspace(0, 1, min(len(px), 20)))
    num_sample = np.quantile(nums, np.linspace(0, 1, min(len(nums), 20)))

    seed_pairs = np.array(np.meshgrid(px_sample, num_sample)).T.reshape(-1, 2)
    if len(seed_pairs) > max_seed_pairs:
        seed_pairs = seed_pairs[
            np.linspace(0, len(seed_pairs) - 1, max_seed_pairs).astype(int)
        ]

    best = {
        "scale": None,
        "seed_pair": None,
        "inliers_count": -1,
        "avg_rel_error": np.inf,
        "matches": [],
    }

    for px_seed, num_seed in seed_pairs:
        if px_seed == 0:
            continue
        s = num_seed / px_seed
        scaled = px * s
        closest, diff, rel = closest_number_np(scaled, nums)
        inliers = rel <= inlier_rel_tol
        inlier_count = inliers.sum()
        avg_rel = np.mean(rel)

        better = inlier_count > best["inliers_count"] or (
            inlier_count == best["inliers_count"] and avg_rel < best["avg_rel_error"]
        )
        if better:
            best.update(
                {
                    "scale": s,
                    "seed_pair": (px_seed, num_seed),
                    "inliers_count": int(inlier_count),
                    "avg_rel_error": float(avg_rel),
                    "matches": list(zip(px, scaled, closest, diff, rel)),
                }
            )

    # ⚙️ refine using least-squares on inliers
    if best["matches"]:
        m = np.array(best["matches"])
        inliers = m[m[:, 4] <= inlier_rel_tol]
        if len(inliers) >= 2:
            refine_px, refine_mm = inliers[:, 0], inliers[:, 2]
            num = np.sum(refine_px * refine_mm)
            den = np.sum(refine_px**2)
            if den > 0:
                refined = num / den
                scaled = px * refined
                _, _, rel = closest_number_np(scaled, nums)
                best["refined_scale"] = refined
                best["refined_avg_rel_error"] = float(np.mean(rel))
                if best["refined_avg_rel_error"] <= best["avg_rel_error"]:
                    best["scale"] = refined
                    best["avg_rel_error"] = best["refined_avg_rel_error"]

    out.update(best)
    logger.info(
        f"estimate_scale_fast → scale={out.get('scale'):.6f}, "
        f"inliers={out.get('inliers_count')}, rel_err={out.get('avg_rel_error'):.4f}"
    )
    return out
