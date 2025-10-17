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
import matplotlib.pyplot as plt

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


def plot_scale_diagnostics(
    pixel_candidates: List[float],
    numbers: List[float],
    result: Dict[str, Any],
    title: str,
):
    if not pixel_candidates or not numbers:
        logger.info("plot_scale_diagnostics skipped: missing data")
        return
    scale = result.get("scale")
    if not scale:
        logger.info("plot_scale_diagnostics skipped: no scale estimate")
        return
    matches = result.get("matches") or []
    px_vals = np.array(pixel_candidates, dtype=float)
    scaled_vals = px_vals * float(scale)
    closest_color = "#ff7f0e"
    normalized_color = "#1f77b4"
    plt.figure(figsize=(8, 5))
    if matches:
        matches_arr = np.array(matches, dtype=float)
        if len(matches_arr) == 0:
            return
        sort_idx = np.argsort(matches_arr[:, 0])
        matches_arr = matches_arr[sort_idx]
        plt.scatter(
            matches_arr[:, 0],
            matches_arr[:, 2],
            label="Closest numbers",
            s=35,
            color=closest_color,
        )
        plt.scatter(
            matches_arr[:, 0],
            matches_arr[:, 1],
            label="Normalized dims",
            marker="x",
            s=55,
            color=normalized_color,
        )
        x_vals = matches_arr[:, 0]
        norm_vals = matches_arr[:, 1]
        closest_vals = matches_arr[:, 2]
        x_min = 0.0
        x_max = max(float(np.max(px_vals)), 1.0)

        def extend_line(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if len(x) <= 1:
                return np.array([x_min, x_max]), np.repeat(y[0] if len(y) else 0.0, 2)
            left_dx = x[1] - x[0]
            right_dx = x[-1] - x[-2]
            left_slope = (y[1] - y[0]) / left_dx if abs(left_dx) > 1e-9 else 0.0
            right_slope = (y[-1] - y[-2]) / right_dx if abs(right_dx) > 1e-9 else 0.0
            y_left = y[0] + left_slope * (x_min - x[0])
            y_right = y[-1] + right_slope * (x_max - x[-1])
            x_extended = np.concatenate(([x_min], x, [x_max]))
            y_extended = np.concatenate(([y_left], y, [y_right]))
            return x_extended, y_extended

        norm_x, norm_y = extend_line(x_vals, norm_vals)
        plt.plot(norm_x, norm_y, color=normalized_color, linewidth=1, alpha=0.6)
        x_span = max(float(np.ptp(matches_arr[:, 0])), 1.0)
        y_span = max(float(np.ptp(matches_arr[:, 1])), 1.0)
        y_offset = y_span * 0.02
        x_offset = x_span * 0.02
        for px_val, scaled_val, closest_val in matches_arr[:, :3]:
            plt.text(
                px_val,
                closest_val - y_offset,
                f"{closest_val:.0f}",
                fontsize=8,
                color=closest_color,
                ha="center",
                va="top",
            )
            plt.text(
                px_val + x_offset,
                scaled_val,
                f"{scaled_val:.0f}",
                fontsize=8,
                color=normalized_color,
                ha="left",
                va="center",
            )
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title(title)
    plt.xlabel("Pixel length")
    plt.ylabel("Physical length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(exist_ok=True)
    safe_title = title.replace(" ", "_").replace("/", "-")
    file_path = output_dir / f"scale_diag_{safe_title}.png"
    plt.savefig(file_path, dpi=150)
    plt.close()
    logger.info("plot saved to %s", file_path)


def main():
    samples_path = Path(__file__).resolve().parent.parent / "dimension_samples.json"
    with samples_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    for sample in samples:
        frames = sample.get("frames", [])
        normalized_frames = normalize_frames_fast(frames)
        numbers = sample.get("numbers", [])
        pixel_candidates: List[float] = []
        for frame in normalized_frames:
            for key in ("normalized_w", "normalized_h"):
                val = frame.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    pixel_candidates.append(float(val))

        result = estimate_scale(pixel_candidates, numbers)
        logger.info(
            "Sample %s (%s): scale=%s, inliers=%s, rel_err=%s",
            sample.get("id"),
            sample.get("name"),
            f"{result.get('scale'):.6f}" if result.get("scale") else None,
            result.get("inliers_count"),
            (
                f"{result.get('avg_rel_error'):.4f}"
                if result.get("avg_rel_error") is not None
                else None
            ),
        )
        plot_scale_diagnostics(
            pixel_candidates,
            numbers,
            result,
            f"Sample {sample.get('id')} – {sample.get('name')}",
        )


if __name__ == "__main__":
    main()
