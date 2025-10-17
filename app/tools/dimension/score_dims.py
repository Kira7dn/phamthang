"""
Dimension Scoring Module

Chấm điểm chất lượng kết quả phân loại dimensions (self-evaluation, không cần ground truth):
1. Consistency: Tính nhất quán giữa các frame (cùng pixel → cùng mm)
2. Frequency: Ưu tiên outer/inner xuất hiện nhiều trong OCR
3. Inner Quality: Tổng inner có khớp với outer_height không
4. Completeness: Đầy đủ outer + inner cho mỗi frame

Overall Score = weighted average của các sub-scores
"""

from collections import Counter
from typing import Dict, List, Optional
import logging

from app.models import Frame

logger = logging.getLogger(__name__)


def score_consistency(results: Dict[int, Dict], frames: List[Frame]) -> Dict[str, float]:
    """
    Chấm điểm tính nhất quán: các frame có cùng pixel dimension phải có cùng mm dimension.

    Logic:
    - Group frames theo normalized_w: các frame cùng nhóm phải có cùng outer_width
    - Group frames theo normalized_h: các frame cùng nhóm phải có cùng outer_height
    - Score per group = 1.0 nếu tất cả cùng giá trị, giảm dần theo số giá trị khác nhau
    - Trả về: {
        'width_consistency': avg score across width groups,
        'height_consistency': avg score across height groups,
        'overall_consistency': avg of width + height
      }

    Args:
        results: Dict[frame_idx, {'width': str, 'height': str, 'inner_heights': [str]}]
        frames: List[frame dicts with normalized_w, normalized_h]

    Returns:
        Dict với các điểm nhất quán
    """
    from collections import defaultdict

    if not results or not frames:
        return {
            "width_consistency": 1.0,
            "height_consistency": 1.0,
            "overall_consistency": 1.0,
        }

    # Group by pixel width
    width_groups = defaultdict(list)
    for i, f in enumerate(frames):
        norm_w = getattr(f, "normalized_w", f.w)
        if i in results:
            width_groups[norm_w].append(results[i].get("width"))

    # Group by pixel height
    height_groups = defaultdict(list)
    for i, f in enumerate(frames):
        norm_h = getattr(f, "normalized_h", f.h)
        if i in results:
            height_groups[norm_h].append(results[i].get("height"))

    # Score width groups
    width_scores = []
    for pixel_w, mm_widths in width_groups.items():
        unique_vals = len(set(mm_widths))
        # Score = 1.0 if all same, decrease by number of unique values
        score = max(0.0, 1.0 - (unique_vals - 1) * 0.3)
        width_scores.append(score)

    # Score height groups
    height_scores = []
    for pixel_h, mm_heights in height_groups.items():
        unique_vals = len(set(mm_heights))
        score = max(0.0, 1.0 - (unique_vals - 1) * 0.3)
        height_scores.append(score)

    return {
        "width_consistency": (
            sum(width_scores) / len(width_scores) if width_scores else 1.0
        ),
        "height_consistency": (
            sum(height_scores) / len(height_scores) if height_scores else 1.0
        ),
        "overall_consistency": (
            (sum(width_scores) + sum(height_scores))
            / (len(width_scores) + len(height_scores))
            if (width_scores or height_scores)
            else 1.0
        ),
    }


def score_frequency_alignment(
    results: Dict[int, Dict], numbers: List[float]
) -> Dict[str, float]:
    """
    Chấm điểm theo tần suất: ưu tiên outer/inner xuất hiện nhiều trong OCR.

    Logic:
    - Đếm tần suất mỗi số trong OCR numbers
    - Outer dimensions nên xuất hiện ít (LOW freq) hoặc vừa phải
    - Inner dimensions nên xuất hiện nhiều (HIGH freq) nếu có nhiều frame
    - Score dựa trên tỷ lệ outer/inner có frequency phù hợp
    - Trả về: {
        'outer_freq_score': điểm outer (cao nếu freq hợp lý),
        'inner_freq_score': điểm inner (cao nếu freq cao),
        'overall_freq_score': avg of outer + inner
      }

    Args:
        results: Dict[frame_idx, {'width': str, 'height': str, 'inner_heights': [str]}]
        numbers: List of all OCR numbers

    Returns:
        Dict với các điểm tần suất
    """
    if not results or not numbers:
        return {
            "outer_freq_score": 0.5,
            "inner_freq_score": 0.5,
            "overall_freq_score": 0.5,
        }

    freq = Counter(numbers)
    num_frames = len(results)

    # Collect all outers and inners
    outer_widths = [results[i].get("width") for i in results]
    outer_heights = [results[i].get("height") for i in results]
    all_inners = []
    for i in results:
        all_inners.extend(results[i].get("inner_heights", []))

    # Score outers: expect LOW to MEDIUM frequency
    # Good: freq <= num_frames (shared or unique per frame)
    # Bad: freq >> num_frames (too common, likely inner)
    outer_scores = []
    for val in set(outer_widths + outer_heights):
        if val:
            val_freq = freq.get(float(val), 0)
            # Ideal: 1 <= freq <= num_frames * 1.5
            if val_freq <= num_frames * 1.5:
                outer_scores.append(1.0)
            else:
                # Penalize high frequency
                penalty = min(1.0, (val_freq - num_frames * 1.5) / (num_frames * 2))
                outer_scores.append(max(0.0, 1.0 - penalty))

    # Score inners: expect HIGH frequency (nhiều frame dùng chung)
    # Good: freq >= num_frames * 0.5 (common across frames)
    # Bad: freq = 1 (unique, unlikely for inner)
    inner_scores = []
    for val in set(all_inners):
        if val:
            val_freq = freq.get(float(val), 0)
            # Ideal: freq >= num_frames * 0.5
            if val_freq >= num_frames * 0.5:
                inner_scores.append(1.0)
            elif val_freq >= 2:
                # Medium freq: partial score
                inner_scores.append(0.7)
            else:
                # freq = 1: low score
                inner_scores.append(0.3)

    return {
        "outer_freq_score": (
            sum(outer_scores) / len(outer_scores) if outer_scores else 0.5
        ),
        "inner_freq_score": (
            sum(inner_scores) / len(inner_scores) if inner_scores else 0.5
        ),
        "overall_freq_score": (
            (sum(outer_scores) + sum(inner_scores))
            / (len(outer_scores) + len(inner_scores))
            if (outer_scores or inner_scores)
            else 0.5
        ),
    }


def score_inner_sum_quality(
    results: Dict[int, Dict], tolerance: float = 0.03
) -> Dict[str, float]:
    """
    Chấm điểm chất lượng tổng inner: sum(inner_heights) có khớp với outer_height không.

    Logic:
    - Với mỗi frame: tính sum(inner_heights)
    - So sánh với outer_height
    - Relative error = |sum(inner) - outer_height| / outer_height
    - Score = 1.0 - min(rel_error, tolerance) / tolerance
    - Trả về: {
        'inner_sum_score': avg score across frames,
        'perfect_sum_matches': số frame có sum(inner) = outer_height
      }

    Args:
        results: Dict[frame_idx, {'width': str, 'height': str, 'inner_heights': [str]}]
        tolerance: ngưỡng relative error tối đa (mặc định 3%)

    Returns:
        Dict với điểm chất lượng tổng inner
    """
    if not results:
        return {"inner_sum_score": 0.0, "perfect_sum_matches": 0, "total_frames": 0}

    scores = []
    perfect_count = 0

    for i in results:
        dims = results[i]
        outer_h = float(dims.get("height", 0))
        inner_heights = [float(x) for x in dims.get("inner_heights", [])]

        if outer_h > 0 and inner_heights:
            sum_inner = sum(inner_heights)
            rel_err = abs(sum_inner - outer_h) / outer_h
            score = max(0.0, 1.0 - min(rel_err, tolerance) / tolerance)
            scores.append(score)

            if abs(sum_inner - outer_h) < 0.01:  # practically zero
                perfect_count += 1
        elif not inner_heights:
            # No inner heights: neutral score
            scores.append(0.5)

    return {
        "inner_sum_score": sum(scores) / len(scores) if scores else 0.0,
        "perfect_sum_matches": perfect_count,
        "total_frames": len(results),
    }


def score_completeness(
    results: Dict[int, Dict], frames: List[Frame]
) -> Dict[str, float]:
    """
    Chấm điểm tính đầy đủ: mỗi frame có đủ outer_width, outer_height, inner_heights không.

    Logic:
    - Mỗi frame cần có width > 0, height > 0
    - Inner_heights có thể rỗng (hợp lệ) hoặc có ít nhất 1 giá trị
    - Score per frame = 1.0 nếu đầy đủ, 0.5 nếu thiếu inner, 0.0 nếu thiếu outer
    - Trả về: {
        'completeness_score': avg score across frames,
        'complete_frames': số frame đầy đủ (outer + inner)
      }

    Args:
        results: Dict[frame_idx, {'width': str, 'height': str, 'inner_heights': [str]}]
        frames: List of frames (để đếm tổng số frame)

    Returns:
        Dict với điểm đầy đủ
    """
    if not results or not frames:
        return {
            "completeness_score": 0.0,
            "complete_frames": 0,
            "total_frames": len(frames) if frames else 0,
        }

    scores = []
    complete_count = 0

    for i in range(len(frames)):
        dims = results.get(i, {})
        w = float(dims.get("width", 0))
        h = float(dims.get("height", 0))
        inner = dims.get("inner_heights", [])

        if w > 0 and h > 0:
            if inner:
                scores.append(1.0)
                complete_count += 1
            else:
                # Outer OK, inner missing
                scores.append(0.7)
        else:
            # Outer missing
            scores.append(0.0)

    return {
        "completeness_score": sum(scores) / len(scores) if scores else 0.0,
        "complete_frames": complete_count,
        "total_frames": len(frames),
    }


def compute_overall_score(
    results: Dict[int, Dict],
    frames: List[Dict],
    numbers: List[float],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, any]:
    """
    Tính tổng điểm chất lượng dimension classification (self-evaluation).

    Logic:
    - Gọi tất cả các hàm score con (không cần ground truth)
    - Tổng hợp theo trọng số (có thể tùy chỉnh)
    - Overall score = weighted average
    - Trả về dict đầy đủ với tất cả sub-scores và overall score

    Trọng số mặc định:
    - consistency: 35% (quan trọng nhất)
    - frequency_alignment: 25%
    - inner_sum_quality: 25%
    - completeness: 15%

    Args:
        results: Dict[frame_idx, {'width': str, 'height': str, 'inner_heights': [str]}]
        frames: List of frames
        numbers: List of OCR numbers
        weights: Optional custom weights dict

    Returns:
        Dict đầy đủ với tất cả scores và thống kê
    """
    if weights is None:
        # Default weights (không có expected)
        weights = {
            "consistency": 0.35,
            "frequency": 0.25,
            "inner_sum": 0.25,
            "completeness": 0.15,
        }

    output = {}

    # 1. Consistency
    consistency = score_consistency(results, frames)
    output["consistency"] = consistency

    # 2. Frequency alignment
    frequency = score_frequency_alignment(results, numbers)
    output["frequency_alignment"] = frequency

    # 3. Inner sum quality
    inner_sum = score_inner_sum_quality(results)
    output["inner_sum_quality"] = inner_sum

    # 4. Completeness
    completeness = score_completeness(results, frames)
    output["completeness"] = completeness

    # 5. Compute overall score
    components = [
        (
            "consistency",
            output["consistency"]["overall_consistency"],
            weights.get("consistency", 0.0),
        ),
        (
            "frequency",
            output["frequency_alignment"]["overall_freq_score"],
            weights.get("frequency", 0.0),
        ),
        (
            "inner_sum",
            output["inner_sum_quality"]["inner_sum_score"],
            weights.get("inner_sum", 0.0),
        ),
        (
            "completeness",
            output["completeness"]["completeness_score"],
            weights.get("completeness", 0.0),
        ),
    ]

    total_weight = sum(w for _, _, w in components)
    if total_weight > 0:
        overall_score = (
            sum(score * weight for _, score, weight in components) / total_weight
        )
    else:
        overall_score = 0.0

    output["overall_score"] = overall_score
    output["score_breakdown"] = {
        name: (score, weight) for name, score, weight in components
    }
    output["weights_used"] = weights

    return output


def print_score_report(score_output: Dict, verbose: bool = True):
    """
    In báo cáo điểm số dễ đọc.

    Args:
        score_output: Output từ compute_overall_score()
        verbose: True = in chi tiết, False = chỉ in tổng quan
    """
    print("\n" + "=" * 60)
    print("DIMENSION CLASSIFICATION SCORE REPORT (Self-Evaluation)")
    print("=" * 60)

    # Overall score
    overall = score_output.get("overall_score", 0.0)
    print(f"\n📊 OVERALL SCORE: {overall:.2%} ({overall*100:.1f}/100)")

    if verbose:
        print("\n--- DETAILED BREAKDOWN ---")

        # Consistency
        cons = score_output.get("consistency", {})
        print(f"\n✓ Consistency: {cons.get('overall_consistency', 0):.2%}")
        print(f"  - Width consistency: {cons.get('width_consistency', 0):.2%}")
        print(f"  - Height consistency: {cons.get('height_consistency', 0):.2%}")

        # Frequency
        freq = score_output.get("frequency_alignment", {})
        print(f"\n✓ Frequency Alignment: {freq.get('overall_freq_score', 0):.2%}")
        print(f"  - Outer freq score: {freq.get('outer_freq_score', 0):.2%}")
        print(f"  - Inner freq score: {freq.get('inner_freq_score', 0):.2%}")

        # Inner sum quality
        inner_sum = score_output.get("inner_sum_quality", {})
        print(f"\n✓ Inner Sum Quality: {inner_sum.get('inner_sum_score', 0):.2%}")
        print(
            f"  - Perfect sum matches: {inner_sum.get('perfect_sum_matches', 0)}/{inner_sum.get('total_frames', 0)}"
        )

        # Completeness
        comp = score_output.get("completeness", {})
        print(f"\n✓ Completeness: {comp.get('completeness_score', 0):.2%}")
        print(
            f"  - Complete frames: {comp.get('complete_frames', 0)}/{comp.get('total_frames', 0)}"
        )

    print("\n" + "=" * 60 + "\n")
