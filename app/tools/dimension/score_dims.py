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
from collections import defaultdict

from app.models import Frame

logger = logging.getLogger(__name__)


def score_consistency(
    results: Dict[int, Dict], frames: List[Frame]
) -> Dict[str, float]:
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
    Chấm điểm theo độ lặp số trong OCR: mọi giá trị (outer hoặc inner) chỉ nên
    xuất hiện tối đa bằng số frame.

    Logic:
    - Chuẩn hóa các số từ OCR và đếm tần suất xuất hiện.
    - Với mỗi giá trị outer/inner, nếu số lần xuất hiện <= số frame → điểm 1.0.
    - Nếu số lần xuất hiện vượt quá số frame → điểm giảm theo tỷ lệ allowable / actual.
    - Kết quả trả về gồm điểm outer, inner và trung bình overall.

    Args:
        results: Dict[frame_idx, {'width': str, 'height': str, 'inner_heights': [str]}]
        numbers: List of all OCR numbers

    Returns:
        Dict với các điểm tần suất
    """
    if not results or not numbers:
        return {
            "outer_freq_score": 0,
            "inner_freq_score": 0,
            "overall_freq_score": 0,
        }

    def _normalize_number(value) -> Optional[float]:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if num != num:
            return None
        return round(num, 4)

    freq: Counter[float] = Counter()
    for num in numbers:
        normalized = _normalize_number(num)
        if normalized is not None:
            freq[normalized] += 1

    num_frames = len(results)

    # Collect all outers and inners
    outer_widths = [results[i].get("width") for i in results]
    outer_heights = [results[i].get("height") for i in results]
    all_inners = []
    for i in results:
        all_inners.extend(results[i].get("inner_heights", []))

    def _score_group(values: List) -> List[float]:
        # Chuẩn hóa và đếm tần suất trong kết quả
        normalized_values_list = [
            _normalize_number(val)
            for val in values
            if _normalize_number(val) is not None
        ]
        if not normalized_values_list:
            return []

        result_freq = Counter(normalized_values_list)
        group_scores: List[float] = []

        for val, count_in_result in result_freq.items():
            val_freq_in_ocr = freq.get(val, 0)
            if val_freq_in_ocr <= 0:
                group_scores.append(0.0)
                continue

            # Cho phép dùng tối đa val_freq_in_ocr * num_frames lần
            allowable = float(val_freq_in_ocr * num_frames)
            score = min(1.0, allowable / float(count_in_result))
            group_scores.append(score)

        return group_scores

    outer_scores = _score_group(outer_widths + outer_heights)
    inner_scores = _score_group(all_inners)

    return {
        "outer_freq_score": (
            sum(outer_scores) / len(outer_scores) if outer_scores else 0
        ),
        "inner_freq_score": (
            sum(inner_scores) / len(inner_scores) if inner_scores else 0
        ),
        "overall_freq_score": (
            (sum(outer_scores) + sum(inner_scores))
            / (len(outer_scores) + len(inner_scores))
            if (outer_scores or inner_scores)
            else 0
        ),
    }


def score_inner_sum_quality(results: Dict[int, Dict]) -> Dict[str, float]:
    """Đánh giá tổng chiều cao các inner so với outer height.

    - Điểm mỗi frame = tỷ lệ `sum(inner) / outer` nếu ≤ 1.
    - Nếu tổng inner vượt outer, phạt nặng: điểm = max(0, 1 - (ratio - 1) * 10).
    - Trả về trung bình điểm, tỷ lệ trung bình và số frame khớp hoàn hảo.
    """

    if not results:
        return {
            "inner_sum_score": 0.0,
            "inner_sum_ratio": 0.0,
            "perfect_sum_matches": 0,
            "total_frames": 0,
        }

    scores: List[float] = []
    ratios: List[float] = []
    perfect_matches = 0

    for dims in results.values():
        outer_h = float(dims.get("height", 0))
        inner_values = [float(x) for x in dims.get("inner_heights", [])]

        if outer_h <= 0:
            scores.append(0.0)
            ratios.append(0.0)
            continue

        if not inner_values:
            scores.append(0.0)
            ratios.append(0.0)
            continue

        inner_sum = sum(inner_values)
        ratio = inner_sum / outer_h
        ratios.append(ratio)

        if abs(inner_sum - outer_h) < 0.01:
            perfect_matches += 1

        if ratio <= 1.0:
            scores.append(ratio)
        else:
            penalty = max(0.0, 1.0 - (ratio - 1.0) * 10.0)
            scores.append(penalty)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0

    return {
        "inner_sum_score": avg_score,
        "inner_sum_ratio": avg_ratio,
        "perfect_sum_matches": perfect_matches,
        "total_frames": len(scores),
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


def calculate_quality_scores(
    result: Dict[int, Dict], frames: List[Frame], numbers_all: List[float]
) -> Dict:
    """Tính toán các chỉ số chất lượng."""

    def _normalize_number(value) -> Optional[float]:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if num != num:
            return None
        return round(num, 4)

    def _compute_unused_numbers() -> tuple[List[float], float]:
        normalized_numbers = [_normalize_number(n) for n in numbers_all]
        normalized_numbers = [n for n in normalized_numbers if n is not None]
        if not normalized_numbers:
            return [], 1.0

        usage_counter = Counter(normalized_numbers)

        def consume(raw_value):
            num = _normalize_number(raw_value)
            if num is None:
                return
            if usage_counter.get(num, 0) > 0:
                usage_counter[num] -= 1

        for dims in result.values():
            consume(dims.get("width"))
            consume(dims.get("height"))
            for inner_value in dims.get("inner_heights", []):
                consume(inner_value)

        unused_values: List[float] = []
        for value, count in usage_counter.items():
            if count > 0:
                unused_values.extend([value] * count)

        used_count = len(normalized_numbers) - len(unused_values)
        score = used_count / len(normalized_numbers)
        unused_values.sort()
        return unused_values, score

    consistency_score = score_consistency(result, frames)
    frequency_score = score_frequency_alignment(result, numbers_all)
    inner_sum_score = score_inner_sum_quality(result)
    completeness_score = score_completeness(result, frames)
    unused_numbers, unused_score = _compute_unused_numbers()

    # Trọng số cho từng thành phần
    weights = {
        "consistency": 0.30,
        "frequency": 0.20,
        "inner_sum": 0.20,
        "completeness": 0.15,
        "unused_numbers": 0.15,
    }

    components = [
        consistency_score["overall_consistency"],
        frequency_score["overall_freq_score"],
        inner_sum_score["inner_sum_score"],
        completeness_score["completeness_score"],
        unused_score,
    ]

    overall_score = sum(s * w for s, w in zip(components, weights.values())) / sum(
        weights.values()
    )

    return {
        "overall": round(overall_score, 4),
        "consistency": round(consistency_score["overall_consistency"], 4),
        "frequency": round(frequency_score["overall_freq_score"], 4),
        "inner_sum": round(inner_sum_score["inner_sum_score"], 4),
        "completeness": round(completeness_score["completeness_score"], 4),
        "unused_numbers_score": round(unused_score, 4),
        "unused_numbers": unused_numbers,
    }


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
