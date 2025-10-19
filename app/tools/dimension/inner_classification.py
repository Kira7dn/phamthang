"""
Các hàm hỗ trợ chọn tổ hợp inner height từ danh sách OCR.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pydantic import BaseModel

from app.models import OCRTextBlock


logger = logging.getLogger(__name__)


def _parse_float(text: str) -> float:
    """Chuyển text thành float, trả về 0.0 nếu không hợp lệ."""
    try:
        return float(str(text).strip())
    except (ValueError, TypeError):
        return 0.0


def _entry_center(block: OCRTextBlock) -> Tuple[float, float]:
    """Tính tâm (x, y) của bounding_box."""
    if not block.bounding_box:
        return 0.0, 0.0
    xs = [v.x or 0.0 for v in block.bounding_box]
    ys = [v.y or 0.0 for v in block.bounding_box]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def _entry_center_x(block: OCRTextBlock) -> float:
    """Lấy tọa độ X trung tâm của bounding_box."""
    return _entry_center(block)[0]


def _estimate_column_count(entries: List[OCRTextBlock], tolerance: float = 10.0) -> int:
    """Ước tính số cột dựa trên tọa độ X của entries."""
    centers = sorted(_entry_center_x(entry) for entry in entries)
    if not centers:
        return 0
    count = 0
    last_center = None
    for cx in centers:
        if last_center is None or abs(cx - last_center) > tolerance:
            count += 1
            last_center = cx
    return count


def _column_factor_from_count(column_count: int) -> float:
    """Tính hệ số điểm dựa trên số cột (1 cột = 1.0, giảm dần)."""
    if column_count <= 0:
        return 0.0
    return max(0.0, 1.0 - (column_count - 1) / 3.0)


def _evaluate_scores(
    total: float,
    target_height: float,
    item_count: int,
    min_items: int,
    balance_divisor: float,
    column_count: int,
) -> Tuple[float, float, float, float]:
    """Đánh giá điểm tổng hợp cho một ứng viên.

    Returns:
        Tuple[score_total, score_balance, score_column, score_overall]
    """
    ratio = total / max(target_height, 1e-6)
    score_total = max(0.0, 1.0 - abs(1.0 - ratio))

    target_count = (
        max(float(min_items), target_height / balance_divisor)
        if balance_divisor > 0
        else float(min_items)
    )
    score_balance = max(0.0, 1.0 - abs(item_count - target_count) / target_count)
    score_column = _column_factor_from_count(column_count)
    score_overall = (
        7.0 * score_total + 0.0 * score_balance + 3.0 * score_column
    ) / 10.0

    return score_total, score_balance, score_column, score_overall


class FrameCandidate(BaseModel):
    """Candidate tổ hợp inner heights."""

    outer_height: int
    inner_heights: List[int]
    total: int
    item_count: int


@dataclass
class CandidateEvaluation:
    """Kết quả đánh giá một ứng viên với điểm chi tiết."""

    score: float
    entries: List[OCRTextBlock]
    score_total: float = 0.0
    score_balance: float = 0.0
    score_column: float = 0.0
    candidate: Optional[FrameCandidate] = None
    origin: str = ""


def _group_by_column(
    entries: List[OCRTextBlock], tolerance_x: float = 10.0
) -> List[List[OCRTextBlock]]:
    """Nhóm entries theo cột dựa trên tọa độ X, sắp xếp giảm dần theo số phần tử."""
    groups: List[List[OCRTextBlock]] = []
    for e in sorted(
        entries, key=lambda e: e.bounding_box[0].x if e.bounding_box else 0
    ):
        bx = e.bounding_box[0].x if e.bounding_box else 0
        found = False
        for g in groups:
            gx = g[0].bounding_box[0].x if g[0].bounding_box else 0
            if abs(bx - gx) <= tolerance_x:
                g.append(e)
                found = True
                break
        if not found:
            groups.append([e])
    groups.sort(key=len, reverse=True)
    return groups


def _sort_entries_by_y(entries: List[OCRTextBlock]) -> List[OCRTextBlock]:
    """Sắp xếp entries theo tọa độ Y."""
    return sorted(entries, key=lambda e: e.bounding_box[0].y if e.bounding_box else 0.0)


def find_inner_height_candidates(
    entries: List[OCRTextBlock],
    outer_height: int,
    min_items: int = 3,
    max_items: int = 10,
) -> List[FrameCandidate]:
    """Tìm các nhóm inner có tổng gần outer_height.

    Loại bỏ trùng lặp dựa trên giá trị tổ hợp thực tế.
    """
    heights = sorted(
        [
            _parse_float(e.text)
            for e in entries
            if _parse_float(e.text) <= outer_height + max(0.005 * outer_height, 1)
        ]
    )
    logger.debug("find_inner_height_candidates › input heights=%s", heights)
    if not heights:
        return []

    candidates: List[FrameCandidate] = []
    seen_combos = set()
    n = len(heights)

    for r in range(min_items, min(max_items, n) + 1):
        for combo in itertools.combinations(heights, r):
            total = sum(combo)
            candidate = FrameCandidate(
                outer_height=outer_height,
                inner_heights=list(combo),
                total=total,
                item_count=r,
            )
            candidates.append(candidate)

    logger.debug(
        "find_inner_height_candidates › generated %d candidates (outer=%d)",
        len(candidates),
        outer_height,
    )
    return candidates


def _map_values_to_entries(
    target_values: List[int],
    entry_pairs: List[Tuple[OCRTextBlock, float]],
    value_tolerance: float = 0.5,
) -> List[OCRTextBlock]:
    """Ánh xạ danh sách giá trị mục tiêu sang OCRTextBlock tương ứng."""
    remaining = list(entry_pairs)
    selected: List[OCRTextBlock] = []

    for value in target_values:
        match_index = next(
            (
                idx
                for idx, (entry, entry_value) in enumerate(remaining)
                if abs(entry_value - value) <= value_tolerance
            ),
            None,
        )
        if match_index is None:
            continue
        selected_entry, _ = remaining.pop(match_index)
        selected.append(selected_entry)

    return selected


def deduplicate_entries(
    entries: List[OCRTextBlock],
    target_height: float,
    y_tolerance: float = 5.0,
    column_tolerance: float = 10.0,
) -> List[OCRTextBlock]:
    """Loại bỏ entries trùng lặp theo cột.

    - Loại bỏ cột đơn có tổng >= target_height (cột outer)
    - Chọn cột chính: ưu tiên nhiều phần tử, gần target_height
    - Với cột phụ: bỏ entry trùng text + cùng Y với cột chính
    """
    if not entries:
        return []

    def col_total(col: List[OCRTextBlock]) -> float:
        return sum(_parse_float(e.text) for e in col)

    col_groups = [
        col
        for col in _group_by_column(entries, column_tolerance)
        if not (len(col) == 1 and col_total(col) >= target_height)
    ]
    if not col_groups:
        return []

    primary_col = max(
        col_groups,
        key=lambda col: (len(col), -abs(col_total(col) - target_height)),
    )

    deduped: List[OCRTextBlock] = list(primary_col)

    for col in col_groups:
        if col is primary_col:
            continue
        for entry in col:
            _, ey = _entry_center(entry)
            duplicate = any(
                entry.text == p.text and abs(_entry_center(p)[1] - ey) <= y_tolerance
                for p in primary_col
            )
            if not duplicate:
                deduped.append(entry)

    deduped.sort(key=lambda e: _entry_center(e)[1])
    return deduped


def inner_height_select(
    inner_entries: List[OCRTextBlock],
    outer_height: OCRTextBlock,
    tolerance: float = 0.05,
    min_items: int = 2,
    max_items: int = 10,
    balance_divisor: float = 400.0,
    column_tolerance: float = 10.0,
) -> List[OCRTextBlock]:
    """Chọn tổ hợp inner_height tốt nhất.

    Quy trình:
    1. Nhóm entries theo cột dựa trên tọa độ X
    2. Chấm điểm từng cột đơn lẻ
    3. Chấm điểm các tổ hợp cột
    4. Trả về ứng viên có điểm cao nhất
    """
    if not inner_entries:
        return []

    outer_height_text = (
        outer_height.text if hasattr(outer_height, "text") else outer_height
    )
    target_height = _parse_float(outer_height_text)
    if target_height <= 0:
        return inner_entries

    col_groups = _group_by_column(inner_entries, column_tolerance)
    logger.debug("inner_height_select › target_height: %.1f", target_height)
    logger.debug(
        "inner_height_select › grouped %d columns: %s",
        len(col_groups),
        [[entry.text for entry in col] for col in col_groups],
    )

    evaluations: List[CandidateEvaluation] = []

    for i, col in enumerate(col_groups):
        pairs_col = [
            (entry, _parse_float(entry.text))
            for entry in col
            if _parse_float(entry.text) > 0
        ]
        total_col = sum(v for _, v in pairs_col)
        diff_ratio = abs(total_col - target_height) / target_height
        logger.debug(
            "inner_height_select › inspect col[%d]: entries=%s total=%.1f diff=%.3f",
            i,
            [entry.text for entry in col],
            total_col,
            diff_ratio,
        )

        # Nếu tổng trong cột này gần đúng target_height ±1% thì trả luôn
        if diff_ratio <= 0.01:
            return _sort_entries_by_y(col)

        if not pairs_col:
            continue

        column_count = _estimate_column_count(col)
        (
            score_total,
            score_balance,
            score_column,
            score_overall,
        ) = _evaluate_scores(
            total_col,
            target_height,
            len(col),
            min_items,
            balance_divisor,
            column_count,
        )
        evaluations.append(
            CandidateEvaluation(
                score=round(score_overall, 3),
                entries=col,
                score_total=round(score_total, 3),
                score_balance=round(score_balance, 3),
                score_column=round(score_column, 3),
                origin=f"col[{i}]",
            )
        )
        candidate_idx = len(evaluations) - 1
        logger.debug(
            "inner_height_select › candidate #%d (%s): total=%.1f diff=%.3f | scores(total=%.3f, balance=%.3f, column=%.3f, overall=%.3f) entries=%s",
            candidate_idx,
            evaluations[-1].origin,
            total_col,
            diff_ratio,
            evaluations[-1].score_total,
            evaluations[-1].score_balance,
            evaluations[-1].score_column,
            evaluations[-1].score,
            [entry.text for entry in evaluations[-1].entries],
        )

        # ---- 3. Nếu chưa đạt, thử kết hợp với các cột khác
        remaining = [(idx, group) for idx, group in enumerate(col_groups) if idx != i]

        if total_col > target_height:
            logger.debug(
                "inner_height_select › skip col[%d]: total=%.1f exceeds target=%.1f",
                i,
                total_col,
                target_height,
            )
            continue  # bỏ hẳn, thử sang cột tiếp theo

        for j_idx, other_col in remaining:
            other_pairs = [
                (e, _parse_float(e.text)) for e in other_col if _parse_float(e.text) > 0
            ]
            partial_target = target_height - total_col

            logger.debug(
                "inner_height_select › combine col[%d]+col[%d]: remaining target=%.1f",
                i,
                j_idx,
                partial_target,
            )
            candidates = find_inner_height_candidates(
                entries=[e for e, _ in other_pairs],
                outer_height=int(round(partial_target)),
                min_items=1,
                max_items=min(max_items, len(other_col)),
            )
            if not candidates:
                logger.debug(
                    "inner_height_select › combine col[%d]+col[%d]: NO CANDIDATES, continue next col",
                    i,
                    j_idx,
                )
                continue

            for candidate in candidates:
                mapped_entries = _map_values_to_entries(
                    candidate.inner_heights, other_pairs
                )
                if not mapped_entries:
                    logger.debug(
                        "inner_height_select › combine col[%d]+col[%d]: candidate %s has no mapping",
                        i,
                        j_idx,
                        candidate.inner_heights,
                    )
                    continue
                column_eval_entries = col + mapped_entries
                total_combined = total_col + candidate.total
                if total_combined > target_height * (1 + tolerance):
                    continue  # bỏ candidate

                diff_combined = abs(total_combined - target_height) / max(
                    target_height, 1e-6
                )
                if diff_combined <= 0.01:
                    logger.debug(
                        "inner_height_select › early success with col[%d]+col[%d]: total=%.1f diff=%.3f",
                        i,
                        j_idx,
                        total_combined,
                        diff_combined,
                    )
                    return _sort_entries_by_y(column_eval_entries)

                column_count = _estimate_column_count(column_eval_entries)
                (
                    score_total,
                    score_balance,
                    score_column,
                    score_overall,
                ) = _evaluate_scores(
                    total_combined,
                    target_height,
                    len(column_eval_entries),
                    min_items,
                    balance_divisor,
                    column_count,
                )
                evaluations.append(
                    CandidateEvaluation(
                        score=round(score_overall, 3),
                        entries=column_eval_entries,
                        score_total=round(score_total, 3),
                        score_balance=round(score_balance, 3),
                        score_column=round(score_column, 3),
                        candidate=candidate,
                        origin=f"col[{i}]+col[{j_idx}]",
                    )
                )
                candidate_idx = len(evaluations) - 1
                logger.debug(
                    "inner_height_select › candidate #%d (%s): total=%.1f diff=%.3f | scores(total=%.3f, balance=%.3f, column=%.3f, overall=%.3f) entries=%s",
                    candidate_idx,
                    evaluations[-1].origin,
                    total_combined,
                    diff_combined,
                    evaluations[-1].score_total,
                    evaluations[-1].score_balance,
                    evaluations[-1].score_column,
                    evaluations[-1].score,
                    [entry.text for entry in evaluations[-1].entries],
                )

    if evaluations:
        best_idx = max(range(len(evaluations)), key=lambda idx: evaluations[idx].score)
        best_eval = evaluations[best_idx]
        logger.debug(
            "inner_height_select › select candidate #%d (%s): overall=%.3f entries=%s",
            best_idx,
            best_eval.origin or "unknown",
            best_eval.score,
            [entry.text for entry in best_eval.entries],
        )
        return _sort_entries_by_y(best_eval.entries)

    return inner_entries
