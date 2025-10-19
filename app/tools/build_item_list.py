from __future__ import annotations

import json
import logging
from pathlib import Path

from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple
import statistics

from pydantic import BaseModel

from app.models import Panel


logger = logging.getLogger("app.tools.build_item_list")


class BOMItem(BaseModel):
    type: str
    size: str
    unit: str
    quantity: float
    note: str = ""


class BuildItemOutput(BaseModel):
    material_list: List[BOMItem]


def infer_hinge_quantity(
    inner_heights: Sequence[float], height_mm: Optional[float]
) -> int:
    """Ước lượng số lượng bản lề dựa theo các đoạn khoảng cách."""
    NEAR_MATCH_THRESHOLD = 0.05
    OUTLIER_STD_MULTIPLIER = 1.5

    # --- Base theo chiều cao ---
    if height_mm and height_mm > 0:
        if height_mm < 1200:
            base_hinges = 2
        elif height_mm < 1800:
            base_hinges = 3
        else:
            base_hinges = 4
    else:
        base_hinges = 3

    segments = [float(v) for v in inner_heights if v and v > 0]
    if not segments:
        return base_hinges

    total_inner = sum(segments)
    avg = statistics.mean(segments)
    stdev = statistics.pstdev(segments) if len(segments) > 1 else 0

    diff_ratio = abs(total_inner - height_mm) / height_mm if height_mm else None

    # --- 1️⃣ Near match: khớp gần tuyệt đối ---
    if diff_ratio is not None and diff_ratio <= NEAR_MATCH_THRESHOLD:
        return max(len(segments) - 1, 2)

    # --- 2️⃣ Lọc outlier ---
    if len(segments) >= 3 and stdev > 0:
        filtered = [
            s for s in segments if abs(s - avg) <= OUTLIER_STD_MULTIPLIER * stdev
        ]
        if filtered:
            segments = filtered
            total_inner = sum(segments)

    ratio = total_inner / height_mm if height_mm else 1.0

    # --- 3️⃣ Nếu nghi ngờ mất OCR (thiếu đoạn nhỏ) ---
    if 0.5 < ratio < 0.85:
        # giữ nguyên số đoạn, không cộng thêm
        return max(len(segments), base_hinges)

    # --- 4️⃣ Fallback: tính gần đúng ---
    hinge_count = len(segments)
    if height_mm and abs(total_inner - height_mm) > avg:
        adjustment = round((height_mm - total_inner) / avg * 0.3)
        hinge_count = max(hinge_count + adjustment, base_hinges)

    return max(min(hinge_count, 6), 2)


def build_item_list(panels: List[Panel]) -> BuildItemOutput:
    """
    Build material list from an panels payload.

    Accepts either:
    - list payload with shape [{"outer_width": float, "outer_height": float, "inner_heights": [float]}]
    """
    logger.info("Building material list from panels payload:")
    for panel in panels:
        logger.info(panel)
    aggregates: Dict[Tuple[str, str, str], Dict[str, float | int]] = {}
    ordered_keys: List[Tuple[str, str, str]] = []

    def accumulate(
        key: Tuple[str, str, str],
        *,
        quantity_increment: float | int,
        count_increment: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = aggregates.get(key)
        if entry is None:
            aggregates[key] = {
                "type": key[0],
                "size": key[1],
                "unit": key[2],
                "quantity": float(quantity_increment),
                "count": count_increment,
            }
            if metadata:
                for meta_key, meta_value in metadata.items():
                    if meta_key in {"hinge_spacings", "hinge_counts"}:
                        aggregates[key][meta_key] = list(meta_value)
                    else:
                        aggregates[key][meta_key] = meta_value
            ordered_keys.append(key)
        else:
            entry["quantity"] += float(quantity_increment)
            entry["count"] += count_increment
            if metadata:
                for meta_key, meta_value in metadata.items():
                    if meta_key in {"hinge_spacings", "hinge_counts"}:
                        container = entry.setdefault(meta_key, [])
                        container.extend(meta_value)
                    else:
                        entry[meta_key] = meta_value

    hinge_total = 0
    for panel in panels:
        width = panel.outer_width
        height = panel.outer_height
        inner_heights = panel.inner_heights
        if not width or not height:
            continue

        area = (float(width) * float(height)) / 1_000_000
        hinge_count = infer_hinge_quantity(inner_heights, float(height))
        key = ("Khung nhôm", f"({width}mm x {height}mm)", "m²")
        accumulate(
            key,
            quantity_increment=round(area, 2),
            metadata={
                "width": width,
                "height": height,
                "hinge_spacings": [tuple(inner_heights)],
                "hinge_counts": [hinge_count],
            },
        )

        hinge_total += hinge_count

    if hinge_total:
        hinge_key = ("Bản lề", "Hinge", "cái")
        accumulate(
            hinge_key, quantity_increment=hinge_total, count_increment=hinge_total
        )

    material_list: List[BOMItem] = []
    for key in ordered_keys:
        entry = aggregates[key]
        size = entry["size"]
        note_parts: List[str] = []
        if entry.get("type") == "Khung nhôm":
            count = entry["count"]
            size = f"{size} * {int(count)}"
            spacings = entry.get("hinge_spacings", [])
            if spacings:
                unique_spacings: List[List[float]] = []
                seen: set[Tuple[float, ...]] = set()
                for spacing in spacings:
                    spacing_tuple = tuple(spacing)
                    if spacing_tuple in seen:
                        continue
                    seen.add(spacing_tuple)
                    unique_spacings.append(list(spacing_tuple))
                note_parts.append(f"Hinge spacing: {unique_spacings}")
            counts = entry.get("hinge_counts", [])
            if counts:
                unique_counts = sorted({int(count) for count in counts})
                note_parts.append(f"Hinges/frame: {unique_counts}")
        note = " | ".join(note_parts)
        quantity_value = round(float(entry["quantity"]), 2)
        material_list.append(
            BOMItem(
                type=str(entry["type"]),
                size=size,
                unit=str(entry["unit"]),
                quantity=quantity_value,
                note=note,
            )
        )

    return BuildItemOutput(material_list=material_list)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    sample_payload = [
        {
            "outer_width": 800,
            "outer_height": 1500,
            "inner_heights": [100.0, 400.0, 500, 400.0, 100],
        },
        {
            "outer_width": 800,
            "outer_height": 1500,
            "inner_heights": [100.0, 400.0, 400.0, 100],
        },
        {
            "outer_width": 800,
            "outer_height": 1500,
            "inner_heights": [100.0, 400.0, 500, 400.0],
        },
        {
            "outer_width": 800,
            "outer_height": 1500,
            "inner_heights": [100.0, 400.0, 1500, 400.0, 100],
        },
        {
            "outer_width": 0,
            "outer_height": 0,
            "inner_heights": [100.0, 400.0, 1500, 400.0, 100],
        },
        {
            "outer_width": 900,
            "outer_height": 2000,
            "inner_heights": [300.0, 400.0, 600, 400.0, 300],
        },
        {
            "outer_width": 900,
            "outer_height": 2000,
            "inner_heights": [300.0, 400.0, 400.0, 300],
        },
        {
            "outer_width": 900,
            "outer_height": 2000,
            "inner_heights": [300.0, 400.0, 600, 400.0],
        },
        {
            "outer_width": 900,
            "outer_height": 2000,
            "inner_heights": [300.0, 400.0, 1600, 400.0, 300],
        },
        {
            "outer_width": 0,
            "outer_height": 0,
            "inner_heights": [300.0, 400.0, 1600, 400.0, 300],
        },
    ]

    # Run with plain dict payload (no external model dependency)
    result = build_item_list(sample_payload)
    result_payload = result.model_dump()
    output_dir = Path("outputs/item_builder")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "build_item_list.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("output: ", output_dir / "build_item_list.json")


if __name__ == "__main__":
    main()
