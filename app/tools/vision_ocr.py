import base64
import os
import json
import re
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Pattern, Tuple

import requests
from dotenv import load_dotenv

from app.models import BoundingRect, OCRTextBlock, OCRVertex
from app.tools.normalize_image import normalize_text
from app.tools.blockprocess import draw_rectangles

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None
    np = None

load_dotenv()  # Load GOOGLE_VISION_API_KEY từ .env

API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
VISION_ENDPOINT = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"


def ocr_text(
    image_content: str,
    output_dir: Optional[Path] = None,
    min_conf: Optional[float] = 0.5,
    whitelist: Optional[str] = r"[0-9]+",
    min_value: Optional[float] = 100.0,  # Filter numbers < min_value (for dimensions)
) -> List[OCRTextBlock]:
    payload = {
        "requests": [
            {
                "image": {"content": image_content},
                "features": [{"type": "TEXT_DETECTION"}],
                "imageContext": {
                    "languageHints": ["vi"],
                    "textDetectionParams": {"enableTextDetectionConfidenceScore": True},
                },
            }
        ]
    }

    try:
        response = requests.post(VISION_ENDPOINT, json=payload, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise ValueError(f"Failed to make request to Vision API: {exc}")

    response_json = response.json()
    vision_responses = response_json.get("responses", [])
    if not vision_responses:
        raise ValueError("No responses received from Vision API")
    annotations = vision_responses[0].get("textAnnotations", [])
    if not annotations:
        raise ValueError("No text annotations received from Vision API")
    full_text = annotations[0].get("description", "")

    # Build lookup for word-level confidences & symbols using fullTextAnnotation
    word_entries: Dict[
        Tuple[str, Tuple[Tuple[Optional[int], Optional[int]], ...]],
        List[Dict[str, Any]],
    ] = {}
    full_text_annotation = vision_responses[0].get("fullTextAnnotation", {})

    # save full text annotation to json
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
        full_text_annotation_path = output_dir / "full_text_annotation.json"
        with open(full_text_annotation_path, "w", encoding="utf-8") as f:
            json.dump(full_text_annotation, f, ensure_ascii=False, indent=4)
        print("Full text annotation saved to:", full_text_annotation_path)

    for page in full_text_annotation.get("pages", []):
        for block in page.get("blocks", []):
            for paragraph in block.get("paragraphs", []):
                for word in paragraph.get("words", []):
                    symbols_data = word.get("symbols", [])
                    word_text = "".join(
                        symbol.get("text", "") for symbol in symbols_data
                    )
                    vertices = word.get("boundingBox", {}).get("vertices", [])
                    key = (
                        word_text,
                        tuple((v.get("x"), v.get("y")) for v in vertices),
                    )

                    symbol_entries: List[Dict[str, Any]] = []
                    symbol_confidences: List[float] = []
                    for symbol in symbols_data:
                        symbol_vertices = symbol.get("boundingBox", {}).get(
                            "vertices", []
                        )
                        confidence_val = symbol.get("confidence")
                        if confidence_val is not None:
                            symbol_confidences.append(confidence_val)
                        symbol_entries.append(
                            {
                                "text": symbol.get("text", ""),
                                "confidence": confidence_val,
                                "vertices": symbol_vertices,
                            }
                        )

                    if symbol_confidences:
                        avg_confidence = sum(symbol_confidences) / len(
                            symbol_confidences
                        )
                    else:
                        avg_confidence = word.get("confidence")

                    entry = {
                        "confidence": avg_confidence,
                        "symbols": symbol_entries,
                    }
                    word_entries.setdefault(key, []).append(entry)

    whitelist_pattern: Optional[Pattern[str]] = (
        re.compile(whitelist) if whitelist else None
    )

    text_blocks: list[OCRTextBlock] = []
    for annotation in annotations[1:]:
        vertices = annotation.get("boundingPoly", {}).get("vertices", [])
        bounding_box = [OCRVertex(x=v.get("x"), y=v.get("y")) for v in vertices]
        desc = annotation.get("description", "")
        key = (
            desc.replace("\n", ""),
            tuple((v.get("x"), v.get("y")) for v in vertices),
        )
        entry_data = None
        if key in word_entries and word_entries[key]:
            entry_data = word_entries[key].pop(0)
        block_confidence = (entry_data or {}).get("confidence")
        symbols_list: List[Dict[str, Any]] = (entry_data or {}).get("symbols", [])

        # Filter symbols based on whitelist and confidence
        filtered_chars: List[str] = []
        filtered_confidences: List[float] = []
        for symbol in symbols_list:
            char = (symbol.get("text") or "").strip()
            if not char:
                continue
            if whitelist_pattern is not None and not whitelist_pattern.fullmatch(char):
                continue
            symbol_conf = symbol.get("confidence")
            if min_conf is not None and symbol_conf is not None:
                if symbol_conf < min_conf:
                    continue
            filtered_chars.append(char)
            if symbol_conf is not None:
                filtered_confidences.append(symbol_conf)

        # Rebuild text from filtered symbols
        rebuilt_text = "".join(filtered_chars)

        # Determine confidence for the block
        if filtered_confidences:
            block_confidence = sum(filtered_confidences) / len(filtered_confidences)
        elif min_conf is not None and block_confidence is not None:
            if block_confidence < min_conf:
                block_confidence = None

        # Skip block if no symbols remain after filtering
        if whitelist_pattern is not None and not rebuilt_text:
            continue
        if min_conf is not None and block_confidence is None and not filtered_chars:
            continue

        text_block = OCRTextBlock(
            text=rebuilt_text or desc,
            bounding_box=bounding_box,
            confidence=block_confidence,
        )
        if min_conf is not None and text_block.confidence is not None:
            if text_block.confidence < min_conf:
                continue
        # Apply whitelist filter first
        if whitelist_pattern is not None:
            normalized_text = text_block.text.strip()
            if not normalized_text or not whitelist_pattern.fullmatch(normalized_text):
                continue

        # Apply min_value filter (if specified)
        # At this point, text should be numeric (passed whitelist)
        if min_value is not None:
            try:
                num_value = float(text_block.text.strip())
                if num_value < min_value:
                    continue  # Skip numbers below threshold
            except (ValueError, TypeError):
                # Should not happen if whitelist=[0-9]+ is used
                # But skip just in case
                continue

        text_blocks.append(text_block)
    if output_dir is not None:
        image_array = np.frombuffer(base64.b64decode(image_content), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is not None:
            rectangles: List[BoundingRect] = []
            labels: List[str] = []
            for block in text_blocks:
                vertices = block.bounding_box
                if len(vertices) >= 2:
                    x_values = [v.x or 0 for v in vertices]
                    y_values = [v.y or 0 for v in vertices]
                    x = min(x_values)
                    y = min(y_values)
                    w = max(x_values) - x
                    h = max(y_values) - y
                    rectangles.append(BoundingRect(x=x, y=y, w=w, h=h))
                    confidence_str = (
                        f" (conf={block.confidence:.2f})"
                        if block.confidence is not None
                        else ""
                    )
                    labels.append(
                        (block.text.strip() or f"Text {len(labels)+1}") + confidence_str
                    )

            annotated = draw_rectangles(
                image,
                rectangles,
                color=(0, 0, 255),
                thickness=2,
                labels=labels,
            )
            output_path = output_dir / "output_vision_ocr.jpg"
            cv2.imwrite(str(output_path), annotated)
            print("Output saved to:", output_path)
            # save text blocks to json
            output_json_path = output_dir / "output_vision_ocr.json"
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    [tb.model_dump() for tb in text_blocks],
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
            print("Output saved to:", output_json_path)
    return text_blocks


# Ví dụ sử dụng
if __name__ == "__main__":
    # img_path = Path("outputs/24a94f1546374c16b54e1e411cc96010/Block 1/00_origin.png")
    # img_path = Path("outputs/c7895b9a60794796bcdb6568edda235b/Block 0/00_origin.png")
    # img_path = Path("outputs/22609b998d2c4ce28e6bceb77b92ffb2/Block 0/00_origin.png")
    # img_path = Path("outputs/22609b998d2c4ce28e6bceb77b92ffb2/normalized/00_origin.png")
    # img_path = Path(
    #     "outputs/test_pipeline_dims_classify/Block 0/normalized_ocr/00_origin.png"
    # )
    img_path = Path("outputs2/pipeline/54173ca3/Block 1/normalized_ocr/00_origin.png")

    output_dir = Path("outputs2", "vision_ocr")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(
            f"Không đọc được ảnh: {img_path}. Kiểm tra đường dẫn và quyền truy cập."
        )
    normalized_image = normalize_text(image, output_dir)
    success, buffer = cv2.imencode(".png", normalized_image)
    if not success:
        raise RuntimeError("Không thể encode ảnh đã normalize sang PNG")
    image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    result = ocr_text(image_base64, output_dir=output_dir)
    if result:
        # save output json
        output_json_path = output_dir / "output_vision_ocr.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(
                [tb.model_dump() for tb in result], f, ensure_ascii=False, indent=4
            )
        print("Output saved to:", output_json_path)
