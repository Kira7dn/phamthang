import base64
import logging
import os
import json
import re
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Pattern, Tuple

import requests
from dotenv import load_dotenv
import cv2
import numpy as np

from app.models import BoundingRect, OCRTextBlock, OCRVertex
from app.tools.blockprocess import draw_rectangles
from app.tools.normalize_image import normalize_text


load_dotenv()  # Load GOOGLE_VISION_API_KEY từ .env

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
VISION_ENDPOINT = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"


def _normalize_ocr_text(text: str) -> str:
    """Normalize OCR text by replacing common misrecognitions."""
    return text.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")


def _save_visualization(
    image: np.ndarray,
    words: List[Dict[str, Any]],
    output_path: Path,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw bounding boxes on image and save."""
    rectangles: List[BoundingRect] = []
    labels: List[str] = []

    for word_data in words:
        vertices = word_data["bounding_box"]
        if len(vertices) >= 2:
            x_values = [v.get("x", 0) for v in vertices]
            y_values = [v.get("y", 0) for v in vertices]
            x = min(x_values)
            y = min(y_values)
            w = max(x_values) - x
            h = max(y_values) - y
            rectangles.append(BoundingRect(x=x, y=y, w=w, h=h))
            confidence_str = (
                f" (conf={word_data['confidence']:.2f})"
                if word_data["confidence"] is not None
                else ""
            )
            labels.append(word_data["text"] + confidence_str)

    annotated = draw_rectangles(
        image, rectangles, color=color, thickness=thickness, labels=labels
    )
    cv2.imwrite(str(output_path), annotated)
    logger.info(f"Saved visualization: {output_path}")


def _extract_tokens_from_fulltext(
    full_text_annotation: Dict[str, Any],
    whitelist_pattern: Optional[Pattern[str]],
    min_conf: Optional[float],
) -> List[Dict[str, Any]]:
    """Extract and filter tokens (numeric words) from fullTextAnnotation."""
    filtered_tokens: List[Dict[str, Any]] = []

    for page in full_text_annotation.get("pages", []):
        for block in page.get("blocks", []):
            for paragraph in block.get("paragraphs", []):
                for word in paragraph.get("words", []):
                    symbols_data = word.get("symbols", [])

                    # Group consecutive valid symbols into tokens
                    tokens: List[Dict[str, Any]] = []
                    current_token_symbols = []
                    current_token_chars = []

                    for symbol in symbols_data:
                        ch = (symbol.get("text", "") or "").strip()
                        if not ch:
                            continue

                        # Check if symbol passes filters
                        is_valid = True
                        if (
                            whitelist_pattern is not None
                            and not whitelist_pattern.fullmatch(ch)
                        ):
                            is_valid = False

                        conf_val = symbol.get("confidence")
                        if (
                            is_valid
                            and min_conf is not None
                            and conf_val is not None
                            and conf_val < min_conf
                        ):
                            is_valid = False

                        if is_valid:
                            # Add to current token
                            current_token_symbols.append(symbol)
                            current_token_chars.append(ch)
                        else:
                            # Invalid symbol breaks the token
                            if current_token_symbols:
                                tokens.append(
                                    {
                                        "symbols": current_token_symbols,
                                        "chars": current_token_chars,
                                    }
                                )
                                current_token_symbols = []
                                current_token_chars = []

                    # Don't forget the last token
                    if current_token_symbols:
                        tokens.append(
                            {
                                "symbols": current_token_symbols,
                                "chars": current_token_chars,
                            }
                        )

                    # Skip if no valid tokens
                    if not tokens:
                        continue

                    # Process each token separately
                    for token in tokens:
                        token_symbols = token["symbols"]
                        token_chars = token["chars"]
                        token_text = "".join(token_chars)

                        # Calculate bounding box from token symbols
                        all_vertices = []
                        for sym in token_symbols:
                            sym_vertices = sym.get("boundingBox", {}).get(
                                "vertices", []
                            )
                            all_vertices.extend(sym_vertices)

                        if not all_vertices:
                            continue

                        # Compute min/max to get bounding box
                        x_coords = [v.get("x", 0) for v in all_vertices]
                        y_coords = [v.get("y", 0) for v in all_vertices]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)

                        # Create bounding box vertices (top-left, top-right, bottom-right, bottom-left)
                        vertices = [
                            {"x": min_x, "y": min_y},
                            {"x": max_x, "y": min_y},
                            {"x": max_x, "y": max_y},
                            {"x": min_x, "y": max_y},
                        ]

                        # Calculate average confidence
                        symbol_confidences = [
                            sym.get("confidence")
                            for sym in token_symbols
                            if sym.get("confidence") is not None
                        ]
                        if symbol_confidences:
                            avg_confidence = sum(symbol_confidences) / len(
                                symbol_confidences
                            )
                        else:
                            avg_confidence = word.get("confidence")

                        # Skip if confidence too low
                        if (
                            min_conf is not None
                            and avg_confidence is not None
                            and avg_confidence < min_conf
                        ):
                            continue

                        # Store filtered token
                        filtered_tokens.append(
                            {
                                "text": token_text,
                                "confidence": avg_confidence,
                                "bounding_box": vertices,
                            }
                        )

    return filtered_tokens


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

    full_text_annotation = vision_responses[0].get("fullTextAnnotation", {})

    # Save full text annotation to json
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        full_text_annotation_path = output_dir / "full_text_annotation.json"
        with open(full_text_annotation_path, "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved full text annotation: {full_text_annotation_path}")

    # Compile whitelist pattern
    whitelist_pattern: Optional[Pattern[str]] = (
        re.compile(whitelist) if whitelist else None
    )

    # Extract tokens from fullTextAnnotation
    filtered_tokens = _extract_tokens_from_fulltext(
        full_text_annotation, whitelist_pattern, min_conf
    )

    # Decode image once for all visualizations
    image = None
    if output_dir is not None and filtered_tokens:
        image_array = np.frombuffer(base64.b64decode(image_content), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Convert filtered_tokens to OCRTextBlock list with min_value filter
    text_blocks: list[OCRTextBlock] = []
    for token_data in filtered_tokens:
        # Normalize text before min_value check
        normalized_text = _normalize_ocr_text(token_data["text"])

        # Apply min_value filter (if specified)
        if min_value is not None:
            try:
                num_value = float(normalized_text.strip())
                if num_value < min_value:
                    continue  # Skip numbers below threshold
            except (ValueError, TypeError):
                continue

        vertices = token_data["bounding_box"]
        bounding_box = [OCRVertex(x=v.get("x"), y=v.get("y")) for v in vertices]

        text_block = OCRTextBlock(
            text=normalized_text,
            bounding_box=bounding_box,
            confidence=token_data["confidence"],
        )
        text_blocks.append(text_block)

    # Save final text_blocks visualization
    if image is not None and text_blocks:
        # Convert text_blocks back to dict format for visualization
        text_blocks_dict = [
            {
                "text": tb.text,
                "confidence": tb.confidence,
                "bounding_box": [{"x": v.x, "y": v.y} for v in tb.bounding_box],
            }
            for tb in text_blocks
        ]
        output_path = output_dir / "output_vision_ocr.jpg"
        _save_visualization(image, text_blocks_dict, output_path, color=(0, 0, 255))

        # Save text blocks to json
        output_json_path = output_dir / "output_vision_ocr.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(
                [tb.model_dump() for tb in text_blocks],
                f,
                ensure_ascii=False,
                indent=4,
            )
        logger.info(f"Saved text blocks JSON: {output_json_path}")

    return text_blocks


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # img_path = Path("outputs/24a94f1546374c16b54e1e411cc96010/Block 1/00_origin.png")
    # img_path = Path("outputs/c7895b9a60794796bcdb6568edda235b/Block 0/00_origin.png")
    # img_path = Path("outputs/22609b998d2c4ce28e6bceb77b92ffb2/Block 0/00_origin.png")
    # img_path = Path("outputs/22609b998d2c4ce28e6bceb77b92ffb2/normalized/00_origin.png")
    # img_path = Path(
    #     "outputs/test_pipeline_dims_classify/Block 0/normalized_ocr/00_origin.png"
    # )
    # img_path = Path("outputs2/pipeline/54173ca3/Block 1/normalized_ocr/00_origin.png")
    img_path = Path(
        "outputs2/pipeline/bcfd4068/Block 0/normalized_frames/00_origin.png"
    )

    output_dir = Path("outputs2", "vision_ocr")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(
            f"Không đọc được ảnh: {img_path}. Kiểm tra đường dẫn và quyền truy cập."
        )
    image = normalize_text(image, output_dir)
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Không thể encode ảnh đã normalize sang PNG")
    image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    ocr_text(image_base64, output_dir=output_dir)
