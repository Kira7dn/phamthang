import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from dotenv import load_dotenv

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, BinaryContent

logger = logging.getLogger("app.agent.llm_ocr")


class NumbersLLMResult(BaseModel):
    numbers: List[float]


class ImageNumbers(BaseModel):
    id: str
    numbers: List[float]


class OCRAggregatedResult(BaseModel):
    images: List[ImageNumbers]


class ImageSet(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    image: Union[np.ndarray, Path]
    frames: Optional[List[Dict[str, Any]]] = Field(default=None)


NUMBERS_SYSTEM_PROMPT = """
You extract all numeric labels from technical drawings.
- Preserve reading order: left→right, then top→bottom.
- Do not sort, deduplicate, or transform values.
- Output must be valid JSON only.
"""


def create_number_prompt() -> str:
    return """
    Output strict JSON only:
    {
      "numbers": [number, ...]
    }
    """


def load_image_bytes(
    image_input: Union[np.ndarray, Path],
) -> List[BinaryContent]:
    # Kiểm tra image_input là Path hay numpy array
    if isinstance(image_input, Path):
        image_input = cv2.imread(str(image_input))
    if image_input is None or (hasattr(image_input, "size") and image_input.size == 0):
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ")
    success, encoded = cv2.imencode(".png", image_input)
    if not success:
        raise ValueError("Không thể mã hóa ảnh đã tiền xử lý")
    return [BinaryContent(data=encoded.tobytes(), media_type="image/png")]


class OCRAgent:
    """Agent phụ trách tách block, gọi LLM phân tích panels và (tùy chọn) lưu kết quả."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        # detect_model_id: str = "google-gla:gemini-2.5-flash-lite",
        # detect_model_id: str = "google-gla:gemini-2.0-flash",
        detect_model_id: str = "openai:gpt-4o",
    ) -> None:
        self.output_dir = output_dir
        self.detect_model_id = detect_model_id
        self.agent_number = Agent(
            detect_model_id,
            output_type=NumbersLLMResult,
            system_prompt=NUMBERS_SYSTEM_PROMPT,
        )

    def run(self, image_set: List[ImageSet]) -> OCRAggregatedResult:
        if self.detect_model_id.startswith("openai:"):
            if not os.getenv("OPENAI_API_KEY"):
                message = "Thiếu biến môi trường OPENAI_API_KEY cho OpenAI API"
                logger.info(f"{message}", file=sys.stderr)
                raise RuntimeError(message)
            logger.info("OPENAI_API_KEY detected, proceeding with analysis")
        elif self.detect_model_id.startswith("google-gla:"):
            if not os.getenv("GOOGLE_API_KEY"):
                message = "Thiếu biến môi trường GOOGLE_API_KEY cho Gemini API"
                logger.info(f"{message}", file=sys.stderr)
                raise RuntimeError(message)
            logger.info("GOOGLE_API_KEY detected, proceeding with analysis")
        else:
            logger.warning(
                "Unknown model provider for %s; make sure proper API key is set",
                self.detect_model_id,
            )
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Output directory prepared", extra={"output_dir": str(self.output_dir)}
            )
        # 2) Gọi agent phân tích cho từng block và gom kết quả
        aggregated = OCRAggregatedResult(images=[])
        for image in image_set:
            shared_context = load_image_bytes(image.image)
            logger.info(f"Running OCR for block {image.id}")
            result = self.agent_number.run_sync(
                [create_number_prompt(), *shared_context]
            )
            if hasattr(result, "output") and isinstance(
                result.output, NumbersLLMResult
            ):
                numbers = result.output.numbers
            else:
                payload = getattr(result, "output", result)
                numbers = NumbersLLMResult.model_validate(payload).numbers
            logger.info(f"Numbers extracted for block {image.id}")

            aggregated.images.append(
                ImageNumbers(
                    id=image.id,
                    numbers=numbers,
                )
            )

        # 3) Lưu JSON kết quả nếu có output_dir
        if self.output_dir:
            result_path = self.output_dir / "ocr_numbers.json"
            result_path.write_text(
                json.dumps(
                    aggregated.model_dump(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Đã lưu kết quả panels ban đầu: {result_path}")
            logger.info(
                "Saved initial panel analysis", extra={"result_path": str(result_path)}
            )
        return aggregated


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Demo input
    images = [
        ImageSet(
            id="0",
            image=Path("outputs/panel_analyze_pipeline/Block 2/00_origin.png"),
        ),
        ImageSet(
            id="1",
            image=Path("outputs/panel_analyze_pipeline/Block 2/00_origin.png"),
        ),
    ]
    output_dir = Path("outputs/panel_agent/analyzer")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    agent = OCRAgent(output_dir=output_dir)
    return agent.run(images)


if __name__ == "__main__":
    main()
