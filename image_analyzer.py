from __future__ import annotations

import shutil
import sys
import os
import json
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from dotenv import load_dotenv

from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent

from panel_verifier import Block, Panel, AggregatedResult

from utils.cluster_image import extract_block_images
from utils.normalize_image import normalize_image


class PanelsLLMResult(BaseModel):
    panels: List[Panel]


def load_image_bytes(
    image_input: Union[np.ndarray, Path], output_dir: Optional[Path] = None
) -> BinaryContent:
    # Kiểm tra image_input là Path hay numpy array
    if isinstance(image_input, Path):
        image_input = cv2.imread(str(image_input))
    processed = normalize_image(image_input, output_dir)
    success, encoded = cv2.imencode(".png", processed)
    if not success:
        raise ValueError("Không thể mã hóa ảnh đã tiền xử lý")
    return BinaryContent(data=encoded.tobytes(), media_type="image/png")


def create_prompt() -> str:
    return """
        You are an **Aluminum Frame Panel Analyst**.

        ### Task: Given a technical drawing of an aluminum-frame assembly, analyze it and return **only JSON**, strictly following this schema:
        {
          "panels": [
            {
              "panel_index": <int>, // Unique index, starting from 1
              "outer_width": <number>, // Outer width in millimetres
              "outer_height": <number>, // Outer height in millimetres
              "inner_heights": [<number>, ...] // List of inner measurements inside height_mm
            }
          ]
        }

        ### Detection rules
        1. A **panel** is any fully enclosed rectangle with a bold/non-dashed outline, treat every panel separately even if they share edges.
        2. Order panels from left→right, then top→bottom.

        ### Extraction rules
        - `outer_width`, `outer_height`: 
            + The outermost dimension labels exactly as written.
            + If a panel has no width or height, refer from the same panel or return `0`.
        - `inner_heights`: 
            + Must include **all values exactly as in the drawing** Preserve original exactly from top -> bottom (e.g. [100,450,450,100]).  
        ### Constraints
        - Do **not** merge or omit any panels.
        - Keep numbers exactly as they appear (no unit conversion, no extra text).
        - Respond with **valid JSON only** (no Markdown, no explanation).
    """


class ImageAnalyzeAgent:
    """Agent phụ trách tách block, gọi LLM phân tích panels và (tùy chọn) lưu kết quả."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        detect_model_id: str = "google-gla:gemini-2.0-flash",
    ) -> None:
        self.output_dir = output_dir
        self.detect_model_id = detect_model_id

        # Khởi tạo agent và prompt một lần cho toàn pipeline
        self.agent = Agent(self.detect_model_id, output_type=PanelsLLMResult)
        self.prompt = create_prompt()

    def run(self, source_image: Union[np.ndarray, Path]) -> AggregatedResult:
        if not os.getenv("GOOGLE_API_KEY"):
            message = "Thiếu biến môi trường GOOGLE_API_KEY cho Gemini API"
            print(message, file=sys.stderr)
            raise RuntimeError(message)

        # Đảm bảo thư mục output tồn tại nếu người dùng cung cấp
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(source_image, Path):
            source_image = cv2.imread(str(source_image))

        # 1) Tách block từ ảnh nguồn
        block_images = extract_block_images(source_image, output_dir=self.output_dir)

        # 2) Gọi agent phân tích cho từng block và gom kết quả
        aggregated = AggregatedResult(blocks=[])
        for idx, image in enumerate(block_images):
            block_dir = (
                Path(self.output_dir, f"Block {str(idx)}") if self.output_dir else None
            )
            processed_image = load_image_bytes(image, block_dir)
            agent_result = self.agent.run_sync([self.prompt, processed_image])
            if isinstance(agent_result.output, PanelsLLMResult):
                panels_result = agent_result.output
            else:
                panels_result = PanelsLLMResult.model_validate(agent_result.output)
            aggregated.blocks.append(
                Block(
                    block_no=str(idx),
                    panels=panels_result.panels,
                )
            )

        # 3) Lưu JSON kết quả nếu có output_dir
        if self.output_dir:
            result_path = self.output_dir / "image_analyze.json"
            result_path.write_text(
                json.dumps(
                    aggregated.model_dump(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Đã lưu kết quả panels ban đầu: {result_path}")
        return aggregated


def main():
    load_dotenv()
    image = Path("assets/z7064219010311_67ae7d4dca697d1842b79755dd0c1b4c.jpg")
    output_dir = Path("outputs/panel_agent/analyzer")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    agent = ImageAnalyzeAgent(output_dir)
    return agent.run(image)


if __name__ == "__main__":
    main()
