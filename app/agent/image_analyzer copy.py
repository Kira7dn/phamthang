import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import List, Optional, Union

import cv2
import numpy as np
from dotenv import load_dotenv

from pydantic_ai import Agent, BinaryContent

from utils.cluster_image import extract_block_images
from utils.normalize_image import normalize_image

# Note: Do not import Pydantic models here to avoid validation. We will aggregate plain dicts.


logger = logging.getLogger("app.agent.image_analyzer")


def load_image_bytes(
    image_input: Union[np.ndarray, Path], output_dir: Optional[Path] = None
) -> BinaryContent:
    # Kiểm tra image_input là Path hay numpy array
    if isinstance(image_input, Path):
        image_input = cv2.imread(str(image_input))
    processed = normalize_image(image_input, output_dir)
    # processed = image_input.copy()
    success, encoded = cv2.imencode(".png", processed)
    if not success:
        raise ValueError("Không thể mã hóa ảnh đã tiền xử lý")
    return BinaryContent(data=encoded.tobytes(), media_type="image/png")


# def create_prompt() -> str:
#     return """
#         ### Task: Extract dimensions of all rectangles (left→right, then top→bottom) in the stacked rectangles, return **only JSON**, strictly following this schema:
#         {
#           "rectangles": [
#             {
#               "outer_width": <number>, // Outer width (label on outer most edge)
#               "outer_height": <number>, // Outer height (label on outer most edge)
#               "inner_heights": [<number>, ...], // List of inner measurements inside outer_height.
#               "other_dimensions": [<number>, ...] // List of other numeric labels not included in inner_heights or outer_height or outer_width.
#             }
#           ]
#         }
#         ### Important
#         - Do **not** merge or omit any rectangles even if they are stacked or shared edges.
#         - Include all numbers appear in the drawing, keep numbers exactly as they appear.
# """
def create_prompt() -> str:
    return """
You are a **Dimension Column Extractor**.

### Task
Hãy quét Identify and list **every numeric dimension label** that appears in the drawing.

Group numbers into **columns** —  
each column includes numbers that share roughly the same **vertical alignment** (same x-position).

Return **only pure JSON** in this format:

{
  "columns": [
    {
      "column_no": <number>,       // Column index (1-based, left → right)
      "dimensions": [<number>, ...] // Numeric labels in this column, ordered from top → bottom
    }
  ]
}

### Rules
- Numbers belong to the same column only if their **x positions align vertically** (within a small visual margin).  
- Include **all visible numeric labels** — do not skip any.  
- Keep numbers **exactly as they appear** (no unit, no rounding, no added text).  
- Return **only pure JSON** — no explanation, no comments.


"""


class ImageAnalyzeAgent:
    """Agent phụ trách tách block, gọi LLM phân tích panel của block và (tùy chọn) lưu kết quả."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        # detect_model_id: str = "google-gla:gemini-2.0-flash",
        # detect_model_id: str = "google-gla:gemini-2.5-flash",
        detect_model_id: str = "openai:gpt-4o",
    ) -> None:
        self.output_dir = output_dir
        self.detect_model_id = detect_model_id

        # Khởi tạo agent và prompt một lần cho toàn pipeline
        self.agent = Agent(self.detect_model_id)
        # self.agent = Agent(self.detect_model_id, output_type=Block)
        self.prompt = create_prompt()

    def run(self, source_image: Union[np.ndarray, Path]) -> dict:
        if not os.getenv("GOOGLE_API_KEY"):
            message = "Thiếu biến môi trường GOOGLE_API_KEY cho Gemini API"
            print(message, file=sys.stderr)
            raise RuntimeError(message)
        logger.info("GOOGLE_API_KEY detected, proceeding with analysis")

        # Đảm bảo thư mục output tồn tại nếu người dùng cung cấp
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Output directory prepared | output_dir=%s",
                str(self.output_dir),
            )
        if isinstance(source_image, Path):
            source_image_path_str = str(source_image)
            source_image = cv2.imread(source_image_path_str)
            logger.info(
                "Loaded source image from path | path=%s",
                source_image_path_str,
            )

        # 1) Tách block từ ảnh nguồn
        # padding_pct: dùng 0.0 để giữ nguyên kích thước, có thể tăng lên (vd 0.05 = 5%) nếu muốn viền trắng
        block_images = extract_block_images(source_image, output_dir=self.output_dir)
        logger.info(
            "Extracted block images | block_count=%s",
            len(block_images),
        )

        # 2) Gọi agent phân tích cho từng block và gom kết quả
        aggregated: dict = {"blocks": []}
        for idx, image in enumerate(block_images):
            block_dir = (
                Path(self.output_dir, f"Block {str(idx)}") if self.output_dir else None
            )
            processed_image = load_image_bytes(image, block_dir)
            logger.info(
                "Running panel analysis agent for block | block_index=%s",
                idx,
            )
            agent_result = self.agent.run_sync([self.prompt, processed_image])
            raw_output = agent_result.output
            if not isinstance(raw_output, str):
                # Coerce non-string outputs to string for consistent storage
                try:
                    raw_output = json.dumps(raw_output, ensure_ascii=False)
                except Exception:
                    raw_output = str(raw_output)
            print(raw_output)

            block_inst = {
                "block_no": str(idx),
                "content": raw_output,
            }

            logger.info(
                "Analysis agent completed | block_index=%s | content_len=%s",
                idx,
                len(raw_output or ""),
            )
            aggregated["blocks"].append(block_inst)

        # 3) Lưu JSON kết quả nếu có output_dir
        if self.output_dir:
            result_path = self.output_dir / "image_analyze.json"
            result_path.write_text(
                json.dumps(aggregated, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Đã lưu kết quả panels ban đầu: {result_path}")
            logger.info(
                "Saved initial panel analysis | result_path=%s",
                str(result_path),
            )
        return aggregated


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    image = Path("outputs/5d430f41d60b422a8385dcbc2e96c66f/Block 1/00_origin.png")
    output_dir = Path("outputs/panel_agent/analyzer")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    agent = ImageAnalyzeAgent(output_dir)
    agent.run(image)


if __name__ == "__main__":
    main()
