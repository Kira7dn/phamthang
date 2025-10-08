from __future__ import annotations

from pathlib import Path
import shutil
from typing import Optional, Union

from dotenv import load_dotenv
import numpy as np

from image_analyzer import ImageAnalyzeAgent
from panel_verifier import AggregatedResult, PanelVerifyAgent


class ExtractPanelPipeline:
    def __init__(
        self,
        analyze_model_id: str = "google-gla:gemini-2.0-flash",
        verify_model_id: str = "openai:o4-mini",
        output_dir: Optional[Path] = None,
    ) -> None:
        self.output_dir = output_dir
        self.analyze_agent = ImageAnalyzeAgent(
            output_dir=output_dir, detect_model_id=analyze_model_id
        )
        self.verify_agent = PanelVerifyAgent(
            output_dir=output_dir, model_id=verify_model_id
        )

    def run(self, image: Union[np.ndarray, Path]) -> AggregatedResult:
        image_analyze = self.analyze_agent.run(image)
        verify_result = self.verify_agent.run(image_analyze)
        return verify_result


def main() -> None:
    load_dotenv()
    # img_path = Path("assets/19b2e788907a1a24436b.jpg")
    # img_path = Path("assets/z7070874630878_585ee684038aad2c9e213817e6749e12.jpg")
    # img_path = Path("assets/z7064219010311_67ae7d4dca697d1842b79755dd0c1b4c.jpg")
    img_path = Path("assets/z7064218874273_30187de327e4ffc9c1886f540a5f2f30.jpg")
    # img_path = Path("assets/z7070874630879_9b10f5140abae79dee0421db84193312.jpg")
    output_dir = Path("outputs/panel_analyze_pipeline")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    pipeline = ExtractPanelPipeline(output_dir=output_dir)
    pipeline.run(img_path)


if __name__ == "__main__":
    main()
