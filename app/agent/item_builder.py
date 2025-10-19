from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from pydantic_ai import Agent, Tool
from dotenv import load_dotenv

from app.tools.build_item_list import BuildItemOutput, build_item_list
from app.models import SimplifiedFrame, Panel


logger = logging.getLogger("app.agent.build_item")


@Tool
def build_item_tool(panels: List[Panel]) -> BuildItemOutput:
    """Aggregate frame items and hinge counts into material_list format."""

    return build_item_list(panels)


BUILD_ITEM_SYSTEM_PROMPT = """
You are an **Aluminum Frame Bill-of-Materials Agent**.

Input JSON contains:
- `panels`: list of panels, each with `panel_index`, `outer_width`, `outer_height`, and `inner_heights` (hinge spacing segments).

Your tasks:
1. Produce an `PanelList` payload suitable for the tool call:
   - Preserve every panel.
   - Ensure each panel includes numeric `outer_width`, `outer_height`, and an `inner_heights` list.
    * Keep the original `inner_heights` sequence when their sum matches `outer_height` within ±5mm.
      * If have big gap between sum of inner_heights compare to outer_height, refer and add that gap as additional segment.
     * Never merge adjacent segments into a new combined value.
     * When inferring new spacing, prioritize reusing values from panels of identical dimensions or build a symmetrical layout while guaranteeing the sum equals `outer_height`.
     * Only leave `inner_heights` empty when no safe inference can be made from available data.
2. Call `build_item_tool` **once** with the final `PanelList` JSON as its argument.
3. Return JSON **only** with the schema:
   {
     "material_list": [...]
   }
   - `material_list` must be exactly the tool output.
   - `material_list` contains the fields `type`, `size`, `unit`, `quantity`, and `note`.
   - Do not include any explanations or extra keys outside the schema.
"""


class BuildItemAgent:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        *,
        model_id: str = "openai:o4-mini",
    ) -> None:
        self.model_id = model_id
        self.output_dir = output_dir
        logger.info(
            "Initializing BuildItemAgent model_id=%s output_dir=%s",
            model_id,
            str(output_dir) if output_dir is not None else "None",
        )
        self.agent = Agent(
            self.model_id,
            output_type=BuildItemOutput,
            system_prompt=BUILD_ITEM_SYSTEM_PROMPT,
            tools=[build_item_tool],
        )

    def _ensure_api_key(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            message = "Thiếu biến môi trường OPENAI_API_KEY cho OpenAI API"
            print(message, file=sys.stderr)
            raise RuntimeError(message)
        logger.info("OPENAI_API_KEY detected, proceeding with BuildItemAgent")

    def run(self, data: List[SimplifiedFrame]) -> BuildItemOutput:
        self._ensure_api_key()
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = [panel for block in data for panel in block.panels]
        payload_str = json.dumps(
            [panel.model_dump() for panel in payload], ensure_ascii=False
        )
        logger.info(
            "Running BuildItemAgent panel_count=%d model=%s",
            len(payload),
            self.model_id,
        )
        result = self.agent.run_sync(payload_str)
        logger.info("BuildItemAgent completed")
        if isinstance(result.output, BuildItemOutput):
            output = result.output
        else:
            output = BuildItemOutput.model_validate(result.output)

        if self.output_dir:
            output_path = self.output_dir / "material_list.json"
            output_path.write_text(
                json.dumps(output.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"Saved BuildItemAgent on {str(output_path)}")
        return output


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Thiếu biến môi trường OPENAI_API_KEY cho OpenAI API", file=sys.stderr)
        sys.exit(1)

    input_path = Path("outputs2/pipeline/8e2aa1c2/Block 0/dimensions_classified.json")
    if not input_path.exists():
        print(
            f"Không tìm thấy file đầu vào: {input_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path("outputs/panel_agent/item_builder")
    # if output_dir.exists():
    #     shutil.rmtree(output_dir)

    aggregated_json = input_path.read_text(encoding="utf-8")
    aggregated = json.loads(aggregated_json)
    panels = aggregated.get("panels")
    frame = {"id": "0", "panels": panels}
    simplified_frames = [SimplifiedFrame(**frame)]
    print(simplified_frames)
    agent = BuildItemAgent(output_dir=output_dir)
    result = agent.run(simplified_frames)
    print(result)


if __name__ == "__main__":
    main()
