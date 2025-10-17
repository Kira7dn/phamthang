from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from pydantic_ai import Agent, Tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.tools.build_item_list import BuildItemOutput, build_item_list
from app.models import SimplifiedFrame, Panel


logger = logging.getLogger("app.agent.build_item")


class Block(BaseModel):
    block_no: str
    panels: list[Panel] = Field(default_factory=list)


class AggregatedResult(BaseModel):
    blocks: list[Block] = Field(default_factory=list)


@Tool
def build_item_tool(aggregated: AggregatedResult) -> BuildItemOutput:
    """Aggregate frame items and hinge counts into material_list format."""

    return build_item_list(aggregated)


BUILD_ITEM_SYSTEM_PROMPT = """
You are an **Aluminum Frame Bill-of-Materials Agent**.

Input JSON contains:
- `blocks`: list of blocks, each with `block_no` and an array of `panels`.
  Each panel provides `panel_index`, `outer_width`, `outer_height`, and `inner_heights` (hinge spacing segments).
- `per_block_overrides`: optional overrides keyed by block number.

Your tasks:
1. Produce an `AggregatedResult` payload suitable for the tool call:
   - Preserve every block and panel.
   - Ensure each panel includes numeric `outer_width`, `outer_height`, and an `inner_heights` list.
     * If a panel is missing spacing data, infer the most reasonable sequence before calling the tool.
       Reuse hinge spacing from panels of identical dimensions, apply `defaults` / `per_block_overrides`, or derive a symmetrical layout that sums to the panel height.
      * When multiple orderings are plausible, reorder the segments so the hinge spacing transitions remain balanced and realistic around the panel.
      * In typical doors, the top and bottom segments are shorter than interior segments; maintain this characteristic and aim for overall symmetry.
     * Only leave `inner_heights` empty when there is no reliable information to infer from.
   - You may carry forward descriptive metadata (material, note, overrides) in your reasoning, but the tool only consumes the geometry fields.
2. Call `build_item_tool` **once** with the final `AggregatedResult` JSON as its argument.
3. Return JSON **only** with the schema:
   {
     "material_list": [...]
   }
   - `material_list` must be exactly the tool output; do not reorder, filter, or edit entries except to ensure valid JSON serialization.
   - `material_list` contains the fields `type`, `size`, `unit`, `quantity`, and `note`. Preserve numeric precision (round quantities to two decimals when needed) and keep the note text unchanged.
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

    def run(self, data) -> BuildItemOutput:
        self._ensure_api_key()
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Accept either list[SimplifiedFrame] or AggregatedResult/dict
        if isinstance(data, list):
            frames: list[SimplifiedFrame] = data
            blocks = [
                Block(
                    block_no=frame.id,
                    panels=[
                        Panel(
                            panel_index=idx,
                            outer_width=panel.outer_width,
                            outer_height=panel.outer_height,
                            inner_heights=panel.inner_heights,
                        )
                        for idx, panel in enumerate(frame.panels)
                    ],
                )
                for frame in frames
            ]
            payload = AggregatedResult(blocks=blocks)
        elif isinstance(data, AggregatedResult):
            payload = data
        elif isinstance(data, dict):
            try:
                payload = AggregatedResult.model_validate(data)
            except Exception:
                payload = AggregatedResult(blocks=[])
        else:
            payload = AggregatedResult(blocks=getattr(data, "blocks", []))
        payload_str = json.dumps(payload.model_dump(), ensure_ascii=False)
        logger.info(
            "Running BuildItemAgent block_count=%d model=%s",
            len(payload.blocks),
            self.model_id,
        )
        result = self.agent.run_sync(payload_str)
        logger.info("BuildItemAgent completed")
        output: BuildItemOutput
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

    input_path = Path("outputs/e0b1fbe9b1be48faa8a7f1e52499324b/stage2.json")
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
    aggregated = AggregatedResult.model_validate_json(aggregated_json)

    agent = BuildItemAgent(output_dir=output_dir)
    result = agent.run(aggregated)

    print(f"Output return type of {type(result)}")


if __name__ == "__main__":
    main()
