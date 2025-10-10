from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent


# Module logger
logger = logging.getLogger("app.agent.panel_verifier")


# ======== Shared schemas (aligned with image_llm.py) ========
class Panel(BaseModel):
    panel_index: int
    outer_width: float | int
    outer_height: float | int
    inner_heights: List[float] = Field(default_factory=list)


class Block(BaseModel):
    block_no: str
    panels: List[Panel]


class AggregatedResult(BaseModel):
    blocks: List[Block]


# ======== Helpers ========


def load_result_json(path: Path) -> AggregatedResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return AggregatedResult.model_validate(data)


# ======== Agent prompts for multi-stage correction ========


def create_partial_fix_prompt() -> str:
    return """
    You are an **Aluminum Door Panel Normalization Agent**.

    Your role is to analyze hinge spacing data extracted from OCR of aluminum door technical drawings.  
    Each panel represents one vertical door frame section, where `inner_heights` describes the vertical distances between hinge positions.

    Goal: Normalize and complete partially specified hinge spacing while keeping panel geometry unchanged.

    Input: AggregatedResult JSON describing panels.  
    Output: AggregatedResult JSON with normalized `inner_heights`.

    Rules:
    1. Keep `outer_width`, `outer_height`, and `panel_index` exactly as provided.
    2. Treat `inner_heights` as hinge spacing segments that sum vertically from top to bottom.
    3. Let `sum_h = sum(inner_heights)`.  
       - If `inner_heights` is empty, leave it empty.  
       - If `inner_heights` is not empty:
         a. If |sum_h - outer_height| ≤ 0.01 * outer_height (≤1%), leave unchanged.  
         b. If 0 < (outer_height - sum_h) ≤ 0.6 * outer_height:
          - Compute `missing_amount = outer_height - sum_h`.  
          - Round to nearest integer.  
          - Insert one new element equal to `missing_amount`, prioritizing a placement that keeps the spacing sequence as symmetrical and realistic as possible (usually at the top or bottom).  
          - Ensure total ≈ `outer_height`.
    4. Keep numeric formats consistent (integers if input integers).
    5. Output **valid JSON only**, no commentary or extra text.
    """


def create_canonical_prompt() -> str:
    return """
    You are an **Aluminum Door Hinge Pattern Standardization Agent**.

    Your role is to ensure consistent hinge spacing across all panels of identical size and to fill missing data using canonical patterns.

    Input: AggregatedResult JSON (after Stage 1 normalization).  
    Output: AggregatedResult JSON with canonical and filled hinge spacing patterns.

    Rules:
    1. Do NOT modify `outer_width`, `outer_height`, or `panel_index`.
    2. Group panels by identical (`block_no`, `outer_height`).
    3. For each group:
       a. Identify all panels with non-empty `inner_heights`.
       b. Select a canonical hinge spacing pattern using these priorities (in order):
          - The pattern that appears most frequently.
          - If multiple patterns have equal frequency, choose the one with the **largest number of segments** (more hinge points = more complete).
          - If still tied, prefer the pattern that is **most symmetric or evenly spaced**.
       c. Replace all non-empty `inner_heights` in the group with the chosen canonical pattern.
       d. For panels with empty `inner_heights`, copy the canonical pattern of their group.
       e. If a group has no canonical pattern (all empty), leave them unchanged (`inner_heights = []`) for manual inspection.
    4. Keep numeric consistency and panel order.
    5. Respond with **valid JSON only**, no explanation or extra text.
    """


class PanelVerifyAgent:

    def __init__(
        self,
        model_id: str = "openai:o4-mini",
        output_dir: Optional[Path] = None,
    ) -> None:
        self.model_id = model_id
        self.output_dir = output_dir
        logger.info(
            "Initializing PanelVerifyAgent | model_id=%s | output_dir=%s",
            model_id,
            str(output_dir) if output_dir else "None",
        )
        self.partial_prompt = create_partial_fix_prompt()
        self.canonical_prompt = create_canonical_prompt()

        self.partial_agent = Agent(
            self.model_id,
            output_type=AggregatedResult,
            system_prompt=self.partial_prompt,
        )
        self.canonical_agent = Agent(
            self.model_id,
            output_type=AggregatedResult,
            system_prompt=self.canonical_prompt,
        )

    def _ensure_api_key(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            message = "Thiếu biến môi trường OPENAI_API_KEY cho OpenAI API"
            print(message, file=sys.stderr)
            raise RuntimeError(message)
        logger.info("OPENAI_API_KEY detected, proceeding with verification")

    def _ensure_output_dirs(self) -> None:
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Verification output directory prepared | output_dir=%s",
            str(self.output_dir),
        )

    def _run_stage(
        self, agent: Agent, aggregated: AggregatedResult
    ) -> AggregatedResult:
        logger.info("Running verification stage | agent=%s", agent.model)
        aggregated_json = aggregated.model_dump()
        aggregated_json_str = json.dumps(aggregated_json, ensure_ascii=False)
        result = agent.run_sync(aggregated_json_str)
        logger.info("Verification stage finished | agent=%s", agent.model)
        if isinstance(result.output, AggregatedResult):
            return result.output
        return AggregatedResult.model_validate(result.output)

    def _save_stage(self, stage: AggregatedResult, filename: str) -> None:
        if not self.output_dir:
            return
        stage_path = self.output_dir / filename
        stage_path.write_text(
            json.dumps(stage.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Đã lưu dữ liệu ({filename}): {stage_path}")
        logger.info(
            "Saved verification stage result | stage_file=%s",
            str(stage_path),
        )

    def run(self, aggregated: AggregatedResult) -> AggregatedResult:
        self._ensure_api_key()
        self._ensure_output_dirs()

        logger.info("Starting panel verification pipeline")
        stage1 = self._run_stage(self.partial_agent, aggregated)
        logger.info("Stage 1 completed")
        self._save_stage(stage1, "stage1.json")

        stage2 = self._run_stage(self.canonical_agent, stage1)
        logger.info("Stage 2 completed")
        self._save_stage(stage2, "stage2.json")
        return stage2


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Thiếu biến môi trường OPENAI_API_KEY cho OpenAI API", file=sys.stderr)
        sys.exit(1)
    output_dir = Path("outputs/panel_agent/verifier")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    analyze_result = Path("outputs/panel_analyze_pipeline/image_analyze.json")
    if not analyze_result.exists():
        print(f"Không tìm thấy file đầu vào: {analyze_result}", file=sys.stderr)
        sys.exit(1)
    agent = PanelVerifyAgent(output_dir=output_dir)

    return agent.run(load_result_json(analyze_result))


if __name__ == "__main__":
    main()
