from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.google import GoogleModelSettings

from app.agent.panel_verifier import Panel


class NumbersLLMResult(BaseModel):
    numbers: List[float]


class PanelsLLMResult(BaseModel):
    panels: List[Panel]


class VerifyLLMResult(BaseModel):
    feedback: str
    final: bool


class PanelDimensions(BaseModel):
    outer_width: float | int
    outer_height: float | int


class PanelDimensionsLLMResult(BaseModel):
    panels: List[PanelDimensions]


class PanelInnerHeights(BaseModel):
    inner_heights: List[float]


class PanelInnerHeightsLLMResult(BaseModel):
    panels: List[PanelInnerHeights]


def _normalize_number(value: float | int | str) -> str:
    decimal_value = Decimal(str(value))
    normalized = decimal_value.normalize()
    normalized_str = format(normalized, "f")
    if "." in normalized_str:
        normalized_str = normalized_str.rstrip("0").rstrip(".")
    return normalized_str or "0"


def _collect_panel_number_counts(panels: List[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        outer_width = panel.get("outer_width")
        if outer_width is not None:
            counts[_normalize_number(outer_width)] += 1
        outer_height = panel.get("outer_height")
        if outer_height is not None:
            counts[_normalize_number(outer_height)] += 1
        inner_heights = panel.get("inner_heights") or []
        for inner in inner_heights:
            if inner is None:
                continue
            counts[_normalize_number(inner)] += 1
    return counts


def python_verify_numbers(
    numbers: List[float], panels: List[dict]
) -> Optional[List[str]]:
    required: Counter[str] = Counter()
    for num in numbers:
        if num is None:
            continue
        required[_normalize_number(num)] += 1
    if not required:
        return None

    present = _collect_panel_number_counts(panels)

    missing_counter: Counter[str] = Counter()
    for value, required_count in required.items():
        missing = required_count - present.get(value, 0)
        if missing > 0:
            missing_counter[value] = missing

    if not missing_counter:
        return None

    missing_values: List[str] = []
    for value, count in missing_counter.items():
        missing_values.extend([value] * count)

    return missing_values


def create_number_prompt() -> str:
    return """
    Output strict JSON only:
    {
      "numbers": [number, ...]
    }
    """


def create_panel_dimensions_prompt(numbers: Optional[List[float]] = None) -> str:
    base = """### Task: Identify every panel in the technical drawing and return **only JSON** using this schema:
{
  "panels": [
    {
      "outer_width": <number>,
      "outer_height": <number>
    }
  ]
}

### Instructions
1. Each panel is a bold, fully enclosed rectangle. Treat shared edges as separate panels.
2. Follow left→right, then top→bottom order.
3. `outer_width` and `outer_height` must be the total outer dimensions from the drawing.
4. Do not include inner segment values in this step.
"""
    if numbers:
        base += (
            "\n### Available numeric labels in drawing:\n"
            + f"{', '.join(map(str, numbers))}\n"
        )
    return base


def create_inner_heights_prompt(
    dimensions: PanelDimensionsLLMResult,
    numbers: Optional[List[float]] = None,
    feedback: Optional[str] = None,
) -> str:
    panels_json = json.dumps(
        [panel.model_dump() for panel in dimensions.panels],
        ensure_ascii=False,
        indent=2,
    )
    base = f"""### Task: For each panel below, fill in the internal vertical segments.
Panels (from previous step):
{panels_json}

Return **only JSON** with this schema:
{{
  "panels": [
    {{ "inner_heights": [<number>, ...] }}
  ]
}}

### Instructions
1. Keep the number of panels identical to the list above.
2. For each panel, list all internal heights (top→bottom) that divide the panel vertically.
3. Do NOT change the outer dimensions. Only return inner heights.
4. If a panel has no internal heights, return an empty list for that panel.
"""
    if numbers:
        base += (
            "\n### Available numeric labels in drawing:\n"
            + f"{', '.join(map(str, numbers))}\n"
        )
    if feedback:
        base += f"\n### Feedback from verification:\n{feedback}\n"
    return base


# System prompts for each agent
NUMBERS_SYSTEM_PROMPT = """
You extract all numeric labels from technical drawings.
- Preserve reading order: left→right, then top→bottom.
- Do not sort, deduplicate, or transform values.
- Output must be valid JSON only.
"""

PANEL_DIMENSIONS_SYSTEM_PROMPT = """
You are a detector of panel outlines.
- Ignore all diagonal lines (if have).
- Identify every bold rectangular panel individually.
- Report only the total outer width and height for each panel.
- Do not include inner segment measurements in this step.
- Follow left→right, then top→bottom ordering.
"""

PANEL_INNER_SYSTEM_PROMPT = """
You add inner vertical segment heights for each panel.
- Ignore all diagonal lines (if have).
- For every provided panel, list the internal vertical measurements from top to bottom.
- Keep panel ordering and outer dimensions exactly as provided.
- Use only numbers that appear in the drawing.
- Output valid JSON only.
"""


class PanelChainAgent:
    def __init__(
        self,
        model: str = "google-gla:gemini-2.0-flash",
        output_dir: Optional[Path] = None,
    ) -> None:
        self.agent_number = Agent(
            model,
            output_type=NumbersLLMResult,
            system_prompt=NUMBERS_SYSTEM_PROMPT,
        )
        self.agent_panel_dimensions = Agent(
            model,
            output_type=PanelDimensionsLLMResult,
            system_prompt=PANEL_DIMENSIONS_SYSTEM_PROMPT,
            model_settings=(
                GoogleModelSettings(
                    google_thinking_config={"include_thoughts": True},
                    temperature=0,
                )
                if model.startswith("google-gla:")
                else None
            ),
        )
        self.agent_panel_inner = Agent(
            model,
            output_type=PanelInnerHeightsLLMResult,
            system_prompt=PANEL_INNER_SYSTEM_PROMPT,
            model_settings=(
                GoogleModelSettings(
                    google_thinking_config={"include_thoughts": True},
                    temperature=0,
                )
                if model.startswith("google-gla:")
                else None
            ),
        )
        self.output_dir = output_dir

    def _extract_numbers(self, shared_context: List[BinaryContent]) -> List[float]:
        res = self.agent_number.run_sync([create_number_prompt(), *shared_context])
        if isinstance(res.output, NumbersLLMResult):
            return res.output.numbers
        try:
            parsed = NumbersLLMResult.model_validate(res.output)
            return parsed.numbers
        except Exception:
            return []

    def _detect_panel_dimensions(
        self,
        shared_context: List[BinaryContent],
        numbers: Optional[List[float]] = None,
    ) -> PanelDimensionsLLMResult:
        prompt = create_panel_dimensions_prompt(numbers)
        res = self.agent_panel_dimensions.run_sync([prompt, *shared_context])
        if isinstance(res.output, PanelDimensionsLLMResult):
            return res.output
        return PanelDimensionsLLMResult.model_validate(res.output)

    def _extract_inner_heights(
        self,
        shared_context: List[BinaryContent],
        dimensions: PanelDimensionsLLMResult,
        numbers: Optional[List[float]] = None,
        feedback: Optional[str] = None,
    ) -> PanelsLLMResult:
        prompt = create_inner_heights_prompt(dimensions, numbers, feedback)
        res = self.agent_panel_inner.run_sync([prompt, *shared_context])
        if isinstance(res.output, PanelInnerHeightsLLMResult):
            inner = res.output
        else:
            inner = PanelInnerHeightsLLMResult.model_validate(res.output)
        return self._combine_dimensions_and_inner(dimensions, inner)

    @staticmethod
    def _combine_dimensions_and_inner(
        dimensions: PanelDimensionsLLMResult,
        inner: PanelInnerHeightsLLMResult,
    ) -> PanelsLLMResult:
        combined: List[Panel] = []
        total = min(len(dimensions.panels), len(inner.panels))
        for idx in range(total):
            dim = dimensions.panels[idx]
            inner_panel = inner.panels[idx]
            combined.append(
                Panel(
                    outer_width=dim.outer_width,
                    outer_height=dim.outer_height,
                    inner_heights=list(inner_panel.inner_heights),
                )
            )
        return PanelsLLMResult(panels=combined)

    def _verify(
        self,
        panels: PanelsLLMResult,
        numbers: Optional[List[float]] = None,
    ) -> VerifyLLMResult:
        panel_dicts = [p.model_dump() for p in panels.panels]

        missing_values = python_verify_numbers(numbers, panel_dicts)
        if missing_values is None:
            return VerifyLLMResult(feedback="", final=True)
        # Tạo feedback thuần Python thay cho agent_verify
        feedback = f"last_result {panel_dicts}, please find and add these missing values: {missing_values}"
        return VerifyLLMResult(feedback=feedback, final=False)

    def _maybe_save(self, path: Optional[Path], name: str, payload: object) -> None:
        if not path:
            return
        path.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def run_chain(
        self, shared_context: List[BinaryContent], *, output_dir: Optional[Path] = None
    ) -> PanelsLLMResult:
        # Resolve effective output dir (instance-level fallback)
        step_dir = output_dir or self.output_dir

        # 1) Extract numbers
        # numbers = self._extract_numbers(shared_context)
        # self._maybe_save(step_dir, "numbers.json", {"numbers": numbers})

        # 2) Detect panel dimensions
        dimensions = self._detect_panel_dimensions(shared_context)
        self._maybe_save(
            step_dir,
            "panel_dimensions.json",
            dimensions.model_dump(),
        )

        # 3) Extract inner heights based on dimensions
        panels = self._extract_inner_heights(shared_context, dimensions)
        self._maybe_save(
            step_dir,
            "loop_1_panels.json",
            PanelsLLMResult(panels=panels.panels).model_dump(),
        )

        # 4) Verify + feedback loop (tối đa 2 lần)
        # for i in range(1, 2 + 1):
        #     verify = self._verify(numbers, panels)
        #     self._maybe_save(step_dir, f"loop_{i}_verify.json", verify.model_dump())
        #     if verify.final:
        #         self._maybe_save(
        #             step_dir,
        #             "panels_final.json",
        #             PanelsLLMResult(panels=panels.panels).model_dump(),
        #         )
        #         return panels
        #     panels = self._extract_inner_heights(
        #         shared_context,
        #         numbers,
        #         dimensions,
        #         verify.feedback,
        #     )
        #     self._maybe_save(
        #         step_dir,
        #         f"loop_{i+1}_panels.json",
        #         PanelsLLMResult(panels=panels.panels).model_dump(),
        #     )
        return panels
