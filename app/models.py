"""
Shared data models for the entire pipeline.
Provides standardized schemas for all pipeline stages.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Stage 1: Frame Detection
# ============================================================================

class FramePixel(BaseModel):
    """Frame coordinates in pixel space."""
    x: int
    y: int
    w: int
    h: int


class Frame(BaseModel):
    """Detected frame with metadata."""
    x: int
    y: int
    w: int
    h: int
    area: float
    aspect: float
    fill_ratio: float
    h_lines_count: int
    v_lines_count: int
    total_lines: int


# ============================================================================
# Stage 2: OCR
# ============================================================================

class OCRImage(BaseModel):
    """OCR result for a single image/block."""
    id: str
    numbers: List[float] = Field(default_factory=list)


class OCRResult(BaseModel):
    """Aggregated OCR results."""
    images: List[OCRImage] = Field(default_factory=list)


# ============================================================================
# Stage 3: Dimension Extraction
# ============================================================================

class Panel(BaseModel):
    """Panel with outer dimensions and inner heights."""
    panel_index: int
    outer_width: float
    outer_height: float
    inner_heights: List[float] = Field(default_factory=list)
    frame_pixel: Optional[FramePixel] = None


class DimensionResult(BaseModel):
    """Dimension extraction result for a block."""
    block_id: str
    panels: List[Panel] = Field(default_factory=list)


# ============================================================================
# Stage 4: Verification & Normalization
# ============================================================================

class VerifiedPanel(BaseModel):
    """Verified and normalized panel."""
    outer_width: float
    outer_height: float
    inner_heights: List[float] = Field(default_factory=list)


class VerifiedBlock(BaseModel):
    """Block with verified panels."""
    block_no: str
    panels: List[VerifiedPanel] = Field(default_factory=list)


class VerifiedResult(BaseModel):
    """Verified and normalized result."""
    blocks: List[VerifiedBlock] = Field(default_factory=list)


# ============================================================================
# Stage 5: Bill of Materials
# ============================================================================

class MaterialItem(BaseModel):
    """Single item in bill of materials."""
    type: str
    size: str
    unit: str
    quantity: float
    note: str = ""


class BillOfMaterials(BaseModel):
    """Complete bill of materials."""
    material_list: List[MaterialItem] = Field(default_factory=list)


# ============================================================================
# Pipeline Result (Complete)
# ============================================================================

class ImageSetResult(BaseModel):
    """Result for a single image set/block."""
    id: str
    frames_count: int = 0
    frames: List[Frame] = Field(default_factory=list)
    panels_count: int = 0
    panels: List[Panel] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Complete pipeline result."""
    ocr: OCRResult
    image_sets: List[ImageSetResult] = Field(default_factory=list)
    verified_result: Optional[VerifiedResult] = None
    bill_of_materials: Optional[BillOfMaterials] = None
    confidence: float = 0.0  # Overall confidence score from dimension extraction
