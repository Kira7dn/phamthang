"""
Shared data models for the entire pipeline.
Provides standardized schemas for all pipeline stages.
"""

from typing import List, Optional, Literal, Dict
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


class OCRVertex(BaseModel):
    x: Optional[int] = None
    y: Optional[int] = None


class OCRTextBlock(BaseModel):
    text: str
    bounding_box: List[OCRVertex] = Field(default_factory=list)
    confidence: Optional[float] = None


class VisionTextDetectionResponse(BaseModel):
    status: Literal["success", "error"]
    full_text: str = ""
    text_blocks: List[OCRTextBlock] = Field(default_factory=list)
    message: Optional[str] = None


# ============================================================================
# Stage 3: Dimension Extraction
# ============================================================================


class BoundingRect(BaseModel):
    """Axis-aligned rectangle in pixel space."""

    x: int
    y: int
    w: int
    h: int


class Panel(BaseModel):
    """Panel with outer dimensions and inner heights."""

    # panel_index: Optional[int] = None
    outer_width: float
    outer_height: float
    inner_heights: List[float] = Field(default_factory=list)
    # frame_pixel: Optional[FramePixel] = None


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


class OCRTextBlocksResult(BaseModel):
    """OCR result with full text blocks."""

    text_blocks: List[Dict] = Field(default_factory=list)


class SimplifiedPanel(BaseModel):
    """Simplified panel with only essential dimensions."""

    outer_width: float
    outer_height: float
    inner_heights: List[float] = Field(default_factory=list)


class SimplifiedFrame(BaseModel):
    """Frame containing multiple panels."""

    id: str
    panels: List[SimplifiedPanel] = Field(default_factory=list)
    quality_scores: Optional[Dict] = None  # Quality scores from dims_classify


class PipelineResult(BaseModel):
    """Complete pipeline result."""

    frames: List[SimplifiedFrame] = Field(
        default_factory=list
    )  # Renamed from image_sets
    bill_of_materials: Optional[List[MaterialItem]] = None
    confidence: float = 0.0  # Overall confidence score from dimension extraction
