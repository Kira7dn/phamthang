from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union
import uuid
import base64

import cv2
from dotenv import load_dotenv
import numpy as np

from app.models import (
    MaterialItem,
    PipelineResult,
    SimplifiedFrame,
    Panel,
)
from app.tools.cluster_image import cluster_blocks
from app.tools.frame_detection import detect_frames
from app.tools.dimension.dims_classify import classify_dimensions
from app.tools.normalize_image import normalize_frame, normalize_text
from app.tools.vision_ocr import ocr_text
from app.tools.build_item_list import build_item_list


logger = logging.getLogger(__name__)


def _get_block_dir(
    output_dir: Optional[Path], block_id: str, subdir: str
) -> Optional[Path]:
    """Helper: Get block output directory."""
    if output_dir:
        path = output_dir / f"Block {block_id}" / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path
    return None


class ExtractPanelPipeline:

    def __init__(
        self,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.output_dir = output_dir

    def run(self, image: Union[np.ndarray, Path]) -> PipelineResult:
        # Kiểm tra image_input là Path hay numpy array
        if isinstance(image, Path):
            image = cv2.imread(str(image))
        image_path_str = str(image) if isinstance(image, Path) else "numpy"
        logger.info(f"Pipeline run started - image_path={image_path_str}")
        # 1) Tách block từ ảnh nguồn
        block_images = cluster_blocks(image, output_dir=self.output_dir)
        logger.info(f"Extracted block images - block_count={len(block_images)}")

        # 2) Xử lý từng block: normalize, frame detection, OCR, dimensions
        simplified_frames: List[SimplifiedFrame] = []  # Build directly in loop

        for i, block_img in enumerate(block_images):
            block_id = str(i)

            # Normalize và detect frames
            frame_dir = _get_block_dir(self.output_dir, block_id, "normalized_frames")
            frame_img, scale_factor = normalize_frame(block_img, frame_dir)

            frame_out_dir = _get_block_dir(self.output_dir, block_id, "frame_detection")
            _, frames = detect_frames(frame_img, frame_out_dir)
            if not frames:
                # Add empty frame
                simplified_frames.append(
                    SimplifiedFrame(id=block_id, panels=[], quality_scores=None)
                )
                continue
            # Scale frame coordinates back to original image dimensions
            for frame in frames:
                frame.x = int(frame.x / scale_factor)
                frame.y = int(frame.y / scale_factor)
                frame.w = int(frame.w / scale_factor)
                frame.h = int(frame.h / scale_factor)
                frame.area = frame.w * frame.h
                frame.aspect = frame.h / frame.w if frame.w > 0 else 0
            logger.info(
                f"Frame detection done - block_id={block_id}, frames_count={len(frames)}"
            )

            # Normalize và run OCR
            ocr_dir = _get_block_dir(self.output_dir, block_id, "normalized_ocr")
            ocr_img, ocr_scale = normalize_text(block_img, ocr_dir)
            # ocr_img = block_img.copy()

            ocr_out_dir = _get_block_dir(self.output_dir, block_id, "vision_ocr")

            # Encode OCR image to base64
            success, buffer = cv2.imencode(".png", ocr_img)
            if not success:
                logger.warning(f"Failed to encode image - block_id={block_id}")
                # Add empty frame
                simplified_frames.append(
                    SimplifiedFrame(id=block_id, panels=[], quality_scores=None)
                )
                continue

            image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

            # Call Vision OCR (returns List[OCRTextBlock])
            ocr_blocks = ocr_text(image_base64, output_dir=ocr_out_dir)

            if ocr_blocks:
                inv_scale = 1.0 / ocr_scale if ocr_scale not in (0.0, None) else 1.0
                for block_data in ocr_blocks:
                    if not block_data.bounding_box:
                        continue
                    for vertex in block_data.bounding_box:
                        if vertex.x is not None:
                            vertex.x = int(round(vertex.x * inv_scale))
                        if vertex.y is not None:
                            vertex.y = int(round(vertex.y * inv_scale))

            # Classify dimensions (ngay sau khi có frames và OCR)
            if ocr_blocks and frames:
                try:
                    # classify_dimensions now returns List[SimplifiedFrame]
                    dims_out_dir = (
                        Path(self.output_dir, f"Block {block_id}")
                        if self.output_dir
                        else None
                    )
                    frame_results = classify_dimensions(
                        ocr_blocks, frames, output_dir=dims_out_dir
                    )
                    if frame_results:
                        # Rename frame IDs và append trực tiếp
                        for idx, frame in enumerate(frame_results):
                            frame.id = f"{block_id}-{idx}"
                            simplified_frames.append(frame)

                        logger.info(
                            f"Dimension classification done - block_id={block_id}"
                        )

                    else:
                        # Add empty frame
                        simplified_frames.append(
                            SimplifiedFrame(id=block_id, panels=[], quality_scores=None)
                        )
                except Exception:
                    # Add empty frame on error
                    simplified_frames.append(
                        SimplifiedFrame(id=block_id, panels=[], quality_scores=None)
                    )
                    logger.exception(
                        "Dimension classification failed for block %s", block_id
                    )
            else:
                # Add empty frame when skipped
                simplified_frames.append(
                    SimplifiedFrame(id=block_id, panels=[], quality_scores=None)
                )
                logger.warning(
                    f"Skipped dimension classification - block_id={block_id}, "
                    f"has_ocr={bool(ocr_blocks)}, has_frames={bool(frames)}"
                )

        # 4) Build bill of materials (using SimplifiedFrame directly)
        panels_payload: list[Panel] = []
        for sf in simplified_frames:
            for p in sf.panels:
                panels_payload.append(
                    Panel(
                        outer_width=p.outer_width,
                        outer_height=p.outer_height,
                        inner_heights=p.inner_heights,
                    )
                )
        items_output = build_item_list(panels_payload)
        logger.info(
            f"Bill of materials built - item_count={len(items_output.material_list)}"
        )

        # 3) Calculate overall confidence from simplified_frames
        total_conf_sum = 0.0
        total_conf_count = 0
        for frame in simplified_frames:
            if frame.quality_scores:
                # Use overall quality score from dims_classify
                total_conf_sum += frame.quality_scores.get("overall", 0.0)
                total_conf_count += 1

        overall_confidence = (
            round(total_conf_sum / total_conf_count, 3) if total_conf_count > 0 else 0.0
        )

        # Build final result
        pipeline_result = PipelineResult(
            frames=simplified_frames,
            bill_of_materials=[
                MaterialItem(
                    type=item.type,
                    size=item.size,
                    unit=item.unit,
                    quantity=item.quantity,
                    note=item.note,
                )
                for item in items_output.material_list
            ],
            confidence=overall_confidence,
        )

        # 6) Save results if output_dir specified
        if self.output_dir:
            # Save complete pipeline result
            result_path = self.output_dir / "pipeline_result.json"
            result_path.write_text(
                pipeline_result.model_dump_json(indent=2), encoding="utf-8"
            )
            logger.info(f"Saved complete pipeline result: {result_path}")

        return pipeline_result


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # img_path = Path("assets/19b2e788907a1a24436b.jpg")
    img_path = Path("assets\z7070874630879_9b10f5140abae79dee0421db84193312.jpg")
    # img_path = Path("assets/z7070874630878_585ee684038aad2c9e213817e6749e12.jpg")
    # img_path = Path("assets/z7064219010311_67ae7d4dca697d1842b79755dd0c1b4c.jpg")
    # img_path = Path("assets/z7064218874273_30187de327e4ffc9c1886f540a5f2f30.jpg")
    # img_path = Path("assets/z7070874630879_9b10f5140abae79dee0421db84193312.jpg")
    # img_path = Path("assets/z7102259936013_b55eb7da65cf594e93eb2b8ff31af7b6.jpg")
    # img_path = Path("assets/z7070874695339_7eec1b9a231e267bca5e9e795f4f630d.jpg")
    # img_path = Path("assets/442f97ecdcf251ac08e3.jpg")
    # img_path = Path("assets/z7064218874273_30187de327e4ffc9c1886f540a5f2f30.jpg")
    # img_path = Path(
    #     "assets/17102025/z7123665442467_dd861eca02ee6a0736de4928efb9420e.jpg"
    # )
    # img_path = Path(
    #     "assets/17102025/z7123768633710_4cbbff36de040eaf5927947043693c29.jpg"
    # )
    # img_path = Path(
    #     "assets/17102025/z7123791968224_14635243a74af983d35821e934ddfcea.jpg"
    # )
    # img_path = Path(
    #     "assets/17102025/z7123797395329_3b42c63e3bbcb86713c86bb3e2a21824.jpg"
    # )

    # Generate unique output directory with UUID
    run_id = uuid.uuid4().hex[:8]  # Use first 8 chars of UUID
    output_dir = Path(f"outputs2/pipeline/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Pipeline run ID: {run_id}")
    logger.info(f"Output directory: {output_dir}")

    pipeline = ExtractPanelPipeline(output_dir=output_dir)
    result = pipeline.run(img_path)

    logger.info(f"Pipeline completed. Results saved to: {output_dir}")
    return result


if __name__ == "__main__":
    main()
