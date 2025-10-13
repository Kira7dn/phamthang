from __future__ import annotations

import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union
import uuid

import cv2
from dotenv import load_dotenv
import numpy as np

from app.agent.item_builder import BuildItemAgent
from app.agent.llm_ocr import OCRAgent, ImageSet
from app.models import (
    MaterialItem,
    Panel,
    VerifiedPanel,
    VerifiedBlock,
    VerifiedResult,
    ImageSetResult,
    PipelineResult,
    OCRResult,
    BillOfMaterials,
    FramePixel,
)
from utils.cluster_image import extract_block_images
from utils.frame_detection import detect_frames_pipeline
from utils.dimension_infer import extract_dimension
from utils.normalize_image import normalize_image


logger = logging.getLogger("app.pipeline")


class ExtractPanelPipeline:

    def __init__(
        self,
        # analyze_model_id: str = "google-gla:gemini-2.0-flash",
        analyze_model_id: str = "openai:gpt-4o",
        item_model_id: str = "openai:gpt-4o-mini",
        output_dir: Optional[Path] = None,
    ) -> None:
        self.output_dir = output_dir
        output_dir_str = str(output_dir) if output_dir else "None"
        logger.info(
            "Initializing ExtractPanelPipeline | output_dir=%s | analyze_model_id=%s | item_model_id=%s",
            output_dir_str,
            analyze_model_id,
            item_model_id,
        )
        self.ocr_agent = OCRAgent(detect_model_id=analyze_model_id)
        self.item_agent = BuildItemAgent(
            output_dir=output_dir,
            model_id=item_model_id,
        )

    def run(self, image: Union[np.ndarray, Path]) -> PipelineResult:
        # Kiểm tra image_input là Path hay numpy array
        if isinstance(image, Path):
            image = cv2.imread(str(image))
        image_path_str = str(image) if isinstance(image, Path) else "numpy"
        logger.info(
            "Pipeline run started | image_path=%s",
            image_path_str,
        )
        # 1) Tách block từ ảnh nguồn
        block_images = extract_block_images(image, output_dir=self.output_dir)
        logger.info("Extracted block images", extra={"block_count": len(block_images)})

        # 2) Normalize tất cả block images để cải thiện chất lượng OCR và frame detection
        normalized_images = []
        for i, block_img in enumerate(block_images):
            normalized_dir = (
                Path(self.output_dir, f"Block {i}", "normalized")
                if self.output_dir
                else None
            )
            normalized_img = normalize_image(block_img, normalized_dir)
            normalized_images.append(normalized_img)
        logger.info("Normalized all block images")

        # 3) Tạo ImageSet cho từng block với ảnh đã normalize
        image_sets = [
            ImageSet(id=str(i), image=img) for i, img in enumerate(normalized_images)
        ]

        # 3) Chạy OCR bất đồng bộ song song với frame detection
        if image_sets:
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Submit OCR vào thread (network-bound, chờ agent call)
                ocr_future = executor.submit(self.ocr_agent.run, image_sets)

                # Trong lúc chờ OCR, chạy frame detection tuần tự trên main thread
                # Note: image_set.image đã được normalize ở bước 2
                for image_set in image_sets:
                    out_dir = (
                        Path(
                            self.output_dir, f"Block {image_set.id}", "frame_detection"
                        )
                        if self.output_dir
                        else None
                    )

                    # Detect frames (image đã được normalize)
                    _, frames = detect_frames_pipeline(image_set.image, out_dir)
                    image_set.frames = frames
                    logger.info(
                        "Frame detection done",
                        extra={"block_id": image_set.id, "frames_count": len(frames)},
                    )

                # Lấy kết quả OCR khi hoàn tất
                ocr_result = ocr_future.result()
                logger.info("OCR stage completed")
        else:
            ocr_result = self.ocr_agent.run(image_sets)
            logger.info("OCR stage completed (no blocks)")

        # 4) Extract dimensions (outer + inner) cho từng ImageSet
        # Store full dimension reports (including debug info)
        dimension_reports = {}

        for image_set in image_sets:
            # Tìm OCR numbers tương ứng với image_set này
            ocr_image = next(
                (img for img in ocr_result.images if img.id == image_set.id), None
            )
            if ocr_image and image_set.frames:
                # Chuẩn bị dữ liệu cho extract_dimension
                dimension_input = {
                    "id": image_set.id,
                    "frames": image_set.frames,
                    "numbers": ocr_image.numbers,
                }

                # Extract dimensions
                dimension_report = extract_dimension(dimension_input)
                if "error" in dimension_report:
                    dimension_reports[image_set.id] = {
                        "frames": [],
                        "error": dimension_report["error"],
                    }
                    logger.warning(
                        "Dimension extraction failed",
                        extra={
                            "block_id": image_set.id,
                            "error": dimension_report["error"],
                        },
                    )
                else:
                    panels = dimension_report.get("frames", [])
                    dimension_reports[image_set.id] = dimension_report
                    logger.info(
                        "Dimension extraction done",
                        extra={
                            "block_id": image_set.id,
                            "panels_count": len(panels),
                            "panels_with_inner": sum(
                                1 for p in panels if p.get("inner_heights")
                            ),
                            "scale": dimension_report.get("scale"),
                        },
                    )
            else:
                dimension_reports[image_set.id] = {"frames": []}
                logger.warning(
                    "Skipped dimension extraction",
                    extra={
                        "block_id": image_set.id,
                        "has_ocr": ocr_image is not None,
                        "has_frames": bool(image_set.frames),
                    },
                )

        logger.info("Frame, OCR, and dimension results aggregated")

        # 5) Convert panels to VerifiedResult format for item builder
        blocks = []
        for image_set in image_sets:
            report = dimension_reports.get(image_set.id, {})
            panels_data = report.get("frames", [])
            if panels_data:
                # Convert dict panels to VerifiedPanel objects
                panel_objects = [
                    VerifiedPanel(
                        outer_width=p["outer_width"],
                        outer_height=p["outer_height"],
                        inner_heights=p.get("inner_heights", []),
                    )
                    for p in panels_data
                ]
                blocks.append(
                    VerifiedBlock(block_no=image_set.id, panels=panel_objects)
                )

        panels_result = VerifiedResult(blocks=blocks)
        logger.info(
            "Prepared panels for BOM",
            extra={
                "blocks_count": len(blocks),
                "total_panels": sum(len(b.panels) for b in blocks),
            },
        )

        # 6) Build bill of materials
        items_result = self.item_agent.run(panels_result)
        logger.info(
            "Bill of materials built",
            extra={"item_count": len(items_result.material_list)},
        )

        # 7) Build final PipelineResult
        # Convert OCR result
        ocr_result_model = OCRResult(
            images=[{"id": img.id, "numbers": img.numbers} for img in ocr_result.images]
        )

        # Convert image sets with panels
        image_set_results = []
        for img_set in image_sets:
            report = dimension_reports.get(img_set.id, {})
            panels_data = report.get("frames", [])
            panels_models = [
                Panel(
                    panel_index=p.get("panel_index", idx),
                    outer_width=p["outer_width"],
                    outer_height=p["outer_height"],
                    inner_heights=p.get("inner_heights", []),
                    frame_pixel=(
                        FramePixel(
                            x=int(img_set.frames[idx]["x"]),
                            y=int(img_set.frames[idx]["y"]),
                            w=int(img_set.frames[idx]["w"]),
                            h=int(img_set.frames[idx]["h"]),
                        )
                        if img_set.frames and idx < len(img_set.frames)
                        else None
                    ),
                )
                for idx, p in enumerate(panels_data)
            ]

            image_set_results.append(
                ImageSetResult(
                    id=img_set.id,
                    frames_count=len(img_set.frames) if img_set.frames else 0,
                    frames=img_set.frames or [],
                    panels_count=len(panels_models),
                    panels=panels_models,
                )
            )

        # Calculate overall confidence from dimension reports
        total_conf_sum = 0.0
        total_conf_count = 0
        for report in dimension_reports.values():
            if "error" in report:
                continue
            for frame in report.get("frames", []):
                # Average of outer width, outer height, and inner confidences
                outer_w_conf = frame.get("outer_width_conf", 0.0)
                outer_h_conf = frame.get("outer_height_conf", 0.0)
                inner_conf = frame.get("inner_conf", 0.0)
                # Weight outer more than inner (outer is more critical)
                frame_conf = (outer_w_conf + outer_h_conf + inner_conf * 0.5) / 2.5
                total_conf_sum += frame_conf
                total_conf_count += 1

        overall_confidence = (
            round(total_conf_sum / total_conf_count, 3) if total_conf_count > 0 else 0.0
        )

        # Build complete pipeline result
        # Convert BOMItem to MaterialItem
        material_items = [
            MaterialItem(
                type=item.type,
                size=item.size,
                unit=item.unit,
                quantity=item.quantity,
                note=item.note,
            )
            for item in items_result.material_list
        ]

        pipeline_result = PipelineResult(
            ocr=ocr_result_model,
            image_sets=image_set_results,
            verified_result=None,  # No verification step
            bill_of_materials=BillOfMaterials(material_list=material_items),
            confidence=overall_confidence,  # Add overall confidence
        )

        # 8) Save results if output_dir specified
        if self.output_dir:
            # Save complete pipeline result
            result_path = self.output_dir / "pipeline_result.json"
            result_path.write_text(
                pipeline_result.model_dump_json(indent=2), encoding="utf-8"
            )
            logger.info(
                "Saved complete pipeline result", extra={"path": str(result_path)}
            )

            # Save bill of materials separately for convenience
            bom_path = self.output_dir / "bill_of_materials.json"
            bom_path.write_text(
                items_result.model_dump_json(indent=2), encoding="utf-8"
            )
            logger.info("Saved bill of materials", extra={"path": str(bom_path)})

        return pipeline_result


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # img_path = Path("assets/19b2e788907a1a24436b.jpg")
    # img_path = Path("assets/z7070874630878_585ee684038aad2c9e213817e6749e12.jpg")
    # img_path = Path("assets/z7064219010311_67ae7d4dca697d1842b79755dd0c1b4c.jpg")
    thin5_thin3 = Path("assets/z7064218874273_30187de327e4ffc9c1886f540a5f2f30.jpg")
    # img_path = Path("assets/z7070874630879_9b10f5140abae79dee0421db84193312.jpg")
    # img_path = Path("assets/z7102259936013_b55eb7da65cf594e93eb2b8ff31af7b6.jpg")
    # img_path = Path("assets/z7070874695339_7eec1b9a231e267bca5e9e795f4f630d.jpg")
    img_path = thin5_thin3

    # Generate unique output directory with UUID
    run_id = uuid.uuid4().hex[:8]  # Use first 8 chars of UUID
    output_dir = Path(f"outputs/pipeline/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Pipeline run ID: {run_id}")
    logger.info(f"Output directory: {output_dir}")

    pipeline = ExtractPanelPipeline(output_dir=output_dir)
    result = pipeline.run(img_path)

    logger.info(f"Pipeline completed. Results saved to: {output_dir}")
    return result


if __name__ == "__main__":
    main()
