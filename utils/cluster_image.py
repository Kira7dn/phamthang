from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import shutil

import cv2
import numpy as np

from utils.image_process import (
    ImagePipeline,
    adaptive_threshold,
    enhance_lines,
    normalize_bg,
    resize_with_limit,
    to_gray,
)


@dataclass
class Block:
    index: int
    bbox: Tuple[int, int, int, int]
    area: int


def cluster_components(
    binary: np.ndarray,
    dilate_kernel: Tuple[int, int] = (25, 25),
    iterations: int = 1,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    clustered = cv2.dilate(binary, kernel, iterations=iterations)
    return clustered


def extract_blocks(
    clustered: np.ndarray,
    min_area: int = 5000,
    min_height: int = 0,
) -> List[Block]:
    contours, _ = cv2.findContours(
        clustered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    blocks: List[Block] = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area or h < min_height:
            continue
        blocks.append(Block(index=idx, bbox=(x, y, w, h), area=area))
    blocks.sort(key=lambda blk: (blk.bbox[1], blk.bbox[0]))
    for new_idx, blk in enumerate(blocks):
        blk.index = new_idx
    return blocks


def draw_blocks(image: np.ndarray, blocks: Sequence[Block]) -> np.ndarray:
    vis = image.copy()
    # Ensure we draw on 3-channel image so BGR colors are visible
    if len(vis.shape) == 2 or (vis.ndim == 3 and vis.shape[2] == 1):
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for blk in blocks:
        x, y, w, h = blk.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"{blk.index}: {w}x{h}"
        cv2.putText(
            vis,
            label,
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def save_block_images(
    image: np.ndarray,
    blocks: Sequence[Block],
    output_dir: Path,
    padding: int = 5,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    block_image_paths: List[Path] = []
    for blk in blocks:
        image_out_path = output_dir / f"block_{blk.index:03d}.png"
        x, y, w, h = blk.bbox
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(image.shape[1], x + w + padding)
        y1 = min(image.shape[0], y + h + padding)
        cropped = image[y0:y1, x0:x1]
        cv2.imwrite(str(image_out_path), cropped)
        block_image_paths.append(image_out_path)
    return block_image_paths


def segment_blocks(
    image: np.ndarray,
    output_dir: Optional[Path] = None,
    dilate_kernel: Tuple[int, int] = (20, 20),
    dilate_iterations: int = 1,
    min_block_area_ratio: float = 0.02,
    min_block_height_ratio: float = 0.02,
    max_aspect_ratio: float = 10.0,
) -> List[Block]:
    blocks: List[Block] = []

    original_height, original_width = image.shape[:2]
    resize_state = {
        "ratio_w": 1.0,
        "ratio_h": 1.0,
        "resized_width": original_width,
        "resized_height": original_height,
    }

    pipeline = ImagePipeline(output_dir)

    def resize_step(img: np.ndarray) -> np.ndarray:
        resized = resize_with_limit(img)
        resized_height, resized_width = resized.shape[:2]
        if resized_width == 0 or resized_height == 0:
            resize_state["ratio_w"] = 1.0
            resize_state["ratio_h"] = 1.0
        else:
            resize_state["ratio_w"] = img.shape[1] / float(resized_width)
            resize_state["ratio_h"] = img.shape[0] / float(resized_height)
        resize_state["resized_width"] = resized_width
        resize_state["resized_height"] = resized_height
        return resized

    def expand(img: np.ndarray) -> np.ndarray:
        return cluster_components(
            img, dilate_kernel=dilate_kernel, iterations=dilate_iterations
        )

    def cluster_annotated(img: np.ndarray) -> np.ndarray:
        nonlocal blocks
        resized_height, resized_width = img.shape[:2]
        image_area = resized_height * resized_width
        adaptive_min_area = max(1, int(image_area * min_block_area_ratio))
        adaptive_min_height = max(8, int(resized_height * min_block_height_ratio))
        adaptive_max_aspect = max_aspect_ratio
        filtered: List[Block] = []
        for blk in extract_blocks(
            img, min_area=adaptive_min_area, min_height=adaptive_min_height
        ):
            x, y, w, h = blk.bbox
            if h == 0:
                continue
            aspect = w / float(h)
            if aspect > adaptive_max_aspect:
                continue
            filtered.append(blk)

        ratio_w = resize_state["ratio_w"]
        ratio_h = resize_state["ratio_h"]
        mapped_blocks: List[Block] = []
        for new_idx, blk in enumerate(filtered):
            x, y, w, h = blk.bbox
            x2 = x + w
            y2 = y + h
            orig_x1 = max(0, int(round(x * ratio_w)))
            orig_y1 = max(0, int(round(y * ratio_h)))
            orig_x2 = min(original_width, int(round(x2 * ratio_w)))
            orig_y2 = min(original_height, int(round(y2 * ratio_h)))
            orig_w = max(0, orig_x2 - orig_x1)
            orig_h = max(0, orig_y2 - orig_y1)
            mapped_blocks.append(
                Block(
                    index=new_idx,
                    bbox=(orig_x1, orig_y1, orig_w, orig_h),
                    area=orig_w * orig_h,
                )
            )
        blocks = mapped_blocks
        return draw_blocks(image, blocks)

    pipeline.add("resize", resize_step)
    pipeline.add("to_gray", to_gray)
    pipeline.add("normalize_bg", normalize_bg)
    pipeline.add("adaptive_threshold", adaptive_threshold)
    pipeline.add("enhance_lines", enhance_lines)
    pipeline.add("expand", expand)
    pipeline.add("cluster", cluster_annotated)

    clustered_image = pipeline.run(image)
    # save clustered image if output_dir is not None
    if output_dir is not None:
        clustered_image_path = output_dir / "clustered_image.png"
        cv2.imwrite(str(clustered_image_path), clustered_image)
        print(f"Saved clustered image to: {clustered_image_path}")
    return blocks


def extract_block_image_paths(
    image: Union[Path, str, np.ndarray],
    output_dir: Optional[Path] = None,
) -> List[Path]:
    # Ensure valid output directory
    if output_dir is None:
        output_dir = Path("outputs/cluster_image")
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = output_dir
    block_dir = output_dir

    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
    blocks = segment_blocks(image, processed_dir)
    block_paths = save_block_images(image, blocks, block_dir)
    return block_paths


def extract_block_images(
    image: Union[Path, str, np.ndarray],
    padding: int = 0,
    output_dir: Optional[Path] = None,
) -> List[np.ndarray]:
    """Segment the input image and return cropped block images as numpy arrays."""

    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
    cluster_dir = Path(output_dir, "cluster_image") if output_dir else None
    # Segment without persisting intermediate artifacts
    blocks = segment_blocks(image, cluster_dir)

    block_images: List[np.ndarray] = []
    for blk in blocks:
        x, y, w, h = blk.bbox
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(image.shape[1], x + w + padding)
        y1 = min(image.shape[0], y + h + padding)
        block_images.append(image[y0:y1, x0:x1].copy())

    return block_images


def main() -> None:
    img_path = Path("assets/19b2e788907a1a24436b.jpg")
    # img_path = Path("assets/z7070874630878_585ee684038aad2c9e213817e6749e12.jpg")
    # img_path = Path("assets/z7064219010311_67ae7d4dca697d1842b79755dd0c1b4c.jpg")
    # img_path = Path("assets/z7064218874273_30187de327e4ffc9c1886f540a5f2f30.jpg")
    # img_path = Path("assets/z7070874630879_9b10f5140abae79dee0421db84193312.jpg")

    output_dir = Path("outputs/cluster_image")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {img_path}")

    block_images = extract_block_images(image)
    print(f"Đã tách {len(block_images)} block (extract_block_images)")

    # Lưu thử từng block ra file để quan sát
    sample_dir = output_dir / "blocks_from_main"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for idx, block_img in enumerate(block_images):
        cv2.imwrite(str(sample_dir / f"block_{idx:03d}.png"), block_img)
    print(f"Đã lưu {len(block_images)} block vào {sample_dir}")


if __name__ == "__main__":
    main()
