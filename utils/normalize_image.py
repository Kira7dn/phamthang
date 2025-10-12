from pathlib import Path
import shutil
from typing import Optional

import numpy as np
import cv2
from utils.image_process import (
    ImagePipeline,
    adaptive_threshold,
    bridge_horizontal,
    clahe,
    add_white_padding,
    invert_background,
    morph_close,
    morph_open,
    normalize_bg,
    remove_small_components,
    resize_with_limit,
    to_gray,
)


def _resize_short_edge(image: np.ndarray, target_short: int) -> np.ndarray:
    h, w = image.shape[:2]
    if target_short <= 0 or min(h, w) == target_short:
        return image
    if h <= w:
        new_h = target_short
        new_w = int(w * (target_short / h))
    else:
        new_w = target_short
        new_h = int(h * (target_short / w))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_image(
    image: np.ndarray,
    output_path: Optional[Path] = None,
    padding_pct: float = 0.05,
    final_short_edge: Optional[int] = None,
) -> np.ndarray:
    if image is None or (hasattr(image, "size") and image.size == 0):
        raise ValueError("normalize_image: input image is empty or None")
    pipeline = ImagePipeline(output_path)

    pipeline.add("resize", lambda image: resize_with_limit(image, 1920, 1920)[0])
    pipeline.add("to_gray", to_gray)
    pipeline.add("normalize_bg", normalize_bg)
    pipeline.add("blur", lambda image: cv2.medianBlur(image, 5))
    pipeline.add("blur", lambda image: cv2.GaussianBlur(image, (5, 5), 0))
    # pipeline.add("clahe", clahe)
    pipeline.add("adaptive_threshold", adaptive_threshold)
    # pipeline.add("invert_background", invert_background)
    pipeline.add("blur", lambda image: cv2.medianBlur(image, 1))
    pipeline.add("blur", lambda image: cv2.GaussianBlur(image, (1, 1), 0))
    pipeline.add(
        "enhance_edges",
        lambda img: cv2.addWeighted(img, 1, cv2.GaussianBlur(img, (0, 0), 3), -0.5, 0),
    )
    # pipeline.add("clahe", clahe)
    pipeline.add(
        "otsu",
        lambda image: cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1],
    )
    pipeline.add("morph_open", lambda image: morph_open(image, (2, 2), 1))
    pipeline.add("morph_close", lambda image: morph_close(image, (3, 3), 2))
    pipeline.add("remove_small_components", remove_small_components)
    # pipeline.add("bridge_horizontal", lambda image: bridge_horizontal(image, (1, 1), 2))
    # pipeline.add("invert_background", invert_background)
    pipeline.add("white_padding", lambda img: add_white_padding(img, padding_pct))
    if final_short_edge is not None:
        pipeline.add(
            "resize_short_edge", lambda img: _resize_short_edge(img, final_short_edge)
        )
    output_image = pipeline.run(image)
    return output_image


if __name__ == "__main__":
    # img_path = Path("outputs/panel_analyze_pipeline/Block 2/00_origin.png")
    # img_path = Path("outputs/5d430f41d60b422a8385dcbc2e96c66f/Block 0/00_origin.png")
    # img_path = Path("outputs/24a94f1546374c16b54e1e411cc96010/Block 1/00_origin.png")
    # img_path = Path("outputs/c7895b9a60794796bcdb6568edda235b/Block 0/00_origin.png")
    # img_path = Path("outputs/22609b998d2c4ce28e6bceb77b92ffb2/Block 0/00_origin.png")
    img_path = Path("assets/block/image.png")

    output_dir = Path("outputs", "normalize_image")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(
            f"Không đọc được ảnh: {img_path}. Kiểm tra đường dẫn và quyền truy cập."
        )
    normalized_image = normalize_image(image, output_dir)
    print("Normalized image saved to:", output_dir)
