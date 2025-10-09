from pathlib import Path
import shutil
from typing import Optional

import numpy as np
import cv2
from utils.image_process import (
    ImagePipeline,
    adaptive_threshold,
    clahe,
    invert_background,
    morph_close,
    morph_open,
    normalize_bg,
    remove_small_components,
    resize_with_limit,
    to_gray,
)


def normalize_image(
    image: np.ndarray,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    pipeline = ImagePipeline(output_path)

    pipeline.add("resize", resize_with_limit)
    pipeline.add("normalize_bg", normalize_bg)
    pipeline.add("to_gray", to_gray)
    pipeline.add("clahe", clahe)
    pipeline.add("blur", lambda image: cv2.medianBlur(image, 3))
    pipeline.add("blur", lambda image: cv2.GaussianBlur(image, (1, 1), 0))
    pipeline.add("adaptive_threshold", adaptive_threshold)
    pipeline.add("morph_open", lambda image: morph_open(image, (1, 1), 1))
    pipeline.add("morph_close", lambda image: morph_close(image, (2, 2), 1))
    pipeline.add("remove_small_components", remove_small_components)
    pipeline.add("invert_background", invert_background)
    output_image = pipeline.run(image)
    return output_image


if __name__ == "__main__":
    # image_path = Path("assets/z7064218874273_30187de327e4ffc9c1886f540a5f2f30.jpg")
    image_path = Path("assets/z7070874630879_9b10f5140abae79dee0421db84193312.jpg")
    output_dir = Path("outputs", "normalize_image")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    image = cv2.imread(image_path)
    normalized_image = normalize_image(image, output_dir)
    print("Normalized image saved to:", output_dir)
