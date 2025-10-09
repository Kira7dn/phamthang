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
    pipeline.add("adaptive_threshold", adaptive_threshold)
    pipeline.add("clahe", clahe)
    pipeline.add("invert_background", invert_background)
    pipeline.add("blur", lambda image: cv2.medianBlur(image, 3))
    pipeline.add("blur", lambda image: cv2.GaussianBlur(image, (1, 1), 0))
    pipeline.add(
        "otsu",
        lambda image: cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1],
    )
    pipeline.add("morph_open", lambda image: morph_open(image, (2, 2), 1))
    pipeline.add("morph_close", lambda image: morph_close(image, (2, 2), 1))
    pipeline.add("remove_small_components", remove_small_components)
    pipeline.add("invert_background", invert_background)
    pipeline.add("clahe", clahe)
    output_image = pipeline.run(image)
    return output_image


if __name__ == "__main__":
    # img_path = Path("assets/z7064219281543_b33d93d5cf3880d2f5f6bab3ed22eb89.jpg")
    # img_path = Path("assets/19b2e788907a1a24436b.jpg")
    # img_path = Path("assets/00_origin.png")
    img_path = Path("assets/block0s.png")
    # img_path = Path("assets/z7064218874273_30187de327e4ffc9c1886f540a5f2f30.jpg")
    # img_path = Path("assets/z7064219010311_67ae7d4dca697d1842b79755dd0c1b4c.jpg")
    # img_path = Path("assets/z7070874630878_585ee684038aad2c9e213817e6749e12.jpg")
    # img_path = Path("assets/z7070874630879_9b10f5140abae79dee0421db84193312.jpg")
    # img_path = Path("assets/z7070871858185_d94ed70d5e13fd0ae4bbf39107e29819.jpg")
    # img_path = Path("assets/z7070874630840_d04d8f5aa9a4d5ff280a48471768c51d.jpg")
    output_dir = Path("outputs", "normalize_image")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    image = cv2.imread(img_path)
    normalized_image = normalize_image(image, output_dir)
    print("Normalized image saved to:", output_dir)
