from __future__ import annotations

import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import HttpUrl
from dotenv import load_dotenv

from app.pipeline import ExtractPanelPipeline


log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
is_production = os.getenv("IS_PRODUCTION", "").lower() == "true"

# In production (Lambda), không tạo/thao tác thư mục outputs hay file log
data_dir = None
if not is_production:
    data_dir = Path("outputs")
    data_dir.mkdir(parents=True, exist_ok=True)

log_handlers = [logging.StreamHandler()]
if data_dir is not None and data_dir.exists():
    log_handlers.append(
        RotatingFileHandler(
            data_dir / "app.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=2,
            encoding="utf-8",
        )
    )

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=log_handlers,
    force=True,
)
logger = logging.getLogger("app.main")

load_dotenv()

app = FastAPI(title="Extract Panel Pipeline API")


def _get_outputs_base_dir() -> Path | None:
    if is_production:
        return None
    base_dir = Path("outputs")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/extract")
async def extract_panel(
    file: UploadFile | None = File(default=None),
    image_url: HttpUrl | None = Query(default=None),
) -> JSONResponse:
    has_file = file is not None
    has_image_url = image_url is not None
    image_url_display = str(image_url) if image_url else "None"
    logger.info(
        "/extract called | has_file=%s | image_url=%s",
        has_file,
        image_url_display,
    )
    if (file is None and image_url is None) or (
        file is not None and image_url is not None
    ):
        logger.warning(
            "Invalid input combination for /extract | has_file=%s | has_image_url=%s",
            has_file,
            has_image_url,
        )
        raise HTTPException(
            status_code=400,
            detail="Cần cung cấp duy nhất một trong hai: file hoặc image_url",
        )

    if file is not None:
        if not file.filename:
            logger.error("Uploaded file missing filename")
            raise HTTPException(status_code=400, detail="Thiếu tên tệp đầu vào")
        content = await file.read()
        if not content:
            logger.error(
                "Uploaded file is empty | uploaded_filename=%s",
                file.filename,
            )
            raise HTTPException(status_code=400, detail="Tệp đầu vào rỗng")
        suffix = Path(file.filename).suffix or ".png"
        await file.close()
        logger.info(
            "Received file upload | uploaded_filename=%s | suffix=%s",
            file.filename,
            suffix,
        )
    else:
        assert image_url is not None
        image_url_str = str(image_url)
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                logger.info(
                    "Downloading image from URL | image_url=%s",
                    image_url_str,
                )
                response = await client.get(image_url_str)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error(
                "Failed to download image | image_url=%s | error=%s",
                image_url_str,
                str(exc),
            )
            raise HTTPException(
                status_code=400,
                detail=f"Không thể tải hình ảnh từ URL: {exc}",
            ) from exc

        content = response.content
        if not content:
            logger.error(
                "Downloaded content is empty | image_url=%s",
                image_url_str,
            )
            raise HTTPException(status_code=400, detail="Nội dung ảnh từ URL rỗng")

        parsed_url = urlparse(image_url_str)
        suffix = Path(parsed_url.path).suffix or ".png"
        logger.info(
            "Downloaded image successfully | image_url=%s | suffix=%s",
            image_url_str,
            suffix,
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    logger.info("Created temporary input file | tmp_path=%s", str(tmp_path))

    output_base = _get_outputs_base_dir()
    output_dir = None
    if output_base is not None:
        output_dir = output_base / uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=False)
        logger.info(
            "Using output directory | output_dir=%s | output_dir_absolute=%s",
            str(output_dir),
            str(output_dir.resolve()),
        )

    try:
        # Lazy import để tránh import cv2 ở cold start (giúp /health không lỗi)
        from app.pipeline import ExtractPanelPipeline

        pipeline = ExtractPanelPipeline(output_dir=output_dir)
        output_dir_display = str(output_dir) if output_dir else "None"
        logger.info(
            "Starting pipeline execution | output_dir=%s",
            output_dir_display,
        )
        result = await asyncio.to_thread(pipeline.run, tmp_path)
        logger.info(
            "Pipeline execution completed | output_dir=%s | confidence=%.3f",
            output_dir_display,
            result.confidence,
        )

        # Build response with conf and materialist
        payload = {
            "conf": result.confidence,
            "materialist": [item.model_dump() for item in result.bill_of_materials],
        }
        return JSONResponse(content=payload)
    except RuntimeError as exc:
        logger.exception("Pipeline execution failed")
    finally:
        tmp_path.unlink(missing_ok=True)
        logger.info("Cleaned up temporary input file | tmp_path=%s", str(tmp_path))


@app.delete("/outputs")
def clear_outputs() -> JSONResponse:
    base_dir = _get_outputs_base_dir()
    if base_dir is None:
        return JSONResponse(
            content={
                "message": "Biến môi trường OUTPUT_DIR chưa được cấu hình, không có dữ liệu để xóa"
            }
        )

    if base_dir.exists():
        shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
    return JSONResponse(content={"message": f"Đã xóa dữ liệu trong {base_dir}"})


@app.get("/health")
def health_check() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


# pylint: disable=import-error
try:
    from mangum import Mangum

    handler = Mangum(app)
    logger.info("Mangum available: Lambda handler created.")
except Exception:
    handler = None
    logger.info("Mangum not available: running in normal server mode.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
