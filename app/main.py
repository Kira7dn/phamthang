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
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

log_handlers = [
    logging.StreamHandler(),
    RotatingFileHandler(
        data_dir / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    ),
]

logging.basicConfig(level=logging.INFO, format=log_format, handlers=log_handlers)
logger = logging.getLogger("app.main")

load_dotenv()

app = FastAPI(title="Extract Panel Pipeline API")


def _get_outputs_base_dir() -> Path | None:
    if os.getenv("IS_PRODUCTION", "").lower() == "true":
        base_dir = Path("outputs")
    else:
        return None
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
    logger.info("/extract called", extra={"has_file": file is not None, "image_url": str(image_url) if image_url else None})
    if (file is None and image_url is None) or (
        file is not None and image_url is not None
    ):
        logger.warning(
            "Invalid input combination for /extract",
            extra={"has_file": file is not None, "has_image_url": image_url is not None},
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
            logger.error("Uploaded file is empty", extra={"filename": file.filename})
            raise HTTPException(status_code=400, detail="Tệp đầu vào rỗng")
        suffix = Path(file.filename).suffix or ".png"
        await file.close()
        logger.info("Received file upload", extra={"filename": file.filename, "suffix": suffix})
    else:
        assert image_url is not None
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                logger.info("Downloading image from URL", extra={"image_url": str(image_url)})
                response = await client.get(str(image_url))
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Failed to download image", extra={"image_url": str(image_url), "error": str(exc)})
            raise HTTPException(
                status_code=400,
                detail=f"Không thể tải hình ảnh từ URL: {exc}",
            ) from exc

        content = response.content
        if not content:
            logger.error("Downloaded content is empty", extra={"image_url": str(image_url)})
            raise HTTPException(status_code=400, detail="Nội dung ảnh từ URL rỗng")

        parsed_url = urlparse(str(image_url))
        suffix = Path(parsed_url.path).suffix or ".png"
        logger.info("Downloaded image successfully", extra={"image_url": str(image_url), "suffix": suffix})

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    logger.info("Created temporary input file", extra={"tmp_path": str(tmp_path)})

    output_base = _get_outputs_base_dir()
    output_dir = None
    if output_base is not None:
        output_dir = output_base / uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=False)
        logger.info("Prepared output directory", extra={"output_dir": str(output_dir)})

    try:
        pipeline = ExtractPanelPipeline(output_dir=output_dir)
        logger.info("Starting pipeline execution", extra={"output_dir": str(output_dir) if output_dir else None})
        result = await asyncio.to_thread(pipeline.run, tmp_path)
        logger.info("Pipeline execution completed", extra={"output_dir": str(output_dir) if output_dir else None})
        payload = result.model_dump()
        return JSONResponse(content=payload)
    except RuntimeError as exc:
        logger.exception("Pipeline execution failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
        logger.info("Cleaned up temporary input file", extra={"tmp_path": str(tmp_path)})


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
