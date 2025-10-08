from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from pipeline import ExtractPanelPipeline

load_dotenv()

app = FastAPI(title="Extract Panel Pipeline API")


def _ensure_outputs_dir() -> Path:
    base_dir = Path("outputs")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


@app.post("/extract")
async def extract_panel(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Thiếu tên tệp đầu vào")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Tệp đầu vào rỗng")

    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)

    output_base = _ensure_outputs_dir()
    output_dir = Path(tempfile.mkdtemp(prefix="panel_api_", dir=output_base))

    try:
        pipeline = ExtractPanelPipeline(output_dir=output_dir)
        result = pipeline.run(tmp_path)
        payload = result.model_dump()
        return JSONResponse(content=payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.delete("/outputs")
def clear_outputs() -> JSONResponse:
    base_dir = Path("outputs")
    if not base_dir.exists():
        return JSONResponse(content={"message": "Không có dữ liệu để xóa"})
    shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    return JSONResponse(content={"message": "Đã xóa dữ liệu outputs"})


@app.get("/health")
def health_check() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})
