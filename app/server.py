"""Local web server. Songs stay on this computer."""

from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.audio_io import ALLOWED_EXTENSIONS
from app.errors import UserFacingError
from app.notation import musicxml_document
from app.pipeline import analyze
from app.stems import DEFAULT_MODE, mode_spec
from app.synth import synthesize_example

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
DEFAULT_DATA = ROOT / "data"
MAX_UPLOAD_BYTES = 120 * 1024 * 1024

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analyze")


def data_root() -> Path:
    import os

    return Path(os.environ.get("MUSIC_ANALYZER_DATA", DEFAULT_DATA))


def jobs_dir() -> Path:
    path = data_root() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_dir(job_id: str) -> Path:
    if not job_id or any(char not in "0123456789abcdef" for char in job_id):
        raise HTTPException(status_code=404, detail="没有找到这次分析。")
    path = jobs_dir() / job_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="没有找到这次分析。")
    return path


def _public_view(job: dict) -> dict:
    view = {
        "id": job["id"],
        "status": job["status"],
        "progress": job.get("progress", 0),
        "step": job.get("step", "read"),
        "message": job.get("message", ""),
        "filename": job.get("filename", ""),
    }
    if job.get("status") == "done":
        view["result"] = job.get("result")
    if job.get("status") == "error":
        view["message"] = job.get("message") or "分析失败了。"
    return view


def _write_snapshot(job: dict) -> None:
    path = jobs_dir() / job["id"] / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = _public_view(job)
    path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")


def _update(job: dict, **fields) -> None:
    with _jobs_lock:
        if "progress" in fields:
            fields["progress"] = max(int(job.get("progress", 0)), int(fields["progress"]))
        job.update(fields)
        try:
            _write_snapshot(job)
        except OSError:
            logger.exception("Could not write job status")


def _run_job(job: dict, source: Path, mode: str) -> None:
    work = jobs_dir() / job["id"]

    def progress(percent: int, step: str, message: str) -> None:
        _update(job, status="running", progress=percent, step=step, message=message)

    try:
        _update(job, status="running", progress=1, step="read", message="开始分析…")
        result = analyze(source, work, progress, mode=mode)
        result_path = work / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        _update(job, status="done", progress=100, step="done", message="分析完成", result=result)
    except UserFacingError as exc:
        logger.info("Job %s stopped: %s", job["id"], exc)
        _update(job, status="error", step="error", message=str(exc))
    except Exception:
        logger.error("Job %s failed\n%s", job["id"], traceback.format_exc())
        _update(
            job,
            status="error",
            step="error",
            message="分析时出了点问题。可以换一首更短的歌再试。如果这是第一次使用，请确认电脑已联网，然后重新运行 start 脚本。",
        )


def _new_job(filename: str) -> dict:
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "step": "read",
        "message": "已收到文件，正在排队…",
        "filename": filename,
        "result": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job
    (jobs_dir() / job_id).mkdir(parents=True, exist_ok=True)
    _write_snapshot(job)
    return job


def _remember_existing_jobs() -> None:
    root = jobs_dir()
    if not root.exists():
        return
    for folder in root.iterdir():
        result_path = folder / "result.json"
        status_path = folder / "status.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            job = {
                "id": folder.name,
                "status": "done",
                "progress": 100,
                "step": "done",
                "message": "分析完成",
                "filename": result.get("filename", ""),
                "result": result,
            }
        elif status_path.exists():
            try:
                saved = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if saved.get("status") in {"queued", "running"}:
                saved["status"] = "error"
                saved["message"] = "程序中途关闭了，这次分析没有完成。请重新选一次文件。"
            job = {
                "id": folder.name,
                "status": saved.get("status", "error"),
                "progress": saved.get("progress", 0),
                "step": saved.get("step", "error"),
                "message": saved.get("message", ""),
                "filename": saved.get("filename", ""),
                "result": saved.get("result"),
            }
        else:
            continue
        _jobs[folder.name] = job


def create_app() -> FastAPI:
    app = FastAPI(title="听音识谱", docs_url=None, redoc_url=None)
    _remember_existing_jobs()

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    def _checked_mode(mode: str) -> str:
        try:
            return str(mode_spec(mode)["id"])
        except UserFacingError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/analyze")
    async def start_analyze(file: UploadFile = File(...), mode: str = Form(DEFAULT_MODE)) -> JSONResponse:
        original = Path(file.filename or "audio").name
        extension = Path(original).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="请上传 mp3、wav、m4a 或 flac 文件。",
            )
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="文件是空的。请换一个音频文件。")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="文件超过 120MB 了。请先把歌曲剪短，或换成 mp3。")
        chosen = _checked_mode(mode)
        job = _new_job(original)
        source = jobs_dir() / job["id"] / f"source{extension}"
        source.write_bytes(payload)
        _executor.submit(_run_job, job, source, chosen)
        return JSONResponse(_public_view(job))

    @app.post("/api/demo")
    def start_demo(mode: str = DEFAULT_MODE) -> JSONResponse:
        chosen = _checked_mode(mode)
        job = _new_job("示例音乐.wav")
        source = jobs_dir() / job["id"] / "source.wav"
        synthesize_example(source)
        _executor.submit(_run_job, job, source, chosen)
        return JSONResponse(_public_view(job))

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            status_path = jobs_dir() / job_id / "status.json"
            if not status_path.exists():
                raise HTTPException(status_code=404, detail="没有找到这次分析。")
            return json.loads(status_path.read_text(encoding="utf-8"))
        return _public_view(job)

    def _safe_stem(stem: str) -> str:
        if not stem or any(char not in "abcdefghijklmnopqrstuvwxyz_" for char in stem):
            raise HTTPException(status_code=404, detail="没有这条音轨。")
        return stem

    @app.get("/api/jobs/{job_id}/audio/{stem}")
    def stem_audio(job_id: str, stem: str) -> FileResponse:
        folder = _job_dir(job_id)
        if stem == "mix":
            path = folder / "mix.wav"
            filename = "mix.wav"
        else:
            _safe_stem(stem)
            path = folder / "stems" / f"{stem}.wav"
            filename = f"{stem}.wav"
        if not path.exists():
            raise HTTPException(status_code=404, detail="音轨还没准备好。")
        return FileResponse(path, media_type="audio/wav", filename=filename)

    @app.get("/api/jobs/{job_id}/midi/{stem}")
    def stem_midi(job_id: str, stem: str) -> FileResponse:
        _safe_stem(stem)
        path = _job_dir(job_id) / "midi" / f"{stem}.mid"
        if not path.exists():
            raise HTTPException(status_code=404, detail="MIDI 还没准备好。")
        return FileResponse(path, media_type="audio/midi", filename=f"{stem}.mid")

    @app.get("/api/jobs/{job_id}/musicxml/{name}")
    def musicxml(job_id: str, name: str, simplify: str = "standard") -> Response:
        _safe_stem(name)
        result_path = _job_dir(job_id) / "result.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                xml = musicxml_document(result, name, simplify)
            except Exception:
                logger.exception("五线谱重新生成失败：%s", name)
                xml = None
            if xml:
                return Response(
                    content=xml,
                    media_type="application/vnd.recordare.musicxml+xml",
                    headers={"Content-Disposition": f'inline; filename="{name}.musicxml"'},
                )
        path = _job_dir(job_id) / "notation" / f"{name}.musicxml"
        if not path.exists():
            raise HTTPException(status_code=404, detail="这份结果是旧的，请重新分析一次，才能看五线谱。")
        return FileResponse(path, media_type="application/vnd.recordare.musicxml+xml", filename=f"{name}.musicxml")

    @app.get("/api/jobs/{job_id}/bundle")
    def bundle(job_id: str) -> FileResponse:
        folder = _job_dir(job_id)
        archive = folder / "music-analysis.zip"
        paths = [folder / "mix.wav"]
        stems = folder / "stems"
        midi = folder / "midi"
        if stems.is_dir():
            paths.extend(sorted(stems.glob("*.wav")))
        if midi.is_dir():
            paths.extend(sorted(midi.glob("*.mid")))
        notation = folder / "notation"
        if notation.is_dir():
            paths.extend(sorted(notation.glob("*.musicxml")))
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in paths:
                if path.exists():
                    handle.write(path, arcname=path.name)
        if not archive.exists():
            raise HTTPException(status_code=404, detail="还没有可以打包的结果。")
        return FileResponse(archive, media_type="application/zip", filename="music-analysis.zip")

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
