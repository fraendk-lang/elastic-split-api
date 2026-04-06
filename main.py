"""
Elastic Split API — AI-powered stem separation using demucs v4.
"""

import io
import os
import uuid
import shutil
import logging
import zipfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
import soundfile as sf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PORT = int(os.environ.get("PORT", "8003"))
_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
# Always allow both www and non-www variants
ALLOWED_ORIGINS = []
for o in _origins:
    o = o.strip()
    ALLOWED_ORIGINS.append(o)
    if o.startswith("https://www."):
        ALLOWED_ORIGINS.append(o.replace("https://www.", "https://"))
    elif o.startswith("https://") and "//www." not in o and o != "*":
        ALLOWED_ORIGINS.append(o.replace("https://", "https://www."))
if "*" in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_DURATION_SECONDS = 600  # 10 minutes
JOB_TTL_MINUTES = int(os.environ.get("JOB_TTL_MINUTES", "30"))
TMP_DIR = "/tmp/elastic-split-jobs"

VALID_FORMATS = {".wav", ".mp3", ".flac", ".ogg"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Elastic Split API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str
    status: str = "processing"          # processing | completed | error
    progress: str = "Uploading..."
    stems: list[str] = field(default_factory=list)
    input_path: str = ""
    output_dir: str = ""
    created_at: float = field(default_factory=time.time)
    duration: float = 0.0
    error: Optional[str] = None
    mode: str = "4stems"

jobs: dict[str, Job] = {}
processing_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds using soundfile (WAV/FLAC) or pydub (MP3/OGG)."""
    ext = Path(file_path).suffix.lower()
    try:
        if ext in (".wav", ".flac"):
            info = sf.info(file_path)
            return info.duration
        else:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0
    except Exception:
        return 0.0


def cleanup_job(job_id: str):
    """Remove job files and entry after TTL expires."""
    job = jobs.pop(job_id, None)
    if job:
        job_dir = os.path.join(TMP_DIR, job_id)
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)
        logger.info(f"Cleaned up job {job_id}")


def run_separation(job: Job):
    """Run demucs separation in a background thread."""
    import separator
    start_time = time.time()
    try:
        job.progress = "Separating stems..."
        stem_names = separator.separate(job.input_path, job.output_dir, job.mode)
        job.stems = stem_names
        job.duration = time.time() - start_time
        job.status = "completed"
        job.progress = "Done"
        logger.info(f"Job {job.job_id} completed in {job.duration:.1f}s — stems: {stem_names}")
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        logger.error(f"Job {job.job_id} failed: {e}")
    finally:
        processing_lock.release()
        # Schedule cleanup
        timer = threading.Timer(JOB_TTL_MINUTES * 60, cleanup_job, args=[job.job_id])
        timer.daemon = True
        timer.start()


def convert_to_mp3(wav_path: str) -> io.BytesIO:
    """Convert a WAV file to 320kbps MP3, return as BytesIO."""
    audio = AudioSegment.from_wav(wav_path)
    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate="320k")
    buf.seek(0)
    return buf

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "online", "service": "Elastic Split API", "version": "1.0.0"}


@app.post("/split")
async def split(
    file: UploadFile = File(...),
    mode: str = Form("4stems"),
):
    # Validate mode
    if mode not in ("2stems", "4stems"):
        raise HTTPException(422, "Invalid mode. Use '2stems' or '4stems'.")

    # Validate file extension
    ext = Path(file.filename or "unknown.wav").suffix.lower()
    if ext not in VALID_FORMATS:
        raise HTTPException(400, f"Invalid format '{ext}'. Accepted: WAV, MP3, FLAC, OGG.")

    # Read file and check size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

    # Acquire processing lock (non-blocking)
    if not processing_lock.acquire(blocking=False):
        raise HTTPException(429, "Please wait, another track is being processed.")

    # Create job
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    input_path = os.path.join(job_dir, f"input{ext}")
    output_dir = os.path.join(job_dir, "stems")
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(contents)

    # Check duration
    duration = get_audio_duration(input_path)
    if duration > MAX_DURATION_SECONDS:
        processing_lock.release()
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, f"Audio too long ({duration:.0f}s). Maximum is 10 minutes.")

    # Create job entry
    job = Job(
        job_id=job_id,
        input_path=input_path,
        output_dir=output_dir,
        mode=mode,
        progress="Starting separation...",
    )
    jobs[job_id] = job

    # Start separation in background thread
    thread = threading.Thread(target=run_separation, args=[job], daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found or expired.")

    response = {
        "job_id": job.job_id,
        "status": job.status,
    }

    if job.status == "processing":
        response["progress"] = job.progress
    elif job.status == "completed":
        response["stems"] = job.stems
        response["duration"] = round(job.duration, 1)
    elif job.status == "error":
        response["error"] = job.error

    return response


@app.get("/download/{job_id}/{stem}")
def download(job_id: str, stem: str, format: str = "wav"):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found or expired.")
    if job.status != "completed":
        raise HTTPException(400, "Job not completed yet.")

    # Handle "all" — ZIP download
    if stem == "all":
        return download_all(job, format)

    # Check stem exists
    if stem not in job.stems:
        raise HTTPException(404, f"Stem '{stem}' not found. Available: {job.stems}")

    wav_path = os.path.join(job.output_dir, f"{stem}.wav")
    if not os.path.exists(wav_path):
        raise HTTPException(404, "Stem file not found on disk.")

    if format == "mp3":
        buf = convert_to_mp3(wav_path)
        return StreamingResponse(
            buf,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f'attachment; filename="elastic-split-{stem}.mp3"'},
        )
    else:
        return StreamingResponse(
            open(wav_path, "rb"),
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="elastic-split-{stem}.wav"'},
        )


def download_all(job: Job, format: str = "wav"):
    """Create a ZIP of all stems and stream it."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem_name in job.stems:
            wav_path = os.path.join(job.output_dir, f"{stem_name}.wav")
            if not os.path.exists(wav_path):
                continue
            if format == "mp3":
                mp3_buf = convert_to_mp3(wav_path)
                zf.writestr(f"elastic-split-{stem_name}.mp3", mp3_buf.read())
            else:
                zf.write(wav_path, f"elastic-split-{stem_name}.wav")

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="elastic-split-stems.zip"'},
    )
