"""
Demucs v4 wrapper for stem separation.
Uses demucs CLI (subprocess) for maximum compatibility.
"""

import os
import subprocess
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("DEMUCS_TIMEOUT_SECONDS", "3600"))
DEMUCS_DEVICE = os.environ.get("DEMUCS_DEVICE", "cpu")
DEMUCS_JOBS = int(os.environ.get("DEMUCS_JOBS", "2"))
DEMUCS_SHIFTS = os.environ.get("DEMUCS_SHIFTS", "1")
# CPU separation often needs ~30–60× realtime; scale timeout with track length.
DEMUCS_CPU_REALTIME_FACTOR = float(os.environ.get("DEMUCS_CPU_REALTIME_FACTOR", "60"))


def separation_timeout(audio_duration_seconds: float) -> int:
    """Minimum 1h, or 60× song length — whichever is larger."""
    scaled = int(max(audio_duration_seconds, 1) * DEMUCS_CPU_REALTIME_FACTOR)
    return max(DEFAULT_TIMEOUT_SECONDS, scaled)


def separate(
    input_path: str,
    output_dir: str,
    mode: str = "4stems",
    *,
    timeout: int | None = None,
    audio_duration_seconds: float = 0,
) -> list[str]:
    """
    Run demucs separation via CLI.
    Blocking — call from a background thread.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build demucs CLI command
    cmd = [
        "python", "-m", "demucs",
        "-n", "htdemucs",
        "--device", DEMUCS_DEVICE,
        "-j", str(DEMUCS_JOBS),
        "--shifts", DEMUCS_SHIFTS,
        "-o", output_dir,
    ]

    if mode == "2stems":
        cmd.extend(["--two-stems", "vocals"])

    cmd.append(input_path)

    run_timeout = timeout if timeout is not None else separation_timeout(audio_duration_seconds)
    logger.info(f"Running demucs (timeout={run_timeout}s): {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=run_timeout,
        )
    except subprocess.TimeoutExpired as e:
        mins = run_timeout // 60
        raise RuntimeError(
            f"Separation timed out after {mins} minutes. "
            "Try a shorter clip or 2-stem mode (vocals/instrumental)."
        ) from e

    if result.returncode != 0:
        logger.error(f"Demucs stderr: {result.stderr}")
        raise RuntimeError(f"Demucs failed: {result.stderr[-500:]}")

    logger.info(f"Demucs stdout: {result.stdout[-200:]}")

    # Demucs outputs to: output_dir/htdemucs/<filename_without_ext>/
    input_name = Path(input_path).stem
    stems_dir = os.path.join(output_dir, "htdemucs", input_name)

    if not os.path.isdir(stems_dir):
        # Try without model subfolder
        stems_dir = os.path.join(output_dir, input_name)
        if not os.path.isdir(stems_dir):
            raise RuntimeError(f"Stems output directory not found. Checked: {output_dir}")

    # Move stems from nested dir to output_dir root
    stem_names = []
    for f in os.listdir(stems_dir):
        if f.endswith(".wav"):
            stem_name = f.replace(".wav", "")
            shutil.move(os.path.join(stems_dir, f), os.path.join(output_dir, f))
            stem_names.append(stem_name)

    # Clean up nested dirs
    htdemucs_dir = os.path.join(output_dir, "htdemucs")
    if os.path.isdir(htdemucs_dir):
        shutil.rmtree(htdemucs_dir, ignore_errors=True)

    logger.info(f"Stems produced: {stem_names}")
    return stem_names
