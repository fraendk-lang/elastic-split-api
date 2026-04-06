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


def separate(input_path: str, output_dir: str, mode: str = "4stems") -> list[str]:
    """
    Run demucs separation via CLI.
    Blocking — call from a background thread.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build demucs CLI command
    cmd = [
        "python", "-m", "demucs",
        "-n", "htdemucs",
        "--device", "cpu",
        "-o", output_dir,
    ]

    if mode == "2stems":
        cmd.extend(["--two-stems", "vocals"])

    cmd.append(input_path)

    logger.info(f"Running demucs: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max
    )

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
