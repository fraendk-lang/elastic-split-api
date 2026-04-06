"""
Demucs v4 wrapper for stem separation.
Pre-loads the htdemucs model at import time to avoid cold-start on first request.
"""

import os
import logging
from pathlib import Path

import torchaudio
import demucs.api

logger = logging.getLogger(__name__)

# Pre-load model at startup (downloads ~300MB on first run)
logger.info("Loading htdemucs model...")
_separator = demucs.api.Separator(model="htdemucs", device="cpu", jobs=4)
logger.info("Model loaded successfully.")


def separate(input_path: str, output_dir: str, mode: str = "4stems") -> list[str]:
    """
    Run demucs separation on an audio file.
    Blocking — call from a background thread.

    Args:
        input_path: Path to the input audio file
        output_dir: Directory to write output stem WAVs
        mode: "4stems" or "2stems"

    Returns:
        List of stem names produced
    """
    os.makedirs(output_dir, exist_ok=True)

    # Always run full 4-stem separation
    origin, separated = _separator.separate_audio_file(Path(input_path))

    if mode == "2stems":
        # Merge drums + bass + other into "instrumental"
        stems = list(separated.keys())
        vocals_key = None
        instrumental_parts = []

        for key in stems:
            if key == "vocals":
                vocals_key = key
            else:
                instrumental_parts.append(separated[key])

        # Sum non-vocal stems
        instrumental = instrumental_parts[0]
        for part in instrumental_parts[1:]:
            instrumental = instrumental + part

        # Save vocals
        vocals_path = os.path.join(output_dir, "vocals.wav")
        torchaudio.save(vocals_path, separated[vocals_key], sample_rate=44100)

        # Save instrumental
        instrumental_path = os.path.join(output_dir, "instrumental.wav")
        torchaudio.save(instrumental_path, instrumental, sample_rate=44100)

        return ["vocals", "instrumental"]
    else:
        # Save all 4 stems
        stem_names = []
        for stem_name, stem_audio in separated.items():
            output_path = os.path.join(output_dir, f"{stem_name}.wav")
            torchaudio.save(output_path, stem_audio, sample_rate=44100)
            stem_names.append(stem_name)

        return stem_names
