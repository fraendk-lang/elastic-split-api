"""
Demucs v4 wrapper for stem separation.
Lazy-loads the htdemucs model on first request to avoid blocking server startup.
"""

import os
import logging
from pathlib import Path

import torchaudio

logger = logging.getLogger(__name__)

_separator = None


def _get_separator():
    """Lazy-load the demucs model on first use."""
    global _separator
    if _separator is None:
        import demucs.api
        logger.info("Loading htdemucs model (first request, may download ~300MB)...")
        _separator = demucs.api.Separator(model="htdemucs", device="cpu", jobs=2)
        logger.info("Model loaded successfully.")
    return _separator


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

    sep = _get_separator()

    # Always run full 4-stem separation
    origin, separated = sep.separate_audio_file(Path(input_path))

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
