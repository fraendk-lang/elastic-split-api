"""
Demucs v4 wrapper for stem separation.
All heavy imports (torch, demucs) are lazy to keep server startup fast and light.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_separator = None


def _get_separator():
    """Lazy-load the demucs model on first use."""
    global _separator
    if _separator is None:
        import demucs.api
        logger.info("Loading htdemucs model (first request, may download ~300MB)...")
        _separator = demucs.api.Separator(model="htdemucs", device="cpu", jobs=1)
        logger.info("Model loaded successfully.")
    return _separator


def separate(input_path: str, output_dir: str, mode: str = "4stems") -> list[str]:
    """
    Run demucs separation on an audio file.
    Blocking — call from a background thread.
    """
    import torchaudio

    os.makedirs(output_dir, exist_ok=True)

    sep = _get_separator()
    origin, separated = sep.separate_audio_file(Path(input_path))

    if mode == "2stems":
        stems = list(separated.keys())
        vocals_key = None
        instrumental_parts = []

        for key in stems:
            if key == "vocals":
                vocals_key = key
            else:
                instrumental_parts.append(separated[key])

        instrumental = instrumental_parts[0]
        for part in instrumental_parts[1:]:
            instrumental = instrumental + part

        torchaudio.save(os.path.join(output_dir, "vocals.wav"), separated[vocals_key], sample_rate=44100)
        torchaudio.save(os.path.join(output_dir, "instrumental.wav"), instrumental, sample_rate=44100)

        return ["vocals", "instrumental"]
    else:
        stem_names = []
        for stem_name, stem_audio in separated.items():
            torchaudio.save(os.path.join(output_dir, f"{stem_name}.wav"), stem_audio, sample_rate=44100)
            stem_names.append(stem_name)
        return stem_names
