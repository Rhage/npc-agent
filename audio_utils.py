"""
audio_utils.py
Handles conversion of numpy audio arrays (from Qwen3-TTS) to OGG files.
Requires ffmpeg to be installed. 

On Windows, ffmpeg is often not on PATH automatically. Install via:
    winget install ffmpeg
    -- or --
    conda install -c conda-forge ffmpeg

If neither works, download ffmpeg manually from https://ffmpeg.org/download.html,
extract it, and set FFMPEG_PATH below to the full path of ffmpeg.exe.
"""

import io
import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import which

# ---------------------------------------------------------------------------
# If ffmpeg is not on your PATH, set this to the full path of ffmpeg.exe.
# Example: FFMPEG_PATH = r"C:\tools\ffmpeg\bin\ffmpeg.exe"
# Leave as None to use PATH auto-detection.
# ---------------------------------------------------------------------------
FFMPEG_PATH = None

def _configure_ffmpeg():
    """Point pydub at ffmpeg, either from PATH or FFMPEG_PATH override."""
    import pydub
    pydub.AudioSegment.sox_path = None  # suppress SoX not found warning
    
    if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):
        pydub.AudioSegment.converter = FFMPEG_PATH
        print(f"[audio_utils] Using ffmpeg at: {FFMPEG_PATH}")
    elif which("ffmpeg"):
        pass  # Already on PATH, pydub will find it automatically
    else:
        raise EnvironmentError(
            "ffmpeg not found on PATH and FFMPEG_PATH is not set.\n"
            "Install ffmpeg (winget install ffmpeg) or set FFMPEG_PATH "
            "in audio_utils.py to the full path of ffmpeg.exe."
        )

# Run once on import
_configure_ffmpeg()


def numpy_to_ogg(wav_array: np.ndarray, sample_rate: int, output_path: str) -> str:
    """
    Convert a numpy audio array to an OGG file and save it to output_path.

    Args:
        wav_array:    1D numpy array of audio samples from Qwen3-TTS.
        sample_rate:  Sample rate returned by the model (typically 24000).
        output_path:  Full path where the .ogg file should be written.

    Returns:
        The output_path on success.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Write numpy array to an in-memory WAV buffer
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, wav_array, sample_rate, format="WAV")
    wav_buffer.seek(0)

    # Step 2: Load WAV buffer into pydub
    audio_segment = AudioSegment.from_wav(wav_buffer)

    # Step 3: Export as OGG via ffmpeg
    # Quality 4 (~128kbps equivalent) is more than sufficient for voice.
    audio_segment.export(output_path, format="ogg", parameters=["-q:a", "4"])

    return output_path


def save_reference_wav(wav_array: np.ndarray, sample_rate: int, output_path: str) -> str:
    """
    Save a numpy audio array as a WAV file (used for voice reference clips).
    We keep reference clips as WAV for lossless quality when cloning.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, wav_array, sample_rate)
    return output_path
