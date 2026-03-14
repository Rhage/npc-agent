"""
tts_engine.py
Wraps FasterQwen3TTS model loading and inference.

Uses the faster-qwen3-tts package which applies CUDA graph capture to bring
generation from ~3.7x slower than realtime down to faster than realtime.

Two models are used:
  - VoiceDesign (1.7B): Used ONCE per character during setup_voices.py.
    Uses the original qwen_tts package (VoiceDesign not in faster-qwen3-tts).
  - FasterQwen3TTS Base: The live inference engine, kept warm in GPU memory.
    Passes ref_audio + ref_text directly — CUDA graphs handle the speedup.

--- CUDA GRAPH NOTE ---
The first generation call after startup triggers CUDA graph capture (~15-20s).
Every call after that runs at full speed. The warmup() method on CloneEngine
handles this on server startup so the first real player request is already fast.

--- ICL CLONING NOTE ---
FasterQwen3TTS uses ICL (In-Context Learning) mode: the reference audio's codec
tokens are fed directly into the transformer context rather than extracting a
static embedding. This means ref_audio and ref_text are passed at generate time,
not pre-cached. The CUDA graph speedup applies regardless.
"""

import os
import torch
import numpy as np
from faster_qwen3_tts import FasterQwen3TTS
from qwen_tts import Qwen3TTSModel as _QwenOriginal  # only for VoiceDesign

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONE_MODEL_LARGE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CLONE_MODEL_SMALL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


class VoiceDesignEngine:
    """
    Wrapper for the VoiceDesign model.
    Only used during setup — not kept in memory during live sessions.
    Uses original qwen_tts since VoiceDesign is not in faster-qwen3-tts.
    """

    def __init__(self, device: str = "cuda:0"):
        print("[VoiceDesign] Loading model... (this may take a moment)")
        self.model = _QwenOriginal.from_pretrained(
            DESIGN_MODEL_ID,
            device_map=device,
            dtype=torch.bfloat16,
        )
        print("[VoiceDesign] Model loaded.")

    def generate_reference(self, text: str, language: str, description: str) -> tuple[np.ndarray, int]:
        """Generate a reference audio clip from a natural language voice description."""
        wavs, sr = self.model.generate_voice_design(
            text=text,
            language=language,
            instruct=description,
        )
        return wavs[0], sr

    def unload(self):
        """Free GPU memory after setup is complete."""
        del self.model
        torch.cuda.empty_cache()
        print("[VoiceDesign] Model unloaded from GPU.")


class CloneEngine:
    """
    The live inference engine using FasterQwen3TTS with CUDA graph capture.

    Characters are registered with their reference WAV path and transcript.
    At generate time, the WAV is passed directly to generate_voice_clone —
    CUDA graphs accelerate the autoregressive decode step regardless.
    """

    def __init__(self, use_large_model: bool = True, device: str = "cuda:0"):
        model_id = CLONE_MODEL_LARGE if use_large_model else CLONE_MODEL_SMALL
        print(f"[CloneEngine] Loading {model_id} via FasterQwen3TTS...")
        self.model = FasterQwen3TTS.from_pretrained(
            model_id,
            device=device,
            dtype=torch.bfloat16,
        )
        # character_id -> {"ref_audio": path, "ref_text": str}
        self._characters: dict[str, dict] = {}
        self._device = device
        print("[CloneEngine] Model ready.")

    def load_character(self, character_id: str, ref_wav_path: str, ref_text: str) -> bool:
        """
        Register a character's reference WAV and transcript.

        Args:
            character_id:  Unique string key for the character.
            ref_wav_path:  Path to the saved reference WAV file.
            ref_text:      Transcript of the reference audio clip.

        Returns:
            True on success, False if the WAV file doesn't exist.
        """
        if not os.path.exists(ref_wav_path):
            print(f"[CloneEngine] WARNING: Reference WAV not found for '{character_id}': {ref_wav_path}")
            return False
        self._characters[character_id] = {
            "ref_audio": ref_wav_path,
            "ref_text": ref_text,
        }
        print(f"[CloneEngine] '{character_id}' registered.")
        return True

    def warmup(self, language: str = "English"):
        """
        Trigger CUDA graph capture with a dummy generation call.
        Must be called after loading characters, before accepting requests.
        The first call is slow (~15-20s) while graphs compile — all subsequent
        calls will be fast.
        """
        if not self._characters:
            print("[CloneEngine] WARNING: No characters loaded, skipping warmup.")
            return

        first_char = next(iter(self._characters))
        char = self._characters[first_char]
        print("[CloneEngine] Running CUDA graph warmup (expect ~15-20s)...")

        _ = self.model.generate_voice_clone(
            text="Warming up.",
            language=language,
            ref_audio=char["ref_audio"],
            ref_text=char["ref_text"],
        )
        print("[CloneEngine] Warmup complete. Subsequent generations will be fast.")

    def generate(self, character_id: str, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        """
        Generate speech for a character using their registered reference audio.

        Args:
            character_id: Must match a key loaded via load_character().
            text:         The dialog line to synthesize.
            language:     Target language.

        Returns:
            Tuple of (wav_array, sample_rate).

        Raises:
            KeyError if the character has no loaded reference.
        """
        if character_id not in self._characters:
            raise KeyError(
                f"No reference loaded for '{character_id}'. "
                "Did setup_voices.py run successfully for this character?"
            )

        char = self._characters[character_id]
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=char["ref_audio"],
            ref_text=char["ref_text"],
        )
        return wavs[0], sr

    def loaded_characters(self) -> list[str]:
        """Return list of character IDs with registered references."""
        return list(self._characters.keys())
