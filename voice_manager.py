"""
voice_manager.py
Owns all TTS orchestration for the NPC Agent server.

Responsibilities:
  - Checking whether a profile has a voice block
  - Generating reference clips lazily (or on demand)
  - Generating OGG audio for a response
  - Returning a Foundry-relative audio path

Keeps server.py clean — all TTS logic lives here.
"""

import os
import asyncio

# TTS dependencies are optional — if not installed, TTS is silently unavailable
try:
    from tts_engine import VoiceDesignEngine, CloneEngine
    from audio_utils import numpy_to_ogg, save_reference_wav
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class VoiceManager:
    def __init__(
        self,
        voices_dir: str,
        foundry_audio_output_path: str,
        foundry_audio_url_prefix: str,
        use_large_model: bool = True,
    ):
        self.voices_dir                 = voices_dir
        self.foundry_audio_output_path  = foundry_audio_output_path
        self.foundry_audio_url_prefix   = foundry_audio_url_prefix
        self.use_large_model            = use_large_model

        self._clone_engine: "CloneEngine | None" = None
        self._file_counter  = 0
        self._engine_lock   = asyncio.Lock()  # prevents concurrent engine init

        os.makedirs(voices_dir, exist_ok=True)
        os.makedirs(foundry_audio_output_path, exist_ok=True)

    # ── Public API ──
    async def warmup(self):
        """
        Pre-load the CloneEngine and trigger CUDA graph capture on startup.
        Eliminates the warmup cost from the first real player request.
        Does nothing if TTS is unavailable or no voices directory exists.
        """
        if not TTS_AVAILABLE:
            return
        if not self.foundry_audio_output_path:  # ← use self. instead of the config constant
            return

        print("[VoiceManager] Starting TTS warmup...")
        engine = await self._get_engine()
        if engine is None:
            return

        # Find any available reference WAV to use for warmup
        wavs = [
            f for f in os.listdir(self.voices_dir)
            if f.endswith(".wav")
        ] if os.path.exists(self.voices_dir) else []

        if not wavs:
            print("[VoiceManager] No reference clips found — skipping CUDA graph warmup.")
            print("[VoiceManager]   Run 'Prep Voice' from the DM console to generate one.")
            return

        # Register the first available character temporarily for warmup
        char_id  = wavs[0].replace(".wav", "")
        ref_text = self._load_ref_text(char_id)

        if char_id not in engine.loaded_characters():
            engine.load_character(char_id, os.path.join(self.voices_dir, wavs[0]), ref_text)

        print("[VoiceManager] Running CUDA graph warmup (expect ~15-20s)...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: engine.warmup())
        print("[VoiceManager] TTS warmup complete.")

    @property
    def available(self) -> bool:
        """True if TTS dependencies are installed."""
        return TTS_AVAILABLE

    def profile_has_voice(self, profile: dict) -> bool:
        """True if the profile JSON contains a voice block."""
        return bool(profile.get("voice"))

    def reference_exists(self, profile_name: str) -> bool:
        """True if a reference WAV already exists for this profile."""
        return os.path.exists(self._ref_wav_path(profile_name))

    async def prepare(self, profile_name: str, profile: dict) -> bool:
        """
        Ensure a reference clip exists for this profile.
        Generates one if missing. Safe to call multiple times — never overwrites.

        Returns True if ready, False if generation failed or no voice block.
        """
        if not TTS_AVAILABLE:
            return False
        if not self.profile_has_voice(profile):
            return False
        if self.reference_exists(profile_name):
            print(f"[VoiceManager] Reference clip already exists for '{profile_name}' — skipping.")
            return True

        return await self._generate_reference(profile_name, profile)

    async def generate_audio(self, profile_name: str, profile: dict, text: str) -> "str | None":
        """
        Generate a voice line for the given profile and text.

        Ensures a reference clip exists first (lazy generation).
        Returns a Foundry-relative audio path on success, None on failure.
        """
        if not TTS_AVAILABLE:
            return None
        if not self.profile_has_voice(profile):
            return None

        # Ensure reference clip exists
        ready = await self.prepare(profile_name, profile)
        if not ready:
            print(f"[VoiceManager] Cannot generate audio — no reference clip for '{profile_name}'.")
            return None

        # Ensure clone engine is loaded
        engine = await self._get_engine()
        if engine is None:
            return None

        # Register character if not already loaded
        if profile_name not in engine.loaded_characters():
            ref_text = self._load_ref_text(profile_name)
            success  = engine.load_character(
                profile_name,
                self._ref_wav_path(profile_name),
                ref_text
            )
            if not success:
                print(f"[VoiceManager] Failed to load character '{profile_name}' into engine.")
                return None

        # Generate
        voice_cfg  = profile["voice"]
        language   = voice_cfg.get("language", "English")

        try:
            loop = asyncio.get_event_loop()
            wav_array, sr = await loop.run_in_executor(
                None,
                lambda: engine.generate(profile_name, text, language)
            )
        except Exception as e:
            print(f"[VoiceManager] TTS generation failed for '{profile_name}': {e}")
            return None

        # Save OGG
        filename   = self._next_filename(profile_name)
        output_path = os.path.join(self.foundry_audio_output_path, filename)
        foundry_src = f"{self.foundry_audio_url_prefix}/{filename}"

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: numpy_to_ogg(wav_array, sr, output_path)
            )
        except Exception as e:
            print(f"[VoiceManager] OGG conversion failed for '{profile_name}': {e}")
            return None

        print(f"[VoiceManager] Generated audio for '{profile_name}': {foundry_src}")
        return foundry_src

    # ── Private helpers ──

    def _ref_wav_path(self, profile_name: str) -> str:
        return os.path.join(self.voices_dir, f"{profile_name}.wav")

    def _ref_text_path(self, profile_name: str) -> str:
        return os.path.join(self.voices_dir, f"{profile_name}.txt")

    def _load_ref_text(self, profile_name: str) -> str:
        path = self._ref_text_path(profile_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read().strip()
        return ""

    def _save_ref_text(self, profile_name: str, text: str):
        with open(self._ref_text_path(profile_name), "w") as f:
            f.write(text)

    def _next_filename(self, profile_name: str) -> str:
        self._file_counter += 1
        return f"{profile_name}_{self._file_counter:04d}.ogg"

    async def _get_engine(self) -> "CloneEngine | None":
        """Lazily initialise the clone engine, thread-safe."""
        if self._clone_engine is not None:
            return self._clone_engine

        async with self._engine_lock:
            # Double-check after acquiring lock
            if self._clone_engine is not None:
                return self._clone_engine

            print("[VoiceManager] Loading CloneEngine...")
            try:
                loop = asyncio.get_event_loop()
                engine = await loop.run_in_executor(
                    None,
                    lambda: CloneEngine(use_large_model=self.use_large_model)
                )
                self._clone_engine = engine
                print("[VoiceManager] CloneEngine ready.")
            except Exception as e:
                print(f"[VoiceManager] Failed to load CloneEngine: {e}")
                return None

        return self._clone_engine

    async def _generate_reference(self, profile_name: str, profile: dict) -> bool:
        """Generate a reference WAV for the profile using VoiceDesign."""
        voice_cfg   = profile["voice"]
        description = voice_cfg.get("description", "")
        language    = voice_cfg.get("language", "English")
        ref_phrase  = voice_cfg.get(
            "ref_phrase",
            "I've been waiting for someone like you. Let's see what you're made of."
        )

        print(f"[VoiceManager] Generating reference clip for '{profile_name}'...")

        try:
            loop = asyncio.get_event_loop()

            def _run():
                design_engine = VoiceDesignEngine()
                wav_array, sr = design_engine.generate_reference(
                    text=ref_phrase,
                    language=language,
                    description=description,
                )
                design_engine.unload()
                return wav_array, sr

            wav_array, sr = await loop.run_in_executor(None, _run)
            save_reference_wav(wav_array, sr, self._ref_wav_path(profile_name))
            self._save_ref_text(profile_name, ref_phrase)
            print(f"[VoiceManager] Reference clip saved for '{profile_name}'.")
            return True

        except Exception as e:
            print(f"[VoiceManager] Reference generation failed for '{profile_name}': {e}")
            return False
