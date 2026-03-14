from dotenv import load_dotenv
import os

load_dotenv()

# ── LLM ──
MODEL_NAME    = os.getenv("MODEL_NAME",    "your-model-name-here")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234")

# ── Server ──
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# ── TTS ──
# Path to the audio output folder INSIDE your FoundryVTT module directory.
# FoundryVTT must be able to serve files from this path.
# Example (Windows): C:/Users/YOU/AppData/Local/FoundryVTT/Data/modules/npc-agent/audio
# Example (Linux):   /home/YOU/foundrydata/Data/modules/npc-agent/audio
FOUNDRY_AUDIO_OUTPUT_PATH = os.getenv("FOUNDRY_AUDIO_OUTPUT_PATH", "")

# The path prefix FoundryVTT uses to serve files from the module.
# Should match: modules/<your-module-name>/audio
FOUNDRY_AUDIO_URL_PREFIX = os.getenv("FOUNDRY_AUDIO_URL_PREFIX", "modules/npc-agent/audio")

# Directory to store reference WAV files alongside the server
VOICES_DIR = os.getenv("VOICES_DIR", "voices")

# True = 1.7B model (better quality), False = 0.6B (faster)
TTS_USE_LARGE_MODEL = os.getenv("TTS_USE_LARGE_MODEL", "true").lower() == "true"
