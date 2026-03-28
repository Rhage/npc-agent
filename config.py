from dotenv import load_dotenv
import os

load_dotenv()

# ── LLM ──
MODEL_NAME    = os.getenv("MODEL_NAME",    "your-model-name-here")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234")

# ── Server ──
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# ── Database ──
DB_PATH      = os.getenv("DB_PATH",      "npc_agent.db")
PROFILES_DIR = os.getenv("PROFILES_DIR", "profiles")   # legacy JSON dir, used by migrate_profiles.py

# ── Knowledge / Embeddings ──
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "multi-qa-mpnet-base-dot-v1")

# ── TTS ──
FOUNDRY_AUDIO_OUTPUT_PATH = os.getenv("FOUNDRY_AUDIO_OUTPUT_PATH", "")
FOUNDRY_AUDIO_URL_PREFIX  = os.getenv("FOUNDRY_AUDIO_URL_PREFIX",  "modules/npc-agent/audio")
VOICES_DIR                = os.getenv("VOICES_DIR",                 "voices")
TTS_USE_LARGE_MODEL       = os.getenv("TTS_USE_LARGE_MODEL", "true").lower() == "true"
