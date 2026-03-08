from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "your-model-name-here")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))