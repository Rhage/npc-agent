from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from chat_agent import ChatAgent
from profiles import load_profile
from voice_manager import VoiceManager
from knowledge_manager import KnowledgeManager
from config import (
    HOST, PORT,
    FOUNDRY_AUDIO_OUTPUT_PATH,
    FOUNDRY_AUDIO_URL_PREFIX,
    VOICES_DIR,
    TTS_USE_LARGE_MODEL,
    EMBEDDING_MODEL,
)
import uvicorn
import asyncio
import json

# KnowledgeManager loads the embedding model — created before ChatAgent
# so it can be passed in at construction time.
knowledge_mgr = KnowledgeManager(EMBEDDING_MODEL)
agent         = ChatAgent(knowledge_mgr=knowledge_mgr)

voice_manager = VoiceManager(
    voices_dir                = VOICES_DIR,
    foundry_audio_output_path = FOUNDRY_AUDIO_OUTPUT_PATH,
    foundry_audio_url_prefix  = FOUNDRY_AUDIO_URL_PREFIX,
    use_large_model           = TTS_USE_LARGE_MODEL,
)

if not voice_manager.available:
    print("[Server] TTS dependencies not found — voice generation disabled.")
elif not FOUNDRY_AUDIO_OUTPUT_PATH:
    print("[Server] FOUNDRY_AUDIO_OUTPUT_PATH not set in .env — voice generation disabled.")

class PendingInterception:
    def __init__(self):
        self.player         = None
        self.profile        = None
        self.message        = None
        self.response       = None
        self.user_id        = None
        self.visible_actors = None
        self.event          = asyncio.Event()
        self.active         = False

    def reset(self):
        self.player         = None
        self.profile        = None
        self.message        = None
        self.response       = None
        self.user_id        = None
        self.visible_actors = None
        self.event.clear()
        self.active         = False

pending       = PendingInterception()
passthrough   = False
tts_enabled   = True
message_queue = asyncio.Queue()

# holds the active WebSocket connections
dm_websocket  = None
vtt_websocket = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(process_queue())
    if tts_enabled and voice_manager.available and FOUNDRY_AUDIO_OUTPUT_PATH:
        asyncio.create_task(voice_manager.warmup())
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    player:  str
    profile: str
    message: str

@app.get("/dm")
async def dm_console():
    return FileResponse("static/dm.html")

# ── WebSocket: DM Console ──
@app.websocket("/ws/dm")
async def ws_dm(websocket: WebSocket):
    global dm_websocket
    await websocket.accept()
    dm_websocket = websocket
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg["type"] == "approved_response":
                # DM has approved — store the (possibly edited) response
                # and signal FoundryVTT that it can proceed
                pending.response = msg["response"]
                pending.event.set()

    except WebSocketDisconnect:
        dm_websocket = None

# ── WebSocket: FoundryVTT ──
async def process_queue():
    while True:
        player, profile, message, user_id, visible_actors, websocket = await message_queue.get()

        formatted_input  = agent.stage_message(player, profile, message)
        approved_profile = profile

        if not passthrough:
            # Step 1 — DM reviews input
            pending.reset()
            pending.player         = player
            pending.profile        = profile
            pending.message        = formatted_input
            pending.user_id        = user_id
            pending.visible_actors = visible_actors
            pending.active         = True

            # Forward to DM console for review
            if dm_websocket:
                await dm_websocket.send_text(json.dumps({
                    "type":    "incoming_message",
                    "player":  player,
                    "profile": profile,
                    "message": formatted_input
                }))

            await pending.event.wait()
            formatted_input = pending.message

        # Call LLM with approved (or unmodified) input
        try:
            response = agent.complete(
                profile_name    = profile,
                formatted_input = formatted_input,
                speaker         = player,
                visible_actors  = visible_actors or []
            )
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type":   "error",
                "detail": str(e)
            }))
            message_queue.task_done()
            if not passthrough:
                pending.reset()
            continue

        if not passthrough:
            # Step 2 — DM reviews response
            pending.reset()
            pending.response = response
            pending.active   = True

            if dm_websocket:
                await dm_websocket.send_text(json.dumps({
                    "type":     "llm_response",
                    "response": response
                }))

            await pending.event.wait()
            response = pending.response
            pending.reset()

        # Generate TTS in parallel with building the response payload
        audio_src = None
        if tts_enabled and voice_manager.available and FOUNDRY_AUDIO_OUTPUT_PATH:
            try:
                npc_profile = load_profile(profile)
                if voice_manager.profile_has_voice(npc_profile):
                    audio_src = await voice_manager.generate_audio(
                        profile_name = profile,
                        profile      = npc_profile,
                        text         = response
                    )
            except Exception as e:
                print(f"[Server] TTS error for '{profile}': {e}")

        # Send to FoundryVTT
        await websocket.send_text(json.dumps({
            "type":      "agent_response",
            "response":  response,
            "profile":   approved_profile,
            "player":    player,
            "userId":    user_id,
            "audio_src": audio_src,
        }))

        message_queue.task_done()

@app.websocket("/ws/vtt")
async def ws_vtt(websocket: WebSocket):
    global vtt_websocket
    await websocket.accept()
    vtt_websocket = websocket
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg["type"] == "player_message":
                await message_queue.put((
                    msg["player"],
                    msg["profile"],
                    msg["message"],
                    msg.get("userId"),
                    msg.get("visibleActors", []),
                    websocket
                ))

    except WebSocketDisconnect:
        vtt_websocket = None
    
# ── HTTP endpoints ──
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        formatted_input = agent.stage_message(
            player=request.player,
            profile_name=request.profile,
            message=request.message
        )
        response = agent.complete(
            profile_name=request.profile,
            formatted_input=formatted_input
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/message")
async def message(request: ChatRequest):
    try:
        response = agent.respond_once(
            player=request.player,
            profile_name=request.profile,
            message=request.message
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/passthrough")
async def set_passthrough(enabled: bool):
    global passthrough
    passthrough = enabled
    return {"passthrough": passthrough}


@app.post("/tts")
async def set_tts(enabled: bool):
    global tts_enabled
    tts_enabled = enabled
    return {"tts_enabled": tts_enabled}


@app.post("/voice/prepare/{profile_name}")
async def prepare_voice(profile_name: str):
    if not voice_manager.available:
        raise HTTPException(status_code=503, detail="TTS dependencies not installed.")
    if not FOUNDRY_AUDIO_OUTPUT_PATH:
        raise HTTPException(status_code=503, detail="FOUNDRY_AUDIO_OUTPUT_PATH not configured.")

    try:
        profile = load_profile(profile_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found.")

    if not voice_manager.profile_has_voice(profile):
        raise HTTPException(status_code=400, detail=f"Profile '{profile_name}' has no voice block.")

    already_existed = voice_manager.reference_exists(profile_name)
    success         = await voice_manager.prepare(profile_name, profile)

    if not success:
        raise HTTPException(status_code=500, detail=f"Reference clip generation failed for '{profile_name}'.")

    return {
        "profile":         profile_name,
        "already_existed": already_existed,
        "ready":           True,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)