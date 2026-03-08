from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from chat_agent import ChatAgent
from config import HOST, PORT, SSL_CERTFILE, SSL_KEYFILE
import uvicorn
import asyncio
import json

agent = ChatAgent()

class PendingInterception:
    def __init__(self):
        self.player   = None
        self.profile  = None
        self.message  = None
        self.response = None
        self.event    = asyncio.Event()
        self.active   = False

    def reset(self):
        self.player   = None
        self.profile  = None
        self.message  = None
        self.response = None
        self.event.clear()
        self.active   = False

pending = PendingInterception()
passthrough = False
message_queue = asyncio.Queue()

# holds the active WebSocket connections
dm_websocket  = None
vtt_websocket = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(process_queue())
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
        player, profile, message, websocket = await message_queue.get()

        formatted_input = agent.stage_message(player, profile, message)
        approved_profile = profile

        if not passthrough:
            # Step 1 — DM reviews input
            pending.reset()
            pending.player  = player
            pending.profile = profile
            pending.message = formatted_input
            pending.active  = True

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
                profile_name=profile,
                formatted_input=formatted_input
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

        # Send to FoundryVTT
        await websocket.send_text(json.dumps({
            "type":     "agent_response",
            "response": response,
            "profile":  approved_profile,
            "player":   player
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
                # Just enqueue — don't process here
                await message_queue.put((
                    msg["player"],
                    msg["profile"],
                    msg["message"],
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
    
if __name__ == "__main__":
    ssl_args = {}
    if SSL_CERTFILE and SSL_KEYFILE:
        ssl_args = {
            "ssl_certfile": SSL_CERTFILE,
            "ssl_keyfile":  SSL_KEYFILE
        }
    uvicorn.run(app, host=HOST, port=PORT, **ssl_args)