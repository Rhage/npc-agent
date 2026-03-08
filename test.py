import asyncio
import websockets
import json

async def test():
    uri = "ws://127.0.0.1:8000/ws/vtt"

    print("Connecting to server...")
    async with websockets.connect(uri) as websocket:
        print("Connected.\n")

        # Simulate a player message
        message = {
            "type":    "player_message",
            "player":  "Aragorn",
            "profile": "riko",
            "message": "Hello there, do you have any supplies for sale?"
        }

        print(f"Sending: {message['message']}\n")
        await websocket.send(json.dumps(message))

        print("Waiting for DM approval...")
        response = await websocket.recv()
        data = json.loads(response)

        if data["type"] == "agent_response":
            print(f"\nApproved response received:\n{data['response']}")
        elif data["type"] == "error":
            print(f"\nError: {data['detail']}")

asyncio.run(test())