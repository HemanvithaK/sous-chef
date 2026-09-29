

# FastAPI is a modern Python web framework. We use it because:
# 1. It supports WebSockets natively (needed for real-time voice)
# 2. It's async by default (needed for handling audio streams)
# 3. It auto-generates API docs at /docs

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# We import our voice pipeline 
from app.voice_pipeline import VoicePipeline
from app.voice.tts import prewarm_fillers

# Create the FastAPI application instance
app = FastAPI(
    title="Sous Chef - Voice Cooking Copilot",
    version="0.1.0",
)
@app.on_event("startup")
async def startup():
    try:
        await prewarm_fillers()
    except Exception as e:
        print(f"Filler prewarm failed (non-fatal): {e}")

# CORS middleware allows the frontend (running on port 3000)
# to talk to the backend (running on port 8000).
# Without this, the browser blocks cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple health check endpoint.
# Hit http://localhost:8000/health to verify the server is running.
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# This is the main WebSocket endpoint.
# The frontend connects here to send/receive voice data.

# WebSocket vs HTTP:
# - HTTP is request-response (client asks, server answers, connection closes)
# - WebSocket stays open — both sides can send data anytime
# - We need this because voice is a continuous stream, not a single request
@app.websocket("/ws/voice")
async def voice_endpoint(ws: WebSocket, session_id: str | None = None):
    await ws.accept()
    print(f"Client connected (session: {session_id or 'new'})")

    pipeline = VoicePipeline(session_id=session_id)

    async def send_event(payload: dict):
        try:
            await ws.send_json(payload)
        except Exception as e:
            print(f"Failed to send event: {e}")

    pipeline.set_event_sender(send_event)

    if pipeline.message_history:
        history = []
        for msg in pipeline.message_history:
            role = "user" if msg.type == "human" else "assistant"
            text = msg.content if isinstance(msg.content, str) else ""
            if text.strip():
                history.append({"role": role, "text": text})
        await ws.send_json({"type": "history", "messages": history})

    try:
        # Step 3: Enter an infinite loop to continuously receive messages.
        # This loop runs until the client disconnects.
        while True:

            # Wait for a message from the browser.
            # The browser sends JSON like:
            #   {"type": "text", "content": "let's make pasta"}
            #   {"type": "audio", "data": "<base64 audio>"}
            data = await ws.receive_json()

            msg_type = data.get("type", "")

            if msg_type == "text":
                user_text = data.get("content", "")
                print(f"User said: {user_text}")
                try:
                    result = await pipeline.process_text(user_text)
                except Exception as e:
                    print(f"process_text failed: {e}")
                    result = {"text": "Something went wrong. Try again?", "audio": None}

                await ws.send_json({
                    "type": "transcript",
                    "role": "assistant",
                    "text": result["text"],
                })
                if result.get("audio"):
                    await ws.send_json({"type": "audio", "data": result["audio"]})

            elif msg_type == "audio":
                audio_base64 = data.get("data", "")

                user_text = await pipeline.transcribe_only(audio_base64)
                if not user_text:
                    continue

                await ws.send_json({
                    "type": "transcript",
                    "role": "user",
                    "text": user_text,
                })

                from app.voice.fillers import random_ack
                from app.voice.tts import synthesize_cached

                ack_audio = await synthesize_cached(random_ack())
                if ack_audio:
                    await ws.send_json({"type": "audio", "data": ack_audio})

                try:
                    result = await pipeline.run_agent_only(user_text)
                except Exception as e:
                    print(f"Agent failed: {e}")
                    result = {"text": "Something went wrong. Try again?", "audio": None}

                await ws.send_json({
                    "type": "transcript",
                    "role": "assistant",
                    "text": result["text"],
                })
                if result.get("audio"):
                    await ws.send_json({"type": "audio", "data": result["audio"]})

    except WebSocketDisconnect:
        # This fires when the browser tab closes or user disconnects.
        # We clean up the pipeline resources.
        print("Client disconnected")

    except Exception as e:
        # Catch any unexpected errors so the server doesn't crash
        print(f"Error in voice endpoint: {e}")

    finally:
        # Always clean up, whether we exit normally or via error
        await pipeline.shutdown()
        print("Pipeline shut down")