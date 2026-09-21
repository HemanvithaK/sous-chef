# backend/app/main.py

# FastAPI is a modern Python web framework. We use it because:
# 1. It supports WebSockets natively (needed for real-time voice)
# 2. It's async by default (needed for handling audio streams)
# 3. It auto-generates API docs at /docs

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# We import our voice pipeline (we'll create this next)
from app.voice_pipeline import VoicePipeline

# Create the FastAPI application instance
app = FastAPI(
    title="Sous Chef - Voice Cooking Copilot",
    version="0.1.0",
)

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
#
# WebSocket vs HTTP:
# - HTTP is request-response (client asks, server answers, connection closes)
# - WebSocket stays open — both sides can send data anytime
# - We need this because voice is a continuous stream, not a single request
@app.websocket("/ws/voice")
async def voice_endpoint(ws: WebSocket):

    # Step 1: Accept the incoming WebSocket connection from the browser
    await ws.accept()
    print("Client connected")

    # Step 2: Create a new voice pipeline for this session.
    # Each connected user gets their own pipeline instance
    # so conversations don't mix.
    pipeline = VoicePipeline()

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
                # User typed a text message (for testing without mic)
                user_text = data.get("content", "")
                print(f"User said: {user_text}")

                # Process through our pipeline:
                # text → Claude LLM → response text
                response = await pipeline.process_text(user_text)

                # Send the response back to the browser
                await ws.send_json({
                    "type": "transcript",
                    "role": "assistant",
                    "text": response,
                })

            elif msg_type == "audio":
                # User sent audio from their microphone
                audio_base64 = data.get("data", "")

                # Process through our full pipeline:
                # audio → Whisper STT → Claude LLM → response text
                user_text, response = await pipeline.process_audio(audio_base64)

                # Send the user's transcription back
                await ws.send_json({
                    "type": "transcript",
                    "role": "user",
                    "text": user_text,
                })

                # Send the assistant's response back
                await ws.send_json({
                    "type": "transcript",
                    "role": "assistant",
                    "text": response,
                })

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