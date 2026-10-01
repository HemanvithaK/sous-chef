"""Sous Chef -- FastAPI entry point."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from app.voice_pipeline import VoicePipeline
from app.voice.fillers import random_ack
from app.voice.tts import prewarm_fillers, synthesize_cached

app = FastAPI(
    title="Sous Chef -- Voice Cooking Copilot",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Generate the short acknowledgment phrases once at boot so the first
    user turn doesn't pay for them. Non-fatal if it fails."""
    try:
        await prewarm_fillers()
    except Exception as e:
        print(f"Filler prewarm failed (non-fatal): {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


def build_history(pipeline: VoicePipeline) -> list[dict]:
    """Turn the stored message history into something safe to show the user.

    Only HumanMessage and AIMessage become bubbles. ToolMessage holds raw
    JSON tool results -- replaying those would leak internals into the
    transcript, which is exactly what the live path already guards against.
    """
    history = []
    for msg in pipeline.message_history:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            continue

        text = msg.content if isinstance(msg.content, str) else ""
        if not text:
            text = pipeline._extract_text(msg.content)
        if text.strip():
            history.append({"role": role, "text": text})
    return history


@app.websocket("/ws/voice")
async def voice_endpoint(ws: WebSocket, session_id: str | None = None):
    await ws.accept()
    print(f"Client connected (session: {session_id or 'new'})")

    pipeline = VoicePipeline(session_id=session_id)

    async def send_event(payload: dict):
        """Handed to the pipeline so timers can speak without the pipeline
        ever holding the socket itself."""
        try:
            await ws.send_json(payload)
        except Exception as e:
            print(f"Failed to send event: {e}")

    pipeline.set_event_sender(send_event)

    # Replay the conversation so a reconnect picks up where it left off.
    history = build_history(pipeline)
    if history:
        await ws.send_json({"type": "history", "messages": history})

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "text":
                user_text = data.get("content", "")
                print(f"User typed: {user_text}")

                try:
                    result = await pipeline.process_text(user_text)
                except Exception as e:
                    print(f"process_text failed: {e}")
                    result = {
                        "text": "Something went wrong. Try again?",
                        "audio": None,
                    }

                await ws.send_json({
                    "type": "transcript",
                    "role": "assistant",
                    "text": result["text"],
                })
                if result.get("audio"):
                    await ws.send_json({
                        "type": "audio",
                        "data": result["audio"],
                    })

            elif msg_type == "audio":
                audio_base64 = data.get("data", "")

                # Transcribe first so the user hears something within a few
                # hundred ms, rather than sitting in silence while the agent
                # thinks and the web search runs.
                user_text = await pipeline.transcribe_only(audio_base64)
                if not user_text:
                    continue

                print(f"User said: {user_text}")

                await ws.send_json({
                    "type": "transcript",
                    "role": "user",
                    "text": user_text,
                })

                ack_audio = await synthesize_cached(random_ack())
                if ack_audio:
                    await ws.send_json({"type": "audio", "data": ack_audio})

                try:
                    result = await pipeline.run_agent_only(user_text)
                except Exception as e:
                    print(f"run_agent_only failed: {e}")
                    result = {
                        "text": "Something went wrong. Try again?",
                        "audio": None,
                    }

                await ws.send_json({
                    "type": "transcript",
                    "role": "assistant",
                    "text": result["text"],
                })
                if result.get("audio"):
                    await ws.send_json({
                        "type": "audio",
                        "data": result["audio"],
                    })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in voice endpoint: {e}")
    finally:
        await pipeline.shutdown()
        print("Pipeline shut down")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )