# backend/app/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from .voice_pipeline import VoicePipeline

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    pipeline = VoicePipeline()

    try:
        while True:
            data = await websocket.receive_json()
            message = Message(**data)
            
            if message.role == "user":
                response = await pipeline.process(message.content)
                await websocket.send_json(response.dict())
            else:
                print(f"Received unexpected message: {message}")
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        await pipeline.shutdown()