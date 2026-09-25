import os
import base64
import io

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.agents.cooking_agent import CookingSession, build_agent
from app.voice.tts import synthesize_speech

load_dotenv()


class VoicePipeline:
    def __init__(self):
        self.session = CookingSession()
        self.agent = build_agent(self.session)
        self.message_history = []

        self.groq_client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            },
            timeout=30.0,
        )

    async def process_text(self, text: str) -> dict:
        return await self._run_agent(text)

    async def process_audio(self, audio_base64: str) -> dict:
        user_text = await self._transcribe(audio_base64)
        if not user_text or not user_text.strip():
            fallback = "Sorry, I didn't catch that. Could you say it again?"
            audio = await synthesize_speech(fallback)
            return {
                "user_text": "",
                "text": fallback,
                "audio": audio,
            }
        result = await self._run_agent(user_text)
        result["user_text"] = user_text
        return result

    async def _run_agent(self, user_text: str) -> dict:
        self.message_history.append(HumanMessage(content=user_text))

        try:
            result = await self.agent.ainvoke(
                {"messages": self.message_history}
            )
            ai_message = result["messages"][-1]
            response_text = self._extract_text(ai_message.content)
            self.message_history = result["messages"]

        except Exception as e:
            print(f"Agent error: {e}")
            response_text = "Sorry, I'm having trouble right now. Try again?"

        audio_b64 = await synthesize_speech(response_text)

        return {
            "text": response_text,
            "audio": audio_b64,
        }

    def _extract_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts).strip()
        return str(content)

    async def _transcribe(self, audio_base64: str) -> str:
        try:
            audio_bytes = base64.b64decode(audio_base64)
            audio_file = ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")

            response = await self.groq_client.post(
                "/audio/transcriptions",
                files={"file": audio_file},
                data={
                    "model": "whisper-large-v3-turbo",
                    "response_format": "text",
                    "language": "en",
                },
            )
            response.raise_for_status()
            return response.text.strip()

        except Exception as e:
            print(f"STT error: {e}")
            return ""

    async def shutdown(self):
        await self.groq_client.aclose()