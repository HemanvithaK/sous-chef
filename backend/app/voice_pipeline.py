import os
import base64
import io

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.agents.cooking_agent import CookingSession, build_agent

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

    async def process_text(self, text: str) -> str:
        return await self._run_agent(text)

    async def process_audio(self, audio_base64: str) -> tuple[str, str]:
        user_text = await self._transcribe(audio_base64)
        if not user_text or not user_text.strip():
            return "", "Sorry, I didn't catch that. Could you say it again?"
        response = await self._run_agent(user_text)
        return user_text, response

    async def _run_agent(self, user_text: str) -> str:
        self.message_history.append(HumanMessage(content=user_text))

        try:
            result = await self.agent.ainvoke(
                {"messages": self.message_history}
            )
            ai_message = result["messages"][-1]
            response_text = ai_message.content

            self.message_history = result["messages"]

            return response_text

        except Exception as e:
            print(f"Agent error: {e}")
            return "Sorry, I'm having trouble right now. Try again?"

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