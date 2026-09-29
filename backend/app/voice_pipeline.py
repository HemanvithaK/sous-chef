import base64
import io
import os

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.agents.cooking_agent import CookingSession, build_agent, is_simple_turn
from app.voice.tts import synthesize_speech

load_dotenv()


class VoicePipeline:
    """
    Owns one cooking session for the lifetime of a WebSocket connection.

    Responsibilities:
      - Speech to text via Groq Whisper
      - Running the LangGraph cooking agent (Sonnet, or Haiku for simple turns)
      - Text to speech via OpenAI
      - Restoring and persisting session state so a reconnect resumes the
        same recipe, step position, and conversation
    """

    def __init__(self, session_id: str | None = None):
        from app.state.store import load_session

        self.session_id = session_id
        restored = load_session(session_id) if session_id else None

        self.session = CookingSession(restored=restored)
        self.agent = build_agent(self.session, model="claude-sonnet-5")
        self.fast_agent = build_agent(
            self.session, model="claude-haiku-4-5-20251001"
        )
        self.message_history = restored["messages"] if restored else []

        if restored:
            recipe = restored.get("current_recipe")
            recipe_name = recipe["name"] if recipe else "no recipe"
            print(
                f"Restored session {session_id}: {recipe_name}, "
                f"step {self.session.current_step}, "
                f"{len(self.message_history)} messages"
            )

        self.groq_client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            timeout=30.0,
        )

    # -- Public entry points -----------------------------------------

    async def process_text(self, text: str) -> dict:
        return await self._run_agent(text)

    async def transcribe_only(self, audio_base64: str) -> str:
        text = await self._transcribe(audio_base64)
        return text.strip() if text else ""

    async def run_agent_only(self, user_text: str) -> dict:
        return await self._run_agent(user_text)

    async def process_audio(self, audio_base64: str) -> dict:
        """Kept for compatibility. The endpoint now splits this into
        transcribe_only + run_agent_only so it can send an instant ack."""
        user_text = await self._transcribe(audio_base64)
        if not user_text or not user_text.strip():
            fallback = "Sorry, I didn't catch that. Could you say it again?"
            audio = await synthesize_speech(fallback)
            return {"user_text": "", "text": fallback, "audio": audio}
        result = await self._run_agent(user_text)
        result["user_text"] = user_text
        return result

    # -- Agent -------------------------------------------------------

    async def _run_agent(self, user_text: str) -> dict:
        self.message_history.append(HumanMessage(content=user_text))

        simple = is_simple_turn(user_text)
        agent = self.fast_agent if simple else self.agent
        print(f"[route] {'HAIKU' if simple else 'SONNET'} <- {user_text[:40]}")

        try:
            result = await agent.ainvoke({"messages": self.message_history})
            ai_message = result["messages"][-1]
            response_text = self._extract_text(ai_message.content)
            self.message_history = result["messages"]
        except Exception as e:
            print(f"Agent error: {e}")
            response_text = "Sorry, I'm having trouble right now. Try again?"

        audio_b64 = await synthesize_speech(response_text)
        self._persist()

        return {"text": response_text, "audio": audio_b64}

    def _extract_text(self, content) -> str:
        """Claude returns a plain string for simple replies, or a list of
        content blocks when tools are involved. Pull out just the text."""
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

    # -- Persistence -------------------------------------------------

    def _persist(self) -> None:
        """Save after every turn, not on disconnect. A crash or a closed
        laptop never fires a clean disconnect, and those are exactly the
        sessions worth keeping."""
        if not self.session_id:
            return
        from app.state.store import save_session

        save_session(
            self.session_id,
            current_recipe=self.session.current_recipe,
            current_step=self.session.current_step,
            active_timers=self.session.active_timers,
            messages=self.message_history,
        )

    # -- Speech to text ----------------------------------------------

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

    # -- Teardown ----------------------------------------------------

    async def shutdown(self):
        self._persist()
        await self.groq_client.aclose()