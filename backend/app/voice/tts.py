import base64
import os

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

VOICE_INSTRUCTIONS = (
    "Speak like a warm, calm cooking companion. Friendly and encouraging, "
    "unhurried but clear. Like a patient friend guiding someone through a recipe."
)


class TextToSpeech:
    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    async def synthesize(self, text: str) -> str | None:
        if not text or not text.strip():
            return None

        try:
            client = self._ensure_client()
            response = await client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=text,
                instructions=VOICE_INSTRUCTIONS,
                response_format="mp3",
            )
            audio_bytes = response.content
            return base64.b64encode(audio_bytes).decode("utf-8")

        except Exception as e:
            print(f"TTS error: {e}")
            return None


_tts = TextToSpeech()


async def synthesize_speech(text: str) -> str | None:
    return await _tts.synthesize(text)