# backend/app/voice_pipeline.py

# This file contains the core voice pipeline logic.
# It handles:
#   1. Speech-to-Text (STT) — converting audio to text using Groq's Whisper
#   2. LLM processing — sending text to Claude and getting a response
#   3. Managing conversation history so Claude remembers context

import os
import base64
import io

# httpx is an async HTTP client. We use it to call the Groq API
# because it supports async/await (unlike the standard requests library).
import httpx

# The official Anthropic SDK for calling Claude.
# We use this instead of raw HTTP because it handles
# auth, retries, and streaming properly.
from anthropic import AsyncAnthropic

# python-dotenv loads environment variables from a .env file
# so we don't hardcode API keys in our source code.
from dotenv import load_dotenv

# Load environment variables from backend/.env
load_dotenv()


class VoicePipeline:
    """
    The voice processing pipeline for Sous Chef.

    This class manages three things:
    1. STT (Speech-to-Text): Converts audio → text using Groq's hosted Whisper
    2. LLM (Language Model): Sends text → Claude → gets response
    3. Conversation history: Keeps track of the chat so Claude has context

    In Phase 1, we handle text and audio input.
    In Phase 2, we'll add TTS (Text-to-Speech) for audio output.
    """

    def __init__(self):
        # Initialize the Anthropic client for Claude API calls.
        # AsyncAnthropic is the async version — needed because
        # our FastAPI server is fully async.
        self.anthropic = AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

        # Initialize an async HTTP client for Groq API calls.
        # We reuse this client across requests (connection pooling)
        # instead of creating a new one each time, which is faster.
        self.groq_client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            },
            timeout=30.0,
        )

        # Conversation history stores all messages in the session.
        # Claude needs the full history to understand context.
        # Example: if the user says "what about garlic?" Claude needs
        # to know they were already talking about a pasta recipe.
        self.conversation_history = []

        # System prompt defines Sous Chef's personality and behavior.
        # This is sent with every Claude API call but never shown to the user.
        self.system_prompt = """You are Sous Chef, a voice-first cooking copilot. 
You help home cooks through recipes hands-free while they cook.

Personality:
- Warm, calm, encouraging. Think: a patient friend who is a great cook.
- Keep responses SHORT. 1-2 sentences max unless they ask for detail.
- Use natural cooking language: "give it another minute", 
  "you want a nice golden brown".

What you do:
1. Walk through recipes step by step. One step at a time.
2. Handle substitutions when someone doesn't have an ingredient.
3. Answer cooking questions — technique, temperature, doneness, food safety.
4. Help coordinate timing when cooking multiple dishes.

What you DON'T do:
- Don't give long explanations unless asked.
- Don't list all ingredients upfront unless asked.
- Don't repeat yourself.

Voice considerations (your responses will be spoken aloud):
- Avoid bullet points, markdown, or special characters.
- Use contractions: "You'll want to..." not "You will want to..."
- Say numbers naturally: "three hundred fifty degrees" not "350°F"."""

    async def process_text(self, text: str) -> str:
        """
        Process a text message from the user.

        Flow: user text → Claude LLM → response text

        Args:
            text: The user's message as a string

        Returns:
            The assistant's response as a string
        """
        return await self._call_llm(text)

    async def process_audio(self, audio_base64: str) -> tuple[str, str]:
        """
        Process an audio message from the user.

        Flow: audio → Groq Whisper STT → Claude LLM → response text

        Args:
            audio_base64: Base64-encoded audio data from the browser mic

        Returns:
            A tuple of (user_text, assistant_response)
        """
        # Step 1: Convert audio to text using Groq's Whisper
        user_text = await self._transcribe(audio_base64)

        if not user_text or not user_text.strip():
            return "", "Sorry, I didn't catch that. Could you say it again?"

        # Step 2: Send transcribed text to Claude
        response = await self._call_llm(user_text)

        return user_text, response

    async def _transcribe(self, audio_base64: str) -> str:
        """
        Convert audio to text using Groq's hosted Whisper model.

        Why Groq?
        - Free tier is generous (enough for development)
        - Whisper large-v3-turbo is the same model, just hosted
        - Much faster than running Whisper locally
        - No GPU required on your machine

        The browser sends audio as base64-encoded WAV data.
        We decode it and send it to Groq's transcription endpoint.
        """
        try:
            # Decode base64 audio string back to raw bytes
            audio_bytes = base64.b64decode(audio_base64)

            # Groq's API expects a file upload, so we wrap
            # the bytes in a file-like object with a filename.
            audio_file = ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")

            # Call Groq's Whisper transcription endpoint.
            # This is compatible with OpenAI's API format.
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

    async def _call_llm(self, user_text: str) -> str:
        """
        Send a message to Claude and get a response.

        Why Claude Sonnet?
        - Excellent at tool use (we'll need this for recipe tools in Phase 2)
        - Good balance of speed and intelligence
        - Supports prompt caching (cuts cost 90% on repeated system prompts)

        We maintain a conversation_history list so Claude sees
        the full context of the cooking session.
        """
        # Add the user's message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        try:
            # Call the Claude API with:
            # - system prompt (personality + rules, sent every time)
            # - full conversation history (so Claude has context)
            # - max_tokens limits response length (short for voice)
            response = await self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=self.system_prompt,
                messages=self.conversation_history,
            )

            # Extract the text from Claude's response.
            # response.content is a list of content blocks;
            # for text responses, we take the first block's text.
            assistant_text = response.content[0].text

            # Add Claude's response to history so the next call
            # includes it as context.
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_text,
            })

            return assistant_text

        except Exception as e:
            print(f"LLM error: {e}")
            return "Sorry, I'm having trouble thinking right now. Try again?"

    async def shutdown(self):
        """
        Clean up resources when the WebSocket disconnects.

        We close the HTTP client to release network connections.
        Without this, connections pile up and eventually error out.
        """
        await self.groq_client.aclose()
        print("Voice pipeline resources released")