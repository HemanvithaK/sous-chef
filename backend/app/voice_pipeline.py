# backend/app/voice_pipeline.py

from pipecat.pipeline import Pipeline
from pipecat.processors import (
    VAD_STRIP, VAD_APPEND_FINISH_FRAME, VAD_APPEND_START_FRAME,
    AudioBuffer, VoskSTTProcessor, WhisperSTTProcessor, 
    TalkingHeadsTTSProcessor, BytesToFloatAudio, FloatToInt16Audio,
    OpenAIChatGPTProcessor, AnthropicClaudeProcessor, 
    TextToSpeechProcessor
)

class VoicePipeline:
    def __init__(self):
        self.pipeline = Pipeline(
            VAD_STRIP,
            AudioBuffer(target_time_len_ms=5000),
            VAD_APPEND_START_FRAME,
            VAD_APPEND_FINISH_FRAME,
            BytesToFloatAudio(sample_width=2, channels=1),
            WhisperSTTProcessor(),
            AnthropicClaudeProcessor(
                model="claude-v1",
                system_prompt="You are a helpful AI assistant.",
                temperature=0.7,
            ),
            TextToSpeechProcessor(
                tts_config={
                    "type": "coqui",
                    "model_name": "tts_models/en/vctk/vits",
                },
            ),
            FloatToInt16Audio(),
        )
    
    async def process(self, message: str) -> str:
        frames = [{"type": "text", "text": message}]
        async for res in self.pipeline.process(frames):
            if res.get("type") == "text":
                response_text = res["text"]
                return {"role": "assistant", "content": response_text}
    
    async def shutdown(self):
        await self.pipeline.close()