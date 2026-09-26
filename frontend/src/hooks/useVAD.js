import { useRef, useCallback, useState } from "react";
import { MicVAD } from "@ricky0123/vad-web";

export function useVAD({ onSpeechEnd, onSpeechStart }) {
  const vadRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const pausedRef = useRef(false);

  const start = useCallback(async () => {
    if (vadRef.current) return;

    const vad = await MicVAD.new({
        baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.31/dist/",
        onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/", 
      onSpeechStart: () => {
        if (pausedRef.current) return;
        setIsSpeaking(true);
        onSpeechStart?.();
      },

      onSpeechEnd: (audio) => {
        if (pausedRef.current) return;
        setIsSpeaking(false);
        onSpeechEnd?.(audio);
      },

      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.35,
      minSpeechFrames: 4,
      redemptionFrames: 12,
    });

    vad.start();
    vadRef.current = vad;
    setIsListening(true);
  }, [onSpeechEnd, onSpeechStart]);

  const stop = useCallback(() => {
    if (vadRef.current) {
      vadRef.current.destroy();
      vadRef.current = null;
    }
    setIsListening(false);
    setIsSpeaking(false);
  }, []);

  const pause = useCallback(() => {
    pausedRef.current = true;
    if (vadRef.current) {
      vadRef.current.pause();
    }
  }, []);

  const resume = useCallback(() => {
    pausedRef.current = false;
    if (vadRef.current) {
      vadRef.current.start();
    }
  }, []);

  return { start, stop, pause, resume, isListening, isSpeaking };
}