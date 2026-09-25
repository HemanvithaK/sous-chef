import React, { useState, useRef, useEffect, useCallback } from "react";
import { Mic, MicOff, Send, ChefHat, Wifi, WifiOff } from "lucide-react";

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState([]);
  const [textInput, setTextInput] = useState("");
  const [error, setError] = useState(null);

  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const transcriptEndRef = useRef(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript]);

  // ── Play TTS audio from the backend ────────────────
  const playAudio = useCallback((base64Mp3) => {
    try {
      const byteChars = atob(base64Mp3);
      const byteNumbers = new Array(byteChars.length);
      for (let i = 0; i < byteChars.length; i++) {
        byteNumbers[i] = byteChars.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: "audio/mpeg" });
      const url = URL.createObjectURL(blob);

      const audio = new Audio(url);
      audio.onended = () => URL.revokeObjectURL(url);
      audio.play().catch((e) => console.error("Audio play failed:", e));
    } catch (e) {
      console.error("Audio decode failed:", e);
    }
  }, []);

  // ── WebSocket connection ───────────────────────────
  const connect = useCallback(() => {
    const ws = new WebSocket(`ws://${window.location.host}/ws/voice`);

    ws.onopen = () => {
      console.log("WebSocket connected");
      setIsConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "transcript") {
        setTranscript((prev) => [
          ...prev,
          { role: data.role, text: data.text, ts: Date.now() },
        ]);
      } else if (data.type === "audio") {
        playAudio(data.data);
      }
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
    };

    ws.onerror = () => {
      setError("Cannot connect to server. Is the backend running?");
      setIsConnected(false);
    };

    wsRef.current = ws;
  }, [playAudio]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  // ── Send text message ──────────────────────────────
  const sendText = useCallback(() => {
    if (!textInput.trim() || !wsRef.current) return;

    setTranscript((prev) => [
      ...prev,
      { role: "user", text: textInput, ts: Date.now() },
    ]);

    wsRef.current.send(
      JSON.stringify({
        type: "text",
        content: textInput,
      })
    );

    setTextInput("");
  }, [textInput]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendText();
    }
  };

  // ── Microphone recording ───────────────────────────
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });

      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });

        const reader = new FileReader();
        reader.onloadend = () => {
          const base64Audio = reader.result.split(",")[1];
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(
              JSON.stringify({
                type: "audio",
                data: base64Audio,
              })
            );
          }
        };
        reader.readAsDataURL(audioBlob);

        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
    } catch (err) {
      if (err.name === "NotAllowedError") {
        setError("Microphone access denied. Please allow mic access.");
      } else {
        setError(`Mic error: ${err.message}`);
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
      setIsRecording(false);
    }
  };

  // ── Render ─────────────────────────────────────────
  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <ChefHat size={32} color="#f97316" />
          <h1 style={styles.title}>Sous Chef</h1>
        </div>
        <div style={styles.headerRight}>
          {isConnected ? (
            <Wifi size={18} color="#4ade80" />
          ) : (
            <WifiOff size={18} color="#78716c" />
          )}
        </div>
      </header>

      {error && (
        <div style={styles.error}>
          {error}
          <button onClick={() => setError(null)} style={styles.errorClose}>
            ✕
          </button>
        </div>
      )}

      <div style={styles.transcript}>
        {transcript.length === 0 && (
          <div style={styles.emptyState}>
            <ChefHat size={64} color="#44403c" />
            <p style={styles.emptyTitle}>Ready to cook?</p>
            <p style={styles.emptySubtitle}>
              Connect and say: "Let's make pasta"
            </p>
          </div>
        )}

        {transcript.map((msg, i) => (
          <div
            key={i}
            style={{
              ...styles.message,
              ...(msg.role === "user"
                ? styles.userMessage
                : styles.assistantMessage),
            }}
          >
            <div
              style={{
                ...styles.bubble,
                ...(msg.role === "user"
                  ? styles.userBubble
                  : styles.assistantBubble),
              }}
            >
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={transcriptEndRef} />
      </div>

      <div style={styles.controls}>
        <button
          onClick={isConnected ? disconnect : connect}
          style={{
            ...styles.connectButton,
            background: isConnected ? "#dc2626" : "#16a34a",
          }}
        >
          {isConnected ? "Disconnect" : "Connect"}
        </button>

        <div style={styles.inputRow}>
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isConnected ? "Type a message..." : "Connect first"}
            disabled={!isConnected}
            style={styles.textInput}
          />
          <button
            onClick={sendText}
            disabled={!isConnected || !textInput.trim()}
            style={{
              ...styles.sendButton,
              opacity: isConnected && textInput.trim() ? 1 : 0.3,
            }}
          >
            <Send size={20} />
          </button>
        </div>

        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={!isConnected}
          style={{
            ...styles.micButton,
            background: isRecording ? "#dc2626" : "#f97316",
            opacity: isConnected ? 1 : 0.3,
            animation: isRecording ? "pulse 1.5s infinite" : "none",
          }}
        >
          {isRecording ? (
            <MicOff size={28} color="white" />
          ) : (
            <Mic size={28} color="white" />
          )}
        </button>
        <p style={styles.micHint}>
          {isRecording
            ? "Recording... tap to stop"
            : isConnected
            ? "Tap to speak"
            : "Connect to start"}
        </p>
      </div>

      <style>{`
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
          70% { box-shadow: 0 0 0 15px rgba(220, 38, 38, 0); }
          100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
        }
      `}</style>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "480px",
    margin: "0 auto",
    padding: "16px",
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "16px",
  },
  headerLeft: { display: "flex", alignItems: "center", gap: "10px" },
  headerRight: { display: "flex", alignItems: "center", gap: "8px" },
  title: { fontSize: "24px", fontWeight: "bold", letterSpacing: "-0.5px" },
  error: {
    background: "rgba(220, 38, 38, 0.15)",
    border: "1px solid rgba(220, 38, 38, 0.3)",
    borderRadius: "10px",
    padding: "12px 16px",
    marginBottom: "12px",
    color: "#fca5a5",
    fontSize: "14px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  errorClose: {
    background: "none",
    border: "none",
    color: "#fca5a5",
    cursor: "pointer",
    fontSize: "16px",
  },
  transcript: {
    flex: 1,
    overflowY: "auto",
    marginBottom: "16px",
    minHeight: "300px",
  },
  emptyState: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    minHeight: "300px",
    gap: "12px",
  },
  emptyTitle: { fontSize: "18px", fontWeight: "600", color: "#a8a29e" },
  emptySubtitle: { fontSize: "14px", color: "#78716c" },
  message: { marginBottom: "10px", display: "flex" },
  userMessage: { justifyContent: "flex-end" },
  assistantMessage: { justifyContent: "flex-start" },
  bubble: {
    maxWidth: "80%",
    padding: "10px 14px",
    borderRadius: "16px",
    fontSize: "14px",
    lineHeight: "1.5",
  },
  userBubble: {
    background: "#ea580c",
    color: "white",
    borderBottomRightRadius: "4px",
  },
  assistantBubble: {
    background: "#292524",
    color: "#e7e5e4",
    borderBottomLeftRadius: "4px",
  },
  controls: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "12px",
    paddingTop: "8px",
    borderTop: "1px solid #292524",
  },
  connectButton: {
    width: "100%",
    padding: "10px",
    borderRadius: "10px",
    border: "none",
    color: "white",
    fontSize: "14px",
    fontWeight: "600",
    cursor: "pointer",
  },
  inputRow: { display: "flex", width: "100%", gap: "8px" },
  textInput: {
    flex: 1,
    padding: "10px 14px",
    borderRadius: "10px",
    border: "1px solid #44403c",
    background: "#292524",
    color: "#fafaf9",
    fontSize: "14px",
    outline: "none",
  },
  sendButton: {
    padding: "10px 14px",
    borderRadius: "10px",
    border: "none",
    background: "#f97316",
    color: "white",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
  },
  micButton: {
    width: "64px",
    height: "64px",
    borderRadius: "50%",
    border: "none",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    marginTop: "8px",
  },
  micHint: { fontSize: "12px", color: "#78716c" },
};
