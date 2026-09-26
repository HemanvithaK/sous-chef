import React, { useState, useRef, useEffect, useCallback } from "react";
import { Mic, Send, ChefHat, Wifi, WifiOff, Ear } from "lucide-react";
import { useVAD } from "./hooks/useVAD";
import { float32ToWavBase64 } from "./lib/audio";

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [handsFree, setHandsFree] = useState(false);
  const [transcript, setTranscript] = useState([]);
  const [textInput, setTextInput] = useState("");
  const [error, setError] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | listening | hearing | thinking | speaking

  const wsRef = useRef(null);
  const transcriptEndRef = useRef(null);
  const vadRef = useRef(null);
  const handsFreeRef = useRef(false);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript]);

  useEffect(() => {
    handsFreeRef.current = handsFree;
  }, [handsFree]);

  // Play TTS audio, pausing the mic while it plays so the mic
  // does not hear Sous Chef's own voice and transcribe it.
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

      if (vadRef.current) vadRef.current.pause();
      setStatus("speaking");

      audio.onended = () => {
        URL.revokeObjectURL(url);
        if (vadRef.current && handsFreeRef.current) {
          vadRef.current.resume();
          setStatus("listening");
        } else {
          setStatus("idle");
        }
      };

      audio.play().catch((e) => {
        console.error("Audio play failed:", e);
        if (vadRef.current && handsFreeRef.current) vadRef.current.resume();
      });
    } catch (e) {
      console.error("Audio decode failed:", e);
    }
  }, []);

  const sendUtterance = useCallback((float32Audio) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const base64Wav = float32ToWavBase64(float32Audio, 16000);
    setStatus("thinking");
    wsRef.current.send(JSON.stringify({ type: "audio", data: base64Wav }));
  }, []);

  const vad = useVAD({
    onSpeechStart: () => setStatus("hearing"),
    onSpeechEnd: (audio) => sendUtterance(audio),
  });

  useEffect(() => {
    vadRef.current = vad;
  }, [vad]);

  const connect = useCallback(() => {
    const ws = new WebSocket(`ws://${window.location.host}/ws/voice`);

    ws.onopen = () => {
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
      setIsConnected(false);
      setHandsFree(false);
      if (vadRef.current) vadRef.current.stop();
      setStatus("idle");
    };

    ws.onerror = () => {
      setError("Cannot connect to server. Is the backend running?");
      setIsConnected(false);
    };

    wsRef.current = ws;
  }, [playAudio]);

  const disconnect = useCallback(() => {
    if (vadRef.current) vadRef.current.stop();
    setHandsFree(false);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
    setStatus("idle");
  }, []);

  const toggleHandsFree = useCallback(async () => {
    if (handsFree) {
      vad.stop();
      setHandsFree(false);
      setStatus("idle");
    } else {
      try {
        await vad.start();
        setHandsFree(true);
        setStatus("listening");
        setError(null);
      } catch (err) {
        setError(`Couldn't start listening: ${err.message}`);
      }
    }
  }, [handsFree, vad]);

  const sendText = useCallback(() => {
    if (!textInput.trim() || !wsRef.current) return;
    setTranscript((prev) => [
      ...prev,
      { role: "user", text: textInput, ts: Date.now() },
    ]);
    wsRef.current.send(JSON.stringify({ type: "text", content: textInput }));
    setTextInput("");
    setStatus("thinking");
  }, [textInput]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendText();
    }
  };

  const statusLabel = {
    idle: isConnected ? "Connected" : "Not connected",
    listening: "Listening... just talk",
    hearing: "Hearing you...",
    thinking: "Thinking...",
    speaking: "Speaking...",
  }[status];

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
            X
          </button>
        </div>
      )}

      <div style={styles.transcript}>
        {transcript.length === 0 && (
          <div style={styles.emptyState}>
            <ChefHat size={64} color="#44403c" />
            <p style={styles.emptyTitle}>Ready to cook?</p>
            <p style={styles.emptySubtitle}>
              Connect, tap Start Listening, and just talk.
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

        <div
          style={{
            ...styles.statusPill,
            color:
              status === "listening" || status === "hearing"
                ? "#4ade80"
                : status === "speaking"
                ? "#f97316"
                : "#a8a29e",
          }}
        >
          {statusLabel}
        </div>

        <button
          onClick={toggleHandsFree}
          disabled={!isConnected}
          style={{
            ...styles.micButton,
            background: handsFree ? "#dc2626" : "#f97316",
            opacity: isConnected ? 1 : 0.3,
            animation: status === "hearing" ? "pulse 1.2s infinite" : "none",
          }}
        >
          {handsFree ? (
            <Ear size={28} color="white" />
          ) : (
            <Mic size={28} color="white" />
          )}
        </button>
        <p style={styles.micHint}>
          {handsFree ? "Tap to stop listening" : "Tap to go hands-free"}
        </p>

        <div style={styles.inputRow}>
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isConnected ? "Or type here..." : "Connect first"}
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
      </div>

      <style>{`
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.5); }
          70% { box-shadow: 0 0 0 18px rgba(74, 222, 128, 0); }
          100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
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
    minHeight: "280px",
  },
  emptyState: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    minHeight: "280px",
    gap: "12px",
  },
  emptyTitle: { fontSize: "18px", fontWeight: "600", color: "#a8a29e" },
  emptySubtitle: {
    fontSize: "14px",
    color: "#78716c",
    textAlign: "center",
    maxWidth: "260px",
  },
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
    gap: "10px",
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
  statusPill: { fontSize: "13px", fontWeight: "600", minHeight: "18px" },
  micButton: {
    width: "72px",
    height: "72px",
    borderRadius: "50%",
    border: "none",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  micHint: { fontSize: "12px", color: "#78716c" },
  inputRow: { display: "flex", width: "100%", gap: "8px", marginTop: "6px" },
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
};
