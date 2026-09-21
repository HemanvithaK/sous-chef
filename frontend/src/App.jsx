// frontend/src/App.jsx

// This is the main component of our app. It handles:
// 1. WebSocket connection to the backend
// 2. Microphone recording and sending audio
// 3. Text input as a fallback
// 4. Displaying the conversation transcript

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Mic, MicOff, Send, ChefHat, Wifi, WifiOff } from "lucide-react";

export default function App() {
  // ── State ──────────────────────────────────────────
  //
  // React state holds all the data our UI needs to render.
  // When state changes, React re-renders the affected parts.

  // Is the WebSocket connected to the backend?
  const [isConnected, setIsConnected] = useState(false);

  // Is the mic currently recording?
  const [isRecording, setIsRecording] = useState(false);

  // Chat transcript — array of {role: "user"|"assistant", text: "..."}
  const [transcript, setTranscript] = useState([]);

  // Text input value (for typing instead of speaking)
  const [textInput, setTextInput] = useState("");

  // Error message to display
  const [error, setError] = useState(null);

  // ── Refs ───────────────────────────────────────────
  //
  // Refs hold values that persist across renders but
  // DON'T trigger re-renders when they change.
  // Perfect for WebSocket connections and media objects.

  // WebSocket connection reference
  const wsRef = useRef(null);

  // MediaRecorder reference (for mic recording)
  const mediaRecorderRef = useRef(null);

  // Audio chunks collected during recording
  const audioChunksRef = useRef([]);

  // Ref to the bottom of the transcript (for auto-scrolling)
  const transcriptEndRef = useRef(null);

  // ── Auto-scroll transcript ────────────────────────
  //
  // Every time a new message is added to the transcript,
  // scroll to the bottom so the user sees the latest message.
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript]);

  // ── WebSocket Connection ──────────────────────────
  //
  // connect() opens a WebSocket to the backend.
  // We use useCallback to prevent recreating this function
  // on every render (performance optimization).

  const connect = useCallback(() => {
    // Create a new WebSocket connection.
    // The URL uses the current host — in development, Vite's
    // proxy forwards /ws/* to localhost:8000 automatically.
    const ws = new WebSocket(`ws://${window.location.host}/ws/voice`);

    // onopen fires when the connection is established
    ws.onopen = () => {
      console.log("WebSocket connected");
      setIsConnected(true);
      setError(null);
    };

    // onmessage fires when the server sends us data
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "transcript") {
        // Add the message to our transcript
        setTranscript((prev) => [
          ...prev,
          { role: data.role, text: data.text, ts: Date.now() },
        ]);
      }
    };

    // onclose fires when the connection is lost
    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
    };

    // onerror fires on connection errors
    ws.onerror = () => {
      setError("Cannot connect to server. Is the backend running?");
      setIsConnected(false);
    };

    // Store the WebSocket in a ref so other functions can use it
    wsRef.current = ws;
  }, []);

  // ── Disconnect ────────────────────────────────────

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  // ── Send Text Message ─────────────────────────────
  //
  // Sends a typed text message to the backend.
  // Used as a fallback when mic isn't available or for testing.

  const sendText = useCallback(() => {
    if (!textInput.trim() || !wsRef.current) return;

    // Add the user's message to the transcript immediately
    // (don't wait for the server — feels faster)
    setTranscript((prev) => [
      ...prev,
      { role: "user", text: textInput, ts: Date.now() },
    ]);

    // Send the message to the backend as JSON
    wsRef.current.send(
      JSON.stringify({
        type: "text",
        content: textInput,
      })
    );

    // Clear the input field
    setTextInput("");
  }, [textInput]);

  // ── Handle Enter Key ──────────────────────────────

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendText();
    }
  };

  // ── Microphone Recording ──────────────────────────
  //
  // startRecording() asks for mic permission, then records audio.
  // stopRecording() stops recording and sends the audio to the backend.
  //
  // We use the MediaRecorder API, which is built into all modern browsers.

  const startRecording = async () => {
    try {
      // Ask the browser for mic access.
      // This triggers the "Allow microphone?" popup on first use.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1, // Mono audio (smaller file, all we need)
          sampleRate: 16000, // 16kHz (what Whisper expects)
          echoCancellation: true, // Reduce echo from speakers
          noiseSuppression: true, // Reduce background noise
        },
      });

      // Create a MediaRecorder that records in webm format.
      // webm is widely supported and Whisper can transcribe it.
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });

      // Clear any previous audio chunks
      audioChunksRef.current = [];

      // ondataavailable fires when the recorder has audio data ready.
      // We collect chunks and combine them when recording stops.
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      // onstop fires when we call mediaRecorder.stop().
      // At this point we have all the audio chunks and can send them.
      mediaRecorder.onstop = async () => {
        // Combine all chunks into one Blob (binary large object)
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });

        // Convert the Blob to base64 so we can send it as JSON.
        // WebSockets can send binary directly, but JSON is easier
        // to debug and extend with metadata.
        const reader = new FileReader();
        reader.onloadend = () => {
          // reader.result looks like "data:audio/webm;base64,SGVsbG8..."
          // We only want the base64 part after the comma.
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

        // Stop all audio tracks to release the microphone.
        // Without this, the browser keeps the mic indicator on.
        stream.getTracks().forEach((track) => track.stop());
      };

      // Start recording
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

  // ── Render ────────────────────────────────────────

  return (
    <div style={styles.container}>
      {/* Header */}
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

      {/* Error Banner */}
      {error && (
        <div style={styles.error}>
          {error}
          <button onClick={() => setError(null)} style={styles.errorClose}>
            ✕
          </button>
        </div>
      )}

      {/* Transcript Area */}
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
              ...(msg.role === "user" ? styles.userMessage : styles.assistantMessage),
            }}
          >
            <div
              style={{
                ...styles.bubble,
                ...(msg.role === "user" ? styles.userBubble : styles.assistantBubble),
              }}
            >
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={transcriptEndRef} />
      </div>

      {/* Bottom Controls */}
      <div style={styles.controls}>
        {/* Connect/Disconnect Button */}
        <button
          onClick={isConnected ? disconnect : connect}
          style={{
            ...styles.connectButton,
            background: isConnected ? "#dc2626" : "#16a34a",
          }}
        >
          {isConnected ? "Disconnect" : "Connect"}
        </button>

        {/* Text Input */}
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

        {/* Mic Button */}
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
          {isRecording ? <MicOff size={28} color="white" /> : <Mic size={28} color="white" />}
        </button>
        <p style={styles.micHint}>
          {isRecording
            ? "Recording... tap to stop"
            : isConnected
            ? "Tap to speak"
            : "Connect to start"}
        </p>
      </div>

      {/* Pulse animation for recording */}
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

// ── Styles ────────────────────────────────────────────
//
// Inline styles instead of CSS modules or Tailwind.
// Keeps everything in one file for Phase 1.

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
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  title: {
    fontSize: "24px",
    fontWeight: "bold",
    letterSpacing: "-0.5px",
  },
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
  emptyTitle: {
    fontSize: "18px",
    fontWeight: "600",
    color: "#a8a29e",
  },
  emptySubtitle: {
    fontSize: "14px",
    color: "#78716c",
  },
  message: {
    marginBottom: "10px",
    display: "flex",
  },
  userMessage: {
    justifyContent: "flex-end",
  },
  assistantMessage: {
    justifyContent: "flex-start",
  },
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
  inputRow: {
    display: "flex",
    width: "100%",
    gap: "8px",
  },
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
  micHint: {
    fontSize: "12px",
    color: "#78716c",
  },
};