import { useState } from "react";
import type { Message } from "../types";
import { DebugPanel } from "./DebugPanel";

interface Props {
  message: Message;
}

export function ChatMessage({ message }: Props) {
  const [showDebug, setShowDebug] = useState(false);
  const isUser = message.role === "user";

  return (
    <div style={{
      padding: "12px 16px",
      background: isUser ? "#1a1f2e" : "#111",
      borderBottom: "1px solid #1e1e1e",
    }}>
      <div style={{
        display: "flex",
        alignItems: "baseline",
        gap: "8px",
        marginBottom: "6px",
      }}>
        <span style={{
          fontSize: "11px",
          fontWeight: "600",
          color: isUser ? "#7dd3fc" : "#86efac",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}>
          {isUser ? "You" : "Clearpath"}
        </span>

        {message.telemetry && (
          <button
            onClick={() => setShowDebug((v) => !v)}
            style={{
              fontSize: "10px",
              color: "#555",
              background: "none",
              border: "1px solid #333",
              borderRadius: "3px",
              padding: "1px 5px",
              cursor: "pointer",
            }}
          >
            {showDebug ? "hide debug" : "debug"}
          </button>
        )}
      </div>

      <div style={{
        color: "#d0d0d0",
        lineHeight: "1.6",
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
      }}>
        {message.streaming ? (
          <span>
            {message.content || ""}
            <span style={{ animation: "blink 1s step-end infinite", opacity: 0.7 }}>|</span>
          </span>
        ) : message.content}
      </div>

      {showDebug && message.telemetry && (
        <DebugPanel telemetry={message.telemetry} />
      )}
    </div>
  );
}