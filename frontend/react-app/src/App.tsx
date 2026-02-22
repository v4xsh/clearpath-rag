import { useEffect, useRef, useState, KeyboardEvent } from "react";
import { ChatMessage } from "./components/ChatMessage";
import { useChat } from "./hooks/useChat";

export default function App() {
  const { messages, isLoading, error, sendMessage, clearSession } = useChat();
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    const q = input.trim();
    if (!q || isLoading) return;
    setInput("");
    sendMessage(q);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "100vh",
      background: "#0d0d0d",
      color: "#e0e0e0",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      maxWidth: "900px",
      margin: "0 auto",
    }}>
      {/* Header */}
      <div style={{
        padding: "14px 20px",
        borderBottom: "1px solid #1e1e1e",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        <div>
          <span style={{ fontWeight: "600", fontSize: "16px", color: "#e0e0e0" }}>Clearpath</span>
          <span style={{ color: "#555", marginLeft: "8px", fontSize: "13px" }}>Support</span>
        </div>
        <button
          onClick={clearSession}
          style={{
            fontSize: "12px",
            color: "#555",
            background: "none",
            border: "1px solid #2a2a2a",
            borderRadius: "4px",
            padding: "4px 10px",
            cursor: "pointer",
          }}
        >
          New session
        </button>
      </div>

      {/* Message list */}
      <div style={{ flex: 1, overflowY: "auto", scrollbarWidth: "thin" }}>
        {messages.length === 0 && (
          <div style={{
            padding: "48px 20px",
            textAlign: "center",
            color: "#444",
            fontSize: "14px",
          }}>
            Ask a question about Clearpath
          </div>
        )}
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        {error && (
          <div style={{ padding: "12px 16px", color: "#f87171", fontSize: "13px" }}>
            Error: {error}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div style={{
        borderTop: "1px solid #1e1e1e",
        padding: "12px 16px",
        display: "flex",
        gap: "10px",
        alignItems: "flex-end",
      }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about Clearpath..."
          rows={1}
          style={{
            flex: 1,
            background: "#1a1a1a",
            border: "1px solid #2a2a2a",
            borderRadius: "6px",
            color: "#e0e0e0",
            padding: "10px 12px",
            fontSize: "14px",
            resize: "none",
            outline: "none",
            lineHeight: "1.5",
            minHeight: "42px",
            maxHeight: "160px",
            overflowY: "auto",
            fontFamily: "inherit",
          }}
          disabled={isLoading}
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
          style={{
            background: isLoading ? "#2a2a2a" : "#2563eb",
            color: isLoading ? "#555" : "#fff",
            border: "none",
            borderRadius: "6px",
            padding: "10px 18px",
            cursor: isLoading ? "not-allowed" : "pointer",
            fontSize: "14px",
            fontWeight: "500",
            minWidth: "70px",
            height: "42px",
          }}
        >
          {isLoading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}