import { useState, useCallback, useRef } from "react";
import type { Message } from "../types";

const API_BASE =
  import.meta.env.VITE_API_BASE ??
  "https://clearpath-rag-production.up.railway.app";

let messageCounter = 0;
const nextId = () => String(++messageCounter);

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const conversationIdRef = useRef<string | null>(null);

  const sendMessage = useCallback(async (question: string) => {
    setError(null);

    const userMessage: Message = {
      id: nextId(),
      role: "user",
      content: question,
    };

    const assistantId = nextId();
    const assistantPlaceholder: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
    };

    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setIsLoading(true);

    abortRef.current = new AbortController();

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortRef.current.signal,
        body: JSON.stringify({
          question,
          conversation_id: conversationIdRef.current ?? undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      // persist conversation id
      if (data.conversation_id) {
        conversationIdRef.current = data.conversation_id;
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: data.answer,
                telemetry: data.metadata, //  contract metadata
                sources: data.sources,
                streaming: false,
              }
            : m
        )
      );
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }

      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);

      setMessages((prev) => prev.filter((m) => m.id !== assistantId));
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearSession = useCallback(() => {
    conversationIdRef.current = null;
    setMessages([]);
    setError(null);
  }, []);

  return { messages, isLoading, error, sendMessage, clearSession };
}