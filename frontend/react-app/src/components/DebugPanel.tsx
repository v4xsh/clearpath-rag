import type { ChatResponse } from "../types";

interface Props {
  telemetry: ChatResponse;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  HIGH: "#22c55e",
  MEDIUM: "#f59e0b",
  LOW: "#ef4444",
};

function MetricRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: "1px solid #2a2a2a" }}>
      <span style={{ color: "#888", fontSize: "12px" }}>{label}</span>
      <span style={{ color: "#e0e0e0", fontSize: "12px", fontFamily: "monospace" }}>{value}</span>
    </div>
  );
}

export function DebugPanel({ telemetry }: Props) {
  const confidenceColor = CONFIDENCE_COLORS[telemetry.confidence] || "#888";

  return (
    <div style={{
      background: "#111",
      border: "1px solid #2a2a2a",
      borderRadius: "6px",
      padding: "12px",
      marginTop: "8px",
      fontSize: "12px",
    }}>
      <div style={{ color: "#555", fontSize: "11px", marginBottom: "8px", letterSpacing: "0.05em" }}>
        DEBUG TELEMETRY
      </div>

      <MetricRow label="Model" value={telemetry.model_used} />
      <MetricRow
        label="Confidence"
        value={<span style={{ color: confidenceColor, fontWeight: "bold" }}>{telemetry.confidence}</span>}
      />
      <MetricRow label="Complexity Score" value={telemetry.complexity_score.toFixed(3)} />
      <MetricRow label="Retrieval Depth (k)" value={telemetry.retrieval_k} />
      <MetricRow label="Coverage Score" value={telemetry.coverage_score.toFixed(4)} />
      <MetricRow label="Attribution Score" value={telemetry.attribution_score.toFixed(4)} />
      <MetricRow label="Tokens In" value={telemetry.tokens_input} />
      <MetricRow label="Tokens Out" value={telemetry.tokens_output} />
      <MetricRow label="Latency" value={`${telemetry.latency_ms.toFixed(0)}ms`} />

      <div style={{ marginTop: "6px", paddingTop: "6px", borderTop: "1px solid #2a2a2a" }}>
        <div style={{ color: "#888", fontSize: "11px", marginBottom: "4px" }}>Routing Reason</div>
        <div style={{ color: "#ccc", fontSize: "11px", lineHeight: "1.4" }}>{telemetry.routing_reason}</div>
      </div>

      {telemetry.flags.length > 0 && (
        <div style={{ marginTop: "6px", paddingTop: "6px", borderTop: "1px solid #2a2a2a" }}>
          <div style={{ color: "#888", fontSize: "11px", marginBottom: "4px" }}>Flags</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
            {telemetry.flags.map((flag) => (
              <span key={flag} style={{
                background: "#2a1a1a",
                color: "#f87171",
                padding: "2px 6px",
                borderRadius: "3px",
                fontSize: "11px",
                fontFamily: "monospace",
              }}>
                {flag}
              </span>
            ))}
          </div>
        </div>
      )}

      {telemetry.sources.length > 0 && (
        <div style={{ marginTop: "6px", paddingTop: "6px", borderTop: "1px solid #2a2a2a" }}>
          <div style={{ color: "#888", fontSize: "11px", marginBottom: "4px" }}>Sources</div>
          {telemetry.sources.map((src) => (
            <div key={src.chunk_id} style={{
              padding: "3px 0",
              color: "#7dd3fc",
              fontSize: "11px",
              cursor: "default",
            }}
              title={`Chunk: ${src.chunk_id} | Page: ${src.page_number}`}
            >
              {src.doc_id} / {src.section_path} (p.{src.page_number})
            </div>
          ))}
        </div>
      )}
    </div>
  );
}