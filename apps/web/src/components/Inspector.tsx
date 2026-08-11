import type { ReactNode } from "react";
import type { ModelNode } from "@llm-explorer/model-ir";
import { componentRegistry } from "../registry.js";

function formatDims(dims: Array<number | string>): string {
  return `[${dims.join(", ")}]`;
}

export function Inspector({ node }: { node: ModelNode | null }) {
  if (!node) {
    return (
      <div className="inspector">
        <div className="empty-hint">Click a component in the graph or tree to inspect it.</div>
      </div>
    );
  }

  const info = componentRegistry[node.type];
  const totalParams = node.parameters.reduce((a, p) => a + (p.slice ? p.logicalShape.reduce((x, y) => x * y, 1) : p.numElements), 0);

  return (
    <div className="inspector">
      <div className="inspector-title" style={{ borderColor: info.color }}>
        <span className="inspector-badge" style={{ background: info.color }}>
          {info.label}
        </span>
        <span className="inspector-name">{node.name}</span>
      </div>

      <Section title="What is it?">
        <p>{info.description}</p>
      </Section>

      {info.formula && (
        <Section title="Show me the math">
          <code className="formula">{info.formula}</code>
        </Section>
      )}

      {(node.inputs.length > 0 || node.outputs.length > 0) && (
        <Section title="Shapes">
          {node.inputs.map((s, i) => (
            <div key={`in-${i}`} className="io-row">
              <span className="io-label">input</span>
              <span className="io-shape">{formatDims(s.dims)}</span>
            </div>
          ))}
          {node.outputs.map((s, i) => (
            <div key={`out-${i}`} className="io-row">
              <span className="io-label">output</span>
              <span className="io-shape">{formatDims(s.dims)}</span>
            </div>
          ))}
        </Section>
      )}

      {node.parameters.length > 0 && (
        <Section title={`Parameters (${totalParams.toLocaleString()})`}>
          {node.parameters.map((p, i) => (
            <div key={i} className="io-row">
              <span className="io-label">{p.slice ? `${p.name} (slice)` : p.name}</span>
              <span className="io-shape">
                {p.logicalShape.join(" × ")} · {p.dtype}
              </span>
            </div>
          ))}
        </Section>
      )}

      {Object.keys(node.metadata).length > 0 && (
        <Section title="Metadata">
          {Object.entries(node.metadata).map(([k, v]) => (
            <div key={k} className="io-row">
              <span className="io-label">{k}</span>
              <span className="io-shape">{typeof v === "object" ? JSON.stringify(v) : String(v)}</span>
            </div>
          ))}
        </Section>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="inspector-section">
      <div className="inspector-section-title">{title}</div>
      {children}
    </div>
  );
}
