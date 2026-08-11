import type { Model } from "@llm-explorer/model-ir";
import { totalParameterCount, totalParameterBytes } from "@llm-explorer/model-ir";

function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function formatBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

export function ModelInfoBar({ model }: { model: Model }) {
  const params = totalParameterCount(model);
  const bytes = totalParameterBytes(model);
  const dtype = Object.values(model.nodes).find((n) => n.parameters.length > 0)?.parameters[0]?.dtype ?? "—";

  const stats: [string, string][] = [
    ["Architecture", model.architecture],
    ["Parameters", formatCount(params)],
    ["Layers", String(model.config.numLayers)],
    ["Attention heads", String(model.config.numHeads)],
    ["Hidden size", String(model.config.hiddenSize)],
    ["Intermediate size", String(model.config.intermediateSize)],
    ["Vocabulary", model.config.vocabSize.toLocaleString()],
    ["Context length", model.config.contextLength.toLocaleString()],
    ["Dtype", dtype],
    ["Weights (in browser)", formatBytes(bytes)],
  ];

  return (
    <div className="model-info-bar">
      <div className="model-info-name">{model.name}</div>
      <div className="model-info-stats">
        {stats.map(([label, value]) => (
          <div key={label} className="model-info-stat">
            <div className="model-info-stat-value">{value}</div>
            <div className="model-info-stat-label">{label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
