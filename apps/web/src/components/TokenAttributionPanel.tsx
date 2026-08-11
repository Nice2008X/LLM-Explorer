import { useEffect, useState } from "react";
import type { Model, ModelAdapter, WeightProvider } from "@llm-explorer/model-ir";
import { computeTokenAttribution, type TokenAttributionResult } from "@llm-explorer/interpretability";
import type { Tokenizer } from "@llm-explorer/tokenizer";

interface Props {
  model: Model;
  weightProvider: WeightProvider;
  adapter: ModelAdapter;
  tokenIds: number[];
  tokenizer: Tokenizer;
}

export function TokenAttributionPanel({ model, weightProvider, adapter, tokenIds, tokenizer }: Props) {
  const [result, setResult] = useState<TokenAttributionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    computeTokenAttribution(model, weightProvider, adapter, tokenIds)
      .then((r) => {
        if (!cancelled) {
          setResult(r);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [model, weightProvider, adapter, tokenIds]);

  if (loading) return <div className="empty-hint">Occluding each token in turn and re-running ({tokenIds.length + 1} forward passes)…</div>;
  if (error) return <div className="inference-error">{error}</div>;
  if (!result) return null;

  const maxAbs = Math.max(...result.entries.map((e) => Math.abs(e.logitDrop)), 1e-9);
  const targetDisplay = tokenizer.decodeToken(result.targetTokenId) || `#${result.targetTokenId}`;

  return (
    <div className="token-attribution">
      <div className="attribution-intro">
        Occlusion attribution toward predicting <strong>"{targetDisplay}"</strong> (the model's actual top prediction, logit {result.baselineLogit.toFixed(3)}).
        Each bar shows how much that token's removal <em>hurt</em> (blue, right) or <em>helped</em> (red, left) the prediction.
      </div>
      <div className="attribution-bars">
        {result.entries.map((e) => {
          const displayToken = tokenizer.decodeToken(tokenIds[e.tokenIndex]) || `#${tokenIds[e.tokenIndex]}`;
          const widthPct = (Math.abs(e.logitDrop) / maxAbs) * 50;
          return (
            <div key={e.tokenIndex} className="attribution-row">
              <span className="attribution-token">{displayToken.trim() || "·"}</span>
              <div className="attribution-track">
                <div className="attribution-center" />
                {e.logitDrop >= 0 ? (
                  <div className="attribution-fill attribution-positive" style={{ width: `${widthPct}%` }} />
                ) : (
                  <div className="attribution-fill attribution-negative" style={{ width: `${widthPct}%` }} />
                )}
              </div>
              <span className="attribution-value">{e.logitDrop >= 0 ? "+" : ""}{e.logitDrop.toFixed(4)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
