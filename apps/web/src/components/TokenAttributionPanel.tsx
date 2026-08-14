import { useEffect, useState } from "react";
import type { Model, ModelAdapter, WeightProvider } from "@llm-explorer/model-ir";
import { computeTokenAttribution, computeHeadAttribution, type TokenAttributionResult, type HeadAttributionResult } from "@llm-explorer/interpretability";
import type { Tokenizer } from "@llm-explorer/tokenizer";

interface Props {
  model: Model;
  weightProvider: WeightProvider;
  adapter: ModelAdapter;
  tokenIds: number[];
  tokenizer: Tokenizer;
  /** Which position's next-token prediction to attribute toward — shared with the Prediction panel/prompt chips, defaults to the last position. */
  selectedTokenIndex: number | null;
  /** Clicking an influential-head row jumps the graph/tree/inspector to that block's Attention node. */
  onSelectNode: (nodeId: string) => void;
  /** Reported whenever this panel's own background computation starts/stops — lets the app show a busy cursor while it runs. */
  onBusyChange?: (busy: boolean) => void;
}

export function TokenAttributionPanel({ model, weightProvider, adapter, tokenIds, tokenizer, selectedTokenIndex, onSelectNode, onBusyChange }: Props) {
  const [result, setResult] = useState<TokenAttributionResult | null>(null);
  const [headResult, setHeadResult] = useState<HeadAttributionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A stale selectedTokenIndex from a longer previous prompt (App only
  // resets it on model change, not on every re-run) would otherwise index
  // past this prompt's logits.
  const predictIndex = Math.min(selectedTokenIndex ?? tokenIds.length - 1, tokenIds.length - 1);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    onBusyChange?.(true);
    Promise.all([
      computeTokenAttribution(model, weightProvider, adapter, tokenIds, { predictIndex }),
      computeHeadAttribution(model, weightProvider, adapter, tokenIds, { predictIndex }),
    ])
      .then(([tokenResult, heads]) => {
        if (!cancelled) {
          setResult(tokenResult);
          setHeadResult(heads);
          setLoading(false);
          onBusyChange?.(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setLoading(false);
          onBusyChange?.(false);
        }
      });
    return () => {
      cancelled = true;
      onBusyChange?.(false);
    };
  }, [model, weightProvider, adapter, tokenIds, predictIndex]);

  if (loading)
    return (
      <div className="loading-hint">
        <span className="spinner" />
        Occluding each token and attention head in turn and re-running…
      </div>
    );
  if (error) return <div className="inference-error">{error}</div>;
  if (!result || !headResult) return null;

  const maxAbs = Math.max(...result.entries.map((e) => Math.abs(e.logitDrop)), 1e-9);
  const targetDisplay = tokenizer.decodeToken(result.targetTokenId) || `#${result.targetTokenId}`;
  const positionNote = predictIndex !== tokenIds.length - 1 ? ` at position ${predictIndex}` : "";

  const topHeads = [...headResult.entries].sort((a, b) => Math.abs(b.logitDrop) - Math.abs(a.logitDrop)).slice(0, 8);
  const maxHeadAbs = Math.max(...topHeads.map((e) => Math.abs(e.logitDrop)), 1e-9);

  return (
    <div className="token-attribution">
      <div className="attribution-intro">
        Occlusion attribution toward predicting <strong>"{targetDisplay}"</strong>{positionNote} (the model's actual top prediction, logit {result.baselineLogit.toFixed(3)}).
        Each bar shows how much removing that token or head <em>hurt</em> (blue, right) or <em>helped</em> (red, left) the prediction.
      </div>
      <div className="attribution-section-title">Most influential tokens</div>
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

      <div className="attribution-section-title">
        Most influential heads
        {headResult.truncated && <span className="attribution-truncated-note"> (showing the first 64 of more combinations in this model)</span>}
      </div>
      <div className="attribution-bars">
        {topHeads.map((e) => {
          const widthPct = (Math.abs(e.logitDrop) / maxHeadAbs) * 50;
          return (
            <button key={`${e.nodeId}:${e.headIndex}`} className="attribution-row attribution-row-clickable" onClick={() => onSelectNode(e.nodeId)}>
              <span className="attribution-token">{e.blockLabel} · Head {e.headIndex}</span>
              <div className="attribution-track">
                <div className="attribution-center" />
                {e.logitDrop >= 0 ? (
                  <div className="attribution-fill attribution-positive" style={{ width: `${widthPct}%` }} />
                ) : (
                  <div className="attribution-fill attribution-negative" style={{ width: `${widthPct}%` }} />
                )}
              </div>
              <span className="attribution-value">{e.logitDrop >= 0 ? "+" : ""}{e.logitDrop.toFixed(4)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
