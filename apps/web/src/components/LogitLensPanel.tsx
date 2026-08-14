import { useEffect, useState } from "react";
import type { Model, WeightProvider, ActivationCapture } from "@llm-explorer/model-ir";
import { computeLogitLens, type LogitLensEntry } from "@llm-explorer/interpretability";
import type { Tokenizer } from "@llm-explorer/tokenizer";

interface Props {
  model: Model;
  weightProvider: WeightProvider;
  capture: ActivationCapture;
  tokenizer: Tokenizer;
  /** Shared with the rest of the app (prompt token chips, Prediction panel, Token Attribution) — clicking a token anywhere moves this same position, instead of each panel tracking its own. */
  selectedTokenIndex: number | null;
  onSelectToken: (i: number) => void;
  /** Reported whenever this panel's own background computation starts/stops — lets the app show a busy cursor while it runs. */
  onBusyChange?: (busy: boolean) => void;
}

export function LogitLensPanel({ model, weightProvider, capture, tokenizer, selectedTokenIndex, onSelectToken, onBusyChange }: Props) {
  // A stale selectedTokenIndex from a longer previous prompt (App only
  // resets it on model change, not on every re-run) would otherwise index
  // past this capture's logits.
  const tokenIndex = Math.min(selectedTokenIndex ?? capture.tokenIds.length - 1, capture.tokenIds.length - 1);
  const [layers, setLayers] = useState<LogitLensEntry[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    onBusyChange?.(true);
    computeLogitLens(model, weightProvider, capture, { tokenIndex, topK: 5 }).then((result) => {
      if (!cancelled) {
        setLayers(result);
        setLoading(false);
        onBusyChange?.(false);
      }
    });
    return () => {
      cancelled = true;
      onBusyChange?.(false);
    };
  }, [model, weightProvider, capture, tokenIndex]);

  const displayTokens = capture.tokenIds.map((id) => tokenizer.decodeToken(id));

  return (
    <div className="logit-lens">
      <div className="logit-lens-header">
        <span>
          Predicting the token <em>after</em>:
        </span>
        <div className="token-chips">
          {displayTokens.map((t, i) => (
            <button key={i} className={"token-chip" + (i === tokenIndex ? " selected" : "")} onClick={() => onSelectToken(i)}>
              {t.trim() === "" ? "·".repeat(Math.max(1, t.length)) : t}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="loading-hint">
          <span className="spinner" />
          Projecting each layer through the LM head…
        </div>
      )}

      {!loading && layers && (
        <div className="logit-lens-layers">
          {layers.map((layer) => (
            <div key={layer.nodeId} className="logit-lens-row">
              <div className="logit-lens-label">{layer.label}</div>
              <div className="logit-lens-bars">
                {layer.topTokens.map((t, i) => (
                  <div key={i} className="logit-lens-token" style={{ opacity: 0.4 + 0.6 * t.prob }} title={`${(t.prob * 100).toFixed(1)}%`}>
                    <span className="logit-lens-token-text">{tokenizer.decodeToken(t.tokenId) || `#${t.tokenId}`}</span>
                    <span className="logit-lens-token-prob">{(t.prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
