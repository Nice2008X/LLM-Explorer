import { useEffect, useState } from "react";
import type { Model, WeightProvider, ActivationCapture } from "@llm-explorer/model-ir";
import { computeLogitLens, type LogitLensEntry } from "@llm-explorer/interpretability";
import type { Tokenizer } from "@llm-explorer/tokenizer";

interface Props {
  model: Model;
  weightProvider: WeightProvider;
  capture: ActivationCapture;
  tokenizer: Tokenizer;
}

export function LogitLensPanel({ model, weightProvider, capture, tokenizer }: Props) {
  const [tokenIndex, setTokenIndex] = useState(capture.tokenIds.length - 1);
  const [layers, setLayers] = useState<LogitLensEntry[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    computeLogitLens(model, weightProvider, capture, { tokenIndex, topK: 5 }).then((result) => {
      if (!cancelled) {
        setLayers(result);
        setLoading(false);
      }
    });
    return () => {
      cancelled = true;
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
            <button key={i} className={"token-chip" + (i === tokenIndex ? " selected" : "")} onClick={() => setTokenIndex(i)}>
              {t.trim() === "" ? "·".repeat(Math.max(1, t.length)) : t}
            </button>
          ))}
        </div>
      </div>

      {loading && <div className="empty-hint">Projecting each layer through the LM head…</div>}

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
