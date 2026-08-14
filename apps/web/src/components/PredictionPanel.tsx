import type { ActivationCapture } from "@llm-explorer/model-ir";
import type { Tokenizer } from "@llm-explorer/tokenizer";
import { topKFromLogits } from "../logits.js";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  result: ActivationCapture;
  tokenizer: Tokenizer;
  selectedTokenIndex: number | null;
  onViewWhy: () => void;
}

/**
 * The "what did the model predict" answer, front and center right after a
 * run — reuses topKFromLogits (already existed, only ever wired into
 * ExperimentPanel's before/after comparison) so this needs no new
 * computation, just a place to show it without digging into a bottom tab.
 */
export function PredictionPanel({ result, tokenizer, selectedTokenIndex, onViewWhy }: Props) {
  const { t } = useTranslation();
  // A stale selectedTokenIndex from a longer previous prompt (App only
  // resets it on model change, not on every re-run) would otherwise index
  // past this result's logits.
  const tokenIndex = Math.min(selectedTokenIndex ?? result.tokenIds.length - 1, result.tokenIds.length - 1);
  const ranked = topKFromLogits(result.logits, tokenIndex, 5);
  const maxProb = ranked[0]?.prob ?? 1;

  return (
    <div className="prediction-panel">
      <div className="prediction-header">
        <span className="prediction-title">{t("prediction.title")}</span>
        <span className="prediction-position">{t("prediction.position").replace("{n}", String(tokenIndex))}</span>
        <button type="button" className="prediction-why-link" onClick={onViewWhy}>
          {t("prediction.why")}
        </button>
      </div>
      <div className="prediction-rows">
        {ranked.map((r) => {
          const display = tokenizer.decodeToken(r.tokenId);
          return (
            <div key={r.tokenId} className="prediction-row">
              <span className="prediction-token">{display.trim() || `#${r.tokenId}`}</span>
              <div className="prediction-bar-track">
                <div className="prediction-bar-fill" style={{ width: `${maxProb > 0 ? (r.prob / maxProb) * 100 : 0}%` }} />
              </div>
              <span className="prediction-pct">{(r.prob * 100).toFixed(1)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
