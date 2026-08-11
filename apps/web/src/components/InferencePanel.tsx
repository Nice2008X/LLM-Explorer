import { useState } from "react";
import type { InferenceState } from "../useInference.js";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  supported: boolean;
  state: InferenceState;
  onRun: (prompt: string) => void;
  selectedTokenIndex: number | null;
  onSelectToken: (i: number) => void;
  compareEnabled: boolean;
  onToggleCompare: () => void;
  promptBState: InferenceState;
  onRunB: (prompt: string) => void;
}

export function InferencePanel({ supported, state, onRun, selectedTokenIndex, onSelectToken, compareEnabled, onToggleCompare, promptBState, onRunB }: Props) {
  const { t } = useTranslation();
  const [prompt, setPrompt] = useState("The cat sat on the");
  const [promptB, setPromptB] = useState("The dog sat on the");

  if (!supported) {
    return (
      <div className="inference-panel inference-panel-disabled">
        {t("inference.noTokenizer")}
      </div>
    );
  }

  return (
    <div className="inference-panel">
      <form
        className="inference-form"
        onSubmit={(e) => {
          e.preventDefault();
          onRun(prompt);
        }}
      >
        <span className="inference-label">{t("inference.promptA")}</span>
        <input value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={t("inference.placeholderA")} />
        <button type="submit" disabled={state.status === "running"}>
          {state.status === "running" ? t("inference.running") : t("inference.run")}
        </button>
        <button type="button" className="compare-toggle" onClick={onToggleCompare}>
          {compareEnabled ? t("inference.hidePromptB") : t("inference.comparePromptB")}
        </button>
      </form>

      {state.status === "error" && <div className="inference-error">{state.error}</div>}

      {state.status === "ready" && state.displayTokens && (
        <div className="token-chips">
          {state.displayTokens.map((t, i) => (
            <button
              key={i}
              className={"token-chip" + (i === selectedTokenIndex ? " selected" : "")}
              onClick={() => onSelectToken(i)}
              title={`position ${i}`}
            >
              {t.trim() === "" ? "·".repeat(Math.max(1, t.length)) : t}
            </button>
          ))}
        </div>
      )}

      {compareEnabled && (
        <form
          className="inference-form prompt-b-form"
          onSubmit={(e) => {
            e.preventDefault();
            onRunB(promptB);
          }}
        >
          <span className="inference-label">{t("inference.promptB")}</span>
          <input value={promptB} onChange={(e) => setPromptB(e.target.value)} placeholder={t("inference.placeholderB")} />
          <button type="submit" disabled={promptBState.status === "running"}>
            {promptBState.status === "running" ? t("inference.running") : t("inference.runB")}
          </button>
        </form>
      )}
      {compareEnabled && promptBState.status === "error" && <div className="inference-error">{promptBState.error}</div>}
      {compareEnabled && promptBState.status === "ready" && promptBState.displayTokens && (
        <div className="token-chips token-chips-b">
          {promptBState.displayTokens.map((t, i) => (
            <span key={i} className="token-chip token-chip-readonly">
              {t.trim() === "" ? "·".repeat(Math.max(1, t.length)) : t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
