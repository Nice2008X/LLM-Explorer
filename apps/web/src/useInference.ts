import { useCallback, useState } from "react";
import type { ActivationCapture, Model, ModelAdapter, WeightProvider } from "@llm-explorer/model-ir";
import type { Tokenizer } from "@llm-explorer/tokenizer";

export interface InferenceState {
  status: "idle" | "running" | "ready" | "error";
  error?: string;
  result?: ActivationCapture;
  displayTokens?: string[];
}

export function useInference(model: Model | undefined, weightProvider: WeightProvider | undefined, adapter: ModelAdapter | undefined, tokenizer: Tokenizer | undefined) {
  const [state, setState] = useState<InferenceState>({ status: "idle" });

  const run = useCallback(
    async (prompt: string) => {
      if (!model || !weightProvider || !tokenizer) return;
      if (!adapter?.runInference) {
        setState({ status: "error", error: `${adapter?.displayName ?? "This adapter"} does not support running inference yet.` });
        return;
      }
      setState({ status: "running" });
      try {
        const { ids, displayTokens } = tokenizer.encode(prompt);
        if (ids.length === 0) throw new Error("Prompt tokenized to zero tokens — try a non-empty prompt.");
        const result = await adapter.runInference(model, weightProvider, ids);
        setState({ status: "ready", result, displayTokens });
      } catch (err) {
        setState({ status: "error", error: err instanceof Error ? err.message : String(err) });
      }
    },
    [model, weightProvider, adapter, tokenizer]
  );

  const reset = useCallback(() => setState({ status: "idle" }), []);

  return { state, run, reset };
}
