import { useCallback, useState } from "react";
import type { Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@llm-explorer/model-ir";
import { peekModelType } from "@llm-explorer/hf-client";
import { loadTokenizer, type Tokenizer } from "@llm-explorer/tokenizer";
import { ADAPTERS } from "./adapters.js";

export interface ModelState {
  status: "idle" | "loading" | "ready" | "error";
  error?: string;
  model?: Model;
  metadata?: ModelMetadata;
  weightProvider?: WeightProvider;
  adapter?: ModelAdapter;
  source?: ModelSource;
  /** Present only if the repo ships a tokenizer.json this app understands — inference/activation features need it, static architecture browsing doesn't. */
  tokenizer?: Tokenizer;
}

export function useModel() {
  const [state, setState] = useState<ModelState>({ status: "idle" });

  const load = useCallback(async (repo: string) => {
    setState({ status: "loading" });
    try {
      const source: ModelSource = { kind: "huggingface", repo: repo.trim() };

      // Read just enough of config.json to know what kind of model this is,
      // *before* any adapter commits to fetching (and possibly misreading)
      // its weights. This is the actual extension point for "lots of
      // different LLM models": each adapter only has to answer canLoad()
      // correctly for its own architecture — nothing else here changes.
      const preview = await peekModelType(source);
      const adapter = ADAPTERS.find((a) => a.canLoad(source, preview));
      if (!adapter) {
        throw new Error(
          `No adapter registered for model_type "${preview.model_type ?? "unknown"}" (architectures: ${(preview.architectures ?? []).join(", ") || "none"}).`
        );
      }

      const metadata = await adapter.loadMetadata(source);
      const model = adapter.buildGraph(metadata);
      const weightProvider = adapter.getWeightProvider(metadata);

      // Best-effort: not every repo ships a tokenizer.json this loader
      // understands, and static architecture/weight browsing doesn't need
      // one at all — only the "run inference" panel does.
      const tokenizer = await loadTokenizer(source).catch(() => undefined);

      setState({ status: "ready", model, metadata, weightProvider, adapter, source, tokenizer });
    } catch (err) {
      setState({ status: "error", error: err instanceof Error ? err.message : String(err) });
    }
  }, []);

  const reset = useCallback(() => setState({ status: "idle" }), []);

  return { state, load, reset };
}
