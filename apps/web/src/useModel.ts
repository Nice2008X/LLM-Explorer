import { useCallback, useState } from "react";
import type { Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@llm-explorer/model-ir";
import { fetchArrayBuffer, hfResolveUrl, peekModelType } from "@llm-explorer/hf-client";
import { loadTokenizer, type Tokenizer } from "@llm-explorer/tokenizer";
import { ADAPTERS } from "./adapters.js";

/** The exact bytes of each source file, kept around purely so "save model to disk" can hand the user back byte-identical files rather than re-serializing anything. */
export interface ModelRawFiles {
  configBytes?: ArrayBuffer;
  weightsBytes?: ArrayBuffer;
  tokenizerBytes?: ArrayBuffer;
}

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
  rawFiles?: ModelRawFiles;
}

/** config.json/tokenizer.json are small — a Hugging Face source hits the IndexedDB cache (already populated by loadMetadata/loadTokenizer above, so this is free) and a local source just reads the file it already has in memory. Missing/failed reads (e.g. no tokenizer.json) resolve to undefined rather than failing the whole load. */
async function readRawFile(source: ModelSource, filename: string): Promise<ArrayBuffer | undefined> {
  try {
    return source.kind === "local" ? source.files[filename] : await fetchArrayBuffer(hfResolveUrl(source, filename));
  } catch {
    return undefined;
  }
}

export function useModel() {
  const [state, setState] = useState<ModelState>({ status: "idle" });

  const loadFromSource = useCallback(async (source: ModelSource) => {
    setState({ status: "loading" });
    try {
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

      // The weights buffer is already sitting in `metadata` — reusing it
      // here avoids a second download of what can be a large file. Only
      // config.json/tokenizer.json need a (cheap) separate read.
      const [configBytes, tokenizerBytes] = await Promise.all([readRawFile(source, "config.json"), readRawFile(source, "tokenizer.json")]);
      const rawFiles: ModelRawFiles = { configBytes, weightsBytes: metadata.weightsBuffer, tokenizerBytes };

      setState({ status: "ready", model, metadata, weightProvider, adapter, source, tokenizer, rawFiles });
    } catch (err) {
      setState({ status: "error", error: err instanceof Error ? err.message : String(err) });
    }
  }, []);

  const load = useCallback((repo: string) => loadFromSource({ kind: "huggingface", repo: repo.trim() }), [loadFromSource]);

  const loadLocalFiles = useCallback(
    async (files: { name: string; config: File; weights: File; tokenizer?: File }) => {
      setState({ status: "loading" });
      try {
        const [configBytes, weightsBytes, tokenizerBytes] = await Promise.all([
          files.config.arrayBuffer(),
          files.weights.arrayBuffer(),
          files.tokenizer?.arrayBuffer(),
        ]);
        const sourceFiles: Record<string, ArrayBuffer> = { "config.json": configBytes, "model.safetensors": weightsBytes };
        if (tokenizerBytes) sourceFiles["tokenizer.json"] = tokenizerBytes;
        await loadFromSource({ kind: "local", name: files.name, files: sourceFiles });
      } catch (err) {
        setState({ status: "error", error: err instanceof Error ? err.message : String(err) });
      }
    },
    [loadFromSource]
  );

  const reset = useCallback(() => setState({ status: "idle" }), []);

  return { state, load, loadLocalFiles, reset };
}
