import { useCallback, useEffect, useState } from "react";
import type { Model, ModelAdapter, ModelMetadata, ModelSource, WeightProvider } from "@tensorium/model-ir";
import { fetchArrayBuffer, hfResolveUrl, peekModelType } from "@tensorium/hf-client";
import { loadTokenizer, type Tokenizer } from "@tensorium/tokenizer";
import { ADAPTERS } from "./adapters.js";

/**
 * Remembers the last Hugging-Face-sourced model across a page reload, so
 * refreshing the "chart frame" page restores it instead of bouncing back to
 * the home screen. Only ever holds a repo id, never local-file bytes —
 * a page can't silently re-read files the user picked before a reload
 * without a fresh user gesture, so local loads are deliberately not
 * persisted here (see loadFromSource/reset below, which clear this on
 * every local load and on an explicit "return to home").
 */
const LAST_REPO_KEY = "app:last-repo";

function readPersistedRepo(): string | null {
  try {
    return window.localStorage.getItem(LAST_REPO_KEY);
  } catch {
    return null;
  }
}

function writePersistedRepo(repo: string | null) {
  try {
    if (repo) window.localStorage.setItem(LAST_REPO_KEY, repo);
    else window.localStorage.removeItem(LAST_REPO_KEY);
  } catch {
    // storage unavailable (private browsing, quota, ...) — refresh just won't restore the model
  }
}

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

      // Hugging-Face-sourced weights are already cached in IndexedDB by URL
      // (see hf-client's fetchCachedArrayBuffer), so re-requesting this repo
      // on the next page load is a cache hit, not a re-download — cheap
      // enough to do automatically. Local sources clear this instead: their
      // bytes only ever live in memory, so there's nothing safe to restore.
      writePersistedRepo(source.kind === "huggingface" ? source.repo : null);

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

  // Explicit "return to home" — also clears the persisted repo, so a
  // refresh of the now-empty home page stays on the home page instead of
  // reloading the model the user just closed.
  const reset = useCallback(() => {
    setState({ status: "idle" });
    writePersistedRepo(null);
  }, []);

  // Restore the last Hugging-Face-sourced model once, on mount — this is
  // what makes refreshing the chart frame page stay there instead of
  // dropping back to the home screen. loadFromSource is stable (useCallback
  // with no deps), so this genuinely only needs to run once; StrictMode's
  // dev-only double-invoke just re-requests the same repo a second time,
  // which is a cache hit, not a real duplicate load.
  useEffect(() => {
    const repo = readPersistedRepo();
    if (repo) loadFromSource({ kind: "huggingface", repo });
  }, [loadFromSource]);

  return { state, load, loadLocalFiles, reset };
}
