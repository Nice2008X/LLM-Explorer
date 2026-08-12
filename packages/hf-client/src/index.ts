import type { ModelMetadata, ModelSource } from "@llm-explorer/model-ir";
import { parseSafetensorsHeader } from "@llm-explorer/tensor-core";
import { fetchCachedArrayBuffer } from "./modelCache.js";

export { MAX_CACHEABLE_BYTES } from "./modelCache.js";

export function hfResolveUrl(source: ModelSource, file: string): string {
  const revision = source.revision ?? "main";
  return `https://huggingface.co/${source.repo}/resolve/${revision}/${file}`;
}

export async function fetchJson<T>(url: string): Promise<T> {
  const bytes = await fetchCachedArrayBuffer(url);
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

export async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  return fetchCachedArrayBuffer(url);
}

export interface HfConfigPreview {
  model_type?: string;
  architectures?: string[];
}

/** Reads just enough of config.json to let ModelAdapter.canLoad decide, before any adapter commits to fetching weights. */
export async function peekModelType(source: ModelSource): Promise<HfConfigPreview> {
  if (source.kind !== "huggingface") throw new Error("peekModelType only supports Hugging Face sources");
  return fetchJson<HfConfigPreview>(hfResolveUrl(source, "config.json"));
}

export interface RawSafetensorsMetadata<TConfig> {
  rawConfig: TConfig;
  weightIndex: ModelMetadata["weightIndex"];
  weightsBuffer: ArrayBuffer;
}

/**
 * The fetch sequence every safetensors-backed adapter needs: raw config.json
 * (typed however that adapter likes) plus the safetensors file, with its
 * header already parsed into a name -> {shape, dtype} index. Each adapter
 * turns `rawConfig` into its own normalized ModelConfig from here.
 */
export async function loadHfSafetensorsMetadata<TConfig>(source: ModelSource): Promise<RawSafetensorsMetadata<TConfig>> {
  if (source.kind !== "huggingface") throw new Error("This adapter only supports Hugging Face sources");

  const rawConfig = await fetchJson<TConfig>(hfResolveUrl(source, "config.json"));
  const weightsBuffer = await fetchArrayBuffer(hfResolveUrl(source, "model.safetensors"));
  const { header } = parseSafetensorsHeader(weightsBuffer);

  const weightIndex: ModelMetadata["weightIndex"] = {};
  for (const [name, entry] of Object.entries(header)) {
    weightIndex[name] = { shape: entry.shape, dtype: entry.dtype };
  }

  return { rawConfig, weightIndex, weightsBuffer };
}
