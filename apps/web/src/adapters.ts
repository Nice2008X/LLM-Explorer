import type { ModelAdapter } from "@llm-explorer/model-ir";
import { GPT2Adapter } from "@llm-explorer/adapter-gpt2";
import { LlamaAdapter } from "@llm-explorer/adapter-llama";
import { MistralAdapter } from "@llm-explorer/adapter-mistral";
import { GemmaAdapter } from "@llm-explorer/adapter-gemma";
import { QwenAdapter } from "@llm-explorer/adapter-qwen";
import { Qwen3Adapter } from "@llm-explorer/adapter-qwen3";
import { PhiAdapter } from "@llm-explorer/adapter-phi";

/**
 * Every architecture the explorer supports. Adding a new one means writing
 * a new adapter package and adding it here — nothing else in the app
 * changes. Mistral, Gemma, Qwen2, Qwen3, and Phi-3/4 are all thin wrappers
 * around `@llm-explorer/adapter-llama-family` (same graph shape and
 * forward pass as Llama, parameterized by their real differences — GQA
 * ratio for Mistral; explicit head_dim, a (1+weight) RMSNorm, and
 * embedding scaling for Gemma; a bias on Q/K/V projections for Qwen2; a
 * per-head QK-Norm for Qwen3; fused Q/K/V and gate/up projections for
 * Phi — rather than separate copies of ~400 lines each).
 */
export const ADAPTERS: ModelAdapter[] = [GPT2Adapter, LlamaAdapter, MistralAdapter, GemmaAdapter, QwenAdapter, Qwen3Adapter, PhiAdapter];

// NOTE: this MVP's WeightProvider downloads the whole safetensors file up
// front (fine for models this size — a few hundred KB to a few MB). A
// full-size checkpoint would work functionally but download the entire
// thing just to read metadata, defeating the lazy-loading design described
// in the project notes — that needs a backend doing true HTTP range reads
// (see README.md "Known limitation"), not a browser-only preset here.
export const PRESET_MODELS = [
  { repo: "hf-internal-testing/tiny-random-gpt2", label: "GPT-2 · tiny-random-gpt2 (5 layers, 4 heads, hidden=32)" },
  { repo: "hf-internal-testing/tiny-random-LlamaForCausalLM", label: "Llama · tiny-random-LlamaForCausalLM (2 layers, 4 heads, hidden=16)" },
  { repo: "yujiepan/mistral-tiny-random", label: "Mistral · mistral-tiny-random (2 layers, GQA 4:2 heads, hidden=8)" },
  { repo: "fxmarty/tiny-random-GemmaForCausalLM", label: "Gemma · tiny-random-GemmaForCausalLM (1 layer, 2 heads, hidden=32)" },
  { repo: "yujiepan/qwen2-tiny-random", label: "Qwen2 · qwen2-tiny-random (2 layers, GQA 4:2 heads, Q/K/V bias)" },
  { repo: "tiny-random/qwen3", label: "Qwen3 · tiny-random/qwen3 (2 layers, GQA 2:1 heads, QK-Norm)" },
  { repo: "tiny-random/phi-4", label: "Phi-4 · tiny-random/phi-4 (2 layers, GQA 2:1 heads, fused QKV + gate/up)" },
];
