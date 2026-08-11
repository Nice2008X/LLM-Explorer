import type { NodeType } from "@llm-explorer/model-ir";

export interface NodeTypeInfo {
  label: string;
  color: string;
  /** "What is it?" — short educational description, independent of any one architecture. */
  description: string;
  /** Optional formula shown in the Inspector's "Show me the math" section. */
  formula?: string;
}

/**
 * The one place that maps an IR NodeType to how it's drawn and explained.
 * A new model adapter (Llama, Mistral, ...) reuses this as-is; it only needs
 * new entries here if it introduces a genuinely new NodeType (e.g. "rope"
 * already has one below, ready for when an adapter emits it).
 */
export const componentRegistry: Record<NodeType, NodeTypeInfo> = {
  model: {
    label: "Model",
    color: "#6b7280",
    description: "The root of the model graph — everything else hangs off this node.",
  },
  block_group: {
    label: "Transformer Blocks",
    color: "#6b7280",
    description: "A repeated stack of identical transformer blocks. Expand a block to see inside it.",
  },
  input: {
    label: "Input",
    color: "#6b7280",
    description: "Token IDs produced by the tokenizer, one integer per input token.",
  },
  embedding: {
    label: "Token Embedding",
    color: "#2563eb",
    description: "Looks up a learned vector for each token ID, turning discrete tokens into continuous vectors.",
    formula: "E = W_e[token_id]",
  },
  positional_embedding: {
    label: "Positional Embedding",
    color: "#2563eb",
    description: "Adds a learned (or computed) vector per sequence position so the model can tell token order apart.",
    formula: "h = E_token + E_position",
  },
  transformer_block: {
    label: "Transformer Block",
    color: "#7c3aed",
    description: "One layer of the model: self-attention lets tokens exchange information, then a feed-forward network transforms each token independently. Both are wrapped in residual connections.",
  },
  layer_norm: {
    label: "LayerNorm",
    color: "#059669",
    description: "Normalizes each token's vector to zero mean / unit variance, then applies a learned scale and shift. Stabilizes training in deep stacks.",
    formula: "y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta",
  },
  rms_norm: {
    label: "RMSNorm",
    color: "#059669",
    description: "A simplified LayerNorm that skips re-centering and only rescales by the root-mean-square — cheaper, and standard in Llama-family models.",
    formula: "y = x / sqrt(mean(x^2) + eps) * gamma",
  },
  attention: {
    label: "Attention",
    color: "#dc2626",
    description: "Lets every token look at every other token in the sequence and pull in relevant information, weighted by learned relevance scores.",
    formula: "Attention(Q, K, V) = softmax(QKᵀ / √d_k) V",
  },
  q_projection: {
    label: "Q Projection",
    color: "#f59e0b",
    description: "Projects each token's vector into a \"query\": what this token is looking for in other tokens.",
    formula: "Q = X W_q",
  },
  k_projection: {
    label: "K Projection",
    color: "#f59e0b",
    description: "Projects each token's vector into a \"key\": what this token offers when other tokens look at it.",
    formula: "K = X W_k",
  },
  v_projection: {
    label: "V Projection",
    color: "#f59e0b",
    description: "Projects each token's vector into a \"value\": the actual content mixed into the output once attention weights are computed.",
    formula: "V = X W_v",
  },
  qkv_projection: {
    label: "QKV Projection",
    color: "#f59e0b",
    description: "A single fused projection producing Query, Key, and Value in one matrix multiply — a common implementation shortcut.",
    formula: "[Q K V] = X W_qkv",
  },
  output_projection: {
    label: "Output Projection",
    color: "#f59e0b",
    description: "Mixes the concatenated per-head attention outputs back into a single vector per token.",
    formula: "Out = Concat(head_1, ..., head_h) W_o",
  },
  rope: {
    label: "RoPE",
    color: "#0891b2",
    description: "Rotary Positional Embedding: rotates Q and K vectors by an angle proportional to sequence position, injecting relative-position information directly into the attention dot product.",
  },
  ffn: {
    label: "Feed Forward",
    color: "#7c3aed",
    description: "Processes each token's vector independently (no cross-token mixing): expand to a larger hidden dimension, apply a nonlinearity, then project back down.",
    formula: "FFN(x) = Linear₂(Activation(Linear₁(x)))",
  },
  linear: {
    label: "Linear",
    color: "#0ea5e9",
    description: "A fully-connected layer: a matrix multiply plus a bias.",
    formula: "y = xW + b",
  },
  activation: {
    label: "Activation",
    color: "#0ea5e9",
    description: "A pointwise nonlinearity applied after a linear layer — without it, stacking linear layers would collapse into one big linear layer.",
  },
  elementwise_mul: {
    label: "Gate (× )",
    color: "#0ea5e9",
    description: "Multiplies two parallel branches together element-by-element. Used by gated FFNs (SwiGLU, GEGLU, ...): one branch is squashed by an activation and acts as a learned gate on the other, un-activated branch.",
    formula: "FFN(x) = down_proj( act(gate_proj(x)) ⊙ up_proj(x) )",
  },
  residual: {
    label: "Residual Add",
    color: "#64748b",
    description: "Adds a sub-layer's input back onto its output. Keeps gradients flowing through very deep stacks and lets each sub-layer learn a small correction rather than a full transformation.",
    formula: "y = x + SubLayer(x)",
  },
  lm_head: {
    label: "LM Head",
    color: "#be185d",
    description: "Projects the final hidden state at each position into a score (logit) for every vocabulary token. Often tied to the token embedding weight, transposed.",
    formula: "logits = h W_eᵀ",
  },
  output: {
    label: "Output",
    color: "#6b7280",
    description: "The raw logits over the vocabulary, one distribution per input position — softmax turns these into next-token probabilities.",
  },
};
