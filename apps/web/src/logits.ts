import type { Tensor } from "@llm-explorer/model-ir";

export interface RankedToken {
  tokenId: number;
  prob: number;
}

/** Softmaxes one sequence position of a [sequence, vocab] logits tensor and returns the top-k tokens. */
export function topKFromLogits(logits: Tensor, tokenIndex: number, k = 5): RankedToken[] {
  const vocab = logits.shape[1];
  const row = Array.from(logits.data.slice(tokenIndex * vocab, (tokenIndex + 1) * vocab));
  const max = Math.max(...row);
  const exps = row.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return row
    .map((_, tokenId) => ({ tokenId, prob: exps[tokenId] / sum }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, k);
}
