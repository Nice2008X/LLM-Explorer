import type { Model, ModelAdapter, WeightProvider } from "@llm-explorer/model-ir";

export interface TokenAttributionEntry {
  tokenIndex: number;
  /** baselineLogit - occludedLogit, for the target token. Positive = this token was supporting the prediction; negative = it was actively working against it. */
  logitDrop: number;
}

export interface TokenAttributionResult {
  targetTokenId: number;
  baselineLogit: number;
  entries: TokenAttributionEntry[];
}

/**
 * Occlusion-based attribution: for each input token in turn, zero out its
 * embedding (leaving position information intact — see
 * `embedNodeIdFor`'s doc comment) and re-run, measuring how much the top
 * prediction's logit drops. A token whose removal hurts the prediction a
 * lot gets a high score.
 *
 * This is one honest, cheap attribution method among several the project
 * notes discuss (gradient-based, integrated gradients, activation
 * patching, logit difference); it does not claim to be the definitive
 * measure of "importance" — occlusion has well-known blind spots (e.g. it
 * can't see redundant/backup signals two tokens both carry). Gradient-based
 * methods would need a from-scratch autodiff layer over every op in
 * nn-ops, which is out of scope here.
 */
export async function computeTokenAttribution(
  model: Model,
  weightProvider: WeightProvider,
  adapter: ModelAdapter,
  tokenIds: number[]
): Promise<TokenAttributionResult> {
  if (!adapter.runInference) throw new Error(`${adapter.displayName} does not support running inference`);

  const S = tokenIds.length;
  const baseline = await adapter.runInference(model, weightProvider, tokenIds);
  const vocab = baseline.logits.shape[1];
  const lastRow = Array.from(baseline.logits.data.slice((S - 1) * vocab, S * vocab));
  const targetTokenId = lastRow.reduce((best, v, i) => (v > lastRow[best] ? i : best), 0);
  const baselineLogit = lastRow[targetTokenId];

  const embedNodeId = embedNodeIdFor(model);

  const entries: TokenAttributionEntry[] = [];
  for (let t = 0; t < S; t++) {
    const occluded = await adapter.runInference(model, weightProvider, tokenIds, [{ nodeId: embedNodeId, operation: "zero", tokenIndex: t }]);
    const occRow = Array.from(occluded.logits.data.slice((S - 1) * vocab, S * vocab));
    entries.push({ tokenIndex: t, logitDrop: baselineLogit - occRow[targetTokenId] });
  }

  return { targetTokenId, baselineLogit, entries };
}

/**
 * GPT-2 keeps token identity ("wte") and position ("wpe") as separate
 * additive tensors, so zeroing "wte" at one row removes only that token's
 * identity. The Llama family folds both into one "embed" node and injects
 * position later via RoPE (computed from the position index alone, not
 * from the embedding values) — so zeroing "embed" at one row has the same
 * "keep position, remove identity" effect there too.
 */
function embedNodeIdFor(model: Model): string {
  return model.nodes["ln_f"] ? "wte" : "embed";
}
