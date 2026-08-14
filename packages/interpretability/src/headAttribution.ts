import type { Model, ModelAdapter, ModelNode, WeightProvider } from "@llm-explorer/model-ir";
import { yieldToBrowser } from "./yield.js";

export interface HeadAttributionEntry {
  /** The owning attention node's id (e.g. "block.1.attn") — selecting this in the graph/tree jumps straight to it. */
  nodeId: string;
  /** The owning transformer block's display name (e.g. "Transformer Block 1"), for UI labels. */
  blockLabel: string;
  headIndex: number;
  logitDrop: number;
}

export interface HeadAttributionResult {
  targetTokenId: number;
  baselineLogit: number;
  predictIndex: number;
  entries: HeadAttributionEntry[];
  /** True if the model has more (block, head) combinations than this function will attribute — see MAX_COMBINATIONS. */
  truncated: boolean;
}

// Every one of this app's tiny-random presets has well under this many
// (block, head) combinations, so they always get full coverage. The cap
// exists for a hypothetical much larger checkpoint loaded via a custom repo
// id — without one, a real model's layer count x head count could turn this
// into thousands of forward passes in a single UI action.
const MAX_COMBINATIONS = 64;

/**
 * Head-level occlusion attribution: for every (transformer block, attention
 * head) pair, zero out just that head's contribution (the same "zero_head"
 * intervention the Experiment tab exposes manually — see
 * nn-ops/intervene.ts's applyHeadIntervention) and re-run, measuring how
 * much the target token's logit drops. Mirrors computeTokenAttribution's
 * occlusion loop exactly, just swapping "which token" for "which head".
 *
 * `headIndex` here always means a *query* head (0..numHeads-1) — attention's
 * output is numHeads*headDim wide regardless of a GQA model's smaller
 * key/value head count, and that's the dimension applyHeadIntervention
 * actually slices.
 */
export async function computeHeadAttribution(
  model: Model,
  weightProvider: WeightProvider,
  adapter: ModelAdapter,
  tokenIds: number[],
  options: { predictIndex?: number } = {}
): Promise<HeadAttributionResult> {
  if (!adapter.runInference) throw new Error(`${adapter.displayName} does not support running inference`);

  const S = tokenIds.length;
  const predictIndex = options.predictIndex ?? S - 1;

  const attentionNodes = Object.values(model.nodes)
    .filter((n) => n.type === "attention")
    .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));

  await yieldToBrowser();
  const baseline = await adapter.runInference(model, weightProvider, tokenIds);
  const vocab = baseline.logits.shape[1];
  const baseRow = Array.from(baseline.logits.data.slice(predictIndex * vocab, (predictIndex + 1) * vocab));
  const targetTokenId = baseRow.reduce((best, v, i) => (v > baseRow[best] ? i : best), 0);
  const baselineLogit = baseRow[targetTokenId];

  const combos: { node: ModelNode; headIndex: number }[] = [];
  for (const node of attentionNodes) {
    const numHeads = Number(node.metadata.numHeads ?? model.config.numHeads);
    for (let h = 0; h < numHeads; h++) combos.push({ node, headIndex: h });
  }
  const truncated = combos.length > MAX_COMBINATIONS;
  const scoped = truncated ? combos.slice(0, MAX_COMBINATIONS) : combos;

  const entries: HeadAttributionEntry[] = [];
  for (const { node, headIndex } of scoped) {
    await yieldToBrowser();
    const occluded = await adapter.runInference(model, weightProvider, tokenIds, [{ nodeId: node.id, operation: "zero_head", headIndex }]);
    const occRow = Array.from(occluded.logits.data.slice(predictIndex * vocab, (predictIndex + 1) * vocab));
    entries.push({
      nodeId: node.id,
      blockLabel: (node.parentId ? model.nodes[node.parentId]?.name : undefined) ?? node.name,
      headIndex,
      logitDrop: baselineLogit - occRow[targetTokenId],
    });
  }

  return { targetTokenId, baselineLogit, predictIndex, entries, truncated };
}
