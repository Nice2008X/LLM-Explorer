import type { Model, ModelEdge } from "@llm-explorer/model-ir";

export function getDescendants(model: Model, id: string): string[] {
  const out: string[] = [];
  const stack = [...model.nodes[id].children];
  while (stack.length) {
    const cur = stack.pop()!;
    out.push(cur);
    stack.push(...model.nodes[cur].children);
  }
  return out;
}

/**
 * Only the leaf (no-children) descendants — the actual computation steps.
 * Purely-organizational container nodes (e.g. "Attention" grouping Q/K/V/Out)
 * are skipped here; collapseEdges bridges over them automatically since it
 * walks each edge endpoint up to its nearest *visible* ancestor. Containers
 * stay fully browsable via the ModelTree, just not as their own graph box.
 */
export function getLeafDescendants(model: Model, id: string): string[] {
  return getDescendants(model, id).filter((d) => model.nodes[d].children.length === 0);
}

function findVisibleAncestor(model: Model, nodeId: string, visible: Set<string>): string | null {
  let cur: string | null = nodeId;
  while (cur) {
    if (visible.has(cur)) return cur;
    cur = model.nodes[cur]?.parentId ?? null;
  }
  return null;
}

/**
 * Given a set of "visible" node ids (some of which are containers standing
 * in for a hidden subtree), re-derives the edges between exactly those
 * nodes by walking each real edge's endpoints up to their nearest visible
 * ancestor. This is what makes level-1 (collapsed blocks) and level-2
 * (one block's internals) both renderable from the same underlying edge
 * list, with no per-architecture special-casing.
 */
export function collapseEdges(model: Model, visibleIds: string[]): ModelEdge[] {
  const visible = new Set(visibleIds);
  const seen = new Set<string>();
  const result: ModelEdge[] = [];
  for (const e of model.edges) {
    const s = findVisibleAncestor(model, e.source, visible);
    const t = findVisibleAncestor(model, e.target, visible);
    if (!s || !t || s === t) continue;
    const key = `${s}->${t}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push({ id: key, source: s, target: t });
  }
  return result;
}

const ELLIPSIS = "__ellipsis__";

export interface Level1Graph {
  nodeIds: string[];
  edges: ModelEdge[];
  ellipsisCount: number;
}

/** The top-level architecture view: input, embeddings, every block (or an abbreviated run of them), head, output. */
export function buildLevel1Graph(model: Model, showAllBlocks: boolean, maxEdgeBlocks = 4): Level1Graph {
  const root = model.nodes[model.rootId];
  const blockGroup = root.children.find((id) => model.nodes[id].type === "block_group");
  const allBlockIds = blockGroup ? model.nodes[blockGroup].children : [];
  const nonBlockChildren = root.children.filter((id) => id !== blockGroup);

  const before = nonBlockChildren.filter((id) => {
    const rank = ["input", "embedding", "positional_embedding"].indexOf(model.nodes[id].type);
    return rank !== -1;
  });
  const after = nonBlockChildren.filter((id) => !before.includes(id));

  let blockIdsToShow = allBlockIds;
  let ellipsisCount = 0;
  if (!showAllBlocks && allBlockIds.length > maxEdgeBlocks * 2 + 1) {
    blockIdsToShow = [...allBlockIds.slice(0, maxEdgeBlocks), ...allBlockIds.slice(allBlockIds.length - maxEdgeBlocks)];
    ellipsisCount = allBlockIds.length - maxEdgeBlocks * 2;
  }

  const firstHalf = ellipsisCount > 0 ? blockIdsToShow.slice(0, maxEdgeBlocks) : blockIdsToShow;
  const secondHalf = ellipsisCount > 0 ? blockIdsToShow.slice(maxEdgeBlocks) : [];

  const nodeIds = [...before, ...firstHalf, ...(ellipsisCount > 0 ? [ELLIPSIS] : []), ...secondHalf, ...after];
  const edges = collapseEdges(model, [...before, ...blockIdsToShow, ...after]);

  if (ellipsisCount > 0) {
    edges.push({ id: "e1", source: firstHalf[firstHalf.length - 1], target: ELLIPSIS });
    edges.push({ id: "e2", source: ELLIPSIS, target: secondHalf[0] });
  }

  return { nodeIds, edges, ellipsisCount };
}

export { ELLIPSIS };

/** The block-detail view: every computation step inside one transformer block. */
export function buildLevel2Graph(model: Model, blockId: string): { nodeIds: string[]; edges: ModelEdge[] } {
  const nodeIds = getLeafDescendants(model, blockId);
  return { nodeIds, edges: collapseEdges(model, nodeIds) };
}
