import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  Background,
  BaseEdge,
  ControlButton,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  type Edge as RFEdge,
  type EdgeProps,
  type Node as RFNode,
  type ReactFlowInstance,
} from "reactflow";
import "reactflow/dist/style.css";
import type { Model, ModelEdge, ModelNode } from "@llm-explorer/model-ir";
import { categoryGlyph, categoryLabel, componentRegistry } from "../registry.js";
import { layeredLayout } from "../layout.js";
import { BLOCK_INPUT, buildLevel1Graph, buildLevel2Graph, ELLIPSIS } from "../graphUtils.js";
import { formatCount } from "../format.js";
import { useLocalStorageState } from "../useLocalStorageState.js";
import { useTranslation } from "./LanguageContext.js";

export type GraphView = { kind: "architecture" } | { kind: "block"; blockId: string };

interface Props {
  model: Model;
  view: GraphView;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onEnterBlock: (blockId: string) => void;
  onExitBlock: () => void;
  /** Whether the surrounding panels (prediction, tree, inspector, bottom) are currently collapsed to give the graph maximum space. */
  isMaxFrame: boolean;
  onToggleMaxFrame: () => void;
}

interface IRNodeData {
  node?: ModelNode;
  label: string;
  sublabel: string;
  glyph?: string;
  dims?: string;
  color: string;
  selected: boolean;
  dimmed: boolean;
  expandable?: boolean;
  inputPorts: number;
  outputPorts: number;
}

/**
 * A node with more than one sibling input/output gets one named Handle per
 * sibling instead of a single shared center point — otherwise every
 * incoming/outgoing edge visually converges on the exact same pixel, which
 * is the main reason a branching FFN/attention block reads as ambiguous
 * curves rather than a clear branch/merge.
 */
function PortHandles({ kind, position, count }: { kind: "target" | "source"; position: Position; count: number }) {
  if (count <= 1) return <Handle type={kind} position={position} style={{ opacity: 0 }} />;
  const prefix = kind === "target" ? "tgt" : "src";
  return (
    <>
      {Array.from({ length: count }, (_, i) => (
        <Handle key={i} type={kind} position={position} id={`${prefix}-${i}`} style={{ opacity: 0, left: `${((i + 1) / (count + 1)) * 100}%` }} />
      ))}
    </>
  );
}

function IRNodeComponent({ data }: { data: IRNodeData }) {
  return (
    <div className={"ir-node nopan nodrag" + (data.selected ? " selected" : "") + (data.dimmed ? " dimmed" : "")} style={{ borderColor: data.color }}>
      <PortHandles kind="target" position={Position.Top} count={data.inputPorts} />
      <div className="ir-node-label">
        {data.glyph && <span className="ir-node-glyph">{data.glyph}</span>}
        {data.label}
      </div>
      <div className="ir-node-sub">{data.sublabel}</div>
      {data.dims && <div className="ir-node-dims">{data.dims}</div>}
      {data.expandable && <div className="ir-node-hint">double-click to expand</div>}
      <PortHandles kind="source" position={Position.Bottom} count={data.outputPorts} />
      {/* Dedicated right-side ports for residual/skip edges routed through the
          side lane (see ResidualEdge below) — kept separate from the
          top/bottom main-flow ports so a residual connection never competes
          with a sibling data-flow edge for the same numbered port. */}
      <Handle type="target" position={Position.Right} id="lane-in" style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Right} id="lane-out" style={{ opacity: 0 }} />
    </div>
  );
}

interface LaneEdgeData {
  laneX: number;
}

/**
 * Residual/skip edges: Block Input and Residual Add are always single-node
 * ranks with nothing else beside them, so exiting straight out their right
 * side and down a vertical lane never has anything to cross.
 */
function ResidualLaneEdge({ id, sourceX, sourceY, targetX, targetY, markerEnd, style, data }: EdgeProps<LaneEdgeData>) {
  const laneX = data?.laneX ?? Math.max(sourceX, targetX) + 60;
  const path = `M ${sourceX},${sourceY} L ${laneX},${sourceY} L ${laneX},${targetY} L ${targetX},${targetY}`;
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />;
}

/**
 * Any other edge that skips over an intermediate rank — e.g. Llama's
 * V Projection -> Output Projection, which bypasses the RoPE rank entirely
 * since RoPE only applies to Q/K. Unlike the residual case, V has siblings
 * (K, Q) sitting right beside it, so exiting straight out its side would cut
 * across them. This uses the *same* Top/Bottom ports an ordinary edge would
 * (still correctly offset among sibling edges), and only detours sideways
 * to `laneX` after already dropping clear of the entire source rank's row —
 * every node in a rank shares its row's height, so once the path is below
 * that row it can travel at any x without crossing a same-rank sibling,
 * then it rises back above the target's row the same way before arriving.
 * `laneX` itself is chosen (see the detour lane computation below) to additionally clear
 * whatever sits in the rank(s) actually being skipped.
 */
function DetourEdge({ id, sourceX, sourceY, targetX, targetY, markerEnd, style, data }: EdgeProps<LaneEdgeData>) {
  const laneX = data?.laneX ?? (sourceX + targetX) / 2;
  const clear = 20; // matches the drop React Flow's own smoothstep uses before its first bend
  const y1 = sourceY + clear;
  const y2 = targetY - clear;
  const path = `M ${sourceX},${sourceY} L ${sourceX},${y1} L ${laneX},${y1} L ${laneX},${y2} L ${targetX},${y2} L ${targetX},${targetY}`;
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />;
}

/** Purely decorative marker at a fan-out/fan-in point — not a real graph node, carries no data, has no handles/edges of its own. */
function JunctionDot() {
  return <div className="graph-junction-dot" />;
}

/**
 * "[n]" bracket glyph for the tensor-shape toggle, drawn as vector paths
 * instead of literal "[" / "]" characters — those two glyphs don't share a
 * baseline in every monospace font, so at the control button's small size
 * they visibly sit at different heights. An SVG keeps both brackets exactly
 * level, and doubles as a plainer stand-in for "array dimensions" (bracket
 * pair around a value) if the toggle's meaning isn't obvious from the text
 * alone.
 */
function ShapeIcon({ active }: { active: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M4.5 2.5h-2v9h2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M9.5 2.5h2v9h-2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      {active && (
        <text x="7" y="8.8" textAnchor="middle" fontSize="6.5" fontFamily="ui-monospace, monospace" fill="currentColor">
          n
        </text>
      )}
    </svg>
  );
}

const nodeTypes = { ir: IRNodeComponent, junction: JunctionDot };
const edgeTypes = { lane: ResidualLaneEdge, detour: DetourEdge };

const EDGE_COLOR = "#94a3b8";
const SKIP_EDGE_COLOR = "#64748b";
/** Rough node width used only to eyeball where a junction dot sits horizontally — nodes auto-size, so this is an approximation, not a measurement. */
const NOMINAL_NODE_WIDTH = 160;
/** Clearance between the widest node a lane has to clear and the lane itself. */
const LANE_GAP = 90;
/** Minimum horizontal separation between two lanes whose vertical runs overlap — keeps concurrent detours (e.g. both block residuals, or a residual and an unrelated skip) from tracing the same line. */
const LANE_SEPARATION = 50;

export function ArchitectureGraph({ model, view, selectedId, onSelect, onEnterBlock, onExitBlock, isMaxFrame, onToggleMaxFrame }: Props) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const rfInstanceRef = useRef<ReactFlowInstance | null>(null);
  const [hoveredEdgeId, setHoveredEdgeId] = useState<string | null>(null);
  // Off by default: a "[sequence_length, 16]" label on every touched edge
  // is genuinely useful when you're chasing shapes, but it's clutter for
  // just browsing the architecture, so it stays opt-in rather than
  // appearing automatically whenever a node gets selected/hovered.
  const [showTensorShapes, setShowTensorShapes] = useLocalStorageState("panel:graph-tensor-shapes", false);

  const { nodeIds, edgeList } = useMemo(() => {
    if (view.kind === "architecture") {
      const g = buildLevel1Graph(model, false);
      return { nodeIds: g.nodeIds, edgeList: g.edges };
    }
    const g = buildLevel2Graph(model, view.blockId);
    return { nodeIds: g.nodeIds, edgeList: g.edges };
  }, [model, view]);

  const positions = useMemo(() => layeredLayout(nodeIds, edgeList), [nodeIds, edgeList]);

  // Any edge that skips over at least one intermediate rank (there's some
  // other node positioned strictly between its source and target) would
  // otherwise draw straight through whatever sits in between — that's the
  // overlap the lane fixes, whether or not the edge is a labeled residual.
  // Edges whose source/target are adjacent ranks don't need it; they never
  // had anything to collide with.
  //
  // Residual ("skip") edges all share one lane, cleared against every node
  // in the whole view: a block's two residuals should read as one
  // continuous line down the side, not jog inward/outward at the rank where
  // one hands off to the other. They exit straight out the source/target's
  // side, which is only safe because those nodes are always single-node
  // ranks with no siblings to cut across.
  //
  // Every other multi-rank edge (e.g. Llama's V Projection -> Output
  // Projection, which skips the RoPE rank since RoPE only applies to Q/K)
  // gets its own local detour instead, cleared only against the nodes it
  // actually spans — a short detour around one node shouldn't be pushed all
  // the way out past whatever's widest in the whole block. Unlike the
  // residual case, a source/target here typically *does* have siblings
  // (V's are K and Q), so the detour is routed via the ordinary Top/Bottom
  // ports rather than a side exit — see DetourEdge above for why that's
  // what keeps it from cutting across them.
  //
  // Each detour also picks whichever side — left or right of the obstacle —
  // costs less total sideways travel from its own source/target position,
  // rather than always going right: V sits at the *left* edge of its rank,
  // so detouring left around RoPE is both shorter and never has to cross
  // K or Q's column at all, whereas detouring right would (harmlessly, once
  // routed below the row — but there's no reason to prefer the longer path).
  // Detours are then greedily nudged further out, away from whichever side
  // they're already on, wherever two lanes' vertical runs would otherwise
  // overlap in y at the same x.
  const { skipLaneXByEdge, detourByEdge } = useMemo(() => {
    type Span = { lo: number; hi: number };
    const spanOf = (e: ModelEdge): Span | null => {
      const sy = positions.get(e.source)?.y;
      const ty = positions.get(e.target)?.y;
      if (sy == null || ty == null || sy === ty) return null;
      const lo = Math.min(sy, ty);
      const hi = Math.max(sy, ty);
      const hasIntervening = nodeIds.some((id) => {
        if (id === e.source || id === e.target) return false;
        const y = positions.get(id)?.y;
        return y != null && y > lo && y < hi;
      });
      return hasIntervening ? { lo, hi } : null;
    };

    const skipLaneXByEdge = new Map<string, number>();
    const placed: { lo: number; hi: number; x: number }[] = [];

    let viewMaxRight = 0;
    for (const id of nodeIds) {
      const p = positions.get(id);
      if (p) viewMaxRight = Math.max(viewMaxRight, p.x + NOMINAL_NODE_WIDTH / 2);
    }
    const skipLaneX = viewMaxRight + LANE_GAP;
    for (const e of edgeList) {
      if (e.label !== "skip") continue;
      const span = spanOf(e);
      if (!span) continue;
      skipLaneXByEdge.set(e.id, skipLaneX);
      placed.push({ ...span, x: skipLaneX });
    }

    const localCandidates = edgeList
      .filter((e) => e.label !== "skip")
      .map((e) => {
        const span = spanOf(e);
        if (!span) return null;
        const sourceX = positions.get(e.source)?.x ?? 0;
        const targetX = positions.get(e.target)?.x ?? 0;
        let obLeft = Infinity;
        let obRight = -Infinity;
        for (const id of nodeIds) {
          if (id === e.source || id === e.target) continue;
          const p = positions.get(id);
          if (p && p.y > span.lo && p.y < span.hi) {
            obLeft = Math.min(obLeft, p.x - NOMINAL_NODE_WIDTH / 2);
            obRight = Math.max(obRight, p.x + NOMINAL_NODE_WIDTH / 2);
          }
        }
        const rightX = obRight + LANE_GAP;
        const leftX = obLeft - LANE_GAP;
        const costRight = Math.abs(rightX - sourceX) + Math.abs(rightX - targetX);
        const costLeft = Math.abs(leftX - sourceX) + Math.abs(leftX - targetX);
        const x = costLeft < costRight ? leftX : rightX;
        return { id: e.id, ...span, x };
      })
      .filter((c): c is { id: string; lo: number; hi: number; x: number } => !!c)
      // Widest span first: a short local detour should never have to dodge
      // a long-spanning one that just happens to be processed first.
      .sort((a, b) => b.hi - b.lo - (a.hi - a.lo));

    const detourByEdge = new Map<string, number>();
    for (const c of localCandidates) {
      let x = c.x;
      const side = x < 0 ? -1 : 1;
      let moved = true;
      while (moved) {
        moved = false;
        for (const p of placed) {
          const overlapsY = c.lo < p.hi && p.lo < c.hi;
          if (overlapsY && Math.abs(x - p.x) < LANE_SEPARATION) {
            x += side * LANE_SEPARATION;
            moved = true;
          }
        }
      }
      placed.push({ lo: c.lo, hi: c.hi, x });
      detourByEdge.set(c.id, x);
    }
    return { skipLaneXByEdge, detourByEdge };
  }, [nodeIds, edgeList, positions]);

  // Multiple edges sharing one source (a branch) or one target (a merge)
  // each get their own named Handle, ordered left-to-right to match their
  // sibling's actual x position — so a port on the left side of a node
  // connects to whichever sibling is laid out on the left, minimizing
  // crossings instead of assigning ports arbitrarily. Only residual/skip
  // edges are excluded here — they use their own dedicated right-side
  // handle instead of a numbered port. Detour edges (V Projection's, etc.)
  // stay in this grouping since they use ordinary Top/Bottom ports too.
  const { sourceHandleByEdge, targetHandleByEdge, outputPortsById, inputPortsById } = useMemo(() => {
    const bySource = new Map<string, ModelEdge[]>();
    const byTarget = new Map<string, ModelEdge[]>();
    for (const e of edgeList) {
      if (skipLaneXByEdge.has(e.id)) continue;
      if (!bySource.has(e.source)) bySource.set(e.source, []);
      bySource.get(e.source)!.push(e);
      if (!byTarget.has(e.target)) byTarget.set(e.target, []);
      byTarget.get(e.target)!.push(e);
    }

    const sourceHandleByEdge = new Map<string, string>();
    const outputPortsById = new Map<string, number>();
    for (const [id, edges] of bySource) {
      outputPortsById.set(id, edges.length);
      if (edges.length <= 1) continue;
      const sorted = [...edges].sort((a, b) => (positions.get(a.target)?.x ?? 0) - (positions.get(b.target)?.x ?? 0));
      sorted.forEach((e, i) => sourceHandleByEdge.set(e.id, `src-${i}`));
    }

    const targetHandleByEdge = new Map<string, string>();
    const inputPortsById = new Map<string, number>();
    for (const [id, edges] of byTarget) {
      inputPortsById.set(id, edges.length);
      if (edges.length <= 1) continue;
      const sorted = [...edges].sort((a, b) => (positions.get(a.source)?.x ?? 0) - (positions.get(b.source)?.x ?? 0));
      sorted.forEach((e, i) => targetHandleByEdge.set(e.id, `tgt-${i}`));
    }

    return { sourceHandleByEdge, targetHandleByEdge, outputPortsById, inputPortsById };
  }, [edgeList, positions, skipLaneXByEdge]);

  // Selecting a node highlights its whole computational neighborhood
  // (every ancestor and descendant reachable through this view's edges)
  // and fades everything else, instead of the selection ring being the
  // only visible feedback.
  const relatedIds = useMemo(() => {
    if (!selectedId || !nodeIds.includes(selectedId)) return null;
    const related = new Set<string>([selectedId]);
    let frontier = [selectedId];
    while (frontier.length) {
      const next: string[] = [];
      for (const cur of frontier) {
        for (const e of edgeList) {
          if (e.target === cur && !related.has(e.source)) {
            related.add(e.source);
            next.push(e.source);
          }
        }
      }
      frontier = next;
    }
    frontier = [selectedId];
    while (frontier.length) {
      const next: string[] = [];
      for (const cur of frontier) {
        for (const e of edgeList) {
          if (e.source === cur && !related.has(e.target)) {
            related.add(e.target);
            next.push(e.target);
          }
        }
      }
      frontier = next;
    }
    return related;
  }, [selectedId, nodeIds, edgeList]);

  const rfNodes: RFNode<IRNodeData>[] = useMemo(
    () =>
      nodeIds.map((id) => {
        const pos = positions.get(id) ?? { x: 0, y: 0 };
        const outputPorts = outputPortsById.get(id) ?? 1;
        const inputPorts = inputPortsById.get(id) ?? 1;
        const dimmed = relatedIds !== null && !relatedIds.has(id);

        if (id === ELLIPSIS) {
          return {
            id,
            type: "ir",
            position: pos,
            draggable: false,
            selectable: false,
            data: { label: "⋯", sublabel: "more blocks (collapsed)", color: "#94a3b8", selected: false, dimmed, inputPorts, outputPorts },
          };
        }
        if (id === BLOCK_INPUT) {
          return {
            id,
            type: "ir",
            position: pos,
            draggable: false,
            selectable: false,
            data: { label: "Block Input", sublabel: "from outside this block", color: "#94a3b8", selected: false, dimmed, inputPorts, outputPorts },
          };
        }

        const node = model.nodes[id];
        const info = componentRegistry[node.type];
        const paramCount = node.parameters.reduce((a, p) => a + p.logicalShape.reduce((x, y) => x * y, 1), 0);
        const categoryText = categoryLabel[info.category] || info.label;
        const sublabel = paramCount > 0 ? `${categoryText} · ${formatCount(paramCount)} params` : categoryText;

        // A "16 -> 32" shape summary is only meaningful where the last
        // dimension actually changes (projections, embeddings) — showing
        // it on every node (where it'd usually just read "32 -> 32") would
        // be noise, not signal.
        let dims: string | undefined;
        if (info.category === "linear") {
          const inDims = node.inputs[0]?.dims;
          const outDims = node.outputs[0]?.dims;
          if (inDims?.length && outDims?.length) dims = `${inDims[inDims.length - 1]} → ${outDims[outDims.length - 1]}`;
        }

        return {
          id,
          type: "ir",
          position: pos,
          draggable: false,
          data: {
            node,
            label: node.name,
            sublabel,
            glyph: categoryGlyph[info.category] || undefined,
            dims,
            color: info.color,
            selected: id === selectedId,
            dimmed,
            expandable: node.type === "transformer_block",
            inputPorts,
            outputPorts,
          },
        };
      }),
    [nodeIds, positions, model, selectedId, relatedIds, outputPortsById, inputPortsById]
  );

  const junctionNodes: RFNode[] = useMemo(() => {
    const junctions: RFNode[] = [];
    for (const [id, count] of outputPortsById) {
      if (count <= 1) continue;
      const pos = positions.get(id);
      if (!pos) continue;
      junctions.push({
        id: `junction-out-${id}`,
        type: "junction",
        position: { x: pos.x + NOMINAL_NODE_WIDTH / 2 - 4, y: pos.y + 96 },
        draggable: false,
        selectable: false,
        data: {},
      });
    }
    for (const [id, count] of inputPortsById) {
      if (count <= 1) continue;
      const pos = positions.get(id);
      if (!pos) continue;
      junctions.push({
        id: `junction-in-${id}`,
        type: "junction",
        position: { x: pos.x + NOMINAL_NODE_WIDTH / 2 - 4, y: pos.y - 24 },
        draggable: false,
        selectable: false,
        data: {},
      });
    }
    return junctions;
  }, [outputPortsById, inputPortsById, positions]);

  const rfEdges: RFEdge[] = useMemo(
    () =>
      edgeList.map((e) => {
        const isSkip = e.label === "skip";
        const dimmed = relatedIds !== null && (!relatedIds.has(e.source) || !relatedIds.has(e.target));
        // Tensor shape is only worth showing when the user has opted in
        // AND is actually looking at this edge (hovered) or at one of its
        // endpoints (selected) — permanently labeling every edge with its
        // shape would be exactly the clutter the doc warns against.
        const showShape = showTensorShapes && (e.id === hoveredEdgeId || e.source === selectedId || e.target === selectedId);
        const shapeDims = showShape ? model.nodes[e.source]?.outputs[0]?.dims : undefined;
        const skipLaneX = skipLaneXByEdge.get(e.id);
        const detourX = detourByEdge.get(e.id);
        const laneX = skipLaneX ?? detourX;
        return {
          id: e.id,
          source: e.source,
          target: e.target,
          sourceHandle: skipLaneX !== undefined ? "lane-out" : sourceHandleByEdge.get(e.id),
          targetHandle: skipLaneX !== undefined ? "lane-in" : targetHandleByEdge.get(e.id),
          type: skipLaneX !== undefined ? "lane" : detourX !== undefined ? "detour" : "smoothstep",
          data: laneX !== undefined ? { laneX } : undefined,
          label: shapeDims?.length ? `[${shapeDims.join(", ")}]` : undefined,
          markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color: isSkip ? SKIP_EDGE_COLOR : EDGE_COLOR },
          style: {
            stroke: isSkip ? SKIP_EDGE_COLOR : EDGE_COLOR,
            strokeWidth: isSkip ? 2 : 1.5,
            // Dash length is defined in flow-space units, which the
            // current zoom then scales down further — at the zoom level
            // "Maximize graph view" lands on for a tall block (~0.5x), a
            // "9 6" dash shrinks to a couple of screen pixels and
            // Chromium's rasterizer blurs it into what reads as a solid
            // line. A chunkier pattern stays legibly dashed even at that
            // zoom (verified down to 0.5x; non-scaling-stroke was tried
            // first but only cancels SVG-native transforms, not the CSS
            // scale React Flow applies to the whole canvas, so it had no
            // effect here).
            ...(isSkip ? { strokeDasharray: "16 10" } : {}),
          },
          className: "graph-edge" + (dimmed ? " graph-edge-dimmed" : ""),
        };
      }),
    [edgeList, hoveredEdgeId, selectedId, model, relatedIds, sourceHandleByEdge, targetHandleByEdge, showTensorShapes, skipLaneXByEdge, detourByEdge]
  );

  const handleNodeClick = useCallback(
    (_: unknown, n: RFNode) => {
      if (n.id === ELLIPSIS || n.id === BLOCK_INPUT || n.type === "junction") return;
      onSelect(n.id);
    },
    [onSelect]
  );

  const handleNodeDoubleClick = useCallback(
    (_: unknown, n: RFNode) => {
      if (n.id === ELLIPSIS || n.id === BLOCK_INPUT || n.type === "junction") return;
      const node = model.nodes[n.id];
      if (node?.type === "transformer_block") onEnterBlock(n.id);
    },
    [model, onEnterBlock]
  );

  const viewKey = view.kind === "architecture" ? "arch" : `block:${view.blockId}`;

  // A node selected elsewhere (most commonly the model tree, which can
  // force-expand and highlight a node the graph never scrolled to) can end
  // up entirely outside the current viewport — the highlight is real but
  // invisible. Pan (never zoom) to bring it fully into view whenever that's
  // the case; skip it when the node's already visible, so clicking a node
  // directly in the graph doesn't yank the camera out from under the
  // user's cursor for no reason.
  useEffect(() => {
    if (!selectedId || !nodeIds.includes(selectedId)) return;
    const instance = rfInstanceRef.current;
    const container = containerRef.current;
    if (!instance || !container) return;

    // Right after a view switch (entering/exiting a block) this node may
    // not be measured yet on the same tick it mounts — wait a frame.
    const raf = requestAnimationFrame(() => {
      const node = instance.getNode(selectedId);
      if (!node) return;
      const width = node.width ?? 220;
      const height = node.height ?? 70;
      const { x: vx, y: vy, zoom } = instance.getViewport();
      const rect = container.getBoundingClientRect();
      const screenX = node.position.x * zoom + vx;
      const screenY = node.position.y * zoom + vy;
      const margin = 24; // keep the node clear of pane edges/controls, not just barely on-screen
      const fullyVisible =
        screenX >= margin && screenY >= margin && screenX + width * zoom <= rect.width - margin && screenY + height * zoom <= rect.height - margin;
      if (fullyVisible) return;

      instance.setCenter(node.position.x + width / 2, node.position.y + height / 2, { zoom, duration: 400 });
    });
    return () => cancelAnimationFrame(raf);
  }, [selectedId, nodeIds]);

  // A couple of the doc's requested shortcuts: Esc backs out of a focused
  // selection (matching the same "Esc = back off" convention the settings
  // and load-model popovers already use), 0 re-fits the view without
  // reaching for the on-canvas button. Ignored while typing anywhere else
  // in the app so pressing "0" in a prompt or search box isn't hijacked.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
      if (e.key === "Escape") onSelect(null);
      else if (e.key === "0") rfInstanceRef.current?.fitView({ padding: 0.2 });
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onSelect]);

  return (
    <div className="architecture-graph" ref={containerRef}>
      {view.kind === "block" && (
        <div className="graph-breadcrumb">
          <button onClick={onExitBlock}>← Architecture</button>
          <span>{model.nodes[view.blockId].name}</span>
        </div>
      )}
      <ReactFlow
        key={viewKey}
        nodes={[...rfNodes, ...junctionNodes] as RFNode[]}
        edges={rfEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodeClick={handleNodeClick}
        onNodeDoubleClick={handleNodeDoubleClick}
        onEdgeMouseEnter={(_, edge) => setHoveredEdgeId(edge.id)}
        onEdgeMouseLeave={() => setHoveredEdgeId(null)}
        onInit={(instance) => {
          rfInstanceRef.current = instance;
        }}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
        nodesConnectable={false}
        zoomOnDoubleClick={false}
        elementsSelectable
      >
        <Background />
        <MiniMap pannable zoomable nodeColor={(n) => (n.data as Partial<IRNodeData> | undefined)?.color ?? "#6b7280"} maskColor="rgba(15, 17, 23, 0.6)" />
        <Controls showInteractive={false}>
          <ControlButton onClick={onToggleMaxFrame} title={isMaxFrame ? t("graph.restorePanels") : t("graph.maximizeGraph")}>
            <span className="control-icon">{isMaxFrame ? "⤡" : "⤢"}</span>
          </ControlButton>
          <ControlButton
            onClick={() => setShowTensorShapes((v) => !v)}
            title={showTensorShapes ? t("graph.hideTensorShapes") : t("graph.showTensorShapes")}
          >
            <span className={"control-icon" + (showTensorShapes ? " active" : "")}>
              <ShapeIcon active={showTensorShapes} />
            </span>
          </ControlButton>
        </Controls>
      </ReactFlow>
    </div>
  );
}
