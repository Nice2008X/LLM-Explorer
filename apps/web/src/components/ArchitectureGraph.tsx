import { useCallback, useEffect, useMemo, useRef } from "react";
import ReactFlow, { Background, Controls, Handle, Position, type Edge as RFEdge, type Node as RFNode, type ReactFlowInstance } from "reactflow";
import "reactflow/dist/style.css";
import type { Model, ModelNode } from "@llm-explorer/model-ir";
import { componentRegistry } from "../registry.js";
import { layeredLayout } from "../layout.js";
import { buildLevel1Graph, buildLevel2Graph, ELLIPSIS } from "../graphUtils.js";

export type GraphView = { kind: "architecture" } | { kind: "block"; blockId: string };

interface Props {
  model: Model;
  view: GraphView;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onEnterBlock: (blockId: string) => void;
  onExitBlock: () => void;
}

interface IRNodeData {
  node?: ModelNode;
  label: string;
  sublabel: string;
  color: string;
  selected: boolean;
  expandable?: boolean;
}

function IRNodeComponent({ data }: { data: IRNodeData }) {
  return (
    <div className={"ir-node nopan nodrag" + (data.selected ? " selected" : "")} style={{ borderColor: data.color }}>
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <div className="ir-node-label">{data.label}</div>
      <div className="ir-node-sub">{data.sublabel}</div>
      {data.expandable && <div className="ir-node-hint">double-click to expand</div>}
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </div>
  );
}

const nodeTypes = { ir: IRNodeComponent };

export function ArchitectureGraph({ model, view, selectedId, onSelect, onEnterBlock, onExitBlock }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rfInstanceRef = useRef<ReactFlowInstance | null>(null);

  const { nodeIds, edgeList } = useMemo(() => {
    if (view.kind === "architecture") {
      const g = buildLevel1Graph(model, false);
      return { nodeIds: g.nodeIds, edgeList: g.edges };
    }
    const g = buildLevel2Graph(model, view.blockId);
    return { nodeIds: g.nodeIds, edgeList: g.edges };
  }, [model, view]);

  const positions = useMemo(() => layeredLayout(nodeIds, edgeList), [nodeIds, edgeList]);

  const rfNodes: RFNode<IRNodeData>[] = useMemo(
    () =>
      nodeIds.map((id) => {
        const pos = positions.get(id) ?? { x: 0, y: 0 };
        if (id === ELLIPSIS) {
          return {
            id,
            type: "ir",
            position: pos,
            draggable: false,
            data: { label: "⋯", sublabel: "more blocks (collapsed)", color: "#94a3b8", selected: false },
          };
        }
        const node = model.nodes[id];
        const info = componentRegistry[node.type];
        const paramCount = node.parameters.reduce((a, p) => a + p.logicalShape.reduce((x, y) => x * y, 1), 0);
        return {
          id,
          type: "ir",
          position: pos,
          draggable: false,
          data: {
            node,
            label: node.name,
            sublabel: paramCount > 0 ? `${paramCount.toLocaleString()} params` : info.label,
            color: info.color,
            selected: id === selectedId,
            expandable: node.type === "transformer_block",
          },
        };
      }),
    [nodeIds, positions, model, selectedId]
  );

  const rfEdges: RFEdge[] = useMemo(
    () =>
      edgeList.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.label,
        style: { stroke: "#94a3b8" },
      })),
    [edgeList]
  );

  const handleNodeClick = useCallback(
    (_: unknown, n: RFNode) => {
      if (n.id === ELLIPSIS) return;
      onSelect(n.id);
    },
    [onSelect]
  );

  const handleNodeDoubleClick = useCallback(
    (_: unknown, n: RFNode) => {
      if (n.id === ELLIPSIS) return;
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
        nodes={rfNodes}
        edges={rfEdges}
        nodeTypes={nodeTypes}
        onNodeClick={handleNodeClick}
        onNodeDoubleClick={handleNodeDoubleClick}
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
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
