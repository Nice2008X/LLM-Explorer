import { useCallback, useMemo } from "react";
import ReactFlow, { Background, Controls, Handle, Position, type Edge as RFEdge, type Node as RFNode } from "reactflow";
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

  return (
    <div className="architecture-graph">
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
