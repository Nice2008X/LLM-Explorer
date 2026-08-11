import { useState } from "react";
import type { Model, ModelNode } from "@llm-explorer/model-ir";

interface Props {
  model: Model;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function ModelTree({ model, selectedId, onSelect }: Props) {
  return (
    <div className="model-tree">
      <TreeNode model={model} nodeId={model.rootId} depth={0} selectedId={selectedId} onSelect={onSelect} />
    </div>
  );
}

function TreeNode({
  model,
  nodeId,
  depth,
  selectedId,
  onSelect,
}: {
  model: Model;
  nodeId: string;
  depth: number;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const node: ModelNode = model.nodes[nodeId];
  const hasChildren = node.children.length > 0;
  // collapse transformer blocks below block 1 by default so a 32-layer model doesn't flood the tree
  const [open, setOpen] = useState(depth < 2);

  return (
    <div>
      <div
        className={"tree-row" + (nodeId === selectedId ? " selected" : "")}
        style={{ paddingLeft: 8 + depth * 14 }}
        onClick={() => onSelect(nodeId)}
      >
        {hasChildren ? (
          <button
            className="tree-toggle"
            onClick={(e) => {
              e.stopPropagation();
              setOpen((o) => !o);
            }}
          >
            {open ? "▾" : "▸"}
          </button>
        ) : (
          <span className="tree-toggle-spacer" />
        )}
        <span className="tree-label">{node.name}</span>
        {node.parameters.length > 0 && <span className="tree-param-dot" title="has weights" />}
      </div>
      {hasChildren && open && (
        <div>
          {node.children.map((childId) => (
            <TreeNode key={childId} model={model} nodeId={childId} depth={depth + 1} selectedId={selectedId} onSelect={onSelect} />
          ))}
        </div>
      )}
    </div>
  );
}
