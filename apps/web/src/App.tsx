import { useEffect, useState } from "react";
import { useModel } from "./useModel.js";
import { useInference } from "./useInference.js";
import { useLocalStorageState } from "./useLocalStorageState.js";
import { useTheme } from "./components/ThemeSwitcher.js";
import { useTranslation } from "./components/LanguageContext.js";
import { SettingsButton, SettingsPanel } from "./components/SettingsPanel.js";
import { ModelLoader } from "./components/ModelLoader.js";
import { ModelInfoBar } from "./components/ModelInfoBar.js";
import { ModelTree } from "./components/ModelTree.js";
import { ArchitectureGraph, type GraphView } from "./components/ArchitectureGraph.js";
import { Inspector } from "./components/Inspector.js";
import { TensorExplorer } from "./components/TensorExplorer.js";
import { InferencePanel } from "./components/InferencePanel.js";
import { LogitLensPanel } from "./components/LogitLensPanel.js";
import { TokenAttributionPanel } from "./components/TokenAttributionPanel.js";
import { ExperimentPanel } from "./components/ExperimentPanel.js";

type BottomTab = "tensor" | "logitlens" | "attribution" | "experiment";

export function App() {
  const { state, load } = useModel();
  const { theme, setTheme } = useTheme();
  const { t } = useTranslation();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<GraphView>({ kind: "architecture" });
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(null);
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [bottomTab, setBottomTab] = useState<BottomTab>("tensor");
  const [treeCollapsed, setTreeCollapsed] = useLocalStorageState("panel:tree-collapsed", false);
  const [inspectorCollapsed, setInspectorCollapsed] = useLocalStorageState("panel:inspector-collapsed", false);
  const [bottomCollapsed, setBottomCollapsed] = useLocalStorageState("panel:bottom-collapsed", false);

  const inference = useInference(state.model, state.weightProvider, state.adapter, state.tokenizer);
  const promptB = useInference(state.model, state.weightProvider, state.adapter, state.tokenizer);

  // A different model can have completely different node ids (fewer/more
  // blocks, different architecture) — stale selection/view referencing the
  // old model's ids would otherwise crash ArchitectureGraph's breadcrumb.
  useEffect(() => {
    setSelectedId(null);
    setView({ kind: "architecture" });
    setSelectedTokenIndex(null);
    setBottomTab("tensor");
    inference.reset();
    promptB.reset();
  }, [state.model]);

  if (state.status !== "ready" || !state.model || !state.weightProvider) {
    return (
      <div className="app-loader-screen">
        <div className="top-right-controls">
          <SettingsButton open={settingsOpen} onToggle={() => setSettingsOpen((v) => !v)} />
          <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} theme={theme} onThemeChange={setTheme} />
        </div>
        <ModelLoader status={state.status} error={state.error} onLoad={load} />
      </div>
    );
  }

  const model = state.model;
  const selectedNode = selectedId ? model.nodes[selectedId] ?? null : null;
  // Defends against the one render where a just-finished model load has
  // landed but `view`/`selectedId` haven't been reset to match yet (the
  // effect above runs after this render, not before it) — without this, a
  // stale blockId from the previous model would crash ArchitectureGraph.
  const safeView: GraphView = view.kind === "block" && !model.nodes[view.blockId] ? { kind: "architecture" } : view;

  const hasResult = inference.state.status === "ready" && !!inference.state.result;
  const analysisTabsEnabled = hasResult && !!state.adapter?.runInference;

  return (
    <div className="app">
      <ModelInfoBar model={model} />
      <div className="top-right-controls">
        <button
          className="load-different"
          onClick={() => {
            const repo = promptForRepo();
            if (repo) load(repo);
          }}
        >
          {t("app.loadDifferentModel")}
        </button>
        <SettingsButton open={settingsOpen} onToggle={() => setSettingsOpen((v) => !v)} />
        <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} theme={theme} onThemeChange={setTheme} />
      </div>
      <InferencePanel
        supported={!!state.tokenizer}
        state={inference.state}
        onRun={inference.run}
        selectedTokenIndex={selectedTokenIndex}
        onSelectToken={setSelectedTokenIndex}
        compareEnabled={compareEnabled}
        onToggleCompare={() => setCompareEnabled((v) => !v)}
        promptBState={promptB.state}
        onRunB={promptB.run}
      />
      <div className="app-body">
        <aside className={"pane pane-tree" + (treeCollapsed ? " collapsed" : "")}>
          <div className="pane-header">
            {!treeCollapsed && <span className="pane-header-title">{t("app.modelTree")}</span>}
            <button className="pane-collapse-btn" onClick={() => setTreeCollapsed((v) => !v)} title={treeCollapsed ? t("app.expandTree") : t("app.collapseTree")}>
              {treeCollapsed ? "›" : "‹"}
            </button>
          </div>
          {treeCollapsed ? (
            <span className="pane-vertical-label">{t("app.modelTree")}</span>
          ) : (
            <div className="pane-tree-body">
              <ModelTree model={model} selectedId={selectedId} onSelect={setSelectedId} />
            </div>
          )}
        </aside>
        <main className="pane pane-graph">
          <ArchitectureGraph
            model={model}
            view={safeView}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onEnterBlock={(blockId) => {
              setView({ kind: "block", blockId });
              setSelectedId(blockId);
            }}
            onExitBlock={() => setView({ kind: "architecture" })}
          />
        </main>
        <aside className={"pane pane-inspector" + (inspectorCollapsed ? " collapsed" : "")}>
          <div className="pane-header">
            <button className="pane-collapse-btn" onClick={() => setInspectorCollapsed((v) => !v)} title={inspectorCollapsed ? t("app.expandInspector") : t("app.collapseInspector")}>
              {inspectorCollapsed ? "‹" : "›"}
            </button>
            {!inspectorCollapsed && <span className="pane-header-title">{t("app.inspector")}</span>}
          </div>
          {inspectorCollapsed ? (
            <span className="pane-vertical-label">{t("app.inspector")}</span>
          ) : (
            <div className="pane-inspector-body">
              <Inspector node={selectedNode} />
            </div>
          )}
        </aside>
      </div>
      <section className={"pane pane-tensor" + (bottomCollapsed ? " collapsed" : "")}>
        <div className="bottom-tabs">
          <button className={bottomTab === "tensor" ? "active" : ""} onClick={() => setBottomTab("tensor")}>
            {t("app.tensorExplorer")}
          </button>
          <button className={bottomTab === "logitlens" ? "active" : ""} disabled={!analysisTabsEnabled} onClick={() => setBottomTab("logitlens")} title={!analysisTabsEnabled ? t("app.runForwardPassFirst") : undefined}>
            {t("app.logitLens")}
          </button>
          <button className={bottomTab === "attribution" ? "active" : ""} disabled={!analysisTabsEnabled} onClick={() => setBottomTab("attribution")} title={!analysisTabsEnabled ? t("app.runForwardPassFirst") : undefined}>
            {t("app.tokenAttribution")}
          </button>
          <button className={bottomTab === "experiment" ? "active" : ""} disabled={!analysisTabsEnabled} onClick={() => setBottomTab("experiment")} title={!analysisTabsEnabled ? t("app.runForwardPassFirst") : undefined}>
            {t("app.experiment")}
          </button>
          <span className="bottom-tabs-spacer" />
          <button className="bottom-collapse-btn" onClick={() => setBottomCollapsed((v) => !v)} title={bottomCollapsed ? t("app.expandPanel") : t("app.collapsePanel")}>
            {bottomCollapsed ? "▴" : "▾"}
          </button>
        </div>

        {!bottomCollapsed && bottomTab === "tensor" && (
          <TensorExplorer
            model={model}
            weightProvider={state.weightProvider}
            selectedNode={selectedNode}
            inference={inference.state}
            selectedTokenIndex={selectedTokenIndex}
            promptBInference={promptB.state}
          />
        )}
        {!bottomCollapsed && bottomTab === "logitlens" && analysisTabsEnabled && state.tokenizer && (
          <LogitLensPanel model={model} weightProvider={state.weightProvider} capture={inference.state.result!} tokenizer={state.tokenizer} />
        )}
        {!bottomCollapsed && bottomTab === "attribution" && analysisTabsEnabled && state.tokenizer && (
          <TokenAttributionPanel model={model} weightProvider={state.weightProvider} adapter={state.adapter!} tokenIds={inference.state.result!.tokenIds} tokenizer={state.tokenizer} />
        )}
        {!bottomCollapsed && bottomTab === "experiment" && analysisTabsEnabled && state.tokenizer && (
          <ExperimentPanel
            model={model}
            weightProvider={state.weightProvider}
            adapter={state.adapter!}
            tokenizer={state.tokenizer}
            selectedNode={selectedNode}
            mainTokenIds={inference.state.result!.tokenIds}
            mainResult={inference.state.result!}
            promptBResult={promptB.state.result ?? null}
          />
        )}
      </section>
    </div>
  );
}

function promptForRepo(): string {
  return window.prompt("Hugging Face model id (GPT-2 or Llama architecture, needs a model.safetensors file):", "hf-internal-testing/tiny-random-LlamaForCausalLM") ?? "";
}
