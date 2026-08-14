import { useMemo, useState } from "react";
import { PRESET_MODELS } from "../adapters.js";
import { useTranslation } from "./LanguageContext.js";

export interface LocalModelFiles {
  name: string;
  config: File;
  weights: File;
  tokenizer?: File;
}

interface Props {
  status: "idle" | "loading" | "ready" | "error";
  error?: string;
  onLoad: (repo: string) => void;
  onLoadLocal: (files: LocalModelFiles) => void;
  /** Presets matching this repo id are left out of the list — used when re-opening the loader for a model that's already loaded, so it isn't offered back as if it were a fresh option. */
  excludeRepo?: string;
  /** Drops the built-in title/subtitle and card chrome (background/border/padding) — used when this is embedded inside a panel that already provides its own header, e.g. the "load a different model" popover. */
  embedded?: boolean;
}

type SourceMode = "huggingface" | "local";

/** Strips a trailing `.safetensors` so a file named `model.safetensors` becomes the model's display name `model`, not a name with a stray extension. */
function defaultModelName(filename: string): string {
  return filename.replace(/\.safetensors$/i, "");
}

export function ModelLoader({ status, error, onLoad, onLoadLocal, excludeRepo, embedded }: Props) {
  const { t } = useTranslation();
  const sortedPresets = useMemo(
    () => PRESET_MODELS.filter((p) => p.repo !== excludeRepo).sort((a, b) => a.label.localeCompare(b.label)),
    [excludeRepo]
  );
  const [repo, setRepo] = useState(() => PRESET_MODELS.find((p) => p.repo !== excludeRepo)?.repo ?? "");
  const [mode, setMode] = useState<SourceMode>("huggingface");
  const [configFile, setConfigFile] = useState<File | null>(null);
  const [weightsFile, setWeightsFile] = useState<File | null>(null);
  const [tokenizerFile, setTokenizerFile] = useState<File | null>(null);

  const canLoadLocal = !!configFile && !!weightsFile && status !== "loading";

  return (
    <div className={"model-loader" + (embedded ? " embedded" : "")}>
      {!embedded && (
        <>
          <div className="model-loader-title">{t("loader.title")}</div>
          <div className="model-loader-sub">{t("loader.subtitle")}</div>
        </>
      )}
      <div className="model-loader-source-tabs">
        <button type="button" className={mode === "huggingface" ? "active" : ""} onClick={() => setMode("huggingface")}>
          {t("loader.sourceHf")}
        </button>
        <button type="button" className={mode === "local" ? "active" : ""} onClick={() => setMode("local")}>
          {t("loader.sourceLocal")}
        </button>
      </div>

      {mode === "huggingface" ? (
        <>
          <form
            className="model-loader-form"
            onSubmit={(e) => {
              e.preventDefault();
              onLoad(repo);
            }}
          >
            <input value={repo} onChange={(e) => setRepo(e.target.value)} placeholder={t("loader.inputPlaceholder")} />
            <button type="submit" disabled={status === "loading"}>
              {status === "loading" ? t("loader.loading") : t("loader.load")}
            </button>
          </form>
          <div className="model-loader-presets">
            {sortedPresets.map((p) => (
              <button
                key={p.repo}
                className="preset-chip"
                onClick={() => {
                  setRepo(p.repo);
                  onLoad(p.repo);
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </>
      ) : (
        <form
          className="model-loader-local-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (!configFile || !weightsFile) return;
            onLoadLocal({ name: defaultModelName(weightsFile.name), config: configFile, weights: weightsFile, tokenizer: tokenizerFile ?? undefined });
          }}
        >
          <p className="model-loader-local-hint">{t("loader.localHint")}</p>
          <label className="model-loader-file-row">
            <span className="model-loader-file-label">{t("loader.localConfig")}</span>
            <span className="model-loader-file-desc">{t("loader.localConfigDesc")}</span>
            <input type="file" accept=".json" onChange={(e) => setConfigFile(e.target.files?.[0] ?? null)} />
          </label>
          <label className="model-loader-file-row">
            <span className="model-loader-file-label">{t("loader.localWeights")}</span>
            <span className="model-loader-file-desc">{t("loader.localWeightsDesc")}</span>
            <input type="file" accept=".safetensors" onChange={(e) => setWeightsFile(e.target.files?.[0] ?? null)} />
          </label>
          <label className="model-loader-file-row">
            <span className="model-loader-file-label">{t("loader.localTokenizer")}</span>
            <span className="model-loader-file-desc">{t("loader.localTokenizerDesc")}</span>
            <input type="file" accept=".json" onChange={(e) => setTokenizerFile(e.target.files?.[0] ?? null)} />
          </label>
          <button type="submit" disabled={!canLoadLocal}>
            {status === "loading" ? t("loader.loading") : t("loader.load")}
          </button>
        </form>
      )}
      {status === "error" && <div className="model-loader-error">{error}</div>}
    </div>
  );
}
