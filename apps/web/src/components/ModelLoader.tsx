import { useMemo, useState } from "react";
import { PRESET_MODELS } from "../adapters.js";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  status: "idle" | "loading" | "ready" | "error";
  error?: string;
  onLoad: (repo: string) => void;
  /** Presets matching this repo id are left out of the list — used when re-opening the loader for a model that's already loaded, so it isn't offered back as if it were a fresh option. */
  excludeRepo?: string;
  /** Drops the built-in title/subtitle and card chrome (background/border/padding) — used when this is embedded inside a panel that already provides its own header, e.g. the "load a different model" popover. */
  embedded?: boolean;
}

export function ModelLoader({ status, error, onLoad, excludeRepo, embedded }: Props) {
  const { t } = useTranslation();
  const sortedPresets = useMemo(
    () => PRESET_MODELS.filter((p) => p.repo !== excludeRepo).sort((a, b) => a.label.localeCompare(b.label)),
    [excludeRepo]
  );
  const [repo, setRepo] = useState(() => PRESET_MODELS.find((p) => p.repo !== excludeRepo)?.repo ?? "");

  return (
    <div className={"model-loader" + (embedded ? " embedded" : "")}>
      {!embedded && (
        <>
          <div className="model-loader-title">{t("loader.title")}</div>
          <div className="model-loader-sub">{t("loader.subtitle")}</div>
        </>
      )}
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
          <button key={p.repo} className="preset-chip" onClick={() => { setRepo(p.repo); onLoad(p.repo); }}>
            {p.label}
          </button>
        ))}
      </div>
      {status === "error" && <div className="model-loader-error">{error}</div>}
    </div>
  );
}
