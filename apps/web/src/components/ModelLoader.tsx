import { useMemo, useState } from "react";
import { PRESET_MODELS } from "../adapters.js";
import { useTranslation } from "./LanguageContext.js";

interface Props {
  status: "idle" | "loading" | "ready" | "error";
  error?: string;
  onLoad: (repo: string) => void;
}

export function ModelLoader({ status, error, onLoad }: Props) {
  const { t } = useTranslation();
  const [repo, setRepo] = useState(PRESET_MODELS[0].repo);
  const sortedPresets = useMemo(() => [...PRESET_MODELS].sort((a, b) => a.label.localeCompare(b.label)), []);

  return (
    <div className="model-loader">
      <div className="model-loader-title">{t("loader.title")}</div>
      <div className="model-loader-sub">{t("loader.subtitle")}</div>
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
