import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App.js";
import { LanguageProvider } from "./components/LanguageContext.js";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <LanguageProvider>
      <App />
    </LanguageProvider>
  </React.StrictMode>
);
