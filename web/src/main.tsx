// IBM Plex, bundled from npm (ADR-063, D3): the files are part of the build, nothing is fetched
// from the internet at run time. Sans for text, Condensed for small capital labels, Mono for numbers.
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans-condensed/500.css";
import "@fontsource/ibm-plex-sans-condensed/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { applyTheme } from "./state/layout";
import "./styles/index.css";

// Set <html data-theme> before the first render, so the page never flashes the wrong theme.
applyTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
