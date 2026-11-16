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
import { useApp } from "./state/store";
import "./theme/tokens.css";
import "./theme/base.css";
import "./theme/environment.css";
import "./theme/network.css";
import "./theme/experiment.css";

// For the screenshot script and for poking at the prototype from the browser console:
// `__nd.setState({ theme: "dark", t: 200 })`, `__nd.getState().selectCar(30)`.
(window as unknown as { __nd: typeof useApp }).__nd = useApp;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
