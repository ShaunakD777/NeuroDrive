import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans-condensed/500.css";
import "@fontsource/ibm-plex-sans-condensed/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { useLive } from "@prod/state/live";
import { usePlayback } from "@prod/state/playback";
import { useSelection } from "@prod/state/selection";

import { App } from "./App";
import { getWorld } from "./source/world";
import { useApp } from "./state/store";
import "./theme/tokens.css";
import "./theme/base.css";
import "./theme/environment.css";
import "./theme/network.css";
import "./theme/experiment.css";

// For the scripts in scripts/ and for poking at the prototype from the browser console:
// `__nd.setState({ theme: "dark", t: 200 })`, `__nd.getState().selectCar(30)`, `__ndWorld()` (the
// snapshot the views draw), `__ndProd.useLive.getState()` (production's live store, live source).
Object.assign(window, { __nd: useApp, __ndWorld: getWorld, __ndProd: { useLive, useSelection, usePlayback } });

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
