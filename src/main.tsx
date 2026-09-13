import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource-variable/outfit";
import "./index.css";
import App from "./App";
import CaptureLab from "./features/capture-lab/CaptureLab";

const normalizedPath = window.location.pathname.replace(/\/+$/, "") || "/";
const RootView = normalizedPath === "/capture-lab" ? CaptureLab : App;

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RootView />
  </React.StrictMode>,
);
