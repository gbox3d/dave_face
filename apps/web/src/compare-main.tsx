import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Compare } from "./Compare";
import "./styles.css";

createRoot(document.getElementById("root")!).render(<StrictMode><Compare /></StrictMode>);
