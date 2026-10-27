import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Recognize } from "./Recognize";
import "./styles.css";

createRoot(document.getElementById("root")!).render(<StrictMode><Recognize /></StrictMode>);
