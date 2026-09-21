// frontend/src/main.jsx

// This is the entry point of our React application.
// It does one thing: mount the App component into the DOM.

import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

// createRoot is the modern React 18+ way to render.
// It replaces the old ReactDOM.render() method.
createRoot(document.getElementById("root")).render(<App />);