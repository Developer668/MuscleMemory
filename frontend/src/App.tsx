import { lazy, Suspense, useEffect } from "react";

import { LandingPage } from "./pages/LandingPage";

const CommandCenter = lazy(async () => {
  const module = await import("./pages/CommandCenter");
  return { default: module.CommandCenter };
});

export default function App() {
  const operator = ["/console", "/app"].includes(window.location.pathname);
  useEffect(() => {
    document.title = operator
      ? "MM-01 Operations | Muscle Memory"
      : "Muscle Memory | One robot. Many worlds.";
  }, [operator]);
  return operator ? (
    <Suspense fallback={<main className="app-loading" aria-label="Loading operations console" />}>
      <CommandCenter />
    </Suspense>
  ) : <LandingPage />;
}
