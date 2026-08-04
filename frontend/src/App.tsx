import { lazy, Suspense, useEffect } from "react";

import { LandingPage } from "./pages/LandingPage";
import { LegalPage } from "./pages/LegalPage";

const CommandCenter = lazy(async () => {
  const module = await import("./pages/CommandCenter");
  return { default: module.CommandCenter };
});

export default function App() {
  const pathname = window.location.pathname;
  const operator = ["/console", "/app"].includes(pathname);
  const legalKind = pathname === "/privacy" ? "privacy" : pathname === "/terms" ? "terms" : null;
  useEffect(() => {
    document.title = operator
      ? "MM-01 Operations | Muscle Memory"
      : legalKind
        ? `${legalKind === "privacy" ? "Privacy" : "Terms"} | Muscle Memory`
      : "Muscle Memory | One robot. Many worlds.";
  }, [legalKind, operator]);
  if (legalKind) return <LegalPage kind={legalKind} />;
  return operator ? (
    <Suspense fallback={<main className="app-loading" aria-label="Loading operations console" />}>
      <CommandCenter />
    </Suspense>
  ) : <LandingPage />;
}
