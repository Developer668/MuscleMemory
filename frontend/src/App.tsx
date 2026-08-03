import { useEffect } from "react";

import { LandingPage } from "./pages/LandingPage";
import { CommandCenter } from "./pages/CommandCenter";

export default function App() {
  const operator = ["/console", "/app"].includes(window.location.pathname);
  useEffect(() => {
    document.title = operator
      ? "MM-01 Operations | Muscle Memory"
      : "Muscle Memory | One robot. Many worlds.";
  }, [operator]);
  return operator ? <CommandCenter /> : <LandingPage />;
}
