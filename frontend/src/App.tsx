import { useEffect } from "react";

import { LandingPage } from "./pages/LandingPage";
import { OperatorConsole } from "./pages/OperatorConsole";

export default function App() {
  const about = window.location.pathname === "/about";
  useEffect(() => {
    document.title = about
      ? "Muscle Memory | One robot. Many worlds."
      : "MM-01 Operations | Muscle Memory";
  }, [about]);
  return about ? <LandingPage /> : <OperatorConsole />;
}
