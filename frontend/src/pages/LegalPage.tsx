import { ArrowLeft } from "lucide-react";

import { BrandMark } from "../components/BrandMark";
import "./LegalPage.css";

type LegalKind = "privacy" | "terms";

export function LegalPage({ kind }: { kind: LegalKind }) {
  const privacy = kind === "privacy";
  return (
    <main className="mm-legal">
      <header className="mm-legal__header">
        <a className="mm-legal__brand" href="/" aria-label="Muscle Memory home"><BrandMark /><strong>Muscle Memory</strong></a>
        <a className="mm-legal__back" href="/"><ArrowLeft size={15} /> Back to system</a>
      </header>
      <article className="mm-legal__article">
        <p className="mm-section-label">Muscle Memory / {privacy ? "Privacy" : "Terms"}</p>
        <h1>{privacy ? "Privacy, kept narrow." : "Terms for the operator surface."}</h1>
        <p className="mm-legal__lede">Last updated August 4, 2026. This page describes the local operator experience in this repository and does not replace a reviewed production agreement.</p>
        {privacy ? <>
          <section><h2>What the console stores</h2><p>The console keeps the bearer credential in the current browser tab's session storage. Episode review notes are sent to the configured Muscle Memory API and stored with their episode reference, operator subject, timestamps, and tags.</p></section>
          <section><h2>What the system records</h2><p>Operational episodes contain simulator telemetry, frame identifiers, policy and world hashes, results, and provider delivery state. The robot checksum travels with each episode so evidence can be checked against the fixed MM-01 bundle.</p></section>
          <section><h2>Provider boundaries</h2><p>Provider credentials remain server-side. The frontend does not display secrets, and unavailable providers are surfaced as unavailable rather than represented by fabricated live results. Review notes are workspace annotations and do not rewrite immutable episode evidence.</p></section>
          <section><h2>Your controls</h2><p>Clear the operator credential from System Settings or close the browser tab to remove the local session token. Server-side episode and note retention is controlled by the deployment operator and its storage policy.</p></section>
        </> : <>
          <section><h2>Use of the system</h2><p>Muscle Memory is an operator and research surface for validated simulated episodes, physical evidence, and high-level task-policy evaluation. It is not a guarantee of real-world robot performance or a substitute for safety review.</p></section>
          <section><h2>Fixed boundaries</h2><p>The MM-01 body, sensors, dimensions, and frozen walking controller are part of the product contract. The learned task policy does not replace locomotion control, and promotion requires measured held-out evidence plus human approval.</p></section>
          <section><h2>Demo and live states</h2><p>The console may offer an explicit synthetic preview when a live runtime is not admitted. Preview output is labeled and must not be treated as provider-backed production evidence.</p></section>
          <section><h2>Operator responsibility</h2><p>Deployment operators are responsible for credentials, provider authorization, retention, access control, and production validation. Configure only worlds, assets, and policy artifacts that have passed the repository's stated gates.</p></section>
        </>}
      </article>
    </main>
  );
}
