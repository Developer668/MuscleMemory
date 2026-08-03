import {
  Activity,
  ArrowDown,
  ArrowRight,
  Check,
  ChevronRight,
  CircleGauge,
  DatabaseZap,
  Play,
  Radio,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Target,
} from "lucide-react";
import {
  motion,
  useReducedMotion,
  useScroll,
  useTransform,
} from "motion/react";
import { useRef, useState } from "react";

import { BrandMark } from "../components/BrandMark";
import { type DemoScenario, LandingWorld } from "../components/LandingWorld";

const scenarios: Array<{
  id: DemoScenario;
  label: string;
  shortLabel: string;
  message: string;
  success: string;
  clearance: string;
  speed: string;
  risk: string;
}> = [
  {
    id: "clear",
    label: "Clear route",
    shortLabel: "Clear",
    message: "Route is clean. Medicine stays level.",
    success: "92%",
    clearance: "0.46 m",
    speed: "0.62 m/s",
    risk: "Nominal",
  },
  {
    id: "laundry",
    label: "Laundry basket",
    shortLabel: "Basket",
    message: "Basket remembered. Widening the turn.",
    success: "84%",
    clearance: "0.27 m",
    speed: "0.31 m/s",
    risk: "Watching",
  },
  {
    id: "low_friction",
    label: "Low friction",
    shortLabel: "Low grip",
    message: "Low grip detected. Slowing before the turn.",
    success: "81%",
    clearance: "0.33 m",
    speed: "0.24 m/s",
    risk: "Mitigated",
  },
];

function PixelMascot({ message, onCycle }: { message: string; onCycle: () => void }) {
  return (
    <div className="mm-mascot-wrap">
      <div className="mm-mascot-bubble" role="status">
        <span>MM-BIT / 01</span>
        <strong>{message}</strong>
      </div>
      <button className="mm-mascot" type="button" onClick={onCycle} title="Change the world">
        <span className="mm-mascot__antenna" />
        <span className="mm-mascot__head">
          <i className="mm-mascot__eye mm-mascot__eye--left" />
          <i className="mm-mascot__eye mm-mascot__eye--right" />
          <i className="mm-mascot__mouth" />
        </span>
        <span className="mm-mascot__body"><i /></span>
        <span className="mm-mascot__arm mm-mascot__arm--left" />
        <span className="mm-mascot__arm mm-mascot__arm--right" />
        <span className="mm-mascot__leg mm-mascot__leg--left" />
        <span className="mm-mascot__leg mm-mascot__leg--right" />
      </button>
    </div>
  );
}

export function LandingPage() {
  const storyRef = useRef<HTMLElement>(null);
  const reduceMotion = useReducedMotion();
  const [scenario, setScenario] = useState<DemoScenario>("laundry");
  const { scrollYProgress } = useScroll({
    target: storyRef,
    offset: ["start start", "end end"],
  });

  const introOpacity = useTransform(scrollYProgress, [0, 0.12, 0.19], [1, 1, 0]);
  const introY = useTransform(scrollYProgress, [0, 0.19], [0, -46]);
  const introVisibility = useTransform(scrollYProgress, (value) => value < 0.2 ? "visible" : "hidden");
  const taskOpacity = useTransform(scrollYProgress, [0.15, 0.22, 0.38, 0.46], [0, 1, 1, 0]);
  const taskY = useTransform(scrollYProgress, [0.15, 0.24, 0.44], [46, 0, -38]);
  const taskVisibility = useTransform(scrollYProgress, (value) => value >= 0.14 && value < 0.48 ? "visible" : "hidden");
  const povOpacity = useTransform(scrollYProgress, [0.44, 0.51, 0.72, 0.79], [0, 1, 1, 0]);
  const povVisibility = useTransform(scrollYProgress, (value) => value >= 0.43 && value < 0.81 ? "visible" : "hidden");
  const memoryOpacity = useTransform(scrollYProgress, [0.76, 0.84, 0.96], [0, 1, 1]);
  const memoryVisibility = useTransform(scrollYProgress, (value) => value >= 0.75 ? "visible" : "hidden");
  const progressScale = useTransform(scrollYProgress, [0, 1], [0, 1]);

  const activeScenario = scenarios.find((item) => item.id === scenario) ?? scenarios[0];

  const runDelivery = () => {
    const story = storyRef.current;
    if (!story) return;
    const top = story.getBoundingClientRect().top + window.scrollY;
    window.scrollTo({ top: top + window.innerHeight * 1.15, behavior: reduceMotion ? "auto" : "smooth" });
  };

  const cycleScenario = () => {
    const index = scenarios.findIndex((item) => item.id === scenario);
    setScenario(scenarios[(index + 1) % scenarios.length].id);
  };

  return (
    <main className="mm-landing" id="top">
      <motion.div className="mm-progress" style={{ scaleX: progressScale }} aria-hidden="true" />

      <section className="mm-story" ref={storyRef} aria-label="MM-01 delivery story">
        <div className="mm-story__sticky">
          <LandingWorld progress={scrollYProgress} scenario={scenario} />
          <div className="mm-world-shade" aria-hidden="true" />
          <div className="mm-film-grain" aria-hidden="true" />

          <header className="mm-header">
            <a className="mm-brand" href="#top" aria-label="Muscle Memory home">
              <BrandMark />
              <strong>MuscleMemory</strong>
            </a>
            <div className="mm-header__status" aria-label="System status">
              <span><i /> Experience online</span>
              <span>MM-01</span>
            </div>
            <a className="mm-console-link" href="/console">
              Live console
              <ChevronRight size={16} />
            </a>
          </header>

          <motion.div className="mm-chapter mm-chapter--intro" style={{ opacity: introOpacity, y: introY, visibility: introVisibility }}>
            <p className="mm-kicker"><span>One robot</span><span>Many worlds</span></p>
            <h1>Experience<br /><em>that moves.</em></h1>
            <p className="mm-intro-copy">
              A household robot that turns every delivery, near miss, and human correction into a safer next run.
            </p>
            <div className="mm-intro-actions">
              <button className="mm-run" type="button" onClick={runDelivery}>
                <Play size={17} fill="currentColor" />
                Run a delivery
              </button>
              <a className="mm-text-link" href="#memory-story">
                See what it remembers <ArrowRight size={17} />
              </a>
            </div>
          </motion.div>

          <motion.div className="mm-chapter mm-chapter--task" style={{ opacity: taskOpacity, y: taskY, visibility: taskVisibility }}>
            <div className="mm-task-index">01 / THE TASK</div>
            <h2>Medicine.<br />Living room.<br /><em>No drama.</em></h2>
            <div className="mm-task-ticket">
              <Target size={19} />
              <div><span>Destination</span><strong>Resident / sofa</strong></div>
              <div><span>Arrival</span><strong>&lt; 30 sec</strong></div>
            </div>
          </motion.div>

          <motion.div className="mm-pov-layer" style={{ opacity: povOpacity, visibility: povVisibility }}>
            <div className="mm-reticle" aria-hidden="true"><span /><span /><span /><span /></div>
            <div className="mm-pov-label"><Radio size={14} /> MM-01 / STEREO COMPOSITE / DEMO</div>
            <aside className="mm-evals" aria-label="Simulated task evaluation">
              <div className="mm-evals__heading">
                <span>Live evals</span>
                <strong>EP-0048</strong>
              </div>
              <div className="mm-eval mm-eval--hero">
                <span>Projected success</span>
                <strong>{activeScenario.success}</strong>
                <i style={{ "--value": activeScenario.success } as React.CSSProperties} />
              </div>
              <div className="mm-eval-grid">
                <div><span>Clearance</span><strong>{activeScenario.clearance}</strong></div>
                <div><span>Speed</span><strong>{activeScenario.speed}</strong></div>
                <div><span>Tray tilt</span><strong>3.2°</strong></div>
                <div><span>Risk</span><strong>{activeScenario.risk}</strong></div>
              </div>
              <div className="mm-policy-action">
                <Activity size={15} />
                <span>Policy action</span>
                <strong>{scenario === "clear" ? "FORWARD" : "SLOW + TURN"}</strong>
              </div>
              <p>Interactive demonstration. Verified episode evidence lives in the console.</p>
            </aside>
          </motion.div>

          <motion.div className="mm-chapter mm-chapter--memory" id="memory-story" style={{ opacity: memoryOpacity, visibility: memoryVisibility }}>
            <div className="mm-task-index">03 / THE MEMORY</div>
            <h2>The home changes.<br /><em>The lesson stays.</em></h2>
            <div className="mm-memory-stream">
              <div><DatabaseZap size={18} /><span>LaserData</span><strong>20 Hz evidence</strong></div>
              <div><CircleGauge size={18} /><span>Failure summary</span><strong>deterministic</strong></div>
              <div><ShieldCheck size={18} /><span>Next policy</span><strong>human gated</strong></div>
            </div>
          </motion.div>

          <div className="mm-scenario-picker" role="group" aria-label="Delivery challenge">
            <span>Change the world</span>
            <div>
              {scenarios.map((item) => (
                <button
                  key={item.id}
                  className={item.id === scenario ? "is-active" : ""}
                  type="button"
                  onClick={() => setScenario(item.id)}
                  aria-pressed={item.id === scenario}
                >
                  {item.shortLabel}
                </button>
              ))}
            </div>
          </div>

          <PixelMascot message={activeScenario.message} onCycle={cycleScenario} />

          <div className="mm-story-rail" aria-hidden="true">
            <span>ORIENT</span><i /><span>DELIVER</span><i /><span>SEE</span><i /><span>REMEMBER</span>
          </div>
          <ArrowDown className="mm-down" size={18} aria-hidden="true" />
        </div>
      </section>

      <section className="mm-proof-strip" aria-label="Promotion standard">
        <div className="mm-proof-strip__lead">
          <span>Measured progress only</span>
          <h2>A new behavior earns its name.</h2>
        </div>
        <div className="mm-proof-number"><strong>20</strong><span>frozen held-out worlds</span></div>
        <div className="mm-proof-number"><strong>0</strong><span>falls permitted</span></div>
        <div className="mm-proof-number"><strong>+20</strong><span>point success lift</span></div>
        <a href="/console"><CircleGauge size={18} /> Inspect evidence <ArrowRight size={17} /></a>
      </section>

      <section className="mm-room-poster" aria-label="MM-01 living room simulation">
        <div className="mm-room-poster__image" role="img" aria-label="Rendered living room from the MM-01 simulation" />
        <div className="mm-room-poster__caption">
          <span>World 07 / warm apartment</span>
          <strong>Same robot.<br />Different room.</strong>
          <p>Furniture can move. Friction can change. The body and walking controller do not.</p>
        </div>
      </section>

      <section className="mm-finale" aria-labelledby="finale-title">
        <div className="mm-finale__spark"><Sparkles size={22} /></div>
        <p>One robot. Many worlds.</p>
        <h2 id="finale-title">
          Give a robot a task<br />and it completes a run.<br />Give it <em>memory</em><br />and every run compounds.
        </h2>
        <strong className="mm-finale__name">MuscleMemory</strong>
        <div className="mm-finale__actions">
          <a className="mm-run mm-run--dark" href="/console">Open live console <ArrowRight size={18} /></a>
          <a className="mm-reset" href="#top"><RotateCcw size={17} /> Replay</a>
        </div>
        <div className="mm-finale__checks">
          <span><Check size={14} /> Fixed MM-01</span>
          <span><Check size={14} /> Evidence-backed learning</span>
          <span><Check size={14} /> Human-gated promotion</span>
        </div>
      </section>

      <footer className="mm-footer">
        <a className="mm-brand" href="#top"><BrandMark /><strong>MuscleMemory</strong></a>
        <span>Safe household delivery, remembered.</span>
        <span>MM-01 / 2026</span>
      </footer>
    </main>
  );
}
