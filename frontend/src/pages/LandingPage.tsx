import {
  Activity,
  ArrowDown,
  ArrowRight,
  Check,
  ChevronRight,
  CircleGauge,
  DatabaseZap,
  Droplets,
  Eye,
  Gauge,
  Play,
  Radio,
  RotateCcw,
  ShieldCheck,
  Target,
  Workflow,
} from "lucide-react";
import {
  motion,
  useReducedMotion,
  useScroll,
  useTransform,
  type MotionValue,
} from "motion/react";
import { useRef } from "react";

import { BrandMark } from "../components/BrandMark";
import { LandingWorld } from "../components/LandingWorld";

function useChapter(
  progress: MotionValue<number>,
  enter: number,
  holdStart: number,
  holdEnd: number,
  exit: number,
) {
  const opacity = useTransform(progress, [enter, holdStart, holdEnd, exit], [0, 1, 1, 0]);
  const y = useTransform(progress, [enter, holdStart, holdEnd, exit], [36, 0, 0, -28]);
  const visibility = useTransform(progress, (value) =>
    value >= enter && value < exit ? "visible" : "hidden",
  );
  return { opacity, y, visibility };
}

export function LandingPage() {
  const storyRef = useRef<HTMLElement>(null);
  const reduceMotion = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: storyRef,
    offset: ["start start", "end end"],
  });

  const introOpacity = useTransform(scrollYProgress, [0, 0.055, 0.09], [1, 1, 0]);
  const introY = useTransform(scrollYProgress, [0, 0.09], [0, -28]);
  const introVisibility = useTransform(scrollYProgress, (value) => value < 0.09 ? "visible" : "hidden");
  const problemStyle = useChapter(scrollYProgress, 0.07, 0.095, 0.14, 0.17);
  const missionStyle = useChapter(scrollYProgress, 0.145, 0.17, 0.215, 0.24);
  const perceptionStyle = useChapter(scrollYProgress, 0.22, 0.245, 0.285, 0.31);
  const actionStyle = useChapter(scrollYProgress, 0.29, 0.32, 0.37, 0.4);
  const memoryStyle = useChapter(scrollYProgress, 0.39, 0.42, 0.465, 0.49);
  const kitchenStyle = useChapter(scrollYProgress, 0.48, 0.515, 0.57, 0.6);
  const washStyle = useChapter(scrollYProgress, 0.585, 0.62, 0.69, 0.72);
  const returnStyle = useChapter(scrollYProgress, 0.705, 0.74, 0.79, 0.82);
  const evaluationStyle = useChapter(scrollYProgress, 0.805, 0.84, 0.885, 0.91);
  const finaleOpacity = useTransform(scrollYProgress, [0.9, 0.94, 1], [0, 1, 1]);
  const finaleY = useTransform(scrollYProgress, [0.9, 0.945], [38, 0]);
  const finaleVisibility = useTransform(scrollYProgress, (value) => value >= 0.895 ? "visible" : "hidden");
  const povOpacity = useTransform(scrollYProgress, [0.22, 0.245, 0.29, 0.315], [0, 1, 1, 0]);
  const progressScale = useTransform(scrollYProgress, [0, 1], [0, 1]);

  const runDelivery = () => {
    const story = storyRef.current;
    if (!story) return;
    const top = story.getBoundingClientRect().top + window.scrollY;
    window.scrollTo({
      top: top + window.innerHeight * 1.25,
      behavior: reduceMotion ? "auto" : "smooth",
    });
  };

  return (
    <main className="mm-landing" id="top">
      <a className="mm-skip-link" href="#story-start">Skip to experience</a>
      <motion.div className="mm-progress" style={{ scaleX: progressScale }} aria-hidden="true" />

      <section className="mm-story" id="story-start" ref={storyRef} aria-label="How Muscle Memory works">
        <div className="mm-story__sticky">
          <LandingWorld progress={scrollYProgress} reducedMotion={Boolean(reduceMotion)} />
          <div className="mm-world-shade" aria-hidden="true" />
          <div className="mm-film-grain" aria-hidden="true" />

          <header className="mm-header">
            <a className="mm-brand" href="#top" aria-label="Muscle Memory home">
              <BrandMark />
              <strong>Muscle Memory</strong>
            </a>
            <div className="mm-header__status" aria-label="System status">
              <span><i /> Experience online</span>
              <span>MM-01 / fixed identity</span>
            </div>
            <a className="mm-console-link" href="/console">Live console <ChevronRight size={16} /></a>
          </header>

          <motion.section className="mm-story-card mm-story-card--intro" style={{ opacity: introOpacity, y: introY, visibility: introVisibility }}>
            <p className="mm-kicker"><span>Physical intelligence</span><span>Built from experience</span></p>
            <h1>Teach agents<br /><em>how the world feels.</em></h1>
            <p className="mm-intro-copy">Generate grounded physical data, turn real interactions into reusable experience, and train RL policies with evidence they can trust.</p>
            <div className="mm-intro-actions">
              <button className="mm-run" type="button" onClick={runDelivery}><Play size={17} fill="currentColor" />Start the run</button>
              <span className="mm-scroll-hint">Scroll to follow MM-01 <ArrowDown size={15} /></span>
            </div>
          </motion.section>

          <motion.article className="mm-story-card mm-story-card--left mm-story-card--problem" style={problemStyle}>
            <div className="mm-card-index"><Gauge size={16} /> The bottleneck</div>
            <h2>Agents can reason.<br /><em>Physics is missing.</em></h2>
            <p>Language describes a task. It does not provide the contact, clearance, friction, or failure data an RL policy needs to learn it.</p>
          </motion.article>

          <motion.article className="mm-story-card mm-story-card--right" style={missionStyle}>
            <div className="mm-card-index"><Target size={16} /> 01 / Delivery</div>
            <h2>Carry medicine.<br /><em>Keep it level.</em></h2>
            <p>The task policy chooses only forward speed, turning rate, or stop. The fixed walking controller handles locomotion.</p>
            <dl className="mm-card-facts">
              <div><dt>Destination</dt><dd>Resident / sofa</dd></div>
              <div><dt>Time limit</dt><dd>30 seconds</dd></div>
              <div><dt>Tray tilt</dt><dd>Below 12°</dd></div>
            </dl>
          </motion.article>

          <motion.div className="mm-pov-layer" style={{ opacity: povOpacity }} aria-hidden="true">
            <div className="mm-reticle"><span /><span /><span /><span /></div>
            <div className="mm-pov-label"><Radio size={13} /> STEREO COMPOSITE / DERIVED DEPTH</div>
          </motion.div>

          <motion.article className="mm-story-card mm-story-card--left" style={perceptionStyle}>
            <div className="mm-card-index"><Eye size={16} /> 02 / Perception</div>
            <h2>See the room.<br /><em>Read the gap.</em></h2>
            <p>Stereo vision becomes 48 depth sectors. The policy sees nearby free space—not a precomputed route.</p>
            <div className="mm-signal-row"><span><i /> Stereo RGB</span><span><i /> Depth sectors</span><span><i /> Tray state</span></div>
          </motion.article>

          <motion.article className="mm-story-card mm-story-card--right" style={actionStyle}>
            <div className="mm-card-index"><Activity size={16} /> 03 / Avoid</div>
            <h2>The table moves<br /><em>the path.</em></h2>
            <p>The robot slows, takes the open edge of the dining area, and keeps its carried medicine stable.</p>
            <dl className="mm-card-facts mm-card-facts--action">
              <div><dt>Forward</dt><dd>0.31 m/s</dd></div>
              <div><dt>Clearance</dt><dd>0.27 m</dd></div>
              <div><dt>Risk</dt><dd>Watching</dd></div>
            </dl>
          </motion.article>

          <motion.article className="mm-story-card mm-story-card--left" style={memoryStyle}>
            <div className="mm-card-index"><DatabaseZap size={16} /> 04 / Remember</div>
            <h2>The run ends.<br /><em>The lesson stays.</em></h2>
            <p>Telemetry links the world, obstacle, action, outcome, correction, and immutable policy version.</p>
            <div className="mm-memory-flow"><span>Episode</span><i /><span>Contact</span><i /><span>Lesson</span><i /><span>Candidate</span></div>
          </motion.article>

          <motion.article className="mm-story-card mm-story-card--right" style={kitchenStyle}>
            <div className="mm-card-index"><Workflow size={16} /> 05 / New task</div>
            <h2>Now, the<br /><em>kitchen.</em></h2>
            <p>The same policy interface transfers to a different goal: reach the sink without rewriting locomotion.</p>
            <dl className="mm-card-facts"><div><dt>Goal</dt><dd>Kitchen sink</dd></div><div><dt>Body</dt><dd>Unchanged</dd></div><div><dt>Context</dt><dd>Reused</dd></div></dl>
          </motion.article>

          <motion.article className="mm-story-card mm-story-card--left" style={washStyle}>
            <div className="mm-card-index"><Droplets size={16} /> 06 / Interaction</div>
            <h2>Observe contact.<br /><em>Record the work.</em></h2>
            <p>At the sink, the scene captures timing, reach, contact, and task completion as structured physical evidence.</p>
            <div className="mm-signal-row"><span><i /> Contact events</span><span><i /> Tool pose</span><span><i /> Completion state</span></div>
          </motion.article>

          <motion.article className="mm-story-card mm-story-card--right" style={returnStyle}>
            <div className="mm-card-index"><RotateCcw size={16} /> 07 / Return</div>
            <h2>Finish cleanly.<br /><em>Reset with context.</em></h2>
            <p>MM-01 returns to the center. The completed task becomes another training episode instead of disposable footage.</p>
          </motion.article>

          <motion.article className="mm-story-card mm-story-card--left" style={evaluationStyle}>
            <div className="mm-card-index"><ShieldCheck size={16} /> 08 / Prove</div>
            <h2>Improvement is<br /><em>measured.</em></h2>
            <p>A candidate runs without A* on frozen held-out worlds. Promotion remains a human decision.</p>
            <div className="mm-gate-grid"><div><strong>20</strong><span>held-out worlds</span></div><div><strong>0</strong><span>falls allowed</span></div><div><CircleGauge size={20} /><span>promotion gate</span></div></div>
          </motion.article>

          <motion.section className="mm-story-card mm-story-card--finale" style={{ opacity: finaleOpacity, y: finaleY, visibility: finaleVisibility }}>
            <span className="mm-finale-label"><Check size={14} /> Two tasks recorded</span>
            <h2>Generate data.<br /><em>Build faster.</em></h2>
            <p>More accurate physical experience for agents and the RL models they train.</p>
            <div className="mm-finale-actions"><a className="mm-run" href="/console">Open live console <ArrowRight size={17} /></a><a className="mm-replay" href="#top"><RotateCcw size={15} /> Replay</a></div>
          </motion.section>

          <nav className="mm-chapter-rail" aria-label="Story progress"><span>Need</span><i /><span>Carry</span><i /><span>See</span><i /><span>Avoid</span><i /><span>Wash</span><i /><span>Prove</span></nav>
        </div>
      </section>

      <section className="mm-problem" aria-labelledby="physical-data-title">
        <p className="mm-section-label">Why Muscle Memory exists</p>
        <div className="mm-problem__lead">
          <h2 id="physical-data-title">AI has a<br />physical-data problem.</h2>
          <p>Agents can plan in words, but embodied policies learn from forces, surfaces, geometry, timing, and failure. That evidence is expensive to collect and usually disappears inside disconnected runs.</p>
        </div>
        <div className="mm-contrast" aria-label="From language descriptions to physical evidence">
          <div><span>Agents receive</span><strong>“Take this to the kitchen.”</strong><p>A clear instruction, with none of the physical detail needed to execute it safely.</p></div>
          <ArrowRight aria-hidden="true" />
          <div><span>RL policies need</span><strong>Clearance. Contact. Friction. Recovery.</strong><p>Grounded, synchronized evidence they can train on and evaluators can inspect.</p></div>
        </div>
      </section>

      <section className="mm-engine" aria-labelledby="engine-title">
        <div className="mm-engine__intro"><p className="mm-section-label">One evidence loop</p><h2 id="engine-title">From task to training data.</h2><p>Muscle Memory turns validated worlds and robot interactions into reusable experience—so teams spend less time assembling pipelines and more time improving policies.</p></div>
        <ol className="mm-engine__steps">
          <li><span>01</span><div><h3>Generate a valid world</h3><p>Build seeded environments with safe physical bounds and deterministic colliders.</p></div></li>
          <li><span>02</span><div><h3>Run the fixed robot</h3><p>Keep the body and walking controller constant while the task policy acts.</p></div></li>
          <li><span>03</span><div><h3>Capture physical evidence</h3><p>Join frames, actions, contacts, outcomes, and failures into one episode record.</p></div></li>
          <li><span>04</span><div><h3>Train, evaluate, decide</h3><p>Create a new candidate, test held-out worlds, then place promotion behind human approval.</p></div></li>
        </ol>
      </section>

      <section className="mm-bottom-cta">
        <p className="mm-section-label">Grounded agents start here</p>
        <h2>Give every run<br /><em>somewhere to go.</em></h2>
        <a className="mm-run" href="/console">Explore the live system <ArrowRight size={18} /></a>
      </section>
    </main>
  );
}
