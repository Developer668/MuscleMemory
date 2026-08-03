import {
  ArrowDownRight,
  ArrowRight,
  Check,
  ChevronRight,
  CircleGauge,
  DatabaseZap,
  Network,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { motion, useReducedMotion, useScroll, useTransform } from "motion/react";

import { BrandMark } from "../components/BrandMark";
import { HeroScene } from "../components/HeroScene";
import { Reveal } from "../components/Reveal";

const memoryLayers = [
  {
    number: "01",
    name: "Live experience",
    product: "LaserData",
    detail: "Every observation and action stays ordered, replayable, and joined to video by frame.",
    icon: DatabaseZap,
    tone: "orange",
  },
  {
    number: "02",
    name: "Explicit experience",
    product: "FalkorDB",
    detail: "Failures become connected lessons across worlds, obstacles, corrections, and policies.",
    icon: Network,
    tone: "blue",
  },
  {
    number: "03",
    name: "Behavioral memory",
    product: "Policy checkpoints",
    detail: "Evaluated behavior is immutable. New experience always earns a new version.",
    icon: ShieldCheck,
    tone: "green",
  },
] as const;

const promotionMetrics = [
  { value: "80%", label: "held-out success" },
  { value: "0", label: "falls permitted" },
  { value: "0.25 m", label: "median clearance" },
  { value: "20 pts", label: "success lift" },
];

export function LandingPage() {
  const reduceMotion = useReducedMotion();
  const { scrollYProgress } = useScroll();
  const lineScale = useTransform(scrollYProgress, [0, 1], [0, 1]);

  return (
    <main id="top">
      <motion.div
        className="scroll-progress"
        style={{ scaleX: reduceMotion ? 1 : lineScale }}
        aria-hidden="true"
      />

      <header className="site-header">
        <a className="brand" href="#top" aria-label="Muscle Memory home">
          <BrandMark />
          <span>Muscle Memory</span>
        </a>
        <nav className="site-nav" aria-label="Primary navigation">
          <a href="#method">Method</a>
          <a href="#memory">Memory</a>
          <a href="#proof">Proof</a>
        </nav>
        <a className="header-action" href="/">
          Open console
          <ChevronRight size={16} strokeWidth={1.8} />
        </a>
      </header>

      <section className="hero" aria-labelledby="hero-title">
        <HeroScene />
        <div className="hero__content">
          <motion.p
            className="eyebrow"
            initial={reduceMotion ? false : { opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.15 }}
          >
            <span className="eyebrow__pulse" />
            Adaptive household robotics
          </motion.p>
          <motion.h1
            id="hero-title"
            initial={reduceMotion ? false : { opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.82, delay: 0.25, ease: [0.2, 0.75, 0.2, 1] }}
          >
            Muscle
            <br />
            <span>Memory</span>
          </motion.h1>
          <motion.div
            className="hero__support"
            initial={reduceMotion ? false : { opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.75, delay: 0.43 }}
          >
            <p className="hero__tagline">One robot. Many worlds. Experience that compounds.</p>
            <p className="hero__description">
              MM-01 learns safe household delivery across changing, physics-valid homes while its
              body, sensors, and walking controller stay fixed.
            </p>
            <div className="hero__actions">
              <a className="button button--primary" href="#platform">
                See the system
                <ArrowDownRight size={18} strokeWidth={1.8} />
              </a>
              <a className="button button--quiet" href="#proof">
                View the promotion standard
                <ArrowRight size={18} strokeWidth={1.8} />
              </a>
            </div>
          </motion.div>
        </div>
        <div className="hero__identity" aria-label="Fixed robot identity">
          <span className="identity-code">MM-01</span>
          <span>Fixed body</span>
          <span>100 Hz gait</span>
          <span>5–10 Hz task policy</span>
        </div>
        <div className="hero__scroll-cue" aria-hidden="true">
          <span>Scroll to compound</span>
          <ArrowDownRight size={18} />
        </div>
      </section>

      <section className="manifesto" id="platform">
        <div className="manifesto__inner">
          <Reveal className="section-kicker">
            <span>01</span>
            The platform
          </Reveal>
          <Reveal className="manifesto__statement" delay={0.08}>
            <p>
              The body stays fixed.
              <br />
              <em>The experience moves forward.</em>
            </p>
          </Reveal>
          <Reveal className="manifesto__aside" delay={0.16}>
            <CircleGauge size={22} strokeWidth={1.5} />
            <p>
              Muscle Memory separates locomotion from learning. MM-01 only learns where to move,
              how quickly to turn, and when to stop.
            </p>
          </Reveal>
        </div>
      </section>

      <section className="learning-loop" id="method" aria-labelledby="method-title">
        <div className="section-heading">
          <Reveal className="section-kicker section-kicker--dark">
            <span>02</span>
            The learning loop
          </Reveal>
          <Reveal delay={0.08}>
            <h2 id="method-title">Experience becomes a safer next run.</h2>
          </Reveal>
          <Reveal className="section-heading__copy" delay={0.14}>
            <p>
              Each validated world adds evidence. Each failure adds context. Nothing is promoted
              until it wins on worlds it has never seen.
            </p>
          </Reveal>
        </div>

        <div className="loop-track">
          <div className="loop-track__line" aria-hidden="true" />
          {[
            ["01", "Generate", "Seeded apartments with deterministic collision geometry."],
            ["02", "Remember", "Telemetry, failure relationships, and human corrections."],
            ["03", "Improve", "A new policy version measured against the same frozen standard."],
          ].map(([number, title, detail], index) => (
            <Reveal className="loop-step" delay={index * 0.1} key={title}>
              <span className="loop-step__number">{number}</span>
              <div className="loop-step__marker" aria-hidden="true" />
              <h3>{title}</h3>
              <p>{detail}</p>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="memory" id="memory" aria-labelledby="memory-title">
        <div className="memory__header">
          <Reveal className="section-kicker">
            <span>03</span>
            The memory stack
          </Reveal>
          <Reveal delay={0.08}>
            <h2 id="memory-title">Three kinds of memory. One compounding system.</h2>
          </Reveal>
        </div>

        <div className="memory__layers">
          {memoryLayers.map((layer, index) => {
            const Icon = layer.icon;
            return (
              <Reveal className={`memory-layer memory-layer--${layer.tone}`} delay={index * 0.08} key={layer.name}>
                <div className="memory-layer__topline">
                  <span>{layer.number}</span>
                  <Icon size={21} strokeWidth={1.5} />
                </div>
                <p className="memory-layer__label">{layer.name}</p>
                <h3>{layer.product}</h3>
                <p className="memory-layer__detail">{layer.detail}</p>
                <div className="memory-layer__signal" aria-hidden="true">
                  {Array.from({ length: 12 }, (_, signalIndex) => (
                    <span key={signalIndex} style={{ "--i": signalIndex } as React.CSSProperties} />
                  ))}
                </div>
              </Reveal>
            );
          })}
        </div>
      </section>

      <section className="proof" id="proof" aria-labelledby="proof-title">
        <div className="proof__visual" aria-hidden="true">
          <div className="proof__lattice" />
          <div className="proof__scan" />
          <div className="proof__gate proof__gate--one" />
          <div className="proof__gate proof__gate--two" />
          <div className="proof__core">
            <ShieldCheck size={48} strokeWidth={1.2} />
          </div>
          <span className="proof__tag proof__tag--one">Frozen split</span>
          <span className="proof__tag proof__tag--two">Paired worlds</span>
          <span className="proof__tag proof__tag--three">Human approval</span>
        </div>
        <div className="proof__content">
          <Reveal className="section-kicker section-kicker--dark">
            <span>04</span>
            Promotion standard
          </Reveal>
          <Reveal delay={0.08}>
            <h2 id="proof-title">Improvement is measured, never implied.</h2>
          </Reveal>
          <Reveal delay={0.14}>
            <p className="proof__intro">
              Candidate policies face twenty held-out worlds. Passing numbers make a policy
              eligible; a human still decides whether it moves forward.
            </p>
          </Reveal>
          <div className="proof__metrics">
            {promotionMetrics.map((metric, index) => (
              <Reveal className="proof-metric" delay={0.18 + index * 0.06} key={metric.label}>
                <strong>{metric.value}</strong>
                <span>{metric.label}</span>
              </Reveal>
            ))}
          </div>
          <Reveal className="proof__checks" delay={0.32}>
            <span><Check size={15} /> Same robot checksum</span>
            <span><Check size={15} /> Same held-out seeds</span>
            <span><Check size={15} /> No training-time teacher</span>
          </Reveal>
        </div>
      </section>

      <section className="closing" aria-labelledby="closing-title">
        <div className="closing__image" aria-hidden="true">
          <img src="/assets/mm01-household-hero.webp" alt="" loading="lazy" />
        </div>
        <div className="closing__shade" />
        <Reveal className="closing__content">
          <span className="closing__icon"><Sparkles size={21} strokeWidth={1.5} /></span>
          <h2 id="closing-title">A lifetime of experience, before the first delivery.</h2>
          <a className="button button--light" href="#top">
            Return to MM-01
            <ArrowRight size={18} strokeWidth={1.8} />
          </a>
        </Reveal>
      </section>

      <footer className="site-footer">
        <a className="brand brand--footer" href="#top">
          <BrandMark />
          <span>Muscle Memory</span>
        </a>
        <p>One robot. Many worlds. Experience that compounds.</p>
        <p className="site-footer__meta">MM-01 · Safe household delivery</p>
      </footer>
    </main>
  );
}
