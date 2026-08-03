import { useEffect, useRef } from "react";

const PARTICLES = [
  { x: 63, y: 18, depth: 1.2 },
  { x: 76, y: 28, depth: 0.8 },
  { x: 55, y: 47, depth: 1.45 },
  { x: 88, y: 61, depth: 0.95 },
  { x: 70, y: 74, depth: 1.3 },
  { x: 92, y: 38, depth: 1.6 },
];

export function HeroScene() {
  const sceneRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return;
    }

    let frame = 0;
    let targetX = 0;
    let targetY = 0;
    let currentX = 0;
    let currentY = 0;

    const paint = () => {
      currentX += (targetX - currentX) * 0.075;
      currentY += (targetY - currentY) * 0.075;
      scene.style.setProperty("--pointer-x", currentX.toFixed(3));
      scene.style.setProperty("--pointer-y", currentY.toFixed(3));
      frame = requestAnimationFrame(paint);
    };

    const handlePointerMove = (event: PointerEvent) => {
      const bounds = scene.getBoundingClientRect();
      targetX = ((event.clientX - bounds.left) / bounds.width - 0.5) * 2;
      targetY = ((event.clientY - bounds.top) / bounds.height - 0.5) * 2;
    };

    const handlePointerLeave = () => {
      targetX = 0;
      targetY = 0;
    };

    scene.addEventListener("pointermove", handlePointerMove);
    scene.addEventListener("pointerleave", handlePointerLeave);
    frame = requestAnimationFrame(paint);

    return () => {
      scene.removeEventListener("pointermove", handlePointerMove);
      scene.removeEventListener("pointerleave", handlePointerLeave);
      cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <div className="hero-scene" ref={sceneRef} aria-hidden="true">
      <div className="hero-scene__media">
        <img
          src="/assets/mm01-household-hero.webp"
          alt=""
          width="1792"
          height="1024"
          fetchPriority="high"
        />
      </div>
      <div className="hero-scene__wash" />
      <div className="hero-scene__grid" />
      <div className="hero-scene__focus">
        <span />
        <span />
        <span />
        <span />
      </div>
      <div className="hero-scene__telemetry">
        {PARTICLES.map((particle, index) => (
          <span
            key={`${particle.x}-${particle.y}`}
            className="telemetry-particle"
            style={
              {
                "--x": `${particle.x}%`,
                "--y": `${particle.y}%`,
                "--depth": particle.depth,
                "--delay": `${index * 180}ms`,
              } as React.CSSProperties
            }
          />
        ))}
      </div>
    </div>
  );
}
