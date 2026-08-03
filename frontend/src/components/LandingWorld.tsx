import { useEffect, useRef } from "react";
import type { MotionValue } from "motion/react";
import * as THREE from "three";

import { buildApartment, makeRobot } from "../operator/RealisticHomeScene";

export type DemoScenario = "clear" | "laundry" | "low_friction";

type LandingWorldProps = {
  progress: MotionValue<number>;
  scenario: DemoScenario;
};

const clamp = (value: number, minimum = 0, maximum = 1) =>
  Math.min(maximum, Math.max(minimum, value));

const smooth = (value: number) => {
  const next = clamp(value);
  return next * next * (3 - 2 * next);
};

function simpleMesh(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  position: [number, number, number],
): THREE.Mesh {
  const item = new THREE.Mesh(geometry, material);
  item.position.set(...position);
  item.castShadow = true;
  item.receiveShadow = true;
  return item;
}

function makeDropObject(kind: "box" | "basket" | "lamp"): THREE.Group {
  const object = new THREE.Group();
  if (kind === "box") {
    const cardboard = new THREE.MeshStandardMaterial({ color: "#b67a45", roughness: 0.88 });
    object.add(simpleMesh(new THREE.BoxGeometry(0.64, 0.52, 0.58), cardboard, [0, 0.26, 0]));
    object.add(simpleMesh(new THREE.BoxGeometry(0.66, 0.035, 0.08), new THREE.MeshStandardMaterial({ color: "#d9b073", roughness: 0.8 }), [0, 0.53, 0]));
  } else if (kind === "basket") {
    const weave = new THREE.MeshStandardMaterial({ color: "#bba078", roughness: 0.94, side: THREE.DoubleSide });
    object.add(simpleMesh(new THREE.CylinderGeometry(0.35, 0.28, 0.58, 28, 1, true), weave, [0, 0.29, 0]));
    object.add(simpleMesh(new THREE.TorusGeometry(0.34, 0.035, 8, 30), weave, [0, 0.58, 0]));
    object.children[1].rotation.x = Math.PI / 2;
    const cloth = new THREE.MeshStandardMaterial({ color: "#5c87a7", roughness: 0.95 });
    object.add(simpleMesh(new THREE.SphereGeometry(0.24, 20, 14), cloth, [0.03, 0.55, 0]));
    object.children[2].scale.set(1, 0.48, 1);
  } else {
    const dark = new THREE.MeshStandardMaterial({ color: "#343536", roughness: 0.42, metalness: 0.58 });
    object.add(simpleMesh(new THREE.CylinderGeometry(0.025, 0.025, 1.6, 12), dark, [0, 0.8, 0]));
    object.add(simpleMesh(new THREE.CylinderGeometry(0.18, 0.32, 0.28, 32, 1, true), dark, [0, 1.52, 0]));
  }
  return object;
}

function makeRoute(): THREE.Line {
  const points = [
    new THREE.Vector3(-2.75, 0.055, 2.25),
    new THREE.Vector3(-1.7, 0.055, 1.4),
    new THREE.Vector3(-0.9, 0.055, 0.45),
    new THREE.Vector3(0.05, 0.055, 0.2),
    new THREE.Vector3(0.9, 0.055, -0.72),
    new THREE.Vector3(2.28, 0.055, -1.83),
  ];
  const curve = new THREE.CatmullRomCurve3(points);
  const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(90));
  const material = new THREE.LineDashedMaterial({
    color: "#77f3cb",
    dashSize: 0.18,
    gapSize: 0.12,
    transparent: true,
    opacity: 0.82,
  });
  const line = new THREE.Line(geometry, material);
  line.computeLineDistances();
  return line;
}

export function LandingWorld({ progress, scenario }: LandingWorldProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const liveRef = useRef({ progress: progress.get(), scenario });

  useEffect(() => {
    liveRef.current.scenario = scenario;
  }, [scenario]);

  useEffect(() => {
    const unsubscribe = progress.on("change", (value) => {
      liveRef.current.progress = value;
    });
    return unsubscribe;
  }, [progress]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#b9c8ca");
    scene.fog = new THREE.FogExp2("#d7dfdc", 0.022);

    const camera = new THREE.PerspectiveCamera(44, 1, 0.04, 80);
    camera.position.set(8.2, 5.4, 8.6);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.domElement.setAttribute("aria-hidden", "true");
    host.appendChild(renderer.domElement);

    const home = buildApartment();
    scene.add(home);

    const robot = makeRobot();
    robot.position.set(-2.75, 0, 2.25);
    scene.add(robot);

    const route = makeRoute();
    scene.add(route);

    const destination = new THREE.Mesh(
      new THREE.RingGeometry(0.28, 0.43, 48),
      new THREE.MeshBasicMaterial({ color: "#ff6e57", transparent: true, opacity: 0.9, side: THREE.DoubleSide }),
    );
    destination.rotation.x = -Math.PI / 2;
    destination.position.set(2.28, 0.07, -1.83);
    scene.add(destination);

    const box = makeDropObject("box");
    box.position.set(-1.2, 4.8, 0.55);
    scene.add(box);
    const basket = makeDropObject("basket");
    basket.position.set(-0.32, 5.5, 0.72);
    scene.add(basket);
    const lamp = makeDropObject("lamp");
    lamp.position.set(3.25, 5.8, 1.1);
    scene.add(lamp);

    const frictionPatch = new THREE.Mesh(
      new THREE.CircleGeometry(0.78, 48),
      new THREE.MeshBasicMaterial({ color: "#77e7ff", transparent: true, opacity: 0.26, side: THREE.DoubleSide }),
    );
    frictionPatch.rotation.x = -Math.PI / 2;
    frictionPatch.position.set(0.25, 0.062, 0.05);
    scene.add(frictionPatch);

    const dotGeometry = new THREE.BufferGeometry();
    const dotPositions = new Float32Array(180 * 3);
    for (let index = 0; index < 180; index += 1) {
      dotPositions[index * 3] = (Math.random() - 0.5) * 9;
      dotPositions[index * 3 + 1] = Math.random() * 3.4;
      dotPositions[index * 3 + 2] = (Math.random() - 0.5) * 7;
    }
    dotGeometry.setAttribute("position", new THREE.BufferAttribute(dotPositions, 3));
    const dots = new THREE.Points(
      dotGeometry,
      new THREE.PointsMaterial({ color: "#b5fff0", size: 0.022, transparent: true, opacity: 0.38 }),
    );
    scene.add(dots);

    scene.add(new THREE.HemisphereLight("#eafcff", "#72513d", 2.25));
    const sun = new THREE.DirectionalLight("#fff0ce", 5.4);
    sun.position.set(2.4, 7.8, -5.8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -7;
    sun.shadow.camera.right = 7;
    sun.shadow.camera.top = 7;
    sun.shadow.camera.bottom = -7;
    scene.add(sun);
    const warm = new THREE.PointLight("#ff9b63", 5.5, 5.5, 2);
    warm.position.set(-2.4, 2.1, 1.8);
    scene.add(warm);

    const routeCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-2.75, 0, 2.25),
      new THREE.Vector3(-1.7, 0, 1.4),
      new THREE.Vector3(-0.9, 0, 0.45),
      new THREE.Vector3(0.05, 0, 0.2),
      new THREE.Vector3(0.9, 0, -0.72),
      new THREE.Vector3(2.28, 0, -1.83),
    ]);
    const widePosition = new THREE.Vector3(8.2, 5.4, 8.6);
    const wideTarget = new THREE.Vector3(0, 0.85, -0.15);
    const overviewPosition = new THREE.Vector3(5.8, 3.15, 6.15);
    const finalPosition = new THREE.Vector3(-6.8, 4.1, 7.2);
    const finalTarget = new THREE.Vector3(0.35, 0.9, -0.4);
    const cameraPosition = new THREE.Vector3();
    const cameraTarget = new THREE.Vector3();
    const forward = new THREE.Vector3();
    const previousPosition = robot.position.clone();
    const clock = new THREE.Clock();
    let pointerX = 0;
    let pointerY = 0;
    let frame = 0;

    const onPointerMove = (event: PointerEvent) => {
      const bounds = host.getBoundingClientRect();
      pointerX = ((event.clientX - bounds.left) / bounds.width - 0.5) * 2;
      pointerY = ((event.clientY - bounds.top) / bounds.height - 0.5) * 2;
    };
    const onPointerLeave = () => {
      pointerX = 0;
      pointerY = 0;
    };
    host.addEventListener("pointermove", onPointerMove);
    host.addEventListener("pointerleave", onPointerLeave);

    const setDrop = (object: THREE.Group, start: number, ground: number, progressValue: number) => {
      const fall = smooth((progressValue - start) / 0.085);
      const bounce = Math.abs(Math.sin(fall * Math.PI * 2.5)) * (1 - fall) * 0.34;
      object.position.y = THREE.MathUtils.lerp(5.2, ground, fall) + bounce;
      object.rotation.y = (1 - fall) * 1.8;
      object.scale.setScalar(0.78 + fall * 0.22);
    };

    const render = () => {
      frame = window.requestAnimationFrame(render);
      const elapsed = clock.getElapsedTime();
      const p = clamp(liveRef.current.progress);
      const mission = smooth((p - 0.14) / 0.58);
      const routePosition = routeCurve.getPoint(mission);
      previousPosition.copy(robot.position);
      robot.position.copy(routePosition);
      forward.copy(routeCurve.getTangent(Math.min(0.999, mission + 0.002))).normalize();
      robot.rotation.y = Math.atan2(forward.x, forward.z);

      const gaitStrength = Math.sin(elapsed * 9.2) * 0.21 * (mission > 0.01 && mission < 0.99 ? 1 : 0);
      robot.getObjectByName("left-leg")!.rotation.x = gaitStrength;
      robot.getObjectByName("right-leg")!.rotation.x = -gaitStrength;
      robot.getObjectByName("left-arm")!.rotation.x = -gaitStrength * 0.35;
      robot.getObjectByName("right-arm")!.rotation.x = gaitStrength * 0.35;

      setDrop(box, 0.16, 0, p);
      setDrop(basket, 0.25, 0, p);
      setDrop(lamp, 0.34, 0, p);
      basket.position.x = liveRef.current.scenario === "laundry" ? -0.28 : -1.95;
      basket.position.z = liveRef.current.scenario === "laundry" ? 0.5 : 1.9;
      frictionPatch.visible = liveRef.current.scenario === "low_friction";
      frictionPatch.material.opacity = 0.2 + Math.sin(elapsed * 3) * 0.07;

      const povMix = smooth((p - 0.48) / 0.1) * (1 - smooth((p - 0.77) / 0.09));
      const finalMix = smooth((p - 0.79) / 0.16);
      cameraPosition.lerpVectors(widePosition, overviewPosition, smooth(p / 0.34));
      cameraTarget.copy(wideTarget).lerp(robot.position.clone().add(new THREE.Vector3(0, 0.9, 0)), smooth((p - 0.08) / 0.28));

      const headPosition = robot.position.clone().add(new THREE.Vector3(0, 1.43, 0));
      const headTarget = headPosition.clone().add(forward.clone().multiplyScalar(3.6));
      cameraPosition.lerp(headPosition, povMix);
      cameraTarget.lerp(headTarget, povMix);
      cameraPosition.lerp(finalPosition, finalMix);
      cameraTarget.lerp(finalTarget, finalMix);
      cameraPosition.x += pointerX * 0.16 * (1 - povMix);
      cameraPosition.y -= pointerY * 0.1 * (1 - povMix);
      camera.position.lerp(cameraPosition, 0.075);
      camera.lookAt(cameraTarget);

      route.visible = p > 0.1 && p < 0.82;
      (route.material as THREE.LineDashedMaterial).dashOffset = -elapsed * 0.35;
      destination.rotation.z += 0.006;
      destination.scale.setScalar(1 + Math.sin(elapsed * 3.2) * 0.08);
      dots.rotation.y = elapsed * 0.008;
      (dots.material as THREE.PointsMaterial).opacity = 0.12 + smooth((p - 0.35) / 0.2) * 0.32;

      renderer.render(scene, camera);
    };

    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    render();

    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
      host.removeEventListener("pointermove", onPointerMove);
      host.removeEventListener("pointerleave", onPointerLeave);
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Points || object instanceof THREE.Line) {
          object.geometry.dispose();
          const surfaces = Array.isArray(object.material) ? object.material : [object.material];
          surfaces.forEach((surface) => {
            const mapped = surface as THREE.MeshStandardMaterial;
            mapped.map?.dispose();
            surface.dispose();
          });
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div
      className="mm-world"
      ref={hostRef}
      role="img"
      aria-label="Interactive 3D living room with MM-01 completing a household delivery"
    />
  );
}
