import { useEffect, useRef } from "react";
import type { MotionValue } from "motion/react";
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

import { makeRobot } from "../operator/RealisticHomeScene";
import { attachImportedFurnishings } from "./ImportedFurnishings";
import { buildTwoStoryHouse, STORY_HEIGHT } from "./TwoStoryHouse";

export type DemoScenario = "clear" | "laundry" | "low_friction";
export type DemoTask = "medicine" | "breakfast" | "kitchen_check" | "parcel";

type LandingWorldProps = {
  progress: MotionValue<number>;
  scenario: DemoScenario;
  task: DemoTask;
};

type TaskWorldConfig = {
  route: Array<[number, number]>;
  color: string;
  payloadColor: string;
};

const TASK_WORLDS: Record<DemoTask, TaskWorldConfig> = {
  medicine: {
    route: [[-4.9, 3.82], [-3.75, 2.72], [-2.15, 1.48], [-0.45, 0.52], [1.42, -0.18], [2.48, -1.42], [3.42, -2.68]],
    color: "#ff6e57",
    payloadColor: "#b84d46",
  },
  breakfast: {
    route: [[-4.72, -0.48], [-3.78, 0.18], [-2.55, 0.5], [-1.42, 0.76], [-0.24, 0.98], [0.68, 1.18]],
    color: "#ffd75f",
    payloadColor: "#d6a542",
  },
  kitchen_check: {
    route: [[4.62, 0.88], [3.18, 0.58], [1.72, 0.18], [0.18, -0.58], [-0.72, -1.26], [-1.08, -1.98], [-1.45, -2.6]],
    color: "#77e7ff",
    payloadColor: "#77e7ff",
  },
  parcel: {
    route: [[4.76, 0.94], [3.32, 0.7], [1.78, 0.52], [0.22, 0.48], [-1.34, 0.8], [-2.35, 1.12], [-3.05, 1.36], [-3.55, 1.55]],
    color: "#ff9a78",
    payloadColor: "#ad7548",
  },
};

const TASK_FOCUS: Record<DemoTask, THREE.Vector3> = {
  medicine: new THREE.Vector3(3.55, 0.82, -3.42),
  breakfast: new THREE.Vector3(0.05, 0.82, 2.73),
  kitchen_check: new THREE.Vector3(-3.72, 0.98, -4.05),
  parcel: new THREE.Vector3(-5.55, 0.58, 2.75),
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

function curveFor(task: DemoTask): THREE.CatmullRomCurve3 {
  return new THREE.CatmullRomCurve3(
    TASK_WORLDS[task].route.map(([x, z]) => new THREE.Vector3(x, 0.08, z)),
  );
}

function makeRoute(curve: THREE.CatmullRomCurve3, color: string): THREE.Line {
  const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(90));
  const material = new THREE.LineDashedMaterial({
    color,
    dashSize: 0.18,
    gapSize: 0.12,
    transparent: true,
    opacity: 0.82,
  });
  const line = new THREE.Line(geometry, material);
  line.computeLineDistances();
  return line;
}

export function LandingWorld({ progress, scenario, task }: LandingWorldProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const liveRef = useRef({ progress: progress.get(), scenario, task });

  useEffect(() => {
    liveRef.current.scenario = scenario;
  }, [scenario]);

  useEffect(() => {
    liveRef.current.task = task;
  }, [task]);

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
    scene.background = new THREE.Color("#aebfc2");
    scene.fog = new THREE.FogExp2("#d5dfde", 0.012);

    const camera = new THREE.PerspectiveCamera(39, 1, 0.04, 120);
    camera.position.set(15.2, 9.5, 15.7);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.02;
    renderer.domElement.setAttribute("aria-hidden", "true");
    host.appendChild(renderer.domElement);

    const environmentGenerator = new THREE.PMREMGenerator(renderer);
    const environment = environmentGenerator.fromScene(new RoomEnvironment(), 0.035).texture;
    scene.environment = environment;

    const house = buildTwoStoryHouse();
    scene.add(house.root);
    const cancelFurnitureLoading = attachImportedFurnishings(house);

    const robot = makeRobot();
    const initialCurve = curveFor(liveRef.current.task);
    robot.position.copy(initialCurve.getPoint(0));
    scene.add(robot);

    const route = makeRoute(initialCurve, TASK_WORLDS[liveRef.current.task].color);
    const routeMaterial = route.material as THREE.LineDashedMaterial;
    scene.add(route);

    const destination = new THREE.Mesh(
      new THREE.RingGeometry(0.28, 0.43, 48),
      new THREE.MeshBasicMaterial({ color: "#ff6e57", transparent: true, opacity: 0.9, side: THREE.DoubleSide }),
    );
    destination.rotation.x = -Math.PI / 2;
    destination.position.copy(initialCurve.getPoint(1));
    destination.position.y = 0.07;
    scene.add(destination);

    const completionHalo = new THREE.Mesh(
      new THREE.RingGeometry(0.44, 0.49, 64),
      new THREE.MeshBasicMaterial({ color: TASK_WORLDS[liveRef.current.task].color, transparent: true, opacity: 0, side: THREE.DoubleSide }),
    );
    completionHalo.rotation.x = -Math.PI / 2;
    completionHalo.position.copy(destination.position);
    scene.add(completionHalo);

    const scanRings = new THREE.Group();
    for (let index = 0; index < 3; index += 1) {
      const scan = new THREE.Mesh(
        new THREE.RingGeometry(0.32, 0.35, 56),
        new THREE.MeshBasicMaterial({ color: "#77e7ff", transparent: true, opacity: 0, side: THREE.DoubleSide }),
      );
      scan.rotation.x = -Math.PI / 2;
      scan.userData.phase = index / 3;
      scanRings.add(scan);
    }
    scene.add(scanRings);

    const breakfastPayload = new THREE.Group();
    breakfastPayload.name = "breakfast-payload";
    breakfastPayload.add(simpleMesh(new THREE.CylinderGeometry(0.13, 0.15, 0.095, 28), new THREE.MeshStandardMaterial({ color: "#ede5d4", roughness: 0.46 }), [-0.1, 1.12, 0.44]));
    breakfastPayload.add(simpleMesh(new THREE.CylinderGeometry(0.075, 0.066, 0.14, 24), new THREE.MeshStandardMaterial({ color: "#d6a542", roughness: 0.52 }), [0.15, 1.14, 0.44]));
    robot.add(breakfastPayload);

    const steam = new THREE.Group();
    for (let index = 0; index < 4; index += 1) {
      const puff = new THREE.Mesh(
        new THREE.SphereGeometry(0.018 + index * 0.004, 10, 8),
        new THREE.MeshBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.42 }),
      );
      puff.userData.phase = index / 4;
      steam.add(puff);
    }
    robot.add(steam);

    const burstCount = 44;
    const burstGeometry = new THREE.BufferGeometry();
    const burstPositions = new Float32Array(burstCount * 3);
    burstGeometry.setAttribute("position", new THREE.BufferAttribute(burstPositions, 3));
    const burstMaterial = new THREE.PointsMaterial({ color: TASK_WORLDS[liveRef.current.task].color, size: 0.065, transparent: true, opacity: 0 });
    const burst = new THREE.Points(burstGeometry, burstMaterial);
    scene.add(burst);

    const box = makeDropObject("box");
    box.position.set(-2.35, 7.8, 1.2);
    scene.add(box);
    const basket = makeDropObject("basket");
    basket.position.set(-0.55, 8.4, 0.72);
    scene.add(basket);
    const lamp = makeDropObject("lamp");
    lamp.position.set(5.45, 8.8, -0.2);
    scene.add(lamp);

    const frictionPatch = new THREE.Mesh(
      new THREE.CircleGeometry(0.78, 48),
      new THREE.MeshBasicMaterial({ color: "#77e7ff", transparent: true, opacity: 0.26, side: THREE.DoubleSide }),
    );
    frictionPatch.rotation.x = -Math.PI / 2;
    frictionPatch.position.set(1.1, 0.062, -0.15);
    scene.add(frictionPatch);

    const dotGeometry = new THREE.BufferGeometry();
    const dotPositions = new Float32Array(180 * 3);
    for (let index = 0; index < 180; index += 1) {
      const seeded = (salt: number) => {
        const value = Math.sin((index + 1) * (12.9898 + salt * 7.13)) * 43758.5453;
        return value - Math.floor(value);
      };
      dotPositions[index * 3] = (seeded(1) - 0.5) * 15;
      dotPositions[index * 3 + 1] = seeded(2) * 7.1;
      dotPositions[index * 3 + 2] = (seeded(3) - 0.5) * 11;
    }
    dotGeometry.setAttribute("position", new THREE.BufferAttribute(dotPositions, 3));
    const dots = new THREE.Points(
      dotGeometry,
      new THREE.PointsMaterial({ color: "#b5fff0", size: 0.022, transparent: true, opacity: 0.38 }),
    );
    scene.add(dots);

    scene.add(new THREE.HemisphereLight("#eafcff", "#6c5140", 1.82));
    const sun = new THREE.DirectionalLight("#fff0ce", 5.05);
    sun.position.set(5.4, 13.5, -10.8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -12;
    sun.shadow.camera.right = 12;
    sun.shadow.camera.top = 12;
    sun.shadow.camera.bottom = -12;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.035;
    scene.add(sun);
    const warm = new THREE.PointLight("#ff9b63", 5.5, 5.5, 2);
    warm.position.set(-4.2, 2.35, -1.6);
    scene.add(warm);
    const upstairsGlow = new THREE.PointLight("#ffcf95", 3.8, 8.5, 2);
    upstairsGlow.position.set(3.6, STORY_HEIGHT + 2.2, -2.4);
    scene.add(upstairsGlow);

    const widePosition = new THREE.Vector3(15.2, 9.5, 15.7);
    const wideTarget = new THREE.Vector3(-0.8, 2.78, -0.15);
    const overviewPosition = new THREE.Vector3(12.6, 5.25, 11.8);
    const finalPosition = new THREE.Vector3(-16.2, 10.4, 17.2);
    const finalTarget = new THREE.Vector3(0.2, 3.0, -0.25);
    const cameraPosition = new THREE.Vector3();
    const cameraTarget = new THREE.Vector3();
    const forward = new THREE.Vector3();
    const previousPosition = robot.position.clone();
    const oldRoutePoint = new THREE.Vector3();
    const newRoutePoint = new THREE.Vector3();
    const morphedRoutePoint = new THREE.Vector3();
    const previousDestination = destination.position.clone();
    const activeDestination = destination.position.clone();
    const previousRouteColor = new THREE.Color(TASK_WORLDS[liveRef.current.task].color);
    const activeRouteColor = previousRouteColor.clone();
    const clock = new THREE.Clock();
    let activeTask = liveRef.current.task;
    let previousCurve = initialCurve;
    let activeCurve = initialCurve;
    let taskTransitionStarted = -10;
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
      object.visible = progressValue >= start;
      const fall = smooth((progressValue - start) / 0.085);
      const bounce = Math.abs(Math.sin(fall * Math.PI * 2.5)) * (1 - fall) * 0.34;
      object.position.y = THREE.MathUtils.lerp(8.6, ground, fall) + bounce;
      object.rotation.y = (1 - fall) * 1.8;
      object.scale.setScalar(0.78 + fall * 0.22);
    };

    const render = () => {
      frame = window.requestAnimationFrame(render);
      const elapsed = clock.getElapsedTime();
      const p = clamp(liveRef.current.progress);
      const mission = smooth((p - 0.14) / 0.58);

      if (liveRef.current.task !== activeTask) {
        previousCurve = activeCurve;
        previousDestination.copy(destination.position);
        previousRouteColor.copy(routeMaterial.color);
        activeTask = liveRef.current.task;
        activeCurve = curveFor(activeTask);
        activeDestination.copy(activeCurve.getPoint(1));
        activeDestination.y = 0.07;
        activeRouteColor.set(TASK_WORLDS[activeTask].color);
        taskTransitionStarted = elapsed;
      }

      const taskMix = smooth((elapsed - taskTransitionStarted) / 0.82);
      previousCurve.getPoint(mission, oldRoutePoint);
      activeCurve.getPoint(mission, newRoutePoint);
      const routePosition = morphedRoutePoint.lerpVectors(oldRoutePoint, newRoutePoint, taskMix);
      previousPosition.copy(robot.position);
      robot.position.copy(routePosition);
      const nextMission = Math.min(0.999, mission + 0.006);
      previousCurve.getPoint(nextMission, oldRoutePoint);
      activeCurve.getPoint(nextMission, newRoutePoint);
      forward.copy(morphedRoutePoint.lerpVectors(oldRoutePoint, newRoutePoint, taskMix)).sub(robot.position).normalize();
      robot.rotation.y = Math.atan2(forward.x, forward.z);

      const routePositions = route.geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let index = 0; index < routePositions.count; index += 1) {
        const routeProgress = index / (routePositions.count - 1);
        previousCurve.getPoint(routeProgress, oldRoutePoint);
        activeCurve.getPoint(routeProgress, newRoutePoint);
        morphedRoutePoint.lerpVectors(oldRoutePoint, newRoutePoint, taskMix);
        routePositions.setXYZ(index, morphedRoutePoint.x, 0.055, morphedRoutePoint.z);
      }
      routePositions.needsUpdate = true;
      route.computeLineDistances();
      routeMaterial.color.lerpColors(previousRouteColor, activeRouteColor, taskMix);
      destination.position.lerpVectors(previousDestination, activeDestination, taskMix);
      (destination.material as THREE.MeshBasicMaterial).color.copy(routeMaterial.color);
      (completionHalo.material as THREE.MeshBasicMaterial).color.copy(routeMaterial.color);
      burstMaterial.color.copy(routeMaterial.color);

      const payload = robot.getObjectByName("delivery-payload");
      const isBreakfast = activeTask === "breakfast";
      const hasParcel = activeTask === "parcel";
      const hasBoxPayload = activeTask === "medicine" || hasParcel;
      if (payload instanceof THREE.Mesh) {
        payload.visible = hasBoxPayload;
        payload.scale.setScalar(hasParcel ? 1.42 : 1);
        (payload.material as THREE.MeshStandardMaterial).color.set(TASK_WORLDS[activeTask].payloadColor);
      }
      breakfastPayload.visible = isBreakfast;
      breakfastPayload.scale.setScalar(isBreakfast ? 0.86 + taskMix * 0.14 : 0.01);
      steam.visible = isBreakfast;
      steam.children.forEach((puff, index) => {
        const phase = (elapsed * 0.46 + puff.userData.phase) % 1;
        puff.position.set(-0.1 + Math.sin(phase * Math.PI * 2 + index) * 0.025, 1.2 + phase * 0.28, 0.44);
        ((puff as THREE.Mesh).material as THREE.MeshBasicMaterial).opacity = Math.sin(phase * Math.PI) * 0.42;
      });

      const gaitStrength = Math.sin(elapsed * 9.2) * 0.21 * (mission > 0.01 && mission < 0.99 ? 1 : 0);
      robot.getObjectByName("left-leg")!.rotation.x = gaitStrength;
      robot.getObjectByName("right-leg")!.rotation.x = -gaitStrength;
      robot.getObjectByName("left-arm")!.rotation.x = -gaitStrength * 0.35;
      robot.getObjectByName("right-arm")!.rotation.x = gaitStrength * 0.35;

      setDrop(box, 0.16, 0, p);
      setDrop(basket, 0.25, 0, p);
      setDrop(lamp, 0.34, 0, p);
      basket.position.x = liveRef.current.scenario === "laundry" ? -0.55 : -5.45;
      basket.position.z = liveRef.current.scenario === "laundry" ? 0.72 : 2.68;
      frictionPatch.visible = liveRef.current.scenario === "low_friction";
      frictionPatch.material.opacity = 0.2 + Math.sin(elapsed * 3) * 0.07;

      const floorLift = smooth((p - 0.12) / 0.15) * (1 - smooth((p - 0.72) / 0.16));
      house.upperFloor.position.y = floorLift * 1.15;

      const povMix = smooth((p - 0.48) / 0.1) * (1 - smooth((p - 0.77) / 0.09));
      const finalMix = smooth((p - 0.79) / 0.16);
      cameraPosition.lerpVectors(widePosition, overviewPosition, smooth(p / 0.34));
      cameraTarget.copy(wideTarget).lerp(robot.position.clone().add(new THREE.Vector3(0, 1.05, 0)), smooth((p - 0.08) / 0.28));

      const headPosition = robot.position.clone()
        .add(new THREE.Vector3(0, 1.49, 0))
        .add(forward.clone().multiplyScalar(0.22));
      const headTarget = headPosition.clone().add(forward.clone().multiplyScalar(3.6));
      headTarget.lerp(TASK_FOCUS[activeTask], smooth((mission - 0.62) / 0.28));
      cameraPosition.lerp(headPosition, povMix);
      cameraTarget.lerp(headTarget, povMix);
      cameraPosition.lerp(finalPosition, finalMix);
      cameraTarget.lerp(finalTarget, finalMix);
      cameraPosition.x += pointerX * 0.3 * (1 - povMix);
      cameraPosition.y -= pointerY * 0.18 * (1 - povMix);
      camera.position.lerp(cameraPosition, 0.075);
      camera.lookAt(cameraTarget);

      route.visible = p > 0.1 && p < 0.82;
      destination.rotation.z += 0.006;
      destination.scale.setScalar(1 + Math.sin(elapsed * 3.2) * 0.08);
      completionHalo.position.copy(destination.position);
      const arrival = clamp((mission - 0.78) / 0.22);
      const arrivalPulse = Math.sin(arrival * Math.PI);
      completionHalo.scale.setScalar(1 + arrival * 4.5);
      (completionHalo.material as THREE.MeshBasicMaterial).opacity = arrivalPulse * 0.72;

      scanRings.visible = activeTask === "kitchen_check" && mission > 0.58;
      scanRings.position.copy(TASK_FOCUS.kitchen_check);
      scanRings.position.y = 1.01;
      scanRings.children.forEach((ring) => {
        const phase = (elapsed * 0.5 + ring.userData.phase) % 1;
        ring.scale.setScalar(0.65 + phase * 3.1);
        ((ring as THREE.Mesh).material as THREE.MeshBasicMaterial).opacity = (1 - phase) * arrival * 0.7;
      });

      burst.position.copy(destination.position);
      for (let index = 0; index < burstCount; index += 1) {
        const angle = index * 2.399963;
        const radius = arrival * (0.25 + (index % 9) * 0.055);
        burstPositions[index * 3] = Math.cos(angle) * radius;
        burstPositions[index * 3 + 1] = arrival * (0.12 + (index % 7) * 0.09);
        burstPositions[index * 3 + 2] = Math.sin(angle) * radius;
      }
      (burstGeometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
      burstMaterial.opacity = arrivalPulse * 0.9;
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
      cancelFurnitureLoading();
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
      environment.dispose();
      environmentGenerator.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div
      className="mm-world"
      ref={hostRef}
      role="img"
      aria-label="Interactive two-story 3D home with MM-01 completing a household delivery"
    />
  );
}
