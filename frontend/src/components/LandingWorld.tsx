import { useEffect, useRef, useState } from "react";
import type { MotionValue } from "motion/react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { buildApartment, makeRobot } from "../operator/RealisticHomeScene";

type LandingWorldProps = {
  progress: MotionValue<number>;
  reducedMotion?: boolean;
};

type LoadState = "loading" | "ready" | "fallback";

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

function makeBasket(): THREE.Group {
  const object = new THREE.Group();
  const weave = new THREE.MeshStandardMaterial({ color: "#a9906e", roughness: 0.94, side: THREE.DoubleSide });
  object.add(simpleMesh(new THREE.CylinderGeometry(0.35, 0.28, 0.58, 28, 1, true), weave, [0, 0.29, 0]));
  object.add(simpleMesh(new THREE.TorusGeometry(0.34, 0.035, 8, 30), weave, [0, 0.58, 0]));
  object.children[1].rotation.x = Math.PI / 2;
  const cloth = new THREE.MeshStandardMaterial({ color: "#667d7c", roughness: 0.95 });
  object.add(simpleMesh(new THREE.SphereGeometry(0.24, 20, 14), cloth, [0.03, 0.55, 0]));
  object.children[2].scale.set(1, 0.48, 1);
  return object;
}

function makeResident(): THREE.Group {
  const resident = new THREE.Group();
  const surface = new THREE.MeshStandardMaterial({ color: "#d8c4aa", roughness: 0.86 });
  resident.add(simpleMesh(new THREE.SphereGeometry(0.13, 24, 18), surface, [0, 1.28, 0]));
  resident.add(simpleMesh(new THREE.CapsuleGeometry(0.16, 0.56, 8, 18), surface, [0, 0.77, 0]));
  resident.add(simpleMesh(new THREE.BoxGeometry(0.72, 0.34, 0.7), new THREE.MeshStandardMaterial({ color: "#5a554e", roughness: 0.9 }), [0, 0.18, 0]));
  resident.name = "resident-destination";
  return resident;
}

function makePayload(): THREE.Group {
  const payload = new THREE.Group();
  payload.name = "delivery-payload";
  const tray = new THREE.MeshStandardMaterial({ color: "#c6c8c0", roughness: 0.3, metalness: 0.45 });
  const pouch = new THREE.MeshStandardMaterial({ color: "#c8f46d", roughness: 0.78 });
  payload.add(simpleMesh(new THREE.BoxGeometry(0.54, 0.025, 0.34), tray, [0, 1.03, 0.27]));
  payload.add(simpleMesh(new THREE.BoxGeometry(0.27, 0.11, 0.2), pouch, [0, 1.1, 0.27]));
  return payload;
}

function makeDishwashingEffect(): THREE.Group {
  const effect = new THREE.Group();
  effect.name = "dishwashing-evidence";
  const plate = simpleMesh(
    new THREE.CylinderGeometry(0.2, 0.2, 0.025, 32),
    new THREE.MeshStandardMaterial({ color: "#ece9df", roughness: 0.36 }),
    [0.34, 1.02, 0],
  );
  plate.rotation.z = Math.PI / 2;
  effect.add(plate);
  const water = new THREE.MeshBasicMaterial({ color: "#98e5e0", transparent: true, opacity: 0.72 });
  for (let index = 0; index < 8; index += 1) {
    effect.add(simpleMesh(new THREE.SphereGeometry(0.018, 10, 8), water, [
      0.27 + (index % 3) * 0.08,
      1.18 + Math.floor(index / 3) * 0.09,
      (index % 2 ? 1 : -1) * 0.08,
    ]));
  }
  effect.visible = false;
  return effect;
}

function makeSensorFan(): THREE.Group {
  const fan = new THREE.Group();
  fan.name = "stereo-depth-fan";
  for (let index = -4; index <= 4; index += 1) {
    const angle = index * 0.12;
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 1.5, 0.08),
      new THREE.Vector3(Math.sin(angle) * 2.6, 1.34, Math.cos(angle) * 2.6),
    ]);
    fan.add(new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: "#c8f46d", transparent: true, opacity: 0.36 }),
    ));
  }
  fan.visible = false;
  return fan;
}

function makeMemoryRings(): THREE.Group {
  const rings = new THREE.Group();
  for (let index = 0; index < 3; index += 1) {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.34 + index * 0.24, 0.36 + index * 0.24, 48),
      new THREE.MeshBasicMaterial({ color: "#c8f46d", transparent: true, opacity: 0.34 - index * 0.08, side: THREE.DoubleSide }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.04 + index * 0.015;
    rings.add(ring);
  }
  rings.visible = false;
  return rings;
}

function deliveryProgress(progress: number): number {
  if (progress < 0.145) return 0;
  if (progress < 0.29) return smooth((progress - 0.145) / 0.145) * 0.5;
  if (progress < 0.32) return 0.5;
  if (progress < 0.41) return THREE.MathUtils.lerp(0.5, 1, smooth((progress - 0.32) / 0.09));
  return 1;
}

function makeRoute(): THREE.Line {
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(4.15, 0.055, 3.1),
    new THREE.Vector3(3.15, 0.055, 3.28),
    new THREE.Vector3(1.85, 0.055, 3.3),
    new THREE.Vector3(0.35, 0.055, 2.95),
    new THREE.Vector3(-0.72, 0.055, 1.72),
    new THREE.Vector3(-0.95, 0.055, 0.18),
    new THREE.Vector3(-1.45, 0.055, -0.9),
    new THREE.Vector3(-1.9, 0.055, -1.35),
  ]);
  const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(90));
  const material = new THREE.LineDashedMaterial({
    color: "#c7ff73",
    dashSize: 0.16,
    gapSize: 0.11,
    transparent: true,
    opacity: 0.9,
  });
  const line = new THREE.Line(geometry, material);
  line.computeLineDistances();
  return line;
}

function normalizeAsset(
  asset: THREE.Object3D,
  targetSize: number,
  sizing: "height" | "footprint",
): THREE.Object3D {
  asset.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(asset);
  const size = bounds.getSize(new THREE.Vector3());
  const sourceSize = sizing === "height" ? size.y : Math.max(size.x, size.z);
  const scale = targetSize / Math.max(sourceSize, 0.001);
  asset.scale.setScalar(scale);
  asset.updateMatrixWorld(true);

  const scaledBounds = new THREE.Box3().setFromObject(asset);
  const center = scaledBounds.getCenter(new THREE.Vector3());
  asset.position.x -= center.x;
  asset.position.z -= center.z;
  asset.position.y -= scaledBounds.min.y;
  asset.updateMatrixWorld(true);
  return asset;
}

function prepareVisualAsset(asset: THREE.Object3D, castShadow: boolean): void {
  asset.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    object.castShadow = castShadow;
    object.receiveShadow = true;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((surface) => {
      if ("envMapIntensity" in surface) {
        (surface as THREE.MeshStandardMaterial).envMapIntensity = 0.7;
      }
    });
  });
}

export function LandingWorld({ progress, reducedMotion = false }: LandingWorldProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [loadProgress, setLoadProgress] = useState(0);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let disposed = false;
    let frame = 0;
    let framePending = false;
    let pageVisible = document.visibilityState === "visible";
    let worldVisible = true;
    let requestFrame = () => {};
    const compactViewport = window.matchMedia("(max-width: 700px)").matches;
    const frameInterval = 1_000 / (compactViewport ? 40 : 60);
    let previousFrameTime = -frameInterval;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#343c38");
    scene.fog = new THREE.FogExp2("#414a45", 0.035);

    const camera = new THREE.PerspectiveCamera(41, 1, 0.04, 80);
    camera.position.set(7.4, 4.6, 7.7);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, compactViewport ? 1.25 : 1.5));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.domElement.setAttribute("aria-hidden", "true");
    host.appendChild(renderer.domElement);

    const worldRoot = new THREE.Group();
    worldRoot.name = "appearance-only-world";
    scene.add(worldRoot);

    const route = makeRoute();
    scene.add(route);

    const destination = new THREE.Mesh(
      new THREE.RingGeometry(0.25, 0.39, 48),
      new THREE.MeshBasicMaterial({ color: "#ff795e", transparent: true, opacity: 0.9, side: THREE.DoubleSide }),
    );
    destination.rotation.x = -Math.PI / 2;
    destination.position.set(-1.9, 0.07, -1.35);
    scene.add(destination);

    const basket = makeBasket();
    basket.position.set(-0.28, 0, 0.5);
    scene.add(basket);

    const resident = makeResident();
    resident.position.set(-2.25, 0, -1.7);
    resident.rotation.y = 0.7;
    scene.add(resident);

    const memoryRings = makeMemoryRings();
    memoryRings.position.set(-0.28, 0, 0.5);
    scene.add(memoryRings);

    const dotGeometry = new THREE.BufferGeometry();
    const dotPositions = new Float32Array(140 * 3);
    for (let index = 0; index < 140; index += 1) {
      dotPositions[index * 3] = (Math.random() - 0.5) * 8.5;
      dotPositions[index * 3 + 1] = Math.random() * 3.2;
      dotPositions[index * 3 + 2] = (Math.random() - 0.5) * 6.5;
    }
    dotGeometry.setAttribute("position", new THREE.BufferAttribute(dotPositions, 3));
    const dots = new THREE.Points(
      dotGeometry,
      new THREE.PointsMaterial({ color: "#dfffd0", size: 0.018, transparent: true, opacity: 0.28 }),
    );
    scene.add(dots);

    scene.add(new THREE.HemisphereLight("#e8f1df", "#43382f", 2.1));
    const sun = new THREE.DirectionalLight("#fff1d6", 4.2);
    sun.position.set(2.4, 8.2, -5.8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.left = -7;
    sun.shadow.camera.right = 7;
    sun.shadow.camera.top = 7;
    sun.shadow.camera.bottom = -7;
    scene.add(sun);
    const warm = new THREE.PointLight("#ff9d75", 3.8, 5.5, 2);
    warm.position.set(-2.4, 2.1, 1.8);
    scene.add(warm);

    const manager = new THREE.LoadingManager();
    manager.onProgress = (_url, loaded, total) => {
      if (!disposed) setLoadProgress(Math.round((loaded / Math.max(total, 1)) * 100));
    };
    const loader = new GLTFLoader(manager);
    let robot: THREE.Group | null = null;
    let sensorFan: THREE.Group | null = null;
    let payload: THREE.Group | null = null;
    let dishwashingEffect: THREE.Group | null = null;
    const appearanceMeshes: THREE.Mesh[] = [];

    const registerCameraOccluders = (root: THREE.Object3D) => {
      root.traverse((object) => {
        if (object instanceof THREE.Mesh) appearanceMeshes.push(object);
      });
    };

    const loadAssets = async () => {
      try {
        const [homeAsset, robotAsset] = await Promise.all([
          loader.loadAsync("/assets/models/loft-interior.glb"),
          loader.loadAsync("/assets/models/mm01-visual-shell.glb"),
        ]);
        if (disposed) return;
        const environmentSphere = homeAsset.scene.getObjectByName("Sphere");
        environmentSphere?.removeFromParent();
        const home = normalizeAsset(homeAsset.scene, 8.6, "footprint");
        const robotVisual = normalizeAsset(robotAsset.scene, 1.72, "height");
        prepareVisualAsset(home, false);
        prepareVisualAsset(robotVisual, true);
        registerCameraOccluders(home);
        robot = new THREE.Group();
        robot.name = "mm01-visual-rig";
        sensorFan = makeSensorFan();
        payload = makePayload();
        dishwashingEffect = makeDishwashingEffect();
        robot.add(robotVisual, sensorFan, payload, dishwashingEffect);
        robot.position.set(4.15, 0, 3.1);
        robot.rotation.y = Math.PI;
        worldRoot.add(home, robot);
        setLoadState("ready");
        requestFrame();
      } catch (error) {
        if (disposed) return;
        console.warn("The supplied visual models could not load; using the cached scene.", error);
        const home = buildApartment();
        const robotVisual = makeRobot();
        registerCameraOccluders(home);
        robot = new THREE.Group();
        sensorFan = makeSensorFan();
        payload = makePayload();
        dishwashingEffect = makeDishwashingEffect();
        robot.add(robotVisual, sensorFan, payload, dishwashingEffect);
        robot.position.set(4.15, 0, 3.1);
        worldRoot.add(home, robot);
        setLoadState("fallback");
        requestFrame();
      }
    };
    void loadAssets();

    const routeCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(4.15, 0, 3.1),
      new THREE.Vector3(3.15, 0, 3.28),
      new THREE.Vector3(1.85, 0, 3.3),
      new THREE.Vector3(0.35, 0, 2.95),
      new THREE.Vector3(-0.72, 0, 1.72),
      new THREE.Vector3(-0.95, 0, 0.18),
      new THREE.Vector3(-1.45, 0, -0.9),
      new THREE.Vector3(-1.9, 0, -1.35),
    ]);
    const kitchenCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-1.9, 0, -1.35),
      new THREE.Vector3(-1.4, 0, -0.65),
      new THREE.Vector3(-0.1, 0, 0.1),
      new THREE.Vector3(1.25, 0, 0.05),
      new THREE.Vector3(2.35, 0, -0.65),
      new THREE.Vector3(2.82, 0, -1.42),
    ]);
    const returnCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(2.82, 0, -1.42),
      new THREE.Vector3(2.25, 0, -0.55),
      new THREE.Vector3(1.25, 0, 0.35),
      new THREE.Vector3(0.05, 0, 0.42),
    ]);
    const widePosition = new THREE.Vector3(7.4, 4.6, 7.7);
    const wideTarget = new THREE.Vector3(0, 0.9, -0.1);
    const finalPosition = new THREE.Vector3(10.4, 7.4, 12.2);
    const finalTarget = new THREE.Vector3(0.05, 0.8, 0.2);
    const cameraPosition = new THREE.Vector3();
    const cameraTarget = new THREE.Vector3();
    const forward = new THREE.Vector3();
    const side = new THREE.Vector3();
    const safeShotPosition = widePosition.clone();
    const raycaster = new THREE.Raycaster();
    const rayDirection = new THREE.Vector3();
    const shotTarget = new THREE.Vector3();
    const clock = new THREE.Clock();
    let shotFrame = 0;
    let pointerX = 0;
    let pointerY = 0;

    const onPointerMove = (event: PointerEvent) => {
      const bounds = host.getBoundingClientRect();
      pointerX = ((event.clientX - bounds.left) / bounds.width - 0.5) * 2;
      pointerY = ((event.clientY - bounds.top) / bounds.height - 0.5) * 2;
      requestFrame();
    };
    const onPointerLeave = () => {
      pointerX = 0;
      pointerY = 0;
      requestFrame();
    };
    host.addEventListener("pointermove", onPointerMove);
    host.addEventListener("pointerleave", onPointerLeave);

    const render = (frameTime: number) => {
      framePending = false;
      if (disposed || !pageVisible || !worldVisible) return;
      if (!reducedMotion && frameTime - previousFrameTime < frameInterval) {
        requestFrame();
        return;
      }
      previousFrameTime = frameTime;
      const elapsed = reducedMotion ? 0 : clock.getElapsedTime();
      const p = clamp(progress.get());
      const delivery = deliveryProgress(p);
      const kitchen = smooth((p - 0.48) / 0.12);
      const returning = smooth((p - 0.72) / 0.1);

      if (robot) {
        let routePosition = routeCurve.getPoint(delivery);
        let routeTangent = routeCurve.getTangent(Math.min(0.999, delivery + 0.002));
        if (p >= 0.48) {
          routePosition = kitchenCurve.getPoint(kitchen);
          routeTangent = kitchenCurve.getTangent(Math.min(0.999, kitchen + 0.002));
        }
        if (p >= 0.72) {
          routePosition = returnCurve.getPoint(returning);
          routeTangent = returnCurve.getTangent(Math.min(0.999, returning + 0.002));
        }
        robot.position.copy(routePosition);
        forward.copy(routeTangent).normalize();
        const walking = (p >= 0.145 && p < 0.29) || (p >= 0.32 && p < 0.41)
          || (p >= 0.48 && p < 0.6) || (p >= 0.72 && p < 0.82);
        const scanning = p >= 0.22 && p < 0.31;
        const washing = p >= 0.6 && p < 0.72;
        robot.rotation.y = Math.atan2(forward.x, forward.z) + (scanning ? Math.sin(elapsed * 1.7) * 0.34 : 0);
        if (p >= 0.4 && p < 0.48) {
          const residentDirection = resident.position.clone().sub(robot.position);
          robot.rotation.y = Math.atan2(residentDirection.x, residentDirection.z);
        }
        if (washing) robot.rotation.y = THREE.MathUtils.lerp(robot.rotation.y, Math.PI / 2, 0.12);
        if (p >= 0.84) robot.rotation.y = THREE.MathUtils.lerp(robot.rotation.y, Math.PI, 0.08);
        robot.position.y = walking
          ? Math.abs(Math.sin(elapsed * 7.4)) * 0.025
          : washing ? Math.sin(elapsed * 2.8) * 0.008 : 0;
      } else {
        forward.set(0, 0, -1);
      }

      const obstacleReveal = smooth((p - 0.28) / 0.045);
      basket.visible = p > 0.275 && p < 0.5;
      basket.scale.setScalar(Math.max(0.001, obstacleReveal));

      if (sensorFan) {
        sensorFan.visible = p >= 0.22 && p < 0.32;
        sensorFan.children.forEach((line, index) => {
          (line as THREE.Line).scale.setScalar(0.86 + Math.sin(elapsed * 2.4 + index * 0.28) * 0.08);
        });
      }
      if (payload) {
        payload.visible = p < 0.405;
        const stabilizedTilt = p >= 0.145 && p < 0.41 ? -Math.sin(elapsed * 3.7) * 0.018 : 0;
        payload.rotation.z = THREE.MathUtils.lerp(payload.rotation.z, stabilizedTilt, 0.08);
        payload.rotation.x = THREE.MathUtils.lerp(payload.rotation.x, stabilizedTilt * 0.45, 0.08);
      }
      if (dishwashingEffect) {
        dishwashingEffect.visible = p >= 0.595 && p < 0.725;
        dishwashingEffect.rotation.y = Math.sin(elapsed * 3.8) * 0.13;
        dishwashingEffect.children.slice(1).forEach((drop, index) => {
          drop.position.y = 1.17 + ((elapsed * 0.24 + index * 0.09) % 0.28);
          drop.scale.setScalar(0.75 + Math.sin(elapsed * 4.2 + index) * 0.2);
        });
      }

      memoryRings.visible = p >= 0.39 && p < 0.49;
      memoryRings.rotation.y = elapsed * 0.12;
      memoryRings.scale.setScalar(0.92 + Math.sin(elapsed * 2) * 0.08);

      const robotPosition = robot?.position ?? routeCurve.getPoint(delivery);
      const compactShot = host.clientWidth <= 700;
      shotTarget.copy(robotPosition).add(new THREE.Vector3(0, compactShot ? 1.38 : 0.92, 0));
      side.set(forward.z, 0, -forward.x).normalize();

      // Re-evaluate a small set of front and profile shots at 10 Hz. The first
      // clear line of sight wins; otherwise use the candidate with the most
      // open space. This keeps the camera out of the loft shell and furniture.
      shotFrame += 1;
      if (shotFrame % 6 === 0 || safeShotPosition.equals(widePosition)) {
        const frontDistance = (p >= 0.595 && p < 0.725 ? 3.0 : 3.5) + (compactShot ? 0.7 : 0);
        const height = (p >= 0.595 && p < 0.725 ? 2.05 : 2.25) + (compactShot ? 0.25 : 0);
        const profileDistance = compactShot ? 4.1 : 3.5;
        const candidates = [
          shotTarget.clone().add(forward.clone().multiplyScalar(frontDistance)).add(side.clone().multiplyScalar(1.05)).setY(height),
          shotTarget.clone().add(forward.clone().multiplyScalar(frontDistance)).add(side.clone().multiplyScalar(-1.05)).setY(height),
          shotTarget.clone().add(side.clone().multiplyScalar(profileDistance)).setY(height + 0.2),
          shotTarget.clone().add(side.clone().multiplyScalar(-profileDistance)).setY(height + 0.2),
          shotTarget.clone().add(forward.clone().multiplyScalar(2.4)).add(side.clone().multiplyScalar(0.9)).setY(3.5),
        ];
        let bestCandidate = candidates[0];
        let bestClearance = -1;
        for (const candidate of candidates) {
          rayDirection.copy(candidate).sub(shotTarget);
          const distance = rayDirection.length();
          raycaster.set(shotTarget, rayDirection.normalize());
          raycaster.far = Math.max(0.1, distance - 0.22);
          const hit = raycaster.intersectObjects(appearanceMeshes, false)[0];
          const clearance = hit ? hit.distance : Number.POSITIVE_INFINITY;
          if (clearance > bestClearance) {
            bestClearance = clearance;
            bestCandidate = candidate;
          }
          if (!hit) break;
        }
        safeShotPosition.copy(bestCandidate);
      }

      const followMix = smooth((p - 0.1) / 0.055);
      const finalMix = smooth((p - 0.9) / 0.07);
      cameraPosition.copy(widePosition).lerp(safeShotPosition, followMix).lerp(finalPosition, finalMix);
      cameraTarget.copy(wideTarget).lerp(shotTarget, followMix).lerp(finalTarget, finalMix);
      cameraPosition.x += pointerX * 0.1 * (1 - finalMix);
      cameraPosition.y -= pointerY * 0.07 * (1 - finalMix);
      if (reducedMotion) camera.position.copy(cameraPosition);
      else camera.position.lerp(cameraPosition, 0.085);
      camera.lookAt(cameraTarget);

      route.visible = p > 0.145 && p < 0.43;
      destination.rotation.z += 0.006;
      destination.scale.setScalar(1 + Math.sin(elapsed * 3.2) * 0.07);
      destination.visible = p > 0.35 && p < 0.49;
      resident.visible = p > 0.33 && p < 0.5;
      dots.rotation.y = elapsed * 0.008;
      const perceptionDots = smooth((p - 0.22) / 0.03) * (1 - smooth((p - 0.32) / 0.03));
      const memoryDots = smooth((p - 0.39) / 0.04) * (1 - smooth((p - 0.49) / 0.04));
      (dots.material as THREE.PointsMaterial).opacity = 0.05 + Math.max(perceptionDots, memoryDots) * 0.34;
      renderer.render(scene, camera);
      if (!reducedMotion) requestFrame();
    };

    requestFrame = () => {
      if (disposed || framePending || !pageVisible || !worldVisible) return;
      framePending = true;
      frame = window.requestAnimationFrame(render);
    };

    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      requestFrame();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(host);
    const visibilityObserver = new IntersectionObserver(([entry]) => {
      worldVisible = entry?.isIntersecting ?? true;
      requestFrame();
    }, { rootMargin: "160px 0px" });
    visibilityObserver.observe(host);
    const onVisibilityChange = () => {
      pageVisible = document.visibilityState === "visible";
      requestFrame();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    const stopProgressListener = progress.on("change", requestFrame);
    resize();
    requestFrame();

    return () => {
      disposed = true;
      window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      visibilityObserver.disconnect();
      stopProgressListener();
      document.removeEventListener("visibilitychange", onVisibilityChange);
      host.removeEventListener("pointermove", onPointerMove);
      host.removeEventListener("pointerleave", onPointerLeave);
      scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh || object instanceof THREE.Points || object instanceof THREE.Line)) return;
        object.geometry.dispose();
        const surfaces = Array.isArray(object.material) ? object.material : [object.material];
        surfaces.forEach((surface) => {
          const mapped = surface as THREE.MeshStandardMaterial;
          mapped.map?.dispose();
          mapped.normalMap?.dispose();
          mapped.roughnessMap?.dispose();
          mapped.metalnessMap?.dispose();
          surface.dispose();
        });
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [progress, reducedMotion]);

  return (
    <div
      className="mm-world"
      ref={hostRef}
      role="img"
      aria-label="Interactive 3D loft with MM-01 delivering medicine, working at the kitchen sink, and returning to the room center"
    >
      {loadState === "loading" && (
        <div className="mm-world-loader" role="status">
          <span>Assembling visual world</span>
          <i><b style={{ transform: `scaleX(${loadProgress / 100})` }} /></i>
          <strong>{loadProgress}%</strong>
        </div>
      )}
      {loadState === "fallback" && (
        <span className="mm-world-fallback">Cached visual world active</span>
      )}
    </div>
  );
}
