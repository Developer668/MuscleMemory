import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RectAreaLight } from "three";

import type { CorrectionPoint } from "./types";

type HomeSceneProps = {
  robotPosition: CorrectionPoint | null;
  robotYaw: number | null;
  path: CorrectionPoint[];
  correction: CorrectionPoint[];
  correctionKind: "route" | "keep_out";
  running: boolean;
};

const ROOM_WIDTH = 8;
const ROOM_DEPTH = 6;

function material(
  color: THREE.ColorRepresentation,
  roughness = 0.72,
  metalness = 0,
): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness });
}

function mesh(
  geometry: THREE.BufferGeometry,
  surface: THREE.Material,
  position: [number, number, number],
  rotation: [number, number, number] = [0, 0, 0],
): THREE.Mesh {
  const item = new THREE.Mesh(geometry, surface);
  item.position.set(...position);
  item.rotation.set(...rotation);
  item.castShadow = true;
  item.receiveShadow = true;
  return item;
}

function roundedBox(
  width: number,
  height: number,
  depth: number,
  radius: number,
): THREE.ExtrudeGeometry {
  const shape = new THREE.Shape();
  const x = -width / 2;
  const y = -height / 2;
  shape.moveTo(x + radius, y);
  shape.lineTo(x + width - radius, y);
  shape.quadraticCurveTo(x + width, y, x + width, y + radius);
  shape.lineTo(x + width, y + height - radius);
  shape.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  shape.lineTo(x + radius, y + height);
  shape.quadraticCurveTo(x, y + height, x, y + height - radius);
  shape.lineTo(x, y + radius);
  shape.quadraticCurveTo(x, y, x + radius, y);
  return new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: true,
    bevelSize: Math.min(radius * 0.35, 0.035),
    bevelThickness: Math.min(radius * 0.35, 0.035),
    bevelSegments: 3,
  });
}

function canvasTexture(
  draw: (context: CanvasRenderingContext2D, size: number) => void,
  repeat: [number, number],
): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2D texture context is unavailable");
  draw(context, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(...repeat);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

function woodTexture(): THREE.CanvasTexture {
  return canvasTexture((context, size) => {
    context.fillStyle = "#9d7049";
    context.fillRect(0, 0, size, size);
    const plankHeight = 64;
    for (let row = 0; row < size / plankHeight; row += 1) {
      const offset = row % 2 ? -96 : 0;
      for (let x = offset; x < size; x += 192) {
        const hue = 28 + ((row * 7 + x / 32) % 8);
        context.fillStyle = `hsl(${hue} 35% ${46 + ((row + x) % 5)}%)`;
        context.fillRect(x + 2, row * plankHeight + 2, 188, plankHeight - 4);
        context.strokeStyle = "rgba(56, 31, 16, .28)";
        context.strokeRect(x + 2, row * plankHeight + 2, 188, plankHeight - 4);
        for (let grain = 0; grain < 5; grain += 1) {
          context.beginPath();
          context.strokeStyle = `rgba(74, 43, 23, ${0.035 + grain * 0.008})`;
          const y = row * plankHeight + 9 + grain * 10;
          context.moveTo(x + 8, y);
          context.bezierCurveTo(x + 55, y - 4, x + 125, y + 5, x + 182, y - 1);
          context.stroke();
        }
      }
    }
  }, [4, 3]);
}

function fabricTexture(base: string, thread: string): THREE.CanvasTexture {
  return canvasTexture((context, size) => {
    context.fillStyle = base;
    context.fillRect(0, 0, size, size);
    context.strokeStyle = thread;
    context.lineWidth = 1;
    for (let index = 0; index < size; index += 5) {
      context.beginPath();
      context.moveTo(index, 0);
      context.lineTo(index, size);
      context.stroke();
      context.beginPath();
      context.moveTo(0, index);
      context.lineTo(size, index);
      context.stroke();
    }
  }, [3, 2]);
}

function addWallArt(scene: THREE.Group, x: number, z: number, tone: string): void {
  const frame = material("#241d18", 0.46, 0.08);
  const canvas = material(tone, 0.82);
  scene.add(mesh(new THREE.BoxGeometry(1.1, 0.72, 0.055), frame, [x, 1.78, z]));
  scene.add(mesh(new THREE.BoxGeometry(0.96, 0.58, 0.062), canvas, [x, 1.78, z - 0.035]));
}

function makePlant(scale = 1): THREE.Group {
  const plant = new THREE.Group();
  const pot = mesh(
    new THREE.CylinderGeometry(0.2 * scale, 0.16 * scale, 0.32 * scale, 28),
    material("#94644b", 0.74),
    [0, 0.16 * scale, 0],
  );
  plant.add(pot);
  const leafMaterial = material("#31563c", 0.68);
  for (let index = 0; index < 8; index += 1) {
    const leaf = mesh(
      new THREE.SphereGeometry(0.13 * scale, 18, 12),
      leafMaterial,
      [Math.sin(index * 1.9) * 0.13 * scale, (0.42 + (index % 3) * 0.1) * scale, Math.cos(index * 1.9) * 0.13 * scale],
      [Math.sin(index) * 0.5, 0, Math.cos(index) * 0.5],
    );
    leaf.scale.set(0.65, 1.75, 0.42);
    plant.add(leaf);
  }
  return plant;
}

function makeChair(color: string): THREE.Group {
  const chair = new THREE.Group();
  const wood = material("#614737", 0.63);
  const seat = material(color, 0.88);
  chair.add(mesh(new THREE.BoxGeometry(0.48, 0.1, 0.48), seat, [0, 0.48, 0]));
  chair.add(mesh(new THREE.BoxGeometry(0.5, 0.72, 0.09), seat, [0, 0.85, 0.2], [-0.08, 0, 0]));
  for (const [x, z] of [[-0.19, -0.18], [0.19, -0.18], [-0.19, 0.18], [0.19, 0.18]]) {
    chair.add(mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.45, 10), wood, [x, 0.225, z]));
  }
  return chair;
}

function buildApartment(): THREE.Group {
  const home = new THREE.Group();
  home.name = "visual-home";

  const floorMap = woodTexture();
  const floorMaterial = new THREE.MeshStandardMaterial({
    map: floorMap,
    color: "#b6875b",
    roughness: 0.58,
    metalness: 0.02,
  });
  home.add(mesh(new THREE.BoxGeometry(ROOM_WIDTH, 0.12, ROOM_DEPTH), floorMaterial, [0, -0.08, 0]));

  const wall = material("#e9e5de", 0.92);
  const trim = material("#f7f4ee", 0.74);
  home.add(mesh(new THREE.BoxGeometry(ROOM_WIDTH, 2.85, 0.14), wall, [0, 1.38, -ROOM_DEPTH / 2]));
  home.add(mesh(new THREE.BoxGeometry(0.14, 2.85, ROOM_DEPTH), wall, [-ROOM_WIDTH / 2, 1.38, 0]));
  home.add(mesh(new THREE.BoxGeometry(ROOM_WIDTH, 0.12, 0.12), trim, [0, 0.04, -ROOM_DEPTH / 2 + 0.08]));
  home.add(mesh(new THREE.BoxGeometry(0.12, 0.12, ROOM_DEPTH), trim, [-ROOM_WIDTH / 2 + 0.08, 0.04, 0]));

  // Window recess, glass, curtains, and a sunlit exterior plane.
  home.add(mesh(new THREE.BoxGeometry(2.25, 1.42, 0.08), material("#c8d8df", 0.08, 0.1), [1.65, 1.58, -3.075]));
  home.add(mesh(new THREE.BoxGeometry(2.45, 0.08, 0.14), trim, [1.65, 2.32, -3.02]));
  home.add(mesh(new THREE.BoxGeometry(0.08, 1.5, 0.14), trim, [0.47, 1.58, -3.02]));
  home.add(mesh(new THREE.BoxGeometry(0.08, 1.5, 0.14), trim, [2.83, 1.58, -3.02]));
  home.add(mesh(new THREE.BoxGeometry(0.055, 1.48, 0.16), material("#d9d4c9", 0.95), [0.32, 1.58, -2.9]));
  home.add(mesh(new THREE.BoxGeometry(0.055, 1.48, 0.16), material("#d9d4c9", 0.95), [2.98, 1.58, -2.9]));

  // Soft wool rug anchors the living area.
  const rugMap = fabricTexture("#67766f", "rgba(238, 238, 225, .08)");
  const rugMaterial = new THREE.MeshStandardMaterial({ map: rugMap, roughness: 0.96 });
  home.add(mesh(new THREE.BoxGeometry(3.25, 0.035, 2.15), rugMaterial, [1.15, 0.025, -0.38]));

  // Sofa with proper base, seat, back, arms, and loose cushions.
  const sofaFabric = fabricTexture("#576962", "rgba(255, 255, 255, .055)");
  const sofaMaterial = new THREE.MeshStandardMaterial({ map: sofaFabric, roughness: 0.92 });
  home.add(mesh(roundedBox(2.45, 0.47, 0.72, 0.12), sofaMaterial, [1.12, 0.39, -2.22], [-Math.PI / 2, 0, 0]));
  home.add(mesh(roundedBox(2.35, 0.62, 0.22, 0.1), sofaMaterial, [1.12, 0.92, -2.51], [-Math.PI / 2, 0, 0]));
  home.add(mesh(roundedBox(0.24, 0.68, 0.7, 0.09), sofaMaterial, [-0.18, 0.55, -2.22], [-Math.PI / 2, 0, 0]));
  home.add(mesh(roundedBox(0.24, 0.68, 0.7, 0.09), sofaMaterial, [2.42, 0.55, -2.22], [-Math.PI / 2, 0, 0]));
  const cushion = material("#c7a078", 0.95);
  home.add(mesh(roundedBox(0.5, 0.45, 0.13, 0.09), cushion, [0.45, 0.78, -2.06], [-Math.PI / 2, 0.15, 0.02]));
  home.add(mesh(roundedBox(0.5, 0.45, 0.13, 0.09), material("#ddd3bd", 0.95), [1.72, 0.78, -2.06], [-Math.PI / 2, -0.12, -0.03]));

  // Coffee table and small objects.
  const oak = material("#745039", 0.52);
  home.add(mesh(roundedBox(1.35, 0.68, 0.09, 0.08), oak, [1.12, 0.43, -0.52], [-Math.PI / 2, 0, 0]));
  for (const [x, z] of [[0.58, -0.78], [1.66, -0.78], [0.58, -0.26], [1.66, -0.26]]) {
    home.add(mesh(new THREE.CylinderGeometry(0.035, 0.05, 0.39, 12), material("#3f332b", 0.5), [x, 0.2, z]));
  }
  home.add(mesh(new THREE.CylinderGeometry(0.12, 0.1, 0.09, 24), material("#ece7dc", 0.4), [0.9, 0.54, -0.5]));
  home.add(mesh(new THREE.BoxGeometry(0.42, 0.05, 0.3), material("#a06448", 0.75), [1.35, 0.52, -0.48], [0, 0.2, 0]));

  // Kitchen run and dining zone.
  const cabinet = material("#c9c2b4", 0.76);
  const counter = material("#464747", 0.28, 0.08);
  for (let index = 0; index < 5; index += 1) {
    home.add(mesh(new THREE.BoxGeometry(0.78, 0.88, 0.62), cabinet, [-3.55, 0.44, -2.55 + index * 0.69]));
    home.add(mesh(new THREE.BoxGeometry(0.03, 0.52, 0.02), material("#8f8b84", 0.3, 0.8), [-3.225, 0.5, -2.55 + index * 0.69]));
  }
  home.add(mesh(new THREE.BoxGeometry(0.78, 0.08, 3.52), counter, [-3.55, 0.92, -1.16]));
  home.add(mesh(new THREE.BoxGeometry(0.62, 1.86, 0.86), material("#d7d7d3", 0.25, 0.52), [-3.47, 0.93, 1.74]));

  home.add(mesh(new THREE.CylinderGeometry(0.77, 0.77, 0.1, 44), oak, [1.95, 0.76, 1.75]));
  home.add(mesh(new THREE.CylinderGeometry(0.12, 0.24, 0.7, 20), material("#4c3d32", 0.52), [1.95, 0.36, 1.75]));
  const diningChairs = [
    [0.95, 1.72, Math.PI / 2],
    [2.95, 1.72, -Math.PI / 2],
    [1.95, 0.78, 0],
  ] as const;
  for (const [x, z, rotation] of diningChairs) {
    const chair = makeChair("#9c7358");
    chair.position.set(x, 0, z);
    chair.rotation.y = rotation;
    home.add(chair);
  }

  // Familiar household clutter, deliberately restrained.
  const basket = new THREE.Group();
  basket.add(mesh(new THREE.CylinderGeometry(0.34, 0.27, 0.55, 28, 1, true), material("#b7986e", 0.88), [0, 0.275, 0]));
  basket.add(mesh(new THREE.TorusGeometry(0.32, 0.035, 8, 28), material("#967750", 0.8), [0, 0.55, 0], [Math.PI / 2, 0, 0]));
  basket.position.set(-2.48, 0, 1.85);
  home.add(basket);
  home.add(mesh(new THREE.BoxGeometry(0.72, 0.54, 0.58), material("#a77b4e", 0.86), [-1.9, 0.27, -0.18], [0, 0.2, 0]));
  home.add(mesh(new THREE.BoxGeometry(0.52, 0.42, 0.48), material("#bb9362", 0.86), [-1.95, 0.75, -0.18], [0, -0.1, 0]));
  home.add(mesh(new THREE.CylinderGeometry(0.3, 0.34, 0.47, 28), material("#746557", 0.8), [3.08, 0.235, 0.5]));

  const plantA = makePlant(1.1);
  plantA.position.set(3.25, 0, -2.2);
  home.add(plantA);
  const plantB = makePlant(0.78);
  plantB.position.set(-2.65, 0, -2.55);
  home.add(plantB);

  addWallArt(home, -1.4, -3.08, "#a87456");
  addWallArt(home, -0.15, -3.08, "#4e6865");

  // Warm practical lights complement the cool window illumination.
  const pendant = new THREE.Group();
  pendant.add(mesh(new THREE.CylinderGeometry(0.018, 0.018, 1.0, 10), material("#292725", 0.35, 0.75), [0, 2.6, 0]));
  pendant.add(mesh(new THREE.CylinderGeometry(0.05, 0.34, 0.24, 32, 1, true), material("#3c3832", 0.38, 0.7), [0, 2.12, 0]));
  const bulb = new THREE.PointLight("#ffc981", 14, 4.6, 2);
  bulb.position.set(0, 2.0, 0);
  bulb.castShadow = true;
  pendant.add(bulb);
  pendant.position.set(1.95, 0, 1.75);
  home.add(pendant);

  return home;
}

function makeRobot(): THREE.Group {
  const robot = new THREE.Group();
  robot.name = "mm-01-visual-body";
  const shell = material("#e7e9e5", 0.34, 0.28);
  const carbon = material("#24292b", 0.3, 0.58);
  const joint = material("#596164", 0.24, 0.72);
  const visor = new THREE.MeshPhysicalMaterial({
    color: "#14282c",
    roughness: 0.06,
    metalness: 0.18,
    transmission: 0.08,
    clearcoat: 1,
  });

  robot.add(mesh(new THREE.CapsuleGeometry(0.22, 0.34, 8, 18), shell, [0, 1.17, 0]));
  robot.add(mesh(roundedBox(0.43, 0.56, 0.26, 0.1), shell, [0, 1.48, -0.13], [-Math.PI / 2, 0, 0]));
  robot.add(mesh(new THREE.SphereGeometry(0.19, 28, 20), shell, [0, 1.89, 0]));
  robot.add(mesh(new THREE.BoxGeometry(0.31, 0.1, 0.16), visor, [0, 1.91, 0.15], [-0.06, 0, 0]));
  for (const x of [-0.09, 0.09]) {
    robot.add(mesh(new THREE.SphereGeometry(0.025, 18, 12), material("#8fe5d0", 0.12, 0.2), [x, 1.92, 0.238]));
  }

  for (const side of [-1, 1]) {
    const arm = new THREE.Group();
    arm.name = side < 0 ? "left-arm" : "right-arm";
    arm.position.set(side * 0.31, 1.58, 0);
    arm.add(mesh(new THREE.CapsuleGeometry(0.07, 0.29, 6, 14), shell, [0, -0.2, 0], [0, 0, side * 0.08]));
    arm.add(mesh(new THREE.SphereGeometry(0.075, 16, 12), joint, [0, -0.42, 0]));
    arm.add(mesh(new THREE.CapsuleGeometry(0.06, 0.28, 6, 14), carbon, [0, -0.62, 0.06], [0.16, 0, 0]));
    robot.add(arm);

    const leg = new THREE.Group();
    leg.name = side < 0 ? "left-leg" : "right-leg";
    leg.position.set(side * 0.13, 1.05, 0);
    leg.add(mesh(new THREE.CapsuleGeometry(0.09, 0.38, 7, 16), shell, [0, -0.28, 0]));
    leg.add(mesh(new THREE.SphereGeometry(0.09, 16, 12), joint, [0, -0.53, 0]));
    leg.add(mesh(new THREE.CapsuleGeometry(0.075, 0.38, 7, 16), carbon, [0, -0.8, 0.015]));
    leg.add(mesh(roundedBox(0.18, 0.31, 0.1, 0.04), shell, [0, -1.08, 0.095], [-Math.PI / 2, 0, 0]));
    robot.add(leg);
  }

  const tray = mesh(roundedBox(0.7, 0.48, 0.055, 0.07), material("#afb7b8", 0.18, 0.72), [0, 1.02, 0.42], [-Math.PI / 2, 0, 0]);
  tray.name = "delivery-tray";
  robot.add(tray);
  robot.add(mesh(roundedBox(0.31, 0.2, 0.1, 0.05), material("#b84d46", 0.72), [0, 1.12, 0.44], [-Math.PI / 2, 0, 0]));
  robot.scale.setScalar(0.72);
  return robot;
}

function routeLine(points: CorrectionPoint[], color: string, dashed = false): THREE.Line | null {
  if (points.length < 2) return null;
  const curvePoints = points.map((point) => new THREE.Vector3(point.x_m - 4, 0.045, point.y_m - 3));
  const geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
  const surface = dashed
    ? new THREE.LineDashedMaterial({ color, linewidth: 2, dashSize: 0.17, gapSize: 0.1, transparent: true, opacity: 0.92 })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.78 });
  const line = new THREE.Line(geometry, surface);
  if (dashed) line.computeLineDistances();
  line.renderOrder = 5;
  return line;
}

export function RealisticHomeScene({
  robotPosition,
  robotYaw,
  path,
  correction,
  correctionKind,
  running,
}: HomeSceneProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const liveRef = useRef({ robotPosition, robotYaw, path, correction, correctionKind, running });

  useEffect(() => {
    liveRef.current = { robotPosition, robotYaw, path, correction, correctionKind, running };
  }, [robotPosition, robotYaw, path, correction, correctionKind, running]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#c7d4d7");
    scene.fog = new THREE.FogExp2("#d8dfdf", 0.018);

    const camera = new THREE.PerspectiveCamera(43, 1, 0.05, 80);
    camera.position.set(8.9, 7.2, 9.4);
    camera.lookAt(0, 0.8, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.055;
    controls.target.set(0, 0.75, -0.25);
    controls.minDistance = 4.7;
    controls.maxDistance = 18;
    controls.minPolarAngle = 0.32;
    controls.maxPolarAngle = Math.PI / 2.04;
    controls.maxAzimuthAngle = Math.PI * 0.95;
    controls.minAzimuthAngle = -Math.PI * 0.45;

    scene.add(buildApartment());
    const robot = makeRobot();
    robot.position.set(-2.8, 0, 2.25);
    robot.rotation.y = Math.PI;
    scene.add(robot);

    const destination = new THREE.Group();
    destination.add(mesh(new THREE.RingGeometry(0.28, 0.42, 48), material("#e56b55", 0.45, 0.1), [0, 0.025, 0], [-Math.PI / 2, 0, 0]));
    const beacon = new THREE.PointLight("#ff806b", 2.2, 2.6, 2);
    beacon.position.set(0, 0.28, 0);
    destination.add(beacon);
    destination.position.set(2.7, 0, -1.86);
    scene.add(destination);

    const hemisphere = new THREE.HemisphereLight("#eaf7ff", "#6a5140", 2.1);
    scene.add(hemisphere);
    const sun = new THREE.DirectionalLight("#fff2d4", 5.2);
    sun.position.set(1.8, 7.5, -6.4);
    sun.target.position.set(0.5, 0, 0.2);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -7;
    sun.shadow.camera.right = 7;
    sun.shadow.camera.top = 7;
    sun.shadow.camera.bottom = -7;
    sun.shadow.bias = -0.00015;
    scene.add(sun, sun.target);
    const windowLight = new RectAreaLight("#cde7ff", 10, 3.0, 1.8);
    windowLight.position.set(1.6, 1.55, -2.78);
    windowLight.lookAt(1.3, 0.6, 0.4);
    scene.add(windowLight);

    let actualLine: THREE.Line | null = null;
    let correctionLine: THREE.Line | null = null;
    let previousPathLength = -1;
    let previousCorrectionKey = "";
    const targetPosition = new THREE.Vector3(-2.8, 0, 2.25);
    let targetYaw = Math.PI;
    const clock = new THREE.Clock();
    let frame = 0;

    const rebuildLines = () => {
      const next = liveRef.current;
      if (next.path.length !== previousPathLength) {
        if (actualLine) {
          scene.remove(actualLine);
          actualLine.geometry.dispose();
          (actualLine.material as THREE.Material).dispose();
        }
        actualLine = routeLine(next.path, "#49cab0");
        if (actualLine) scene.add(actualLine);
        previousPathLength = next.path.length;
      }
      const correctionKey = `${next.correctionKind}:${next.correction.map((point) => `${point.x_m}:${point.y_m}`).join("|")}`;
      if (correctionKey !== previousCorrectionKey) {
        if (correctionLine) {
          scene.remove(correctionLine);
          correctionLine.geometry.dispose();
          (correctionLine.material as THREE.Material).dispose();
        }
        correctionLine = routeLine(next.correction, next.correctionKind === "keep_out" ? "#ef7763" : "#ffd36a", true);
        if (correctionLine) scene.add(correctionLine);
        previousCorrectionKey = correctionKey;
      }
    };

    const render = () => {
      frame = window.requestAnimationFrame(render);
      const next = liveRef.current;
      if (next.robotPosition) {
        targetPosition.set(next.robotPosition.x_m - 4, 0, next.robotPosition.y_m - 3);
      }
      if (next.robotYaw !== null) targetYaw = -next.robotYaw + Math.PI / 2;
      robot.position.lerp(targetPosition, 0.075);
      const delta = Math.atan2(Math.sin(targetYaw - robot.rotation.y), Math.cos(targetYaw - robot.rotation.y));
      robot.rotation.y += delta * 0.08;

      const elapsed = clock.getElapsedTime();
      const walking = next.running && robot.position.distanceTo(targetPosition) > 0.025;
      const gait = walking ? Math.sin(elapsed * 8.4) * 0.18 : 0;
      const leftLeg = robot.getObjectByName("left-leg");
      const rightLeg = robot.getObjectByName("right-leg");
      const leftArm = robot.getObjectByName("left-arm");
      const rightArm = robot.getObjectByName("right-arm");
      if (leftLeg && rightLeg && leftArm && rightArm) {
        leftLeg.rotation.x = gait;
        rightLeg.rotation.x = -gait;
        leftArm.rotation.x = -gait * 0.42;
        rightArm.rotation.x = gait * 0.42;
      }
      destination.rotation.y += 0.003;
      beacon.intensity = 1.7 + Math.sin(elapsed * 2.3) * 0.45;
      rebuildLines();
      controls.update();
      renderer.render(scene, camera);
    };

    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(host);
    resize();
    render();

    return () => {
      window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      controls.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          for (const item of materials) {
            const standard = item as THREE.MeshStandardMaterial;
            standard.map?.dispose();
            item.dispose();
          }
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div className="ops-home-scene" ref={hostRef}>
      <div className="ops-scene-vignette" aria-hidden="true" />
      <div className="ops-scene-caption">
        <span>INTERACTIVE HOME VIEW</span>
        <strong>Drag to orbit · Scroll to inspect</strong>
      </div>
    </div>
  );
}
