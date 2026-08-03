import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

import { buildTwoStoryHouse } from "./src/components/TwoStoryHouse";

const host = document.getElementById("host")!;
const scene = new THREE.Scene();
scene.background = new THREE.Color("#aebfc2");

const camera = new THREE.PerspectiveCamera(39, 1280 / 860, 0.04, 200);
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(1280, 860, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.02;
host.appendChild(renderer.domElement);

const generator = new THREE.PMREMGenerator(renderer);
scene.environment = generator.fromScene(new RoomEnvironment(), 0.035).texture;

const house = buildTwoStoryHouse();
scene.add(house.root);

scene.add(new THREE.HemisphereLight("#eafcff", "#6c5140", 1.82));
const sun = new THREE.DirectionalLight("#fff0ce", 5.05);
sun.position.set(5.4, 13.5, -10.8);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.left = -12;
sun.shadow.camera.right = 12;
sun.shadow.camera.top = 12;
sun.shadow.camera.bottom = -12;
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.035;
scene.add(sun);

const views: Record<string, [THREE.Vector3, THREE.Vector3]> = {
  wide: [new THREE.Vector3(15.2, 9.5, 15.7), new THREE.Vector3(-0.8, 2.78, -0.15)],
  stairs: [new THREE.Vector3(5.2, 4.6, 9.4), new THREE.Vector3(-1.1, 2.1, 1.9)],
  stairsClose: [new THREE.Vector3(2.6, 3.0, 6.4), new THREE.Vector3(-1.1, 1.7, 2.2)],
  upper: [new THREE.Vector3(11.0, 8.4, 10.6), new THREE.Vector3(-0.6, 3.9, -0.4)],
  upperFront: [new THREE.Vector3(1.4, 7.2, 12.0), new THREE.Vector3(-0.6, 3.4, 1.0)],
  ground: [new THREE.Vector3(10.4, 6.2, 11.2), new THREE.Vector3(0.2, 0.7, -0.2)],
  floorLine: [new THREE.Vector3(9.0, 1.05, 6.2), new THREE.Vector3(0.0, 0.55, -1.0)],
  beds: [new THREE.Vector3(9.4, 5.3, 4.2), new THREE.Vector3(1.2, 3.5, -2.4)],
  liftedUpper: [new THREE.Vector3(15.2, 9.5, 15.7), new THREE.Vector3(-0.8, 2.78, -0.15)],
};

declare global {
  interface Window {
    setView: (name: string) => void;
  }
}

window.setView = (name: string) => {
  const [position, target] = views[name] ?? views.wide;
  house.upperFloor.position.y = name === "liftedUpper" ? 1.15 : 0;
  camera.position.copy(position);
  camera.lookAt(target);
  renderer.render(scene, camera);
};

const query = new URLSearchParams(location.search);
window.setView(query.get("view") ?? "wide");
// Re-render a few times so async textures/env map are certainly in the buffer,
// then publish the caller's nonce so the screenshot driver knows this document
// (not the previous one) is the thing on screen.
let ticks = 0;
const settle = () => {
  renderer.render(scene, camera);
  ticks += 1;
  if (ticks < 6) requestAnimationFrame(settle);
  else document.title = `ready:${query.get("n") ?? ""}`;
};
settle();
