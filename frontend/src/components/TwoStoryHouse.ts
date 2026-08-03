import * as THREE from "three";

export const HOUSE_WIDTH = 13.2;
export const HOUSE_DEPTH = 9.2;
export const STORY_HEIGHT = 3.12;

export type TwoStoryHouseModel = {
  root: THREE.Group;
  upperFloor: THREE.Group;
};

function surface(
  color: THREE.ColorRepresentation,
  roughness = 0.72,
  metalness = 0,
): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness });
}

function item(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  position: [number, number, number],
  rotation: [number, number, number] = [0, 0, 0],
): THREE.Mesh {
  const result = new THREE.Mesh(geometry, material);
  result.position.set(...position);
  result.rotation.set(...rotation);
  result.castShadow = true;
  result.receiveShadow = true;
  return result;
}

function addBox(
  parent: THREE.Group,
  dimensions: [number, number, number],
  position: [number, number, number],
  material: THREE.Material,
  rotation: [number, number, number] = [0, 0, 0],
): THREE.Mesh {
  const result = item(new THREE.BoxGeometry(...dimensions), material, position, rotation);
  parent.add(result);
  return result;
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

function oakTexture(): THREE.CanvasTexture {
  return canvasTexture((context, size) => {
    context.fillStyle = "#93623d";
    context.fillRect(0, 0, size, size);
    const plank = 56;
    for (let row = 0; row < Math.ceil(size / plank); row += 1) {
      const y = row * plank;
      const offset = row % 2 === 0 ? -70 : 0;
      for (let x = offset; x < size; x += 170) {
        const light = 42 + ((row * 11 + Math.floor(x / 20)) % 9);
        context.fillStyle = `hsl(27 42% ${light}%)`;
        context.fillRect(x + 2, y + 2, 166, plank - 4);
        context.strokeStyle = "rgba(51, 28, 15, .34)";
        context.strokeRect(x + 2, y + 2, 166, plank - 4);
        for (let grain = 0; grain < 4; grain += 1) {
          context.beginPath();
          context.strokeStyle = `rgba(55, 28, 14, ${0.05 + grain * 0.012})`;
          context.moveTo(x + 8, y + 10 + grain * 11);
          context.bezierCurveTo(x + 54, y + 4 + grain * 12, x + 112, y + 17 + grain * 9, x + 160, y + 9 + grain * 11);
          context.stroke();
        }
      }
    }
  }, [6, 4]);
}

function tileTexture(base: string, grout: string, cells = 8): THREE.CanvasTexture {
  return canvasTexture((context, size) => {
    context.fillStyle = grout;
    context.fillRect(0, 0, size, size);
    const cell = size / cells;
    for (let y = 0; y < cells; y += 1) {
      for (let x = 0; x < cells; x += 1) {
        const variation = ((x * 13 + y * 7) % 5) * 2;
        context.fillStyle = base;
        context.globalAlpha = 0.91 + variation / 100;
        context.fillRect(x * cell + 3, y * cell + 3, cell - 6, cell - 6);
      }
    }
    context.globalAlpha = 1;
  }, [3, 3]);
}

function fabricTexture(base: string, thread: string): THREE.CanvasTexture {
  return canvasTexture((context, size) => {
    context.fillStyle = base;
    context.fillRect(0, 0, size, size);
    context.strokeStyle = thread;
    for (let index = 0; index < size; index += 6) {
      context.beginPath();
      context.moveTo(index, 0);
      context.lineTo(index, size);
      context.stroke();
      context.beginPath();
      context.moveTo(0, index);
      context.lineTo(size, index);
      context.stroke();
    }
  }, [3, 3]);
}

function makeWindow(width: number, height: number): THREE.Group {
  const window = new THREE.Group();
  const glass = new THREE.MeshPhysicalMaterial({
    color: "#b9d9e6",
    roughness: 0.08,
    metalness: 0.05,
    transmission: 0.26,
    transparent: true,
    opacity: 0.66,
    clearcoat: 1,
  });
  const frame = surface("#f5f2ea", 0.64);
  addBox(window, [width, height, 0.045], [0, 0, 0], glass);
  addBox(window, [width + 0.16, 0.08, 0.1], [0, height / 2 + 0.04, 0], frame);
  addBox(window, [width + 0.16, 0.08, 0.1], [0, -height / 2 - 0.04, 0], frame);
  addBox(window, [0.08, height, 0.1], [-width / 2 - 0.04, 0, 0], frame);
  addBox(window, [0.08, height, 0.1], [width / 2 + 0.04, 0, 0], frame);
  addBox(window, [0.055, height, 0.08], [0, 0, 0.035], frame);
  addBox(window, [width, 0.055, 0.08], [0, 0, 0.035], frame);
  return window;
}

function makePlant(scale = 1): THREE.Group {
  const plant = new THREE.Group();
  plant.add(item(
    new THREE.CylinderGeometry(0.22 * scale, 0.17 * scale, 0.34 * scale, 28),
    surface("#8d5c43", 0.82),
    [0, 0.17 * scale, 0],
  ));
  const green = surface("#294d38", 0.72);
  for (let index = 0; index < 10; index += 1) {
    const leaf = item(
      new THREE.SphereGeometry(0.13 * scale, 18, 12),
      green,
      [
        Math.sin(index * 1.8) * 0.17 * scale,
        (0.44 + (index % 4) * 0.1) * scale,
        Math.cos(index * 1.8) * 0.17 * scale,
      ],
      [Math.sin(index) * 0.42, 0, Math.cos(index) * 0.42],
    );
    leaf.scale.set(0.62, 1.72, 0.44);
    plant.add(leaf);
  }
  return plant;
}

function makeDiningChair(upholstery: THREE.Material): THREE.Group {
  const chair = new THREE.Group();
  const wood = surface("#4d3527", 0.52);
  addBox(chair, [0.48, 0.1, 0.48], [0, 0.48, 0], upholstery);
  addBox(chair, [0.5, 0.68, 0.09], [0, 0.83, 0.19], upholstery, [-0.07, 0, 0]);
  for (const x of [-0.19, 0.19]) {
    for (const z of [-0.18, 0.18]) {
      chair.add(item(new THREE.CylinderGeometry(0.024, 0.032, 0.44, 10), wood, [x, 0.22, z]));
    }
  }
  return chair;
}

function makeSofa(): THREE.Group {
  const sofa = new THREE.Group();
  const fabric = new THREE.MeshStandardMaterial({
    map: fabricTexture("#536961", "rgba(255,255,255,.06)"),
    color: "#667d74",
    roughness: 0.94,
  });
  const cushion = surface("#d7c6ab", 0.95);
  addBox(sofa, [3.05, 0.46, 0.92], [0, 0.4, 0], fabric);
  addBox(sofa, [2.9, 0.72, 0.24], [0, 0.9, -0.36], fabric, [-0.07, 0, 0]);
  addBox(sofa, [0.24, 0.67, 0.88], [-1.52, 0.56, 0], fabric);
  addBox(sofa, [0.24, 0.67, 0.88], [1.52, 0.56, 0], fabric);
  for (const x of [-0.92, 0, 0.92]) addBox(sofa, [0.78, 0.18, 0.72], [x, 0.69, 0.04], fabric);
  addBox(sofa, [0.52, 0.42, 0.14], [-0.9, 0.88, -0.03], cushion, [0, 0.08, -0.06]);
  addBox(sofa, [0.52, 0.42, 0.14], [0.94, 0.88, -0.03], surface("#c58062", 0.95), [0, -0.1, 0.05]);
  return sofa;
}

function makeBed(width: number, depth: number, color: string): THREE.Group {
  const bed = new THREE.Group();
  const frame = surface("#6a4935", 0.58);
  const linen = new THREE.MeshStandardMaterial({
    map: fabricTexture(color, "rgba(255,255,255,.09)"),
    color,
    roughness: 0.97,
  });
  addBox(bed, [width, 0.28, depth], [0, 0.34, 0], frame);
  addBox(bed, [width - 0.12, 0.28, depth - 0.14], [0, 0.58, 0], linen);
  addBox(bed, [width, 1.12, 0.16], [0, 0.78, -depth / 2 + 0.02], surface("#83705f", 0.9));
  addBox(bed, [width - 0.18, 0.08, depth * 0.48], [0, 0.75, 0.38], surface("#a9bdba", 0.95));
  for (const x of [-width * 0.24, width * 0.24]) {
    addBox(bed, [width * 0.4, 0.15, 0.44], [x, 0.82, -depth * 0.28], surface("#efe9de", 0.98));
  }
  return bed;
}

function makePendant(color: string): THREE.Group {
  const pendant = new THREE.Group();
  addBox(pendant, [0.025, 0.8, 0.025], [0, 0.4, 0], surface("#262827", 0.3, 0.78));
  pendant.add(item(
    new THREE.CylinderGeometry(0.08, 0.32, 0.28, 32, 1, true),
    surface(color, 0.36, 0.55),
    [0, 0.02, 0],
  ));
  const bulb = new THREE.Mesh(
    new THREE.SphereGeometry(0.07, 18, 12),
    new THREE.MeshStandardMaterial({ color: "#fff4cf", emissive: "#ffcf86", emissiveIntensity: 2.6 }),
  );
  bulb.position.y = -0.1;
  pendant.add(bulb);
  return pendant;
}

function makeResident(): THREE.Group {
  const resident = new THREE.Group();
  resident.name = "delivery-resident";
  const skin = surface("#a76d52", 0.78);
  const shirt = surface("#6c8198", 0.92);
  const trousers = surface("#3f474d", 0.9);
  const shoe = surface("#262a2b", 0.56);
  resident.add(item(new THREE.CapsuleGeometry(0.18, 0.42, 7, 18), shirt, [0, 1.23, 0], [0.08, 0, 0]));
  resident.add(item(new THREE.SphereGeometry(0.17, 24, 18), skin, [0, 1.68, 0.02]));
  const hair = item(new THREE.SphereGeometry(0.174, 24, 14, 0, Math.PI * 2, 0, Math.PI * 0.52), surface("#4c382e", 0.86), [0, 1.72, 0.01]);
  hair.rotation.x = -0.18;
  resident.add(hair);
  for (const side of [-1, 1]) {
    resident.add(item(new THREE.CapsuleGeometry(0.055, 0.36, 6, 12), skin, [side * 0.23, 1.2, 0.09], [0.52, 0, side * 0.18]));
    resident.add(item(new THREE.CapsuleGeometry(0.085, 0.38, 7, 14), trousers, [side * 0.12, 0.78, 0.22], [Math.PI / 2, 0, 0]));
    resident.add(item(new THREE.CapsuleGeometry(0.075, 0.36, 7, 14), trousers, [side * 0.12, 0.43, 0.48], [0.12, 0, 0]));
    resident.add(item(new THREE.BoxGeometry(0.17, 0.1, 0.28), shoe, [side * 0.12, 0.18, 0.61]));
  }
  return resident;
}

function addBackWall(
  parent: THREE.Group,
  baseY: number,
  windows: Array<{ x: number; width: number }>,
): void {
  const wall = surface("#e8e5dc", 0.94);
  const height = 1.52;
  const sill = 0.72;
  const top = STORY_HEIGHT - sill - height;
  addBox(parent, [HOUSE_WIDTH, sill, 0.16], [0, baseY + sill / 2, -HOUSE_DEPTH / 2], wall);
  addBox(parent, [HOUSE_WIDTH, top, 0.16], [0, baseY + sill + height + top / 2, -HOUSE_DEPTH / 2], wall);
  let cursor = -HOUSE_WIDTH / 2;
  for (const opening of [...windows].sort((a, b) => a.x - b.x)) {
    const left = opening.x - opening.width / 2;
    if (left > cursor) {
      addBox(parent, [left - cursor, height, 0.16], [(left + cursor) / 2, baseY + sill + height / 2, -HOUSE_DEPTH / 2], wall);
    }
    const window = makeWindow(opening.width - 0.12, height - 0.12);
    window.position.set(opening.x, baseY + sill + height / 2, -HOUSE_DEPTH / 2 + 0.02);
    parent.add(window);
    cursor = opening.x + opening.width / 2;
  }
  if (cursor < HOUSE_WIDTH / 2) {
    addBox(parent, [HOUSE_WIDTH / 2 - cursor, height, 0.16], [(HOUSE_WIDTH / 2 + cursor) / 2, baseY + sill + height / 2, -HOUSE_DEPTH / 2], wall);
  }
}

function addLeftWall(parent: THREE.Group, baseY: number): void {
  const wall = surface("#e7e3da", 0.94);
  addBox(parent, [0.16, STORY_HEIGHT, HOUSE_DEPTH], [-HOUSE_WIDTH / 2, baseY + STORY_HEIGHT / 2, 0], wall);
  for (const z of [-2.4, 2.1]) {
    const window = makeWindow(1.65, 1.38);
    window.rotation.y = Math.PI / 2;
    window.position.set(-HOUSE_WIDTH / 2 + 0.02, baseY + 1.55, z);
    parent.add(window);
  }
}

function addTrim(parent: THREE.Group, baseY: number): void {
  const trim = surface("#f7f4ec", 0.72);
  addBox(parent, [HOUSE_WIDTH, 0.12, 0.12], [0, baseY + 0.07, -HOUSE_DEPTH / 2 + 0.11], trim);
  addBox(parent, [0.12, 0.12, HOUSE_DEPTH], [-HOUSE_WIDTH / 2 + 0.11, baseY + 0.07, 0], trim);
}

function addKitchen(parent: THREE.Group): void {
  const cabinet = surface("#c6c0b4", 0.7);
  const charcoal = surface("#343838", 0.24, 0.12);
  const steel = surface("#c4c8c6", 0.22, 0.72);
  for (let index = 0; index < 7; index += 1) {
    const x = -5.85 + index * 0.7;
    addBox(parent, [0.66, 0.86, 0.64], [x, 0.43, -4.12], cabinet);
    addBox(parent, [0.028, 0.28, 0.026], [x + 0.24, 0.48, -3.79], steel);
    if (index !== 3) addBox(parent, [0.66, 0.72, 0.38], [x, 2.05, -4.3], cabinet);
  }
  addBox(parent, [4.88, 0.08, 0.72], [-3.75, 0.91, -4.08], charcoal);
  addBox(parent, [0.82, 1.92, 0.74], [-6.08, 0.96, -3.72], steel);
  addBox(parent, [0.72, 0.05, 0.58], [-3.72, 0.955, -4.05], surface("#191d1c", 0.3, 0.55));
  for (let burner = 0; burner < 4; burner += 1) {
    const ring = item(new THREE.TorusGeometry(0.1, 0.014, 8, 24), surface("#606767", 0.3, 0.7), [-3.94 + (burner % 2) * 0.42, 0.99, -4.18 + Math.floor(burner / 2) * 0.27], [Math.PI / 2, 0, 0]);
    parent.add(ring);
  }
  addBox(parent, [2.75, 0.88, 0.92], [-3.25, 0.44, -1.85], cabinet);
  addBox(parent, [2.88, 0.09, 1.02], [-3.25, 0.92, -1.85], surface("#ebe4d8", 0.34));
  const sink = item(new THREE.CylinderGeometry(0.31, 0.31, 0.05, 32), steel, [-3.72, 0.99, -1.85], [Math.PI / 2, 0, 0]);
  sink.scale.z = 0.72;
  parent.add(sink);
  const faucet = item(new THREE.TorusGeometry(0.19, 0.025, 8, 24, Math.PI), steel, [-3.72, 1.18, -2.12], [0, 0, Math.PI / 2]);
  parent.add(faucet);
  for (const x of [-4.02, -2.5]) {
    const stool = new THREE.Group();
    stool.add(item(new THREE.CylinderGeometry(0.26, 0.26, 0.08, 28), surface("#7a573f", 0.58), [0, 0.68, 0]));
    stool.add(item(new THREE.CylinderGeometry(0.045, 0.055, 0.66, 12), surface("#333635", 0.35, 0.55), [0, 0.33, 0]));
    stool.position.set(x, 0, -1.16);
    parent.add(stool);
  }
  for (const x of [-3.85, -2.65]) {
    const pendant = makePendant("#725d47");
    pendant.position.set(x, 2.45, -1.84);
    parent.add(pendant);
  }
}

function addLivingRoom(parent: THREE.Group): void {
  const rug = new THREE.MeshStandardMaterial({
    map: fabricTexture("#687b77", "rgba(235,245,238,.08)"),
    color: "#718782",
    roughness: 0.97,
  });
  addBox(parent, [4.75, 0.035, 3.15], [3.55, 0.035, -1.62], rug);
  const sofa = makeSofa();
  sofa.position.set(3.55, 0, -3.65);
  parent.add(sofa);
  const resident = makeResident();
  resident.position.set(3.55, 0.03, -3.42);
  parent.add(resident);
  addBox(parent, [1.55, 0.11, 0.84], [3.55, 0.5, -1.6], surface("#724b34", 0.5));
  for (const x of [2.96, 4.14]) {
    for (const z of [-1.9, -1.3]) {
      parent.add(item(new THREE.CylinderGeometry(0.028, 0.04, 0.45, 10), surface("#282b2a", 0.28, 0.7), [x, 0.24, z]));
    }
  }
  addBox(parent, [2.35, 0.52, 0.44], [3.55, 0.26, 0.05], surface("#6a4b38", 0.58));
  addBox(parent, [1.62, 0.9, 0.08], [3.55, 1.25, -0.18], surface("#171b1b", 0.12, 0.38));
  addBox(parent, [1.5, 0.04, 0.03], [3.55, 0.78, -0.11], surface("#2b2f2f", 0.22, 0.65));
  const plant = makePlant(1.18);
  plant.position.set(5.62, 0, -3.78);
  parent.add(plant);
  const floorLamp = new THREE.Group();
  floorLamp.add(item(new THREE.CylinderGeometry(0.025, 0.035, 1.62, 12), surface("#282b2b", 0.3, 0.72), [0, 0.81, 0]));
  floorLamp.add(item(new THREE.CylinderGeometry(0.18, 0.34, 0.3, 32, 1, true), surface("#c5ad89", 0.78), [0, 1.55, 0]));
  floorLamp.position.set(5.45, 0, -0.2);
  parent.add(floorLamp);
}

function addDining(parent: THREE.Group): void {
  const oak = surface("#76503a", 0.52);
  addBox(parent, [2.8, 0.14, 1.28], [0.05, 0.78, 2.73], oak);
  for (const x of [-1.05, 1.15]) {
    for (const z of [2.3, 3.16]) parent.add(item(new THREE.CylinderGeometry(0.05, 0.07, 0.72, 12), surface("#393332", 0.38, 0.55), [x, 0.38, z]));
  }
  const upholstery = surface("#a7795e", 0.9);
  const placements: Array<[number, number, number]> = [
    [-1.55, 2.43, Math.PI / 2],
    [-1.55, 3.05, Math.PI / 2],
    [1.65, 2.43, -Math.PI / 2],
    [1.65, 3.05, -Math.PI / 2],
    [0.05, 1.82, 0],
    [0.05, 3.65, Math.PI],
  ];
  for (const [x, z, rotation] of placements) {
    const chair = makeDiningChair(upholstery);
    chair.position.set(x, 0, z);
    chair.rotation.y = rotation;
    parent.add(chair);
  }
  const pendant = makePendant("#353332");
  pendant.scale.setScalar(1.25);
  pendant.position.set(0.05, 2.35, 2.73);
  parent.add(pendant);
  parent.add(item(new THREE.CylinderGeometry(0.16, 0.12, 0.13, 24), surface("#ede8dd", 0.42), [0.05, 0.93, 2.72]));
}

function addFoyerAndStudy(parent: THREE.Group): void {
  const door = surface("#5d4131", 0.58);
  addBox(parent, [1.08, 2.35, 0.12], [-4.95, 1.18, 4.48], door);
  addBox(parent, [0.76, 0.04, 0.04], [-4.95, 1.52, 4.4], surface("#3b2b22", 0.42));
  parent.add(item(new THREE.SphereGeometry(0.055, 16, 12), surface("#b89a55", 0.24, 0.7), [-4.55, 1.12, 4.38]));
  addBox(parent, [1.45, 0.46, 0.46], [-5.55, 0.27, 2.75], surface("#8c6c53", 0.72));
  addBox(parent, [1.32, 0.11, 0.4], [-5.55, 0.56, 2.75], surface("#c5ae8c", 0.94));
  for (const x of [-0.42, 0.42]) {
    addBox(parent, [0.08, 1.1, 0.08], [-5.55 + x, 1.34, 2.46], surface("#2e3130", 0.3, 0.65));
  }
  addBox(parent, [1.18, 0.65, 0.08], [-5.55, 1.35, 2.45], surface("#b6a07d", 0.86));

  const studyRug = new THREE.MeshStandardMaterial({ map: fabricTexture("#7e6e77", "rgba(255,255,255,.06)"), roughness: 0.97 });
  addBox(parent, [2.9, 0.03, 2.35], [4.72, 0.03, 2.72], studyRug);
  addBox(parent, [2.18, 0.11, 0.78], [4.78, 0.78, 3.62], surface("#6e4c38", 0.5));
  for (const x of [3.83, 5.73]) {
    for (const z of [3.35, 3.88]) parent.add(item(new THREE.CylinderGeometry(0.03, 0.045, 0.73, 10), surface("#242827", 0.28, 0.72), [x, 0.38, z]));
  }
  addBox(parent, [0.98, 0.58, 0.08], [4.78, 1.22, 3.56], surface("#1c2222", 0.12, 0.4));
  addBox(parent, [0.65, 0.04, 0.2], [4.78, 0.91, 3.38], surface("#272b2b", 0.24, 0.65));
  const chair = makeDiningChair(surface("#4c675f", 0.9));
  chair.position.set(4.78, 0, 2.78);
  chair.rotation.y = Math.PI;
  parent.add(chair);
  addBox(parent, [1.76, 2.15, 0.32], [6.02, 1.08, 1.65], surface("#604837", 0.62));
  for (let shelf = 0; shelf < 4; shelf += 1) addBox(parent, [1.62, 0.06, 0.36], [6.02, 0.34 + shelf * 0.52, 1.45], surface("#302a26", 0.48));
  for (let book = 0; book < 12; book += 1) {
    const shelf = book % 3;
    const x = 5.38 + (book % 4) * 0.28;
    addBox(parent, [0.18, 0.3 + (book % 2) * 0.08, 0.19], [x, 0.55 + shelf * 0.52, 1.39], surface(["#a05d4c", "#506b69", "#c29a5d"][book % 3], 0.84));
  }
}

function addStairs(parent: THREE.Group): void {
  const tread = surface("#795039", 0.54);
  const stringer = surface("#e6e0d4", 0.78);
  const rail = surface("#353735", 0.34, 0.55);
  const steps = 15;
  for (let index = 0; index < steps; index += 1) {
    const height = (index + 1) * (STORY_HEIGHT / steps);
    const z = 3.86 - index * 0.235;
    addBox(parent, [1.58, height, 0.29], [-1.12, height / 2, z], stringer);
    addBox(parent, [1.68, 0.075, 0.34], [-1.12, height + 0.035, z], tread);
    if (index % 2 === 0) {
      parent.add(item(new THREE.CylinderGeometry(0.025, 0.025, 0.88, 9), rail, [-0.22, height + 0.44, z]));
    }
  }
  const handrail = addBox(parent, [0.07, 0.07, 3.56], [-0.22, 2.1, 2.22], rail, [-0.66, 0, 0]);
  handrail.rotation.x = -0.66;
}

function addUpperArchitecture(upper: THREE.Group): void {
  const oak = new THREE.MeshStandardMaterial({ map: oakTexture(), color: "#a4744e", roughness: 0.57 });
  const wall = surface("#e9e5dc", 0.94);
  const tile = new THREE.MeshStandardMaterial({ map: tileTexture("#d6d2ca", "#aeb0ac", 9), roughness: 0.72 });
  addBox(upper, [HOUSE_WIDTH, 0.16, 4.92], [0, STORY_HEIGHT, -2.14], oak);
  addBox(upper, [4.9, 0.16, 4.28], [-4.15, STORY_HEIGHT, 2.44], oak);
  addBox(upper, [6.2, 0.16, 4.28], [3.5, STORY_HEIGHT, 2.44], oak);
  addBox(upper, [3.2, 0.02, 3.1], [-4.55, STORY_HEIGHT + 0.1, 2.55], tile);
  addBackWall(upper, STORY_HEIGHT, [
    { x: -4.55, width: 2.0 },
    { x: -0.9, width: 1.8 },
    { x: 3.75, width: 2.55 },
  ]);
  addLeftWall(upper, STORY_HEIGHT);
  addTrim(upper, STORY_HEIGHT);
  addBox(upper, [0.14, 2.9, 4.72], [0.35, STORY_HEIGHT + 1.45, -2.18], wall);
  addBox(upper, [6.05, 2.9, 0.14], [-3.58, STORY_HEIGHT + 1.45, 0.66], wall);
  addBox(upper, [3.85, 2.9, 0.14], [4.68, STORY_HEIGHT + 1.45, 0.66], wall);
  addBox(upper, [1.2, 2.9, 0.14], [0.95, STORY_HEIGHT + 1.45, 0.66], wall);
  const railing = surface("#333735", 0.34, 0.6);
  for (let index = 0; index < 9; index += 1) {
    upper.add(item(new THREE.CylinderGeometry(0.022, 0.022, 0.88, 8), railing, [-0.2, STORY_HEIGHT + 0.5, 0.95 + index * 0.36]));
  }
  addBox(upper, [0.07, 0.07, 3.05], [-0.2, STORY_HEIGHT + 0.96, 2.38], railing);
}

function addUpperRooms(upper: THREE.Group): void {
  const master = makeBed(2.75, 2.2, "#d8d1c4");
  master.position.set(3.7, STORY_HEIGHT, -2.55);
  upper.add(master);
  for (const x of [2.03, 5.37]) {
    addBox(upper, [0.62, 0.48, 0.5], [x, STORY_HEIGHT + 0.24, -3.5], surface("#6c4b39", 0.56));
    upper.add(item(new THREE.CylinderGeometry(0.1, 0.18, 0.3, 24, 1, true), surface("#c5aa7f", 0.86), [x, STORY_HEIGHT + 0.84, -3.5]));
  }
  const masterRug = new THREE.MeshStandardMaterial({ map: fabricTexture("#7a8882", "rgba(255,255,255,.06)"), roughness: 0.96 });
  addBox(upper, [3.7, 0.025, 2.95], [3.7, STORY_HEIGHT + 0.03, -2.22], masterRug);
  addBox(upper, [1.8, 2.2, 0.58], [5.62, STORY_HEIGHT + 1.1, -0.2], surface("#786553", 0.7));
  addBox(upper, [1.6, 1.85, 0.05], [5.62, STORY_HEIGHT + 1.18, -0.52], new THREE.MeshPhysicalMaterial({ color: "#bacfd5", roughness: 0.08, metalness: 0.2, clearcoat: 1 }));

  const secondBed = makeBed(2.2, 2.05, "#a8bec3");
  secondBed.position.set(-2.25, STORY_HEIGHT, -2.7);
  upper.add(secondBed);
  addBox(upper, [1.72, 0.1, 0.65], [-5.18, STORY_HEIGHT + 0.77, -1.16], surface("#76513a", 0.54));
  for (const x of [-5.88, -4.48]) {
    for (const z of [-1.42, -0.94]) upper.add(item(new THREE.CylinderGeometry(0.025, 0.038, 0.72, 9), surface("#292c2b", 0.3, 0.7), [x, STORY_HEIGHT + 0.37, z]));
  }
  addBox(upper, [0.78, 0.52, 0.06], [-5.18, STORY_HEIGHT + 1.12, -1.21], surface("#1b2121", 0.12, 0.4));

  const porcelain = surface("#eeeae1", 0.34);
  const bathroomY = STORY_HEIGHT;
  addBox(upper, [1.95, 0.62, 0.6], [-5.22, bathroomY + 0.31, 3.9], surface("#8a806f", 0.62));
  addBox(upper, [1.98, 0.08, 0.64], [-5.22, bathroomY + 0.66, 3.9], porcelain);
  for (const x of [-5.68, -4.76]) {
    upper.add(item(new THREE.SphereGeometry(0.22, 24, 16), porcelain, [x, bathroomY + 0.72, 3.9], [Math.PI / 2, 0, 0]));
  }
  addBox(upper, [1.74, 0.85, 0.05], [-5.22, bathroomY + 1.52, 4.16], new THREE.MeshPhysicalMaterial({ color: "#b9d2d8", roughness: 0.04, metalness: 0.18, clearcoat: 1 }));
  addBox(upper, [2.15, 0.56, 0.86], [-3.72, bathroomY + 0.28, 1.24], porcelain);
  addBox(upper, [2.0, 0.07, 0.7], [-3.72, bathroomY + 0.58, 1.24], porcelain);
  const showerGlass = new THREE.MeshPhysicalMaterial({ color: "#cbe5e8", roughness: 0.05, transmission: 0.35, transparent: true, opacity: 0.48, clearcoat: 1 });
  addBox(upper, [1.5, 1.95, 0.05], [-5.78, bathroomY + 1.02, 1.62], showerGlass);
  addBox(upper, [0.05, 1.95, 1.35], [-5.03, bathroomY + 1.02, 0.96], showerGlass);
  upper.add(item(new THREE.CylinderGeometry(0.19, 0.24, 0.45, 28), porcelain, [-2.38, bathroomY + 0.31, 2.75]));
  addBox(upper, [0.52, 0.12, 0.65], [-2.38, bathroomY + 0.59, 2.89], porcelain);

  addBox(upper, [1.65, 0.45, 0.5], [1.5, STORY_HEIGHT + 0.24, 2.76], surface("#8b6d53", 0.72));
  addBox(upper, [1.45, 0.08, 0.44], [1.5, STORY_HEIGHT + 0.51, 2.76], surface("#c4aa87", 0.92));
  const plant = makePlant(0.82);
  plant.position.set(3.2, STORY_HEIGHT, 2.9);
  upper.add(plant);
  for (const x of [-3.5, 2.9]) {
    const pendant = makePendant("#8c785d");
    pendant.position.set(x, STORY_HEIGHT + 2.38, -2.0);
    upper.add(pendant);
  }
}

export function buildTwoStoryHouse(): TwoStoryHouseModel {
  const root = new THREE.Group();
  root.name = "two-story-visual-house";
  const ground = new THREE.Group();
  ground.name = "ground-floor";
  const upperFloor = new THREE.Group();
  upperFloor.name = "upper-floor";
  root.add(ground, upperFloor);

  const lawn = new THREE.MeshStandardMaterial({
    map: canvasTexture((context, size) => {
      context.fillStyle = "#526e4d";
      context.fillRect(0, 0, size, size);
      for (let index = 0; index < 900; index += 1) {
        const x = (index * 83) % size;
        const y = (index * 149) % size;
        context.fillStyle = index % 3 === 0 ? "rgba(203,220,155,.08)" : "rgba(26,55,31,.07)";
        context.fillRect(x, y, 2, 7);
      }
    }, [7, 5]),
    color: "#607a57",
    roughness: 0.98,
  });
  addBox(root, [18.4, 0.18, 14.2], [0, -0.5, 0], lawn);
  const concrete = surface("#a9aaa3", 0.91);
  addBox(root, [1.45, 0.12, 3.1], [-4.95, -0.35, 5.76], concrete);
  addBox(root, [2.0, 0.2, 0.95], [-4.95, -0.26, 4.76], concrete);
  for (const x of [-6.05, -3.85, 5.25, 6.05]) {
    const shrub = makePlant(0.62);
    shrub.position.set(x, -0.4, 5.15);
    root.add(shrub);
  }

  const oak = new THREE.MeshStandardMaterial({ map: oakTexture(), color: "#a4744e", roughness: 0.57, metalness: 0.01 });
  const foundation = surface("#71675f", 0.84);
  addBox(ground, [HOUSE_WIDTH + 0.36, 0.3, HOUSE_DEPTH + 0.36], [0, -0.18, 0], foundation);
  addBox(ground, [HOUSE_WIDTH, 0.16, HOUSE_DEPTH], [0, 0, 0], oak);
  addBackWall(ground, 0, [
    { x: -4.5, width: 2.0 },
    { x: -0.3, width: 2.0 },
    { x: 4.0, width: 2.8 },
  ]);
  addLeftWall(ground, 0);
  addTrim(ground, 0);

  const wall = surface("#e9e5dc", 0.94);
  addBox(ground, [0.14, 2.95, 2.65], [2.65, 1.48, 3.28], wall);
  addBox(ground, [3.85, 2.95, 0.14], [4.58, 1.48, 1.0], wall);
  addBox(ground, [1.1, 2.95, 0.14], [0.35, 1.48, 1.0], wall);
  addBox(ground, [2.1, 2.95, 0.14], [-4.95, 1.48, 1.0], wall);

  addKitchen(ground);
  addLivingRoom(ground);
  addDining(ground);
  addFoyerAndStudy(ground);
  addStairs(ground);
  addUpperArchitecture(upperFloor);
  addUpperRooms(upperFloor);

  const firstLight = new THREE.RectAreaLight("#ffd6a0", 5.5, 4.5, 3.2);
  firstLight.position.set(2.8, 2.72, -1.8);
  firstLight.rotation.x = -Math.PI / 2;
  root.add(firstLight);
  const kitchenLight = new THREE.RectAreaLight("#ffe4bd", 4.5, 3.6, 2.4);
  kitchenLight.position.set(-3.7, 2.64, -2.1);
  kitchenLight.rotation.x = -Math.PI / 2;
  root.add(kitchenLight);
  const upperLight = new THREE.RectAreaLight("#ffd8a8", 4.1, 4.8, 2.8);
  upperLight.position.set(2.8, STORY_HEIGHT + 2.55, -2.0);
  upperLight.rotation.x = -Math.PI / 2;
  upperFloor.add(upperLight);

  return { root, upperFloor };
}
