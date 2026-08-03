import * as THREE from "three";

export const HOUSE_WIDTH = 13.2;
export const HOUSE_DEPTH = 9.2;
export const STORY_HEIGHT = 3.12;

/**
 * Every wall, trim and piece of furniture in this file is positioned against the
 * *walking surface* of its storey: y = 0 downstairs, y = STORY_HEIGHT upstairs.
 * Floor slabs therefore hang below that line rather than straddling it.
 */
const FLOOR_THICKNESS = 0.16;
const GROUND_SOFFIT = -FLOOR_THICKNESS;
const UPPER_SOFFIT = STORY_HEIGHT - FLOOR_THICKNESS;
const WALL_HEIGHT = 2.9;
const DOOR_HEIGHT = 2.1;

/**
 * One straight flight of stairs, plus the hole it rises through. The stairwell
 * opening, the guard rails, the landing and the upper partitions are all derived
 * from these constants, so the flight cannot drift out of its own hole again.
 */
const STAIR_RISERS = 15;
const STAIR_RISE = STORY_HEIGHT / STAIR_RISERS;
const STAIR_GOING = 0.27;
const STAIR_WIDTH = 1.3;
const STAIR_CENTER_X = -1.2;
const STAIR_WEST_X = STAIR_CENTER_X - STAIR_WIDTH / 2;
const STAIR_EAST_X = STAIR_CENTER_X + STAIR_WIDTH / 2;
/** Nosing of the bottom tread, and the floor edge the top tread lands against. */
const STAIR_BOTTOM_Z = 4.1;
const STAIR_TOP_Z = STAIR_BOTTOM_Z - (STAIR_RISERS - 1) * STAIR_GOING;
const HANDRAIL_HEIGHT = 0.9;
const HANDRAIL_THICKNESS = 0.07;

/** Upper-floor partitions and the door openings punched through them. */
const WEST_PARTITION_X = -2.85;
const EAST_PARTITION_X = 0.35;
const CROSS_PARTITION_Z = 0.66;
const PARTITION_THICKNESS = 0.14;
const BEDROOM_DOOR_Z: [number, number] = [-1.25, -0.35];
/** Bathroom door, stopping short of the well so a pier can carry the guard rail. */
const BATHROOM_DOOR_X: [number, number] = [WEST_PARTITION_X + 0.08, -2.05];
/** Where the two guard rails stand: just clear of the well on the slab side. */
const WEST_RAIL_X = STAIR_WEST_X - 0.06;
const EAST_RAIL_X = STAIR_EAST_X + 0.045;

export type TwoStoryHouseModel = {
  root: THREE.Group;
  groundFloor: THREE.Group;
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

/**
 * A gap in a wall. `full` gaps run floor to ceiling (a stairwell, an open plan
 * threshold); the rest get a header and casing so they read as a door.
 */
type Opening = { from: number; to: number; full?: boolean };

function addOpeningTrim(
  parent: THREE.Group,
  along: "x" | "z",
  offset: number,
  baseY: number,
  opening: Opening,
  casing: THREE.Material,
): void {
  const depth = PARTITION_THICKNESS + 0.05;
  const mid = (opening.from + opening.to) / 2;
  const span = opening.to - opening.from;
  for (const edge of [opening.from, opening.to]) {
    const size: [number, number, number] = along === "z"
      ? [depth, DOOR_HEIGHT, 0.07]
      : [0.07, DOOR_HEIGHT, depth];
    const at: [number, number, number] = along === "z"
      ? [offset, baseY + DOOR_HEIGHT / 2, edge]
      : [edge, baseY + DOOR_HEIGHT / 2, offset];
    addBox(parent, size, at, casing);
  }
  const lintel: [number, number, number] = along === "z"
    ? [depth, 0.07, span + 0.14]
    : [span + 0.14, 0.07, depth];
  const lintelAt: [number, number, number] = along === "z"
    ? [offset, baseY + DOOR_HEIGHT, mid]
    : [mid, baseY + DOOR_HEIGHT, offset];
  addBox(parent, lintel, lintelAt, casing);
}

/**
 * Builds one partition as a run of panels between its openings, so a wall can
 * never float over a hole in the floor and every threshold gets a real head.
 */
function addPartition(
  parent: THREE.Group,
  along: "x" | "z",
  offset: number,
  span: [number, number],
  baseY: number,
  openings: Opening[],
  material: THREE.Material,
  casing?: THREE.Material,
): void {
  const panel = (from: number, to: number, height: number, bottom: number) => {
    if (to - from <= 0.001 || height <= 0.001) return;
    const size: [number, number, number] = along === "z"
      ? [PARTITION_THICKNESS, height, to - from]
      : [to - from, height, PARTITION_THICKNESS];
    const at: [number, number, number] = along === "z"
      ? [offset, bottom + height / 2, (from + to) / 2]
      : [(from + to) / 2, bottom + height / 2, offset];
    addBox(parent, size, at, material);
  };

  let cursor = span[0];
  for (const opening of [...openings].sort((a, b) => a.from - b.from)) {
    panel(cursor, opening.from, WALL_HEIGHT, baseY);
    if (!opening.full) {
      panel(opening.from, opening.to, WALL_HEIGHT - DOOR_HEIGHT, baseY + DOOR_HEIGHT);
      if (casing) addOpeningTrim(parent, along, offset, baseY, opening, casing);
    }
    cursor = opening.to;
  }
  panel(cursor, span[1], WALL_HEIGHT, baseY);
}

/** Guard rail along one edge of the stairwell, standing on the floor it protects. */
function addWellRailing(
  parent: THREE.Group,
  x: number,
  span: [number, number],
  baseY: number,
): void {
  const railing = surface("#333735", 0.34, 0.6);
  const [from, to] = span;
  const clear = HANDRAIL_HEIGHT - HANDRAIL_THICKNESS / 2;
  const count = Math.max(2, Math.round((to - from) / 0.17));
  for (let index = 0; index <= count; index += 1) {
    const z = from + ((to - from) * index) / count;
    parent.add(item(
      new THREE.CylinderGeometry(0.022, 0.022, clear, 8),
      railing,
      [x, baseY + clear / 2, z],
    ));
  }
  addBox(
    parent,
    [HANDRAIL_THICKNESS, HANDRAIL_THICKNESS, to - from],
    [x, baseY + HANDRAIL_HEIGHT - HANDRAIL_THICKNESS / 2, (from + to) / 2],
    railing,
  );
  for (const z of [from, to]) {
    parent.add(item(
      new THREE.CylinderGeometry(0.032, 0.032, HANDRAIL_HEIGHT, 10),
      railing,
      [x, baseY + HANDRAIL_HEIGHT / 2, z],
    ));
  }
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

function plasterTexture(base: string, fleck: string): THREE.CanvasTexture {
  return canvasTexture((context, size) => {
    context.fillStyle = base;
    context.fillRect(0, 0, size, size);
    for (let index = 0; index < 1200; index += 1) {
      const x = (index * 73) % size;
      const y = (index * 181) % size;
      const radius = 0.4 + ((index * 17) % 9) / 10;
      context.fillStyle = fleck;
      context.globalAlpha = 0.025 + (index % 5) * 0.008;
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
    }
    context.globalAlpha = 1;
  }, [2.2, 1.8]);
}

function makeCurtains(width: number, height: number, color: string): THREE.Group {
  const curtains = new THREE.Group();
  const textile = new THREE.MeshStandardMaterial({
    map: fabricTexture(color, "rgba(255,255,255,.045)"),
    color,
    roughness: 0.98,
    side: THREE.DoubleSide,
  });
  const rod = surface("#2b2d2c", 0.32, 0.7);
  curtains.add(item(new THREE.CylinderGeometry(0.022, 0.022, width + 0.42, 12), rod, [0, height / 2 + 0.12, 0], [0, 0, Math.PI / 2]));
  for (const side of [-1, 1]) {
    const panel = item(new THREE.BoxGeometry(width * 0.24, height, 0.045, 8, 16, 1), textile, [side * width * 0.42, 0, 0]);
    const positions = panel.geometry.attributes.position;
    for (let index = 0; index < positions.count; index += 1) {
      const x = positions.getX(index);
      const y = positions.getY(index);
      positions.setZ(index, positions.getZ(index) + Math.sin((x + y * 0.18) * 32) * 0.025);
    }
    positions.needsUpdate = true;
    panel.geometry.computeVertexNormals();
    curtains.add(panel);
  }
  return curtains;
}

function makeFramedPrint(
  width: number,
  height: number,
  colors: [string, string, string],
  seed: number,
): THREE.Group {
  const artwork = new THREE.Group();
  const frame = surface("#382d25", 0.48);
  addBox(artwork, [width + 0.1, height + 0.1, 0.045], [0, 0, 0], frame);
  const texture = canvasTexture((context, size) => {
    context.fillStyle = colors[0];
    context.fillRect(0, 0, size, size);
    context.fillStyle = colors[1];
    context.beginPath();
    context.arc(size * (0.35 + seed * 0.04), size * 0.42, size * 0.26, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = colors[2];
    context.save();
    context.translate(size * 0.62, size * 0.66);
    context.rotate(seed * 0.23);
    context.fillRect(-size * 0.22, -size * 0.12, size * 0.44, size * 0.24);
    context.restore();
  }, [1, 1]);
  const print = item(
    new THREE.PlaneGeometry(width, height),
    new THREE.MeshStandardMaterial({ map: texture, roughness: 0.86 }),
    [0, 0, 0.026],
  );
  artwork.add(print);
  return artwork;
}

function makeClock(): THREE.Group {
  const clock = new THREE.Group();
  const rim = surface("#302e2a", 0.34, 0.55);
  const face = surface("#eee9dc", 0.56);
  clock.add(item(new THREE.CylinderGeometry(0.3, 0.3, 0.055, 36), rim, [0, 0, 0], [Math.PI / 2, 0, 0]));
  clock.add(item(new THREE.CircleGeometry(0.255, 36), face, [0, 0, 0.031]));
  addBox(clock, [0.018, 0.17, 0.018], [0, 0.075, 0.047], rim, [0, 0, -0.42]);
  addBox(clock, [0.014, 0.12, 0.018], [0.045, -0.025, 0.049], rim, [0, 0, 1.05]);
  return clock;
}

function makeGardenTree(): THREE.Group {
  const tree = new THREE.Group();
  tree.name = "garden-tree";
  const bark = new THREE.MeshStandardMaterial({
    map: plasterTexture("#5d4533", "#1e1712"),
    color: "#75543d",
    roughness: 0.96,
  });
  tree.add(item(new THREE.CylinderGeometry(0.22, 0.33, 3.45, 18), bark, [0, 1.72, 0]));
  const leafColors = ["#315c3c", "#3f7047", "#557d4a"];
  for (let index = 0; index < 11; index += 1) {
    const crown = item(
      new THREE.IcosahedronGeometry(0.72 + (index % 3) * 0.09, 2),
      surface(leafColors[index % leafColors.length], 0.9),
      [Math.sin(index * 1.7) * 0.78, 3.1 + (index % 4) * 0.35, Math.cos(index * 1.7) * 0.62],
    );
    crown.scale.set(1.08, 0.92, 1);
    tree.add(crown);
  }
  return tree;
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
  sofa.name = "procedural-sofa";
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
  // Recessed plinth carrying the frame, so the bed rests on the floor.
  addBox(bed, [width - 0.16, 0.2, depth - 0.16], [0, 0.1, 0], surface("#4f382a", 0.7));
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
    pendant.position.set(x, UPPER_SOFFIT - 0.8, -1.84);
    parent.add(pendant);
  }
}

function addLivingRoom(parent: THREE.Group): void {
  const rug = new THREE.MeshStandardMaterial({
    map: fabricTexture("#687b77", "rgba(235,245,238,.08)"),
    color: "#718782",
    roughness: 0.97,
  });
  addBox(parent, [4.75, 0.035, 3.15], [3.55, 0.0175, -1.62], rug);
  const sofa = makeSofa();
  sofa.position.set(3.55, 0, -3.65);
  parent.add(sofa);
  const resident = makeResident();
  resident.position.set(3.55, 0.03, -3.42);
  parent.add(resident);
  const coffeeTable = new THREE.Group();
  coffeeTable.name = "procedural-coffee-table";
  addBox(coffeeTable, [1.55, 0.11, 0.84], [3.55, 0.5, -1.6], surface("#724b34", 0.5));
  for (const x of [2.96, 4.14]) {
    for (const z of [-1.9, -1.3]) {
      coffeeTable.add(item(new THREE.CylinderGeometry(0.028, 0.04, 0.45, 10), surface("#282b2a", 0.28, 0.7), [x, 0.24, z]));
    }
  }
  parent.add(coffeeTable);
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

/**
 * The dining room sits east of the stair hall, between the flight and the study
 * wall. Chairs face the table: makeDiningChair puts its back at local +z, so a
 * seat north of the table is turned by PI and a seat west of it by -PI/2.
 */
const DINING_X = 1.4;
const DINING_Z = 3.1;
const DINING_TOP = 0.85;

function addDining(parent: THREE.Group): void {
  const oak = surface("#76503a", 0.52);
  const diningTable = new THREE.Group();
  diningTable.name = "procedural-dining-table";
  addBox(diningTable, [2.0, 0.14, 1.2], [DINING_X, DINING_TOP - 0.07, DINING_Z], oak);
  for (const x of [DINING_X - 0.78, DINING_X + 0.78]) {
    for (const z of [DINING_Z - 0.42, DINING_Z + 0.42]) {
      diningTable.add(item(new THREE.CylinderGeometry(0.05, 0.07, DINING_TOP - 0.14, 12), surface("#393332", 0.38, 0.55), [x, (DINING_TOP - 0.14) / 2, z]));
    }
  }
  parent.add(diningTable);
  const upholstery = surface("#a7795e", 0.9);
  const placements: Array<[number, number, number]> = [
    [DINING_X - 0.55, DINING_Z - 0.88, Math.PI + 0.06],
    [DINING_X + 0.55, DINING_Z - 0.88, Math.PI - 0.04],
    [DINING_X - 0.55, DINING_Z + 0.88, -0.05],
    [DINING_X + 0.55, DINING_Z + 0.88, 0.07],
  ];
  for (const [x, z, rotation] of placements) {
    const chair = makeDiningChair(upholstery);
    chair.position.set(x, 0, z);
    chair.rotation.y = rotation;
    parent.add(chair);
  }
  const pendant = makePendant("#353332");
  pendant.scale.setScalar(1.25);
  pendant.position.set(DINING_X, UPPER_SOFFIT - 1.0, DINING_Z);
  parent.add(pendant);
  parent.add(item(new THREE.CylinderGeometry(0.16, 0.12, 0.13, 24), surface("#ede8dd", 0.42), [DINING_X, DINING_TOP + 0.065, DINING_Z - 0.01]));
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
  addBox(parent, [2.9, 0.03, 2.35], [4.72, 0.015, 2.72], studyRug);
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
  const bookshelf = new THREE.Group();
  bookshelf.name = "procedural-study-bookshelf";
  addBox(bookshelf, [1.76, 2.15, 0.32], [6.02, 1.08, 1.65], surface("#604837", 0.62));
  for (let shelf = 0; shelf < 4; shelf += 1) addBox(bookshelf, [1.62, 0.06, 0.36], [6.02, 0.34 + shelf * 0.52, 1.45], surface("#302a26", 0.48));
  for (let book = 0; book < 12; book += 1) {
    const shelf = book % 3;
    const x = 5.38 + (book % 4) * 0.28;
    addBox(bookshelf, [0.18, 0.3 + (book % 2) * 0.08, 0.19], [x, 0.55 + shelf * 0.52, 1.39], surface(["#a05d4c", "#506b69", "#c29a5d"][book % 3], 0.84));
  }
  parent.add(bookshelf);
}

function addBookStack(
  parent: THREE.Group,
  position: [number, number, number],
  rotation = 0,
  count = 3,
): void {
  const colors = ["#b95f4c", "#426b6b", "#d1a24f", "#6a5574"];
  for (let index = 0; index < count; index += 1) {
    addBox(
      parent,
      [0.38 - index * 0.025, 0.045, 0.26],
      [position[0] + index * 0.015, position[1] + index * 0.052, position[2]],
      surface(colors[index % colors.length], 0.86),
      [0, rotation + index * 0.09, 0],
    );
  }
}

function addLivedInClutter(ground: THREE.Group, upper: THREE.Group): void {
  const cardboard = surface("#ad7548", 0.92);
  const ceramic = surface("#e9e2d5", 0.38);
  const clothBlue = surface("#496f83", 0.96);
  const clothCoral = surface("#b96f5b", 0.96);

  // Entryway overflow, banked against the walls so the hall keeps a clear lane
  // from the front door past the console table.
  addBox(ground, [0.62, 0.42, 0.48], [-3.45, 0.21, 4.22], cardboard, [0, -0.14, 0]);
  addBox(ground, [0.44, 0.28, 0.36], [-3.3, 0.14, 3.55], cardboard, [0, 0.2, 0]);
  const tote = item(new THREE.SphereGeometry(0.29, 18, 12), clothCoral, [-6.25, 0.267, 4.05], [0, 0.35, 0]);
  tote.scale.set(0.72, 0.92, 0.42);
  ground.add(tote);
  addBookStack(ground, [-5.47, 0.6375, 2.74], 0.12, 3);

  // A used living room: blanket, mug, magazines, and a toy left in the route's periphery.
  addBox(ground, [0.78, 0.035, 0.54], [2.66, 0.83, -3.44], clothCoral, [0.18, 0.12, -0.18]);
  ground.add(item(new THREE.CylinderGeometry(0.09, 0.075, 0.13, 20), ceramic, [3.12, 0.63, -1.52]));
  addBookStack(ground, [3.88, 0.59, -1.61], -0.18, 2);
  const toy = new THREE.Group();
  addBox(toy, [0.28, 0.16, 0.18], [0, 0.14, 0], surface("#d6a542", 0.74));
  for (const x of [-0.11, 0.11]) {
    for (const z of [-0.07, 0.07]) toy.add(item(new THREE.CylinderGeometry(0.045, 0.045, 0.04, 12), surface("#303535", 0.56), [x, 0.07, z], [Math.PI / 2, 0, 0]));
  }
  toy.position.set(2.1, 0, -0.47);
  toy.rotation.y = -0.35;
  ground.add(toy);

  // Kitchen and dining surfaces are in use, not showroom-clean.
  addBox(ground, [0.52, 0.035, 0.32], [-2.62, 1.0, -1.82], surface("#8d5a3d", 0.66), [0, 0.1, 0]);
  for (let index = 0; index < 4; index += 1) {
    ground.add(item(new THREE.SphereGeometry(0.075 + index * 0.006, 16, 12), surface(["#d8894b", "#b74d3d", "#d3b34d"][index % 3], 0.78), [-2.8 + index * 0.13, 1.08, -1.82 + (index % 2) * 0.06]));
  }
  addBookStack(ground, [DINING_X - 0.45, DINING_TOP + 0.023, DINING_Z - 0.15], 0.24, 2);
  ground.add(item(new THREE.CylinderGeometry(0.13, 0.11, 0.12, 20), ceramic, [DINING_X + 0.4, DINING_TOP + 0.06, DINING_Z + 0.1]));

  // Upstairs has laundry, rumpled clothes, bedside reading, and scattered blocks.
  const laundry = item(new THREE.SphereGeometry(0.34, 18, 12), clothBlue, [2.45, STORY_HEIGHT + 0.177, 0.02]);
  laundry.scale.set(1.1, 0.52, 0.82);
  upper.add(laundry);
  for (let index = 0; index < 6; index += 1) {
    const color = ["#d5a53e", "#a65c4e", "#4f7770", "#d7d2c5"][index % 4];
    const block = addBox(upper, [0.16, 0.16, 0.16], [-3.5 + (index % 3) * 0.24, STORY_HEIGHT + 0.1, 2.05 + Math.floor(index / 3) * 0.22], surface(color, 0.8));
    block.rotation.y = index * 0.31;
  }
  addBookStack(upper, [2.04, STORY_HEIGHT + 0.503, -3.5], -0.1, 3);
  addBox(upper, [0.92, 0.035, 0.66], [-4.6, STORY_HEIGHT + 0.7375, -2.1], clothBlue, [-0.16, 0.08, 0.13]);
  addBox(upper, [0.68, 0.035, 0.42], [-4.52, STORY_HEIGHT + 0.6775, 3.82], clothCoral, [0, 0.06, 0]);
  for (let index = 0; index < 3; index += 1) {
    const height = 0.18 + index * 0.05;
    upper.add(item(new THREE.CylinderGeometry(0.035, 0.04, height, 14), surface(["#d8b45a", "#668c82", "#bd7160"][index], 0.58), [-5.84 + index * 0.16, STORY_HEIGHT + 0.66 + height / 2, 3.88]));
  }
}

function addArchitecturalCharacter(root: THREE.Group, ground: THREE.Group, upper: THREE.Group): void {
  const sagePlaster = new THREE.MeshStandardMaterial({
    map: plasterTexture("#6f7b75", "#26362f"),
    color: "#75827c",
    roughness: 0.96,
  });
  const bluePlaster = new THREE.MeshStandardMaterial({
    map: plasterTexture("#788a91", "#2b3b40"),
    color: "#85979d",
    roughness: 0.96,
  });
  const rosePlaster = new THREE.MeshStandardMaterial({
    map: plasterTexture("#8e706a", "#4b302b"),
    color: "#96766f",
    roughness: 0.96,
  });
  const tile = new THREE.MeshStandardMaterial({
    map: tileTexture("#c7d3ce", "#7f8f8b", 12),
    color: "#d4ddd8",
    roughness: 0.62,
  });
  const trim = surface("#eee9df", 0.74);

  // Room finishes run the full solid section of the wall they sit on: from the top
  // of the skirting to the head of the wall, and never across a door or the well.
  const finish = (
    parent: THREE.Group,
    span: [number, number],
    baseY: number,
    z: number,
    material: THREE.Material,
  ) => {
    const skirting = 0.11;
    addBox(
      parent,
      [span[1] - span[0], WALL_HEIGHT - skirting, 0.025],
      [(span[0] + span[1]) / 2, baseY + skirting + (WALL_HEIGHT - skirting) / 2, z],
      material,
    );
    addBox(parent, [span[1] - span[0], skirting, 0.06], [(span[0] + span[1]) / 2, baseY + skirting / 2, z - 0.018], trim);
  };
  finish(ground, [2.72, HOUSE_WIDTH / 2], 0, 1.081, sagePlaster);
  finish(upper, [-HOUSE_WIDTH / 2, BATHROOM_DOOR_X[0]], STORY_HEIGHT, 0.742, bluePlaster);
  finish(upper, [EAST_PARTITION_X - PARTITION_THICKNESS / 2, HOUSE_WIDTH / 2], STORY_HEIGHT, 0.742, rosePlaster);
  addBox(ground, [4.78, 0.77, 0.035], [-3.75, 1.42, -4.49], tile);
  addBox(ground, [4.82, 0.58, 0.03], [3.98, 0.33, -4.49], sagePlaster);

  // Front door casing.
  for (const x of [-5.56, -4.34]) addBox(ground, [0.09, 2.48, 0.13], [x, 1.25, 4.42], trim);
  addBox(ground, [1.31, 0.09, 0.13], [-4.95, 2.46, 4.42], trim);

  const livingCurtains = makeCurtains(2.8, 1.66, "#b98a6f");
  livingCurtains.position.set(4.0, 1.55, -4.46);
  ground.add(livingCurtains);
  const diningCurtains = makeCurtains(2.0, 1.58, "#566f72");
  diningCurtains.position.set(-0.3, 1.55, -4.46);
  ground.add(diningCurtains);
  const bedroomCurtains = makeCurtains(2.55, 1.62, "#c7b59c");
  bedroomCurtains.position.set(3.75, STORY_HEIGHT + 1.55, -4.46);
  upper.add(bedroomCurtains);

  const studyPrint = makeFramedPrint(0.68, 0.88, ["#ded6c7", "#d56f52", "#3d6664"], 1);
  studyPrint.position.set(4.03, 1.72, 1.105);
  ground.add(studyPrint);
  const studyPrintTwo = makeFramedPrint(0.54, 0.7, ["#d9d4c8", "#d8ab54", "#5b7482"], 2);
  studyPrintTwo.position.set(5.18, 1.58, 1.105);
  studyPrintTwo.rotation.z = -0.025;
  ground.add(studyPrintTwo);
  const bedroomPrint = makeFramedPrint(0.9, 0.62, ["#e2d6c8", "#365f60", "#c97958"], 3);
  bedroomPrint.position.set(4.7, STORY_HEIGHT + 1.7, 0.766);
  upper.add(bedroomPrint);
  const clock = makeClock();
  clock.position.set(5.86, 2.07, 1.112);
  ground.add(clock);

  // The garden is intentionally imperfect: uneven pavers, an established tree, and mixed planters.
  const tree = makeGardenTree();
  // Stood in the front yard clear of the gable: beside the house its crown reached
  // through the left wall and the upper floor slab.
  tree.position.set(-7.6, -0.41, 6.2);
  root.add(tree);
  const paver = surface("#b0aca1", 0.94);
  for (let index = 0; index < 7; index += 1) {
    addBox(
      root,
      [0.78 + (index % 2) * 0.12, 0.06, 0.52],
      [-4.95 + Math.sin(index * 1.8) * 0.11, -0.37, 6.15 + index * 0.72],
      paver,
      [0, (index % 3 - 1) * 0.065, 0],
    );
  }
  for (const [x, z, color] of [
    [-6.45, 4.95, "#b76252"],
    [5.72, 5.2, "#d2a348"],
    [6.38, 4.88, "#8f6683"],
  ] as Array<[number, number, string]>) {
    root.add(item(new THREE.CylinderGeometry(0.19, 0.16, 0.25, 18), surface("#7b5540", 0.86), [x, -0.27, z]));
    for (let petal = 0; petal < 5; petal += 1) {
      root.add(item(new THREE.SphereGeometry(0.07, 12, 8), surface(color, 0.82), [x + Math.sin(petal * 1.26) * 0.1, -0.02 + (petal % 2) * 0.05, z + Math.cos(petal * 1.26) * 0.1]));
    }
  }

  const doormat = new THREE.MeshStandardMaterial({
    map: fabricTexture("#6d5440", "rgba(238,218,183,.11)"),
    color: "#795f49",
    roughness: 1,
  });
  addBox(root, [1.2, 0.035, 0.62], [-4.95, -0.1425, 4.96], doormat, [0, 0.03, 0]);
}

/**
 * One straight flight rising toward -z. There are STAIR_RISERS risers but only
 * STAIR_RISERS - 1 treads: the upper floor itself is the last walking surface,
 * which is what stops the top tread from intersecting the slab.
 *
 * The handrail is generated from the flight's own pitch and the balusters are cut
 * to reach it, so the rail can never slope against the steps or drift off the
 * treads the way a hand-tuned rotation did.
 */
function addStairs(parent: THREE.Group): void {
  const tread = surface("#795039", 0.54);
  const stringer = surface("#e6e0d4", 0.78);
  const rail = surface("#353735", 0.34, 0.55);
  const treadThickness = 0.05;
  const nosing = 0.03;
  const railX = STAIR_EAST_X - 0.05;
  const pitch = Math.atan2(STAIR_RISE, STAIR_GOING);
  // Height of the rail's centre line directly above a tread's walking surface.
  const railCentreAt = (z: number) =>
    STAIR_RISE + HANDRAIL_HEIGHT + (STAIR_BOTTOM_Z - z) * (STAIR_RISE / STAIR_GOING);

  for (let index = 0; index < STAIR_RISERS - 1; index += 1) {
    const walking = (index + 1) * STAIR_RISE;
    const front = STAIR_BOTTOM_Z - index * STAIR_GOING;
    const back = front - STAIR_GOING;
    // Solid carriage below, tread board on top: together they fill 0 .. walking.
    addBox(
      parent,
      [STAIR_WIDTH, walking - treadThickness, STAIR_GOING],
      [STAIR_CENTER_X, (walking - treadThickness) / 2, (front + back) / 2],
      stringer,
    );
    addBox(
      parent,
      [STAIR_WIDTH, treadThickness, STAIR_GOING + nosing],
      [STAIR_CENTER_X, walking - treadThickness / 2, (front + nosing + back) / 2],
      tread,
    );

    if (index % 2 === 0) {
      const z = (front + back) / 2;
      const clear = railCentreAt(z) - HANDRAIL_THICKNESS / 2 - walking;
      parent.add(item(
        new THREE.CylinderGeometry(0.022, 0.022, clear, 9),
        rail,
        [railX, walking + clear / 2, z],
      ));
    }
  }

  // Riser board closing the gap between the top tread and the landing soffit.
  const topTread = (STAIR_RISERS - 1) * STAIR_RISE;
  addBox(
    parent,
    [STAIR_WIDTH, UPPER_SOFFIT - topTread, 0.06],
    [STAIR_CENTER_X, (UPPER_SOFFIT + topTread) / 2, STAIR_TOP_Z - 0.03],
    stringer,
  );

  const run = (STAIR_RISERS - 1) * STAIR_GOING;
  const rise = (STAIR_RISERS - 1) * STAIR_RISE;
  const handrail = addBox(
    parent,
    [HANDRAIL_THICKNESS, HANDRAIL_THICKNESS, Math.hypot(run, rise)],
    [railX, (railCentreAt(STAIR_BOTTOM_Z) + railCentreAt(STAIR_TOP_Z)) / 2, (STAIR_BOTTOM_Z + STAIR_TOP_Z) / 2],
    rail,
    [pitch, 0, 0],
  );
  handrail.name = "stair-handrail";
  // Newel posts anchoring each end of the rail to the flight.
  parent.add(item(
    new THREE.CylinderGeometry(0.038, 0.038, railCentreAt(STAIR_BOTTOM_Z), 12),
    rail,
    [railX, railCentreAt(STAIR_BOTTOM_Z) / 2, STAIR_BOTTOM_Z - 0.06],
  ));
  parent.add(item(
    new THREE.CylinderGeometry(0.038, 0.038, railCentreAt(STAIR_TOP_Z) - STORY_HEIGHT + 0.4, 12),
    rail,
    [railX, STORY_HEIGHT - 0.4 + (railCentreAt(STAIR_TOP_Z) - STORY_HEIGHT + 0.4) / 2, STAIR_TOP_Z + 0.08],
  ));
}

function addUpperArchitecture(upper: THREE.Group): void {
  const oak = new THREE.MeshStandardMaterial({ map: oakTexture(), color: "#a4744e", roughness: 0.57 });
  const wall = surface("#e9e5dc", 0.94);
  const casing = surface("#f4f0e7", 0.7);
  const tile = new THREE.MeshStandardMaterial({ map: tileTexture("#d6d2ca", "#aeb0ac", 9), roughness: 0.72 });
  const west = -HOUSE_WIDTH / 2;
  const east = HOUSE_WIDTH / 2;
  const back = -HOUSE_DEPTH / 2;
  const front = HOUSE_DEPTH / 2;

  // Three slabs: the bedroom half, then two front slabs flanking the stairwell.
  const slab = (span: [number, number], depth: [number, number]) =>
    addBox(
      upper,
      [span[1] - span[0], FLOOR_THICKNESS, depth[1] - depth[0]],
      [(span[0] + span[1]) / 2, UPPER_SOFFIT + FLOOR_THICKNESS / 2, (depth[0] + depth[1]) / 2],
      oak,
    );
  slab([west, east], [back, STAIR_TOP_Z]);
  slab([west, STAIR_WEST_X], [STAIR_TOP_Z, front]);
  slab([STAIR_EAST_X, east], [STAIR_TOP_Z, front]);
  addBox(upper, [3.2, 0.02, 3.1], [-4.55, STORY_HEIGHT + 0.01, 2.55], tile);

  addBackWall(upper, STORY_HEIGHT, [
    { x: -4.55, width: 2.0 },
    { x: -0.9, width: 1.8 },
    { x: 3.75, width: 2.55 },
  ]);
  addLeftWall(upper, STORY_HEIGHT);
  addTrim(upper, STORY_HEIGHT);

  // Bedroom 2 and the master each open onto the landing through a cased door.
  addPartition(upper, "z", WEST_PARTITION_X, [back, CROSS_PARTITION_Z], STORY_HEIGHT,
    [{ from: BEDROOM_DOOR_Z[0], to: BEDROOM_DOOR_Z[1] }], wall, casing);
  addPartition(upper, "z", EAST_PARTITION_X, [back, CROSS_PARTITION_Z], STORY_HEIGHT,
    [{ from: BEDROOM_DOOR_Z[0], to: BEDROOM_DOOR_Z[1] }], wall, casing);
  // The cross wall carries the bathroom door, then opens full height across the
  // stairwell and the landing beside it, so no guard rail has to cross a doorway.
  addPartition(upper, "x", CROSS_PARTITION_Z, [west, east], STORY_HEIGHT, [
    { from: BATHROOM_DOOR_X[0], to: BATHROOM_DOOR_X[1] },
    { from: STAIR_WEST_X, to: EAST_PARTITION_X - PARTITION_THICKNESS / 2, full: true },
  ], wall, casing);

  // Guard rails stand on the slabs either side of the well, never inside it. The
  // west rail starts at the pier beside the bathroom door; the east rail runs the
  // full length of the landing from the head of the flight.
  const railFront = front - 0.05;
  addWellRailing(upper, WEST_RAIL_X, [CROSS_PARTITION_Z + PARTITION_THICKNESS / 2, railFront], STORY_HEIGHT);
  addWellRailing(upper, EAST_RAIL_X, [STAIR_TOP_Z, railFront], STORY_HEIGHT);
}

function addUpperRooms(upper: THREE.Group): void {
  const master = makeBed(2.75, 2.2, "#d8d1c4");
  master.position.set(3.7, STORY_HEIGHT, -2.55);
  upper.add(master);
  for (const x of [2.03, 5.37]) {
    addBox(upper, [0.62, 0.48, 0.5], [x, STORY_HEIGHT + 0.24, -3.5], surface("#6c4b39", 0.56));
    upper.add(item(new THREE.CylinderGeometry(0.1, 0.18, 0.3, 24, 1, true), surface("#c5aa7f", 0.86), [x, STORY_HEIGHT + 0.63, -3.5]));
  }
  const masterRug = new THREE.MeshStandardMaterial({ map: fabricTexture("#7a8882", "rgba(255,255,255,.06)"), roughness: 0.96 });
  addBox(upper, [3.7, 0.025, 2.95], [3.7, STORY_HEIGHT + 0.0125, -2.22], masterRug);
  const wardrobe = new THREE.Group();
  wardrobe.name = "procedural-master-wardrobe";
  addBox(wardrobe, [1.8, 2.2, 0.58], [5.62, STORY_HEIGHT + 1.1, -0.2], surface("#786553", 0.7));
  addBox(wardrobe, [1.6, 1.85, 0.05], [5.62, STORY_HEIGHT + 1.18, -0.52], new THREE.MeshPhysicalMaterial({ color: "#bacfd5", roughness: 0.08, metalness: 0.2, clearcoat: 1 }));
  upper.add(wardrobe);

  const secondBed = makeBed(2.2, 2.05, "#a8bec3");
  secondBed.name = "procedural-second-bed";
  secondBed.position.set(-4.5, STORY_HEIGHT, -2.7);
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
  // Tub kept west of the doorway so the threshold stays walkable.
  addBox(upper, [1.9, 0.56, 0.86], [-4.05, bathroomY + 0.28, 1.24], porcelain);
  addBox(upper, [1.75, 0.07, 0.7], [-4.05, bathroomY + 0.58, 1.24], porcelain);
  const showerGlass = new THREE.MeshPhysicalMaterial({ color: "#cbe5e8", roughness: 0.05, transmission: 0.35, transparent: true, opacity: 0.48, clearcoat: 1 });
  addBox(upper, [1.5, 1.95, 0.05], [-5.78, bathroomY + 0.985, 1.62], showerGlass);
  addBox(upper, [0.05, 1.95, 1.35], [-5.03, bathroomY + 0.985, 0.96], showerGlass);
  upper.add(item(new THREE.CylinderGeometry(0.19, 0.24, 0.45, 28), porcelain, [-2.38, bathroomY + 0.225, 2.75]));
  addBox(upper, [0.52, 0.12, 0.65], [-2.38, bathroomY + 0.51, 2.89], porcelain);

  addBox(upper, [1.65, 0.45, 0.5], [1.5, STORY_HEIGHT + 0.225, 2.76], surface("#8b6d53", 0.72));
  addBox(upper, [1.45, 0.08, 0.44], [1.5, STORY_HEIGHT + 0.49, 2.76], surface("#c4aa87", 0.92));
  const plant = makePlant(0.82);
  plant.position.set(3.2, STORY_HEIGHT, 2.9);
  upper.add(plant);
  for (const x of [-4.0, 2.9]) {
    const pendant = makePendant("#8c785d");
    pendant.position.set(x, STORY_HEIGHT + WALL_HEIGHT - 0.8, -2.0);
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
  addBox(ground, [HOUSE_WIDTH + 0.36, 0.3, HOUSE_DEPTH + 0.36], [0, GROUND_SOFFIT - 0.15, 0], foundation);
  addBox(ground, [HOUSE_WIDTH, FLOOR_THICKNESS, HOUSE_DEPTH], [0, GROUND_SOFFIT / 2, 0], oak);
  addBackWall(ground, 0, [
    { x: -4.5, width: 2.0 },
    { x: -0.3, width: 2.0 },
    { x: 4.0, width: 2.8 },
  ]);
  addLeftWall(ground, 0);
  addTrim(ground, 0);

  const wall = surface("#e9e5dc", 0.94);
  const casing = surface("#f4f0e7", 0.7);
  // Dining room / study divider, with the study's doorway at its open end.
  const spineFace = 1.0 + PARTITION_THICKNESS / 2;
  addPartition(ground, "z", 2.65, [spineFace, HOUSE_DEPTH / 2], 0,
    [{ from: spineFace, to: 1.96 }], wall, casing);
  // The spine wall. The flight passes through it full height, so the openings
  // either side have to be wide enough to walk round the stair: the foyer reaches
  // the kitchen to the west, and the kitchen reaches the dining room to the east.
  addPartition(ground, "x", 1.0, [-HOUSE_WIDTH / 2, HOUSE_WIDTH / 2], 0, [
    { from: -3.9, to: STAIR_WEST_X - 0.1 },
    { from: STAIR_WEST_X - 0.1, to: STAIR_EAST_X + 0.1, full: true },
    { from: STAIR_EAST_X + 0.1, to: 2.58 },
  ], wall, casing);

  addKitchen(ground);
  addLivingRoom(ground);
  addDining(ground);
  addFoyerAndStudy(ground);
  addStairs(ground);
  addUpperArchitecture(upperFloor);
  addUpperRooms(upperFloor);
  addLivedInClutter(ground, upperFloor);
  addArchitecturalCharacter(root, ground, upperFloor);

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

  return { root, groundFloor: ground, upperFloor };
}
