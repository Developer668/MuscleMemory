// Geometry audit for the two-story landing house. Run from frontend/ with node.
import type { Object3D } from "three";

const noopContext = new Proxy(
  {},
  {
    get(_target, property) {
      if (property === "canvas") return { width: 512, height: 512 };
      if (property === "globalAlpha" || property === "fillStyle" || property === "strokeStyle") return "";
      return () => undefined;
    },
    set() {
      return true;
    },
  },
);

(globalThis as Record<string, unknown>).document = {
  createElement: () => ({
    width: 512,
    height: 512,
    getContext: () => noopContext,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    style: {},
  }),
};
(globalThis as Record<string, unknown>).window = { devicePixelRatio: 1 };
(globalThis as Record<string, unknown>).HTMLCanvasElement = class {};
(globalThis as Record<string, unknown>).ImageData = class {};

const THREE = await import("three");
const { buildTwoStoryHouse, HOUSE_WIDTH, HOUSE_DEPTH, STORY_HEIGHT } = await import(
  "./src/components/TwoStoryHouse.ts"
);

type Entry = {
  path: string;
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
};

const house = buildTwoStoryHouse();
house.root.updateMatrixWorld(true);

const entries: Entry[] = [];
const pathOf = (object: Object3D): string => {
  const parts: string[] = [];
  let node: Object3D | null = object;
  while (node && node !== house.root) {
    parts.unshift(node.name || node.type);
    node = node.parent;
  }
  return parts.join("/");
};

house.root.traverse((object: Object3D) => {
  if (!(object instanceof THREE.Mesh)) return;
  const box = new THREE.Box3().setFromObject(object);
  entries.push({
    path: pathOf(object),
    min: { x: box.min.x, y: box.min.y, z: box.min.z },
    max: { x: box.max.x, y: box.max.y, z: box.max.z },
  });
});

const r = (value: number) => Number(value.toFixed(3));
const fmt = (e: Entry) =>
  `x[${r(e.min.x)},${r(e.max.x)}] y[${r(e.min.y)},${r(e.max.y)}] z[${r(e.min.z)},${r(e.max.z)}]`;

console.log(`meshes=${entries.length} HOUSE_WIDTH=${HOUSE_WIDTH} HOUSE_DEPTH=${HOUSE_DEPTH} STORY=${STORY_HEIGHT}`);

const GROUND_TOP = 0.08;
const UPPER_TOP = STORY_HEIGHT + 0.08;

console.log(`\n== floor surfaces: groundTop=${GROUND_TOP} upperTop=${r(UPPER_TOP)}`);

// --- 1. Objects resting neither on a floor nor on another object -------------
console.log("\n== A. bottom-of-object vs nearest floor plane (gap>0.02 or sink>0.02)");
const floors = [GROUND_TOP, UPPER_TOP];
const suspects: Array<[string, number, number]> = [];
for (const entry of entries) {
  const bottom = entry.min.y;
  let best = Infinity;
  let plane = 0;
  for (const f of floors) {
    if (Math.abs(bottom - f) < Math.abs(best)) {
      best = bottom - f;
      plane = f;
    }
  }
  if (Math.abs(best) > 0.02 && Math.abs(best) < 0.9) suspects.push([entry.path, plane, best]);
}
const grouped = new Map<string, Array<[string, number, number]>>();
for (const s of suspects) {
  const key = s[0].split("/")[0] + "|" + r(s[1]) + "|" + r(s[2]);
  if (!grouped.has(key)) grouped.set(key, []);
  grouped.get(key)!.push(s);
}
for (const [key, list] of grouped) {
  const [top, plane, delta] = key.split("|");
  console.log(`  ${top} plane=${plane} delta=${delta} (${list.length} mesh) e.g. ${list[0][0]}`);
}

// --- 2. Stair envelope vs upper-floor slab opening ---------------------------
console.log("\n== B. stair envelope vs upper floor slabs");
const stairMeshes = entries.filter((e) => e.min.y < 0.2 && e.max.y > 1.0 && e.path.startsWith("ground-floor"));
const stairEnv = { minX: Infinity, maxX: -Infinity, minZ: Infinity, maxZ: -Infinity, maxY: -Infinity };
for (const e of entries) {
  // stringers/treads live around x in [-2,-0.2]
  if (e.min.x > -2.2 && e.max.x < -0.1 && e.min.y < 3.3 && e.max.y > 0.1 && e.min.z > 0.2) {
    stairEnv.minX = Math.min(stairEnv.minX, e.min.x);
    stairEnv.maxX = Math.max(stairEnv.maxX, e.max.x);
    stairEnv.minZ = Math.min(stairEnv.minZ, e.min.z);
    stairEnv.maxZ = Math.max(stairEnv.maxZ, e.max.z);
    stairEnv.maxY = Math.max(stairEnv.maxY, e.max.y);
  }
}
console.log(`  stair-ish envelope: x[${r(stairEnv.minX)},${r(stairEnv.maxX)}] z[${r(stairEnv.minZ)},${r(stairEnv.maxZ)}] maxY=${r(stairEnv.maxY)}`);
console.log(`  tall meshes rising from ground (${stairMeshes.length}):`);
for (const e of stairMeshes.slice(0, 6)) console.log(`    ${e.path} ${fmt(e)}`);

const slabs = entries.filter(
  (e) => e.path.startsWith("upper-floor") && e.max.y - e.min.y < 0.2 && e.min.y > 2.9 && e.min.y < 3.15 && e.max.x - e.min.x > 3,
);
console.log("  upper slabs:");
for (const e of slabs) console.log(`    ${fmt(e)}`);

// --- 3. Explicit interpenetration of stairs into slabs -----------------------
console.log("\n== C. stair meshes intersecting upper slabs");
let hits = 0;
for (const e of entries) {
  if (!(e.min.x > -2.2 && e.max.x < -0.1 && e.min.z > 0.2)) continue;
  for (const s of slabs) {
    const ox = Math.min(e.max.x, s.max.x) - Math.max(e.min.x, s.min.x);
    const oy = Math.min(e.max.y, s.max.y) - Math.max(e.min.y, s.min.y);
    const oz = Math.min(e.max.z, s.max.z) - Math.max(e.min.z, s.min.z);
    if (ox > 0.001 && oy > 0.001 && oz > 0.001) {
      hits += 1;
      if (hits <= 8) console.log(`  ${e.path} ${fmt(e)} overlap dx=${r(ox)} dy=${r(oy)} dz=${r(oz)}`);
    }
  }
}
console.log(`  total stair/slab intersections: ${hits}`);

// --- 4. Route walkability ----------------------------------------------------
const ROUTES: Record<string, Array<[number, number]>> = {
  medicine: [[-4.9, 3.82], [-3.75, 2.72], [-2.15, 1.48], [-0.45, 0.52], [1.42, -0.18], [2.48, -1.42], [3.42, -2.68]],
  breakfast: [[-4.72, -0.48], [-3.78, 0.18], [-2.55, 0.5], [-1.42, 0.76], [-0.24, 0.98], [0.68, 1.18]],
  kitchen_check: [[4.62, 0.88], [3.18, 0.58], [1.72, 0.18], [0.18, -0.58], [-0.72, -1.26], [-1.08, -1.98], [-1.45, -2.6]],
  parcel: [[4.76, 0.94], [3.32, 0.7], [1.78, 0.52], [0.22, 0.48], [-1.34, 0.8], [-2.35, 1.12], [-3.05, 1.36], [-3.55, 1.55]],
};

const ROBOT_RADIUS = 0.3;
const ROBOT_HEIGHT = 1.4;

// Obstacles: anything on the ground floor or root that occupies the walking band.
const obstacles = entries.filter((e) => {
  if (e.path.startsWith("upper-floor")) return false;
  if (e.max.y <= GROUND_TOP + 0.05) return false; // rugs / floor slab
  if (e.min.y > ROBOT_HEIGHT) return false; // overhead
  return true;
});

console.log(`\n== D. route clearance (robot r=${ROBOT_RADIUS}, obstacles=${obstacles.length})`);
for (const [name, points] of Object.entries(ROUTES)) {
  const curve = new THREE.CatmullRomCurve3(points.map(([x, z]) => new THREE.Vector3(x, GROUND_TOP, z)));
  const samples = curve.getPoints(240);
  const blocking = new Map<string, { count: number; worst: number; box: Entry }>();
  for (const p of samples) {
    for (const o of obstacles) {
      const dx = Math.max(o.min.x - p.x, 0, p.x - o.max.x);
      const dz = Math.max(o.min.z - p.z, 0, p.z - o.max.z);
      const distance = Math.hypot(dx, dz);
      if (distance < ROBOT_RADIUS) {
        const key = o.path;
        const previous = blocking.get(key);
        const penetration = ROBOT_RADIUS - distance;
        if (!previous || penetration > previous.worst) blocking.set(key, { count: (previous?.count ?? 0) + 1, worst: penetration, box: o });
        else previous.count += 1;
      }
    }
  }
  console.log(`  ${name}: ${blocking.size} obstacles within ${ROBOT_RADIUS}m of centreline`);
  const sorted = [...blocking.entries()].sort((a, b) => b[1].worst - a[1].worst);
  for (const [key, info] of sorted.slice(0, 10)) {
    console.log(`    pen=${r(info.worst)} n=${info.count} h=${r(info.box.max.y - GROUND_TOP)} ${key} ${fmt(info.box)}`);
  }
}

// --- 5. Handrail vs stair nosing line ---------------------------------------
console.log("\n== E. handrail / baluster placement");
for (const e of entries) {
  if (e.path.includes("ground-floor") && e.min.x > -0.35 && e.max.x < -0.05 && e.max.y > 1.0) {
    console.log(`  ${e.path} ${fmt(e)}`);
  }
}
