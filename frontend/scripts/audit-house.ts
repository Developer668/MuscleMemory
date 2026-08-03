/**
 * Structural audit of the landing house: run with `npm run audit:house`.
 *
 * These are the invariants the geometry kept breaking silently — floor datums,
 * the stair fitting its own well, rails standing on something, every room having a
 * way in, and the robot's routes not walking through walls. Nothing here renders,
 * so it runs in plain node with a stub canvas.
 */
const noopContext = new Proxy({}, {
  get(_t, property) {
    if (property === "canvas") return { width: 512, height: 512 };
    if (property === "globalAlpha" || property === "fillStyle" || property === "strokeStyle") return "";
    return () => undefined;
  },
  set() { return true; },
});
(globalThis as Record<string, unknown>).document = {
  createElement: () => ({ width: 512, height: 512, getContext: () => noopContext, style: {} }),
};

const THREE = await import("three");
const { buildTwoStoryHouse, HOUSE_DEPTH, STORY_HEIGHT } = await import("../src/components/TwoStoryHouse.ts");

type Entry = { path: string; min: THREE.Vector3; max: THREE.Vector3 };

const house = buildTwoStoryHouse();
house.root.updateMatrixWorld(true);

const entries: Entry[] = [];
const pathOf = (object: THREE.Object3D): string => {
  const parts: string[] = [];
  let node: THREE.Object3D | null = object;
  while (node && node !== house.root) { parts.unshift(node.name || node.type); node = node.parent; }
  return parts.join("/");
};
house.root.traverse((object) => {
  if (!(object instanceof THREE.Mesh)) return;
  const box = new THREE.Box3().setFromObject(object);
  entries.push({ path: pathOf(object), min: box.min.clone(), max: box.max.clone() });
});

const r = (v: number) => Number(v.toFixed(3));
const fmt = (e: Entry) => `x[${r(e.min.x)},${r(e.max.x)}] y[${r(e.min.y)},${r(e.max.y)}] z[${r(e.min.z)},${r(e.max.z)}]`;
const overlap = (a: Entry, b: Entry) => {
  const dx = Math.min(a.max.x, b.max.x) - Math.max(a.min.x, b.min.x);
  const dy = Math.min(a.max.y, b.max.y) - Math.max(a.min.y, b.min.y);
  const dz = Math.min(a.max.z, b.max.z) - Math.max(a.min.z, b.min.z);
  return Math.min(dx, dy, dz) > 0.002 ? { dx, dy, dz } : null;
};

let failures = 0;
const check = (name: string, ok: boolean, detail = "") => {
  if (!ok) failures += 1;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
};

console.log(`meshes=${entries.length}\n`);

// --- 1. Floor slab top surfaces land exactly on the storey datum -------------
const slabs = entries.filter((e) =>
  Math.abs(e.max.y - e.min.y - 0.16) < 1e-6 && (e.max.x - e.min.x) > 3 && (e.max.z - e.min.z) > 3);
console.log("floor slabs:");
for (const s of slabs) console.log(`  ${fmt(s)}`);
check("ground slab top == 0", slabs.some((s) => Math.abs(s.max.y) < 1e-6));
check("upper slab tops == STORY_HEIGHT",
  slabs.filter((s) => s.max.y > 1).every((s) => Math.abs(s.max.y - STORY_HEIGHT) < 1e-6));

// --- 2. Nothing pierces a floor slab ----------------------------------------
const slabHits: string[] = [];
const insideEnvelope = (e: Entry) =>
  e.min.x > -6.45 && e.max.x < 6.45 && e.min.z > -4.45 && e.max.z < 4.45;
for (const e of entries) {
  if (slabs.includes(e)) continue;
  if (e.path.endsWith("stair-handrail")) continue;
  if (!insideEnvelope(e)) continue;
  for (const s of slabs) {
    const o = overlap(e, s);
    if (o && e.min.y < s.max.y - 0.005 && e.max.y > s.min.y + 0.005) {
      slabHits.push(`${e.path} ${fmt(e)} dy=${r(o.dy)}`);
    }
  }
}
check("no mesh punches through a floor slab", slabHits.length === 0, slabHits.slice(0, 6).join(" | "));

// --- 3. Stair treads and handrail agree -------------------------------------
const stairTreads = entries.filter((e) =>
  e.min.x > -2.0 && e.max.x < -0.4 && e.min.z > 0.2 &&
  Math.abs(e.max.y - e.min.y - 0.05) < 1e-6 && Math.abs(e.max.z - e.min.z - 0.30) < 1e-6);
stairTreads.sort((a, b) => b.min.z - a.min.z);
console.log(`\nstair treads: ${stairTreads.length}`);
if (stairTreads.length) {
  console.log(`  lowest  ${fmt(stairTreads[0])}`);
  console.log(`  highest ${fmt(stairTreads[stairTreads.length - 1])}`);
}
check("treads rise as z decreases",
  stairTreads.length > 5 && stairTreads.every((t, i) => i === 0 || t.max.y > stairTreads[i - 1].max.y));
check("top tread does not exceed the upper floor",
  stairTreads.length > 0 && stairTreads[stairTreads.length - 1].max.y < STORY_HEIGHT + 1e-9);

const handrail = entries
  .filter((e) => e.min.x > -1.2 && e.max.x < -0.3 && (e.max.z - e.min.z) > 2 && (e.max.y - e.min.y) > 2)
  .sort((a, b) => (b.max.z - b.min.z) - (a.max.z - a.min.z))[0];
if (handrail) {
  console.log(`  handrail ${fmt(handrail)}`);
  const topTread = stairTreads[stairTreads.length - 1].max.y;
  check("handrail tops out at the head of the flight",
    handrail.max.y > topTread && handrail.max.y < topTread + 1.4,
    `railTop=${r(handrail.max.y)} topTread=${r(topTread)}`);
} else {
  check("handrail found", false);
}

// --- 4. Balusters stand on a tread ------------------------------------------
const balusters = entries.filter((e) =>
  (e.max.x - e.min.x) < 0.06 && (e.max.y - e.min.y) > 0.5 && (e.max.y - e.min.y) < 1.3 &&
  e.min.x > -1.0 && e.max.x < -0.3 && e.min.z > 0.2 && e.min.y > 0.1 && e.min.y < 3.1);
console.log(`\nstair balusters: ${balusters.length}`);
const unsupported = balusters.filter((b) => {
  const cx = (b.min.x + b.max.x) / 2;
  const cz = (b.min.z + b.max.z) / 2;
  return !stairTreads.some((t) =>
    cx > t.min.x - 0.01 && cx < t.max.x + 0.01 && cz > t.min.z - 0.01 && cz < t.max.z + 0.01 &&
    Math.abs(t.max.y - b.min.y) < 0.02);
});
check("every stair baluster stands on a tread", balusters.length > 3 && unsupported.length === 0,
  unsupported.slice(0, 3).map(fmt).join(" | "));

// --- 5. Nothing floats inside the stairwell void ----------------------------
const WELL = { minX: -1.85, maxX: -0.55, minZ: 0.32, maxZ: HOUSE_DEPTH / 2 };
const overWell = entries.filter((e) => {
  if (e.min.y < STORY_HEIGHT - 0.02) return false;
  const insideX = e.min.x > WELL.minX + 0.02 && e.max.x < WELL.maxX - 0.02;
  const insideZ = e.min.z > WELL.minZ + 0.02 && e.max.z < WELL.maxZ - 0.02;
  return insideX && insideZ;
});
check("nothing sits inside the stairwell void", overWell.length === 0,
  overWell.slice(0, 5).map((e) => `${e.path} ${fmt(e)}`).join(" | "));

// --- 6. Doorways are walkable ----------------------------------------------
const bandFree = (x: number, z: number, fromY: number, toY: number) =>
  !entries.some((e) =>
    x > e.min.x - 0.26 && x < e.max.x + 0.26 &&
    z > e.min.z - 0.26 && z < e.max.z + 0.26 &&
    e.max.y > fromY && e.min.y < toY);
const doorways: Array<[string, number, number, number]> = [
  ["ground kitchen<->hall", -2.9, 1.0, 0],
  ["ground study door", 2.65, 1.5, 0],
  ["upper bedroom2 door", -2.6, -0.8, STORY_HEIGHT],
  ["upper master door", 0.35, -0.8, STORY_HEIGHT],
  ["upper bathroom door", -2.45, 0.66, STORY_HEIGHT],
  ["upper front room door", -0.1, 0.66, STORY_HEIGHT],
];
console.log("");
for (const [label, x, z, baseY] of doorways) {
  check(`doorway clear: ${label}`, bandFree(x, z, baseY + 0.2, baseY + 1.6));
}

// --- 6b. No wall intersects the flight -------------------------------------
const flight: Entry = {
  path: "flight",
  min: new THREE.Vector3(-1.85, 0, 0.26),
  max: new THREE.Vector3(-0.55, STORY_HEIGHT, 4.13),
};
const wallsInFlight = entries.filter((e) => {
  if (e.min.x > -2.0 && e.max.x < -0.4 && e.min.z > 0.2) return false; // the stair itself
  if (e.path.endsWith("stair-handrail")) return false;
  if (e.max.y - e.min.y < 1.5) return false;                            // not a wall
  return overlap(e, flight) !== null;
});
check("no wall intersects the stair flight", wallsInFlight.length === 0,
  wallsInFlight.slice(0, 4).map((e) => `${e.path} ${fmt(e)}`).join(" | "));

// --- 6c. Garden planting clears the building ------------------------------
const envelope: Entry = {
  path: "house",
  min: new THREE.Vector3(-6.68, -0.2, -4.68),
  max: new THREE.Vector3(6.6, 6.1, 4.6),
};
const foliageInHouse = entries.filter((e) =>
  e.path.startsWith("garden-tree") && overlap(e, envelope) !== null);
check("garden planting clears the house", foliageInHouse.length === 0,
  foliageInHouse.slice(0, 3).map(fmt).join(" | "));

// --- 7. Robot route clearance ----------------------------------------------
const { TASK_WORLDS, ROBOT_RADIUS } = await import("../src/components/demoRoutes.ts");
const ROUTES = Object.fromEntries(
  Object.entries(TASK_WORLDS).map(([name, config]) => [name, config.route]),
);
{
  const RADIUS = ROBOT_RADIUS;
  const obstacles = entries.filter((e) =>
    !e.path.startsWith("upper-floor") && e.max.y > 0.06 && e.min.y < 1.4);
  console.log(`\nroute clearance (r=${RADIUS}, obstacles=${obstacles.length})`);
  for (const [name, points] of Object.entries(ROUTES)) {
    const curve = new THREE.CatmullRomCurve3(points.map(([x, z]) => new THREE.Vector3(x, 0, z)));
    const worst = new Map<string, { pen: number; box: Entry }>();
    for (const p of curve.getPoints(400)) {
      for (const o of obstacles) {
        const dx = Math.max(o.min.x - p.x, 0, p.x - o.max.x);
        const dz = Math.max(o.min.z - p.z, 0, p.z - o.max.z);
        const pen = RADIUS - Math.hypot(dx, dz);
        if (pen > 0) {
          const prev = worst.get(o.path + fmt(o));
          if (!prev || pen > prev.pen) worst.set(o.path + fmt(o), { pen, box: o });
        }
      }
    }
    const list = [...worst.values()].sort((a, b) => b.pen - a.pen);
    check(`route ${name} is clear`, list.length === 0,
      list.slice(0, 4).map((h) => `pen=${r(h.pen)} h=${r(h.box.max.y)} ${fmt(h.box)}`).join(" | "));
  }
}

console.log(`\n${failures === 0 ? "ALL STRUCTURAL CHECKS PASS" : `${failures} STRUCTURAL FAILURE(S)`}`);
process.exit(failures === 0 ? 0 : 1);
