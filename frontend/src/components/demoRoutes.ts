import * as THREE from "three";

export type DemoTask = "medicine" | "breakfast" | "kitchen_check" | "parcel";

export type TaskWorldConfig = {
  route: Array<[number, number]>;
  color: string;
  payloadColor: string;
};

/**
 * Ground-floor delivery routes, in metres on the house's own axes.
 *
 * These walk the house's actual doorways. The staircase occupies the middle of the
 * spine wall, so anything crossing between the foyer and the dining room has to go
 * round it through the kitchen — a straight line between those rooms is a wall.
 *
 * `npm run audit:house` asserts every route keeps a robot-radius clearance from
 * the built geometry, so edit these together with the house, not on their own.
 */
export const TASK_WORLDS: Record<DemoTask, TaskWorldConfig> = {
  medicine: {
    route: [[-4.95, 4.05], [-4.5, 3.5], [-4.35, 2.65], [-3.95, 1.85], [-2.95, 1.45], [-2.85, 0.5], [-2.2, -0.4], [-0.7, -0.65], [0.55, -1.5], [1.45, -2.55], [1.5, -3.35]],
    color: "#ff6e57",
    payloadColor: "#b84d46",
  },
  breakfast: {
    route: [[-3.72, -3.15], [-2.6, -3.05], [-1.5, -2.8], [-1.25, -1.9], [-1.15, -0.9], [-0.55, -0.15], [-0.02, 0.5], [0.0, 1.3], [0.0, 2.1], [-0.02, 2.9]],
    color: "#ffd75f",
    payloadColor: "#d6a542",
  },
  kitchen_check: {
    route: [[5.6, 2.4], [4.4, 1.9], [3.4, 1.6], [2.65, 1.5], [2.0, 1.35], [1.2, 0.85], [-0.2, -0.2], [-1.35, -0.7], [-1.45, -1.9], [-1.6, -2.95], [-2.7, -3.2], [-3.72, -3.2]],
    color: "#77e7ff",
    payloadColor: "#77e7ff",
  },
  parcel: {
    route: [[5.6, 2.45], [4.4, 1.9], [3.4, 1.6], [2.65, 1.5], [2.0, 1.35], [1.2, 0.85], [-0.3, -0.15], [-1.8, -0.35], [-2.85, 0.3], [-2.95, 1.45], [-3.6, 2.1], [-4.45, 2.65]],
    color: "#ff9a78",
    payloadColor: "#ad7548",
  },
};

/** What the robot looks at once it arrives: the delivery surface for each task. */
export const TASK_FOCUS: Record<DemoTask, THREE.Vector3> = {
  medicine: new THREE.Vector3(3.55, 0.74, -3.42),
  breakfast: new THREE.Vector3(1.4, 0.74, 3.1),
  kitchen_check: new THREE.Vector3(-3.72, 0.9, -4.05),
  parcel: new THREE.Vector3(-5.55, 0.5, 2.75),
};

/** Robot footprint radius used for both the walk and the clearance audit. */
export const ROBOT_RADIUS = 0.28;
