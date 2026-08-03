import type {
  EpisodeDetail,
  EpisodeSummary,
  LiveEpisodeOptions,
  LiveEpisodeStatus,
  ProviderHealth,
  ServiceHealth,
  TelemetryRecord,
} from "./types";

const EPISODE_ID = "synthetic-house-loop-mm01";
const POLICY_ID = "delivery-v2-sensor-fusion-hysteresis";
const WORLD_ID = "apartment-synthetic-loop-17";
const LOOP_SAMPLES = 900;

type RoutineMove = "walk" | "scan" | "yield" | "turn" | "handoff" | "return";

type RoutineWaypoint = {
  label: string;
  move: RoutineMove;
  point: readonly [number, number];
};

// These are visual-only house coordinates. They deliberately follow the actual
// ground-floor routes and never change the fixed robot or simulator world state.
const LOOP_ROUTE: readonly RoutineWaypoint[] = [
  { label: "Entry scan", move: "scan", point: [-4.95, 4.05] },
  { label: "Clear the entry", move: "walk", point: [-4.5, 3.5] },
  { label: "Pass the boot rack", move: "yield", point: [-4.35, 2.65] },
  { label: "Hallway turn", move: "turn", point: [-3.95, 1.85] },
  { label: "Clear the football", move: "walk", point: [-2.95, 1.45] },
  { label: "Cross the kitchen", move: "walk", point: [-2.85, 0.5] },
  { label: "Depth sweep", move: "scan", point: [-2.2, -0.4] },
  { label: "Living room approach", move: "walk", point: [-0.7, -0.65] },
  { label: "Sofa clearance", move: "yield", point: [0.55, -1.5] },
  { label: "Delivery approach", move: "walk", point: [1.45, -2.55] },
  { label: "Medicine handoff", move: "handoff", point: [3.42, -2.68] },
  { label: "Lounge return", move: "return", point: [4.7, -2.15] },
  { label: "Window scan", move: "scan", point: [4.9, -0.82] },
  { label: "Dining turn", move: "turn", point: [3.9, 0.48] },
  { label: "Table clearance", move: "yield", point: [2.7, 1.5] },
  { label: "Dining circuit", move: "walk", point: [1.4, 3.1] },
  { label: "Kitchen glance", move: "scan", point: [0.0, 2.9] },
  { label: "Hall return", move: "return", point: [-0.2, 1.5] },
  { label: "Entry approach", move: "walk", point: [-1.8, 0.3] },
  { label: "Reset route", move: "return", point: [-2.95, 1.45] },
  { label: "Entry reset", move: "turn", point: [-3.6, 2.1] },
  { label: "Loop complete", move: "scan", point: [-4.45, 2.65] },
];

export const localProviders: ProviderHealth[] = [
  {
    provider: "Browser demo loop",
    state: "simulation",
    detail: "Synthetic telemetry rendered locally for the interactive house tour.",
    checked_at: new Date().toISOString(),
    evidence_id: null,
  },
];

export const localHealth: ServiceHealth = {
  service: "muscle-memory-api",
  api_version: "v1",
  state: "simulation",
  providers: localProviders,
  checked_at: new Date().toISOString(),
};

export const localEpisode: EpisodeSummary = {
  episode_id: EPISODE_ID,
  kind: "demo",
  state: "created",
  robot_checksum: "mm01-frozen-demo-visualization",
  world_id: WORLD_ID,
  world_hash: "synthetic-house-loop-v2",
  policy_id: POLICY_ID,
  policy_hash: "synthetic-demo-policy-v2",
  opened_at: new Date().toISOString(),
  closed_at: null,
};

export const localDetail: EpisodeDetail = {
  episode: localEpisode,
  telemetry_records: 0,
  provider_delivery: "simulation",
  result: null,
  failure_ids: [],
  correction_ids: [],
};

export const localLiveOptions: LiveEpisodeOptions = {
  enabled: true,
  unavailable_reason: null,
  mode: "training",
  catalog_id: "browser-synthetic-house-loop",
  catalog_sha256: "browser-synthetic-house-loop-v2",
  seeds: [17],
  policies: [
    {
      policy_id: POLICY_ID,
      policy_hash: "synthetic-demo-policy-v2",
      evaluated_episode_count: 0,
      promotable: false,
      deployment_status: "candidate_live_test",
      is_default: true,
    },
  ],
  default_policy_id: POLICY_ID,
  video_products: ["third_person", "left_eye_rgb", "right_eye_rgb", "stereo_composite", "derived_depth", "simulator_debug_segmentation"],
  maximum_duration_seconds: 90,
};

function loopProgress(sequence: number): number {
  return ((sequence % LOOP_SAMPLES) + LOOP_SAMPLES) % LOOP_SAMPLES / LOOP_SAMPLES;
}

function interpolate(sequence: number): { x: number; y: number; yaw: number; move: RoutineMove; label: string; segment: number } {
  const progress = loopProgress(sequence);
  const scaled = progress * LOOP_ROUTE.length;
  const segment = Math.floor(scaled);
  const fraction = scaled - segment;
  const current = LOOP_ROUTE[segment];
  const next = LOOP_ROUTE[(segment + 1) % LOOP_ROUTE.length];
  const x = current.point[0] + (next.point[0] - current.point[0]) * fraction;
  const y = current.point[1] + (next.point[1] - current.point[1]) * fraction;
  return {
    x,
    y,
    yaw: Math.atan2(next.point[1] - current.point[1], next.point[0] - current.point[0]),
    move: current.move,
    label: current.label,
    segment,
  };
}

function motionProfile(move: RoutineMove, phase: number): { speed: number; turn: number; stop: number; clearance: number; tilt: number } {
  const sway = Math.sin(phase * 0.73);
  if (move === "handoff") return { speed: 0, turn: sway * 0.025, stop: 0.98, clearance: 0.64, tilt: 1.1 };
  if (move === "scan") return { speed: 0.13, turn: 0.34 + sway * 0.07, stop: 0.28, clearance: 0.57, tilt: 1.5 };
  if (move === "yield") return { speed: 0.19, turn: sway * 0.16, stop: 0.38, clearance: 0.31 + Math.abs(sway) * 0.035, tilt: 2.5 };
  if (move === "turn") return { speed: 0.18, turn: 0.46 + sway * 0.08, stop: 0.18, clearance: 0.42, tilt: 2.0 };
  if (move === "return") return { speed: 0.38, turn: sway * 0.12, stop: 0.04, clearance: 0.47, tilt: 2.3 };
  return { speed: 0.46 + sway * 0.03, turn: sway * 0.11, stop: 0.02, clearance: 0.5 + sway * 0.035, tilt: 2.2 + Math.abs(sway) * 0.5 };
}

export function localRoutineRecord(sequence: number): TelemetryRecord {
  const route = interpolate(sequence);
  const phase = sequence / 6;
  const profile = motionProfile(route.move, phase);
  const cycle = Math.floor(sequence / LOOP_SAMPLES) + 1;
  const battery = 86.8 - loopProgress(sequence) * 0.18;
  return {
    episode_id: EPISODE_ID,
    world_id: WORLD_ID,
    policy_id: POLICY_ID,
    sequence,
    sim_time_seconds: sequence / 10,
    event_time: Date.now() / 1000,
    failure_type: null,
    frame_id: `demo-frame-${String(sequence).padStart(5, "0")}`,
    frame_join_key: "frame_id",
    signal_use: "Used by policy",
    delivery: "simulation",
    payload_checksum: `synthetic-loop-${sequence}`,
    payload: {
      position_x_m: route.x,
      position_y_m: route.y,
      yaw_rad: route.yaw,
      current_obstacle_clearance_m: profile.clearance,
      tray_tilt_degrees: profile.tilt,
      policy_action: { forward_speed_mps: profile.speed, turning_rate_rad_s: profile.turn, stop_probability: profile.stop },
      rocketride_step: "run_episode",
      routine_mode: "synthetic_demo_loop",
      routine_label: route.label,
      routine_move: route.move,
      loop_cycle: cycle,
      loop_progress: loopProgress(sequence),
      synthetic_notice: "Browser-generated visualization only. Not simulator or provider evidence.",
    },
    sensors: [
      { category: "Stereo vision and depth", signal_use: "Used by policy", available: true, values: { derived_depth_sectors_m: Array.from({ length: 32 }, (_, index) => 0.42 + ((index * 17 + sequence * 3) % 39) / 17) } },
      { category: "Linkwise IMUs", signal_use: "Used by policy", available: true, values: { torso_pitch_degrees: 0.55 + Math.sin(phase) * 0.24, pelvis_yaw_rate: profile.turn } },
      { category: "Joint position and effort", signal_use: "Used by policy", available: true, values: { mean_effort_nm: 19.6 + profile.speed * 15 + Math.sin(phase) * 1.5 } },
      { category: "Foot contacts", signal_use: "Used by policy", available: true, values: { left_contact: Math.sin(phase * 2) >= 0, right_contact: Math.sin(phase * 2) < 0 } },
      { category: "Wrist force and tray balance", signal_use: "Used by policy", available: true, values: { tray_tilt_degrees: profile.tilt, payload_secure: true } },
      { category: "Hand pressure and slip", signal_use: "Used by policy", available: true, values: { package_slip_detected: false, grip_pressure_kpa: route.move === "handoff" ? 38.6 : 42.4 } },
      { category: "Microphone activity", signal_use: "Logged only", available: false, values: { status: "unavailable in fixed profile" } },
      { category: "Battery and energy", signal_use: "Logged only", available: true, values: { battery_percent: battery, power_watts: 392 + profile.speed * 74 + Math.sin(phase) * 16 } },
    ],
  };
}

export function localLiveStatus(sequence: number, active: boolean): LiveEpisodeStatus {
  return {
    episode_id: EPISODE_ID,
    phase: active ? "running" : "closed",
    health: active ? "healthy" : "terminal",
    world_id: WORLD_ID,
    policy_id: POLICY_ID,
    policy_hash: "synthetic-demo-policy-v2",
    policy_promotable: false,
    simulation_time_seconds: sequence / 10,
    wall_elapsed_seconds: sequence / 10,
    wall_clock_lag_seconds: 0,
    telemetry_records: sequence + 1,
    video_frames: 0,
    dropped_video_frames: 0,
    last_frame_id: `demo-frame-${String(sequence).padStart(5, "0")}`,
    provider_state: "simulation",
    completion_reason: active ? null : "Synthetic demo paused",
    success: null,
    failed_reasons: [],
    graph_provider_complete: false,
    telemetry_provider_complete: false,
    error_type: null,
    detail: "Browser-generated synthetic telemetry. No backend provider was contacted.",
    video_streams: {
      third_person: "",
      left_eye_rgb: "",
      right_eye_rgb: "",
      stereo_composite: "",
      derived_depth: "",
      simulator_debug_segmentation: "",
    },
  };
}
