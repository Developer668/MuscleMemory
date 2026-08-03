import type {
  EpisodeDetail,
  EpisodeSummary,
  LiveEpisodeOptions,
  LiveEpisodeStatus,
  ProviderHealth,
  ServiceHealth,
  TelemetryRecord,
} from "./types";

const EPISODE_ID = "local-routine-medicine-delivery";
const POLICY_ID = "delivery-v2-sensor-fusion-hysteresis";
const WORLD_ID = "apartment-local-routine-17";

export const localProviders: ProviderHealth[] = [
  {
    provider: "Local routine simulator",
    state: "simulation",
    detail: "Synthetic browser telemetry for local interface testing.",
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
  robot_checksum: "mm01-frozen-local-demo",
  world_id: WORLD_ID,
  world_hash: "local-routine-world-v1",
  policy_id: POLICY_ID,
  policy_hash: "local-policy-v2",
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
  catalog_id: "local-browser-routine",
  catalog_sha256: "local-browser-routine-v1",
  seeds: [17],
  policies: [
    {
      policy_id: POLICY_ID,
      policy_hash: "local-policy-v2",
      evaluated_episode_count: 0,
      promotable: false,
      deployment_status: "candidate_live_test",
      is_default: true,
    },
  ],
  default_policy_id: POLICY_ID,
  video_products: ["third_person", "left_eye_rgb", "right_eye_rgb", "stereo_composite", "derived_depth", "simulator_debug_segmentation"],
  maximum_duration_seconds: 30,
};

const route: Array<[number, number]> = [
  [-4.6, 3.6], [-3.7, 2.8], [-2.6, 1.7], [-1.5, 0.8], [-0.3, 0.2], [0.9, -0.8], [1.5, -2.1], [1.5, -3.1],
];

function interpolate(progress: number): [number, number, number] {
  const scaled = Math.max(0, Math.min(0.999, progress)) * (route.length - 1);
  const index = Math.floor(scaled);
  const fraction = scaled - index;
  const current = route[index];
  const next = route[Math.min(route.length - 1, index + 1)];
  const x = current[0] + (next[0] - current[0]) * fraction;
  const y = current[1] + (next[1] - current[1]) * fraction;
  return [x, y, Math.atan2(next[1] - current[1], next[0] - current[0])];
}

export function localRoutineRecord(sequence: number): TelemetryRecord {
  const progress = Math.min(sequence / 300, 1);
  const [x, y, yaw] = interpolate(progress);
  const phase = sequence / 6;
  const speed = progress >= 0.96 ? 0 : 0.42 + Math.sin(phase) * 0.025;
  const clearance = 0.34 + Math.sin(phase * 0.7) * 0.045;
  const tilt = 2.1 + Math.abs(Math.sin(phase * 0.9)) * 1.25;
  return {
    episode_id: EPISODE_ID,
    world_id: WORLD_ID,
    policy_id: POLICY_ID,
    sequence,
    sim_time_seconds: sequence / 10,
    event_time: Date.now() / 1000,
    failure_type: null,
    frame_id: `local-frame-${String(sequence).padStart(5, "0")}`,
    frame_join_key: "frame_id",
    signal_use: "Used by policy",
    delivery: "simulation",
    payload_checksum: `local-${sequence}`,
    payload: {
      position_x_m: x,
      position_y_m: y,
      yaw_rad: yaw,
      current_obstacle_clearance_m: clearance,
      tray_tilt_degrees: tilt,
      policy_action: { forward_speed_mps: speed, turning_rate_rad_s: Math.sin(phase * 0.5) * 0.12, stop_probability: progress >= 0.96 ? 0.98 : 0.02 },
      rocketride_step: progress < 0.1 ? "validate_world" : progress < 0.82 ? "run_episode" : "summarize_telemetry",
      routine_mode: "local_synthetic",
    },
    sensors: [
      { category: "Stereo vision and depth", signal_use: "Used by policy", available: true, values: { derived_depth_sectors_m: Array.from({ length: 32 }, (_, index) => 0.48 + ((index * 13 + sequence) % 34) / 18) } },
      { category: "Linkwise IMUs", signal_use: "Used by policy", available: true, values: { torso_pitch_degrees: 0.7 + Math.sin(phase) * 0.2 } },
      { category: "Joint position and effort", signal_use: "Used by policy", available: true, values: { mean_effort_nm: 18.4 + Math.sin(phase) * 1.3 } },
      { category: "Foot contacts", signal_use: "Used by policy", available: true, values: { left_contact: sequence % 10 < 5, right_contact: sequence % 10 >= 5 } },
      { category: "Wrist force and tray balance", signal_use: "Used by policy", available: true, values: { tray_tilt_degrees: tilt } },
      { category: "Hand pressure and slip", signal_use: "Used by policy", available: true, values: { package_slip_detected: false, grip_pressure_kpa: 41.2 } },
      { category: "Microphone activity", signal_use: "Logged only", available: false, values: { status: "unavailable in fixed profile" } },
      { category: "Battery and energy", signal_use: "Logged only", available: true, values: { battery_percent: 87.4 - progress * 0.3, power_watts: 412 + Math.sin(phase) * 14 } },
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
    policy_hash: "local-policy-v2",
    policy_promotable: false,
    simulation_time_seconds: sequence / 10,
    wall_elapsed_seconds: sequence / 10,
    wall_clock_lag_seconds: 0,
    telemetry_records: sequence + 1,
    video_frames: 0,
    dropped_video_frames: 0,
    last_frame_id: `local-frame-${String(sequence).padStart(5, "0")}`,
    provider_state: "simulation",
    completion_reason: active ? null : "local routine stopped",
    success: null,
    failed_reasons: [],
    graph_provider_complete: false,
    telemetry_provider_complete: false,
    error_type: null,
    detail: "Local synthetic telemetry. No backend provider was contacted.",
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
