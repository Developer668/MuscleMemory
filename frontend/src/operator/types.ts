export type ProviderState =
  | "unconfigured"
  | "configured"
  | "healthy"
  | "degraded"
  | "end_to_end_verified"
  | "simulation"
  | "cached";

export type SignalUse = "Used by policy" | "Logged only" | "Simulator ground truth";

export interface ProviderHealth {
  provider: string;
  state: ProviderState;
  detail: string;
  checked_at: string;
  evidence_id: string | null;
}

export interface ServiceHealth {
  service: "muscle-memory-api";
  api_version: "v1";
  state: ProviderState;
  providers: ProviderHealth[];
  checked_at: string;
}

export type MemoryGraphOwner =
  | "system"
  | "World & Physics Agent"
  | "Failure & Curriculum Agent"
  | "Safety & Evaluation Agent";

export interface MemoryGraphNode {
  id: string;
  label: string;
  record_kind: string;
  owner: MemoryGraphOwner;
  properties: Record<string, unknown>;
}

export interface MemoryGraphEdge {
  id: string;
  source: string;
  target: string;
  relationship: string;
}

export interface MemoryGraphSnapshot {
  exposure: "operational_only";
  provider: "FalkorDB";
  provider_state: ProviderState;
  graph_name: string;
  source: "falkordb" | "local_cache";
  provider_checked_at: string;
  refreshed_at: string;
  fact_count: number;
  nodes: MemoryGraphNode[];
  edges: MemoryGraphEdge[];
}

export interface EpisodeSummary {
  episode_id: string;
  kind: "training" | "development_evaluation" | "demo";
  state: "created" | "running" | "succeeded" | "failed" | "aborted";
  robot_checksum: string;
  world_id: string;
  world_hash: string;
  policy_id: string;
  policy_hash: string;
  opened_at: string;
  closed_at: string | null;
}

export interface EpisodeList {
  exposure: "operational_only";
  items: EpisodeSummary[];
  next_cursor: string | null;
}

export interface EpisodeDetail {
  episode: EpisodeSummary;
  telemetry_records: number;
  provider_delivery: ProviderState;
  result: Record<string, unknown> | null;
  failure_ids: string[];
  correction_ids: string[];
}

export interface SensorReading {
  category: string;
  signal_use: SignalUse;
  available: boolean;
  values: unknown;
}

export interface TelemetryRecord {
  episode_id: string;
  world_id: string;
  policy_id: string;
  sequence: number;
  sim_time_seconds: number;
  event_time: number;
  failure_type: string | null;
  frame_id: string | null;
  frame_join_key: "frame_id";
  signal_use: SignalUse;
  sensors: SensorReading[];
  payload: Record<string, unknown>;
  payload_checksum: string;
  delivery: ProviderState;
}

export interface TelemetryPage {
  episode_id: string;
  cadence_hz: 20;
  records: TelemetryRecord[];
  next_sequence: number | null;
}

export interface ReplayPage {
  episode_id: string;
  frame_join_key: "frame_id";
  records: TelemetryRecord[];
  next_sequence: number | null;
}

export type VideoProduct =
  | "third_person"
  | "left_eye_rgb"
  | "right_eye_rgb"
  | "stereo_composite"
  | "derived_depth"
  | "simulator_debug_segmentation";

export interface LivePolicyOption {
  policy_id: string;
  policy_hash: string;
  evaluated_episode_count: number;
  promotable: boolean;
  deployment_status: "stable_deployed" | "candidate_live_test";
  is_default: boolean;
}

export interface LiveEpisodeOptions {
  enabled: boolean;
  unavailable_reason: string | null;
  mode: "training";
  catalog_id: string | null;
  catalog_sha256: string | null;
  seeds: number[];
  policies: LivePolicyOption[];
  default_policy_id: string | null;
  video_products: VideoProduct[];
  maximum_duration_seconds: number | null;
}

export type LiveEpisodePhase =
  | "queued"
  | "starting"
  | "running"
  | "cancelling"
  | "closed"
  | "failed";

export interface LiveEpisodeStatus {
  episode_id: string;
  phase: LiveEpisodePhase;
  health: "starting" | "healthy" | "degraded" | "terminal" | "failed";
  world_id: string;
  policy_id: string;
  policy_hash: string;
  policy_promotable: boolean;
  simulation_time_seconds: number;
  wall_elapsed_seconds: number;
  wall_clock_lag_seconds: number;
  telemetry_records: number;
  video_frames: number;
  dropped_video_frames: number;
  last_frame_id: string | null;
  provider_state: string | null;
  completion_reason: string | null;
  success: boolean | null;
  failed_reasons: string[];
  graph_provider_complete: boolean | null;
  telemetry_provider_complete: boolean | null;
  error_type: string | null;
  detail: string | null;
  video_streams: Record<VideoProduct, string>;
}

export interface LiveStreamMessage {
  schema_version: "muscle-memory.live.v1";
  kind: "telemetry" | "status";
  episode_id: string;
  cadence_hz: 20;
  frame_join_key: "frame_id";
  frame_id: string | null;
  telemetry: TelemetryRecord | null;
  status: Record<string, unknown> | null;
  emitted_at: string;
  dropped_before: number;
}

export interface PendingApproval {
  requirement_id: string;
  kind:
    | "uncertain_physical_properties"
    | "reward_change"
    | "curriculum_change"
    | "policy_promotion"
    | "policy_rollback"
    | "correction";
  summary: string;
  blocking: true;
  plan_digest: string | null;
  run_id: string | null;
  created_at: string;
}

export interface PendingApprovalList {
  items: PendingApproval[];
}

export interface CorrectionPoint {
  x_m: number;
  y_m: number;
}

export interface CorrectionView {
  correction_id: string;
  episode_id: string;
  failure_id: string;
  kind: "route" | "keep_out";
  state: "pending" | "approved" | "rejected";
  submitted_by: string;
  created_at: string;
  graph_delivery: ProviderState;
}

export interface PolicyMetrics {
  episode_count: number;
  success_rate: number;
  collision_rate: number;
  falls: number;
  median_clearance_m: number;
  median_path_efficiency: number;
}

export interface PolicySummary {
  policy_id: string;
  policy_hash: string;
  evaluated: boolean;
  evaluation_scope: "development" | "held_out_aggregate" | "none";
  metrics: PolicyMetrics | null;
  immutable: boolean;
}

export interface PolicySummaryList {
  items: PolicySummary[];
}

export interface TaskPolicyTrainingMetrics {
  training_episode_count: number;
  validation_episode_count: number;
  training_sample_count: number;
  validation_sample_count: number;
  best_epoch: number;
  training_command_accuracy: number;
  validation_command_accuracy: number;
  validation_loss: number;
  validation_forward_mae_mps: number;
  validation_turning_mae_rad_s: number;
  validation_stop_mae: number;
}

export interface TaskPolicyTrainingJob {
  job_id: string;
  policy_id: string;
  state: "queued" | "running" | "completed" | "failed";
  epochs: number;
  seed: number;
  dataset_sha256: string;
  training_data_split: "training";
  robot_component: "high_level_task_policy";
  promotion_status: "not_evaluated";
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  checkpoint_sha256: string | null;
  evidence_sha256: string | null;
  metrics: TaskPolicyTrainingMetrics | null;
  error_type: string | null;
}

export interface TaskPolicyTrainingJobList {
  items: TaskPolicyTrainingJob[];
}

export interface PromotionEligibility {
  baseline_policy_id: string;
  candidate_policy_id: string;
  held_out_episode_count: number;
  checks: Record<string, boolean>;
  numerically_eligible: boolean;
  human_approval_required: true;
  promotion_applied: boolean;
  evidence_hash: string | null;
}

export interface ApiErrorPayload {
  error?: {
    code?: string;
    message?: string;
    request_id?: string;
  };
}

export type StreamState = "idle" | "connecting" | "live" | "closed" | "error";
