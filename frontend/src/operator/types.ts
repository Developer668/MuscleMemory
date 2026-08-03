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

