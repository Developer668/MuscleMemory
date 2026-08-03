import {
  Activity,
  AlertTriangle,
  BatteryCharging,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Cloud,
  Crosshair,
  Eye,
  EyeOff,
  Footprints,
  Gauge,
  Hand,
  KeyRound,
  Layers3,
  Mic2,
  Network,
  Pause,
  Play,
  Radio,
  RefreshCw,
  RotateCcw,
  Route,
  ScanLine,
  ShieldCheck,
  Square,
  Undo2,
  X,
  XCircle,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { BrandMark } from "../components/BrandMark";
import { RealisticHomeScene } from "../operator/RealisticHomeScene";
import { useOperatorData } from "../operator/useOperatorData";
import type {
  CorrectionPoint,
  PolicyMetrics,
  ProviderState,
  SensorReading,
  SignalUse,
  TelemetryRecord,
} from "../operator/types";

const SENSOR_CATEGORIES = [
  "Stereo vision and depth",
  "Linkwise IMUs",
  "Joint position and effort",
  "Foot contacts",
  "Wrist force and tray balance",
  "Hand pressure and slip",
  "Microphone activity",
  "Battery and energy",
] as const;

const SENSOR_DEFAULT_USE: Record<(typeof SENSOR_CATEGORIES)[number], SignalUse> = {
  "Stereo vision and depth": "Used by policy",
  "Linkwise IMUs": "Used by policy",
  "Joint position and effort": "Used by policy",
  "Foot contacts": "Used by policy",
  "Wrist force and tray balance": "Used by policy",
  "Hand pressure and slip": "Used by policy",
  "Microphone activity": "Logged only",
  "Battery and energy": "Logged only",
};

const VIDEO_FEEDS = [
  ["left_eye_rgb", "Left eye RGB"],
  ["right_eye_rgb", "Right eye RGB"],
  ["stereo_composite", "Stereo composite"],
  ["derived_depth", "Derived depth"],
  ["simulator_debug_segmentation", "Simulator debug segmentation"],
] as const;

type ViewBox = { x: number; y: number; width: number; height: number };

function stateLabel(state: string): string {
  return state.replaceAll("_", " ");
}

function shortId(value: string | null | undefined, length = 10): string {
  if (!value) return "Unavailable";
  return value.length <= length ? value : `${value.slice(0, length)}…`;
}

function formatTime(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "--:--.---";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nestedValue(source: unknown, paths: string[]): unknown {
  for (const path of paths) {
    let current: unknown = source;
    for (const segment of path.split(".")) {
      if (!isRecord(current)) {
        current = undefined;
        break;
      }
      current = current[segment];
    }
    if (current !== undefined && current !== null) return current;
  }
  return undefined;
}

function numberValue(source: unknown, paths: string[]): number | null {
  const value = nestedValue(source, paths);
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringValue(source: unknown, paths: string[]): string | null {
  const value = nestedValue(source, paths);
  return typeof value === "string" && value.trim() ? value : null;
}

function extractPosition(record: TelemetryRecord | null | undefined): CorrectionPoint | null {
  if (!record) return null;
  const x = numberValue(record.payload, [
    "position_x_m",
    "position.x_m",
    "robot_position.x_m",
    "pose.x_m",
  ]);
  const y = numberValue(record.payload, [
    "position_y_m",
    "position.y_m",
    "robot_position.y_m",
    "pose.y_m",
  ]);
  return x === null || y === null ? null : { x_m: x, y_m: y };
}

function extractYaw(record: TelemetryRecord | null | undefined): number | null {
  if (!record) return null;
  return numberValue(record.payload, [
    "yaw_radians",
    "yaw_rad",
    "rotation.yaw_radians",
    "robot_pose.yaw_radians",
    "pose.yaw_radians",
  ]);
}

function metricFromSensors(record: TelemetryRecord | null, paths: string[]): number | null {
  if (!record) return null;
  for (const sensor of record.sensors) {
    const value = numberValue(sensor.values, paths);
    if (value !== null) return value;
  }
  return numberValue(record.payload, paths);
}

function booleanFromRecord(record: TelemetryRecord | null, paths: string[]): boolean | null {
  if (!record) return null;
  for (const sensor of record.sensors) {
    const value = nestedValue(sensor.values, paths);
    if (typeof value === "boolean") return value;
  }
  const value = nestedValue(record.payload, paths);
  return typeof value === "boolean" ? value : null;
}

function actionLabel(record: TelemetryRecord | null): string {
  if (!record) return "Unavailable";
  const action = nestedValue(record.payload, ["policy_action", "action"]);
  if (typeof action === "string") return action;
  if (isRecord(action)) {
    const forward = numberValue(action, ["forward_speed", "forward_speed_mps"]);
    const turning = numberValue(action, ["turning_rate", "turning_rate_rad_s"]);
    const stop = numberValue(action, ["stop_probability"]);
    const parts = [
      forward === null ? null : `${forward.toFixed(2)} m/s`,
      turning === null ? null : `${turning.toFixed(2)} rad/s`,
      stop === null ? null : `stop ${(stop * 100).toFixed(0)}%`,
    ].filter(Boolean);
    return parts.length ? parts.join(" · ") : "Recorded action";
  }
  return "Unavailable";
}

function safeMediaUrl(value: unknown): string | null {
  if (typeof value !== "string" || value === "video_stream") return null;
  if (value.startsWith("/") || value.startsWith("https://") || value.startsWith("http://")) {
    return value;
  }
  return null;
}

function mediaUrl(record: TelemetryRecord | null, key: string): string | null {
  if (!record) return null;
  const stereo = record.sensors.find((sensor) => sensor.category === "Stereo vision and depth");
  return safeMediaUrl(
    nestedValue(stereo?.values, [
      `${key}.stream_url`,
      `${key}.url`,
      key,
    ]) ??
      nestedValue(record.payload, [
        `video_metadata.${key}.stream_url`,
        `video_metadata.${key}.url`,
        `video_metadata.${key}`,
      ]),
  );
}

function depthSectors(record: TelemetryRecord | null): number[] {
  if (!record) return [];
  const stereo = record.sensors.find((sensor) => sensor.category === "Stereo vision and depth");
  const raw = nestedValue(stereo?.values, ["derived_depth_sectors", "derived_depth_sectors_m"]);
  if (!Array.isArray(raw)) return [];
  return raw.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
}

function previewValue(value: unknown): string {
  if (value === null || value === undefined) return "Unavailable";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "string") return value === "video_stream" ? "Direct stream" : value;
  if (Array.isArray(value)) {
    const numeric = value.filter((item): item is number => typeof item === "number");
    if (numeric.length === value.length && numeric.length) {
      return `${numeric.length} values · ${Math.min(...numeric).toFixed(2)}–${Math.max(...numeric).toFixed(2)}`;
    }
    return `${value.length} values`;
  }
  if (isRecord(value)) return `${Object.keys(value).length} channels`;
  return String(value);
}

function sensorHighlights(sensor: SensorReading): [string, string][] {
  if (!sensor.available || !isRecord(sensor.values)) return [];
  return Object.entries(sensor.values)
    .slice(0, 3)
    .map(([key, value]) => [key.replaceAll("_", " "), previewValue(value)]);
}

function sensorIcon(category: string) {
  if (category.startsWith("Stereo")) return Eye;
  if (category.startsWith("Linkwise")) return Activity;
  if (category.startsWith("Joint")) return Gauge;
  if (category.startsWith("Foot")) return Footprints;
  if (category.startsWith("Wrist")) return Layers3;
  if (category.startsWith("Hand")) return Hand;
  if (category.startsWith("Microphone")) return Mic2;
  return BatteryCharging;
}

function StatusDot({ state }: { state: ProviderState | "live" | "closed" | "error" | "idle" | "connecting" }) {
  return <span className={`ops-status-dot ops-state--${state}`} aria-hidden="true" />;
}

function ProviderStrip({ providers }: { providers: ReturnType<typeof useOperatorData>["providers"] }) {
  const sponsorNames = ["LaserData", "FalkorDB", "Guild.ai", "RocketRide"];
  return (
    <div className="ops-provider-strip" aria-label="Sponsor provider health">
      {sponsorNames.map((name) => {
        const provider = providers.find((item) => item.provider.toLowerCase().includes(name.split(".")[0].toLowerCase()));
        return (
          <span key={name} className="ops-provider" title={provider?.detail || `${name} not reported`}>
            <StatusDot state={provider?.state || "unconfigured"} />
            <span>{name}</span>
            <strong>{stateLabel(provider?.state || "unconfigured")}</strong>
          </span>
        );
      })}
    </div>
  );
}

function TopBar({ data }: { data: ReturnType<typeof useOperatorData> }) {
  const [showToken, setShowToken] = useState(false);
  const episode = data.detail?.episode;
  return (
    <header className="ops-topbar">
      <a className="ops-brand" href="/" aria-label="Muscle Memory product overview">
        <BrandMark />
        <span>
          <strong>Muscle Memory</strong>
          <small>MM-01 operations</small>
        </span>
      </a>

      <div className="ops-topbar__facts" aria-label="Current episode facts">
        <div><span>Policy</span><strong>{episode?.policy_id || "No episode"}</strong></div>
        <div><span>World</span><strong>{episode?.world_id || "Unavailable"}</strong></div>
        <div><span>Mode</span><strong>{episode ? stateLabel(episode.kind) : "Unavailable"}</strong></div>
        <div><span>Task</span><strong>{episode ? stateLabel(episode.state) : "Idle"}</strong></div>
      </div>

      <div className="ops-topbar__actions">
        <label className="ops-token" title="Credential is kept in this browser tab only">
          <KeyRound size={14} />
          <input
            type={showToken ? "text" : "password"}
            value={data.token}
            onChange={(event) => data.setToken(event.target.value)}
            placeholder="Operator credential"
            autoComplete="off"
            aria-label="Operator bearer credential"
          />
          <button
            type="button"
            onClick={() => setShowToken((visible) => !visible)}
            title={showToken ? "Hide credential" : "Show credential"}
            aria-label={showToken ? "Hide credential" : "Show credential"}
          >
            {showToken ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </label>
        <button
          type="button"
          className="ops-icon-button"
          onClick={() => void data.refresh()}
          title="Refresh backend state"
          aria-label="Refresh backend state"
        >
          <RefreshCw size={16} />
        </button>
        <a className="ops-text-link" href="/">Overview</a>
      </div>
      <ProviderStrip providers={data.providers} />
    </header>
  );
}

function EpisodePicker({ data }: { data: ReturnType<typeof useOperatorData> }) {
  return (
    <section className="ops-episode-bar" aria-label="Episode selection">
      <label>
        <Radio size={15} />
        <span>Episode</span>
        <select
          value={data.selectedEpisodeId}
          onChange={(event) => data.setSelectedEpisodeId(event.target.value)}
          disabled={!data.episodes.length}
        >
          {!data.episodes.length && <option value="">No operational episodes</option>}
          {data.episodes.map((episode) => (
            <option key={episode.episode_id} value={episode.episode_id}>
              {episode.episode_id} · {episode.state}
            </option>
          ))}
        </select>
      </label>
      <span className="ops-stream-state">
        <StatusDot state={data.streamState} />
        {data.streamState === "live" ? "20 Hz stream connected" : `Stream ${data.streamState}`}
      </span>
      <span><Cloud size={14} /> {data.detail ? stateLabel(data.detail.provider_delivery) : "No delivery record"}</span>
      <span><ShieldCheck size={14} /> Robot {shortId(data.detail?.episode.robot_checksum, 12)}</span>
      {data.droppedMessages > 0 && (
        <span className="ops-warning"><AlertTriangle size={14} /> {data.droppedMessages} dropped before latest</span>
      )}
    </section>
  );
}

function sensorRail(record: TelemetryRecord | null): SensorReading[] {
  return SENSOR_CATEGORIES.map((category) => {
    const reading = record?.sensors.find((item) => item.category === category);
    return reading || {
      category,
      signal_use: SENSOR_DEFAULT_USE[category],
      available: false,
      values: null,
    };
  });
}

function SensorRail({ record }: { record: TelemetryRecord | null }) {
  return (
    <aside className="ops-sensor-rail" aria-labelledby="sensor-title">
      <div className="ops-section-heading">
        <div><Activity size={15} /><span>Sensor rail</span></div>
        <strong id="sensor-title">8 / 8 categories</strong>
      </div>
      <div className="ops-sensor-list">
        {sensorRail(record).map((sensor, index) => {
          const Icon = sensorIcon(sensor.category);
          const highlights = sensorHighlights(sensor);
          return (
            <details className="ops-sensor-card" key={sensor.category} open={index < 2}>
              <summary>
                <span className="ops-sensor-icon"><Icon size={16} /></span>
                <span className="ops-sensor-name">
                  <strong>{sensor.category}</strong>
                  <small className={`ops-use-label ops-use-label--${sensor.signal_use.toLowerCase().replaceAll(" ", "-")}`}>
                    {sensor.signal_use}
                  </small>
                </span>
                <span className={sensor.available ? "ops-available" : "ops-unavailable"}>
                  {sensor.available ? "Live" : "Unavailable"}
                </span>
                <ChevronDown size={14} className="ops-details-chevron" />
              </summary>
              <div className="ops-sensor-body">
                {highlights.length ? highlights.map(([label, value]) => (
                  <div key={label}><span>{label}</span><strong>{value}</strong></div>
                )) : <p>No values in the selected event.</p>}
              </div>
            </details>
          );
        })}
      </div>
    </aside>
  );
}

function viewBoxFor(points: CorrectionPoint[]): ViewBox {
  if (!points.length) return { x: 0, y: 0, width: 10, height: 6 };
  const xs = points.map((point) => point.x_m);
  const ys = points.map((point) => point.y_m);
  const minX = Math.min(0, ...xs) - 0.5;
  const minY = Math.min(0, ...ys) - 0.5;
  const maxX = Math.max(10, ...xs) + 0.5;
  const maxY = Math.max(6, ...ys) + 0.5;
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

function CorrectionToolbar({
  data,
  points,
  setPoints,
  kind,
  setKind,
}: {
  data: ReturnType<typeof useOperatorData>;
  points: CorrectionPoint[];
  setPoints: (points: CorrectionPoint[]) => void;
  kind: "route" | "keep_out";
  setKind: (kind: "route" | "keep_out") => void;
}) {
  const [failureId, setFailureId] = useState(data.detail?.failure_ids[0] || "");
  const enoughPoints = points.length >= (kind === "route" ? 2 : 3);
  const canSubmit = Boolean(data.token && failureId && enoughPoints && !data.mutationBusy);
  return (
    <div className="ops-correction-toolbar">
      <div className="ops-segmented" aria-label="Correction geometry">
        <button type="button" className={kind === "route" ? "is-active" : ""} onClick={() => setKind("route")}>
          <Route size={14} /> Route
        </button>
        <button type="button" className={kind === "keep_out" ? "is-active" : ""} onClick={() => setKind("keep_out")}>
          <Square size={14} /> Keep-out
        </button>
      </div>
      <label className="ops-failure-select">
        <span>Failure</span>
        <select value={failureId} onChange={(event) => setFailureId(event.target.value)} disabled={!data.detail?.failure_ids.length}>
          {!data.detail?.failure_ids.length && <option value="">None attached</option>}
          {data.detail?.failure_ids.map((id) => <option key={id} value={id}>{id}</option>)}
        </select>
      </label>
      <span className="ops-correction-count">{points.length} points</span>
      <button
        type="button"
        className="ops-icon-button"
        onClick={() => setPoints(points.slice(0, -1))}
        disabled={!points.length}
        title="Undo point"
        aria-label="Undo correction point"
      ><Undo2 size={15} /></button>
      <button
        type="button"
        className="ops-icon-button"
        onClick={() => setPoints([])}
        disabled={!points.length}
        title="Clear correction"
        aria-label="Clear correction"
      ><RotateCcw size={15} /></button>
      <button
        type="button"
        className="ops-command-button"
        disabled={!canSubmit}
        onClick={() => void data.submitCorrection(failureId, kind, points)}
        title={
          !failureId ? "No deterministic failure is attached to this episode" :
          !data.token ? "Operator credential required" :
          !enoughPoints ? "More points required" : "Submit blocking human correction"
        }
      >
        <Check size={15} /> Submit correction
      </button>
    </div>
  );
}

function SimulationView({
  data,
  record,
  pathRecords,
}: {
  data: ReturnType<typeof useOperatorData>;
  record: TelemetryRecord | null;
  pathRecords: TelemetryRecord[];
}) {
  const [points, setPoints] = useState<CorrectionPoint[]>([]);
  const [kind, setKind] = useState<"route" | "keep_out">("route");
  const svgRef = useRef<SVGSVGElement>(null);
  const positions = useMemo(
    () => pathRecords.map(extractPosition).filter((point): point is CorrectionPoint => point !== null),
    [pathRecords],
  );
  const viewBox = useMemo(() => viewBoxFor([...positions, ...points]), [positions, points]);
  const latestPosition = extractPosition(record || pathRecords.at(-1));

  const addPoint = (event: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const bounds = svg.getBoundingClientRect();
    const x = viewBox.x + ((event.clientX - bounds.left) / bounds.width) * viewBox.width;
    const y = viewBox.y + ((event.clientY - bounds.top) / bounds.height) * viewBox.height;
    setPoints([...points, { x_m: Number(x.toFixed(3)), y_m: Number(y.toFixed(3)) }]);
  };

  const path = positions.map((point) => `${point.x_m},${point.y_m}`).join(" ");
  const correctionPath = points.map((point) => `${point.x_m},${point.y_m}`).join(" ");
  const video = mediaUrl(record, "third_person");

  return (
    <section className="ops-simulation" aria-labelledby="simulation-title">
      <div className="ops-section-heading ops-section-heading--overlay">
        <div><Crosshair size={15} /><span id="simulation-title">Third-person simulation</span></div>
        <strong>{record ? `frame ${shortId(record.frame_id, 18)}` : "No frame joined"}</strong>
      </div>
      <div className="ops-simulation__stage">
        <RealisticHomeScene
          robotPosition={latestPosition}
          robotYaw={extractYaw(record)}
          path={positions}
          correction={points}
          correctionKind={kind}
          running={data.detail?.episode.state === "running"}
        />
        {video && (
          <figure className="ops-stage-live-feed">
            <img src={video} alt="Live third-person simulator camera" />
            <figcaption><Radio size={12} /> Simulator camera · {shortId(record?.frame_id, 14)}</figcaption>
          </figure>
        )}
        <svg
          ref={svgRef}
          className="ops-world-plot ops-world-plot--inset"
          viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
          preserveAspectRatio="xMidYMid meet"
          onPointerDown={addPoint}
          aria-label="Telemetry trajectory and correction plane"
        >
          <defs>
            <pattern id="ops-grid" width="1" height="1" patternUnits="userSpaceOnUse">
              <path d="M 1 0 L 0 0 0 1" fill="none" stroke="currentColor" strokeWidth="0.012" />
            </pattern>
          </defs>
          <rect x={viewBox.x} y={viewBox.y} width={viewBox.width} height={viewBox.height} fill="url(#ops-grid)" />
          {path && <polyline className="ops-actual-path" points={path} fill="none" />}
          {correctionPath && (
            kind === "keep_out" && points.length > 2
              ? <polygon className="ops-correction-shape" points={correctionPath} />
              : <polyline className="ops-correction-line" points={correctionPath} fill="none" />
          )}
          {points.map((point, index) => (
            <g key={`${point.x_m}:${point.y_m}:${index}`}>
              <circle className="ops-correction-point" cx={point.x_m} cy={point.y_m} r="0.11" />
              <text x={point.x_m + 0.14} y={point.y_m - 0.14}>{index + 1}</text>
            </g>
          ))}
          {latestPosition && (
            <g className="ops-robot-position" transform={`translate(${latestPosition.x_m} ${latestPosition.y_m})`}>
              <circle r="0.22" />
              <path d="M 0 -0.34 L 0.18 0.08 L 0 0 L -0.18 0.08 Z" />
            </g>
          )}
        </svg>
        {!positions.length && (
          <div className="ops-spatial-status">
            <ScanLine size={15} />
            <span>Waiting for simulator ground-truth pose</span>
          </div>
        )}
        <div className="ops-stage-readout">
          <span><i className="ops-key ops-key--path" /> Telemetry path</span>
          <span><i className="ops-key ops-key--correction" /> Human correction</span>
          <span>X/Y in meters</span>
        </div>
        <div className="ops-stage-clock">
          <small>SIM TIME</small>
          <strong>{formatTime(record?.sim_time_seconds)}</strong>
          <span>{record ? `SEQ ${record.sequence}` : "NO EVENT"}</span>
        </div>
      </div>
      <CorrectionToolbar
        key={data.detail?.failure_ids.join("|") || "no-failure"}
        data={data}
        points={points}
        setPoints={setPoints}
        kind={kind}
        setKind={setKind}
      />
      {data.correction && (
        <div className="ops-inline-success">
          <CheckCircle2 size={15} /> {data.correction.correction_id} · {data.correction.state} human gate
        </div>
      )}
    </section>
  );
}

function DepthFeed({ sectors }: { sectors: number[] }) {
  if (!sectors.length) return null;
  const max = Math.max(...sectors, 0.01);
  return (
    <div className="ops-depth-bars" aria-label={`${sectors.length} derived depth sectors`}>
      {sectors.map((sector, index) => (
        <span
          key={`${sector}:${index}`}
          style={{ height: `${Math.max(8, Math.min(100, (sector / max) * 100))}%` }}
          title={`${sector.toFixed(2)} m`}
        />
      ))}
    </div>
  );
}

function VideoFeed({ feedKey, label, record, featured }: {
  feedKey: string;
  label: string;
  record: TelemetryRecord | null;
  featured: boolean;
}) {
  const url = mediaUrl(record, feedKey);
  const sectors = feedKey === "derived_depth" ? depthSectors(record) : [];
  const available = Boolean(url || sectors.length);
  return (
    <figure className={`ops-video-feed ${featured ? "ops-video-feed--featured" : ""}`}>
      <div className="ops-video-feed__surface">
        {url && <img src={url} alt={`${label} direct simulator feed`} />}
        {!url && sectors.length > 0 && <DepthFeed sectors={sectors} />}
        {!available && <div className="ops-video-unavailable"><EyeOff size={17} /><span>Unavailable</span></div>}
        <span className="ops-feed-scan" aria-hidden="true" />
      </div>
      <figcaption>
        <span>{label}</span>
        <small>{record?.frame_id ? shortId(record.frame_id, 15) : "No frame ID"}</small>
      </figcaption>
    </figure>
  );
}

function RobotPov({ record }: { record: TelemetryRecord | null }) {
  const speed = metricFromSensors(record, ["forward_speed_mps", "speed_mps", "forward_speed"]);
  const tilt = metricFromSensors(record, ["tray_tilt_degrees", "current_tray_tilt_degrees"]);
  const clearance = metricFromSensors(record, ["current_obstacle_clearance_m", "obstacle_clearance_m"]);
  const collision = Boolean(record?.failure_type) || booleanFromRecord(record, ["collision", "body_collision"]) === true;
  const direction = stringValue(record?.payload, ["destination_direction", "goal_direction"]);
  return (
    <section className="ops-pov" aria-labelledby="pov-title">
      <div className="ops-section-heading">
        <div><Bot size={15} /><span id="pov-title">Robot POV</span></div>
        <strong>{record?.frame_id ? "Joined by frame_id" : "Awaiting frame"}</strong>
      </div>
      <div className="ops-pov-grid">
        {VIDEO_FEEDS.map(([key, label], index) => (
          <VideoFeed key={key} feedKey={key} label={label} record={record} featured={index === 2} />
        ))}
      </div>
      <div className="ops-hud" aria-label="Robot state">
        <div><span>Action</span><strong>{actionLabel(record)}</strong></div>
        <div><span>Destination</span><strong>{direction || "Unavailable"}</strong></div>
        <div><span>Clearance</span><strong>{clearance === null ? "Unavailable" : `${clearance.toFixed(2)} m`}</strong></div>
        <div><span>Speed</span><strong>{speed === null ? "Unavailable" : `${speed.toFixed(2)} m/s`}</strong></div>
        <div><span>Tray tilt</span><strong>{tilt === null ? "Unavailable" : `${tilt.toFixed(1)}°`}</strong></div>
        <div className={collision ? "ops-hud-alert" : ""}><span>Collision</span><strong>{collision ? "Detected" : record ? "None reported" : "Unavailable"}</strong></div>
      </div>
    </section>
  );
}

function HumanGates({ data }: { data: ReturnType<typeof useOperatorData> }) {
  return (
    <section className="ops-gates" aria-labelledby="gates-title">
      <div className="ops-section-heading">
        <div><ShieldCheck size={15} /><span id="gates-title">Human gates</span></div>
        <strong>{data.approvals.length} blocking</strong>
      </div>
      <div className="ops-gate-list">
        {!data.approvals.length && (
          <div className="ops-empty-row"><CheckCircle2 size={16} /><span>No pending decisions</span></div>
        )}
        {data.approvals.map((approval) => (
          <article className="ops-gate" key={approval.requirement_id}>
            <div>
              <span className="ops-gate-kind">{stateLabel(approval.kind)}</span>
              <strong>{approval.summary}</strong>
              <small>{shortId(approval.requirement_id, 18)} · {new Date(approval.created_at).toLocaleString()}</small>
            </div>
            <div className="ops-gate-actions">
              <button
                type="button"
                className="ops-reject-button"
                disabled={!data.token || data.mutationBusy === approval.requirement_id}
                onClick={() => void data.decideApproval(approval.requirement_id, "reject")}
                title={data.token ? "Reject this proposal" : "Operator credential required"}
              ><X size={14} /> Reject</button>
              <button
                type="button"
                className="ops-approve-button"
                disabled={!data.token || data.mutationBusy === approval.requirement_id}
                onClick={() => void data.decideApproval(approval.requirement_id, "approve")}
                title={data.token ? "Approve this proposal" : "Operator credential required"}
              ><Check size={14} /> Approve</button>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function MetricCell({ label, baseline, candidate, format }: {
  label: string;
  baseline: number | null;
  candidate: number | null;
  format: (value: number) => string;
}) {
  return (
    <div className="ops-metric-cell">
      <span>{label}</span>
      <strong>{baseline === null ? "—" : format(baseline)}</strong>
      <strong>{candidate === null ? "—" : format(candidate)}</strong>
    </div>
  );
}

function metric(metrics: PolicyMetrics | null | undefined, key: keyof PolicyMetrics): number | null {
  return metrics ? metrics[key] : null;
}

function PolicyEvidence({ data }: { data: ReturnType<typeof useOperatorData> }) {
  const baseline = data.policies.find((policy) => policy.policy_id === data.baselinePolicyId);
  const candidate = data.policies.find((policy) => policy.policy_id === data.candidatePolicyId);
  const heldOut = data.policies.filter((policy) => policy.evaluation_scope === "held_out_aggregate");
  const checks = Object.entries(data.eligibility?.checks || {});
  return (
    <section className="ops-policy" aria-labelledby="policy-title">
      <div className="ops-section-heading">
        <div><Network size={15} /><span id="policy-title">Held-out policy evidence</span></div>
        <strong>{heldOut.length ? `${heldOut.length} evaluated` : "Evidence unavailable"}</strong>
      </div>
      <div className="ops-policy-selectors">
        <label><span>Baseline V0</span><select value={data.baselinePolicyId} onChange={(event) => data.setBaselinePolicyId(event.target.value)}>
          <option value="">Select baseline</option>
          {heldOut.map((policy) => <option key={policy.policy_id} value={policy.policy_id}>{policy.policy_id} · held-out aggregate</option>)}
        </select></label>
        <label><span>Candidate V1</span><select value={data.candidatePolicyId} onChange={(event) => data.setCandidatePolicyId(event.target.value)}>
          <option value="">Select candidate</option>
          {heldOut.map((policy) => <option key={policy.policy_id} value={policy.policy_id}>{policy.policy_id} · held-out aggregate</option>)}
        </select></label>
      </div>
      <div className="ops-metric-head"><span>Metric</span><strong>V0</strong><strong>V1</strong></div>
      <div className="ops-metric-grid">
        <MetricCell label="Episodes" baseline={metric(baseline?.metrics, "episode_count")} candidate={metric(candidate?.metrics, "episode_count")} format={(v) => String(v)} />
        <MetricCell label="Success" baseline={metric(baseline?.metrics, "success_rate")} candidate={metric(candidate?.metrics, "success_rate")} format={(v) => `${(v * 100).toFixed(0)}%`} />
        <MetricCell label="Collisions" baseline={metric(baseline?.metrics, "collision_rate")} candidate={metric(candidate?.metrics, "collision_rate")} format={(v) => `${(v * 100).toFixed(0)}%`} />
        <MetricCell label="Falls" baseline={metric(baseline?.metrics, "falls")} candidate={metric(candidate?.metrics, "falls")} format={(v) => String(v)} />
        <MetricCell label="Median clearance" baseline={metric(baseline?.metrics, "median_clearance_m")} candidate={metric(candidate?.metrics, "median_clearance_m")} format={(v) => `${v.toFixed(2)} m`} />
        <MetricCell label="Path efficiency" baseline={metric(baseline?.metrics, "median_path_efficiency")} candidate={metric(candidate?.metrics, "median_path_efficiency")} format={(v) => v.toFixed(2)} />
      </div>
      <div className="ops-promotion-state">
        <div className={data.eligibility?.numerically_eligible ? "is-eligible" : "is-blocked"}>
          {data.eligibility?.numerically_eligible ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
          <span>
            <strong>{data.eligibility?.promotion_applied ? "Promotion applied" : data.eligibility?.numerically_eligible ? "Numerically eligible · human approval required" : "Promotion blocked"}</strong>
            <small>{data.eligibility ? `${data.eligibility.held_out_episode_count} paired held-out episodes` : "No exact paired eligibility record"}</small>
          </span>
        </div>
        {checks.length > 0 && <div className="ops-check-list">
          {checks.map(([label, passed]) => <span key={label} className={passed ? "passed" : "failed"}>{passed ? <Check size={12} /> : <X size={12} />}{stateLabel(label)}</span>)}
        </div>}
      </div>
    </section>
  );
}

function extractProgress(record: TelemetryRecord | null): number | null {
  const value = numberValue(record?.payload, ["completion_progress", "progress", "reward_progress"]);
  if (value === null) return null;
  return Math.max(0, Math.min(1, value > 1 ? value / 100 : value));
}

function Timeline({
  data,
  activeIndex,
  setActiveIndex,
  playing,
  setPlaying,
}: {
  data: ReturnType<typeof useOperatorData>;
  activeIndex: number;
  setActiveIndex: (index: number) => void;
  playing: boolean;
  setPlaying: (playing: boolean) => void;
}) {
  const records = data.records;
  const current = records[activeIndex] || records.at(-1) || null;
  const progress = extractProgress(current);
  const reward = numberValue(current?.payload, ["reward", "total_reward"]);
  const lesson = stringValue(current?.payload, ["retrieved_lesson", "falkordb_lesson", "lesson"]);
  const workflow = stringValue(current?.payload, ["rocketride_workflow_step", "workflow_step"]);
  const laser = data.providers.find((provider) => provider.provider.toLowerCase().includes("laser"));
  return (
    <section className="ops-timeline" aria-labelledby="timeline-title">
      <div className="ops-timeline__head">
        <div className="ops-section-heading">
          <div><Zap size={15} /><span id="timeline-title">LaserData timeline</span></div>
          <strong>{records.length} immutable events · 20 Hz</strong>
        </div>
        <span className="ops-laser-state"><StatusDot state={laser?.state || "unconfigured"} /> {stateLabel(laser?.state || "unconfigured")}</span>
      </div>
      <div className="ops-timeline__track">
        <div className="ops-track-line" />
        {records.map((record, index) => (
          <button
            type="button"
            key={record.sequence}
            className={`ops-event-marker ${record.failure_type ? "is-failure" : ""} ${index === activeIndex ? "is-active" : ""}`}
            style={{ left: `${records.length <= 1 ? 0 : (index / (records.length - 1)) * 100}%` }}
            onClick={() => setActiveIndex(index)}
            title={`Sequence ${record.sequence} · ${formatTime(record.sim_time_seconds)}${record.failure_type ? ` · ${record.failure_type}` : ""}`}
            aria-label={`Replay event ${record.sequence}`}
          />
        ))}
      </div>
      <div className="ops-replay-controls">
        <button type="button" className="ops-icon-button" onClick={() => setActiveIndex(Math.max(0, activeIndex - 1))} disabled={!records.length || activeIndex <= 0} title="Previous event" aria-label="Previous replay event"><ChevronLeft size={16} /></button>
        <button type="button" className="ops-play-button" onClick={() => setPlaying(!playing)} disabled={records.length < 2} title={playing ? "Pause replay" : "Play replay"} aria-label={playing ? "Pause replay" : "Play replay"}>{playing ? <Pause size={15} /> : <Play size={15} />}</button>
        <button type="button" className="ops-icon-button" onClick={() => setActiveIndex(Math.min(records.length - 1, activeIndex + 1))} disabled={!records.length || activeIndex >= records.length - 1} title="Next event" aria-label="Next replay event"><ChevronRight size={16} /></button>
        <input
          type="range"
          min={0}
          max={Math.max(0, records.length - 1)}
          value={Math.max(0, Math.min(activeIndex, records.length - 1))}
          onChange={(event) => setActiveIndex(Number(event.target.value))}
          disabled={!records.length}
          aria-label="Replay event position"
        />
        <strong>{current ? formatTime(current.sim_time_seconds) : "--:--.---"}</strong>
        <span>{current ? `SEQ ${current.sequence}` : "NO EVENT"}</span>
      </div>
      <div className="ops-timeline-readouts">
        <div><span>Reward</span><strong>{reward === null ? "Unavailable" : reward.toFixed(3)}</strong></div>
        <div className="ops-progress-readout"><span>Completion</span><strong>{progress === null ? "Unavailable" : `${(progress * 100).toFixed(0)}%`}</strong><i>{progress !== null && <b style={{ width: `${progress * 100}%` }} />}</i></div>
        <div><span>Failure</span><strong>{current?.failure_type ? stateLabel(current.failure_type) : current ? "None reported" : "Unavailable"}</strong></div>
        <div><span>FalkorDB lesson</span><strong>{lesson || "No lesson attached"}</strong></div>
        <div><span>RocketRide step</span><strong>{workflow ? stateLabel(workflow) : "No workflow attached"}</strong></div>
      </div>
    </section>
  );
}

export function OperatorConsole() {
  const data = useOperatorData();
  const [selection, setSelection] = useState({ episodeId: "", index: 0 });
  const [playing, setPlaying] = useState(false);
  const activeIndex = selection.episodeId === data.selectedEpisodeId
    ? Math.max(0, Math.min(selection.index, data.records.length - 1))
    : Math.max(0, data.records.length - 1);
  const setActiveIndex = useCallback((index: number) => {
    setSelection({ episodeId: data.selectedEpisodeId, index });
  }, [data.selectedEpisodeId]);

  useEffect(() => {
    if (!playing || data.records.length < 2) return;
    const timer = window.setTimeout(() => {
      if (activeIndex >= data.records.length - 1) setPlaying(false);
      else setActiveIndex(activeIndex + 1);
    }, 50);
    return () => window.clearTimeout(timer);
  }, [playing, data.records.length, activeIndex, setActiveIndex]);

  const selectedRecord = data.records[activeIndex] || data.latestRecord;
  const pathRecords = data.records.slice(0, Math.max(0, activeIndex) + 1);

  return (
    <main className="operator-app">
      <TopBar data={data} />
      <EpisodePicker data={data} />

      {data.issue && (
        <div className="ops-system-message ops-system-message--warning" role="status">
          <AlertTriangle size={15} /><span>{data.issue}</span>
          <button type="button" onClick={() => void data.refresh()}><RefreshCw size={14} /> Retry</button>
        </div>
      )}
      {data.mutationIssue && (
        <div className="ops-system-message ops-system-message--error" role="alert">
          <XCircle size={15} /><span>{data.mutationIssue}</span>
        </div>
      )}

      {!data.selectedEpisodeId && (
        <div className="ops-system-message" role="status">
          {data.loading ? <RefreshCw className="is-spinning" size={15} /> : <Radio size={15} />}
          <span>{data.loading ? "Connecting to the operations API" : "No operational episode is available; live surfaces remain explicitly unavailable."}</span>
        </div>
      )}

      <div className="ops-workspace">
        <SensorRail record={selectedRecord} />
        <div className="ops-primary-column">
          <SimulationView key={data.selectedEpisodeId || "no-episode"} data={data} record={selectedRecord} pathRecords={pathRecords} />
          <PolicyEvidence data={data} />
        </div>
        <div className="ops-right-column">
          <RobotPov record={selectedRecord} />
          <HumanGates data={data} />
        </div>
      </div>
      <Timeline
        data={data}
        activeIndex={activeIndex}
        setActiveIndex={setActiveIndex}
        playing={playing}
        setPlaying={setPlaying}
      />
    </main>
  );
}
