import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Bot,
  Box,
  Check,
  ChevronDown,
  Circle,
  Cloud,
  Database,
  Eye,
  EyeOff,
  GitBranch,
  GripVertical,
  KeyRound,
  Lock,
  Maximize2,
  Network,
  Play,
  Radio,
  RefreshCw,
  Route,
  Search,
  Settings,
  ShieldCheck,
  Square,
  Waypoints,
  Workflow,
  X,
} from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import { BrandMark } from "../components/BrandMark";
import { RealisticHomeScene } from "../operator/RealisticHomeScene";
import type {
  CorrectionPoint,
  MemoryGraphNode as MemoryGraphNodeData,
  ProviderHealth,
  SensorReading,
  SignalUse,
  TelemetryRecord,
  VideoProduct,
} from "../operator/types";
import { useOperatorData, type OperatorData } from "../operator/useOperatorData";
import "./CommandCenter.css";

type WorkspaceView = "operations" | "memory" | "rocketride" | "settings";
type ManualCommand = "hold" | "forward" | "reverse" | "left" | "right";

const VIEW_LABELS: Record<WorkspaceView, string> = {
  operations: "Operations",
  memory: "Memory Graph",
  rocketride: "RocketRide",
  settings: "System Settings",
};

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

const VIDEO_FEEDS: Array<{ id: VideoProduct; label: string }> = [
  { id: "left_eye_rgb", label: "Left RGB" },
  { id: "right_eye_rgb", label: "Right RGB" },
  { id: "stereo_composite", label: "Stereo" },
  { id: "derived_depth", label: "Depth" },
  { id: "simulator_debug_segmentation", label: "Debug seg" },
];

const PIPELINE_STEPS = [
  { label: "Validate world", owner: "World & Physics", gate: false },
  { label: "Run episode", owner: "RocketRide", gate: false },
  { label: "Summarize telemetry", owner: "RocketRide", gate: false },
  { label: "Query graph memory", owner: "RocketRide", gate: false },
  { label: "Select curriculum", owner: "Failure & Curriculum", gate: true },
  { label: "Train candidate", owner: "RocketRide", gate: false },
  { label: "Evaluate candidate", owner: "Safety & Evaluation", gate: false },
  { label: "Promote or roll back", owner: "Safety & Evaluation", gate: true },
] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nestedValue(source: unknown, paths: string[]): unknown {
  for (const path of paths) {
    let value: unknown = source;
    for (const segment of path.split(".")) {
      if (!isRecord(value)) {
        value = undefined;
        break;
      }
      value = value[segment];
    }
    if (value !== undefined && value !== null) return value;
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

function safeMediaUrl(value: unknown): string | null {
  if (typeof value !== "string" || value === "video_stream") return null;
  return /^(https?:\/\/|\/)/.test(value) ? value : null;
}

function mediaUrl(record: TelemetryRecord | null, product: string): string | null {
  if (!record) return null;
  const frames = record.payload.video_frames;
  if (Array.isArray(frames)) {
    const frame = frames.find(
      (item) => isRecord(item) && item.frame_id === record.frame_id,
    ) ?? frames.at(-1);
    if (isRecord(frame) && Array.isArray(frame.products)) {
      const match = frame.products.find(
        (item) => isRecord(item) && item.product === product,
      );
      if (isRecord(match)) {
        const direct = safeMediaUrl(match.stream_url) ?? safeMediaUrl(match.frame_url);
        if (direct) return direct;
      }
    }
  }
  const stereo = record.sensors.find((sensor) => sensor.category === "Stereo vision and depth");
  return safeMediaUrl(
    nestedValue(stereo?.values, [`${product}.stream_url`, `${product}.url`, product]) ??
      nestedValue(record.payload, [
        `video_metadata.${product}.stream_url`,
        `video_metadata.${product}.url`,
        `video_metadata.${product}`,
      ]),
  );
}

function depthSectors(record: TelemetryRecord | null): number[] {
  if (!record) return [];
  const stereo = record.sensors.find((sensor) => sensor.category === "Stereo vision and depth");
  const raw = nestedValue(stereo?.values, ["derived_depth_sectors", "derived_depth_sectors_m"]);
  return Array.isArray(raw)
    ? raw.filter((value): value is number => typeof value === "number" && Number.isFinite(value))
    : [];
}

function extractPosition(record: TelemetryRecord | null): CorrectionPoint | null {
  if (!record) return null;
  const x = numberValue(record.payload, [
    "position_x_m",
    "simulator_pose.position_x_m",
    "robot_position.x_m",
    "pose.x_m",
  ]);
  const y = numberValue(record.payload, [
    "position_y_m",
    "simulator_pose.position_y_m",
    "robot_position.y_m",
    "pose.y_m",
  ]);
  return x === null || y === null ? null : { x_m: x, y_m: y };
}

function extractYaw(record: TelemetryRecord | null): number | null {
  if (!record) return null;
  return numberValue(record.payload, [
    "yaw_rad",
    "simulator_pose.yaw_rad",
    "robot_yaw_rad",
    "pose.yaw_rad",
  ]);
}

function providerFor(data: OperatorData, name: string): ProviderHealth | null {
  return data.providers.find((provider) => provider.provider.toLowerCase() === name.toLowerCase()) ?? null;
}

function readableState(value: string | null | undefined): string {
  return value ? value.replaceAll("_", " ") : "Unavailable";
}

function shortId(value: string | null | undefined, length = 13): string {
  if (!value) return "Unavailable";
  return value.length <= length ? value : `${value.slice(0, length)}…`;
}

function compactSensorValue(sensor: SensorReading | undefined): string {
  if (!sensor?.available) return "Unavailable";
  const value = sensor.values;
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === "boolean") return value ? "Active" : "Inactive";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return `${value.length} samples`;
  if (isRecord(value)) {
    const entry = Object.entries(value).find(([, item]) =>
      typeof item === "number" || typeof item === "boolean" || typeof item === "string",
    );
    if (entry) {
      const [key, item] = entry;
      const readable = typeof item === "number" && !Number.isInteger(item) ? item.toFixed(2) : String(item);
      return `${key.replaceAll("_", " ")} ${readable}`;
    }
    return "Structured telemetry";
  }
  return "Available";
}

function currentPipelineIndex(record: TelemetryRecord | null): number {
  const raw = stringValue(record?.payload, ["rocketride_step", "workflow_step", "pipeline_step"]);
  if (!raw) return -1;
  const normalized = raw.toLowerCase().replaceAll("_", " ").replaceAll("-", " ");
  return PIPELINE_STEPS.findIndex(({ label }) => normalized.includes(label.toLowerCase()));
}

function Panel({
  title,
  icon,
  meta,
  className = "",
  children,
}: {
  title: string;
  icon: ReactNode;
  meta?: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`cc-panel ${className}`}>
      <header className="cc-panel__header">
        <div className="cc-panel__title"><GripVertical size={14} /><span>{icon}</span><strong>{title}</strong></div>
        <div className="cc-panel__tools">{meta && <span>{meta}</span>}<Maximize2 size={13} /></div>
      </header>
      <div className="cc-panel__body">{children}</div>
    </section>
  );
}

function EmptyState({ icon, title, detail }: { icon: ReactNode; title: string; detail: string }) {
  return (
    <div className="cc-empty">
      <span>{icon}</span>
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  );
}

function Header({
  view,
  setView,
  menuOpen,
  setMenuOpen,
}: {
  view: WorkspaceView;
  setView: (view: WorkspaceView) => void;
  menuOpen: boolean;
  setMenuOpen: (open: boolean) => void;
}) {
  return (
    <header className="cc-header">
      <button className="cc-brand" type="button" onClick={() => setView("operations")}>
        <BrandMark />
        <span><strong>Muscle Memory</strong><small>{VIEW_LABELS[view]}</small></span>
      </button>
      <div className="cc-menu-wrap">
        <button
          className={`cc-settings-button ${menuOpen ? "is-open" : ""}`}
          type="button"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-expanded={menuOpen}
          aria-label="Open workspace settings"
        >
          {menuOpen ? <X size={19} /> : <Settings size={19} />}
        </button>
        {menuOpen && (
          <nav className="cc-view-menu" aria-label="Workspace pages">
            {(Object.keys(VIEW_LABELS) as WorkspaceView[]).map((item) => (
              <button
                type="button"
                key={item}
                className={view === item ? "is-active" : ""}
                onClick={() => { setView(item); setMenuOpen(false); }}
              >
                {item === "operations" && <Bot size={17} />}
                {item === "memory" && <Network size={17} />}
                {item === "rocketride" && <Workflow size={17} />}
                {item === "settings" && <Settings size={17} />}
                <span>{VIEW_LABELS[item]}</span>
                {view === item && <i />}
              </button>
            ))}
          </nav>
        )}
      </div>
    </header>
  );
}

function DirectionPad({
  enabled,
  command,
  onCommand,
  onToggle,
  canEnable,
}: {
  enabled: boolean;
  command: ManualCommand;
  onCommand: (command: ManualCommand) => void;
  onToggle: () => void;
  canEnable: boolean;
}) {
  return (
    <div className={`cc-intervention ${enabled ? "is-enabled" : ""}`}>
      <div className="cc-intervention__heading">
        <span>Human intervention</span>
        <button
          type="button"
          role="switch"
          aria-checked={enabled}
          onClick={onToggle}
          aria-label={enabled ? "Hide human intervention joystick" : "Show human intervention joystick"}
          title={enabled ? "Hide joystick" : "Show joystick"}
        ><i /></button>
      </div>
      <span className="cc-intervention__state">
        {enabled ? (canEnable ? `Local intent: ${command} · not sent` : "Intent preview unavailable") : "Intent preview hidden"}
      </span>
      {enabled && (
        <div className="cc-dpad" aria-label="High-level movement intent preview">
          <button type="button" disabled={!canEnable} onClick={() => onCommand("forward")} aria-label="Forward"><ArrowUp /></button>
          <button type="button" disabled={!canEnable} onClick={() => onCommand("left")} aria-label="Turn left"><ArrowLeft /></button>
          <button className="cc-dpad__hold" type="button" disabled={!canEnable} onClick={() => onCommand("hold")} aria-label="Stop"><Square /></button>
          <button type="button" disabled={!canEnable} onClick={() => onCommand("right")} aria-label="Turn right"><ArrowRight /></button>
          <button type="button" disabled={!canEnable} onClick={() => onCommand("reverse")} aria-label="Reverse"><ArrowDown /></button>
        </div>
      )}
    </div>
  );
}

function RobotVideo({ record }: { record: TelemetryRecord | null }) {
  const [feed, setFeed] = useState<VideoProduct>("stereo_composite");
  const url = mediaUrl(record, feed);
  const action = record ? numberValue(record.payload, ["policy_action.forward_speed_mps"]) : null;
  const battery = record
    ? numberValue(record.sensors.find((sensor) => sensor.category === "Battery and energy")?.values, ["battery_percent"])
    : null;
  return (
    <div className="cc-pov-stack">
      <div className="cc-pov-tabs" role="tablist" aria-label="Robot POV products">
        {VIDEO_FEEDS.map((item) => (
          <button
            type="button"
            role="tab"
            aria-selected={feed === item.id}
            className={feed === item.id ? "is-active" : ""}
            key={item.id}
            onClick={() => setFeed(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="cc-pov-view">
        {url ? (
          <div className="cc-video"><img src={url} alt={`MM-01 ${feed.replaceAll("_", " ")} feed`} /><span>Joined by frame_id</span></div>
        ) : feed === "derived_depth" ? (
          <DepthView record={record} />
        ) : record?.payload.routine_mode === "local_synthetic" ? (
          <div className="cc-telemetry-readout">
            <small>Local synthetic telemetry</small>
            <strong>{action === null ? "0.00" : action.toFixed(2)} <span>m/s command</span></strong>
            <div><span>Frame</span><b>{shortId(record.frame_id, 15)}</b></div>
            <div><span>Battery</span><b>{battery === null ? "--" : `${battery.toFixed(1)}%`}</b></div>
          </div>
        ) : (
          <EmptyState icon={<EyeOff size={21} />} title={`${VIDEO_FEEDS.find((item) => item.id === feed)?.label} unavailable`} detail="Waiting for a frame joined by frame_id." />
        )}
      </div>
    </div>
  );
}

function DepthView({ record }: { record: TelemetryRecord | null }) {
  const url = mediaUrl(record, "derived_depth");
  const sectors = depthSectors(record);
  if (url) return <div className="cc-video"><img src={url} alt="Stereo-derived depth feed" /><span>Stereo-derived depth</span></div>;
  if (!sectors.length) return <EmptyState icon={<EyeOff size={21} />} title="Depth unavailable" detail="No stereo-derived sectors in this event." />;
  const maximum = Math.max(...sectors, 0.01);
  return (
    <div className="cc-depth" aria-label={`${sectors.length} stereo-derived depth sectors`}>
      {sectors.map((sector, index) => <i key={`${sector}:${index}`} style={{ height: `${Math.max(7, (sector / maximum) * 100)}%` }} />)}
    </div>
  );
}

function SensorStrip({ record }: { record: TelemetryRecord | null }) {
  return (
    <section className="cc-sensor-strip" aria-label="All eight MM-01 sensor categories">
      {SENSOR_CATEGORIES.map((category) => {
        const sensor = record?.sensors.find((item) => item.category === category);
        const signalUse = sensor?.signal_use ?? SENSOR_DEFAULT_USE[category];
        return (
          <div className={`cc-sensor-chip is-${signalUse.toLowerCase().replaceAll(" ", "-")}`} key={category}>
            <span>{category}</span>
            <strong>{compactSensorValue(sensor)}</strong>
            <small>{signalUse}</small>
          </div>
        );
      })}
    </section>
  );
}

function OperationsView({ data }: { data: OperatorData }) {
  const [intervention, setIntervention] = useState(false);
  const [manualCommand, setManualCommand] = useState<ManualCommand>("hold");
  const latest = data.latestRecord;
  const path = useMemo(
    () => data.records.map(extractPosition).filter((point): point is CorrectionPoint => point !== null),
    [data.records],
  );
  const running = Boolean(
    data.detail?.episode.state === "running" ||
      (data.liveStatus && ["queued", "starting", "running", "cancelling"].includes(data.liveStatus.phase)),
  );
  const canPreviewIntent = true;
  const thirdPersonVideo = mediaUrl(latest, "third_person");
  const failure = data.detail?.failure_ids[0] ?? latest?.failure_type;
  const curriculumGate = data.approvals.find((approval) => approval.kind === "curriculum_change");
  const policyGate = data.approvals.find((approval) => approval.kind === "policy_promotion" || approval.kind === "policy_rollback");
  const clearance = numberValue(latest?.payload, ["current_obstacle_clearance_m", "obstacle_clearance_m"]);
  const tilt = numberValue(latest?.payload, ["tray_tilt_degrees", "current_tray_tilt_degrees"]);
  const currentStep = currentPipelineIndex(latest);

  return (
    <div className="cc-operations">
      <section className="cc-world-panel">
        <header className="cc-world-panel__header">
          <div><Route size={15} /><strong>World</strong><span>Third-person view</span></div>
          <div className="cc-world-actions">
            <span><i className={`cc-dot cc-dot--${running ? "live" : "unconfigured"}`} />{running ? "Routine active" : data.isLocalRoutine ? "Local simulator" : "Visual staging"}</span>
            <button
              type="button"
              className="cc-start-routine"
              disabled={running || Boolean(data.mutationBusy) || !data.liveOptions?.enabled || (!data.isLocalRoutine && !data.token)}
              onClick={() => void data.startLiveEpisode()}
            ><Play size={13} /> Start routine</button>
          </div>
        </header>
        <div className="cc-world-stage">
          <RealisticHomeScene
            robotPosition={extractPosition(latest)}
            robotYaw={extractYaw(latest)}
            path={path}
            correction={[]}
            correctionKind="route"
            running={running}
          />
          {thirdPersonVideo && (
            <figure className="cc-direct-evidence">
              <img src={thirdPersonVideo} alt="Direct third-person MuJoCo evidence frame" />
              <figcaption><Radio size={11} /> Direct frame · {shortId(latest?.frame_id, 10)}</figcaption>
            </figure>
          )}
          <DirectionPad
            enabled={intervention}
            command={manualCommand}
            canEnable={canPreviewIntent}
            onCommand={setManualCommand}
            onToggle={() => { setIntervention(!intervention); setManualCommand("hold"); }}
          />
        </div>
      </section>

      <aside className="cc-adaptive-grid" aria-label="Adaptive agent and video windows">
        <Panel title="Robot POV" icon={<Eye size={14} />} meta={latest?.frame_id ? shortId(latest.frame_id) : "No frame"} className="cc-panel--media">
          <RobotVideo record={latest} />
        </Panel>
        <Panel title="Derived depth" icon={<Waypoints size={14} />} meta="Used by policy" className="cc-panel--media">
          <DepthView record={latest} />
        </Panel>
        <Panel title="World & Physics" icon={<Box size={14} />} meta={data.detail ? "Attached" : "Waiting"}>
          <div className="cc-agent-copy">
            <span>World validation</span>
            <strong>{data.detail ? data.detail.episode.world_id : "No validated world attached"}</strong>
            <p>{data.detail ? "Physics uses approved deterministic colliders." : "A world must pass every validation gate before use."}</p>
          </div>
        </Panel>
        <Panel title="Failure & Curriculum" icon={<AlertTriangle size={14} />} meta={curriculumGate ? "Human gate" : "Idle"}>
          <div className="cc-agent-copy">
            <span>Recent failure</span>
            <strong>{failure ? readableState(failure) : "No failure attached"}</strong>
            <p>{curriculumGate?.summary ?? "No curriculum proposal is waiting for approval."}</p>
          </div>
        </Panel>
        <Panel title="Safety & Evaluation" icon={<ShieldCheck size={14} />} meta={data.eligibility ? "Measured" : "Waiting"}>
          <div className="cc-check-list">
            <div><span>Clearance</span><strong>{clearance === null ? "Unavailable" : `${clearance.toFixed(2)} m`}</strong></div>
            <div><span>Tray tilt</span><strong>{tilt === null ? "Unavailable" : `${tilt.toFixed(1)}°`}</strong></div>
            <div><span>Promotion</span><strong>{policyGate ? "Approval required" : data.eligibility ? (data.eligibility.numerically_eligible ? "Eligible · gated" : "Blocked") : "No paired evidence"}</strong></div>
          </div>
        </Panel>
        <Panel title="Active workflow" icon={<Workflow size={14} />} meta={currentStep >= 0 ? "Active" : "No run"}>
          <div className="cc-mini-pipeline">
            {PIPELINE_STEPS.slice(0, 4).map((step, index) => (
              <div key={step.label} className={index === currentStep ? "is-active" : index < currentStep ? "is-complete" : ""}>
                <i>{index < currentStep ? <Check size={10} /> : index + 1}</i><span>{step.label}</span>
              </div>
            ))}
          </div>
        </Panel>
      </aside>
      <SensorStrip record={latest} />
    </div>
  );
}

function GraphCanvas({ data }: { data: OperatorData }) {
  const graph = data.memoryGraph;
  const [selected, setSelected] = useState("memory:explicit");
  const [query, setQuery] = useState("");
  const [kindFilter, setKindFilter] = useState<string | null>(null);
  const allNodes = useMemo(() => graph?.nodes ?? [], [graph]);
  const nodeById = useMemo(
    () => Object.fromEntries(allNodes.map((node) => [node.id, node])),
    [allNodes],
  );
  const active = nodeById[selected] ?? nodeById["memory:explicit"] ?? allNodes[0];
  const structuralKinds = new Set(["memory_provider", "runtime_agent", "fixed_robot"]);
  const normalizedQuery = query.trim().toLowerCase();
  const structuralNodes = allNodes.filter((node) => structuralKinds.has(node.record_kind));
  const matchingFacts = allNodes.filter((node) => {
    if (structuralKinds.has(node.record_kind)) return false;
    if (kindFilter && node.record_kind !== kindFilter) return false;
    if (!normalizedQuery) return true;
    return `${node.label} ${node.record_kind} ${node.owner} ${JSON.stringify(node.properties)}`
      .toLowerCase()
      .includes(normalizedQuery);
  });
  const visibleNodes = [...structuralNodes, ...matchingFacts.slice(-24)];
  const visibleIds = new Set(visibleNodes.map((node) => node.id));
  const visibleEdges = (graph?.edges ?? []).filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
  );
  const relationships = (graph?.edges ?? []).filter(
    (edge) => edge.source === active?.id || edge.target === active?.id,
  );
  const kinds = [...new Set(allNodes.map((node) => node.record_kind))]
    .filter((kind) => !structuralKinds.has(kind));
  const factsByOwner = new Map<string, MemoryGraphNodeData[]>();
  for (const node of visibleNodes) {
    if (structuralKinds.has(node.record_kind)) continue;
    const group = factsByOwner.get(node.owner) ?? [];
    group.push(node);
    factsByOwner.set(node.owner, group);
  }

  const tone = (node: MemoryGraphNodeData): string => {
    const byKind: Record<string, string> = {
      memory_provider: "mint",
      fixed_robot: "mint",
      world: "blue",
      obstacle: "violet",
      episode: "green",
      failure: "coral",
      correction: "amber",
      lesson: "mint",
      evaluated_policy: "blue",
      outperformance: "amber",
    };
    if (node.record_kind === "runtime_agent") {
      if (node.owner === "World & Physics Agent") return "blue";
      if (node.owner === "Failure & Curriculum Agent") return "violet";
      return "amber";
    }
    return byKind[node.record_kind] ?? "mint";
  };
  const nodeIcon = (node: MemoryGraphNodeData): ReactNode => {
    if (node.record_kind === "memory_provider") return <Database />;
    if (node.record_kind === "fixed_robot") return <Bot />;
    if (node.record_kind === "runtime_agent") return <Network />;
    if (node.record_kind === "evaluated_policy") return <ShieldCheck />;
    if (node.record_kind === "failure") return <AlertTriangle />;
    if (node.record_kind === "lesson") return <Activity />;
    return <Circle />;
  };
  const position = (node: MemoryGraphNodeData): { x: number; y: number } => {
    if (node.id === "memory:explicit") return { x: 50, y: 9 };
    if (node.id === "robot:mm-01") return { x: 50, y: 91 };
    if (node.id === "agent:world-physics") return { x: 18, y: 29 };
    if (node.id === "agent:failure-curriculum") return { x: 50, y: 29 };
    if (node.id === "agent:safety-evaluation") return { x: 82, y: 29 };
    const centers: Record<string, number> = {
      "World & Physics Agent": 18,
      "Failure & Curriculum Agent": 50,
      "Safety & Evaluation Agent": 82,
      system: 50,
    };
    const group = factsByOwner.get(node.owner) ?? [node];
    const index = Math.max(0, group.findIndex((item) => item.id === node.id));
    const columns = node.owner === "system" ? 2 : 3;
    const spacing = node.owner === "system" ? 10 : 9;
    return {
      x: centers[node.owner] + (index % columns - (columns - 1) / 2) * spacing,
      y: 53 + Math.floor(index / columns) * 17,
    };
  };
  const points = Object.fromEntries(visibleNodes.map((node) => [node.id, position(node)]));
  const nodeSummary = (node: MemoryGraphNodeData): string => {
    const count = node.properties.fact_count;
    if (typeof count === "number") return `${count} ${count === 1 ? "fact" : "facts"}`;
    if (node.record_kind === "fixed_robot") return "Fixed identity";
    return readableState(node.record_kind);
  };
  const displayValue = (value: unknown): string => {
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
    if (typeof value === "string") return value.length > 32 ? `${value.slice(0, 32)}…` : readableState(value);
    return JSON.stringify(value);
  };
  return (
    <div className="cc-graph-layout">
      <section className="cc-graph-surface">
        <div className="cc-view-heading">
          <div><h1>Live memory graph</h1><p>Operational FalkorDB facts across the three agent roles. Explicit memory never enters the control path.</p></div>
          <div className="cc-graph-heading-tools">
            <span className={`cc-provider-label is-${graph?.provider_state ?? "unconfigured"}`}><i />{graph ? `${graph.source === "falkordb" ? "FalkorDB" : "Local mirror"} · ${graph.fact_count} facts` : "Loading memory"}</span>
            <div className="cc-search"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} aria-label="Search live memory graph" placeholder="Search live facts" /></div>
          </div>
        </div>
        <div className="cc-node-filters" aria-label="Node families">
          <button type="button" className={kindFilter === null ? "is-active" : ""} onClick={() => setKindFilter(null)}>All facts <b>{graph?.fact_count ?? 0}</b></button>
          {kinds.map((kind) => (
            <button type="button" key={kind} className={kindFilter === kind ? "is-active" : ""} onClick={() => setKindFilter(kindFilter === kind ? null : kind)}>
              {readableState(kind)} <b>{allNodes.filter((node) => node.record_kind === kind).length}</b>
            </button>
          ))}
        </div>
        <div className="cc-graph-canvas">
          {!graph && <EmptyState icon={<Database size={24} />} title="Loading operational memory" detail="Reading the backend graph snapshot." />}
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            {visibleEdges.map((edge) => {
              const from = points[edge.source];
              const to = points[edge.target];
              return <line key={edge.id} x1={from.x} y1={from.y} x2={to.x} y2={to.y} />;
            })}
          </svg>
          {visibleEdges.map((edge) => {
            const from = points[edge.source];
            const to = points[edge.target];
            return <span key={`${edge.id}:label`} className="cc-edge-label" style={{ left: `${(from.x + to.x) / 2}%`, top: `${(from.y + to.y) / 2}%` }}>{edge.relationship}</span>;
          })}
          {visibleNodes.map((node) => {
            const point = points[node.id];
            return (
            <button
              type="button"
              key={node.id}
              className={`cc-graph-node is-${tone(node)} ${structuralKinds.has(node.record_kind) ? "is-structure" : "is-fact"} ${selected === node.id ? "is-selected" : ""}`}
              style={{ left: `${point.x}%`, top: `${point.y}%` }}
              onClick={() => setSelected(node.id)}
              aria-pressed={selected === node.id}
            >
              <i>{nodeIcon(node)}</i>
              <strong>{node.label.replace(" Agent", "")}</strong><span>{nodeSummary(node)}</span>
            </button>
          )})}
          {graph && graph.fact_count === 0 && <p className="cc-graph-empty-note">No operational facts yet. The live topology will populate after a validated episode closes.</p>}
        </div>
      </section>
      <aside className="cc-inspector">
        <Panel title="Selected live node" icon={<Circle size={14} />} meta={active ? readableState(active.record_kind) : "Loading"}>
          {active && <div className={`cc-selected-node is-${tone(active)}`}><i>{nodeIcon(active)}</i><div><strong>{active.label}</strong><span>{active.owner === "system" ? nodeSummary(active) : active.owner}</span></div></div>}
        </Panel>
        <Panel title="Node data" icon={<Database size={14} />} meta={graph ? `Refreshed ${new Date(graph.refreshed_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}` : "Loading"}>
          <div className="cc-inspector-list">
            {active && Object.entries(active.properties).slice(0, 9).map(([key, value]) => <div key={key}><span>{readableState(key)}</span><strong title={String(value)}>{displayValue(value)}</strong></div>)}
            {!active && <div><span>Status</span><strong>Loading snapshot</strong></div>}
          </div>
        </Panel>
        <Panel title="Live relationships" icon={<GitBranch size={14} />} meta={`${relationships.length} connected`}>
          <div className="cc-relationship-list">
            {active && relationships.map((edge) => (
              <div key={edge.id}><span>{edge.relationship}</span><strong>{nodeById[edge.source === active.id ? edge.target : edge.source]?.label ?? "Related node"}</strong></div>
            ))}
            {!relationships.length && <p>No relationships in the current operational snapshot.</p>}
          </div>
        </Panel>
      </aside>
    </div>
  );
}

function RocketRideView({ data, goToSettings }: { data: OperatorData; goToSettings: () => void }) {
  const currentIndex = currentPipelineIndex(data.latestRecord);
  const rocketride = providerFor(data, "rocketride.ai");
  const [selectedStep, setSelectedStep] = useState(Math.max(0, currentIndex));
  const selected = PIPELINE_STEPS[selectedStep];
  const relevantGates = data.approvals.filter((approval) =>
    approval.kind === "curriculum_change" || approval.kind === "policy_promotion" || approval.kind === "policy_rollback",
  );
  return (
    <div className="cc-pipeline-layout">
      <section className="cc-pipeline-surface">
        <div className="cc-view-heading">
          <div><h1>RocketRide</h1><p>RocketRide executes. Guild.ai decides.</p></div>
          <span className={`cc-provider-label is-${rocketride?.state ?? "unconfigured"}`}><i />{readableState(rocketride?.state)}</span>
        </div>
        <div className="cc-pipeline-list">
          {PIPELINE_STEPS.map((step, index) => {
            const status = currentIndex < 0 ? "waiting" : index < currentIndex ? "complete" : index === currentIndex ? "active" : "waiting";
            const gate = step.gate && relevantGates.length > 0;
            return (
              <button type="button" key={step.label} onClick={() => setSelectedStep(index)} className={`cc-pipeline-step is-${status} ${selectedStep === index ? "is-selected" : ""}`}>
                <i className="cc-pipeline-step__status">{status === "complete" ? <Check /> : index + 1}</i>
                <div><strong>{step.label}</strong><span>{step.owner}</span></div>
                <small>{gate ? <><Lock size={12} /> Human gate</> : status}</small>
                <ChevronDown size={15} />
              </button>
            );
          })}
        </div>
      </section>
      <aside className="cc-inspector">
        <Panel title="Current step" icon={<Activity size={14} />} meta={currentIndex < 0 ? "No active run" : `Step ${currentIndex + 1}`}>
          <div className="cc-current-step"><i>{selectedStep + 1}</i><div><strong>{selected.label}</strong><span>{selected.owner}</span></div></div>
          <p className="cc-muted-copy">Select a step to inspect its immutable place in the approved pipeline.</p>
        </Panel>
        <Panel title="Run evidence" icon={<GitBranch size={14} />} meta={data.detail ? "Attached" : "Unavailable"}>
          <div className="cc-inspector-list">
            <div><span>Episode</span><strong>{shortId(data.detail?.episode.episode_id)}</strong></div>
            <div><span>World</span><strong>{shortId(data.detail?.episode.world_id)}</strong></div>
            <div><span>Policy</span><strong>{shortId(data.detail?.episode.policy_id)}</strong></div>
            <div><span>Events</span><strong>{data.detail ? String(data.detail.telemetry_records) : "Unavailable"}</strong></div>
          </div>
        </Panel>
        <Panel title="Human gates" icon={<Lock size={14} />} meta={`${relevantGates.length} blocking`}>
          <div className="cc-gates">
            {relevantGates.map((gate) => (
              <div key={gate.requirement_id}><span>{readableState(gate.kind)}</span><strong>{gate.summary}</strong><button type="button" disabled={!data.token || Boolean(data.mutationBusy)} onClick={() => void data.decideApproval(gate.requirement_id, "approve")}>Approve</button></div>
            ))}
            {!relevantGates.length && <p>No curriculum or policy decision is waiting.</p>}
            {!data.token && <button type="button" className="cc-configure-link" onClick={goToSettings}><KeyRound size={13} /> Configure operator access</button>}
          </div>
        </Panel>
      </aside>
    </div>
  );
}

function Toggle({ checked, onChange, label, detail }: { checked: boolean; onChange: () => void; label: string; detail: string }) {
  return (
    <button type="button" className="cc-setting-toggle" role="switch" aria-checked={checked} onClick={onChange}>
      <span><strong>{label}</strong><small>{detail}</small></span><i><b /></i>
    </button>
  );
}

function SettingsView({
  data,
  showOverlays,
  setShowOverlays,
  compactWindows,
  setCompactWindows,
}: {
  data: OperatorData;
  showOverlays: boolean;
  setShowOverlays: (value: boolean) => void;
  compactWindows: boolean;
  setCompactWindows: (value: boolean) => void;
}) {
  const liveActive = Boolean(data.liveStatus && ["queued", "starting", "running", "cancelling"].includes(data.liveStatus.phase));
  return (
    <div className="cc-settings-view">
      <div className="cc-view-heading"><div><h1>System settings</h1><p>Configure operator access, workspace presentation, and admitted runtime choices.</p></div></div>
      <div className="cc-settings-grid">
        <Panel title="Operator access" icon={<KeyRound size={14} />} meta={data.token ? "Configured" : "Required for writes"}>
          <label className="cc-field"><span>Bearer credential</span><input type="password" value={data.token} onChange={(event) => data.setToken(event.target.value)} placeholder="Stored in this browser tab only" autoComplete="off" /></label>
          <p className="cc-muted-copy">Needed for approvals, corrections, and starting an admitted live episode.</p>
        </Panel>
        <Panel title="Workspace" icon={<Settings size={14} />} meta="Local preferences">
          <Toggle checked={showOverlays} onChange={() => setShowOverlays(!showOverlays)} label="World overlays" detail="Show path, clearance, and contact visualization." />
          <Toggle checked={compactWindows} onChange={() => setCompactWindows(!compactWindows)} label="Compact windows" detail="Reduce spacing when the viewport is constrained." />
        </Panel>
        <Panel title="Live runtime" icon={<Play size={14} />} meta={data.liveOptions?.enabled ? "Admitted" : "Unavailable"}>
          <div className="cc-runtime-fields">
            <label><span>Episode</span><select value={data.selectedEpisodeId} onChange={(event) => data.setSelectedEpisodeId(event.target.value)} disabled={!data.episodes.length}><option value="">No operational episode</option>{data.episodes.map((episode) => <option value={episode.episode_id} key={episode.episode_id}>{episode.episode_id}</option>)}</select></label>
            <label><span>World seed</span><select value={data.liveSeed ?? ""} onChange={(event) => data.setLiveSeed(Number(event.target.value))} disabled={!data.liveOptions?.enabled || liveActive}><option value="">Unavailable</option>{data.liveOptions?.seeds.map((seed) => <option key={seed}>{seed}</option>)}</select></label>
            <label><span>Evaluated policy</span><select value={data.livePolicyId} onChange={(event) => data.setLivePolicyId(event.target.value)} disabled={!data.liveOptions?.enabled || liveActive}><option value="">Unavailable</option>{data.liveOptions?.policies.map((policy) => <option value={policy.policy_id} key={policy.policy_id}>{policy.policy_id}</option>)}</select></label>
          </div>
          <div className="cc-runtime-actions">
            <button type="button" disabled={(!data.isLocalRoutine && !data.token) || !data.liveOptions?.enabled || liveActive || Boolean(data.mutationBusy)} onClick={() => void data.startLiveEpisode()}><Play size={13} /> Start routine</button>
            <button type="button" disabled={!liveActive || Boolean(data.mutationBusy)} onClick={() => void data.cancelLiveEpisode()}><Square size={12} /> Cancel</button>
          </div>
        </Panel>
        <Panel title="Fixed robot" icon={<Bot size={14} />} meta="Read-only">
          <div className="cc-inspector-list">
            <div><span>Identity</span><strong>MM-01</strong></div>
            <div><span>Checksum</span><strong>{shortId(data.detail?.episode.robot_checksum)}</strong></div>
            <div><span>Walking controller</span><strong>Frozen · 100 Hz</strong></div>
            <div><span>Task outputs</span><strong>Speed · turn · stop</strong></div>
          </div>
        </Panel>
        <Panel title="Providers" icon={<Cloud size={14} />} meta={`${data.providers.length} reported`} className="cc-settings-providers">
          <div className="cc-provider-list">
            {data.providers.slice(0, 6).map((provider) => <div key={provider.provider}><i className={`cc-dot cc-dot--${provider.state}`} /><span><strong>{provider.provider}</strong><small>{provider.detail}</small></span><b>{readableState(provider.state)}</b></div>)}
          </div>
        </Panel>
      </div>
    </div>
  );
}

export function CommandCenter() {
  const data = useOperatorData();
  const [view, setView] = useState<WorkspaceView>("operations");
  const [menuOpen, setMenuOpen] = useState(false);
  const [showOverlays, setShowOverlays] = useState(true);
  const [compactWindows, setCompactWindows] = useState(false);

  return (
    <main className={`command-center ${compactWindows ? "is-compact" : ""} ${showOverlays ? "has-overlays" : "no-overlays"}`}>
      <Header view={view} setView={setView} menuOpen={menuOpen} setMenuOpen={setMenuOpen} />
      {data.issue && <div className="cc-alert"><AlertTriangle size={14} /><span>{data.issue}</span><button type="button" onClick={() => void data.refresh()}><RefreshCw size={13} /> Retry</button></div>}
      {data.mutationIssue && <div className="cc-alert cc-alert--error"><AlertTriangle size={14} /><span>{data.mutationIssue}</span></div>}
      <div className="cc-content">
        {view === "operations" && <OperationsView data={data} />}
        {view === "memory" && <GraphCanvas data={data} />}
        {view === "rocketride" && <RocketRideView data={data} goToSettings={() => setView("settings")} />}
        {view === "settings" && <SettingsView data={data} showOverlays={showOverlays} setShowOverlays={setShowOverlays} compactWindows={compactWindows} setCompactWindows={setCompactWindows} />}
      </div>
    </main>
  );
}
