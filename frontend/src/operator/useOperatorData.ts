import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiRequestError, liveSocketUrl, operatorApi } from "./api";
import { localDetail, localEpisode, localHealth, localLiveOptions, localLiveStatus, localRoutineRecord } from "./localRoutine";
import type {
  CorrectionPoint,
  CorrectionView,
  EpisodeDetail,
  EpisodeReviewNote,
  EpisodeSummary,
  LiveEpisodeOptions,
  LiveEpisodeStatus,
  LiveStreamMessage,
  MemoryGraphSnapshot,
  PendingApproval,
  PolicySummary,
  PromotionEligibility,
  ProviderHealth,
  ServiceHealth,
  StreamState,
  TaskPolicyTrainingJob,
  TelemetryRecord,
} from "./types";

const OPERATOR_TOKEN_KEY = "muscle-memory.operator-token";

function errorMessage(error: unknown): string {
  if (error instanceof ApiRequestError && error.requestId) {
    return `${error.message} · request ${error.requestId}`;
  }
  return error instanceof Error ? error.message : "The backend request failed";
}

function mergeRecords(
  current: TelemetryRecord[],
  incoming: TelemetryRecord,
): TelemetryRecord[] {
  const prior = current.findIndex((record) => record.sequence === incoming.sequence);
  if (prior >= 0) {
    const updated = current.slice();
    updated[prior] = incoming;
    return updated;
  }
  return [...current, incoming].sort((a, b) => a.sequence - b.sequence).slice(-2000);
}

export interface OperatorData {
  health: ServiceHealth | null;
  memoryGraph: MemoryGraphSnapshot | null;
  providers: ProviderHealth[];
  episodes: EpisodeSummary[];
  selectedEpisodeId: string;
  setSelectedEpisodeId: (episodeId: string) => void;
  detail: EpisodeDetail | null;
  notes: EpisodeReviewNote[];
  notesLoading: boolean;
  records: TelemetryRecord[];
  latestRecord: TelemetryRecord | null;
  approvals: PendingApproval[];
  policies: PolicySummary[];
  trainingJobs: TaskPolicyTrainingJob[];
  trainingJob: TaskPolicyTrainingJob | null;
  trainingEpochs: number;
  setTrainingEpochs: (epochs: number) => void;
  trainingSeed: number;
  setTrainingSeed: (seed: number) => void;
  liveOptions: LiveEpisodeOptions | null;
  liveStatus: LiveEpisodeStatus | null;
  liveSeed: number | null;
  setLiveSeed: (seed: number) => void;
  livePolicyId: string;
  setLivePolicyId: (policyId: string) => void;
  baselinePolicyId: string;
  setBaselinePolicyId: (policyId: string) => void;
  candidatePolicyId: string;
  setCandidatePolicyId: (policyId: string) => void;
  eligibility: PromotionEligibility | null;
  streamState: StreamState;
  droppedMessages: number;
  loading: boolean;
  issue: string | null;
  token: string;
  setToken: (token: string) => void;
  correction: CorrectionView | null;
  mutationBusy: string | null;
  mutationIssue: string | null;
  isLocalRoutine: boolean;
  isSyntheticDemo: boolean;
  refresh: () => Promise<void>;
  startDemoLoop: () => void;
  exitDemoLoop: () => Promise<void>;
  startLiveEpisode: () => Promise<void>;
  startTrainingJob: () => Promise<void>;
  cancelLiveEpisode: () => Promise<void>;
  decideApproval: (requirementId: string, verdict: "approve" | "reject") => Promise<void>;
  submitCorrection: (
    failureId: string,
    kind: "route" | "keep_out",
    points: CorrectionPoint[],
  ) => Promise<void>;
  createNote: (body: string, tags: string[]) => Promise<void>;
  archiveNote: (noteId: string) => Promise<void>;
  exportEpisode: () => void;
}

export function useOperatorData(): OperatorData {
  const [health, setHealth] = useState<ServiceHealth | null>(null);
  const [memoryGraph, setMemoryGraph] = useState<MemoryGraphSnapshot | null>(null);
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [selectedEpisodeId, setSelectedEpisodeId] = useState("");
  const [detail, setDetail] = useState<EpisodeDetail | null>(null);
  const [notes, setNotes] = useState<EpisodeReviewNote[]>([]);
  const [notesLoading, setNotesLoading] = useState(false);
  const [records, setRecords] = useState<TelemetryRecord[]>([]);
  const [approvals, setApprovals] = useState<PendingApproval[]>([]);
  const [policies, setPolicies] = useState<PolicySummary[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<TaskPolicyTrainingJob[]>([]);
  const [trainingJob, setTrainingJob] = useState<TaskPolicyTrainingJob | null>(null);
  const [trainingEpochs, setTrainingEpochs] = useState(180);
  const [trainingSeed, setTrainingSeed] = useState(668);
  const [liveOptions, setLiveOptions] = useState<LiveEpisodeOptions | null>(null);
  const [liveStatus, setLiveStatus] = useState<LiveEpisodeStatus | null>(null);
  const [liveSeed, setLiveSeed] = useState<number | null>(null);
  const [livePolicyId, setLivePolicyId] = useState("");
  const [baselinePolicyId, setBaselinePolicyId] = useState("");
  const [candidatePolicyId, setCandidatePolicyId] = useState("");
  const [eligibility, setEligibility] = useState<PromotionEligibility | null>(null);
  const [streamState, setStreamState] = useState<StreamState>("idle");
  const [droppedMessages, setDroppedMessages] = useState(0);
  const [loading, setLoading] = useState(true);
  const [issue, setIssue] = useState<string | null>(null);
  const [tokenValue, setTokenValue] = useState(() =>
    window.sessionStorage.getItem(OPERATOR_TOKEN_KEY) || "",
  );
  const [correction, setCorrection] = useState<CorrectionView | null>(null);
  const [mutationBusy, setMutationBusy] = useState<string | null>(null);
  const [mutationIssue, setMutationIssue] = useState<string | null>(null);
  const [isLocalRoutine, setIsLocalRoutine] = useState(false);
  const [isSyntheticDemo, setIsSyntheticDemo] = useState(false);
  const localSequence = useRef(0);
  const syntheticDemoRef = useRef(false);
  const refresh = useCallback(async () => {
    if (syntheticDemoRef.current) {
      setLoading(false);
      return;
    }
    const [healthResult, graphResult, episodeResult, approvalsResult, policiesResult, liveResult, trainingResult] =
      await Promise.allSettled([
        operatorApi.health(),
        operatorApi.memoryGraph(),
        operatorApi.episodes(),
        operatorApi.approvals(),
        operatorApi.policies(),
        operatorApi.liveOptions(),
        operatorApi.trainingJobs(),
      ]);

    const failures: string[] = [];
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    else failures.push(errorMessage(healthResult.reason));

    if (graphResult.status === "fulfilled") setMemoryGraph(graphResult.value);
    else failures.push(errorMessage(graphResult.reason));

    if (episodeResult.status === "fulfilled") {
      const nextEpisodes = episodeResult.value.items;
      setEpisodes(nextEpisodes);
      if (!nextEpisodes.length) setNotes([]);
      setSelectedEpisodeId((current) =>
        nextEpisodes.some((episode) => episode.episode_id === current)
          ? current
          : nextEpisodes[0]?.episode_id || "",
      );
    } else {
      failures.push(errorMessage(episodeResult.reason));
    }

    if (approvalsResult.status === "fulfilled") setApprovals(approvalsResult.value.items);
    else failures.push(errorMessage(approvalsResult.reason));

    if (policiesResult.status === "fulfilled") {
      const nextPolicies = policiesResult.value.items;
      setPolicies(nextPolicies);
      const heldOut = nextPolicies.filter((policy) => policy.evaluation_scope === "held_out_aggregate");
      const baseline = heldOut.find((policy) => /(^|[-_])v?0($|[-_])/i.test(policy.policy_id));
      const candidate = heldOut.find((policy) => /(^|[-_])v?1($|[-_])/i.test(policy.policy_id));
      setBaselinePolicyId((current) => current || baseline?.policy_id || heldOut[0]?.policy_id || "");
      setCandidatePolicyId(
        (current) => current || candidate?.policy_id || heldOut[1]?.policy_id || "",
      );
    } else {
      failures.push(errorMessage(policiesResult.reason));
    }

    if (liveResult.status === "fulfilled") {
      const options = liveResult.value;
      setLiveOptions(options);
      setLiveSeed((current) =>
        current !== null && options.seeds.includes(current)
          ? current
          : options.seeds[0] ?? null,
      );
      setLivePolicyId((current) =>
        options.policies.some((policy) => policy.policy_id === current)
          ? current
          : options.default_policy_id || "",
      );
    } else {
      failures.push(errorMessage(liveResult.reason));
    }

    if (trainingResult.status === "fulfilled") {
      const nextJobs = trainingResult.value.items;
      setTrainingJobs(nextJobs);
      setTrainingJob((current) =>
        nextJobs.find((job) => job.job_id === current?.job_id) ||
        nextJobs.find((job) => job.state === "queued" || job.state === "running") ||
        nextJobs[0] ||
        null,
      );
    } else {
      failures.push(errorMessage(trainingResult.reason));
    }

    if (healthResult.status === "rejected") {
      setHealth(localHealth);
      setEpisodes([localEpisode]);
      setSelectedEpisodeId(localEpisode.episode_id);
      setDetail(localDetail);
      setRecords([localRoutineRecord(0)]);
      setLiveOptions(localLiveOptions);
      setLiveSeed(17);
      setLivePolicyId(localLiveOptions.default_policy_id || "");
      setStreamState("idle");
      setIsLocalRoutine(true);
      setIsSyntheticDemo(false);
      setIssue(null);
    } else {
      setIsLocalRoutine(false);
      setIsSyntheticDemo(false);
      setIssue(failures.length ? [...new Set(failures)].join(" · ") : null);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    const timer = window.setInterval(() => void refresh(), 10_000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refresh]);

  useEffect(() => {
    if (!isLocalRoutine || liveStatus?.phase !== "running") return;
    const timer = window.setInterval(() => {
      localSequence.current += 3;
      const record = localRoutineRecord(localSequence.current);
      setRecords((current) => [...current, record].slice(-2000));
      setLiveStatus(localLiveStatus(localSequence.current, true));
    }, 300);
    return () => window.clearInterval(timer);
  }, [isLocalRoutine, liveStatus?.phase]);

  useEffect(() => {
    if (isLocalRoutine || !selectedEpisodeId) return;
    let active = true;
    void (async () => {
      try {
        const nextDetail = await operatorApi.episode(selectedEpisodeId);
        if (!active) return;
        setDetail(nextDetail);
        const page =
          nextDetail.episode.state === "running"
            ? await operatorApi.telemetry(selectedEpisodeId)
            : await operatorApi.replay(selectedEpisodeId);
        if (active) setRecords(page.records);
      } catch (error) {
        const liveIsOpening =
          liveStatus?.episode_id === selectedEpisodeId &&
          ["queued", "starting"].includes(liveStatus.phase);
        if (active && !liveIsOpening) setIssue(errorMessage(error));
      }
    })();
    return () => {
      active = false;
    };
  }, [isLocalRoutine, liveStatus?.episode_id, liveStatus?.phase, selectedEpisodeId]);

  useEffect(() => {
    if (isLocalRoutine || !selectedEpisodeId) return;
    let active = true;
    void operatorApi
      .episodeNotes(selectedEpisodeId)
      .then((result) => {
        if (active) setNotes(result.items);
      })
      .catch((error) => {
        if (active) setMutationIssue(errorMessage(error));
      })
      .finally(() => {
        if (active) setNotesLoading(false);
      });
    return () => {
      active = false;
    };
  }, [isLocalRoutine, selectedEpisodeId]);

  useEffect(() => {
    if (isLocalRoutine ||
      !liveStatus ||
      !["queued", "starting", "running", "cancelling"].includes(liveStatus.phase)
    ) {
      return;
    }
    let active = true;
    const poll = async () => {
      try {
        const status = await operatorApi.liveEpisodeStatus(liveStatus.episode_id);
        if (!active) return;
        setLiveStatus(status);
        if (["closed", "failed"].includes(status.phase)) await refresh();
      } catch (error) {
        if (active) setIssue(errorMessage(error));
      }
    };
    const initial = window.setTimeout(() => void poll(), 150);
    const timer = window.setInterval(() => void poll(), 500);
    return () => {
      active = false;
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [isLocalRoutine, liveStatus, refresh]);

  useEffect(() => {
    if (isLocalRoutine || !selectedEpisodeId) return;
    let stopped = false;
    let reconnectTimer = 0;
    let socket: WebSocket | null = null;

    const connect = () => {
      if (stopped) return;
      setStreamState("connecting");
      socket = new WebSocket(liveSocketUrl(selectedEpisodeId));
      socket.addEventListener("open", () => setStreamState("live"));
      socket.addEventListener("message", (event) => {
        try {
          const message = JSON.parse(String(event.data)) as LiveStreamMessage;
          if (message.episode_id !== selectedEpisodeId) return;
          setDroppedMessages(message.dropped_before);
          if (message.kind === "telemetry" && message.telemetry) {
            setRecords((current) => mergeRecords(current, message.telemetry as TelemetryRecord));
          }
        } catch {
          setStreamState("error");
        }
      });
      socket.addEventListener("close", (event) => {
        if (stopped) return;
        setStreamState(event.code === 4404 ? "error" : "closed");
        reconnectTimer = window.setTimeout(connect, 3000);
      });
      socket.addEventListener("error", () => setStreamState("error"));
    };
    connect();
    return () => {
      stopped = true;
      window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [isLocalRoutine, selectedEpisodeId]);

  useEffect(() => {
    if (!baselinePolicyId || !candidatePolicyId || baselinePolicyId === candidatePolicyId) {
      return;
    }
    let active = true;
    void operatorApi
      .promotionEligibility(baselinePolicyId, candidatePolicyId)
      .then((result) => {
        if (active) setEligibility(result);
      })
      .catch((error: unknown) => {
        if (active) {
          setEligibility(null);
          setIssue(errorMessage(error));
        }
      });
    return () => {
      active = false;
    };
  }, [baselinePolicyId, candidatePolicyId]);

  const setToken = useCallback((value: string) => {
    setTokenValue(value);
    if (value) window.sessionStorage.setItem(OPERATOR_TOKEN_KEY, value);
    else window.sessionStorage.removeItem(OPERATOR_TOKEN_KEY);
  }, []);

  const decideApproval = useCallback(
    async (requirementId: string, verdict: "approve" | "reject") => {
      setMutationBusy(requirementId);
      setMutationIssue(null);
      try {
        await operatorApi.decideApproval(requirementId, verdict, tokenValue);
        const next = await operatorApi.approvals();
        setApprovals(next.items);
      } catch (error) {
        setMutationIssue(errorMessage(error));
      } finally {
        setMutationBusy(null);
      }
    },
    [tokenValue],
  );

  const submitCorrection = useCallback(
    async (failureId: string, kind: "route" | "keep_out", points: CorrectionPoint[]) => {
      if (!selectedEpisodeId) return;
      setMutationBusy("correction");
      setMutationIssue(null);
      try {
        const result = await operatorApi.submitCorrection(
          selectedEpisodeId,
          failureId,
          kind,
          points,
          tokenValue,
        );
        setCorrection(result);
        const next = await operatorApi.approvals();
        setApprovals(next.items);
        setDetail(await operatorApi.episode(selectedEpisodeId));
      } catch (error) {
        setMutationIssue(errorMessage(error));
      } finally {
        setMutationBusy(null);
      }
    },
    [selectedEpisodeId, tokenValue],
  );

  const createNote = useCallback(
    async (body: string, tags: string[]) => {
      if (!selectedEpisodeId || isLocalRoutine) {
        setMutationIssue("Review notes require a persisted operational episode.");
        return;
      }
      if (!tokenValue) {
        setMutationIssue("Configure an operator credential before adding a review note.");
        return;
      }
      setMutationBusy("note-create");
      setMutationIssue(null);
      try {
        const note = await operatorApi.createEpisodeNote(
          selectedEpisodeId,
          body,
          tags,
          tokenValue,
        );
        setNotes((current) => [note, ...current]);
      } catch (error) {
        setMutationIssue(errorMessage(error));
      } finally {
        setMutationBusy(null);
      }
    },
    [isLocalRoutine, selectedEpisodeId, tokenValue],
  );

  const archiveNote = useCallback(
    async (noteId: string) => {
      if (!selectedEpisodeId || isLocalRoutine || !tokenValue) {
        setMutationIssue("Configure an operator credential to archive a review note.");
        return;
      }
      setMutationBusy(`note-archive-${noteId}`);
      setMutationIssue(null);
      try {
        await operatorApi.archiveEpisodeNote(selectedEpisodeId, noteId, tokenValue);
        setNotes((current) => current.filter((note) => note.note_id !== noteId));
      } catch (error) {
        setMutationIssue(errorMessage(error));
      } finally {
        setMutationBusy(null);
      }
    },
    [isLocalRoutine, selectedEpisodeId, tokenValue],
  );

  const exportEpisode = useCallback(() => {
    if (!detail || isLocalRoutine) {
      setMutationIssue("Episode export requires a persisted operational episode.");
      return;
    }
    const payload = {
      schema_version: "muscle-memory.episode-export.v1",
      exported_at: new Date().toISOString(),
      episode: detail.episode,
      detail,
      telemetry: records,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${detail.episode.episode_id}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }, [detail, isLocalRoutine, records]);

  const startLiveEpisode = useCallback(async () => {
    if (liveSeed === null || !livePolicyId) return;
    setMutationBusy("live-start");
    setMutationIssue(null);
    try {
      if (isLocalRoutine) {
        localSequence.current = 0;
        setDetail({ ...localDetail, episode: { ...localEpisode, state: "running" } });
        setRecords([localRoutineRecord(0)]);
        setLiveStatus(localLiveStatus(0, true));
        setSelectedEpisodeId(localEpisode.episode_id);
        return;
      }
      const started = await operatorApi.startLiveEpisode(
        liveSeed,
        livePolicyId,
        tokenValue,
      );
      setLiveStatus(started);
      setDetail(null);
      setRecords([]);
      setSelectedEpisodeId(started.episode_id);
      await new Promise((resolve) => window.setTimeout(resolve, 200));
      await refresh();
      setSelectedEpisodeId(started.episode_id);
    } catch (error) {
      setMutationIssue(errorMessage(error));
    } finally {
      setMutationBusy(null);
    }
  }, [isLocalRoutine, livePolicyId, liveSeed, refresh, tokenValue]);

  const startDemoLoop = useCallback(() => {
    syntheticDemoRef.current = true;
    localSequence.current = 0;
    setIsLocalRoutine(true);
    setIsSyntheticDemo(true);
    setHealth(localHealth);
    setEpisodes([localEpisode]);
    setSelectedEpisodeId(localEpisode.episode_id);
    setDetail({ ...localDetail, episode: { ...localEpisode, state: "running" } });
    setNotes([]);
    setRecords([localRoutineRecord(0)]);
    setLiveOptions(localLiveOptions);
    setLiveSeed(17);
    setLivePolicyId(localLiveOptions.default_policy_id || "");
    setLiveStatus(localLiveStatus(0, true));
    setStreamState("idle");
    setIssue(null);
    setMutationIssue(null);
  }, []);

  const exitDemoLoop = useCallback(async () => {
    syntheticDemoRef.current = false;
    setIsSyntheticDemo(false);
    setIsLocalRoutine(false);
    setLiveStatus(null);
    setDetail(null);
    setNotes([]);
    setRecords([]);
    setCorrection(null);
    setIssue(null);
    setMutationIssue(null);
    setLoading(true);
    await refresh();
  }, [refresh]);

  const cancelLiveEpisode = useCallback(async () => {
    if (!liveStatus) return;
    setMutationBusy("live-cancel");
    setMutationIssue(null);
    try {
      if (isLocalRoutine) {
        setLiveStatus(localLiveStatus(localSequence.current, false));
        setDetail({ ...localDetail, episode: { ...localEpisode, state: "aborted" } });
        return;
      }
      setLiveStatus(
        await operatorApi.cancelLiveEpisode(liveStatus.episode_id, tokenValue),
      );
    } catch (error) {
      setMutationIssue(errorMessage(error));
    } finally {
      setMutationBusy(null);
    }
  }, [isLocalRoutine, liveStatus, tokenValue]);

  useEffect(() => {
    if (!trainingJob || !["queued", "running"].includes(trainingJob.state)) return;
    let active = true;
    const poll = async () => {
      try {
        const next = await operatorApi.trainingJob(trainingJob.job_id);
        if (!active) return;
        setTrainingJob(next);
        setTrainingJobs((current) => [next, ...current.filter((job) => job.job_id !== next.job_id)]);
      } catch (error) {
        if (active) setMutationIssue(errorMessage(error));
      }
    };
    const initial = window.setTimeout(() => void poll(), 150);
    const timer = window.setInterval(() => void poll(), 500);
    return () => {
      active = false;
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [trainingJob]);

  const startTrainingJob = useCallback(async () => {
    setMutationBusy("training-start");
    setMutationIssue(null);
    try {
      const started = await operatorApi.startTrainingJob(
        trainingEpochs,
        trainingSeed,
        tokenValue,
      );
      setTrainingJob(started);
      setTrainingJobs((current) => [started, ...current.filter((job) => job.job_id !== started.job_id)]);
    } catch (error) {
      setMutationIssue(errorMessage(error));
    } finally {
      setMutationBusy(null);
    }
  }, [tokenValue, trainingEpochs, trainingSeed]);

  const latestRecord = records.at(-1) || null;
  const providers = useMemo(() => health?.providers || [], [health]);

  return {
    health,
    memoryGraph,
    providers,
    episodes,
    selectedEpisodeId,
    setSelectedEpisodeId,
    detail,
    notes,
    notesLoading,
    records,
    latestRecord,
    approvals,
    policies,
    trainingJobs,
    trainingJob,
    trainingEpochs,
    setTrainingEpochs,
    trainingSeed,
    setTrainingSeed,
    liveOptions,
    liveStatus,
    liveSeed,
    setLiveSeed,
    livePolicyId,
    setLivePolicyId,
    baselinePolicyId,
    setBaselinePolicyId,
    candidatePolicyId,
    setCandidatePolicyId,
    eligibility:
      eligibility?.baseline_policy_id === baselinePolicyId &&
      eligibility.candidate_policy_id === candidatePolicyId
        ? eligibility
        : null,
    streamState,
    droppedMessages,
    loading,
    issue,
    token: tokenValue,
    setToken,
    correction,
    mutationBusy,
    mutationIssue,
    isLocalRoutine,
    isSyntheticDemo,
    refresh,
    startDemoLoop,
    exitDemoLoop,
    startLiveEpisode,
    startTrainingJob,
    cancelLiveEpisode,
    decideApproval,
    submitCorrection,
    createNote,
    archiveNote,
    exportEpisode,
  };
}
