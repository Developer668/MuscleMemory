import type {
  ApiErrorPayload,
  CorrectionPoint,
  CorrectionView,
  EpisodeDetail,
  EpisodeList,
  PendingApprovalList,
  PolicySummaryList,
  PromotionEligibility,
  ReplayPage,
  ServiceHealth,
  TelemetryPage,
} from "./types";

const configuredBase = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE = (configuredBase || "/api/v1").replace(/\/$/, "");

export class ApiRequestError extends Error {
  readonly status: number;
  readonly requestId: string | null;

  constructor(message: string, status: number, requestId: string | null) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.requestId = requestId;
  }
}

async function request<T>(path: string, init?: RequestInit, token?: string): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Accept", "application/json");
  if (init?.body) headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });
  if (!response.ok) {
    const payload = await response
      .json()
      .then((value: unknown) => value as ApiErrorPayload)
      .catch(() => null);
    throw new ApiRequestError(
      payload?.error?.message || `Request failed with status ${response.status}`,
      response.status,
      payload?.error?.request_id || response.headers.get("X-Request-ID"),
    );
  }
  return (await response.json()) as T;
}

export const operatorApi = {
  health: () => request<ServiceHealth>("/health"),
  episodes: () => request<EpisodeList>("/episodes?limit=200"),
  episode: (episodeId: string) =>
    request<EpisodeDetail>(`/episodes/${encodeURIComponent(episodeId)}`),
  telemetry: (episodeId: string) =>
    request<TelemetryPage>(`/episodes/${encodeURIComponent(episodeId)}/telemetry?limit=2000`),
  replay: (episodeId: string) =>
    request<ReplayPage>(`/episodes/${encodeURIComponent(episodeId)}/replay?limit=2000`),
  approvals: () => request<PendingApprovalList>("/approvals/pending"),
  policies: () => request<PolicySummaryList>("/policies"),
  promotionEligibility: (baselineId: string, candidateId: string) => {
    const query = new URLSearchParams({
      baseline_policy_id: baselineId,
      candidate_policy_id: candidateId,
    });
    return request<PromotionEligibility>(`/policies/promotion-eligibility?${query.toString()}`);
  },
  decideApproval: (requirementId: string, verdict: "approve" | "reject", token: string) =>
    request(
      `/approvals/${encodeURIComponent(requirementId)}/decision`,
      { method: "POST", body: JSON.stringify({ verdict, note: "Operator console decision" }) },
      token,
    ),
  submitCorrection: (
    episodeId: string,
    failureId: string,
    kind: "route" | "keep_out",
    points: CorrectionPoint[],
    token: string,
  ) =>
    request<CorrectionView>(
      `/episodes/${encodeURIComponent(episodeId)}/corrections`,
      {
        method: "POST",
        body: JSON.stringify({
          failure_id: failureId,
          kind,
          points,
          description: `Operator ${kind === "route" ? "route" : "keep-out"} correction`,
        }),
      },
      token,
    ),
};

export function liveSocketUrl(episodeId: string): string {
  const configured = import.meta.env.VITE_WS_BASE_URL?.trim();
  if (configured) {
    return `${configured.replace(/\/$/, "")}/episodes/${encodeURIComponent(episodeId)}/live`;
  }
  if (/^https?:\/\//.test(API_BASE)) {
    const url = new URL(API_BASE);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = `${url.pathname.replace(/\/$/, "")}/episodes/${encodeURIComponent(episodeId)}/live`;
    return url.toString();
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${API_BASE}/episodes/${encodeURIComponent(episodeId)}/live`;
}
