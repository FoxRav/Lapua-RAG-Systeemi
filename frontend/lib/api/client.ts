// Thin typed wrapper around fetch for the Lapua-RAG API.
//
// Why not openapi-fetch? The API surface is small (3 endpoints) and we
// want fine-grained control over error messages surfaced to the user.

import type { components } from "./openapi";

export type RagAnswer = components["schemas"]["RagAnswer"];
export type RagSource = components["schemas"]["RagSource"];
export type ChunkDetail = components["schemas"]["ChunkDetail"];
// AnswerMode + AbstainReason are inline Literal unions in the OpenAPI
// spec (Pydantic Literals don't get their own schema entry). We re-derive
// them from the QueryRequest body / RagAnswer field for type-safety.
export type AnswerMode = NonNullable<components["schemas"]["QueryRequest"]["mode"]>;
export type AbstainReason = NonNullable<RagAnswer["abstain_reason"]>;

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080").replace(/\/$/, "");

/** Stable, actionable error class so the UI can branch on cause. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(
  path: string,
  init: RequestInit & { signal?: AbortSignal } = {},
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // body wasn't JSON; keep statusText
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  async query(payload: {
    query: string;
    tenant?: string;
    mode?: AnswerMode;
  }): Promise<RagAnswer> {
    return request<RagAnswer>("/v1/query", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async getChunk(chunkId: string): Promise<ChunkDetail> {
    return request<ChunkDetail>(`/v1/chunks/${encodeURIComponent(chunkId)}`);
  },

  async getSystemStats(): Promise<Record<string, unknown>> {
    return request("/v1/system/stats");
  },

  async getSystemVersion(): Promise<Record<string, unknown>> {
    return request("/v1/system/version");
  },

  async getSystemCoverage(): Promise<Record<string, unknown>> {
    return request("/v1/system/coverage");
  },

  /** Build the streaming PDF URL with optional `#page=N` anchor. */
  pdfUrl(docId: string, pageNo?: number): string {
    const base = `${API_BASE}/v1/documents/${encodeURIComponent(docId)}/source`;
    return pageNo ? `${base}#page=${pageNo}&zoom=page-fit` : base;
  },
};
