// LocalStorage-backed query history. Pure module — exposes a small API
// the QueryHistory component layers on top of.

import type { AnswerMode } from "@/lib/api/client";

const STORAGE_KEY = "lapua-rag.history.v1";
const MAX_ENTRIES = 20;

export type HistoryEntry = {
  readonly id: string;
  readonly query: string;
  readonly mode: AnswerMode;
  readonly tenant: string | null;
  readonly timestamp: number;
};

/** Returns history newest-first; empty array on SSR or parse errors. */
export function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    // Defensive: drop entries missing required fields rather than throw.
    return parsed.filter(
      (e): e is HistoryEntry =>
        typeof e === "object" &&
        e !== null &&
        typeof (e as HistoryEntry).query === "string" &&
        typeof (e as HistoryEntry).timestamp === "number",
    );
  } catch {
    return [];
  }
}

export function pushHistory(entry: Omit<HistoryEntry, "id" | "timestamp">): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  const full: HistoryEntry = {
    ...entry,
    id: crypto.randomUUID(),
    timestamp: Date.now(),
  };
  const next = [full, ...loadHistory()].slice(0, MAX_ENTRIES);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  return next;
}

export function clearHistory(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(STORAGE_KEY);
}
