"use client";

import { useCallback, useEffect, useState } from "react";
import type { AnswerMode } from "@/lib/api/client";
import { type HistoryEntry, loadHistory } from "@/lib/history";
import { Header } from "@/components/header";
import { ChatPanel } from "@/components/chat-panel";
import { SystemSidebar } from "@/components/system-sidebar";
import { QueryHistory } from "@/components/query-history";

const MODE_STORAGE_KEY = "lapua-rag.answer-mode.v1";
// Default to extract for new sessions: it's the v0.6 bridge that gives one
// coherent cited answer even while lapua-llm-v2 over-abstains in synth mode.
const DEFAULT_MODE: AnswerMode = "extract";

function readPersistedMode(): AnswerMode {
  if (typeof window === "undefined") return DEFAULT_MODE;
  const v = window.localStorage.getItem(MODE_STORAGE_KEY);
  return v === "synth" || v === "retrieve" || v === "extract" ? v : DEFAULT_MODE;
}

export default function HomePage(): React.ReactNode {
  // Hydration-safe: initial render uses the SSR default, then we rehydrate
  // from LocalStorage in an effect to avoid mismatched markup.
  const [mode, setMode] = useState<AnswerMode>(DEFAULT_MODE);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [preset, setPreset] = useState<string | null>(null);

  useEffect(() => {
    // Intentional client-side hydration from LocalStorage. SSR cannot know
    // the persisted mode/history, so we render a sane default first and
    // upgrade after mount. The new react-hooks/set-state-in-effect rule
    // doesn't have an exemption for hydration patterns yet.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMode(readPersistedMode());
    setHistory(loadHistory());
  }, []);

  const onModeChange = useCallback((next: AnswerMode) => {
    setMode(next);
    window.localStorage.setItem(MODE_STORAGE_KEY, next);
  }, []);

  const refreshHistory = useCallback(() => setHistory(loadHistory()), []);

  return (
    <div className="flex flex-col h-dvh">
      <Header />
      <div className="mx-auto flex w-full max-w-7xl flex-1 min-h-0 gap-4 px-4 py-4">
        <main className="flex-1 min-w-0 flex flex-col rounded-lg border bg-card overflow-hidden">
          <ChatPanel
            mode={mode}
            preset={preset}
            onPresetConsumed={() => setPreset(null)}
            onHistoryChange={refreshHistory}
          />
        </main>
        <div className="hidden lg:block w-[320px] shrink-0 space-y-4 overflow-y-auto">
          <SystemSidebar mode={mode} onModeChange={onModeChange} />
          <QueryHistory
            entries={history}
            onSelect={(e) => setPreset(e.query)}
            onCleared={refreshHistory}
          />
        </div>
      </div>
    </div>
  );
}
