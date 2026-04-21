"use client";

import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { Loader2, Send } from "lucide-react";
import { toast } from "sonner";
import {
  api,
  ApiError,
  type AggregateResult,
  type AnswerMode,
  type RagAnswer,
} from "@/lib/api/client";
import { classifyQuery } from "@/lib/classify-query";
import { pushHistory } from "@/lib/history";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AnswerCard } from "@/components/answer-card";
import { AggregateCard } from "@/components/aggregate-card";

type ChatTurn =
  | {
      readonly kind: "rag";
      readonly id: string;
      readonly query: string;
      readonly mode: AnswerMode;
      readonly answer: RagAnswer | null;
      readonly status: "pending" | "ok" | "error";
      readonly error?: string;
    }
  | {
      readonly kind: "aggregate";
      readonly id: string;
      readonly query: string;
      readonly result: AggregateResult | null;
      readonly status: "pending" | "ok" | "error";
      readonly error?: string;
    };

type Props = {
  mode: AnswerMode;
  /** Optional preset to inject into the input (e.g. from QueryHistory). */
  preset?: string | null;
  onPresetConsumed?: () => void;
  onHistoryChange?: () => void;
};

export function ChatPanel({
  mode,
  preset,
  onPresetConsumed,
  onHistoryChange,
}: Props): React.ReactNode {
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // History → input prefill. Synchronous setState in an effect is the
  // standard way to mirror an external "command" prop into local state;
  // the new react-hooks/set-state-in-effect rule doesn't yet model this.
  useEffect(() => {
    if (preset != null) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setInput(preset);
      onPresetConsumed?.();
      inputRef.current?.focus();
    }
  }, [preset, onPresetConsumed]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const ragMutation = useMutation({
    mutationFn: (payload: { query: string; mode: AnswerMode }) =>
      api.query({ query: payload.query, mode: payload.mode }),
  });
  const aggregateMutation = useMutation({
    mutationFn: (payload: { query: string }) => api.aggregate({ query: payload.query }),
  });

  const pending = ragMutation.isPending || aggregateMutation.isPending;

  async function submit(query: string) {
    const trimmed = query.trim();
    if (trimmed.length < 3) {
      toast.error("Kysymyksen on oltava vähintään 3 merkkiä.");
      return;
    }
    const id = crypto.randomUUID();
    const kind = classifyQuery(trimmed);
    setInput("");

    if (kind === "aggregate") {
      const pendingTurn: ChatTurn = {
        kind: "aggregate",
        id,
        query: trimmed,
        result: null,
        status: "pending",
      };
      setTurns((t) => [...t, pendingTurn]);
      try {
        const result = await aggregateMutation.mutateAsync({ query: trimmed });
        setTurns((t) =>
          t.map((tt) =>
            tt.id === id && tt.kind === "aggregate"
              ? { ...tt, result, status: "ok" }
              : tt,
          ),
        );
        pushHistory({ query: trimmed, mode, tenant: null });
        onHistoryChange?.();
      } catch (err) {
        const msg =
          err instanceof ApiError ? `${err.status} ${err.detail}` : (err as Error).message;
        setTurns((t) =>
          t.map((tt) => (tt.id === id ? { ...tt, status: "error", error: msg } : tt)),
        );
        toast.error(`Aggregointikysely epäonnistui: ${msg}`);
      }
      return;
    }

    const pendingTurn: ChatTurn = {
      kind: "rag",
      id,
      query: trimmed,
      mode,
      answer: null,
      status: "pending",
    };
    setTurns((t) => [...t, pendingTurn]);

    try {
      const answer = await ragMutation.mutateAsync({ query: trimmed, mode });
      setTurns((t) =>
        t.map((tt) =>
          tt.id === id && tt.kind === "rag" ? { ...tt, answer, status: "ok" } : tt,
        ),
      );
      pushHistory({ query: trimmed, mode, tenant: null });
      onHistoryChange?.();
    } catch (err) {
      const msg =
        err instanceof ApiError ? `${err.status} ${err.detail}` : (err as Error).message;
      setTurns((t) =>
        t.map((tt) => (tt.id === id ? { ...tt, status: "error", error: msg } : tt)),
      );
      toast.error(`Kysely epäonnistui: ${msg}`);
    }
  }

  function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    submit(input);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      submit(input);
    }
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {turns.length === 0 && (
          <Card className="border-dashed bg-card/50">
            <CardContent className="p-6 text-center text-muted-foreground space-y-2">
              <div className="text-base font-medium text-foreground">
                Tervetuloa Lapua-RAG Systeemiin
              </div>
              <div className="text-sm">
                Kysy kysymys korpuksen sisällöstä — esim.{" "}
                <em>“Kuka valittiin Jytyn pääluottamusmieheksi Lapualla?”</em> tai{" "}
                <em>“Kuinka monta päätöstä Kai Pöntiselle?”</em>
              </div>
              <div className="text-xs">
                Aktiivinen tila: <span className="font-mono uppercase">{mode}</span>. Vaihda
                oikeasta sivupalkista. Aggregaattikysymykset ohjataan automaattisesti
                /v1/aggregate-endpointiin.
              </div>
            </CardContent>
          </Card>
        )}

        {turns.map((t) => {
          if (t.status === "pending") {
            return (
              <Card key={t.id} className="overflow-hidden">
                <CardContent className="p-4 space-y-3">
                  <div className="text-sm text-muted-foreground">
                    Kysymys: <span className="text-foreground">{t.query}</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Loader2 className="h-3 w-3 animate-spin" />{" "}
                    {t.kind === "aggregate"
                      ? "Lasketaan Systeemistä…"
                      : "Haetaan ja rerankataan…"}
                  </div>
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-1/2" />
                  <Skeleton className="h-20 w-full" />
                </CardContent>
              </Card>
            );
          }
          if (t.status === "error") {
            return (
              <Card key={t.id} className="border-destructive/40">
                <CardContent className="p-4 space-y-2">
                  <div className="text-sm">
                    Kysymys: <span className="text-foreground">{t.query}</span>
                  </div>
                  <div className="text-sm text-destructive">{t.error}</div>
                </CardContent>
              </Card>
            );
          }
          if (t.kind === "aggregate") {
            return <AggregateCard key={t.id} query={t.query} result={t.result!} />;
          }
          return <AnswerCard key={t.id} answer={t.answer!} mode={t.mode} query={t.query} />;
        })}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={onSubmit}
        className="border-t bg-card p-4 flex items-end gap-2"
      >
        <Textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Esitä kysymys Systeemille… (Ctrl+Enter lähettää)"
          className="min-h-[60px] resize-none flex-1"
          disabled={pending}
        />
        <Button type="submit" disabled={pending} size="lg">
          {pending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </form>
    </div>
  );
}
