"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertTriangle, Info, ShieldX } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { SourceCard } from "@/components/source-card";
import type { RagAnswer, AnswerMode, AbstainReason } from "@/lib/api/client";

type Props = {
  answer: RagAnswer;
  mode: AnswerMode;
  query: string;
};

const ABSTAIN_COPY: Record<AbstainReason, { title: string; tone: string; Icon: typeof AlertTriangle }> = {
  no_context: {
    title: "Ei haettavissa olevaa lähdettä",
    tone: "border-amber-500/40 bg-amber-500/5",
    Icon: Info,
  },
  below_threshold: {
    title: "Lähteet liian heikkoja luotettavaan vastaukseen",
    tone: "border-amber-500/40 bg-amber-500/5",
    Icon: AlertTriangle,
  },
  model_refused: {
    title: "Malli kieltäytyi vastaamasta",
    tone: "border-destructive/40 bg-destructive/5",
    Icon: ShieldX,
  },
};

function ScoreMeter({ score }: { score: number | null | undefined }): React.ReactNode {
  if (score == null) return null;
  // Reranker scores typically span ~[-5, +5]; normalise to [0, 1] for the bar.
  const pct = Math.max(0, Math.min(1, (score + 5) / 10));
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span className="font-mono">top score {score.toFixed(3)}</span>
      <div className="h-1.5 w-24 rounded-full bg-muted overflow-hidden">
        <div
          className="h-full bg-primary transition-all"
          style={{ width: `${pct * 100}%` }}
        />
      </div>
    </div>
  );
}

export function AnswerCard({ answer, mode, query }: Props): React.ReactNode {
  const sources = answer.lahteet ?? [];
  const isAbstain = answer.abstained;
  const abstainCopy = isAbstain && answer.abstain_reason ? ABSTAIN_COPY[answer.abstain_reason] : null;

  return (
    <Card className="overflow-hidden">
      <CardHeader className="space-y-2 pb-3">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <CardTitle className="text-sm text-muted-foreground font-normal">
            Kysymys: <span className="text-foreground font-medium">{query}</span>
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant={mode === "synth" ? "default" : "secondary"} className="uppercase">
              {mode}
            </Badge>
            <ScoreMeter score={answer.max_source_score} />
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {abstainCopy && (
          <div className={`flex gap-3 rounded-md border p-3 text-sm ${abstainCopy.tone}`}>
            <abstainCopy.Icon className="h-4 w-4 mt-0.5 shrink-0" />
            <div>
              <div className="font-medium">{abstainCopy.title}</div>
              <div className="text-muted-foreground mt-0.5">{answer.perustelut}</div>
            </div>
          </div>
        )}

        {!isAbstain && (
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer.johtopaatos}</ReactMarkdown>
            {answer.perustelut && (
              <>
                <Separator className="my-3" />
                <div className="text-sm text-muted-foreground">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer.perustelut}</ReactMarkdown>
                </div>
              </>
            )}
          </div>
        )}

        {sources.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Lähteet · {sources.length}
            </div>
            <div className="space-y-2">
              {sources.map((s, i) => (
                <SourceCard
                  key={s.chunk_id ?? `${s.doc_id}-${s.page_no}-${i}`}
                  source={s}
                  rank={i + 1}
                  // Top source's score is the only per-source signal we
                  // get back from the API; subsequent sources show no badge.
                  score={i === 0 ? answer.max_source_score : null}
                />
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
