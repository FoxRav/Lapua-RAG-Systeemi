"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Copy, Check, FileText, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { api, type RagSource } from "@/lib/api/client";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PdfViewer } from "@/components/pdf-viewer";

type Props = {
  source: RagSource;
  rank: number;
  /**
   * Optional rerank score to surface. The API doesn't expose per-source
   * scores in retrieve mode (only `max_source_score` for the top result),
   * so this is undefined for non-top sources.
   */
  score?: number | null;
};

function scoreVariant(s: number | null | undefined): "default" | "secondary" | "outline" {
  if (s == null) return "outline";
  if (s >= 0.5) return "default";
  if (s >= 0) return "secondary";
  return "outline";
}

export function SourceCard({ source, rank, score }: Props): React.ReactNode {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const fullChunk = useQuery({
    queryKey: ["chunk", source.chunk_id],
    queryFn: () => api.getChunk(source.chunk_id!),
    enabled: expanded && !!source.chunk_id,
  });

  const copyDocId = async () => {
    await navigator.clipboard.writeText(source.doc_id);
    setCopied(true);
    toast.success("doc_id kopioitu");
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2 flex-wrap text-sm">
            <Badge variant="outline" className="font-mono text-xs">
              #{rank}
            </Badge>
            <button
              onClick={copyDocId}
              className="font-mono text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
              title="Kopioi doc_id"
            >
              {source.doc_id}
              {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
            </button>
            <Badge variant="secondary">s. {source.page_no}</Badge>
            {source.section_id && <Badge variant="outline">{source.section_id}</Badge>}
            {score != null && (
              <Badge variant={scoreVariant(score)} title="Reranker score">
                {score.toFixed(3)}
              </Badge>
            )}
          </div>
          <PdfViewer
            docId={source.doc_id}
            pageNo={source.page_no}
            trigger={
              <Button size="sm" variant="outline" className="h-7">
                <FileText className="h-3 w-3 mr-1" /> PDF
              </Button>
            }
          />
        </div>

        <div className="text-sm leading-relaxed whitespace-pre-wrap break-words">
          {expanded && source.chunk_id ? (
            fullChunk.isLoading ? (
              <span className="inline-flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                Ladataan koko chunkia…
              </span>
            ) : fullChunk.isError ? (
              <span className="text-destructive">
                Chunkin lataus epäonnistui: {(fullChunk.error as Error).message}
              </span>
            ) : (
              fullChunk.data?.text ?? source.snippet
            )
          ) : (
            source.snippet
          )}
        </div>

        {source.chunk_id && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setExpanded((v) => !v)}
            className="text-xs h-7 -ml-2 text-muted-foreground"
          >
            {expanded ? "Näytä vähemmän" : "Näytä koko chunk"}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
