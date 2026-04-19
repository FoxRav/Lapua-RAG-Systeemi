"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, FileStack, GitCommitVertical, Loader2 } from "lucide-react";
import { api, type AnswerMode } from "@/lib/api/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";

type Props = {
  mode: AnswerMode;
  onModeChange: (mode: AnswerMode) => void;
};

// Description and short label kept here so the SystemSidebar UI is the
// single source of truth for what each mode does — no scattered copies.
const MODE_DESCRIPTIONS: Record<AnswerMode, string> = {
  extract:
    "Extract: LoRA lainaa relevantin virkkeen, Python rakentaa yhden vastauksen (silta v3:lle).",
  retrieve: "Retrieve: top-N lähteet sellaisinaan, ei LLM-syntheesiä.",
  synth: "Synthesis: LLM tiivistää vastauksen lähteistä (vaatii toimivan LoRAn).",
};

type StatsResponse = {
  documents?: number;
  chunks?: number;
  tokens?: number;
  by_status?: Array<{ status: string; count: number }>;
};

type CoverageResponse = {
  coverage?: Array<{ doc_type: string; count: number }>;
};

type VersionResponse = {
  version?: string;
  build_hash?: string;
};

function StatBlock({ label, value }: { label: string; value: string | number }): React.ReactNode {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="font-mono text-sm">{value}</span>
    </div>
  );
}

export function SystemSidebar({ mode, onModeChange }: Props): React.ReactNode {
  const stats = useQuery({ queryKey: ["stats"], queryFn: () => api.getSystemStats() as Promise<StatsResponse> });
  const coverage = useQuery({
    queryKey: ["coverage"],
    queryFn: () => api.getSystemCoverage() as Promise<CoverageResponse>,
  });
  const version = useQuery({
    queryKey: ["version"],
    queryFn: () => api.getSystemVersion() as Promise<VersionResponse>,
  });

  return (
    <aside className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Activity className="h-4 w-4" /> Tila
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-2">
            <Label className="text-sm font-normal">Vastaustapa</Label>
            <Tabs
              value={mode}
              onValueChange={(value) => {
                // Base UI Tabs returns `string | number`; we know our values.
                if (value === "extract" || value === "retrieve" || value === "synth") {
                  onModeChange(value);
                }
              }}
            >
              <TabsList className="grid grid-cols-3 w-full">
                <TabsTrigger value="extract">Extract</TabsTrigger>
                <TabsTrigger value="retrieve">Retrieve</TabsTrigger>
                <TabsTrigger value="synth">Synth</TabsTrigger>
              </TabsList>
            </Tabs>
            <div className="text-xs text-muted-foreground leading-relaxed">
              {MODE_DESCRIPTIONS[mode]}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <FileStack className="h-4 w-4" /> Korpus
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {stats.isLoading ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="h-3 w-3 animate-spin" /> ladataan…
            </div>
          ) : stats.isError ? (
            <div className="text-xs text-destructive">Statsien lataus epäonnistui</div>
          ) : (
            <>
              <StatBlock label="Dokumentit" value={stats.data?.documents ?? "—"} />
              <StatBlock label="Chunkit" value={stats.data?.chunks ?? "—"} />
              <StatBlock label="Tokenit" value={(stats.data?.tokens ?? 0).toLocaleString("fi-FI")} />
              {stats.data?.by_status && stats.data.by_status.length > 0 && (
                <>
                  <Separator className="my-2" />
                  <div className="space-y-1">
                    {stats.data.by_status.map((s) => (
                      <StatBlock key={s.status} label={s.status} value={s.count} />
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {coverage.data?.coverage && coverage.data.coverage.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Kattavuus / dokumenttityyppi</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            {coverage.data.coverage.map((c) => (
              <StatBlock key={c.doc_type} label={c.doc_type} value={c.count} />
            ))}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <GitCommitVertical className="h-4 w-4" /> Versio
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {version.isLoading ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="h-3 w-3 animate-spin" /> ladataan…
            </div>
          ) : version.isError ? (
            <div className="text-xs text-destructive">Versionhaku epäonnistui</div>
          ) : (
            <div className="space-y-1">
              {version.data?.version && (
                <Badge variant="outline" className="font-mono text-xs">
                  v{version.data.version}
                </Badge>
              )}
              {version.data?.build_hash && (
                <div className="font-mono text-xs text-muted-foreground truncate">
                  {String(version.data.build_hash).slice(0, 12)}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </aside>
  );
}
