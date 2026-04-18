"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, FileStack, GitCommitVertical, Loader2 } from "lucide-react";
import { api, type AnswerMode } from "@/lib/api/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";

type Props = {
  mode: AnswerMode;
  onModeChange: (mode: AnswerMode) => void;
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
          <div className="flex items-center justify-between gap-3">
            <div className="space-y-0.5">
              <Label htmlFor="mode-toggle" className="text-sm font-normal">
                Vastaustapa
              </Label>
              <div className="text-xs text-muted-foreground">
                {mode === "synth"
                  ? "Synthesis: LLM tiivistää lähteistä"
                  : "Retrieve: top-N lähteet sellaisinaan"}
              </div>
            </div>
            <Switch
              id="mode-toggle"
              checked={mode === "synth"}
              onCheckedChange={(v) => onModeChange(v ? "synth" : "retrieve")}
            />
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
