"use client";

import { Calculator, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { AggregateResult } from "@/lib/api/client";

type Props = {
  query: string;
  result: AggregateResult;
};

// Locale-aware formatters; lazy-initialise via module-level constants
// so we don't rebuild them on every render.
const EUR_FORMATTER = new Intl.NumberFormat("fi-FI", {
  style: "currency",
  currency: "EUR",
  minimumFractionDigits: 2,
});
const COUNT_FORMATTER = new Intl.NumberFormat("fi-FI");

function formatValue(result: AggregateResult): string {
  if (result.value == null) return "–";
  if (result.result_type === "sum") {
    return EUR_FORMATTER.format(result.value);
  }
  return COUNT_FORMATTER.format(result.value);
}

export function AggregateCard({ query, result }: Props): React.ReactNode {
  const isUnsupported = result.result_type === "not_supported";
  return (
    <Card className="overflow-hidden">
      <CardHeader className="space-y-2 pb-3">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <CardTitle className="text-sm text-muted-foreground font-normal">
            Kysymys: <span className="text-foreground font-medium">{query}</span>
          </CardTitle>
          <Badge variant={isUnsupported ? "secondary" : "default"} className="uppercase">
            {result.result_type === "count" && "Lukumäärä"}
            {result.result_type === "sum" && "Summa"}
            {isUnsupported && "Ei tuettu"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {isUnsupported ? (
          <div className="flex gap-3 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
            <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
            <div>
              <div className="font-medium">Aggregointi ei tue tätä kysymystä</div>
              <div className="text-muted-foreground mt-0.5">{result.explanation}</div>
            </div>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-3">
              <Calculator className="h-6 w-6 text-primary shrink-0" />
              <div className="text-3xl font-bold tabular-nums">
                {formatValue(result)}
              </div>
            </div>
            <p className="text-sm text-muted-foreground">{result.explanation}</p>
            {result.entity && (
              <p className="text-xs text-muted-foreground">
                Rajaus: <span className="font-mono">{result.entity}</span>
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
