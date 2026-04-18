"use client";

import { History, Trash2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { type HistoryEntry, clearHistory } from "@/lib/history";

type Props = {
  entries: HistoryEntry[];
  onSelect: (entry: HistoryEntry) => void;
  onCleared: () => void;
};

const fmt = new Intl.DateTimeFormat("fi-FI", {
  hour: "2-digit",
  minute: "2-digit",
  day: "2-digit",
  month: "2-digit",
});

export function QueryHistory({ entries, onSelect, onCleared }: Props): React.ReactNode {
  return (
    <Card>
      <CardHeader className="pb-3 flex-row items-center justify-between space-y-0">
        <CardTitle className="text-sm flex items-center gap-2">
          <History className="h-4 w-4" /> Historia
        </CardTitle>
        {entries.length > 0 && (
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            onClick={() => {
              clearHistory();
              onCleared();
            }}
          >
            <Trash2 className="h-3 w-3 mr-1" /> Tyhjennä
          </Button>
        )}
      </CardHeader>
      <CardContent>
        {entries.length === 0 ? (
          <div className="text-xs text-muted-foreground italic">Ei vielä kyselyjä.</div>
        ) : (
          <ScrollArea className="h-[280px] pr-2">
            <ul className="space-y-1">
              {entries.map((e) => (
                <li key={e.id}>
                  <button
                    onClick={() => onSelect(e)}
                    className="w-full text-left rounded-md p-2 hover:bg-accent transition-colors"
                  >
                    <div className="text-sm line-clamp-2">{e.query}</div>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="outline" className="text-[10px] uppercase h-4 px-1.5">
                        {e.mode}
                      </Badge>
                      <span className="text-[10px] text-muted-foreground">
                        {fmt.format(new Date(e.timestamp))}
                      </span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}
