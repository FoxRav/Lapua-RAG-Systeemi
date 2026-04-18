"use client";

import { useTheme } from "next-themes";
import { Moon, Sun, Monitor } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

const ORDER = ["light", "dark", "system"] as const;

export function ThemeToggle(): React.ReactNode {
  const { theme, setTheme } = useTheme();

  const current = (ORDER as readonly string[]).includes(theme ?? "")
    ? (theme as (typeof ORDER)[number])
    : "system";
  const next = ORDER[(ORDER.indexOf(current) + 1) % ORDER.length];

  const Icon = current === "light" ? Sun : current === "dark" ? Moon : Monitor;

  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(next)}
            aria-label={`Switch theme (now: ${current})`}
          >
            <Icon className="h-4 w-4" />
          </Button>
        }
      />
      <TooltipContent>Tema: {current} → {next}</TooltipContent>
    </Tooltip>
  );
}
