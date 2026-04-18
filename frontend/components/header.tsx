"use client";

import { Database } from "lucide-react";
import Link from "next/link";
import { ThemeToggle } from "@/components/theme-toggle";

export function Header(): React.ReactNode {
  return (
    <header className="border-b bg-card sticky top-0 z-30 backdrop-blur supports-[backdrop-filter]:bg-card/80">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-4 px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <Database className="h-5 w-5 text-primary" />
          <span>Lapua-RAG</span>
          <span className="text-muted-foreground text-sm font-normal">· Systeemi</span>
        </Link>
        <div className="flex items-center gap-2">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
