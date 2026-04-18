"use client";

import { ExternalLink } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api/client";

type Props = {
  docId: string;
  pageNo: number;
  // ReactElement (not ReactNode) because Base UI's `render` prop on
  // DialogTrigger needs a single element it can clone the trigger props
  // into. Strings or fragments aren't valid here.
  trigger?: React.ReactElement;
};

/**
 * Lazy PDF viewer modal: builds the streaming URL and embeds an iframe
 * that hands rendering to the browser's built-in PDF viewer (Chrome/Edge
 * /Firefox all support `#page=N` anchors).
 */
export function PdfViewer({ docId, pageNo, trigger }: Props): React.ReactNode {
  const url = api.pdfUrl(docId, pageNo);

  const triggerNode = trigger ?? (
    <Button size="sm" variant="outline">
      Avaa PDF s. {pageNo}
    </Button>
  );

  return (
    <Dialog>
      <DialogTrigger render={triggerNode} />
      <DialogContent className="max-w-5xl h-[85vh] flex flex-col gap-2 p-0">
        <DialogHeader className="px-4 pt-4">
          <DialogTitle className="flex items-center gap-2 text-base">
            <span className="font-mono text-xs text-muted-foreground">{docId}</span>
            <span>· s. {pageNo}</span>
            <a
              href={url}
              target="_blank"
              rel="noreferrer"
              className="ml-auto text-xs text-primary hover:underline inline-flex items-center gap-1"
            >
              Avaa uudessa välilehdessä <ExternalLink className="h-3 w-3" />
            </a>
          </DialogTitle>
        </DialogHeader>
        <iframe
          src={url}
          className="flex-1 w-full rounded-b-lg border-t bg-muted"
          title={`PDF ${docId} s. ${pageNo}`}
        />
      </DialogContent>
    </Dialog>
  );
}
