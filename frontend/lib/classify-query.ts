// Deterministic classifier that routes user questions between the RAG
// path (`/v1/query`) and the aggregate SQL path (`/v1/aggregate`).
//
// We prefer a regex list over an LLM call for two reasons:
//  1. The rule set is small and the questions are predictable in Finnish
//     municipal vocabulary ("kuinka monta", "paljonko rahaa", ...).
//  2. Routing must be instantaneous; no second network hop before the
//     user sees "loading".

export type QueryClass = "rag" | "aggregate";

// Ordering matters: the first pattern that matches wins. Aggregate-only
// phrases come first, then monetary patterns that would otherwise also
// trigger on the RAG path.
const AGGREGATE_PATTERNS: ReadonlyArray<RegExp> = [
  /\bkuinka\s+monta\b/i,
  /\bkuinka\s+monessa\b/i,
  /\bmontako\b/i,
  /\blukum[aä]{1,2}r[aä]\b/i,
  /\bkuinka\s+paljon\s+(rahaa|euroa|maksoi|kustansi|budjetoitiin)\b/i,
  /\bpaljonko\s+(rahaa|euroa|maksoi|kustansi)\b/i,
  /\byhteens[aä]\s+(euroa|€|\d)/i,
  /\bkokonaissumma\b/i,
  /\bkuinka\s+moni\s+(henkil[oö]|j[aä]sen|valtuutettu)\b/i,
];

export function classifyQuery(query: string): QueryClass {
  const trimmed = query.trim();
  if (trimmed.length === 0) return "rag";
  for (const pattern of AGGREGATE_PATTERNS) {
    if (pattern.test(trimmed)) return "aggregate";
  }
  return "rag";
}
