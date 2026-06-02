# Extraction Prompt — Prototype Verdict

**Question answered:** What extraction prompt reliably produces verbatim spans
(payloads intact, code blocks whole, distractors dropped, intra-chunk trimmed)
from a local LLM given realistic RAG chunks + query + task_intent?

**Answer:** The 7-rule prompt in `_proto_extract.py` (v4) passes all criteria.

## Validated prompt rules (copy these into the production Extractor)

1. Copy spans VERBATIM — character-for-character, all punctuation preserved.
2. Each code block (``` ... ```) must be copied whole — never truncate mid-string.
3. Within a chunk, select only the code blocks and surrounding sentences that
   directly address the task intent. Stop before any code block whose operation
   is unrelated to the task.
4. If a chunk contains multiple relevant code blocks, include all of them in one
   span (preserve any brief prose between them so the span is contiguous).
5. Omit any chunk with no material directly relevant to the task intent.
6. Output a JSON array. Each element: exactly two keys — "doc_id" and "span".
7. No extra keys, no commentary, no markdown outside the JSON.

## What the tests showed

- **Scenario A** (direct login bypass): model selected `sqli-basics.md` only,
  included both the string-tautology block AND the numeric-parameter block once
  rule 4 was present. Dropped XSS, URL-encoding, union, and error-based chunks.

- **Scenario B** (no visible output — confirm injectable then bypass): model
  selected two chunks. `blind-sqli.md` span contained the SLEEP block only —
  correctly excluded the boolean-extraction block and the sqlmap `--dbs` block
  (rule 3). `sqli-basics.md` span as above.

- **Verbatim check**: all spans passed mid-span fingerprint check in both runs.

## Prompt iteration log

| Version | Change | Result |
|---------|--------|--------|
| v1 | Initial 6-rule prompt | 1 span, 1 code block from sqli-basics.md. Distractors dropped. |
| v2 | Added "include ALL relevant code blocks" rule | No change — model still selected 1 block. |
| v3 | Added Scenario B (two-chunk test) | Scenario B: 2 chunks ✓ but blind-sqli.md span over-included sqlmap block. |
| v4 | Added intra-chunk trim rule (stop before unrelated blocks) | Both scenarios pass all checks. |

## Production notes

- Output budget enforcement (ADR-0002 §2) is NOT in this prompt — add it at the
  call site, not inside the prompt. Truncating a span mid-payload is forbidden;
  enforce at span-emit time, not inside the LLM instruction.
- The prompt works on the model used in this project's `LLM_DEFAULT_MODEL`.
  Re-validate if switching to a weaker model (8B range).
- Graceful degradation (timeout → return raw top-k) belongs in the calling code,
  not in the prompt.
