import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

VERIFIER_MODEL = "claude-haiku-4-5-20251001"

VERIFIER_PROMPT = """You are a retrieval verifier for a cooking substitution database.

IMPORTANT: Each database entry is keyed by the ingredient the user is MISSING, and \
lists what to use in its place. So the entry for "cornstarch" contains substitutes \
FOR cornstarch.

This means phrasings like "instead of X", "besides X", "alternative to X", \
"replacement for X", or "I don't have X" are all asking for the entry named X. \
Do not reject the entry for X just because the user wants to avoid X — that entry \
is exactly what answers their question.

Your job: decide which candidate entries actually address the user's question.

A candidate matches when:
- It is the ingredient the user named, in any phrasing (singular, plural, or a close \
form: "lemons" matches "lemon juice", "shallots" matches "shallot").
- It is a direct synonym of that ingredient.
- The user described a function rather than naming an ingredient (for example \
"something to thicken my sauce" or "I need an acid"), and the candidate's role is \
that function. Multiple candidates may match here — return all of them.

A candidate does NOT match when:
- It is merely in the same category. Saffron and garlic are both aromatics: not a \
match. Tamarind and soy sauce are both umami condiments: not a match. Star anise \
and thai basil both have anise notes: not a match.
- The user named an ingredient that simply is not in the candidate list.

When uncertain, do not match. A wrong substitution ruins food. Returning nothing is \
the correct answer when the database does not cover the ingredient.

Respond with ONLY a JSON object:
{"matches": [list of matching candidate indices]}

Return {"matches": []} if none match.

Output the JSON object and nothing else. No explanation, no preamble, no trailing \
commentary."""

def extract_json_object(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        char = text[i]

        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])

    raise ValueError(f"Unbalanced JSON in response: {text[:200]}")

class RetrievalVerifier:
    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    def verify(self, query: str, candidates: list[dict]) -> list[int]:
        if not candidates:
            return []

        lines = []
        for i, c in enumerate(candidates):
            lines.append(f"[{i}] {c['ingredient']} — {c.get('role', '')}")
        candidate_block = "\n".join(lines)

        user_message = (
            f"User asked: {query}\n\n"
            f"Candidates:\n{candidate_block}\n\n"
            f"Which candidates are the same ingredient the user asked about?"
        )

        try:
            client = self._ensure_client()
            response = client.messages.create(
                model=VERIFIER_MODEL,
                max_tokens=100,
                system=VERIFIER_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text
            parsed = extract_json_object(raw)
            matches = parsed.get("matches", [])
            return [i for i in matches if isinstance(i, int) and 0 <= i < len(candidates)]

        except Exception as e:
            print(f"Verifier error (failing closed): {e}")
            print(f"  raw response was: {raw[:300] if 'raw' in dir() else 'no response'}")
            return []


_verifier = RetrievalVerifier()


def verify_candidates(query: str, candidates: list[dict]) -> list[int]:
    return _verifier.verify(query, candidates)