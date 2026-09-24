import json
import os
from typing import Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from app.recipes.parser import ParsedRecipe, extract_json_object


load_dotenv()

EXTRACTOR_MODEL = "claude-haiku-4-5-20251001"

EXTRACTOR_PROMPT = """You extract structured recipe data from unstructured text \
(recipe blog posts, pasted recipes, printed instructions).

Return a JSON object with this exact shape:
{
  "name": "recipe title as string",
  "ingredients": ["ingredient 1 with quantity", "ingredient 2 with quantity", ...],
  "steps": ["step 1 as complete sentence", "step 2 as complete sentence", ...],
  "servings": integer or null,
  "total_minutes": integer or null
}

Rules:
- Ingredients: keep quantities and units together. "2 tablespoons olive oil", not \
"olive oil" alone.
- Steps: one action per step. If the source combines "chop the onion and heat the \
oil" into one paragraph, split them. Rewrite for clarity if the source is confusing.
- Steps must be readable aloud as-is. Avoid abbreviations ("tbsp" → "tablespoon"), \
symbols ("350°F" → "three hundred fifty degrees Fahrenheit"), and bullet points.
- If a field is genuinely missing from the source (no serving size given, no total \
time listed), use null. Do not guess.
- Ignore ads, comments, story text, related-recipe links, and other blog cruft. \
Extract only the recipe itself.

If the text does not contain a recognizable recipe at all, return:
{"error": "no recipe found"}

Output the JSON object and nothing else. No preamble, no explanation, no code \
fences."""


class LLMExtractor:
    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    async def extract(self, raw_text: str, source: str) -> Optional[ParsedRecipe]:
        if not raw_text or len(raw_text.strip()) < 50:
            return None

        truncated = raw_text[:15000]

        try:
            client = self._ensure_client()
            response = await client.messages.create(
                model=EXTRACTOR_MODEL,
                max_tokens=2000,
                system=EXTRACTOR_PROMPT,
                messages=[{"role": "user", "content": truncated}],
            )
            raw = response.content[0].text
            parsed = extract_json_object(raw)

            if "error" in parsed:
                return None

            if not parsed.get("ingredients") or not parsed.get("steps"):
                return None

            return ParsedRecipe(
                name=parsed.get("name") or "Untitled recipe",
                ingredients=parsed["ingredients"],
                steps=parsed["steps"],
                servings=parsed.get("servings"),
                total_minutes=parsed.get("total_minutes"),
                source=source,
                parser_used="llm",
            )

        except Exception as e:
            print(f"LLM extractor error: {e}")
            return None


_extractor = LLMExtractor()


async def extract_recipe(raw_text: str, source: str) -> Optional[ParsedRecipe]:
    return await _extractor.extract(raw_text, source)