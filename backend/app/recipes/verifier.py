import os
from typing import Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from app.recipes.parser import ParsedRecipe, extract_json_object


load_dotenv()

VERIFIER_MODEL = "claude-haiku-4-5-20251001"

VERIFIER_PROMPT = """You are checking whether a parsed recipe actually matches \
what the user asked for.

You will receive:
- The user's request (a recipe name or description)
- A recipe that was found and parsed from the web

Your job: decide whether the recipe genuinely matches the request.

A recipe matches when:
- It is the same dish, even if named slightly differently (e.g. "spaghetti aglio \
e olio" matches "aglio olio pasta", "chocolate chip cookies" matches "classic \
chocolate chip cookies").
- It is a well-known regional variant of the same dish (e.g. "Hyderabadi biryani" \
matches "chicken biryani").
- The user described the dish and this recipe delivers it.

A recipe does NOT match when:
- It is a different dish from the same category. Pikelets are not sourdough \
pancakes — they use baking powder, not a sourdough starter. Pizza and pancakes \
are both flat breads but not substitutes.
- Key ingredients the user named are absent. "Sourdough pancakes" requires a \
sourdough starter; a recipe without one is not a match.
- The core dish is different, not just supplemented. Adding kale to aglio e olio \
does NOT make it a different dish — it is a variation. But replacing chicken \
with tofu in butter chicken DOES change the dish. The test: if you removed the \
extras, would the core recipe still be what the user asked for? If yes, match.

When uncertain, do not match. A wrong recipe is worse than no recipe — the user \
will follow the wrong steps and end up with the wrong dish.

Respond with ONLY a JSON object:
{"match": true or false, "reason": "brief explanation for logs"}

Output the JSON object and nothing else. No preamble, no code fences."""


class RecipeVerifier:
    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    async def verify(self, query: str, recipe: ParsedRecipe) -> bool:
        ingredients_preview = ", ".join(recipe.ingredients[:10])
        if len(recipe.ingredients) > 10:
            ingredients_preview += f", ... ({len(recipe.ingredients) - 10} more)"

        user_message = (
            f"User asked for: {query}\n\n"
            f"Recipe found:\n"
            f"  Name: {recipe.name}\n"
            f"  Ingredients: {ingredients_preview}\n\n"
            f"Does this recipe match what the user asked for?"
        )

        try:
            client = self._ensure_client()
            response = await client.messages.create(
                model=VERIFIER_MODEL,
                max_tokens=150,
                system=VERIFIER_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text
            parsed = extract_json_object(raw)

            is_match = bool(parsed.get("match", False))
            reason = parsed.get("reason", "no reason given")

            print(f"    Verifier: match={is_match} ({reason})")
            return is_match

        except Exception as e:
            print(f"    Recipe verifier error (failing closed): {e}")
            return False


_verifier = RecipeVerifier()


async def verify_recipe_match(query: str, recipe: ParsedRecipe) -> bool:
    return await _verifier.verify(query, recipe)