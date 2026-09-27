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

QUALITY_PROMPT = """You are judging whether a recipe is detailed enough to follow \
hands-free by voice, while cooking.

The user cannot see the recipe. They only hear one step at a time. So each step \
must tell them what to do AND how to know when it is done.

A recipe is GOOD ENOUGH when:
- Steps describe concrete actions with enough detail to act on.
- There are cues for timing, heat level, or doneness somewhere in the steps.
- A reasonably confident home cook could follow it without looking anything up.

A recipe is NOT GOOD ENOUGH when:
- Steps are one-liners that assume you can see the full recipe ("thicken the \
cream", "make the sauce", "cook the chicken").
- It refers to things not included ("see notes", "follow package directions").
- Critical steps are missing entirely — it jumps from raw ingredients to finished \
dish.

Judge the recipe as a whole. A single short step among detailed ones is fine.

Respond with ONLY a JSON object:
{"good_enough": true or false, "reason": "one short sentence"}

Output the JSON and nothing else."""


class QualityJudge:
    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    async def judge(self, recipe: ParsedRecipe) -> bool:
        steps_text = "\n".join(
            f"{i}. {s}" for i, s in enumerate(recipe.steps[:12], 1)
        )
        user_message = (
            f"Recipe: {recipe.name}\n\n"
            f"Steps:\n{steps_text}\n\n"
            f"Is this detailed enough to follow by voice while cooking?"
        )

        try:
            client = self._ensure_client()
            response = await client.messages.create(
                model=VERIFIER_MODEL,
                max_tokens=150,
                system=QUALITY_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            parsed = extract_json_object(response.content[0].text)
            good = bool(parsed.get("good_enough", False))
            print(f"    Quality judge: {good} ({parsed.get('reason', '')})")
            return good
        except Exception as e:
            print(f"    Quality judge error (accepting by default): {e}")
            return True


_quality_judge = QualityJudge()


async def judge_recipe_quality(recipe: ParsedRecipe) -> bool:
    return await _quality_judge.judge(recipe)

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