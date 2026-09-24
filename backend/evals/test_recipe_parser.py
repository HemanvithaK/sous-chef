import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.recipes.loader import load_recipe


URL_CASES = [
    {
        "url": "https://www.bonappetit.com/recipe/bas-best-chocolate-chip-cookies",
        "expected_keywords": ["chocolate chip", "cookie"],
    },
    {
        "url": "https://www.simplyrecipes.com/recipes/homemade_pizza/",
        "expected_keywords": ["pizza"],
    },
    {
        "url": "https://cooking.nytimes.com/recipes/1015178-plum-torte",
        "expected_keywords": ["plum", "torte"],
    },
]

RECIPE_NAME_CASES = [
    {"query": "chicken biryani", "expected_keywords": ["biryani"]},
    {"query": "chocolate chip cookies", "expected_keywords": ["chocolate chip", "cookie"]},
    {"query": "pasta aglio e olio", "expected_keywords": ["aglio", "olio"]},
    {"query": "butter chicken", "expected_keywords": ["butter chicken", "makhani"]},
    {"query": "sourdough pancakes", "expected_keywords": ["sourdough"]},
    {"query": "beef wellington", "expected_keywords": ["wellington"]},
    {"query": "pad thai", "expected_keywords": ["pad thai"]},
]

RAW_TEXT_CASE = """
Simple Scrambled Eggs

Serves 1. Takes about 5 minutes.

You need:
- 3 large eggs
- 1 tablespoon butter
- Salt and pepper to taste

Instructions:
1. Crack the eggs into a bowl and whisk until fully combined.
2. Melt butter in a nonstick pan over medium-low heat.
3. Pour in eggs. Let them sit for 20 seconds, then push gently with a spatula.
4. When eggs look about 70 percent set but still glossy, pull the pan off heat.
5. Season with salt and pepper. Serve immediately.
"""


def check_match(recipe_name: str, expected_keywords: list[str]) -> tuple[bool, str]:
    name_lower = recipe_name.lower()
    for keyword in expected_keywords:
        if keyword.lower() in name_lower:
            return True, f"matched '{keyword}'"
    return False, f"none of {expected_keywords} found in '{recipe_name}'"


async def test_urls():
    print("=" * 70)
    print("URL PARSING (direct URL, no relevance check needed)")
    print("=" * 70)

    passed = 0
    for case in URL_CASES:
        url = case["url"]
        print(f"\nURL: {url}")
        try:
            recipe = await load_recipe(url)
            if not recipe:
                print("  FAILED - no recipe returned")
                continue

            matched, note = check_match(recipe.name, case["expected_keywords"])
            status = "PASS" if matched else "WRONG"
            print(f"  [{status}] {recipe.name} ({note})")
            print(f"  Parser: {recipe.parser_used}, Source: {recipe.source}")
            if matched:
                passed += 1

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nURL cases: {passed}/{len(URL_CASES)} correct")
    return passed, len(URL_CASES)


async def test_recipe_names():
    print("\n" + "=" * 70)
    print("RECIPE NAME SEARCH (verifier active)")
    print("=" * 70)

    passed = 0
    silent_wrong = 0
    honest_failures = 0

    for case in RECIPE_NAME_CASES:
        query = case["query"]
        print(f"\nQuery: {query}")
        try:
            recipe = await load_recipe(query)
            if not recipe:
                print("  [HONEST FAIL] No recipe found or all were rejected")
                honest_failures += 1
                continue

            matched, note = check_match(recipe.name, case["expected_keywords"])
            if matched:
                print(f"  [PASS] {recipe.name} ({note})")
                passed += 1
            else:
                print(f"  [SILENT WRONG] {recipe.name} ({note})")
                silent_wrong += 1

        except Exception as e:
            print(f"  ERROR: {e}")

    total = len(RECIPE_NAME_CASES)
    print(f"\nRecipe name cases:")
    print(f"  Correct:       {passed}/{total}")
    print(f"  Silent wrong:  {silent_wrong}/{total}  <- the metric that matters most")
    print(f"  Honest fail:   {honest_failures}/{total}  <- acceptable, better than wrong")
    return passed, silent_wrong, honest_failures, total


async def test_raw_text():
    print("\n" + "=" * 70)
    print("RAW TEXT (LLM extraction, no search)")
    print("=" * 70)

    recipe = await load_recipe(RAW_TEXT_CASE)
    if recipe:
        print(f"  RESULT: {recipe.name}")
        print(f"  Ingredients: {len(recipe.ingredients)}, Steps: {len(recipe.steps)}")
        return True
    else:
        print("  FAILED")
        return False


async def main():
    url_passed, url_total = await test_urls()
    name_passed, silent_wrong, honest_fail, name_total = await test_recipe_names()
    raw_ok = await test_raw_text()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"URL parsing:       {url_passed}/{url_total} correct")
    print(f"Recipe search:     {name_passed}/{name_total} correct")
    print(f"  silent wrong:    {silent_wrong}  <- must be 0 for production")
    print(f"  honest failures: {honest_fail}  <- acceptable")
    print(f"Raw text parsing:  {'PASS' if raw_ok else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())