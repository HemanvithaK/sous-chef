import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.recipes.loader import load_recipe


URL_CASES = [
    "https://www.bonappetit.com/recipe/bas-best-chocolate-chip-cookies",
    "https://www.simplyrecipes.com/recipes/homemade_pizza/",
    "https://cooking.nytimes.com/recipes/1015178-plum-torte",
]

RECIPE_NAME_CASES = [
    "chicken biryani",
    "chocolate chip cookies",
    "pasta aglio e olio",
    "butter chicken",
    "sourdough pancakes",
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


async def test_urls():
    print("=" * 68)
    print("URL PARSING (direct URL, no search)")
    print("=" * 68)

    for url in URL_CASES:
        print(f"\nURL: {url}")
        try:
            recipe = await load_recipe(url)
            if recipe:
                print(f"  RESULT: {recipe.name}")
                print(f"  Parser: {recipe.parser_used}")
                print(f"  Source: {recipe.source}")
                print(f"  Ingredients: {len(recipe.ingredients)}, Steps: {len(recipe.steps)}")
            else:
                print("  FAILED even after search fallback")
        except Exception as e:
            print(f"  ERROR: {e}")


async def test_recipe_names():
    print("\n" + "=" * 68)
    print("RECIPE NAME SEARCH (no URL, search the web)")
    print("=" * 68)

    for name in RECIPE_NAME_CASES:
        print(f"\nQuery: {name}")
        try:
            recipe = await load_recipe(name)
            if recipe:
                print(f"  RESULT: {recipe.name}")
                print(f"  Parser: {recipe.parser_used}")
                print(f"  Source: {recipe.source}")
                print(f"  Ingredients: {len(recipe.ingredients)}, Steps: {len(recipe.steps)}")
            else:
                print("  FAILED - no working recipe found")
        except Exception as e:
            print(f"  ERROR: {e}")


async def test_raw_text():
    print("\n" + "=" * 68)
    print("RAW TEXT (LLM extraction, no search)")
    print("=" * 68)

    recipe = await load_recipe(RAW_TEXT_CASE)
    if recipe:
        print(f"  RESULT: {recipe.name}")
        print(f"  Parser: {recipe.parser_used}")
        print(f"  Ingredients: {len(recipe.ingredients)}, Steps: {len(recipe.steps)}")
    else:
        print("  FAILED")


async def main():
    await test_urls()
    await test_recipe_names()
    await test_raw_text()


if __name__ == "__main__":
    asyncio.run(main())