from typing import Optional

from app.recipes.parser import (
    ParsedRecipe,
    extract_url,
    fetch_html,
    html_to_text,
    is_url,
    try_scraper,
)
from app.recipes.llm_extractor import extract_recipe
from app.recipes.search import search_recipe


MAX_SEARCH_ATTEMPTS = 5


async def load_recipe(user_input: str) -> Optional[ParsedRecipe]:
    text = user_input.strip()

    if is_url(text):
        recipe = await _load_from_url(text)
        if recipe:
            return recipe
        return await _search_and_load(text)

    embedded_url = extract_url(text)
    if embedded_url:
        recipe = await _load_from_url(embedded_url)
        if recipe:
            return recipe

    recipe = await extract_recipe(text, source="user_input")
    if recipe:
        return recipe

    return await _search_and_load(text)


async def _load_from_url(url: str) -> Optional[ParsedRecipe]:
    scraped = await try_scraper(url)
    if scraped:
        return scraped

    try:
        html = await fetch_html(url)
    except Exception:
        return None

    text = html_to_text(html)
    if not text:
        return None

    return await extract_recipe(text, source=url)


async def _search_and_load(query: str) -> Optional[ParsedRecipe]:
    from app.recipes.verifier import verify_recipe_match

    print(f"Searching web for: {query}")
    urls = await search_recipe(query, max_results=MAX_SEARCH_ATTEMPTS)

    if not urls:
        print("Search returned no results")
        return None

    print(f"Search returned {len(urls)} URLs, trying each...")

    for i, url in enumerate(urls, 1):
        print(f"  [{i}/{len(urls)}] Trying {url}")
        recipe = await _load_from_url(url)

        if not recipe:
            continue

        print(f"    Parsed via {recipe.parser_used}: {recipe.name}")

        is_match = await verify_recipe_match(query, recipe)
        if is_match:
            print(f"  Verified match, using this recipe")
            return recipe
        else:
            print(f"  Rejected by verifier, trying next")

    print("All search results either failed to parse or were rejected as non-matches")
    return None