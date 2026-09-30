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


async def load_recipe(
    user_input: str,
    constraints: list[str] | None = None,
) -> Optional[ParsedRecipe]:
    text = user_input.strip()

    if is_url(text):
        recipe = await _load_from_url(text)
        if recipe:
            return recipe
        return await _search_and_load(text, constraints=constraints)

    embedded_url = extract_url(text)
    if embedded_url:
        recipe = await _load_from_url(embedded_url)
        if recipe:
            return recipe

    recipe = await extract_recipe(text, source="user_input")
    if recipe:
        return recipe

    return await _search_and_load(text, constraints=constraints)


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


async def _search_and_load(
    query: str,
    constraints: list[str] | None = None,
) -> Optional[ParsedRecipe]:
    from app.recipes.verifier import verify_recipe_match, judge_recipe_quality
    from app.recipes.quality import assess_quality

    if constraints:
        print(f"Searching web for: {query} (constraints: {', '.join(constraints)})")
    else:
        print(f"Searching web for: {query}")

    urls = await search_recipe(
        query, max_results=MAX_SEARCH_ATTEMPTS, constraints=constraints
    )

    if not urls:
        print("Search returned no results")
        return None

    print(f"Search returned {len(urls)} URLs, trying each...")
    best_fallback = None
    best_score = 0.0

    for i, url in enumerate(urls, 1):
        print(f"  [{i}/{len(urls)}] Trying {url}")
        recipe = await _load_from_url(url)
        if not recipe:
            continue

        print(f"    Parsed via {recipe.parser_used}: {recipe.name}")

        is_match = await verify_recipe_match(query, recipe, constraints=constraints)
        if not is_match:
            print("    Rejected: not a match")
            continue

        report = assess_quality(recipe)
        print(
            f"    Quality: {report.score} "
            f"({report.step_count} steps, avg {report.avg_step_length} chars)"
        )
        if report.reasons:
            print(f"      issues: {', '.join(report.reasons)}")

        if report.passed:
            print("    Accepted: relevant and detailed")
            return recipe

        if report.needs_llm_check:
            if await judge_recipe_quality(recipe):
                print("    Accepted: borderline but judged usable")
                return recipe

        print("    Rejected: too thin, trying next")
        if report.score > best_score:
            best_fallback = recipe
            best_score = report.score

    if best_fallback:
        print(f"All results were thin. Using best available (score {best_score}).")
        return best_fallback

    print("No usable recipe found")
    return None