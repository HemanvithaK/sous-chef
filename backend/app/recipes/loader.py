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


async def load_recipe(user_input: str) -> Optional[ParsedRecipe]:
    text = user_input.strip()

    if is_url(text):
        return await _load_from_url(text)

    url = extract_url(text)
    if url:
        recipe = await _load_from_url(url)
        if recipe:
            return recipe

    return await extract_recipe(text, source="user_input")


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