import os
from typing import Optional

from dotenv import load_dotenv
from tavily import AsyncTavilyClient

load_dotenv()


ALLOWED_RECIPE_DOMAINS = [
    "allrecipes.com",
    "simplyrecipes.com",
    "food.com",
    "epicurious.com",
    "foodnetwork.com",
    "bonappetit.com",
    "kingarthurbaking.com",
    "myrecipes.com",
    "tasteofhome.com",
    "delish.com",
    "eatingwell.com",
    "cookieandkate.com",
    "budgetbytes.com",
    "smittenkitchen.com",
    "onceuponachef.com",
    "loveandlemons.com",
    "minimalistbaker.com",
    "recipetineats.com",
    "themediterraneandish.com",
    "gimmesomeoven.com",
]


class RecipeSearcher:
    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise RuntimeError("TAVILY_API_KEY not set in environment")
            self._client = AsyncTavilyClient(api_key=api_key)
        return self._client

    async def search(self, recipe_query: str, max_results: int = 5) -> list[str]:
        client = self._ensure_client()

        search_query = f"{recipe_query} recipe ingredients instructions"

        try:
            response = await client.search(
                query=search_query,
                search_depth="basic",
                max_results=max_results,
                include_domains=ALLOWED_RECIPE_DOMAINS,
            )
            results = response.get("results", [])
            return [r["url"] for r in results if "url" in r]

        except Exception as e:
            print(f"Tavily search error: {e}")
            return []


_searcher = RecipeSearcher()


async def search_recipe(recipe_query: str, max_results: int = 5) -> list[str]:
    return await _searcher.search(recipe_query, max_results=max_results)