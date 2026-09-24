import re
from dataclasses import dataclass, asdict
from typing import Optional

import httpx
from bs4 import BeautifulSoup
import json


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

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

URL_PATTERN = re.compile(r"https?://\S+")


@dataclass
class ParsedRecipe:
    name: str
    ingredients: list[str]
    steps: list[str]
    servings: Optional[int]
    total_minutes: Optional[int]
    source: str
    parser_used: str

    def to_dict(self) -> dict:
        return asdict(self)


def is_url(text: str) -> bool:
    text = text.strip()
    return text.startswith("http://") or text.startswith("https://")


def extract_url(text: str) -> Optional[str]:
    match = URL_PATTERN.search(text)
    return match.group(0) if match else None


async def fetch_html(url: str, timeout: float = 10.0) -> str:
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=timeout,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


async def try_scraper(url: str) -> Optional[ParsedRecipe]:
    from recipe_scrapers import scrape_html
    try:
        from recipe_scrapers import SchemaOrgException, WebsiteNotImplementedError
    except ImportError:
        from recipe_scrapers._exceptions import (
            SchemaOrgException,
            WebsiteNotImplementedError,
        )

    try:
        html = await fetch_html(url)
        scraper = scrape_html(html, org_url=url)

        return ParsedRecipe(
            name=scraper.title() or "Untitled recipe",
            ingredients=scraper.ingredients() or [],
            steps=[s.strip() for s in scraper.instructions().split("\n") if s.strip()],
            servings=_safe_int(scraper.yields()),
            total_minutes=scraper.total_time() or None,
            source=url,
            parser_used="scraper",
        )

    except WebsiteNotImplementedError:
        return None
    except SchemaOrgException:
        return None
    except Exception as e:
        print(f"Scraper unexpected error for {url}: {e}")
        return None


def _safe_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None