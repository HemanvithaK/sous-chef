import re
from dataclasses import dataclass

from app.recipes.parser import ParsedRecipe


TIME_PATTERN = re.compile(
    r"\b\d+\s*(?:to\s*\d+\s*)?(?:second|sec|minute|min|hour|hr)s?\b"
    r"|\b(?:thirty|forty|fifty|sixty|ten|fifteen|twenty|five|two|three|four)\s+"
    r"(?:second|minute|hour)s?\b",
    re.IGNORECASE,
)

TEMP_PATTERN = re.compile(
    r"\b\d{2,3}\s*(?:degrees|°|f\b|c\b)"
    r"|\b(?:low|medium|high|medium-low|medium-high)\s+heat\b"
    r"|\bpreheat\b",
    re.IGNORECASE,
)

DONENESS_PATTERN = re.compile(
    r"\b(?:until|when)\s+\w+"
    r"|\bgolden\b|\btender\b|\bfragrant\b|\bbrowned?\b|\bsoft(?:ened)?\b"
    r"|\bthickened\b|\bbubbl\w+\b|\bcrisp\w*\b|\bwilted?\b|\bset\b",
    re.IGNORECASE,
)

VAGUE_STEPS = re.compile(
    r"^\s*(?:see (?:notes|above|below)|as (?:directed|needed)|"
    r"repeat|continue|follow (?:the )?(?:package|instructions)|enjoy|serve)\s*\.?\s*$",
    re.IGNORECASE,
)


@dataclass
class QualityReport:
    passed: bool
    score: float
    step_count: int
    avg_step_length: int
    has_times: bool
    has_temps: bool
    has_doneness: bool
    ingredient_count: int
    reasons: list[str]
    needs_llm_check: bool


def assess_quality(recipe: ParsedRecipe) -> QualityReport:
    steps = [s for s in recipe.steps if s and s.strip()]
    step_count = len(steps)
    ingredient_count = len(recipe.ingredients)

    if step_count == 0:
        return QualityReport(
            passed=False, score=0.0, step_count=0, avg_step_length=0,
            has_times=False, has_temps=False, has_doneness=False,
            ingredient_count=ingredient_count,
            reasons=["no steps at all"], needs_llm_check=False,
        )

    total_len = sum(len(s) for s in steps)
    avg_len = total_len // step_count

    all_text = " ".join(steps)
    has_times = bool(TIME_PATTERN.search(all_text))
    has_temps = bool(TEMP_PATTERN.search(all_text))
    has_doneness = bool(DONENESS_PATTERN.search(all_text))

    vague_count = sum(1 for s in steps if VAGUE_STEPS.match(s))

    reasons = []
    score = 0.0

    if avg_len >= 100:
        score += 0.35
    elif avg_len >= 60:
        score += 0.20
    elif avg_len >= 35:
        score += 0.08
    else:
        reasons.append(f"steps too short (avg {avg_len} chars)")

    if step_count >= 5:
        score += 0.20
    elif step_count >= 3:
        score += 0.10
    else:
        reasons.append(f"only {step_count} steps")

    if has_times:
        score += 0.15
    else:
        reasons.append("no timing cues")

    if has_temps:
        score += 0.10
    else:
        reasons.append("no heat or temperature cues")

    if has_doneness:
        score += 0.15
    else:
        reasons.append("no doneness cues")

    if ingredient_count >= 4:
        score += 0.05
    else:
        reasons.append(f"only {ingredient_count} ingredients")

    if vague_count > 0:
        score -= 0.15 * vague_count
        reasons.append(f"{vague_count} vague or placeholder steps")

    score = max(0.0, min(1.0, score))

    passed = score >= 0.55
    needs_llm_check = 0.35 <= score < 0.55

    return QualityReport(
        passed=passed,
        score=round(score, 2),
        step_count=step_count,
        avg_step_length=avg_len,
        has_times=has_times,
        has_temps=has_temps,
        has_doneness=has_doneness,
        ingredient_count=ingredient_count,
        reasons=reasons,
        needs_llm_check=needs_llm_check,
    )