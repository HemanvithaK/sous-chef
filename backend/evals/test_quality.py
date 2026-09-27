import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.recipes.parser import ParsedRecipe
from app.recipes.quality import assess_quality


THIN = ParsedRecipe(
    name="Fettuccine Alfredo",
    ingredients=["cream", "butter", "parmesan", "fettuccine"],
    steps=[
        "Thicken the cream.",
        "Add the cheese.",
        "Toss with pasta.",
        "Serve.",
    ],
    servings=4, total_minutes=30, source="test", parser_used="test",
)

GOOD = ParsedRecipe(
    name="Spaghetti Aglio e Olio",
    ingredients=["8 oz spaghetti", "6 cloves garlic", "1/3 cup olive oil",
                 "1/2 tsp red pepper flakes", "1/4 cup parsley", "salt"],
    steps=[
        "Bring a large pot of well-salted water to a rolling boil, then add the "
        "spaghetti and cook one minute short of the package time.",
        "While the pasta cooks, heat the olive oil in a large skillet over "
        "medium-low heat, add the sliced garlic and red pepper flakes, and cook "
        "slowly for about four to five minutes until the garlic is golden.",
        "Before draining, scoop out about half a cup of the starchy pasta water.",
        "Add the drained pasta to the garlic oil with a splash of pasta water and "
        "toss aggressively over medium heat for about a minute until the sauce "
        "turns silky and coats every strand.",
        "Kill the heat, toss in the parsley, and finish with a drizzle of olive oil.",
    ],
    servings=2, total_minutes=20, source="test", parser_used="test",
)

BORDERLINE = ParsedRecipe(
    name="Simple Omelette",
    ingredients=["3 eggs", "butter", "salt", "cheese"],
    steps=[
        "Whisk the eggs with a pinch of salt.",
        "Melt butter in a pan over medium heat.",
        "Pour in eggs and cook until the edges set.",
        "Add cheese, fold, and slide onto a plate.",
    ],
    servings=1, total_minutes=10, source="test", parser_used="test",
)


def show(label, recipe):
    r = assess_quality(recipe)
    verdict = "PASS" if r.passed else ("LLM CHECK" if r.needs_llm_check else "REJECT")
    print(f"\n{label}: {recipe.name}")
    print(f"  Verdict: {verdict}  (score {r.score})")
    print(f"  Steps: {r.step_count}, avg length {r.avg_step_length}")
    print(f"  times={r.has_times} temps={r.has_temps} doneness={r.has_doneness}")
    if r.reasons:
        print(f"  Issues: {', '.join(r.reasons)}")


if __name__ == "__main__":
    print("=" * 60)
    print("QUALITY HEURISTIC CALIBRATION")
    print("=" * 60)
    show("THIN (should REJECT)", THIN)
    show("GOOD (should PASS)", GOOD)
    show("BORDERLINE (either is fine)", BORDERLINE)