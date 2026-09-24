import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.retriever import (
    search_substitutions,
    search_substitutions_verified,
    close_retriever,
)


POSITIVE_CASES = [
    {"query": "I don't have fish sauce", "expected": "fish sauce"},
    {"query": "out of buttermilk for pancakes", "expected": "buttermilk"},
    {"query": "no heavy cream, what else works", "expected": "heavy cream"},
    {"query": "vegan alternative to eggs in baking", "expected": "eggs"},
    {"query": "what can thicken a sauce instead of cornstarch", "expected": "cornstarch"},
    {"query": "substitute for thai basil", "expected": "thai basil"},
    {"query": "I ran out of butter", "expected": "butter"},
    {"query": "no white wine for deglazing", "expected": "white wine"},
    {"query": "something instead of parmesan cheese", "expected": "parmesan"},
    {"query": "gluten free soy sauce option", "expected": "soy sauce"},
    {"query": "I need acid but no lemons", "expected": "lemon juice"},
    {"query": "dairy free sour cream", "expected": "sour cream"},
    {"query": "no baking powder in the cupboard", "expected": "baking powder"},
    {"query": "out of shallots", "expected": "shallot"},
    {"query": "replacement for brown sugar in cookies", "expected": "brown sugar"},
    {"query": "what makes things spicy besides red pepper flakes", "expected": "red pepper flakes"},
    {"query": "no mirin for teriyaki", "expected": "mirin"},
    {"query": "tahini alternative for hummus", "expected": "tahini"},
    {"query": "korean chili paste substitute", "expected": "gochujang"},
    {"query": "out of vanilla", "expected": "vanilla extract"},
]

OUT_OF_CORPUS_CASES = [
    "substitute for saffron",
    "what can I use instead of tamarind paste",
    "no star anise, what else",
    "out of pomegranate molasses",
    "replacement for fenugreek leaves",
    "I don't have kaffir lime leaves",
    "substitute for nori sheets",
    "out of espresso powder for brownies",
]

OFF_TOPIC_CASES = [
    "how do I sharpen a knife",
    "what temperature to roast a chicken",
    "motor oil substitute",
    "how long does pasta take to boil",
]


def stage1_recall():
    print("=" * 70)
    print("STAGE 1 -- RETRIEVAL RECALL (verifier off)")
    print("Goal: the right answer must be in the candidates")
    print("=" * 70)

    hits = 0
    for case in POSITIVE_CASES:
        results = search_substitutions(case["query"], top_k=3)
        got = [r["ingredient"] for r in results]
        found = case["expected"] in got
        hits += found
        status = "PASS" if found else "MISS"
        print(f"[{status}] {case['query'][:42]:<42} -> {got}")

    n = len(POSITIVE_CASES)
    print(f"\nRecall@3: {hits}/{n} = {hits/n:.1%}")
    return hits / n


def stage2_positives():
    print("\n" + "=" * 70)
    print("STAGE 2 -- END TO END (verifier on)")
    print("Goal: correct ingredient survives verification")
    print("=" * 70)

    correct = 0
    latencies = []

    for case in POSITIVE_CASES:
        t0 = time.time()
        result = search_substitutions_verified(case["query"], top_k=3)
        latencies.append(time.time() - t0)

        verified = [r["ingredient"] for r in result["verified"]]
        ok = case["expected"] in verified
        correct += ok

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['query'][:42]:<42} -> {verified} [{result['stage']}]")

    n = len(POSITIVE_CASES)
    avg_ms = sum(latencies) / len(latencies) * 1000
    print(f"\nEnd-to-end accuracy: {correct}/{n} = {correct/n:.1%}")
    print(f"Average latency: {avg_ms:.0f} ms")
    return correct / n, avg_ms


def stage2_rejections(cases, label):
    print("\n" + "=" * 70)
    print(f"STAGE 2 -- {label} (verifier on)")
    print("Goal: return nothing")
    print("=" * 70)

    correct = 0
    for query in cases:
        result = search_substitutions_verified(query, top_k=3)
        verified = [r["ingredient"] for r in result["verified"]]
        retrieved = [c["ingredient"] for c in result["candidates"]]

        if not verified:
            correct += 1
            print(f"[PASS] {query[:42]:<42} -> rejected (retrieval had {retrieved})")
        else:
            print(f"[FAIL] {query[:42]:<42} -> {verified}")

    n = len(cases)
    print(f"\n{label} rejection: {correct}/{n} = {correct/n:.1%}")
    return correct / n


if __name__ == "__main__":
    recall = stage1_recall()
    accuracy, latency = stage2_positives()
    ooc = stage2_rejections(OUT_OF_CORPUS_CASES, "OUT-OF-CORPUS")
    off = stage2_rejections(OFF_TOPIC_CASES, "OFF-TOPIC")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Stage 1 recall@3:          {recall:.1%}")
    print(f"Stage 2 accuracy:          {accuracy:.1%}")
    print(f"Out-of-corpus rejection:   {ooc:.1%}")
    print(f"Off-topic rejection:       {off:.1%}")
    print(f"Avg latency per query:     {latency:.0f} ms")

    close_retriever()