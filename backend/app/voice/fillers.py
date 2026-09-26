import random

ACK_PHRASES = [
    "One sec.",
    "Got it.",
    "Let me check.",
    "On it.",
    "Sure thing.",
]

SEARCH_PHRASES = [
    "Let me find that recipe.",
    "Looking that up now.",
    "Finding a good one for you.",
]


def random_ack() -> str:
    return random.choice(ACK_PHRASES)


def random_search_ack() -> str:
    return random.choice(SEARCH_PHRASES)