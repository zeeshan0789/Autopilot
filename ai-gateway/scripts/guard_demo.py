import numpy as np

from app.cache.embedder import Embedder
from app.cache.guard import check_guard, is_time_sensitive

# For display only. The real threshold is tuned on a labeled sample in step P2.6.
ILLUSTRATIVE_THRESHOLD = 0.85

# (new prompt, cached prompt, should the cache be allowed to reuse the answer?)
PAIRS = [
    ("Which city is the capital of France?", "What is the capital of France?", True),
    ("I forgot my password, how can I change it?", "How do I reset my password?", True),
    ("What was Apple revenue in 2025?", "What was Apple revenue in 2024?", False),
    ("What is the capital of Germany?", "What is the capital of France?", False),
    ("Convert 200 USD to PKR", "Convert 100 USD to PKR", False),
    ("Is it not safe to eat raw eggs?", "Is it safe to eat raw eggs?", False),
    ("Top ten movies of all time", "Top five movies of all time", False),
]


def main():
    embedder = Embedder()
    print(f"Illustrative threshold: {ILLUSTRATIVE_THRESHOLD}")
    print()

    for new_text, cached_text, reusable in PAIRS:
        similarity = float(np.dot(embedder.encode(new_text), embedder.encode(cached_text)))
        guard = check_guard(new_text, cached_text)

        by_threshold = similarity >= ILLUSTRATIVE_THRESHOLD
        final = by_threshold and guard.ok

        print(f"similarity {similarity:.3f}")
        print(f"  new:    {new_text}")
        print(f"  cached: {cached_text}")
        print(f"  threshold alone: {'HIT' if by_threshold else 'miss'}")
        print(f"  guard: {guard.reason}")
        print(f"  final: {'HIT' if final else 'miss'}   (answer should be reusable: {reusable})")
        if final and not reusable:
            print("  !!! FALSE HIT")
        print()

    for text in ["What is the weather today?", "What is the capital of France?"]:
        print(f"time sensitive={is_time_sensitive(text)}  {text}")


main()