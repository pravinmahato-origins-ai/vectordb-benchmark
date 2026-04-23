#!/usr/bin/env python3
import argparse
import json
import random
import uuid
from datetime import datetime, timezone

import numpy as np

CATEGORIES = [f"cat_{i}" for i in range(10)]
DIMENSION = 1536
DEFAULT_COUNT = 1700

SENTENCE_TEMPLATES = [
    "The {adj} {noun} quickly {verb} the {obj}.",
    "A {adj} researcher discovered a new {noun} near the {obj}.",
    "Several {noun} units were deployed to {verb} the {obj}.",
]
ADJS = ["efficient", "robust", "adaptive", "scalable", "distributed"]
NOUNS = ["system", "cluster", "model", "vector", "index", "pipeline"]
VERBS = ["optimised", "evaluated", "indexed", "transformed", "queried"]
OBJS = ["dataset", "embedding", "shard", "replica", "query"]


def random_sentence() -> str:
    tpl = random.choice(SENTENCE_TEMPLATES)
    return tpl.format(
        adj=random.choice(ADJS),
        noun=random.choice(NOUNS),
        verb=random.choice(VERBS),
        obj=random.choice(OBJS),
    )


def unit_vector(dim: int) -> list[float]:
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--output", default="dataset.jsonl")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    with open(args.output, "w") as f:
        for _ in range(args.count):
            record = {
                "id": str(uuid.uuid4()),
                "embedding": unit_vector(DIMENSION),
                "text": random_sentence(),
                "category": random.choice(CATEGORIES),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {args.count} records to {args.output}")


if __name__ == "__main__":
    main()
