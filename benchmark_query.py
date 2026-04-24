#!/usr/bin/env python3
"""
Usage: python benchmark_query.py [--adapters pgvector,pgvanilla,sqlite_vec,weaviate,pinecone]
Runs 200 query rounds per adapter (ANN + filtered), emits results/query_<adapter>_<ts>.json
"""
import argparse
import json
import random
from benchmark_utils import build_result, write_result, load_adapters

ROUNDS = 200
TOP_K = 10
DEFAULT_ADAPTERS = ["pgvector", "sqlite_vec", "weaviate"]


def run_queries(adapter_name: str, adapter, records: list[dict], rounds: int, label: str = "baseline") -> list[dict]:
    vectors = [r["embedding"] for r in records]
    categories = list({r["category"] for r in records})

    adapter.setup()
    results = []

    latencies_ann = []
    for _ in range(rounds):
        v = random.choice(vectors)
        _, ms = adapter.query(v, top_k=TOP_K)
        latencies_ann.append(ms)
    results.append(build_result(adapter_name, "query_ann", latencies_ann, clients=1, label=label))

    latencies_filt = []
    for _ in range(rounds):
        v = random.choice(vectors)
        cat = random.choice(categories)
        _, ms = adapter.query(v, top_k=TOP_K, filters={"category": cat})
        latencies_filt.append(ms)
    results.append(build_result(adapter_name, "query_filtered", latencies_filt, clients=1, label=label))

    adapter.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", default=",".join(DEFAULT_ADAPTERS))
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    with open(args.dataset) as f:
        records = [json.loads(line) for line in f if line.strip()]

    adapter_names = [a.strip() for a in args.adapters.split(",")]
    adapters = load_adapters(adapter_names)

    for name, adapter in adapters.items():
        results = run_queries(name, adapter, records, args.rounds, label=args.label)
        for r in results:
            path = write_result(r, prefix="query")
            print(f"[{name}][{r['operation']}] p50={r['p50_ms']:.1f}ms p99={r['p99_ms']:.1f}ms → {path}")


if __name__ == "__main__":
    main()
