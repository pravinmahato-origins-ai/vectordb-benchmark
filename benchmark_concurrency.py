#!/usr/bin/env python3
"""
Usage: python benchmark_concurrency.py [--adapters ...] [--clients 1,5,10,25]
Runs 100 queries at each concurrency level using ThreadPoolExecutor.
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from benchmark_utils import build_result, write_result, load_adapters

DEFAULT_CLIENTS = [1, 5, 10, 25]
QUERIES_PER_LEVEL = 100
TOP_K = 10
DEFAULT_ADAPTERS = ["pgvector", "sqlite_vec", "weaviate"]


def run_level(adapter, vectors: list[list[float]], n_clients: int, n_queries: int) -> tuple[list[float], float]:
    def single_query(_):
        v = random.choice(vectors)
        _, ms = adapter.query(v, top_k=TOP_K)
        return ms

    latencies = []
    t_wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_clients) as pool:
        futures = [pool.submit(single_query, i) for i in range(n_queries)]
        for f in as_completed(futures):
            latencies.append(f.result())
    wall_time_s = time.perf_counter() - t_wall_start
    return latencies, wall_time_s


def run_concurrency(adapter_name: str, adapter, records: list[dict], client_levels: list[int], label: str = "baseline") -> list[dict]:
    vectors = [r["embedding"] for r in records]

    adapter.setup()
    results = []
    for n in client_levels:
        latencies, wall_time_s = run_level(adapter, vectors, n_clients=n, n_queries=QUERIES_PER_LEVEL)
        r = build_result(adapter_name, "concurrency", latencies, clients=n, wall_time_s=wall_time_s, label=label)
        results.append(r)
        print(f"  [{adapter_name}] clients={n} p50={r['p50_ms']:.1f}ms p99={r['p99_ms']:.1f}ms qps={r['qps']}")
    adapter.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", default=",".join(DEFAULT_ADAPTERS))
    parser.add_argument("--clients", default="1,5,10,25")
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    client_levels = [int(c) for c in args.clients.split(",")]

    with open(args.dataset) as f:
        records = [json.loads(line) for line in f if line.strip()]

    adapter_names = [a.strip() for a in args.adapters.split(",")]
    adapters = load_adapters(adapter_names)

    for name, adapter in adapters.items():
        results = run_concurrency(name, adapter, records, client_levels, label=args.label)
        for r in results:
            write_result(r, prefix="concurrency")


if __name__ == "__main__":
    main()
