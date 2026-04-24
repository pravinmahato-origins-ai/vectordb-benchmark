#!/usr/bin/env python3
"""
Usage: python benchmark_insert.py [--adapters pgvector,pgvanilla,sqlite_vec,weaviate,pinecone]
Reads dataset.jsonl, inserts all records in batches of 100, emits results/insert_<adapter>_<ts>.json
"""
import argparse
import json
import time
from tqdm import tqdm
from benchmark_utils import build_result, write_result, load_adapters

BATCH_SIZE = 100
DEFAULT_ADAPTERS = ["pgvector", "sqlite_vec", "weaviate"]


def run_insert(adapter_name: str, adapter, records: list[dict], label: str = "baseline") -> dict:
    adapter.reset()
    batches = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]
    batch_latencies = []
    t_total_start = time.perf_counter()

    for batch in tqdm(batches, desc=f"Inserting [{adapter_name}]"):
        t0 = time.perf_counter()
        adapter.insert_batch(batch)
        batch_latencies.append((time.perf_counter() - t0) * 1000)

    total_ms = (time.perf_counter() - t_total_start) * 1000
    records_per_sec = len(records) / (total_ms / 1000)

    result = build_result(
        adapter=adapter_name,
        operation="insert",
        latencies_ms=batch_latencies,
        clients=1,
        extra={
            "total_records": len(records),
            "total_ms": round(total_ms, 2),
            "records_per_sec": round(records_per_sec, 2),
            "batch_size": BATCH_SIZE,
        },
        label=label,
    )
    adapter.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", default=",".join(DEFAULT_ADAPTERS))
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    with open(args.dataset) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} records from {args.dataset}")
    adapter_names = [a.strip() for a in args.adapters.split(",")]
    adapters = load_adapters(adapter_names)

    for name, adapter in adapters.items():
        result = run_insert(name, adapter, records, label=args.label)
        path = write_result(result, prefix="insert")
        print(f"[{name}] {result['records_per_sec']} rec/s → {path}")


if __name__ == "__main__":
    main()
