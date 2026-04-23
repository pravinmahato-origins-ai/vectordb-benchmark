#!/usr/bin/env python3
"""
Usage: python benchmark_compare.py --before baseline --after optimized
Compares benchmark results between two labeled runs.
"""
import argparse
import glob
import json
import os

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

OPERATIONS = ["insert", "query_ann", "query_filtered", "concurrency"]


def load_results(results_dir: str) -> list[dict]:
    records = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            with open(path) as f:
                r = json.load(f)
            r.setdefault("label", "baseline")
            records.append(r)
        except (json.JSONDecodeError, OSError):
            continue
    return records


def pick_latest(records: list[dict], label: str, adapter: str, operation: str) -> dict | None:
    matches = [
        r for r in records
        if r.get("label") == label
        and r.get("adapter") == adapter
        and r.get("operation") == operation
    ]
    if not matches:
        return None
    return max(matches, key=lambda r: r.get("timestamp", ""))


def delta_pct(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return (after - before) / before * 100


def colorize(val: float, lower_is_better: bool) -> str:
    s = f"{val:+.1f}%"
    if lower_is_better:
        if val < -1:
            return f"{GREEN}{s}{RESET}"
        if val > 1:
            return f"{RED}{s}{RESET}"
    else:
        if val > 1:
            return f"{GREEN}{s}{RESET}"
        if val < -1:
            return f"{RED}{s}{RESET}"
    return s


def get_throughput(record: dict) -> float:
    """Use records_per_sec for insert; qps for all others."""
    if record.get("operation") == "insert":
        return record.get("records_per_sec", record.get("qps", 0))
    return record.get("qps", 0)


def print_table(operation: str, rows: list[dict]) -> None:
    if not rows:
        return
    print(f"\n=== {operation.upper()} ===")
    print(
        f"{'adapter':<14}  {'bef p50':>8} {'aft p50':>8} {'Δ%':>8}"
        f"   {'bef p99':>8} {'aft p99':>8} {'Δ%':>8}"
        f"   {'bef thru':>9} {'aft thru':>9} {'Δ%':>8}"
    )
    print("-" * 95)
    for row in rows:
        dp50 = delta_pct(row["bef_p50"], row["aft_p50"])
        dp99 = delta_pct(row["bef_p99"], row["aft_p99"])
        dthr = delta_pct(row["bef_thr"], row["aft_thr"])
        print(
            f"{row['adapter']:<14}"
            f"  {row['bef_p50']:>8.1f} {row['aft_p50']:>8.1f} {colorize(dp50, lower_is_better=True)}"
            f"   {row['bef_p99']:>8.1f} {row['aft_p99']:>8.1f} {colorize(dp99, lower_is_better=True)}"
            f"   {row['bef_thr']:>9.1f} {row['aft_thr']:>9.1f} {colorize(dthr, lower_is_better=False)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results between two labels")
    parser.add_argument("--before", default="baseline")
    parser.add_argument("--after", default="optimized")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    records = load_results(args.results_dir)
    if not records:
        print(f"No result files found in {args.results_dir}/")
        return

    adapters = sorted({r["adapter"] for r in records})

    for operation in OPERATIONS:
        rows = []
        for adapter in adapters:
            before = pick_latest(records, args.before, adapter, operation)
            after = pick_latest(records, args.after, adapter, operation)
            if before is None or after is None:
                continue
            rows.append({
                "adapter": adapter,
                "bef_p50": before.get("p50_ms", 0),
                "aft_p50": after.get("p50_ms", 0),
                "bef_p99": before.get("p99_ms", 0),
                "aft_p99": after.get("p99_ms", 0),
                "bef_thr": get_throughput(before),
                "aft_thr": get_throughput(after),
            })
        print_table(operation, rows)

    print()


if __name__ == "__main__":
    main()
