import json
import os
import statistics
from datetime import datetime, timezone


def percentile(data: list[float], p: int) -> float:
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = max(0, int(p / 100 * n) - 1)
    return float(sorted_data[idx])


def build_result(
    adapter: str,
    operation: str,
    latencies_ms: list[float],
    clients: int = 1,
    extra: dict | None = None,
    wall_time_s: float | None = None,
    label: str = "baseline",
) -> dict:
    total_s = wall_time_s if wall_time_s is not None else sum(latencies_ms) / 1000
    qps = len(latencies_ms) / total_s if total_s > 0 else 0
    record = {
        "adapter": adapter,
        "operation": operation,
        "label": label,
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "qps": round(qps, 2),
        "clients": clients,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        record.update(extra)
    return record


def write_result(record: dict, prefix: str, output_dir: str = "results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{prefix}_{record['adapter']}_{ts}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def load_adapters(names: list[str]) -> dict:
    """Instantiate adapters from environment variables."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    adapters = {}
    if "pgvector" in names:
        from adapters.pgvector import PgvectorAdapter
        adapters["pgvector"] = PgvectorAdapter(url=os.environ["POSTGRES_URL"])
    if "pgvanilla" in names:
        from adapters.pgvanilla import PgvanillaAdapter
        adapters["pgvanilla"] = PgvanillaAdapter(url=os.environ["POSTGRES_VANILLA_URL"])
    if "sqlite_vec" in names:
        from adapters.sqlite_vec import SqliteVecAdapter
        adapters["sqlite_vec"] = SqliteVecAdapter(path=os.getenv("SQLITE_PATH", "./benchmark.db"))
    if "weaviate" in names:
        from adapters.weaviate import WeaviateAdapter
        adapters["weaviate"] = WeaviateAdapter(url=os.getenv("WEAVIATE_URL", "http://localhost:8080"))
    if "pinecone" in names:
        from adapters.pinecone import PineconeAdapter
        adapters["pinecone"] = PineconeAdapter(
            api_key=os.environ["PINECONE_API_KEY"],
            index_name=os.environ["PINECONE_INDEX_NAME"],
        )
    return adapters
