# Vector DB Benchmark — Optimisation Report

**Dataset:** 1,700 records, 1536-dimensional embeddings  
**Adapters:** pgvector, sqlite_vec, weaviate, pinecone  
**Comparison:** `baseline` (original code) vs `optimized_v2` (all changes applied)

---

## How to Run

```bash
# Run all adapters
python benchmark_insert.py      --adapters pgvector,sqlite_vec,weaviate,pinecone --label optimized_v2
python benchmark_query.py       --adapters pgvector,sqlite_vec,weaviate,pinecone --label optimized_v2
python benchmark_concurrency.py --adapters pgvector,sqlite_vec,weaviate,pinecone --label optimized_v2

# Compare
python benchmark_compare.py --before baseline --after optimized_v2
```

Open `results.ipynb` and run all cells to see charts and scorecards.

---

## What Changed in `optimized_v2`

All changes are consolidated into a single label. The table below maps each change to the file it lives in.

### pgvector (`adapters/pgvector.py`)

| Change | Detail |
|--------|--------|
| Insert: `execute_batch` → `COPY FROM STDIN` | Streams the entire batch as a tab-delimited buffer instead of row-by-row parameterised INSERTs. Faster for bulk loads. |
| Query: `hnsw.ef_search` moved to connection URL | Previously `SET hnsw.ef_search=40` was issued on every query (one extra round-trip each time). Now injected as a session GUC via the connection URL `options` parameter — set once per connection, zero per-query overhead. |
| Postgres server tuning (`docker-compose.yml`) | `shared_buffers=256MB`, `work_mem=64MB`, `effective_cache_size=512MB` passed as startup flags. |

### pinecone (`adapters/pinecone.py`)

| Change | Detail |
|--------|--------|
| Parallel upsert | Batch is split into 4 chunks and upserted concurrently via `ThreadPoolExecutor`. Reduces wall time for network-bound inserts. |
| `pool_threads=4` on index open | Allows the Pinecone client to multiplex HTTP connections. |

### sqlite_vec (`adapters/sqlite_vec.py`)

| Change | Detail |
|--------|--------|
| `PRAGMA cache_size = -65536` | 64 MB in-process page cache — keeps hot index pages in RAM. |
| `PRAGMA mmap_size = 268435456` | 256 MB memory-mapped I/O — eliminates syscall overhead for reads. |
| `PRAGMA temp_store = MEMORY` | Temp tables and sort buffers stay in RAM. |
| `PRAGMA synchronous = NORMAL` | Avoids a full fsync on every commit; safe with WAL mode. |
| `PRAGMA wal_autocheckpoint = 10000` | Checkpoints every 10,000 pages instead of the default 1,000. Prevents checkpoint events from blocking concurrent readers under high load. |

### weaviate (`adapters/weaviate.py`)

| Change | Detail |
|--------|--------|
| Insert: `insert_many` with pre-built list | Reverted from `batch.dynamic()` which added streaming overhead that dominated at fixed 100-record batches. `insert_many` sends the full list in one gRPC call. |
| Collection handle cached (`self._col`) | `collections.get(COLLECTION_NAME)` called once in `reset()`/`setup()` and stored. Previously re-fetched on every `insert_batch` and `query` call. |

---

## Notebook Structure

The notebook (`results.ipynb`) tracks the following charts and tables. All cells read from `results/*.json` and automatically pick up both labels.

| Cell | What it shows |
|------|--------------|
| **Chart 1** | Insert throughput (records/sec) per adapter — tallest bar = best bulk ingestion |
| **Chart 2** | ANN query latency (p50/p95/p99) — lower = better; p99 shows tail consistency |
| **Chart 3** | Filtered query latency (p50/p95/p99) — same as Chart 2 with a metadata filter applied |
| **Chart 4** | Concurrency degradation — p99 vs client count (1/5/10/25); flatter line = better under load |
| **Chart 5** | Query QPS side-by-side for ANN and filtered — taller = better throughput |
| **Scorecard** | Normalized 0–1 per metric, overall rank per adapter |
| **Radar chart** | All 7 dimensions at once — larger polygon = better balanced adapter |
| **Use-case summary** | Best adapter per use case (bulk insert, low latency, high QPS, concurrency) |
| **Before/After bars** | p50 and p99 side-by-side for baseline vs optimized_v2 per operation |
| **Δ% table** | Per-adapter delta table with green/red highlighting (green = improvement) |
| **Comparison scorecard** | Scorecard run separately for each label to show rank changes |

---

## Baseline Results

These are the untuned numbers used as the reference point for all comparisons.

### Insert

| Adapter | rec/s | p50 ms | p99 ms | total time |
|---------|-------|--------|--------|------------|
| pgvector | 192.5 | 553.4 | 738.2 | 8,831 ms |
| pinecone | 8.3 | 12,426 | 19,358 | 205,355 ms |
| sqlite_vec | 3,232 | 33.5 | 54.5 | 526 ms |
| weaviate | 3,375 | 27.6 | 38.6 | 504 ms |

### Query ANN (200 rounds, single client, top-k=10)

| Adapter | p50 ms | p95 ms | p99 ms | QPS |
|---------|--------|--------|--------|-----|
| pgvector | 1.48 | 2.21 | 2.99 | 643 |
| pinecone | 302.9 | 395.2 | 599.6 | 3.11 |
| sqlite_vec | 3.18 | 3.93 | 4.03 | 305 |
| weaviate | 1.58 | 3.28 | 4.62 | 554 |

### Query Filtered (200 rounds, single client, top-k=10)

| Adapter | p50 ms | p95 ms | p99 ms | QPS |
|---------|--------|--------|--------|-----|
| pgvector | 1.29 | 1.70 | 2.33 | 739 |
| pinecone | 301.7 | 336.4 | 598.5 | 3.19 |
| sqlite_vec | 3.70 | 4.52 | 4.64 | 260 |
| weaviate | 1.01 | 1.55 | 2.10 | 936 |

### Concurrency (p50 ms / QPS)

| Adapter | 1 client | 5 clients | 10 clients | 25 clients |
|---------|----------|-----------|------------|------------|
| pgvector | 1.92 ms / 361 | 5.97 ms / 193 | 5.41 ms / 191 | 5.93 ms / 177 |
| pinecone | 301 ms / 3.13 | 309 ms / 13.2 | 309 ms / 25.5 | 314 ms / 36.0 |
| sqlite_vec | 3.78 ms / 251 | 11.65 ms / 404 | 18.34 ms / 478 | 42.17 ms / 215 |
| weaviate | 2.04 ms / 385 | 4.44 ms / 915 | 10.42 ms / 813 | 21.76 ms / 616 |

---

## optimized_v2 Results

> Run the benchmarks with `--label optimized_v2` to populate this section.

---

## Glossary

| Term | Meaning |
|------|---------|
| **p50** | Median latency — the typical query experience |
| **p95** | 95th percentile — starts to reveal occasional slow outliers |
| **p99** | 99th percentile — worst-case most users will ever see |
| **QPS** | Queries per second — throughput |
| **rec/s** | Records per second — insert throughput |
| **ANN** | Approximate nearest neighbour — core vector search |
| **Filtered** | ANN + metadata filter (e.g. `category = "X"`) applied |
| **Concurrency** | Multiple clients querying simultaneously |
