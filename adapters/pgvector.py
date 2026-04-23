import io
import time
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse, quote
from adapters.base import VectorDBAdapter


def _escape_copy(s: str) -> str:
    """Escape special characters for PostgreSQL COPY text format."""
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def _url_with_guc(url: str, key: str, val: int) -> str:
    p = urlparse(url)
    qs = parse_qs(p.query, keep_blank_values=True)
    new_opt = f"-c {key}={val}"
    qs["options"] = [qs["options"][0] + " " + new_opt] if "options" in qs else [new_opt]
    return urlunparse(p._replace(query=urlencode(qs, doseq=True, quote_via=quote)))


class PgvectorAdapter(VectorDBAdapter):
    def __init__(self, url: str, ef_search: int = 40):
        self._url = _url_with_guc(url, "hnsw.ef_search", ef_search)
        self._pool = None

    def reset(self) -> None:
        self._pool = ThreadedConnectionPool(1, 32, self._url)
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute("DROP TABLE IF EXISTS embeddings")
                cur.execute("""
                    CREATE TABLE embeddings (
                        id TEXT PRIMARY KEY,
                        embedding vector(1536),
                        text TEXT,
                        category TEXT,
                        ts TIMESTAMPTZ
                    )
                """)
                cur.execute("""
                    CREATE INDEX ON embeddings
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """)
            conn.commit()
        finally:
            self._pool.putconn(conn)

    def setup(self) -> None:
        self._pool = ThreadedConnectionPool(1, 32, self._url)

    def insert_batch(self, records: list[dict]) -> None:
        conn = self._pool.getconn()
        try:
            buf = io.StringIO()
            for r in records:
                buf.write("\t".join([
                    r["id"],
                    str(r["embedding"]),
                    _escape_copy(r["text"]),
                    r["category"],
                    r["timestamp"],
                ]) + "\n")
            buf.seek(0)
            with conn.cursor() as cur:
                cur.copy_expert(
                    "COPY embeddings (id, embedding, text, category, ts) FROM STDIN",
                    buf,
                )
            conn.commit()
        finally:
            self._pool.putconn(conn)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> tuple[list, float]:
        vec_str = str(vector)
        if filters and "category" in filters:
            sql = """
                SELECT id, text, category,
                       embedding <=> %s::vector AS distance
                FROM embeddings
                WHERE category = %s
                ORDER BY distance LIMIT %s
            """
            params = (vec_str, filters["category"], top_k)
        else:
            sql = """
                SELECT id, text, category,
                       embedding <=> %s::vector AS distance
                FROM embeddings
                ORDER BY distance LIMIT %s
            """
            params = (vec_str, top_k)

        conn = self._pool.getconn()
        try:
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            latency_ms = (time.perf_counter() - t0) * 1000
        finally:
            self._pool.putconn(conn)
        results = [{"id": r[0], "text": r[1], "category": r[2], "distance": r[3]} for r in rows]
        return results, latency_ms

    def close(self) -> None:
        if self._pool:
            self._pool.closeall()
            self._pool = None

    def teardown(self) -> None:
        if self._pool:
            conn = self._pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("DROP TABLE IF EXISTS embeddings")
                conn.commit()
            finally:
                self._pool.putconn(conn)
            self._pool.closeall()
            self._pool = None
