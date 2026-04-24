import io
import time

from psycopg2.pool import ThreadedConnectionPool

from adapters.base import VectorDBAdapter


def _escape_copy(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


class PgvanillaAdapter(VectorDBAdapter):
    def __init__(self, url: str):
        self._url = url
        self._pool = None

    def reset(self) -> None:
        self._pool = ThreadedConnectionPool(1, 32, self._url)
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS embeddings")
                cur.execute("""
                    CREATE TABLE embeddings (
                        id TEXT PRIMARY KEY,
                        embedding float8[],
                        text TEXT,
                        category TEXT,
                        ts TIMESTAMPTZ
                    )
                """)
                cur.execute("CREATE INDEX ON embeddings (category)")
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
                pg_array = "{" + ",".join(str(v) for v in r["embedding"]) + "}"
                buf.write("\t".join([
                    r["id"],
                    pg_array,
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
        pg_array = "{" + ",".join(str(v) for v in vector) + "}"

        cosine_sql = """
            1.0 - (
                (SELECT SUM(e * q) FROM unnest(embedding, %s::float8[]) AS t(e, q))
                / NULLIF(
                    SQRT((SELECT SUM(e * e) FROM unnest(embedding) AS t(e)))
                    * SQRT((SELECT SUM(q * q) FROM unnest(%s::float8[]) AS t(q))),
                    0
                )
            )
        """

        if filters and "category" in filters:
            sql = f"""
                SELECT id, text, category, {cosine_sql} AS distance
                FROM embeddings
                WHERE category = %s
                ORDER BY distance LIMIT %s
            """
            params = (pg_array, pg_array, filters["category"], top_k)
        else:
            sql = f"""
                SELECT id, text, category, {cosine_sql} AS distance
                FROM embeddings
                ORDER BY distance LIMIT %s
            """
            params = (pg_array, pg_array, top_k)

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
