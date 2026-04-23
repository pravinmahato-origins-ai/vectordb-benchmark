import struct
import threading
import time
import sqlite3
import sqlite_vec
from adapters.base import VectorDBAdapter


def _pack_vector(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


class SqliteVecAdapter(VectorDBAdapter):
    def __init__(self, path: str = "./benchmark.db"):
        self._path = path
        self._conn = None
        self._local = threading.local()
        self._worker_conns: list[sqlite3.Connection] = []
        self._worker_conns_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA cache_size = -65536")       # 64 MB page cache
        conn.execute("PRAGMA mmap_size = 268435456")     # 256 MB mmap I/O
        conn.execute("PRAGMA temp_store = MEMORY")       # temp tables in RAM
        conn.execute("PRAGMA synchronous = NORMAL")      # safe with WAL, avoids full fsync
        conn.execute("PRAGMA wal_autocheckpoint = 10000") # checkpoint every 10000 pages, avoids blocking readers under concurrent load
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating one if needed."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
            with self._worker_conns_lock:
                self._worker_conns.append(conn)
        return conn

    def reset(self) -> None:
        self._conn = self._connect()
        self._conn.execute("DROP TABLE IF EXISTS embeddings_meta")
        self._conn.execute("DROP TABLE IF EXISTS vec_embeddings")
        self._conn.execute("""
            CREATE TABLE embeddings_meta (
                id TEXT PRIMARY KEY,
                text TEXT,
                category TEXT,
                ts TEXT
            )
        """)
        self._conn.execute("""
            CREATE VIRTUAL TABLE vec_embeddings USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[1536]
            )
        """)
        self._conn.commit()

    def setup(self) -> None:
        self._conn = self._connect()

    def insert_batch(self, records: list[dict]) -> None:
        meta = [(r["id"], r["text"], r["category"], r["timestamp"]) for r in records]
        vecs = [(r["id"], _pack_vector(r["embedding"])) for r in records]
        self._conn.executemany(
            "INSERT INTO embeddings_meta (id, text, category, ts) VALUES (?, ?, ?, ?)", meta
        )
        self._conn.executemany(
            "INSERT INTO vec_embeddings (id, embedding) VALUES (?, ?)", vecs
        )
        self._conn.commit()

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> tuple[list, float]:
        packed = _pack_vector(vector)
        if filters and "category" in filters:
            sql = """
                SELECT v.id, m.text, m.category, v.distance
                FROM vec_embeddings v
                JOIN embeddings_meta m ON m.id = v.id
                WHERE v.embedding MATCH ?
                  AND k = ?
                  AND m.category = ?
                ORDER BY v.distance
            """
            params = (packed, top_k * 10, filters["category"])
        else:
            sql = """
                SELECT v.id, m.text, m.category, v.distance
                FROM vec_embeddings v
                JOIN embeddings_meta m ON m.id = v.id
                WHERE v.embedding MATCH ?
                  AND k = ?
                ORDER BY v.distance
                LIMIT ?
            """
            params = (packed, top_k, top_k)

        t0 = time.perf_counter()
        cur = self._get_conn().execute(sql, params)
        rows = cur.fetchall()
        latency_ms = (time.perf_counter() - t0) * 1000

        if filters:
            rows = rows[:top_k]

        results = [{"id": r[0], "text": r[1], "category": r[2], "distance": r[3]} for r in rows]
        return results, latency_ms

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        with self._worker_conns_lock:
            for conn in self._worker_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._worker_conns.clear()
        self._local = threading.local()

    def teardown(self) -> None:
        if self._conn:
            self._conn.execute("DROP TABLE IF EXISTS vec_embeddings")
            self._conn.execute("DROP TABLE IF EXISTS embeddings_meta")
            self._conn.commit()
            self._conn.close()
            self._conn = None
        with self._worker_conns_lock:
            for conn in self._worker_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._worker_conns.clear()
        self._local = threading.local()
