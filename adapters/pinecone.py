import math
import time
from concurrent.futures import ThreadPoolExecutor
from pinecone import Pinecone, ServerlessSpec
from adapters.base import VectorDBAdapter


class PineconeAdapter(VectorDBAdapter):
    def __init__(self, api_key: str, index_name: str):
        self._api_key = api_key
        self._index_name = index_name
        self._index = None

    def reset(self) -> None:
        pc = Pinecone(api_key=self._api_key)

        if self._index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self._index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(self._index_name).status["ready"]:
                time.sleep(1)

        self._index = pc.Index(self._index_name, pool_threads=4)
        try:
            self._index.delete(delete_all=True)
        except Exception:
            pass

    def setup(self) -> None:
        pc = Pinecone(api_key=self._api_key)
        self._index = pc.Index(self._index_name, pool_threads=4)

    def insert_batch(self, records: list[dict]) -> None:
        vectors = [
            {
                "id": r["id"],
                "values": r["embedding"],
                "metadata": {
                    "text": r["text"],
                    "category": r["category"],
                    "timestamp": r["timestamp"],
                },
            }
            for r in records
        ]
        chunk_size = max(1, math.ceil(len(vectors) / 4))
        chunks = [vectors[i:i + chunk_size] for i in range(0, len(vectors), chunk_size)]
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(self._index.upsert, vectors=chunk) for chunk in chunks]
            for f in futures:
                f.result()

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> tuple[list, float]:
        pinecone_filter = None
        if filters and "category" in filters:
            pinecone_filter = {"category": {"$eq": filters["category"]}}

        t0 = time.perf_counter()
        response = self._index.query(
            vector=vector,
            top_k=top_k,
            filter=pinecone_filter,
            include_metadata=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        results = [
            {
                "id": m.id,
                "text": m.metadata.get("text"),
                "category": m.metadata.get("category"),
                "distance": 1.0 - m.score,
            }
            for m in response.matches
        ]
        return results, latency_ms

    def close(self) -> None:
        self._index = None

    def teardown(self) -> None:
        # Pinecone index is cloud-managed; teardown is a no-op to avoid deleting production data
        self._index = None
