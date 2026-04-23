import time
from urllib.parse import urlparse
import weaviate
import weaviate.classes as wvc
from adapters.base import VectorDBAdapter

COLLECTION_NAME = "Embeddings"


class WeaviateAdapter(VectorDBAdapter):
    def __init__(self, url: str = "http://localhost:8080"):
        self._url = url
        self._client = None

    def reset(self) -> None:
        parsed = urlparse(self._url)
        self._client = weaviate.connect_to_local(
            host=parsed.hostname,
            port=parsed.port or 8080,
        )
        if self._client.collections.exists(COLLECTION_NAME):
            self._client.collections.delete(COLLECTION_NAME)
        self._client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                ef_construction=64,
                max_connections=16,
            ),
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="ts", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="ext_id", data_type=wvc.config.DataType.TEXT),
            ],
        )

    def setup(self) -> None:
        parsed = urlparse(self._url)
        self._client = weaviate.connect_to_local(
            host=parsed.hostname,
            port=parsed.port or 8080,
        )

    def insert_batch(self, records: list[dict]) -> None:
        col = self._client.collections.get(COLLECTION_NAME)
        objects = [
            wvc.data.DataObject(
                properties={
                    "ext_id": r["id"],
                    "text": r["text"],
                    "category": r["category"],
                    "ts": r["timestamp"],
                },
                vector=r["embedding"],
            )
            for r in records
        ]
        col.data.insert_many(objects)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> tuple[list, float]:
        col = self._client.collections.get(COLLECTION_NAME)
        weaviate_filter = None
        if filters and "category" in filters:
            weaviate_filter = wvc.query.Filter.by_property("category").equal(filters["category"])

        t0 = time.perf_counter()
        response = col.query.near_vector(
            near_vector=vector,
            limit=top_k,
            filters=weaviate_filter,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        results = [
            {
                "id": o.properties.get("ext_id"),
                "text": o.properties.get("text"),
                "category": o.properties.get("category"),
                "distance": o.metadata.distance,
            }
            for o in response.objects
        ]
        return results, latency_ms

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def teardown(self) -> None:
        if self._client:
            if self._client.collections.exists(COLLECTION_NAME):
                self._client.collections.delete(COLLECTION_NAME)
            self._client.close()
            self._client = None
