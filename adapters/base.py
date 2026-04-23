from abc import ABC, abstractmethod


class VectorDBAdapter(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Drop existing data, recreate schema/index, and connect. Used by insert benchmark."""

    @abstractmethod
    def setup(self) -> None:
        """Connect to existing schema without wiping data. Used by query/concurrency benchmarks."""

    @abstractmethod
    def insert_batch(self, records: list[dict]) -> None:
        """Insert a batch of records. Each record has keys: id, embedding, text, category, timestamp."""

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> tuple[list, float]:
        """Return (results, latency_ms). results is a list of matched records."""

    @abstractmethod
    def close(self) -> None:
        """Close connections without dropping data."""

    @abstractmethod
    def teardown(self) -> None:
        """Drop table/collection and close all connections."""
