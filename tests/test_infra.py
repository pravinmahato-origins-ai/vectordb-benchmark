import os
import pytest
import psycopg2
import weaviate

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5433/vectordb")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

@pytest.mark.integration
def test_postgres_reachable():
    conn = psycopg2.connect(POSTGRES_URL)
    conn.close()

@pytest.mark.integration
def test_weaviate_reachable():
    client = weaviate.connect_to_local(host="localhost", port=8080)
    assert client.is_ready()
    client.close()
