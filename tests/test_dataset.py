import json
import math
import subprocess
import os
import pytest

DATASET_PATH = "dataset.jsonl"

@pytest.fixture(scope="module", autouse=True)
def generate():
    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
    subprocess.run(["python", "generate_dataset.py", "--count", "50"], check=True)

def load_records():
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]

def test_record_count():
    records = load_records()
    assert len(records) == 50

def test_schema_fields():
    records = load_records()
    required = {"id", "embedding", "text", "category", "timestamp"}
    for r in records:
        assert required <= r.keys()

def test_embedding_dimension():
    records = load_records()
    for r in records:
        assert len(r["embedding"]) == 1536

def test_embedding_unit_normalised():
    records = load_records()
    for r in records[:5]:
        norm = math.sqrt(sum(x**2 for x in r["embedding"]))
        assert abs(norm - 1.0) < 1e-5

def test_categories_within_allowed():
    CATEGORIES = {f"cat_{i}" for i in range(10)}
    records = load_records()
    for r in records:
        assert r["category"] in CATEGORIES

def test_id_is_uuid4_format():
    import uuid
    records = load_records()
    for r in records:
        uuid.UUID(r["id"], version=4)  # raises if invalid
