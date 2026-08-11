from __future__ import annotations

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orbit.memory.config import MemoryConfig
from orbit.memory.errors import MemoryModelError, MemorySchemaError, MemoryValidationError
from orbit.memory.models import MemoryModels
from orbit.memory.service import MemoryService, memory_identity, normalize_memory_text
from orbit.memory.store import QdrantMemoryStore
from orbit.memory.types import MemoryCandidate, MemoryRecord


class FakeVector:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": list(range(len(text.split()) + 2))}


class FakeEmbedder:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return FakeVector([1.0] * self.dimension)


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((list(pairs), kwargs))
        return self.scores


class FakeServiceModels:
    def __init__(self):
        self.validated = []
        self.embedded_memories = []
        self.embedded_queries = []
        self.rerank_calls = []

    def validate_memory(self, text):
        self.validated.append(text)

    def embed_memory(self, text):
        self.embedded_memories.append(text)
        return [0.1, 0.2, 0.3]

    def embed_query(self, query):
        self.embedded_queries.append(query)
        return [0.3, 0.2, 0.1]

    def rerank(self, query, passages):
        self.rerank_calls.append((query, list(passages)))
        scores = {"first": 0.1, "second": 0.9, "third": 0.4}
        return [scores[p] for p in passages]


class FakeServiceStore:
    def __init__(self):
        self.records = {}
        self.upserts = []
        self.search_limit = None

    def get(self, point_id):
        return self.records.get(point_id)

    def upsert(self, record, vector):
        self.records[record.id] = record
        self.upserts.append((record, vector))

    def search(self, vector, *, limit):
        self.search_limit = limit
        self.search_vector = vector
        return [
            MemoryCandidate(make_record("1", "first"), 0.99),
            MemoryCandidate(make_record("2", "second"), 0.80),
            MemoryCandidate(make_record("3", "third"), 0.70),
        ]


def make_record(point_id, text):
    return MemoryRecord(
        id=point_id,
        text=text,
        stored_at="2026-08-11T00:00:00Z",
        content_hash="hash-" + point_id,
        schema_version=1,
        embedding_model="test-model",
    )


class FakeDistance:
    COSINE = "Cosine"


class FakeVectorParams:
    def __init__(self, *, size, distance):
        self.size = size
        self.distance = distance


class FakePointStruct:
    def __init__(self, *, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload


class FakeQdrantModels:
    Distance = FakeDistance
    VectorParams = FakeVectorParams
    PointStruct = FakePointStruct


class FakeQdrantClient:
    def __init__(self, *, exists=False, size=3, distance="Cosine"):
        self.exists = exists
        self.size = size
        self.distance = distance
        self.created = []
        self.points = {}
        self.query_response = SimpleNamespace(points=[])
        self.query_calls = []

    def collection_exists(self, name):
        self.collection_name = name
        return self.exists

    def create_collection(self, *, collection_name, vectors_config):
        self.exists = True
        self.size = vectors_config.size
        self.distance = vectors_config.distance
        self.created.append((collection_name, vectors_config))
        return True

    def get_collection(self, name):
        del name
        vectors = SimpleNamespace(size=self.size, distance=self.distance)
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

    def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
        del collection_name, with_payload, with_vectors
        return [self.points[i] for i in ids if i in self.points]

    def upsert(self, *, collection_name, points, wait):
        del collection_name, wait
        for point in points:
            self.points[point.id] = SimpleNamespace(id=point.id, payload=point.payload)

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return self.query_response


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.config = MemoryConfig(vector_size=3, retrieval_candidates=20)
        self.models = FakeServiceModels()
        self.store = FakeServiceStore()
        self.service = MemoryService(self.config, self.models, self.store)

    def test_normalization_and_identity_are_deterministic(self):
        first = normalize_memory_text("  User   likes\npytest  ")
        second = normalize_memory_text("User likes pytest")
        self.assertEqual(first, second)
        self.assertEqual(memory_identity(first), memory_identity(second))

    def test_add_deduplicates_before_second_embedding(self):
        first = self.service.add("  User   likes pytest ")
        second = self.service.add("User likes pytest")

        self.assertFalse(first.already_stored)
        self.assertTrue(second.already_stored)
        self.assertEqual(len(self.store.upserts), 1)
        self.assertEqual(self.models.validated, ["User likes pytest"])
        self.assertEqual(self.models.embedded_memories, ["User likes pytest"])

    def test_retrieve_dense_candidates_then_reranks(self):
        hits = self.service.retrieve("testing preference", limit=2)

        self.assertEqual(self.models.embedded_queries, ["testing preference"])
        self.assertEqual(self.store.search_limit, 20)
        self.assertEqual(
            self.models.rerank_calls,
            [("testing preference", ["first", "second", "third"])],
        )
        self.assertEqual([hit.record.text for hit in hits], ["second", "third"])
        self.assertEqual([hit.rank for hit in hits], [1, 2])

    def test_limit_is_clamped(self):
        hits = self.service.retrieve("query", limit=999)
        self.assertEqual(len(hits), 3)
        self.assertEqual(self.store.search_limit, 40)

    def test_empty_inputs_are_rejected(self):
        with self.assertRaises(MemoryValidationError):
            self.service.add("   ")
        with self.assertRaises(MemoryValidationError):
            self.service.retrieve("\n\t")


class MemoryModelsTests(unittest.TestCase):
    def setUp(self):
        self.config = MemoryConfig(vector_size=3, max_memory_tokens=20, max_query_tokens=20)
        self.models = MemoryModels(self.config)
        self.embedder = FakeEmbedder(dimension=3)
        self.reranker = FakeReranker([0.2, 0.8])
        self.models._embedder = self.embedder
        self.models._reranker = self.reranker

    def test_e5_prefixes_and_normalization_are_used(self):
        self.models.validate_memory("stored fact")
        memory_vector = self.models.embed_memory("stored fact")
        query_vector = self.models.embed_query("what was stored?")

        self.assertEqual(memory_vector, [1.0, 1.0, 1.0])
        self.assertEqual(query_vector, [1.0, 1.0, 1.0])
        self.assertEqual(self.embedder.calls[0][0], "passage: stored fact")
        self.assertEqual(self.embedder.calls[1][0], "query: what was stored?")
        for _, kwargs in self.embedder.calls:
            self.assertTrue(kwargs["normalize_embeddings"])
            self.assertFalse(kwargs["show_progress_bar"])

    def test_reranker_receives_raw_query_passage_pairs(self):
        scores = self.models.rerank("query", ["one", "two"])
        self.assertEqual(scores, [0.2, 0.8])
        pairs, kwargs = self.reranker.calls[0]
        self.assertEqual(pairs, [("query", "one"), ("query", "two")])
        self.assertEqual(kwargs["batch_size"], self.config.rerank_batch_size)

    def test_oversized_memory_is_rejected_without_silent_truncation(self):
        self.models.config = MemoryConfig(vector_size=3, max_memory_tokens=5)
        with self.assertRaises(MemoryValidationError):
            self.models.validate_memory("one two three four")

    def test_models_load_local_only(self):
        seen = {}

        class LocalSentenceTransformer(FakeEmbedder):
            def __init__(self, name, **kwargs):
                super().__init__(dimension=3)
                seen["embedder"] = (name, kwargs)

        class LocalCrossEncoder(FakeReranker):
            def __init__(self, name, **kwargs):
                super().__init__([0.5])
                seen["reranker"] = (name, kwargs)

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = LocalSentenceTransformer
        fake_module.CrossEncoder = LocalCrossEncoder

        models = MemoryModels(MemoryConfig(vector_size=3, device="cpu"))
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            models._get_embedder()
            models._get_reranker()

        self.assertTrue(seen["embedder"][1]["local_files_only"])
        self.assertTrue(seen["reranker"][1]["local_files_only"])
        self.assertEqual(seen["embedder"][1]["device"], "cpu")
        self.assertEqual(seen["reranker"][1]["device"], "cpu")

    def test_dimension_mismatch_fails_at_model_load(self):
        class WrongDimension(FakeEmbedder):
            def __init__(self, name, **kwargs):
                del name, kwargs
                super().__init__(dimension=2)

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = WrongDimension
        fake_module.CrossEncoder = FakeReranker
        models = MemoryModels(MemoryConfig(vector_size=3))
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            with self.assertRaises(MemoryModelError):
                models._get_embedder()


class QdrantMemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.config = MemoryConfig(vector_size=3, collection_name="test_memory")

    def test_creates_collection_with_cosine_schema(self):
        client = FakeQdrantClient(exists=False)
        store = QdrantMemoryStore(
            self.config,
            client=client,
            models_module=FakeQdrantModels,
        )

        store.ensure_collection()

        self.assertEqual(len(client.created), 1)
        _, params = client.created[0]
        self.assertEqual(params.size, 3)
        self.assertEqual(params.distance, "Cosine")

    def test_existing_incompatible_collection_is_not_recreated(self):
        client = FakeQdrantClient(exists=True, size=2)
        store = QdrantMemoryStore(
            self.config,
            client=client,
            models_module=FakeQdrantModels,
        )

        with self.assertRaises(MemorySchemaError):
            store.ensure_collection()
        self.assertEqual(client.created, [])

    def test_upsert_and_get_round_trip(self):
        client = FakeQdrantClient(exists=True)
        store = QdrantMemoryStore(
            self.config,
            client=client,
            models_module=FakeQdrantModels,
        )
        record = make_record("abc", "remember this")

        store.upsert(record, [0.1, 0.2, 0.3])
        loaded = store.get("abc")

        self.assertEqual(loaded, record)

    def test_query_points_returns_payload_candidates(self):
        client = FakeQdrantClient(exists=True)
        record = make_record("abc", "remember this")
        client.query_response = SimpleNamespace(
            points=[SimpleNamespace(id="abc", payload=record.payload(), score=0.88)]
        )
        store = QdrantMemoryStore(
            self.config,
            client=client,
            models_module=FakeQdrantModels,
        )

        results = store.search([0.1, 0.2, 0.3], limit=20)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record, record)
        self.assertAlmostEqual(results[0].vector_score, 0.88)
        self.assertEqual(client.query_calls[0]["limit"], 20)


if __name__ == "__main__":
    unittest.main()
