"""Unit tests for chunking and citations, and integration tests against real Weaviate.

Integration tests use deterministic embeddings, so no API key is needed. They run when
Weaviate is reachable at WEAVIATE_URL (see docker-compose.yml) and skip otherwise.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest
import weaviate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag

TEST_COLLECTION = "EnterpriseChunkTest"


def fake_embed(texts):
    """Hash words into a small bag-of-words vector, so shared words mean nearby vectors."""
    vectors = []
    for text in texts:
        vector = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


def hit(number: int) -> rag.Hit:
    chunk = rag.Chunk(f"doc{number}.md", "Doc", "policy", "legal", 0, "text")
    return rag.Hit(chunk, 0.5)


def test_split_words_overlaps_and_covers_all_words():
    words = [str(n) for n in range(100)]
    chunks = rag.split_words(" ".join(words), size=40, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert {word for chunk in chunks for word in chunk.split()} == set(words)
    assert chunks[-1].split()[-1] == "99"


def test_load_chunks_reads_metadata_for_every_sample_document():
    chunks = rag.load_chunks(ROOT / "data")
    assert {chunk.doc_type for chunk in chunks} == {"policy", "product", "runbook", "support"}
    assert len({chunk.source for chunk in chunks}) == 6
    assert len({(chunk.source, chunk.chunk_index) for chunk in chunks}) == len(chunks)


def test_parse_document_rejects_missing_metadata(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text("---\ntitle: Bad\n---\nbody", encoding="utf-8")
    with pytest.raises(ValueError, match="doc_type"):
        rag.parse_document(path)


def test_extract_citations_keeps_order_and_drops_unknown_numbers():
    hits = [hit(1), hit(2)]
    cited = rag.extract_citations("A [2]. B [9]. C [1]. D [2].", hits)
    assert cited == [hits[1], hits[0]]


def test_generate_answer_without_hits_skips_the_model():
    def complete(system, user):
        raise AssertionError("model must not be called")

    assert rag.generate_answer("q", [], complete).text == rag.NO_EVIDENCE_ANSWER


def test_generate_answer_refuses_uncited_text():
    answer = rag.generate_answer("q", [hit(1)], lambda system, user: "Trust me.")
    assert answer.text == rag.NO_EVIDENCE_ANSWER
    assert answer.citations == []


def test_generate_answer_returns_cited_hits():
    hits = [hit(1), hit(2)]
    answer = rag.generate_answer("q", hits, lambda system, user: "Yes [2].")
    assert answer.text == "Yes [2]."
    assert answer.citations == [hits[1]]


@pytest.fixture(scope="module")
def client():
    try:
        connection = rag.connect()
    except weaviate.exceptions.WeaviateBaseError as error:
        pytest.skip(f"Weaviate is not reachable: {error}")
    rag.recreate_collection(connection, TEST_COLLECTION)
    rag.index_chunks(connection, rag.load_chunks(ROOT / "data"), fake_embed, TEST_COLLECTION)
    yield connection
    connection.collections.delete(TEST_COLLECTION)
    connection.close()


def run_search(client, query, **kwargs):
    return rag.search(client, query, fake_embed, name=TEST_COLLECTION, **kwargs)


def test_indexing_stores_every_chunk(client):
    expected = len(rag.load_chunks(ROOT / "data"))
    total = client.collections.get(TEST_COLLECTION).aggregate.over_all(total_count=True)
    assert total.total_count == expected


def test_hybrid_search_finds_the_relevant_document(client):
    hits = run_search(client, "payments_failover feature flag")
    assert hits[0].chunk.source == "payments-outage-runbook.md"
    assert hits[0].score > 0


def test_keyword_only_search_matches_exact_terms(client):
    hits = run_search(client, "HMAC-SHA256 webhooks", alpha=0.0)
    assert hits[0].chunk.source == "atlas-api-product-guide.md"


def test_doc_type_filter_excludes_other_types(client):
    hits = run_search(client, "failover", filters=rag.build_filters(doc_types=["policy"]))
    assert hits
    assert {hit.chunk.doc_type for hit in hits} == {"policy"}


def test_filters_on_different_fields_are_combined(client):
    filters = rag.build_filters(doc_types=["runbook"], teams=["sre"])
    hits = run_search(client, "failover", filters=filters)
    assert {hit.chunk.source for hit in hits} == {"payments-outage-runbook.md"}


def test_source_filter_and_no_match(client):
    filters = rag.build_filters(sources=["billing-support-faq.md"])
    assert {hit.chunk.source for hit in run_search(client, "refund", filters=filters)} == {
        "billing-support-faq.md"
    }
    filters = rag.build_filters(doc_types=["policy"], teams=["sre"])
    assert run_search(client, "refund", filters=filters) == []


def test_answer_cites_a_real_retrieved_chunk(client):
    hits = run_search(client, "How long are security audit logs retained?")
    answer = rag.generate_answer("q", hits, lambda system, user: "13 months [1].")
    assert answer.citations[0].chunk.source == "data-retention-policy.md"
