"""Tests for assessment checks, plus retrieval and persistence on a real Chroma database.

Embeddings are deterministic, so no API key is needed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
from llama_index.core.embeddings import BaseEmbedding

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import compliance


class HashEmbedding(BaseEmbedding):
    """Hash words into a small bag-of-words vector, so shared words mean nearby vectors."""

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * 128
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)


def source(number: int, kind: str) -> compliance.Source:
    return compliance.Source(number, f"{kind}.md", kind, "Title", "text", 0.5)


SOURCES = [source(1, "requirements"), source(2, "evidence")]


def model_reply(*findings: dict) -> compliance.Completer:
    reply = json.dumps({"summary": "Summary.", "findings": list(findings)})
    return lambda prompt: reply


def finding(status: str, requirements=(1,), evidence=(2,), gap="") -> dict:
    return {
        "obligation": "Do the thing.",
        "status": status,
        "requirement_citations": list(requirements),
        "evidence_citations": list(evidence),
        "gap": gap,
    }


def test_load_nodes_labels_kinds_and_has_unique_ids():
    nodes = compliance.load_nodes(ROOT / "data")
    assert {node.metadata["kind"] for node in nodes} == {"requirements", "evidence"}
    assert len({node.node_id for node in nodes}) == len(nodes)


def test_supported_finding_is_kept():
    result = compliance.assess("q", SOURCES, model_reply(finding("met")))
    assert result.findings[0].status == "met"
    assert result.findings[0].evidence_citations == [2]


def test_met_without_evidence_becomes_missing():
    result = compliance.assess("q", SOURCES, model_reply(finding("met", evidence=())))
    assert result.findings[0].status == "missing"
    assert result.findings[0].gap


def test_weak_keeps_the_model_gap_when_downgraded():
    reply = model_reply(finding("weak", evidence=(), gap="Report is stale."))
    assert compliance.assess("q", SOURCES, reply).findings[0].gap == "Report is stale."


def test_requirement_cited_as_evidence_does_not_count():
    result = compliance.assess("q", SOURCES, model_reply(finding("met", evidence=(1,))))
    assert result.findings[0].status == "missing"
    assert result.findings[0].evidence_citations == []


def test_invented_citation_does_not_count():
    result = compliance.assess("q", SOURCES, model_reply(finding("met", evidence=(99,))))
    assert result.findings[0].status == "missing"


def test_finding_without_a_valid_requirement_is_dropped():
    result = compliance.assess("q", SOURCES, model_reply(finding("met", requirements=(2,))))
    assert result.findings == []
    assert result.summary == compliance.NO_REQUIREMENTS_SUMMARY


def test_model_is_skipped_without_requirements():
    def complete(prompt):
        raise AssertionError("model must not be called")

    result = compliance.assess("q", [source(1, "evidence")], complete)
    assert result.summary == compliance.NO_REQUIREMENTS_SUMMARY


def test_parse_handles_reasoning_and_fences():
    raw = '<think>{not json}</think>\n```json\n{"summary": "Done."}\n```'
    assert compliance.parse_assessment(raw).summary == "Done."


@pytest.mark.parametrize(
    "raw",
    ["no json here", '{"summary": ', '{"findings": []}', '{"summary": "s", "findings": [{}]}'],
)
def test_parse_rejects_bad_output(raw):
    with pytest.raises(ValueError):
        compliance.parse_assessment(raw)


def test_prompt_formats_sources_containing_braces():
    sources = [compliance.Source(1, "a.md", "requirements", "T", "{literal}", 0.5)]
    assert "{literal}" in compliance.build_prompt("q {x}", sources)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("chroma"))
    nodes = compliance.load_nodes(ROOT / "data")
    assert compliance.build_index(nodes, HashEmbedding(), path) == len(nodes)
    return path


def test_retrieval_returns_both_kinds_with_sequential_numbers(db_path):
    index = compliance.load_index(HashEmbedding(), db_path)
    sources = compliance.retrieve(index, "SOC 2 report for vendors", top_k=3)
    assert [item.number for item in sources] == [1, 2, 3, 4, 5, 6]
    assert [item.kind for item in sources] == ["requirements"] * 3 + ["evidence"] * 3
    assert sources[0].file == "vendor-risk-standard.md"
    assert any(item.file == "vendor-register.md" for item in sources)


def test_rebuilding_does_not_duplicate_chunks(db_path):
    nodes = compliance.load_nodes(ROOT / "data")
    assert compliance.build_index(nodes, HashEmbedding(), db_path) == len(nodes)


def test_empty_database_gives_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="ingest.py"):
        compliance.load_index(HashEmbedding(), str(tmp_path))


def test_index_survives_a_process_restart(db_path):
    script = f"""
import sys
sys.path[:0] = [{str(ROOT)!r}, {str(Path(__file__).parent)!r}]
import compliance
from test_compliance import HashEmbedding
index = compliance.load_index(HashEmbedding(), {db_path!r})
print(compliance.retrieve(index, "regulator notification 72 hours", top_k=1)[0].file)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "incident-response-policy.md"
