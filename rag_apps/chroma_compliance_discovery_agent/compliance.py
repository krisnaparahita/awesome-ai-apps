"""Chroma indexing, evidence retrieval, and checked compliance assessments."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel, Field, ValidationError

DEFAULT_NEBIUS_API_BASE = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_CHAT_MODEL = "MiniMaxAI/MiniMax-M3"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_CHROMA_PATH = "chroma_db"
COLLECTION = "compliance_documents"
KINDS = ("requirements", "evidence")
NO_REQUIREMENTS_SUMMARY = "No indexed requirement covers this question."

Completer = Callable[[str], str]
Status = Literal["met", "weak", "missing"]


class Finding(BaseModel):
    """One obligation and how well the indexed evidence supports it."""

    obligation: str
    status: Status
    requirement_citations: list[int] = Field(default_factory=list)
    evidence_citations: list[int] = Field(default_factory=list)
    gap: str = ""


class Assessment(BaseModel):
    """A summary plus the findings that survived citation checks."""

    summary: str
    findings: list[Finding] = Field(default_factory=list)


@dataclass(frozen=True)
class Source:
    """A retrieved chunk, numbered by its position in the prompt."""

    number: int
    file: str
    kind: str
    title: str
    text: str
    score: float


def parse_document(path: Path) -> tuple[str, str]:
    """Return a markdown document's title (its first heading) and text."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^# (.+)$", text)
    if not match:
        raise ValueError(f"{path.name} has no top-level heading")
    return match.group(1).strip(), text


def load_nodes(data_dir: Path) -> list[TextNode]:
    """Chunk every document by section. The folder name decides the document's kind."""
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)
    nodes: list[TextNode] = []
    for kind in KINDS:
        for path in sorted((data_dir / kind).glob("*.md")):
            title, text = parse_document(path)
            sections = [s for s in re.split(r"(?m)^(?=#{1,3} )", text) if s.strip()]
            pieces = [piece for section in sections for piece in splitter.split_text(section)]
            nodes.extend(
                TextNode(
                    id_=f"{kind}/{path.name}:{index}",
                    text=piece,
                    metadata={"file": path.name, "kind": kind, "title": title},
                )
                for index, piece in enumerate(pieces)
            )
    return nodes


def chroma_path() -> str:
    """Return the on-disk location of the Chroma database."""
    return os.getenv("CHROMA_PATH", "").strip() or DEFAULT_CHROMA_PATH


def build_index(nodes: Sequence[TextNode], embed_model: BaseEmbedding, path: str) -> int:
    """Rebuild the persistent collection from nodes. Returns the number stored."""
    client = chromadb.PersistentClient(path=path)
    if COLLECTION in [collection.name for collection in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION)
    store = ChromaVectorStore(chroma_collection=collection)
    VectorStoreIndex(
        list(nodes),
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=embed_model,
    )
    return collection.count()


def load_index(embed_model: BaseEmbedding, path: str) -> VectorStoreIndex:
    """Open the persisted index without re-embedding anything."""
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(COLLECTION)
    if collection.count() == 0:
        raise RuntimeError("The index is empty. Run `python ingest.py` first.")
    store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(store, embed_model=embed_model)


def retrieve(index: VectorStoreIndex, question: str, top_k: int = 4) -> list[Source]:
    """Retrieve requirements and evidence separately, so that neither crowds out the other."""
    sources: list[Source] = []
    for kind in KINDS:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="kind", value=kind)])
        retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
        for result in retriever.retrieve(question):
            sources.append(
                Source(
                    number=len(sources) + 1,
                    file=result.node.metadata["file"],
                    kind=kind,
                    title=result.node.metadata["title"],
                    text=result.node.get_content(),
                    score=result.score or 0.0,
                )
            )
    return sources


PROMPT = """You are a compliance analyst. Assess the question using only the numbered \
sources. Requirement sources say what must be done. Evidence sources show what was done.

For each obligation that applies to the question, report:
- obligation: the obligation in one sentence
- status: "met" if the evidence fully shows it was done, "weak" if the evidence is partial, \
out of date, or indirect, "missing" if no evidence addresses it
- requirement_citations: numbers of the requirement sources that state it
- evidence_citations: numbers of the evidence sources that support it
- gap: what is missing or weak, or "" if the status is "met"

Reply with JSON only, in this shape:
{{"summary": "...", "findings": [{{"obligation": "...", "status": "met", \
"requirement_citations": [1], "evidence_citations": [5], "gap": ""}}]}}

Sources:
{sources}

Question: {question}"""


def build_prompt(question: str, sources: Sequence[Source]) -> str:
    """Number the retrieved chunks so the model can cite them."""
    rendered = "\n\n".join(
        f"[{source.number}] ({source.kind}) {source.file}\n{source.text}" for source in sources
    )
    return PROMPT.format(sources=rendered, question=question)


def parse_assessment(raw: str) -> Assessment:
    """Parse model output, tolerating reasoning blocks and code fences around the JSON."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("The model did not return JSON")
    try:
        return Assessment.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"The model returned an invalid assessment: {error}") from error


def check_finding(finding: Finding, sources: Sequence[Source]) -> Finding | None:
    """Enforce the citation rules on one finding.

    Citations must point at a retrieved source of the right kind. A finding with no valid
    requirement citation is dropped, because the obligation is ungrounded. A "met" or "weak"
    finding with no valid evidence citation becomes "missing", whatever the model claimed.
    """
    kinds = {source.number: source.kind for source in sources}
    requirements = [n for n in finding.requirement_citations if kinds.get(n) == "requirements"]
    evidence = [n for n in finding.evidence_citations if kinds.get(n) == "evidence"]
    if not requirements:
        return None
    status, gap = finding.status, finding.gap
    if not evidence and status != "missing":
        status, gap = "missing", gap or "The model cited no valid evidence for this obligation."
    return finding.model_copy(
        update={
            "status": status,
            "gap": gap,
            "requirement_citations": requirements,
            "evidence_citations": evidence,
        }
    )


def assess(question: str, sources: Sequence[Source], complete: Completer) -> Assessment:
    """Produce a checked assessment. Skips the model when no requirement was retrieved."""
    if not any(source.kind == "requirements" for source in sources):
        return Assessment(summary=NO_REQUIREMENTS_SUMMARY)
    assessment = parse_assessment(complete(build_prompt(question, sources)))
    checked = (check_finding(finding, sources) for finding in assessment.findings)
    findings = [finding for finding in checked if finding is not None]
    if not findings:
        return Assessment(summary=NO_REQUIREMENTS_SUMMARY)
    return Assessment(summary=assessment.summary, findings=findings)


def nebius_api_base() -> str:
    """Return the Token Factory endpoint, honoring the NEBIUS_API_BASE override."""
    return os.getenv("NEBIUS_API_BASE", "").strip() or DEFAULT_NEBIUS_API_BASE


def nebius_api_key() -> str:
    """Return the Nebius API key, or fail with a clear message."""
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set NEBIUS_API_KEY in .env")
    return api_key


def nebius_embed_model() -> BaseEmbedding:
    """Create the Nebius embedding model."""
    from llama_index.embeddings.nebius import NebiusEmbedding

    return NebiusEmbedding(
        model_name=os.getenv("RAG_EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL,
        api_key=nebius_api_key(),
        api_base=nebius_api_base(),
    )


def nebius_completer() -> Completer:
    """Create a completion function backed by a Nebius chat model."""
    from llama_index.llms.nebius import NebiusLLM

    llm = NebiusLLM(
        model=os.getenv("RAG_MODEL", "").strip() or DEFAULT_CHAT_MODEL,
        api_key=nebius_api_key(),
        api_base=nebius_api_base(),
        temperature=0,
    )
    return lambda prompt: llm.complete(prompt).text
