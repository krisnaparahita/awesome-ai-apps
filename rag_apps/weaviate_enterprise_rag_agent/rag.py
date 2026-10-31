"""Weaviate storage, hybrid retrieval, and cited answer generation."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import weaviate
from openai import OpenAI
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery
from weaviate.util import generate_uuid5

DEFAULT_NEBIUS_API_BASE = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_CHAT_MODEL = "MiniMaxAI/MiniMax-M3"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
COLLECTION = "EnterpriseChunk"
CHUNK_WORDS = 180
CHUNK_OVERLAP = 30
NO_EVIDENCE_ANSWER = "I could not find this in the indexed documents."

Embedder = Callable[[Sequence[str]], list[list[float]]]
Completer = Callable[[str, str], str]


@dataclass(frozen=True)
class Chunk:
    """A section of a source document, ready to index."""

    source: str
    title: str
    doc_type: str
    team: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk and its hybrid relevance score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    """Generated text and the retrieved chunks it actually cites."""

    text: str
    citations: list[Hit]


def parse_document(path: Path) -> tuple[dict[str, str], str]:
    """Split a markdown file into its front matter fields and body."""
    match = re.match(r"---\n(.*?)\n---\n(.*)", path.read_text(encoding="utf-8"), re.DOTALL)
    if not match:
        raise ValueError(f"{path.name} has no front matter")
    lines = [line.split(":", 1) for line in match.group(1).splitlines() if ":" in line]
    fields = {key.strip(): value.strip() for key, value in lines}
    missing = {"title", "doc_type", "team"} - fields.keys()
    if missing:
        raise ValueError(f"{path.name} is missing front matter: {sorted(missing)}")
    return fields, match.group(2)


def split_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words) - overlap, step)]


def load_chunks(data_dir: Path) -> list[Chunk]:
    """Read every markdown document and chunk it by section."""
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")):
        fields, body = parse_document(path)
        sections = [s for s in re.split(r"(?m)^(?=#{1,3} )", body) if s.strip()]
        texts = [text for section in sections for text in split_words(section)]
        chunks.extend(
            Chunk(path.name, fields["title"], fields["doc_type"], fields["team"], index, text)
            for index, text in enumerate(texts)
        )
    return chunks


def connect() -> weaviate.WeaviateClient:
    """Connect to Weaviate Cloud when an API key is set, otherwise to a local instance."""
    url = os.getenv("WEAVIATE_URL", "").strip() or "http://localhost:8080"
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()
    if api_key:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=url, auth_credentials=Auth.api_key(api_key)
        )
    parsed = urlparse(url)
    return weaviate.connect_to_local(host=parsed.hostname or "localhost", port=parsed.port or 8080)


def recreate_collection(client: weaviate.WeaviateClient, name: str = COLLECTION) -> None:
    """Drop and rebuild the collection. Vectors are supplied by this app, not by Weaviate."""
    client.collections.delete(name)
    keyword = {"data_type": DataType.TEXT, "tokenization": Tokenization.FIELD}
    client.collections.create(
        name,
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", index_searchable=False, **keyword),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="doc_type", index_searchable=False, **keyword),
            Property(name="team", index_searchable=False, **keyword),
            Property(name="chunk_index", data_type=DataType.INT),
        ],
    )


def index_chunks(
    client: weaviate.WeaviateClient,
    chunks: Sequence[Chunk],
    embed: Embedder,
    name: str = COLLECTION,
) -> int:
    """Embed chunks and write them. Returns the number written."""
    vectors = embed([chunk.text for chunk in chunks])
    collection = client.collections.get(name)
    with collection.batch.fixed_size(batch_size=64) as batch:
        for chunk, vector in zip(chunks, vectors, strict=True):
            batch.add_object(
                uuid=generate_uuid5(f"{chunk.source}:{chunk.chunk_index}"),
                properties={
                    "text": chunk.text,
                    "source": chunk.source,
                    "title": chunk.title,
                    "doc_type": chunk.doc_type,
                    "team": chunk.team,
                    "chunk_index": chunk.chunk_index,
                },
                vector=vector,
            )
    failed = collection.batch.failed_objects
    if failed:
        raise RuntimeError(f"{len(failed)} chunks failed to index: {failed[0].message}")
    return len(chunks)


def build_filters(
    doc_types: Iterable[str] = (), teams: Iterable[str] = (), sources: Iterable[str] = ()
) -> Filter | None:
    """AND together one any-of filter per metadata field that has values."""
    selected = {"doc_type": list(doc_types), "team": list(teams), "source": list(sources)}
    filters = [Filter.by_property(key).contains_any(val) for key, val in selected.items() if val]
    return Filter.all_of(filters) if filters else None


def search(
    client: weaviate.WeaviateClient,
    query: str,
    embed: Embedder,
    *,
    filters: Filter | None = None,
    alpha: float = 0.5,
    limit: int = 5,
    name: str = COLLECTION,
) -> list[Hit]:
    """Run hybrid search. alpha=0 is pure BM25 keyword search, alpha=1 pure vector search."""
    response = client.collections.get(name).query.hybrid(
        query=query,
        vector=embed([query])[0],
        alpha=alpha,
        limit=limit,
        filters=filters,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        return_metadata=MetadataQuery(score=True),
    )
    return [
        Hit(
            Chunk(
                source=obj.properties["source"],
                title=obj.properties["title"],
                doc_type=obj.properties["doc_type"],
                team=obj.properties["team"],
                chunk_index=int(obj.properties["chunk_index"]),
                text=obj.properties["text"],
            ),
            obj.metadata.score or 0.0,
        )
        for obj in response.objects
    ]


SYSTEM_PROMPT = (
    "You answer questions about internal company documents. Use only the numbered "
    "sources. Put a citation such as [1] after every factual claim. If the sources do "
    f"not contain the answer, reply exactly: {NO_EVIDENCE_ANSWER}"
)


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    """Number the retrieved chunks so the model can cite them."""
    sources = "\n\n".join(
        f"[{number}] {hit.chunk.source} ({hit.chunk.doc_type}, {hit.chunk.team})\n{hit.chunk.text}"
        for number, hit in enumerate(hits, start=1)
    )
    return f"Sources:\n{sources}\n\nQuestion: {question}"


def extract_citations(text: str, hits: Sequence[Hit]) -> list[Hit]:
    """Return cited hits in first-cited order, ignoring numbers that match no source."""
    numbers = dict.fromkeys(int(number) for number in re.findall(r"\[(\d+)\]", text))
    return [hits[number - 1] for number in numbers if 1 <= number <= len(hits)]


def generate_answer(question: str, hits: Sequence[Hit], complete: Completer) -> Answer:
    """Answer from retrieved chunks. Skips the model when nothing was retrieved, and
    refuses an answer that cites none of the sources, since it can't be grounded."""
    if not hits:
        return Answer(NO_EVIDENCE_ANSWER, [])
    text = complete(SYSTEM_PROMPT, build_prompt(question, hits)).strip()
    citations = extract_citations(text, hits)
    return Answer(text, citations) if citations else Answer(NO_EVIDENCE_ANSWER, [])


def nebius_client() -> OpenAI:
    """Create an OpenAI-compatible client for Nebius Token Factory."""
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set NEBIUS_API_KEY in .env")
    base_url = os.getenv("NEBIUS_API_BASE", "").strip() or DEFAULT_NEBIUS_API_BASE
    return OpenAI(api_key=api_key, base_url=base_url)


def nebius_embedder(client: OpenAI) -> Embedder:
    """Build an embedding function backed by Nebius."""
    model = os.getenv("RAG_EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL

    def embed(texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            batch = list(texts[start : start + 32])
            response = client.embeddings.create(model=model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    return embed


def nebius_completer(client: OpenAI) -> Completer:
    """Build a chat completion function backed by Nebius."""
    model = os.getenv("RAG_MODEL", "").strip() or DEFAULT_CHAT_MODEL

    def complete(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    return complete
