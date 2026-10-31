"""Rebuild the Weaviate collection from the documents in data/."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from rag import (
    connect,
    index_chunks,
    load_chunks,
    nebius_client,
    nebius_embedder,
    recreate_collection,
)

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    load_dotenv()
    chunks = load_chunks(DATA_DIR)
    embed = nebius_embedder(nebius_client())
    client = connect()
    try:
        recreate_collection(client)
        count = index_chunks(client, chunks, embed)
    finally:
        client.close()
    print(f"Indexed {count} chunks from {DATA_DIR}")


if __name__ == "__main__":
    main()
