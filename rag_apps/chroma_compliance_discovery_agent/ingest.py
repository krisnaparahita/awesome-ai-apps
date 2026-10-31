"""Rebuild the Chroma index from the documents in data/."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from compliance import build_index, chroma_path, load_nodes, nebius_embed_model

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    load_dotenv()
    count = build_index(load_nodes(DATA_DIR), nebius_embed_model(), chroma_path())
    print(f"Indexed {count} chunks into {chroma_path()}")


if __name__ == "__main__":
    main()
