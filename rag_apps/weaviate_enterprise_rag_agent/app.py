"""Streamlit interface for the Weaviate enterprise RAG agent."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from ingest import DATA_DIR
from rag import (
    build_filters,
    connect,
    generate_answer,
    load_chunks,
    nebius_client,
    nebius_completer,
    nebius_embedder,
    search,
)

load_dotenv()
st.set_page_config(page_title="Enterprise RAG with Weaviate")
st.title("Enterprise RAG with Weaviate")

chunks = load_chunks(DATA_DIR)

with st.sidebar:
    st.header("Filters")
    doc_types = st.multiselect("Document type", sorted({chunk.doc_type for chunk in chunks}))
    teams = st.multiselect("Team", sorted({chunk.team for chunk in chunks}))
    sources = st.multiselect("Source", sorted({chunk.source for chunk in chunks}))
    st.header("Retrieval")
    alpha = st.slider("Keyword (0) to vector (1)", 0.0, 1.0, 0.5, 0.05)
    limit = st.slider("Chunks to retrieve", 1, 10, 5)

question = st.chat_input("Ask about policies, product docs, runbooks, or support articles")
if question:
    st.chat_message("user").write(question)
    llm_client = nebius_client()
    client = connect()
    try:
        hits = search(
            client,
            question,
            nebius_embedder(llm_client),
            filters=build_filters(doc_types, teams, sources),
            alpha=alpha,
            limit=limit,
        )
    finally:
        client.close()
    answer = generate_answer(question, hits, nebius_completer(llm_client))
    with st.chat_message("assistant"):
        st.write(answer.text)
        for number, hit in enumerate(hits, start=1):
            cited = "cited" if hit in answer.citations else "not cited"
            with st.expander(f"[{number}] {hit.chunk.source} ({hit.score:.2f}, {cited})"):
                st.write(hit.chunk.text)
