"""Streamlit interface for the Chroma compliance discovery agent."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from compliance import (
    assess,
    chroma_path,
    load_index,
    nebius_completer,
    nebius_embed_model,
    retrieve,
)

STATUS_LABELS = {"met": "🟢 Met", "weak": "🟡 Weak evidence", "missing": "🔴 Missing evidence"}

load_dotenv()
st.set_page_config(page_title="Compliance Discovery")
st.title("Compliance Discovery with Chroma")
st.caption("Finds obligations and evidence gaps in the indexed documents. Not legal advice.")

with st.sidebar:
    top_k = st.slider("Chunks per document kind", 1, 8, 4)

question = st.chat_input("Ask about obligations, controls, vendors, or audit evidence")
if question:
    st.chat_message("user").write(question)
    try:
        index = load_index(nebius_embed_model(), chroma_path())
        sources = retrieve(index, question, top_k)
        assessment = assess(question, sources, nebius_completer())
    except (RuntimeError, ValueError) as error:
        st.error(str(error))
        st.stop()

    with st.chat_message("assistant"):
        st.write(assessment.summary)
        for finding in assessment.findings:
            st.subheader(STATUS_LABELS[finding.status])
            st.write(finding.obligation)
            if finding.gap:
                st.warning(finding.gap)
            cited = finding.requirement_citations + finding.evidence_citations
            st.caption("Sources: " + " ".join(f"[{number}]" for number in cited))
        with st.expander("Retrieved sources"):
            for source in sources:
                st.markdown(f"**[{source.number}] {source.file}** ({source.kind})")
                st.write(source.text)
