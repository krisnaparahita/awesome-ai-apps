# Chroma Compliance Discovery Agent

> Ask what your policies require and whether your audit evidence backs it up, with citations and a list of the gaps.

A runnable RAG application for security, compliance, and audit teams. It indexes requirement documents (policies and standards) and evidence documents (review records, reports, and registers) in a persistent Chroma database, then answers natural-language questions by listing each obligation, rating its evidence, and citing the source chunks. Built with Chroma, LlamaIndex, Nebius Token Factory, and Streamlit.

This is a demonstration of evidence discovery, not legal or audit advice.

## 🚀 Features

- **Obligation summaries**: Turns policy text into a list of the obligations that apply to a question.
- **Evidence gap detection**: Rates each obligation as met, weak, or missing, and says what is lacking.
- **Checked citations**: Citations must point at retrieved sources of the right kind. An obligation with no requirement citation is dropped, and a "met" or "weak" rating with no evidence citation becomes "missing".
- **Balanced retrieval**: Requirements and evidence are retrieved separately, so a long policy can't crowd out the evidence.
- **Persistent index**: The Chroma database lives on disk and survives app restarts, so nothing is re-embedded.

## 🛠️ Tech Stack

- **Python**: Core programming language
- **Streamlit**: Chat interface
- **LlamaIndex**: Chunking, indexing, retrieval, and model access
- **Chroma**: Persistent vector database
- **Nebius Token Factory**: Embeddings and chat completions

## Workflow

1. `ingest.py` chunks the documents in `data/` by section. The folder sets each chunk's kind: `data/requirements/` or `data/evidence/`.
2. Chunks are embedded with Nebius and written to the Chroma database in `chroma_db/`.
3. For each question, `compliance.retrieve` runs one filtered search per kind.
4. The model returns a JSON assessment with cited findings.
5. `compliance.assess` validates the JSON and applies the citation rules before anything is shown.

## 📦 Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) or pip for package management
- An API key for [Nebius Token Factory](https://tokenfactory.nebius.com/)

### Environment Variables

Copy `.env.example` to `.env` and set your key:

```env
NEBIUS_API_KEY="your_nebius_api_key"
```

`CHROMA_PATH`, `RAG_MODEL`, `RAG_EMBEDDING_MODEL`, and `NEBIUS_API_BASE` are optional overrides.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Arindam200/awesome-ai-apps.git
   cd awesome-ai-apps/rag_apps/chroma_compliance_discovery_agent
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

   or with pip:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   pip install -e .
   ```

## ⚙️ Usage

1. **Index the sample documents.** This rebuilds the collection, so rerun it after changing `data/`:

   ```bash
   uv run python ingest.py
   ```

2. **Run the application:**

   ```bash
   uv run streamlit run app.py
   ```

3. **Open your browser** to `http://localhost:8501` and try the questions below. The sample documents contain deliberate gaps.

   - `Are we meeting our quarterly access review obligation?` (a system was skipped and one sign-off is pending)
   - `Do all vendors that process customer data meet the vendor standard?` (one report is stale and one agreement is missing)
   - `Have we run an incident response tabletop exercise this year?` (no evidence exists)
   - `Is MFA enforced for production access?` (fully evidenced)

### Using your own documents

Put markdown files with a top-level `#` heading in `data/requirements/` or `data/evidence/`, then rerun `ingest.py`.

### Tests

The tests use deterministic embeddings and a real on-disk Chroma database, so they need no API key.

```bash
uv run --extra test pytest
```

## 📂 Project Structure

```
chroma_compliance_discovery_agent/
├── data/
│   ├── evidence/         # Sample audit evidence
│   └── requirements/     # Sample policies and standards
├── tests/                # Assessment, retrieval, and persistence tests
├── .env.example          # Environment variable template
├── app.py                # Streamlit interface
├── compliance.py         # Indexing, retrieval, and checked assessments
├── ingest.py             # Rebuilds the Chroma index from data/
├── pyproject.toml        # Python dependencies
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See the [CONTRIBUTING.md](https://github.com/Arindam200/awesome-ai-apps/blob/main/CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Arindam200/awesome-ai-apps/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

- [Chroma](https://www.trychroma.com/) for the vector database.
- [LlamaIndex](https://www.llamaindex.ai/) for orchestration.
- [Nebius Token Factory](https://tokenfactory.nebius.com/) for inference.
