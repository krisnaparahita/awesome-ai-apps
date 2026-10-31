# Weaviate Enterprise RAG Agent

> Ask questions across internal documents and get answers with citations, using Weaviate hybrid search and metadata filters.

A runnable RAG application for teams that need to search policies, product docs, incident runbooks, and support articles in one place. Exact terms such as `payments_failover` and natural-language questions both work, because Weaviate combines BM25 keyword search with vector search in a single query. Built with Weaviate, Nebius Token Factory, and Streamlit.

## 🚀 Features

- **Hybrid retrieval**: Weaviate BM25 and vector search in one query, with a slider to weight keywords against vectors.
- **Metadata filtering**: Restrict a question by document type, team, or source file.
- **Relevance scoring**: Every retrieved chunk shows its normalized hybrid score.
- **Grounded answers**: Answers cite numbered chunks. Citations are checked against what was retrieved, and an answer that cites nothing is replaced with a refusal.
- **Local or cloud**: Runs on a local Weaviate container or on Weaviate Cloud.

## 🛠️ Tech Stack

- **Python**: Core programming language
- **Streamlit**: Chat interface
- **Weaviate**: Vector and keyword storage, hybrid search, and filtering
- **Nebius Token Factory**: Embeddings and chat completions through its OpenAI-compatible API

## Workflow

1. `ingest.py` reads the markdown files in `data/`, using their front matter (`title`, `doc_type`, `team`) as metadata.
2. Documents are split by section into overlapping chunks, embedded with Nebius, and stored in Weaviate. The app supplies the vectors, so no Weaviate vectorizer module is needed.
3. For each question, `rag.search` runs a hybrid query with the selected filters.
4. The top chunks are numbered and sent to the model, which must cite them.
5. The app shows the answer and each retrieved chunk, marked as cited or not cited.

## 📦 Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) or pip for package management
- Docker for local Weaviate, or a [Weaviate Cloud](https://weaviate.io/developers/wcs) cluster
- An API key for [Nebius Token Factory](https://tokenfactory.nebius.com/)

### Environment Variables

Copy `.env.example` to `.env` and set your key:

```env
NEBIUS_API_KEY="your_nebius_api_key"
WEAVIATE_URL="http://localhost:8080"
WEAVIATE_API_KEY=""
```

For Weaviate Cloud, set `WEAVIATE_URL` to your cluster URL and fill in `WEAVIATE_API_KEY`. With a key set, the app connects to the cloud. Without one, it connects to `WEAVIATE_URL` as a local instance.

`RAG_MODEL`, `RAG_EMBEDDING_MODEL`, and `NEBIUS_API_BASE` are optional overrides.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Arindam200/awesome-ai-apps.git
   cd awesome-ai-apps/rag_apps/weaviate_enterprise_rag_agent
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

3. **Start Weaviate** (skip this for Weaviate Cloud):

   ```bash
   docker compose up -d
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

3. **Open your browser** to `http://localhost:8501` and try:

   - `What do I do if the card processor is down?`
   - `How long are security audit logs retained?` with Document type set to `policy`
   - `How many days can I work from abroad?`
   - `What is our parental leave policy?` (not in the documents, so the app should refuse)

### Using your own documents

Add markdown files to `data/` with this front matter, then rerun `ingest.py`:

```markdown
---
title: Expense Policy
doc_type: policy
team: finance
---
```

### Tests

The tests use deterministic embeddings, so they need no API key. The integration tests run against Weaviate at `WEAVIATE_URL` and skip if it is unreachable.

```bash
uv run --extra test pytest
```

## 📂 Project Structure

```
weaviate_enterprise_rag_agent/
├── data/                 # Sample enterprise documents
├── tests/                # Unit and Weaviate integration tests
├── .env.example          # Environment variable template
├── app.py                # Streamlit interface
├── docker-compose.yml    # Local Weaviate
├── ingest.py             # Rebuilds the collection from data/
├── pyproject.toml        # Python dependencies
├── rag.py                # Retrieval, filtering, and answer generation
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See the [CONTRIBUTING.md](https://github.com/Arindam200/awesome-ai-apps/blob/main/CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Arindam200/awesome-ai-apps/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

- [Weaviate](https://weaviate.io/) for hybrid search.
- [Nebius Token Factory](https://tokenfactory.nebius.com/) for inference.
