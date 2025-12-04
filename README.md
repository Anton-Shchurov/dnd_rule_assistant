# D&D Rule Assistant (RAG)

**A production-ready Retrieval-Augmented Generation (RAG) system for Dungeons & Dragons 5e rules.**

This project demonstrates the implementation of an advanced RAG pipeline designed to answer complex rule questions from the *Dungeon Master's Guide* (DMG). It goes beyond simple text matching by handling complex PDF layouts, implementing hybrid search with reranking, and providing a polished Telegram Bot interface.

---

## 🚀 Project Essence

The goal was to create an assistant capable of navigating the intricate and often cross-referenced rules of D&D. Unlike standard RAG tutorials, this project tackles real-world data challenges: multi-column PDF layouts, tables, and the need for precise, context-aware retrieval.

**Key Features:**
- **High-Fidelity Parsing**: Converts complex PDFs into structured Markdown using **Docling** (with OCR support).
- **Smart Retrieval**: Uses dense vector embeddings (Qdrant) with a two-stage reranking process.
- **Context Optimization**: Dynamically adjusts retrieval depth (`k*4` candidates) to maximize recall before reranking.
- **Production Ready**: Fully Dockerized application with a Telegram interface.

---

## 💡 Challenges & Solutions

During development, I encountered several engineering challenges. Here is how they were solved:

| Challenge | Solution |
|-----------|----------|
| **Complex PDF Layouts** | Standard parsers failed on D&D's multi-column layout. I utilized **Docling** (with OCR) to accurately extract text, tables, and headers, converting them into clean Markdown while preserving the document structure. |
| **Retrieval Accuracy** | Simple vector search often missed specific rule keywords. I implemented a **Two-Stage Pipeline**: retrieving a broad set of candidates (`initial_k = 4 * k`) via dense embeddings, then filtering them with a **Cross-Encoder Reranker**. |
| **Parameter Tuning** | Balancing recall and context window size was tricky. I experimented with various `top_k` and `retrieval_k` values, settling on a dynamic expansion strategy to capture relevant context without overwhelming the LLM. |
| **Context Relevance** | To reduce hallucinations, the reranker re-scores the retrieved chunks, ensuring the LLM receives only the most pertinent information. |
| **Deployment** | To ensure the system runs reliably on any server, I containerized the entire application (Bot + Vector DB) using **Docker** and **Docker Compose**. |

---

## 🛠 Tech Stack

- **Language**: Python 3.12
- **RAG & Database**: Qdrant (Vector Store), OpenAI API (Embeddings & Generation)
- **Parsing**: Docling
- **Interface**: Aiogram (Telegram Bot), Typer (CLI)
- **Infrastructure**: Docker, Docker Compose, Poetry

---

## 📂 Project Structure

```text
dnd_rule_assistant/
├── docker-compose.yml      # Service orchestration (Bot + Qdrant)
├── Dockerfile              # Production image definition
├── pyproject.toml          # Dependency management
├── src/
│   ├── core/               # RAG pipelines and business logic
│   ├── providers/          # Clients for OpenAI, Qdrant
│   └── interfaces/         # Telegram Bot and CLI entry points
├── configs/                # Configuration files
└── data/                   # (Excluded from git) Processed knowledge base
```

---

## ⚡ Quick Start

The project is designed to be deployed with a single command. **The vector database is automatically restored from a snapshot** — no manual indexing required.

### Prerequisites
- Docker & Docker Compose
- OpenAI API Key
- Telegram Bot Token

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Anton-Shchurov/dnd_rule_assistant.git
   cd dnd_rule_assistant
   ```

2. **Configure Environment**:
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=sk-...
   TELEGRAM_BOT_TOKEN=123456:ABC...
   ```

3. **Run with Docker**:
   ```bash
   docker-compose up -d --build
   ```
   On first launch, the system automatically restores the pre-built vector database from a snapshot.

4. **Interact**:
   Open your bot in Telegram and send `/start`. Ask any rule question, e.g., *"How does grappling work?"*

---

## 🔧 Development

### Creating a New Snapshot

If you update the knowledge base, create a new snapshot:
```bash
poetry run python -m dnd_rag.interfaces.cli snapshot --collection dnd_rule_assistant
```

---

*Created by Anton Shchurov as a portfolio project demonstrating advanced Agentic RAG systems.*