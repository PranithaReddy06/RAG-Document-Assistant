# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their content. It retrieves relevant document information and generates context-grounded answers using a locally running LLM.

## Features

- Upload and index PDF documents.
- Extract and split PDF content into manageable chunks.
- Generate embeddings using HuggingFace Sentence Transformers.
- Store and retrieve document embeddings using ChromaDB.
- Perform semantic similarity search.
- Rerank retrieved chunks using FlashRank.
- Generate answers using Ollama.
- Validate generated answers and provide a fallback when information is unavailable.
- Support document-specific question answering.
- Prevent duplicate PDF indexing.
- Persistent storage for uploaded PDFs and ChromaDB.
- REST APIs built with FastAPI.
- Docker and Docker Compose support.
- API testing using Pytest.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| RAG Framework | LangChain |
| LLM | Ollama |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Database | ChromaDB |
| Reranking | FlashRank |
| PDF Processing | PyPDF |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |
| Frontend | HTML, CSS, JavaScript |

## Project Structure

```text
RAG-Document-Assistant/
│
├── app/
│   ├── core/
│   │   └── config.py
│   ├── rag/
│   │   ├── ingestion.py
│   │   ├── service.py
│   │   └── vectorstore.py
│   ├── schemas/
│   │   └── question.py
│   └── main.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
└── .gitignore
```

## How It Works

1. User uploads a PDF document.
2. The application extracts text from the PDF.
3. Text is split into smaller chunks.
4. HuggingFace generates embeddings for the chunks.
5. Embeddings and metadata are stored in ChromaDB.
6. User submits a question.
7. ChromaDB retrieves relevant chunks using similarity search.
8. FlashRank reranks the retrieved results.
9. Ollama generates an answer using the retrieved context.
10. The application validates the answer and returns the response with source information.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload-pdf` | Upload and index a PDF |
| GET | `/documents` | Retrieve uploaded documents |
| POST | `/ask` | Ask questions about a document |

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/PranithaReddy06/RAG-Document-Assistant.git
cd RAG-Document-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv myenv
```

Activate it on Windows:

```powershell
myenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
LLM_MODEL=your_ollama_model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_BASE_URL=http://localhost:11434
```

Replace `your_ollama_model` with the Ollama model you have installed.

Make sure Ollama is running and the configured model is available.

### 5. Start the backend

```bash
uvicorn app.main:app --reload
```

### 6. Access API documentation

Open:

http://127.0.0.1:8000/docs

## Docker Setup

Make sure Docker Desktop and Ollama are running.

Configure `OLLAMA_BASE_URL` in `.env` as:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Build and start the application:

```bash
docker compose up -d --build
```

Stop containers:

```bash
docker compose down
```

The Docker Compose configuration uses persistent volumes for ChromaDB and uploaded PDFs, so data remains available after container recreation.

API documentation:

http://localhost:8000/docs

## Running Tests

```bash
pytest -v
```

## Future Improvements

- Automated RAG evaluation.
- Authentication and document-level access control.
- Background document processing.
- Production deployment and monitoring.

---

**Developed by:** Pranitha Reddy

**GitHub:** https://github.com/PranithaReddy06