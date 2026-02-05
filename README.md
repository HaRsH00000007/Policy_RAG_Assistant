# Policy_RAG_Assistant

# Policy RAG Assistant

A minimal Retrieval-Augmented Generation system for answering questions about company policy documents.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Groq API key:
```bash
export GROQ_API_KEY="your-key-here"
```

3. Add policy documents to `data/policies/` (PDF, TXT, or MD files)

## Usage

### Streamlit UI
```bash
streamlit run app.py
```

### CLI
```bash
python main.py "What is the vacation policy?"
```

## Architecture

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: ChromaDB (local)
- **LLM**: Groq API with llama-3.1-8b-instant
- **Chunking**: 500 words with 100-word overlap

## Features

- Document loading (PDF/TXT/MD)
- Semantic search with ChromaDB
- Prompt engineering comparison
- JSON-structured responses with confidence scores
- Query logging
