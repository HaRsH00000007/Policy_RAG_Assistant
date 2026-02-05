# Policy_RAG_Assistant

A minimal Retrieval-Augmented Generation (RAG) system that answers questions about company policy documents using grounded retrieval and structured prompting.

This project focuses on **prompt engineering, hallucination reduction, and evaluation**, rather than complex UI or heavy frameworks.

---

## Overview

The Policy RAG Assistant allows users to upload policy documents (PDF, TXT, MD) and ask questions about them.

The system:

- Retrieves relevant document chunks using semantic search
- Generates grounded answers using an LLM
- Avoids hallucinations using strict prompt design
- Provides structured evaluation metrics for responses

---

## Architecture Overview
User Question
│
▼
Semantic Retrieval (ChromaDB)
│
▼
Top-K Relevant Chunks
│
▼
Prompt Builder (Initial / Improved)
│
▼
Groq LLM (Llama 3.1)
│
▼
Structured JSON Response + Evaluation


