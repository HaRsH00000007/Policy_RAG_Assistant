from groq import Groq
from typing import List, Dict
from src.vectorstore import VectorStore
from src.prompts import get_prompt
from src.utils import safe_json_parse, log_query, get_groq_api_key, evaluate_response

import os
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    """Main RAG pipeline for question answering."""

    def __init__(self, vector_store: VectorStore, model: str = "llama-3.1-8b-instant"):
        """Initialize RAG pipeline."""
        self.vector_store = vector_store
        self.model = model
        self.client = Groq(api_key=get_groq_api_key())

    def query(self, question: str, prompt_type: str = "improved", top_k: int = 5) -> Dict:
        """
        Answer a question using RAG.
        """

        # ------------------------------------------------
        # 1️⃣ Retrieve relevant documents
        # ------------------------------------------------
        retrieved_chunks = self.vector_store.search(question, top_k=top_k)

        # Apply simple reranking (BONUS FEATURE)
        if retrieved_chunks:
            retrieved_chunks = self.rerank_simple(retrieved_chunks, question)

        # ------------------------------------------------
        # 2️⃣ Handle case where nothing retrieved
        # ------------------------------------------------
        if not retrieved_chunks:
            response = {
                "answer": "I don't know based on the provided documents.",
                "evidence": [],
                "confidence": "Low",
                "retrieved_chunks": []
            }

            #  Add evaluation metrics
            evaluation = evaluate_response(question, response, prompt_type)
            response["evaluation"] = evaluation

            log_query(question, [], response, prompt_type)
            return response

        # ------------------------------------------------
        # 3️⃣ Build context
        # ------------------------------------------------
        context = self._build_context(retrieved_chunks)

        # (Optional safety) Prevent overly long context
        context = context[:4000]

        # ------------------------------------------------
        # 4️⃣ Create prompt
        # ------------------------------------------------
        prompt = get_prompt(prompt_type, context, question)

        # ------------------------------------------------
        # 5️⃣ Call Groq API
        # ------------------------------------------------
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  #  more deterministic for RAG
                max_tokens=1024
            )

            response_text = completion.choices[0].message.content

            # ------------------------------------------------
            # 6️⃣ Parse response
            # ------------------------------------------------
            if prompt_type == "improved":
                parsed = safe_json_parse(response_text)

                if parsed:
                    response = {
                        "answer": parsed.get("answer", response_text),
                        "evidence": parsed.get("evidence", []),
                        "confidence": parsed.get("confidence", "Medium"),
                        "retrieved_chunks": retrieved_chunks
                    }
                else:
                    # Fallback if JSON parsing fails
                    response = {
                        "answer": response_text,
                        "evidence": [],
                        "confidence": "Medium",
                        "retrieved_chunks": retrieved_chunks
                    }
            else:
                response = {
                    "answer": response_text,
                    "evidence": [],
                    "confidence": "N/A",
                    "retrieved_chunks": retrieved_chunks
                }

            # ------------------------------------------------
            #  7️⃣ Add Evaluation Metrics (NEW)
            # ------------------------------------------------
            evaluation = evaluate_response(question, response, prompt_type)
            response["evaluation"] = evaluation

            # ------------------------------------------------
            # 8️⃣ Log Query
            # ------------------------------------------------
            log_query(question, retrieved_chunks, response, prompt_type)

            return response

        except Exception as e:
            print(f"Error calling LLM: {e}")

            response = {
                "answer": "The system encountered an error while generating a response.",
                "evidence": [],
                "confidence": "Low",
                "retrieved_chunks": retrieved_chunks
            }

            evaluation = evaluate_response(question, response, prompt_type)
            response["evaluation"] = evaluation

            return response

    # ------------------------------------------------
    # Helper: Build Context
    # ------------------------------------------------
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            text = chunk["text"]
            context_parts.append(f"[Document {i} - {source}]\n{text}\n")

        return "\n".join(context_parts)

    # ------------------------------------------------
    # BONUS: Simple Reranker
    # ------------------------------------------------
    def rerank_simple(self, chunks: List[Dict], question: str) -> List[Dict]:
        """
        Simple reranking based on keyword overlap.
        """
        question_words = set(question.lower().split())

        for chunk in chunks:
            text_words = set(chunk["text"].lower().split())
            overlap = len(question_words & text_words)
            chunk["keyword_score"] = overlap

        reranked = sorted(
            chunks,
            key=lambda x: (x.get("keyword_score", 0), -x.get("score", 0)),
            reverse=True
        )

        return reranked
