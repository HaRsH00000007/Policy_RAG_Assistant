import os
import json
from datetime import datetime
from pathlib import Path


def ensure_directories():
    """Create necessary directories if they don't exist."""
    Path("data/policies").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("chroma_db").mkdir(parents=True, exist_ok=True)


def log_query(question, retrieved_chunks, response, prompt_type="improved"):
    """Log query details to JSONL file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "prompt_type": prompt_type,
        "num_chunks_retrieved": len(retrieved_chunks),
        "chunks": [
            {
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "metadata": chunk.get("metadata", {})
            }
            for chunk in retrieved_chunks
        ],
        "response": response
    }

    log_file = "logs/queries.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def get_groq_api_key():
    """Get Groq API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return api_key


def safe_json_parse(text):
    """Safely parse JSON from LLM response."""
    try:
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
        return None
    except Exception:
        return None


# ============================================================
# ⭐ NEW: Simple RAG Evaluation Metrics
# ============================================================

def evaluate_response(question: str, response: dict, prompt_type: str) -> dict:
    """
    Generate simple evaluation metrics for RAG output.

    Metrics:
    - Accuracy (basic heuristic)
    - Groundedness (based on evidence presence)
    - Hallucination Risk
    - Prompt Version
    """

    answer = response.get("answer", "")
    evidence = response.get("evidence", [])

    # ---------------------------
    # Accuracy (simple heuristic)
    # ---------------------------
    if isinstance(answer, str) and answer.startswith("I don't know"):
        accuracy = "⚠️"
    else:
        accuracy = "✅"

    # ---------------------------
    # Groundedness
    # ---------------------------
    groundedness = "✅" if evidence else "⚠️"

    # ---------------------------
    # Hallucination Risk
    # ---------------------------
    if isinstance(answer, str) and answer.startswith("I don't know"):
        hallucination = "LOW"
    elif evidence:
        hallucination = "LOW"
    else:
        hallucination = "MEDIUM"

    evaluation = {
        "Accuracy": accuracy,
        "Groundedness": groundedness,
        "Hallucination Risk": hallucination,
        "Prompt Version": prompt_type
    }

    return evaluation
