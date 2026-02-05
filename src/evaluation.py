import json
from pathlib import Path
from typing import List, Dict


def load_queries_log(log_file: str = "logs/queries.jsonl") -> List[Dict]:
    """Load all logged queries."""
    queries = []
    if not Path(log_file).exists():
        return queries
    
    with open(log_file, "r") as f:
        for line in f:
            queries.append(json.loads(line))
    
    return queries


def analyze_confidence_distribution(log_file: str = "logs/queries.jsonl") -> Dict:
    """Analyze confidence score distribution from logs."""
    queries = load_queries_log(log_file)
    
    confidence_counts = {"High": 0, "Medium": 0, "Low": 0, "N/A": 0}
    
    for query in queries:
        confidence = query.get("response", {}).get("confidence", "N/A")
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
    
    return {
        "total_queries": len(queries),
        "confidence_distribution": confidence_counts
    }


def compare_prompts(question: str, rag_pipeline) -> Dict:
    """Compare initial vs improved prompt responses."""
    initial_response = rag_pipeline.query(question, prompt_type="initial")
    improved_response = rag_pipeline.query(question, prompt_type="improved")
    
    return {
        "question": question,
        "initial": initial_response,
        "improved": improved_response
    }