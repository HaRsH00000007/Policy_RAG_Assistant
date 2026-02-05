from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks based on word count.
    
    Args:
        text: Input text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        if end >= len(words):
            break
        
        start = end - overlap
    
    return chunks


def chunk_documents(documents: List[dict], chunk_size: int = 500, overlap: int = 100) -> List[dict]:
    """
    Chunk multiple documents while preserving metadata.
    
    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    chunked_docs = []
    
    for doc in documents:
        text = doc["text"]
        metadata = doc.get("metadata", {})
        
        chunks = chunk_text(text, chunk_size, overlap)
        
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            })
    
    return chunked_docs