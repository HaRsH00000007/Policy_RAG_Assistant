import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List


class VectorStore:
    """Simple ChromaDB wrapper for document storage and retrieval."""
    
    def __init__(self, collection_name: str = "policy_docs", persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB and embedding model."""
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection_name = collection_name
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[dict]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
        """
        if not documents:
            print("No documents to add")
            return
        
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} chunks to vector store")
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search for relevant documents.
        
        Returns:
            List of dicts with 'text', 'metadata', and 'score' keys
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Format results
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                documents.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": results["distances"][0][i] if results["distances"] else 0
                })
        
        return documents
    
    def reset(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store reset")
    
    def count(self) -> int:
        """Get count of documents in collection."""
        return self.collection.count()