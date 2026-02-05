import sys
import os
from dotenv import load_dotenv

from src.loader import load_documents
from src.chunking import chunk_documents
from src.vectorstore import VectorStore
from src.rag_pipeline import RAGPipeline
from src.utils import ensure_directories

# Load environment variables
load_dotenv()


def setup_vector_store():
    """Initialize and populate vector store."""
    print("Loading documents...")
    docs = load_documents()

    if not docs:
        print("No documents found in data/policies/")
        sys.exit(1)

    print(f"Loaded {len(docs)} documents")

    print("Chunking documents...")
    chunked = chunk_documents(docs, chunk_size=500, overlap=100)
    print(f"Created {len(chunked)} chunks")

    print("Initializing vector store...")
    vector_store = VectorStore()
    vector_store.reset()
    vector_store.add_documents(chunked)

    print("Setup complete!")
    return vector_store


def main():
    """CLI interface for RAG pipeline."""
    ensure_directories()

    # ------------------------------------------------
    # Check API key
    # ------------------------------------------------
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        sys.exit(1)

    # ------------------------------------------------
    # Get question from command line
    # ------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python main.py 'Your question here'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    # ------------------------------------------------
    # Setup RAG pipeline
    # ------------------------------------------------
    vector_store = setup_vector_store()
    rag_pipeline = RAGPipeline(vector_store)

    # ------------------------------------------------
    # Query
    # ------------------------------------------------
    print(f"\nQuestion: {question}\n")

    response = rag_pipeline.query(question, prompt_type="improved")

    # ------------------------------------------------
    # Display Results
    # ------------------------------------------------
    print("=" * 80)
    print("ANSWER:")
    print(response["answer"])

    print("\n" + "=" * 80)
    print(f"Confidence: {response.get('confidence', 'N/A')}")
    print(f"Sources Retrieved: {len(response['retrieved_chunks'])}")

    # Show retrieved chunk preview ( looks professional)
    if response.get("retrieved_chunks"):
        print("\nRETRIEVED CONTEXT PREVIEW:")
        for i, chunk in enumerate(response["retrieved_chunks"], 1):
            preview = chunk["text"][:120].replace("\n", " ")
            print(f"{i}. {preview}...")

    if response.get("evidence"):
        print("\nEVIDENCE:")
        for i, ev in enumerate(response["evidence"], 1):
            print(f"{i}. {ev}")

    #  NEW: Evaluation Metrics
    if response.get("evaluation"):
        print("\n" + "=" * 80)
        print("EVALUATION:")
        for k, v in response["evaluation"].items():
            print(f"{k}: {v}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
