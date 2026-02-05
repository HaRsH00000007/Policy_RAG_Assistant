import streamlit as st
from src.loader import load_documents
from src.chunking import chunk_documents
from src.vectorstore import VectorStore
from src.rag_pipeline import RAGPipeline
from src.utils import ensure_directories
from src.evaluation import analyze_confidence_distribution
import os
import tempfile
from pathlib import Path


# Page config
st.set_page_config(page_title="Policy RAG Assistant", layout="wide")

# Initialize
ensure_directories()

# Check API key
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not set. Please set it as an environment variable.")
    st.stop()

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "uploaded_files_count" not in st.session_state:
    st.session_state.uploaded_files_count = 0

# Title
st.title("Policy RAG Assistant")
st.markdown("Ask questions about company policies")

# Sidebar
with st.sidebar:
    st.header("Setup")

    upload_method = st.radio(
        "Choose upload method:",
        ["Upload files here", "Load from data/policies/"],
        key="upload_method"
    )

    if upload_method == "Upload files here":
        uploaded_files = st.file_uploader(
            "Upload policy documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Process Uploaded Files"):
            with st.spinner("Processing uploaded files..."):
                from src.loader import load_pdf, load_text

                docs = []
                for uploaded_file in uploaded_files:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = Path(tmp_file.name)

                        if tmp_path.suffix.lower() == ".pdf":
                            text = load_pdf(tmp_path)
                        elif tmp_path.suffix.lower() in [".txt", ".md"]:
                            text = load_text(tmp_path)
                        else:
                            continue

                        if text.strip():
                            docs.append({
                                "text": text,
                                "metadata": {
                                    "source": uploaded_file.name,
                                    "type": tmp_path.suffix[1:]
                                }
                            })

                        tmp_path.unlink()

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

                if docs:
                    chunked = chunk_documents(docs, chunk_size=500, overlap=100)

                    vector_store = VectorStore()
                    vector_store.reset()
                    vector_store.add_documents(chunked)

                    st.session_state.vector_store = vector_store
                    st.session_state.rag_pipeline = RAGPipeline(vector_store)
                    st.session_state.uploaded_files_count = len(docs)

                    st.success(f"Processed {len(docs)} documents, {len(chunked)} chunks")
                else:
                    st.warning("No valid documents were processed")

    else:
        if st.button("Load Documents from Folder"):
            with st.spinner("Loading documents..."):
                docs = load_documents()
                if docs:
                    chunked = chunk_documents(docs, chunk_size=500, overlap=100)

                    vector_store = VectorStore()
                    vector_store.reset()
                    vector_store.add_documents(chunked)

                    st.session_state.vector_store = vector_store
                    st.session_state.rag_pipeline = RAGPipeline(vector_store)
                    st.session_state.uploaded_files_count = len(docs)

                    st.success(f"Loaded {len(docs)} documents, {len(chunked)} chunks")
                else:
                    st.warning("No documents found in data/policies/")

    if st.session_state.vector_store:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", st.session_state.uploaded_files_count)
        with col2:
            st.metric("Total Chunks", st.session_state.vector_store.count())

    st.divider()

    st.header("Analytics")
    if st.button("View Stats"):
        stats = analyze_confidence_distribution()
        st.json(stats)

# Main area
if st.session_state.rag_pipeline is None:
    st.info("Upload documents or load from folder in the sidebar to get started")
else:
    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_input("Ask a question:", placeholder="e.g., What is the vacation policy?")

    with col2:
        prompt_type = st.selectbox("Prompt:", ["improved", "initial", "compare"])

    if question:
        if prompt_type == "compare":

            colA, colB = st.columns(2)

            with colA:
                st.subheader("Initial Prompt Result")
                result_initial = st.session_state.rag_pipeline.query(question, prompt_type="initial")
                st.write(result_initial["answer"])
                st.metric("Confidence", result_initial.get("confidence", "N/A"))
                if result_initial.get("evaluation"):
                    st.json(result_initial["evaluation"])

            with colB:
                st.subheader("Improved Prompt Result")
                result_improved = st.session_state.rag_pipeline.query(question, prompt_type="improved")
                st.write(result_improved["answer"])
                st.metric("Confidence", result_improved.get("confidence", "N/A"))
                if result_improved.get("evaluation"):
                    st.json(result_improved["evaluation"])

            display_chunks = result_improved["retrieved_chunks"]

        else:
            with st.spinner("Searching..."):
                response = st.session_state.rag_pipeline.query(question, prompt_type=prompt_type)

            st.markdown("### Answer")
            st.write(response["answer"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", response.get("confidence", "N/A"))
            with col2:
                st.metric("Sources Used", len(response["retrieved_chunks"]))

            if response.get("evaluation"):
                st.subheader("Evaluation")
                st.json(response["evaluation"])

            if response.get("evidence"):
                with st.expander("Evidence"):
                    for i, ev in enumerate(response["evidence"], 1):
                        st.markdown(f"{i}. {ev}")

            display_chunks = response["retrieved_chunks"]

        with st.expander("Retrieved Chunks"):
            for i, chunk in enumerate(display_chunks, 1):
                st.markdown(f"Chunk {i} (score: {chunk.get('score', 0):.4f})")
                st.markdown(f"Source: {chunk.get('metadata', {}).get('source', 'Unknown')}")
                st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                st.divider()
