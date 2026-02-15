import streamlit as st
import os
from dotenv import load_dotenv
from ingestion.pdf_loader import load_pdf_text
from ingestion.chunker import chunk_text
from embeddings.embedder import embed_texts
from vector_db.faiss_store import build_faiss_index
from retrieval.search import search
from llm.generator import generate_answer
load_dotenv()

st.set_page_config(page_title="Hybrid RAG Demo", layout="wide")

st.title("📄 Hybrid RAG – Compliance Assistant")
st.write("First load the document and then ask questions.")

# -----------------------------
# Session state (cache objects)
# -----------------------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

# -----------------------------
# Load document button
# -----------------------------
st.sidebar.header("📂 Document Setup")

if st.sidebar.button("Load Document"):
    with st.spinner("Loading and indexing document..."):
        text = load_pdf_text(os.getenv("INPUT_FILE"))
        chunks = chunk_text(text)
        vectors = embed_texts(chunks)
        index = build_faiss_index(vectors)

        st.session_state.index = index
        st.session_state.chunks = chunks

    st.sidebar.success("Document indexed successfully!")

# -----------------------------
# Question input
# -----------------------------
query = st.text_input("🔍 Ask a question:")

# -----------------------------
# Run RAG
# -----------------------------
if query and st.session_state.index:
    with st.spinner("Searching and generating answer..."):
        retrieved_chunks = search(
            query,
            st.session_state.index,
            st.session_state.chunks,
            top_k=3
        )

        context = "\n".join(retrieved_chunks)
        answer = generate_answer(query, context)

    st.subheader("✅ Answer")
    st.write(answer)

    with st.expander("📚 Retrieved Context"):
        for i, chunk in enumerate(retrieved_chunks, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(chunk)

elif query and not st.session_state.index:
    st.warning("Please load the document first from the sidebar.")
