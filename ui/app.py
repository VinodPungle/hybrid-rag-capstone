import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv
from ingestion.pdf_loader import load_pdf_text
from ingestion.chunker import chunk_text
from embeddings.embedder import embed_texts
from vector_db.faiss_store import build_faiss_index
from retrieval.hybrid_search import hybrid_search
from graph_db.entity_extractor import extract_graph_from_chunks
from graph_db.neo4j_store import get_driver, build_knowledge_graph, fetch_graph_visual_data
from streamlit_agraph import agraph, Node, Edge, Config
from llm.generator import generate_answer
load_dotenv()

st.set_page_config(page_title="Hybrid RAG Demo", layout="wide")

st.title("📄 Hybrid RAG – Documents Understanding Assistant")
st.write("First load the document and then ask questions. Uses both vector search and knowledge graph (GraphRAG) for enhanced retrieval.")

# -----------------------------
# Session state (cache objects)
# -----------------------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.neo4j_driver = None

# -----------------------------
# Upload / Load document
# -----------------------------
st.sidebar.header("📂 Document Setup")

uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if st.sidebar.button("Load Document"):
    # Determine PDF source: uploaded file or INPUT_FILE env var
    if uploaded_file is not None:
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = tmp_path
        st.sidebar.info(f"Using uploaded file: {uploaded_file.name}")
    elif os.getenv("INPUT_FILE"):
        pdf_path = os.getenv("INPUT_FILE")
        st.sidebar.info(f"Using env file: {os.path.basename(pdf_path)}")
    else:
        st.sidebar.error("Please upload a PDF or set INPUT_FILE in .env")
        st.stop()

    with st.spinner("Loading and indexing document..."):
        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text)
        vectors = embed_texts(chunks)
        index = build_faiss_index(vectors)

        st.session_state.index = index
        st.session_state.chunks = chunks

    st.sidebar.success("Vector index built!")

    with st.spinner("Building knowledge graph (GraphRAG)..."):
        try:
            entities, relationships = extract_graph_from_chunks(chunks)
            driver = get_driver()
            build_knowledge_graph(driver, entities, relationships)
            st.session_state.neo4j_driver = driver
            st.sidebar.success(f"Knowledge graph built: {len(entities)} entities, {len(relationships)} relationships.")
        except Exception as e:
            st.sidebar.warning(f"Graph build failed (vector search still available): {e}")
            st.session_state.neo4j_driver = None

# -----------------------------
# Question input
# -----------------------------
query = st.text_input("🔍 Ask a question:")

# -----------------------------
# Run Hybrid RAG
# -----------------------------
if query and st.session_state.index:
    with st.spinner("Searching and generating answer..."):
        results = hybrid_search(
            query,
            st.session_state.index,
            st.session_state.chunks,
            driver=st.session_state.neo4j_driver,
            top_k=3
        )

        answer = generate_answer(query, results["combined_context"])

    st.subheader("✅ Answer")
    st.write(answer)

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📚 Vector Search Results", expanded=True):
            for i, chunk in enumerate(results["vector_results"], 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)

    with col2:
        with st.expander("🔗 Knowledge Graph Context", expanded=True):
            if results["graph_results"]:
                for i, ctx in enumerate(results["graph_results"], 1):
                    st.markdown(f"**Relationship {i}:** {ctx}")
            else:
                st.info("No graph context found for this query.")

elif query and not st.session_state.index:
    st.warning("Please load the document first from the sidebar.")

# -----------------------------
# Knowledge Graph Visualization
# -----------------------------
ENTITY_COLORS = {
    "Person": "#FF6B6B",
    "Organization": "#4ECDC4",
    "Committee": "#45B7D1",
    "Role": "#96CEB4",
    "Regulation": "#FFEAA7",
    "Process": "#DDA0DD",
    "Document": "#98D8C8",
    "Concept": "#F7DC6F",
}

if st.session_state.neo4j_driver:
    st.divider()
    st.subheader("🔗 Knowledge Graph Visualization")

    try:
        graph_nodes, graph_edges = fetch_graph_visual_data(st.session_state.neo4j_driver, limit=150)

        if graph_nodes and graph_edges:
            nodes = [
                Node(
                    id=n["name"],
                    label=n["name"],
                    size=20,
                    color=ENTITY_COLORS.get(n["type"], "#97C2FC"),
                    title=f"{n['name']} ({n['type']})"
                )
                for n in graph_nodes
            ]
            edges = [
                Edge(
                    source=e["source"],
                    target=e["target"],
                    label=e["relation"],
                    color="#888888"
                )
                for e in graph_edges
            ]

            config = Config(
                width=900,
                height=600,
                directed=True,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=True,
            )

            agraph(nodes=nodes, edges=edges, config=config)

            # Legend
            st.markdown("**Legend:**")
            legend_cols = st.columns(len(ENTITY_COLORS))
            for col, (etype, color) in zip(legend_cols, ENTITY_COLORS.items()):
                col.markdown(
                    f'<span style="color:{color}; font-size:20px;">&#9679;</span> {etype}',
                    unsafe_allow_html=True
                )
        else:
            st.info("No graph data available. Load a document first.")
    except Exception as e:
        st.warning(f"Could not render graph: {e}")
