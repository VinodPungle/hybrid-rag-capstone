"""
Streamlit UI for the Hybrid RAG Pipeline

Provides a web interface for:
  - Uploading PDF documents or using INPUT_FILE from .env
  - Asking questions with hybrid search (vector + knowledge graph)
  - Viewing answers with source context from both search methods
  - Visualizing the knowledge graph with interactive node/edge rendering

UI configuration (title, description, colors, graph viz settings)
is driven by config.yaml → ui section.
"""

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

# [Step 1] Import config loader — UI settings from config.yaml
from config.settings import get as cfg

load_dotenv()

# [Step 1] Load UI config section (page_title, app_title, description, colors, graph_viz)
_ui = cfg("ui")

# [Step 1] Page title and app title from config.yaml instead of hardcoded values
st.set_page_config(page_title=_ui["page_title"], layout="wide")

st.title(f"📄 {_ui['app_title']}")
# [Step 1] App description from config.yaml
st.write(_ui["app_description"])

# -----------------------------
# Session state (cache objects)
# -----------------------------
# These persist across Streamlit reruns within the same browser session
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.neo4j_driver = None

# -----------------------------
# Upload / Load document
# -----------------------------
st.sidebar.header("📂 Document Setup")

# [Feature 1] File uploader allows users to upload PDFs directly in the UI
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if st.sidebar.button("Load Document"):
    # Determine PDF source: uploaded file takes priority, then INPUT_FILE env var
    if uploaded_file is not None:
        # [Feature 1] Save uploaded file to a temp directory for processing
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = tmp_path
        st.sidebar.info(f"Using uploaded file: {uploaded_file.name}")
    elif os.getenv("INPUT_FILE"):
        # Fallback to INPUT_FILE environment variable
        pdf_path = os.getenv("INPUT_FILE")
        st.sidebar.info(f"Using env file: {os.path.basename(pdf_path)}")
    else:
        st.sidebar.error("Please upload a PDF or set INPUT_FILE in .env")
        st.stop()

    # Pipeline: load PDF → chunk → embed → build FAISS index
    with st.spinner("Loading and indexing document..."):
        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text)
        vectors = embed_texts(chunks)
        index = build_faiss_index(vectors)

        st.session_state.index = index
        st.session_state.chunks = chunks

    st.sidebar.success("Vector index built!")

    # Build knowledge graph (non-fatal: vector search still works if this fails)
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
# [Feature 2] Ask button with validation — prevents blank query submission
ask_button = st.button("Ask")

# -----------------------------
# Run Hybrid RAG
# -----------------------------
# [Feature 2] Input validation: check for blank query and missing document
if ask_button and not query.strip():
    st.warning("Please enter a question before clicking Ask.")
elif ask_button and not st.session_state.index:
    st.warning("Please load the document first from the sidebar.")
elif ask_button and query.strip() and st.session_state.index:
    with st.spinner("Searching and generating answer..."):
        # Run hybrid search: vector similarity + knowledge graph
        results = hybrid_search(
            query,
            st.session_state.index,
            st.session_state.chunks,
            driver=st.session_state.neo4j_driver,
            top_k=3
        )

        # Generate answer using combined context from both search methods
        answer = generate_answer(query, results["combined_context"])

    st.subheader("✅ Answer")
    st.write(answer)

    # Display results from both search methods side by side
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


# -----------------------------
# Knowledge Graph Visualization
# -----------------------------
# [Step 1] Entity colors from config.yaml instead of hardcoded dict
ENTITY_COLORS = _ui["entity_colors"]

if st.session_state.neo4j_driver:
    st.divider()
    st.subheader("🔗 Knowledge Graph Visualization")

    try:
        # [Step 1] Limit now defaults from config.yaml → retrieval.graph_viz_limit
        graph_nodes, graph_edges = fetch_graph_visual_data(st.session_state.neo4j_driver)

        if graph_nodes and graph_edges:
            # Convert graph data to streamlit-agraph Node/Edge objects
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

            # [Step 1] Graph visualization settings from config.yaml → ui.graph_viz
            _gv = _ui["graph_viz"]
            config = Config(
                width=_gv["width"],
                height=_gv["height"],
                directed=_gv["directed"],
                physics=_gv["physics"],
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=True,
            )

            agraph(nodes=nodes, edges=edges, config=config)

            # Legend: color-coded entity types
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
