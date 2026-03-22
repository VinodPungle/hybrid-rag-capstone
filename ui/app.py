"""
Streamlit UI for the Hybrid RAG Pipeline (Production Setup)

In production mode, the UI is a thin client that calls the FastAPI backend
instead of running the pipeline directly. This separation of concerns means:
  - The pipeline runs in one place (FastAPI) — single source of truth
  - All requests flow through the API — metrics, logging, and monitoring capture everything
  - The UI only handles presentation — no pipeline imports needed
  - Multiple UIs (web, mobile, CLI) can share the same backend

API endpoints used:
  - GET  /health → check if service is up and document is loaded
  - POST /ingest → upload PDF to build vector index + knowledge graph
  - POST /ask    → ask a question, get answer with source context
  - GET  /graph  → fetch knowledge graph nodes/edges for visualization

UI configuration (title, description, colors, graph viz settings)
is driven by config.yaml → ui section.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests
from dotenv import load_dotenv
from streamlit_agraph import agraph, Node, Edge, Config

# [Step 1] Import config loader — UI settings from config.yaml
from config.settings import get as cfg

load_dotenv()

# [Step 1] Load UI config section (page_title, app_title, description, colors, graph_viz)
_ui = cfg("ui")

# [Production] FastAPI backend URL — configurable via env var or defaults to localhost:8000
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# [Step 1] Page title and app title from config.yaml instead of hardcoded values
st.set_page_config(page_title=_ui["page_title"], layout="wide")

st.title(f"📄 {_ui['app_title']}")
# [Step 1] App description from config.yaml
st.write(_ui["app_description"])

# -----------------------------
# Session state
# -----------------------------
# Track whether a document has been loaded (via API health check)
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
    st.session_state.graph_available = False

# [Production] Check API health on load to sync state
try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
    st.session_state.document_loaded = health.get("document_loaded", False)
    st.session_state.graph_available = health.get("graph_available", False)
except Exception:
    pass

# -----------------------------
# Sidebar: API connection status
# -----------------------------
st.sidebar.header("📂 Document Setup")

# [Production] Show API connection status in sidebar
try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=3).json()
    st.sidebar.success(f"API connected ({API_BASE_URL})")
    if health.get("document_loaded"):
        st.sidebar.info("Document is already loaded in the API.")
except Exception:
    st.sidebar.error(f"API not reachable at {API_BASE_URL}")
    st.sidebar.caption("Start the API with: `uvicorn api.app:app --port 8000`")

# -----------------------------
# Upload / Load document
# -----------------------------
# [Feature 1] File uploader allows users to upload PDFs directly in the UI
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if st.sidebar.button("Load Document"):
    if uploaded_file is not None:
        # [Production] Send uploaded file to FastAPI /ingest endpoint
        with st.spinner("Uploading and indexing document via API..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                resp = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.document_loaded = True
                    st.session_state.graph_available = data.get("entities", 0) > 0
                    st.sidebar.success(
                        f"Document ingested: {data['chunks']} chunks, "
                        f"{data['entities']} entities, "
                        f"{data['relationships']} relationships."
                    )
                    st.sidebar.caption(data["message"])
                else:
                    st.sidebar.error(f"Ingestion failed: {resp.json().get('detail', resp.text)}")
            except requests.ConnectionError:
                st.sidebar.error(f"Cannot connect to API at {API_BASE_URL}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
    elif os.getenv("INPUT_FILE"):
        # [Production] For INPUT_FILE, read locally and send to API
        pdf_path = os.getenv("INPUT_FILE")
        if not os.path.exists(pdf_path):
            st.sidebar.error(f"File not found: {pdf_path}")
        else:
            with st.spinner("Uploading and indexing document via API..."):
                try:
                    with open(pdf_path, "rb") as f:
                        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
                        resp = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.document_loaded = True
                        st.session_state.graph_available = data.get("entities", 0) > 0
                        st.sidebar.success(
                            f"Document ingested: {data['chunks']} chunks, "
                            f"{data['entities']} entities, "
                            f"{data['relationships']} relationships."
                        )
                        st.sidebar.caption(data["message"])
                    else:
                        st.sidebar.error(f"Ingestion failed: {resp.json().get('detail', resp.text)}")
                except requests.ConnectionError:
                    st.sidebar.error(f"Cannot connect to API at {API_BASE_URL}")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
    else:
        st.sidebar.error("Please upload a PDF or set INPUT_FILE in .env")

# -----------------------------
# Question input
# -----------------------------
query = st.text_input("🔍 Ask a question:")
# [Feature 2] Ask button with validation — prevents blank query submission
ask_button = st.button("Ask")

# -----------------------------
# Run Hybrid RAG via API
# -----------------------------
# [Feature 2] Input validation: check for blank query and missing document
if ask_button and not query.strip():
    st.warning("Please enter a question before clicking Ask.")
elif ask_button and not st.session_state.document_loaded:
    st.warning("Please load the document first from the sidebar.")
elif ask_button and query.strip() and st.session_state.document_loaded:
    with st.spinner("Searching and generating answer..."):
        try:
            # [Production] Call FastAPI /ask endpoint instead of running pipeline directly
            resp = requests.post(
                f"{API_BASE_URL}/ask",
                json={"query": query},
                timeout=60,
            )
            if resp.status_code == 200:
                result = resp.json()
            else:
                st.error(f"API error: {resp.json().get('detail', resp.text)}")
                st.stop()
        except requests.ConnectionError:
            st.error(f"Cannot connect to API at {API_BASE_URL}")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.subheader("✅ Answer")
    st.write(result["answer"])

    # [Production] Show latency from the API response
    st.caption(f"Response time: {result['latency_s']}s")

    # Display results from both search methods side by side
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📚 Vector Search Results", expanded=True):
            for i, chunk in enumerate(result["vector_results"], 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)

    with col2:
        with st.expander("🔗 Knowledge Graph Context", expanded=True):
            if result["graph_results"]:
                for i, ctx in enumerate(result["graph_results"], 1):
                    st.markdown(f"**Relationship {i}:** {ctx}")
            else:
                st.info("No graph context found for this query.")


# -----------------------------
# Knowledge Graph Visualization
# -----------------------------
# [Step 1] Entity colors from config.yaml instead of hardcoded dict
ENTITY_COLORS = _ui["entity_colors"]

if st.session_state.graph_available:
    st.divider()
    st.subheader("🔗 Knowledge Graph Visualization")

    try:
        # [Production] Fetch graph data from FastAPI /graph endpoint
        resp = requests.get(f"{API_BASE_URL}/graph", timeout=30)
        if resp.status_code == 200:
            graph_data = resp.json()
            graph_nodes = graph_data.get("nodes", [])
            graph_edges = graph_data.get("edges", [])

            if graph_nodes and graph_edges:
                # Convert API response to streamlit-agraph Node/Edge objects
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
    except requests.ConnectionError:
        st.warning(f"Cannot connect to API at {API_BASE_URL} for graph data.")
    except Exception as e:
        st.warning(f"Could not render graph: {e}")
