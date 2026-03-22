"""
FastAPI REST API for the Hybrid RAG Pipeline (Step 4)

Exposes the RAG pipeline as a REST API with three endpoints:
  - GET  /health  → Check if the service is running and a document is loaded
  - POST /ingest  → Upload a PDF to build the vector index + knowledge graph
  - POST /ask     → Ask a question and get an answer with source context

Also exposes GET /metrics for Prometheus scraping (Step 9).

Run with: uvicorn api.app:app --reload
Swagger UI available at: http://localhost:8000/docs
"""

import os
import time
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# [Step 9] Auto-instruments all FastAPI routes with Prometheus HTTP metrics
from prometheus_fastapi_instrumentator import Instrumentator

# [Step 1] Import config loader
from config.settings import get as cfg

# [Step 2] Import structured logger
from utils.logger import get_logger

# [Step 9] Import custom Prometheus metrics for LLMOps monitoring
from monitoring.prometheus_metrics import (
    RAG_REQUESTS_TOTAL, RAG_REQUEST_LATENCY,
    DOCUMENTS_INGESTED, CHUNKS_CREATED,
    GRAPH_ENTITIES, GRAPH_RELATIONSHIPS,
    VECTOR_RESULTS_COUNT, GRAPH_RESULTS_COUNT,
)

# Pipeline module imports
from ingestion.pdf_loader import load_pdf_text
from ingestion.chunker import chunk_text
from embeddings.embedder import embed_texts
from vector_db.faiss_store import build_faiss_index
from retrieval.hybrid_search import hybrid_search
from graph_db.entity_extractor import extract_graph_from_chunks
from graph_db.neo4j_store import get_driver, build_knowledge_graph
from llm.generator import generate_answer

load_dotenv()
logger = get_logger(__name__)

# --------------- In-memory state ---------------
# Stores the current FAISS index, chunks, and Neo4j driver between requests.
# These are populated by /ingest and consumed by /ask.
_state = {
    "index": None,
    "chunks": None,
    "neo4j_driver": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler: logs startup and cleans up Neo4j driver on shutdown."""
    logger.info("FastAPI server starting up.")
    yield
    # Cleanup: close Neo4j connection when the server shuts down
    if _state["neo4j_driver"]:
        _state["neo4j_driver"].close()
        logger.info("Neo4j driver closed.")


app = FastAPI(
    title="Hybrid RAG API",
    description="REST API for the Hybrid RAG pipeline (vector + knowledge graph).",
    version="1.0.0",
    lifespan=lifespan,
)

# [Step 9] Expose /metrics endpoint for Prometheus to scrape
# This auto-tracks HTTP request count, latency, and response size for all routes
Instrumentator().instrument(app).expose(app)


# --------------- Request / Response models ---------------
# Pydantic models for request validation and response serialization

class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""
    query: str = Field(..., min_length=1, description="The question to ask")


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""
    answer: str
    vector_results: list[str]
    graph_results: list[str]
    latency_s: float


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""
    chunks: int
    entities: int
    relationships: int
    message: str


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    document_loaded: bool


# --------------- Endpoints ---------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check: returns service status and whether a document is loaded."""
    return HealthResponse(
        status="ok",
        document_loaded=_state["index"] is not None,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF and build the vector index + knowledge graph.

    Pipeline: PDF → text extraction → chunking → embedding → FAISS index
              → entity extraction → Neo4j knowledge graph
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    start = time.perf_counter()

    # Save uploaded file to a temp path so pdf_loader can read it
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_path, "wb") as f:
        f.write(file.file.read())

    # Pipeline: load → chunk → embed → index
    text = load_pdf_text(tmp_path)
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")

    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    index = build_faiss_index(vectors)

    # Store in memory for subsequent /ask calls
    _state["index"] = index
    _state["chunks"] = chunks

    # Build knowledge graph (non-fatal: vector search still works if this fails)
    entity_count = 0
    rel_count = 0
    try:
        entities, relationships = extract_graph_from_chunks(chunks)
        driver = get_driver()
        build_knowledge_graph(driver, entities, relationships)
        _state["neo4j_driver"] = driver
        entity_count = len(entities)
        rel_count = len(relationships)
    except Exception as e:
        logger.warning("Graph build failed (vector search still available): %s", e)
        _state["neo4j_driver"] = None

    elapsed = time.perf_counter() - start
    logger.info("Ingestion complete in %.2fs: %d chunks, %d entities, %d relationships.",
                elapsed, len(chunks), entity_count, rel_count)

    # [Step 9] Record Prometheus metrics for the ingestion
    DOCUMENTS_INGESTED.inc()
    CHUNKS_CREATED.set(len(chunks))
    GRAPH_ENTITIES.set(entity_count)
    GRAPH_RELATIONSHIPS.set(rel_count)
    RAG_REQUESTS_TOTAL.labels(endpoint="ingest").inc()
    RAG_REQUEST_LATENCY.labels(endpoint="ingest").observe(elapsed)

    return IngestResponse(
        chunks=len(chunks),
        entities=entity_count,
        relationships=rel_count,
        message=f"Document ingested successfully in {elapsed:.1f}s.",
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Ask a question against the loaded document.

    Runs hybrid search (vector + graph), then generates an answer
    using the combined context from both search methods.
    """
    if _state["index"] is None:
        raise HTTPException(status_code=400, detail="No document loaded. Call /ingest first.")

    start = time.perf_counter()

    # Run hybrid search: vector similarity + knowledge graph traversal
    results = hybrid_search(
        request.query,
        _state["index"],
        _state["chunks"],
        driver=_state["neo4j_driver"],
    )

    # Generate answer using the combined context from both search methods
    answer = generate_answer(request.query, results["combined_context"])

    elapsed = time.perf_counter() - start
    logger.info("Query answered in %.2fs: '%s'", elapsed, request.query[:80])

    # [Step 9] Record Prometheus metrics for the query
    RAG_REQUESTS_TOTAL.labels(endpoint="ask").inc()
    RAG_REQUEST_LATENCY.labels(endpoint="ask").observe(elapsed)
    VECTOR_RESULTS_COUNT.observe(len(results["vector_results"]))
    GRAPH_RESULTS_COUNT.observe(len(results["graph_results"]))

    return AskResponse(
        answer=answer,
        vector_results=results["vector_results"],
        graph_results=results["graph_results"],
        latency_s=round(elapsed, 3),
    )
