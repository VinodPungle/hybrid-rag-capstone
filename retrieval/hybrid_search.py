"""
Hybrid Search Module

Combines vector search (FAISS) and knowledge graph search (Neo4j)
to produce a merged context for LLM answer generation. This is the
core of the Hybrid RAG approach.

Parameters default to config.yaml → retrieval section.
"""

from retrieval.search import search
from retrieval.graph_search import graph_search

# [Step 1] Import config loader for retrieval defaults
from config.settings import get as cfg

# [Step 2] Import structured logger (replaces print() for warnings)
from utils.logger import get_logger

logger = get_logger(__name__)


def hybrid_search(query, index, chunks, driver=None, top_k=None, graph_limit=None):
    """
    Combine vector search results with knowledge graph context.

    Runs both search methods and merges results into a single context
    string that the LLM uses to generate answers.

    Args:
        query: The user's question
        index: FAISS index for vector search
        chunks: List of text chunks
        driver: Neo4j driver for graph search (None = skip graph or auto-create)
        top_k: Number of vector search results
               (defaults to config.yaml → retrieval.vector_top_k)
        graph_limit: Max graph relationships per entity match
                     (defaults to config.yaml → retrieval.graph_limit)

    Returns:
        dict with keys:
            - "vector_results": list of chunk strings from vector search
            - "graph_results": list of relationship strings from graph search
            - "combined_context": merged context string for the LLM
    """
    # [Step 1] Load defaults from config.yaml if not explicitly provided
    _ret = cfg("retrieval")
    if top_k is None:
        top_k = _ret["vector_top_k"]
    if graph_limit is None:
        graph_limit = _ret["graph_limit"]

    # Vector search: find the most similar document chunks
    vector_results = search(query, index, chunks, top_k=top_k)

    # Graph search: find related entities/relationships in the knowledge graph
    try:
        graph_results = graph_search(query, driver=driver, limit=graph_limit)
    except Exception as e:
        # [Step 2] Graceful degradation: log warning and fall back to vector-only
        logger.warning("Graph search failed, falling back to vector-only: %s", e)
        graph_results = []

    # Build combined context string with labeled sections for the LLM
    context_parts = []

    if vector_results:
        context_parts.append("=== Document Context ===")
        context_parts.extend(vector_results)

    if graph_results:
        context_parts.append("\n=== Knowledge Graph Context ===")
        context_parts.extend(graph_results)

    combined_context = "\n".join(context_parts)

    return {
        "vector_results": vector_results,
        "graph_results": graph_results,
        "combined_context": combined_context
    }
