from retrieval.search import search
from retrieval.graph_search import graph_search


def hybrid_search(query, index, chunks, driver=None, top_k=3, graph_limit=15):
    """
    Combine vector search results with knowledge graph context.

    Args:
        query: The user's question
        index: FAISS index for vector search
        chunks: List of text chunks
        driver: Neo4j driver for graph search
        top_k: Number of vector search results
        graph_limit: Max graph relationships per entity match

    Returns:
        dict with keys:
            - "vector_results": list of chunk strings from vector search
            - "graph_results": list of relationship strings from graph search
            - "combined_context": merged context string for the LLM
    """
    # Vector search
    vector_results = search(query, index, chunks, top_k=top_k)

    # Graph search
    try:
        graph_results = graph_search(query, driver=driver, limit=graph_limit)
    except Exception as e:
        print(f"Warning: Graph search failed, falling back to vector-only: {e}")
        graph_results = []

    # Build combined context
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
