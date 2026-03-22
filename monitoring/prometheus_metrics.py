"""
Custom Prometheus Metrics for LLMOps Monitoring (Step 9)

Defines all custom metrics that are tracked across the Hybrid RAG pipeline.
These metrics are scraped by Prometheus and visualized in Grafana.

Metric types used:
  - Counter: monotonically increasing values (requests, tokens, documents)
  - Histogram: distribution of values with configurable buckets (latency, result counts)
  - Gauge: current values that can go up and down (chunk/entity/relationship counts)

Metrics are recorded in:
  - api/app.py: request counts, latency, ingestion stats, result counts
  - llm/generator.py: generation call tokens and latency
  - graph_db/entity_extractor.py: extraction call tokens and latency
"""

from prometheus_client import Counter, Histogram, Gauge

# --- Request metrics ---
# Tracks total API requests and their latency, labeled by endpoint (ingest/ask)
RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total number of RAG queries processed",
    ["endpoint"],
)

RAG_REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "End-to-end latency for RAG queries",
    ["endpoint"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# --- LLM token metrics ---
# Tracks token consumption per LLM call, labeled by call type and token type.
# call_type: "generation" (answer gen) or "entity_extraction" (GraphRAG)
# token_type: "prompt" or "completion"
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens consumed by LLM calls",
    ["call_type", "token_type"],
)

# Tracks latency of individual LLM API calls (not end-to-end request latency)
LLM_CALL_LATENCY = Histogram(
    "llm_call_latency_seconds",
    "Latency of individual LLM API calls",
    ["call_type"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# --- Pipeline metrics ---
# Tracks document ingestion counts and current index/graph sizes
DOCUMENTS_INGESTED = Counter(
    "documents_ingested_total",
    "Total number of documents ingested",
)

CHUNKS_CREATED = Gauge(
    "chunks_created_current",
    "Number of chunks in the current document index",
)

GRAPH_ENTITIES = Gauge(
    "graph_entities_current",
    "Number of entities in the current knowledge graph",
)

GRAPH_RELATIONSHIPS = Gauge(
    "graph_relationships_current",
    "Number of relationships in the current knowledge graph",
)

# --- Retrieval metrics ---
# Tracks how many results each search method returns per query
VECTOR_RESULTS_COUNT = Histogram(
    "vector_results_count",
    "Number of vector search results returned per query",
    buckets=[0, 1, 2, 3, 5, 10],
)

GRAPH_RESULTS_COUNT = Histogram(
    "graph_results_count",
    "Number of graph search results returned per query",
    buckets=[0, 1, 3, 5, 10, 15, 30],
)


def record_llm_call(call_type, response, latency_s):
    """
    Record Prometheus metrics for an LLM API call.

    Called from llm/generator.py and graph_db/entity_extractor.py
    after each Azure OpenAI API call.

    Args:
        call_type: "generation" or "entity_extraction"
        response: OpenAI-style response with .usage attribute
        latency_s: Wall-clock time for the API call in seconds
    """
    # Record call latency in the histogram
    LLM_CALL_LATENCY.labels(call_type=call_type).observe(latency_s)

    # Record token usage if available in the response
    if hasattr(response, "usage") and response.usage:
        LLM_TOKENS_TOTAL.labels(call_type=call_type, token_type="prompt").inc(
            response.usage.prompt_tokens
        )
        LLM_TOKENS_TOTAL.labels(call_type=call_type, token_type="completion").inc(
            response.usage.completion_tokens
        )
