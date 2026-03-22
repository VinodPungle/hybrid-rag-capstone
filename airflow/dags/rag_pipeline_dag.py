"""
Airflow DAG for the Hybrid RAG Pipeline (Step 10)

Orchestrates the full RAG pipeline as a directed acyclic graph (DAG) with 7 tasks:

  load_document → chunk_text → ┬─ create_embeddings → build_index ──┬─→ run_evaluation
                                └─ extract_entities → build_graph ──┘

The pipeline splits into two parallel branches after chunking:
  - Branch 1: embedding generation → FAISS index build (vector search path)
  - Branch 2: entity extraction → Neo4j knowledge graph build (GraphRAG path)
Both branches converge at the evaluation task.

Data passing strategy:
  - Small data (text, file paths, counts): passed via Airflow XCom
  - Large data (chunks, vectors, graph data): serialized to temp JSON files,
    with file paths passed via XCom

Trigger: manual (schedule=None) or set a cron expression for periodic re-ingestion.
"""

import os
import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# [Step 10] Default args applied to all tasks in the DAG
default_args = {
    "owner": "llmops",
    "depends_on_past": False,       # Each run is independent
    "email_on_failure": False,
    "retries": 1,                   # Retry once on failure (e.g., transient API errors)
    "retry_delay": timedelta(minutes=2),
}

# [Step 10] Define the DAG — manual trigger, no backfill
dag = DAG(
    dag_id="hybrid_rag_pipeline",
    default_args=default_args,
    description="End-to-end Hybrid RAG pipeline: ingest, embed, index, graph, evaluate",
    schedule=None,  # Manual trigger or set a cron (e.g., "@daily")
    start_date=datetime(2025, 1, 1),
    catchup=False,  # Don't backfill missed runs
    tags=["llmops", "rag"],
)


def task_load_document(**context):
    """
    Task 1: Load PDF and extract text.

    Reads INPUT_FILE from environment, extracts text via pdf_loader,
    and pushes the text to XCom (truncated to 500K chars for XCom size limits).
    """
    from dotenv import load_dotenv
    load_dotenv()
    from ingestion.pdf_loader import load_pdf_text

    input_file = os.getenv("INPUT_FILE")
    if not input_file or not os.path.exists(input_file):
        raise FileNotFoundError(f"INPUT_FILE not set or not found: {input_file}")

    text = load_pdf_text(input_file)
    if not text or not text.strip():
        raise ValueError("No text extracted from PDF.")

    # Push text via XCom (small data) — truncated to avoid XCom size limit
    context["ti"].xcom_push(key="document_text", value=text[:500000])
    context["ti"].xcom_push(key="input_file", value=input_file)
    return f"Loaded {len(text)} characters from {os.path.basename(input_file)}"


def task_chunk_text(**context):
    """
    Task 2: Chunk the document text.

    Pulls text from XCom, splits into chunks using config-driven parameters,
    and saves chunks to a temp file (too large for XCom).
    """
    from ingestion.chunker import chunk_text

    # Pull text from the previous task via XCom
    text = context["ti"].xcom_pull(task_ids="load_document", key="document_text")
    chunks = chunk_text(text)

    # Save chunks to a temp file (large data) — pass file path via XCom instead
    chunks_path = "/tmp/rag_chunks.json"
    with open(chunks_path, "w") as f:
        json.dump(chunks, f)

    context["ti"].xcom_push(key="chunks_path", value=chunks_path)
    context["ti"].xcom_push(key="chunk_count", value=len(chunks))
    return f"Created {len(chunks)} chunks"


def task_create_embeddings(**context):
    """
    Task 3a (Branch 1): Generate embeddings for all chunks via Azure OpenAI.

    Reads chunks from temp file, generates embedding vectors,
    and saves them to a temp file for the next task.
    """
    from embeddings.embedder import embed_texts

    # Load chunks from temp file (path passed via XCom)
    chunks_path = context["ti"].xcom_pull(task_ids="chunk_text", key="chunks_path")
    with open(chunks_path, "r") as f:
        chunks = json.load(f)

    vectors = embed_texts(chunks)

    # Save vectors to temp file (large data)
    vectors_path = "/tmp/rag_vectors.json"
    with open(vectors_path, "w") as f:
        json.dump(vectors, f)

    context["ti"].xcom_push(key="vectors_path", value=vectors_path)
    return f"Generated {len(vectors)} embeddings"


def task_build_index(**context):
    """
    Task 4a (Branch 1): Build the FAISS vector index from embeddings.

    Loads vectors from temp file, builds a FAISS index,
    and serializes it to disk for the evaluation task.
    """
    from vector_db.faiss_store import build_faiss_index
    import faiss
    import numpy as np

    # Load embedding vectors from temp file
    vectors_path = context["ti"].xcom_pull(task_ids="create_embeddings", key="vectors_path")
    with open(vectors_path, "r") as f:
        vectors = json.load(f)

    index = build_faiss_index(vectors)

    # Serialize FAISS index to disk — needed by the evaluation task
    index_path = "/tmp/rag_faiss.index"
    faiss.write_index(index, index_path)

    context["ti"].xcom_push(key="index_path", value=index_path)
    return f"FAISS index built with {index.ntotal} vectors"


def task_extract_entities(**context):
    """
    Task 3b (Branch 2): Extract entities and relationships from chunks via LLM.

    Uses the entity extraction prompt from config.yaml to identify entities
    and their relationships, saving the graph data to a temp file.
    """
    from graph_db.entity_extractor import extract_graph_from_chunks

    # Load chunks from temp file (shared with Branch 1)
    chunks_path = context["ti"].xcom_pull(task_ids="chunk_text", key="chunks_path")
    with open(chunks_path, "r") as f:
        chunks = json.load(f)

    entities, relationships = extract_graph_from_chunks(chunks)

    # Save graph data to temp file for the next task
    graph_path = "/tmp/rag_graph.json"
    with open(graph_path, "w") as f:
        json.dump({"entities": entities, "relationships": relationships}, f)

    context["ti"].xcom_push(key="graph_path", value=graph_path)
    context["ti"].xcom_push(key="entity_count", value=len(entities))
    context["ti"].xcom_push(key="relationship_count", value=len(relationships))
    return f"Extracted {len(entities)} entities, {len(relationships)} relationships"


def task_build_knowledge_graph(**context):
    """
    Task 4b (Branch 2): Store entities and relationships in Neo4j.

    Loads graph data from temp file and writes it to the Neo4j database.
    The driver is closed in a finally block to prevent connection leaks.
    """
    from graph_db.neo4j_store import get_driver, build_knowledge_graph

    # Load extracted graph data from temp file
    graph_path = context["ti"].xcom_pull(task_ids="extract_entities", key="graph_path")
    with open(graph_path, "r") as f:
        graph_data = json.load(f)

    driver = get_driver()
    try:
        build_knowledge_graph(driver, graph_data["entities"], graph_data["relationships"])
    finally:
        # Always close the driver, even if graph build fails
        driver.close()

    return "Knowledge graph built in Neo4j"


def task_run_evaluation(**context):
    """
    Task 5 (Convergence): Run evaluation metrics on test cases.

    This task runs after both branches complete. It:
      1. Loads pipeline artifacts (chunks + FAISS index) from temp files
      2. Optionally connects to Neo4j for hybrid search
      3. For each test case: runs hybrid search → generates answer → computes metrics
      4. Saves results to a temp file

    [Step 8] Uses the evaluate() function which runs all config-enabled metrics.
    """
    from evaluation.metrics import evaluate
    from retrieval.hybrid_search import hybrid_search
    from llm.generator import generate_answer
    from graph_db.neo4j_store import get_driver
    import faiss
    import numpy as np

    # Load pipeline artifacts from temp files (created by earlier tasks)
    chunks_path = context["ti"].xcom_pull(task_ids="chunk_text", key="chunks_path")
    index_path = context["ti"].xcom_pull(task_ids="build_index", key="index_path")

    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    # Deserialize FAISS index from disk
    index = faiss.read_index(index_path)

    # Try to connect to Neo4j for hybrid search (non-fatal if unavailable)
    driver = None
    try:
        driver = get_driver()
    except Exception:
        pass

    # Load test cases from the evaluation directory
    test_cases_path = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "test_cases.json")
    if not os.path.exists(test_cases_path):
        return "No test cases found, skipping evaluation."

    with open(test_cases_path, "r") as f:
        test_cases = json.load(f)

    # Run each test case through the full pipeline and compute metrics
    results = []
    for tc in test_cases:
        search_results = hybrid_search(tc["query"], index, chunks, driver=driver)
        answer = generate_answer(tc["query"], search_results["combined_context"])
        # [Step 8] Compute all enabled metrics (BLEU, ROUGE, faithfulness)
        scores = evaluate(tc["reference_answer"], answer, context=search_results["combined_context"])
        results.append({"query": tc["query"], "scores": scores})

    # Save evaluation results to temp file
    eval_output_path = "/tmp/rag_eval_results.json"
    with open(eval_output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Cleanup Neo4j connection
    if driver:
        driver.close()

    context["ti"].xcom_push(key="eval_results_path", value=eval_output_path)
    return f"Evaluation complete: {len(results)} test cases"


# --- [Step 10] Define Airflow tasks using PythonOperator ---
# Each task wraps a pipeline function and passes data via XCom or temp files

load_doc = PythonOperator(
    task_id="load_document",
    python_callable=task_load_document,
    dag=dag,
)

chunk = PythonOperator(
    task_id="chunk_text",
    python_callable=task_chunk_text,
    dag=dag,
)

embed = PythonOperator(
    task_id="create_embeddings",
    python_callable=task_create_embeddings,
    dag=dag,
)

build_idx = PythonOperator(
    task_id="build_index",
    python_callable=task_build_index,
    dag=dag,
)

extract_ents = PythonOperator(
    task_id="extract_entities",
    python_callable=task_extract_entities,
    dag=dag,
)

build_graph = PythonOperator(
    task_id="build_knowledge_graph",
    python_callable=task_build_knowledge_graph,
    dag=dag,
)

run_eval = PythonOperator(
    task_id="run_evaluation",
    python_callable=task_run_evaluation,
    dag=dag,
)

# --- [Step 10] Pipeline DAG dependency graph ---
# load → chunk → [embed → index, extract → graph] → evaluate
# Two parallel branches after chunking converge at evaluation
load_doc >> chunk
chunk >> [embed, extract_ents]        # Split into two parallel branches
embed >> build_idx                     # Branch 1: embeddings → FAISS index
extract_ents >> build_graph            # Branch 2: entities → Neo4j graph
[build_idx, build_graph] >> run_eval   # Both branches must complete before evaluation
